#!/usr/bin/env python3
"""Extract, transcribe, translate, or sync subtitles for video files.

Checks for embedded SRT subtitles first and extracts them.
If none are found, uses faster-whisper for transcription.
Use --sync to synchronize existing SRT files to video audio via ffsubsync.
Use --translate-subs to translate SRT files via the Claude API.
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _setup_cuda_paths():
    """Add pip-installed NVIDIA libraries to LD_LIBRARY_PATH.

    Needed when the system CUDA version (e.g. 13) does not match
    the version ctranslate2/faster-whisper was built against (CUDA 12).
    """
    try:
        import nvidia.cublas
        import nvidia.cudnn
    except ImportError:
        return

    lib_paths = []
    for pkg in [nvidia.cublas, nvidia.cudnn]:
        pkg_dir = Path(pkg.__path__[0]) / "lib"
        if pkg_dir.is_dir():
            lib_paths.append(str(pkg_dir))

    if not lib_paths:
        return

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    new = ":".join(lib_paths)
    os.environ["LD_LIBRARY_PATH"] = f"{new}:{existing}" if existing else new


_setup_cuda_paths()

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".webm", ".mov", ".ts"}
SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".sub"}


def check_dependencies(needs_ffmpeg=True, needs_whisper=False, needs_sync=False):
    """Check that required tools and libraries are available. Exits with instructions if not."""
    missing = []

    if needs_ffmpeg:
        if not shutil.which("ffmpeg"):
            missing.append(("ffmpeg", "Install via package manager (apt/pacman/brew/choco)"))
        if not shutil.which("ffprobe"):
            missing.append(("ffprobe", "Installed together with ffmpeg"))

    if needs_whisper:
        try:
            from faster_whisper import WhisperModel  # noqa: F401
        except ImportError:
            missing.append(("faster-whisper", "pip install faster-whisper"))

    if needs_sync:
        try:
            import ffsubsync  # noqa: F401
        except ImportError:
            missing.append(("ffsubsync", "pip install ffsubsync"))

    if missing:
        print("Missing dependencies:", file=sys.stderr)
        for name, fix in missing:
            print(f"  - {name}: {fix}", file=sys.stderr)
        sys.exit(1)


def sync_subtitles(video_path: Path, srt_path: Path, output_path: Path | None) -> bool:
    """Synchronize an SRT file to video audio using ffsubsync."""
    if output_path is None:
        output_path = srt_path.with_suffix(".synced.srt")

    print(f"  Syncing: {srt_path.name}")
    print(f"  Against: {video_path.name}")
    print(f"  Saving to: {output_path.name}")

    cmd = [
        sys.executable, "-m", "ffsubsync",
        str(video_path),
        "-i", str(srt_path),
        "-o", str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Sync error: {result.stderr.strip()}", file=sys.stderr)
        return False

    # Show offset if ffsubsync reports it
    for line in result.stderr.splitlines():
        if "offset" in line.lower() or "framerate" in line.lower():
            print(f"  {line.strip()}")

    print(f"  Sync complete!")
    return True


def ffprobe_subtitle_streams(video_path: Path) -> list[dict]:
    """Return list of subtitle streams in the video file."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "s",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    data = json.loads(result.stdout)
    return data.get("streams", [])


def extract_subtitles(video_path: Path, output_path: Path) -> bool:
    """Extract the first SRT stream from the video file."""
    cmd = [
        "ffmpeg", "-y", "-v", "warning",
        "-i", str(video_path),
        "-map", "0:s:0",
        "-f", "srt",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Extraction error: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def _load_whisper_model(whisper_cls, model_name: str, compute_type: str = "int8",
                        device: str = "auto"):
    """Load a Whisper model with informative progress messages."""
    # CPU does not support float16
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_cached = (cache_dir / f"models--Systran--faster-whisper-{model_name}").exists()

    if model_cached:
        print(f"  Loading Whisper model '{model_name}' ({compute_type}, {device})...")
    else:
        print(f"  Downloading Whisper model '{model_name}' "
              f"(first run, may take several minutes)...")

    model = whisper_cls(model_name, device=device, compute_type=compute_type)
    print(f"  Model ready.")
    return model


def _format_duration(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _run_whisper(whisper_model, wav_path: str, video_path: Path,
                 output_path: Path | None, force: bool,
                 language: str | None) -> tuple[bool, list[str], Path | None]:
    """Run Whisper transcription. Returns (already_exists, srt_lines, output_path).

    Raises RuntimeError on CUDA issues (OOM, missing libraries).
    """
    kwargs = {}
    if language:
        kwargs["language"] = language

    segments, info = whisper_model.transcribe(wav_path, **kwargs)
    detected_lang = info.language
    total_dur = info.duration
    print(f"  Detected language: {detected_lang} (confidence: {info.language_probability:.1%})")
    print(f"  Duration: {_format_duration(total_dur)}")

    # Determine output path from detected language if not specified
    if output_path is None:
        output_path = video_path.parent / (video_path.stem + f".{detected_lang}.srt")
        if output_path.exists() and not force:
            print(f"  Already exists: {output_path.name}")
            return True, [], output_path

    print(f"  Saving to: {output_path.name}")

    srt_lines = []
    for i, segment in enumerate(segments, start=1):
        start_ts = format_srt_timestamp(segment.start)
        end_ts = format_srt_timestamp(segment.end)
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_ts} --> {end_ts}")
        srt_lines.append(segment.text.strip())
        srt_lines.append("")
        if total_dur > 0:
            pct = min(segment.end / total_dur * 100, 100)
            print(f"\r  Transcribing... {pct:.0f}% "
                  f"({_format_duration(segment.end)}/{_format_duration(total_dur)})",
                  end="", flush=True)

    print()  # Newline after progress
    return False, srt_lines, output_path


def transcribe_with_whisper(video_path: Path, output_path: Path | None,
                            model_name: str, language: str | None,
                            whisper_model=None, force: bool = False,
                            compute_type: str = "int8",
                            device: str = "auto") -> bool:
    """Transcribe audio with faster-whisper and save as SRT.

    If output_path is None, the filename is determined from the detected language.
    If whisper_model is None, the model is loaded automatically.
    On CUDA OOM, automatically falls back to CPU.
    """
    if whisper_model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("Error: faster-whisper is not installed.", file=sys.stderr)
            print("Run: pip install faster-whisper", file=sys.stderr)
            return False
        whisper_model = _load_whisper_model(WhisperModel, model_name, compute_type, device)

    # Extract audio to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    print(f"  Extracting audio...")
    cmd = [
        "ffmpeg", "-y", "-v", "warning",
        "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Audio extraction error: {result.stderr.strip()}", file=sys.stderr)
        Path(wav_path).unlink(missing_ok=True)
        return False

    print(f"  Transcribing...")

    try:
        already_exists, srt_lines, output_path = _run_whisper(
            whisper_model, wav_path, video_path, output_path, force, language
        )
        if already_exists:
            Path(wav_path).unlink(missing_ok=True)
            return True

    except RuntimeError as e:
        err = str(e)
        if "not found or cannot be loaded" in err:
            print(f"\n  Error: CUDA library missing: {e}", file=sys.stderr)
            print(f"  Run: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12", file=sys.stderr)
            Path(wav_path).unlink(missing_ok=True)
            return False

        if "out of memory" in err.lower():
            print(f"  GPU out of memory - falling back to CPU...", file=sys.stderr)
            try:
                from faster_whisper import WhisperModel
                cpu_type = "int8" if compute_type == "float16" else compute_type
                whisper_model = _load_whisper_model(
                    WhisperModel, model_name, cpu_type, "cpu"
                )
                already_exists, srt_lines, output_path = _run_whisper(
                    whisper_model, wav_path, video_path, output_path, force, language
                )
                if already_exists:
                    Path(wav_path).unlink(missing_ok=True)
                    return True
            except Exception as cpu_err:
                print(f"\n  Error: Transcription also failed on CPU: {cpu_err}",
                      file=sys.stderr)
                print(f"  Try: -m medium or -m small", file=sys.stderr)
                Path(wav_path).unlink(missing_ok=True)
                return False
        else:
            Path(wav_path).unlink(missing_ok=True)
            raise

    output_path.write_text("\n".join(srt_lines), encoding="utf-8")

    # Cleanup
    Path(wav_path).unlink(missing_ok=True)
    return True


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


LANG_NAMES = {
    "sv": "Swedish", "en": "English", "da": "Danish", "no": "Norwegian",
    "fi": "Finnish", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "ja": "Japanese", "zh": "Chinese",
}


def parse_srt(srt_path: Path) -> list[dict]:
    """Parse an SRT file into a list of {index, start, end, text}."""
    content = srt_path.read_text(encoding="utf-8-sig")  # Handles BOM
    segments = []
    blocks = content.strip().split("\n\n")

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        timestamp_line = lines[1].strip()
        if " --> " not in timestamp_line:
            continue
        start, end = timestamp_line.split(" --> ", 1)
        text = "\n".join(lines[2:])
        segments.append({
            "index": index,
            "start": start.strip(),
            "end": end.strip(),
            "text": text,
        })

    return segments


def write_srt(segments: list[dict], output_path: Path):
    """Write a segment list as an SRT file."""
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{seg['start']} --> {seg['end']}")
        lines.append(seg["text"])
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def opensubtitles_hash(file_path: Path) -> str:
    """Compute the OpenSubtitles hash for a video file."""
    block_size = 65536
    file_size = file_path.stat().st_size

    if file_size < block_size * 2:
        raise ValueError(f"File too small for hash computation ({file_size} bytes)")

    hash_val = file_size
    with open(file_path, 'rb') as f:
        for _ in range(block_size // 8):
            buf = f.read(8)
            hash_val += struct.unpack('<q', buf)[0]
            hash_val &= 0xFFFFFFFFFFFFFFFF

        f.seek(max(0, file_size - block_size))
        for _ in range(block_size // 8):
            buf = f.read(8)
            hash_val += struct.unpack('<q', buf)[0]
            hash_val &= 0xFFFFFFFFFFFFFFFF

    return f"{hash_val:016x}"


def _opensubtitles_search(api_key: str, **params) -> list[dict]:
    """Search the OpenSubtitles API."""
    base_url = "https://api.opensubtitles.com/api/v1/subtitles"
    query_str = urllib.parse.urlencode(params)
    url = f"{base_url}?{query_str}"

    req = urllib.request.Request(url, headers={
        "Api-Key": api_key,
        "User-Agent": "subtitle-tool v1.0",
    })

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        return data.get("data", [])
    except urllib.error.HTTPError as e:
        print(f"  OpenSubtitles API error: {e.code} {e.reason}", file=sys.stderr)
        return []
    except urllib.error.URLError as e:
        print(f"  Network error: {e.reason}", file=sys.stderr)
        return []


def _opensubtitles_download(file_id: int, output_path: Path, api_key: str) -> bool:
    """Download a subtitle file from OpenSubtitles.com."""
    url = "https://api.opensubtitles.com/api/v1/download"
    req_data = json.dumps({"file_id": file_id}).encode()
    req = urllib.request.Request(url, data=req_data, headers={
        "Api-Key": api_key,
        "User-Agent": "subtitle-tool v1.0",
        "Content-Type": "application/json",
        "Accept": "application/json",
    })

    try:
        with urllib.request.urlopen(req) as resp:
            dl_data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"  Download error: {e.code} {e.reason}", file=sys.stderr)
        return False

    dl_link = dl_data.get("link")
    if not dl_link:
        print("  No download link in API response", file=sys.stderr)
        return False

    try:
        urllib.request.urlretrieve(dl_link, str(output_path))
        return True
    except Exception as e:
        print(f"  Download failed: {e}", file=sys.stderr)
        return False


def fetch_opensubtitles(video_path: Path, api_key: str, force: bool = False) -> bool:
    """Search and download Swedish and English subtitles from OpenSubtitles.com."""
    print(f"\nOpenSubtitles: {video_path.name}")

    try:
        file_hash = opensubtitles_hash(video_path)
    except ValueError as e:
        print(f"  {e}", file=sys.stderr)
        return False
    print(f"  Hash: {file_hash}")

    languages = {"en": "English", "sv": "Swedish"}
    any_downloaded = False

    for lang, lang_name in languages.items():
        output_path = video_path.parent / (video_path.stem + f".{lang}.srt")

        if output_path.exists() and not force:
            print(f"  {lang_name} already exists: {output_path.name}")
            continue

        # Search by hash
        results = _opensubtitles_search(api_key, moviehash=file_hash, languages=lang)
        hash_matched = [r for r in results if r["attributes"].get("moviehash_match")]
        non_hash = [r for r in results if not r["attributes"].get("moviehash_match")]

        # Fallback: search by filename if no results
        if not results:
            query = video_path.stem.replace(".", " ").replace("_", " ").replace("-", " ")
            results = _opensubtitles_search(api_key, query=query, languages=lang)
            non_hash = results

        if hash_matched:
            best = hash_matched[0]
            attrs = best["attributes"]
            file_id = attrs["files"][0]["file_id"]
            release = attrs.get("release", "unknown")
            print(f"  {lang_name} (hash OK): {release}")
            if _opensubtitles_download(file_id, output_path, api_key):
                print(f"  Saved: {output_path.name}")
                any_downloaded = True
        elif non_hash:
            best = non_hash[0]
            attrs = best["attributes"]
            file_id = attrs["files"][0]["file_id"]
            release = attrs.get("release", "unknown")
            print(f"\n  WARNING: {lang_name} subtitle found but hash does NOT match!")
            print(f"  Release: {release}")
            print(f"  The subtitle may not match your video file exactly.")
            try:
                answer = input(f"  Download anyway? (y/n): ").strip().lower()
            except EOFError:
                answer = "n"
            if answer in ("j", "ja", "y", "yes"):
                if _opensubtitles_download(file_id, output_path, api_key):
                    print(f"  Saved: {output_path.name}")
                    any_downloaded = True
            else:
                print(f"  Skipping {lang_name} subtitle.")
        else:
            print(f"  No {lang_name} subtitle found")

    return any_downloaded


def translate_batch_claude(texts: list[str], source_lang: str, target_lang: str,
                           api_key: str, model: str) -> list[str]:
    """Translate a batch of subtitle lines via the Claude API."""
    source_name = LANG_NAMES.get(source_lang, source_lang)
    target_name = LANG_NAMES.get(target_lang, target_lang)

    # Build numbered input (join multi-line entries with spaces)
    numbered = "\n".join(
        f"{i+1}: {t.replace(chr(10), ' ')}" for i, t in enumerate(texts)
    )

    prompt = (
        f"Translate the following numbered subtitle lines from {source_name} "
        f"to {target_name}.\n"
        f"Rules:\n"
        f"- Return ONLY the translations, numbered exactly like the input.\n"
        f"- Preserve the original meaning and tone.\n"
        f"- Keep proper nouns unchanged.\n"
        f"- Do not add explanations or notes.\n\n"
        f"{numbered}"
    )

    body = json.dumps({
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        print(f"  Claude API error: {e.code} {e.reason}", file=sys.stderr)
        if error_body:
            print(f"  {error_body[:200]}", file=sys.stderr)
        raise
    except urllib.error.URLError as e:
        print(f"  Network error: {e.reason}", file=sys.stderr)
        raise

    response_text = data["content"][0]["text"]

    # Parse numbered response
    result = []
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip "N: " prefix
        colon_pos = line.find(": ")
        if colon_pos != -1 and colon_pos < 6 and line[:colon_pos].isdigit():
            result.append(line[colon_pos + 2:])
        else:
            result.append(line)

    return result


def _translate_output_path(srt_path: Path, target_lang: str) -> Path:
    """Determine output filename for translation."""
    stem = srt_path.stem
    # Check if stem ends with .xx (language code)
    parts = stem.rsplit(".", 1)
    if len(parts) == 2 and len(parts[1]) == 2 and parts[1].isalpha():
        return srt_path.parent / f"{parts[0]}.{target_lang}.srt"
    return srt_path.parent / f"{stem}.{target_lang}.srt"


def translate_subtitles(srt_path: Path, target_lang: str, api_key: str,
                        model: str, force: bool) -> bool:
    """Translate an SRT file to another language via the Claude API."""
    output_path = _translate_output_path(srt_path, target_lang)

    if output_path.exists() and not force:
        print(f"  Already exists: {output_path.name} (use --force to overwrite)")
        return True

    print(f"\nTranslating: {srt_path.name} -> {output_path.name}")

    segments = parse_srt(srt_path)
    if not segments:
        print("  No subtitles to translate", file=sys.stderr)
        return False

    # Detect source language from filename
    stem_parts = srt_path.stem.rsplit(".", 1)
    if len(stem_parts) == 2 and len(stem_parts[1]) == 2 and stem_parts[1].isalpha():
        source_lang = stem_parts[1]
    else:
        source_lang = "en"

    target_name = LANG_NAMES.get(target_lang, target_lang)
    print(f"  Language: {source_lang} -> {target_lang} ({target_name})")
    print(f"  Model: {model}")
    print(f"  Subtitle count: {len(segments)}")

    # Batch and translate
    batch_size = 40
    translated_texts = []
    total = len(segments)
    total_batches = (total + batch_size - 1) // batch_size

    for i in range(0, total, batch_size):
        batch = [s["text"] for s in segments[i:i + batch_size]]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} lines)...")

        try:
            result = translate_batch_claude(
                batch, source_lang, target_lang, api_key, model
            )
        except Exception:
            print(f"  Aborting translation.", file=sys.stderr)
            return False

        # Validate line count
        if len(result) != len(batch):
            print(f"  Warning: got {len(result)} lines, expected {len(batch)}",
                  file=sys.stderr)
            while len(result) < len(batch):
                result.append(batch[len(result)])
            result = result[:len(batch)]

        translated_texts.extend(result)

    # Build output segments with original timestamps
    output_segments = []
    for seg, text in zip(segments, translated_texts):
        output_segments.append({
            "index": seg["index"],
            "start": seg["start"],
            "end": seg["end"],
            "text": text,
        })

    write_srt(output_segments, output_path)
    print(f"  Done! Saved: {output_path.name} ({total} lines translated)")
    return True


def process_file(video_path: Path, force: bool, model: str,
                 language: str | None, only_whisper: bool = False,
                 whisper_model=None, compute_type: str = "int8",
                 device: str = "auto") -> bool:
    """Process a video file - extract or transcribe subtitles."""
    print(f"\nProcessing: {video_path.name}")

    if not only_whisper:
        # Check for embedded subtitles
        streams = ffprobe_subtitle_streams(video_path)
        if streams:
            codec = streams[0].get("codec_name", "unknown")
            lang = streams[0].get("tags", {}).get("language", "und")
            output_path = video_path.parent / (video_path.stem + f".{lang}.srt")
            if output_path.exists() and not force:
                print(f"  Already exists: {output_path.name}")
                return True
            print(f"  Found embedded subtitle: {codec} ({lang})")
            print(f"  Extracting to: {output_path.name}")
            return extract_subtitles(video_path, output_path)
        print(f"  No embedded subtitles - starting Whisper transcription")
    else:
        print(f"  Starting Whisper transcription (--only-whisper)")

    # If language specified, determine output path upfront; otherwise auto-detect
    if language:
        output_path = video_path.parent / (video_path.stem + f".{language}.srt")
        if output_path.exists() and not force:
            print(f"  Already exists: {output_path.name}")
            return True
    else:
        output_path = None  # Determined by detected language

    return transcribe_with_whisper(video_path, output_path, model, language,
                                   whisper_model, force, compute_type, device)


def main():
    parser = argparse.ArgumentParser(
        description="Extract, transcribe, or translate subtitles for video files"
    )
    parser.add_argument(
        "path",
        help="Path to video file or directory (batch mode)"
    )
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Whisper language (default: auto-detect)"
    )
    parser.add_argument(
        "-m", "--model",
        default="large-v3",
        help="Whisper model (default: large-v3)"
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Whisper compute type: int8, float16, float32 (default: int8)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for Whisper: auto, cuda, cpu (default: auto)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing subtitle files"
    )
    parser.add_argument(
        "--sync",
        metavar="SRT",
        help="Sync an SRT file to video audio (requires: video_path + --sync sub.srt)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file for --sync (default: <input>.synced.srt)"
    )
    parser.add_argument(
        "--only-whisper",
        action="store_true",
        help="Skip embedded subtitles, always use Whisper"
    )
    parser.add_argument(
        "--opensubtitles",
        action="store_true",
        help="Download Swedish and English subtitles from OpenSubtitles.com"
    )
    parser.add_argument(
        "--translate-subs",
        metavar="SRT",
        help="Translate an SRT file to another language via Claude API (requires: --to)"
    )
    parser.add_argument(
        "--to",
        default=None,
        help="Target language for --translate-subs (e.g. sv, en, de, fr)"
    )
    parser.add_argument(
        "--translate-model",
        default="claude-haiku-4-5-20251001",
        help="Claude model for translation (default: claude-haiku-4-5-20251001)"
    )
    args = parser.parse_args()

    # Check dependencies based on mode
    if args.sync:
        check_dependencies(needs_ffmpeg=True, needs_sync=True)
    elif not args.opensubtitles and not args.translate_subs:
        check_dependencies(needs_ffmpeg=True, needs_whisper=args.only_whisper)

    target = Path(args.path)

    # Sync mode: sync an SRT to video audio
    if args.sync:
        srt_path = Path(args.sync)
        if not target.is_file():
            print(f"Video file not found: {target}", file=sys.stderr)
            sys.exit(1)
        if not srt_path.is_file():
            print(f"SRT file not found: {srt_path}", file=sys.stderr)
            sys.exit(1)
        out = Path(args.output) if args.output else None
        success = sync_subtitles(target, srt_path, out)
        sys.exit(0 if success else 1)

    # OpenSubtitles mode: download subtitles
    if args.opensubtitles:
        api_key = os.environ.get("OPENSUBTITLES_API_KEY")
        if not api_key:
            print("Error: OPENSUBTITLES_API_KEY is not set.", file=sys.stderr)
            print("See README for instructions.", file=sys.stderr)
            sys.exit(1)

        if target.is_file():
            if target.suffix.lower() not in VIDEO_EXTENSIONS:
                print(f"Warning: {target.suffix} is not a known video format", file=sys.stderr)
            success = fetch_opensubtitles(target, api_key, args.force)
            sys.exit(0 if success else 1)
        elif target.is_dir():
            video_files = sorted(
                f for f in target.iterdir()
                if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
            )
            if not video_files:
                print(f"No video files found in: {target}")
                sys.exit(1)
            print(f"Found {len(video_files)} video files in: {target}")
            for vf in video_files:
                fetch_opensubtitles(vf, api_key, args.force)
            sys.exit(0)
        else:
            print(f"Path not found: {target}", file=sys.stderr)
            sys.exit(1)

    # Translate mode: translate SRT via Claude API
    if args.translate_subs:
        if not args.to:
            print("Error: --translate-subs requires --to (target language)", file=sys.stderr)
            print("Example: --translate-subs movie.en.srt --to sv", file=sys.stderr)
            sys.exit(1)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
            print("See README for instructions.", file=sys.stderr)
            sys.exit(1)

        srt_target = Path(args.translate_subs)
        if srt_target.is_file():
            success = translate_subtitles(
                srt_target, args.to, api_key, args.translate_model, args.force
            )
            sys.exit(0 if success else 1)
        elif srt_target.is_dir():
            srt_files = sorted(
                f for f in srt_target.iterdir()
                if f.is_file() and f.suffix.lower() == ".srt"
            )
            if not srt_files:
                print(f"No SRT files found in: {srt_target}")
                sys.exit(1)
            print(f"Found {len(srt_files)} SRT files in: {srt_target}")
            ok = 0
            for sf in srt_files:
                if translate_subtitles(
                    sf, args.to, api_key, args.translate_model, args.force
                ):
                    ok += 1
            print(f"\nDone! {ok}/{len(srt_files)} translated.")
            sys.exit(0 if ok == len(srt_files) else 1)
        else:
            print(f"SRT file not found: {srt_target}", file=sys.stderr)
            sys.exit(1)

    if target.is_file():
        if target.suffix.lower() not in VIDEO_EXTENSIONS:
            print(f"Warning: {target.suffix} is not a known video format", file=sys.stderr)
        success = process_file(target, args.force, args.model, args.language,
                               args.only_whisper, compute_type=args.compute_type,
                               device=args.device)
        sys.exit(0 if success else 1)

    elif target.is_dir():
        video_files = sorted(
            f for f in target.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not video_files:
            print(f"No video files found in: {target}")
            sys.exit(1)

        print(f"Found {len(video_files)} video files in: {target}")

        # Load Whisper model once for batch mode
        whisper_model = None
        try:
            from faster_whisper import WhisperModel
            whisper_model = _load_whisper_model(WhisperModel, args.model,
                                                args.compute_type, args.device)
        except ImportError:
            pass  # Handled per file if Whisper is needed

        ok = 0
        fail = 0
        for vf in video_files:
            if process_file(vf, args.force, args.model, args.language,
                            args.only_whisper, whisper_model, args.compute_type,
                            args.device):
                ok += 1
            else:
                fail += 1

        print(f"\nDone! {ok} succeeded, {fail} failed out of {len(video_files)} files.")
        sys.exit(0 if fail == 0 else 1)

    else:
        print(f"Path not found: {target}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
