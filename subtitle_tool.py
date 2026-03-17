#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "faster-whisper",
#     "ffsubsync",
#     "setuptools<81",
#     "nvidia-cublas-cu12",
#     "nvidia-cudnn-cu12",
# ]
# ///
"""Extract, transcribe, translate, or sync subtitles for video files.

Checks for embedded SRT subtitles first and extracts them.
If none are found, uses faster-whisper for transcription.
Use --sync to synchronize existing SRT files to video audio via ffsubsync.
Use --translate-subs to translate SRT files via the Claude API.
"""

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _setup_cuda_paths():
    """Add pip-installed NVIDIA libraries to LD_LIBRARY_PATH and pre-load them.

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

    # Set LD_LIBRARY_PATH for subprocesses
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    new = ":".join(lib_paths)
    os.environ["LD_LIBRARY_PATH"] = f"{new}:{existing}" if existing else new

    # Pre-load CUDA libraries so ctranslate2 can find them via dlopen
    import ctypes
    for lib_dir in lib_paths:
        for so_file in sorted(Path(lib_dir).glob("*.so*")):
            try:
                ctypes.CDLL(str(so_file), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_setup_cuda_paths()

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".webm", ".mov", ".ts"}
SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".sub"}

# ANSI color codes (disabled if output is not a terminal)
if sys.stderr.isatty() and sys.stdout.isatty():
    C_RESET = "\033[0m"
    C_BOLD = "\033[1m"
    C_RED = "\033[31m"
    C_GREEN = "\033[32m"
    C_YELLOW = "\033[33m"
    C_CYAN = "\033[36m"
    C_DIM = "\033[2m"
else:
    C_RESET = C_BOLD = C_RED = C_GREEN = C_YELLOW = C_CYAN = C_DIM = ""


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
        print(f"{C_RED}Missing dependencies:{C_RESET}", file=sys.stderr)
        for name, fix in missing:
            print(f"  {C_RED}-{C_RESET} {C_BOLD}{name}{C_RESET}: {fix}", file=sys.stderr)
        sys.exit(1)


def sync_subtitles(video_path: Path, srt_path: Path, output_path: Path | None) -> bool:
    """Synchronize an SRT file to video audio using ffsubsync."""
    if output_path is None:
        output_path = srt_path  # Overwrite the original file

    print(f"  Syncing: {C_BOLD}{srt_path.name}{C_RESET}")
    print(f"  Against: {C_BOLD}{video_path.name}{C_RESET}")
    if output_path != srt_path:
        print(f"  Saving to: {C_CYAN}{output_path.name}{C_RESET}")

    cmd = ["ffsubsync"]
    cmd += [str(video_path), "-i", str(srt_path), "-o", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  {C_RED}Sync error:{C_RESET} {result.stderr.strip()}", file=sys.stderr)
        return False

    # Show offset if ffsubsync reports it
    for line in result.stderr.splitlines():
        if "offset" in line.lower() or "framerate" in line.lower():
            print(f"  {C_DIM}{line.strip()}{C_RESET}")

    print(f"  {C_GREEN}Sync complete!{C_RESET}")
    return True


# Bitmap subtitle codecs that cannot be extracted to SRT
_BITMAP_SUB_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"}


def ffprobe_subtitle_streams(video_path: Path) -> list[dict]:
    """Return list of text-based subtitle streams in the video file.

    Bitmap formats (PGS, VobSub, DVB) are excluded as they cannot be
    converted to SRT.
    """
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
    streams = data.get("streams", [])
    # Filter out bitmap subtitle formats
    return [s for s in streams if s.get("codec_name") not in _BITMAP_SUB_CODECS]


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
        print(f"  {C_RED}Extraction error:{C_RESET} {result.stderr.strip()}", file=sys.stderr)
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
        print(f"  {C_CYAN}Loading Whisper model '{model_name}' ({compute_type}, {device})...{C_RESET}")
    else:
        print(f"  {C_CYAN}Downloading Whisper model '{model_name}' "
              f"(first run, may take several minutes)...{C_RESET}")

    model = whisper_cls(model_name, device=device, compute_type=compute_type)
    print(f"  {C_GREEN}Model ready.{C_RESET}")
    return model


def _format_duration(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# Patterns that indicate hallucinated (non-speech) content
_HALLUCINATION_RE = re.compile(
    r"^("
    r"©.*|"                          # Copyright lines
    r"[\u266a-\u266f\u2669]+.*|"     # Music note characters
    r"¶+.*|"                         # Pilcrow symbols
    r"Tack.*för att.*tittade?.*|"    # "Thanks for watching" in Swedish
    r"Thank.* for watching.*|"       # "Thanks for watching" in English
    r"Undertext.*av.*|"              # "Subtitled by" in Swedish
    r"Subtitl.*by.*|"               # "Subtitled by" in English
    r"transcript.*|"                 # Transcript credits
    r"\.{3,}"                        # Just dots
    r")$",
    re.IGNORECASE,
)

# Max characters per subtitle line before splitting (industry standard: 42)
_MAX_LINE_CHARS = 42


def _is_hallucination(text: str, duration: float = 0) -> bool:
    """Check if a segment looks like a Whisper hallucination."""
    if _HALLUCINATION_RE.match(text):
        return True
    # Very short segments that are just noise
    if len(text) <= 2 and not text.isalpha():
        return True
    # Long silent gap with short text (music/credits filler)
    if duration > 15 and len(text) < 30:
        return True
    return False


def _split_at_word_boundaries(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of max_chars at word boundaries."""
    words = text.split()
    parts = []
    current = []
    current_len = 0
    for word in words:
        added_len = len(word) + (1 if current else 0)
        if current_len + added_len > max_chars and current:
            parts.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += added_len
    if current:
        parts.append(" ".join(current))
    return parts


def _distribute_time(parts: list[str], start: float, end: float) -> list[dict]:
    """Distribute time proportionally across text parts."""
    total_chars = sum(len(p) for p in parts)
    duration = end - start
    result = []
    t = start
    for part in parts:
        part_dur = duration * (len(part) / total_chars) if total_chars > 0 else 0
        result.append({"start": t, "end": t + part_dur, "text": part.strip()})
        t += part_dur
    return result


def _split_long_segment(start: float, end: float, text: str) -> list[dict]:
    """Split a long subtitle segment into shorter ones.

    Tries sentence boundaries first, then commas, then hard word-boundary split.
    """
    if len(text) <= _MAX_LINE_CHARS:
        return [{"start": start, "end": end, "text": text}]

    # Try sentence boundaries (. ! ? followed by space and uppercase)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÅÄÖ])', text)
    if len(parts) > 1:
        # Recursively split any parts that are still too long
        result = []
        for p in _distribute_time(parts, start, end):
            result.extend(_split_long_segment(p["start"], p["end"], p["text"]))
        return result

    # Try commas/semicolons
    parts = re.split(r'(?<=[,;])\s+', text)
    if len(parts) > 1:
        result = []
        for p in _distribute_time(parts, start, end):
            result.extend(_split_long_segment(p["start"], p["end"], p["text"]))
        return result

    # Hard split at word boundaries
    parts = _split_at_word_boundaries(text, _MAX_LINE_CHARS)
    if len(parts) > 1:
        return _distribute_time(parts, start, end)

    return [{"start": start, "end": end, "text": text}]


def _merge_short_segments(segments: list[tuple]) -> list[tuple]:
    """Merge consecutive short segments into longer ones.

    Prevents excessive fragmentation from Whisper (e.g. 14 tiny segments
    for one sentence).
    """
    if not segments:
        return segments

    merged = [segments[0]]
    for start, end, text in segments[1:]:
        prev_start, prev_end, prev_text = merged[-1]
        prev_dur = prev_end - prev_start
        gap = start - prev_end

        # Merge if: previous segment is short, gap is tiny, and combined text fits
        combined = f"{prev_text} {text}"
        if (prev_dur < 3 and gap < 0.5 and len(prev_text) < 30
                and len(combined) <= _MAX_LINE_CHARS):
            merged[-1] = (prev_start, end, combined)
        else:
            merged.append((start, end, text))

    return merged


def _run_whisper(whisper_model, wav_path: str, video_path: Path,
                 output_path: Path | None, force: bool,
                 language: str | None) -> tuple[bool, list[str], Path | None]:
    """Run Whisper transcription. Returns (already_exists, srt_lines, output_path).

    Raises RuntimeError on CUDA issues (OOM, missing libraries).
    """
    kwargs = {
        "condition_on_previous_text": False,  # Prevent hallucination loops
        "no_speech_threshold": 0.6,
        "hallucination_silence_threshold": 2,  # Skip silent segments >2s
    }
    if language:
        kwargs["language"] = language

    segments, info = whisper_model.transcribe(wav_path, **kwargs)
    detected_lang = info.language
    total_dur = info.duration
    print(f"  Detected language: {C_BOLD}{detected_lang}{C_RESET} {C_DIM}(confidence: {info.language_probability:.1%}){C_RESET}")
    print(f"  Duration: {C_BOLD}{_format_duration(total_dur)}{C_RESET}")

    # Determine output path from detected language if not specified
    if output_path is None:
        output_path = video_path.parent / (video_path.stem + f".{detected_lang}.srt")
        if output_path.exists() and not force:
            print(f"  {C_YELLOW}Already exists:{C_RESET} {output_path.name}")
            return True, [], output_path

    print(f"  Saving to: {C_CYAN}{output_path.name}{C_RESET}")

    # Collect and filter segments
    raw_segments = []
    prev_text = ""
    repeat_count = 0
    filtered = 0
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        # Skip repeated text (hallucination loops)
        if text == prev_text:
            repeat_count += 1
            if repeat_count >= 2:
                filtered += 1
                continue
        else:
            repeat_count = 0
        prev_text = text

        # Skip hallucination patterns
        if _is_hallucination(text, segment.end - segment.start):
            filtered += 1
            continue

        raw_segments.append((segment.start, segment.end, text))

        if total_dur > 0:
            pct = min(segment.end / total_dur * 100, 100)
            print(f"\r  {C_CYAN}Transcribing... {pct:.0f}% "
                  f"({_format_duration(segment.end)}/{_format_duration(total_dur)}){C_RESET}",
                  end="", flush=True)

    print()  # Newline after progress

    # Merge consecutive short fragments
    merged_segments = _merge_short_segments(raw_segments)
    merged_count = len(raw_segments) - len(merged_segments)

    # Split long segments
    final_segments = []
    for start, end, text in merged_segments:
        final_segments.extend(_split_long_segment(start, end, text))
    split_count = len(final_segments) - len(merged_segments)

    # Build SRT output
    srt_lines = []
    for idx, seg in enumerate(final_segments, start=1):
        start_ts = format_srt_timestamp(seg["start"])
        end_ts = format_srt_timestamp(seg["end"])
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{start_ts} --> {end_ts}")
        srt_lines.append(seg["text"])
        srt_lines.append("")

    stats = []
    if filtered:
        stats.append(f"filtered {filtered} hallucination(s)")
    if merged_count:
        stats.append(f"merged {merged_count} fragment(s)")
    if split_count:
        stats.append(f"split {split_count} long segment(s)")
    if stats:
        print(f"  {C_DIM}{', '.join(stats).capitalize()}{C_RESET}")

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
            print(f"{C_RED}Error: faster-whisper is not installed.{C_RESET}", file=sys.stderr)
            print(f"Run: {C_BOLD}pip install faster-whisper{C_RESET}", file=sys.stderr)
            return False
        whisper_model = _load_whisper_model(WhisperModel, model_name, compute_type, device)

    # Extract audio to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    print(f"  {C_CYAN}Extracting audio...{C_RESET}")
    cmd = [
        "ffmpeg", "-y", "-v", "warning",
        "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  {C_RED}Audio extraction error:{C_RESET} {result.stderr.strip()}", file=sys.stderr)
        Path(wav_path).unlink(missing_ok=True)
        return False

    print(f"  {C_CYAN}Transcribing...{C_RESET}")

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
            print(f"\n  {C_RED}Error: CUDA library missing:{C_RESET} {e}", file=sys.stderr)
            print(f"  Run: {C_BOLD}pip install nvidia-cublas-cu12 nvidia-cudnn-cu12{C_RESET}", file=sys.stderr)
            Path(wav_path).unlink(missing_ok=True)
            return False

        if "out of memory" in err.lower():
            print(f"  {C_YELLOW}GPU out of memory - falling back to CPU...{C_RESET}", file=sys.stderr)
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
                print(f"\n  {C_RED}Error: Transcription also failed on CPU:{C_RESET} {cpu_err}",
                      file=sys.stderr)
                print(f"  Try: {C_BOLD}-m medium{C_RESET} or {C_BOLD}-m small{C_RESET}", file=sys.stderr)
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
    # Try UTF-8 first, fall back to common encodings for older subtitle files
    for enc in ("utf-8-sig", "latin-1"):
        try:
            content = srt_path.read_text(encoding=enc)
            break
        except (UnicodeDecodeError, ValueError):
            continue
    else:
        raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Cannot decode {srt_path}")
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


def _opensubtitles_request(api_key: str, url: str, data: bytes | None = None,
                           token: str | None = None) -> dict | None:
    """Make an OpenSubtitles API request with retry on rate limit."""
    headers = {
        "Api-Key": api_key,
        "User-Agent": "subtitle-tool v1.0",
    }
    if data:
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for attempt in range(1, 4):
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Read retry-after header or default to 2s * attempt
                retry_after = int(e.headers.get("Retry-After", attempt * 2))
                print(f"  {C_YELLOW}Rate limited - waiting {retry_after}s...{C_RESET}")
                time.sleep(retry_after)
                continue
            print(f"  {C_RED}OpenSubtitles API error:{C_RESET} {e.code} {e.reason}",
                  file=sys.stderr)
            if e.code == 401:
                print(f"  Set OPENSUBTITLES_USERNAME and OPENSUBTITLES_PASSWORD",
                      file=sys.stderr)
            return None
        except urllib.error.URLError as e:
            print(f"  {C_RED}Network error:{C_RESET} {e.reason}", file=sys.stderr)
            return None

    print(f"  {C_RED}Rate limit exceeded, try again later{C_RESET}", file=sys.stderr)
    return None


def _opensubtitles_search(api_key: str, **params) -> list[dict]:
    """Search the OpenSubtitles API."""
    base_url = "https://api.opensubtitles.com/api/v1/subtitles"
    query_str = urllib.parse.urlencode(params)
    url = f"{base_url}?{query_str}"
    data = _opensubtitles_request(api_key, url)
    return data.get("data", []) if data else []


def _opensubtitles_login(api_key: str, username: str, password: str) -> str | None:
    """Login to OpenSubtitles and return a JWT token."""
    url = "https://api.opensubtitles.com/api/v1/login"
    body = json.dumps({"username": username, "password": password}).encode()
    data = _opensubtitles_request(api_key, url, data=body)
    return data.get("token") if data else None


def _opensubtitles_download(file_id: int, output_path: Path, api_key: str,
                            token: str | None = None) -> bool:
    """Download a subtitle file from OpenSubtitles.com."""
    url = "https://api.opensubtitles.com/api/v1/download"
    body = json.dumps({"file_id": file_id}).encode()
    dl_data = _opensubtitles_request(api_key, url, data=body, token=token)
    if not dl_data:
        return False

    dl_link = dl_data.get("link")
    if not dl_link:
        print(f"  {C_RED}No download link in API response{C_RESET}", file=sys.stderr)
        return False

    try:
        dl_req = urllib.request.Request(dl_link, headers={
            "User-Agent": "subtitle-tool v1.0",
        })
        with urllib.request.urlopen(dl_req) as resp:
            output_path.write_bytes(resp.read())
        return True
    except Exception as e:
        print(f"  {C_RED}Download failed:{C_RESET} {e}", file=sys.stderr)
        return False


def fetch_opensubtitles(video_path: Path, api_key: str, force: bool = False,
                        token: str | None = None) -> bool:
    """Search and download Swedish and English subtitles from OpenSubtitles.com."""
    print(f"\n{C_BOLD}OpenSubtitles: {video_path.name}{C_RESET}")

    try:
        file_hash = opensubtitles_hash(video_path)
    except ValueError as e:
        print(f"  {e}", file=sys.stderr)
        return False
    print(f"  Hash: {C_DIM}{file_hash}{C_RESET}")

    languages = {"en": "English", "sv": "Swedish"}
    any_found = False

    for lang, lang_name in languages.items():
        output_path = video_path.parent / (video_path.stem + f".{lang}.srt")

        if output_path.exists() and not force:
            print(f"  {C_YELLOW}{lang_name} already exists:{C_RESET} {output_path.name}")
            any_found = True
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
            print(f"  {C_GREEN}{lang_name} (hash OK):{C_RESET} {release}")
            if _opensubtitles_download(file_id, output_path, api_key, token):
                print(f"  {C_GREEN}Saved:{C_RESET} {output_path.name}")
                any_found = True
        elif non_hash:
            best = non_hash[0]
            attrs = best["attributes"]
            file_id = attrs["files"][0]["file_id"]
            release = attrs.get("release", "unknown")
            print(f"  {C_YELLOW}{lang_name} (hash mismatch):{C_RESET} {release}")
            if _opensubtitles_download(file_id, output_path, api_key, token):
                print(f"  {C_GREEN}Saved:{C_RESET} {output_path.name}")
                print(f"  {C_CYAN}Auto-syncing to video...{C_RESET}")
                sync_subtitles(video_path, output_path, None)
                any_found = True
        else:
            print(f"  {C_DIM}No {lang_name} subtitle found{C_RESET}")

    return any_found


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
        "max_tokens": 8192,
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

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
            break
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            if e.code == 529 or e.code == 429:
                # Overloaded or rate limited - retry with backoff
                if attempt < max_retries:
                    wait = attempt * 15
                    print(f"  {C_YELLOW}API busy ({e.code}) - retrying in {wait}s (attempt {attempt}/{max_retries})...{C_RESET}")
                    time.sleep(wait)
                    req = urllib.request.Request(req.full_url, data=body, headers=dict(req.headers))
                    continue
            print(f"  {C_RED}Claude API error:{C_RESET} {e.code} {e.reason}", file=sys.stderr)
            if error_body:
                print(f"  {error_body[:200]}", file=sys.stderr)
            raise
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            if attempt < max_retries:
                wait = attempt * 10
                print(f"  {C_YELLOW}Connection error - retrying in {wait}s (attempt {attempt}/{max_retries})...{C_RESET}")
                time.sleep(wait)
                req = urllib.request.Request(req.full_url, data=body, headers=dict(req.headers))
                continue
            print(f"  {C_RED}Network error:{C_RESET} {e}", file=sys.stderr)
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
        print(f"  {C_YELLOW}Already exists:{C_RESET} {output_path.name} (use --force to overwrite)")
        return True

    print(f"\n{C_BOLD}Translating:{C_RESET} {srt_path.name} -> {C_CYAN}{output_path.name}{C_RESET}")

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
    print(f"  Language: {C_BOLD}{source_lang} -> {target_lang}{C_RESET} ({target_name})")
    print(f"  Model: {C_DIM}{model}{C_RESET}")
    print(f"  Subtitle count: {C_BOLD}{len(segments)}{C_RESET}")

    # Translate in batches to stay within API output token limits and avoid timeouts
    batch_size = 200
    all_texts = [s["text"] for s in segments]
    translated_texts = []
    total_batches = (len(all_texts) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(all_texts), batch_size):
        batch = all_texts[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        if total_batches > 1:
            print(f"  {C_CYAN}Translating batch {batch_num}/{total_batches} ({len(batch)} lines)...{C_RESET}")
        else:
            print(f"  {C_CYAN}Translating {len(batch)} lines...{C_RESET}")

        try:
            result = translate_batch_claude(
                batch, source_lang, target_lang, api_key, model
            )
        except Exception as e:
            print(f"  {C_RED}Translation error:{C_RESET} {e}", file=sys.stderr)
            return False

        # Validate line count
        if len(result) != len(batch):
            print(f"  {C_YELLOW}Warning: got {len(result)} lines, expected {len(batch)}{C_RESET}",
                  file=sys.stderr)
            while len(result) < len(batch):
                result.append(batch[len(result)])
            result = result[:len(batch)]

        translated_texts.extend(result)

        # Small delay between batches to respect rate limits
        if batch_num < total_batches:
            time.sleep(2)

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
    print(f"  {C_GREEN}Done! Saved: {output_path.name} ({len(segments)} lines translated){C_RESET}")
    return True


def process_file(video_path: Path, force: bool, model: str,
                 language: str | None, only_whisper: bool = False,
                 whisper_model=None, compute_type: str = "int8",
                 device: str = "auto",
                 os_api_key: str | None = None,
                 os_token: str | None = None) -> bool:
    """Process a video file - extract, download, or transcribe subtitles.

    Priority: 1) embedded subs, 2) OpenSubtitles, 3) Whisper transcription.
    """
    print(f"\n{C_BOLD}Processing: {video_path.name}{C_RESET}")

    if not only_whisper:
        # Step 1: Check for embedded subtitles
        streams = ffprobe_subtitle_streams(video_path)
        if streams:
            codec = streams[0].get("codec_name", "unknown")
            lang = streams[0].get("tags", {}).get("language", "und")
            output_path = video_path.parent / (video_path.stem + f".{lang}.srt")
            if output_path.exists() and not force:
                print(f"  {C_YELLOW}Already exists:{C_RESET} {output_path.name}")
                return True
            print(f"  Found embedded subtitle: {C_BOLD}{codec}{C_RESET} ({lang})")
            print(f"  Extracting to: {C_CYAN}{output_path.name}{C_RESET}")
            return extract_subtitles(video_path, output_path)

        # Step 2: Try OpenSubtitles if credentials are available
        if os_api_key:
            if fetch_opensubtitles(video_path, os_api_key, force, os_token):
                return True

        print(f"  {C_DIM}No subtitles found - falling back to Whisper{C_RESET}")
    else:
        print(f"  Starting Whisper transcription {C_DIM}(--only-whisper){C_RESET}")

    # Step 3: Whisper transcription
    if language:
        output_path = video_path.parent / (video_path.stem + f".{language}.srt")
        if output_path.exists() and not force:
            print(f"  {C_YELLOW}Already exists:{C_RESET} {output_path.name}")
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
        "-c", "--compute-type",
        default="int8",
        help="Whisper compute type: int8, float16, float32 (default: int8)"
    )
    parser.add_argument(
        "-d", "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for Whisper: auto, cuda, cpu (default: auto)"
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Overwrite existing subtitle files"
    )
    parser.add_argument(
        "-s", "--sync",
        metavar="SRT",
        help="Sync an SRT file to video audio (requires: video_path + --sync sub.srt)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file for --sync (default: overwrite input)"
    )
    parser.add_argument(
        "-w", "--only-whisper",
        action="store_true",
        help="Skip embedded subtitles, always use Whisper"
    )
    parser.add_argument(
        "--opensubtitles",
        action="store_true",
        help="Download Swedish and English subtitles from OpenSubtitles.com"
    )
    parser.add_argument(
        "-t", "--translate-subs",
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

    # OpenSubtitles login (if credentials are available)
    os_api_key = os.environ.get("OPENSUBTITLES_API_KEY")
    os_token = None
    if os_api_key:
        os_user = os.environ.get("OPENSUBTITLES_USERNAME")
        os_pass = os.environ.get("OPENSUBTITLES_PASSWORD")
        if os_user and os_pass:
            print(f"{C_CYAN}OpenSubtitles: logging in as {os_user}...{C_RESET}")
            os_token = _opensubtitles_login(os_api_key, os_user, os_pass)
            if os_token:
                print(f"{C_GREEN}OpenSubtitles: login OK{C_RESET}")
            else:
                print(f"{C_YELLOW}OpenSubtitles: login failed, downloads may fail{C_RESET}")

    # OpenSubtitles-only mode: just download subtitles
    if args.opensubtitles:
        if not os_api_key:
            print(f"{C_RED}Error: OPENSUBTITLES_API_KEY is not set.{C_RESET}", file=sys.stderr)
            print(f"See README for instructions.", file=sys.stderr)
            sys.exit(1)
        if target.is_file():
            if target.suffix.lower() not in VIDEO_EXTENSIONS:
                print(f"{C_YELLOW}Warning: {target.suffix} is not a known video format{C_RESET}", file=sys.stderr)
            success = fetch_opensubtitles(target, os_api_key, args.force, os_token)
            sys.exit(0 if success else 1)
        elif target.is_dir():
            video_files = sorted(
                f for f in target.rglob("*")
                if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
            )
            if not video_files:
                print(f"No video files found in: {target}")
                sys.exit(1)
            print(f"Found {C_BOLD}{len(video_files)}{C_RESET} video files in: {target}")
            for vf in video_files:
                fetch_opensubtitles(vf, os_api_key, args.force, os_token)
                time.sleep(1)  # Rate limit courtesy
            sys.exit(0)
        else:
            print(f"Path not found: {target}", file=sys.stderr)
            sys.exit(1)

    # Translate mode: translate SRT via Claude API
    if args.translate_subs:
        if not args.to:
            print(f"{C_RED}Error: --translate-subs requires --to (target language){C_RESET}", file=sys.stderr)
            print("Example: --translate-subs movie.en.srt --to sv", file=sys.stderr)
            sys.exit(1)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print(f"{C_RED}Error: ANTHROPIC_API_KEY is not set.{C_RESET}", file=sys.stderr)
            print(f"See README for instructions.", file=sys.stderr)
            sys.exit(1)

        srt_target = Path(args.translate_subs)
        if srt_target.is_file():
            success = translate_subtitles(
                srt_target, args.to, api_key, args.translate_model, args.force
            )
            sys.exit(0 if success else 1)
        elif srt_target.is_dir():
            srt_files = sorted(
                f for f in srt_target.rglob("*")
                if f.is_file() and f.suffix.lower() == ".srt"
            )
            if not srt_files:
                print(f"No SRT files found in: {srt_target}")
                sys.exit(1)
            print(f"Found {C_BOLD}{len(srt_files)}{C_RESET} SRT files in: {srt_target}")
            ok = 0
            for sf in srt_files:
                if translate_subtitles(
                    sf, args.to, api_key, args.translate_model, args.force
                ):
                    ok += 1
            print(f"\n{C_GREEN}Done!{C_RESET} {ok}/{len(srt_files)} translated.")
            sys.exit(0 if ok == len(srt_files) else 1)
        else:
            print(f"SRT file not found: {srt_target}", file=sys.stderr)
            sys.exit(1)

    if target.is_file():
        if target.suffix.lower() not in VIDEO_EXTENSIONS:
            print(f"{C_YELLOW}Warning: {target.suffix} is not a known video format{C_RESET}", file=sys.stderr)
        success = process_file(target, args.force, args.model, args.language,
                               args.only_whisper, compute_type=args.compute_type,
                               device=args.device,
                               os_api_key=os_api_key, os_token=os_token)
        sys.exit(0 if success else 1)

    elif target.is_dir():
        video_files = sorted(
            f for f in target.rglob("*")
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
                            args.device, os_api_key, os_token):
                ok += 1
            else:
                fail += 1

        print(f"\n{C_GREEN}Done!{C_RESET} {ok} succeeded, {f'{C_RED}{fail}{C_RESET}' if fail else '0'} failed out of {len(video_files)} files.")
        sys.exit(0 if fail == 0 else 1)

    else:
        print(f"Path not found: {target}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
