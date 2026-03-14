#!/usr/bin/env python3
"""Extrahera, transkribera, oversatt eller synkronisera undertexter fran videofiler.

Kollar forst om videofilen har inbaddade SRT-undertexter och extraherar dem.
Om inga undertexter finns anvands faster-whisper for transkribering.
Med --sync kan befintliga SRT-filer synkroniseras mot videons ljud via ffsubsync.
Med --translate-subs kan SRT-filer oversattas via Claude API.
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

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".webm", ".mov", ".ts"}
SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".sub"}


def check_dependencies(needs_ffmpeg=True, needs_whisper=False, needs_sync=False):
    """Kontrollera att verktyg och bibliotek finns. Avslutar med instruktioner om nagot saknas."""
    missing = []

    if needs_ffmpeg:
        if not shutil.which("ffmpeg"):
            missing.append(("ffmpeg", "Installera via pakethanteraren (apt/pacman/brew/choco)"))
        if not shutil.which("ffprobe"):
            missing.append(("ffprobe", "Installeras tillsammans med ffmpeg"))

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
        print("Saknade beroenden:", file=sys.stderr)
        for name, fix in missing:
            print(f"  - {name}: {fix}", file=sys.stderr)
        sys.exit(1)


def sync_subtitles(video_path: Path, srt_path: Path, output_path: Path | None) -> bool:
    """Synkronisera en SRT-fil mot videons ljud med ffsubsync."""
    if output_path is None:
        output_path = srt_path.with_suffix(".synced.srt")

    print(f"  Synkroniserar: {srt_path.name}")
    print(f"  Mot video: {video_path.name}")
    print(f"  Sparar till: {output_path.name}")

    cmd = [
        sys.executable, "-m", "ffsubsync",
        str(video_path),
        "-i", str(srt_path),
        "-o", str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Fel vid synkronisering: {result.stderr.strip()}", file=sys.stderr)
        return False

    # Visa offset om ffsubsync rapporterar det
    for line in result.stderr.splitlines():
        if "offset" in line.lower() or "framerate" in line.lower():
            print(f"  {line.strip()}")

    print(f"  Synkronisering klar!")
    return True


def ffprobe_subtitle_streams(video_path: Path) -> list[dict]:
    """Returnerar lista med undertextstroemmar i videofilen."""
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
    """Extrahera forsta SRT-stroemmen fran videofilen."""
    cmd = [
        "ffmpeg", "-y", "-v", "warning",
        "-i", str(video_path),
        "-map", "0:s:0",
        "-f", "srt",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Fel vid extrahering: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def _load_whisper_model(whisper_cls, model_name: str, compute_type: str = "int8"):
    """Ladda Whisper-modell med informativ progress."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_cached = (cache_dir / f"models--Systran--faster-whisper-{model_name}").exists()

    if model_cached:
        print(f"  Laddar Whisper-modell '{model_name}' ({compute_type})...")
    else:
        print(f"  Laddar ner Whisper-modell '{model_name}' "
              f"(forsta gangen, kan ta flera minuter)...")

    model = whisper_cls(model_name, device="cuda", compute_type=compute_type)
    print(f"  Modell redo.")
    return model


def transcribe_with_whisper(video_path: Path, output_path: Path | None,
                            model_name: str, language: str | None,
                            whisper_model=None, force: bool = False,
                            compute_type: str = "int8") -> bool:
    """Transkribera ljud med faster-whisper och spara som SRT.

    Om output_path ar None bestams filnamnet fran detekterat sprak.
    Om whisper_model ar None laddas modellen automatiskt.
    """
    if whisper_model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("Fel: faster-whisper ar inte installerat.", file=sys.stderr)
            print("Kor: pip install faster-whisper", file=sys.stderr)
            return False
        whisper_model = _load_whisper_model(WhisperModel, model_name, compute_type)

    # Extrahera ljud till temporar WAV-fil
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    print(f"  Extraherar ljud...")
    cmd = [
        "ffmpeg", "-y", "-v", "warning",
        "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Fel vid ljudextrahering: {result.stderr.strip()}", file=sys.stderr)
        Path(wav_path).unlink(missing_ok=True)
        return False

    print(f"  Transkriberar...")
    kwargs = {}
    if language:
        kwargs["language"] = language

    try:
        segments, info = whisper_model.transcribe(wav_path, **kwargs)
        detected_lang = info.language
        print(f"  Detekterat sprak: {detected_lang} (sannolikhet: {info.language_probability:.1%})")

        # Bestam output-sokvag fran detekterat sprak om inte angiven
        if output_path is None:
            output_path = video_path.parent / (video_path.stem + f".{detected_lang}.srt")
            if output_path.exists() and not force:
                print(f"  Finns redan: {output_path.name}")
                Path(wav_path).unlink(missing_ok=True)
                return True

        print(f"  Sparar till: {output_path.name}")

        # Generera SRT
        srt_lines = []
        for i, segment in enumerate(segments, start=1):
            start_ts = format_srt_timestamp(segment.start)
            end_ts = format_srt_timestamp(segment.end)
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_ts} --> {end_ts}")
            srt_lines.append(segment.text.strip())
            srt_lines.append("")

    except RuntimeError as e:
        err = str(e)
        Path(wav_path).unlink(missing_ok=True)
        if "not found or cannot be loaded" in err:
            print(f"\n  Fel: CUDA-bibliotek saknas: {e}", file=sys.stderr)
            print(f"  Kor: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12", file=sys.stderr)
            return False
        if "out of memory" in err.lower():
            print(f"\n  Fel: GPU-minnet racker inte till.", file=sys.stderr)
            print(f"  Forslag:", file=sys.stderr)
            print(f"    --compute-type int8    (halverar minnesanvandning)", file=sys.stderr)
            print(f"    -m medium              (mindre modell)", file=sys.stderr)
            print(f"    -m small               (minsta brukbara modell)", file=sys.stderr)
            return False
        raise

    output_path.write_text("\n".join(srt_lines), encoding="utf-8")

    # Stadning
    Path(wav_path).unlink(missing_ok=True)
    return True


def format_srt_timestamp(seconds: float) -> str:
    """Konvertera sekunder till SRT-tidsstampel (HH:MM:SS,mmm)."""
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
    """Parsa SRT-fil till lista av {index, start, end, text}."""
    content = srt_path.read_text(encoding="utf-8-sig")  # Hanterar BOM
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
    """Skriv segment-lista som SRT-fil."""
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{seg['start']} --> {seg['end']}")
        lines.append(seg["text"])
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def opensubtitles_hash(file_path: Path) -> str:
    """Berakna OpenSubtitles-hash for en videofil."""
    block_size = 65536
    file_size = file_path.stat().st_size

    if file_size < block_size * 2:
        raise ValueError(f"Filen ar for liten for hash-berakning ({file_size} bytes)")

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
    """Intern: sok OpenSubtitles API."""
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
        print(f"  OpenSubtitles API-fel: {e.code} {e.reason}", file=sys.stderr)
        return []
    except urllib.error.URLError as e:
        print(f"  Natverksfel: {e.reason}", file=sys.stderr)
        return []


def _opensubtitles_download(file_id: int, output_path: Path, api_key: str) -> bool:
    """Intern: ladda hem en undertextfil fran OpenSubtitles.com."""
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
        print(f"  Nedladdningsfel: {e.code} {e.reason}", file=sys.stderr)
        return False

    dl_link = dl_data.get("link")
    if not dl_link:
        print("  Ingen nedladdningslink i API-svaret", file=sys.stderr)
        return False

    try:
        urllib.request.urlretrieve(dl_link, str(output_path))
        return True
    except Exception as e:
        print(f"  Fel vid nedladdning: {e}", file=sys.stderr)
        return False


def fetch_opensubtitles(video_path: Path, api_key: str, force: bool = False) -> bool:
    """Sok och ladda hem svenska och engelska undertexter fran OpenSubtitles.com."""
    print(f"\nOpenSubtitles: {video_path.name}")

    try:
        file_hash = opensubtitles_hash(video_path)
    except ValueError as e:
        print(f"  {e}", file=sys.stderr)
        return False
    print(f"  Hash: {file_hash}")

    languages = {"en": "engelska", "sv": "svenska"}
    any_downloaded = False

    for lang, lang_name in languages.items():
        output_path = video_path.parent / (video_path.stem + f".{lang}.srt")

        if output_path.exists() and not force:
            print(f"  {lang_name.capitalize()} finns redan: {output_path.name}")
            continue

        # Sok med hash
        results = _opensubtitles_search(api_key, moviehash=file_hash, languages=lang)
        hash_matched = [r for r in results if r["attributes"].get("moviehash_match")]
        non_hash = [r for r in results if not r["attributes"].get("moviehash_match")]

        # Fallback: sok pa filnamn om inga resultat
        if not results:
            query = video_path.stem.replace(".", " ").replace("_", " ").replace("-", " ")
            results = _opensubtitles_search(api_key, query=query, languages=lang)
            non_hash = results

        if hash_matched:
            best = hash_matched[0]
            attrs = best["attributes"]
            file_id = attrs["files"][0]["file_id"]
            release = attrs.get("release", "okand")
            print(f"  {lang_name.capitalize()} (hash OK): {release}")
            if _opensubtitles_download(file_id, output_path, api_key):
                print(f"  Sparad: {output_path.name}")
                any_downloaded = True
        elif non_hash:
            best = non_hash[0]
            attrs = best["attributes"]
            file_id = attrs["files"][0]["file_id"]
            release = attrs.get("release", "okand")
            print(f"\n  VARNING: {lang_name.capitalize()} undertext hittad men hash stammer INTE!")
            print(f"  Release: {release}")
            print(f"  Undertexten kanske inte matchar din videofil exakt.")
            try:
                answer = input(f"  Ladda hem anda? (j/n): ").strip().lower()
            except EOFError:
                answer = "n"
            if answer in ("j", "ja", "y", "yes"):
                if _opensubtitles_download(file_id, output_path, api_key):
                    print(f"  Sparad: {output_path.name}")
                    any_downloaded = True
            else:
                print(f"  Hoppar over {lang_name} undertext.")
        else:
            print(f"  Ingen {lang_name} undertext hittad")

    return any_downloaded


def translate_batch_claude(texts: list[str], source_lang: str, target_lang: str,
                           api_key: str, model: str) -> list[str]:
    """Oversatt en batch undertextrader via Claude API."""
    source_name = LANG_NAMES.get(source_lang, source_lang)
    target_name = LANG_NAMES.get(target_lang, target_lang)

    # Bygg numrerad input (join multi-line entries med mellanslag)
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
        print(f"  Claude API-fel: {e.code} {e.reason}", file=sys.stderr)
        if error_body:
            print(f"  {error_body[:200]}", file=sys.stderr)
        raise
    except urllib.error.URLError as e:
        print(f"  Natverksfel: {e.reason}", file=sys.stderr)
        raise

    response_text = data["content"][0]["text"]

    # Parsa numrerat svar
    result = []
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Ta bort "N: " prefix
        colon_pos = line.find(": ")
        if colon_pos != -1 and colon_pos < 6 and line[:colon_pos].isdigit():
            result.append(line[colon_pos + 2:])
        else:
            result.append(line)

    return result


def _translate_output_path(srt_path: Path, target_lang: str) -> Path:
    """Bestam output-filnamn for oversattning."""
    stem = srt_path.stem
    # Kolla om stem slutar med .xx (sprakkod)
    parts = stem.rsplit(".", 1)
    if len(parts) == 2 and len(parts[1]) == 2 and parts[1].isalpha():
        return srt_path.parent / f"{parts[0]}.{target_lang}.srt"
    return srt_path.parent / f"{stem}.{target_lang}.srt"


def translate_subtitles(srt_path: Path, target_lang: str, api_key: str,
                        model: str, force: bool) -> bool:
    """Oversatt en SRT-fil till annat sprak via Claude API."""
    output_path = _translate_output_path(srt_path, target_lang)

    if output_path.exists() and not force:
        print(f"  Finns redan: {output_path.name} (anvand --force for att skriva over)")
        return True

    print(f"\nOversatter: {srt_path.name} -> {output_path.name}")

    segments = parse_srt(srt_path)
    if not segments:
        print("  Inga undertexter att oversatta", file=sys.stderr)
        return False

    # Detektera kallsprak fran filnamn
    stem_parts = srt_path.stem.rsplit(".", 1)
    if len(stem_parts) == 2 and len(stem_parts[1]) == 2 and stem_parts[1].isalpha():
        source_lang = stem_parts[1]
    else:
        source_lang = "en"

    target_name = LANG_NAMES.get(target_lang, target_lang)
    print(f"  Sprak: {source_lang} -> {target_lang} ({target_name})")
    print(f"  Modell: {model}")
    print(f"  Antal undertexter: {len(segments)}")

    # Batcha och oversatt
    batch_size = 40
    translated_texts = []
    total = len(segments)
    total_batches = (total + batch_size - 1) // batch_size

    for i in range(0, total, batch_size):
        batch = [s["text"] for s in segments[i:i + batch_size]]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} rader)...")

        try:
            result = translate_batch_claude(
                batch, source_lang, target_lang, api_key, model
            )
        except Exception:
            print(f"  Avbryter oversattning.", file=sys.stderr)
            return False

        # Validera antal rader
        if len(result) != len(batch):
            print(f"  Varning: fick {len(result)} rader, forvantade {len(batch)}",
                  file=sys.stderr)
            while len(result) < len(batch):
                result.append(batch[len(result)])
            result = result[:len(batch)]

        translated_texts.extend(result)

    # Bygg output-segment med originaltidsstamplar
    output_segments = []
    for seg, text in zip(segments, translated_texts):
        output_segments.append({
            "index": seg["index"],
            "start": seg["start"],
            "end": seg["end"],
            "text": text,
        })

    write_srt(output_segments, output_path)
    print(f"  Klar! Sparad: {output_path.name} ({total} rader oversatta)")
    return True


def process_file(video_path: Path, force: bool, model: str,
                 language: str | None, only_whisper: bool = False,
                 whisper_model=None, compute_type: str = "int8") -> bool:
    """Behandla en videofil - extrahera eller transkribera undertexter."""
    print(f"\nBehandlar: {video_path.name}")

    if not only_whisper:
        # Kolla efter inbaddade undertexter
        streams = ffprobe_subtitle_streams(video_path)
        if streams:
            codec = streams[0].get("codec_name", "okant")
            lang = streams[0].get("tags", {}).get("language", "und")
            output_path = video_path.parent / (video_path.stem + f".{lang}.srt")
            if output_path.exists() and not force:
                print(f"  Finns redan: {output_path.name}")
                return True
            print(f"  Hittade inbaddad undertext: {codec} ({lang})")
            print(f"  Extraherar till: {output_path.name}")
            return extract_subtitles(video_path, output_path)
        print(f"  Inga inbaddade undertexter - startar Whisper-transkribering")
    else:
        print(f"  Startar Whisper-transkribering (--only-whisper)")

    # Om sprak angivet: bestam output-sokvag direkt, annars auto-detect
    if language:
        output_path = video_path.parent / (video_path.stem + f".{language}.srt")
        if output_path.exists() and not force:
            print(f"  Finns redan: {output_path.name}")
            return True
    else:
        output_path = None  # Bestams av detekterat sprak

    return transcribe_with_whisper(video_path, output_path, model, language,
                                   whisper_model, force, compute_type)


def main():
    parser = argparse.ArgumentParser(
        description="Extrahera eller transkribera undertexter fran videofiler"
    )
    parser.add_argument(
        "path",
        help="Sokvag till videofil eller katalog (batch-lage)"
    )
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Whisper-sprak (default: auto-detect)"
    )
    parser.add_argument(
        "-m", "--model",
        default="large-v3",
        help="Whisper-modell (default: large-v3)"
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Whisper-berakningstyp: int8, float16, float32 (default: int8)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skriv over befintliga .en.srt-filer"
    )
    parser.add_argument(
        "--sync",
        metavar="SRT",
        help="Synka en SRT-fil mot videons ljud (krav: video_path + --sync sub.srt)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output-fil for --sync (default: <input>.synced.srt)"
    )
    parser.add_argument(
        "--only-whisper",
        action="store_true",
        help="Hoppa over inbaddade undertexter, anvand alltid Whisper"
    )
    parser.add_argument(
        "--opensubtitles",
        action="store_true",
        help="Ladda hem svenska och engelska undertexter fran OpenSubtitles.com"
    )
    parser.add_argument(
        "--translate-subs",
        metavar="SRT",
        help="Oversatt en SRT-fil till annat sprak via Claude API (krav: --to)"
    )
    parser.add_argument(
        "--to",
        default=None,
        help="Malsprak for --translate-subs (t.ex. sv, en, de, fr)"
    )
    parser.add_argument(
        "--translate-model",
        default="claude-haiku-4-5-20251001",
        help="Claude-modell for oversattning (default: claude-haiku-4-5-20251001)"
    )
    args = parser.parse_args()

    # Kontrollera beroenden baserat pa lage
    if args.sync:
        check_dependencies(needs_ffmpeg=True, needs_sync=True)
    elif not args.opensubtitles and not args.translate_subs:
        check_dependencies(needs_ffmpeg=True, needs_whisper=args.only_whisper)

    target = Path(args.path)

    # Sync-lage: synka en SRT mot en video
    if args.sync:
        srt_path = Path(args.sync)
        if not target.is_file():
            print(f"Videofil finns inte: {target}", file=sys.stderr)
            sys.exit(1)
        if not srt_path.is_file():
            print(f"SRT-fil finns inte: {srt_path}", file=sys.stderr)
            sys.exit(1)
        out = Path(args.output) if args.output else None
        success = sync_subtitles(target, srt_path, out)
        sys.exit(0 if success else 1)

    # OpenSubtitles-lage: ladda hem undertexter
    if args.opensubtitles:
        api_key = os.environ.get("OPENSUBTITLES_API_KEY")
        if not api_key:
            print("Fel: OPENSUBTITLES_API_KEY ar inte satt.", file=sys.stderr)
            print("Se README for instruktioner.", file=sys.stderr)
            sys.exit(1)

        if target.is_file():
            if target.suffix.lower() not in VIDEO_EXTENSIONS:
                print(f"Varning: {target.suffix} ar inte en kand videofiltyp", file=sys.stderr)
            success = fetch_opensubtitles(target, api_key, args.force)
            sys.exit(0 if success else 1)
        elif target.is_dir():
            video_files = sorted(
                f for f in target.iterdir()
                if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
            )
            if not video_files:
                print(f"Inga videofiler hittades i: {target}")
                sys.exit(1)
            print(f"Hittade {len(video_files)} videofiler i: {target}")
            for vf in video_files:
                fetch_opensubtitles(vf, api_key, args.force)
            sys.exit(0)
        else:
            print(f"Sokvagen finns inte: {target}", file=sys.stderr)
            sys.exit(1)

    # Translate-lage: oversatt SRT via Claude API
    if args.translate_subs:
        if not args.to:
            print("Fel: --translate-subs kraver --to (malsprak)", file=sys.stderr)
            print("Exempel: --translate-subs film.en.srt --to sv", file=sys.stderr)
            sys.exit(1)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Fel: ANTHROPIC_API_KEY ar inte satt.", file=sys.stderr)
            print("Se README for instruktioner.", file=sys.stderr)
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
                print(f"Inga SRT-filer hittades i: {srt_target}")
                sys.exit(1)
            print(f"Hittade {len(srt_files)} SRT-filer i: {srt_target}")
            ok = 0
            for sf in srt_files:
                if translate_subtitles(
                    sf, args.to, api_key, args.translate_model, args.force
                ):
                    ok += 1
            print(f"\nKlart! {ok}/{len(srt_files)} oversatta.")
            sys.exit(0 if ok == len(srt_files) else 1)
        else:
            print(f"SRT-filen finns inte: {srt_target}", file=sys.stderr)
            sys.exit(1)

    if target.is_file():
        if target.suffix.lower() not in VIDEO_EXTENSIONS:
            print(f"Varning: {target.suffix} ar inte en kand videofiltyp", file=sys.stderr)
        success = process_file(target, args.force, args.model, args.language,
                               args.only_whisper, compute_type=args.compute_type)
        sys.exit(0 if success else 1)

    elif target.is_dir():
        video_files = sorted(
            f for f in target.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not video_files:
            print(f"Inga videofiler hittades i: {target}")
            sys.exit(1)

        print(f"Hittade {len(video_files)} videofiler i: {target}")

        # Ladda Whisper-modell en gang for batch-lage
        whisper_model = None
        try:
            from faster_whisper import WhisperModel
            whisper_model = _load_whisper_model(WhisperModel, args.model, args.compute_type)
        except ImportError:
            pass  # Hanteras per fil om Whisper behovs

        ok = 0
        fail = 0
        for vf in video_files:
            if process_file(vf, args.force, args.model, args.language,
                            args.only_whisper, whisper_model, args.compute_type):
                ok += 1
            else:
                fail += 1

        print(f"\nKlart! {ok} lyckades, {fail} misslyckades av {len(video_files)} filer.")
        sys.exit(0 if fail == 0 else 1)

    else:
        print(f"Sokvagen finns inte: {target}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
