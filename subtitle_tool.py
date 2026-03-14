#!/usr/bin/env python3
"""Extrahera, transkribera eller synkronisera undertexter fran videofiler.

Kollar forst om videofilen har inbaddade SRT-undertexter och extraherar dem.
Om inga undertexter finns anvands faster-whisper for transkribering.
Med --sync kan befintliga SRT-filer synkroniseras mot videons ljud via ffsubsync.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".webm", ".mov", ".ts"}
SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".sub"}


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


def transcribe_with_whisper(video_path: Path, output_path: Path,
                            model_name: str, language: str | None) -> bool:
    """Transkribera ljud med faster-whisper och spara som SRT."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Fel: faster-whisper ar inte installerat.", file=sys.stderr)
        print("Kor: pip install faster-whisper", file=sys.stderr)
        return False

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

    print(f"  Laddar Whisper-modell '{model_name}'...")
    model = WhisperModel(model_name, device="cuda", compute_type="float16")

    print(f"  Transkriberar...")
    kwargs = {}
    if language:
        kwargs["language"] = language

    segments, info = model.transcribe(wav_path, **kwargs)
    detected_lang = info.language
    print(f"  Detekterat sprak: {detected_lang} (sannolikhet: {info.language_probability:.1%})")

    # Generera SRT
    srt_lines = []
    for i, segment in enumerate(segments, start=1):
        start_ts = format_srt_timestamp(segment.start)
        end_ts = format_srt_timestamp(segment.end)
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_ts} --> {end_ts}")
        srt_lines.append(segment.text.strip())
        srt_lines.append("")

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


def process_file(video_path: Path, force: bool, model: str,
                 language: str | None) -> bool:
    """Behandla en videofil - extrahera eller transkribera undertexter."""
    output_path = video_path.with_suffix("").with_suffix(".en.srt")

    # Hantera dubbla suffix korrekt (.en.srt)
    # Path.with_suffix tar bort sista suffixet, sa vi bygger manuellt
    output_path = video_path.parent / (video_path.stem + ".en.srt")

    if output_path.exists() and not force:
        print(f"  Finns redan: {output_path.name} (anvand --force for att skriva over)")
        return True

    print(f"\nBehandlar: {video_path.name}")

    # Kolla efter inbaddade undertexter
    streams = ffprobe_subtitle_streams(video_path)
    if streams:
        codec = streams[0].get("codec_name", "okant")
        lang = streams[0].get("tags", {}).get("language", "okant")
        print(f"  Hittade inbaddad undertext: {codec} ({lang})")
        print(f"  Extraherar till: {output_path.name}")
        return extract_subtitles(video_path, output_path)
    else:
        print(f"  Inga inbaddade undertexter - startar Whisper-transkribering")
        print(f"  Sparar till: {output_path.name}")
        return transcribe_with_whisper(video_path, output_path, model, language)


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
    args = parser.parse_args()

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

    if target.is_file():
        if target.suffix.lower() not in VIDEO_EXTENSIONS:
            print(f"Varning: {target.suffix} ar inte en kand videofiltyp", file=sys.stderr)
        success = process_file(target, args.force, args.model, args.language)
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
        ok = 0
        fail = 0
        for vf in video_files:
            if process_file(vf, args.force, args.model, args.language):
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
