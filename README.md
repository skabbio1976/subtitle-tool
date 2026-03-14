# subtitle-tool

Extract, transcribe, translate, or download subtitles for video files.

## Features

- **Extract** embedded SRT subtitles from video files (via ffmpeg)
- **Transcribe** audio to subtitles with faster-whisper (GPU-accelerated)
- **Translate** SRT files between languages via Claude API
- **Download** Swedish and English subtitles from OpenSubtitles.com
- **Sync** existing SRT files to video audio (via ffsubsync)
- **Batch mode** - process all video/SRT files in a directory

## Requirements

- Python 3.10+
- ffmpeg and ffprobe
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (for transcription)
- [ffsubsync](https://github.com/smacke/ffsubsync) (for sync)
- CUDA-compatible GPU (for Whisper)

```
pip install -r requirements.txt
```

## API keys

Some features require API keys, which are read from environment variables.
**Never put API keys directly in commands or scripts.**

### Anthropic (Claude) - for `--translate-subs`

1. Create an account at [console.anthropic.com](https://console.anthropic.com/)
2. Go to **API Keys** and create a new key
3. Set the environment variable:

```bash
# bash / zsh (~/.bashrc or ~/.zshrc)
export ANTHROPIC_API_KEY="sk-ant-..."

# fish (~/.config/fish/config.fish)
set -Ux ANTHROPIC_API_KEY "sk-ant-..."

# Windows (PowerShell, permanent for current user)
[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-...", "User")
```

### Hugging Face (optional) - for faster Whisper model downloads

Whisper models are downloaded from Hugging Face Hub on first run. Setting a
token removes rate limits and speeds up downloads. Without a token you may see
a warning — this is harmless, downloads still work.

1. Create an account at [huggingface.co](https://huggingface.co/)
2. Go to **Settings > Access Tokens** and create a token
3. Set the environment variable:

```bash
# bash / zsh (~/.bashrc or ~/.zshrc)
export HF_TOKEN="hf_..."

# fish (~/.config/fish/config.fish)
set -Ux HF_TOKEN "hf_..."

# Windows (PowerShell, permanent for current user)
[Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_...", "User")
```

### OpenSubtitles - for `--opensubtitles`

1. Create an account at [opensubtitles.com](https://www.opensubtitles.com/)
2. Go to [API consumers](https://www.opensubtitles.com/en/consumers) and register an app
3. Set the environment variable:

```bash
# bash / zsh (~/.bashrc or ~/.zshrc)
export OPENSUBTITLES_API_KEY="your-key-here"

# fish (~/.config/fish/config.fish)
set -Ux OPENSUBTITLES_API_KEY "your-key-here"

# Windows (PowerShell, permanent for current user)
[Environment]::SetEnvironmentVariable("OPENSUBTITLES_API_KEY", "your-key-here", "User")
```

## Usage

### Extract or transcribe

```bash
# Single file - extracts embedded subs, otherwise uses Whisper
python subtitle_tool.py video.mkv

# All video files in a directory
python subtitle_tool.py /path/to/videos/

# Force Whisper (skip embedded subtitle check)
python subtitle_tool.py --only-whisper video.mkv

# Specify language and model
python subtitle_tool.py -l sv -m large-v3 video.mkv

# Overwrite existing .en.srt files
python subtitle_tool.py --force video.mkv
```

### Translate subtitles

Translate SRT files between languages using the Claude API.
Requires `ANTHROPIC_API_KEY` (see above).

```bash
# Translate English SRT to Swedish (uses Claude Haiku by default)
python subtitle_tool.py --translate-subs movie.en.srt --to sv .

# Use a different Claude model
python subtitle_tool.py --translate-subs movie.en.srt --to sv --translate-model claude-sonnet-4-6 .

# Batch - translate all SRT files in a directory
python subtitle_tool.py --translate-subs /path/to/subs/ --to sv .
```

Output filename is determined automatically: `movie.en.srt` becomes `movie.sv.srt`.
The source language is detected from the filename (e.g., `.en.srt` = English).

### Download from OpenSubtitles.com

Downloads both Swedish (.sv.srt) and English (.en.srt) subtitles.
Requires `OPENSUBTITLES_API_KEY` (see above).

```bash
# Single file
python subtitle_tool.py --opensubtitles video.mkv

# Batch - all files in a directory
python subtitle_tool.py --opensubtitles /path/to/videos/
```

If the file hash matches on OpenSubtitles, subtitles are downloaded automatically.
If the hash does not match, a warning is shown and you can choose to download or skip.

### Sync subtitles

```bash
# Sync an SRT file to the video's audio
python subtitle_tool.py video.mkv --sync subtitle.srt

# Specify output file
python subtitle_tool.py video.mkv --sync subtitle.srt -o synced.srt
```

## Building a standalone binary

If you don't want to install Python and dependencies globally, you can build a
single executable with PyInstaller:

```bash
# Create a venv and install everything
python -m venv .venv
source .venv/bin/activate        # bash/zsh
# source .venv/bin/activate.fish  # fish
# .venv\Scripts\activate          # Windows

pip install faster-whisper ffsubsync pyinstaller

# Build single-file binary
pyinstaller --onefile subtitle_tool.py

# The binary is now in dist/
./dist/subtitle_tool --help
```

> **Note:** The binary will be large (~300-500 MB) because it bundles the Python
> runtime and all dependencies including CUDA/cuDNN libraries from faster-whisper.
> Whisper model files are **not** included — they are downloaded on first run.
> ffmpeg/ffprobe must still be installed separately on the system.

## Supported formats

Video: .mkv, .mp4, .avi, .webm, .mov, .ts
