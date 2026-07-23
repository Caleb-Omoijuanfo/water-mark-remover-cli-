# VideoClean

CLI tool for removing watermarks from videos using local inpainting models.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (dependency management)
- FFmpeg (`ffmpeg` and `ffprobe` on `PATH`)

## Setup

```bash
# Install system dependencies (macOS)
brew install ffmpeg

# Create venv and install package + deps
uv sync

# Optional: install dev/test extras
uv sync --extra dev
```

## Run The CLI

Use either of these forms:

```bash
# Preferred: run via uv in the project environment
uv run videoclean --help

# Alternative
python -m videoclean --help
```

## Command Overview

- `info`: show video metadata (resolution, fps, duration, codecs)
- `preview`: generate an image preview of the selected watermark region
- `remove`: remove the selected region and write a cleaned video
- `version`: print CLI version

## Typical Workflow

1. Inspect the source video

```bash
uv run videoclean info input.mp4
```

2. Pick and preview a region (`x1,y1,x2,y2`)

```bash
uv run videoclean preview input.mp4 \
	--region 20,20,300,120 \
	--output preview.png \
	--mask-padding 2
```

3. Run watermark removal

```bash
uv run videoclean remove input.mp4 \
	--region 20,20,300,120 \
	--output cleaned.mp4
```

## Important Options

`remove` options:

- `--output, -o`: output video path (default is `<input>_cleaned.<ext>`)
- `--region, -r`: required region in `x1,y1,x2,y2`
- `--mask-padding`: expand the region by N pixels on each side
- `--model, -m`: inpainting backend (default: `opencv`)
- `--device`: `auto`, `cpu`, `cuda`, or `mps`
- `--crf`: output quality from `0..51` (lower is higher quality, default `18`)
- `--keep-temp`: keep temp workspace for debugging

`preview` options:

- `--frame`: preview a specific frame index (default `0`)
- `--time`: preview by timestamp in seconds (overrides `--frame`)
- `--save-mask`: also save a binary mask image next to preview output

## Notes

- Regions are validated against video bounds; invalid regions return a readable
  error and a non-zero exit code.
- FFmpeg (`ffmpeg` and `ffprobe`) must be installed and available on `PATH`.
- The tool preserves video resolution and attempts to preserve duration/audio.

## Example Commands

```bash
videoclean info input.mp4
videoclean preview input.mp4 --region 20,20,300,120 --output preview.png
videoclean remove input.mp4 --output cleaned.mp4 --region 20,20,300,120
```

See `Implementation.md` for the phased build plan.
