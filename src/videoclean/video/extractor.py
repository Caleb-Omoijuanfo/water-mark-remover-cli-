"""Frame extraction from video files via FFmpeg."""

from __future__ import annotations

import logging
from pathlib import Path

from videoclean.exceptions import FFmpegError, InvalidVideoError
from videoclean.video.ffmpeg import run_ffmpeg
from videoclean.video.metadata import VideoMetadata, probe

log = logging.getLogger(__name__)

# Zero-padded frame pattern used for extract and encode symmetry.
FRAME_PATTERN = "frame_%06d.png"
FRAME_GLOB = "frame_*.png"


def extract_frames(
    video_path: Path | str,
    output_dir: Path | str,
    *,
    metadata: VideoMetadata | None = None,
    image_ext: str = "png",
) -> list[Path]:
    """Extract all frames from *video_path* into *output_dir*.

    Frames are written as ``frame_000001.png``, ``frame_000002.png``, …

    Parameters
    ----------
    video_path:
        Source video file.
    output_dir:
        Directory that will receive frame images (created if missing).
    metadata:
        Optional pre-probed metadata (avoids a second ffprobe call).
    image_ext:
        Image extension/format (``png`` or ``jpg``). PNG is lossless and
        preferred for the MVP pipeline.

    Returns
    -------
    list[Path]
        Sorted list of extracted frame paths.
    """
    source = Path(video_path).expanduser().resolve()
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if not source.is_file():
        raise InvalidVideoError(f"Video file not found: {source}")

    meta = metadata or probe(source)
    if image_ext not in {"png", "jpg", "jpeg"}:
        raise FFmpegError(f"Unsupported frame image format: {image_ext!r}")

    pattern = dest / f"frame_%06d.{ 'jpg' if image_ext == 'jpeg' else image_ext }"

    # -vsync 0 / -fps_mode passthrough: one output image per input frame.
    args = [
        "-i",
        str(source),
        "-fps_mode",
        "passthrough",
        str(pattern),
    ]
    log.info(
        "Extracting frames from %s (%s, %.3fs @ %.3f fps) → %s",
        source.name,
        meta.resolution,
        meta.duration,
        meta.fps,
        dest,
    )
    run_ffmpeg(args)

    frames = list_frames(dest, image_ext=image_ext)
    if not frames:
        raise FFmpegError(f"No frames were extracted from: {source}")

    log.info("Extracted %d frame(s)", len(frames))
    return frames


def list_frames(directory: Path | str, *, image_ext: str = "png") -> list[Path]:
    """Return sorted frame paths in *directory* matching the naming pattern."""
    root = Path(directory)
    ext = "jpg" if image_ext == "jpeg" else image_ext
    frames = sorted(root.glob(f"frame_*.{ext}"))
    return frames


def extract_single_frame(
    video_path: Path | str,
    output_path: Path | str,
    *,
    frame_index: int = 0,
    time_seconds: float | None = None,
    metadata: VideoMetadata | None = None,
) -> Path:
    """Extract a single frame from *video_path* to *output_path*.

    Parameters
    ----------
    video_path:
        Source video file.
    output_path:
        Destination image path (``.png`` recommended).
    frame_index:
        Zero-based frame index to extract. Ignored when *time_seconds* is set.
    time_seconds:
        Optional timestamp (seconds) to seek before grabbing one frame.
    metadata:
        Optional pre-probed metadata (used for bounds checks on frame_index).
    """
    source = Path(video_path).expanduser().resolve()
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not source.is_file():
        raise InvalidVideoError(f"Video file not found: {source}")

    if time_seconds is not None:
        if time_seconds < 0:
            raise FFmpegError(f"time_seconds must be >= 0, got {time_seconds}")
        args = [
            "-ss",
            f"{time_seconds:.6f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            str(dest),
        ]
    else:
        if frame_index < 0:
            raise FFmpegError(f"frame_index must be >= 0, got {frame_index}")
        meta = metadata or probe(source)
        if meta.frame_count is not None and frame_index >= meta.frame_count:
            raise FFmpegError(
                f"frame_index {frame_index} is out of range "
                f"(video has {meta.frame_count} frame(s))."
            )
        # select filter uses zero-based frame numbers.
        args = [
            "-i",
            str(source),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-frames:v",
            "1",
            str(dest),
        ]

    log.info(
        "Extracting frame from %s → %s (index=%s time=%s)",
        source.name,
        dest,
        frame_index if time_seconds is None else "n/a",
        time_seconds,
    )
    run_ffmpeg(args)

    if not dest.is_file() or dest.stat().st_size == 0:
        raise FFmpegError(f"Failed to extract frame from: {source}")

    return dest
