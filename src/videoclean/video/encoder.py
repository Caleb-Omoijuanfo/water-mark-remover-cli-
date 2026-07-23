"""Frame reassembly, audio extraction, and final mux via FFmpeg."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from videoclean.exceptions import FFmpegError, InvalidVideoError
from videoclean.video.extractor import list_frames
from videoclean.video.ffmpeg import run_ffmpeg
from videoclean.video.metadata import VideoMetadata, probe

log = logging.getLogger(__name__)


def encode_frames(
    frames_dir: Path | str,
    output_path: Path | str,
    *,
    fps: float,
    image_ext: str = "png",
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """Reassemble frame images into an H.264 (libx264) video without audio.

    Parameters
    ----------
    frames_dir:
        Directory containing ``frame_%06d.<ext>`` images.
    output_path:
        Destination video path (parent dirs created as needed).
    fps:
        Frame rate for the output stream.
    image_ext:
        Frame image extension (must match extracted frames).
    crf:
        libx264 constant rate factor (lower = higher quality).
    preset:
        libx264 encoding preset.
    """
    source_dir = Path(frames_dir)
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if fps <= 0:
        raise FFmpegError(f"fps must be positive, got {fps}")

    frames = list_frames(source_dir, image_ext=image_ext)
    if not frames:
        raise FFmpegError(f"No frames found in: {source_dir}")

    ext = "jpg" if image_ext == "jpeg" else image_ext
    pattern = source_dir / f"frame_%06d.{ext}"

    args = [
        "-framerate",
        f"{fps:.6f}".rstrip("0").rstrip("."),
        "-i",
        str(pattern),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    log.info(
        "Encoding %d frame(s) @ %.3f fps → %s",
        len(frames),
        fps,
        dest,
    )
    run_ffmpeg(args)

    if not dest.is_file() or dest.stat().st_size == 0:
        raise FFmpegError(f"Encoder produced no output at: {dest}")

    return dest


def extract_audio(
    video_path: Path | str,
    audio_path: Path | str,
    *,
    metadata: VideoMetadata | None = None,
) -> Path | None:
    """Extract the audio stream from *video_path*.

    Returns
    -------
    Path | None
        Path to the extracted audio file, or ``None`` if the source has no audio.
    """
    source = Path(video_path).expanduser().resolve()
    dest = Path(audio_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    meta = metadata or probe(source)
    if not meta.has_audio:
        log.info("No audio stream in %s; skipping audio extraction", source.name)
        return None

    # Prefer AAC in M4A for broad mux compatibility.
    args = [
        "-i",
        str(source),
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(dest),
    ]
    log.info("Extracting audio from %s → %s", source.name, dest)
    run_ffmpeg(args)

    if not dest.is_file() or dest.stat().st_size == 0:
        raise FFmpegError(f"Audio extraction produced no output at: {dest}")

    return dest


def mux_video_audio(
    video_path: Path | str,
    output_path: Path | str,
    *,
    audio_path: Path | str | None = None,
) -> Path:
    """Mux a video-only stream with optional audio into the final container.

    If *audio_path* is ``None`` or missing, the video is copied (re-muxed) to
    *output_path* without an audio track.
    """
    video = Path(video_path)
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not video.is_file():
        raise InvalidVideoError(f"Video file not found: {video}")

    audio: Path | None = Path(audio_path) if audio_path is not None else None
    if audio is not None and not audio.is_file():
        log.warning("Audio file missing (%s); muxing video only", audio)
        audio = None

    if audio is None:
        if video.resolve() == dest.resolve():
            return dest
        shutil.copy2(video, dest)
        log.info("Wrote video-only output → %s", dest)
        return dest

    # Avoid -shortest: re-encoded AAC is often a few frames shorter than the
    # video, and -shortest would truncate the picture stream (duration drift).
    # Without it, FFmpeg uses the longest stream; we then cap with -t when the
    # caller needs an exact duration (pipeline verifies roughly).
    args = [
        "-i",
        str(video),
        "-i",
        str(audio),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        str(dest),
    ]
    log.info("Muxing video + audio → %s", dest)
    run_ffmpeg(args)

    if not dest.is_file() or dest.stat().st_size == 0:
        raise FFmpegError(f"Mux produced no output at: {dest}")

    return dest


def assemble_video(
    frames_dir: Path | str,
    output_path: Path | str,
    *,
    fps: float,
    source_video: Path | str | None = None,
    audio_path: Path | str | None = None,
    video_only_path: Path | str | None = None,
    image_ext: str = "png",
    crf: int = 18,
) -> Path:
    """Encode frames and mux audio into the final output video.

    If *audio_path* is not provided but *source_video* is, audio is extracted
    from the source first.
    """
    dest = Path(output_path)
    intermediate = (
        Path(video_only_path)
        if video_only_path is not None
        else dest.with_suffix(".video_only.mp4")
    )

    encode_frames(
        frames_dir,
        intermediate,
        fps=fps,
        image_ext=image_ext,
        crf=crf,
    )

    resolved_audio: Path | None
    if audio_path is not None:
        resolved_audio = Path(audio_path)
    elif source_video is not None:
        # Extract beside intermediate if caller did not supply audio.
        side_audio = intermediate.with_suffix(".m4a")
        resolved_audio = extract_audio(source_video, side_audio)
    else:
        resolved_audio = None

    try:
        return mux_video_audio(intermediate, dest, audio_path=resolved_audio)
    finally:
        # Drop intermediate when it is a sibling helper file we created.
        if video_only_path is None and intermediate.exists() and intermediate != dest:
            intermediate.unlink(missing_ok=True)
