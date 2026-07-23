"""Video metadata extraction via ffprobe."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from videoclean.exceptions import FFmpegError, InvalidVideoError
from videoclean.video.ffmpeg import run_ffprobe

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Probe results for a single video file."""

    path: Path
    width: int
    height: int
    duration: float
    fps: float
    frame_count: int | None
    video_codec: str
    pixel_format: str | None
    has_audio: bool
    audio_codec: str | None
    audio_sample_rate: int | None
    audio_channels: int | None
    bit_rate: int | None
    format_name: str | None

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


def probe(path: Path | str) -> VideoMetadata:
    """Extract metadata from a video file using ffprobe JSON output.

    Raises
    ------
    InvalidVideoError
        If the file is missing, unreadable, or has no video stream.
    FFmpegError
        If ffprobe fails for another reason.
    """
    video_path = Path(path).expanduser().resolve()
    if not video_path.exists():
        raise InvalidVideoError(f"Video file not found: {video_path}")
    if not video_path.is_file():
        raise InvalidVideoError(f"Path is not a file: {video_path}")

    raw = _ffprobe_json(video_path)
    return parse_probe_data(raw, video_path)


def parse_probe_data(data: dict[str, Any], path: Path) -> VideoMetadata:
    """Build :class:`VideoMetadata` from ffprobe JSON (testable without I/O)."""
    streams = data.get("streams") or []
    fmt = data.get("format") or {}

    video_stream = _first_stream(streams, "video")
    if video_stream is None:
        raise InvalidVideoError(f"No video stream found in: {path}")

    audio_stream = _first_stream(streams, "audio")

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise InvalidVideoError(
            f"Invalid video dimensions ({width}x{height}) in: {path}"
        )

    fps = _parse_frame_rate(
        video_stream.get("avg_frame_rate")
        or video_stream.get("r_frame_rate")
        or "0/1"
    )
    if fps <= 0:
        raise InvalidVideoError(f"Could not determine frame rate for: {path}")

    duration = _parse_duration(video_stream, fmt)
    frame_count = _parse_frame_count(video_stream, duration, fps)

    bit_rate: int | None = None
    if fmt.get("bit_rate"):
        try:
            bit_rate = int(fmt["bit_rate"])
        except (TypeError, ValueError):
            bit_rate = None

    audio_codec = None
    audio_sample_rate = None
    audio_channels = None
    if audio_stream is not None:
        audio_codec = audio_stream.get("codec_name")
        if audio_stream.get("sample_rate"):
            try:
                audio_sample_rate = int(audio_stream["sample_rate"])
            except (TypeError, ValueError):
                audio_sample_rate = None
        if audio_stream.get("channels") is not None:
            try:
                audio_channels = int(audio_stream["channels"])
            except (TypeError, ValueError):
                audio_channels = None

    return VideoMetadata(
        path=path,
        width=width,
        height=height,
        duration=duration,
        fps=fps,
        frame_count=frame_count,
        video_codec=str(video_stream.get("codec_name") or "unknown"),
        pixel_format=video_stream.get("pix_fmt"),
        has_audio=audio_stream is not None,
        audio_codec=audio_codec,
        audio_sample_rate=audio_sample_rate,
        audio_channels=audio_channels,
        bit_rate=bit_rate,
        format_name=fmt.get("format_name"),
    )


def _ffprobe_json(path: Path) -> dict[str, Any]:
    result = run_ffprobe(
        [
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ]
    )
    stdout = (result.stdout or "").strip()
    if not stdout:
        raise InvalidVideoError(f"ffprobe returned empty output for: {path}")
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"ffprobe returned invalid JSON for: {path}") from exc
    if not isinstance(data, dict):
        raise FFmpegError(f"ffprobe JSON root is not an object for: {path}")
    return data


def _first_stream(streams: list[dict[str, Any]], codec_type: str) -> dict[str, Any] | None:
    for stream in streams:
        if stream.get("codec_type") == codec_type:
            return stream
    return None


def _parse_frame_rate(rate: str) -> float:
    text = str(rate).strip()
    if not text or text == "0/0":
        return 0.0
    try:
        if "/" in text:
            num_s, den_s = text.split("/", 1)
            num = float(num_s)
            den = float(den_s)
            if den == 0:
                return 0.0
            return num / den
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _parse_duration(video_stream: dict[str, Any], fmt: dict[str, Any]) -> float:
    for source in (video_stream.get("duration"), fmt.get("duration")):
        if source is None:
            continue
        try:
            value = float(source)
            if value > 0:
                return value
        except (TypeError, ValueError):
            continue
    # Fall back to nb_frames / fps if available.
    try:
        nb = video_stream.get("nb_frames")
        if nb is not None:
            frames = int(nb)
            fps = _parse_frame_rate(
                video_stream.get("avg_frame_rate")
                or video_stream.get("r_frame_rate")
                or "0/1"
            )
            if frames > 0 and fps > 0:
                return frames / fps
    except (TypeError, ValueError):
        pass
    return 0.0


def _parse_frame_count(
    video_stream: dict[str, Any],
    duration: float,
    fps: float,
) -> int | None:
    nb = video_stream.get("nb_frames")
    if nb is not None and str(nb).isdigit():
        count = int(nb)
        if count > 0:
            return count
    if duration > 0 and fps > 0:
        return max(1, round(duration * fps))
    return None
