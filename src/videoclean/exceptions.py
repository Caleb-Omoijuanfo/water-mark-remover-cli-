"""Custom exception hierarchy for VideoClean.

All user-facing failures should raise a subclass of :class:`VideoCleanError`
so the CLI can print a human-readable message without a raw stack trace.
"""

from __future__ import annotations


class VideoCleanError(Exception):
    """Base error for all VideoClean failures."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class InvalidRegionError(VideoCleanError):
    """Raised when a region string or coordinates are invalid."""


class InvalidVideoError(VideoCleanError):
    """Raised when a video file is missing, unreadable, or malformed."""


class FFmpegNotFoundError(VideoCleanError):
    """Raised when ffmpeg or ffprobe is not available on PATH."""


class FFmpegError(VideoCleanError):
    """Raised when an ffmpeg/ffprobe subprocess fails."""


class ProcessingError(VideoCleanError):
    """Raised when the processing pipeline fails mid-job."""


class ModelError(VideoCleanError):
    """Raised when model loading or inference fails."""


class ConfigError(VideoCleanError):
    """Raised when configuration or CLI options are invalid."""
