"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from videoclean.video.ffmpeg import ensure_ffmpeg_available, run_ffmpeg


@pytest.fixture(scope="session")
def ffmpeg_available() -> None:
    """Skip tests that need FFmpeg when binaries are missing."""
    try:
        ensure_ffmpeg_available()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"FFmpeg not available: {exc}")


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory, ffmpeg_available: None) -> Path:
    """Generate a short test video with audio (session-scoped).

    Specs: 320x240, 2 seconds, 10 fps, sine-wave AAC audio.
    """
    out_dir = tmp_path_factory.mktemp("fixtures")
    path = out_dir / "sample_with_audio.mp4"

    # Color bars + sine audio, short and deterministic.
    run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=10:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ]
    )
    assert path.is_file() and path.stat().st_size > 0
    return path


@pytest.fixture(scope="session")
def sample_video_no_audio(
    tmp_path_factory: pytest.TempPathFactory,
    ffmpeg_available: None,
) -> Path:
    """Generate a short silent test video."""
    out_dir = tmp_path_factory.mktemp("fixtures_silent")
    path = out_dir / "sample_no_audio.mp4"
    run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=10:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ]
    )
    assert path.is_file()
    return path
