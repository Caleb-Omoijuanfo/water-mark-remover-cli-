"""Unit tests for FFmpeg binary discovery helpers."""

from __future__ import annotations

import pytest

from videoclean.exceptions import FFmpegNotFoundError
from videoclean.video import ffmpeg as ffmpeg_mod


def test_find_ffmpeg_and_ffprobe() -> None:
    ffmpeg_mod.reset_cached_paths()
    try:
        ff = ffmpeg_mod.find_ffmpeg()
        fp = ffmpeg_mod.find_ffprobe()
    except FFmpegNotFoundError:
        pytest.skip("FFmpeg not installed")
    assert ff
    assert fp
    assert "ffmpeg" in ff
    assert "ffprobe" in fp


def test_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    ffmpeg_mod.reset_cached_paths()
    monkeypatch.setattr(ffmpeg_mod.shutil, "which", lambda _name: None)
    with pytest.raises(FFmpegNotFoundError, match="ffmpeg"):
        ffmpeg_mod.find_ffmpeg()
    ffmpeg_mod.reset_cached_paths()
