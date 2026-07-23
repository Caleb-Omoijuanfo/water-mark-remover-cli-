"""Integration: extract → encode → mux preserves duration, resolution, audio."""

from __future__ import annotations

from pathlib import Path

import pytest

from videoclean.video.metadata import probe
from videoclean.video.roundtrip import roundtrip_video


@pytest.mark.usefixtures("ffmpeg_available")
class TestRoundtrip:
    def test_roundtrip_preserves_duration_resolution_audio(
        self,
        sample_video: Path,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "roundtrip.mp4"
        result = roundtrip_video(sample_video, output)
        assert result == output.resolve()
        assert output.is_file()
        assert output.stat().st_size > 0

        src = probe(sample_video)
        out = probe(output)

        assert out.width == src.width
        assert out.height == src.height
        assert out.has_audio is True
        assert src.has_audio is True

        # Duration should match within a small tolerance (container/codec rounding).
        assert out.duration == pytest.approx(src.duration, abs=0.15)

        # FPS should be close (exact equality may fail due to float formatting).
        assert out.fps == pytest.approx(src.fps, rel=0.02)

        # Frame count: allow ±1 due to encoder boundary rounding.
        if src.frame_count is not None and out.frame_count is not None:
            assert abs(out.frame_count - src.frame_count) <= 1

    def test_roundtrip_without_audio(
        self,
        sample_video_no_audio: Path,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "silent_roundtrip.mp4"
        roundtrip_video(sample_video_no_audio, output)

        src = probe(sample_video_no_audio)
        out = probe(output)

        assert out.width == src.width
        assert out.height == src.height
        assert out.has_audio is False
        assert out.duration == pytest.approx(src.duration, abs=0.15)

    def test_roundtrip_rejects_same_path(
        self,
        sample_video: Path,
    ) -> None:
        from videoclean.exceptions import InvalidVideoError

        with pytest.raises(InvalidVideoError, match="differ"):
            roundtrip_video(sample_video, sample_video)
