"""Unit tests for VideoMetadata parsing (no FFmpeg required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from videoclean.exceptions import InvalidVideoError
from videoclean.video.metadata import parse_probe_data


def _base_probe(
    *,
    width: int = 640,
    height: int = 360,
    fps: str = "30/1",
    duration: str = "2.0",
    nb_frames: str | None = "60",
    has_audio: bool = True,
) -> dict:
    streams: list[dict] = [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": width,
            "height": height,
            "avg_frame_rate": fps,
            "r_frame_rate": fps,
            "pix_fmt": "yuv420p",
            "duration": duration,
        }
    ]
    if nb_frames is not None:
        streams[0]["nb_frames"] = nb_frames
    if has_audio:
        streams.append(
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "44100",
                "channels": 2,
            }
        )
    return {
        "streams": streams,
        "format": {
            "duration": duration,
            "bit_rate": "1000000",
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        },
    }


class TestParseProbeData:
    def test_basic_video_with_audio(self) -> None:
        path = Path("/tmp/sample.mp4")
        meta = parse_probe_data(_base_probe(), path)
        assert meta.path == path
        assert meta.width == 640
        assert meta.height == 360
        assert meta.resolution == "640x360"
        assert meta.fps == pytest.approx(30.0)
        assert meta.duration == pytest.approx(2.0)
        assert meta.frame_count == 60
        assert meta.video_codec == "h264"
        assert meta.pixel_format == "yuv420p"
        assert meta.has_audio is True
        assert meta.audio_codec == "aac"
        assert meta.audio_sample_rate == 44100
        assert meta.audio_channels == 2
        assert meta.bit_rate == 1_000_000

    def test_video_without_audio(self) -> None:
        meta = parse_probe_data(_base_probe(has_audio=False), Path("x.mp4"))
        assert meta.has_audio is False
        assert meta.audio_codec is None

    def test_fractional_fps(self) -> None:
        meta = parse_probe_data(_base_probe(fps="30000/1001"), Path("x.mp4"))
        assert meta.fps == pytest.approx(30000 / 1001)

    def test_frame_count_from_duration(self) -> None:
        data = _base_probe(duration="1.5", fps="24/1", nb_frames=None)
        meta = parse_probe_data(data, Path("x.mp4"))
        assert meta.frame_count == 36  # round(1.5 * 24)

    def test_missing_video_stream(self) -> None:
        data = {
            "streams": [{"codec_type": "audio", "codec_name": "aac"}],
            "format": {"duration": "1.0"},
        }
        with pytest.raises(InvalidVideoError, match="No video stream"):
            parse_probe_data(data, Path("x.mp4"))

    def test_invalid_dimensions(self) -> None:
        with pytest.raises(InvalidVideoError, match="dimensions"):
            parse_probe_data(_base_probe(width=0, height=0), Path("x.mp4"))

    def test_zero_fps(self) -> None:
        with pytest.raises(InvalidVideoError, match="frame rate"):
            parse_probe_data(_base_probe(fps="0/0"), Path("x.mp4"))
