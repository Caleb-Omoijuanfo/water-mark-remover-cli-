"""Integration: full remove pipeline preserves media properties."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from videoclean.cli import app
from videoclean.core.job import ProcessingConfig
from videoclean.core.pipeline import run_pipeline
from videoclean.core.job import JobStatus
from videoclean.region import Region
from videoclean.video.metadata import probe

runner = CliRunner()


def _draw_watermark(video: Path, out: Path, region: Region) -> Path:
    """Copy *video* frames with a solid rectangle burned into *region*."""
    from videoclean.core.job import JobWorkspace
    from videoclean.video.encoder import encode_frames, extract_audio, mux_video_audio
    from videoclean.video.extractor import extract_frames, list_frames

    meta = probe(video)
    with JobWorkspace.create() as ws:
        extract_frames(video, ws.frames_dir, metadata=meta)
        for frame_path in list_frames(ws.frames_dir):
            img = cv2.imread(str(frame_path))
            assert img is not None
            img[region.y1: region.y2, region.x1: region.x2] = (0, 0, 255)
            cv2.imwrite(str(frame_path), img)
        audio = extract_audio(video, ws.audio_path, metadata=meta)
        encode_frames(ws.frames_dir, ws.video_only_path, fps=meta.fps)
        mux_video_audio(ws.video_only_path, out, audio_path=audio)
    return out


@pytest.mark.usefixtures("ffmpeg_available")
class TestPipelineIntegration:
    def test_run_pipeline_end_to_end(
        self,
        sample_video: Path,
        tmp_path: Path,
    ) -> None:
        region = Region(20, 20, 80, 50)
        watermarked = tmp_path / "watermarked.mp4"
        _draw_watermark(sample_video, watermarked, region)

        output = tmp_path / "cleaned.mp4"
        cfg = ProcessingConfig(
            input_path=watermarked,
            output_path=output,
            region=region,
            mask_padding=2,
            model="opencv",
            device="cpu",
            keep_temp=False,
        )
        result = run_pipeline(cfg, show_progress=False)

        assert result.output_path == output.resolve()
        assert output.is_file() and output.stat().st_size > 0
        assert result.frame_count > 0
        assert result.job_state is not None
        assert result.job_state.status is JobStatus.COMPLETED
        assert result.job_state.progress == pytest.approx(1.0)

        src = probe(watermarked)
        out = probe(output)
        assert out.width == src.width
        assert out.height == src.height
        assert out.has_audio is src.has_audio
        # Double encode (fixture burn-in + clean) can drift slightly; allow 0.35s.
        assert out.duration == pytest.approx(src.duration, abs=0.35)
        assert out.fps == pytest.approx(src.fps, rel=0.05)
        if src.frame_count is not None and out.frame_count is not None:
            assert abs(out.frame_count - src.frame_count) <= 1

        # Spot-check: first cleaned frame should not be pure red in the region.
        from videoclean.video.extractor import extract_single_frame

        frame_path = tmp_path / "check.png"
        extract_single_frame(output, frame_path, frame_index=0, metadata=out)
        frame = cv2.imread(str(frame_path))
        assert frame is not None
        patch = frame[region.y1: region.y2, region.x1: region.x2]
        # Mean channel distance from pure red should be large after inpaint
        # against the testsrc background (not a solid red fill).
        mean_bgr = patch.mean(axis=(0, 1))
        pure_red = np.array([0.0, 0.0, 255.0])
        assert np.linalg.norm(mean_bgr - pure_red) > 30.0

    def test_cli_remove_success(
        self,
        sample_video: Path,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "cli_cleaned.mp4"
        result = runner.invoke(
            app,
            [
                "remove",
                str(sample_video),
                "--region",
                "10,10,60,40",
                "--output",
                str(out),
                "--mask-padding",
                "1",
                "--model",
                "opencv",
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.is_file()
        assert "Done" in result.output or "done" in result.output.lower()
        combined = (result.output or "").lower()
        assert "stub" not in combined

        src = probe(sample_video)
        out_meta = probe(out)
        assert out_meta.width == src.width
        assert out_meta.height == src.height
        assert out_meta.has_audio is src.has_audio
        assert out_meta.duration == pytest.approx(src.duration, abs=0.2)

    def test_cli_remove_invalid_region(
        self,
        sample_video: Path,
        tmp_path: Path,
    ) -> None:
        result = runner.invoke(
            app,
            [
                "remove",
                str(sample_video),
                "--region",
                "0,0,500,10",
                "--output",
                str(tmp_path / "x.mp4"),
            ],
        )
        assert result.exit_code == 1
        combined = (result.output or "") + (result.stderr or "")
        assert "Traceback" not in combined
        assert "outside" in combined.lower() or "Error" in combined

    def test_cli_remove_requires_region(
        self,
        sample_video: Path,
    ) -> None:
        result = runner.invoke(app, ["remove", str(sample_video)])
        assert result.exit_code != 0
