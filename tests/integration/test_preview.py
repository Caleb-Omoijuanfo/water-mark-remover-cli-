"""Integration tests for `videoclean preview` and mask validation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from videoclean.cli import app
from videoclean.masks.manual import MASK_INPAINT
from videoclean.masks.preview import render_preview_image
from videoclean.region import Region
from videoclean.video.metadata import probe

runner = CliRunner()


@pytest.mark.usefixtures("ffmpeg_available")
class TestPreviewIntegration:
    def test_render_preview_image_file(
        self,
        sample_video: Path,
        tmp_path: Path,
    ) -> None:
        meta = probe(sample_video)
        # Region inside 320x240 sample.
        region = Region(20, 20, 100, 60)
        out = tmp_path / "preview.png"

        result = render_preview_image(
            sample_video,
            region,
            out,
            padding=2,
            save_mask=True,
        )
        assert result == out.resolve() or result == out
        assert out.is_file()
        assert out.stat().st_size > 0

        image = cv2.imread(str(out), cv2.IMREAD_COLOR)
        assert image is not None
        assert image.shape[1] == meta.width
        assert image.shape[0] == meta.height

        mask_path = out.with_name("preview_mask.png")
        assert mask_path.is_file()
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert mask is not None
        # Padded region: 18,18 → 102,62
        assert mask[18:62, 18:102].min() == MASK_INPAINT
        assert int(mask[0, 0]) == 0

    def test_cli_preview_success(
        self,
        sample_video: Path,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "cli_preview.png"
        result = runner.invoke(
            app,
            [
                "preview",
                str(sample_video),
                "--region",
                "10,10,80,50",
                "--output",
                str(out),
                "--mask-padding",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.is_file()
        assert "Wrote preview" in result.output

        image = cv2.imread(str(out))
        assert image is not None
        assert image.shape[:2] == (240, 320)

    def test_cli_preview_invalid_region_human_readable(
        self,
        sample_video: Path,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "bad.png"
        result = runner.invoke(
            app,
            [
                "preview",
                str(sample_video),
                "--region",
                "0,0,9999,10",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 1
        # Human-readable error, not a Python traceback.
        combined = (result.output or "") + (result.stderr or "")
        assert "outside" in combined.lower() or "Error" in combined
        assert "Traceback" not in combined
        assert not out.exists()

    def test_cli_preview_malformed_region(
        self,
        sample_video: Path,
        tmp_path: Path,
    ) -> None:
        result = runner.invoke(
            app,
            [
                "preview",
                str(sample_video),
                "--region",
                "not-a-region",
                "--output",
                str(tmp_path / "x.png"),
            ],
        )
        assert result.exit_code == 1
        combined = (result.output or "") + (result.stderr or "")
        assert "Traceback" not in combined
        assert "Error" in combined or "region" in combined.lower()

    def test_cli_remove_validates_region(
        self,
        sample_video: Path,
        tmp_path: Path,
    ) -> None:
        # Invalid region: clear error (full pipeline covered in test_pipeline).
        bad = runner.invoke(
            app,
            [
                "remove",
                str(sample_video),
                "--region",
                "0,0,500,10",
                "--output",
                str(tmp_path / "out.mp4"),
            ],
        )
        assert bad.exit_code == 1
        combined = (bad.output or "") + (bad.stderr or "")
        assert "Traceback" not in combined
        assert "outside" in combined.lower() or "Error" in combined
