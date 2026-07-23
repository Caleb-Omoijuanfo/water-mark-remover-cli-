"""Unit tests for OpenCV frame inpainting (no FFmpeg / no CLI)."""

from __future__ import annotations

import numpy as np
import pytest

from videoclean.exceptions import ConfigError, ModelError
from videoclean.inpainting.frame_inpainter import FrameInpaintingEngine
from videoclean.masks.manual import MASK_INPAINT, create_mask
from videoclean.region import Region


def _solid_frame(
    height: int = 64,
    width: int = 64,
    color: tuple[int, int, int] = (40, 80, 160),
) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


class TestFrameInpaintingEngine:
    def test_load_and_process_frame(self) -> None:
        engine = FrameInpaintingEngine(radius=3)
        assert not engine.is_loaded
        engine.load()
        assert engine.is_loaded
        engine.load()  # idempotent

        frame = _solid_frame()
        # Paint a contrasting block that should be removed.
        frame[20:40, 20:40] = (0, 0, 255)
        mask = create_mask(64, 64, Region(20, 20, 40, 40))

        out = engine.process_frame(frame, mask)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8
        # Outside mask unchanged.
        assert np.array_equal(out[0, 0], frame[0, 0])
        # Inside mask should no longer be pure red (inpaint fills from neighbors).
        center = out[30, 30]
        assert not np.array_equal(center, np.array([0, 0, 255], dtype=np.uint8))

    def test_empty_mask_returns_copy(self) -> None:
        engine = FrameInpaintingEngine()
        engine.load()
        frame = _solid_frame()
        mask = np.zeros((64, 64), dtype=np.uint8)
        out = engine.process_frame(frame, mask)
        assert np.array_equal(out, frame)
        assert out is not frame

    def test_rejects_mismatched_mask(self) -> None:
        engine = FrameInpaintingEngine()
        engine.load()
        frame = _solid_frame(32, 32)
        mask = np.zeros((16, 16), dtype=np.uint8)
        with pytest.raises(ModelError, match="does not match"):
            engine.process_frame(frame, mask)

    def test_rejects_bad_radius(self) -> None:
        with pytest.raises(ConfigError, match="radius"):
            FrameInpaintingEngine(radius=0)

    def test_process_video_writes_frames(self, tmp_path) -> None:
        import cv2

        frames_dir = tmp_path / "frames"
        out_dir = tmp_path / "out"
        frames_dir.mkdir()

        for i in range(1, 4):
            frame = _solid_frame()
            frame[10:20, 10:20] = (255, 255, 255)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.png"), frame)

        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:20, 10:20] = MASK_INPAINT

        engine = FrameInpaintingEngine()
        written = engine.process_video(frames_dir, out_dir, mask)
        assert len(written) == 3
        for path in written:
            assert path.is_file()
            img = cv2.imread(str(path))
            assert img is not None
            assert img.shape == (64, 64, 3)

    def test_no_cli_or_pipeline_imports(self) -> None:
        """Model module must not depend on CLI / pipeline (Phase 4 AC)."""
        import videoclean.inpainting.frame_inpainter as mod

        forbidden = {
            "videoclean.cli",
            "videoclean.core.pipeline",
            "videoclean.core.job",
        }
        module_names = set(mod.__dict__.get("__name__", "").split()) | set(
            getattr(m, "__name__", "")
            for m in mod.__dict__.values()
            if hasattr(m, "__name__")
        )
        # Stronger: inspect the module's import graph via sys.modules after import.
        import sys

        # Any submodule loaded solely because of frame_inpainter should not be CLI.
        loaded = {
            name
            for name in sys.modules
            if name.startswith("videoclean.")
        }
        # frame_inpainter may share package with others already loaded by tests;
        # assert the source file itself has no forbidden imports.
        source = Path_read(mod.__file__)
        for name in forbidden:
            assert name not in source, f"forbidden import of {name}"


def Path_read(path: str | None) -> str:
    from pathlib import Path

    assert path is not None
    return Path(path).read_text(encoding="utf-8")
