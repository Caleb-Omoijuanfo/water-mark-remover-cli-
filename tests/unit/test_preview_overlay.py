"""Unit tests for preview overlay rendering (no FFmpeg)."""

from __future__ import annotations

import numpy as np
import pytest

from videoclean.exceptions import InvalidRegionError
from videoclean.masks.manual import MASK_INPAINT
from videoclean.masks.preview import render_overlay
from videoclean.region import Region


def test_render_overlay_shapes_and_mask() -> None:
    frame = np.full((100, 200, 3), 128, dtype=np.uint8)
    region = Region(20, 10, 60, 40)
    overlay, mask, effective = render_overlay(frame, region, padding=0)

    assert overlay.shape == frame.shape
    assert mask.shape == (100, 200)
    assert effective == region
    assert mask[10:40, 20:60].min() == MASK_INPAINT
    # Masked pixels should differ from the flat gray background (red tint).
    assert not np.array_equal(overlay[20, 30], frame[20, 30])
    # Unmasked pixel stays the same (aside from possible label area at top).
    assert np.array_equal(overlay[90, 180], frame[90, 180])


def test_render_overlay_with_padding() -> None:
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    region = Region(20, 20, 40, 40)
    overlay, mask, effective = render_overlay(frame, region, padding=5)
    assert effective.as_tuple() == (15, 15, 45, 45)
    assert mask[15:45, 15:45].min() == MASK_INPAINT
    assert overlay is not None


def test_render_overlay_rejects_out_of_bounds() -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    region = Region(0, 0, 60, 10)
    with pytest.raises(InvalidRegionError, match="outside"):
        render_overlay(frame, region)
