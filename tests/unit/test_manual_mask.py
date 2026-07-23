"""Unit tests for manual mask generation."""

from __future__ import annotations

import numpy as np
import pytest

from videoclean.exceptions import ConfigError, InvalidRegionError
from videoclean.masks.manual import (
    MASK_INPAINT,
    MASK_KEEP,
    create_mask,
    create_mask_for_frame,
    expand_region,
    validate_region_for_video,
)
from videoclean.region import Region


class TestCreateMask:
    def test_basic_region(self) -> None:
        region = Region(10, 20, 40, 50)
        mask = create_mask(100, 200, region)
        assert mask.shape == (100, 200)
        assert mask.dtype == np.uint8
        assert mask[20:50, 10:40].min() == MASK_INPAINT
        assert mask[20:50, 10:40].max() == MASK_INPAINT
        # Outside region is keep.
        assert mask[0, 0] == MASK_KEEP
        assert mask[99, 199] == MASK_KEEP
        # Boundary: x2/y2 are exclusive.
        assert mask[49, 39] == MASK_INPAINT
        assert mask[50, 10] == MASK_KEEP
        assert mask[20, 40] == MASK_KEEP

    def test_full_frame(self) -> None:
        region = Region(0, 0, 64, 48)
        mask = create_mask(48, 64, region)
        assert mask.sum() == 64 * 48 * MASK_INPAINT

    def test_with_padding(self) -> None:
        region = Region(10, 10, 20, 20)
        mask = create_mask(100, 100, region, padding=5)
        # Effective: 5,5 → 25,25
        assert mask[5:25, 5:25].min() == MASK_INPAINT
        assert mask[4, 10] == MASK_KEEP
        assert mask[10, 4] == MASK_KEEP

    def test_padding_clamped_to_edges(self) -> None:
        region = Region(2, 2, 10, 10)
        mask = create_mask(50, 50, region, padding=10)
        # Clamped to 0,0 → 20,20
        assert mask[0:20, 0:20].min() == MASK_INPAINT
        assert int(mask.sum()) == 20 * 20 * MASK_INPAINT

    def test_negative_padding(self) -> None:
        region = Region(10, 10, 20, 20)
        with pytest.raises(ConfigError, match="padding"):
            create_mask(100, 100, region, padding=-1)

    def test_region_outside_frame(self) -> None:
        region = Region(0, 0, 200, 10)
        with pytest.raises(InvalidRegionError, match="outside"):
            create_mask(100, 100, region)

    def test_create_mask_for_frame(self) -> None:
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        region = Region(5, 5, 15, 25)
        mask = create_mask_for_frame(frame, region)
        assert mask.shape == (80, 120)
        assert mask[5:25, 5:15].max() == MASK_INPAINT


class TestExpandRegion:
    def test_no_padding(self) -> None:
        region = Region(10, 10, 30, 40)
        assert expand_region(region, 0, 100, 100) == region

    def test_expand(self) -> None:
        region = Region(10, 10, 30, 40)
        expanded = expand_region(region, 3, 100, 100)
        assert expanded.as_tuple() == (7, 7, 33, 43)


class TestValidateRegionForVideo:
    def test_valid(self) -> None:
        region = Region(0, 0, 100, 50)
        effective = validate_region_for_video(region, 320, 240, padding=0)
        assert effective == region

    def test_invalid_bounds(self) -> None:
        region = Region(0, 0, 400, 50)
        with pytest.raises(InvalidRegionError, match="outside"):
            validate_region_for_video(region, 320, 240)
