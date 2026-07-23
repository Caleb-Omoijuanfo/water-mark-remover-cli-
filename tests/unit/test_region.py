"""Unit tests for Region parsing and validation."""

from __future__ import annotations

import pytest

from videoclean.exceptions import InvalidRegionError
from videoclean.region import Region


class TestRegionParse:
    def test_valid_basic(self) -> None:
        region = Region.parse("20,20,300,120")
        assert region.x1 == 20
        assert region.y1 == 20
        assert region.x2 == 300
        assert region.y2 == 120

    def test_valid_with_whitespace(self) -> None:
        region = Region.parse("  10,  20 ,30 , 40  ")
        assert region.as_tuple() == (10, 20, 30, 40)

    def test_valid_zero_origin(self) -> None:
        region = Region.parse("0,0,100,50")
        assert region.as_tuple() == (0, 0, 100, 50)

    def test_width_height_area(self) -> None:
        region = Region.parse("10,20,110,70")
        assert region.width == 100
        assert region.height == 50
        assert region.area == 5000

    def test_str_roundtrip(self) -> None:
        region = Region.parse("5,6,7,8")
        assert str(region) == "5,6,7,8"
        assert Region.parse(str(region)) == region

    def test_empty_string(self) -> None:
        with pytest.raises(InvalidRegionError, match="empty"):
            Region.parse("")

    def test_whitespace_only(self) -> None:
        with pytest.raises(InvalidRegionError, match="empty"):
            Region.parse("   ")

    def test_too_few_components(self) -> None:
        with pytest.raises(InvalidRegionError, match="exactly 4"):
            Region.parse("1,2,3")

    def test_too_many_components(self) -> None:
        with pytest.raises(InvalidRegionError, match="exactly 4"):
            Region.parse("1,2,3,4,5")

    def test_non_integer(self) -> None:
        with pytest.raises(InvalidRegionError, match="integer"):
            Region.parse("1,2,3.5,4")

    def test_non_numeric(self) -> None:
        with pytest.raises(InvalidRegionError, match="integer"):
            Region.parse("a,b,c,d")

    def test_x1_not_less_than_x2(self) -> None:
        with pytest.raises(InvalidRegionError, match="x1 < x2"):
            Region.parse("100,10,100,50")

    def test_x1_greater_than_x2(self) -> None:
        with pytest.raises(InvalidRegionError, match="x1 < x2"):
            Region.parse("200,10,100,50")

    def test_y1_not_less_than_y2(self) -> None:
        with pytest.raises(InvalidRegionError, match="y1 < y2"):
            Region.parse("10,50,100,50")

    def test_y1_greater_than_y2(self) -> None:
        with pytest.raises(InvalidRegionError, match="y1 < y2"):
            Region.parse("10,80,100,50")

    def test_negative_coordinate(self) -> None:
        with pytest.raises(InvalidRegionError, match=">= 0"):
            Region.parse("-1,0,10,10")

    def test_direct_constructor_validates(self) -> None:
        with pytest.raises(InvalidRegionError, match="x1 < x2"):
            Region(x1=5, y1=0, x2=5, y2=10)


class TestRegionValidateBounds:
    def test_inside_bounds(self) -> None:
        region = Region.parse("10,10,100,80")
        region.validate_bounds(1920, 1080)  # should not raise

    def test_exact_frame_size(self) -> None:
        region = Region.parse("0,0,1920,1080")
        region.validate_bounds(1920, 1080)

    def test_outside_width(self) -> None:
        region = Region.parse("0,0,2000,100")
        with pytest.raises(InvalidRegionError, match="outside"):
            region.validate_bounds(1920, 1080)

    def test_outside_height(self) -> None:
        region = Region.parse("0,0,100,2000")
        with pytest.raises(InvalidRegionError, match="outside"):
            region.validate_bounds(1920, 1080)

    def test_invalid_frame_dimensions(self) -> None:
        region = Region.parse("0,0,10,10")
        with pytest.raises(InvalidRegionError, match="positive"):
            region.validate_bounds(0, 1080)
