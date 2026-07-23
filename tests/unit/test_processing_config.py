"""Unit tests for ProcessingConfig."""

from __future__ import annotations

from pathlib import Path

import pytest

from videoclean.core.job import ProcessingConfig, default_output_path
from videoclean.exceptions import ConfigError
from videoclean.region import Region


def test_default_output_path() -> None:
    assert default_output_path(Path("clip.mp4")) == Path("clip_cleaned.mp4")
    assert default_output_path(Path("/tmp/a/b.mov")).name == "b_cleaned.mov"


def test_validate_basic_ok() -> None:
    cfg = ProcessingConfig(
        input_path=Path("in.mp4"),
        output_path=Path("out.mp4"),
        region=Region(0, 0, 10, 10),
    )
    cfg.validate_basic()


def test_validate_basic_rejects_bad_crf() -> None:
    cfg = ProcessingConfig(
        input_path=Path("in.mp4"),
        output_path=Path("out.mp4"),
        region=Region(0, 0, 10, 10),
        crf=99,
    )
    with pytest.raises(ConfigError, match="crf"):
        cfg.validate_basic()


def test_resolved_paths(tmp_path: Path) -> None:
    inp = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    cfg = ProcessingConfig(
        input_path=inp,
        output_path=out,
        region=Region(1, 1, 5, 5),
    ).resolved()
    assert cfg.input_path.is_absolute()
    assert cfg.output_path.is_absolute()
