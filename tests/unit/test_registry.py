"""Unit tests for the inpainting model registry."""

from __future__ import annotations

import pytest

from videoclean.exceptions import ConfigError
from videoclean.inpainting.base import InpaintingEngine
from videoclean.inpainting.frame_inpainter import FrameInpaintingEngine
from videoclean.models.registry import (
    DEFAULT_MODEL,
    get_engine,
    list_models,
    register,
    reset_registry,
    unregister,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    reset_registry()
    yield
    reset_registry()


def test_default_models_registered() -> None:
    models = list_models()
    assert DEFAULT_MODEL in models
    assert "opencv" in models
    assert "cv2" in models


def test_get_engine_default() -> None:
    engine = get_engine()
    assert isinstance(engine, FrameInpaintingEngine)
    assert isinstance(engine, InpaintingEngine)


def test_get_engine_by_name() -> None:
    engine = get_engine("opencv", radius=5)
    assert isinstance(engine, FrameInpaintingEngine)
    assert engine.radius == 5


def test_unknown_model() -> None:
    with pytest.raises(ConfigError, match="Unknown model"):
        get_engine("does-not-exist")


def test_register_custom() -> None:
    class Dummy(InpaintingEngine):
        name = "dummy"

        def load(self) -> None:
            self._loaded = True

        def process_frame(self, frame, mask):  # type: ignore[no-untyped-def]
            return frame

    register("dummy", lambda **_: Dummy())
    assert "dummy" in list_models()
    engine = get_engine("dummy")
    assert isinstance(engine, Dummy)
    unregister("dummy")
    assert "dummy" not in list_models()
