"""Simple inpainting model registry for ``--model`` selection."""

from __future__ import annotations

import logging
from typing import Callable

from videoclean.exceptions import ConfigError, ModelError
from videoclean.inpainting.base import InpaintingEngine

log = logging.getLogger(__name__)

EngineFactory = Callable[..., InpaintingEngine]

DEFAULT_MODEL = "opencv"

# name -> factory
_REGISTRY: dict[str, EngineFactory] = {}
_BUILTINS_REGISTERED = False


def register(name: str, factory: EngineFactory, *, overwrite: bool = False) -> None:
    """Register an engine factory under *name* (case-insensitive)."""
    key = _normalize(name)
    if not key:
        raise ConfigError("Model name must be a non-empty string")
    if key in _REGISTRY and not overwrite:
        raise ConfigError(f"Model {name!r} is already registered")
    _REGISTRY[key] = factory
    log.debug("Registered inpainting model: %s", key)


def unregister(name: str) -> None:
    """Remove a model from the registry (mainly for tests)."""
    _REGISTRY.pop(_normalize(name), None)


def list_models() -> list[str]:
    """Return sorted registered model names."""
    _ensure_builtins()
    return sorted(_REGISTRY)


def get_engine(name: str | None = None, **kwargs: object) -> InpaintingEngine:
    """Instantiate an engine by registry name.

    Parameters
    ----------
    name:
        Model key (default: :data:`DEFAULT_MODEL`).
    **kwargs:
        Forwarded to the engine factory (e.g. ``radius``, ``device``).
    """
    _ensure_builtins()
    key = _normalize(name or DEFAULT_MODEL)
    factory = _REGISTRY.get(key)
    if factory is None:
        available = ", ".join(list_models()) or "(none)"
        raise ConfigError(
            f"Unknown model {name!r}. Available models: {available}."
        )
    try:
        engine = factory(**kwargs)
    except TypeError as exc:
        raise ModelError(
            f"Failed to create model {key!r} with kwargs {kwargs}: {exc}"
        ) from exc
    if not isinstance(engine, InpaintingEngine):
        raise ModelError(
            f"Factory for {key!r} returned {type(engine).__name__}, "
            f"expected InpaintingEngine"
        )
    return engine


def _normalize(name: str) -> str:
    return str(name).strip().lower()


def _ensure_builtins() -> None:
    """Lazily register built-in engines (avoids import cycles at module load)."""
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    from videoclean.inpainting.frame_inpainter import FrameInpaintingEngine

    def _opencv_factory(**kwargs: object) -> InpaintingEngine:
        # Filter kwargs FrameInpaintingEngine accepts.
        allowed = {"radius", "algorithm", "device"}
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return FrameInpaintingEngine(**filtered)  # type: ignore[arg-type]

    # Always install builtins (overwrite=False only matters if user registered first).
    if "opencv" not in _REGISTRY:
        register("opencv", _opencv_factory)
    if "cv2" not in _REGISTRY:
        register("cv2", _opencv_factory)
    if "telea" not in _REGISTRY:

        def _telea_factory(**kwargs: object) -> InpaintingEngine:
            merged = dict(kwargs)
            merged["algorithm"] = "telea"
            return _opencv_factory(**merged)

        register("telea", _telea_factory)

    _BUILTINS_REGISTERED = True


def reset_registry() -> None:
    """Clear registry state (for tests)."""
    global _BUILTINS_REGISTERED
    _REGISTRY.clear()
    _BUILTINS_REGISTERED = False
