"""OpenCV Telea/NS frame-by-frame inpainting engine (MVP placeholder)."""

from __future__ import annotations

import logging
from typing import Literal

import cv2
import numpy as np

from videoclean.exceptions import ConfigError, ModelError
from videoclean.inpainting.base import InpaintingEngine

log = logging.getLogger(__name__)

InpaintAlgorithm = Literal["telea", "ns"]

_ALGORITHM_MAP: dict[str, int] = {
    "telea": cv2.INPAINT_TELEA,
    "ns": cv2.INPAINT_NS,
}


class FrameInpaintingEngine(InpaintingEngine):
    """Per-frame inpainting via :func:`cv2.inpaint`.

    This is the MVP placeholder backend so the full pipeline can be validated
    end-to-end before integrating heavier AI models. It runs on CPU only.
    """

    name = "opencv"

    def __init__(
        self,
        *,
        radius: int = 3,
        algorithm: InpaintAlgorithm = "telea",
        device: str = "cpu",
    ) -> None:
        """
        Parameters
        ----------
        radius:
            Inpaint radius in pixels (OpenCV ``inpaintRadius``).
        algorithm:
            ``"telea"`` (fast, default) or ``"ns"`` (Navier-Stokes).
        device:
            Accepted for API compatibility with future GPU engines. Only
            ``"cpu"`` is supported here.
        """
        super().__init__()
        if radius < 1:
            raise ConfigError(f"inpaint radius must be >= 1, got {radius}")
        if algorithm not in _ALGORITHM_MAP:
            raise ConfigError(
                f"Unknown inpaint algorithm {algorithm!r}; "
                f"expected one of {sorted(_ALGORITHM_MAP)}"
            )
        if device not in {"cpu", "auto"}:
            # OpenCV inpaint is CPU-only; allow "auto" so pipeline device
            # selection does not break this engine.
            log.warning(
                "FrameInpaintingEngine ignores device=%r (CPU-only backend)",
                device,
            )
        self.radius = radius
        self.algorithm = algorithm
        self.device = "cpu"
        self._cv_flags = _ALGORITHM_MAP[algorithm]

    def load(self) -> None:
        """No weights to load; marks the engine ready."""
        if self._loaded:
            return
        log.debug(
            "Loaded FrameInpaintingEngine (algorithm=%s radius=%d)",
            self.algorithm,
            self.radius,
        )
        self._loaded = True

    def process_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Inpaint *frame* where *mask* is non-zero using OpenCV."""
        self._ensure_loaded()
        self._validate_frame_mask(frame, mask)

        # OpenCV treats any non-zero mask pixel as a region to reconstruct.
        # Empty mask → return a copy (nothing to do).
        if not np.any(mask):
            return frame.copy()

        try:
            result = cv2.inpaint(frame, mask, self.radius, self._cv_flags)
        except cv2.error as exc:  # pragma: no cover - defensive
            raise ModelError(f"OpenCV inpaint failed: {exc}") from exc

        if result is None or result.shape != frame.shape:
            raise ModelError(
                f"OpenCV inpaint returned unexpected result "
                f"(shape={None if result is None else result.shape})"
            )
        return result
