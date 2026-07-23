"""Abstract inpainting engine interface.

Model / backend code must only depend on this module (plus numpy/OpenCV),
never on the CLI or pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from videoclean.exceptions import ModelError


class InpaintingEngine(ABC):
    """Backend-agnostic contract for watermark inpainting.

    Lifecycle::

        engine = SomeEngine(...)
        engine.load()
        out = engine.process_frame(frame, mask)
        # or engine.process_video(...)
        engine.unload()
    """

    #: Registry / CLI name for this engine (override in subclasses).
    name: str = "base"

    def __init__(self) -> None:
        self._loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """True after a successful :meth:`load`."""
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load weights / prepare the backend.

        Idempotent: calling ``load`` when already loaded is a no-op.
        """

    def unload(self) -> None:
        """Release resources. Safe to call when not loaded."""
        self._loaded = False

    @abstractmethod
    def process_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Inpaint a single BGR frame.

        Parameters
        ----------
        frame:
            ``uint8`` BGR image of shape ``(H, W, 3)``.
        mask:
            ``uint8`` single-channel mask of shape ``(H, W)`` where
            ``0`` = keep original pixels and ``255`` (or any non-zero) =
            reconstruct.

        Returns
        -------
        np.ndarray
            Inpainted BGR frame, same shape and dtype as *frame*.
        """

    def process_video(
        self,
        frames_dir: Path | str,
        output_dir: Path | str,
        mask: np.ndarray,
        *,
        image_ext: str = "png",
        frame_paths: Sequence[Path] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[Path]:
        """Inpaint every frame in *frames_dir* (or *frame_paths*).

        Default implementation loops :meth:`process_frame`. Subclasses may
        override for batch / temporal models.

        Parameters
        ----------
        frames_dir:
            Directory containing source frames (``frame_%06d.<ext>``).
        output_dir:
            Directory for written inpainted frames (created if needed).
        mask:
            Static binary mask applied to every frame.
        image_ext:
            Frame image extension when discovering frames via *frames_dir*.
        frame_paths:
            Optional explicit ordered list of frame paths (skips directory scan).
        progress_callback:
            Optional ``callback(current_index, total)`` invoked after each frame
            (1-based current index). Not used by the CLI progress UI, which is
            stage-based.

        Returns
        -------
        list[Path]
            Sorted list of written frame paths.
        """
        self._ensure_loaded()

        source_dir = Path(frames_dir)
        dest_dir = Path(output_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        if frame_paths is None:
            # Local discovery only — do not import video/pipeline modules here
            # so model code stays decoupled from the rest of the package.
            ext = "jpg" if image_ext == "jpeg" else image_ext
            paths = sorted(source_dir.glob(f"frame_*.{ext}"))
        else:
            paths = list(frame_paths)

        if not paths:
            raise ModelError(f"No frames found to inpaint in: {source_dir}")

        self._validate_mask(mask)

        written: list[Path] = []
        total = len(paths)
        for index, frame_path in enumerate(paths, start=1):
            out_path = dest_dir / frame_path.name
            self._process_frame_file(frame_path, mask, out_path)
            written.append(out_path)
            if progress_callback is not None:
                progress_callback(index, total)

        return written

    # ------------------------------------------------------------------
    # Helpers shared by implementations
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def _validate_frame_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ModelError(
                f"Expected BGR frame shape (H, W, 3), got {frame.shape}"
            )
        if frame.dtype != np.uint8:
            raise ModelError(f"Expected uint8 frame, got dtype {frame.dtype}")
        self._validate_mask(mask)
        if mask.shape[:2] != frame.shape[:2]:
            raise ModelError(
                f"Mask shape {mask.shape[:2]} does not match frame "
                f"{frame.shape[:2]}"
            )

    @staticmethod
    def _validate_mask(mask: np.ndarray) -> None:
        if mask.ndim != 2:
            raise ModelError(
                f"Expected single-channel mask (H, W), got shape {mask.shape}"
            )
        if mask.dtype != np.uint8:
            raise ModelError(f"Expected uint8 mask, got dtype {mask.dtype}")

    def _process_frame_file(
        self,
        frame_path: Path,
        mask: np.ndarray,
        output_path: Path,
    ) -> None:
        import cv2

        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ModelError(f"Failed to read frame: {frame_path}")
        result = self.process_frame(frame, mask)
        if not cv2.imwrite(str(output_path), result):
            raise ModelError(f"Failed to write inpainted frame: {output_path}")
