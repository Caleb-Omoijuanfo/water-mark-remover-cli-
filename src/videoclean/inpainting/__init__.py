"""Inpainting engines (frame and video backends)."""

from videoclean.inpainting.base import InpaintingEngine
from videoclean.inpainting.frame_inpainter import FrameInpaintingEngine

__all__ = [
    "FrameInpaintingEngine",
    "InpaintingEngine",
]
