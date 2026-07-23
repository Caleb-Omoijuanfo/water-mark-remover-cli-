"""Watermark mask generation (manual, tracking, detection)."""

from videoclean.masks.manual import (
    MASK_INPAINT,
    MASK_KEEP,
    create_mask,
    create_mask_for_frame,
    expand_region,
    validate_region_for_video,
)
from videoclean.masks.preview import render_overlay, render_preview_image

__all__ = [
    "MASK_INPAINT",
    "MASK_KEEP",
    "create_mask",
    "create_mask_for_frame",
    "expand_region",
    "render_overlay",
    "render_preview_image",
    "validate_region_for_video",
]
