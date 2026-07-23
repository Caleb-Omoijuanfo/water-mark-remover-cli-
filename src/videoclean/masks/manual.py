"""Manual watermark masks from user-specified regions."""

from __future__ import annotations

import logging

import numpy as np

from videoclean.exceptions import ConfigError, InvalidRegionError
from videoclean.region import Region

log = logging.getLogger(__name__)

# Mask convention used throughout the pipeline:
#   0   = keep original pixels
#   255 = reconstruct / inpaint
MASK_KEEP = 0
MASK_INPAINT = 255


def expand_region(
    region: Region,
    padding: int,
    frame_width: int,
    frame_height: int,
) -> Region:
    """Expand *region* by *padding* pixels on each side, clamped to the frame.

    Parameters
    ----------
    region:
        Base region (must already lie inside the frame).
    padding:
        Non-negative pixel padding applied to all four sides.
    frame_width / frame_height:
        Frame dimensions used for clamping.
    """
    if padding < 0:
        raise ConfigError(f"mask padding must be >= 0, got {padding}")
    if padding == 0:
        region.validate_bounds(frame_width, frame_height)
        return region

    region.validate_bounds(frame_width, frame_height)

    x1 = max(0, region.x1 - padding)
    y1 = max(0, region.y1 - padding)
    x2 = min(frame_width, region.x2 + padding)
    y2 = min(frame_height, region.y2 + padding)

    expanded = Region(x1=x1, y1=y1, x2=x2, y2=y2)
    log.debug(
        "Expanded region %s by padding=%d → %s (frame %dx%d)",
        region,
        padding,
        expanded,
        frame_width,
        frame_height,
    )
    return expanded


def create_mask(
    frame_height: int,
    frame_width: int,
    region: Region,
    *,
    padding: int = 0,
) -> np.ndarray:
    """Create a binary uint8 mask for a single frame.

    Returns an array of shape ``(frame_height, frame_width)`` where pixels
    inside the (optionally padded) region are ``255`` (inpaint) and all
    others are ``0`` (keep).
    """
    if frame_height <= 0 or frame_width <= 0:
        raise InvalidRegionError(
            f"Frame dimensions must be positive, got {frame_width}x{frame_height}."
        )

    effective = expand_region(region, padding, frame_width, frame_height)
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    # Half-open [y1:y2, x1:x2] matches Region convention.
    mask[effective.y1 : effective.y2, effective.x1 : effective.x2] = MASK_INPAINT
    return mask


def create_mask_for_frame(
    frame: np.ndarray,
    region: Region,
    *,
    padding: int = 0,
) -> np.ndarray:
    """Create a mask matching the spatial size of *frame* (H×W×C or H×W)."""
    if frame.ndim not in (2, 3):
        raise ConfigError(f"Expected a 2D or 3D frame array, got shape {frame.shape}")
    height, width = frame.shape[:2]
    return create_mask(height, width, region, padding=padding)


def validate_region_for_video(
    region: Region,
    frame_width: int,
    frame_height: int,
    *,
    padding: int = 0,
) -> Region:
    """Validate *region* against video bounds and return the effective (padded) region.

    The base region must lie fully inside the frame. Padding may expand it up
    to the frame edges.
    """
    if padding < 0:
        raise ConfigError(f"mask padding must be >= 0, got {padding}")
    region.validate_bounds(frame_width, frame_height)
    return expand_region(region, padding, frame_width, frame_height)
