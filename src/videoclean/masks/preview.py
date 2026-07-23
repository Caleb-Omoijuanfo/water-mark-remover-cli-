"""Render preview images showing the watermark region and mask overlay."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from videoclean.exceptions import InvalidVideoError, ProcessingError
from videoclean.masks.manual import MASK_INPAINT, create_mask, validate_region_for_video
from videoclean.region import Region
from videoclean.video.extractor import extract_single_frame
from videoclean.video.metadata import VideoMetadata, probe

log = logging.getLogger(__name__)

# BGR colors for OpenCV drawing.
_COLOR_REGION_BOX = (0, 255, 0)  # green — user region
_COLOR_PADDED_BOX = (0, 165, 255)  # orange — padded region
_COLOR_MASK_TINT = (0, 0, 255)  # red — inpaint area
_DEFAULT_ALPHA = 0.4


def render_overlay(
    frame_bgr: np.ndarray,
    region: Region,
    *,
    padding: int = 0,
    alpha: float = _DEFAULT_ALPHA,
) -> tuple[np.ndarray, np.ndarray, Region]:
    """Compose a preview frame with mask tint and region rectangles.

    Returns
    -------
    overlay, mask, effective_region
        ``overlay`` is a BGR image ready to save; ``mask`` is the binary mask;
        ``effective_region`` is the region after padding/clamp.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ProcessingError(
            f"Expected a BGR frame (H, W, 3), got shape {frame_bgr.shape}"
        )

    height, width = frame_bgr.shape[:2]
    effective = validate_region_for_video(
        region, width, height, padding=padding
    )
    mask = create_mask(height, width, region, padding=padding)

    overlay = frame_bgr.copy()
    tint = np.zeros_like(overlay)
    tint[:, :] = _COLOR_MASK_TINT

    mask_bool = mask == MASK_INPAINT
    if mask_bool.any():
        blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, tint, alpha, 0)
        overlay[mask_bool] = blended[mask_bool]

    # Padded box (if padding expanded the region).
    if effective.as_tuple() != region.as_tuple():
        cv2.rectangle(
            overlay,
            (effective.x1, effective.y1),
            (max(effective.x1, effective.x2 - 1), max(effective.y1, effective.y2 - 1)),
            _COLOR_PADDED_BOX,
            2,
        )

    # User-specified region box.
    cv2.rectangle(
        overlay,
        (region.x1, region.y1),
        (max(region.x1, region.x2 - 1), max(region.y1, region.y2 - 1)),
        _COLOR_REGION_BOX,
        2,
    )

    # Corner label for quick visual confirmation.
    label = f"region {region}"
    if padding:
        label += f"  pad={padding} → {effective}"
    _draw_label(overlay, label, org=(max(8, region.x1), max(20, region.y1 - 8)))

    return overlay, mask, effective


def render_preview_image(
    video_path: Path | str,
    region: Region,
    output_path: Path | str,
    *,
    padding: int = 0,
    frame_index: int = 0,
    time_seconds: float | None = None,
    metadata: VideoMetadata | None = None,
    alpha: float = _DEFAULT_ALPHA,
    save_mask: bool = False,
) -> Path:
    """Extract a frame, render region/mask overlay, and write *output_path*.

    Parameters
    ----------
    video_path:
        Source video.
    region:
        User-specified watermark region (validated against frame size).
    output_path:
        Destination image path (``.png`` / ``.jpg``).
    padding:
        Extra pixels around the region included in the mask.
    frame_index:
        Zero-based frame index to preview (ignored if *time_seconds* is set).
    time_seconds:
        Optional timestamp to seek before grabbing a frame.
    metadata:
        Optional pre-probed metadata.
    alpha:
        Opacity of the red mask tint (0–1).
    save_mask:
        If True, also write a binary mask next to the preview
        (``<stem>_mask.png``).
    """
    source = Path(video_path).expanduser().resolve()
    dest = Path(output_path).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not source.is_file():
        raise InvalidVideoError(f"Video file not found: {source}")

    meta = metadata or probe(source)
    region.validate_bounds(meta.width, meta.height)

    # Extract one frame into a sibling temp file next to the destination.
    frame_tmp = dest.with_name(f".{dest.stem}_frame_tmp.png")
    try:
        extract_single_frame(
            source,
            frame_tmp,
            frame_index=frame_index,
            time_seconds=time_seconds,
            metadata=meta,
        )
        frame = cv2.imread(str(frame_tmp), cv2.IMREAD_COLOR)
        if frame is None:
            raise ProcessingError(f"Failed to read extracted frame: {frame_tmp}")

        # Guard against rare dimension mismatches after decode.
        fh, fw = frame.shape[:2]
        if fw != meta.width or fh != meta.height:
            log.warning(
                "Frame size %dx%d differs from probed %s; validating against frame",
                fw,
                fh,
                meta.resolution,
            )
            region.validate_bounds(fw, fh)

        overlay, mask, effective = render_overlay(
            frame, region, padding=padding, alpha=alpha
        )

        if not cv2.imwrite(str(dest), overlay):
            raise ProcessingError(f"Failed to write preview image: {dest}")

        if save_mask:
            mask_path = dest.with_name(f"{dest.stem}_mask{dest.suffix or '.png'}")
            if not cv2.imwrite(str(mask_path), mask):
                raise ProcessingError(f"Failed to write mask image: {mask_path}")
            log.info("Wrote mask preview: %s", mask_path)

        log.info(
            "Wrote preview %s (region=%s effective=%s padding=%d)",
            dest,
            region,
            effective,
            padding,
        )
        return dest
    finally:
        if frame_tmp.exists():
            frame_tmp.unlink(missing_ok=True)


def _draw_label(
    image: np.ndarray,
    text: str,
    *,
    org: tuple[int, int],
) -> None:
    """Draw a small filled label behind text for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    # Keep label inside the frame.
    x = min(max(0, x), max(0, image.shape[1] - tw - 4))
    y = min(max(th + 2, y), image.shape[0] - 2)
    cv2.rectangle(
        image,
        (x - 2, y - th - 2),
        (x + tw + 2, y + baseline + 2),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
