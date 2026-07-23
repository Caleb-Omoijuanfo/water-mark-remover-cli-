"""Compute device selection.

Phase 6 expands this with CUDA / MPS auto-detection. For Phase 5 the pipeline
always resolves to CPU so the OpenCV MVP backend works everywhere.
"""

from __future__ import annotations

import logging
from typing import Literal

from videoclean.exceptions import ConfigError

log = logging.getLogger(__name__)

DeviceName = Literal["cpu", "cuda", "mps"]

SUPPORTED_DEVICES: tuple[str, ...] = ("cpu", "cuda", "mps", "auto")


def resolve_device(requested: str = "auto") -> DeviceName:
    """Resolve a user-requested device string to a concrete device.

    Parameters
    ----------
    requested:
        ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.

    Returns
    -------
    DeviceName
        Concrete device. Phase 5 always returns ``"cpu"`` for ``"auto"``;
        explicit ``"cuda"`` / ``"mps"`` are accepted but may be ignored by
        CPU-only engines (with a warning at engine construction).
    """
    key = (requested or "auto").strip().lower()
    if key not in SUPPORTED_DEVICES:
        raise ConfigError(
            f"Unknown device {requested!r}. "
            f"Expected one of: {', '.join(SUPPORTED_DEVICES)}."
        )

    if key == "auto":
        # Phase 6: probe torch.cuda / torch.backends.mps.
        log.debug("Device auto-select → cpu (GPU detection lands in Phase 6)")
        return "cpu"

    if key in {"cuda", "mps"}:
        log.debug("Device requested=%s (availability checked in Phase 6)", key)
        return key  # type: ignore[return-value]

    return "cpu"


def describe_device(device: str) -> str:
    """Human-readable device label for CLI display."""
    labels = {
        "cpu": "CPU",
        "cuda": "CUDA (NVIDIA GPU)",
        "mps": "MPS (Apple Silicon)",
    }
    return labels.get(device, device)
