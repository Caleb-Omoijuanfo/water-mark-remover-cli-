"""Thin subprocess wrappers for ffmpeg and ffprobe.

All FFmpeg invocations in VideoClean should go through this module.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from videoclean.exceptions import FFmpegError, FFmpegNotFoundError

log = logging.getLogger(__name__)

_ffmpeg_path: str | None = None
_ffprobe_path: str | None = None


def find_ffmpeg() -> str:
    """Locate the ``ffmpeg`` binary on PATH."""
    global _ffmpeg_path
    if _ffmpeg_path is not None:
        return _ffmpeg_path
    path = shutil.which("ffmpeg")
    if path is None:
        raise FFmpegNotFoundError(
            "ffmpeg was not found on PATH. Install FFmpeg "
            "(e.g. `brew install ffmpeg` or `apt install ffmpeg`) and retry."
        )
    _ffmpeg_path = path
    return path


def find_ffprobe() -> str:
    """Locate the ``ffprobe`` binary on PATH."""
    global _ffprobe_path
    if _ffprobe_path is not None:
        return _ffprobe_path
    path = shutil.which("ffprobe")
    if path is None:
        raise FFmpegNotFoundError(
            "ffprobe was not found on PATH. Install FFmpeg "
            "(e.g. `brew install ffmpeg` or `apt install ffmpeg`) and retry."
        )
    _ffprobe_path = path
    return path


def ensure_ffmpeg_available() -> tuple[str, str]:
    """Ensure both ffmpeg and ffprobe are available; return their paths."""
    return find_ffmpeg(), find_ffprobe()


def run_ffmpeg(
    args: Sequence[str],
    *,
    check: bool = True,
    capture_output: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``ffmpeg`` with the given arguments (excluding the binary name).

    Always passes ``-hide_banner`` and ``-loglevel error`` unless the caller
    already supplied ``-loglevel``.
    """
    binary = find_ffmpeg()
    cmd = [binary, "-hide_banner", "-y"]
    # Allow caller to override loglevel by placing it first in args.
    if not any(a == "-loglevel" for a in args):
        cmd.extend(["-loglevel", "error"])
    cmd.extend(args)
    return _run(cmd, check=check, capture_output=capture_output, timeout=timeout)


def run_ffprobe(
    args: Sequence[str],
    *,
    check: bool = True,
    capture_output: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``ffprobe`` with the given arguments (excluding the binary name)."""
    binary = find_ffprobe()
    cmd = [binary, "-hide_banner"]
    if not any(a == "-loglevel" for a in args):
        cmd.extend(["-loglevel", "error"])
    cmd.extend(args)
    return _run(cmd, check=check, capture_output=capture_output, timeout=timeout)


def _run(
    cmd: list[str],
    *,
    check: bool,
    capture_output: bool,
    timeout: float | None,
) -> subprocess.CompletedProcess[str]:
    log.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        # Binary vanished between which() and exec.
        name = Path(cmd[0]).name
        raise FFmpegNotFoundError(
            f"{name} was not found on PATH. Install FFmpeg and retry."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(
            f"Command timed out after {timeout}s: {' '.join(cmd)}"
        ) from exc

    if check and result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise FFmpegError(f"Command failed ({cmd[0]}): {detail}")

    return result


def reset_cached_paths() -> None:
    """Clear cached binary paths (for tests)."""
    global _ffmpeg_path, _ffprobe_path
    _ffmpeg_path = None
    _ffprobe_path = None
