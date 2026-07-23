"""Rich-based logging setup for the VideoClean CLI."""

from __future__ import annotations

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

_console: Optional[Console] = None
_configured: bool = False


def get_console() -> Console:
    """Return the shared Rich console used for CLI output."""
    global _console
    if _console is None:
        _console = Console(stderr=False)
    return _console


def setup_logging(*, verbose: bool = False) -> None:
    """Configure application logging.

    Parameters
    ----------
    verbose:
        When True, log at DEBUG level; otherwise INFO.
    """
    global _configured

    level = logging.DEBUG if verbose else logging.INFO
    console = get_console()

    handler = RichHandler(
        console=console,
        show_time=verbose,
        show_path=verbose,
        rich_tracebacks=verbose,
        markup=True,
    )
    handler.setLevel(level)

    root = logging.getLogger()
    # Replace existing handlers so repeated setup (e.g. tests) stays clean.
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet noisy third-party loggers unless debugging.
    if not verbose:
        for name in ("urllib3", "PIL", "torch", "asyncio"):
            logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True
    logging.debug("Logging configured (verbose=%s)", verbose)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (ensures setup has been called at least once)."""
    if not _configured:
        setup_logging(verbose=False)
    return logging.getLogger(name)
