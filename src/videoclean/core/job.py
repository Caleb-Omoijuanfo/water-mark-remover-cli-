"""Job workspace, processing config, and temporary directory lifecycle."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from videoclean.exceptions import ConfigError, InvalidVideoError
from videoclean.region import Region

log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Lifecycle status for a single processing job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobState:
    """Track stage-level state for an active pipeline job."""

    total_stages: int
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: JobStatus = JobStatus.PENDING
    current_stage: str | None = None
    completed_stages: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None

    def begin_stage(self, stage: str) -> None:
        """Mark job as running and set the active stage label."""
        if self.started_at is None:
            self.started_at = datetime.now(timezone.utc)
        self.status = JobStatus.RUNNING
        self.current_stage = stage

    def finish_stage(self) -> None:
        """Advance stage progress by one."""
        self.completed_stages = min(
            self.completed_stages + 1, self.total_stages)

    def mark_completed(self) -> None:
        """Set terminal success state."""
        self.status = JobStatus.COMPLETED
        self.current_stage = None
        self.completed_stages = self.total_stages
        self.finished_at = datetime.now(timezone.utc)

    def mark_failed(self, message: str) -> None:
        """Set terminal failure state with a user-safe error message."""
        self.status = JobStatus.FAILED
        self.error = message
        self.finished_at = datetime.now(timezone.utc)

    @property
    def progress(self) -> float:
        """Return normalized progress in the range [0.0, 1.0]."""
        if self.total_stages <= 0:
            return 0.0
        return self.completed_stages / float(self.total_stages)


def default_temp_root() -> Path:
    """Return the root directory for job workspaces.

    Default: ``<system-temp>/videoclean/``. Override with ``VIDEOCLEAN_TEMP``.
    """
    base = os.environ.get("VIDEOCLEAN_TEMP")
    if base:
        return Path(base)
    return Path(tempfile.gettempdir()) / "videoclean"


def default_output_path(input_path: Path) -> Path:
    """Derive ``<stem>_cleaned<suffix>`` next to the input file."""
    path = Path(input_path)
    return path.with_name(f"{path.stem}_cleaned{path.suffix or '.mp4'}")


@dataclass
class ProcessingConfig:
    """User-facing options for a single ``remove`` job.

    Paths are stored as given; the pipeline resolves and validates them.
    """

    input_path: Path
    output_path: Path
    region: Region
    mask_padding: int = 0
    model: str = "opencv"
    device: str = "auto"
    keep_temp: bool = False
    crf: int = 18
    inpaint_radius: int = 3

    def resolved(self) -> ProcessingConfig:
        """Return a copy with expanded absolute paths."""
        return ProcessingConfig(
            input_path=Path(self.input_path).expanduser().resolve(),
            output_path=Path(self.output_path).expanduser().resolve(),
            region=self.region,
            mask_padding=self.mask_padding,
            model=self.model,
            device=self.device,
            keep_temp=self.keep_temp,
            crf=self.crf,
            inpaint_radius=self.inpaint_radius,
        )

    def validate_basic(self) -> None:
        """Validate options that do not require probing the video."""
        if self.mask_padding < 0:
            raise ConfigError(
                f"mask padding must be >= 0, got {self.mask_padding}")
        if self.crf < 0 or self.crf > 51:
            raise ConfigError(f"crf must be in 0..51, got {self.crf}")
        if self.inpaint_radius < 1:
            raise ConfigError(
                f"inpaint radius must be >= 1, got {self.inpaint_radius}"
            )
        if not str(self.model).strip():
            raise ConfigError("model name must be a non-empty string")


@dataclass
class JobWorkspace:
    """Per-job temporary directory layout.

    Layout::

        job-<uuid>/
          frames/          # extracted source frames
          processed/       # frames after inpainting
          audio.m4a        # extracted audio (if present)
          video_only.mp4   # re-encoded video without audio
          mask.png         # static binary mask (optional debug artifact)
    """

    root: Path
    job_id: str
    keep_temp: bool = False
    _cleaned: bool = field(default=False, init=False, repr=False)

    @classmethod
    def create(
        cls,
        *,
        keep_temp: bool = False,
        temp_root: Path | None = None,
    ) -> JobWorkspace:
        """Create a new job directory under the temp root."""
        job_id = uuid.uuid4().hex
        root = (temp_root or default_temp_root()) / f"job-{job_id}"
        root.mkdir(parents=True, exist_ok=False)
        (root / "frames").mkdir()
        (root / "processed").mkdir()
        log.debug("Created job workspace: %s", root)
        return cls(root=root, job_id=job_id, keep_temp=keep_temp)

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def processed_dir(self) -> Path:
        return self.root / "processed"

    @property
    def audio_path(self) -> Path:
        return self.root / "audio.m4a"

    @property
    def video_only_path(self) -> Path:
        return self.root / "video_only.mp4"

    @property
    def mask_path(self) -> Path:
        return self.root / "mask.png"

    def cleanup(self, *, force: bool = False) -> None:
        """Remove the job directory unless ``keep_temp`` is set (or force)."""
        if self._cleaned:
            return
        if self.keep_temp and not force:
            log.info("Keeping temp workspace: %s", self.root)
            return
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=False)
            log.debug("Removed job workspace: %s", self.root)
        self._cleaned = True

    def __enter__(self) -> JobWorkspace:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        # Keep temps on failure when keep_temp is True; always keep if keep_temp.
        # On success (no exception) cleanup unless keep_temp.
        # On failure: keep if keep_temp (debug), else still cleanup by default
        # unless keep_temp was requested for debug of failures too.
        if exc_type is not None and self.keep_temp:
            log.info(
                "Preserving temp workspace after failure (--keep-temp): %s",
                self.root,
            )
            return
        if exc_type is not None and not self.keep_temp:
            # Cleanup on failure unless user asked to keep.
            self.cleanup()
            return
        self.cleanup()


def ensure_io_paths(input_path: Path, output_path: Path) -> tuple[Path, Path]:
    """Resolve and validate input/output paths for a processing job."""
    source = Path(input_path).expanduser().resolve()
    dest = Path(output_path).expanduser().resolve()

    if not source.exists():
        raise InvalidVideoError(f"Video file not found: {source}")
    if not source.is_file():
        raise InvalidVideoError(f"Path is not a file: {source}")
    if source == dest:
        raise InvalidVideoError(
            "Output path must differ from the input path to avoid overwriting "
            f"the source: {source}"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    return source, dest
