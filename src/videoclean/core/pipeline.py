"""End-to-end watermark removal pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from typing import Callable

import cv2
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from videoclean.core.job import (
    JobWorkspace,
    JobState,
    ProcessingConfig,
    ensure_io_paths,
)
from videoclean.exceptions import ProcessingError
from videoclean.masks.manual import create_mask, validate_region_for_video
from videoclean.models.registry import get_engine
from videoclean.region import Region
from videoclean.utils.device import describe_device, resolve_device
from videoclean.utils.logging import get_console
from videoclean.video.encoder import encode_frames, extract_audio, mux_video_audio
from videoclean.video.extractor import extract_frames
from videoclean.video.ffmpeg import ensure_ffmpeg_available
from videoclean.video.metadata import VideoMetadata, probe

log = logging.getLogger(__name__)

# Ordered stages shown in the Rich progress bar (one step each, not per-frame).
PIPELINE_STAGES: tuple[str, ...] = (
    "Validate inputs",
    "Probe metadata",
    "Validate region",
    "Select device",
    "Create workspace",
    "Extract frames",
    "Extract audio",
    "Build mask",
    "Load model",
    "Inpaint frames",
    "Encode video",
    "Mux audio",
    "Verify output",
    "Cleanup",
)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Outcome of a successful :func:`run_pipeline` call."""

    output_path: Path
    input_path: Path
    region: Region
    effective_region: Region
    metadata: VideoMetadata
    model: str
    device: str
    frame_count: int
    job_id: str
    kept_temp: Path | None = None
    job_state: JobState | None = None


ProgressCallback = Callable[[str, int, int], None]
"""Optional ``(stage_name, completed_stages, total_stages)`` hook."""


def run_pipeline(
    config: ProcessingConfig,
    *,
    console: Console | None = None,
    show_progress: bool = True,
    progress_callback: ProgressCallback | None = None,
    temp_root: Path | None = None,
) -> PipelineResult:
    """Run the full remove pipeline for *config*.

    Stages (progress bar advances once per stage, not per frame)::

        validate → metadata → region → device → temp dir → extract frames →
        extract audio → mask → load model → inpaint → encode → mux →
        verify → cleanup
    """
    cfg = config.resolved()
    cfg.validate_basic()
    ui = console if console is not None else get_console()

    total = len(PIPELINE_STAGES)
    completed = 0
    job_state = JobState(total_stages=total)
    progress: Progress | None = None
    task_id: TaskID = TaskID(0)
    workspace: JobWorkspace | None = None

    def begin_stage(name: str) -> None:
        job_state.begin_stage(name)
        log.info("[%d/%d] %s", completed + 1, total, name)
        if progress is not None:
            progress.update(task_id, description=f"[cyan]{name}[/cyan]")

    def finish_stage(name: str) -> None:
        nonlocal completed
        completed += 1
        job_state.finish_stage()
        if progress_callback is not None:
            progress_callback(name, completed, total)
        if progress is not None:
            progress.advance(task_id)

    if show_progress:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=ui,
            transient=False,
        )
        progress.start()
        task_id = cast(
            TaskID,
            progress.add_task("[cyan]Starting…[/cyan]", total=total),
        )

    try:
        # 1. Validate inputs + FFmpeg
        begin_stage("Validate inputs")
        ensure_ffmpeg_available()
        source, dest = ensure_io_paths(cfg.input_path, cfg.output_path)
        finish_stage("Validate inputs")

        # 2. Metadata
        begin_stage("Probe metadata")
        meta = probe(source)
        finish_stage("Probe metadata")

        # 3. Region
        begin_stage("Validate region")
        effective = validate_region_for_video(
            cfg.region,
            meta.width,
            meta.height,
            padding=cfg.mask_padding,
        )
        finish_stage("Validate region")

        # 4. Device
        begin_stage("Select device")
        device = resolve_device(cfg.device)
        finish_stage("Select device")

        region_note = str(cfg.region)
        if effective.as_tuple() != cfg.region.as_tuple():
            region_note = f"{cfg.region} → {effective} (padded)"
        # Print summary once we know meta/device (below the progress bar area).
        log.info(
            "Job plan: %s %s → %s | model=%s device=%s region=%s frames≈%s",
            source.name,
            meta.resolution,
            dest.name,
            cfg.model,
            device,
            region_note,
            meta.frame_count,
        )

        # 5. Workspace
        begin_stage("Create workspace")
        workspace = JobWorkspace.create(
            keep_temp=cfg.keep_temp,
            temp_root=temp_root,
        )
        finish_stage("Create workspace")

        # 6. Extract frames
        begin_stage("Extract frames")
        frames = extract_frames(source, workspace.frames_dir, metadata=meta)
        if not frames:
            raise ProcessingError(f"No frames extracted from: {source}")
        finish_stage("Extract frames")

        # 7. Audio
        begin_stage("Extract audio")
        audio = extract_audio(source, workspace.audio_path, metadata=meta)
        finish_stage("Extract audio")

        # 8. Mask
        begin_stage("Build mask")
        mask = create_mask(
            meta.height,
            meta.width,
            cfg.region,
            padding=cfg.mask_padding,
        )
        if cfg.keep_temp:
            cv2.imwrite(str(workspace.mask_path), mask)
        finish_stage("Build mask")

        # 9. Load model
        begin_stage("Load model")
        engine = get_engine(
            cfg.model,
            radius=cfg.inpaint_radius,
            device=device,
        )
        engine.load()
        finish_stage("Load model")

        # 10. Inpaint (stage-level progress only — not per-frame)
        begin_stage("Inpaint frames")
        try:
            processed = engine.process_video(
                workspace.frames_dir,
                workspace.processed_dir,
                mask,
            )
        finally:
            engine.unload()

        if len(processed) != len(frames):
            raise ProcessingError(
                f"Inpainting produced {len(processed)} frame(s) but "
                f"{len(frames)} were extracted"
            )
        finish_stage("Inpaint frames")

        # 11. Encode
        begin_stage("Encode video")
        encode_frames(
            workspace.processed_dir,
            workspace.video_only_path,
            fps=meta.fps,
            crf=cfg.crf,
        )
        finish_stage("Encode video")

        # 12. Mux
        begin_stage("Mux audio")
        mux_video_audio(
            workspace.video_only_path,
            dest,
            audio_path=audio,
        )
        finish_stage("Mux audio")

        # 13. Verify
        begin_stage("Verify output")
        if not dest.is_file() or dest.stat().st_size == 0:
            raise ProcessingError(f"Pipeline failed to produce output: {dest}")
        out_meta = probe(dest)
        if out_meta.width != meta.width or out_meta.height != meta.height:
            raise ProcessingError(
                f"Output resolution {out_meta.resolution} does not match "
                f"input {meta.resolution}"
            )
        finish_stage("Verify output")

        # 14. Cleanup
        begin_stage("Cleanup")
        job_id = workspace.job_id
        job_state.job_id = job_id
        kept: Path | None = workspace.root if cfg.keep_temp else None
        workspace.cleanup()
        workspace = None  # prevent double-cleanup in except
        finish_stage("Cleanup")
        job_state.mark_completed()

        result = PipelineResult(
            output_path=dest,
            input_path=source,
            region=cfg.region,
            effective_region=effective,
            metadata=meta,
            model=cfg.model,
            device=device,
            frame_count=len(frames),
            job_id=job_id,
            kept_temp=kept,
            job_state=job_state,
        )

        log.info(
            "Pipeline complete: %s (%d frames, model=%s, device=%s)",
            dest,
            result.frame_count,
            cfg.model,
            device,
        )
        return result
    except Exception as exc:
        job_state.mark_failed(str(exc))
        if workspace is not None:
            if cfg.keep_temp:
                log.info(
                    "Preserving temp workspace after failure (--keep-temp): %s",
                    workspace.root,
                )
            else:
                workspace.cleanup()
        raise
    finally:
        if progress is not None:
            progress.stop()
