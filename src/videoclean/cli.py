"""VideoClean command-line interface (Typer)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import typer
from rich.table import Table
from typer.core import TyperGroup

from videoclean import __version__
from videoclean.core.job import ProcessingConfig, default_output_path
from videoclean.core.pipeline import run_pipeline
from videoclean.exceptions import ConfigError, InvalidVideoError, VideoCleanError
from videoclean.masks.manual import validate_region_for_video
from videoclean.masks.preview import render_preview_image
from videoclean.models.registry import DEFAULT_MODEL, list_models
from videoclean.region import Region
from videoclean.utils.device import describe_device
from videoclean.utils.logging import get_console, get_logger, setup_logging
from videoclean.video.ffmpeg import ensure_ffmpeg_available
from videoclean.video.metadata import probe


class _FriendlyErrorGroup(TyperGroup):
    """Convert :class:`VideoCleanError` into a clean non-zero CLI exit.

    Ensures invalid regions and other user-facing failures print a readable
    message (never a raw stack trace), both for the real entry point and when
    the Typer app is invoked directly (e.g. tests).
    """

    def invoke(self, ctx: Any) -> Any:
        try:
            return super().invoke(ctx)
        except VideoCleanError as exc:
            _print_error(exc)
            raise typer.Exit(code=1) from None


app = typer.Typer(
    name="videoclean",
    help="Remove watermarks from videos using local inpainting models.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    cls=_FriendlyErrorGroup,
)

log = get_logger(__name__)


def _print_error(exc: VideoCleanError) -> None:
    """Print a human-readable error to the shared console."""
    if not logging.getLogger().handlers:
        setup_logging(verbose=False)
    get_console().print(f"[bold red]Error:[/bold red] {exc.message}")


@app.callback()
def global_options(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose (DEBUG) logging.",
        show_default=False,
    ),
) -> None:
    """VideoClean — local video watermark removal."""
    setup_logging(verbose=verbose)


@app.command("info")
def info_cmd(
    input_path: Path = typer.Argument(
        ...,
        metavar="INPUT",
        help="Path to the input video file.",
        exists=False,
        dir_okay=False,
        readable=False,
    ),
) -> None:
    """Show metadata for a video file (via ffprobe)."""
    console = get_console()
    ensure_ffmpeg_available()

    path = input_path.expanduser().resolve()
    if not path.exists():
        raise InvalidVideoError(f"Video file not found: {path}")
    if not path.is_file():
        raise InvalidVideoError(f"Path is not a file: {path}")

    meta = probe(path)
    size_bytes = path.stat().st_size
    size_human = _format_bytes(size_bytes)

    table = Table(title="Video info", show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Path", str(path))
    table.add_row("Name", path.name)
    table.add_row("Size", f"{size_human} ({size_bytes:,} bytes)")
    table.add_row("Resolution", meta.resolution)
    table.add_row("Duration", f"{meta.duration:.3f} s")
    table.add_row("FPS", f"{meta.fps:.3f}")
    table.add_row(
        "Frames",
        str(meta.frame_count) if meta.frame_count is not None else "unknown",
    )
    table.add_row("Video codec", meta.video_codec)
    table.add_row("Pixel format", meta.pixel_format or "unknown")
    if meta.has_audio:
        audio_bits = meta.audio_codec or "unknown"
        if meta.audio_sample_rate:
            audio_bits += f", {meta.audio_sample_rate} Hz"
        if meta.audio_channels:
            audio_bits += f", {meta.audio_channels} ch"
        table.add_row("Audio", audio_bits)
    else:
        table.add_row("Audio", "(none)")
    if meta.format_name:
        table.add_row("Format", meta.format_name)
    if meta.bit_rate:
        table.add_row("Bit rate", f"{meta.bit_rate:,} bps")

    console.print(table)
    log.debug("info completed for %s", path)


@app.command("remove")
def remove_cmd(
    input_path: Path = typer.Argument(
        ...,
        metavar="INPUT",
        help="Path to the input video file.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path for the cleaned output video "
        "(default: <input>_cleaned.<ext>).",
    ),
    region: str = typer.Option(
        ...,
        "--region",
        "-r",
        help="Watermark region as x1,y1,x2,y2 (e.g. 20,20,300,120).",
    ),
    mask_padding: int = typer.Option(
        0,
        "--mask-padding",
        help="Extra pixels to expand the mask around the region (default: 0).",
        min=0,
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        "-m",
        help=f"Inpainting model (default: {DEFAULT_MODEL}). "
        f"Available: {', '.join(list_models())}.",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Compute device: auto, cpu, cuda, or mps (default: auto).",
    ),
    keep_temp: bool = typer.Option(
        False,
        "--keep-temp",
        help="Keep temporary working files after processing.",
    ),
    crf: int = typer.Option(
        18,
        "--crf",
        help="libx264 CRF quality (0–51, lower = higher quality; default: 18).",
        min=0,
        max=51,
    ),
) -> None:
    """Remove a watermark region from a video via local inpainting."""
    console = get_console()
    ensure_ffmpeg_available()

    path = input_path.expanduser().resolve()
    if not path.exists():
        raise InvalidVideoError(f"Video file not found: {path}")
    if not path.is_file():
        raise InvalidVideoError(f"Path is not a file: {path}")

    parsed_region = Region.parse(region)
    out = (output if output is not None else default_output_path(path)).expanduser()

    console.print(
        f"[bold]Removing watermark[/bold] from [cyan]{path.name}[/cyan] → "
        f"[cyan]{out}[/cyan]"
    )
    console.print(
        f"  region={parsed_region}  padding={mask_padding}  "
        f"model={model}  device={device}"
    )

    config = ProcessingConfig(
        input_path=path,
        output_path=out,
        region=parsed_region,
        mask_padding=mask_padding,
        model=model,
        device=device,
        keep_temp=keep_temp,
        crf=crf,
    )
    result = run_pipeline(config, console=console, show_progress=True)

    console.print(
        f"[green]Done.[/green] Wrote [bold]{result.output_path}[/bold] "
        f"({result.frame_count} frames, {result.metadata.resolution}, "
        f"{describe_device(result.device)})"
    )
    if result.effective_region.as_tuple() != result.region.as_tuple():
        console.print(
            f"  effective region after padding: {result.effective_region}"
        )
    if result.kept_temp is not None:
        console.print(f"  [dim]temp kept:[/dim] {result.kept_temp}")


@app.command("preview")
def preview_cmd(
    input_path: Path = typer.Argument(
        ...,
        metavar="INPUT",
        help="Path to the input video file.",
    ),
    region: str = typer.Option(
        ...,
        "--region",
        "-r",
        help="Watermark region as x1,y1,x2,y2 (e.g. 20,20,300,120).",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path for the preview image (default: preview.png).",
    ),
    mask_padding: int = typer.Option(
        0,
        "--mask-padding",
        help="Extra pixels to expand the mask around the region (default: 0).",
        min=0,
    ),
    frame_index: int = typer.Option(
        0,
        "--frame",
        help="Zero-based frame index to preview (default: 0).",
        min=0,
    ),
    time: Optional[float] = typer.Option(
        None,
        "--time",
        help="Optional timestamp in seconds (overrides --frame).",
    ),
    save_mask: bool = typer.Option(
        False,
        "--save-mask",
        help="Also write a binary mask image next to the preview.",
    ),
) -> None:
    """Render a preview of the watermark region and mask overlay."""
    console = get_console()
    ensure_ffmpeg_available()

    path = input_path.expanduser().resolve()
    if not path.exists():
        raise InvalidVideoError(f"Video file not found: {path}")
    if not path.is_file():
        raise InvalidVideoError(f"Path is not a file: {path}")

    if time is not None and time < 0:
        raise ConfigError(f"--time must be >= 0, got {time}")

    parsed_region = Region.parse(region)
    out = (output or Path("preview.png")).expanduser()

    meta = probe(path)
    # Fail fast with a clear bounds error before extracting a frame.
    validate_region_for_video(
        parsed_region,
        meta.width,
        meta.height,
        padding=mask_padding,
    )

    result = render_preview_image(
        path,
        parsed_region,
        out,
        padding=mask_padding,
        frame_index=frame_index,
        time_seconds=time,
        metadata=meta,
        save_mask=save_mask,
    )

    console.print(f"[green]Wrote preview:[/green] {result}")
    console.print(
        f"  region: {parsed_region}  "
        f"({parsed_region.width}x{parsed_region.height})  "
        f"frame={meta.resolution}  padding={mask_padding}"
    )
    if save_mask:
        mask_path = result.with_name(f"{result.stem}_mask{result.suffix or '.png'}")
        console.print(f"  mask:   {mask_path}")


@app.command("version")
def version_cmd() -> None:
    """Print the VideoClean version and exit."""
    get_console().print(f"videoclean {__version__}")


def _format_bytes(num: int) -> str:
    """Format a byte count as a human-readable string."""
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num} B"


def main() -> None:
    """CLI entry point with human-readable error handling.

    Command-level :class:`VideoCleanError` instances are handled by
    :class:`_FriendlyErrorGroup`. This safety net covers failures that escape
    the Typer group (e.g. during very early startup).
    """
    try:
        app()
    except VideoCleanError as exc:
        _print_error(exc)
        raise SystemExit(1) from None
    except typer.Exit:
        raise
    except SystemExit:
        raise


if __name__ == "__main__":
    main()
