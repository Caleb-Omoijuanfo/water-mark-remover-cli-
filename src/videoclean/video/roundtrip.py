"""No-AI video round-trip: extract frames → re-encode → restore audio."""

from __future__ import annotations

import logging
from pathlib import Path

from videoclean.core.job import JobWorkspace
from videoclean.exceptions import InvalidVideoError
from videoclean.video.encoder import encode_frames, extract_audio, mux_video_audio
from videoclean.video.extractor import extract_frames
from videoclean.video.metadata import VideoMetadata, probe

log = logging.getLogger(__name__)


def roundtrip_video(
    input_path: Path | str,
    output_path: Path | str,
    *,
    keep_temp: bool = False,
    metadata: VideoMetadata | None = None,
) -> Path:
    """Round-trip a video through frame extract → encode → audio mux.

    No masking or inpainting is applied. Used to validate the FFmpeg pipeline
    (Phase 2 acceptance) and as the skeleton for the full processing path.

    Parameters
    ----------
    input_path:
        Source video.
    output_path:
        Destination for the reconstructed video.
    keep_temp:
        If True, preserve the job temp directory after completion.
    metadata:
        Optional pre-probed metadata.

    Returns
    -------
    Path
        Resolved path to the output video.
    """
    source = Path(input_path).expanduser().resolve()
    dest = Path(output_path).expanduser().resolve()

    if not source.is_file():
        raise InvalidVideoError(f"Video file not found: {source}")
    if source == dest:
        raise InvalidVideoError(
            "Output path must differ from the input path to avoid overwriting "
            f"the source: {source}"
        )

    meta = metadata or probe(source)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with JobWorkspace.create(keep_temp=keep_temp) as workspace:
        log.info(
            "Round-trip job %s: %s → %s",
            workspace.job_id,
            source.name,
            dest,
        )

        extract_frames(source, workspace.frames_dir, metadata=meta)
        audio = extract_audio(source, workspace.audio_path, metadata=meta)
        encode_frames(
            workspace.frames_dir,
            workspace.video_only_path,
            fps=meta.fps,
        )
        mux_video_audio(
            workspace.video_only_path,
            dest,
            audio_path=audio,
        )

    if not dest.is_file():
        raise InvalidVideoError(f"Round-trip failed to produce output: {dest}")

    log.info("Round-trip complete: %s", dest)
    return dest
