"""Video I/O: FFmpeg wrappers, metadata, frame extract/encode, mux."""

from videoclean.video.encoder import assemble_video, encode_frames, extract_audio, mux_video_audio
from videoclean.video.extractor import extract_frames, extract_single_frame, list_frames
from videoclean.video.ffmpeg import ensure_ffmpeg_available, find_ffmpeg, find_ffprobe
from videoclean.video.metadata import VideoMetadata, probe
from videoclean.video.roundtrip import roundtrip_video

__all__ = [
    "VideoMetadata",
    "assemble_video",
    "encode_frames",
    "ensure_ffmpeg_available",
    "extract_audio",
    "extract_frames",
    "extract_single_frame",
    "find_ffmpeg",
    "find_ffprobe",
    "list_frames",
    "mux_video_audio",
    "probe",
    "roundtrip_video",
]
