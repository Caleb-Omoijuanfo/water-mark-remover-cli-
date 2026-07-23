"""Unit tests for JobWorkspace temp lifecycle."""

from __future__ import annotations

from pathlib import Path

from videoclean.core.job import JobWorkspace


def test_create_layout(tmp_path: Path) -> None:
    ws = JobWorkspace.create(temp_root=tmp_path)
    assert ws.root.exists()
    assert ws.frames_dir.is_dir()
    assert ws.processed_dir.is_dir()
    assert ws.root.name.startswith("job-")
    assert ws.audio_path.name == "audio.m4a"
    assert ws.video_only_path.name == "video_only.mp4"
    ws.cleanup(force=True)
    assert not ws.root.exists()


def test_cleanup_respects_keep_temp(tmp_path: Path) -> None:
    ws = JobWorkspace.create(temp_root=tmp_path, keep_temp=True)
    root = ws.root
    ws.cleanup()
    assert root.exists()
    ws.cleanup(force=True)
    assert not root.exists()


def test_context_manager_cleans_on_success(tmp_path: Path) -> None:
    with JobWorkspace.create(temp_root=tmp_path) as ws:
        root = ws.root
        assert root.exists()
    assert not root.exists()


def test_context_manager_keeps_on_failure_when_requested(tmp_path: Path) -> None:
    root: Path | None = None
    try:
        with JobWorkspace.create(temp_root=tmp_path, keep_temp=True) as ws:
            root = ws.root
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert root is not None and root.exists()
    # manual force cleanup
    JobWorkspace(root=root, job_id="x", keep_temp=False).cleanup(force=True)
