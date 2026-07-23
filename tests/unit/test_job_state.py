"""Unit tests for JobState lifecycle tracking."""

from __future__ import annotations

from videoclean.core.job import JobState, JobStatus


def test_job_state_progress_lifecycle() -> None:
    state = JobState(total_stages=4)
    assert state.status is JobStatus.PENDING
    assert state.progress == 0.0

    state.begin_stage("Validate inputs")
    assert state.status is JobStatus.RUNNING
    assert state.current_stage == "Validate inputs"
    assert state.started_at is not None

    state.finish_stage()
    assert state.completed_stages == 1
    assert state.progress == 0.25

    state.mark_completed()
    assert state.status is JobStatus.COMPLETED
    assert state.completed_stages == 4
    assert state.progress == 1.0
    assert state.current_stage is None
    assert state.finished_at is not None


def test_job_state_mark_failed() -> None:
    state = JobState(total_stages=3)
    state.begin_stage("Inpaint frames")
    state.finish_stage()
    state.mark_failed("boom")

    assert state.status is JobStatus.FAILED
    assert state.error == "boom"
    assert state.finished_at is not None
