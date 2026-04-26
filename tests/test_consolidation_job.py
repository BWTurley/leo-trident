"""Tests for src.jobs.consolidation.nightly_consolidation_job."""
from __future__ import annotations

from unittest.mock import patch

from src.jobs import consolidation as job

SAMPLE_SUMMARY = {
    "facts_extracted": [
        {"action": "ADD", "fact": "f1"},
        {"action": "UPDATE", "fact": "f2"},
        {"action": "NOOP", "fact": "f3"},
        {"action": "DELETE", "fact": "f4"},
        "not-a-dict",
    ],
    "tier_changes": [
        {"chunk_id": "a", "from_tier": "warm", "to_tier": "cold"},
        {"chunk_id": "b", "from_tier": "cold", "to_tier": "warm"},
        {"chunk_id": "c", "from_tier": "warm", "to_tier": "cold"},
    ],
    "errors": [],
}


def test_nightly_consolidation_job_happy_path():
    metric_calls: list[tuple] = []
    notify_calls: list[str] = []

    def fake_log_metric(name, value, tags=None, **kw):
        metric_calls.append((name, value, tags))

    def fake_notify(text, parse_mode="Markdown"):
        notify_calls.append(text)
        return True

    with patch.object(job, "_safe_consolidate_call", return_value=SAMPLE_SUMMARY), \
         patch.object(job, "log_metric", side_effect=fake_log_metric), \
         patch.object(job, "notify_telegram", side_effect=fake_notify):
        result = job.nightly_consolidation_job()

    assert result["ok"] is True
    assert result["chunks_pruned"] == 2  # two to_tier == cold
    assert result["chunks_merged"] == 2  # ADD + UPDATE
    assert result["duration_ms"] >= 0

    names = [c[0] for c in metric_calls]
    assert "consolidation.duration_ms" in names
    assert "consolidation.chunks_pruned" in names
    assert "consolidation.chunks_merged" in names

    by_name = {c[0]: c[1] for c in metric_calls}
    assert by_name["consolidation.chunks_pruned"] == 2
    assert by_name["consolidation.chunks_merged"] == 2

    assert len(notify_calls) == 1
    msg = notify_calls[0]
    assert "🌙" in msg
    assert "pruned 2" in msg
    assert "merged 2" in msg
    assert "ms" in msg


def test_nightly_consolidation_job_handles_exception():
    metric_calls: list[tuple] = []
    notify_calls: list[str] = []

    def boom():
        raise RuntimeError("kaboom")

    def fake_log_metric(name, value, tags=None, **kw):
        metric_calls.append((name, value, tags))

    def fake_notify(text, parse_mode="Markdown"):
        notify_calls.append(text)
        return True

    with patch.object(job, "_safe_consolidate_call", side_effect=boom), \
         patch.object(job, "log_metric", side_effect=fake_log_metric), \
         patch.object(job, "notify_telegram", side_effect=fake_notify):
        # Must not raise.
        result = job.nightly_consolidation_job()

    assert result["ok"] is False
    assert "kaboom" in result.get("error", "")
    # Failure path still emits a duration metric (with status=error tag).
    assert any(c[0] == "consolidation.duration_ms" for c in metric_calls)
    err_metric = next(c for c in metric_calls if c[0] == "consolidation.duration_ms")
    assert (err_metric[2] or {}).get("status") == "error"
    # And alerts the user.
    assert len(notify_calls) == 1
    assert "FAILED" in notify_calls[0]
    assert "kaboom" in notify_calls[0]


def test_register_default_jobs_registers_consolidation():
    from src import jobs, scheduler

    # Clean slate to avoid pollution from other tests.
    scheduler.unregister("consolidation_nightly")
    jobs.register_default_jobs()

    names = [j["name"] for j in scheduler.jobs()]
    assert "consolidation_nightly" in names

    entry = next(j for j in scheduler.jobs() if j["name"] == "consolidation_nightly")
    assert entry["schedule"] == "@daily 03:00"

    scheduler.unregister("consolidation_nightly")
