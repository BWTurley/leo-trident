"""Tests for src.quality — drift telemetry / golden-query MRR/Recall."""
from __future__ import annotations

import json

import pytest

from src import quality

# ── Fixtures ──────────────────────────────────────────────────────────────

GOLDEN_TINY = [
    {"query": "submarine", "expected_chunk_ids": ["pennsylvania"]},
    {"query": "job", "expected_chunk_ids": ["asme"]},
    {"query": "business", "expected_chunk_ids": ["platelabs"]},
    {"query": "persona", "expected_chunk_ids": ["leo", "goth"]},
]


class FakeTrident:
    """Maps a query string -> a list of result dicts."""

    def __init__(self, mapping: dict[str, list[dict]]):
        self.mapping = mapping
        self.calls: list[tuple[str, int]] = []

    def query(self, text: str, top_k: int = 10, **kw):
        self.calls.append((text, top_k))
        return list(self.mapping.get(text, []))


# ── run_quality_snapshot ──────────────────────────────────────────────────

def test_run_quality_snapshot_with_mock_trident():
    """Hand-built scenario: 4 queries, 3 hits, ranks 1, 3, miss, 2.

    MRR = (1/1 + 1/3 + 0 + 1/2) / 4 = (1.0 + 0.3333 + 0 + 0.5) / 4 = 0.4583
    Recall@10 = 3/4 = 0.75
    """
    fake = FakeTrident({
        "submarine": [
            {"chunk_id": "uss-pennsylvania-001", "text": "boat"},      # rank 1 hit
        ],
        "job": [
            {"chunk_id": "x", "text": "blah"},
            {"chunk_id": "y", "text": "more"},
            {"chunk_id": "asme-trainee-7", "text": "code"},            # rank 3 hit
        ],
        "business": [
            {"chunk_id": "n", "text": "not a match"},
        ],
        "persona": [
            {"chunk_id": "z", "text": "irrelevant"},
            {"chunk_id": "a", "text": "leo dommy mommy"},              # rank 2 hit (text)
        ],
    })

    snap = quality.run_quality_snapshot(trident=fake, golden=GOLDEN_TINY)

    assert snap["n_queries"] == 4
    assert snap["k"] == 10
    assert snap["recall_at_10"] == pytest.approx(0.75)
    expected_mrr = (1.0 + 1 / 3 + 0.0 + 0.5) / 4
    assert snap["mrr_at_10"] == pytest.approx(expected_mrr)

    per = snap["per_query"]
    assert per[0]["rank"] == 1 and per[0]["hit"] is True
    assert per[1]["rank"] == 3 and per[1]["hit"] is True
    assert per[2]["rank"] is None and per[2]["hit"] is False
    assert per[3]["rank"] == 2 and per[3]["hit"] is True


def test_run_quality_snapshot_handles_query_exception():
    class Boom:
        def query(self, text, top_k=10, **kw):
            raise RuntimeError("nope")

    snap = quality.run_quality_snapshot(
        trident=Boom(),
        golden=[{"query": "x", "expected_chunk_ids": ["y"]}],
    )
    assert snap["n_queries"] == 1
    assert snap["mrr_at_10"] == 0.0
    assert snap["recall_at_10"] == 0.0
    assert snap["per_query"][0]["hit"] is False


def test_load_golden_queries_real_fixture():
    items = quality.load_golden_queries()
    assert isinstance(items, list)
    assert 8 <= len(items) <= 12
    for it in items:
        assert "query" in it
        assert "expected_chunk_ids" in it
        assert isinstance(it["expected_chunk_ids"], list)


def test_load_golden_queries_missing(tmp_path):
    items = quality.load_golden_queries(path=tmp_path / "nope.json")
    assert items == []


def test_load_golden_queries_bad_shape(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"not": "a list"}))
    assert quality.load_golden_queries(path=p) == []


# ── daily_quality_job ─────────────────────────────────────────────────────

def _patch_metrics(monkeypatch, *, baseline_mrr=None, baseline_recall=None):
    """Wire up log_metric/notify/_rolling_baseline mocks; return capture dicts."""
    logged: list[tuple[str, float, dict | None]] = []
    notifications: list[str] = []

    def fake_log_metric(name, value, tags=None, base_path=None):
        logged.append((name, value, tags))

    def fake_notify(text, parse_mode="Markdown"):
        notifications.append(text)
        return True

    def fake_baseline(metric_name, days=7, skip_today=True):
        if metric_name == "quality.mrr":
            return baseline_mrr
        if metric_name == "quality.recall":
            return baseline_recall
        return None

    monkeypatch.setattr(quality, "log_metric", fake_log_metric)
    monkeypatch.setattr(quality, "notify_telegram", fake_notify)
    monkeypatch.setattr(quality, "_rolling_baseline", fake_baseline)

    return logged, notifications


def test_daily_quality_job_logs_metrics(monkeypatch):
    logged, notifications = _patch_metrics(monkeypatch)

    monkeypatch.setattr(
        quality,
        "run_quality_snapshot",
        lambda: {"mrr_at_10": 0.6, "recall_at_10": 0.7, "n_queries": 5, "k": 10,
                 "per_query": [], "ts": "now"},
    )

    out = quality.daily_quality_job()
    assert out["ok"] is True
    assert out["mrr_at_10"] == 0.6
    assert out["recall_at_10"] == 0.7
    assert out["alerted"] is False

    names = [n for n, _, _ in logged]
    assert "quality.mrr" in names
    assert "quality.recall" in names
    # Tags include k=10
    by_name = {n: (v, t) for n, v, t in logged}
    assert by_name["quality.mrr"] == (0.6, {"k": 10})
    assert by_name["quality.recall"] == (0.7, {"k": 10})
    assert notifications == []


def test_alert_on_regression(monkeypatch):
    """Today=0.5 vs baseline=0.8 → 37.5% drop → alert fires."""
    logged, notifications = _patch_metrics(
        monkeypatch, baseline_mrr=0.8, baseline_recall=0.8,
    )
    monkeypatch.setattr(
        quality,
        "run_quality_snapshot",
        lambda: {"mrr_at_10": 0.5, "recall_at_10": 0.5, "n_queries": 10,
                 "k": 10, "per_query": [], "ts": "now"},
    )

    out = quality.daily_quality_job()
    assert out["ok"] is True
    assert out["alerted"] is True
    assert len(notifications) == 1
    msg = notifications[0]
    assert "regression" in msg.lower()
    assert "MRR@10" in msg
    assert "Recall@10" in msg


def test_no_alert_when_stable(monkeypatch):
    """Today=0.78 vs baseline=0.8 → 2.5% drop → no alert."""
    logged, notifications = _patch_metrics(
        monkeypatch, baseline_mrr=0.8, baseline_recall=0.8,
    )
    monkeypatch.setattr(
        quality,
        "run_quality_snapshot",
        lambda: {"mrr_at_10": 0.78, "recall_at_10": 0.78, "n_queries": 10,
                 "k": 10, "per_query": [], "ts": "now"},
    )

    out = quality.daily_quality_job()
    assert out["ok"] is True
    assert out["alerted"] is False
    assert notifications == []


def test_no_alert_when_no_baseline(monkeypatch):
    """Fresh deploy: no historical data → no alert even if today is low."""
    logged, notifications = _patch_metrics(
        monkeypatch, baseline_mrr=None, baseline_recall=None,
    )
    monkeypatch.setattr(
        quality,
        "run_quality_snapshot",
        lambda: {"mrr_at_10": 0.1, "recall_at_10": 0.1, "n_queries": 10,
                 "k": 10, "per_query": [], "ts": "now"},
    )

    out = quality.daily_quality_job()
    assert out["alerted"] is False
    assert notifications == []


def test_daily_quality_job_swallows_snapshot_errors(monkeypatch):
    def boom():
        raise RuntimeError("explode")

    monkeypatch.setattr(quality, "run_quality_snapshot", boom)
    out = quality.daily_quality_job()
    assert out["ok"] is False
    assert out["error"] == "RuntimeError"


# ── Rolling baseline ──────────────────────────────────────────────────────

def test_rolling_baseline_uses_metrics_sink(monkeypatch, tmp_path):
    """Write some yesterday metrics, ensure baseline averages them."""
    from datetime import datetime, timedelta, timezone

    from src.service import metrics as ms

    base = tmp_path
    (base / "data" / "metrics").mkdir(parents=True)

    # Write 3 days of history
    today = datetime.now(timezone.utc).date()
    for i, val in enumerate([0.8, 0.9, 0.7], start=1):
        day = (today - timedelta(days=i)).isoformat()
        fp = base / "data" / "metrics" / f"{day}.jsonl"
        fp.write_text(json.dumps(
            {"ts": "x", "name": "quality.mrr", "value": val}
        ) + "\n")

    monkeypatch.setattr(ms, "_DEFAULT_BASE_PATH", base)

    avg = quality._rolling_baseline("quality.mrr", days=7)
    assert avg == pytest.approx((0.8 + 0.9 + 0.7) / 3)


# ── Endpoint ──────────────────────────────────────────────────────────────

def test_admin_quality_snapshot_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    from src.service import api as service_api

    fixed = {"mrr_at_10": 0.42, "recall_at_10": 0.5, "n_queries": 4, "k": 10,
             "per_query": [], "ts": "now"}
    monkeypatch.setattr(
        "src.quality.run_quality_snapshot", lambda *a, **kw: fixed,
    )

    client = TestClient(service_api.app)
    resp = client.post("/admin/quality/snapshot")
    assert resp.status_code == 200
    body = resp.json()
    assert body["mrr_at_10"] == 0.42
    assert body["recall_at_10"] == 0.5
    assert body["n_queries"] == 4
