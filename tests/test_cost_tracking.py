"""Tests for Wave 2E — per-namespace cost tracking."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src import cost_tracking
from src.cost_tracking import (
    cost_breakdown,
    log_embed_call,
    weekly_digest_job,
)
from src.pricing import cost_for


def test_cost_for_known_model():
    assert cost_for("text-embedding-3-large", 10000) == pytest.approx(
        10000 / 1000 * 0.00013
    )


def test_cost_for_unknown_model_zero():
    assert cost_for("mystery-model", 100) == 0.0


def test_cost_for_zero_or_negative_tokens():
    assert cost_for("text-embedding-3-large", 0) == 0.0
    assert cost_for("text-embedding-3-large", -5) == 0.0


def test_log_embed_call(monkeypatch):
    calls: list[tuple] = []

    def fake_log_metric(name, value, tags=None):
        calls.append((name, value, tags))

    monkeypatch.setattr(cost_tracking, "log_metric", fake_log_metric)
    log_embed_call("voyage-3", 5000, "leo-trident", "text")

    assert len(calls) == 2
    names = [c[0] for c in calls]
    assert "embed.tokens" in names
    assert "embed.cost_usd" in names

    tokens_call = next(c for c in calls if c[0] == "embed.tokens")
    cost_call = next(c for c in calls if c[0] == "embed.cost_usd")

    assert tokens_call[1] == 5000
    assert tokens_call[2] == {
        "model": "voyage-3",
        "namespace": "leo-trident",
        "source_kind": "text",
    }
    expected = 5000 / 1000 * 0.00018
    assert cost_call[1] == pytest.approx(expected)
    assert cost_call[2] == {"model": "voyage-3", "namespace": "leo-trident"}


def test_log_embed_call_zero_tokens_noop(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(
        cost_tracking, "log_metric",
        lambda *a, **kw: calls.append((a, kw)),
    )
    log_embed_call("voyage-3", 0)
    assert calls == []


def test_cost_breakdown(tmp_path: Path, monkeypatch):
    """Write fake metrics for today and assert aggregation."""
    metrics_dir = tmp_path / "data" / "metrics"
    metrics_dir.mkdir(parents=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    fp = metrics_dir / f"{today}.jsonl"

    events = [
        {"ts": "x", "name": "embed.cost_usd", "value": 0.10,
         "tags": {"model": "voyage-3", "namespace": "leo"}},
        {"ts": "x", "name": "embed.cost_usd", "value": 0.05,
         "tags": {"model": "voyage-3", "namespace": "leo"}},
        {"ts": "x", "name": "embed.cost_usd", "value": 0.02,
         "tags": {"model": "text-embedding-3-large", "namespace": "asme"}},
        # noise — should be ignored
        {"ts": "x", "name": "embed.tokens", "value": 1234,
         "tags": {"model": "voyage-3", "namespace": "leo"}},
        {"ts": "x", "name": "scheduler.run", "value": 1.0},
    ]
    with open(fp, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    # Repoint metrics base_path → tmp_path. read_metrics resolves base_path
    # via src.config.BASE_PATH; monkeypatch that.
    import src.service.metrics as metrics_mod
    monkeypatch.setattr(metrics_mod, "_DEFAULT_BASE_PATH", tmp_path)

    report = cost_breakdown(1)
    assert report["days"] == 1
    assert report["by_namespace"]["leo"] == pytest.approx(0.15)
    assert report["by_namespace"]["asme"] == pytest.approx(0.02)
    assert report["by_model"]["voyage-3"] == pytest.approx(0.15)
    assert report["by_model"]["text-embedding-3-large"] == pytest.approx(0.02)
    assert report["total_usd"] == pytest.approx(0.17)


def test_cost_breakdown_multi_day(tmp_path: Path, monkeypatch):
    metrics_dir = tmp_path / "data" / "metrics"
    metrics_dir.mkdir(parents=True)

    def write(day: str, val: float, ns: str):
        fp = metrics_dir / f"{day}.jsonl"
        with open(fp, "a") as f:
            f.write(json.dumps({
                "ts": "x", "name": "embed.cost_usd", "value": val,
                "tags": {"model": "voyage-3", "namespace": ns},
            }) + "\n")

    today = datetime.now(timezone.utc).date()
    write(today.isoformat(), 0.01, "ns_today")
    write((today - timedelta(days=2)).isoformat(), 0.04, "ns_old")
    # Outside the 2-day window → must be excluded.
    write((today - timedelta(days=10)).isoformat(), 99.0, "ns_ancient")

    import src.service.metrics as metrics_mod
    monkeypatch.setattr(metrics_mod, "_DEFAULT_BASE_PATH", tmp_path)

    report = cost_breakdown(2)
    assert report["total_usd"] == pytest.approx(0.01)
    report3 = cost_breakdown(3)
    assert report3["total_usd"] == pytest.approx(0.05)
    assert "ns_ancient" not in report3["by_namespace"]


def test_weekly_digest_format(monkeypatch):
    sent_msgs: list[str] = []

    def fake_breakdown(days=7):
        return {
            "days": 7,
            "by_namespace": {"leo": 0.42, "asme": 0.10},
            "by_model": {"voyage-3": 0.52},
            "total_usd": 0.52,
        }

    def fake_notify(text, parse_mode="Markdown"):
        sent_msgs.append(text)
        return True

    monkeypatch.setattr(cost_tracking, "cost_breakdown", fake_breakdown)
    import src.notify as notify_mod
    monkeypatch.setattr(notify_mod, "notify_telegram", fake_notify)

    result = weekly_digest_job()
    assert result["sent"] is True
    assert len(sent_msgs) == 1
    msg = sent_msgs[0]
    assert "$" in msg
    assert "0.5200" in msg  # total formatted
    assert "leo" in msg
    assert "voyage-3" in msg
