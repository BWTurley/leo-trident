"""
Leo Trident — Metrics Sink

Append-only JSONL metrics, one file per day at data/metrics/{YYYY-MM-DD}.jsonl.
Fire-and-forget: never raises.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import BASE_PATH as _DEFAULT_BASE_PATH

logger = logging.getLogger(__name__)


def log_metric(name: str, value: float | int, tags: dict = None,
               base_path: Path = None) -> None:
    """
    Append a metric line to data/metrics/{YYYY-MM-DD}.jsonl.
    Each line: {"ts": "...", "name": "...", "value": ..., "tags": {...}}
    Never raises; best-effort.
    """
    try:
        bp = base_path or _DEFAULT_BASE_PATH
        metrics_dir = bp / "data" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = metrics_dir / f"{today}.jsonl"

        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "name": name,
            "value": value,
        }
        if tags:
            entry["tags"] = tags

        with open(filepath, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.debug("Metric write failed (best-effort): %s", e)


def read_metrics(date: str | None = None,
                 name_prefix: str | None = None,
                 base_path: Path = None) -> list[dict]:
    """
    Read JSONL metrics for `date` (YYYY-MM-DD, default today UTC).
    Optionally filter to entries whose `name` starts with `name_prefix`.
    Returns [] if the file doesn't exist or on any read error.
    """
    try:
        bp = base_path or _DEFAULT_BASE_PATH
        day = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = bp / "data" / "metrics" / f"{day}.jsonl"
        if not filepath.exists():
            return []
        out: list[dict] = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if name_prefix and not str(entry.get("name", "")).startswith(name_prefix):
                    continue
                out.append(entry)
        return out
    except Exception as e:
        logger.debug("read_metrics failed: %s", e)
        return []


def rollup_metric(name: str, date: str | None = None,
                  agg: str = "sum",
                  base_path: Path = None) -> float:
    """
    Aggregate all events with `name` on `date` (default today UTC).
    agg ∈ {"sum","avg","max","min","count"}. Returns 0.0 if no events.
    """
    entries = [e for e in read_metrics(date=date, base_path=base_path)
               if e.get("name") == name]
    if not entries:
        return 0.0
    values = [float(e.get("value", 0) or 0) for e in entries]
    if agg == "sum":
        return float(sum(values))
    if agg == "avg":
        return float(sum(values) / len(values))
    if agg == "max":
        return float(max(values))
    if agg == "min":
        return float(min(values))
    if agg == "count":
        return float(len(values))
    raise ValueError(f"unknown agg: {agg}")
