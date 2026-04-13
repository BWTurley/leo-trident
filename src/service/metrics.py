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
