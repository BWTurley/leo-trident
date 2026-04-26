"""
Leo Trident — Nightly Sleep-Time Consolidation Job

Wraps `SleepTimeConsolidator.run()` with metrics + Telegram notification.
Designed to be invoked by `src.scheduler` on a `@daily 03:00` cadence, or
inline via the `/admin/consolidate/run-now` admin endpoint.

Contract:
- Never raises. Failures are logged + reported to Telegram.
- Always emits these metrics:
    consolidation.duration_ms
    consolidation.chunks_pruned     (tier demotions to 'cold')
    consolidation.chunks_merged     (facts with action ADD or UPDATE)
- Returns the metric counts dict so callers (admin endpoint) can surface them.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from src.notify import notify_telegram
from src.service.metrics import log_metric

logger = logging.getLogger(__name__)


def _safe_consolidate_call() -> dict[str, Any]:
    """Instantiate and run the consolidator. Returns its summary dict.

    Kept as a tiny seam so tests can monkeypatch this without having to
    reach inside the consolidator module.
    """
    from src.memory.consolidator import SleepTimeConsolidator

    consolidator = SleepTimeConsolidator()
    return consolidator.run() or {}


def _count_pruned(summary: dict[str, Any]) -> int:
    """Number of chunks demoted to cold during tier management."""
    pruned = 0
    for change in summary.get("tier_changes", []) or []:
        if isinstance(change, dict) and change.get("to_tier") == "cold":
            pruned += 1
    return pruned


def _count_merged(summary: dict[str, Any]) -> int:
    """Number of facts that resulted in a merge-style write (ADD/UPDATE).

    NOOP and DELETE are excluded because they do not produce a merged
    memory record.
    """
    merged = 0
    for fact in summary.get("facts_extracted", []) or []:
        if not isinstance(fact, dict):
            continue
        action = str(fact.get("action", "")).upper()
        if action in {"ADD", "UPDATE"}:
            merged += 1
    return merged


def nightly_consolidation_job() -> dict[str, Any]:
    """Run a full sleep-time consolidation cycle.

    Emits metrics and a Telegram notification on both success and failure.
    Never raises. Returns a dict with the metric counts:
        {duration_ms, chunks_pruned, chunks_merged, ok, error?}
    """
    started = time.monotonic()
    result: dict[str, Any] = {
        "ok": False,
        "duration_ms": 0,
        "chunks_pruned": 0,
        "chunks_merged": 0,
    }

    try:
        summary = _safe_consolidate_call()
        duration_ms = int((time.monotonic() - started) * 1000)
        pruned = _count_pruned(summary)
        merged = _count_merged(summary)

        result.update(
            ok=True,
            duration_ms=duration_ms,
            chunks_pruned=pruned,
            chunks_merged=merged,
        )

        log_metric("consolidation.duration_ms", duration_ms)
        log_metric("consolidation.chunks_pruned", pruned)
        log_metric("consolidation.chunks_merged", merged)

        msg = (
            f"🌙 Trident consolidation: pruned {pruned}, "
            f"merged {merged} in {duration_ms}ms"
        )
        errors = summary.get("errors") or []
        if errors:
            msg += f" ({len(errors)} non-fatal errors)"
        try:
            notify_telegram(msg)
        except Exception:  # noqa: BLE001 - notify already swallows, belt + suspenders
            logger.exception("consolidation: notify_telegram raised unexpectedly")

        logger.info(
            "consolidation: ok pruned=%d merged=%d duration_ms=%d",
            pruned, merged, duration_ms,
        )
        return result

    except Exception as e:  # noqa: BLE001 - we promise never to raise
        duration_ms = int((time.monotonic() - started) * 1000)
        result["duration_ms"] = duration_ms
        result["error"] = f"{type(e).__name__}: {e}"

        # Best-effort: still record the duration so dashboards see the failed run.
        try:
            log_metric(
                "consolidation.duration_ms",
                duration_ms,
                {"status": "error"},
            )
        except Exception:  # noqa: BLE001
            pass

        logger.exception("consolidation: nightly job failed")

        try:
            notify_telegram(
                f"⚠️ Trident consolidation FAILED after {duration_ms}ms: "
                f"{type(e).__name__}: {e}"
            )
        except Exception:  # noqa: BLE001
            logger.exception("consolidation: failure notify raised unexpectedly")

        return result


__all__ = ["nightly_consolidation_job"]
