"""
Leo Trident — Drift Telemetry / Quality Snapshot

Daily golden-query MRR/Recall harness. Loads a small fixture of
canonical Q+A pairs, runs them against the live retriever, computes
MRR@10 and Recall@10, logs to the metrics sink, and fires a Telegram
alert when today's score regresses >15% vs. the 7-day rolling avg.

Non-fatal: nothing here raises into the scheduler.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from src.notify import notify_telegram
from src.service.metrics import log_metric, rollup_metric

logger = logging.getLogger(__name__)

# tests/fixtures/golden_queries.json relative to repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
_GOLDEN_PATH = _REPO_ROOT / "tests" / "fixtures" / "golden_queries.json"

_TOP_K = 10
_REGRESSION_THRESHOLD = 0.15  # 15% drop vs. rolling baseline
_BASELINE_WINDOW_DAYS = 7


def load_golden_queries(path: Path | None = None) -> list[dict]:
    """Load the golden-query fixture. Returns [] on any error."""
    p = path or _GOLDEN_PATH
    try:
        with open(p) as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("golden_queries: expected list, got %s", type(data))
            return []
        return data
    except Exception as e:  # noqa: BLE001
        logger.warning("golden_queries: load failed: %s", e)
        return []


def _result_matches(result: dict, expected_substrs: list[str]) -> bool:
    """True if any expected substring appears in the result's chunk_id or text."""
    hay = " ".join(
        str(result.get(k, "") or "")
        for k in ("chunk_id", "paragraph_id", "text", "content", "section", "part")
    ).lower()
    return any(sub.lower() in hay for sub in expected_substrs if sub)


def _rank_of_first_match(results: list[dict], expected: list[str]) -> Optional[int]:
    for i, r in enumerate(results):
        if _result_matches(r, expected):
            return i + 1  # 1-indexed
    return None


def _resolve_query_callable(trident: Any) -> Callable[[str, int], list[dict]]:
    """Coerce a trident-like input into a (text, top_k) -> list[dict] callable."""
    if trident is None:
        from src.api import LeoTrident

        trident = LeoTrident()
    if callable(trident) and not hasattr(trident, "query"):
        return trident  # already a function
    return lambda text, top_k: trident.query(text=text, top_k=top_k)


def run_quality_snapshot(trident: Any = None,
                         golden: list[dict] | None = None,
                         top_k: int = _TOP_K) -> dict:
    """
    Run all golden queries and compute MRR@K and Recall@K.

    `trident` may be:
      * None — a fresh LeoTrident is constructed.
      * A LeoTrident-like object exposing .query(text, top_k).
      * A plain callable (text, top_k) -> list[dict] — useful in tests.

    Returns:
        {
          "mrr_at_10": float,
          "recall_at_10": float,
          "n_queries": int,
          "k": int,
          "per_query": [
              {"query": ..., "rank": int|None, "hit": bool, "n_results": int}
          ],
          "ts": ISO-8601 UTC,
        }
    """
    queries = golden if golden is not None else load_golden_queries()
    query_fn = _resolve_query_callable(trident)

    per_query: list[dict] = []
    reciprocal_ranks: list[float] = []
    hits = 0

    for entry in queries:
        q = entry.get("query", "")
        expected = entry.get("expected_chunk_ids") or []
        try:
            results = list(query_fn(q, top_k) or [])
        except Exception as e:  # noqa: BLE001
            logger.warning("golden query %r failed: %s", q, e)
            results = []

        results = results[:top_k]
        rank = _rank_of_first_match(results, expected)
        hit = rank is not None
        if hit:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        per_query.append({
            "query": q,
            "rank": rank,
            "hit": hit,
            "n_results": len(results),
        })

    n = len(queries)
    mrr = (sum(reciprocal_ranks) / n) if n else 0.0
    recall = (hits / n) if n else 0.0

    return {
        "mrr_at_10": mrr,
        "recall_at_10": recall,
        "n_queries": n,
        "k": top_k,
        "per_query": per_query,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def _rolling_baseline(metric_name: str, days: int = _BASELINE_WINDOW_DAYS,
                      skip_today: bool = True) -> Optional[float]:
    """Average of `metric_name` over the last `days` days (avg-of-daily-avgs).

    Returns None if there are no historical samples to compare against.
    """
    today = datetime.now(timezone.utc).date()
    start = 1 if skip_today else 0
    daily_avgs: list[float] = []
    for i in range(start, start + days):
        day = (today - timedelta(days=i)).isoformat()
        # rollup_metric returns 0.0 if no events; we want to skip absent days
        # so re-check via read_metrics presence.
        from src.service.metrics import read_metrics
        if not [e for e in read_metrics(date=day) if e.get("name") == metric_name]:
            continue
        daily_avgs.append(rollup_metric(metric_name, date=day, agg="avg"))
    if not daily_avgs:
        return None
    return sum(daily_avgs) / len(daily_avgs)


def daily_quality_job() -> dict:
    """
    Scheduled entry point. Computes today's snapshot, logs metrics, and
    fires a Telegram alert if MRR or Recall regressed >15% vs the
    rolling 7-day baseline. Always non-fatal — exceptions are swallowed
    and a status dict is returned.
    """
    try:
        snap = run_quality_snapshot()
    except Exception as e:  # noqa: BLE001
        logger.exception("daily_quality_job: snapshot failed")
        return {"ok": False, "error": type(e).__name__, "detail": str(e)}

    try:
        log_metric("quality.mrr", snap["mrr_at_10"], tags={"k": snap["k"]})
        log_metric("quality.recall", snap["recall_at_10"], tags={"k": snap["k"]})
    except Exception as e:  # noqa: BLE001
        logger.warning("daily_quality_job: log_metric failed: %s", e)

    alerted = False
    try:
        regressions: list[str] = []
        for label, key, metric_name in (
            ("MRR@10", "mrr_at_10", "quality.mrr"),
            ("Recall@10", "recall_at_10", "quality.recall"),
        ):
            today_val = float(snap.get(key, 0.0) or 0.0)
            baseline = _rolling_baseline(metric_name)
            if baseline is None or baseline <= 0:
                continue
            drop = (baseline - today_val) / baseline
            if drop > _REGRESSION_THRESHOLD:
                regressions.append(
                    f"• *{label}*: today={today_val:.3f} "
                    f"vs 7d-avg={baseline:.3f} (−{drop*100:.1f}%)"
                )

        if regressions:
            text = (
                "⚠️ *Leo Trident quality regression*\n"
                + "\n".join(regressions)
                + f"\n_n_queries_={snap['n_queries']}"
            )
            notify_telegram(text)
            alerted = True
    except Exception as e:  # noqa: BLE001
        logger.warning("daily_quality_job: alert path failed: %s", e)

    return {
        "ok": True,
        "mrr_at_10": snap["mrr_at_10"],
        "recall_at_10": snap["recall_at_10"],
        "n_queries": snap["n_queries"],
        "alerted": alerted,
    }


__all__ = [
    "load_golden_queries",
    "run_quality_snapshot",
    "daily_quality_job",
]
