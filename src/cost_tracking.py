"""
Leo Trident — Per-namespace Cost Tracking

Embed-call accounting plus a weekly Telegram digest.

Public API:
    log_embed_call(model, tokens, namespace='default', source_kind='text')
    cost_breakdown(days=7) -> dict
    weekly_digest_job()  # scheduled

TODO: wire log_embed_call into actual embedder calls (src/ingest/embedder.py).
The embedder is currently un-instrumented to keep its external API stable;
once we add a thin wrapper there we get cost tracking for free.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.pricing import cost_for
from src.service.metrics import log_metric, read_metrics

logger = logging.getLogger(__name__)


def log_embed_call(
    model: str,
    tokens: int,
    namespace: str = "default",
    source_kind: str = "text",
) -> None:
    """Record an embed call as two metric events: tokens & cost_usd."""
    try:
        tokens = int(tokens)
    except Exception:
        tokens = 0
    if tokens <= 0:
        return
    tags_tokens = {
        "model": model,
        "namespace": namespace,
        "source_kind": source_kind,
    }
    tags_cost = {"model": model, "namespace": namespace}
    try:
        log_metric("embed.tokens", tokens, tags_tokens)
        log_metric("embed.cost_usd", cost_for(model, tokens), tags_cost)
    except Exception as e:  # pragma: no cover
        logger.debug("log_embed_call failed: %s", e)


def cost_breakdown(days: int = 7) -> dict:
    """Aggregate `embed.cost_usd` events over the trailing `days` days.

    Returns:
        {
          "days": N,
          "by_namespace": {ns: usd, ...},
          "by_model":     {model: usd, ...},
          "total_usd":    float,
        }
    """
    days = max(1, int(days))
    by_ns: dict[str, float] = {}
    by_model: dict[str, float] = {}
    total = 0.0

    today = datetime.now(timezone.utc).date()
    for i in range(days):
        day = (today - timedelta(days=i)).isoformat()
        try:
            entries = read_metrics(date=day, name_prefix="embed.cost_usd")
        except Exception as e:
            logger.debug("cost_breakdown: read failed %s: %s", day, e)
            continue
        for e in entries:
            if e.get("name") != "embed.cost_usd":
                continue
            try:
                v = float(e.get("value", 0) or 0)
            except Exception:
                continue
            tags = e.get("tags") or {}
            ns = tags.get("namespace", "default")
            model = tags.get("model", "unknown")
            by_ns[ns] = by_ns.get(ns, 0.0) + v
            by_model[model] = by_model.get(model, 0.0) + v
            total += v

    return {
        "days": days,
        "by_namespace": by_ns,
        "by_model": by_model,
        "total_usd": total,
    }


def _format_digest(report: dict) -> str:
    days = report.get("days", 7)
    total = report.get("total_usd", 0.0)
    lines = [f"*Leo Trident — Cost Digest ({days}d)*", ""]
    lines.append(f"Total: ${total:.4f}")
    lines.append("")
    by_ns = report.get("by_namespace") or {}
    if by_ns:
        lines.append("*By namespace:*")
        for ns, v in sorted(by_ns.items(), key=lambda kv: -kv[1]):
            lines.append(f"  • {ns}: ${v:.4f}")
        lines.append("")
    by_model = report.get("by_model") or {}
    if by_model:
        lines.append("*By model:*")
        for m, v in sorted(by_model.items(), key=lambda kv: -kv[1]):
            lines.append(f"  • {m}: ${v:.4f}")
    return "\n".join(lines)


def weekly_digest_job() -> dict:
    """Compute weekly cost breakdown and post to Telegram. Never fatal."""
    try:
        report = cost_breakdown(7)
    except Exception as e:
        logger.exception("weekly_digest_job: cost_breakdown failed: %s", e)
        report = {"days": 7, "by_namespace": {}, "by_model": {}, "total_usd": 0.0}

    text = _format_digest(report)
    sent = False
    try:
        from src.notify import notify_telegram
        sent = bool(notify_telegram(text))
    except Exception as e:
        logger.exception("weekly_digest_job: notify failed: %s", e)

    try:
        log_metric(
            "digest.sent",
            1 if sent else 0,
            {"kind": "cost_weekly", "ok": str(sent).lower()},
        )
    except Exception:  # pragma: no cover
        pass

    return {"ok": True, "sent": sent, "report": report}


__all__ = [
    "log_embed_call",
    "cost_breakdown",
    "weekly_digest_job",
]
