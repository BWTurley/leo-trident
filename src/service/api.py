"""
Leo Trident — HTTP API for the Hermes memory plugin.

Four POST endpoints wrapping LeoTrident:
  /query, /log_turn, /ingest_fact, /search_conversations

Bind to 127.0.0.1 only. Never expose raw exceptions — log and return
a sanitized detail.
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(title="Leo Trident API", docs_url=None, redoc_url=None)

_trident = None
_trident_lock = Lock()


def _get_trident():
    """Lazily construct and cache a single LeoTrident instance."""
    global _trident
    if _trident is None:
        with _trident_lock:
            if _trident is None:
                from src.api import LeoTrident
                _trident = LeoTrident()
    return _trident


def reset_trident_for_tests():
    """Drop the cached LeoTrident so tests can swap in a fake."""
    global _trident
    _trident = None


# ── Request models ───────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1)
    top_k: int = Field(default=8, ge=1, le=100)
    use_rerank: bool = True
    use_relevance_judge: bool = False
    include_conversations: bool = True
    # Reply-aware retrieval (Wave 1C): optional text of the message the user
    # is replying to. When non-empty, results are re-ranked using a blend of
    # similarity to ``text`` and to ``reply_context``. None/empty == disabled
    # (backwards compatible).
    reply_context: Optional[str] = None
    reply_alpha: float = Field(default=0.7, ge=0.0, le=1.0)


class LogTurnRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    user: str = Field(..., min_length=1)
    assistant: str = Field(..., min_length=1)
    ts: Optional[str] = None


class IngestFactRequest(BaseModel):
    category: str = Field(..., min_length=1)
    key: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class SearchConversationsRequest(BaseModel):
    text: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    session_id: Optional[str] = None


# ── Error handling ───────────────────────────────────────────────────────

def _error(status: int, msg: str, detail: Optional[str] = None) -> JSONResponse:
    body: dict[str, Any] = {"error": msg}
    if detail:
        body["detail"] = detail
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(HTTPException)
async def _http_exc_handler(request: Request, exc: HTTPException):
    return _error(exc.status_code, str(exc.detail))


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return _error(500, "internal_error", type(exc).__name__)


# Pydantic validation errors → 400 with sanitized detail
from fastapi.exceptions import RequestValidationError  # noqa: E402


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    msg = first.get("msg", "invalid request")
    return _error(400, "invalid_request", f"{loc}: {msg}" if loc else msg)


# ── Endpoints ────────────────────────────────────────────────────────────

@app.post("/query")
def query(req: QueryRequest):
    t0 = time.monotonic()
    try:
        trident = _get_trident()
        results = trident.query(
            text=req.text,
            top_k=req.top_k,
            use_rerank=req.use_rerank,
            use_relevance_judge=req.use_relevance_judge,
            include_conversations=req.include_conversations,
        )
    except Exception as e:
        logger.exception("query failed")
        return _error(500, "query_failed", type(e).__name__)

    # Reply-aware re-rank (Wave 1C). Best-effort: if anything goes wrong
    # (missing embeddings on results, embedder unavailable, etc.) we log and
    # fall through to the un-boosted results.
    if req.reply_context and req.reply_context.strip() and results:
        try:
            embedder = trident._get_embedder()
            q_vec = embedder.embed_query(req.text, dim=768)
            r_vec = embedder.embed_query(req.reply_context, dim=768)
            # Results from LeoTrident.query() do not currently carry the
            # raw embedding vector. Fetch from LanceDB by chunk_id when
            # possible.
            # TODO(reply-aware): plumb embeddings through LeoTrident.query()
            # so we don't need this side-channel lookup.
            from src.retrieval.reply_aware import boost_with_reply_context
            chunk_ids = [r.get("chunk_id") for r in results if r.get("chunk_id")]
            embed_map: dict[str, list[float]] = {}
            if chunk_ids:
                try:
                    tbl = trident._get_lance_table()  # type: ignore[attr-defined]
                    rows = tbl.search().where(
                        "chunk_id IN ({})".format(
                            ",".join(f"'{c}'" for c in chunk_ids)
                        )
                    ).limit(len(chunk_ids)).to_list()
                    for row in rows:
                        cid = row.get("chunk_id")
                        emb = row.get("vector_768") or row.get("embedding")
                        if cid and emb is not None:
                            embed_map[cid] = list(emb)
                except Exception as e:
                    logger.debug("reply_aware: embedding lookup skipped: %s", e)
            enriched = [
                {**r, "embedding": embed_map.get(r.get("chunk_id"))}
                for r in results
            ]
            if any(r["embedding"] is not None for r in enriched):
                results = boost_with_reply_context(
                    enriched, q_vec, r_vec, alpha=req.reply_alpha
                )
                # Strip raw embedding from the response payload to keep it small.
                for r in results:
                    r.pop("embedding", None)
        except Exception as e:
            logger.warning("reply_aware boost failed: %s", e)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return {
        "results": list(results) if results else [],
        "stub_embedder": bool(getattr(_get_trident(), "_using_stub", False)),
        "query_ms": elapsed_ms,
    }


@app.post("/log_turn")
def log_turn(req: LogTurnRequest):
    try:
        trident = _get_trident()
        conn = trident._get_conn()
    except Exception as e:
        logger.exception("log_turn: db open failed")
        return _error(500, "db_unavailable", type(e).__name__)

    try:
        if req.ts:
            created_at = req.ts
        else:
            created_at = datetime.now(timezone.utc).isoformat()

        row = conn.execute(
            "SELECT COALESCE(MAX(turn_index), -1) + 1 AS next FROM conversation_logs "
            "WHERE session_id = ?",
            (req.session_id,),
        ).fetchone()
        turn_index = int(row["next"] if row is not None else 0)

        user_log_id = str(uuid.uuid4())
        asst_log_id = str(uuid.uuid4())

        conn.execute(
            "INSERT INTO conversation_logs (log_id, session_id, turn_index, role, content, created_at) "
            "VALUES (?, ?, ?, 'user', ?, ?)",
            (user_log_id, req.session_id, turn_index, req.user, created_at),
        )
        conn.execute(
            "INSERT INTO conversation_logs (log_id, session_id, turn_index, role, content, created_at) "
            "VALUES (?, ?, ?, 'assistant', ?, ?)",
            (asst_log_id, req.session_id, turn_index, req.assistant, created_at),
        )
        conn.commit()
        return {"ok": True, "turn_id": turn_index}
    except Exception as e:
        logger.exception("log_turn insert failed")
        return _error(500, "log_turn_failed", type(e).__name__)
    finally:
        try:
            conn.close()
        except Exception:
            pass


@app.post("/ingest_fact")
def ingest_fact(req: IngestFactRequest):
    paragraph_id = f"fact:{req.category}:{req.key}"
    text = f"[{req.category}] {req.key}: {req.value}"

    try:
        trident = _get_trident()
        chunk_id = trident.ingest_text(
            text=text,
            paragraph_id=paragraph_id,
            section="fact",
            part=req.category,
            source="fact",
        )
    except Exception as e:
        logger.exception("ingest_fact failed")
        return _error(500, "ingest_failed", type(e).__name__)

    return {"ok": True, "paragraph_id": paragraph_id, "chunk_id": chunk_id}


@app.post("/admin/consolidate/run-now")
def admin_consolidate_run_now():
    """Trigger nightly consolidation synchronously and return metric counts."""
    try:
        from src.jobs.consolidation import nightly_consolidation_job
        result = nightly_consolidation_job()
    except Exception as e:
        # nightly_consolidation_job promises not to raise, but be defensive.
        logger.exception("admin_consolidate_run_now: unexpected raise")
        return _error(500, "consolidate_failed", type(e).__name__)
    return result


@app.get("/admin/cost")
def admin_cost(days: int = 7):
    """Per-namespace + per-model embed cost breakdown over trailing N days."""
    try:
        from src.cost_tracking import cost_breakdown
        return cost_breakdown(days)
    except Exception as e:
        logger.exception("admin_cost failed")
        return _error(500, "cost_breakdown_failed", type(e).__name__)


@app.post("/admin/quality/snapshot")
def admin_quality_snapshot():
    """Run the golden-query quality snapshot synchronously and return the result."""
    try:
        from src.quality import run_quality_snapshot
        return run_quality_snapshot()
    except Exception as e:
        logger.exception("admin_quality_snapshot failed")
        return _error(500, "quality_snapshot_failed", type(e).__name__)


@app.post("/search_conversations")
def search_conversations(req: SearchConversationsRequest):
    try:
        trident = _get_trident()
        results = trident.search_conversations(
            text=req.text,
            top_k=req.top_k,
            session_id=req.session_id,
        )
    except Exception as e:
        logger.exception("search_conversations failed")
        return _error(500, "search_failed", type(e).__name__)

    return {"results": list(results) if results else []}
