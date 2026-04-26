"""
Reply-aware retrieval re-ranking.

Given a list of retrieval results (each carrying an embedding) plus the
original query embedding and the embedding of the message being replied to,
re-score each result as a convex combination of cosine similarity to the
query and to the reply context, then re-sort.

This lets a chat-time query like "what about that?" still pull in chunks
related to the parent message in the reply chain, without losing the
primary intent of the user's new question.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _to_vec(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    return arr


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def boost_with_reply_context(
    results: list[dict],
    query_embedding: Sequence[float],
    reply_context_embedding: Sequence[float] | None,
    alpha: float = 0.7,
) -> list[dict]:
    """Re-rank ``results`` using a blend of query and reply-context similarity.

    Each result must contain an ``embedding`` field (list/array of floats).
    Results lacking an embedding are scored 0.0 for that component.

    New score::

        score = alpha * cos(result, query) + (1 - alpha) * cos(result, reply_ctx)

    The function returns a new list (does not mutate the input list order),
    sorted by ``reply_aware_score`` descending. Each returned dict has a
    ``reply_aware_score`` key added.

    If ``reply_context_embedding`` is None, this degenerates to ranking by
    similarity to the query alone (with weight ``alpha``); callers usually
    just skip calling this function in that case.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    q = _to_vec(query_embedding)
    r = _to_vec(reply_context_embedding) if reply_context_embedding is not None else None

    scored: list[dict] = []
    for item in results:
        emb = item.get("embedding")
        if emb is None:
            sim_q = 0.0
            sim_r = 0.0
        else:
            v = _to_vec(emb)
            sim_q = _cosine(v, q)
            sim_r = _cosine(v, r) if r is not None else 0.0
        score = alpha * sim_q + (1.0 - alpha) * sim_r
        new_item = dict(item)
        new_item["reply_aware_score"] = score
        scored.append(new_item)

    scored.sort(key=lambda d: d["reply_aware_score"], reverse=True)
    return scored
