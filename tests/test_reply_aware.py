"""Tests for src/retrieval/reply_aware.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.reply_aware import boost_with_reply_context


def _mk(chunk_id: str, embedding):
    return {"chunk_id": chunk_id, "embedding": list(embedding), "score": 0.0}


def test_boost_changes_order():
    # Query points along x; result1 is most similar to query.
    # Reply context points along y; result3 is most similar to reply_ctx.
    # With alpha=0.3 (heavier reply weight), result3 should jump to first.
    query = [1.0, 0.0, 0.0]
    reply_ctx = [0.0, 1.0, 0.0]

    results = [
        _mk("r1", [0.95, 0.10, 0.0]),   # very close to query
        _mk("r2", [0.70, 0.50, 0.0]),   # mid
        _mk("r3", [0.10, 0.99, 0.0]),   # very close to reply ctx
    ]

    boosted = boost_with_reply_context(results, query, reply_ctx, alpha=0.3)
    assert boosted[0]["chunk_id"] == "r3"
    # And every result has the new field.
    for item in boosted:
        assert "reply_aware_score" in item


def test_alpha_zero_uses_only_reply_context():
    query = [1.0, 0.0]
    reply_ctx = [0.0, 1.0]

    results = [
        _mk("a", [1.0, 0.0]),  # cos(query)=1, cos(reply)=0
        _mk("b", [0.0, 1.0]),  # cos(query)=0, cos(reply)=1
    ]

    boosted = boost_with_reply_context(results, query, reply_ctx, alpha=0.0)
    assert boosted[0]["chunk_id"] == "b"
    assert boosted[0]["reply_aware_score"] == pytest.approx(1.0)
    assert boosted[1]["reply_aware_score"] == pytest.approx(0.0)


def test_alpha_one_preserves_query_order():
    query = [1.0, 0.0]
    reply_ctx = [0.0, 1.0]

    # Sorted descending by similarity to query already.
    results = [
        _mk("a", [1.0, 0.0]),
        _mk("b", [0.7, 0.7]),
        _mk("c", [0.1, 1.0]),
    ]

    boosted = boost_with_reply_context(results, query, reply_ctx, alpha=1.0)
    assert [r["chunk_id"] for r in boosted] == ["a", "b", "c"]
    # Scores should equal cos(query) only.
    for orig, out in zip(results, boosted):
        v = np.asarray(orig["embedding"])
        q = np.asarray(query)
        expected = float(np.dot(v, q) / (np.linalg.norm(v) * np.linalg.norm(q)))
        assert out["reply_aware_score"] == pytest.approx(expected)


def test_no_reply_context_embedding_is_none():
    """When reply_context_embedding is None, score collapses to alpha * sim(query)."""
    query = [1.0, 0.0]
    results = [
        _mk("a", [1.0, 0.0]),
        _mk("b", [0.0, 1.0]),
    ]
    boosted = boost_with_reply_context(results, query, None, alpha=0.7)
    assert boosted[0]["chunk_id"] == "a"
    assert boosted[0]["reply_aware_score"] == pytest.approx(0.7)


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        boost_with_reply_context([], [1.0], [1.0], alpha=1.5)


# TODO(reply-aware-api): add a TestClient-based test asserting POST /query
# with no `reply_context` field returns the same shape as before. Skipped
# here because constructing or mocking a full LeoTrident inside the test
# harness needs a fixture that doesn't yet exist.
