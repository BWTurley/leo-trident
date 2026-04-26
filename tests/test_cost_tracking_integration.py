"""Integration: Embedder calls log_embed_call after encoding."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.ingest import embedder as embedder_mod


def _make_embedder_with_fake_model():
    """Build an Embedder whose underlying SentenceTransformer is mocked,
    so we don't download / run the real model in CI."""
    emb = embedder_mod.Embedder(model_name="nomic-ai/nomic-embed-text-v1.5")
    fake_model = MagicMock()

    # Return a (n, 768) float tensor-like ndarray; F.layer_norm/F.normalize
    # both accept torch tensors, so use torch directly.
    import torch

    def _encode(texts, **kw):
        return torch.ones((len(texts), 768), dtype=torch.float32)

    fake_model.encode.side_effect = _encode
    emb._model = fake_model
    return emb


def test_embed_invokes_log_embed_call():
    emb = _make_embedder_with_fake_model()

    with patch.object(embedder_mod, "log_embed_call") as mock_log:
        out = emb.embed(["hello world", "foo bar"], dim=64, namespace="leo-trident")

    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 64)

    mock_log.assert_called_once()
    kwargs = mock_log.call_args.kwargs
    assert kwargs["namespace"] == "leo-trident"
    assert kwargs["model"] == "nomic-embed-text-v1.5"  # HF org prefix stripped
    assert kwargs["source_kind"] == "text"
    # len("hello world")=11 + len("foo bar")=7 → 18 chars // 4 = 4 tokens
    assert kwargs["tokens"] == (11 + 7) // 4


def test_embed_query_tracks_with_query_source_kind():
    emb = _make_embedder_with_fake_model()

    with patch.object(embedder_mod, "log_embed_call") as mock_log:
        v = emb.embed_query("what is leo", dim=64, namespace="ns-q")

    assert v.shape == (64,)
    mock_log.assert_called_once()
    kwargs = mock_log.call_args.kwargs
    assert kwargs["namespace"] == "ns-q"
    assert kwargs["source_kind"] == "query"


def test_embed_documents_tracks_with_document_source_kind():
    emb = _make_embedder_with_fake_model()

    with patch.object(embedder_mod, "log_embed_call") as mock_log:
        out = emb.embed_documents(["doc one", "doc two"], dim=64, namespace="ns-d")

    assert out.shape == (2, 64)
    mock_log.assert_called_once()
    kwargs = mock_log.call_args.kwargs
    assert kwargs["namespace"] == "ns-d"
    assert kwargs["source_kind"] == "document"
