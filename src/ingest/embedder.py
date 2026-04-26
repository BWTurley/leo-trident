"""
Leo Trident — Embedder
Uses nomic-ai/nomic-embed-text-v1.5 with Matryoshka dimensions (64/256/768).
CPU only, batch size 32.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from src.config import EMBED_DEVICE
from src.cost_tracking import log_embed_call

VALID_DIMS = {64, 256, 768}
DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"

logger = logging.getLogger(__name__)


def _estimate_tokens(texts: List[str]) -> int:
    """Rough token estimate: chars / 4. Good enough for cost accounting."""
    return sum(len(t) for t in texts) // 4


def _pricing_model_name(model_name: str) -> str:
    """Strip HF org prefix so the pricing table key matches."""
    return model_name.split("/", 1)[-1]


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    trust_remote_code=True,
                    device=EMBED_DEVICE,
                )
            except Exception as e:
                logger.warning(
                    f"Embedder device {EMBED_DEVICE!r} init failed ({e}), falling back to CPU"
                )
                self._model = SentenceTransformer(
                    self.model_name,
                    trust_remote_code=True,
                    device="cpu",
                )
        return self._model

    def _encode_raw(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Return full 768-d normalized embeddings."""
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        # nomic Matryoshka: apply layer_norm before truncation
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def _truncate(self, embeddings: np.ndarray, dim: int) -> np.ndarray:
        if dim not in VALID_DIMS:
            raise ValueError(f"dim must be one of {VALID_DIMS}, got {dim}")
        truncated = embeddings[:, :dim]
        # re-normalize after truncation
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return truncated / norms

    def _track_cost(
        self,
        texts: List[str],
        namespace: str,
        source_kind: str,
    ) -> None:
        try:
            log_embed_call(
                model=_pricing_model_name(self.model_name),
                tokens=_estimate_tokens(texts),
                namespace=namespace,
                source_kind=source_kind,
            )
        except Exception as e:  # pragma: no cover
            logger.debug("embed cost tracking failed: %s", e)

    def embed(
        self,
        texts: List[str],
        dim: int = 768,
        namespace: str = "global",
        source_kind: str = "text",
    ) -> np.ndarray:
        """Embed texts at specified Matryoshka dimension.

        TODO: callers currently default to namespace='global'; thread the real
        per-ingest/query namespace through once we plumb it from upstream.
        """
        raw = self._encode_raw(texts)
        out = self._truncate(raw, dim)
        self._track_cost(texts, namespace=namespace, source_kind=source_kind)
        return out

    def embed_query(
        self,
        text: str,
        dim: int = 768,
        namespace: str = "global",
    ) -> np.ndarray:
        """Embed a single query with nomic search_query prefix."""
        prefixed = f"search_query: {text}"
        # _track_cost runs inside embed(); pass query source_kind through.
        return self.embed(
            [prefixed],
            dim=dim,
            namespace=namespace,
            source_kind="query",
        )[0]

    def embed_documents(
        self,
        texts: List[str],
        dim: int = 768,
        namespace: str = "global",
    ) -> np.ndarray:
        """Embed documents with nomic search_document prefix."""
        prefixed = [f"search_document: {t}" for t in texts]
        return self.embed(
            prefixed,
            dim=dim,
            namespace=namespace,
            source_kind="document",
        )
