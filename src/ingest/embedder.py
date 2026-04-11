"""
Leo Trident — Embedder
Uses nomic-ai/nomic-embed-text-v1.5 with Matryoshka dimensions (64/256/768).
CPU only, batch size 32.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List

VALID_DIMS = {64, 256, 768}
DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
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

    def embed(self, texts: List[str], dim: int = 768) -> np.ndarray:
        """Embed texts at specified Matryoshka dimension."""
        raw = self._encode_raw(texts)
        return self._truncate(raw, dim)

    def embed_query(self, text: str, dim: int = 768) -> np.ndarray:
        """Embed a single query with nomic search_query prefix."""
        prefixed = f"search_query: {text}"
        return self.embed([prefixed], dim=dim)[0]

    def embed_documents(self, texts: List[str], dim: int = 768) -> np.ndarray:
        """Embed documents with nomic search_document prefix."""
        prefixed = [f"search_document: {t}" for t in texts]
        return self.embed(prefixed, dim=dim)
