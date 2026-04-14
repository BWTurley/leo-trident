"""
Stub Embedder — returns deterministic random normalized vectors.
Used when sentence-transformers is unavailable (CI, CPU-only test env).
"""
from __future__ import annotations

import hashlib
from typing import List

import numpy as np

VALID_DIMS = {64, 256, 768}


class StubEmbedder:
    """Deterministic stub: same text always returns the same vector."""

    def _text_to_vec(self, text: str, dim: int) -> np.ndarray:
        seed = int(hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim).astype(np.float32)
        v /= np.linalg.norm(v)
        return v

    def embed(self, texts: List[str], dim: int = 768) -> np.ndarray:
        if dim not in VALID_DIMS:
            raise ValueError(f"dim must be one of {VALID_DIMS}")
        return np.stack([self._text_to_vec(t, dim) for t in texts])

    def embed_query(self, text: str, dim: int = 768) -> np.ndarray:
        return self._text_to_vec(f"search_query: {text}", dim)

    def embed_documents(self, texts: List[str], dim: int = 768) -> np.ndarray:
        return self.embed([f"search_document: {t}" for t in texts], dim=dim)
