"""
Leo Trident — BGE Reranker
Uses BAAI/bge-reranker-v2-m3 via transformers cross-encoder.
Graceful fallback to RRF score order if model unavailable.
"""
from __future__ import annotations
import logging
from typing import List

logger = logging.getLogger(__name__)


class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._available = False
        self._try_load()

    def _try_load(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
            )
            self._model.eval()
            self._available = True
            logger.info(f"BGEReranker loaded: {self.model_name}")
        except Exception as e:
            logger.warning(f"BGEReranker unavailable ({e}), will use score fallback")
            self._available = False

    def rerank(self, query: str, candidates: List[dict], top_k: int = 10) -> List[dict]:
        """
        Rerank candidates using BGE cross-encoder.
        Falls back to existing score if model unavailable.
        candidates: list of dicts with at least 'content' and 'score' keys.
        """
        if not candidates:
            return []

        if not self._available:
            # Fallback: sort by existing RRF score descending
            sorted_cands = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
            return sorted_cands[:top_k]

        import torch

        pairs = [[query, c.get("content", "")] for c in candidates]
        try:
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self._model(**inputs).logits.squeeze(-1)
                scores = torch.sigmoid(logits).cpu().numpy().tolist()

            ranked = sorted(
                zip(scores, candidates),
                key=lambda x: x[0],
                reverse=True,
            )
            results = []
            for score, cand in ranked[:top_k]:
                c = dict(cand)
                c["rerank_score"] = float(score)
                results.append(c)
            return results
        except Exception as e:
            logger.error(f"BGEReranker.rerank error: {e}, falling back to score sort")
            sorted_cands = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
            return sorted_cands[:top_k]
