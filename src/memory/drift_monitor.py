"""
Leo Trident — Drift Monitor (Phase 5)
Tracks embedding drift and anchor integrity over time.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from src.config import BASE_PATH as _DEFAULT_BASE_PATH

logger = logging.getLogger(__name__)


class DriftMonitor:
    def __init__(self, base_path: str | Path = None):
        base_path = Path(base_path) if base_path else _DEFAULT_BASE_PATH
        self.base_path = Path(base_path)
        self.db_path = self.base_path / "data" / "leo_trident.db"
        self.vault_path = self.base_path / "vault" / "_system"
        self._lt = None

    def _get_lt(self):
        if self._lt is None:
            from src.api import LeoTrident
            self._lt = LeoTrident(base_path=str(self.base_path))
        return self._lt

    def check_anchors(self) -> dict:
        """SHA-256 verify all anchors in vault/_system/anchors.json."""
        anchors_path = self.vault_path / "anchors.json"
        try:
            with open(anchors_path) as f:
                anchors = json.load(f)
        except Exception as e:
            return {"ok": False, "violations": [f"Cannot load anchors.json: {e}"]}

        violations = []

        def _check(items: list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                rule = item.get("rule") or item.get("fact", "")
                stored = item.get("hash", "")
                if not stored:
                    continue
                computed = hashlib.sha256(rule.encode()).hexdigest()
                if computed != stored:
                    violations.append({
                        "rule": rule,
                        "stored": stored,
                        "computed": computed,
                    })

        _check(anchors.get("asme_safety_pins", {}).get("never", []))
        _check(anchors.get("asme_safety_pins", {}).get("always", []))
        _check(anchors.get("core_facts", []))

        if violations:
            logger.error("ANCHOR VIOLATIONS DETECTED: %d", len(violations))
        else:
            logger.info("All anchors verified OK")

        return {"ok": len(violations) == 0, "violations": violations}

    def compute_psi(
        self,
        baseline_embeddings: np.ndarray,
        current_embeddings: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Population Stability Index between two embedding distributions.
        PSI = Σ (actual% - expected%) * ln(actual% / expected%)
        Computes PSI per dimension and returns the mean.
        Alert threshold: PSI > 0.1
        """
        eps = 1e-8
        n_dims = baseline_embeddings.shape[1]
        psi_values = []

        for d in range(n_dims):
            base_col = baseline_embeddings[:, d]
            curr_col = current_embeddings[:, d]

            # Bin edges from combined range
            all_vals = np.concatenate([base_col, curr_col])
            bins = np.linspace(all_vals.min(), all_vals.max() + eps, n_bins + 1)

            base_hist, _ = np.histogram(base_col, bins=bins)
            curr_hist, _ = np.histogram(curr_col, bins=bins)

            base_pct = base_hist / (base_hist.sum() + eps)
            curr_pct = curr_hist / (curr_hist.sum() + eps)

            # Avoid log(0)
            base_pct = np.where(base_pct == 0, eps, base_pct)
            curr_pct = np.where(curr_pct == 0, eps, curr_pct)

            psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
            psi_values.append(float(psi))

        mean_psi = float(np.mean(psi_values))
        if mean_psi > 0.1:
            logger.warning("PSI alert: mean PSI=%.4f exceeds threshold 0.1", mean_psi)
        return mean_psi

    def embedding_drift_report(self, sample_size: int = 100) -> dict:
        """
        Sample chunks from LanceDB, re-embed them, and compare cosine similarity
        between stored and fresh embeddings.
        Returns {mean_cosine_sim, min_cosine_sim, drift_detected: bool}.
        """
        try:
            import lancedb
        except ImportError:
            return {
                "mean_cosine_sim": None,
                "min_cosine_sim": None,
                "drift_detected": False,
                "error": "lancedb not installed",
            }

        try:
            lance_path = self.base_path / "data" / "lancedb"
            db = lancedb.connect(str(lance_path))

            # Try cold table first (768d), fall back to warm (256d)
            table_name = "chunks_cold"
            if table_name not in db.table_names():
                table_name = "chunks_warm"
            if table_name not in db.table_names():
                return {
                    "mean_cosine_sim": None,
                    "min_cosine_sim": None,
                    "drift_detected": False,
                    "error": "No LanceDB tables found",
                }

            tbl = db.open_table(table_name)
            df = tbl.to_pandas()
            if df.empty:
                return {
                    "mean_cosine_sim": None,
                    "min_cosine_sim": None,
                    "drift_detected": False,
                    "error": "Table is empty",
                }

            # Sample up to sample_size rows
            sample = df.sample(min(sample_size, len(df)), random_state=42)

            # Re-embed sample texts
            from src.ingest.embedder import Embedder
            embedder = Embedder()

            dim = 768 if "768" in table_name or "cold" in table_name else 256
            texts = sample["content"].tolist() if "content" in sample.columns else []
            if not texts:
                return {
                    "mean_cosine_sim": None,
                    "min_cosine_sim": None,
                    "drift_detected": False,
                    "error": "No content column found",
                }

            fresh_embeddings = embedder.embed_documents(texts, dim=dim)

            # Compare stored vs fresh
            stored_col = "vector" if "vector" in sample.columns else sample.columns[-1]
            stored_embeddings = np.array(sample[stored_col].tolist())

            cosines = []
            for stored, fresh in zip(stored_embeddings, fresh_embeddings):
                stored_n = stored / (np.linalg.norm(stored) + 1e-8)
                fresh_n = fresh / (np.linalg.norm(fresh) + 1e-8)
                cosines.append(float(np.dot(stored_n, fresh_n)))

            mean_sim = float(np.mean(cosines))
            min_sim = float(np.min(cosines))
            drift_detected = mean_sim < 0.95 or min_sim < 0.80

            if drift_detected:
                logger.warning(
                    "Embedding drift detected: mean_cosine=%.4f, min_cosine=%.4f",
                    mean_sim, min_sim,
                )

            return {
                "mean_cosine_sim": mean_sim,
                "min_cosine_sim": min_sim,
                "drift_detected": drift_detected,
                "n_sampled": len(cosines),
            }

        except Exception as e:
            logger.warning("embedding_drift_report failed: %s", e)
            return {
                "mean_cosine_sim": None,
                "min_cosine_sim": None,
                "drift_detected": False,
                "error": str(e),
            }
