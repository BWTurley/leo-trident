"""
Leo Trident — Personalized PageRank (PPR)
Runs PPR over the ASME cross-reference graph using scipy sparse matrices
loaded from SQLite graph_edges table. No graph database required.

References:
  - HippoRAG 2 (arXiv:2502.14802) — PPR + RRF for multi-hop retrieval
  - fast-pagerank (github: asajadi/fast-pagerank)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _try_import_scipy():
    try:
        from scipy.sparse import csr_matrix
        return csr_matrix
    except ImportError:
        raise ImportError("scipy is required for PPR: pip install scipy")


def _try_import_fast_pagerank():
    try:
        from fast_pagerank import pagerank_power
        return pagerank_power
    except ImportError:
        logger.warning(
            "fast-pagerank not installed. Falling back to numpy power iteration. "
            "Install with: pip install fast-pagerank"
        )
        return None


class ASMEGraphPPR:
    """
    Loads the ASME cross-reference graph from SQLite into a scipy CSR matrix
    and runs Personalized PageRank to identify related paragraphs.

    The graph is stored in RAM as a CSR matrix (~3MB for 80K edges at 18K nodes).
    Load once at startup; refresh only after new ASME ingestion.
    """

    def __init__(
        self,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        """
        Args:
            damping: PageRank damping factor (alpha). 0.85 is standard.
            max_iter: Maximum power iteration steps.
            tol: Convergence tolerance.
        """
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol

        # Populated by load_from_sqlite()
        self._matrix: Optional[object] = None          # scipy CSR
        self._node_ids: list[str] = []                 # paragraph IDs in matrix order
        self._node_index: dict[str, int] = {}          # paragraph_id → row/col index
        self._n_nodes: int = 0
        self._n_edges: int = 0

    # ── Graph loading ─────────────────────────────────────────────────

    def load_from_sqlite(self, conn, edition_year: Optional[int] = None) -> "ASMEGraphPPR":
        """
        Load graph edges from SQLite into a scipy CSR matrix.

        Args:
            conn: sqlite3.Connection with graph_edges table.
            edition_year: if provided, filter to a specific ASME edition.

        Returns:
            self (for chaining)
        """
        csr_matrix = _try_import_scipy()

        # Load edges
        if edition_year is not None:
            rows = conn.execute(
                "SELECT source_id, target_id, weight FROM graph_edges "
                "WHERE edition_year = ? OR edition_year IS NULL",
                (edition_year,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT source_id, target_id, weight FROM graph_edges"
            ).fetchall()

        if not rows:
            logger.warning("No graph edges found in database. PPR will return empty results.")
            self._matrix = None
            return self

        # Build node vocabulary
        nodes = set()
        for row in rows:
            nodes.add(row[0])  # source_id
            nodes.add(row[1])  # target_id

        self._node_ids = sorted(nodes)
        self._node_index = {nid: i for i, nid in enumerate(self._node_ids)}
        n = len(self._node_ids)
        self._n_nodes = n
        self._n_edges = len(rows)

        # Build sparse matrix
        row_indices = []
        col_indices = []
        weights = []

        for edge_row in rows:
            src = edge_row[0]
            tgt = edge_row[1]
            w = float(edge_row[2]) if edge_row[2] is not None else 1.0
            if src in self._node_index and tgt in self._node_index:
                row_indices.append(self._node_index[src])
                col_indices.append(self._node_index[tgt])
                weights.append(w)
                # Add reverse edge (undirected similarity)
                row_indices.append(self._node_index[tgt])
                col_indices.append(self._node_index[src])
                weights.append(w)

        self._matrix = csr_matrix(
            (weights, (row_indices, col_indices)),
            shape=(n, n),
            dtype=np.float32,
        )

        # Log edge-type distribution for observability
        try:
            dist_rows = conn.execute(
                "SELECT reference_type, COUNT(*) FROM graph_edges GROUP BY reference_type"
            ).fetchall()
            dist = {r[0]: r[1] for r in dist_rows}
            total = sum(dist.values()) or 1
            dist_str = ", ".join(f"{k}={v} ({100*v//total}%)" for k, v in sorted(dist.items()))
            logger.info(f"PPR edge distribution: {dist_str}")
        except Exception:
            pass  # reference_type column may not exist on very old DBs pre-migration

        logger.info(
            f"PPR graph loaded: {n} nodes, {self._n_edges} edges "
            f"({self._matrix.nnz} matrix entries, "
            f"~{self._matrix.data.nbytes / 1024 / 1024:.1f} MB)"
        )
        return self

    @classmethod
    def from_sqlite(cls, conn, **kwargs) -> "ASMEGraphPPR":
        """Factory: create and load in one call."""
        return cls(**kwargs).load_from_sqlite(conn)

    # ── Query ─────────────────────────────────────────────────────────

    def query(
        self,
        seed_paragraph_ids: list[str],
        top_k: int = 50,
        seed_weights: Optional[dict[str, float]] = None,
    ) -> list[tuple[str, float]]:
        """
        Run Personalized PageRank seeded from given paragraph IDs.

        Args:
            seed_paragraph_ids: paragraphs to use as personalization seeds.
            top_k: number of top-ranked paragraphs to return.
            seed_weights: optional per-seed weight (default: uniform).

        Returns:
            List of (paragraph_id, ppr_score) sorted by score descending.
            Seeds themselves are excluded from results.
        """
        if self._matrix is None or self._n_nodes == 0:
            logger.warning("PPR graph not loaded — returning empty results")
            return []

        # Filter seeds to known nodes
        known_seeds = [s for s in seed_paragraph_ids if s in self._node_index]
        if not known_seeds:
            logger.debug(f"None of the seed nodes are in the graph: {seed_paragraph_ids}")
            return []

        # Build personalization vector
        personalization = np.zeros(self._n_nodes, dtype=np.float32)
        for seed in known_seeds:
            idx = self._node_index[seed]
            w = seed_weights.get(seed, 1.0) if seed_weights else 1.0
            personalization[idx] += w

        # Normalize
        total = personalization.sum()
        if total == 0:
            return []
        personalization /= total

        # Run PPR
        scores = self._run_ppr(personalization)

        # Collect results, excluding seeds
        seed_set = set(known_seeds)
        results = [
            (self._node_ids[i], float(scores[i]))
            for i in range(self._n_nodes)
            if self._node_ids[i] not in seed_set and scores[i] > 1e-9
        ]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def query_by_embedding_similarity(
        self,
        query_vector: np.ndarray,
        node_embeddings: dict[str, np.ndarray],
        top_seeds: int = 5,
        top_k: int = 50,
    ) -> list[tuple[str, float]]:
        """
        Identify seed nodes via embedding similarity, then run PPR.
        This avoids the need for NER or LLM extraction of paragraph IDs.

        Args:
            query_vector: query embedding (normalized).
            node_embeddings: dict of paragraph_id → embedding vector.
            top_seeds: how many top-similarity nodes to use as seeds.
            top_k: PPR result count.
        """
        # Compute cosine similarity between query and all node embeddings
        similarities = []
        for pid, emb in node_embeddings.items():
            if pid not in self._node_index:
                continue
            # Assume inputs are unit-normalized
            sim = float(np.dot(query_vector, emb))
            similarities.append((pid, sim))

        similarities.sort(key=lambda x: -x[1])
        seeds = [pid for pid, _ in similarities[:top_seeds]]
        seed_weights = {pid: sim for pid, sim in similarities[:top_seeds]}

        return self.query(seeds, top_k=top_k, seed_weights=seed_weights)

    # ── Internals ─────────────────────────────────────────────────────

    def _run_ppr(self, personalization: np.ndarray) -> np.ndarray:
        """Run PPR via fast-pagerank if available, else numpy power iteration."""
        pagerank_power = _try_import_fast_pagerank()

        if pagerank_power is not None:
            # fast-pagerank: pagerank_power(G, p=damping, personalize=v, tol=tol)
            try:
                scores = pagerank_power(
                    self._matrix,
                    p=self.damping,
                    personalize=personalization,
                    tol=self.tol,
                )
                return scores
            except Exception as e:
                logger.warning(f"fast-pagerank failed ({e}), falling back to power iteration")

        # Numpy fallback: power iteration
        return self._numpy_ppr(personalization)

    def _numpy_ppr(self, personalization: np.ndarray) -> np.ndarray:
        """
        Pure numpy Personalized PageRank via power iteration.
        r = α · A · r + (1 - α) · v
        """
        n = self._n_nodes
        # Row-normalize adjacency matrix
        A = self._matrix.astype(np.float64)
        row_sums = np.array(A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0  # avoid divide-by-zero for dangling nodes
        # Normalize rows
        from scipy.sparse import diags
        D_inv = diags(1.0 / row_sums)
        A_norm = D_inv @ A

        r = personalization.copy().astype(np.float64)
        v = personalization.astype(np.float64)

        for _ in range(self.max_iter):
            r_new = self.damping * A_norm.T.dot(r) + (1.0 - self.damping) * v
            delta = np.abs(r_new - r).sum()
            r = r_new
            if delta < self.tol:
                break

        return r.astype(np.float32)

    # ── Diagnostics ───────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "nodes": self._n_nodes,
            "edges": self._n_edges,
            "matrix_entries": int(self._matrix.nnz) if self._matrix is not None else 0,
            "matrix_mb": round(
                self._matrix.data.nbytes / 1024 / 1024, 2
            ) if self._matrix is not None else 0,
            "damping": self.damping,
        }

    def __repr__(self) -> str:
        return f"ASMEGraphPPR(nodes={self._n_nodes}, edges={self._n_edges}, α={self.damping})"


# ── CLI self-test ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent))
    from src.schema import init_schema
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    try:
        conn = init_schema(db_path)

        # Seed some edges
        test_edges = [
            ('UG-22', 'UW-11', 1.0),
            ('UG-22', 'UCS-66', 1.0),
            ('UW-11', 'QW-200', 1.0),
            ('UW-11', 'UW-51', 1.0),
            ('UCS-66', 'UG-84', 1.0),
            ('QW-200', 'QW-403', 1.0),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO graph_edges (source_id, target_id, weight) VALUES (?,?,?)",
            test_edges,
        )
        conn.commit()

        ppr = ASMEGraphPPR.from_sqlite(conn)
        print(f"Graph: {ppr.stats}")

        results = ppr.query(['UG-22'], top_k=10)
        print("\nPPR from UG-22 (top-10):")
        for pid, score in results:
            print(f"  {pid:20s}  {score:.6f}")

    finally:
        conn.close()
        os.unlink(db_path)
