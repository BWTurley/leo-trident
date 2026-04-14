"""
Leo Trident — Reciprocal Rank Fusion (RRF)
Fuses ranked lists from BM25, dense vector search, and PPR.

RRF_score(d) = Σ_r [ 1 / (k + rank_r(d)) ]   with k = 60  (default)

References:
  - Cormack et al. (2009) — original RRF paper
  - HippoRAG 2 (arXiv:2502.14802) — PPR + RRF for multi-hop retrieval
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

ALLOWED_FTS_TABLES = frozenset({"chunks_fts", "asme_fts"})


@dataclass
class RankedResult:
    """A single document/chunk in a ranked list."""
    doc_id: str
    score: float = 0.0           # original retrieval score (BM25 / cosine / PPR)
    rank: int = 0                # 1-indexed rank within its source list
    content: Optional[str] = None
    paragraph_id: Optional[str] = None
    section: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class FusedResult:
    """A document after RRF fusion, with component rank details."""
    doc_id: str
    rrf_score: float
    ranks: dict[str, int]        # {"bm25": 3, "dense": 7, "ppr": 12}
    scores: dict[str, float]     # original scores per source
    content: Optional[str] = None
    paragraph_id: Optional[str] = None
    section: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def appeared_in(self) -> list[str]:
        return list(self.ranks.keys())

    def __repr__(self) -> str:
        sources = ', '.join(f'{s}@{r}' for s, r in sorted(self.ranks.items()))
        return f'FusedResult({self.doc_id!r}, rrf={self.rrf_score:.4f}, [{sources}])'


class ReciprocalRankFusion:
    """
    Merges multiple ranked lists via Reciprocal Rank Fusion.

    Usage:
        rrf = ReciprocalRankFusion(k=60)
        fused = rrf.fuse({
            "bm25":  [RankedResult("UG-22", score=12.3), ...],
            "dense": [RankedResult("UG-22", score=0.91), ...],
            "ppr":   [RankedResult("UW-11", score=0.05), ...],
        })
        top_10 = fused[:10]
    """

    def __init__(self, k: int = 60):
        """
        Args:
            k: smoothing constant. k=60 is the original RRF recommendation.
               Larger k reduces the impact of top-ranked items.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k

    def fuse(
        self,
        ranked_lists: dict[str, list[RankedResult]],
        weights: Optional[dict[str, float]] = None,
        top_n: Optional[int] = None,
    ) -> list[FusedResult]:
        """
        Fuse multiple ranked lists.

        Args:
            ranked_lists: dict mapping source name → ranked list (ordered best-first).
                          Items are automatically assigned ranks 1..N.
            weights: optional per-source weight multiplier (default 1.0 for all).
                     E.g. {"bm25": 0.8, "dense": 1.0, "ppr": 1.2}
            top_n: if set, return only top N results after fusion.

        Returns:
            List of FusedResult sorted by RRF score descending.
        """
        if weights is None:
            weights = {}

        # Accumulate per-document RRF scores
        # doc_id → {"rrf_score": float, "ranks": {src: rank}, "scores": {src: score},
        #           "result": RankedResult}
        accumulated: dict[str, dict] = {}

        for source_name, results in ranked_lists.items():
            w = weights.get(source_name, 1.0)
            for rank_idx, item in enumerate(results):
                rank = rank_idx + 1  # 1-indexed
                rrf_contribution = w / (self.k + rank)

                if item.doc_id not in accumulated:
                    accumulated[item.doc_id] = {
                        "rrf_score": 0.0,
                        "ranks": {},
                        "scores": {},
                        "result": item,
                    }

                acc = accumulated[item.doc_id]
                acc["rrf_score"] += rrf_contribution
                acc["ranks"][source_name] = rank
                acc["scores"][source_name] = item.score

                # Prefer richer result objects (with content)
                if item.content and not acc["result"].content:
                    acc["result"] = item

        # Build FusedResult list
        fused = []
        for doc_id, acc in accumulated.items():
            ref = acc["result"]
            fused.append(FusedResult(
                doc_id=doc_id,
                rrf_score=acc["rrf_score"],
                ranks=acc["ranks"],
                scores=acc["scores"],
                content=ref.content,
                paragraph_id=ref.paragraph_id,
                section=ref.section,
                metadata=ref.metadata,
            ))

        # Sort by RRF score descending; break ties by doc_id for determinism
        fused.sort(key=lambda x: (-x.rrf_score, x.doc_id))

        if top_n is not None:
            fused = fused[:top_n]

        return fused

    def fuse_simple(
        self,
        *ranked_lists: list[str],
        source_names: Optional[list[str]] = None,
        weights: Optional[dict[str, float]] = None,
        top_n: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """
        Simplified interface: accepts lists of doc_id strings (already ranked).
        Returns list of (doc_id, rrf_score) tuples sorted by score.

        Args:
            *ranked_lists: positional ranked lists of doc_id strings.
            source_names: optional names for each list (default: "source_0", "source_1", ...).
        """
        names = source_names or [f"source_{i}" for i in range(len(ranked_lists))]
        input_dict = {}
        for name, ranked in zip(names, ranked_lists):
            input_dict[name] = [
                RankedResult(doc_id=doc_id, rank=i + 1)
                for i, doc_id in enumerate(ranked)
            ]
        fused = self.fuse(input_dict, weights=weights, top_n=top_n)
        return [(r.doc_id, r.rrf_score) for r in fused]


# ── Convenience wrappers ──────────────────────────────────────────────────────

def rrf_fuse(
    bm25_results: list[RankedResult],
    dense_results: list[RankedResult],
    ppr_results: Optional[list[RankedResult]] = None,
    k: int = 60,
    weights: Optional[dict[str, float]] = None,
    top_n: Optional[int] = None,
) -> list[FusedResult]:
    """
    Standard Leo Trident 3-way fusion (BM25 + dense + PPR).
    PPR results are optional — if None, fuses only BM25 + dense.
    """
    rrf = ReciprocalRankFusion(k=k)
    ranked = {
        "bm25": bm25_results,
        "dense": dense_results,
    }
    if ppr_results is not None:
        ranked["ppr"] = ppr_results

    return rrf.fuse(ranked, weights=weights, top_n=top_n)


def bm25_from_sqlite(
    conn,
    query: str,
    limit: int = 100,
    table: str = 'chunks_fts',
) -> list[RankedResult]:
    """
    Run BM25 search against SQLite FTS5 asme_fts table.
    Returns ranked list of RankedResult.
    """
    if table not in ALLOWED_FTS_TABLES:
        raise ValueError(f"table must be one of {sorted(ALLOWED_FTS_TABLES)}, got {table!r}")
    rows = conn.execute(
        f"""
        SELECT f.chunk_id, rank,
               c.paragraph_id, c.section, c.content
        FROM {table} f
        JOIN asme_chunks c ON c.chunk_id = f.chunk_id
        WHERE {table} MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()

    results = []
    for i, row in enumerate(rows):
        results.append(RankedResult(
            doc_id=row['chunk_id'],
            score=float(-row['rank']),   # FTS5 rank is negative
            rank=i + 1,
            content=row['content'],
            paragraph_id=row['paragraph_id'],
            section=row['section'],
        ))
    return results


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    rrf = ReciprocalRankFusion(k=60)

    # Toy example from the spec
    bm25 = [RankedResult(f"doc_{i}", score=10 - i) for i in range(10)]
    dense = [RankedResult(f"doc_{i*2 % 10}", score=0.9 - i * 0.08) for i in range(10)]
    ppr = [RankedResult(f"doc_{(i+1) % 10}", score=0.1 - i * 0.005) for i in range(10)]

    results = rrf_fuse(bm25, dense, ppr, k=60, top_n=5)
    print("Top-5 RRF results:")
    for r in results:
        print(f"  {r}")
