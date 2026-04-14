"""
Leo Trident — Unified Query & Ingest API
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import os

import numpy as np

from src.config import BASE_PATH as _DEFAULT_BASE_PATH

logger = logging.getLogger(__name__)


class LeoTrident:
    def __init__(self, base_path: str | Path = None):
        self.base_path = Path(base_path) if base_path else _DEFAULT_BASE_PATH
        self.data_path = self.base_path / "data"
        self.db_path = self.data_path / "leo_trident.db"
        self.lance_path = self.data_path / "lancedb"
        self.vault_path = self.base_path / "vault"

        self._embedder = None
        self._using_stub = False
        self._bm25 = None
        self._dense_cold = None
        self._dense_warm = None
        self._ppr = None
        self._reranker = None
        self._fusion = None

    # ── lazy loaders ─────────────────────────────────────────────────

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from src.ingest.embedder import Embedder
                self._embedder = Embedder()
                self._using_stub = False
            except Exception as e:
                if os.environ.get("LEO_ALLOW_STUB_EMBEDDER") != "1":
                    raise RuntimeError(
                        f"Real embedder failed to load ({e}). "
                        "Set LEO_ALLOW_STUB_EMBEDDER=1 to fall back to random "
                        "vectors (test/CI only — produces meaningless results)."
                    ) from e
                from src.ingest.stub_embedder import StubEmbedder
                logger.error("USING STUB EMBEDDER — search results are random")
                self._embedder = StubEmbedder()
                self._using_stub = True
        return self._embedder

    def _get_bm25(self):
        if self._bm25 is None:
            from src.retrieval.bm25 import BM25Retriever
            self._bm25 = BM25Retriever(str(self.db_path))
        return self._bm25

    def _get_dense_cold(self):
        if self._dense_cold is None:
            from src.retrieval.dense import DenseRetriever
            self._dense_cold = DenseRetriever(str(self.lance_path), table_name="chunks_cold")
        return self._dense_cold

    def _get_dense_warm(self):
        if self._dense_warm is None:
            from src.retrieval.dense import DenseRetriever
            self._dense_warm = DenseRetriever(str(self.lance_path), table_name="chunks_warm")
        return self._dense_warm

    def _get_ppr(self):
        if self._ppr is None:
            from src.retrieval.ppr import ASMEGraphPPR
            from src.schema import create_connection
            conn = create_connection(self.db_path)
            try:
                self._ppr = ASMEGraphPPR().load_from_sqlite(conn)
            finally:
                conn.close()
        return self._ppr

    def _get_reranker(self):
        if self._reranker is None:
            from src.retrieval.reranker import BGEReranker
            self._reranker = BGEReranker()
        return self._reranker

    def _get_fusion(self):
        if self._fusion is None:
            from src.retrieval.fusion import ReciprocalRankFusion
            self._fusion = ReciprocalRankFusion()
        return self._fusion

    def _get_conn(self) -> sqlite3.Connection:
        from src.schema import create_connection
        return create_connection(self.db_path)

    # ── query helpers ──────────────────────────────────────────────────

    def _run_bm25(self, text: str) -> List[dict]:
        """BM25 search via FTS5. Returns ranked results or empty list on failure."""
        try:
            return self._get_bm25().search(text, top_k=100)
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            return []

    def _run_dense(self, vec_768, vec_256) -> List[dict]:
        """Dense vector search across cold (768d) and warm (256d) tables."""
        results = []
        try:
            results = self._get_dense_cold().search(vec_768, top_k=100)
        except Exception as e:
            logger.warning(f"Dense cold search failed: {e}")
        try:
            warm = self._get_dense_warm().search(vec_256, top_k=50)
            results = results + warm
        except Exception as e:
            logger.warning(f"Dense warm search failed: {e}")
        return results

    def _run_ppr(self, bm25_results: List[dict]) -> List[dict]:
        """PPR search seeded by top BM25 paragraph IDs."""
        results = []
        try:
            seeds = [r["paragraph_id"] for r in bm25_results[:5] if r.get("paragraph_id")]
            if not seeds:
                return []
            raw_ppr = self._get_ppr().query(seeds, top_k=100)
            conn_ppr = self._get_conn()
            try:
                for pid, score in raw_ppr:
                    row = conn_ppr.execute(
                        "SELECT chunk_id, paragraph_id, content FROM asme_chunks WHERE paragraph_id=? LIMIT 1",
                        (pid,)
                    ).fetchone()
                    if row:
                        results.append({
                            "chunk_id": row["chunk_id"],
                            "paragraph_id": row["paragraph_id"],
                            "content": row["content"],
                            "score": score,
                        })
            finally:
                conn_ppr.close()
        except Exception as e:
            logger.warning(f"PPR search failed: {e}")
        return results

    def _fuse_and_enrich(self, bm25_results: List[dict], dense_results: List[dict],
                         ppr_results: List[dict], text: str,
                         include_conversations: bool) -> List[dict]:
        """RRF fusion + DB content enrichment. Returns candidate dicts."""
        from src.retrieval.fusion import RankedResult

        def to_ranked(results: List[dict], score_key: str = "score") -> List[RankedResult]:
            return [
                RankedResult(
                    doc_id=r.get("chunk_id", str(i)),
                    score=r.get(score_key, 0.0),
                    rank=i + 1,
                    content=r.get("content", ""),
                    paragraph_id=r.get("paragraph_id", ""),
                )
                for i, r in enumerate(results)
            ]

        lists = {}
        if bm25_results:
            lists["bm25"] = to_ranked(bm25_results, "rank")
        if dense_results:
            lists["dense"] = to_ranked(dense_results, "score")
        if ppr_results:
            lists["ppr"] = to_ranked(ppr_results, "score")

        if include_conversations:
            try:
                conv_results = self.search_conversations(text, top_k=5)
                if conv_results:
                    lists["conversations"] = [
                        RankedResult(
                            doc_id=f"log:{cr['log_id']}",
                            score=cr.get("rank", 0.0),
                            rank=i + 1,
                            content=cr["content"],
                            paragraph_id="",
                        )
                        for i, cr in enumerate(conv_results)
                    ]
            except Exception as e:
                logger.warning(f"Conversation search failed: {e}")

        if not lists:
            return []

        fused = self._get_fusion().fuse(lists)

        # Enrich content from DB
        conn = self._get_conn()
        chunk_content = {}
        chunk_para = {}
        try:
            ids = [f.doc_id for f in fused[:30]]
            if not ids:
                return []
            placeholders = ",".join("?" * len(ids))
            rows = conn.execute(
                f"SELECT chunk_id, paragraph_id, content FROM asme_chunks WHERE chunk_id IN ({placeholders})",
                ids,
            ).fetchall()
            for row in rows:
                chunk_content[row["chunk_id"]] = row["content"]
                chunk_para[row["chunk_id"]] = row["paragraph_id"]
        except Exception as e:
            logger.warning(f"DB content lookup failed: {e}")
        finally:
            conn.close()

        return [
            {
                "chunk_id": f.doc_id,
                "paragraph_id": f.paragraph_id or chunk_para.get(f.doc_id, ""),
                "content": f.content or chunk_content.get(f.doc_id, ""),
                "score": f.rrf_score,
                "source": list(f.ranks.keys()),
            }
            for f in fused[:30]
        ]

    # ── query ─────────────────────────────────────────────────────────

    def query(self, text: str, top_k: int = 10, use_rerank: bool = True,
              use_relevance_judge: bool = False,
              include_conversations: bool = False) -> List[dict]:
        """
        Hybrid BM25 + dense + PPR query with RRF fusion and optional BGE rerank.
        Returns top_k results with: chunk_id, paragraph_id, content, score, source.
        """
        _t0 = time.monotonic()
        embedder = self._get_embedder()
        query_vec_768 = embedder.embed_query(text, dim=768)
        query_vec_256 = embedder.embed_query(text, dim=256)

        bm25_results = self._run_bm25(text)
        dense_results = self._run_dense(query_vec_768, query_vec_256)
        ppr_results = self._run_ppr(bm25_results)

        candidates = self._fuse_and_enrich(
            bm25_results, dense_results, ppr_results, text, include_conversations
        )
        if not candidates:
            return []

        # Optional BGE rerank
        if use_rerank:
            try:
                reranker = self._get_reranker()
                candidates = reranker.rerank(text, candidates, top_k=top_k)
            except Exception as e:
                logger.warning(f"Reranker failed: {e}")
                candidates = candidates[:top_k]
        else:
            candidates = candidates[:top_k]

        # Optional relevance judgment
        if use_relevance_judge and candidates:
            try:
                from src.retrieval.relevance_judge import ReferenceRelevanceJudge
                judge = ReferenceRelevanceJudge()
                conn = self._get_conn()
                try:
                    candidates = judge.judge(text, candidates, conn)
                finally:
                    conn.close()
            except Exception as e:
                logger.warning(f"Relevance judge failed: {e} — returning unannotated results")

        if getattr(self, "_using_stub", False):
            for c in candidates:
                c["warning"] = "stub_embedder_random_vectors"

        # Metrics
        try:
            from src.service.metrics import log_metric
            latency_ms = round((time.monotonic() - _t0) * 1000, 1)
            log_metric("query.latency_ms", latency_ms, tags={
                "rerank": use_rerank,
                "judge": use_relevance_judge,
                "result_count": len(candidates),
            })
        except (ImportError, OSError) as e:
            logger.debug(f"Metrics logging skipped: {e}")

        return candidates

    # ── ingest ────────────────────────────────────────────────────────

    def ingest_text(
        self,
        text: str,
        paragraph_id: str,
        section: str,
        part: str,
        edition_year: int = 2025,
    ) -> str:
        """
        Parse + embed + store in LanceDB + FTS5 + graph.
        Returns chunk_id.
        """
        # Generate chunk_id
        chunk_id = f"{section}_{paragraph_id}_{edition_year}"
        chunk_id = chunk_id.replace(" ", "_").replace("/", "-")

        content_hash = hashlib.sha256(text.encode()).hexdigest()
        now = datetime.now(timezone.utc).isoformat()

        # Embed at 256d (warm) and 768d (cold)
        embedder = self._get_embedder()
        vec_768 = embedder.embed_documents([text], dim=768)[0]
        vec_256 = embedder.embed_documents([text], dim=256)[0]

        # Store in SQLite
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO asme_chunks
                   (chunk_id, paragraph_id, section, part, edition_year,
                    content, content_hash, no_forget, mandatory, content_type,
                    cross_refs, raptor_level, embedding_dim, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,1,1,'normative','[]',0,768,?,?)""",
                (chunk_id, paragraph_id, section, part, edition_year,
                 text, content_hash, now, now),
            )
            conn.commit()
        finally:
            conn.close()

        # Store in LanceDB (cold = 768d, warm = 256d)
        try:
            import lancedb
            import pyarrow as pa
            db = lancedb.connect(str(self.lance_path))

            for table_name, vec in [("chunks_cold", vec_768), ("chunks_warm", vec_256)]:
                try:
                    table = db.open_table(table_name)
                    # Remove existing entry (LanceDB doesn't have DELETE by filter easily in older versions)
                    # Just add — duplicates will be ranked lower
                    new_row = {
                        "chunk_id": chunk_id,
                        "paragraph_id": paragraph_id,
                        "section": section,
                        "content_type": "normative",
                        "content": text,
                        "no_forget": True,
                        "tier": "cold" if table_name == "chunks_cold" else "warm",
                        "edition_year": edition_year,
                        "created_at": now,
                        "vector": vec.tolist(),
                    }
                    table.add([new_row])
                except Exception as e:
                    logger.warning(f"LanceDB {table_name} insert error: {e}")
        except Exception as e:
            logger.warning(f"LanceDB ingest error: {e}")

        # Index in BM25 (chunks_fts via BM25Retriever)
        try:
            bm25 = self._get_bm25()
            bm25.index_chunk(chunk_id, paragraph_id, text)
        except Exception as e:
            logger.warning(f"BM25 index error: {e}")

        # Extract and store cross-reference edges via the parser
        try:
            from src.ingest.asme_parser import ASMEParser, GraphEdge, REFERENCE_TYPE_WEIGHTS
            parser = ASMEParser(edition_year=edition_year)
            refs = parser.extract_cross_refs_with_context(text, self_id=paragraph_id)
            if refs:
                edges = [
                    GraphEdge(
                        source_id=paragraph_id,
                        target_id=r['ref_id'],
                        edge_type='cross_ref',
                        reference_type=r['reference_type'],
                        citation_text=r['citation_text'],
                        context=r['context'],
                        weight=REFERENCE_TYPE_WEIGHTS[r['reference_type']],
                        edition_year=edition_year,
                    )
                    for r in refs
                ]
                conn2 = self._get_conn()
                try:
                    ASMEParser.insert_edges(conn2, edges)
                finally:
                    conn2.close()
        except Exception as e:
            logger.warning(f"Graph edge extraction failed: {e}")

        # Reload dense retriever handles
        self._dense_cold = None
        self._dense_warm = None

        return chunk_id

    # ── conversation search ──────────────────────────────────────────

    def search_conversations(self, text: str, top_k: int = 10,
                             hours: Optional[int] = None,
                             session_id: Optional[str] = None) -> list[dict]:
        """
        Search conversation history via FTS5.
        Returns list of dicts with: log_id, session_id, role, content, created_at, rank.
        """
        conn = self._get_conn()
        try:
            # Quote each token for FTS5 (handles hyphens like UW-51)
            fts_query = " ".join(f'"{w}"' for w in text.split())

            sql = (
                "SELECT cl.log_id, cl.session_id, cl.role, cl.content, cl.created_at, "
                "       logs_fts.rank "
                "FROM logs_fts "
                "JOIN conversation_logs cl ON cl.log_id = logs_fts.log_id "
                "WHERE logs_fts MATCH ?"
            )
            params: list = [fts_query]

            if hours is not None:
                cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
                sql += " AND cl.created_at >= ?"
                params.append(cutoff.isoformat())

            if session_id is not None:
                sql += " AND cl.session_id = ?"
                params.append(session_id)

            sql += " ORDER BY logs_fts.rank LIMIT ?"
            params.append(top_k)

            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"Conversation search failed: {e}")
            return []
        finally:
            conn.close()

