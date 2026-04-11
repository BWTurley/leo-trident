"""
Leo Trident — BM25 Retriever (SQLite FTS5)
Preserves ASME identifiers like UG-22, UW-11.a.2 via unicode61 tokenchars.
"""
from __future__ import annotations
import sqlite3
from typing import List


class BM25Retriever:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_fts_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_fts_table(self):
        """Create or rebuild FTS5 table with correct tokenizer."""
        with self._connect() as conn:
            # Check if table exists with wrong tokenizer (hyphen-first breaks unicode61)
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='chunks_fts'"
            ).fetchone()
            if row and "\'-._\'" in row[0]:
                # Broken tokenizer — drop and recreate
                conn.execute("DROP TABLE IF EXISTS chunks_fts")
                conn.commit()
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(
                    chunk_id UNINDEXED,
                    paragraph_id UNINDEXED,
                    content,
                    tokenize="unicode61"
                )
            """)
            conn.commit()

    def index_chunk(self, chunk_id: str, paragraph_id: str, content: str):
        """Insert or replace a chunk in the FTS index."""
        with self._connect() as conn:
            # Remove existing entry for this chunk_id
            conn.execute(
                "DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,)
            )
            conn.execute(
                "INSERT INTO chunks_fts(chunk_id, paragraph_id, content) VALUES (?, ?, ?)",
                (chunk_id, paragraph_id, content.lower()),
            )
            conn.commit()

    def search(self, query: str, top_k: int = 100) -> List[dict]:
        """
        BM25 search via FTS5.
        Returns list of dicts with keys: chunk_id, paragraph_id, rank, content.
        """
        # Escape special FTS5 characters; lowercase for case-insensitive match
        safe_query = query.lower().replace('"', '""')
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT chunk_id, paragraph_id, content, rank
                    FROM chunks_fts
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (safe_query, top_k),
                ).fetchall()
                return [
                    {
                        "chunk_id": row["chunk_id"],
                        "paragraph_id": row["paragraph_id"],
                        "rank": float(row["rank"]),
                        "content": row["content"],
                    }
                    for row in rows
                ]
        except sqlite3.OperationalError:
            return []
