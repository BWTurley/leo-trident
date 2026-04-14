"""
Leo Trident — Dense (LanceDB) Retriever
Supports warm (256d) and cold (768d) tables.
"""
from __future__ import annotations
import numpy as np
from typing import List
import lancedb


class DenseRetriever:
    def __init__(self, db_path: str, table_name: str = "cold"):
        self.db_path = db_path
        self.table_name = table_name
        self._db = None
        self._table = None

    def _get_table(self):
        if self._table is None:
            self._db = lancedb.connect(self.db_path)
            self._table = self._db.open_table(self.table_name)
        return self._table

    def search(self, query_vec: np.ndarray, top_k: int = 100) -> List[dict]:
        """
        Vector search using cosine similarity.
        Returns list of dicts with keys: chunk_id, paragraph_id, score, content.
        """
        try:
            table = self._get_table()
            results = (
                table.search(query_vec.tolist())
                .metric("cosine")
                .limit(top_k)
                .to_list()
            )
            output = []
            for row in results:
                output.append({
                    "chunk_id": row.get("chunk_id", ""),
                    "paragraph_id": row.get("paragraph_id", ""),
                    "score": float(1.0 - row.get("_distance", 1.0)),  # cosine: 1 - distance
                    "content": row.get("content", ""),
                })
            return output
        except Exception:
            # Table might not exist yet
            return []

    def reload(self):
        """Force reload table handle (after ingest)."""
        self._table = None
        self._db = None
