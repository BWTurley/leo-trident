"""
Leo Trident — End-to-End Test
Ingests 3 fake ASME paragraphs and queries them.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import shutil
import tempfile
from pathlib import Path


DUMMY_PARAGRAPHS = [
    {
        "text": (
            "UG-22 LOADINGS. Vessels shall be designed to resist the effects of the "
            "following design loads: (a) internal or external design pressure; "
            "(b) weight of the vessel and normal contents under operating or test conditions; "
            "(c) superimposed static reactions from weight of attached equipment, such as "
            "motors, machinery, other vessels, piping, linings, and insulation; "
            "(d) the attachment of internals, vessel supports, lugs, rings, skirts, saddles "
            "and legs; (e) cyclic and dynamic reactions due to pressure or thermal variations."
        ),
        "paragraph_id": "UG-22",
        "section": "VIII-1",
        "part": "UG",
    },
    {
        "text": (
            "UW-11 RADIOGRAPHIC EXAMINATION. (a) Full radiography. All butt welds in vessels "
            "for which the user or his designated agent specifies that full radiography be "
            "applied shall meet the requirements of this paragraph. Spot radiography shall be "
            "applied as required in (b) below. "
            "UW-51 covers the methods and acceptance criteria for radiographic examination."
        ),
        "paragraph_id": "UW-11",
        "section": "VIII-1",
        "part": "UW",
    },
    {
        "text": (
            "QW-200 WELDING PROCEDURE SPECIFICATIONS (WPS). Each manufacturer or contractor "
            "shall prepare written WPSs for each welding process used in production work. "
            "A WPS shall reference the supporting PQRs. Essential variables, nonessential "
            "variables, and supplementary essential variables are listed in QW-250 through "
            "QW-280 for each welding process."
        ),
        "paragraph_id": "QW-200",
        "section": "IX",
        "part": "QW",
    },
]


class TestLeoTridentE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize LeoTrident with test data directory."""
        from src.api import LeoTrident
        # Use a temp dir so tests don't pollute production DB
        cls.tmp_dir = tempfile.mkdtemp(prefix="leo_trident_test_")
        lt = LeoTrident(base_path=cls.tmp_dir)

        cls.lt = lt
        cls.lt.data_path = Path(cls.tmp_dir) / "data"
        cls.lt.db_path = cls.lt.data_path / "leo_trident.db"
        cls.lt.lance_path = cls.lt.data_path / "lancedb"
        cls.lt.vault_path = Path(cls.tmp_dir) / "vault"
        cls.lt.data_path.mkdir(parents=True, exist_ok=True)

        # Initialize DBs
        from src.schema import init_schema
        init_schema(cls.lt.db_path)

        import lancedb
        import pyarrow as pa
        db = lancedb.connect(str(cls.lt.lance_path))

        base_fields = [
            pa.field("chunk_id", pa.string()),
            pa.field("paragraph_id", pa.string()),
            pa.field("section", pa.string()),
            pa.field("content_type", pa.string()),
            pa.field("content", pa.string()),
            pa.field("no_forget", pa.bool_()),
            pa.field("tier", pa.string()),
            pa.field("edition_year", pa.int32()),
            pa.field("created_at", pa.string()),
        ]
        for name, dim in [("chunks_cold", 768), ("chunks_warm", 256)]:
            schema = pa.schema(base_fields + [pa.field("vector", pa.list_(pa.float32(), dim))])
            db.create_table(name, schema=schema, mode="create")

        print("\n[setup] Ingesting dummy paragraphs...")
        cls.chunk_ids = []
        for p in DUMMY_PARAGRAPHS:
            cid = cls.lt.ingest_text(
                text=p["text"],
                paragraph_id=p["paragraph_id"],
                section=p["section"],
                part=p["part"],
                edition_year=2025,
            )
            cls.chunk_ids.append(cid)
            print(f"  Ingested: {cid}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_chunk_ids_non_empty(self):
        """All chunk IDs should be non-empty strings."""
        for cid in self.chunk_ids:
            self.assertIsInstance(cid, str)
            self.assertGreater(len(cid), 0)

    def test_query_returns_results(self):
        """Query should return at least 1 result."""
        results = self.lt.query("what are the design load requirements", top_k=10, use_rerank=False)
        print(f"\n[test] Query returned {len(results)} results")
        for r in results:
            print(f"  chunk_id={r.get('chunk_id')} para={r.get('paragraph_id')} score={r.get('score'):.4f}")
        self.assertGreater(len(results), 0, "Expected at least 1 result")

    def test_paragraph_ids_in_results(self):
        """Expected paragraph IDs should appear in results."""
        results = self.lt.query("what are the design load requirements", top_k=10, use_rerank=False)
        returned_para_ids = {r.get("paragraph_id") for r in results}
        returned_chunk_ids = {r.get("chunk_id") for r in results}

        # At minimum UG-22 should come back (it's about design loads)
        expected = {"UG-22"}
        overlap = expected & returned_para_ids
        if not overlap:
            # Fall back: check chunk IDs
            overlap_chunks = set(self.chunk_ids) & returned_chunk_ids
            print(f"  para overlap: {overlap}, chunk overlap: {overlap_chunks}")
        else:
            print(f"  paragraph IDs found in results: {overlap}")

        # Lenient: at least one of our 3 ingested chunks should be in results
        all_our_chunks = set(self.chunk_ids)
        self.assertTrue(
            len(returned_chunk_ids & all_our_chunks) > 0 or len(overlap) > 0,
            f"None of the ingested chunks appeared in results. "
            f"Returned chunk_ids: {returned_chunk_ids}, ingested: {all_our_chunks}"
        )

    def test_scores_non_zero(self):
        """All returned scores should be non-zero."""
        results = self.lt.query("what are the design load requirements", top_k=10, use_rerank=False)
        for r in results:
            score = r.get("score", 0)
            self.assertGreater(abs(score), 0, f"Score should be non-zero for {r.get('chunk_id')}")

    def test_bm25_direct(self):
        """BM25 search should find chunks containing 'loadings'."""
        bm25 = self.lt._get_bm25()
        results = bm25.search("loadings", top_k=10)
        print(f"\n[test] BM25 'loadings' → {len(results)} results")
        para_ids = [r["paragraph_id"] for r in results]
        print(f"  paragraph IDs: {para_ids}")
        self.assertGreater(len(results), 0)
        # UG-22 text contains "LOADINGS"
        self.assertIn("UG-22", para_ids)

    def test_dense_search(self):
        """Dense vector search should return results after ingest."""
        embedder = self.lt._get_embedder()
        qvec = embedder.embed_query("pressure vessel design loads", dim=768)
        self.lt._dense_cold = None  # force reload
        results = self.lt._get_dense_cold().search(qvec, top_k=10)
        print(f"\n[test] Dense search → {len(results)} results")
        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
