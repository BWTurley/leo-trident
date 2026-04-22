"""
Real (non-mocked) round-trip test for /ingest_fact -> /query.

Verifies that a fact ingested via the /ingest_fact endpoint can be retrieved
via /query against the same DB. Uses the stub embedder (BM25 carries the
recall here, not dense vectors).
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["LEO_ALLOW_STUB_EMBEDDER"] = "1"

from fastapi.testclient import TestClient

from src.schema import init_schema
from src.service import api as api_module


class IngestFactRoundTripTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        (base / "data").mkdir(parents=True, exist_ok=True)
        (base / "data" / "lancedb").mkdir(exist_ok=True)
        vault = base / "vault" / "_system"
        vault.mkdir(parents=True, exist_ok=True)
        (vault / "hot.json").write_text('{"_meta": {"version": 2}}')
        (vault / "anchors.json").write_text(
            '{"_meta": {"version": 2}, '
            '"asme_safety_pins": {"never": [], "always": []}, '
            '"core_facts": []}'
        )

        init_schema(base / "data" / "leo_trident.db").close()

        from src.api import LeoTrident
        api_module.reset_trident_for_tests()
        api_module._trident = LeoTrident(base_path=str(base))

        self.client = TestClient(api_module.app, raise_server_exceptions=False)

    def tearDown(self):
        api_module.reset_trident_for_tests()
        self.tmp.cleanup()

    def test_ingest_fact_then_query_round_trip(self):
        ingest = self.client.post(
            "/ingest_fact",
            json={
                "category": "hsb",
                "key": "start_date",
                "value": "2026-05-11",
            },
        )
        self.assertEqual(ingest.status_code, 200, ingest.text)
        body = ingest.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["paragraph_id"], "fact:hsb:start_date")

        q = self.client.post(
            "/query",
            json={
                "text": "hsb start date",
                "top_k": 10,
                "use_rerank": False,
                "use_relevance_judge": False,
                "include_conversations": False,
            },
        )
        self.assertEqual(q.status_code, 200, q.text)
        results = q.json()["results"]
        self.assertGreater(len(results), 0, "expected ingested fact in results")

        para_ids = {r.get("paragraph_id", "") for r in results}
        self.assertIn("fact:hsb:start_date", para_ids)


if __name__ == "__main__":
    unittest.main()
