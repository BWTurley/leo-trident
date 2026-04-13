"""
Tests for Phase 8 — Health Endpoint
"""
import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient


def _seed_health_env(tmp_dir: str) -> Path:
    """Create a valid BASE_PATH for health checks."""
    base = Path(tmp_dir)
    data = base / "data"
    data.mkdir(parents=True, exist_ok=True)

    # SQLite
    db_path = data / "leo_trident.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS asme_chunks (
        chunk_id TEXT PRIMARY KEY, paragraph_id TEXT, content TEXT,
        section TEXT, part TEXT, edition_year INTEGER,
        content_hash TEXT, no_forget INTEGER DEFAULT 1,
        mandatory INTEGER DEFAULT 1, content_type TEXT DEFAULT 'normative',
        cross_refs TEXT DEFAULT '[]', raptor_level INTEGER DEFAULT 0,
        embedding_dim INTEGER DEFAULT 768,
        created_at TEXT, updated_at TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS graph_edges (
        edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id TEXT, target_id TEXT, edge_type TEXT DEFAULT 'cross_ref',
        reference_type TEXT DEFAULT 'mandatory',
        citation_text TEXT, context TEXT, weight REAL DEFAULT 1.0,
        edition_year INTEGER, created_at TEXT,
        UNIQUE(source_id, target_id, edge_type))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS tier_registry (
        memory_id TEXT PRIMARY KEY, content_type TEXT, tier TEXT,
        n_visit INTEGER DEFAULT 0, last_accessed TEXT,
        heat_score REAL DEFAULT 0.0, stability_days REAL DEFAULT 1.0,
        retention_r REAL DEFAULT 1.0, retention_at TEXT,
        no_forget INTEGER DEFAULT 0, version INTEGER DEFAULT 1,
        vault_path TEXT, created_at TEXT, updated_at TEXT)""")
    # Seed data
    conn.execute(
        "INSERT INTO asme_chunks (chunk_id, paragraph_id, content, content_hash, created_at, updated_at) "
        "VALUES ('c1', 'UG-22', 'Test', 'abc', '2026-04-13', '2026-04-13')"
    )
    conn.execute(
        "INSERT INTO asme_chunks (chunk_id, paragraph_id, content, content_hash, content_type, created_at, updated_at) "
        "VALUES ('c2', 'UG-27', 'Test2', 'def', 'normative', '2026-04-13', '2026-04-13')"
    )
    conn.execute(
        "INSERT INTO graph_edges (source_id, target_id, reference_type) VALUES ('UG-22', 'UW-11', 'mandatory')"
    )
    conn.execute(
        "INSERT INTO tier_registry (memory_id, content_type, tier) VALUES ('c1', 'asme_chunk', 'warm')"
    )
    conn.commit()
    conn.close()

    # LanceDB
    lance = data / "lancedb"
    lance.mkdir()
    (lance / "marker").write_text("exists")

    # vault/_system with valid anchors
    vault_sys = base / "vault" / "_system"
    vault_sys.mkdir(parents=True)

    rule = "waive UG-99 hydrostatic test"
    fact = "Primary focus: ASME BPVC Section VIII Division 1"
    anchors = {
        "_meta": {"version": 2},
        "asme_safety_pins": {
            "never": [{"rule": rule, "hash": hashlib.sha256(rule.encode()).hexdigest()}],
            "always": [],
        },
        "core_facts": [
            {"fact": fact, "hash": hashlib.sha256(fact.encode()).hexdigest()}
        ],
    }
    with open(vault_sys / "anchors.json", "w") as f:
        json.dump(anchors, f)

    # consolidation log
    with open(vault_sys / "consolidation_log.json", "w") as f:
        json.dump({"runs": [], "last_run": "2026-04-11T02:00:14Z"}, f)

    return base


class TestHealth(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.base = _seed_health_env(self.tmp)
        from src.service import health
        health.set_base_path(self.base)
        self.client = TestClient(health.app)

    def test_health_returns_200(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["checks"]["sqlite_readable"])
        self.assertTrue(data["checks"]["lancedb_readable"])
        self.assertTrue(data["checks"]["anchors_intact"])

    def test_health_returns_503_on_tampered_anchors(self):
        # Tamper anchors
        anchors_path = self.base / "vault" / "_system" / "anchors.json"
        with open(anchors_path) as f:
            anchors = json.load(f)
        anchors["asme_safety_pins"]["never"][0]["hash"] = "bad" * 16
        with open(anchors_path, "w") as f:
            json.dump(anchors, f)

        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 503)
        self.assertFalse(resp.json()["checks"]["anchors_intact"])

    def test_stats_returns_corpus_counts(self):
        resp = self.client.get("/stats")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["corpus"]["asme_chunks"], 2)
        self.assertEqual(data["corpus"]["graph_edges"], 1)
        self.assertEqual(data["tiers"]["warm"], 1)
        self.assertEqual(data["last_consolidation"], "2026-04-11T02:00:14Z")

    def test_version_returns_phase(self):
        resp = self.client.get("/version")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["phase"], "8")
        self.assertIn("6a", data["schema_migrations"])


if __name__ == "__main__":
    unittest.main()
