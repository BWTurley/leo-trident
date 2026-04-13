"""
Tests for Phase 8 — Backup Job
"""
import json
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _seed_test_env(tmp_dir: str) -> Path:
    """Create a seeded BASE_PATH with SQLite DB and LanceDB directory."""
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
        reference_type TEXT DEFAULT 'unclassified',
        citation_text TEXT, context TEXT, weight REAL DEFAULT 1.0,
        edition_year INTEGER, created_at TEXT,
        UNIQUE(source_id, target_id, edge_type))""")
    conn.execute(
        "INSERT INTO asme_chunks (chunk_id, paragraph_id, content, content_hash, created_at, updated_at) "
        "VALUES ('test_chunk_1', 'UG-22', 'Test content', 'abc123', '2026-04-13', '2026-04-13')"
    )
    conn.execute(
        "INSERT INTO graph_edges (source_id, target_id) VALUES ('UG-22', 'UW-11')"
    )
    conn.commit()
    conn.close()

    # Fake LanceDB directory
    lance = data / "lancedb"
    lance.mkdir()
    (lance / "test_table.lance").mkdir()
    (lance / "test_table.lance" / "data.bin").write_text("fake lance data")

    # vault/_system
    vault_sys = base / "vault" / "_system"
    vault_sys.mkdir(parents=True)
    (vault_sys / "anchors.json").write_text('{"_meta": {"version": 2}}')
    (vault_sys / "hot.json").write_text('{"_meta": {"version": 2}}')

    return base


class TestBackup(unittest.TestCase):

    def test_backup_creates_valid_snapshot(self):
        tmp = tempfile.mkdtemp()
        base = _seed_test_env(tmp)

        from scripts.backup import main
        backup_dir = main(base_path=base, retention_days=30)

        # Backup dir exists
        self.assertTrue(backup_dir.exists())

        # Manifest is valid
        manifest_path = backup_dir / "manifest.json"
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["asme_chunk_count"], 1)
        self.assertEqual(manifest["graph_edge_count"], 1)
        self.assertGreater(manifest["sqlite_bytes"], 0)
        self.assertGreater(manifest["lancedb_bytes"], 0)

        # SQLite copy is readable
        backup_db = backup_dir / "leo_trident.db"
        self.assertTrue(backup_db.exists())
        conn = sqlite3.connect(str(backup_db))
        count = conn.execute("SELECT COUNT(*) FROM asme_chunks").fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)

        # LanceDB copy has same structure
        lance_copy = backup_dir / "lancedb"
        self.assertTrue(lance_copy.exists())
        self.assertTrue((lance_copy / "test_table.lance" / "data.bin").exists())

        # vault_system copy
        vault_copy = backup_dir / "vault_system"
        self.assertTrue(vault_copy.exists())
        self.assertTrue((vault_copy / "anchors.json").exists())

    def test_prune_removes_old_backups(self):
        tmp = tempfile.mkdtemp()
        base = _seed_test_env(tmp)

        from scripts.backup import prune_backups

        # Create a fake old backup directory with a past timestamp
        old_ts = "20260101T030000Z"
        old_dir = base / "backups" / old_ts
        old_dir.mkdir(parents=True)
        (old_dir / "manifest.json").write_text('{}')
        self.assertTrue(old_dir.exists())

        # Create a fresh backup
        from scripts.backup import main
        fresh = main(base_path=base, retention_days=30)
        self.assertTrue(fresh.exists())

        # Old one should have been pruned (it's >30 days old)
        self.assertFalse(old_dir.exists())
        # Fresh one should still exist
        self.assertTrue(fresh.exists())

    def test_dry_run_creates_nothing(self):
        tmp = tempfile.mkdtemp()
        base = _seed_test_env(tmp)

        from scripts.backup import main
        result = main(base_path=base, retention_days=30, dry_run=True)

        # Backup dir should not exist
        self.assertFalse(result.exists())


if __name__ == "__main__":
    unittest.main()
