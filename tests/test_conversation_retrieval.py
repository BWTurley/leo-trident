"""
Tests for Phase 8 — Conversation Retrieval via FTS5
"""
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _seed_conversation_db(tmp_dir: str) -> Path:
    """Create a BASE_PATH with conversation_logs + FTS5 + minimal schema."""
    base = Path(tmp_dir)
    data = base / "data"
    data.mkdir(parents=True, exist_ok=True)

    db_path = data / "leo_trident.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")

    # Minimal tables
    conn.execute("""CREATE TABLE IF NOT EXISTS asme_chunks (
        chunk_id TEXT PRIMARY KEY, paragraph_id TEXT, content TEXT,
        section TEXT, part TEXT, edition_year INTEGER,
        content_hash TEXT, no_forget INTEGER DEFAULT 1,
        mandatory INTEGER DEFAULT 1, content_type TEXT DEFAULT 'normative',
        cross_refs TEXT DEFAULT '[]', raptor_level INTEGER DEFAULT 0,
        embedding_dim INTEGER DEFAULT 768, created_at TEXT, updated_at TEXT)""")
    conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        chunk_id UNINDEXED, paragraph_id, content, tokenize="unicode61")""")
    conn.execute("""CREATE TABLE IF NOT EXISTS graph_edges (
        edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id TEXT, target_id TEXT, edge_type TEXT DEFAULT 'cross_ref',
        reference_type TEXT DEFAULT 'unclassified',
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
    conn.execute("""CREATE TABLE IF NOT EXISTS conversation_logs (
        log_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
        turn_index INTEGER NOT NULL, role TEXT NOT NULL,
        content TEXT NOT NULL, token_count INTEGER,
        topics TEXT DEFAULT '[]', asme_refs TEXT DEFAULT '[]',
        created_at TEXT, consolidated INTEGER DEFAULT 0,
        consolidation_at TEXT)""")
    conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS logs_fts USING fts5(
        log_id UNINDEXED, content, tokenize="unicode61")""")

    # FTS sync trigger
    conn.execute("""CREATE TRIGGER IF NOT EXISTS logs_fts_insert
        AFTER INSERT ON conversation_logs BEGIN
        INSERT INTO logs_fts(log_id, content) VALUES (new.log_id, new.content);
        END""")

    now = datetime.now(timezone.utc)
    old = (now - timedelta(hours=2)).isoformat()
    recent = now.isoformat()

    # Session 1 — recent, mentions UW-51
    conn.execute(
        "INSERT INTO conversation_logs (log_id, session_id, turn_index, role, content, created_at) "
        "VALUES ('log1', 'sess-A', 0, 'user', 'What are the UW-51 spot radiographic examination requirements?', ?)",
        (recent,),
    )
    conn.execute(
        "INSERT INTO conversation_logs (log_id, session_id, turn_index, role, content, created_at) "
        "VALUES ('log2', 'sess-A', 1, 'assistant', 'UW-51 requires spot RT for category joints per UW-11.', ?)",
        (recent,),
    )

    # Session 2 — old, mentions UCS-66
    conn.execute(
        "INSERT INTO conversation_logs (log_id, session_id, turn_index, role, content, created_at) "
        "VALUES ('log3', 'sess-B', 0, 'user', 'How does UCS-66 impact MDMT calculations?', ?)",
        (old,),
    )
    conn.execute(
        "INSERT INTO conversation_logs (log_id, session_id, turn_index, role, content, created_at) "
        "VALUES ('log4', 'sess-B', 1, 'assistant', 'UCS-66 provides impact test exemption curves.', ?)",
        (old,),
    )

    # Session 1 again — mentions UG-99
    conn.execute(
        "INSERT INTO conversation_logs (log_id, session_id, turn_index, role, content, created_at) "
        "VALUES ('log5', 'sess-A', 2, 'user', 'What about UG-99 hydrostatic test procedures?', ?)",
        (recent,),
    )

    conn.commit()
    conn.close()

    # LanceDB dir (needed for LeoTrident init)
    (data / "lancedb").mkdir(exist_ok=True)

    # vault/_system
    vault = base / "vault" / "_system"
    vault.mkdir(parents=True, exist_ok=True)
    (vault / "hot.json").write_text('{"_meta": {"version": 2}}')
    (vault / "anchors.json").write_text('{"_meta": {"version": 2}, "asme_safety_pins": {"never": [], "always": []}, "core_facts": []}')

    return base


class TestConversationRetrieval(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.base = _seed_conversation_db(self.tmp)

    def _lt(self):
        from src.api import LeoTrident
        return LeoTrident(base_path=str(self.base))

    def test_search_conversations_basic(self):
        lt = self._lt()
        results = lt.search_conversations("UW-51")
        self.assertGreater(len(results), 0)
        contents = [r["content"] for r in results]
        self.assertTrue(any("UW-51" in c for c in contents))

    def test_search_conversations_time_filter(self):
        lt = self._lt()
        # hours=1 should only get recent turns
        results = lt.search_conversations("UCS-66", hours=1)
        # The UCS-66 turns are 2 hours old — should be excluded
        self.assertEqual(len(results), 0)

    def test_search_conversations_session_filter(self):
        lt = self._lt()
        results = lt.search_conversations("UW-51", session_id="sess-A")
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r["session_id"], "sess-A")

        # Wrong session
        results_b = lt.search_conversations("UW-51", session_id="sess-B")
        self.assertEqual(len(results_b), 0)

    def test_query_include_conversations(self):
        lt = self._lt()
        results = lt.query("UW-51", top_k=10, use_rerank=False,
                           include_conversations=True)
        # Should include conversation results with "conversations" in source
        conv_results = [r for r in results if "conversations" in r.get("source", [])]
        # With just conversation data and no ASME chunks, we should get conversation results
        self.assertGreater(len(conv_results), 0)

    def test_query_default_excludes_conversations(self):
        lt = self._lt()
        results = lt.query("UW-51", top_k=10, use_rerank=False)
        # Default should NOT include conversations
        conv_results = [r for r in results if "conversations" in r.get("source", [])]
        self.assertEqual(len(conv_results), 0)


if __name__ == "__main__":
    unittest.main()
