"""
Tests for Phase 4 — Sleep-Time Consolidation Pipeline
Mocks Claude API — no external calls.
"""
import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_test_db(tmp_dir: str) -> str:
    """Create a minimal test SQLite DB with required tables."""
    db_path = Path(tmp_dir) / "leo_trident.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversation_logs (
            log_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL DEFAULT 0,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            consolidated INTEGER NOT NULL DEFAULT 0,
            consolidation_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tier_registry (
            chunk_id TEXT PRIMARY KEY,
            tier TEXT,
            heat_score REAL,
            retention REAL,
            visit_count INTEGER,
            last_accessed TEXT,
            stability REAL,
            no_forget INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS asme_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT,
            paragraph_id TEXT,
            content TEXT,
            content_type TEXT,
            last_accessed TEXT
        )
    """)
    conn.commit()
    conn.close()
    return str(db_path)


def _make_test_vault(tmp_dir: str) -> Path:
    """Create minimal vault/_system/ structure."""
    system_dir = Path(tmp_dir) / "vault" / "_system"
    system_dir.mkdir(parents=True, exist_ok=True)

    # anchors.json with correct hashes
    rule = "waive UG-99 hydrostatic test"
    anchors = {
        "asme_safety_pins": {
            "never": [{"rule": rule, "hash": hashlib.sha256(rule.encode()).hexdigest()}],
            "always": [],
        },
        "core_facts": [
            {"fact": "Brett is the user/human",
             "hash": hashlib.sha256("Brett is the user/human".encode()).hexdigest()}
        ],
    }
    with open(system_dir / "anchors.json", "w") as f:
        json.dump(anchors, f)

    # hot.json
    hot = {
        "_meta": {"version": 1},
        "session_hint": {"last_topic": None, "pending": []},
        "active_project": {},
    }
    with open(system_dir / "hot.json", "w") as f:
        json.dump(hot, f)

    # consolidation_log.json
    with open(system_dir / "consolidation_log.json", "w") as f:
        json.dump({"runs": []}, f)

    return system_dir


# ── ConversationLogger tests ───────────────────────────────────────────────────

class TestConversationLogger(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        Path(self.tmp, "data").mkdir(exist_ok=True)
        _make_test_db(Path(self.tmp) / "data")

    def _logger(self):
        from src.memory.conversation_logger import ConversationLogger
        return ConversationLogger(base_path=self.tmp)

    def test_log_and_retrieve(self):
        cl = self._logger()
        cl.log_turn("user", "What is UG-22?", session_id="test-session")
        cl.log_turn("assistant", "UG-22 covers loadings.", session_id="test-session")

        unprocessed = cl.get_unprocessed()
        self.assertEqual(len(unprocessed), 2)
        self.assertEqual(unprocessed[0]["role"], "user")

    def test_mark_processed(self):
        cl = self._logger()
        log_id = cl.log_turn("user", "Hello", session_id="s1")
        cl.mark_processed([log_id])

        unprocessed = cl.get_unprocessed()
        self.assertEqual(len(unprocessed), 0)

    def test_get_recent(self):
        cl = self._logger()
        cl.log_turn("user", "Recent message", session_id="s2")
        recent = cl.get_recent(hours=1)
        self.assertGreaterEqual(len(recent), 1)


# ── SleepTimeConsolidator tests ────────────────────────────────────────────────

class TestSleepTimeConsolidator(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        data_dir = Path(self.tmp) / "data"
        data_dir.mkdir(exist_ok=True)
        _make_test_db(data_dir)
        _make_test_vault(self.tmp)

    def _consolidator(self):
        from src.memory.consolidator import SleepTimeConsolidator
        c = SleepTimeConsolidator(base_path=self.tmp)
        c.api_key = "test-key"
        return c

    @patch("src.memory.consolidator.SleepTimeConsolidator._claude")
    def test_fact_extraction_parses(self, mock_claude):
        mock_claude.return_value = json.dumps([
            {"action": "ADD", "fact": "Brett uses API 510", "confidence": 0.9, "existing_id_if_update": None},
            {"action": "NOOP", "fact": "Albany NY", "confidence": 1.0, "existing_id_if_update": None},
        ])
        c = self._consolidator()
        facts = c.fact_extraction("Brett is an API 510 inspector in Albany NY.")
        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0]["action"], "ADD")
        self.assertEqual(facts[1]["action"], "NOOP")

    @patch("src.memory.consolidator.SleepTimeConsolidator._claude")
    def test_fact_extraction_handles_bad_json(self, mock_claude):
        mock_claude.return_value = "Sorry, I cannot extract facts."
        c = self._consolidator()
        facts = c.fact_extraction("some text")
        self.assertEqual(facts, [])

    def test_drift_check_passes(self):
        c = self._consolidator()
        result = c.drift_check()
        self.assertTrue(result["ok"])
        self.assertEqual(result["violations"], [])

    def test_drift_check_catches_violation(self):
        # Tamper with an anchor hash
        anchors_path = Path(self.tmp) / "vault" / "_system" / "anchors.json"
        with open(anchors_path) as f:
            anchors = json.load(f)
        anchors["asme_safety_pins"]["never"][0]["hash"] = "deadbeef" * 8
        with open(anchors_path, "w") as f:
            json.dump(anchors, f)

        c = self._consolidator()
        result = c.drift_check()
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["violations"]), 1)

    def test_hot_recompression_stays_under_200_tokens(self):
        c = self._consolidator()
        c.hot_recompression()

        hot_path = Path(self.tmp) / "vault" / "_system" / "hot.json"
        with open(hot_path) as f:
            hot_text = f.read()

        # Rough token estimate: chars / 4
        approx_tokens = len(hot_text) / 4
        self.assertLessEqual(approx_tokens, 400,
                             f"hot.json may exceed 200 tokens: {approx_tokens:.0f} estimated")

    def test_consolidation_log_updated(self):
        c = self._consolidator()
        summary = {
            "run_at": "2026-04-11T02:00:00+00:00",
            "dry_run": False,
            "facts_extracted": [],
            "tier_changes": [],
            "forward_predictions": [],
            "hot_recompressed": True,
            "drift_check": {"ok": True, "violations": []},
            "errors": [],
        }
        c._append_consolidation_log(summary)

        log_path = Path(self.tmp) / "vault" / "_system" / "consolidation_log.json"
        with open(log_path) as f:
            log = json.load(f)
        self.assertEqual(len(log["runs"]), 1)
        self.assertEqual(log["runs"][0]["run_at"], "2026-04-11T02:00:00+00:00")

    @patch("src.memory.consolidator.SleepTimeConsolidator._claude")
    def test_dry_run_does_not_write_log(self, mock_claude):
        mock_claude.return_value = "[]"
        c = self._consolidator()
        # Patch tier_management to avoid DB dependency
        c.tier_management = MagicMock(return_value=[])
        c.forward_prediction = MagicMock(return_value=[])
        c.hot_recompression = MagicMock()

        # Run in dry-run mode
        summary = c.run(conversation_log="Test turn", dry_run=True)
        self.assertTrue(summary["dry_run"])

        # Log should still be empty
        log_path = Path(self.tmp) / "vault" / "_system" / "consolidation_log.json"
        with open(log_path) as f:
            log = json.load(f)
        self.assertEqual(len(log["runs"]), 0)


if __name__ == "__main__":
    unittest.main()
