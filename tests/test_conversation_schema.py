"""
Tests for conversation_logs schema migration (task 02).

Covers:
- fresh DB gets the table
- migration is idempotent
- /log_turn-style writes flow through to LeoTrident.search_conversations
- FTS5 stays in sync on insert / update / delete
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["LEO_ALLOW_STUB_EMBEDDER"] = "1"

from src.schema import init_schema


def _table_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name=? AND type IN ('table','view')",
        (name,),
    ).fetchone()
    return bool(row)


def _trigger_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name=? AND type='trigger'", (name,)
    ).fetchone()
    return bool(row)


def _index_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name=? AND type='index'", (name,)
    ).fetchone()
    return bool(row)


class ConversationSchemaTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "leo_trident.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_table_created_fresh_db(self):
        conn = init_schema(self.db_path)
        try:
            self.assertTrue(_table_exists(conn, "conversation_logs"))
            self.assertTrue(_table_exists(conn, "logs_fts"))

            cols = {row[1] for row in conn.execute("PRAGMA table_info(conversation_logs)")}
            for required in {"log_id", "session_id", "turn_index", "role",
                             "content", "created_at"}:
                self.assertIn(required, cols)

            self.assertTrue(_index_exists(conn, "idx_logs_session"))
            self.assertTrue(_index_exists(conn, "idx_logs_session_ts"))
            self.assertTrue(_trigger_exists(conn, "logs_fts_insert"))
            self.assertTrue(_trigger_exists(conn, "logs_fts_delete"))
            self.assertTrue(_trigger_exists(conn, "logs_fts_update"))
        finally:
            conn.close()

    def test_migration_idempotent(self):
        c1 = init_schema(self.db_path)
        c1.execute(
            "INSERT INTO conversation_logs (log_id, session_id, turn_index, role, content, created_at) "
            "VALUES (?, 's1', 0, 'user', 'hello', ?)",
            (str(uuid.uuid4()), datetime.now(timezone.utc).isoformat()),
        )
        c1.commit()
        (count_before,) = c1.execute("SELECT COUNT(*) FROM conversation_logs").fetchone()
        c1.close()

        # Re-run migration on the same DB — must not raise, must not duplicate
        c2 = init_schema(self.db_path)
        try:
            (count_after,) = c2.execute("SELECT COUNT(*) FROM conversation_logs").fetchone()
            self.assertEqual(count_before, count_after)
            self.assertEqual(count_after, 1)

            triggers = [
                row[0] for row in c2.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger' "
                    "AND tbl_name='conversation_logs'"
                )
            ]
            # Each trigger is defined once even after a second run
            self.assertEqual(sorted(triggers), sorted(set(triggers)))
        finally:
            c2.close()

    def test_log_turn_flows_to_search(self):
        """The critical integration: rows inserted with the /log_turn shape
        must come back from LeoTrident.search_conversations."""
        # Build a base_path the way LeoTrident expects.
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

        db_path = base / "data" / "leo_trident.db"
        conn = init_schema(db_path)
        now = datetime.now(timezone.utc).isoformat()
        # Insert /log_turn shape: paired user/assistant rows sharing turn_index.
        rows = [
            ("s1", 0, "user", "what is the bzqx_distinctive_keyword spec?"),
            ("s1", 0, "assistant", "the bzqx_distinctive_keyword spec is XYZ."),
            ("s1", 1, "user", "any other notes?"),
        ]
        for session_id, turn_index, role, content in rows:
            conn.execute(
                "INSERT INTO conversation_logs (log_id, session_id, turn_index, "
                "role, content, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), session_id, turn_index, role, content, now),
            )
        conn.commit()
        conn.close()

        from src.api import LeoTrident
        lt = LeoTrident(base_path=str(base))
        results = lt.search_conversations(text="bzqx_distinctive_keyword")

        self.assertEqual(len(results), 2, results)
        contents = [r["content"] for r in results]
        self.assertTrue(all("bzqx_distinctive_keyword" in c for c in contents))

        scoped = lt.search_conversations(
            text="bzqx_distinctive_keyword", session_id="s1"
        )
        self.assertEqual(len(scoped), 2)

    def test_fts_sync(self):
        conn = init_schema(self.db_path)
        try:
            log_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO conversation_logs (log_id, session_id, turn_index, "
                "role, content, created_at) VALUES (?, 's1', 0, 'user', 'first text', ?)",
                (log_id, now),
            )
            conn.commit()
            (n,) = conn.execute(
                "SELECT COUNT(*) FROM logs_fts WHERE log_id = ?", (log_id,)
            ).fetchone()
            self.assertEqual(n, 1, "insert trigger missed")

            # UPDATE → FTS row reflects new content
            conn.execute(
                "UPDATE conversation_logs SET content = 'second text' WHERE log_id = ?",
                (log_id,),
            )
            conn.commit()
            row = conn.execute(
                "SELECT content FROM logs_fts WHERE log_id = ?", (log_id,)
            ).fetchone()
            self.assertEqual(row[0], "second text", "update trigger missed")

            # DELETE → FTS row gone
            conn.execute("DELETE FROM conversation_logs WHERE log_id = ?", (log_id,))
            conn.commit()
            (n,) = conn.execute(
                "SELECT COUNT(*) FROM logs_fts WHERE log_id = ?", (log_id,)
            ).fetchone()
            self.assertEqual(n, 0, "delete trigger missed")
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
