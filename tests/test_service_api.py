"""
Tests for src/service/api.py — the Hermes-plugin-facing HTTP endpoints.
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from src.service import api as api_module


def _seed_db(path: Path) -> None:
    """Create just enough schema for log_turn."""
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversation_logs (
            log_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER,
            topics TEXT DEFAULT '[]',
            asme_refs TEXT DEFAULT '[]',
            created_at DATETIME,
            consolidated BOOLEAN DEFAULT FALSE
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS logs_fts USING fts5(
            log_id UNINDEXED, content, tokenize="unicode61"
        );
    """)
    conn.commit()
    conn.close()


class _FakeTrident:
    """Stand-in for LeoTrident that exposes _get_conn to a real sqlite file."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._using_stub = False
        self.query = MagicMock()
        self.ingest_text = MagicMock()
        self.search_conversations = MagicMock()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn


class ServiceApiTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.db"
        _seed_db(self.db_path)

        self.fake = _FakeTrident(self.db_path)
        api_module.reset_trident_for_tests()
        api_module._trident = self.fake

        self.client = TestClient(api_module.app, raise_server_exceptions=False)

    def tearDown(self):
        api_module.reset_trident_for_tests()
        self.tmp.cleanup()

    # ── /query ──────────────────────────────────────────────────────────

    def test_query_happy_path(self):
        canned = [
            {"chunk_id": "c1", "paragraph_id": "UG-22", "content": "x", "score": 0.9},
            {"chunk_id": "c2", "paragraph_id": "UW-11", "content": "y", "score": 0.5},
        ]
        self.fake.query.return_value = canned

        resp = self.client.post(
            "/query",
            json={"text": "pressure vessel design", "top_k": 5},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["results"], canned)
        self.assertFalse(body["stub_embedder"])
        self.assertIn("query_ms", body)
        self.assertIsInstance(body["query_ms"], int)

        # kwargs forwarded
        self.fake.query.assert_called_once()
        kwargs = self.fake.query.call_args.kwargs
        self.assertEqual(kwargs["text"], "pressure vessel design")
        self.assertEqual(kwargs["top_k"], 5)

    def test_query_missing_text_returns_400(self):
        resp = self.client.post("/query", json={"top_k": 5})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.json())

    def test_query_empty_text_returns_400(self):
        resp = self.client.post("/query", json={"text": ""})
        self.assertEqual(resp.status_code, 400)

    # ── /log_turn ───────────────────────────────────────────────────────

    def test_log_turn_inserts_row(self):
        resp = self.client.post(
            "/log_turn",
            json={"session_id": "s1", "user": "hi", "assistant": "hello"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["turn_id"], 0)

        # A second call in the same session should increment turn_id.
        resp2 = self.client.post(
            "/log_turn",
            json={"session_id": "s1", "user": "again", "assistant": "sure"},
        )
        self.assertEqual(resp2.json()["turn_id"], 1)

        conn = sqlite3.connect(str(self.db_path))
        (count,) = conn.execute("SELECT COUNT(*) FROM conversation_logs").fetchone()
        self.assertEqual(count, 4)  # 2 calls × (user + assistant)
        rows = conn.execute(
            "SELECT role, content FROM conversation_logs WHERE session_id='s1' AND turn_index=0 "
            "ORDER BY role"
        ).fetchall()
        self.assertEqual(rows, [("assistant", "hello"), ("user", "hi")])
        conn.close()

    def test_log_turn_empty_strings_returns_400(self):
        resp = self.client.post(
            "/log_turn",
            json={"session_id": "s1", "user": "", "assistant": "ok"},
        )
        self.assertEqual(resp.status_code, 400)

        resp = self.client.post(
            "/log_turn",
            json={"session_id": "", "user": "hi", "assistant": "ok"},
        )
        self.assertEqual(resp.status_code, 400)

    # ── /ingest_fact ────────────────────────────────────────────────────

    def test_ingest_fact_creates_paragraph(self):
        self.fake.ingest_text.return_value = "fact_profile_role_2025"

        resp = self.client.post(
            "/ingest_fact",
            json={
                "category": "profile",
                "key": "role",
                "value": "principal engineer",
                "confidence": 0.9,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["paragraph_id"], "fact:profile:role")

        self.fake.ingest_text.assert_called_once()
        kwargs = self.fake.ingest_text.call_args.kwargs
        self.assertEqual(kwargs["source"], "fact")
        self.assertEqual(kwargs["paragraph_id"], "fact:profile:role")
        self.assertIn("principal engineer", kwargs["text"])

    def test_ingest_fact_missing_key_returns_400(self):
        resp = self.client.post(
            "/ingest_fact",
            json={"category": "profile", "value": "x"},
        )
        self.assertEqual(resp.status_code, 400)

    # ── /search_conversations ───────────────────────────────────────────

    def test_search_conversations_happy_path(self):
        canned = [
            {
                "log_id": "abc",
                "session_id": "s1",
                "role": "user",
                "content": "about pressure vessels",
                "created_at": "2026-04-21T12:00:00Z",
                "rank": -1.23,
            }
        ]
        self.fake.search_conversations.return_value = canned

        resp = self.client.post(
            "/search_conversations",
            json={"text": "pressure", "top_k": 3, "session_id": "s1"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json(), {"results": canned})

        kwargs = self.fake.search_conversations.call_args.kwargs
        self.assertEqual(kwargs["text"], "pressure")
        self.assertEqual(kwargs["top_k"], 3)
        self.assertEqual(kwargs["session_id"], "s1")

    def test_search_conversations_missing_text_returns_400(self):
        resp = self.client.post("/search_conversations", json={"top_k": 3})
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
