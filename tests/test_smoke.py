"""Tests for scripts/smoke.sh — exercises the harness against a mock server.

Spins up a small FastAPI app on an ephemeral port that mimics the Trident
HTTP surface, seeds a temp SQLite with the conversation_logs schema, and
runs scripts/smoke.sh against it with SKIP_SYSTEMD=1. We assert exit code
and the [PASS]/[FAIL] lines for both the happy path and failure modes.
"""
from __future__ import annotations

import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_SH = REPO_ROOT / "scripts" / "smoke.sh"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_app(state: dict[str, Any]) -> FastAPI:
    """A mock Trident HTTP surface controlled by `state`.

    state keys:
      health_status: int (default 200)
      embedder: str ("real" | "stub" | "missing", default "real")
      asme_chunks: int (default 0)
      query_hits: int | "echo" — number of synthetic hits to return for /query.
                  "echo" means: return the count of conversation_logs that
                  contain the query text (so we exercise log_turn round-trip).
      stub_embedder_in_query: bool (default False) — sets stub_embedder field
                             in /query response.
      logs: list[dict] — populated by /log_turn calls.
      facts: list[dict] — populated by /ingest_fact calls.
    """
    app = FastAPI()

    @app.get("/health")
    def health():
        kind = state.get("embedder", "real")
        checks = {
            "sqlite_readable": True,
            "lancedb_readable": True,
            "anchors_intact": True,
            "embedder_loaded": kind != "missing",
            "embedder": kind,
        }
        code = state.get("health_status", 200)
        return JSONResponse(content={"status": "ok", "checks": checks}, status_code=code)

    @app.get("/stats")
    def stats():
        return {
            "corpus": {"asme_chunks": state.get("asme_chunks", 0), "graph_edges": 0,
                        "edge_types": {}, "reference_types": {},
                        "code_cases": 0, "interpretations": 0},
            "tiers": {"hot": 0, "warm": 0, "cold": 0},
        }

    @app.post("/log_turn")
    async def log_turn(request: Request):
        body = await request.json()
        state.setdefault("logs", []).append(body)
        return {"ok": True, "turn_id": len(state["logs"]) - 1}

    @app.post("/search_conversations")
    async def search_conversations(request: Request):
        body = await request.json()
        text = body.get("text", "")
        hits = [log for log in state.get("logs", [])
                if text in log.get("user", "") or text in log.get("assistant", "")]
        return {"results": [{"log_id": str(i), "content": h.get("user", ""),
                             "session_id": h.get("session_id", ""),
                             "role": "user", "created_at": "now", "rank": -1.0}
                            for i, h in enumerate(hits)]}

    @app.post("/query")
    async def query(request: Request):
        body = await request.json()
        text = body.get("text", "")
        hits_setting = state.get("query_hits", 1)
        if hits_setting == "echo":
            hit_count = sum(1 for log in state.get("logs", [])
                            if text in log.get("user", "") or text in log.get("assistant", ""))
            hit_count += sum(1 for f in state.get("facts", []) if text in f.get("value", ""))
        else:
            hit_count = int(hits_setting)
        results = [{"chunk_id": f"c{i}", "content": f"hit {i}", "score": 1.0,
                    "source": ["bm25"], "paragraph_id": ""}
                   for i in range(hit_count)]
        return {"results": results,
                "stub_embedder": bool(state.get("stub_embedder_in_query", False)),
                "query_ms": 1}

    @app.post("/ingest_fact")
    async def ingest_fact(request: Request):
        body = await request.json()
        state.setdefault("facts", []).append(body)
        pid = f"fact:{body.get('category', '')}:{body.get('key', '')}"
        return {"ok": True, "paragraph_id": pid, "chunk_id": f"chunk_{len(state['facts'])}"}

    return app


@contextmanager
def _mock_server(state: dict[str, Any]):
    port = _free_port()
    app = _build_app(state)
    config = uvicorn.Config(app, host="127.0.0.1", port=port,
                             log_level="error", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if server.started:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("mock server did not start in time")
    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _seed_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE conversation_logs (
            log_id TEXT PRIMARY KEY, session_id TEXT, turn_index INTEGER,
            role TEXT, content TEXT, created_at TEXT, consolidated INTEGER DEFAULT 0
        );
        CREATE INDEX idx_logs_session ON conversation_logs(session_id);
        CREATE INDEX idx_logs_created ON conversation_logs(created_at);
        CREATE VIRTUAL TABLE logs_fts USING fts5(log_id UNINDEXED, content);
    """)
    conn.commit()
    conn.close()


def _run_smoke(url: str, db_path: Path, env_extra: dict[str, str] | None = None,
               timeout: int = 30) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update({
        "TRIDENT_URL": url,
        "LEO_DB_PATH": str(db_path),
        "SKIP_SYSTEMD": "1",
    })
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(SMOKE_SH)],
        capture_output=True, text=True, env=env, timeout=timeout,
    )


class TestSmokeHarness(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.db = self.tmp / "leo_trident.db"
        _seed_db(self.db)

    def test_happy_path_all_pass(self):
        state = {"asme_chunks": 42, "query_hits": "echo"}
        with _mock_server(state) as port:
            result = _run_smoke(f"http://127.0.0.1:{port}", self.db)
        self.assertEqual(result.returncode, 0,
                         msg=f"stdout={result.stdout}\nstderr={result.stderr}")
        for tag in ("[1/7] health endpoint", "[2/7] stats endpoint",
                     "[3/7] schema sanity", "[4/7] log_turn round-trip",
                     "[5/7] query endpoint", "[6/7] ingest_fact round-trip",
                     "[7/7] systemd service"):
            self.assertIn(tag, result.stdout)
        self.assertIn("[PASS]", result.stdout)
        self.assertNotIn("[FAIL]", result.stdout)
        self.assertIn("SMOKE PASSED", result.stdout)
        self.assertIn("skipped (SKIP_SYSTEMD=1)", result.stdout)

    def test_service_unreachable_fails_step_1(self):
        # Use a definitely-unbound port — no server started.
        port = _free_port()
        result = _run_smoke(f"http://127.0.0.1:{port}", self.db, timeout=20)
        self.assertEqual(result.returncode, 1)
        self.assertIn("[1/7] health endpoint", result.stdout + result.stderr)
        self.assertIn("[FAIL]", result.stderr)
        self.assertIn("service unreachable", result.stderr)
        self.assertIn("SMOKE FAILED", result.stderr)

    def test_stub_embedder_fails_step_1(self):
        state = {"embedder": "stub"}
        with _mock_server(state) as port:
            result = _run_smoke(f"http://127.0.0.1:{port}", self.db)
        self.assertEqual(result.returncode, 1)
        self.assertIn("[FAIL]", result.stderr)
        self.assertIn("embedder=stub", result.stderr)

    def test_missing_schema_fails_step_3(self):
        # Empty DB without conversation_logs.
        bare_db = self.tmp / "bare.db"
        sqlite3.connect(str(bare_db)).close()
        state = {"asme_chunks": 0, "query_hits": 1}
        with _mock_server(state) as port:
            result = _run_smoke(f"http://127.0.0.1:{port}", bare_db)
        self.assertEqual(result.returncode, 1)
        self.assertIn("[1/7] health endpoint", result.stdout)
        self.assertIn("[2/7] stats endpoint", result.stdout)
        self.assertIn("[3/7] schema sanity", result.stderr)
        self.assertIn("missing schema object", result.stderr)

    def test_query_no_hits_fails_step_5(self):
        # Mock returns 0 hits even when text is logged.
        state = {"asme_chunks": 0, "query_hits": 0}
        with _mock_server(state) as port:
            result = _run_smoke(f"http://127.0.0.1:{port}", self.db)
        self.assertEqual(result.returncode, 1)
        self.assertIn("[5/7] query endpoint", result.stderr)
        self.assertIn("expected >=1 hit", result.stderr)

    def test_query_stub_flag_fails_step_5(self):
        state = {"asme_chunks": 0, "query_hits": 1,
                 "stub_embedder_in_query": True}
        with _mock_server(state) as port:
            result = _run_smoke(f"http://127.0.0.1:{port}", self.db)
        self.assertEqual(result.returncode, 1)
        self.assertIn("[5/7] query endpoint", result.stderr)
        self.assertIn("stub_embedder", result.stderr)


if __name__ == "__main__":
    unittest.main()
