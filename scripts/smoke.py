#!/usr/bin/env python3
"""Leo Trident smoke harness — JSON-emitting Python variant.

Same checks as scripts/smoke.sh but returns a structured summary suitable
for CI / cron jobs. Exits 0 on all-pass, 1 on first failure.

Usage:
    python scripts/smoke.py
    LEO_TRIDENT_HOME=... TRIDENT_URL=... python scripts/smoke.py --json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

DEFAULT_URL = "http://127.0.0.1:8765"
DEFAULT_HOME = Path(os.environ.get("LEO_TRIDENT_HOME", str(Path.home() / "leo_trident")))
SERVICE_NAME = os.environ.get("LEO_SERVICE_NAME", "leo-trident.service")
SKIP_SYSTEMD = os.environ.get("SKIP_SYSTEMD") == "1"


def _post(url: str, payload: dict[str, Any], timeout: float = 20.0) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode() or "{}")


def _get(url: str, timeout: float = 5.0) -> tuple[int, dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read().decode() or "{}")
        except Exception:
            data = {}
        return e.code, data


def run(url: str, home: Path) -> dict[str, Any]:
    nonce = uuid.uuid4().hex[:16]
    keyword = f"smoketokn{nonce}"
    session_id = f"smoke-{nonce}"
    fact_value = f"canary_{nonce}_unique"
    db_path = Path(os.environ.get("LEO_DB_PATH", str(home / "data" / "leo_trident.db")))

    steps: list[dict[str, Any]] = []

    def step(name: str, fn: Callable[[], str]) -> bool:
        t0 = time.monotonic()
        try:
            detail = fn()
            steps.append({
                "step": name, "status": "PASS", "detail": detail,
                "ms": int((time.monotonic() - t0) * 1000),
            })
            return True
        except Exception as e:  # noqa: BLE001
            steps.append({
                "step": name, "status": "FAIL", "detail": str(e),
                "ms": int((time.monotonic() - t0) * 1000),
            })
            return False

    def s_health() -> str:
        try:
            code, data = _get(f"{url}/health", timeout=15.0)
        except Exception as e:
            raise RuntimeError(f"service unreachable at {url}: {e}")
        if code != 200:
            raise RuntimeError(f"expected 200, got {code}")
        kind = data.get("checks", {}).get("embedder", "unknown")
        if kind == "stub":
            raise RuntimeError("embedder=stub (random vectors — NO-GO for prod)")
        if kind != "real":
            raise RuntimeError(f"embedder={kind} (expected real)")
        return f"200 ok, embedder=real"

    def s_stats() -> str:
        code, data = _get(f"{url}/stats")
        if code != 200:
            raise RuntimeError(f"expected 200, got {code}")
        chunks = data.get("corpus", {}).get("asme_chunks")
        if chunks is None:
            raise RuntimeError("missing corpus.asme_chunks")
        return f"valid JSON, asme_chunks={chunks}"

    def s_schema() -> str:
        if not db_path.exists():
            raise RuntimeError(f"db missing at {db_path}")
        conn = sqlite3.connect(str(db_path))
        try:
            names = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE name IN "
                "('conversation_logs','logs_fts','idx_logs_session','idx_logs_created')"
            )}
        finally:
            conn.close()
        for required in ("conversation_logs", "logs_fts", "idx_logs_session", "idx_logs_created"):
            if required not in names:
                raise RuntimeError(f"missing schema object: {required}")
        return "conversation_logs + indexes present"

    def s_log_turn() -> str:
        code, data = _post(f"{url}/log_turn", {
            "session_id": session_id,
            "user": f"ping {keyword}",
            "assistant": f"pong {keyword}",
        })
        if code != 200 or not data.get("ok"):
            raise RuntimeError(f"log_turn failed: {code} {data}")
        code, data = _post(f"{url}/search_conversations", {
            "text": keyword, "top_k": 5, "session_id": session_id,
        })
        if code != 200:
            raise RuntimeError(f"search_conversations: {code}")
        hits = len(data.get("results") or [])
        if hits < 1:
            raise RuntimeError(f"expected >=1 hit for '{keyword}', got {hits}")
        return f"logged + retrieved ({hits} hits)"

    def s_query() -> str:
        code, data = _post(f"{url}/query", {
            "text": keyword, "top_k": 5,
            "use_rerank": False, "include_conversations": True,
        })
        if code != 200:
            raise RuntimeError(f"query: {code}")
        if data.get("stub_embedder"):
            raise RuntimeError("stub_embedder=true in /query response")
        hits = len(data.get("results") or [])
        if hits < 1:
            raise RuntimeError(f"expected >=1 hit for '{keyword}', got {hits}")
        return f"{hits} hits for '{keyword}'"

    def s_ingest_fact() -> str:
        code, data = _post(f"{url}/ingest_fact", {
            "category": "smoke", "key": f"smokefact_{nonce}", "value": fact_value,
        })
        if code != 200 or not data.get("ok"):
            raise RuntimeError(f"ingest_fact: {code} {data}")
        code, data = _post(f"{url}/query", {
            "text": fact_value, "top_k": 10,
            "use_rerank": False, "include_conversations": False,
        })
        if code != 200:
            raise RuntimeError(f"query for fact: {code}")
        hits = len(data.get("results") or [])
        if hits < 1:
            raise RuntimeError(f"ingested fact not retrievable (0 hits)")
        return f"ingested + retrieved ({hits} hits)"

    def s_systemd() -> str:
        if SKIP_SYSTEMD:
            return "skipped (SKIP_SYSTEMD=1)"
        if not shutil.which("systemctl"):
            raise RuntimeError("systemctl not found (set SKIP_SYSTEMD=1 to skip)")
        out = subprocess.run(
            ["systemctl", "--user", "is-active", SERVICE_NAME],
            capture_output=True, text=True, timeout=5,
        )
        state = (out.stdout or "").strip()
        if state != "active":
            raise RuntimeError(f"{SERVICE_NAME} is '{state}', expected 'active'")
        return f"{SERVICE_NAME} active"

    sequence = [
        ("health endpoint", s_health),
        ("stats endpoint", s_stats),
        ("schema sanity", s_schema),
        ("log_turn round-trip", s_log_turn),
        ("query endpoint", s_query),
        ("ingest_fact round-trip", s_ingest_fact),
        ("systemd service", s_systemd),
    ]

    t_start = time.monotonic()
    for name, fn in sequence:
        if not step(name, fn):
            break
    elapsed = int((time.monotonic() - t_start) * 1000)
    passed = all(s["status"] == "PASS" for s in steps) and len(steps) == len(sequence)
    return {
        "passed": passed,
        "elapsed_ms": elapsed,
        "steps": steps,
        "url": url,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("TRIDENT_URL", DEFAULT_URL))
    ap.add_argument("--home", default=str(DEFAULT_HOME), type=Path)
    ap.add_argument("--json", action="store_true", help="emit JSON summary only")
    args = ap.parse_args()

    summary = run(args.url, args.home)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for i, s in enumerate(summary["steps"], 1):
            tag = "[PASS]" if s["status"] == "PASS" else "[FAIL]"
            print(f"[{i}/7] {s['step']:<32} {tag} {s['detail']}")
        verdict = "PASSED" if summary["passed"] else "FAILED"
        print(f"\nSMOKE {verdict} — {len(summary['steps'])}/7 steps in "
              f"{summary['elapsed_ms']}ms.")

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
