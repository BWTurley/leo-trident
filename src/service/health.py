"""
Leo Trident — Health HTTP endpoint

Tiny FastAPI app exposing system state for monitoring. Binds to 127.0.0.1
by default. Read-only; never mutates state.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.config import BASE_PATH as _DEFAULT_BASE_PATH

logger = logging.getLogger(__name__)

app = FastAPI(title="Leo Trident Health", docs_url=None, redoc_url=None)

# Resolved once at import; overridable via set_base_path() for testing
_base_path: Path = _DEFAULT_BASE_PATH


def set_base_path(p: Path):
    """Override base_path for testing."""
    global _base_path
    _base_path = p


def _db_path() -> Path:
    return _base_path / "data" / "leo_trident.db"


def _lance_path() -> Path:
    return _base_path / "data" / "lancedb"


def _vault_system() -> Path:
    return _base_path / "vault" / "_system"


def _read_conn() -> sqlite3.Connection:
    """Open a fresh read-only SQLite connection."""
    conn = sqlite3.connect(str(_db_path()), timeout=5)
    conn.execute("PRAGMA query_only=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 1)


def _check_anchors() -> bool:
    """Quick SHA-256 check on anchors.json."""
    anchors_path = _vault_system() / "anchors.json"
    try:
        with open(anchors_path) as f:
            anchors = json.load(f)
    except Exception:
        return False

    def _check(items: list) -> bool:
        for item in items:
            if not isinstance(item, dict):
                continue
            rule = item.get("rule") or item.get("fact", "")
            stored = item.get("hash", "")
            if not stored:
                continue
            if hashlib.sha256(rule.encode()).hexdigest() != stored:
                return False
        return True

    ok = _check(anchors.get("asme_safety_pins", {}).get("never", []))
    ok = ok and _check(anchors.get("asme_safety_pins", {}).get("always", []))
    ok = ok and _check(anchors.get("core_facts", []))
    return ok


def _check_embedder() -> bool:
    """Check if the embedding model can be loaded."""
    try:
        from src.ingest.embedder import Embedder
        Embedder()
        return True
    except Exception:
        try:
            from src.ingest.stub_embedder import StubEmbedder
            StubEmbedder()
            return True
        except Exception:
            return False


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    checks = {
        "sqlite_readable": _db_path().exists(),
        "lancedb_readable": _lance_path().exists(),
        "anchors_intact": _check_anchors(),
        "embedder_loaded": _check_embedder(),
    }

    all_ok = all(checks.values())
    status_code = 200 if all_ok else 503

    return JSONResponse(
        content={
            "status": "ok" if all_ok else "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
        },
        status_code=status_code,
    )


@app.get("/stats")
def stats():
    corpus = {
        "asme_chunks": 0,
        "graph_edges": 0,
        "edge_types": {},
        "reference_types": {},
        "code_cases": 0,
        "interpretations": 0,
    }
    tiers = {"hot": 0, "warm": 0, "cold": 0}
    last_consolidation = None
    last_backup = None

    if _db_path().exists():
        try:
            conn = _read_conn()

            corpus["asme_chunks"] = conn.execute(
                "SELECT COUNT(*) FROM asme_chunks"
            ).fetchone()[0]

            corpus["graph_edges"] = conn.execute(
                "SELECT COUNT(*) FROM graph_edges"
            ).fetchone()[0]

            for row in conn.execute(
                "SELECT edge_type, COUNT(*) as cnt FROM graph_edges GROUP BY edge_type"
            ):
                corpus["edge_types"][row["edge_type"]] = row["cnt"]

            for row in conn.execute(
                "SELECT reference_type, COUNT(*) as cnt FROM graph_edges GROUP BY reference_type"
            ):
                corpus["reference_types"][row["reference_type"]] = row["cnt"]

            corpus["code_cases"] = conn.execute(
                "SELECT COUNT(*) FROM asme_chunks WHERE content_type='code_case'"
            ).fetchone()[0]

            corpus["interpretations"] = conn.execute(
                "SELECT COUNT(*) FROM asme_chunks WHERE content_type='interpretation'"
            ).fetchone()[0]

            for row in conn.execute(
                "SELECT tier, COUNT(*) as cnt FROM tier_registry GROUP BY tier"
            ):
                tiers[row["tier"]] = row["cnt"]

            conn.close()
        except Exception as e:
            logger.warning("Stats DB read failed: %s", e)

    # Last consolidation
    cons_log = _vault_system() / "consolidation_log.json"
    if cons_log.exists():
        try:
            with open(cons_log) as f:
                log = json.load(f)
            last_consolidation = log.get("last_run")
        except Exception:
            pass

    # Last backup
    backups_dir = _base_path / "backups"
    if backups_dir.exists():
        backup_dirs = sorted(
            [d for d in backups_dir.iterdir() if d.is_dir()], reverse=True
        )
        if backup_dirs:
            manifest = backup_dirs[0] / "manifest.json"
            if manifest.exists():
                try:
                    with open(manifest) as f:
                        m = json.load(f)
                    last_backup = m.get("timestamp")
                except Exception:
                    pass

    return {
        "corpus": corpus,
        "tiers": tiers,
        "last_consolidation": last_consolidation,
        "last_backup": last_backup,
        "disk": {
            "sqlite_mb": _dir_size_mb(_db_path().parent) if _db_path().exists() else 0.0,
            "lancedb_mb": _dir_size_mb(_lance_path()),
            "vault_mb": _dir_size_mb(_vault_system()),
        },
    }


@app.get("/version")
def version():
    git_sha = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            git_sha = result.stdout.strip()
    except Exception:
        pass

    return {
        "phase": "8",
        "schema_migrations": ["6a", "6b"],
        "git_sha": git_sha,
    }
