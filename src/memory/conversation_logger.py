"""
Leo Trident — Conversation Logger
Logs conversation turns to SQLite for sleep-time consolidation to process.
Matches the existing schema (conversation_logs table with log_id, created_at, consolidated).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from src.config import BASE_PATH as _DEFAULT_BASE_PATH

logger = logging.getLogger(__name__)


class ConversationLogger:
    """Log conversation turns to SQLite and retrieve them for consolidation."""

    def __init__(self, base_path: str | Path = None):
        base_path = Path(base_path) if base_path else _DEFAULT_BASE_PATH
        self.db_path = Path(base_path) / "data" / "leo_trident.db"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def log_turn(self, role: str, content: str, session_id: Optional[str] = None,
                 turn_index: int = 0) -> str:
        """Insert a conversation turn into conversation_logs. Returns log_id (UUID)."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        log_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO conversation_logs
                   (log_id, session_id, turn_index, role, content, created_at, consolidated)
                   VALUES (?, ?, ?, ?, ?, ?, 0)""",
                (log_id, session_id, turn_index, role, content, now),
            )
            conn.commit()
        return log_id

    def get_recent(self, hours: int = 4) -> list[dict]:
        """Fetch turns from the last N hours."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversation_logs WHERE created_at >= ? ORDER BY created_at ASC",
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_unprocessed(self) -> list[dict]:
        """Fetch turns not yet processed by consolidation."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversation_logs WHERE consolidated = 0 ORDER BY created_at ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_processed(self, turn_ids: list[str]):
        """Mark turns as consolidated (by log_id strings)."""
        if not turn_ids:
            return
        placeholders = ",".join("?" * len(turn_ids))
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                f"UPDATE conversation_logs SET consolidated=1, consolidation_at=? "
                f"WHERE log_id IN ({placeholders})",
                [now] + list(turn_ids),
            )
            conn.commit()
        logger.info("Marked %d turns as processed", len(turn_ids))
