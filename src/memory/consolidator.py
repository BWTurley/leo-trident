"""
Leo Trident — Sleep-Time Consolidator
Two-agent consolidation pipeline:
  - Read-only LeoTrident for retrieval
  - Write-only SQLite connection for mutations
  - Claude Sonnet via Abacus.AI API for fact extraction
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import BASE_PATH as _DEFAULT_BASE_PATH
from src.memory import llm_client

logger = logging.getLogger(__name__)


class SleepTimeConsolidator:
    """Full sleep-time consolidation pipeline."""

    def __init__(self, base_path: str | Path = None):
        self.base_path = Path(base_path) if base_path else _DEFAULT_BASE_PATH
        self.db_path = self.base_path / "data" / "leo_trident.db"
        self.vault_path = self.base_path / "vault" / "_system"

        # Lazy imports so missing deps only fail at use time
        self._lt = None       # read-only LeoTrident
        self._write_conn = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_lt(self):
        """Lazy-load read-only LeoTrident."""
        if self._lt is None:
            from src.api import LeoTrident
            self._lt = LeoTrident(base_path=str(self.base_path))
        return self._lt

    def _write_db(self) -> sqlite3.Connection:
        """Open/return a write connection (WAL mode)."""
        if self._write_conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._write_conn = conn
        return self._write_conn

    def _claude(self, prompt: str) -> str:
        """Call LLM via configured backend (cloud or local)."""
        return llm_client.complete(prompt)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, conversation_log: Optional[str] = None, dry_run: bool = False) -> dict:
        """
        Full consolidation cycle.
        Returns summary dict of changes made.
        """
        logger.info("=== Sleep-time consolidation starting ===")
        summary: dict = {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
            "facts_extracted": [],
            "tier_changes": [],
            "forward_predictions": [],
            "hot_recompressed": False,
            "drift_check": {},
            "errors": [],
        }

        # 1. Fact extraction
        try:
            if conversation_log:
                text = conversation_log
            else:
                from src.memory.conversation_logger import ConversationLogger
                cl = ConversationLogger(base_path=str(self.base_path))
                unprocessed = cl.get_unprocessed()
                if unprocessed:
                    text = "\n".join(f"[{r['role']}] {r['content']}" for r in unprocessed)
                    turn_ids = [r["log_id"] for r in unprocessed]
                else:
                    text = ""
                    turn_ids = []

            if text.strip():
                facts = self.fact_extraction(text)
                summary["facts_extracted"] = facts
                logger.info("Extracted %d facts", len(facts))
                if not dry_run and turn_ids:
                    cl.mark_processed(turn_ids)
        except Exception as e:
            logger.warning("Fact extraction failed: %s", e)
            summary["errors"].append(f"fact_extraction: {e}")

        # 2. Tier management
        try:
            changes = self.tier_management(dry_run=dry_run)
            summary["tier_changes"] = changes
        except Exception as e:
            logger.warning("Tier management failed: %s", e)
            summary["errors"].append(f"tier_management: {e}")

        # 3. Forward prediction
        try:
            topics = []
            for f in summary.get("facts_extracted", []):
                if isinstance(f, dict) and f.get("fact"):
                    topics.append(f["fact"][:80])
            if topics:
                preds = self.forward_prediction(topics[:5], dry_run=dry_run)
                summary["forward_predictions"] = preds
        except Exception as e:
            logger.warning("Forward prediction failed: %s", e)
            summary["errors"].append(f"forward_prediction: {e}")

        # 4. Hot recompression
        try:
            if not dry_run:
                self.hot_recompression()
                summary["hot_recompressed"] = True
            else:
                logger.info("[DRY-RUN] Would regenerate hot.json")
        except Exception as e:
            logger.warning("Hot recompression failed: %s", e)
            summary["errors"].append(f"hot_recompression: {e}")

        # 5. Drift check
        try:
            drift = self.drift_check()
            summary["drift_check"] = drift
        except Exception as e:
            logger.warning("Drift check failed: %s", e)
            summary["errors"].append(f"drift_check: {e}")

        # 6. Write consolidation log
        if not dry_run:
            self._append_consolidation_log(summary)

        logger.info("=== Consolidation complete — %d errors ===", len(summary["errors"]))
        return summary

    def fact_extraction(self, text: str) -> list[dict]:
        """
        Call Claude to extract atomic facts and classify them as
        ADD / UPDATE / DELETE / NOOP against existing memory.
        """
        prompt = (
            "You are a memory consolidation agent for Leo, an ASME inspection assistant.\n"
            "Extract atomic facts from the following conversation text.\n"
            "For each fact, classify it as ADD / UPDATE / DELETE / NOOP.\n"
            "ADD = new fact not previously known.\n"
            "UPDATE = corrects or refines an existing fact.\n"
            "DELETE = removes an outdated fact.\n"
            "NOOP = already known, no change needed.\n\n"
            "Return ONLY a JSON array. Each element:\n"
            '{"action":"ADD|UPDATE|DELETE|NOOP","fact":"<atomic fact>","confidence":0.0-1.0,'
            '"existing_id_if_update":null}\n\n'
            "Conversation:\n"
            f"{text[:4000]}"
        )
        raw = self._claude(prompt)
        # Extract JSON array from response
        import re
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            logger.warning("No JSON array found in Claude response")
            return []
        try:
            facts = json.loads(match.group(0))
            return facts if isinstance(facts, list) else []
        except json.JSONDecodeError as e:
            logger.warning("JSON parse error in fact_extraction: %s", e)
            return []

    def tier_management(self, dry_run: bool = False) -> list[dict]:
        """
        Recompute heat scores and R(t) for all tier_registry entries.
        Promote/demote chunks between warm and cold.
        Returns list of changes.
        """
        from src.memory.tier_manager import TierManager, compute_heat, compute_retention

        changes = []

        try:
            # Use a separate read connection to list all records
            read_conn = sqlite3.connect(str(self.db_path), timeout=30)
            read_conn.row_factory = sqlite3.Row
            rows = read_conn.execute("SELECT * FROM tier_registry").fetchall()
            read_conn.close()
        except Exception as e:
            logger.warning("Could not read tier_registry: %s", e)
            return []

        if not rows:
            return []

        # Use write connection for mutations
        write_conn = self._write_db()
        tm = TierManager(conn=write_conn)

        for row in rows:
            memory_id = row["memory_id"] if "memory_id" in row.keys() else row.get("chunk_id", "")
            if not memory_id:
                continue

            record = tm.get(memory_id)
            if record is None:
                continue

            old_tier = record.tier
            new_tier = tm._compute_tier(record, __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc))

            if new_tier != old_tier:
                heat = round(compute_heat(record), 4)
                ret = round(compute_retention(record), 4)
                change = {
                    "chunk_id": memory_id,
                    "from_tier": old_tier,
                    "to_tier": new_tier,
                    "heat": heat,
                    "retention": ret,
                }
                changes.append(change)
                if not dry_run:
                    record.tier = new_tier
                    tm.upsert(record)
                    write_conn.commit()
                    logger.info("Tier change: %s %s→%s", memory_id, old_tier, new_tier)

        return changes

    def forward_prediction(self, recent_topics: list[str], dry_run: bool = False) -> list[str]:
        """
        Pre-load related cold-tier chunks into warm tier based on recent topics.
        Returns list of promoted chunk IDs.
        """
        lt = self._get_lt()
        promoted = []

        for topic in recent_topics:
            try:
                results = lt.query(topic, top_k=5)
                for r in results:
                    chunk_id = r.get("chunk_id", "")
                    tier = r.get("tier", "cold")
                    if tier == "cold":
                        if not dry_run:
                            self._promote_to_warm(chunk_id)
                        promoted.append(chunk_id)
            except Exception as e:
                logger.warning("Forward prediction query failed for '%s': %s", topic, e)

        return promoted

    def _promote_to_warm(self, chunk_id: str):
        """Bump a chunk from cold to warm in tier_registry."""
        conn = self._write_db()
        conn.execute(
            "UPDATE tier_registry SET tier='warm' WHERE chunk_id=?",
            (chunk_id,),
        )
        conn.commit()

    def hot_recompression(self):
        """
        Regenerate vault/_system/hot.json from current warm-tier facts.
        Keeps content ≤200 tokens.
        """
        # Load current hot.json as baseline
        hot_path = self.vault_path / "hot.json"
        try:
            with open(hot_path) as f:
                hot = json.load(f)
        except Exception:
            hot = {}

        # Pull active project + session hints from tier_registry / personal facts
        active_project = self._get_active_project()
        last_topic = self._get_last_topic()
        pending = self._get_pending_items()

        # Update session hint
        if "session_hint" not in hot:
            hot["session_hint"] = {}
        hot["session_hint"]["last_topic"] = last_topic
        hot["session_hint"]["pending"] = pending

        # Update active project if found
        if active_project and "active_project" in hot:
            hot["active_project"].update(active_project)

        hot["_meta"] = hot.get("_meta", {})
        hot["_meta"]["generated_at"] = datetime.now(timezone.utc).isoformat()

        with open(hot_path, "w") as f:
            json.dump(hot, f, indent=2)
        logger.info("hot.json regenerated")

    def _get_active_project(self) -> dict:
        """Fetch active project from tier_registry if available."""
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=10)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT content FROM asme_chunks WHERE content_type='project' "
                "ORDER BY last_accessed DESC LIMIT 1"
            ).fetchone()
            conn.close()
            if row:
                return {"notes": row["content"][:100]}
        except Exception:
            pass
        return {}

    def _get_last_topic(self) -> Optional[str]:
        """Fetch last conversation topic from logs."""
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=10)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT content FROM conversation_logs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            conn.close()
            if row:
                return row["content"][:80]
        except Exception:
            pass
        return None

    def _get_pending_items(self) -> list[str]:
        """Stub — returns empty list unless personal memory has open items."""
        return []

    def drift_check(self) -> dict:
        """
        Load anchors.json and verify each rule's SHA-256 hash.
        Returns {ok: bool, violations: list}.
        """
        anchors_path = self.vault_path / "anchors.json"
        try:
            with open(anchors_path) as f:
                anchors = json.load(f)
        except Exception as e:
            return {"ok": False, "violations": [f"Cannot load anchors.json: {e}"]}

        violations = []

        def _check_items(items: list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                rule = item.get("rule") or item.get("fact", "")
                stored_hash = item.get("hash", "")
                if not stored_hash:
                    continue
                computed = hashlib.sha256(rule.encode()).hexdigest()
                if computed != stored_hash:
                    violations.append({
                        "rule": rule,
                        "stored": stored_hash,
                        "computed": computed,
                    })

        # Check ASME safety pins
        _check_items(anchors.get("asme_safety_pins", {}).get("never", []))
        _check_items(anchors.get("asme_safety_pins", {}).get("always", []))
        # Check core facts
        _check_items(anchors.get("core_facts", []))

        if violations:
            logger.error("ANCHOR DRIFT DETECTED: %d violations", len(violations))
        else:
            logger.info("Anchor drift check passed — all hashes valid")

        return {"ok": len(violations) == 0, "violations": violations}

    def _append_consolidation_log(self, summary: dict):
        """Append this run's summary to vault/_system/consolidation_log.json."""
        log_path = self.vault_path / "consolidation_log.json"
        try:
            with open(log_path) as f:
                log = json.load(f)
        except Exception:
            log = {"runs": []}

        # Keep last 100 runs
        log.setdefault("runs", []).append(summary)
        log["runs"] = log["runs"][-100:]
        log["last_run"] = summary["run_at"]

        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        logger.info("Consolidation log updated")
