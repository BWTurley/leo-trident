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
import sqlite3
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
        import threading
        self._write_lock = threading.Lock()

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

        # 7. Metrics
        try:
            from src.service.metrics import log_metric
            log_metric("consolidation.errors", len(summary["errors"]))
            log_metric("consolidation.facts_extracted", len(summary["facts_extracted"]))
            log_metric("consolidation.tier_changes", len(summary["tier_changes"]))
        except (ImportError, OSError) as e:
            logger.debug("Metrics logging skipped: %s", e)

        logger.info("=== Consolidation complete — %d errors ===", len(summary["errors"]))
        return summary

    def fact_extraction(self, text: str) -> list[dict]:
        """
        Call Claude to extract atomic facts and classify them as
        ADD / UPDATE / DELETE / NOOP against existing memory.
        """
        if len(text) > 4000:
            logger.warning("Truncating consolidation input from %d to 4000 chars", len(text))
        prompt = (
            "You are a memory consolidation agent for Leo Trident, an ASME retrieval system.\n"
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
        # Try strict JSON first, then fall back to extracting the largest [...] span.
        raw_stripped = raw.strip()
        # Strip markdown code fences if present
        if raw_stripped.startswith("```"):
            lines = raw_stripped.split("\n")
            raw_stripped = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        try:
            facts = json.loads(raw_stripped)
        except json.JSONDecodeError:
            # Find the outermost [...] using bracket counting, not regex
            start = raw.find("[")
            if start == -1:
                logger.warning("No JSON array found. Raw response (first 500 chars): %s", raw[:500])
                return []
            depth = 0
            end = -1
            for i in range(start, len(raw)):
                if raw[i] == "[": depth += 1
                elif raw[i] == "]":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end == -1:
                logger.warning("Unbalanced brackets in response: %s", raw[:500])
                return []
            try:
                facts = json.loads(raw[start:end])
            except json.JSONDecodeError as e:
                logger.warning("JSON parse error: %s. Raw: %s", e, raw[:500])
                return []

        return facts if isinstance(facts, list) else []

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
                    with self._write_lock:
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
        with self._write_lock:
            conn = self._write_db()
            conn.execute(
                "UPDATE tier_registry SET tier='warm' WHERE chunk_id=?",
                (chunk_id,),
            )
            conn.commit()

    def hot_recompression(self):
        """
        Re-read hot.json and update only _meta.generated_at.
        hot.json is now a static safety-pins file; no agent fields to regenerate.
        """
        hot_path = self.vault_path / "hot.json"
        try:
            with open(hot_path) as f:
                hot = json.load(f)
        except (json.JSONDecodeError, OSError):
            hot = {}

        hot.setdefault("_meta", {})
        hot["_meta"]["generated_at"] = datetime.now(timezone.utc).isoformat()

        with open(hot_path, "w") as f:
            json.dump(hot, f, indent=2)
        logger.info("hot.json regenerated")

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
        except (json.JSONDecodeError, OSError):
            log = {"runs": []}

        # Keep last 100 runs
        log.setdefault("runs", []).append(summary)
        log["runs"] = log["runs"][-100:]
        log["last_run"] = summary["run_at"]

        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        logger.info("Consolidation log updated")
