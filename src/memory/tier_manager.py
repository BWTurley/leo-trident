"""
Leo Trident — Tier Manager
Manages memory tier transitions (hot/warm/cold) using:
  - Heat score:  H(m) = α·N_visit + β·R_recency + γ·L_interaction
  - FSRS forgetting curve: R(t) = (1 + t/S)^(-0.5)

References:
  - MemoryOS (arXiv:2506.06326) — heat-score eviction
  - FSRS (github: open-spaced-repetition/free-spaced-repetition-scheduler) — retention model
  - TiMem (arXiv:2601.02845) — adaptive budget allocation
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Heat score weights (α, β, γ) — tunable
ALPHA = 0.4   # N_visit weight
BETA  = 0.4   # R_recency weight
GAMMA = 0.2   # L_interaction weight

# Tier transition thresholds
COLD_TO_WARM_HEAT   = 0.4   # or ≥2 accesses in 7 days
WARM_TO_HOT_HEAT    = 0.7   # or ≥5 accesses in 3 days
HOT_TO_WARM_R       = 0.7   # R(t) < 0.7 or 7 days no access → demote hot
WARM_TO_COLD_R      = 0.5   # R(t) < 0.5 or 30 days no access → demote warm

# FSRS initial stability (days)
STABILITY_INIT = 1.0         # new personal facts
STABILITY_MAX  = 365.0       # after many accesses

# Tier names
TIER_HOT   = "hot"
TIER_WARM  = "warm"
TIER_COLD  = "cold"


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class MemoryRecord:
    memory_id: str
    content_type: str        # personal_fact / project_note / episodic_log / asme_chunk
    tier: str
    n_visit: int = 0
    last_accessed: Optional[datetime] = None
    created_at: Optional[datetime] = None
    heat_score: float = 0.0
    stability_days: float = STABILITY_INIT
    retention_r: float = 1.0
    retention_at: Optional[datetime] = None
    no_forget: bool = False
    version: int = 1
    vault_path: Optional[str] = None
    # Extra: for heat score computation
    interaction_depth: float = 0.0   # L_interaction average

    def __post_init__(self):
        now = datetime.now(timezone.utc)
        if self.created_at is None:
            self.created_at = now
        if self.last_accessed is None:
            self.last_accessed = now

    def to_db_row(self) -> dict:
        d = asdict(self)
        for key in ('last_accessed', 'created_at', 'retention_at'):
            if d[key] is not None and isinstance(d[key], datetime):
                d[key] = d[key].isoformat()
        return d


# ── Core Functions ────────────────────────────────────────────────────────────

def compute_heat(
    n_visit: int,
    last_accessed: datetime,
    interaction_depth: float = 0.0,
    now: Optional[datetime] = None,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> float:
    """
    Compute heat score: H(m) = α·N_visit_norm + β·R_recency + γ·L_interaction

    Args:
        n_visit: total access count
        last_accessed: datetime of last access
        interaction_depth: average interaction depth (0–1 normalized)
        now: current time (defaults to utcnow)
        alpha, beta, gamma: weight coefficients (must sum to 1.0)

    Returns:
        Heat score in [0, 1]
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # Normalize visit count via sigmoid-like mapping
    # 0 visits → 0, 10 visits → ~0.91
    n_norm = 1.0 - (1.0 / (1.0 + n_visit * 0.1))

    # Recency: R_recency = 1 / (1 + days_since_last_access)
    if last_accessed.tzinfo is None:
        last_accessed = last_accessed.replace(tzinfo=timezone.utc)
    days_since = max(0.0, (now - last_accessed).total_seconds() / 86400.0)
    r_recency = 1.0 / (1.0 + days_since)

    # Interaction depth normalized to [0, 1]
    l_interaction = min(1.0, max(0.0, interaction_depth))

    heat = alpha * n_norm + beta * r_recency + gamma * l_interaction
    return round(float(heat), 6)


def compute_retention(
    stability_days: float,
    last_accessed: Optional[datetime] = None,
    now: Optional[datetime] = None,
) -> float:
    """
    FSRS power-law forgetting curve: R(t) = (1 + t/S)^(-0.5)

    Args:
        stability_days: S — stability in days (grows with each review)
        last_accessed: when memory was last accessed
        now: current time

    Returns:
        Retention R in (0, 1] — closer to 1 = well-retained
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if last_accessed is None:
        return 1.0

    if last_accessed.tzinfo is None:
        last_accessed = last_accessed.replace(tzinfo=timezone.utc)

    t = max(0.0, (now - last_accessed).total_seconds() / 86400.0)
    r = (1.0 + t / max(stability_days, 0.001)) ** (-0.5)
    return round(float(r), 6)


def update_stability(stability_days: float, n_visit: int) -> float:
    """
    Grow stability with each successful review (simplified FSRS).
    S grows roughly as: S_new = S * (1 + 0.1 * n_visit)
    Capped at STABILITY_MAX.
    """
    new_s = stability_days * (1.0 + 0.1 * max(1, n_visit))
    return min(new_s, STABILITY_MAX)


# ── Tier Manager ──────────────────────────────────────────────────────────────

class TierManager:
    """
    Manages heat scores, retention, and tier transitions for all memory records.
    Reads/writes to SQLite tier_registry table.
    """

    def __init__(
        self,
        conn,
        alpha: float = ALPHA,
        beta: float = BETA,
        gamma: float = GAMMA,
    ):
        self.conn = conn
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    # ── CRUD ──────────────────────────────────────────────────────────

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        row = self.conn.execute(
            "SELECT * FROM tier_registry WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def upsert(self, record: MemoryRecord) -> None:
        row = record.to_db_row()
        self.conn.execute("""
            INSERT OR REPLACE INTO tier_registry
            (memory_id, content_type, tier, n_visit, last_accessed,
             created_at, heat_score, stability_days, retention_r,
             retention_at, no_forget, version, vault_path, updated_at)
            VALUES
            (:memory_id, :content_type, :tier, :n_visit, :last_accessed,
             :created_at, :heat_score, :stability_days, :retention_r,
             :retention_at, :no_forget, :version, :vault_path,
             CURRENT_TIMESTAMP)
        """, row)

    def register(
        self,
        memory_id: Optional[str] = None,
        content_type: str = 'personal_fact',
        initial_tier: str = TIER_COLD,
        no_forget: bool = False,
        vault_path: Optional[str] = None,
    ) -> MemoryRecord:
        """Create a new memory record in the registry."""
        if memory_id is None:
            memory_id = str(uuid.uuid4())

        record = MemoryRecord(
            memory_id=memory_id,
            content_type=content_type,
            tier=initial_tier,
            no_forget=no_forget,
            vault_path=vault_path,
        )
        self.upsert(record)
        self.conn.commit()
        return record

    # ── Access recording ──────────────────────────────────────────────

    def record_access(
        self,
        memory_id: str,
        interaction_depth: float = 0.5,
    ) -> Optional[MemoryRecord]:
        """
        Record an access event: increment visit count, update heat + retention,
        and potentially promote the memory tier.
        """
        record = self.get(memory_id)
        if record is None:
            logger.warning(f"record_access: memory_id {memory_id} not found")
            return None

        now = datetime.now(timezone.utc)
        record.n_visit += 1
        record.last_accessed = now
        record.interaction_depth = (
            (record.interaction_depth * (record.n_visit - 1) + interaction_depth)
            / record.n_visit
        )

        # Update heat score
        record.heat_score = compute_heat(
            record.n_visit, record.last_accessed,
            record.interaction_depth, now=now,
            alpha=self.alpha, beta=self.beta, gamma=self.gamma,
        )

        # Update retention + stability
        record.stability_days = update_stability(record.stability_days, record.n_visit)
        record.retention_r = compute_retention(record.stability_days, now, now)
        record.retention_at = now
        record.version += 1

        # Check promotion
        old_tier = record.tier
        record.tier = self._compute_tier(record, now)
        if record.tier != old_tier:
            logger.info(f"Tier transition: {memory_id} {old_tier} → {record.tier}")

        self.upsert(record)
        self.conn.commit()
        return record

    # ── Batch tier refresh ────────────────────────────────────────────

    def refresh_all_tiers(self, batch_size: int = 500) -> dict[str, int]:
        """
        Recompute heat scores and retention for all records.
        Promotes/demotes as needed.
        Returns counts of transitions.
        """
        now = datetime.now(timezone.utc)
        transitions: dict[str, int] = {}

        offset = 0
        while True:
            rows = self.conn.execute(
                "SELECT * FROM tier_registry LIMIT ? OFFSET ?",
                (batch_size, offset),
            ).fetchall()
            if not rows:
                break

            for row in rows:
                record = self._row_to_record(row)
                old_tier = record.tier

                # Recompute
                record.heat_score = compute_heat(
                    record.n_visit, record.last_accessed,
                    record.interaction_depth, now=now,
                    alpha=self.alpha, beta=self.beta, gamma=self.gamma,
                )
                record.retention_r = compute_retention(
                    record.stability_days, record.last_accessed, now
                )
                record.retention_at = now

                record.tier = self._compute_tier(record, now)

                if record.tier != old_tier:
                    key = f"{old_tier}→{record.tier}"
                    transitions[key] = transitions.get(key, 0) + 1
                    record.version += 1
                    self.upsert(record)

            offset += batch_size

        self.conn.commit()
        return transitions

    def get_tier(self, tier: str, limit: int = 100) -> list[MemoryRecord]:
        """Fetch all records in a given tier, sorted by heat score desc."""
        rows = self.conn.execute(
            "SELECT * FROM tier_registry WHERE tier = ? ORDER BY heat_score DESC LIMIT ?",
            (tier, limit),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_hot_context(self) -> list[MemoryRecord]:
        return self.get_tier(TIER_HOT, limit=50)

    def get_warm_candidates(self, limit: int = 200) -> list[MemoryRecord]:
        return self.get_tier(TIER_WARM, limit=limit)

    # ── Tier logic ────────────────────────────────────────────────────

    def _compute_tier(self, record: MemoryRecord, now: datetime) -> str:
        """
        Determine target tier for a record based on heat score, R(t), and rules.
        no_forget=True items (ASME normative) are never deleted but CAN be demoted warm→cold.
        """
        current = record.tier

        # ── Promotion rules ──────────────────────────────────────────
        if current == TIER_COLD:
            days_window = 7
            if record.n_visit > 0 and record.last_accessed:
                la = record.last_accessed
                if la.tzinfo is None:
                    la = la.replace(tzinfo=timezone.utc)
                recent_days = (now - la).total_seconds() / 86400.0
                if record.heat_score >= COLD_TO_WARM_HEAT or (
                    record.n_visit >= 2 and recent_days <= days_window
                ):
                    return TIER_WARM

        elif current == TIER_WARM:
            if record.last_accessed:
                la = record.last_accessed
                if la.tzinfo is None:
                    la = la.replace(tzinfo=timezone.utc)
                recent_days = (now - la).total_seconds() / 86400.0
                if record.heat_score >= WARM_TO_HOT_HEAT or (
                    record.n_visit >= 5 and recent_days <= 3
                ):
                    return TIER_HOT

        # ── Demotion rules ───────────────────────────────────────────
        if current == TIER_HOT:
            if record.last_accessed:
                la = record.last_accessed
                if la.tzinfo is None:
                    la = la.replace(tzinfo=timezone.utc)
                days_since = (now - la).total_seconds() / 86400.0
                if record.retention_r < HOT_TO_WARM_R or days_since >= 7:
                    return TIER_WARM

        elif current == TIER_WARM:
            if record.last_accessed:
                la = record.last_accessed
                if la.tzinfo is None:
                    la = la.replace(tzinfo=timezone.utc)
                days_since = (now - la).total_seconds() / 86400.0
                if record.retention_r < WARM_TO_COLD_R or days_since >= 30:
                    return TIER_COLD

        return current  # no change

    # ── Row conversion ────────────────────────────────────────────────

    @staticmethod
    def _row_to_record(row) -> MemoryRecord:
        def parse_dt(s):
            if s is None:
                return None
            if isinstance(s, datetime):
                return s
            try:
                dt = datetime.fromisoformat(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except (ValueError, TypeError):
                return None

        return MemoryRecord(
            memory_id=row['memory_id'],
            content_type=row['content_type'],
            tier=row['tier'],
            n_visit=row['n_visit'] or 0,
            last_accessed=parse_dt(row['last_accessed']),
            created_at=parse_dt(row['created_at']),
            heat_score=row['heat_score'] or 0.0,
            stability_days=row['stability_days'] or STABILITY_INIT,
            retention_r=row['retention_r'] or 1.0,
            retention_at=parse_dt(row['retention_at']),
            no_forget=bool(row['no_forget']),
            version=row['version'] or 1,
            vault_path=row['vault_path'],
        )

    # ── Stats ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        rows = self.conn.execute("""
            SELECT tier,
                   COUNT(*) as count,
                   AVG(heat_score) as avg_heat,
                   AVG(retention_r) as avg_retention
            FROM tier_registry
            GROUP BY tier
        """).fetchall()
        return {
            row['tier']: {
                'count': row['count'],
                'avg_heat': round(row['avg_heat'] or 0, 4),
                'avg_retention': round(row['avg_retention'] or 0, 4),
            }
            for row in rows
        }


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys, os, tempfile
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent))
    from src.schema import init_schema

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    try:
        conn = init_schema(db_path)
        tm = TierManager(conn)

        # Register a memory
        rec = tm.register('test-001', content_type='personal_fact', initial_tier=TIER_COLD)
        print(f"Registered: {rec.memory_id} tier={rec.tier}")

        # Simulate accesses to trigger promotion
        for i in range(3):
            rec = tm.record_access('test-001', interaction_depth=0.8)
            print(f"  Access {i+1}: heat={rec.heat_score:.4f} R={rec.retention_r:.4f} tier={rec.tier}")

        print(f"\nStats: {tm.stats()}")

        # Test heat and retention functions
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=30)
        print(f"\nRetention after 30 days (S=1): {compute_retention(1.0, old, now):.4f}")
        print(f"Retention after 30 days (S=30): {compute_retention(30.0, old, now):.4f}")
        print(f"Heat (10 visits, 1 day ago): {compute_heat(10, now - timedelta(days=1)):.4f}")

    finally:
        conn.close()
        os.unlink(db_path)
