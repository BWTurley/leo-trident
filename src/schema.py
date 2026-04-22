"""
Leo Trident SQLite Schema
Full schema with WAL mode, FTS5, and all Phase 0 tables.
"""

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA synchronous=NORMAL;

-- ============================================================
-- ASME Corpus Chunks
-- ============================================================
CREATE TABLE IF NOT EXISTS asme_chunks (
    chunk_id        TEXT PRIMARY KEY,          -- e.g. "VIII-1_UG-22_a_2025"
    paragraph_id    TEXT NOT NULL,             -- e.g. "UG-22(a)"
    section         TEXT,                      -- "VIII-1", "IX", "V", "I", "II"
    part            TEXT,                      -- "UG", "UW", "UCS", "QW", etc.
    edition_year    INTEGER,                   -- 2021, 2023, 2025
    valid_from      DATE,
    valid_to        DATE,                      -- NULL = currently active
    content         TEXT NOT NULL,
    content_hash    TEXT NOT NULL,             -- SHA-256
    no_forget       BOOLEAN DEFAULT TRUE,
    mandatory       BOOLEAN DEFAULT TRUE,
    content_type    TEXT DEFAULT 'normative',  -- normative / informative / appendix
    cross_refs      TEXT DEFAULT '[]',         -- JSON array of paragraph IDs
    raptor_level    INTEGER DEFAULT 0,         -- 0=leaf, 1=group, 2=part, 3=section
    embedding_dim   INTEGER DEFAULT 768,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_asme_paragraph ON asme_chunks(paragraph_id);
CREATE INDEX IF NOT EXISTS idx_asme_section   ON asme_chunks(section);
CREATE INDEX IF NOT EXISTS idx_asme_part      ON asme_chunks(part);
CREATE INDEX IF NOT EXISTS idx_asme_edition   ON asme_chunks(edition_year);
CREATE INDEX IF NOT EXISTS idx_asme_valid     ON asme_chunks(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_asme_type      ON asme_chunks(content_type);

-- FTS5 for BM25 keyword search over ASME content
-- tokenchars preserves ASME identifiers like UG-22, UW-11.a.2
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    paragraph_id,
    content,
    tokenize="unicode61"
);

-- ============================================================
-- Graph Edges — ASME Cross-Reference Adjacency
-- ============================================================
CREATE TABLE IF NOT EXISTS graph_edges (
    edge_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id       TEXT NOT NULL,   -- paragraph_id or chunk_id
    target_id       TEXT NOT NULL,   -- paragraph_id or chunk_id
    edge_type       TEXT DEFAULT 'cross_ref',  -- cross_ref / hierarchy / related
    reference_type  TEXT DEFAULT 'unclassified',  -- mandatory / conditional / informational / unclassified
    citation_text   TEXT,
    context         TEXT,
    weight          REAL DEFAULT 1.0,
    edition_year    INTEGER,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, target_id, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type   ON graph_edges(edge_type);

-- ============================================================
-- Tier Registry — Heat Scores, R(t), Tier Assignments
-- ============================================================
CREATE TABLE IF NOT EXISTS tier_registry (
    memory_id       TEXT PRIMARY KEY,
    content_type    TEXT NOT NULL,    -- personal_fact / project_note / episodic_log / asme_chunk
    tier            TEXT NOT NULL,    -- hot / warm / cold
    n_visit         INTEGER DEFAULT 0,
    last_accessed   DATETIME,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    -- Heat score components
    heat_score      REAL DEFAULT 0.0,
    -- FSRS forgetting curve
    stability_days  REAL DEFAULT 1.0,  -- S in R(t) = (1 + t/S)^-0.5
    retention_r     REAL DEFAULT 1.0,  -- R(t) last computed value
    retention_at    DATETIME,          -- when R(t) was last computed
    -- Flags
    no_forget       BOOLEAN DEFAULT FALSE,
    version         INTEGER DEFAULT 1,
    vault_path      TEXT,              -- relative path in Obsidian vault
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tier_tier       ON tier_registry(tier);
CREATE INDEX IF NOT EXISTS idx_tier_heat       ON tier_registry(heat_score DESC);
CREATE INDEX IF NOT EXISTS idx_tier_accessed   ON tier_registry(last_accessed);
CREATE INDEX IF NOT EXISTS idx_tier_type       ON tier_registry(content_type);

-- ============================================================
-- Conversation Logs
-- ============================================================
CREATE TABLE IF NOT EXISTS conversation_logs (
    log_id          TEXT PRIMARY KEY,   -- UUID
    session_id      TEXT NOT NULL,
    turn_index      INTEGER NOT NULL,
    role            TEXT NOT NULL,      -- user / assistant / system
    content         TEXT NOT NULL,
    token_count     INTEGER,
    topics          TEXT DEFAULT '[]',  -- JSON array of detected topics
    asme_refs       TEXT DEFAULT '[]',  -- JSON array of ASME paragraphs cited
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    consolidated    BOOLEAN DEFAULT FALSE,
    consolidation_at DATETIME
);

CREATE INDEX IF NOT EXISTS idx_logs_session    ON conversation_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_logs_created    ON conversation_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_logs_session_ts ON conversation_logs(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_logs_pending    ON conversation_logs(consolidated) WHERE consolidated=FALSE;

-- FTS5 over conversation logs
CREATE VIRTUAL TABLE IF NOT EXISTS logs_fts USING fts5(
    log_id UNINDEXED,
    content,
    tokenize="unicode61"
);

-- ============================================================
-- Personal Memory Facts (warm/cold personal knowledge)
-- ============================================================
CREATE TABLE IF NOT EXISTS personal_facts (
    fact_id         TEXT PRIMARY KEY,  -- UUID
    category        TEXT NOT NULL,     -- profile / project / contact / procedure
    key             TEXT NOT NULL,
    value           TEXT NOT NULL,
    confidence      REAL DEFAULT 1.0,
    source          TEXT,              -- conversation session that yielded this fact
    operation       TEXT DEFAULT 'ADD',  -- ADD / UPDATE / DELETE (Mem0-style)
    version         INTEGER DEFAULT 1,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_facts_category ON personal_facts(category);
CREATE INDEX IF NOT EXISTS idx_facts_key      ON personal_facts(key);
"""

FTS_SYNC_TRIGGERS = """
-- Keep FTS5 in sync with asme_chunks
CREATE TRIGGER IF NOT EXISTS asme_fts_insert AFTER INSERT ON asme_chunks BEGIN
    INSERT INTO chunks_fts(chunk_id, paragraph_id, content)
    VALUES (new.chunk_id, new.paragraph_id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS asme_fts_delete AFTER DELETE ON asme_chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = old.chunk_id;
END;

CREATE TRIGGER IF NOT EXISTS asme_fts_update AFTER UPDATE ON asme_chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = old.chunk_id;
    INSERT INTO chunks_fts(chunk_id, paragraph_id, content)
    VALUES (new.chunk_id, new.paragraph_id, new.content);
END;

-- Keep FTS5 in sync with conversation_logs
CREATE TRIGGER IF NOT EXISTS logs_fts_insert AFTER INSERT ON conversation_logs BEGIN
    INSERT INTO logs_fts(log_id, content) VALUES (new.log_id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS logs_fts_delete AFTER DELETE ON conversation_logs BEGIN
    DELETE FROM logs_fts WHERE log_id = old.log_id;
END;

CREATE TRIGGER IF NOT EXISTS logs_fts_update AFTER UPDATE ON conversation_logs BEGIN
    DELETE FROM logs_fts WHERE log_id = old.log_id;
    INSERT INTO logs_fts(log_id, content) VALUES (new.log_id, new.content);
END;
"""


def create_connection(db_path: str | Path, read_only: bool = False) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode and appropriate settings."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if read_only:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.execute("PRAGMA query_only=ON")
    else:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _migrate_graph_edges(conn):
    """Add Phase 6a columns to graph_edges if they don't exist yet."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(graph_edges)")}
    migrations = [
        ("reference_type", "ALTER TABLE graph_edges ADD COLUMN reference_type TEXT DEFAULT 'unclassified'"),
        ("citation_text",  "ALTER TABLE graph_edges ADD COLUMN citation_text TEXT"),
        ("context",        "ALTER TABLE graph_edges ADD COLUMN context TEXT"),
    ]
    for col, sql in migrations:
        if col not in cols:
            conn.execute(sql)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_reftype ON graph_edges(reference_type)")


def init_schema(db_path: str | Path) -> sqlite3.Connection:
    """Initialize the full schema. Idempotent — safe to call on existing DB."""
    conn = create_connection(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.executescript(FTS_SYNC_TRIGGERS)
    _migrate_graph_edges(conn)
    conn.commit()
    return conn


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "leo_trident.db"
    conn = init_schema(db)
    print(f"Schema initialized at {db}")
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    for t in tables:
        print(f"  ✓ {t['name']}")
    conn.close()
