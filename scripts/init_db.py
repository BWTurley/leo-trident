#!/usr/bin/env python3
"""
Leo Trident — Database Initializer
Initializes SQLite (full schema) + LanceDB tables at 64/256/768 dimensions.
"""

import sys
import json
import argparse
from pathlib import Path

# Add parent to path for src imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.schema import init_schema


def init_lancedb(lance_path: Path) -> None:
    """Initialize LanceDB with hot/warm/cold tables at 64/256/768 dimensions."""
    try:
        import lancedb
        import pyarrow as pa
    except ImportError:
        print("⚠  lancedb or pyarrow not installed — skipping LanceDB init")
        print("   Run: pip install lancedb pyarrow")
        return

    lance_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(lance_path))

    # Schema per tier — Matryoshka dimensions
    tier_configs = [
        ("hot",  64),
        ("warm", 256),
        ("cold", 768),
    ]

    base_fields = [
        pa.field("chunk_id",      pa.string()),
        pa.field("paragraph_id",  pa.string()),
        pa.field("section",       pa.string()),
        pa.field("content_type",  pa.string()),
        pa.field("content",       pa.string()),
        pa.field("no_forget",     pa.bool_()),
        pa.field("tier",          pa.string()),
        pa.field("edition_year",  pa.int32()),
        pa.field("created_at",    pa.string()),
    ]

    for tier_name, dim in tier_configs:
        table_name = f"chunks_{tier_name}"
        if table_name in db.table_names():
            print(f"  ✓ LanceDB table '{table_name}' already exists — skipping")
            continue

        schema = pa.schema(base_fields + [
            pa.field("vector", pa.list_(pa.float32(), dim))
        ])

        # Create empty table with schema
        db.create_table(table_name, schema=schema, mode="create")
        print(f"  ✓ LanceDB table '{table_name}' created ({dim}d vectors)")

    # Also create a personal memory table (warm tier, 256d)
    personal_table = "personal_warm"
    if personal_table not in db.table_names():
        personal_schema = pa.schema([
            pa.field("fact_id",      pa.string()),
            pa.field("category",     pa.string()),
            pa.field("key",          pa.string()),
            pa.field("value",        pa.string()),
            pa.field("confidence",   pa.float32()),
            pa.field("created_at",   pa.string()),
            pa.field("vector",       pa.list_(pa.float32(), 256)),
        ])
        db.create_table(personal_table, schema=personal_schema, mode="create")
        print(f"  ✓ LanceDB table '{personal_table}' created (256d vectors)")

    print(f"\n  LanceDB initialized at: {lance_path}")
    print(f"  Tables: {db.table_names()}")


def seed_vault_system(vault_path: Path) -> None:
    """Create vault _system directory and placeholder files if missing."""
    system_dir = vault_path / "_system"
    system_dir.mkdir(parents=True, exist_ok=True)

    hot_path = system_dir / "hot.json"
    if not hot_path.exists():
        # Minimal seed — will be overwritten by real hot.json from vault/_system/hot.json
        hot_seed = {
            "version": 1,
            "token_budget": 200,
            "persona": "",
            "safety_pins": [],
            "active_project": {},
            "session_hint": {}
        }
        hot_path.write_text(json.dumps(hot_seed, indent=2))
        print(f"  ✓ Seeded {hot_path}")

    anchors_path = system_dir / "anchors.json"
    if not anchors_path.exists():
        print(f"  ⚠  {anchors_path} not found — run vault init to create it")

    consolidation_log = system_dir / "consolidation_log.json"
    if not consolidation_log.exists():
        consolidation_log.write_text(json.dumps({"runs": []}, indent=2))
        print(f"  ✓ Seeded {consolidation_log}")


def main():
    parser = argparse.ArgumentParser(description="Initialize Leo Trident databases")
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data"),
        help="Directory for SQLite and LanceDB storage",
    )
    parser.add_argument(
        "--vault-dir",
        default=str(ROOT / "vault"),
        help="Path to Obsidian vault root",
    )
    parser.add_argument(
        "--skip-lancedb",
        action="store_true",
        help="Skip LanceDB initialization",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── SQLite ────────────────────────────────────────────────────────
    db_path = data_dir / "leo_trident.db"
    print(f"\n── SQLite ─────────────────────────────────────────────")
    print(f"   Path: {db_path}")
    conn = init_schema(db_path)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','shadow') ORDER BY name"
    ).fetchall()
    for t in tables:
        print(f"  ✓ {t['name']}")
    conn.close()
    print(f"  WAL mode: active")

    # ── LanceDB ───────────────────────────────────────────────────────
    if not args.skip_lancedb:
        lance_path = data_dir / "lancedb"
        print(f"\n── LanceDB ────────────────────────────────────────────")
        print(f"   Path: {lance_path}")
        init_lancedb(lance_path)

    # ── Vault ────────────────────────────────────────────────────────
    vault_path = Path(args.vault_dir)
    print(f"\n── Vault ──────────────────────────────────────────────")
    print(f"   Path: {vault_path}")
    seed_vault_system(vault_path)

    print("\n✅  Leo Trident initialization complete.\n")


if __name__ == "__main__":
    main()
