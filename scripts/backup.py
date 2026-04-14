#!/usr/bin/env python3
"""
Leo Trident — Backup Job

Snapshots SQLite (via .backup API, which handles WAL safely) and copies
the LanceDB directory. Writes to $LEO_TRIDENT_HOME/backups/{timestamp}/
and prunes older-than-N-day backups.

Usage:
    python scripts/backup.py                     # default retention
    python scripts/backup.py --retention-days 30
    python scripts/backup.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import socket
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BASE_PATH as _DEFAULT_BASE_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [backup] %(message)s")
logger = logging.getLogger(__name__)


def _dir_size(path: Path) -> int:
    """Total bytes of all files under path."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main(base_path: Path = None, retention_days: int = 30, dry_run: bool = False) -> Path:
    """
    Run a full backup. Returns the backup directory path.
    Raises on critical failure.
    """
    base_path = base_path or _DEFAULT_BASE_PATH
    data_path = base_path / "data"
    db_path = data_path / "leo_trident.db"
    lance_path = data_path / "lancedb"
    vault_system = base_path / "vault" / "_system"

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = base_path / "backups" / ts
    if dry_run:
        logger.info("[DRY-RUN] Would create backup at %s", backup_dir)
    else:
        backup_dir.mkdir(parents=True, exist_ok=True)

    # 1. SQLite backup (WAL-safe)
    sqlite_dest = backup_dir / "leo_trident.db"
    if db_path.exists():
        if dry_run:
            logger.info("[DRY-RUN] Would backup SQLite %s", db_path)
        else:
            src_conn = sqlite3.connect(str(db_path))
            dst_conn = sqlite3.connect(str(sqlite_dest))
            src_conn.backup(dst_conn)
            dst_conn.close()
            src_conn.close()
            logger.info("SQLite backed up: %s", sqlite_dest)
    else:
        logger.warning("SQLite DB not found at %s — skipping", db_path)

    # 2. LanceDB copy
    lance_dest = backup_dir / "lancedb"
    if lance_path.exists():
        if dry_run:
            logger.info("[DRY-RUN] Would copy LanceDB %s", lance_path)
        else:
            shutil.copytree(str(lance_path), str(lance_dest))
            logger.info("LanceDB backed up: %s", lance_dest)
    else:
        logger.warning("LanceDB directory not found at %s — skipping", lance_path)

    # 3. vault/_system/ copy
    vault_dest = backup_dir / "vault_system"
    if vault_system.exists():
        if dry_run:
            logger.info("[DRY-RUN] Would copy vault/_system/")
        else:
            shutil.copytree(str(vault_system), str(vault_dest))
            logger.info("Vault system backed up: %s", vault_dest)
    else:
        logger.warning("vault/_system/ not found — skipping")

    # 4. Write manifest
    if not dry_run:
        # Count ASME chunks and graph edges from the backup DB
        asme_count = 0
        edge_count = 0
        if sqlite_dest.exists():
            conn = sqlite3.connect(str(sqlite_dest))
            try:
                asme_count = conn.execute("SELECT COUNT(*) FROM asme_chunks").fetchone()[0]
            except Exception:
                pass
            try:
                edge_count = conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
            except Exception:
                pass
            conn.close()

        manifest = {
            "timestamp": ts,
            "sqlite_bytes": sqlite_dest.stat().st_size if sqlite_dest.exists() else 0,
            "lancedb_bytes": _dir_size(lance_dest),
            "vault_system_bytes": _dir_size(vault_dest),
            "asme_chunk_count": asme_count,
            "graph_edge_count": edge_count,
            "schema_version": "phase8",
            "host": socket.gethostname(),
        }
        manifest_path = backup_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Manifest written: %s", manifest_path)

    # 5. Prune old backups
    prune_backups(base_path, retention_days, dry_run=dry_run)

    logger.info("Backup complete: %s", backup_dir)
    return backup_dir


def prune_backups(base_path: Path, retention_days: int, dry_run: bool = False):
    """Remove backup directories older than retention_days."""
    backups_root = base_path / "backups"
    if not backups_root.exists():
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

    for entry in sorted(backups_root.iterdir()):
        if not entry.is_dir():
            continue
        # Parse timestamp from directory name (YYYYMMDDTHHMMSSz)
        try:
            dir_ts = datetime.strptime(entry.name, "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue

        if dir_ts < cutoff:
            if dry_run:
                logger.info("[DRY-RUN] Would prune old backup: %s", entry.name)
            else:
                shutil.rmtree(entry)
                logger.info("Pruned old backup: %s", entry.name)


def prune_metrics(base_path: Path, retention_days: int = 90, dry_run: bool = False):
    """Remove metrics JSONL files older than retention_days."""
    metrics_dir = base_path / "data" / "metrics"
    if not metrics_dir.exists():
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

    for f in sorted(metrics_dir.glob("*.jsonl")):
        try:
            file_date = datetime.strptime(f.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if file_date < cutoff:
            if dry_run:
                logger.info("[DRY-RUN] Would prune old metrics: %s", f.name)
            else:
                f.unlink()
                logger.info("Pruned old metrics file: %s", f.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leo Trident backup")
    parser.add_argument("--retention-days", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        main(retention_days=args.retention_days, dry_run=args.dry_run)
    except Exception as e:
        logger.error("Backup failed: %s", e, exc_info=True)
        sys.exit(1)
