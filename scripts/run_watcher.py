#!/usr/bin/env python3
"""Long-running vault watcher daemon. Reads .md changes and re-indexes."""
import logging
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api import LeoTrident
from src.config import VAULT_PATH
from src.ingest.file_watcher import VaultWatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("watcher")

lt = LeoTrident()


def on_change(path: Path):
    if "_system" in path.parts:
        return
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return
        lt.ingest_text(
            text=text,
            paragraph_id=path.stem,
            section="Personal",
            part=path.parent.name,
            edition_year=2025,
        )
        log.info(f"Re-indexed {path}")
    except Exception as e:
        log.error(f"Failed to re-index {path}: {e}")


def main():
    watcher = VaultWatcher(vault_path=str(VAULT_PATH), on_md_change=on_change)
    if not watcher.start():
        log.error("Failed to start watcher")
        sys.exit(1)

    def _shutdown(*_):
        watcher.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    log.info(f"Watching {VAULT_PATH}")
    while watcher.is_running():
        time.sleep(5)


if __name__ == "__main__":
    main()
