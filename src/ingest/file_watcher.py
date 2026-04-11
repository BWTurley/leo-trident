"""
Leo Trident — Vault File Watcher
Watches vault/ for .md file changes and re-indexes content.
Debounce 500ms. Logs to vault/_system/consolidation_log.json.
"""
from __future__ import annotations
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logger.warning("watchdog not installed — file watcher disabled")


class VaultEventHandler(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    def __init__(
        self,
        on_md_change: Callable[[Path], None],
        debounce_ms: int = 500,
    ):
        if WATCHDOG_AVAILABLE:
            super().__init__()
        self.on_md_change = on_md_change
        self.debounce_ms = debounce_ms
        self._pending: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def _debounced_trigger(self, path: str):
        def _run():
            with self._lock:
                self._pending.pop(path, None)
            try:
                self.on_md_change(Path(path))
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")

        with self._lock:
            existing = self._pending.get(path)
            if existing:
                existing.cancel()
            timer = threading.Timer(self.debounce_ms / 1000.0, _run)
            self._pending[path] = timer
            timer.start()

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self._debounced_trigger(event.src_path)

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self._debounced_trigger(event.src_path)


class VaultWatcher:
    def __init__(
        self,
        vault_path: str,
        on_md_change: Callable[[Path], None],
        log_path: Optional[str] = None,
    ):
        self.vault_path = Path(vault_path)
        self.on_md_change = on_md_change
        self.log_path = Path(log_path) if log_path else (
            self.vault_path / "_system" / "consolidation_log.json"
        )
        self._observer = None

    def _handle_change(self, path: Path):
        logger.info(f"Vault change detected: {path}")
        self._log_change(path)
        self.on_md_change(path)

    def _log_change(self, path: Path):
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.log_path.exists():
                log = json.loads(self.log_path.read_text())
            else:
                log = {"runs": []}

            log["runs"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "file": str(path),
                "event": "file_change",
            })
            # Keep last 500 entries
            log["runs"] = log["runs"][-500:]
            self.log_path.write_text(json.dumps(log, indent=2))
        except Exception as e:
            logger.error(f"Failed to write consolidation log: {e}")

    def start(self) -> bool:
        if not WATCHDOG_AVAILABLE:
            logger.error("watchdog not available — cannot start watcher")
            return False
        if not self.vault_path.exists():
            logger.error(f"Vault path does not exist: {self.vault_path}")
            return False

        handler = VaultEventHandler(self._handle_change)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.vault_path), recursive=True)
        self._observer.start()
        logger.info(f"VaultWatcher started: {self.vault_path}")
        return True

    def stop(self):
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("VaultWatcher stopped")

    def is_running(self) -> bool:
        return self._observer is not None and self._observer.is_alive()
