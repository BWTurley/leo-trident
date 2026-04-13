"""
Tests for Phase 8 — Metrics Logging
"""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMetrics(unittest.TestCase):

    def test_log_metric_writes_valid_jsonl(self):
        tmp = Path(tempfile.mkdtemp())
        from src.service.metrics import log_metric

        log_metric("test.latency", 42.5, tags={"foo": "bar"}, base_path=tmp)

        metrics_dir = tmp / "data" / "metrics"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = metrics_dir / f"{today}.jsonl"

        self.assertTrue(filepath.exists())
        with open(filepath) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)

        entry = json.loads(lines[0])
        self.assertEqual(entry["name"], "test.latency")
        self.assertEqual(entry["value"], 42.5)
        self.assertEqual(entry["tags"]["foo"], "bar")
        self.assertIn("ts", entry)

    def test_multiple_calls_append(self):
        tmp = Path(tempfile.mkdtemp())
        from src.service.metrics import log_metric

        log_metric("m1", 1, base_path=tmp)
        log_metric("m2", 2, base_path=tmp)
        log_metric("m3", 3, base_path=tmp)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = tmp / "data" / "metrics" / f"{today}.jsonl"
        with open(filepath) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 3)

    def test_no_raise_on_readonly_dir(self):
        # Use a non-existent deep path that can't be created due to file collision
        tmp = Path(tempfile.mkdtemp())
        blocker = tmp / "data" / "metrics"
        blocker.mkdir(parents=True)
        # Create a file where we'd want a directory — this blocks mkdir
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = blocker / f"{today}.jsonl"

        # Make the file read-only (best-effort on Windows)
        filepath.write_text("")
        try:
            os.chmod(str(filepath), 0o444)
        except Exception:
            pass

        from src.service.metrics import log_metric
        # Should not raise
        try:
            # On most systems writing to a read-only file will fail,
            # but log_metric swallows exceptions
            log_metric("should.not.raise", 99, base_path=tmp)
        except Exception:
            self.fail("log_metric raised on write failure")
        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(str(filepath), 0o644)
            except Exception:
                pass

    def test_metric_pruning(self):
        tmp = Path(tempfile.mkdtemp())
        metrics_dir = tmp / "data" / "metrics"
        metrics_dir.mkdir(parents=True)

        # Create old and recent metric files
        old_date = (datetime.now(timezone.utc) - timedelta(days=100)).strftime("%Y-%m-%d")
        recent_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        (metrics_dir / f"{old_date}.jsonl").write_text('{"name":"old"}\n')
        (metrics_dir / f"{recent_date}.jsonl").write_text('{"name":"recent"}\n')

        from scripts.backup import prune_metrics
        prune_metrics(tmp, retention_days=90)

        self.assertFalse((metrics_dir / f"{old_date}.jsonl").exists())
        self.assertTrue((metrics_dir / f"{recent_date}.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
