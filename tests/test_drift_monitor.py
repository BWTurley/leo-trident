"""
Tests for Phase 5 — DriftMonitor
"""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_anchors(tmp_dir: str, tamper: bool = False) -> Path:
    system_dir = Path(tmp_dir) / "vault" / "_system"
    system_dir.mkdir(parents=True, exist_ok=True)

    rule_never = "waive UG-99 hydrostatic test"
    rule_always = "cite ASME paragraph IDs in all code references"
    fact = "Primary focus: ASME BPVC Section VIII Division 1"

    anchors = {
        "_meta": {"version": 2, "format": "Leo Trident anchors v2"},
        "asme_safety_pins": {
            "never": [
                {"rule": rule_never,
                 "hash": "bad_hash" if tamper else hashlib.sha256(rule_never.encode()).hexdigest()}
            ],
            "always": [
                {"rule": rule_always,
                 "hash": hashlib.sha256(rule_always.encode()).hexdigest()}
            ],
        },
        "core_facts": [
            {"fact": fact, "hash": hashlib.sha256(fact.encode()).hexdigest()}
        ],
    }
    with open(system_dir / "anchors.json", "w") as f:
        json.dump(anchors, f)
    return system_dir


class TestDriftMonitorAnchors(unittest.TestCase):

    def test_check_anchors_passes(self):
        tmp = tempfile.mkdtemp()
        _make_anchors(tmp, tamper=False)
        from src.memory.drift_monitor import DriftMonitor
        dm = DriftMonitor(base_path=tmp)
        result = dm.check_anchors()
        self.assertTrue(result["ok"])
        self.assertEqual(result["violations"], [])

    def test_check_anchors_detects_tamper(self):
        tmp = tempfile.mkdtemp()
        _make_anchors(tmp, tamper=True)
        from src.memory.drift_monitor import DriftMonitor
        dm = DriftMonitor(base_path=tmp)
        result = dm.check_anchors()
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["violations"]), 1)
        self.assertIn("waive UG-99", result["violations"][0]["rule"])

    def test_check_anchors_missing_file(self):
        tmp = tempfile.mkdtemp()
        # No anchors.json created
        from src.memory.drift_monitor import DriftMonitor
        dm = DriftMonitor(base_path=tmp)
        result = dm.check_anchors()
        self.assertFalse(result["ok"])
        self.assertTrue(len(result["violations"]) > 0)


class TestDriftMonitorPSI(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.mkdtemp()
        self.dm_cls = None
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.memory.drift_monitor import DriftMonitor
        self.dm = DriftMonitor(base_path=tmp)

    def test_psi_identical_distributions(self):
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((50, 4))
        psi = self.dm.compute_psi(emb, emb.copy())
        # Identical distributions should have PSI ≈ 0
        self.assertAlmostEqual(psi, 0.0, places=3)

    def test_psi_different_distributions(self):
        rng = np.random.default_rng(42)
        base = rng.standard_normal((100, 4))
        curr = rng.standard_normal((100, 4)) + 2.0  # shifted
        psi = self.dm.compute_psi(base, curr)
        # Different distributions should have higher PSI
        self.assertGreater(psi, 0.0)

    def test_psi_alert_threshold(self):
        rng = np.random.default_rng(42)
        base = rng.standard_normal((100, 4))
        curr = rng.standard_normal((100, 4)) + 5.0  # very shifted
        psi = self.dm.compute_psi(base, curr)
        # Should be well above threshold 0.1
        self.assertGreater(psi, 0.1)


class TestEmbeddingDriftReport(unittest.TestCase):

    def test_embedding_drift_report_no_lancedb(self):
        """Should return graceful error when LanceDB or tables not present."""
        tmp = tempfile.mkdtemp()
        # No LanceDB
        from src.memory.drift_monitor import DriftMonitor
        dm = DriftMonitor(base_path=tmp)
        result = dm.embedding_drift_report()
        # Should not raise — just return with error or None values
        self.assertIn("mean_cosine_sim", result)
        self.assertIn("drift_detected", result)
        self.assertFalse(result["drift_detected"])

    def test_embedding_drift_report_runs(self):
        """Integration smoke test: should run without raising."""
        tmp = tempfile.mkdtemp()
        from src.memory.drift_monitor import DriftMonitor
        dm = DriftMonitor(base_path=tmp)
        try:
            result = dm.embedding_drift_report()
            self.assertIsInstance(result, dict)
        except Exception as e:
            self.fail(f"embedding_drift_report raised unexpectedly: {e}")


if __name__ == "__main__":
    unittest.main()
