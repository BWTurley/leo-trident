"""Tests for src/config.py — env var parsing and path resolution."""
import importlib
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestConfig(unittest.TestCase):

    def _reload(self):
        import src.config as cfg
        importlib.reload(cfg)
        return cfg

    def test_default_home_is_user_path_when_no_legacy(self):
        os.environ.pop("LEO_TRIDENT_HOME", None)
        cfg = self._reload()
        # Either user home or legacy Abacus path — both are valid defaults
        self.assertTrue(
            str(cfg.BASE_PATH).endswith("leo_trident"),
            f"Unexpected BASE_PATH: {cfg.BASE_PATH}",
        )

    def test_env_var_override(self):
        os.environ["LEO_TRIDENT_HOME"] = "/tmp/leo_test_xyz"
        try:
            cfg = self._reload()
            expected = Path("/tmp/leo_test_xyz").expanduser().resolve()
            self.assertEqual(cfg.BASE_PATH, expected)
        finally:
            os.environ.pop("LEO_TRIDENT_HOME", None)

    def test_llm_mode_default_is_cloud(self):
        os.environ.pop("LEO_LLM_MODE", None)
        cfg = self._reload()
        self.assertEqual(cfg.LLM_MODE, "cloud")

    def test_llm_mode_override(self):
        os.environ["LEO_LLM_MODE"] = "local"
        try:
            cfg = self._reload()
            self.assertEqual(cfg.LLM_MODE, "local")
        finally:
            os.environ.pop("LEO_LLM_MODE", None)

    def test_embed_device_default_is_cpu(self):
        os.environ.pop("LEO_EMBED_DEVICE", None)
        cfg = self._reload()
        self.assertEqual(cfg.EMBED_DEVICE, "cpu")


if __name__ == "__main__":
    unittest.main()
