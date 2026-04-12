"""Tests for src/memory/llm_client.py — dispatch logic."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestLLMClient(unittest.TestCase):

    @patch("src.memory.llm_client.LLM_MODE", "cloud")
    @patch("src.memory.llm_client.ABACUS_API_KEY", "test-key")
    @patch("src.memory.llm_client.httpx.post")
    def test_cloud_dispatch(self, mock_post):
        mock_post.return_value = MagicMock(
            json=lambda: {"choices": [{"message": {"content": "cloud-response"}}]},
            raise_for_status=lambda: None,
        )
        from src.memory import llm_client
        result = llm_client.complete("hello")
        self.assertEqual(result, "cloud-response")
        called_url = mock_post.call_args[0][0]
        self.assertIn("abacus", called_url)

    @patch("src.memory.llm_client.LLM_MODE", "local")
    @patch("src.memory.llm_client.httpx.post")
    def test_local_dispatch(self, mock_post):
        mock_post.return_value = MagicMock(
            json=lambda: {"message": {"content": "local-response"}},
            raise_for_status=lambda: None,
        )
        from src.memory import llm_client
        result = llm_client.complete("hello")
        self.assertEqual(result, "local-response")
        called_url = mock_post.call_args[0][0]
        self.assertIn("11434", called_url)

    @patch("src.memory.llm_client.LLM_MODE", "cloud")
    @patch("src.memory.llm_client.ABACUS_API_KEY", "")
    def test_cloud_missing_key_raises(self):
        from src.memory import llm_client
        with self.assertRaises(RuntimeError):
            llm_client.complete("hello")

    @patch("src.memory.llm_client.LLM_MODE", "nonsense")
    def test_unknown_mode_raises(self):
        from src.memory import llm_client
        with self.assertRaises(ValueError):
            llm_client.complete("hello")


if __name__ == "__main__":
    unittest.main()
