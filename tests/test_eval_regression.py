"""
Regression test: runs the eval and asserts metrics don't drop below baseline.

Baseline is committed in tests/fixtures/eval_baseline.json. Update it
intentionally when you've verified the new numbers are better.
"""
import json
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _ollama_reachable() -> bool:
    """Check if Ollama is reachable for local LLM mode."""
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


class TestEvalRegression(unittest.TestCase):
    """Run eval against synthetic corpus and assert metrics above baseline."""

    BASELINE_PATH = Path(__file__).parent / "fixtures" / "eval_baseline.json"
    QUESTIONS_PATH = Path(__file__).parent / "fixtures" / "eval_questions.json"
    CORPUS_DIR = Path(__file__).parent / "fixtures" / "synthetic_corpus"

    @classmethod
    def setUpClass(cls):
        # Import eval runner functions
        from scripts.run_eval import setup_eval_instance, run_eval

        # Load questions (skip bi_temporal stubs)
        with open(cls.QUESTIONS_PATH) as f:
            cls.questions = [q for q in json.load(f) if not q.get("skip")]

        # Load baseline
        if not cls.BASELINE_PATH.exists():
            raise unittest.SkipTest(
                "No baseline found — run scripts/run_eval.py first to generate one"
            )
        with open(cls.BASELINE_PATH) as f:
            cls.baseline = json.load(f)

        # Set up eval instance and run (no relevance judge — doesn't need LLM)
        cls.tmp_dir = tempfile.mkdtemp(prefix="leo_eval_regtest_")
        try:
            cls.lt = setup_eval_instance(cls.tmp_dir, cls.CORPUS_DIR)
            cls.results = run_eval(
                cls.lt,
                cls.questions,
                use_rerank=False,
                use_judge=False,
            )
        except Exception:
            shutil.rmtree(cls.tmp_dir, ignore_errors=True)
            raise

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmp_dir"):
            shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def _get_baseline_metric(self, category: str, metric: str) -> float:
        by_cat = self.baseline.get("aggregate", {}).get("by_category", {})
        return by_cat.get(category, {}).get(metric, 0.0)

    def _get_current_metric(self, category: str, metric: str) -> float:
        by_cat = self.results.get("aggregate", {}).get("by_category", {})
        return by_cat.get(category, {}).get(metric, 0.0)

    def _assert_no_regression(self, category: str, metric: str, slack: float = 0.05):
        """Assert current metric >= baseline - slack."""
        baseline_val = self._get_baseline_metric(category, metric)
        current_val = self._get_current_metric(category, metric)
        self.assertGreaterEqual(
            current_val,
            baseline_val - slack,
            f"{category} {metric}: current {current_val:.3f} < baseline "
            f"{baseline_val:.3f} - {slack} slack",
        )

    def test_single_hop_recall_at_5(self):
        self._assert_no_regression("single_hop_factual", "recall_at_5")

    def test_single_hop_recall_at_10(self):
        self._assert_no_regression("single_hop_factual", "recall_at_10")

    def test_multi_hop_recall_at_5(self):
        self._assert_no_regression("multi_hop", "recall_at_5")

    def test_multi_hop_recall_at_10(self):
        self._assert_no_regression("multi_hop", "recall_at_10")

    def test_paragraph_id_lookup_recall_at_5(self):
        self._assert_no_regression("paragraph_id_lookup", "recall_at_5")

    def test_overall_mrr(self):
        baseline_mrr = self.baseline.get("aggregate", {}).get("mrr", 0.0)
        current_mrr = self.results.get("aggregate", {}).get("mrr", 0.0)
        self.assertGreaterEqual(
            current_mrr,
            baseline_mrr - 0.05,
            f"Overall MRR: current {current_mrr:.3f} < baseline {baseline_mrr:.3f} - 0.05",
        )


if __name__ == "__main__":
    unittest.main()
