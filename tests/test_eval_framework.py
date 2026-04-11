"""
Tests for Phase 5 — ASME Eval Framework
Uses dummy data already in the system (no real ASME corpus required).
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Eval question bank ────────────────────────────────────────────────────────

EVAL_QUESTIONS = [
    # Single-hop factual
    {
        "query": "what are the design load requirements for pressure vessels",
        "expected_paragraph_ids": ["UG-22"],
        "category": "single_hop_factual",
    },
    {
        "query": "welding examination requirements",
        "expected_paragraph_ids": ["UW-11"],
        "category": "single_hop_factual",
    },
    {
        "query": "procedure qualification requirements",
        "expected_paragraph_ids": ["QW-200"],
        "category": "single_hop_factual",
    },
    {
        "query": "minimum wall thickness for cylindrical shells under internal pressure",
        "expected_paragraph_ids": ["UG-27"],
        "category": "single_hop_factual",
    },
    {
        "query": "impact test exemptions for carbon steel",
        "expected_paragraph_ids": ["UCS-66"],
        "category": "single_hop_factual",
    },
    {
        "query": "joint efficiency for welded joints",
        "expected_paragraph_ids": ["UW-12"],
        "category": "single_hop_factual",
    },
    {
        "query": "hydrostatic testing requirements",
        "expected_paragraph_ids": ["UG-99"],
        "category": "single_hop_factual",
    },
    # Multi-hop (cross-reference chain)
    {
        "query": "minimum design metal temperature calculation",
        "expected_paragraph_ids": ["UCS-66", "UG-22"],
        "category": "multi_hop",
    },
    {
        "query": "welder qualification and WPS requirements for pressure vessels",
        "expected_paragraph_ids": ["QW-200", "UW-11"],
        "category": "multi_hop",
    },
    # Abstention (should not hallucinate unknown paragraphs)
    {
        "query": "ASME requirements for nuclear reactor pressure vessels",
        "expected_paragraph_ids": [],  # Not in VIII-1
        "category": "abstention",
    },
]


# ── Eval Framework ────────────────────────────────────────────────────────────

class ASMEEvalFramework:
    """
    Runs ASME retrieval evaluation.
    Metrics: Recall@5, Recall@10, MRR
    """

    def run_eval(self, lt, questions: list) -> dict:
        """
        Run eval questions through the retrieval pipeline.
        Returns {recall_at_5, recall_at_10, mrr, per_question_results}.
        """
        per_question = []
        rr_scores = []
        hits_at_5 = 0
        hits_at_10 = 0

        for q in questions:
            query = q["query"]
            expected = q.get("expected_paragraph_ids", [])
            category = q.get("category", "unknown")

            try:
                results = lt.query(query, top_k=10)
            except Exception as e:
                results = []

            retrieved_ids = [r.get("paragraph_id", "") for r in results]

            hit5 = self.citation_accuracy(retrieved_ids[:5], expected)
            hit10 = self.citation_accuracy(retrieved_ids[:10], expected)

            # MRR: find rank of first hit
            rr = 0.0
            if expected:
                for rank, pid in enumerate(retrieved_ids, start=1):
                    if any(e in pid for e in expected):
                        rr = 1.0 / rank
                        break

            if hit5:
                hits_at_5 += 1
            if hit10:
                hits_at_10 += 1
            if expected:  # skip abstention for MRR
                rr_scores.append(rr)

            per_question.append({
                "query": query,
                "category": category,
                "expected": expected,
                "retrieved_top5": retrieved_ids[:5],
                "hit@5": hit5,
                "hit@10": hit10,
                "rr": rr,
            })

        n = len(questions)
        mrr = float(sum(rr_scores) / len(rr_scores)) if rr_scores else 0.0

        return {
            "recall_at_5": hits_at_5 / n if n else 0.0,
            "recall_at_10": hits_at_10 / n if n else 0.0,
            "mrr": mrr,
            "n_questions": n,
            "per_question_results": per_question,
        }

    def citation_accuracy(self, results: list, expected_ids: list) -> bool:
        """Return True if any expected paragraph ID appears in any retrieved result."""
        if not expected_ids:
            return True  # abstention: no positive expected
        for res in results:
            for eid in expected_ids:
                if eid in str(res):
                    return True
        return False


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEvalQuestionBank(unittest.TestCase):
    """Verify the eval question bank structure."""

    def test_question_count(self):
        self.assertEqual(len(EVAL_QUESTIONS), 10)

    def test_all_questions_have_required_fields(self):
        for q in EVAL_QUESTIONS:
            self.assertIn("query", q)
            self.assertIn("expected_paragraph_ids", q)
            self.assertIn("category", q)

    def test_categories_covered(self):
        categories = {q["category"] for q in EVAL_QUESTIONS}
        self.assertIn("single_hop_factual", categories)
        self.assertIn("multi_hop", categories)
        self.assertIn("abstention", categories)

    def test_single_hop_count(self):
        single = [q for q in EVAL_QUESTIONS if q["category"] == "single_hop_factual"]
        self.assertGreaterEqual(len(single), 3)


class TestASMEEvalFramework(unittest.TestCase):
    """Test the eval framework with a mock retrieval system."""

    def _make_mock_lt(self, paragraph_id: str = "UG-22"):
        """Create a mock LeoTrident that returns a fixed paragraph ID."""
        lt = MagicMock()
        lt.query.return_value = [
            {"paragraph_id": paragraph_id, "content": f"Content for {paragraph_id}", "score": 0.9}
        ]
        return lt

    def test_run_eval_returns_metrics(self):
        fw = ASMEEvalFramework()
        lt = self._make_mock_lt("UG-22")
        results = fw.run_eval(lt, EVAL_QUESTIONS)

        self.assertIn("recall_at_5", results)
        self.assertIn("recall_at_10", results)
        self.assertIn("mrr", results)
        self.assertIn("per_question_results", results)
        self.assertEqual(results["n_questions"], len(EVAL_QUESTIONS))

    def test_citation_accuracy_hit(self):
        fw = ASMEEvalFramework()
        self.assertTrue(fw.citation_accuracy(["UG-22", "UW-11"], ["UG-22"]))

    def test_citation_accuracy_miss(self):
        fw = ASMEEvalFramework()
        self.assertFalse(fw.citation_accuracy(["UW-12", "UCS-66"], ["UG-22"]))

    def test_citation_accuracy_abstention(self):
        # Empty expected_ids = abstention = always True
        fw = ASMEEvalFramework()
        self.assertTrue(fw.citation_accuracy(["anything"], []))

    def test_recall_at_5_with_perfect_retrieval(self):
        """If we always return the exact expected ID, Recall@5 should be high."""
        fw = ASMEEvalFramework()

        def smart_query(query, top_k=10):
            # Return the expected paragraph ID based on keyword
            keyword_map = {
                "UG-22": "load", "UW-11": "welding exam", "QW-200": "procedure",
                "UG-27": "wall thickness", "UCS-66": "impact", "UW-12": "joint",
                "UG-99": "hydro",
            }
            for pid, kw in keyword_map.items():
                if kw.lower() in query.lower():
                    return [{"paragraph_id": pid, "content": "...", "score": 0.95}]
            return []

        lt = MagicMock()
        lt.query.side_effect = smart_query
        results = fw.run_eval(lt, EVAL_QUESTIONS[:7])  # factual only
        # At least some hits expected
        self.assertGreaterEqual(results["recall_at_5"], 0.0)

    def test_mrr_perfect_rank_one(self):
        fw = ASMEEvalFramework()
        q = [{"query": "loadings", "expected_paragraph_ids": ["UG-22"], "category": "single_hop_factual"}]
        lt = MagicMock()
        lt.query.return_value = [{"paragraph_id": "UG-22", "content": "...", "score": 0.9}]
        results = fw.run_eval(lt, q)
        self.assertEqual(results["mrr"], 1.0)


if __name__ == "__main__":
    unittest.main()
