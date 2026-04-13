"""
Tests for Phase 6b — Reference Relevance Judge
Mocks llm_client.complete — no external calls.
"""
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import init_schema
from src.retrieval.relevance_judge import ReferenceRelevanceJudge


def _seed_db(tmp_dir):
    """Create a DB with three paragraphs and typed edges from UG-22."""
    db_path = Path(tmp_dir) / "leo_trident.db"
    conn = init_schema(db_path)
    # Paragraphs
    paragraphs = [
        ('VIII-1_UG-22_2025', 'UG-22', 'Design loads for pressure vessels include internal pressure, weight, and seismic reactions.'),
        ('VIII-1_UW-12_2025', 'UW-12', 'Joint efficiency E shall be determined per this table.'),
        ('VIII-1_UCS-66_2025', 'UCS-66', 'Impact test exemptions for carbon steel.'),
        ('VIII-1_UG-99_2025', 'UG-99', 'Hydrostatic test procedures.'),
    ]
    for cid, pid, content in paragraphs:
        conn.execute(
            """INSERT INTO asme_chunks
               (chunk_id, paragraph_id, section, part, edition_year,
                content, content_hash, embedding_dim)
               VALUES (?,?,?,?,?,?,?,768)""",
            (cid, pid, 'VIII-1', 'UG', 2025, content, 'hash_' + cid),
        )
    # Edges from UG-22
    edges = [
        ('UG-22', 'UW-12',   'cross_ref', 'mandatory',     'shall comply with UW-12',      'Vessels shall comply with UW-12 for joint efficiency.', 2.0),
        ('UG-22', 'UCS-66',  'cross_ref', 'conditional',   'except as permitted by UCS-66', 'Except as permitted by UCS-66, impact testing is required.', 1.0),
        ('UG-22', 'UG-99',   'cross_ref', 'informational', 'see also UG-99',                'See also UG-99 for hydrotest guidance.', 0.3),
    ]
    for src, tgt, et, rt, ct, ctx, w in edges:
        conn.execute(
            """INSERT INTO graph_edges
               (source_id, target_id, edge_type, reference_type,
                citation_text, context, weight, edition_year)
               VALUES (?,?,?,?,?,?,?,2025)""",
            (src, tgt, et, rt, ct, ctx, w),
        )
    conn.commit()
    return conn, db_path


class TestRelevanceJudge(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.conn, self.db_path = _seed_db(self.tmp)

    def test_empty_results_no_llm_calls(self):
        with patch('src.retrieval.relevance_judge.llm_client.complete') as mock_llm:
            judge = ReferenceRelevanceJudge()
            out = judge.judge("anything", [], self.conn)
            self.assertEqual(out, [])
            mock_llm.assert_not_called()

    def test_primary_with_no_edges_returns_empty_refs(self):
        results = [{'paragraph_id': 'UW-12', 'content': 'Joint efficiency...'}]
        with patch('src.retrieval.relevance_judge.llm_client.complete') as mock_llm:
            judge = ReferenceRelevanceJudge()
            out = judge.judge("joint efficiency", results, self.conn)
            self.assertEqual(out[0]['references'], [])
            mock_llm.assert_not_called()

    @patch('src.retrieval.relevance_judge.llm_client.complete')
    def test_llm_happy_path(self, mock_llm):
        mock_llm.return_value = json.dumps([
            {"paragraph_id": "UW-12",  "relevance": "required",   "reason": "joint efficiency directly affects thickness calc"},
            {"paragraph_id": "UCS-66", "relevance": "irrelevant", "reason": "not about seismic"},
            {"paragraph_id": "UG-99",  "relevance": "optional",   "reason": "related testing context"},
        ])
        results = [{'paragraph_id': 'UG-22', 'content': 'Design loads include seismic...'}]
        judge = ReferenceRelevanceJudge()
        out = judge.judge("seismic design loads", results, self.conn)

        self.assertEqual(mock_llm.call_count, 1)
        refs = out[0]['references']
        self.assertEqual(len(refs), 3)
        by_id = {r['paragraph_id']: r for r in refs}
        self.assertEqual(by_id['UW-12']['relevance'], 'required')
        self.assertEqual(by_id['UCS-66']['relevance'], 'irrelevant')
        self.assertEqual(by_id['UG-99']['relevance'], 'optional')
        # Reference type should be attached from the edge
        self.assertEqual(by_id['UW-12']['reference_type'], 'mandatory')

    @patch('src.retrieval.relevance_judge.llm_client.complete')
    def test_malformed_json_falls_back(self, mock_llm):
        mock_llm.return_value = "Sorry, I cannot comply."
        results = [{'paragraph_id': 'UG-22', 'content': 'Design loads...'}]
        judge = ReferenceRelevanceJudge()
        out = judge.judge("seismic", results, self.conn)
        refs = out[0]['references']
        # Fallback uses reference_type defaults
        by_id = {r['paragraph_id']: r for r in refs}
        self.assertEqual(by_id['UW-12']['relevance'], 'required')        # mandatory
        self.assertEqual(by_id['UCS-66']['relevance'], 'optional')        # conditional
        self.assertEqual(by_id['UG-99']['relevance'], 'irrelevant')       # informational
        self.assertIn('fallback', by_id['UW-12']['reason'].lower())

    @patch('src.retrieval.relevance_judge.llm_client.complete')
    def test_wrong_length_array_falls_back(self, mock_llm):
        mock_llm.return_value = json.dumps([
            {"paragraph_id": "UW-12", "relevance": "required", "reason": "x"},
        ])  # only 1 entry for 3 edges
        results = [{'paragraph_id': 'UG-22', 'content': 'Design loads...'}]
        judge = ReferenceRelevanceJudge()
        out = judge.judge("anything", results, self.conn)
        refs = out[0]['references']
        self.assertEqual(len(refs), 3)
        # All should be fallback
        self.assertTrue(all('fallback' in r['reason'].lower() for r in refs))

    @patch('src.retrieval.relevance_judge.llm_client.complete')
    def test_llm_exception_falls_back_no_raise(self, mock_llm):
        mock_llm.side_effect = RuntimeError("connection refused")
        results = [{'paragraph_id': 'UG-22', 'content': 'Design loads...'}]
        judge = ReferenceRelevanceJudge()
        # Should not raise
        out = judge.judge("anything", results, self.conn)
        self.assertEqual(len(out[0]['references']), 3)

    @patch('src.retrieval.relevance_judge.llm_client.complete')
    def test_invalid_relevance_value_falls_back(self, mock_llm):
        mock_llm.return_value = json.dumps([
            {"paragraph_id": "UW-12",  "relevance": "maybe",     "reason": "x"},
            {"paragraph_id": "UCS-66", "relevance": "required",  "reason": "x"},
            {"paragraph_id": "UG-99",  "relevance": "optional",  "reason": "x"},
        ])
        results = [{'paragraph_id': 'UG-22', 'content': 'Design loads...'}]
        judge = ReferenceRelevanceJudge()
        out = judge.judge("anything", results, self.conn)
        # One invalid value poisons the whole response → fallback
        self.assertTrue(all('fallback' in r['reason'].lower()
                            for r in out[0]['references']))

    @patch('src.retrieval.relevance_judge.llm_client.complete')
    def test_strips_markdown_fences(self, mock_llm):
        mock_llm.return_value = (
            "```json\n"
            + json.dumps([
                {"paragraph_id": "UW-12",  "relevance": "required",   "reason": "a"},
                {"paragraph_id": "UCS-66", "relevance": "optional",   "reason": "b"},
                {"paragraph_id": "UG-99",  "relevance": "irrelevant", "reason": "c"},
            ])
            + "\n```"
        )
        results = [{'paragraph_id': 'UG-22', 'content': 'Design loads...'}]
        judge = ReferenceRelevanceJudge()
        out = judge.judge("anything", results, self.conn)
        refs = out[0]['references']
        self.assertEqual(refs[0]['relevance'], 'required')
        self.assertNotIn('fallback', refs[0]['reason'].lower())

    @patch('src.retrieval.relevance_judge.llm_client.complete')
    def test_max_refs_per_result_caps_and_prioritizes(self, mock_llm):
        # Add 5 more informational edges from UG-22 so it has 8 total
        for i, tgt in enumerate(['X-1', 'X-2', 'X-3', 'X-4', 'X-5']):
            self.conn.execute(
                """INSERT INTO asme_chunks
                   (chunk_id, paragraph_id, section, part, edition_year,
                    content, content_hash, embedding_dim)
                   VALUES (?,?,?,?,?,?,?,768)""",
                (f'c_{tgt}', tgt, 'VIII-1', 'X', 2025, f'content of {tgt}', f'h_{tgt}'),
            )
            self.conn.execute(
                """INSERT INTO graph_edges
                   (source_id, target_id, edge_type, reference_type,
                    citation_text, context, weight, edition_year)
                   VALUES (?,?,?,?,?,?,?,2025)""",
                ('UG-22', tgt, 'cross_ref', 'informational',
                 f'see {tgt}', f'see {tgt} for info', 0.3),
            )
        self.conn.commit()

        mock_llm.return_value = json.dumps([
            {"paragraph_id": "x", "relevance": "required", "reason": "x"}
        ] * 3)  # cap=3 for this test

        judge = ReferenceRelevanceJudge(max_refs_per_result=3)
        results = [{'paragraph_id': 'UG-22', 'content': 'Design loads...'}]
        judge.judge("anything", results, self.conn)

        # Inspect what was in the prompt: mandatory + conditional should win over informational
        prompt_sent = mock_llm.call_args[0][0]
        self.assertIn('UW-12', prompt_sent)    # mandatory
        self.assertIn('UCS-66', prompt_sent)   # conditional
        # Only one informational slot remains; UG-99 or one of X-* could fill it
        informationals_in_prompt = sum(
            1 for p in ['UG-99', 'X-1', 'X-2', 'X-3', 'X-4', 'X-5']
            if p in prompt_sent
        )
        self.assertEqual(informationals_in_prompt, 1)


class TestAPIIntegration(unittest.TestCase):
    """Smoke test: use_relevance_judge=False is the default and unchanged."""

    def test_default_query_unchanged(self):
        # The existing test_e2e.py covers this; just assert the parameter exists
        # and defaults to False.
        import inspect
        from src.api import LeoTrident
        sig = inspect.signature(LeoTrident.query)
        self.assertIn('use_relevance_judge', sig.parameters)
        self.assertEqual(sig.parameters['use_relevance_judge'].default, False)


if __name__ == "__main__":
    unittest.main()
