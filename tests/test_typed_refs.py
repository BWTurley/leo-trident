"""
Tests for Phase 6a — Typed Cross-Reference Edges
"""
import os
import sys
import sqlite3
import tempfile
import shutil
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import init_schema
from src.ingest.asme_parser import (
    ASMEParser, ASMEChunk, GraphEdge,
    classify_reference, extract_context_window,
    REFERENCE_TYPE_WEIGHTS,
)


class TestClassifyReference(unittest.TestCase):

    def test_mandatory_shall_comply(self):
        assert classify_reference("vessels shall comply with UW-12") == 'mandatory'

    def test_mandatory_in_accordance_with(self):
        assert classify_reference("examined in accordance with UW-51") == 'mandatory'

    def test_mandatory_as_required_by(self):
        assert classify_reference("as required by UCS-66(a)") == 'mandatory'

    def test_mandatory_requirements_of(self):
        assert classify_reference("meet the requirements of UW-11") == 'mandatory'

    def test_conditional_except_as_permitted(self):
        assert classify_reference("except as permitted by UCS-66") == 'conditional'

    def test_conditional_when_applicable(self):
        # 'when applicable' wins over 'see' because conditional is checked before informational
        assert classify_reference("when applicable, see UG-22") == 'conditional'

    def test_informational_see_also(self):
        assert classify_reference("see also Appendix D") == 'informational'

    def test_informational_refer_to(self):
        assert classify_reference("refer to UG-99 for guidance") == 'informational'

    def test_unclassified_bare_citation(self):
        assert classify_reference("The thickness (UG-27) varies") == 'unclassified'


class TestExtractCrossRefsWithContext(unittest.TestCase):

    def test_mixed_citations_in_one_paragraph(self):
        text = (
            "Vessels shall comply with UW-12 for joint efficiency. "
            "Except as permitted by UCS-66, impact testing is required. "
            "See also UG-22 for design loads."
        )
        parser = ASMEParser()
        refs = parser.extract_cross_refs_with_context(text)
        by_id = {r['ref_id']: r for r in refs}
        assert by_id['UW-12']['reference_type'] == 'mandatory'
        assert by_id['UCS-66']['reference_type'] == 'conditional'
        assert by_id['UG-22']['reference_type'] == 'informational'

    def test_dedup_keeps_highest_priority(self):
        text = "See UW-12. Vessels shall comply with UW-12 per this section."
        parser = ASMEParser()
        refs = parser.extract_cross_refs_with_context(text)
        uw12_refs = [r for r in refs if r['ref_id'] == 'UW-12']
        assert len(uw12_refs) == 1
        assert uw12_refs[0]['reference_type'] == 'mandatory'

    def test_self_reference_excluded(self):
        text = "UG-22 covers loadings. Vessels shall comply with UG-22."
        parser = ASMEParser()
        refs = parser.extract_cross_refs_with_context(text, self_id='UG-22')
        assert all(r['ref_id'] != 'UG-22' for r in refs)

    def test_backward_compat_wrapper(self):
        text = "Vessels shall comply with UW-12. See also UG-22."
        parser = ASMEParser()
        ids = parser.extract_cross_refs(text)
        assert isinstance(ids, list)
        assert all(isinstance(x, str) for x in ids)
        assert set(ids) == {'UW-12', 'UG-22'}


class TestGraphEdgeRoundTrip(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="leo_typed_refs_")
        self.db_path = os.path.join(self.tmp_dir, "test.db")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_insert_and_read_all_columns(self):
        conn = init_schema(self.db_path)
        edge = GraphEdge(
            source_id='UG-22',
            target_id='UW-12',
            edge_type='cross_ref',
            reference_type='mandatory',
            citation_text='shall comply with UW-12',
            context='Vessels shall comply with UW-12 for joint efficiency.',
            weight=2.0,
            edition_year=2025,
        )
        ASMEParser.insert_edges(conn, [edge])
        row = conn.execute(
            "SELECT source_id, target_id, edge_type, reference_type, "
            "citation_text, context, weight, edition_year FROM graph_edges LIMIT 1"
        ).fetchone()
        assert row['source_id'] == 'UG-22'
        assert row['target_id'] == 'UW-12'
        assert row['reference_type'] == 'mandatory'
        assert row['citation_text'] == 'shall comply with UW-12'
        assert 'shall comply' in row['context']
        assert row['weight'] == 2.0
        assert row['edition_year'] == 2025
        conn.close()

    def test_migration_adds_columns_to_existing_db(self):
        # Create a DB with the OLD graph_edges schema (no new columns)
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE graph_edges (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT DEFAULT 'cross_ref',
                weight REAL DEFAULT 1.0,
                edition_year INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_id, target_id, edge_type)
            )
        """)
        conn.commit()
        conn.close()

        # Now run init_schema which should add the new columns via migration
        conn = init_schema(self.db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(graph_edges)")}
        assert 'reference_type' in cols
        assert 'citation_text' in cols
        assert 'context' in cols
        conn.close()

    def test_weight_derived_from_reference_type(self):
        text = (
            "Vessels shall comply with UW-12 for joint efficiency. "
            "See also UG-22 for design loads."
        )
        parser = ASMEParser(edition_year=2025)
        chunks = parser.parse_text(text, paragraph_id='UG-99')
        edges = ASMEParser.chunks_to_edges(chunks)
        by_target = {e.target_id: e for e in edges}
        # UW-12 should be mandatory → weight 2.0
        assert by_target['UW-12'].weight == 2.0
        assert by_target['UW-12'].reference_type == 'mandatory'
        # UG-22 should be informational → weight 0.3
        assert by_target['UG-22'].weight == 0.3
        assert by_target['UG-22'].reference_type == 'informational'


class TestPPRWithTypedEdges(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="leo_ppr_typed_")
        self.db_path = os.path.join(self.tmp_dir, "test.db")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_mandatory_outweighs_informational(self):
        from src.retrieval.ppr import ASMEGraphPPR

        conn = init_schema(self.db_path)
        # A -> B (mandatory, weight 2.0)
        # A -> C (informational, weight 0.3)
        conn.execute(
            "INSERT INTO graph_edges (source_id, target_id, edge_type, reference_type, weight) "
            "VALUES ('A', 'B', 'cross_ref', 'mandatory', 2.0)"
        )
        conn.execute(
            "INSERT INTO graph_edges (source_id, target_id, edge_type, reference_type, weight) "
            "VALUES ('A', 'C', 'cross_ref', 'informational', 0.3)"
        )
        conn.commit()

        ppr = ASMEGraphPPR.from_sqlite(conn)
        results = ppr.query(['A'], top_k=10)
        scores = {pid: score for pid, score in results}

        assert 'B' in scores, f"B not found in PPR results: {results}"
        assert 'C' in scores, f"C not found in PPR results: {results}"
        assert scores['B'] > scores['C'], (
            f"Expected B ({scores['B']}) > C ({scores['C']}) "
            f"since mandatory weight (2.0) > informational (0.3)"
        )
        conn.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
