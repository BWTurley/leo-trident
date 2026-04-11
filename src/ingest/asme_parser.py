"""
ASME BPVC Parser
Parses pre-split plain-text paragraph files, extracts paragraph IDs,
cross-references, and hierarchical metadata. Outputs structured dicts
suitable for direct insertion into asme_chunks + graph_edges.
"""

import re
import json
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

# ── Regex Patterns ────────────────────────────────────────────────────────────

# Primary paragraph ID patterns for ASME BPVC sections
PARAGRAPH_ID_PATTERNS = [
    # Section VIII Div 1 — UG, UW, UCS, UNF, UHA, UHT, ULT, UCL, etc.
    r'\b(U[A-Z]{1,3}-\d{1,4}(?:\.[a-z0-9]+)*(?:\([a-z0-9]+\))*)\b',
    # Section IX — QW, QB
    r'\b(Q[BW]-\d{1,4}(?:\.[a-z0-9]+)*(?:\([a-z0-9]+\))*)\b',
    # Section I — PG, PW, PB, etc.
    r'\b(P[A-Z]{1,2}-\d{1,4}(?:\.[a-z0-9]+)*(?:\([a-z0-9]+\))*)\b',
    # Section V — Article numbers (T-110, T-150, etc.)
    r'\b(T-\d{3,4}(?:\.[a-z0-9]+)*(?:\([a-z0-9]+\))*)\b',
    # Generic fallback: 1-3 uppercase letters + dash + 1-4 digits
    r'\b([A-Z]{1,3}-\d{1,4}(?:\.\d+)*(?:\([a-z0-9]\))*)\b',
]

# Cross-reference extraction patterns
CROSS_REF_PATTERNS = [
    r'(?:see|refer to|per|as required by|in accordance with|subject to)\s+'
    r'([A-Z]{1,3}-\d{1,4}(?:\.[a-z0-9]+)*(?:\([a-z0-9]+\))*)',
    r'(?:see also|as defined in|as specified in)\s+'
    r'([A-Z]{1,3}-\d{1,4}(?:\.[a-z0-9]+)*(?:\([a-z0-9]+\))*)',
    # Inline references like "(see UG-99)"
    r'\(see\s+([A-Z]{1,3}-\d{1,4}(?:\.[a-z0-9]+)*(?:\([a-z0-9]+\))*)\)',
    # "requirements of UW-11"
    r'requirements of\s+([A-Z]{1,3}-\d{1,4}(?:\.[a-z0-9]+)*(?:\([a-z0-9]+\))*)',
    # Bare references following a comma or "and" — less reliable
    r'(?:and|or),?\s+([A-Z]{1,3}-\d{1,4}(?:\.\d+)+(?:\([a-z0-9]\))?)\b',
]

# Section / part detection from paragraph prefix
SECTION_MAP = {
    'UG': ('VIII-1', 'UG'), 'UW': ('VIII-1', 'UW'), 'UF': ('VIII-1', 'UF'),
    'UCS': ('VIII-1', 'UCS'), 'UNF': ('VIII-1', 'UNF'), 'UHA': ('VIII-1', 'UHA'),
    'UHT': ('VIII-1', 'UHT'), 'ULT': ('VIII-1', 'ULT'), 'UCL': ('VIII-1', 'UCL'),
    'UCD': ('VIII-1', 'UCD'), 'UIG': ('VIII-1', 'UIG'),
    'QW': ('IX', 'QW'), 'QB': ('IX', 'QB'),
    'PG': ('I', 'PG'), 'PW': ('I', 'PW'), 'PB': ('I', 'PB'), 'PFH': ('I', 'PFH'),
    'T': ('V', 'T'),
}

# Whether a paragraph prefix is typically mandatory
MANDATORY_PREFIXES = {
    'UG', 'UW', 'UCS', 'UNF', 'UHA', 'UHT', 'ULT', 'UCL',
    'QW', 'QB', 'PG', 'PW',
}

# ASME hierarchy levels for common parts
HIERARCHY = {
    'VIII-1': {
        'UG': ('Subsection A', 'Part UG — General Requirements'),
        'UW': ('Subsection B', 'Part UW — Welded Pressure Vessels'),
        'UF': ('Subsection B', 'Part UF — Forged Pressure Vessels'),
        'UB': ('Subsection B', 'Part UB — Brazed Pressure Vessels'),
        'UCS': ('Subsection C', 'Part UCS — Carbon and Low Alloy Steel'),
        'UNF': ('Subsection C', 'Part UNF — Nonferrous Materials'),
        'UHA': ('Subsection C', 'Part UHA — High Alloy Steel'),
        'UHT': ('Subsection C', 'Part UHT — Ferritic Steels with Tensile Props'),
        'ULT': ('Subsection C', 'Part ULT — Low Temperature Operation'),
        'UCL': ('Subsection C', 'Part UCL — Clad and Lined Material'),
    },
    'IX': {
        'QW': ('Part QW', 'Welding'),
        'QB': ('Part QB', 'Brazing'),
    },
    'I': {
        'PG': ('Part PG', 'General Requirements'),
        'PW': ('Part PW', 'Requirements for Boilers Fabricated by Welding'),
    },
    'V': {
        'T': ('Subsection A', 'NDE Methods'),
    },
}


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class ASMEChunk:
    chunk_id: str
    paragraph_id: str
    section: str
    part: str
    edition_year: int
    content: str
    content_hash: str
    cross_refs: list[str] = field(default_factory=list)
    no_forget: bool = True
    mandatory: bool = True
    content_type: str = 'normative'
    raptor_level: int = 0
    subsection: str = ''
    part_title: str = ''
    # Optional: subparagraph label if split from parent
    subparagraph: str = ''
    # Hierarchy metadata
    parent_id: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d['cross_refs'] = json.dumps(d['cross_refs'])
        return d

    def to_db_row(self) -> dict:
        """Return dict matching asme_chunks table columns."""
        return {
            'chunk_id': self.chunk_id,
            'paragraph_id': self.paragraph_id,
            'section': self.section,
            'part': self.part,
            'edition_year': self.edition_year,
            'valid_from': f'{self.edition_year}-07-01',
            'valid_to': None,
            'content': self.content,
            'content_hash': self.content_hash,
            'no_forget': self.no_forget,
            'mandatory': self.mandatory,
            'content_type': self.content_type,
            'cross_refs': json.dumps(self.cross_refs),
            'raptor_level': self.raptor_level,
            'embedding_dim': 768,
        }


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    edge_type: str = 'cross_ref'
    weight: float = 1.0
    edition_year: Optional[int] = None

    def to_db_row(self) -> dict:
        return asdict(self)


# ── Core Parser ───────────────────────────────────────────────────────────────

class ASMEParser:
    """
    Parses pre-split plain text ASME paragraphs into structured chunks.

    Input expectations:
      - Each file = one paragraph (e.g. UG-22.txt, QW-200.txt)
      - OR: a single file with paragraphs separated by blank lines
      - File name may encode paragraph ID: "UG-22.txt" → paragraph_id = "UG-22"
      - Content is already text-extracted (no PDF parsing needed here)
    """

    # Compiled regex for performance
    _para_id_re = re.compile(
        '|'.join(PARAGRAPH_ID_PATTERNS), re.IGNORECASE
    )
    _xref_re = re.compile(
        '|'.join(CROSS_REF_PATTERNS), re.IGNORECASE
    )
    _subpara_re = re.compile(r'^\s*\(([a-z0-9]+)\)\s+', re.MULTILINE)

    def __init__(self, edition_year: int = 2025):
        self.edition_year = edition_year

    # ── ID extraction ──────────────────────────────────────────────────

    def extract_paragraph_id(self, text: str, filename: str = '') -> Optional[str]:
        """
        Identify the primary paragraph ID for a chunk.
        Prefers filename hint, then first occurrence in text body.
        """
        # Try filename first (most reliable)
        if filename:
            stem = Path(filename).stem
            # Direct match: "UG-22", "QW-200", "UW-11"
            direct = re.match(r'^([A-Z]{1,3}-\d{1,4})', stem, re.IGNORECASE)
            if direct:
                return direct.group(1).upper()

        # Search text for the first standalone paragraph ID at line start
        for line in text.splitlines()[:20]:  # header zone
            m = re.match(r'^\s*([A-Z]{1,3}-\d{1,4}(?:\.[a-z0-9]+)*(?:\([a-z0-9]+\))?)\b',
                         line.strip())
            if m:
                return m.group(1)

        # Fall back: first match anywhere in text
        m = self._para_id_re.search(text)
        if m:
            # Return whichever capture group matched
            return next(g for g in m.groups() if g is not None)

        return None

    def extract_cross_refs(self, text: str, self_id: Optional[str] = None) -> list[str]:
        """Extract all cross-referenced paragraph IDs from text."""
        refs = set()
        for m in self._xref_re.finditer(text):
            # One of the capture groups will be non-None
            ref = next((g for g in m.groups() if g), None)
            if ref:
                ref = ref.strip().upper()
                if self_id and ref == self_id:
                    continue  # skip self-references
                refs.add(ref)

        # Also scan for any paragraph IDs in text that look like references
        for m in self._para_id_re.finditer(text):
            ref = next((g for g in m.groups() if g), None)
            if ref:
                ref = ref.strip().upper()
                if self_id and ref == self_id:
                    continue
                refs.add(ref)

        return sorted(refs)

    # ── Hierarchy resolution ───────────────────────────────────────────

    def resolve_hierarchy(self, paragraph_id: str) -> tuple[str, str, str, str]:
        """
        Return (section, part, subsection, part_title) for a paragraph ID.
        E.g. "UG-22" → ("VIII-1", "UG", "Subsection A", "Part UG — General Requirements")
        """
        prefix_match = re.match(r'^([A-Z]+)', paragraph_id)
        if not prefix_match:
            return ('UNKNOWN', 'UNKNOWN', '', '')

        prefix = prefix_match.group(1)
        if prefix in SECTION_MAP:
            section, part = SECTION_MAP[prefix]
        else:
            section, part = 'UNKNOWN', prefix

        subsection = ''
        part_title = ''
        if section in HIERARCHY and part in HIERARCHY[section]:
            subsection, part_title = HIERARCHY[section][part]

        return section, part, subsection, part_title

    def is_mandatory(self, paragraph_id: str, content: str = '') -> bool:
        """Heuristic: most UG/UW/UCS/QW paragraphs are mandatory."""
        prefix_match = re.match(r'^([A-Z]+)', paragraph_id)
        if not prefix_match:
            return True
        prefix = prefix_match.group(1)
        if prefix in MANDATORY_PREFIXES:
            return True
        # Nonmandatory appendices
        if re.search(r'nonmandatory|appendix [A-Z]', content[:200], re.IGNORECASE):
            return False
        return True

    # ── Chunking ──────────────────────────────────────────────────────

    def chunk_paragraph(self, text: str, paragraph_id: str,
                        max_tokens: int = 500) -> list[str]:
        """
        Split a long paragraph at subparagraph boundaries (a), (b), (c)...
        Short adjacent subparagraphs are merged with parent if under threshold.
        Returns list of text chunks.
        """
        # Rough token estimate: words * 1.3
        word_count = len(text.split())
        approx_tokens = int(word_count * 1.3)

        if approx_tokens <= max_tokens:
            return [text.strip()]

        # Split at subparagraph markers
        parts = self._subpara_re.split(text)
        if len(parts) <= 1:
            # No subparagraph markers — hard split at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            current = []
            current_words = 0
            for sent in sentences:
                w = len(sent.split())
                if current_words + w > max_tokens // 1.3 and current:
                    chunks.append(' '.join(current))
                    current = [sent]
                    current_words = w
                else:
                    current.append(sent)
                    current_words += w
            if current:
                chunks.append(' '.join(current))
            return [c.strip() for c in chunks if c.strip()]

        # Merge small subparagraphs
        chunks = []
        current = parts[0]  # preamble text before first subpara
        i = 1
        while i < len(parts) - 1:
            label = parts[i]
            body = parts[i + 1]
            combined = f"({label}) {body}"
            w = len((current + combined).split())
            if w * 1.3 < max_tokens:
                current += "\n" + combined
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = combined
            i += 2

        if current.strip():
            chunks.append(current.strip())

        return chunks or [text.strip()]

    # ── Main parse entry points ────────────────────────────────────────

    def parse_file(self, filepath: Path) -> list[ASMEChunk]:
        """Parse a single paragraph file → list of ASMEChunk objects."""
        text = filepath.read_text(encoding='utf-8', errors='replace').strip()
        if not text:
            return []

        paragraph_id = self.extract_paragraph_id(text, filepath.name)
        if not paragraph_id:
            logger.warning(f"Could not determine paragraph_id for {filepath.name}")
            paragraph_id = filepath.stem.upper()

        return self._build_chunks(text, paragraph_id, str(filepath))

    def parse_text(self, text: str, paragraph_id: Optional[str] = None,
                   source: str = '') -> list[ASMEChunk]:
        """Parse raw text (already extracted). paragraph_id can be provided explicitly."""
        text = text.strip()
        if not text:
            return []
        if not paragraph_id:
            paragraph_id = self.extract_paragraph_id(text, source)
        if not paragraph_id:
            logger.warning(f"Could not determine paragraph_id from text (source={source})")
            paragraph_id = 'UNKNOWN'
        return self._build_chunks(text, paragraph_id, source)

    def parse_directory(self, directory: Path,
                        glob: str = '*.txt') -> list[ASMEChunk]:
        """Parse all text files in a directory."""
        chunks = []
        for fp in sorted(directory.glob(glob)):
            try:
                chunks.extend(self.parse_file(fp))
            except Exception as e:
                logger.error(f"Failed to parse {fp}: {e}")
        return chunks

    def parse_bulk_text(self, text: str, delimiter: str = '\n\n---\n\n') -> list[ASMEChunk]:
        """
        Parse a single file containing multiple paragraphs separated by delimiter.
        Each section should start with paragraph ID on first line.
        """
        chunks = []
        for block in text.split(delimiter):
            block = block.strip()
            if block:
                chunks.extend(self.parse_text(block))
        return chunks

    # ── Internal ──────────────────────────────────────────────────────

    def _build_chunks(self, text: str, paragraph_id: str,
                      source: str) -> list[ASMEChunk]:
        paragraph_id = paragraph_id.upper()
        section, part, subsection, part_title = self.resolve_hierarchy(paragraph_id)
        mandatory = self.is_mandatory(paragraph_id, text)

        # Content type detection
        content_type = 'normative'
        if re.search(r'nonmandatory appendix', text[:300], re.IGNORECASE):
            content_type = 'appendix_nonmandatory'
        elif re.search(r'^appendix\b', text[:100], re.IGNORECASE):
            content_type = 'appendix'

        # Split into sub-chunks if needed
        text_chunks = self.chunk_paragraph(text, paragraph_id)

        result = []
        for i, chunk_text in enumerate(text_chunks):
            suffix = f'_{i}' if len(text_chunks) > 1 else ''
            chunk_id = f'{section}_{paragraph_id}{suffix}_{self.edition_year}'.replace(' ', '_')

            cross_refs = self.extract_cross_refs(chunk_text, paragraph_id)
            content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()

            chunk = ASMEChunk(
                chunk_id=chunk_id,
                paragraph_id=f'{paragraph_id}{suffix}' if suffix else paragraph_id,
                section=section,
                part=part,
                edition_year=self.edition_year,
                content=chunk_text,
                content_hash=content_hash,
                cross_refs=cross_refs,
                no_forget=True,
                mandatory=mandatory,
                content_type=content_type,
                raptor_level=0,
                subsection=subsection,
                part_title=part_title,
            )
            result.append(chunk)

        return result

    # ── Graph edge extraction ──────────────────────────────────────────

    @staticmethod
    def chunks_to_edges(chunks: list[ASMEChunk]) -> list[GraphEdge]:
        """Convert cross-reference data from chunks into GraphEdge list."""
        edges = []
        for chunk in chunks:
            for ref in chunk.cross_refs:
                edges.append(GraphEdge(
                    source_id=chunk.paragraph_id,
                    target_id=ref,
                    edge_type='cross_ref',
                    weight=1.0,
                    edition_year=chunk.edition_year,
                ))
        return edges

    # ── DB insert helpers ──────────────────────────────────────────────

    @staticmethod
    def insert_chunks(conn, chunks: list[ASMEChunk]) -> int:
        """Insert chunks into asme_chunks table. Returns count inserted."""
        rows = [c.to_db_row() for c in chunks]
        conn.executemany("""
            INSERT OR REPLACE INTO asme_chunks
            (chunk_id, paragraph_id, section, part, edition_year,
             valid_from, valid_to, content, content_hash,
             no_forget, mandatory, content_type, cross_refs, raptor_level, embedding_dim)
            VALUES
            (:chunk_id, :paragraph_id, :section, :part, :edition_year,
             :valid_from, :valid_to, :content, :content_hash,
             :no_forget, :mandatory, :content_type, :cross_refs, :raptor_level, :embedding_dim)
        """, rows)
        conn.commit()
        return len(rows)

    @staticmethod
    def insert_edges(conn, edges: list[GraphEdge]) -> int:
        """Insert graph edges. Returns count inserted."""
        rows = [e.to_db_row() for e in edges]
        conn.executemany("""
            INSERT OR IGNORE INTO graph_edges
            (source_id, target_id, edge_type, weight, edition_year)
            VALUES (:source_id, :target_id, :edge_type, :weight, :edition_year)
        """, rows)
        conn.commit()
        return len(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Parse ASME text files')
    parser.add_argument('input', help='File or directory to parse')
    parser.add_argument('--edition', type=int, default=2025)
    parser.add_argument('--output', help='Output JSON file (default: stdout)')
    parser.add_argument('--glob', default='*.txt', help='File glob pattern')
    args = parser.parse_args()

    p = ASMEParser(edition_year=args.edition)
    inp = Path(args.input)

    if inp.is_dir():
        chunks = p.parse_directory(inp, glob=args.glob)
    else:
        chunks = p.parse_file(inp)

    output = [c.to_dict() for c in chunks]

    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f'Wrote {len(chunks)} chunks to {args.output}')
    else:
        print(json.dumps(output, indent=2))
