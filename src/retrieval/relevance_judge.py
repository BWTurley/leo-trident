"""
Leo Trident — Reference Relevance Judge

For each reranked result, uses the LLM to decide which of the result's
outgoing cross-references are required / optional / irrelevant for the
current query. Best-effort enrichment — never raises.

Dispatches through src.memory.llm_client.complete() so it works in both
LEO_LLM_MODE=cloud and LEO_LLM_MODE=local.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from typing import Optional

from src.memory import llm_client

logger = logging.getLogger(__name__)


# Priority for choosing which citations to send when a primary has more
# than max_refs_per_result. Matches REFERENCE_TYPE_WEIGHTS intent.
_REF_TYPE_PRIORITY = {
    'mandatory':     3,
    'conditional':   2,
    'informational': 1,
    'unclassified':  0,
}

# Map reference_type → fallback relevance when the LLM fails
_FALLBACK_RELEVANCE = {
    'mandatory':     'required',
    'conditional':   'optional',
    'informational': 'irrelevant',
    'unclassified':  'optional',
}


PROMPT_TEMPLATE = """\
You are a retrieval-relevance judge for ASME Boiler & Pressure Vessel Code queries.

USER QUERY:
{query}

PRIMARY PARAGRAPH: {primary_id}
{primary_content}

This paragraph cites the following other paragraphs. For each citation, decide
whether its content is REQUIRED, OPTIONAL, or IRRELEVANT for answering the
query above.

- REQUIRED: the query cannot be correctly answered without reading this paragraph
- OPTIONAL: helpful background but not needed for a correct answer
- IRRELEVANT: cited in {primary_id} but unrelated to this specific query

CITATIONS:
{enumerated_citations}

Respond with ONLY a JSON array, one object per citation, in the SAME ORDER as
listed above. No prose before or after. Schema:
[{{"paragraph_id": "...", "relevance": "required|optional|irrelevant", "reason": "one sentence"}}]
"""


class ReferenceRelevanceJudge:
    def __init__(self, max_refs_per_result: int = 10,
                 max_primary_content_chars: int = 1500,
                 max_citation_content_chars: int = 400):
        self.max_refs_per_result = max_refs_per_result
        self.max_primary_content_chars = max_primary_content_chars
        self.max_citation_content_chars = max_citation_content_chars

    def judge(self, query: str, results: list[dict],
              conn: sqlite3.Connection) -> list[dict]:
        """
        Mutate each result in-place by adding result['references'] = [...].
        Returns the same list. Never raises.
        """
        if not results:
            return results

        for result in results:
            try:
                result['references'] = self._judge_one(query, result, conn)
            except Exception as e:
                logger.warning(
                    f"Relevance judge failed for {result.get('paragraph_id', '?')}: {e}"
                )
                result['references'] = []
        return results

    def _judge_one(self, query: str, result: dict,
                   conn: sqlite3.Connection) -> list[dict]:
        primary_id = result.get('paragraph_id', '')
        if not primary_id:
            return []

        edges = self._fetch_edges(conn, primary_id)
        if not edges:
            return []

        # Cap at max_refs_per_result, preferring higher-priority types
        if len(edges) > self.max_refs_per_result:
            edges.sort(
                key=lambda e: _REF_TYPE_PRIORITY.get(e['reference_type'], 0),
                reverse=True,
            )
            edges = edges[:self.max_refs_per_result]

        # Fetch content for each target
        target_ids = [e['target_id'] for e in edges]
        contents = self._fetch_contents(conn, target_ids)

        # Build enumerated citations block
        lines = []
        for i, edge in enumerate(edges, 1):
            tid = edge['target_id']
            body = contents.get(tid, '(content not found)')
            body = body[:self.max_citation_content_chars]
            citation_phrase = edge.get('citation_text') or edge.get('context') or ''
            citation_phrase = citation_phrase[:200]
            lines.append(
                f'{i}. {tid} [{edge["reference_type"]}] — cited as: "{citation_phrase}"\n'
                f'   Content: {body}'
            )
        enumerated = '\n\n'.join(lines)

        primary_content = (result.get('content') or '')[:self.max_primary_content_chars]
        prompt = PROMPT_TEMPLATE.format(
            query=query,
            primary_id=primary_id,
            primary_content=primary_content,
            enumerated_citations=enumerated,
        )

        # Call LLM; on ANY failure, fall back to reference_type defaults
        try:
            raw = llm_client.complete(prompt, max_tokens=1024, temperature=0.0)
            parsed = self._parse_response(raw, edges)
        except Exception as e:
            logger.warning(f"LLM call failed in relevance judge: {e}")
            parsed = None

        if parsed is None:
            parsed = self._fallback(edges)

        # Attach content and reference_type to each entry
        for entry, edge in zip(parsed, edges):
            entry.setdefault('reference_type', edge['reference_type'])
            entry.setdefault('content', contents.get(edge['target_id'], '')[:self.max_citation_content_chars])

        return parsed

    def _fetch_edges(self, conn: sqlite3.Connection,
                     source_id: str) -> list[dict]:
        try:
            rows = conn.execute(
                """SELECT target_id, reference_type, citation_text, context,
                          edge_type, weight
                   FROM graph_edges
                   WHERE source_id = ? AND edge_type = 'cross_ref'""",
                (source_id,),
            ).fetchall()
        except sqlite3.Error as e:
            logger.warning(f"Edge fetch failed for {source_id}: {e}")
            return []
        return [dict(r) for r in rows]

    def _fetch_contents(self, conn: sqlite3.Connection,
                        paragraph_ids: list[str]) -> dict[str, str]:
        if not paragraph_ids:
            return {}
        placeholders = ','.join('?' * len(paragraph_ids))
        try:
            rows = conn.execute(
                f"""SELECT paragraph_id, content FROM asme_chunks
                    WHERE paragraph_id IN ({placeholders})""",
                paragraph_ids,
            ).fetchall()
        except sqlite3.Error as e:
            logger.warning(f"Content fetch failed: {e}")
            return {}
        # If same paragraph_id has multiple chunks, keep the first
        out: dict[str, str] = {}
        for r in rows:
            pid = r['paragraph_id']
            if pid not in out:
                out[pid] = r['content'] or ''
        return out

    def _parse_response(self, raw: str,
                        edges: list[dict]) -> Optional[list[dict]]:
        """Parse JSON array from LLM response. Return None if malformed."""
        # Strip markdown fences if present
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip(),
                         flags=re.MULTILINE)
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(data, list) or len(data) != len(edges):
            return None
        # Validate shape
        valid_relevance = {'required', 'optional', 'irrelevant'}
        out = []
        for entry, edge in zip(data, edges):
            if not isinstance(entry, dict):
                return None
            pid = entry.get('paragraph_id') or edge['target_id']
            rel = entry.get('relevance', '').lower()
            if rel not in valid_relevance:
                return None
            out.append({
                'paragraph_id': pid,
                'relevance': rel,
                'reason': entry.get('reason', '')[:500],
            })
        return out

    def _fallback(self, edges: list[dict]) -> list[dict]:
        """Map reference_type → relevance deterministically."""
        return [
            {
                'paragraph_id': e['target_id'],
                'relevance': _FALLBACK_RELEVANCE.get(e['reference_type'], 'optional'),
                'reason': f"(fallback: {e['reference_type']} citation; LLM unavailable)",
            }
            for e in edges
        ]
