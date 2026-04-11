# Leo Trident — Memory System


---

## What It Is

Leo Trident is a **local-first, three-tier memory and retrieval system** designed to give an AI agent persistent, semantically searchable memory across sessions — with special support for the ASME Boiler and Pressure Vessel Code (BPVC).

It runs entirely on commodity hardware (currently Abacus.AI cloud, migrating to Mac Mini M5). No cloud vector databases. No graph servers. No dependencies that can't be swapped for local equivalents.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    LEO (AI Agent)                           │
│                Claude Sonnet via Abacus.AI                  │
└──────────────────────┬──────────────────────────────────────┘
                       │  query / ingest
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              LeoTrident API  (src/api.py)                   │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────────┐   │
│  │  BM25    │   │  Dense   │   │  PPR Graph           │   │
│  │ (FTS5)   │   │ (LanceDB)│   │  (scipy CSR)         │   │
│  └────┬─────┘   └────┬─────┘   └──────────┬───────────┘   │
│       └──────────────┴──────────────────────┘              │
│                       │                                     │
│              RRF Fusion  (src/retrieval/fusion.py)          │
│                       │                                     │
│              BGE Reranker  (src/retrieval/reranker.py)      │
│                       │                                     │
│               Top-K Results returned to Leo                 │
└─────────────────────────────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐    ┌──────────────────────┐
│    LanceDB       │    │     SQLite           │
│  (vector store)  │    │  (FTS5 + metadata    │
│                  │    │   + graph edges)     │
│  chunks_hot 64d  │    │                      │
│  chunks_warm256d │    │  asme_chunks         │
│  chunks_cold768d │    │  chunks_fts          │
│  personal_warm   │    │  graph_edges         │
└──────────────────┘    │  tier_registry       │
                        │  conversation_logs   │
                        └──────────────────────┘
                                   ▲
                        ┌──────────┘
                        │
              ┌─────────────────────┐
              │   Markdown Vault    │
              │  (plain .md files)  │
              │                     │
              │  vault/Personal/    │
              │  vault/ASME-BPVC/   │
              │  vault/Sessions/    │
              │  vault/_system/     │
              │    hot.json         │
              │    anchors.json     │
              └─────────────────────┘
```

---

## Three-Tier Memory

The core design is a **hot / warm / cold** memory hierarchy. Every piece of information lives in exactly one tier based on how often it's accessed.

### HOT Tier — Always On (≤200 tokens)

**What lives here:** Brett's identity, active project context, ASME safety pins, session hints.

**How it works:** `vault/_system/hot.json` is read at the start of every agent turn and injected directly into the prompt. No retrieval needed — it's always there.

**Format:**
```
[PERSONA]
Brett | ASME QC Inspector → HSB Authorized Inspector | VIII-1 focus | Albany NY

[SAFETY PINS] 🔒
NEVER: waive UG-99 hydro | skip UW-51 spot RT | ignore PQR essential vars
ALWAYS: cite paragraph IDs | flag Code Edition year | verify MDMT per UCS-66

[ACTIVE PROJECT]
Project: {id} | Client: {name} | Code: VIII-1 {edition}

[SESSION HINT]
Last: {topic} | Pending: {items}
```

**Promotion rule:** Only updated by sleep-time consolidation. Content must score H > 0.7 heat score or be accessed ≥5 times in 3 days.

---

### WARM Tier — Fast Retrieval (500–1,500 tokens/query)

**What lives here:** Session summaries, frequently-accessed ASME paragraphs, active project notes, recent personal facts.

**How it works:** 256-dimensional vector search via LanceDB (`chunks_warm` table) + SQLite FTS5 keyword search. Results fused via RRF.

**Promotion rule:** Cold content promoted to warm when heat score H > 0.4 or accessed ≥2 times in 7 days.

---

### COLD Tier — Deep Storage (2,000–4,000 tokens on explicit retrieval)

**What lives here:** Full ASME BPVC corpus (Sections I, II, V, VIII-1, IX), episodic conversation logs, archived project notes, rarely-accessed personal facts.

**How it works:** 768-dimensional vector search via LanceDB (`chunks_cold` table) + FTS5 + PPR graph traversal across ASME cross-reference edges. All three signals fused via RRF, then reranked by BGE-Reranker-v2-M3.

**ASME content rule:** All normative ASME content has `no_forget=True` — it never gets deleted, only moves between warm and cold.

---

## Retrieval Pipeline (4 Stages)

### Stage 1: Broad Recall (parallel)

Three signals fire simultaneously:

| Signal | Implementation | Returns |
|--------|---------------|---------|
| **BM25** | SQLite FTS5 (`chunks_fts`) | Top-100 by keyword rank |
| **Dense** | LanceDB cosine similarity (768d) | Top-100 by vector similarity |
| **PPR** | scipy CSR sparse PageRank on `graph_edges` table | Top-50 by graph activation |

PPR seeds are identified by embedding similarity between the query and ASME paragraph IDs. Activation spreads through the cross-reference graph (e.g., a query about UG-22 activates UW-11, UCS-66, and QW-200 via graph edges).

### Stage 2: Fusion

Reciprocal Rank Fusion merges the three result lists:

```
RRF_score(doc) = Σ [ 1 / (k + rank(doc)) ]   where k = 60
```

k=60 prevents any single high-rank result from dominating. Works on ranks, not raw scores — no normalization needed.

### Stage 3: Rerank

BGE-Reranker-v2-M3 (`BAAI/bge-reranker-v2-m3`) scores the top-30 RRF candidates as query-passage pairs. Cross-encoder architecture — sees both query and passage together, produces a single relevance logit. Runs on CPU (~500ms for 30 candidates).

Falls back to RRF-ordered results if model download fails.

### Stage 4: Return

Top-K results (default 10) returned with: `chunk_id`, `paragraph_id`, `content`, `score`, `source` (bm25/dense/ppr/reranked).

**Total latency: ~300–600ms on CPU.**

---

## Embeddings: Matryoshka Architecture

All embeddings use `nomic-ai/nomic-embed-text-v1.5` with Matryoshka Representation Learning. This means a single model produces embeddings at multiple dimensions, where smaller dimensions are nested inside larger ones.

| Tier | Dimension | Use | Quality vs 768d |
|------|-----------|-----|-----------------|
| HOT | 64d | Quick relevance check | ~85–88% |
| WARM | 256d | Session/recent retrieval | ~95–97% |
| COLD | 768d | Full ASME corpus search | 100% (baseline) |

**Required:** Apply `F.layer_norm` before dimension truncation (nomic-specific requirement for Matryoshka to work correctly).

Prefixes used per nomic spec:
- Query: `"search_query: {text}"`
- Documents: `"search_document: {text}"`

---

## ASME  Integration

### Paragraph Structure

The ASME BPVC uses a hierarchical numbering system that maps cleanly to the vault structure:

```
Section VIII Division 1
├── Subsection A
│   └── Part UG (General Requirements)
│       ├── UG-22 (Loadings)
│       ├── UG-27 (Thickness of Shells Under Internal Pressure)
│       └── ...
├── Subsection B
│   └── Part UW (Welding)
│       ├── UW-11 (Radiographic and Ultrasonic Examination)
│       └── UW-12 (Joint Efficiencies)
└── Subsection C
    └── Part UCS (Carbon and Low Alloy Steel)
        └── UCS-66 (Impact Test Exemptions)

Section IX (Welding Qualifications)
└── Part QW
    ├── QW-200 (General)
    └── QW-250 (Variable Tables — one per welding process)
```

### Ingestion Pipeline

When ASME text arrives (plain text, pre-split by paragraph):

1. **Parse** — `asme_parser.py` extracts paragraph IDs via regex (`[A-Z]{1,4}-\d{1,4}`), identifies cross-references (`see UW-11`, `per UCS-66`), builds hierarchy metadata
2. **Embed** — `embedder.py` generates 64d, 256d, and 768d vectors using nomic-embed-text-v1.5
3. **Store** — chunk stored in:
   - `asme_chunks` SQLite table (metadata + bi-temporal versioning)
   - `chunks_fts` FTS5 index (keyword search)
   - `chunks_cold` LanceDB table (768d vectors)
   - `graph_edges` SQLite table (cross-reference edges for PPR)
4. **Flag** — `no_forget=True`, `content_type="normative"` set automatically

### Bi-Temporal Versioning

Every chunk knows which code edition it belongs to:

```sql
chunk_id    = "VIII-1_UG-22_a_2025"
paragraph_id = "UG-22(a)"
edition_year = 2025
valid_from   = 2025-07-01
valid_to     = NULL  -- NULL means current
```

Querying "what did UG-22 say in 2021 vs 2023" = `WHERE paragraph_id='UG-22' AND edition_year IN (2021, 2023)`.

---

## Vault Structure

```
leo_trident/vault/
├── _system/
│   ├── hot.json              # 200-token hot context (auto-generated)
│   ├── anchors.json          # LOCKED facts with SHA-256 hashes
│   ├── consolidation_log.json # Sleep-time audit trail
│   └── tier_registry/        # SQLite metadata lives here
├── Personal/
│   ├── Profile.md            # Brett's identity, certs, preferences
│   └── Projects/             # Active and archived project notes
├── ASME-BPVC/
│   ├── _index.md             # Edition tracking
│   ├── Section-VIII-Div1/
│   │   ├── Subsection-A/Part-UG/   # UG-22.md, UG-27.md, etc.
│   │   ├── Subsection-B/Part-UW/
│   │   └── Subsection-C/
│   └── Section-IX/
│       ├── Part-QW/
│       └── Part-QB/
├── Summaries/                # RAPTOR-generated hierarchical summaries
└── Sessions/                 # Conversation logs (auto-generated)
```

File watcher (`file_watcher.py`) monitors `vault/` for `.md` changes. On change: debounce 500ms → re-parse → re-embed → update LanceDB + FTS5 index.

---

## Heat Score & Forgetting Curve

### Heat Score

Controls tier promotion. Higher = more likely to move to warmer tier.

```
H(m) = α·N_visit + β·R_recency + γ·L_interaction
```

Where:
- `N_visit` = access count
- `R_recency` = `1 / (1 + days_since_last_access)`
- `L_interaction` = average interaction depth (how many turns used this memory)
- Weights: α=0.5, β=0.3, γ=0.2

### FSRS Forgetting Curve

Controls demotion. Based on spaced-repetition research.

```
R(t) = (1 + t/S)^(-0.5)
```

Where `t` = days since last access, `S` = stability (grows with repeated access, starts at 1 day).

| Transition | Trigger |
|---|---|
| Cold → Warm | H > 0.4 or ≥2 accesses in 7 days |
| Warm → Hot | H > 0.7 or ≥5 accesses in 3 days |
| Hot → Warm | R(t) < 0.7 or 7 days inactive |
| Warm → Cold | R(t) < 0.5 or 30 days inactive |

ASME normative content: `no_forget=True` — exempt from R(t) demotion, never deleted.

---

## Locked Anchors

`vault/_system/anchors.json` contains facts that **cannot be modified by sleep-time consolidation**. Each anchor has a SHA-256 hash of its content. If the consolidator tries to overwrite an anchor, the hash won't match and the change is rejected.

Current anchors:
- Core identity (Brett's name, role, location)
- ASME safety pins (never waive UG-99 hydro, always cite paragraph IDs, etc.)

To add or change an anchor, Brett must explicitly approve it — it's a deliberate human-in-the-loop gate.

---

## Sleep-Time Consolidation

Runs asynchronously (not during conversations). Uses Claude Sonnet via Abacus.AI API.

**Triggers:**
- 30 minutes of inactivity
- Nightly at 2 AM EST
- Warm tier exceeds 50K tokens
- Explicit `!evolve` or "consolidate" command

**What it does:**
1. **Fact extraction** — reads conversation logs, classifies each new fact as ADD / UPDATE / DELETE / NOOP against existing memories
2. **Tier management** — recomputes heat scores and R(t), promotes/demotes content
3. **Forward prediction** — if Brett has been working on a vessel project, pre-loads relevant UG-22, UW-12, and Section IX welding quals into warm tier
4. **Hot recompression** — regenerates the 200-token hot context from warm candidates

**Race condition prevention:** SQLite WAL mode. Conversational agent uses read-only connection (`PRAGMA query_only=ON`). Consolidator uses write connection. Readers never block writers.

---

## File Reference

| File | Purpose |
|------|---------|
| `src/api.py` | Main interface — `LeoTrident.query()`, `LeoTrident.ingest_text()` |
| `src/schema.py` | SQLite table definitions, WAL mode, FTS5 indexes |
| `src/ingest/asme_parser.py` | ASME paragraph-boundary parser, cross-reference extractor |
| `src/ingest/embedder.py` | nomic-embed-text-v1.5, Matryoshka 64/256/768d |
| `src/ingest/file_watcher.py` | watchdog vault sync → LanceDB + FTS5 |
| `src/retrieval/bm25.py` | SQLite FTS5 BM25 keyword search |
| `src/retrieval/dense.py` | LanceDB cosine similarity vector search |
| `src/retrieval/ppr.py` | scipy CSR PageRank on ASME cross-reference graph |
| `src/retrieval/fusion.py` | Reciprocal Rank Fusion (k=60) |
| `src/retrieval/reranker.py` | BGE-Reranker-v2-M3, CPU, fallback to RRF order |
| `src/memory/tier_manager.py` | Heat scores, FSRS forgetting curve, tier transitions |
| `scripts/init_db.py` | Initialize SQLite schema + LanceDB tables |
| `scripts/ingest_asme.py` | CLI for batch ASME corpus ingestion |
| `vault/_system/hot.json` | 200-token hot context (always injected) |
| `vault/_system/anchors.json` | SHA-256 locked identity + safety pins |

---

## Setup

```bash
# Install dependencies
pip install lancedb pyarrow sentence-transformers FlagEmbedding \
            fast-pagerank scipy numpy watchdog anthropic httpx pytest

# Initialize databases
cd /home/ubuntu/leo_trident
python3 scripts/init_db.py

# Run tests
python3 -m pytest tests/ -v

# Ingest ASME text (when available)
python3 scripts/ingest_asme.py --input /path/to/section-viii-div1/ --section VIII-1

# Query
python3 -c "
from src.api import LeoTrident
lt = LeoTrident()
results = lt.query('minimum wall thickness formula for cylindrical shells', top_k=5)
for r in results:
    print(r['paragraph_id'], r['score'], r['content'][:100])
"
```

---

## Migration Path to Mac Mini M5

The system is designed to swap cloud API calls for local Ollama with zero architecture changes:

| Component | Now (Abacus) | Mac Mini M5 |
|---|---|---|
| Embedding model | sentence-transformers (CPU) | nomic-embed-text via Ollama |
| Sleep LLM | Claude Sonnet via Abacus API | Qwen 2.5 14B Q5_K_M via Ollama |
| BGE Reranker | CPU FP32 | CPU or Metal GPU |
| Main agent LLM | Claude Sonnet via Abacus API | Keep on Abacus or local |
| Storage | LanceDB + SQLite on disk | Same — no changes |

---

## Current Status

| Phase | Status |
|-------|--------|
| Phase 0: Foundation (LanceDB + SQLite schema) | ✅ Complete |
| Phase 1: Personal knowledge migration | ✅ Complete |
| Phase 2: Full retrieval pipeline | ✅ Complete — 6/6 tests passing |
| Phase 3: ASME corpus ingestion | ⏳ Waiting on text (VIII-1 + IX first) |
| Phase 4: Sleep-time consolidation v2 | ✅ Complete |
| Phase 5: Eval + hardening | ✅ Complete — 34/34 tests passing |

---

## Phase 5: Eval Metrics

### Anchor Integrity (April 11, 2026)
- **SHA-256 violations:** 0
- **Rules checked:** 9 (4 NEVER + 5 ALWAYS + core_facts)
- **Status:** ✅ All anchors verified

### Consolidation Test Coverage
| Test Suite | Tests | Status |
|---|---|---|
| `test_consolidation.py` | 10 | ✅ Pass |
| `test_drift_monitor.py` | 8 | ✅ Pass |
| `test_eval_framework.py` | 10 | ✅ Pass |
| `test_e2e.py` | 6 | ✅ Pass |
| **Total** | **34** | **✅ All pass** |

### ASME Eval Framework
- 10 question eval bank covering: single-hop factual (7), multi-hop (2), abstention (1)
- Metrics: Recall@5, Recall@10, MRR
- *Full metrics pending real ASME corpus ingestion (Phase 3)*

---

*Built April 10–11, 2026*
