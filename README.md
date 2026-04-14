# Leo Trident — Retrieval Service for Technical Corpora


---

## What This Is For

Leo Trident is a **local-first retrieval and memory system** built for densely cross-referenced technical corpora. It combines vector search, keyword search, and graph traversal into a single pipeline optimized for documents where paragraphs reference each other extensively.

**Primary use case:** ASME Boiler and Pressure Vessel Code (BPVC), where a query about shell thickness (UG-27) must also surface joint efficiency (UW-12), radiographic examination (UW-11), and impact testing (UCS-66) through cross-reference graph edges.

**Generalizes to:** Legal codes, regulatory standards, ISO/ANSI specifications, building codes, military standards — any corpus with hierarchical structure and dense internal cross-references.

It runs entirely on commodity hardware (Mac Mini M5, any Linux box). No cloud vector databases. No graph servers. No dependencies that can't be swapped for local equivalents.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│           Caller (LLM agent, CLI, API, etc.)                │
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
│              Reference Relevance Judge (optional)           │
│                       │                                     │
│               Top-K Results returned to caller              │
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
```

---

## Integration

```python
from src.api import LeoTrident

lt = LeoTrident()
results = lt.query(
    "welding examination requirements for austenitic stainless steel",
    top_k=10,
    use_rerank=True,
    use_relevance_judge=True,
)
for r in results:
    print(f"{r['paragraph_id']} (score={r['score']:.3f})")
    for ref in r.get('references', []):
        if ref['relevance'] == 'required':
            print(f"  → also consult {ref['paragraph_id']}: {ref['reason']}")
```

### Conversation History Search

```python
# Search past conversations
results = lt.search_conversations("UW-51", hours=24)

# Include conversation history in retrieval results
results = lt.query("UW-51 spot RT", include_conversations=True)
```

---

## Three-Tier Memory

Every piece of information lives in exactly one tier based on how often it's accessed.

### HOT Tier — Always On (≤200 tokens)

**What lives here:** Safety-critical retrieval constraints (safety pins).

**How it works:** `vault/_system/hot.json` contains output constraints injected into retrieval context. No retrieval needed — it's always available.

**Format:**
```json
{
  "safety_pins": {
    "never": ["return content contradicting UG-99 hydrostatic test requirements", "..."],
    "always": ["return paragraph IDs with all code references", "..."]
  }
}
```

**Promotion rule:** Only updated by sleep-time consolidation. Content must score H > 0.7 heat score or be accessed ≥5 times in 3 days.

---

### WARM Tier — Fast Retrieval (500–1,500 tokens/query)

**What lives here:** Session summaries, frequently-accessed ASME paragraphs, active project notes.

**How it works:** 256-dimensional vector search via LanceDB (`chunks_warm` table) + SQLite FTS5 keyword search. Results fused via RRF.

**Promotion rule:** Cold content promoted to warm when heat score H > 0.4 or accessed ≥2 times in 7 days.

---

### COLD Tier — Deep Storage (2,000–4,000 tokens on explicit retrieval)

**What lives here:** Full ASME BPVC corpus, episodic conversation logs, archived notes.

**How it works:** 768-dimensional vector search via LanceDB (`chunks_cold` table) + FTS5 + PPR graph traversal across ASME cross-reference edges. All three signals fused via RRF, then reranked by BGE-Reranker-v2-M3.

**ASME content rule:** All normative ASME content has `no_forget=True` — it never gets deleted, only moves between warm and cold.

---

## Retrieval Pipeline (4 Stages)

### Stage 1: Broad Recall (parallel)

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

### Stage 4: Reference Relevance Judgment (optional)

For each reranked result, an LLM judges which of the result's outgoing cross-references are required, optional, or irrelevant for the specific query. Falls back to deterministic reference-type mapping when the LLM is unavailable.

### Stage 5: Return

Top-K results (default 10) returned with: `chunk_id`, `paragraph_id`, `content`, `score`, `source`, and optionally `references`.

**Total latency: ~300–600ms on CPU.**

---

## Embeddings: Matryoshka Architecture

All embeddings use `nomic-ai/nomic-embed-text-v1.5` with Matryoshka Representation Learning. A single model produces embeddings at multiple dimensions, where smaller dimensions are nested inside larger ones.

| Tier | Dimension | Use | Quality vs 768d |
|------|-----------|-----|-----------------|
| HOT | 64d | Quick relevance check | ~85–88% |
| WARM | 256d | Session/recent retrieval | ~95–97% |
| COLD | 768d | Full corpus search | 100% (baseline) |

**Required:** Apply `F.layer_norm` before dimension truncation (nomic-specific requirement for Matryoshka to work correctly).

Prefixes used per nomic spec:
- Query: `"search_query: {text}"`
- Documents: `"search_document: {text}"`

---

## ASME Integration

### Paragraph Structure

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

1. **Parse** — `asme_parser.py` extracts paragraph IDs via regex, identifies cross-references, classifies reference types (mandatory/conditional/informational), builds hierarchy metadata
2. **Embed** — `embedder.py` generates 64d, 256d, and 768d vectors using nomic-embed-text-v1.5
3. **Store** — chunk stored in SQLite (metadata + bi-temporal versioning), FTS5 index, LanceDB tables, and graph_edges
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

---

## Heat Score & Forgetting Curve

### Heat Score

Controls tier promotion. Higher = more likely to move to warmer tier.

```
H(m) = α·N_visit + β·R_recency + γ·L_interaction
```

Where: `N_visit` = access count, `R_recency` = `1 / (1 + days_since_last_access)`, `L_interaction` = average interaction depth. Weights: α=0.5, β=0.3, γ=0.2.

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

## Locked Anchors (Data Integrity)

`vault/_system/anchors.json` contains data-integrity safeguards for safety-critical corpus content. Each anchor has a SHA-256 hash. If the consolidator or any automated process tries to modify an anchor, the hash won't match and the change is rejected.

Current anchors protect:
- **ASME safety pins:** retrieval constraints ensuring safety-critical content (UG-99 hydrostatic test, UW-51 spot RT, PQR essential variables, MDMT per UCS-66) is never omitted or contradicted
- **Corpus scope facts:** what the system's primary domain coverage is

The drift monitor (`src/memory/drift_monitor.py`) verifies anchor integrity on every consolidation run and at health check time.

---

## Sleep-Time Consolidation

Runs asynchronously (not during conversations). Uses configurable LLM backend.

**Triggers:**
- 30 minutes of inactivity
- Nightly at 2 AM EST
- Warm tier exceeds 50K tokens

**What it does:**
1. **Fact extraction** — reads conversation logs, classifies each new fact as ADD / UPDATE / DELETE / NOOP against existing memories
2. **Tier management** — recomputes heat scores and R(t), promotes/demotes content
3. **Forward prediction** — pre-loads related cold-tier chunks into warm tier based on recent topics
4. **Hot recompression** — updates timestamp on safety-pins context
5. **Drift check** — verifies anchor SHA-256 hashes
6. **Metrics** — logs consolidation stats to JSONL

**Race condition prevention:** SQLite WAL mode. Conversational readers use read-only connections. Consolidator uses write connection. Readers never block writers.

---

## Health and Operations

### Health Endpoint

Always-on FastAPI service at `127.0.0.1:8765`:

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | 200 OK if SQLite + LanceDB + anchors + embedder are healthy; 503 otherwise |
| `GET /stats` | Corpus counts, tier distribution, disk usage, last consolidation/backup times |
| `GET /version` | Phase marker, schema migrations, git SHA |

### Backup Job

`scripts/backup.py` runs weekly (Sunday 3 AM via launchd):
- WAL-safe SQLite `.backup()` API
- LanceDB directory copy (append-only, safe to copy live)
- `vault/_system/` snapshot
- `manifest.json` with counts and sizes
- Configurable retention (default 30 days)
- Metrics file pruning (90 days)

### Metrics

JSONL files at `data/metrics/{YYYY-MM-DD}.jsonl`. Instrumented paths:
- `query.latency_ms` — end-to-end retrieval latency with rerank/judge tags
- `judge.fallback` — whether the relevance judge fell back to deterministic mapping
- `consolidation.*` — error count, facts extracted, tier changes per run

### Deployment (macOS)

Four launchd agents in `deploy/macos/`:
- `com.leotrident.consolidate.plist` — nightly 2 AM consolidation
- `com.leotrident.watcher.plist` — always-on vault file watcher
- `com.leotrident.backup.plist` — weekly Sunday 3 AM backup
- `com.leotrident.health.plist` — always-on health endpoint

See `deploy/macos/README.md` for setup instructions.

---

## File Reference

| File | Purpose |
|------|---------|
| `src/api.py` | Main interface — `query()`, `ingest_text()`, `search_conversations()` |
| `src/schema.py` | SQLite table definitions, WAL mode, FTS5 indexes |
| `src/config.py` | Environment variable resolution, path defaults |
| `src/ingest/asme_parser.py` | ASME paragraph-boundary parser, cross-reference extractor, reference type classification |
| `src/ingest/embedder.py` | nomic-embed-text-v1.5, Matryoshka 64/256/768d |
| `src/ingest/file_watcher.py` | watchdog vault sync → LanceDB + FTS5 |
| `src/retrieval/bm25.py` | SQLite FTS5 BM25 keyword search |
| `src/retrieval/dense.py` | LanceDB cosine similarity vector search |
| `src/retrieval/ppr.py` | scipy CSR PageRank on cross-reference graph |
| `src/retrieval/fusion.py` | Reciprocal Rank Fusion (k=60) |
| `src/retrieval/reranker.py` | BGE-Reranker-v2-M3, CPU, fallback to RRF order |
| `src/retrieval/relevance_judge.py` | LLM-based cross-reference relevance classification |
| `src/memory/tier_manager.py` | Heat scores, FSRS forgetting curve, tier transitions |
| `src/memory/consolidator.py` | Sleep-time consolidation pipeline |
| `src/memory/drift_monitor.py` | Embedding drift detection + anchor integrity |
| `src/memory/conversation_logger.py` | Conversation turn logging for FTS5 search |
| `src/service/health.py` | FastAPI health/stats/version endpoints |
| `src/service/metrics.py` | JSONL metrics sink |
| `scripts/backup.py` | Scheduled backup with retention |
| `scripts/run_health.py` | Health endpoint launcher |
| `scripts/init_db.py` | Initialize SQLite schema + LanceDB tables |
| `scripts/ingest_asme.py` | CLI for batch ASME corpus ingestion |

---

## Configuration

All runtime behavior is controlled via environment variables (or a `.env` file in the repo root).

| Variable | Default | Purpose |
|---|---|---|
| `LEO_TRIDENT_HOME` | `~/leo_trident` | Project root |
| `LEO_LLM_MODE` | `cloud` | `cloud` = Abacus Claude, `local` = Ollama |
| `LEO_EMBED_DEVICE` | `cpu` | `cpu`, `mps` (Apple Silicon), or `cuda` |
| `OLLAMA_URL` | `http://localhost:11434` | Used when `LEO_LLM_MODE=local` |
| `LEO_CONSOLIDATION_MODEL` | `qwen2.5:14b-instruct-q5_K_M` | Ollama model for consolidation |
| `ABACUS_API_KEY` | — | Required when `LEO_LLM_MODE=cloud` |

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize databases
python scripts/init_db.py

# Run tests
python -m pytest tests/ -v

# Ingest ASME text (when available)
python scripts/ingest_asme.py --input /path/to/section-viii-div1/ --section VIII-1

# Start health endpoint
python scripts/run_health.py
# → http://127.0.0.1:8765/health
```

---

## Current Status

| Phase | Status |
|-------|--------|
| Phase 0: Foundation (LanceDB + SQLite schema) | Complete |
| Phase 1: Personal knowledge migration | Complete |
| Phase 2: Full retrieval pipeline | Complete |
| Phase 3: ASME corpus ingestion | Waiting on text |
| Phase 4: Sleep-time consolidation v2 | Complete |
| Phase 5: Eval + hardening | Complete |
| Phase 6a: Typed cross-references | Complete |
| Phase 6b: Relevance judge | Complete |
| Phase 7: Eval framework | Complete |
| Phase 8: Operational hardening | Complete |

---

## Troubleshooting

### "Real embedder failed to load" error
You're missing `sentence-transformers` or one of its dependencies (likely `torch` or `transformers`). Either:
- Install the full stack: `pip install sentence-transformers torch transformers einops`
- Or for CI/test only, set `LEO_ALLOW_STUB_EMBEDDER=1` (warning: search results will be random).

### Search returns unrelated results
Check whether you're running with the stub embedder. Look for a `warning: stub_embedder_random_vectors` field on each result, or grep logs for "USING STUB EMBEDDER".

### `git status` shows changes in `vault/_system/` after every consolidation
Those files are runtime state and should be gitignored. If you see them tracked, your `.gitignore` is missing the `vault/_system/` line. Re-run the audit-remediation steps.

### Cloud LLM mode silently uses an old key
If you have `/home/ubuntu/.openclaw/openclaw.json` on your machine, it's used as a fallback when `ABACUS_API_KEY` is empty. Either delete that file or explicitly set `ABACUS_API_KEY` in `.env`.

---

*Built April 2026*
