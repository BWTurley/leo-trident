#!/usr/bin/env python3
"""
Leo Trident -- Eval Runner

Ingests the synthetic corpus, runs the eval question bank, prints metrics,
and writes JSON results to data/eval_runs/.

Usage:
    python3 scripts/run_eval.py                         # full run
    python3 scripts/run_eval.py --no-rerank             # skip BGE rerank
    python3 scripts/run_eval.py --no-judge              # skip relevance judge
    python3 scripts/run_eval.py --category multi_hop    # run one category only
    python3 scripts/run_eval.py --compare {prev}.json   # diff vs prior run
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CORPUS_DIR = ROOT / "tests" / "fixtures" / "synthetic_corpus"
METADATA_PATH = CORPUS_DIR / "_metadata.json"
QUESTIONS_PATH = ROOT / "tests" / "fixtures" / "eval_questions.json"
OUTPUT_DIR = ROOT / "data" / "eval_runs"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_eval_instance(
    tmp_dir: str | Path,
    corpus_dir: str | Path | None = None,
) -> "LeoTrident":
    """
    Create a fully-initialized LeoTrident instance in *tmp_dir* with the
    synthetic corpus ingested.  Returns the ready-to-query instance.

    Exported so regression tests can call this directly.
    """
    tmp_dir = Path(tmp_dir)
    corpus_path = Path(corpus_dir) if corpus_dir else CORPUS_DIR

    from src.api import LeoTrident
    from src.schema import init_schema

    lt = LeoTrident(base_path=str(tmp_dir))
    lt.data_path = tmp_dir / "data"
    lt.db_path = lt.data_path / "leo_trident.db"
    lt.lance_path = lt.data_path / "lancedb"
    lt.vault_path = tmp_dir / "vault"
    lt.data_path.mkdir(parents=True, exist_ok=True)

    # SQLite schema
    init_schema(lt.db_path)

    # LanceDB tables (mirror test_e2e.py)
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(str(lt.lance_path))
    base_fields = [
        pa.field("chunk_id", pa.string()),
        pa.field("paragraph_id", pa.string()),
        pa.field("section", pa.string()),
        pa.field("content_type", pa.string()),
        pa.field("content", pa.string()),
        pa.field("no_forget", pa.bool_()),
        pa.field("tier", pa.string()),
        pa.field("edition_year", pa.int32()),
        pa.field("created_at", pa.string()),
    ]
    for name, dim in [("chunks_cold", 768), ("chunks_warm", 256)]:
        schema = pa.schema(
            base_fields + [pa.field("vector", pa.list_(pa.float32(), dim))]
        )
        db.create_table(name, schema=schema, mode="create")

    # Ingest synthetic corpus
    from src.ingest.asme_parser import ASMEParser

    parser = ASMEParser(edition_year=2025)

    txt_files = sorted(corpus_path.glob("*.txt"))
    for fp in txt_files:
        para_id = fp.stem  # e.g. "UG-22"
        text = fp.read_text(encoding="utf-8").strip()
        if not text:
            continue
        section, part, _subsection, _part_title = parser.resolve_hierarchy(para_id)
        lt.ingest_text(
            text=text,
            paragraph_id=para_id,
            section=section,
            part=part,
            edition_year=2025,
        )

    return lt


def _count_edges(lt: "LeoTrident") -> tuple[int, dict[str, int]]:
    """Return (total_edges, {reference_type: count})."""
    from src.schema import create_connection

    conn = create_connection(lt.db_path, read_only=True)
    try:
        total = conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
        rows = conn.execute(
            "SELECT reference_type, COUNT(*) AS cnt FROM graph_edges GROUP BY reference_type"
        ).fetchall()
        dist = {r["reference_type"]: r["cnt"] for r in rows}
    finally:
        conn.close()
    return total, dist


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------

def _eval_question(
    lt: "LeoTrident",
    q: dict,
    use_rerank: bool,
    use_judge: bool,
) -> dict:
    """
    Run a single question through the pipeline and return per-question metrics.
    """
    expected: list[str] = q.get("expected_paragraph_ids", [])
    forbidden: list[str] = q.get("forbidden_paragraph_ids", [])
    required_refs: list[str] = q.get("required_references", [])
    category: str = q.get("category", "unknown")
    query_text: str = q["query"]

    t0 = time.perf_counter()
    try:
        results = lt.query(
            query_text,
            top_k=10,
            use_rerank=use_rerank,
            use_relevance_judge=use_judge,
        )
        error = None
    except Exception as exc:
        results = []
        error = str(exc)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    returned_pids = [r.get("paragraph_id", "") for r in results]

    # hit@k
    hit5 = any(pid in expected for pid in returned_pids[:5]) if expected else False
    hit10 = any(pid in expected for pid in returned_pids[:10]) if expected else False

    # reciprocal rank
    rr = 0.0
    if expected:
        for rank, pid in enumerate(returned_pids, 1):
            if pid in expected:
                rr = 1.0 / rank
                break

    # abstention check (all scores below 0.3 or empty)
    abstention_correct = False
    if category == "abstention":
        if not results:
            abstention_correct = True
        else:
            abstention_correct = all(
                r.get("score", 1.0) < 0.3 for r in results
            )

    # relevance accuracy (only when judge is active and category matches)
    required_hit = None
    forbidden_avoid = None
    fallback_used = False

    if use_judge and category == "reference_relevance":
        # Flatten all references from all returned results
        all_refs: list[dict] = []
        for r in results:
            all_refs.extend(r.get("references", []))

        ref_map: dict[str, str] = {}  # paragraph_id -> relevance
        ref_reasons: list[str] = []
        for ref in all_refs:
            pid = ref.get("paragraph_id", "")
            rel = ref.get("relevance", "")
            reason = ref.get("reason", "")
            if pid:
                ref_map[pid] = rel
            ref_reasons.append(reason)

        # required_hit: fraction of required_references marked 'required'
        if required_refs:
            hits = sum(1 for pid in required_refs if ref_map.get(pid) == "required")
            required_hit = hits / len(required_refs)

        # forbidden_avoid: fraction of forbidden_paragraph_ids NOT marked 'required'
        if forbidden:
            avoided = sum(
                1 for pid in forbidden
                if ref_map.get(pid, "irrelevant") in ("optional", "irrelevant")
            )
            forbidden_avoid = avoided / len(forbidden)

        # fallback detection
        fallback_used = any("fallback" in reason.lower() for reason in ref_reasons)

    entry = {
        "id": q.get("id", ""),
        "category": category,
        "query": query_text,
        "expected": expected,
        "returned_pids": returned_pids,
        "hit5": hit5,
        "hit10": hit10,
        "rr": rr,
        "latency_ms": round(elapsed_ms, 1),
        "error": error,
    }
    if category == "abstention":
        entry["abstention_correct"] = abstention_correct
    if required_hit is not None:
        entry["required_hit"] = required_hit
    if forbidden_avoid is not None:
        entry["forbidden_avoid"] = forbidden_avoid
    if use_judge and category == "reference_relevance":
        entry["fallback_used"] = fallback_used

    return entry


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def _aggregate(per_question: list[dict]) -> dict:
    """Compute aggregate metrics from the per-question list."""
    if not per_question:
        return {}

    # Overall (exclude abstention from hit/rr aggregation)
    non_abstention = [q for q in per_question if q["category"] != "abstention"]
    hits5 = [q["hit5"] for q in non_abstention if "hit5" in q]
    hits10 = [q["hit10"] for q in non_abstention if "hit10" in q]
    rrs = [q["rr"] for q in non_abstention if "rr" in q]
    latencies = [q["latency_ms"] for q in per_question if q.get("latency_ms")]

    recall5 = statistics.mean(hits5) if hits5 else 0.0
    recall10 = statistics.mean(hits10) if hits10 else 0.0
    mrr = statistics.mean(rrs) if rrs else 0.0

    # Per category
    cats: dict[str, list[dict]] = {}
    for q in per_question:
        cats.setdefault(q["category"], []).append(q)

    by_category: dict[str, dict] = {}
    for cat, qs in sorted(cats.items()):
        if cat == "abstention":
            # Use abstention_correct rate instead of hit@k
            correct = [q.get("abstention_correct", False) for q in qs]
            rate = round(statistics.mean(correct), 3) if correct else 0.0
            by_category[cat] = {
                "n": len(qs),
                "recall_at_5": rate,
                "recall_at_10": rate,
                "mrr": None,  # MRR not applicable for abstention
            }
        else:
            c5 = [q["hit5"] for q in qs if "hit5" in q]
            c10 = [q["hit10"] for q in qs if "hit10" in q]
            crr = [q["rr"] for q in qs if "rr" in q]
            by_category[cat] = {
                "n": len(qs),
                "recall_at_5": round(statistics.mean(c5), 3) if c5 else 0.0,
                "recall_at_10": round(statistics.mean(c10), 3) if c10 else 0.0,
                "mrr": round(statistics.mean(crr), 3) if crr else 0.0,
            }

    # Relevance judge metrics
    req_hits = [q["required_hit"] for q in per_question if q.get("required_hit") is not None]
    forb_avoids = [q["forbidden_avoid"] for q in per_question if q.get("forbidden_avoid") is not None]
    fallbacks = [q.get("fallback_used", False) for q in per_question if "fallback_used" in q]

    relevance = {
        "required_hit": round(statistics.mean(req_hits), 3) if req_hits else None,
        "forbidden_avoid": round(statistics.mean(forb_avoids), 3) if forb_avoids else None,
        "fallback_rate": round(sum(fallbacks) / len(fallbacks), 3) if fallbacks else None,
    }

    # Latency
    lat_median = round(statistics.median(latencies), 1) if latencies else 0.0
    lat_sorted = sorted(latencies)
    p95_idx = int(len(lat_sorted) * 0.95)
    lat_p95 = round(lat_sorted[min(p95_idx, len(lat_sorted) - 1)], 1) if lat_sorted else 0.0

    errors = sum(1 for q in per_question if q.get("error"))

    return {
        "recall_at_5": round(recall5, 3),
        "recall_at_10": round(recall10, 3),
        "mrr": round(mrr, 3),
        "by_category": by_category,
        "relevance": relevance,
        "latency_ms": {"median": lat_median, "p95": lat_p95},
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Core eval runner (importable)
# ---------------------------------------------------------------------------

def run_eval(
    lt: "LeoTrident",
    questions: list[dict],
    use_rerank: bool = True,
    use_judge: bool = True,
) -> dict:
    """
    Run all *questions* against *lt* and return the full results dict.
    Exported for regression tests.
    """
    per_question: list[dict] = []
    total = len(questions)

    for i, q in enumerate(questions, 1):
        qid = q.get("id", f"q{i}")
        sys.stdout.write(f"\r  [{i}/{total}] {qid}...")
        sys.stdout.flush()
        entry = _eval_question(lt, q, use_rerank=use_rerank, use_judge=use_judge)
        per_question.append(entry)

    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()

    agg = _aggregate(per_question)

    # Corpus stats
    n_paragraphs = len(list(CORPUS_DIR.glob("*.txt")))
    edge_count, edge_dist = _count_edges(lt)

    run_at = datetime.now(timezone.utc).isoformat()
    llm_mode = os.environ.get("LEO_LLM_MODE", "local")

    return {
        "run_at": run_at,
        "config": {
            "rerank": use_rerank,
            "judge": use_judge,
            "llm_mode": llm_mode,
        },
        "corpus": {
            "paragraphs": n_paragraphs,
            "edges": edge_count,
            "edge_type_dist": edge_dist,
        },
        "aggregate": agg,
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _try_rich_table(results: dict) -> bool:
    """Attempt to print with rich. Returns True if successful."""
    try:
        from rich.console import Console
        from rich.table import Table

        import io, sys as _sys
        # Force UTF-8 on Windows to avoid cp1252 encoding errors with rich
        if _sys.platform == "win32":
            _sys.stdout = io.TextIOWrapper(
                _sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
        console = Console(force_terminal=True)
    except ImportError:
        return False

    agg = results["aggregate"]
    cfg = results["config"]
    corpus = results["corpus"]
    pq = results["per_question"]
    run_at = results["run_at"]

    total = len(pq)
    skipped = results.get("skipped_count", 0)
    run_count = total

    console.print()
    console.rule("[bold]Leo Trident Eval[/bold]", style="bright_cyan")
    console.print(f"  Run:        {run_at}")
    console.print(
        f"  Corpus:     {corpus['paragraphs']} paragraphs, "
        f"{corpus['edges']} typed edges"
    )
    console.print(
        f"  Questions:  {total + skipped} total ({run_count} run, {skipped} skipped)"
    )
    console.print(
        f"  Config:     rerank={cfg['rerank']}, judge={cfg['judge']}, "
        f"mode={cfg['llm_mode']}"
    )
    console.print()

    # Per-category table
    table = Table(title="Per-category metrics", show_lines=True)
    table.add_column("Category", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("R@5", justify="right")
    table.add_column("R@10", justify="right")
    table.add_column("MRR", justify="right")

    for cat, m in sorted(agg["by_category"].items()):
        mrr_str = "N/A" if m["mrr"] is None else f"{m['mrr']:.3f}"
        table.add_row(
            cat,
            str(m["n"]),
            f"{m['recall_at_5']:.3f}",
            f"{m['recall_at_10']:.3f}",
            mrr_str,
        )

    # Overall row
    table.add_row(
        "[bold]OVERALL[/bold]",
        str(run_count),
        f"[bold]{agg['recall_at_5']:.3f}[/bold]",
        f"[bold]{agg['recall_at_10']:.3f}[/bold]",
        f"[bold]{agg['mrr']:.3f}[/bold]",
    )
    console.print(table)

    # Relevance judge block
    rel = agg.get("relevance", {})
    if any(v is not None for v in rel.values()):
        console.print()
        console.print("  [bold]Relevance judge:[/bold]")
        if rel.get("required_hit") is not None:
            console.print(f"    accuracy (required):    {rel['required_hit']:.2f}")
        if rel.get("forbidden_avoid") is not None:
            console.print(f"    accuracy (irrelevant):  {rel['forbidden_avoid']:.2f}")
        if rel.get("fallback_rate") is not None:
            console.print(f"    fallback rate:          {rel['fallback_rate']:.2f}")

    # Latency
    lat = agg.get("latency_ms", {})
    console.print()
    console.print("  [bold]Latency (median / p95):[/bold]")
    console.print(
        f"    retrieval:              {lat.get('median', 0):.0f}ms / "
        f"{lat.get('p95', 0):.0f}ms"
    )

    # Errors
    console.print()
    console.print(f"  Errors: {agg.get('errors', 0)}")
    console.rule(style="bright_cyan")
    console.print()

    return True


def _print_plain(results: dict) -> None:
    """Fallback plain-text printer."""
    agg = results["aggregate"]
    cfg = results["config"]
    corpus = results["corpus"]
    pq = results["per_question"]
    run_at = results["run_at"]

    total = len(pq)
    skipped = results.get("skipped_count", 0)
    run_count = total

    sep = "=" * 55
    print()
    print(sep)
    print(f"  Leo Trident Eval -- {run_at}")
    print(sep)
    print(
        f"  Corpus:     {corpus['paragraphs']} paragraphs, "
        f"{corpus['edges']} typed edges"
    )
    print(
        f"  Questions:  {total + skipped} total ({run_count} run, {skipped} skipped)"
    )
    print(
        f"  Config:     rerank={cfg['rerank']}, judge={cfg['judge']}, "
        f"mode={cfg['llm_mode']}"
    )
    print()
    print("  Per-category metrics:")

    # Header
    hdr = (
        f"  {'Category':<24} {'N':>4} {'R@5':>8} {'R@10':>9} {'MRR':>7}"
    )
    print(hdr)
    print("  " + "-" * len(hdr.strip()))

    for cat, m in sorted(agg["by_category"].items()):
        mrr_str = "    N/A" if m["mrr"] is None else f"{m['mrr']:>7.3f}"
        print(
            f"  {cat:<24} {m['n']:>4} "
            f"{m['recall_at_5']:>8.3f} {m['recall_at_10']:>9.3f} "
            f"{mrr_str}"
        )

    print(
        f"  {'OVERALL':<24} {run_count:>4} "
        f"{agg['recall_at_5']:>8.3f} {agg['recall_at_10']:>9.3f} "
        f"{agg['mrr']:>7.3f}"
    )

    # Relevance
    rel = agg.get("relevance", {})
    if any(v is not None for v in rel.values()):
        print()
        print("  Relevance judge:")
        if rel.get("required_hit") is not None:
            print(f"    accuracy (required):    {rel['required_hit']:.2f}")
        if rel.get("forbidden_avoid") is not None:
            print(f"    accuracy (irrelevant):  {rel['forbidden_avoid']:.2f}")
        if rel.get("fallback_rate") is not None:
            print(f"    fallback rate:          {rel['fallback_rate']:.2f}")

    lat = agg.get("latency_ms", {})
    print()
    print("  Latency (median / p95):")
    print(
        f"    retrieval:              {lat.get('median', 0):.0f}ms / "
        f"{lat.get('p95', 0):.0f}ms"
    )

    print()
    print(f"  Errors: {agg.get('errors', 0)}")
    print(sep)
    print()


def print_results(results: dict) -> None:
    if not _try_rich_table(results):
        _print_plain(results)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def print_comparison(current: dict, prior_path: str) -> None:
    """Load a prior run and print a delta table."""
    prior = json.loads(Path(prior_path).read_text(encoding="utf-8"))
    prior_agg = prior.get("aggregate", {})
    cur_agg = current.get("aggregate", {})

    prior_ts = prior.get("run_at", "unknown")
    print(f"\n  Comparison vs {prior_ts}:")
    print(f"  {'category':<24} {'metric':<10} {'old':>8} -> {'new':>8}  (delta)")
    print("  " + "-" * 70)

    regressions: list[str] = []

    # Overall
    for metric in ("recall_at_5", "recall_at_10", "mrr"):
        old_val = prior_agg.get(metric, 0.0)
        new_val = cur_agg.get(metric, 0.0)
        delta = new_val - old_val
        flag = " ***" if delta < -0.03 else ""
        print(
            f"  {'OVERALL':<24} {metric:<10} {old_val:>8.3f} -> "
            f"{new_val:>8.3f}  ({delta:+.3f}){flag}"
        )
        if delta < -0.03:
            regressions.append(f"OVERALL/{metric}: {delta:+.3f}")

    # Per category
    all_cats = set(
        list(prior_agg.get("by_category", {}).keys())
        + list(cur_agg.get("by_category", {}).keys())
    )
    for cat in sorted(all_cats):
        prior_cat = prior_agg.get("by_category", {}).get(cat, {})
        cur_cat = cur_agg.get("by_category", {}).get(cat, {})
        for metric in ("recall_at_5", "recall_at_10", "mrr"):
            old_val = prior_cat.get(metric, 0.0)
            new_val = cur_cat.get(metric, 0.0)
            delta = new_val - old_val
            flag = " ***" if delta < -0.03 else ""
            print(
                f"  {cat:<24} {metric:<10} {old_val:>8.3f} -> "
                f"{new_val:>8.3f}  ({delta:+.3f}){flag}"
            )
            if delta < -0.03:
                regressions.append(f"{cat}/{metric}: {delta:+.3f}")

    if regressions:
        print()
        print("  REGRESSIONS (>3pp):")
        for r in regressions:
            print(f"    - {r}")
    else:
        print("\n  No regressions detected.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leo Trident eval runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip BGE rerank stage",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip relevance judge stage",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Run only questions matching this category",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a prior eval JSON to diff against",
    )
    args = parser.parse_args()

    use_rerank = not args.no_rerank
    use_judge = not args.no_judge

    # Load questions
    if not QUESTIONS_PATH.exists():
        print(f"ERROR: Questions file not found: {QUESTIONS_PATH}", file=sys.stderr)
        sys.exit(1)

    questions: list[dict] = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    # Filter
    skipped = 0
    filtered: list[dict] = []
    for q in questions:
        if q.get("skip", False):
            skipped += 1
            continue
        if args.category and q.get("category") != args.category:
            skipped += 1
            continue
        filtered.append(q)

    if not filtered:
        print("No questions to run after filtering.", file=sys.stderr)
        sys.exit(1)

    print(f"  Loading {len(filtered)} questions ({skipped} skipped)...")

    # Setup temp instance and ingest corpus
    tmp_dir = tempfile.mkdtemp(prefix="leo_eval_")
    print(f"  Temp dir: {tmp_dir}")
    print("  Setting up eval instance and ingesting corpus...")

    try:
        lt = setup_eval_instance(tmp_dir)
        print("  Corpus ingested. Running eval...")

        results = run_eval(lt, filtered, use_rerank=use_rerank, use_judge=use_judge)
        results["skipped_count"] = skipped

        # Print
        print_results(results)

        # Write output
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = OUTPUT_DIR / f"{ts}.json"
        out_path.write_text(
            json.dumps(results, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"  Results written to: {out_path}")

        # Compare if requested
        if args.compare:
            compare_path = Path(args.compare)
            if not compare_path.exists():
                # Try inside OUTPUT_DIR
                compare_path = OUTPUT_DIR / args.compare
            if compare_path.exists():
                print_comparison(results, str(compare_path))
            else:
                print(
                    f"  WARNING: comparison file not found: {args.compare}",
                    file=sys.stderr,
                )

    finally:
        # Clean up temp dir
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
