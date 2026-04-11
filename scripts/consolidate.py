#!/usr/bin/env python3
"""
Leo Trident — Manual Consolidation CLI

Usage:
    python3 scripts/consolidate.py                  # full run
    python3 scripts/consolidate.py --dry-run        # show changes, don't write
    python3 scripts/consolidate.py --check-anchors  # only drift_check
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from project root or scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("consolidate")


def main():
    parser = argparse.ArgumentParser(description="Leo Trident sleep-time consolidation")
    parser.add_argument("--dry-run", action="store_true", help="Show changes, don't write")
    parser.add_argument("--check-anchors", action="store_true", help="Only run anchor drift check")
    args = parser.parse_args()

    from src.memory.consolidator import SleepTimeConsolidator

    consolidator = SleepTimeConsolidator()

    if args.check_anchors:
        print("\n=== ANCHOR DRIFT CHECK ===")
        result = consolidator.drift_check()
        print(json.dumps(result, indent=2))
        if result["ok"]:
            print("\n✅ All anchors verified — no drift detected.")
            sys.exit(0)
        else:
            print(f"\n❌ VIOLATIONS FOUND: {len(result['violations'])}")
            for v in result["violations"]:
                print(f"  Rule: {v['rule']}")
                print(f"    stored:   {v['stored']}")
                print(f"    computed: {v['computed']}")
            sys.exit(1)

    # Full consolidation run
    if args.dry_run:
        print("\n=== DRY-RUN MODE — no writes will be performed ===")
    else:
        print("\n=== SLEEP-TIME CONSOLIDATION ===")

    summary = consolidator.run(dry_run=args.dry_run)

    print(f"\nRun at:          {summary['run_at']}")
    print(f"Dry-run:         {summary['dry_run']}")
    print(f"Facts extracted: {len(summary.get('facts_extracted', []))}")
    print(f"Tier changes:    {len(summary.get('tier_changes', []))}")
    print(f"FWD predictions: {len(summary.get('forward_predictions', []))}")
    print(f"Hot recompressed:{summary.get('hot_recompressed', False)}")

    drift = summary.get("drift_check", {})
    if drift:
        status = "✅ OK" if drift.get("ok") else f"❌ {len(drift.get('violations',[]))} violations"
        print(f"Anchor drift:    {status}")

    if summary.get("errors"):
        print(f"\nErrors ({len(summary['errors'])}):")
        for e in summary["errors"]:
            print(f"  • {e}")

    if summary.get("tier_changes"):
        print("\nTier changes:")
        for c in summary["tier_changes"]:
            print(f"  {c['chunk_id']}: {c['from_tier']} → {c['to_tier']}  "
                  f"(heat={c['heat']}, retention={c['retention']})")

    if summary.get("facts_extracted"):
        print("\nFacts extracted:")
        for f in summary["facts_extracted"][:10]:
            print(f"  [{f.get('action','?')}] {f.get('fact','')[:80]}")
        if len(summary["facts_extracted"]) > 10:
            print(f"  ... and {len(summary['facts_extracted'])-10} more")

    print()
    if not drift.get("ok", True):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
