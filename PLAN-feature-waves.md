# Trident Feature Waves — Execution Plan

**Status:** in-flight (dispatched 2026-04-26 by Leo on Brett's `/GO`)
**Repo:** `~/leo/leo-trident` (`BWTurley/leo-trident`)

## Wave 0 — Foundation (single agent)
- Branch: `leo/foundation-scheduler-notify`
- Adds: `src/scheduler.py` (in-process APScheduler-style), `src/notify.py` (Telegram delivery), extends existing `src/service/metrics.py` with daily-rollup query helpers + `metric_event_with_tags()` ergonomic wrapper.
- Acceptance: importable, scheduler runs heartbeat, Telegram delivery proven, tests pass.

## Wave 1 — Parallel features (3 agents)
- A. `leo/feat-sleep-consolidation-cron` — schedule existing consolidator + summary Telegram
- B. `leo/feat-multimodal-ingest` — PDF + image ingest (Anthropic Haiku for image captions)
- C. `leo/feat-reply-aware-retrieval` — `/query` accepts `reply_context`, weighted RRF re-rank

## Wave 2 — Observability (2 agents)
- D. `leo/feat-drift-telemetry` — golden-query MRR/Recall daily, Telegram alert on regression
- E. `leo/feat-cost-tracking` — per-namespace embed accounting, weekly digest

## Hard rules for every subagent
- Branch from `main`. Push, open PR.
- DO NOT touch `src/api.py` beyond extending one named endpoint.
- Run `ruff check . && pytest -q` before commit; if either fails, push as DRAFT PR.
- All new code has at least one unit test.
- Commit on `BWTurley` GitHub account using existing `gh` CLI auth (admin permission verified).

## Cost ceiling
~$3–6 in subagent tokens total. Auto-merge with `--admin --squash --delete-branch`.
