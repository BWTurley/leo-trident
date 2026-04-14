# Audit Remediation — 2026-04-13

Branch: fix/audit-remediation-2026-04-13

## Fixes applied
- CRITICAL: Removed vault/Personal/Profile.md and vault/_system/*.json from tracking; added templates and bootstrap
- CRITICAL: Git history rewritten to purge Profile.md from all past commits
- HIGH: Stub embedder is now opt-in via LEO_ALLOW_STUB_EMBEDDER
- HIGH: bm25_from_sqlite validates table parameter against allowlist
- HIGH: Legacy /home/ubuntu/.openclaw key fallback marked DEPRECATED with narrowed exception handling
- HIGH: BGE reranker pinned to revision SHA
- MEDIUM: Empty-list guard in api.py query enrichment
- MEDIUM: Narrowed broad except blocks (debug -> warning, except Exception: pass -> narrowed types)
- MEDIUM: Robust JSON extraction in fact_extraction (bracket counting, code fence stripping)
- MEDIUM: Truncation logging in fact_extraction
- MEDIUM: Thread lock on shared write connection in consolidator
- MEDIUM: GitHub Actions CI workflow added
- MEDIUM: query() refactored into _run_bm25, _run_dense, _run_ppr, _fuse_and_enrich helpers
- LOW: ruff auto-fix (import sorting, unused imports)
- LOW: MD5 usedforsecurity=False in stub_embedder
- LOW: MIT LICENSE added
- LOW: pyproject.toml added
- LOW: README troubleshooting section added

## Verification results
- bandit: HIGH=0, MEDIUM=4 (all B608 false positives on parameterized SQL)
- ruff: 0 errors on modified files; pre-existing warnings in untouched files
- pip-audit: no known vulnerabilities
- pytest: 92 passed (matches baseline)

## Manual actions still required
- [ ] Force-push history rewrite: git push origin --force --all && git push origin --force --tags
- [ ] Replace [REPLACE WITH NAME OR LEAVE AS-IS] in LICENSE
- [ ] Decide whether to delete /home/ubuntu/.openclaw/openclaw.json on Abacus host
- [ ] Fill in vault/Personal/Profile.md locally with personal data (now gitignored)
- [ ] Rotate any credentials that were mentioned in the old Profile.md
