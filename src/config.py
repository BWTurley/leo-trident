"""Central config — single source of truth for paths, runtime mode, devices.

Environment variables (all optional, sensible defaults):
    LEO_TRIDENT_HOME    Project root (default: ~/leo_trident)
    LEO_LLM_MODE        "cloud" or "local" (default: cloud for backward-compat)
    LEO_EMBED_DEVICE    "cpu", "mps", or "cuda" (default: cpu)
    OLLAMA_URL          http://localhost:11434
    LEO_CONSOLIDATION_MODEL   qwen2.5:14b-instruct-q5_K_M
    LEO_EMBED_MODEL     nomic-embed-text
    ABACUS_ENDPOINT     https://routellm.abacus.ai/v1
    ABACUS_API_KEY      (required if LEO_LLM_MODE=cloud)
"""
from __future__ import annotations

import os
from pathlib import Path

# Load .env if present — never fails if missing
try:
    from dotenv import load_dotenv
    _repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(_repo_root / ".env", override=False)
except ImportError:
    pass


def _resolve_home() -> Path:
    env = os.environ.get("LEO_TRIDENT_HOME")
    if env:
        return Path(env).expanduser().resolve()
    # Abacus back-compat: if the legacy path exists and no env var is set, use it
    legacy = Path("/home/ubuntu/leo_trident")
    if legacy.exists():
        return legacy
    return (Path.home() / "leo_trident").resolve()


BASE_PATH: Path = _resolve_home()
DATA_PATH: Path = BASE_PATH / "data"
DB_PATH: Path = DATA_PATH / "leo_trident.db"
LANCE_PATH: Path = DATA_PATH / "lancedb"
VAULT_PATH: Path = BASE_PATH / "vault"
SYSTEM_PATH: Path = VAULT_PATH / "_system"

LLM_MODE: str = os.environ.get("LEO_LLM_MODE", "cloud")
EMBED_DEVICE: str = os.environ.get("LEO_EMBED_DEVICE", "cpu")

OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_CONSOLIDATION_MODEL: str = os.environ.get(
    "LEO_CONSOLIDATION_MODEL", "qwen2.5:14b-instruct-q5_K_M"
)
OLLAMA_EMBED_MODEL: str = os.environ.get("LEO_EMBED_MODEL", "nomic-embed-text")

ABACUS_ENDPOINT: str = os.environ.get(
    "ABACUS_ENDPOINT", "https://routellm.abacus.ai/v1"
)
ABACUS_API_KEY: str = os.environ.get("ABACUS_API_KEY", "")
ABACUS_MODEL: str = os.environ.get("ABACUS_MODEL", "claude-sonnet-4-6")

# Legacy key-file fallback for Abacus back-compat (read once if env is empty)
if not ABACUS_API_KEY and LLM_MODE == "cloud":
    _legacy_key_path = Path("/home/ubuntu/.openclaw/openclaw.json")
    if _legacy_key_path.exists():
        try:
            import json
            with open(_legacy_key_path) as f:
                cfg = json.load(f)
            ABACUS_API_KEY = cfg["models"]["providers"]["abacus"]["apiKey"]
        except Exception:
            pass


def _bootstrap_personal_files():
    """Copy *.example.* templates into place if the real files don't exist."""
    for ex in (BASE_PATH / "vault" / "_system").glob("*.example.json"):
        real = ex.with_name(ex.name.replace(".example", ""))
        if not real.exists():
            real.write_text(ex.read_text())
    prof_ex = BASE_PATH / "vault" / "Personal" / "Profile.example.md"
    prof = BASE_PATH / "vault" / "Personal" / "Profile.md"
    if prof_ex.exists() and not prof.exists():
        prof.write_text(prof_ex.read_text())


_bootstrap_personal_files()
