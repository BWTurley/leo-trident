"""LLM client with cloud (Abacus) / local (Ollama) dispatch.

Selected by LEO_LLM_MODE env var. Default is 'cloud' for back-compat.
"""
from __future__ import annotations

import logging

import httpx

from src.config import (
    LLM_MODE,
    OLLAMA_URL, OLLAMA_CONSOLIDATION_MODEL,
    ABACUS_ENDPOINT, ABACUS_API_KEY, ABACUS_MODEL,
)

logger = logging.getLogger(__name__)


def complete(prompt: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
    """Dispatch to the configured backend. Returns raw text response."""
    if LLM_MODE == "local":
        return _ollama_complete(prompt, max_tokens, temperature)
    if LLM_MODE == "cloud":
        return _abacus_complete(prompt, max_tokens, temperature)
    raise ValueError(f"Unknown LEO_LLM_MODE: {LLM_MODE!r} (expected 'cloud' or 'local')")


def _ollama_complete(prompt: str, max_tokens: int, temperature: float) -> str:
    resp = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_CONSOLIDATION_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _abacus_complete(prompt: str, max_tokens: int, temperature: float) -> str:
    if not ABACUS_API_KEY:
        raise RuntimeError(
            "LEO_LLM_MODE=cloud but ABACUS_API_KEY is empty. "
            "Set it in .env or environment."
        )
    resp = httpx.post(
        f"{ABACUS_ENDPOINT}/chat/completions",
        headers={
            "Authorization": f"Bearer {ABACUS_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": ABACUS_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
