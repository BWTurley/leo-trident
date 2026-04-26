"""
Leo Trident — Model Pricing Table

Static USD-per-1k-tokens table for embedders & generative models we touch.
Local models are 0.0. Unknown models default to 0.0 (free).

Only `cost_for(model, tokens)` is part of the public API.
"""
from __future__ import annotations

# USD per 1,000 input tokens
PRICE_TABLE: dict[str, float] = {
    # Local embedders — free.
    "bge-m3": 0.0,
    "BAAI/bge-m3": 0.0,
    "nomic-embed-text-v1.5": 0.0,
    # Hosted embedders.
    "voyage-3": 0.00018,
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013,
    # Generative (input pricing) — used for image captions, etc.
    "claude-haiku-4-5": 0.00025,
    "claude-sonnet-4-7": 0.003,
}


def cost_for(model: str, tokens: int) -> float:
    """Return USD cost for `tokens` against `model`. Unknown model → 0.0."""
    if not model or tokens <= 0:
        return 0.0
    price = PRICE_TABLE.get(model, 0.0)
    return (tokens / 1000.0) * price


__all__ = ["PRICE_TABLE", "cost_for"]
