"""Image ingest adapter — caption an image via Anthropic API for semantic search.

Uses urllib (stdlib) so we don't depend on the anthropic SDK shape, which
also makes mocking trivial in tests.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-haiku-4-5"
ANTHROPIC_VERSION = "2023-06-01"

CAPTION_PROMPT = (
    "Describe this image in detail for a semantic search index. "
    "Include subjects, setting, text visible in the image, colors, "
    "notable objects, and any contextual cues. Return only the description, "
    "no preamble."
)

_EXT_MEDIA_TYPE = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _media_type_for(path: Path) -> str:
    return _EXT_MEDIA_TYPE.get(path.suffix.lower(), "image/png")


def caption_image(path: str | Path) -> str:
    """Return a detailed caption for the image suitable for semantic search.

    If ANTHROPIC_API_KEY is not set, returns a placeholder string rather
    than raising, so ingest pipelines stay resilient on dev machines.
    """
    p = Path(path)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return f"[image: {p.name}] (no caption — ANTHROPIC_API_KEY not set)"

    try:
        img_bytes = p.read_bytes()
    except OSError as e:
        logger.warning("caption_image: failed to read %s: %s", p, e)
        return f"[image: {p.name}] (no caption — could not read file)"

    b64 = base64.standard_b64encode(img_bytes).decode("ascii")
    media_type = _media_type_for(p)

    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }
        ],
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=data,
        method="POST",
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.warning("caption_image: API call failed: %s", e)
        return f"[image: {p.name}] (no caption — API error: {type(e).__name__})"

    # Response shape: {"content": [{"type": "text", "text": "..."}], ...}
    for block in payload.get("content", []) or []:
        if block.get("type") == "text" and block.get("text"):
            return block["text"].strip()
    return f"[image: {p.name}] (no caption — empty response)"
