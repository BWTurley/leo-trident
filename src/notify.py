"""
Leo Trident — Telegram Notifier

Thin wrapper around Telegram Bot sendMessage. Never raises; returns bool.
Reads TELEGRAM_BOT_TOKEN from env, falling back to ~/.secrets/leo_persona.env.
Reads TELEGRAM_CHAT_ID from env, defaulting to "8561774202" (Brett's home chat).
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_CHAT_ID = "8561774202"
_MAX_LEN = 4000
_TRUNC_SUFFIX = "…[truncated]"
_SECRETS_PATH = Path.home() / ".secrets" / "leo_persona.env"


def _read_secret_from_file(key: str, path: Path = _SECRETS_PATH) -> Optional[str]:
    try:
        if not path.exists():
            return None
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key:
                v = v.strip().strip('"').strip("'")
                return v
        return None
    except Exception as e:  # noqa: BLE001
        logger.debug("notify: secret read failed for %s: %s", key, e)
        return None


def _get_token() -> Optional[str]:
    return os.environ.get("TELEGRAM_BOT_TOKEN") or _read_secret_from_file(
        "TELEGRAM_BOT_TOKEN"
    )


def _get_chat_id() -> str:
    return (
        os.environ.get("TELEGRAM_CHAT_ID")
        or _read_secret_from_file("TELEGRAM_CHAT_ID")
        or _DEFAULT_CHAT_ID
    )


def _truncate(text: str) -> str:
    if len(text) <= _MAX_LEN:
        return text
    keep = _MAX_LEN - len(_TRUNC_SUFFIX)
    return text[:keep] + _TRUNC_SUFFIX


def notify_telegram(text: str, parse_mode: str = "Markdown") -> bool:
    """
    Send a Telegram message. Returns True on HTTP 200, False on any failure.
    Never raises.
    """
    token = _get_token()
    if not token:
        logger.warning("notify: TELEGRAM_BOT_TOKEN not configured")
        return False
    chat_id = _get_chat_id()
    body = {
        "chat_id": chat_id,
        "text": _truncate(text),
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            status = getattr(resp, "status", resp.getcode())
            if status == 200:
                return True
            logger.error("notify: telegram returned status=%s", status)
            return False
    except urllib.error.HTTPError as e:
        logger.error("notify: HTTPError %s: %s", e.code, e.reason)
        return False
    except Exception as e:  # noqa: BLE001
        logger.error("notify: send failed: %s", e)
        return False


__all__ = ["notify_telegram"]
