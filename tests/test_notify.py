"""Tests for src.notify."""
from unittest import mock

from src import notify


class _FakeResp:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def getcode(self):
        return self.status


def test_notify_telegram_success(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")

    with mock.patch.object(notify.urllib.request, "urlopen",
                           return_value=_FakeResp(200)) as m:
        ok = notify.notify_telegram("hi")
    assert ok is True
    assert m.called


def test_notify_telegram_http_error(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")

    import urllib.error
    err = urllib.error.HTTPError(
        url="http://x", code=500, msg="boom", hdrs=None, fp=None,
    )

    def boom(*a, **kw):
        raise err

    with mock.patch.object(notify.urllib.request, "urlopen", side_effect=boom):
        ok = notify.notify_telegram("hi")
    assert ok is False  # no raise


def test_notify_telegram_no_token(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    # point secret file path at a guaranteed-missing location
    monkeypatch.setattr(notify, "_SECRETS_PATH",
                        notify.Path("/nonexistent/leo_persona.env"))
    assert notify.notify_telegram("hi") is False


def test_truncate_long_message():
    long = "x" * 5000
    out = notify._truncate(long)
    assert len(out) <= 4000
    assert out.endswith("…[truncated]")
