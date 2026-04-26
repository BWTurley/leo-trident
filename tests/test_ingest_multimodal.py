"""Tests for multimodal ingest adapters (PDF + image)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from src.ingest.image import caption_image
from src.ingest.pdf import ingest_pdf


def _make_tiny_pdf(path: Path) -> None:
    """Build a 1-page PDF with pypdf — no extra deps needed."""
    from pypdf import PdfWriter
    from pypdf.generic import (
        ArrayObject,
        DecodedStreamObject,
        DictionaryObject,
        FloatObject,
        NameObject,
        NumberObject,
    )

    writer = PdfWriter()
    # Add a blank page; we only need ingest_pdf to return >0 page-strings.
    writer.add_blank_page(width=200, height=200)

    # Inject a tiny content stream so extract_text has *something* to parse,
    # though even a blank page is enough for the assertion `len(pages) > 0`.
    page = writer.pages[0]
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 12 Tf 50 100 Td (Hello PDF) Tj ET")
    page[NameObject("/Contents")] = content

    # Bare-bones font resource so the content stream is well-formed.
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    resources = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
    })
    page[NameObject("/Resources")] = resources
    page[NameObject("/MediaBox")] = ArrayObject(
        [NumberObject(0), NumberObject(0), FloatObject(200), FloatObject(200)]
    )

    with open(path, "wb") as f:
        writer.write(f)


def test_pdf_ingest(tmp_path: Path) -> None:
    pdf_path = tmp_path / "tiny.pdf"
    _make_tiny_pdf(pdf_path)

    pages = ingest_pdf(pdf_path)
    assert isinstance(pages, list)
    assert len(pages) > 0
    assert all(isinstance(p, str) for p in pages)


def test_image_caption_no_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    img = tmp_path / "anything.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)  # not a valid PNG, fine

    out = caption_image(img)
    assert out.startswith("[image:")
    assert "no caption" in out


def test_caption_image_with_mocked_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-123")

    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)

    fake_caption = "A red apple sitting on a wooden desk next to a notebook."
    fake_response = {
        "id": "msg_x",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": fake_caption}],
    }

    class FakeResp:
        def __init__(self, data: bytes):
            self._data = data

        def read(self) -> bytes:
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        # Verify the request was constructed correctly.
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResp(json.dumps(fake_response).encode("utf-8"))

    import src.ingest.image as image_mod

    monkeypatch.setattr(image_mod.urllib.request, "urlopen", fake_urlopen)

    out = caption_image(img)
    assert out == fake_caption
    assert "api.anthropic.com" in captured["url"]
    # Body should contain image block with base64 data.
    msg = captured["body"]["messages"][0]
    assert msg["role"] == "user"
    assert any(b.get("type") == "image" for b in msg["content"])
    # Silence unused import warning in some lints.
    _ = io.BytesIO
