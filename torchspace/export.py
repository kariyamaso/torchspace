"""Rendering/export of the self-contained viewer.

The viewer HTML (three.js inlined, no network access needed) ships inside the
wheel as package data and is loaded via importlib.resources — so exports and
notebook display work identically from a pip install, a checkout, or Colab.
"""
from __future__ import annotations

import html as _htmlmod
import json
from importlib import resources
from typing import Any

_TOKEN = "/*__TORCHSPACE_DATA__*/"
_TITLE = "__TORCHSPACE_TITLE__"


def _template() -> str:
    return (resources.files("torchspace") / "assets" / "viewer.html").read_text(
        encoding="utf-8")


def render_html(irs: list[dict[str, Any]], title: str = "TorchSpace") -> str:
    """Returns the complete, self-contained viewer HTML as a string."""
    payload = json.dumps({"models": irs}, ensure_ascii=False)
    # "<" only occurs inside JSON string literals, where the "\\u003c"
    # escape is equivalent — neutralises "</script>", "<!--" and any other
    # markup the HTML script-data parser would act on.
    payload = payload.replace("<", "\\u003c")
    # split at the data token first: the title substitution then cannot
    # touch the payload, and a title containing the literal token (or an
    # IR containing the literal title marker) cannot confuse either step.
    head, _, tail = _template().partition(_TOKEN)
    escaped_title = _htmlmod.escape(title)
    head = head.replace(_TITLE, escaped_title)
    tail = tail.replace(_TITLE, escaped_title)
    return head + f"window.TORCHSPACE_DATA = {payload};" + tail


def export_html(irs: list[dict[str, Any]], path: str,
                title: str = "TorchSpace") -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_html(irs, title))
    return path


def iframe_html(irs: list[dict[str, Any]], title: str = "TorchSpace",
                height: int = 650, width: str = "100%") -> str:
    """Sandboxed-iframe wrapper for notebook front-ends (Jupyter, Colab,
    VS Code). srcdoc keeps everything inline: no server, no network, no CDN —
    which is exactly why the viewer inlines three.js instead of loading it."""
    import re
    height = int(height)
    if not re.fullmatch(r"[0-9.]+(px|%|em|rem|vw|vh)?", str(width)):
        raise ValueError(f"width must be a plain CSS length, got {width!r}")
    doc = render_html(irs, title)
    esc = _htmlmod.escape(doc, quote=True)
    return (
        f'<iframe srcdoc="{esc}" sandbox="allow-scripts" '
        f'style="width:{width};height:{height}px;border:1px solid #232a3a;'
        f'border-radius:10px;background:#0b0e14" '
        f'title="TorchSpace — {_htmlmod.escape(title)}"></iframe>'
    )
