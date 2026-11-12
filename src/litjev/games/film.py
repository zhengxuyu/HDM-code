# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minimal Labs
# Adapted from jevlike examples/film/build-film.mjs at
# 94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452; rewritten for the LitJev trace schema.
"""Build a self-contained replay. Never fabricate upstream attention activations."""

import json
from pathlib import Path


def _inline(template: Path, placeholder: str, trace: dict, output: Path):
    """Put the trace inside the page, so the result is one file that opens anywhere.

    Script delimiters are escaped, including any that arrive inside an imported
    trace's own strings -- a slow mode's reasoning text is model output and goes
    straight into this page.
    """
    payload = json.dumps(trace, allow_nan=False)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = template.read_text().replace(f'"{placeholder}"', payload, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)


def build_compare(trace_path: Path, output: Path):
    """The fast-against-slow page, with one run's trace baked in."""
    trace = json.loads(trace_path.read_text())
    if trace.get("schema_version") != "litjev.compare.v1" or not trace.get("rows"):
        raise ValueError("Expected a non-empty litjev.compare.v1 trace")
    _inline(Path(__file__).parents[1] / "static" / "compare.html",
            "__LITJEV_COMPARE__", trace, output)


def build_film(trace_path: Path, output: Path):
    trace = json.loads(trace_path.read_text())
    if trace.get("schema_version") != "litjev.trace.v1" or not trace.get("decisions"):
        raise ValueError("Expected a non-empty litjev.trace.v1 trace")
    template = (Path(__file__).parents[1] / "static" / "film.html").read_text()
    # Escape script delimiters, including malicious action labels in an imported trace.
    payload = json.dumps(trace, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e")
    payload = payload.replace("&", "\\u0026")
    html = template.replace('"__LITJEV_TRACE__"', payload, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
