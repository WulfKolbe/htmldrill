"""Command handlers — return prose strings (PDFDRILL/CHATDRILL convention).

A small ``Ctx`` carries the resolved args; each handler takes it, does ONE cheap
thing, appends to the sidecar, and returns text. Idempotency is structural: a
command records a fact and, on re-run, detects it and returns the cached view
unless ``--force``.

L0 tier (this module): every command except ``fetch`` operates on the snapshot
``fetch`` captured — no network, no headless browser. Snapshot commands hard-gate
on the ``FETCHED`` fact and tell the user to run ``fetch`` first if it's absent
(they are deliberately NOT wired as planner ``requires:`` of ``fetch``, so
``--ensure`` can never trigger a network round-trip).
"""
from __future__ import annotations

import json
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import planner
from .parse import html as H
from .sidecar import Sidecar, work_root
from .sources import fetch as F
from .sources import render as R

# -- facts (module scope) ----------------------------------------------------
FETCHED = "FETCHED"
SIZE_KNOWN = "SIZE_KNOWN"
META_KNOWN = "META_KNOWN"
CANONICAL_KNOWN = "CANONICAL_KNOWN"
LINKS_KNOWN = "LINKS_KNOWN"
JSONLD_KNOWN = "JSONLD_KNOWN"
MICRODATA_KNOWN = "MICRODATA_KNOWN"
OPENGRAPH_KNOWN = "OPENGRAPH_KNOWN"
FEEDS_KNOWN = "FEEDS_KNOWN"
OUTLINE_KNOWN = "OUTLINE_KNOWN"
# L1 (render)
RENDERED = "RENDERED"
TEXT_KNOWN = "TEXT_KNOWN"
COMPARED = "COMPARED"
# L5 (model)
MODEL_BUILT = "MODEL_BUILT"
# L6 (projectors — offline)
TIDDLERS_BUILT = "TIDDLERS_BUILT"
MD_BUILT = "MD_BUILT"
LLMTEXT_BUILT = "LLMTEXT_BUILT"


@dataclass
class Ctx:
    url: Optional[str] = None         # URL / local path, or (for status/steps) an id prefix
    work: Optional[str] = None
    force: bool = False
    as_json: bool = False
    target: Optional[str] = None      # for `steps`
    out: Optional[str] = None
    ua: Optional[str] = None
    timeout: float = F.DEFAULT_TIMEOUT
    window: str = "1280,900"          # render viewport


# -- id resolution -----------------------------------------------------------

def _resolve_id(ctx: Ctx) -> str:
    """Map the positional (a URL/path or an existing id prefix) to a sidecar id."""
    if not ctx.url:
        raise ValueError("a URL (or an existing sidecar id) is required")
    cand = F.local_id_for(ctx.url)
    root = work_root(ctx.work)
    if (root / f"{cand}.htmldrill.json").exists():
        return cand
    # `ctx.url` may be an absolute path (a non-relative glob pattern, which
    # Path.glob rejects); only try prefix-resolution when it's a bare id token.
    if "/" not in ctx.url and "\\" not in ctx.url:
        return resolve_local_id(ctx.url, ctx.work)
    return cand


def _load_snapshot(ctx: Ctx) -> tuple[Sidecar, str]:
    sc = Sidecar(_resolve_id(ctx), work=ctx.work)
    if not sc.has(FETCHED):
        raise FileNotFoundError(
            f"no fetched snapshot for {ctx.url!r} — run `htmldrill fetch {ctx.url}` first")
    html = sc.read_blob("raw.html") or ""
    return sc, html


def _prev(sc: Sidecar, fact: str) -> str:
    return ",".join(sorted(sc.facts - {fact})) or "INIT"


def _base_url(sc: Sidecar) -> str:
    return sc.get_evidence("final_url") or sc.get_evidence("url") or ""


# -- fetch (the only network command) ----------------------------------------

def cmd_fetch(ctx: Ctx) -> str:
    if not ctx.url:
        raise ValueError("usage: htmldrill fetch <url>")
    lid = F.local_id_for(ctx.url)
    sc = Sidecar(lid, work=ctx.work)
    if sc.has(FETCHED) and not ctx.force:
        return _fetch_report(sc, cached=True)
    t0 = time.perf_counter()
    res = F.fetch(ctx.url, timeout=ctx.timeout, ua=ctx.ua)
    sc.write_blob("raw.html", res.text)
    sc.write_blob("headers.json", json.dumps(res.headers, indent=2, ensure_ascii=False))
    cost_ms = (time.perf_counter() - t0) * 1000
    sc.set_evidence("url", ctx.url)
    sc.set_evidence("final_url", res.final_url)
    sc.set_evidence("status", res.status)
    sc.set_evidence("content_type", res.content_type)
    sc.set_evidence("bytes", len(res.body))
    sc.set_layer("raw_html", {"path": "raw.html", "format": "text/html"})
    sc.add_fact(FETCHED)
    sc.log_transition("fetch", "INIT", FETCHED, cost_ms,
                      f"{res.status} {len(res.body)}B {res.content_type}")
    sc.save()
    return _fetch_report(sc, cached=False)


def _fetch_report(sc: Sidecar, cached: bool) -> str:
    ev = sc.evidence
    tag = "cached" if cached else "fetched"
    return (f"{tag} {ev.get('url')}\n"
            f"  id:          {sc.local_id}\n"
            f"  status:      {ev.get('status')}\n"
            f"  final url:   {ev.get('final_url')}\n"
            f"  content-type:{ev.get('content_type')}\n"
            f"  size:        {ev.get('bytes')} bytes\n"
            f"  snapshot:    {sc.blob_path('raw.html')}\n"
            f"  next: meta · links · jsonld · outline · size")


# -- snapshot introspection (no network) -------------------------------------

def _guess_framework(html: str) -> str:
    # Patterns must be SPECIFIC markers (attributes/globals/generator meta), not
    # bare words like "vue"/"angular" — those substring-match chat/JSON payloads
    # and produce false positives (a 62MB ChatGPT export "is Vue + Angular";
    # a TiddlyWiki "is Vue"). We also detect TiddlyWiki, the most common real input.
    pats = [
        ("Next.js", r"__NEXT_DATA__|/_next/static/"),
        ("Nuxt", r"window\.__NUXT__"),
        ("React", r"data-reactroot|react-dom\.production|__REACT_DEVTOOLS"),
        ("Vue", r"data-v-[0-9a-f]{8}\b|__VUE__|vue\.runtime"),
        ("Angular", r"\bng-version=|_nghost-|\bng-app="),
        ("Svelte", r"\bsvelte-[0-9a-z]{6}\b"),
        ("TiddlyWiki",
         r'application-name"\s+content="TiddlyWiki"|tiddlywiki-tiddler-store|'
         r'id="storeArea"|generator"\s+content="TiddlyWiki"'),
    ]
    hits = [name for name, p in pats if re.search(p, html, re.I)]
    return ", ".join(hits) if hits else "none detected"


def cmd_size(ctx: Ctx) -> str:
    sc, html = _load_snapshot(ctx)
    c = H.collect(html)
    nbytes = sc.get_evidence("bytes", len(html.encode("utf-8", "replace")))
    nlines = html.count("\n") + 1
    fw = _guess_framework(html)
    visible_text = sum(len(t) for _, t in c.anchors) + sum(len(t) for _, t in c.headings)
    # Shallow render-required heuristic: a framework shell with almost no static
    # text/headings is probably JS-rendered (the OCR-analog escalation signal).
    needs_render = bool(fw != "none detected" and len(c.headings) <= 1 and visible_text < 200)
    sc.set_evidence("tag_count", c.tag_count)
    sc.set_evidence("framework", fw)
    sc.set_evidence("needs_render", needs_render)
    sc.add_fact(SIZE_KNOWN)
    sc.log_transition("size", _prev(sc, SIZE_KNOWN), SIZE_KNOWN, 0,
                      f"{nbytes}B {c.tag_count} tags fw={fw}")
    sc.save()
    verdict = ("LIKELY JS-RENDERED — static markup is thin; `render` recommended (M1)"
               if needs_render else "static markup looks sufficient — no render needed")
    return (f"{sc.local_id}: {nbytes} bytes, {nlines} lines, ~{c.tag_count} tags\n"
            f"  framework:  {fw}\n"
            f"  headings:   {len(c.headings)}   anchors: {len(c.anchors)}\n"
            f"  verdict:    {verdict}")


def cmd_headers(ctx: Ctx) -> str:
    sc, _ = _load_snapshot(ctx)
    raw = sc.read_blob("headers.json")
    if not raw:
        return "no captured headers (snapshot has no headers.json)."
    headers = json.loads(raw)
    lines = [f"response headers for {sc.local_id} ({len(headers)}):"]
    for k, v in headers.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def cmd_meta(ctx: Ctx) -> str:
    sc, html = _load_snapshot(ctx)
    c = H.collect(html)
    meta = H.extract_meta(c)
    sc.set_evidence("meta", meta)
    sc.add_fact(META_KNOWN)
    sc.log_transition("meta", _prev(sc, META_KNOWN), META_KNOWN, 0, f"{len(meta)} keys")
    sc.save()
    if not meta and not c.title:
        return f"{sc.local_id}: no <meta> tags or <title> found."
    lines = [f"{sc.local_id}: <title> {c.title!r}"]
    if meta:
        lines.append(f"  {len(meta)} meta key(s):")
        for k, v in meta.items():
            lines.append(f"    {k}: {v[:100]}")
    else:
        lines.append("  (no <meta> tags, but a <title> is present)")
    return "\n".join(lines)


def cmd_canonical(ctx: Ctx) -> str:
    sc, html = _load_snapshot(ctx)
    c = H.collect(html)
    canon = H.extract_canonical(c)
    sc.set_evidence("canonical", canon)
    sc.add_fact(CANONICAL_KNOWN)
    sc.log_transition("canonical", _prev(sc, CANONICAL_KNOWN), CANONICAL_KNOWN, 0, "")
    sc.save()
    if not canon:
        return f"{sc.local_id}: no <link rel=canonical> and no og:url."
    return f"{sc.local_id}:\n" + "\n".join(f"  {k}: {v}" for k, v in canon.items())


# A URL stops at whitespace, quotes, brackets, backslashes (JS escapes), and
# common code punctuation — so we don't harvest `http://\\\\n…` or `http://$(x`
# escape-mangled fragments out of JS string literals as if they were real links.
_URL_RE = re.compile(r"https?://[^\s\"'<>)\]}\\(|]+")


def _clean_url(u: str) -> str:
    """Decode the handful of HTML entities that leak into harvested URLs and trim
    trailing punctuation, so the invisible-URL list is clean."""
    import html as _html
    return _html.unescape(u).rstrip(".,;")


def cmd_links(ctx: Ctx) -> str:
    sc, html = _load_snapshot(ctx)
    c = H.collect(html)
    base = _base_url(sc)
    links = H.extract_links(c, base_url=base)
    # KILLER CASE: URLs present in the raw markup (data-*, inline JSON, <link>,
    # JS string literals) that never surface as a visible <a href> anchor — the
    # HTML analog of pdfdrill reading invisible links out of the annotation layer.
    anchor_set = {u for u, _ in links["internal"] + links["external"] + links["other"]}
    raw_urls = {_clean_url(u) for u in _URL_RE.findall(html)}
    # keep only plausible hosts (a dot in the authority) — drops `http://<wikiname`,
    # `http://$(userName`, `http://127.0.0.1:8080` template noise stays but garbage
    # with no real host is dropped.
    hidden = sorted(
        u for u in raw_urls
        if u not in anchor_set and "." in u.split("//", 1)[-1].split("/", 1)[0])
    sc.set_evidence("link_counts", {k: len(v) for k, v in links.items()})
    sc.set_evidence("hidden_link_count", len(hidden))
    sc.add_fact(LINKS_KNOWN)
    sc.log_transition("links", _prev(sc, LINKS_KNOWN), LINKS_KNOWN, 0,
                      f"{len(anchor_set)} anchors, {len(hidden)} hidden")
    sc.save()
    lines = [f"{sc.local_id}: {len(links['internal'])} internal, "
             f"{len(links['external'])} external, {len(links['other'])} other anchors"]
    for u, t in links["external"][:8]:
        lines.append(f"  → {u[:78]}  {('· ' + t[:30]) if t else ''}")
    if hidden:
        lines.append(f"  ⚠ {len(hidden)} URL(s) in the markup but NOT visible anchors "
                     f"(data-*/inline-JSON/JS literals):")
        for u in hidden[:8]:
            lines.append(f"      {u[:80]}")
    return "\n".join(lines)


def cmd_jsonld(ctx: Ctx) -> str:
    sc, html = _load_snapshot(ctx)
    c = H.collect(html)
    blocks = H.extract_jsonld(c)
    types: list[str] = []
    for b in blocks:
        if b["ok"]:
            data = b["data"]
            for node in (data if isinstance(data, list) else [data]):
                if isinstance(node, dict) and node.get("@type"):
                    t = node["@type"]
                    types.extend(t if isinstance(t, list) else [t])
    sc.set_evidence("jsonld_types", types)
    sc.set_evidence("jsonld_blocks", len(blocks))
    sc.add_fact(JSONLD_KNOWN)
    sc.log_transition("jsonld", _prev(sc, JSONLD_KNOWN), JSONLD_KNOWN, 0,
                      f"{len(blocks)} blocks {types}")
    sc.save()
    if not blocks:
        return f"{sc.local_id}: no <script type=application/ld+json> blocks."
    lines = [f"{sc.local_id}: {len(blocks)} JSON-LD block(s), @types: "
             f"{', '.join(types) or '(none)'}"]
    for i, b in enumerate(blocks):
        if b["ok"]:
            data = b["data"]
            top = data[0] if isinstance(data, list) and data else data
            keys = list(top.keys()) if isinstance(top, dict) else type(data).__name__
            lines.append(f"  [{i}] ok — top-level keys: {keys}")
        else:
            lines.append(f"  [{i}] PARSE ERROR: {b['error']} ({b['raw_len']} bytes)")
    return "\n".join(lines)


def cmd_microdata(ctx: Ctx) -> str:
    sc, html = _load_snapshot(ctx)
    c = H.collect(html)
    items = H.extract_microdata(c)
    sc.set_evidence("microdata_types", [m["itemtype"] for m in items])
    sc.add_fact(MICRODATA_KNOWN)
    sc.log_transition("microdata", _prev(sc, MICRODATA_KNOWN), MICRODATA_KNOWN, 0,
                      f"{len(items)} items")
    sc.save()
    if not items:
        return f"{sc.local_id}: no microdata (itemscope/itemtype) found."
    lines = [f"{sc.local_id}: {len(items)} microdata item(s):"]
    for m in items[:10]:
        lines.append(f"  {m['itemtype']}  props: {', '.join(m['props'][:8]) or '(none)'}")
    return "\n".join(lines)


def cmd_opengraph(ctx: Ctx) -> str:
    sc, html = _load_snapshot(ctx)
    c = H.collect(html)
    og = H.extract_opengraph(c)
    sc.set_evidence("opengraph", og)
    sc.add_fact(OPENGRAPH_KNOWN)
    sc.log_transition("opengraph", _prev(sc, OPENGRAPH_KNOWN), OPENGRAPH_KNOWN, 0,
                      f"{len(og)} props")
    sc.save()
    if not og:
        return f"{sc.local_id}: no OpenGraph/Twitter-card meta."
    return f"{sc.local_id}: {len(og)} og:/twitter: prop(s)\n" + \
        "\n".join(f"  {k}: {v[:100]}" for k, v in og.items())


def cmd_feeds(ctx: Ctx) -> str:
    sc, html = _load_snapshot(ctx)
    c = H.collect(html)
    feeds = H.extract_feeds(c)
    sc.set_evidence("feeds", feeds)
    sc.add_fact(FEEDS_KNOWN)
    sc.log_transition("feeds", _prev(sc, FEEDS_KNOWN), FEEDS_KNOWN, 0, f"{len(feeds)}")
    sc.save()
    if not feeds:
        return f"{sc.local_id}: no RSS/Atom feed links."
    lines = [f"{sc.local_id}: {len(feeds)} feed(s):"]
    for f in feeds:
        lines.append(f"  {f['type']:<24} {f['href']}  {f['title']}")
    return "\n".join(lines)


def cmd_outline(ctx: Ctx) -> str:
    sc, html = _load_snapshot(ctx)
    c = H.collect(html)
    heads = H.extract_outline(c)
    sc.set_evidence("heading_count", len(heads))
    sc.add_fact(OUTLINE_KNOWN)
    sc.log_transition("outline", _prev(sc, OUTLINE_KNOWN), OUTLINE_KNOWN, 0,
                      f"{len(heads)} headings")
    sc.save()
    if not heads:
        return f"{sc.local_id}: no headings (h1–h6) in the static markup."
    lines = [f"{sc.local_id}: heading outline ({len(heads)}):"]
    for lvl, txt in heads:
        lines.append(f"  {'  ' * (lvl - 1)}h{lvl} {txt[:80]}")
    return "\n".join(lines)


# -- L1 render (headless escalation) -----------------------------------------

def cmd_render(ctx: Ctx) -> str:
    """The expensive, escalation-gated layer: materialize the page with headless
    Chrome and snapshot the rendered DOM + a screenshot. Like `fetch`, it is the
    only other command allowed to touch the network — never auto-ensured."""
    if not ctx.url:
        raise ValueError("usage: htmldrill render <url>")
    lid = F.local_id_for(ctx.url)
    sc = Sidecar(lid, work=ctx.work)
    if sc.has(RENDERED) and not ctx.force:
        return _render_report(sc, cached=True)
    t0 = time.perf_counter()
    res = R.render(ctx.url, timeout=ctx.timeout, window=ctx.window)
    cost_ms = (time.perf_counter() - t0) * 1000
    sc.write_blob("rendered.html", res.dom)
    has_shot = False
    if res.screenshot:
        sc.write_blob_bytes("screenshot.png", res.screenshot)
        has_shot = True
    c = H.collect(res.dom)
    sc.set_evidence("url", sc.get_evidence("url") or ctx.url)
    sc.set_evidence("rendered_bytes", len(res.dom.encode("utf-8", "replace")))
    sc.set_evidence("rendered_tags", c.tag_count)
    sc.set_evidence("chrome", res.chrome)
    sc.set_evidence("has_screenshot", has_shot)
    sc.set_layer("rendered_html", {"path": "rendered.html", "format": "text/html"})
    sc.add_fact(RENDERED)
    sc.log_transition("render", _prev(sc, RENDERED), RENDERED, cost_ms,
                      f"{c.tag_count} tags shot={has_shot} via {Path(res.chrome).name}")
    sc.save()
    return _render_report(sc, cached=False)


def _render_report(sc: Sidecar, cached: bool) -> str:
    ev = sc.evidence
    tag = "cached render" if cached else "rendered"
    static = ev.get("tag_count")
    delta = (f"  static tags: {static} → rendered: {ev.get('rendered_tags')} "
             f"(Δ{ev.get('rendered_tags', 0) - static:+d})\n" if static is not None else "")
    shot = sc.blob_path("screenshot.png")
    return (f"{tag} {ev.get('url')}\n"
            f"  id:          {sc.local_id}\n"
            f"  rendered:    {ev.get('rendered_bytes')} bytes, {ev.get('rendered_tags')} tags\n"
            f"{delta}"
            f"  chrome:      {ev.get('chrome')}\n"
            f"  screenshot:  {shot if ev.get('has_screenshot') else '(none)'}\n"
            f"  next: dom · text · compare · screenshot")


def _rendered_or_static(sc: Sidecar) -> tuple[str, str]:
    """(html, source-label) — prefer the rendered DOM, fall back to the static snapshot."""
    if sc.has(RENDERED):
        return sc.read_blob("rendered.html") or "", "rendered"
    if sc.has(FETCHED):
        return sc.read_blob("raw.html") or "", "static"
    raise FileNotFoundError("nothing captured yet — run `fetch` or `render` first")


def cmd_dom(ctx: Ctx) -> str:
    sc = Sidecar(_resolve_id(ctx), work=ctx.work)
    if not sc.has(RENDERED):
        raise FileNotFoundError(
            f"no render for {ctx.url!r} — run `htmldrill render {ctx.url}` first")
    c = H.collect(sc.read_blob("rendered.html") or "")
    lines = [f"{sc.local_id}: rendered DOM — ~{c.tag_count} tags, "
             f"{len(c.headings)} headings, {len(c.anchors)} anchors"]
    if sc.has(FETCHED):
        sc_static = H.collect(sc.read_blob("raw.html") or "")
        lines.append(f"  vs static: {sc_static.tag_count} tags "
                     f"(Δ{c.tag_count - sc_static.tag_count:+d}), "
                     f"{len(sc_static.anchors)} anchors "
                     f"(Δ{len(c.anchors) - len(sc_static.anchors):+d})")
        if c.tag_count > sc_static.tag_count * 1.3:
            lines.append("  → JS injected substantial DOM after load (SPA-like).")
    return "\n".join(lines)


def cmd_text(ctx: Ctx) -> str:
    sc = Sidecar(_resolve_id(ctx), work=ctx.work)
    html, source = _rendered_or_static(sc)
    text = H.extract_text(html)
    sc.write_blob("text.txt", text)
    sc.set_evidence("text_chars", len(text))
    sc.set_evidence("text_source", source)
    sc.add_fact(TEXT_KNOWN)
    sc.log_transition("text", _prev(sc, TEXT_KNOWN), TEXT_KNOWN, 0,
                      f"{len(text)} chars from {source}")
    sc.save()
    preview = text[:400] + ("…" if len(text) > 400 else "")
    return (f"{sc.local_id}: {len(text)} chars of visible text (from the {source} DOM)\n"
            f"  → {sc.blob_path('text.txt')}\n\n{preview}")


def cmd_screenshot(ctx: Ctx) -> str:
    sc = Sidecar(_resolve_id(ctx), work=ctx.work)
    if not (sc.has(RENDERED) and sc.has_blob("screenshot.png")):
        raise FileNotFoundError(
            f"no screenshot for {ctx.url!r} — run `htmldrill render {ctx.url}` first")
    p = sc.blob_path("screenshot.png")
    return f"{sc.local_id}: screenshot {p.stat().st_size} bytes\n  → {p}"


def cmd_compare(ctx: Ctx) -> str:
    """Three-way fidelity: static-DOM text | rendered-DOM text | screenshot — the
    HTML analog of pdfdrill's LaTeX|KaTeX|image QC. Surfaces content that exists
    only after render (SPA bodies) or only in the markup (SEO data never painted)."""
    sc = Sidecar(_resolve_id(ctx), work=ctx.work)
    static_html = sc.read_blob("raw.html") if sc.has(FETCHED) else None
    rendered_html = sc.read_blob("rendered.html") if sc.has(RENDERED) else None
    if not static_html and not rendered_html:
        raise FileNotFoundError(
            f"nothing to compare for {ctx.url!r} — run `fetch` and/or `render` first")
    static_txt = H.extract_text(static_html) if static_html else ""
    rendered_txt = H.extract_text(rendered_html) if rendered_html else ""
    sl, rl = len(static_txt), len(rendered_txt)
    has_shot = sc.has_blob("screenshot.png")
    delta = rl - sl
    if static_html and rendered_html:
        if delta > max(200, sl * 0.3):
            verdict = "content appears mostly AFTER render — static scrape would miss it (SPA)"
        elif sl > rl + 200:
            verdict = "static markup carries text the render dropped (e.g. noscript/SEO body)"
        else:
            verdict = "static and rendered text broadly agree — static scrape is sufficient"
    else:
        verdict = ("only the rendered DOM captured — run `fetch` for the static side"
                   if rendered_html else
                   "only the static snapshot captured — run `render` for the rendered side")
    rows = (f"  static-DOM text:   {sl:>7} chars  {'✓' if static_html else '—'}\n"
            f"  rendered-DOM text: {rl:>7} chars  {'✓' if rendered_html else '—'}\n"
            f"  screenshot:        {'present' if has_shot else 'absent':>7}  "
            f"{'✓' if has_shot else '—'}")
    html_report = (f"<!DOCTYPE html><meta charset=utf-8><title>compare {sc.local_id}</title>"
                   f"<h1>compare: {sc.local_id}</h1><table border=1 cellpadding=6>"
                   f"<tr><th>source</th><th>chars</th></tr>"
                   f"<tr><td>static-DOM</td><td>{sl}</td></tr>"
                   f"<tr><td>rendered-DOM</td><td>{rl}</td></tr>"
                   f"<tr><td>screenshot</td><td>{'yes' if has_shot else 'no'}</td></tr>"
                   f"</table><p>{verdict}</p>")
    sc.write_blob("compare.html", html_report)
    sc.set_evidence("compare_delta_chars", delta)
    sc.add_fact(COMPARED)
    sc.log_transition("compare", _prev(sc, COMPARED), COMPARED, 0, f"Δ{delta} chars")
    sc.save()
    return (f"{sc.local_id}: static | rendered | screenshot fidelity\n{rows}\n"
            f"  Δ rendered−static: {delta:+d} chars\n  verdict: {verdict}\n"
            f"  → {sc.blob_path('compare.html')}")


# -- L5 model (offline docmodel ingestion) -----------------------------------

def cmd_model(ctx: Ctx) -> str:
    """Lift the captured DOM into a shared docmodel Document and persist it as
    ``model.docmodel.json``. OFFLINE: requires a prior ``fetch`` or ``render``
    (it NEVER touches the network / auto-ensures a fetch — like the snapshot
    commands, it hard-gates on the captured-snapshot fact). Idempotent: skips when
    MODEL_BUILT unless ``--force``."""
    sc = Sidecar(_resolve_id(ctx), work=ctx.work)
    if not (sc.has(RENDERED) or sc.has(FETCHED)):
        raise FileNotFoundError(
            f"no snapshot for {ctx.url!r} — run `htmldrill fetch {ctx.url}` "
            f"(or `render`) first; `model` is offline and won't fetch.")
    if sc.has(MODEL_BUILT) and sc.has_blob("model.docmodel.json") and not ctx.force:
        return _model_report(sc, cached=True)

    html, source = _rendered_or_static(sc)        # prefer rendered DOM, else static
    bibkey = sc.local_id.upper()

    # docmodel is the one external bridge — import lazily, only inside `model`.
    from .ingest_dom import build_document
    t0 = time.perf_counter()
    doc = build_document(html, bibkey=bibkey, local_id=sc.local_id)
    cost_ms = (time.perf_counter() - t0) * 1000

    sc.write_blob("model.docmodel.json",
                  json.dumps(doc.to_dict(), indent=2, ensure_ascii=False))

    by_type: dict[str, int] = {}
    for o in doc.objects.values():
        by_type[o.type] = by_type.get(o.type, 0) + 1
    sc.set_evidence("model_source", source)
    sc.set_evidence("model_object_count", len(doc.objects))
    sc.set_evidence("model_objects_by_type", by_type)
    sc.set_evidence("model_bibkey", bibkey)
    sc.set_layer("docmodel", {"path": "model.docmodel.json", "format": "application/json"})
    sc.add_fact(MODEL_BUILT)
    sc.log_transition("model", _prev(sc, MODEL_BUILT), MODEL_BUILT, cost_ms,
                      f"{len(doc.objects)} objs from {source} {by_type}")
    sc.save()
    return _model_report(sc, cached=False)


def _model_report(sc: Ctx | Sidecar, cached: bool) -> str:  # type: ignore[name-defined]
    ev = sc.evidence
    by_type = ev.get("model_objects_by_type", {})
    tag = "cached model" if cached else "built model"
    lines = [
        f"{tag} {sc.local_id}: docmodel Document with "
        f"{ev.get('model_object_count', 0)} object(s) "
        f"(from the {ev.get('model_source', '?')} DOM, bibkey {ev.get('model_bibkey')})",
        f"  → {sc.blob_path('model.docmodel.json')}",
    ]
    if by_type:
        lines.append("  objects by type:")
        for t, n in sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"    {t:<12} {n}")
    lines.append("  next: load with docmodel.Document.from_dict and project")
    return "\n".join(lines)


# -- L6 projectors (offline) — run pdfdrill's REAL docops projectors ----------
#
# THE PAYOFF: lift the persisted docmodel Document and feed it to pdfdrill's
# canonical projectors (PlainText / TiddlyWiki / LLMCompact / LLMText), the SAME
# operators it runs over PDFs. These are OFFLINE (no network), so `model` is a
# safe `requires:` prerequisite (--ensure auto-runs it; `model` itself errors if
# there's no captured snapshot — never a fetch). Each is idempotent via a fact.

#: projector classname → dotted docops module (a subset of docops' own registry;
#: we reuse the loader/registry so the wiring matches pdfdrill exactly).
_PROJECTORS = {
    "tiddlers": "TiddlyWikiProjector",
    "md": "LLMCompactProjector",
    "llmtext": "PlainTextProjector",
}


def _load_document(sc: Sidecar):
    """Load the persisted docmodel Document (== docops.main.load_document)."""
    if not sc.has_blob("model.docmodel.json"):
        raise FileNotFoundError(
            f"no docmodel for {sc.local_id!r} — run `htmldrill model {sc.local_id}` "
            f"first (offline; needs a prior fetch/render).")
    from .ingest_dom import ensure_pdfdrill  # re-export of _core.ensure_pdfdrill
    ensure_pdfdrill()
    from docmodel import Document  # noqa: WPS433
    raw = sc.read_blob("model.docmodel.json") or ""
    return Document.from_dict(json.loads(raw))


def _make_projector(classname: str):
    """Instantiate a docops projector via its loader registry (no class import)."""
    from .ingest_dom import ensure_pdfdrill
    ensure_pdfdrill()
    import importlib

    from docops.base import OperatorConfig
    from docops.loader import DEFAULT_REGISTRY

    modname = DEFAULT_REGISTRY[classname]
    mod = importlib.import_module(modname)
    cls = getattr(mod, classname)
    return cls(OperatorConfig(op="projector", classname=classname))


def _run_projector(ctx: Ctx, *, name: str, classname: str, out_blob: str,
                   fact: str) -> str:
    """Shared projector body: gate on the fact (idempotent / --force), load the
    persisted Document, run the projector, write the output blob, return prose."""
    sc = Sidecar(_resolve_id(ctx), work=ctx.work)
    if sc.has(fact) and sc.has_blob(out_blob) and not ctx.force:
        return _projector_report(sc, name=name, classname=classname,
                                 out_blob=out_blob, cached=True)
    doc = _load_document(sc)
    proj = _make_projector(classname)
    t0 = time.perf_counter()
    result = proj.project(doc)
    cost_ms = (time.perf_counter() - t0) * 1000
    if not isinstance(result, str):
        result = result.decode("utf-8") if isinstance(result, bytes) else str(result)
    sc.write_blob(out_blob, result)
    sc.set_evidence(f"{name}_classname", classname)
    sc.set_evidence(f"{name}_bytes", len(result.encode("utf-8", "replace")))
    sc.set_layer(name, {"path": out_blob, "format": "text/plain"})
    sc.add_fact(fact)
    sc.log_transition(name, _prev(sc, fact), fact, cost_ms,
                      f"{classname} → {out_blob} ({len(result)} chars)")
    sc.save()
    return _projector_report(sc, name=name, classname=classname,
                             out_blob=out_blob, cached=False)


def _projector_report(sc: Sidecar, *, name: str, classname: str,
                      out_blob: str, cached: bool) -> str:
    tag = f"cached {name}" if cached else name
    nbytes = sc.get_evidence(f"{name}_bytes", "?")
    return (f"{tag} {sc.local_id}: {classname} → {out_blob} ({nbytes} bytes)\n"
            f"  → {sc.blob_path(out_blob)}")


def _tiddlers_dir(ctx: Ctx, sc: Sidecar) -> Path:
    """Per-tiddler output dir. Defaults INSIDE this target's blob dir so runs are
    isolated per target — never the shared repo-root ``tiddlers/`` (which pooled
    every target's files into the working tree). Overridable via --out /
    $HTMLDRILL_TIDDLERS."""
    explicit = ctx.out or os.environ.get("HTMLDRILL_TIDDLERS")
    return Path(explicit) if explicit else (sc.blob_dir / "tiddlers")


def cmd_tiddlers(ctx: Ctx) -> str:
    """Project the docmodel Document through pdfdrill's TiddlyWikiProjector,
    writing the importable ``tiddlers.json`` blob (a JSON array of tiddler dicts)
    plus individual ``.tid`` files into ``./tiddlers/`` (like chatdrill). OFFLINE;
    requires ``model`` (auto-ensured). Idempotent via TIDDLERS_BUILT / --force."""
    sc = Sidecar(_resolve_id(ctx), work=ctx.work)
    out_dir = _tiddlers_dir(ctx, sc)
    if sc.has(TIDDLERS_BUILT) and sc.has_blob("tiddlers.json") and not ctx.force:
        n = sc.get_evidence("tiddler_count", "?")
        return (f"cached tiddlers {sc.local_id}: {n} tiddler(s) — skipped. "
                f"--force to redo.\n  → {sc.blob_path('tiddlers.json')}")
    doc = _load_document(sc)
    proj = _make_projector("TiddlyWikiProjector")
    t0 = time.perf_counter()
    blob = proj.project(doc)                       # JSON array string
    cost_ms = (time.perf_counter() - t0) * 1000
    tids = json.loads(blob)

    sc.write_blob("tiddlers.json", blob)
    # individual .tid-ish files into the live wiki folder (trivial: one JSON each)
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in tids:
        fn = re.sub(r"[^\w.\-]+", "_", str(t.get("title", "untitled"))) + ".tid.json"
        (out_dir / fn).write_text(
            json.dumps(t, indent=2, ensure_ascii=False), encoding="utf-8")

    sc.set_evidence("tiddler_count", len(tids))
    sc.set_layer("tiddlers", {"path": "tiddlers.json", "format": "tiddlywiki/json"})
    sc.add_fact(TIDDLERS_BUILT)
    sc.log_transition("tiddlers", _prev(sc, TIDDLERS_BUILT), TIDDLERS_BUILT,
                      cost_ms, f"{len(tids)} tiddlers via TiddlyWikiProjector")
    sc.save()
    return (f"tiddlers {sc.local_id}: {len(tids)} tiddler(s) via TiddlyWikiProjector "
            f"in {cost_ms:.0f} ms\n"
            f"  → import blob: {sc.blob_path('tiddlers.json')}\n"
            f"  → per-tiddler files in {out_dir}/")


def cmd_md(ctx: Ctx) -> str:
    """Project the docmodel Document through pdfdrill's LLMCompactProjector —
    token-optimized markdown for LLM ingestion — writing ``md.md``. OFFLINE;
    requires ``model`` (auto-ensured). Idempotent via MD_BUILT / --force."""
    return _run_projector(ctx, name="md", classname="LLMCompactProjector",
                          out_blob="md.md", fact=MD_BUILT)


def cmd_llmtext(ctx: Ctx) -> str:
    """Project the docmodel Document through pdfdrill's PlainTextProjector — a
    flat, flow-ordered text dump — writing ``llm.txt``. OFFLINE; requires
    ``model`` (auto-ensured). Idempotent via LLMTEXT_BUILT / --force."""
    return _run_projector(ctx, name="llmtext", classname="PlainTextProjector",
                          out_blob="llm.txt", fact=LLMTEXT_BUILT)


# -- state / planning / diagnostics ------------------------------------------

def cmd_artifacts(ctx: Ctx) -> str:
    sc = Sidecar(_resolve_id(ctx), work=ctx.work)
    if not sc.blob_dir.exists():
        return f"{sc.local_id}: no blobs yet — run `htmldrill fetch {ctx.url}`."
    lines = [f"{sc.local_id}: blobs in {sc.blob_dir}"]
    for p in sorted(sc.blob_dir.iterdir()):
        if p.is_file():
            lines.append(f"  {p.name:<22} {p.stat().st_size:>9} bytes")
    return "\n".join(lines)


def cmd_status(ctx: Ctx) -> str:
    sc = Sidecar(_resolve_id(ctx), work=ctx.work)
    if not sc.json_path.exists():
        return (f"no sidecar for {ctx.url!r} yet — nothing built. "
                f"Run `htmldrill fetch {ctx.url}`.")
    facts = ", ".join(sorted(sc.facts)) or "(none)"
    ev = sc.evidence
    lines = [
        f"sidecar {sc.json_path}",
        f"  url:         {ev.get('url', '?')}",
        f"  facts:       {facts}",
        f"  status:      {ev.get('status', '?')}   bytes: {ev.get('bytes', '?')}   "
        f"framework: {ev.get('framework', '?')}",
        f"  transitions: {len(sc.transitions)}",
    ]
    for t in sc.transitions[-6:]:
        lines.append(f"    {t['ts']}  {t['node']}: {t['from']} → {t['to']}  "
                     f"({t['cost_ms']} ms) {t['detail']}")
    return "\n".join(lines)


def cmd_steps(ctx: Ctx) -> str:
    # `url` is optional for steps: with no target, plan against a fresh (empty)
    # sidecar so we still describe the generic prerequisite chain.
    lid = _resolve_id(ctx) if ctx.url else "(no-target)"
    sc = Sidecar(lid, work=ctx.work)
    return planner.describe(ctx.target, sc)


def cmd_doctor(ctx: Ctx) -> str:
    checks = []
    checks.append(("python >= 3.10", sys.version_info >= (3, 10), platform.python_version()))
    try:
        import html.parser  # noqa: F401
        checks.append(("stdlib html.parser", True, "ok"))
    except Exception as e:  # pragma: no cover
        checks.append(("stdlib html.parser", False, str(e)))
    try:
        import yaml  # noqa: F401
        checks.append(("pyyaml (planner manifest)", True, "ok"))
    except Exception:
        checks.append(("pyyaml (planner manifest)", False, "pip install pyyaml"))
    root = work_root(ctx.work)
    try:
        root.mkdir(parents=True, exist_ok=True)
        writable = os.access(root, os.W_OK)
    except Exception:
        writable = False
    checks.append((f"work dir writable ({root})", writable, "ok" if writable else "NOT writable"))
    chrome = R.find_chrome()
    checks.append(("headless chrome (M1 render)", bool(chrome),
                   chrome or "none — set $HTMLDRILL_CHROME (L0 still works without it)"))
    lines = ["htmldrill doctor:"]
    for name, ok, detail in checks:
        lines.append(f"  [{'✓' if ok else '✗'}] {name:<32} {detail}")
    # Chrome is optional (L0 needs none); don't let its absence fail the verdict.
    ok_all = all(ok for name, ok, _ in checks if "chrome" not in name)
    lines.append("  → all systems go." if ok_all else "  → fix the ✗ items above.")
    return "\n".join(lines)


def cmd_config(ctx: Ctx) -> str:
    return (
        "htmldrill effective config:\n"
        f"  work dir:    {work_root(ctx.work)}  (--work / $HTMLDRILL_WORK)\n"
        f"  user-agent:  {ctx.ua or F.DEFAULT_UA}  ($HTMLDRILL_UA)\n"
        f"  timeout:     {ctx.timeout}s  ($HTMLDRILL_TIMEOUT)\n"
        f"  python:      {platform.python_version()} @ {sys.executable}"
    )


# handler registry (name → fn), used by the CLI and the planner's --ensure
HANDLERS = {
    "fetch": cmd_fetch,
    "size": cmd_size,
    "headers": cmd_headers,
    "meta": cmd_meta,
    "canonical": cmd_canonical,
    "links": cmd_links,
    "jsonld": cmd_jsonld,
    "microdata": cmd_microdata,
    "opengraph": cmd_opengraph,
    "feeds": cmd_feeds,
    "outline": cmd_outline,
    "render": cmd_render,
    "dom": cmd_dom,
    "text": cmd_text,
    "screenshot": cmd_screenshot,
    "compare": cmd_compare,
    "model": cmd_model,
    "tiddlers": cmd_tiddlers,
    "md": cmd_md,
    "llmtext": cmd_llmtext,
    "artifacts": cmd_artifacts,
    "status": cmd_status,
    "steps": cmd_steps,
    "doctor": cmd_doctor,
    "config": cmd_config,
}
