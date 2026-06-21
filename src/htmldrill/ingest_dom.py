"""ingest_dom — the L4→L5 alpha-producer: HTML DOM → docmodel Document.

This is htmldrill's structural bridge into the shared intermediate representation.
:func:`build_document` takes rendered-or-static HTML plus a ``bibkey`` and walks
the block spine (``parse.html.walk_blocks``) into a real
:class:`docmodel.Document`:

  * one **stream** (``html_text``) holding one anchor per block, the block text in
    its payload (``text`` + ``_line_index`` so downstream line-rebuilders work);
  * one **DocObject** per block, typed (Heading / Paragraph / ListItem /
    CodeBlock / Table / Figure / Link), carrying a ``surface``
    :class:`Realization` whose ``start``/``end`` point at that block's anchor;
  * ``props`` mapped per type (heading ``level`` + ``caption``; figure
    ``src``/``alt``; link ``href``; table ``rows``), plus a ``flow_index`` int on
    every object so the projectors include and order them deterministically.

It imports docmodel via :func:`htmldrill._core.ensure_pdfdrill` — the single
external bridge — and matches docmodel's REAL constructors (Stream.append returns
an Anchor; Realization.stream is the stream NAME string; start/end are Anchor
objects from THAT stream).
"""
from __future__ import annotations

from typing import Optional

from ._core import ensure_pdfdrill
from .parse import html as H

#: the stream name htmldrill writes its block anchors into
HTML_STREAM = "html_text"

#: HTML block type → the docmodel/docops object-type VOCABULARY the projectors
#: actually consume (TiddlyWikiProjector iterates objects_of_type("Section"),
#: ("Paragraph"), ("Picture"), ("Table"), ("ListItem") — it has no "Heading"/
#: "Figure"/"Link" types, so those must be translated or they vanish from output.
_TYPE_MAP = {
    "Heading": "Section",      # handled specially (hierarchy), listed for clarity
    "Paragraph": "Paragraph",
    "ListItem": "ListItem",
    "CodeBlock": "Paragraph",  # no CodeBlock type downstream; keep its text as prose
    "Table": "Table",
    "Figure": "Picture",
}


def _blocks_from_tiddlywiki_store(store: list[dict]) -> list["H.Block"]:
    """Turn a TiddlyWiki content-tiddler store into the block spine: each tiddler
    becomes a Heading (its title) + a Paragraph (its text). Tiny macro/template
    shims (the <120-char ``<$latex>``/``<$link>`` shadow tiddlers) are skipped so
    real prose is what surfaces, not scaffolding."""
    blocks: list[H.Block] = []
    for t in store:
        title = str(t.get("title", "")).strip()
        text = str(t.get("text", "")).strip()
        # skip the macro/widget template tiddlers (short, all-caps shim titles like
        # FO/CIT/PARA whose body is a single <$widget …/> call)
        if len(text) < 120 and text.startswith("<$"):
            continue
        if not text:
            continue
        if title:
            blocks.append(H.Block(type="Heading", text=title, props={"level": 2}))
        blocks.append(H.Block(type="Paragraph", text=text))
    return blocks


def _props_for(block: H.Block, flow_index: int) -> dict:
    """Type-specific props (+ flow_index) for a non-Section block's DocObject."""
    p: dict = {"flow_index": flow_index, "text": block.text}
    t = block.type
    if t == "Figure":
        p["caption"] = block.props.get("alt", "")
        p["src"] = block.props.get("src", "")
        p["alt"] = block.props.get("alt", "")
    elif t == "Link":
        p["href"] = block.props.get("href", "")
    elif t == "Table":
        p["rows"] = block.props.get("rows", [])
    return p


def build_document(html: str, bibkey: str, local_id: Optional[str] = None):
    """Build and return a docmodel :class:`Document` from page HTML.

    ``bibkey`` lands in ``doc.meta['bibkey']`` (load-bearing for the projectors).
    ``local_id`` is recorded in meta for provenance. Returns the Document; the
    caller persists it with ``json.dump(doc.to_dict())``.
    """
    ensure_pdfdrill()
    from docmodel import Document, DocObject, Realization  # noqa: WPS433

    doc = Document()
    doc.meta["bibkey"] = bibkey
    if local_id:
        doc.meta["local_id"] = local_id
    doc.meta["source"] = "htmldrill"
    # htmldrill has no pages; declare one so meta['num_pages'] is sane downstream.
    doc.meta["num_pages"] = 1

    stream = doc.ensure_stream(HTML_STREAM)
    blocks = H.walk_blocks(html)

    # Fallback for JS-rendered single-file apps whose static body is empty: a
    # TiddlyWiki carries all its content in an inline JSON tiddler store, which we
    # can read offline (no browser). If the structural walk found essentially
    # nothing but a store exists, build the spine from the store instead — turning
    # the silent-empty-output case into a real document.
    if sum(len(b.text) for b in blocks) < 200:
        store = H.tiddlywiki_store(html)
        if store:
            blocks = _blocks_from_tiddlywiki_store(store)

    # Section hierarchy: an HTML heading at level L closes every open section of
    # level >= L (its siblings/deeper kin) and nests under the nearest shallower
    # one. Headings become Section objects (the projector's unit); intervening
    # blocks attach to the current section via props['parent_section'] so the
    # TiddlyWiki/markdown projectors render the document's real structure rather
    # than a flat list. This is the vocabulary alignment the M3 verifier flagged.
    section_stack: list[tuple[int, object]] = []   # (level, Section DocObject)
    current_section_id: Optional[str] = None
    section_seq = 0

    for i, block in enumerate(blocks):
        # one anchor per block, carrying its text; payload keys mirror what the
        # projectors read off mathpix_lines (text + a line index for ordering).
        anchor = stream.append(text=block.text, type=block.type, _line_index=i)

        if block.type == "Heading":
            level = int(block.props.get("level", 1))
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            parent_id = section_stack[-1][1].id if section_stack else None
            section_seq += 1
            obj = DocObject(type="Section", props={
                "flow_index": i, "text": block.text, "level": level,
                "caption": block.text, "section_number": str(section_seq),
            })
            if parent_id:
                obj.parent = parent_id          # projector reads Section.parent
            obj.add_realization(
                Realization(stream=HTML_STREAM, start=anchor, end=anchor, role="surface"))
            doc.add(obj)
            section_stack.append((level, obj))
            current_section_id = obj.id
            continue

        mapped = _TYPE_MAP.get(block.type, block.type)
        props = _props_for(block, i)
        if current_section_id:
            props["parent_section"] = current_section_id   # nest under the heading
        obj = DocObject(type=mapped, props=props)
        obj.add_realization(
            Realization(stream=HTML_STREAM, start=anchor, end=anchor, role="surface"))
        doc.add(obj)

    doc.meta["title"] = next(
        (b.text for b in blocks if b.type == "Heading"), bibkey)
    doc.meta["block_count"] = len(blocks)
    doc.meta["section_count"] = section_seq
    return doc
