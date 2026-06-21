"""Corpus regression tests — REAL local pages, offline, deterministic.

These guard the real-world failures hardened in M-fix: the structural walk
dropping all non-anchor text (interactive pages whose content lives in
<div>/<button>/<label>), the TiddlyWiki single-file SPA whose content lives only
in an inline JSON store, and the link/meta/status edge cases. Every page is a
SMALL real file copied into tests/corpus/ — no network, no mocks, no Chrome.

Runnable two ways:
    python3 -m pytest tests/test_corpus.py
    PYTHONPATH=src python3 tests/test_corpus.py
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from htmldrill.parse import html as H          # noqa: E402
from htmldrill import commands as C            # noqa: E402
from htmldrill.sidecar import Sidecar          # noqa: E402

CORPUS = Path(__file__).resolve().parent / "corpus"
AWK = CORPUS / "awk-compare.html"
BITFIELDS = CORPUS / "bitfields.html"
WKOLBE = CORPUS / "wkolbe.html"
TW = CORPUS / "tw-mini.html"


def _pdfdrill_available() -> bool:
    try:
        from htmldrill._core import ensure_pdfdrill
        ensure_pdfdrill()
        import docops.loader  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


# -- the data-loss bug: generic-container text must NOT be dropped --------------

def test_awk_walk_captures_div_button_text():
    """The AWK page wraps all body text in <div>/<button>/<textarea> — none in
    <p>. The structural walk must still recover that interactive UI text."""
    raw = AWK.read_text(encoding="utf-8")
    blocks = H.walk_blocks(raw)
    joined = " ".join(b.text for b in blocks)
    # real labels/buttons that used to vanish (only the 4 <a> links survived)
    assert "AWK Script" in joined
    assert "Run AWK" in joined
    assert "Example 1" in joined
    # and the 4 real anchors are still there
    assert sum(1 for b in blocks if b.type == "Link") == 4
    # block text should now be comparable to the visible-text extraction, not ~0
    assert sum(len(b.text) for b in blocks) > 100


def test_bitfields_form_text_present():
    """ADS-B decoder: form labels/buttons must survive into the block spine."""
    raw = BITFIELDS.read_text(encoding="utf-8")
    blocks = H.walk_blocks(raw)
    joined = " ".join(b.text for b in blocks)
    assert "ADS-B" in joined or "Decode" in joined
    assert sum(len(b.text) for b in blocks) > 100


def test_wkolbe_links_internal_classified():
    """A local file:// site's relative nav link must classify as internal, not
    fall into 'other' (the empty-base_host bug)."""
    raw = WKOLBE.read_text(encoding="utf-8")
    c = H.collect(raw)
    base = WKOLBE.as_uri()
    links = H.extract_links(c, base_url=base)
    assert links["internal"], "expected at least one internal link on the local site"


# -- the TiddlyWiki SPA: content lives only in the JSON store ------------------

def test_tiddlywiki_store_recovered_offline():
    raw = TW.read_text(encoding="utf-8")
    store = H.tiddlywiki_store(raw)
    titles = {t["title"] for t in store}
    assert "Introduction" in titles and "Methods" in titles
    assert "$:/core" not in titles               # system tiddler dropped


# -- model + projectors must yield REAL text on these pages -------------------

def _fetch_model(work: str, path: Path) -> C.Ctx:
    ctx = C.Ctx(url=str(path), work=work)
    C.cmd_fetch(ctx)
    C.cmd_model(ctx)
    return ctx


def test_awk_model_has_objects():
    raw = AWK.read_text(encoding="utf-8")
    from htmldrill.ingest_dom import build_document
    if not _pdfdrill_available():
        print("    (skip: no pdfdrill docmodel)", end="")
        return
    doc = build_document(raw, bibkey="AWK")
    assert len(doc.objects) > 5, "model dropped the page body"


def test_tiddlywiki_model_recovers_store_content():
    if not _pdfdrill_available():
        print("    (skip: no pdfdrill docmodel)", end="")
        return
    raw = TW.read_text(encoding="utf-8")
    from htmldrill.ingest_dom import build_document
    doc = build_document(raw, bibkey="TW")
    # 2 content tiddlers → 2 Section + 2 Paragraph (PARA shim + $:/core dropped)
    assert len(doc.objects) >= 4
    texts = " ".join(str(o.props.get("text", "")) for o in doc.objects.values())
    assert "quick brown fox" in texts


def test_projector_yields_real_text_on_awk():
    if not _pdfdrill_available():
        print("    (skip: no pdfdrill docmodel)", end="")
        return
    with tempfile.TemporaryDirectory() as work:
        ctx = _fetch_model(work, AWK)
        C.cmd_llmtext(ctx)
        sc = Sidecar(C._resolve_id(ctx), work=work)
        txt = sc.read_blob("llm.txt") or ""
        assert "AWK Script" in txt or "Run AWK" in txt, "projector emitted no body text"


# -- status / meta edge cases that used to crash or mislead -------------------

def test_status_by_path_does_not_crash():
    """status on an absolute local path used to raise NotImplementedError
    (non-relative glob); it must now resolve the id and report state."""
    with tempfile.TemporaryDirectory() as work:
        ctx = C.Ctx(url=str(AWK), work=work)
        C.cmd_fetch(ctx)
        out = C.cmd_status(ctx)
        assert "facts:" in out and "FETCHED" in out


def test_meta_reports_title_without_meta_tags():
    """A page with a <title> but zero <meta> tags must still report the title."""
    html = "<html><head><title>Only A Title</title></head><body>x</body></html>"
    with tempfile.TemporaryDirectory() as work:
        p = Path(work) / "titleonly.html"
        p.write_text(html, encoding="utf-8")
        ctx = C.Ctx(url=str(p), work=work)
        C.cmd_fetch(ctx)
        out = C.cmd_meta(ctx)
        assert "Only A Title" in out
        assert "no <meta> tags or <title>" not in out


def test_framework_no_false_positive_on_plain_text():
    """A big plain page with the words 'vue'/'angular' in body text must NOT be
    flagged as a framework (the substring false-positive bug)."""
    from htmldrill.commands import _guess_framework
    html = "<html><body><p>" + ("vue angular react " * 5000) + "</p></body></html>"
    assert _guess_framework(html) == "none detected"


def test_tiddlers_write_to_isolated_dir():
    """Per-tiddler files must land inside the target's blob dir, not the CWD."""
    if not _pdfdrill_available():
        print("    (skip: no pdfdrill docmodel)", end="")
        return
    with tempfile.TemporaryDirectory() as work:
        ctx = _fetch_model(work, TW)
        out = C.cmd_tiddlers(ctx)
        sc = Sidecar(C._resolve_id(ctx), work=work)
        tdir = sc.blob_dir / "tiddlers"
        assert tdir.is_dir() and any(tdir.iterdir()), "no per-tiddler files written"
        assert str(tdir) in out


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{passed}/{len(fns)} passed")
    return 0 if passed == len(fns) else 1


if __name__ == "__main__":
    raise SystemExit(_run_all())
