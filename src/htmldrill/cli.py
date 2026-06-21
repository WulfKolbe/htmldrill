"""htmldrill CLI — flat, prose-returning (PDFDRILL/CHATDRILL convention).

Shallow-first: start with the free L0 tier on a captured snapshot —

  htmldrill fetch   <url>                  # the only network step; snapshots raw HTML
  htmldrill size    <url>                  # bytes/tags/framework + render-needed verdict
  htmldrill meta    <url>                  # <meta> + <title>
  htmldrill links   <url>                  # anchors + the invisible-URL killer case
  htmldrill jsonld  <url>                  # structured data lifted straight from markup
  htmldrill outline <url>                  # h1–h6 tree
  htmldrill status  <url> | steps <cmd> <url>
  htmldrill doctor | config

Every command returns prose; quote it back to the user. Snapshot commands need a
prior `fetch` (they refuse to touch the network themselves).
"""
from __future__ import annotations

import argparse
import sys

from . import planner
from .commands import HANDLERS, Ctx
from .sidecar import Sidecar
from .sources import fetch as F


def _ctx(args) -> Ctx:
    return Ctx(
        url=getattr(args, "url", None),
        work=getattr(args, "work", None),
        force=getattr(args, "force", False),
        as_json=getattr(args, "json", False),
        target=getattr(args, "target", None),
        out=getattr(args, "out", None),
        ua=getattr(args, "ua", None),
        timeout=getattr(args, "timeout", F.DEFAULT_TIMEOUT),
        window=getattr(args, "window", "1280,900"),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="htmldrill",
        description="Token-economical drill-down toolkit for HTML / web documents.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def work_arg(p):
        p.add_argument("--work", help="artifact root (default: $HTMLDRILL_WORK or ./drills)")

    def url_arg(p):
        p.add_argument("url", help="URL, local .html path, or an existing sidecar id prefix")

    # fetch — the only network command
    p = sub.add_parser("fetch", help="fetch the URL; snapshot raw HTML + headers")
    url_arg(p); work_arg(p)
    p.add_argument("--force", action="store_true", help="re-fetch even if FETCHED")
    p.add_argument("--ua", help="override User-Agent")
    p.add_argument("--timeout", type=float, default=F.DEFAULT_TIMEOUT)
    p.set_defaults(cmd="fetch")

    # snapshot introspection commands (uniform: <url> [--work] [--force] [--ensure])
    SNAP = {
        "size": "bytes/tags/framework + static-vs-render verdict",
        "headers": "captured HTTP response headers",
        "meta": "<meta> tags + <title>",
        "canonical": "<link rel=canonical> / og:url",
        "links": "anchors split internal/external + invisible-URL killer case",
        "jsonld": "application/ld+json blocks (@types + keys)",
        "microdata": "itemscope/itemtype/itemprop items",
        "opengraph": "og:* / twitter:* meta",
        "feeds": "RSS/Atom feed links",
        "outline": "h1–h6 heading tree",
    }
    for name, helptext in SNAP.items():
        p = sub.add_parser(name, help=helptext)
        url_arg(p); work_arg(p)
        p.add_argument("--force", action="store_true", help=f"recompute even if cached")
        p.add_argument("--ensure", action="store_true",
                       help="auto-run missing OFFLINE prerequisites first")
        p.set_defaults(cmd=name)

    # render — the headless escalation (network, like fetch; never auto-ensured)
    p = sub.add_parser("render", help="headless-render the page; snapshot DOM + screenshot")
    url_arg(p); work_arg(p)
    p.add_argument("--force", action="store_true", help="re-render even if RENDERED")
    p.add_argument("--timeout", type=float, default=45.0)
    p.add_argument("--window", default="1280,900", help="viewport WxH (default 1280,900)")
    p.set_defaults(cmd="render")

    # render-derived views (operate on the rendered/static snapshot, no network)
    RENDER_VIEWS = {
        "dom": "rendered DOM stats vs the static markup",
        "text": "visible text (rendered DOM preferred, static fallback)",
        "screenshot": "report the captured screenshot path",
        "compare": "static | rendered | screenshot fidelity table",
    }
    for name, helptext in RENDER_VIEWS.items():
        p = sub.add_parser(name, help=helptext)
        url_arg(p); work_arg(p)
        p.add_argument("--force", action="store_true", help="recompute even if cached")
        p.add_argument("--ensure", action="store_true",
                       help="auto-run missing OFFLINE prerequisites first")
        p.set_defaults(cmd=name)

    # model — the offline L5 docmodel ingestion (no network)
    p = sub.add_parser("model", help="build a shared docmodel Document from the captured DOM")
    url_arg(p); work_arg(p)
    p.add_argument("--force", action="store_true", help="rebuild even if MODEL_BUILT")
    p.add_argument("--ensure", action="store_true",
                   help="auto-run missing OFFLINE prerequisites first")
    p.set_defaults(cmd="model")

    # projectors (offline) — run pdfdrill's REAL docops projectors over the model
    PROJECTORS = {
        "tiddlers": "TiddlyWikiProjector → tiddlers.json (+ ./tiddlers/ files)",
        "md": "LLMCompactProjector → md.md (token-optimized markdown)",
        "llmtext": "PlainTextProjector → llm.txt (flat flow-ordered text)",
    }
    for name, helptext in PROJECTORS.items():
        p = sub.add_parser(name, help=helptext)
        url_arg(p); work_arg(p)
        p.add_argument("--force", action="store_true", help="re-project even if built")
        p.add_argument("--ensure", action="store_true",
                       help="auto-run missing OFFLINE prerequisites first (model)")
        p.set_defaults(cmd=name)

    # artifacts / status — state views
    p = sub.add_parser("artifacts", help="list the blobs captured for this target")
    url_arg(p); work_arg(p); p.set_defaults(cmd="artifacts")

    p = sub.add_parser("status", help="show the sidecar state")
    url_arg(p); work_arg(p); p.set_defaults(cmd="status")

    # steps — planner view. `url` is optional: with only a command we describe the
    # generic chain; with a url/id too we describe it against that target's state.
    p = sub.add_parser("steps", help="show the prerequisite chain for a command")
    p.add_argument("target", help="the command to plan for (e.g. size)")
    p.add_argument("url", nargs="?", default=None,
                   help="optional URL / local path / sidecar id to plan against")
    work_arg(p); p.set_defaults(cmd="steps")

    # doctor / config — diagnostics (no target)
    p = sub.add_parser("doctor", help="environment self-check")
    work_arg(p); p.set_defaults(cmd="doctor")

    p = sub.add_parser("config", help="print the effective configuration")
    work_arg(p); p.set_defaults(cmd="config")

    args = ap.parse_args(argv)
    ctx = _ctx(args)

    try:
        # --ensure: run missing OFFLINE prerequisites before the target. Network
        # steps (fetch) are never declared as requires, so this can't hit the net.
        if getattr(args, "ensure", False):
            tid = F.local_id_for(ctx.url)
            sc = Sidecar(tid, work=ctx.work)
            planner.ensure(args.cmd, sc, HANDLERS, ctx)
        print(HANDLERS[args.cmd](ctx))
        return 0
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
