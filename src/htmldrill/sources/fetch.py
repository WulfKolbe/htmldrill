"""L0 transport — fetch raw HTML, stdlib only (``urllib``), zero dependencies.

The ONLY network boundary in M0. Everything downstream operates on the snapshot
this writes (``raw.html`` + ``headers.json`` blobs), never the live network, so
re-runs are deterministic and replay against the captured bytes.

A URL is not a filesystem path, so the *local id* — the sidecar key — is derived
deterministically from the normalized URL: a readable slug + a short blake2b hash
(collision-resistant, stable across runs). Local files and ``file://`` URLs are
accepted too, which keeps the whole L0 tier testable with no network.
"""
from __future__ import annotations

import gzip
import hashlib
import os
import re
import zlib
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DEFAULT_UA = os.environ.get(
    "HTMLDRILL_UA", "htmldrill/0.1 (+https://github.com/WulfKolbe/htmldrill)")
DEFAULT_TIMEOUT = float(os.environ.get("HTMLDRILL_TIMEOUT", "20"))

#: Hard cap on bytes we will read/keep (live fetch and local file). A multi-hundred
#: -MB page degrades gracefully (clear error) instead of hanging/OOM-ing the parser.
MAX_BYTES = int(os.environ.get("HTMLDRILL_MAX_BYTES", str(256 * 1024 * 1024)))


def _decompress(body: bytes, encoding: str) -> bytes:
    """Decode a Content-Encoding'd body (gzip / deflate / br). Best-effort: if a
    decoder is unavailable or the stream is not actually compressed, return the
    bytes unchanged rather than corrupting the snapshot."""
    enc = (encoding or "").lower().strip()
    try:
        if enc in ("gzip", "x-gzip"):
            return gzip.decompress(body)
        if enc == "deflate":
            try:
                return zlib.decompress(body)
            except zlib.error:
                return zlib.decompress(body, -zlib.MAX_WBITS)  # raw deflate
        if enc == "br":
            import brotli  # type: ignore  # optional dependency
            return brotli.decompress(body)
    except Exception:
        return body
    return body


def normalize_url(url: str) -> str:
    """Light normalization for stable id derivation (not full canonicalization)."""
    u = url.strip()
    if "://" not in u and not u.startswith("/") and "." in u.split("/")[0]:
        u = "https://" + u            # bare host like example.com → https://
    return u


def is_local(url: str) -> bool:
    if url.startswith("file://"):
        return True
    if "://" in url:
        return False
    return True                        # no scheme → treat as a local path


def local_id_for(url: str) -> str:
    """Deterministic sidecar key: <slug>-<hash8> from the normalized URL/path."""
    norm = normalize_url(url)
    p = urlparse(norm)
    if p.scheme in ("http", "https"):
        base = (p.netloc + p.path).rstrip("/")
    else:
        base = Path(norm.replace("file://", "")).name or norm
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", base).strip("-").lower()[:40] or "page"
    h = hashlib.blake2b(norm.encode("utf-8"), digest_size=4).hexdigest()
    return f"{slug}-{h}"


class FetchResult:
    def __init__(self, url: str, final_url: str, status: int,
                 headers: dict, body: bytes, content_type: str):
        self.url = url
        self.final_url = final_url
        self.status = status
        self.headers = headers
        self.body = body
        self.content_type = content_type

    @property
    def text(self) -> str:
        # Charset resolution order: Content-Type header → <meta charset> in the
        # first chunk of the body → utf-8 (lenient). This recovers pages that
        # declare their encoding only in markup (very common).
        enc = None
        m = re.search(r"charset=([\w\-]+)", self.content_type, re.I)
        if m:
            enc = m.group(1)
        if not enc:
            head = self.body[:4096]
            m2 = re.search(rb"charset=[\"']?([\w\-]+)", head, re.I)
            if m2:
                enc = m2.group(1).decode("ascii", "ignore")
        enc = enc or "utf-8"
        try:
            return self.body.decode(enc, errors="replace")
        except (LookupError, TypeError):
            return self.body.decode("utf-8", errors="replace")


def fetch(url: str, timeout: float = DEFAULT_TIMEOUT,
          ua: Optional[str] = None) -> FetchResult:
    """Fetch http(s) URL (following redirects) or read a local file / file://."""
    norm = normalize_url(url)
    if is_local(norm):
        path = Path(norm.replace("file://", "")).expanduser().resolve()
        size = path.stat().st_size
        if size > MAX_BYTES:
            raise ValueError(
                f"{path} is {size} bytes — exceeds HTMLDRILL_MAX_BYTES ({MAX_BYTES}); "
                f"raise the limit to process it.")
        body = path.read_bytes()
        ctype = "text/html; charset=utf-8"
        return FetchResult(url, path.as_uri(), 200,
                           {"Content-Type": ctype, "Content-Length": str(len(body))},
                           body, ctype)
    req = Request(norm, headers={
        "User-Agent": ua or DEFAULT_UA,
        "Accept": "text/html,application/xhtml+xml,*/*",
        # Advertise the encodings we can actually undo — otherwise some origins
        # send gzip anyway and urllib hands back the raw compressed bytes.
        "Accept-Encoding": "gzip, deflate",
    })
    with urlopen(req, timeout=timeout) as resp:        # noqa: S310 — http(s) only above
        # Bounded read: never pull more than MAX_BYTES + 1 (the +1 detects overflow).
        body = resp.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise ValueError(
                f"{norm} response exceeds HTMLDRILL_MAX_BYTES ({MAX_BYTES}); "
                f"raise the limit to process it.")
        headers = {k: v for k, v in resp.headers.items()}
        body = _decompress(body, resp.headers.get("Content-Encoding", ""))
        ctype = resp.headers.get("Content-Type", "text/html")
        return FetchResult(url, resp.geturl(), resp.status, headers, body, ctype)
