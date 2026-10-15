"""The exam web app (plan §8): a Starlette app serving the search page, a small JSON API, and
the library's exam-copy videos and PDFs.

Endpoints (all GET):

* ``/``                         the search page (``static/index.html``)
* ``/api/status``               library, index fingerprint, and what the no-AI guard checked
* ``/api/courses``              courses with file counts and weeks, for the filters
* ``/api/search?q=...``         results; optional ``course``, ``kind`` (comma-separated),
                                ``week_from``, ``week_to``
* ``/api/passage/{id}``         one passage in full, with its neighbours' ids
* ``/files/{source_id}``        the source's exam-copy video or PDF (HTTP Range supported)
* ``/static/...``, ``/pdfjs/...``  the page's own assets and the bundled PDF.js viewer

Endpoints are ``async`` and share one read-only SQLite connection, so every query runs on
the event loop's thread, one at a time (each takes milliseconds).
"""

from __future__ import annotations

import hashlib
import html
import mimetypes
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from studybuddy_exam import __version__, guard
from studybuddy_exam.index_format import IndexInfo
from studybuddy_exam.query import parse
from studybuddy_exam.search import KINDS, LIMIT, MARK_CLOSE, MARK_OPEN, Filters, Hit, Searcher

STATIC = Path(__file__).parent / "static"

# The browser may load only what this app serves: no CDN, no outside requests at all.
# PDF.js needs blob:/data: URLs for fonts and images, a worker, and WebAssembly decoders.
CSP = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self' 'wasm-unsafe-eval'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: blob:",
        "font-src 'self' data:",
        "media-src 'self' blob:",
        "worker-src 'self' blob:",
        "connect-src 'self'",
        "frame-src 'self'",
        "frame-ancestors 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
    ]
)
_SECURITY_HEADERS = [
    (b"content-security-policy", CSP.encode()),
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-resource-policy", b"same-origin"),
]


class SecurityHeaders:
    """Adds the CSP and related headers to every response, and no-store to API responses."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        is_api = scope["path"].startswith("/api/")

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(_SECURITY_HEADERS)
                if is_api:
                    headers.append((b"cache-control", b"no-store"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def create_app(conn: sqlite3.Connection, info: IndexInfo, *, host: str = "127.0.0.1", port: int = 8765) -> Starlette:
    searcher = Searcher(conn)
    library_root = info.library_root
    index_sha = file_sha256(info.path)

    async def homepage(request: Request) -> Response:
        return FileResponse(STATIC / "index.html", media_type="text/html")

    async def status(request: Request) -> Response:
        blocked = guard.loaded_blocked_modules()
        return JSONResponse(
            {
                "app": {"name": "studybuddy-exam", "version": __version__},
                "library": {"name": info.meta.get("library_name") or library_root.name, "root": str(library_root)},
                "index": {
                    "path": str(info.path),
                    "sha256": index_sha,
                    "built_at": info.meta.get("built_at"),
                    "courses": int(info.meta.get("courses", 0)),
                    "sources": int(info.meta.get("sources", 0)),
                    "passages": int(info.meta.get("passages", 0)),
                },
                "guard": {
                    "installed": guard.installed(),
                    "ai_modules_loaded": blocked,
                    "network": "loopback only" if guard.installed() else "not restricted",
                    "listening": f"{host}:{port}",
                },
            }
        )

    async def courses(request: Request) -> Response:
        return JSONResponse(searcher.courses())

    async def search(request: Request) -> Response:
        params = request.query_params
        try:
            filters = Filters(
                course=params.get("course") or None,
                kinds=_kinds(params.get("kind")),
                week_from=_int(params.get("week_from")),
                week_to=_int(params.get("week_to")),
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        result = searcher.search(params.get("q", "")[:500], filters)
        terms = result.query.highlight_terms
        if result.corrected and result.suggestion:
            terms = parse(result.suggestion).highlight_terms  # the hits are for the corrected question
        return JSONResponse(
            {
                "query": result.query.text,
                "terms": terms,
                "suggestion": result.suggestion,
                "corrected": result.corrected,
                "count": len(result.hits),
                "limited": len(result.hits) >= LIMIT,  # there may be more than the ones shown
                "results": [_hit_json(h, terms) for h in result.hits],
            }
        )

    async def passage(request: Request) -> Response:
        row = searcher.passage(request.path_params["passage_id"])
        if row is None:
            return JSONResponse({"error": "no such passage"}, status_code=404)
        previous, following = searcher.neighbours(row["source_id"], row["ord"])
        data = {k: row[k] for k in row.keys() if k not in ("view_path",)}
        hit = Hit(
            passage_id=row["id"], source_id=row["source_id"], course_slug=row["course_slug"],
            course_title=row["course_title"], kind=row["kind"], title=row["source_title"], week=row["week"],
            date=row["date"], t_start_ms=row["t_start_ms"], t_end_ms=row["t_end_ms"], slide_no=row["slide_no"],
            page_no=row["page_no"], heading_path=row["heading_path"], origin=row["origin"], snippet="",
            match="", has_view=row["view_path"] is not None,
        )
        terms = parse(request.query_params.get("q", "")).highlight_terms
        data.update(
            {
                "title": row["source_title"],
                "heading_parts": _heading_parts(row["heading_path"]),
                "has_view": hit.has_view,
                "file_url": f"/files/{row['source_id']}" if hit.has_view else None,
                "open_url": _open_url(hit, terms),
                "previous_id": previous,
                "next_id": following,
            }
        )
        return JSONResponse(data)

    async def source_file(request: Request) -> Response:
        row = searcher.source(request.path_params["source_id"])
        if row is None or not row["view_path"]:
            return JSONResponse({"error": "no file to show for this source"}, status_code=404)
        path = (library_root / row["view_path"]).resolve()
        if not path.is_relative_to(library_root.resolve()) or not path.is_file():
            return JSONResponse({"error": "the file is missing from the library"}, status_code=404)
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type, content_disposition_type="inline")

    routes = [
        Route("/", homepage),
        Route("/api/status", status),
        Route("/api/courses", courses),
        Route("/api/search", search),
        Route("/api/passage/{passage_id:int}", passage),
        Route("/files/{source_id:int}", source_file),
        Mount("/static", StaticFiles(directory=STATIC), name="static"),
    ]
    if (STATIC / "pdfjs").is_dir():
        routes.append(Mount("/pdfjs", StaticFiles(directory=STATIC / "pdfjs"), name="pdfjs"))
    middleware = [
        Middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"]),
        Middleware(SecurityHeaders),
    ]
    return Starlette(routes=routes, middleware=middleware)


def _kinds(value: str | None) -> frozenset[str] | None:
    if not value:
        return None
    kinds = frozenset(k.strip() for k in value.split(",") if k.strip())
    if unknown := kinds - set(KINDS):
        raise ValueError(f"unknown kind(s): {', '.join(sorted(unknown))}")
    return kinds or None


def _int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"not a number: {value!r}") from None


def _hit_json(hit: Hit, terms: list[str]) -> dict[str, Any]:
    data = asdict(hit)
    data["snippet_html"] = html.escape(data.pop("snippet")).replace(MARK_OPEN, "<mark>").replace(MARK_CLOSE, "</mark>")
    data["heading_parts"] = _heading_parts(hit.heading_path)
    data["file_url"] = f"/files/{hit.source_id}" if hit.has_view else None
    data["open_url"] = _open_url(hit, terms)
    return data


def _heading_parts(path: str | None) -> list[str]:
    """"Chapter 2 > 2.1 Glycolysis" -> ["Chapter 2", "2.1 Glycolysis"] (the builder joins with " > ")."""
    return [part for part in (path or "").split(" > ") if part]


def _open_url(hit: Hit, terms: list[str]) -> str | None:
    """Where clicking the result goes: the video at its time, or the PDF viewer at its page."""
    if not hit.has_view:
        return None
    if hit.kind == "video":
        start = max(0, (hit.t_start_ms or 0) // 1000)
        return f"/files/{hit.source_id}#t={start}"
    page = hit.slide_no or hit.page_no or 1
    fragment = urlencode({"page": page, "search": " ".join(terms)}, quote_via=quote)
    return f"/pdfjs/web/viewer.html?file={quote(f'/files/{hit.source_id}', safe='')}#{fragment}"
