"""The search page and the vendored PDF.js work under the server's strict CSP (plan §8).

No browser: these read the files. The page may load only scripts and styles that this app
serves, with no inline script, no inline event handlers and no URLs to other hosts, because
the exam runs with Wi-Fi off and the CSP would block them anyway.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "studybuddy_exam" / "static"
PDFJS = STATIC / "pdfjs"
PAGE_FILES = ("index.html", "app.js", "app.css")

URL = re.compile(r"""(?i)\bhttps?://[^\s"'<>()`]+""")
PROTOCOL_RELATIVE = re.compile(r"""(?i)(?:src|href)\s*=\s*["']?//|url\(\s*["']?//|import\s*\(?\s*["']//|from\s*["']//""")
LOCAL_HOSTS = {"127.0.0.1", "localhost"}


class _Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[dict[str, str | None]] = []
        self.inline_scripts: list[str] = []
        self.links: list[dict[str, str | None]] = []
        self.handlers: list[str] = []
        self.styles: list[str] = []
        self._in_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        self.handlers += [f"<{tag} {name}>" for name, _ in attrs if name.startswith("on")]
        if tag == "script":
            self.scripts.append(attributes)
            self._in_script = "src" not in attributes
        elif tag == "link":
            self.links.append(attributes)
        elif tag == "style":
            self.styles.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_script and data.strip():
            self.inline_scripts.append(data.strip())


@pytest.fixture(scope="module")
def page() -> _Page:
    parser = _Page()
    parser.feed((STATIC / "index.html").read_text(encoding="utf-8"))
    return parser


def _local_file(url: str) -> Path:
    """The file under static/ that a page URL such as /static/app.js is served from."""
    parts = urlsplit(url)
    assert not parts.scheme and not parts.netloc, f"{url} is not a local path"
    path = parts.path
    for prefix, folder in (("/static/", STATIC), ("/pdfjs/", PDFJS)):
        if path.startswith(prefix):
            return folder / path.removeprefix(prefix)
    raise AssertionError(f"{url} is not served by the exam app")


def test_no_inline_scripts_or_event_handlers(page: _Page) -> None:
    assert page.scripts, "the page loads no script"
    assert page.inline_scripts == [], "inline <script> blocks are blocked by the CSP"
    assert all(s.get("src") for s in page.scripts)
    assert page.handlers == [], f"inline event handlers are blocked by the CSP: {page.handlers}"
    assert page.styles == [], "keep styles in app.css"


def test_scripts_and_stylesheets_are_local_files(page: _Page) -> None:
    urls = [s["src"] for s in page.scripts] + [link["href"] for link in page.links if link.get("href")]
    assert any(u.endswith("app.js") for u in urls) and any(u.endswith("app.css") for u in urls)
    for url in urls:
        assert url and _local_file(url).is_file(), f"{url} does not exist under static/"


@pytest.mark.parametrize("name", PAGE_FILES)
def test_no_urls_to_other_hosts(name: str) -> None:
    text = (STATIC / name).read_text(encoding="utf-8")
    foreign = [u for u in URL.findall(text) if urlsplit(u).hostname not in LOCAL_HOSTS]
    foreign += PROTOCOL_RELATIVE.findall(text)
    assert foreign == [], f"{name} refers to other hosts: {foreign}"


def test_app_script_avoids_eval() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    for banned in ("eval(", "new Function(", "setTimeout(\"", "setInterval(\"", "document.write("):
        assert banned not in script


def test_vendored_pdfjs_is_complete_and_trimmed() -> None:
    for required in ("web/viewer.html", "web/viewer.mjs", "web/viewer.css", "build/pdf.mjs", "build/pdf.worker.mjs",
                     "LICENSE", "STUDYBUDDY-VENDORED.txt", "web/locale/locale.json", "web/locale/en-US/viewer.ftl"):
        assert (PDFJS / required).is_file(), f"pdfjs/{required} is missing"
    files = [p for p in PDFJS.rglob("*") if p.is_file()]
    assert not [p for p in files if p.suffix == ".map"], "source maps should be trimmed"
    assert not (PDFJS / "web" / "compressed.tracemonkey-pldi-09.pdf").exists(), "the sample PDF should be trimmed"
    locales = {p.name for p in (PDFJS / "web" / "locale").iterdir() if p.is_dir()}
    assert locales == {"en-US"}
    assert '"en-us":"en-US/viewer.ftl"' in (PDFJS / "web" / "locale" / "locale.json").read_text().replace(" ", "")
    assert sum(p.stat().st_size for p in files) < 10 * 1024 * 1024, "PDF.js should stay well under 10 MB"
    assert "6.4.299" in (PDFJS / "STUDYBUDDY-VENDORED.txt").read_text()
