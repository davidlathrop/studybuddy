"""The web app: API shapes, file serving with Range, and the browser-side lockdown."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
from starlette.testclient import TestClient

from studybuddy_exam.index_format import open_index
from studybuddy_exam.server import CSP, create_app


@pytest.fixture(scope="module")
def client(index_file: Path) -> TestClient:
    conn, info = open_index(index_file)
    with TestClient(create_app(conn, info, port=8765), base_url="http://127.0.0.1:8765") as c:
        yield c


def test_status(client: TestClient, index_file: Path) -> None:
    data = client.get("/api/status").json()
    assert data["library"]["name"] == "Test Library"
    assert data["index"]["passages"] == 11 and len(data["index"]["sha256"]) == 64
    assert data["guard"]["ai_modules_loaded"] == []
    assert data["guard"]["listening"] == "127.0.0.1:8765"


def test_search_json(client: TestClient) -> None:
    data = client.get("/api/search", params={"q": "krebs cycle"}).json()
    assert data["terms"] == ["krebs", "cycle"] and data["count"] == len(data["results"]) > 0
    video = next(r for r in data["results"] if r["kind"] == "video")
    assert video["open_url"] == f"/files/{video['source_id']}#t={video['t_start_ms'] // 1000}"
    slide = next(r for r in data["results"] if r["kind"] == "pptx")
    url = urlsplit(slide["open_url"])
    assert url.path == "/pdfjs/web/viewer.html"
    assert parse_qs(url.query)["file"] == [f"/files/{slide['source_id']}"]
    fragment = parse_qs(unquote(url.fragment))
    assert fragment["page"] == ["6"] and fragment["search"] == ["krebs cycle"]
    assert "<mark>Krebs</mark>" in slide["snippet_html"]


def test_snippets_are_escaped(client: TestClient, index_file: Path) -> None:
    data = client.get("/api/search", params={"q": "cost-push"}).json()
    assert "<mark>" in data["results"][0]["snippet_html"]
    assert "<script" not in str(data)


def test_text_only_sources_have_no_links(client: TestClient) -> None:
    hit = client.get("/api/search", params={"q": "tariffs"}).json()["results"][0]
    assert hit["has_view"] is False and hit["open_url"] is None and hit["file_url"] is None


def test_corrected_search_highlights_the_corrected_words(client: TestClient) -> None:
    data = client.get("/api/search", params={"q": "fotosynthesis"}).json()
    assert data["corrected"] and data["suggestion"] == "photosynthesis" and data["terms"] == ["photosynthesis"]


def test_search_filters_and_errors(client: TestClient) -> None:
    data = client.get("/api/search", params={"q": "elasticity tariffs", "course": "econ201", "kind": "docx"}).json()
    assert {r["source_id"] for r in data["results"]} == {12}
    assert client.get("/api/search", params={"q": "x", "kind": "exe"}).status_code == 400
    assert client.get("/api/search", params={"q": "x", "week_from": "three"}).status_code == 400
    assert client.get("/api/search").json()["results"] == []


def test_courses_and_passage(client: TestClient) -> None:
    assert [c["slug"] for c in client.get("/api/courses").json()] == ["bio101", "econ201"]
    passage = client.get("/api/passage/2").json()
    assert passage["previous_id"] == 1 and passage["next_id"] == 3 and passage["file_url"] == "/files/10"
    assert "view_path" not in passage
    assert client.get("/api/passage/999").status_code == 404


def test_files_support_range_requests(client: TestClient, library: Path) -> None:
    whole = client.get("/files/10")
    assert whole.status_code == 200 and whole.headers["content-type"] == "video/mp4"
    part = client.get("/files/10", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206 and part.headers["content-range"] == f"bytes 100-199/{len(whole.content)}"
    assert part.content == whole.content[100:200]
    assert client.get("/files/11").headers["content-type"] == "application/pdf"


def test_files_only_for_sources_with_a_view(client: TestClient) -> None:
    assert client.get("/files/12").status_code == 404  # text-only source
    assert client.get("/files/999").status_code == 404
    assert client.get("/files/..%2F..%2Fetc%2Fpasswd").status_code == 404  # not an id: no route


def test_every_response_is_locked_down(client: TestClient) -> None:
    for path in ("/", "/api/status", "/files/10", "/static/index.html"):
        headers = client.get(path).headers
        assert headers["content-security-policy"] == CSP, path
        assert headers["x-content-type-options"] == "nosniff"
    assert "default-src 'self'" in CSP and "http" not in CSP
    assert client.get("/api/status").headers["cache-control"] == "no-store"


def test_foreign_host_names_are_refused(client: TestClient) -> None:
    assert client.get("/api/status", headers={"host": "evil.example"}).status_code == 400
    assert client.get("/api/status", headers={"host": "localhost:8765"}).status_code == 200


def test_heading_parts_limits_and_passage_links(client: TestClient) -> None:
    data = client.get("/api/search", params={"q": "tariffs"}).json()
    assert data["limited"] is False
    assert data["results"][0]["heading_parts"] == ["Inflation", "Cost-push"]
    passage = client.get("/api/passage/6", params={"q": "krebs"}).json()
    assert passage["title"] == "Week 4 slides" and passage["heading_parts"] == []
    assert passage["open_url"].startswith("/pdfjs/web/viewer.html?file=%2Ffiles%2F11#page=6&search=krebs")
