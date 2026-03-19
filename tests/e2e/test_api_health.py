"""Playwright tests — API health, status, and OpenAPI docs (sections 1, 21).

Pure API tests (health, status, openapi.json) use httpx directly so they are
not subject to the browser request context timeout or interference from
preceding UI operations that may keep the server event loop busy.
"""

import httpx
from playwright.sync_api import Page, expect

BASE_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# Section 1: Health and Status  (httpx — no browser needed)
# ---------------------------------------------------------------------------


def test_health_endpoint():
    """1.1 /health returns 200 with status ok."""
    resp = httpx.get(f"{BASE_URL}/health", timeout=10)
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"


def test_status_endpoint():
    """1.2 /status returns expected fields."""
    resp = httpx.get(f"{BASE_URL}/status", timeout=10)
    assert resp.status_code == 200
    body = resp.json()
    assert "app_name" in body
    assert "vectorstore_backend" in body
    assert "document_count" in body
    assert body["app_name"] != ""


# ---------------------------------------------------------------------------
# Section 21: OpenAPI Documentation
# ---------------------------------------------------------------------------


def test_swagger_ui_loads(page: Page):
    """21.1 Swagger UI renders with all endpoints."""
    page.goto(f"{BASE_URL}/docs")
    expect(page).to_have_title("Document Assistant (D.O.T.) - Swagger UI")
    page.wait_for_selector(".opblock", timeout=10_000)
    count = page.locator(".opblock").count()
    assert count >= 5, f"Expected >= 5 endpoints in Swagger UI, found {count}"


def test_openapi_json():
    """21.2 /openapi.json returns valid schema with all expected paths."""
    resp = httpx.get(f"{BASE_URL}/openapi.json", timeout=10)
    assert resp.status_code == 200
    paths = resp.json().get("paths", {})
    for expected in ["/health", "/status", "/ingest", "/chat", "/chat/stream"]:
        assert expected in paths, f"Missing path {expected!r} in OpenAPI schema"


def test_redoc_loads(page: Page):
    """21.3 ReDoc renders correctly."""
    page.goto(f"{BASE_URL}/redoc")
    page.wait_for_selector("redoc-wrap, .menu-content, [data-role='search']", timeout=15_000)
    expect(page.locator("redoc-wrap, .redoc-wrap")).to_be_visible()
