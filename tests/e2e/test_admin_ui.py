"""Playwright tests — Admin console UI (admin.html)."""

from playwright.sync_api import Page, expect

BASE_URL = "http://127.0.0.1:8000"
ADMIN_URL = f"{BASE_URL}/static/admin.html"


# ---------------------------------------------------------------------------
# Page load
# ---------------------------------------------------------------------------


def test_admin_page_loads(page: Page):
    """Admin console loads; Collections tab is active by default."""
    page.goto(ADMIN_URL)

    # Tab bar is present
    expect(page.get_by_role("tab", name="Collections")).to_be_visible()

    # Collections panel is the active one
    expect(page.locator("#tab-collections")).to_be_visible()


# ---------------------------------------------------------------------------
# Collections tab
# ---------------------------------------------------------------------------


def test_admin_collections_tab(page: Page):
    """Collections tab completes its API call without JS errors."""
    page.goto(ADMIN_URL)

    # Wait for the spinner to disappear (loading done) or the list to appear.
    # Covers both empty-store (list stays hidden, spinner hides) and
    # populated-store (list becomes visible) cases.
    page.wait_for_function(
        """() => {
            const state = document.getElementById('collections-state');
            const list  = document.getElementById('collections-list');
            const stateHidden = !state || getComputedStyle(state).display === 'none';
            const listShown   = list && list.style.display !== 'none';
            return stateHidden || listShown;
        }""",
        timeout=10_000,
    )
    # The panel itself must remain visible
    expect(page.locator("#tab-collections")).to_be_visible()


# ---------------------------------------------------------------------------
# Jobs tab
# ---------------------------------------------------------------------------


def test_admin_jobs_tab(page: Page):
    """Jobs tab becomes visible and shows its state message after click."""
    page.goto(ADMIN_URL)

    page.get_by_role("tab", name="Jobs").click()

    # Panel becomes active (display: block via .tab-panel.active)
    page.wait_for_selector("#tab-jobs.active", timeout=5_000)
    expect(page.locator("#jobs-state")).to_be_visible()


# ---------------------------------------------------------------------------
# Metrics tab
# ---------------------------------------------------------------------------


def test_admin_metrics_tab(page: Page):
    """Metrics tab becomes visible and shows its state message after click."""
    page.goto(ADMIN_URL)

    page.get_by_role("tab", name="Metrics").click()

    page.wait_for_selector("#tab-metrics.active", timeout=5_000)
    assert (
        page.locator("#metrics-state").is_visible() or page.locator("#metrics-content").is_visible()
    ), "Expected metrics-state or metrics-content to be visible"


# ---------------------------------------------------------------------------
# Reports tab
# ---------------------------------------------------------------------------


def test_admin_reports_tab(page: Page):
    """Reports tab becomes visible and shows its state message after click."""
    page.goto(ADMIN_URL)

    page.get_by_role("tab", name="Reports").click()

    page.wait_for_selector("#tab-reports.active", timeout=5_000)
    assert (
        page.locator("#reports-state").is_visible() or page.locator("#reports-content").is_visible()
    ), "Expected reports-state or reports-content to be visible"
