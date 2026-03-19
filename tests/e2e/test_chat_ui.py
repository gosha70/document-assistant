"""Playwright tests — Chat UI (manual test plan section 6)."""

from playwright.sync_api import Page, expect

BASE_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# 6.1  Page loads
# ---------------------------------------------------------------------------


def test_chat_page_loads(page: Page):
    """6.1 Root redirects to /static/index.html; core UI elements present."""
    page.goto(BASE_URL)
    expect(page).to_have_url(f"{BASE_URL}/static/index.html")

    # Branding title
    expect(page.locator(".header-title")).to_contain_text("D.O.T")

    # Input area
    expect(page.locator("#msg-input")).to_be_visible()

    # Send button
    expect(page.locator("#send-btn")).to_be_visible()

    # Attach button (the visible button; the file input is hidden)
    expect(page.locator("#attach-btn")).to_be_visible()


# ---------------------------------------------------------------------------
# 6.3  Shift+Enter inserts newline; Enter submits
# ---------------------------------------------------------------------------


def test_shift_enter_newline(page: Page):
    """6.3 Shift+Enter adds a newline but does NOT send the message."""
    page.goto(BASE_URL)
    textarea = page.locator("#msg-input")
    textarea.click()
    textarea.type("line one")
    textarea.press("Shift+Enter")
    textarea.type("line two")

    value = textarea.input_value()
    assert "\n" in value, "Shift+Enter should insert a newline"

    # No user-sent message bubble should have appeared
    assert page.locator(".user-message").count() == 0, "Shift+Enter must not submit the form"


# ---------------------------------------------------------------------------
# 6.5  File upload via UI — supported type
# ---------------------------------------------------------------------------


def test_file_upload_supported_type(page: Page, tmp_path):
    """6.5 Attaching a .txt file triggers ingestion and shows a status message."""
    import uuid

    # Use a unique filename so the server never sees it as a duplicate
    # across repeated test runs against the same persistent collection.
    unique_name = f"e2e_sample_{uuid.uuid4().hex[:8]}.txt"
    sample = tmp_path / unique_name
    sample.write_text("Python is a programming language created by Guido van Rossum.")

    page.goto(BASE_URL)

    with page.expect_file_chooser() as fc_info:
        page.locator("#attach-btn").click()
    fc_info.value.set_files(str(sample))

    # If the server detects a duplicate (file with this name existed from a
    # prior run) the chat UI shows a modal — dismiss it automatically.
    page.wait_for_timeout(800)
    dup_modal = page.locator("#dup-modal")
    if dup_modal.evaluate("el => getComputedStyle(el).display !== 'none'"):
        page.locator("#dup-add-btn").click()

    # First ingest loads the embedding model — allow up to 60s
    page.wait_for_selector(".doc-status-done, .doc-status-error", timeout=60_000)
    assert page.locator(".doc-status-done").count() > 0, "Expected a successful ingestion status bubble for a .txt file"


# ---------------------------------------------------------------------------
# 6.6  File upload — unsupported type
# ---------------------------------------------------------------------------


def test_file_upload_unsupported_type(page: Page, tmp_path):
    """6.6 Attaching a .png file shows a doc-status-error bubble."""
    bad_file = tmp_path / "photo.png"
    bad_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    page.goto(BASE_URL)

    with page.expect_file_chooser() as fc_info:
        page.locator("#attach-btn").click()
    fc_info.value.set_files(str(bad_file))

    page.wait_for_selector(".doc-status-error, .error-bubble", timeout=10_000)
    assert (
        page.locator(".doc-status-error, .error-bubble").count() > 0
    ), "Expected an error status bubble for an unsupported file type"


# ---------------------------------------------------------------------------
# 6.8  Auth token persists in localStorage
# ---------------------------------------------------------------------------


def test_auth_token_saved_in_localstorage(page: Page):
    """6.8 Pasting a token via the settings gear saves it to localStorage."""
    page.goto(BASE_URL)

    # Open settings modal
    page.locator("#settings-open-btn").click()
    expect(page.locator("#settings-modal")).to_be_visible()

    # Enter token and save
    page.locator("#token-input").fill("test-bearer-token-123")
    page.locator("#settings-save-btn").click()

    # Modal should close
    expect(page.locator("#settings-modal")).not_to_be_visible()

    # Value must be persisted under the app's localStorage key
    stored = page.evaluate("localStorage.getItem('dot_token')")
    assert stored == "test-bearer-token-123", f"Token not persisted in localStorage['dot_token']; got: {stored!r}"


# ---------------------------------------------------------------------------
# 6.9  Multiple messages stay visible
# ---------------------------------------------------------------------------


def test_multiple_messages_visible(page: Page):
    """6.9 Sending multiple messages keeps them all visible in the chat box."""
    page.goto(BASE_URL)
    textarea = page.locator("#msg-input")
    send = page.locator("#send-btn")

    for i in range(3):
        textarea.fill(f"Test message {i + 1}")
        send.click()
        page.wait_for_timeout(200)

    user_messages = page.locator(".user-message")
    assert user_messages.count() >= 3, f"Expected >= 3 user messages, found {user_messages.count()}"
