"""The public deployment must expose no admin surface at all.

`app/main.py` mounts `web/` as a catch-all at "/", so any file in `web/` is
world-reachable by construction, and any route registered before that mount wins
over it. Both facts are load-bearing here.

These tests pin the boundary in three independent ways, because each one alone
can be undone by an ordinary mistake:

  1. the four admin paths 404 when ADMIN_API is off (the behaviour that matters);
  2. no admin file sits inside `web/` (so the 404 does not depend on route order);
  3. /auth/config carries the publishable key and never a secret one.

Nothing here tests authorization — tests/test_admin_surface.py is about whether
the routes EXIST. tests/test_auth.py covers who may use them once they do, and
that is unaffected by the flag: require_admin still verifies the session against
Supabase and checks the allowlist.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

ADMIN_PATHS = ["/admin.html", "/admin.css", "/admin/generations", "/admin/config"]


@pytest.fixture
def client():
    """A client for the app as this environment configures it.

    Note what this fixture does NOT do: it cannot turn ADMIN_API off. The routes
    are registered at import time from `settings.admin_api`, and `app.main` is
    already imported by the time a test runs, so monkeypatching the environment
    here would change nothing and only look like it worked. Instead the tests skip
    when the flag is on, and say so.
    """
    import app.main as main
    if main.settings.admin_api:
        pytest.skip("ADMIN_API is enabled in this environment (.env or shell); the "
                    "route tests describe the deployed configuration, where it is unset")
    return TestClient(main.app)


# ── 1. the paths are absent ──────────────────────────────────

@pytest.mark.parametrize("path", ADMIN_PATHS)
def test_admin_paths_are_absent_by_default(client, path):
    """404, not 401. A 401 would confirm the endpoint exists."""
    assert client.get(path).status_code == 404, (
        f"{path} is reachable on a default-configured server"
    )


def test_admin_hide_is_indistinguishable_from_a_path_that_never_existed(client):
    """POST /admin/hide must not answer differently from POST /anything-else.

    It returns 405, not 404, and that is fine: the static mount at "/" catches every
    unmatched POST, so `POST /nonsense` returns 405 too. Asserting the two are equal
    is a stronger claim than asserting a particular number — it says the response
    carries no information about whether an admin endpoint is behind it. A 401 here
    would fail this test, and should, because 401 confirms the route exists.
    """
    admin = client.post("/admin/hide", json={"created_at": "x", "hidden": True})
    control = client.post("/definitely-not-a-route", json={"created_at": "x"})
    assert admin.status_code == control.status_code
    assert admin.status_code not in (200, 401, 403), (
        f"POST /admin/hide answered {admin.status_code}, which tells a caller the "
        f"endpoint is there"
    )


def test_the_old_config_path_is_gone_not_aliased(client):
    """/admin/config moved to /auth/config and was deliberately not aliased.

    The payload was never the problem; the path was. Keeping an alias would keep
    the thing that made a public endpoint read as an admin leak.
    """
    assert client.get("/admin/config").status_code == 404


# ── 2. nothing admin-shaped is inside the served directory ───

def test_no_admin_file_is_served_from_web():
    """`web/` is mounted as a catch-all, so a file here needs no route to leak.

    This is the check that does not depend on route ordering. If someone moves the
    mount above the route definitions, or reverts the move, test 1 can start
    passing for the wrong reason while this one still fails.
    """
    stray = sorted(p.name for p in WEB.rglob("admin*") if p.is_file())
    assert not stray, (
        f"{stray} sits inside the public static mount. The admin view belongs in "
        f"tools/admin/, which app/main.py serves only when ADMIN_API is set."
    )


def test_the_admin_view_still_exists_outside_the_mount():
    """Severing it from the public site must not mean losing it."""
    for name in ("admin.html", "admin.css", "README.md"):
        assert (ROOT / "tools" / "admin" / name).is_file(), f"tools/admin/{name} missing"


# ── 3. what /auth/config may say ─────────────────────────────

def test_auth_config_is_public_and_carries_no_secret(client):
    """The publishable key is meant to ship to browsers; the secret key is not.

    A browser cannot start a Supabase session without the publishable key, so this
    endpoint is unauthenticated on purpose. What keeps that safe is row-level
    security, not secrecy — see the route's own docstring.
    """
    r = client.get("/auth/config")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"enabled", "supabase_url", "anon_key"}, (
        f"unexpected fields in /auth/config: {sorted(body)}"
    )
    blob = " ".join(str(v) for v in body.values()).lower()
    for marker in ("service_role", "sb_secret_", "secret_key"):
        assert marker not in blob, f"/auth/config leaked something matching {marker!r}"


def test_auth_config_does_not_advertise_admin(client):
    """Nothing public needs to know whether an admin allowlist is configured."""
    assert "admin_enabled" not in client.get("/auth/config").json()


def test_the_frontend_asks_for_the_new_path():
    """A stale fetch URL in auth.js breaks sign-in silently for every visitor."""
    js = (WEB / "auth.js").read_text()
    assert 'fetch("/auth/config")' in js
    # the old path may still be NAMED in a comment explaining the move; what must not
    # survive is a call to it
    assert 'fetch("/admin/config")' not in js
    # and every page loading it must bust the cache, or a cached copy keeps
    # requesting the route that no longer exists
    for page in list(WEB.glob("*.html")) + [ROOT / "tools/admin/admin.html"]:
        text = page.read_text()
        if "auth.js" in text:
            assert "auth.js?v=3" in text, f"{page.name} loads a stale auth.js version"
