"""There is no admin surface — not a disabled one, not a flagged one, none.

The web admin view and the `/admin/*` endpoints were removed on 2026-09-07. An
earlier version of this file tested that they were absent *unless* `ADMIN_API` was
set. That flag is gone too, because it was the weak part: a checkbox in the Space's
environment variables would have republished the whole surface, and nothing in the
deployment would have complained.

So these tests assert absence unconditionally. They also check the two things that
made the surface easy to republish by accident:

  1. `app/main.py` mounts `web/` as a catch-all at "/", so a file dropped in that
     directory is world-reachable with no route and no error;
  2. the code that existed only to gate the admin routes is gone, so it cannot be
     wired back up to a new route by half-remembering that it was safe.

`tests/test_auth.py` still covers `verify_token`, which `/generate` uses. Removing
the allowlist did not remove sign-in.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

GONE_PATHS = ["/admin.html", "/admin.css", "/admin/config",
              "/admin/generations", "/admin/hide"]


@pytest.fixture
def client():
    import app.main as main
    return TestClient(main.app)


# ── 1. nothing answers on an admin path ──────────────────────

@pytest.mark.parametrize("path", GONE_PATHS)
def test_no_admin_path_answers(client, path):
    """404 for a GET. Never 200, and never 401 — a 401 confirms a route is there."""
    assert client.get(path).status_code == 404, f"{path} is reachable"


def test_admin_hide_post_is_indistinguishable_from_a_path_that_never_existed(client):
    """The removed endpoint was a POST, so a GET alone does not prove it is gone.

    It answers 405 rather than 404 because the static mount catches every unmatched
    POST — `POST /definitely-not-a-route` answers 405 too. Asserting the two are
    equal is stronger than asserting a number: it says the response carries no
    information about whether an endpoint is behind it.
    """
    admin = client.post("/admin/hide", json={"created_at": "x", "hidden": True})
    control = client.post("/definitely-not-a-route", json={"created_at": "x"})
    assert admin.status_code == control.status_code
    assert admin.status_code not in (200, 401, 403)


def test_no_route_in_the_app_is_registered_under_admin(client):
    """Absence at the app level, not just at the HTTP level.

    A path can 404 because a route is missing OR because a handler chose to return
    404. This distinguishes them: the route table itself must contain no /admin
    path, so there is nothing to accidentally re-enable.
    """
    import app.main as main
    admin_routes = [r.path for r in main.app.routes
                    if getattr(r, "path", "").startswith("/admin")]
    assert admin_routes == [], f"routes still registered: {admin_routes}"


# ── 2. nothing admin-shaped can be served by the static mount ─

def test_no_admin_file_exists_anywhere_in_the_repo():
    """`web/` is a catch-all mount, so a file there needs no route to be public.

    Checked across the whole tree rather than only `web/`, because the previous fix
    moved these files to `tools/admin/` and kept them — which left a page that a
    single misconfigured variable would serve. They are deleted now; git history has
    them.
    """
    stray = sorted(
        str(p.relative_to(ROOT)) for p in ROOT.rglob("admin*")
        if p.is_file()
        and ".git/" not in str(p)
        and "steering_arena/" not in str(p.relative_to(ROOT))
        and p.suffix in {".html", ".css", ".js"}
    )
    assert not stray, f"admin front-end files still present: {stray}"


# ── 3. the gate code is gone, not merely unused ───────────────

def test_the_allowlist_gate_is_removed():
    """Leaving require_admin behind invites wiring it to a new route on the belief
    that it was the safe part. It was sound, but the surface it guarded is what got
    removed, so the guard goes with it."""
    from app import userauth
    for name in ("require_admin", "admin_emails"):
        assert not hasattr(userauth, name), f"userauth.{name} still exists"


def test_the_admin_settings_are_removed():
    """A setting that reads as if it controls access, but does not, is worse than no
    setting. ADMIN_API in particular was one checkbox from republishing everything."""
    from app.config import Settings
    fields = set(Settings.model_fields)
    assert "admin_api" not in fields
    assert "admin_emails" not in fields


def test_only_one_column_projection_can_leave_the_server():
    """ADMIN_FIELDS was the wider of two projections and is gone; PUBLIC_FIELDS is
    now the only column list served over HTTP, which is less to audit."""
    import app.main as main
    assert not hasattr(main, "ADMIN_FIELDS")
    assert set(main.PUBLIC_FIELDS) == {"created_at", "arm", "handle", "prompt",
                                       "continuation"}


# ── 4. moderation survived the removal ───────────────────────

def test_moderation_is_still_possible_without_the_endpoint():
    """`hidden` filters the public feed and nothing else in the app could set it, so
    deleting /admin/hide without a replacement would have made an abusive row
    permanent."""
    script = ROOT / "scripts" / "moderate_generation.py"
    assert script.is_file(), "scripts/moderate_generation.py is missing"
    src = script.read_text()
    for token in ("hide", "unhide", "supabase_service_key", "generation_events"):
        assert token in src, f"{token!r} missing from the moderation script"


# ── 5. what the public site still exposes, on purpose ────────

def test_auth_config_is_public_and_carries_no_secret(client):
    """A browser cannot start a Supabase session without the publishable key, so
    this endpoint is unauthenticated by design. What keeps it safe is row-level
    security, not secrecy — see docs/how-to/verify-the-public-surface.md."""
    r = client.get("/auth/config")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"enabled", "supabase_url", "anon_key"}, sorted(body)
    blob = " ".join(str(v) for v in body.values()).lower()
    for marker in ("service_role", "sb_secret_", "secret_key"):
        assert marker not in blob, f"/auth/config leaked something matching {marker!r}"


def test_auth_config_does_not_advertise_admin(client):
    assert "admin_enabled" not in client.get("/auth/config").json()


def test_the_frontend_asks_for_the_new_path():
    """A stale fetch URL in auth.js breaks sign-in silently for every visitor."""
    js = (WEB / "auth.js").read_text()
    assert 'fetch("/auth/config")' in js
    # the old path may still be named in a comment explaining the move; a CALL to it
    # must not survive
    assert 'fetch("/admin/config")' not in js
    for page in WEB.glob("*.html"):
        text = page.read_text()
        if "auth.js" in text:
            assert "auth.js?v=3" in text, f"{page.name} loads a stale auth.js version"
