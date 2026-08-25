"""Admin log view — authorization only. The page shows text strangers typed into the
public demo, so every one of these is about refusing to show it."""

from types import SimpleNamespace

import pytest

from app import userauth as adminauth


def settings(**over):
    """Stub with the same browser_key() indirection as app.config.Settings, so these
    tests exercise the real lookup (publishable key preferred, anon accepted)."""
    base = dict(admin_emails="Boss@Example.com", supabase_url="https://p.supabase.co",
                supabase_publishable_key="", supabase_anon_key="anon-key")
    base.update(over)
    ns = SimpleNamespace(**base)
    ns.browser_key = lambda: ns.supabase_publishable_key or ns.supabase_anon_key
    return ns


class FakeResponse:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload


def patch_user(monkeypatch, status, payload=None):
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse(status, payload))


def test_emails_are_parsed_case_insensitively():
    assert adminauth.admin_emails(settings(admin_emails=" A@x.com , B@Y.com ")) == {"a@x.com", "b@y.com"}


# ── fail closed ──────────────────────────────────────────────

def test_no_allowlist_denies_everyone(monkeypatch):
    patch_user(monkeypatch, 200, {"id": "u", "email": "boss@example.com"})
    with pytest.raises(adminauth.AuthError):
        adminauth.require_admin("tok", settings(admin_emails=""))


def test_missing_token_denied():
    with pytest.raises(adminauth.AuthError):
        adminauth.require_admin("", settings())


def test_any_verified_account_may_generate(monkeypatch):
    """/generate needs only a real session — the allowlist is an ADMIN gate, not a
    barrier to using the public demo."""
    patch_user(monkeypatch, 200, {"id": "uid-1", "email": "stranger@elsewhere.com"})
    assert adminauth.verify_token("tok", settings())["id"] == "uid-1"


def test_generate_still_refuses_an_unverified_session(monkeypatch):
    patch_user(monkeypatch, 401, {})
    with pytest.raises(adminauth.AuthError):
        adminauth.verify_token("bad", settings())


def test_user_hash_is_stable_and_not_the_id():
    h1 = adminauth.user_hash("uid-1", "salt")
    h2 = adminauth.user_hash("uid-1", "salt")
    assert h1 == h2 and "uid-1" not in h1
    assert h1 != adminauth.user_hash("uid-2", "salt")


def test_unconfigured_supabase_denied():
    with pytest.raises(adminauth.AuthError):
        adminauth.require_admin("tok", settings(supabase_anon_key="", supabase_publishable_key=""))


def test_supabase_unreachable_denies_rather_than_admits(monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "get", boom)
    with pytest.raises(adminauth.AuthError):
        adminauth.require_admin("tok", settings())


def test_rejected_token_denied(monkeypatch):
    patch_user(monkeypatch, 401, {})
    with pytest.raises(adminauth.AuthError):
        adminauth.require_admin("expired", settings())


def test_valid_token_outside_the_allowlist_denied(monkeypatch):
    """A real Supabase account is not authorization — anyone can make one."""
    patch_user(monkeypatch, 200, {"id": "u", "email": "stranger@elsewhere.com"})
    with pytest.raises(adminauth.AuthError):
        adminauth.require_admin("tok", settings())


# ── the one accept path ──────────────────────────────────────

def test_allowlisted_email_accepted_case_insensitively(monkeypatch):
    patch_user(monkeypatch, 200, {"id": "u", "email": "BOSS@example.com"})
    assert adminauth.require_admin("tok", settings()) == "boss@example.com"


# ── what may leave the server ────────────────────────────────

def test_admin_response_fields_exclude_identifiers():
    from app.main import ADMIN_FIELDS
    assert "ip_hash" not in ADMIN_FIELDS and "prompt_hash" not in ADMIN_FIELDS
    assert "user_hash" not in ADMIN_FIELDS
    assert {"prompt", "continuation", "arm", "research_consent"} <= set(ADMIN_FIELDS)


def test_public_feed_shows_a_handle_but_never_an_account():
    """The feed is world-readable. A player-chosen handle is meant to be there; anything
    that could tie a generation back to a person is not."""
    from app.main import PUBLIC_FIELDS
    assert set(PUBLIC_FIELDS) == {"created_at", "arm", "handle", "prompt", "continuation"}
    for leaky in ("ip_hash", "prompt_hash", "user_hash", "email", "consent_version"):
        assert leaky not in PUBLIC_FIELDS


def test_generation_handles_follow_the_leaderboard_rules():
    """One handle policy for the whole site, not two."""
    from app.errors import ValidationError
    from app.submission import validate_handle
    assert validate_handle(" AAA ") == "AAA"
    for bad in ("", "x" * 33, "<script>", "nope@example.com"):
        with pytest.raises(ValidationError):
            validate_handle(bad)


def test_either_key_name_configures_sign_in():
    """Supabase renamed anon -> publishable; both names must work, new one preferred."""
    from app.config import Settings
    only_new = Settings(supabase_publishable_key="sb_publishable_x", supabase_anon_key="")
    only_old = Settings(supabase_publishable_key="", supabase_anon_key="eyJlegacy")
    both = Settings(supabase_publishable_key="sb_publishable_x", supabase_anon_key="eyJlegacy")
    assert only_new.browser_key() == "sb_publishable_x"
    assert only_old.browser_key() == "eyJlegacy"
    assert both.browser_key() == "sb_publishable_x"
    assert Settings(supabase_publishable_key="", supabase_anon_key="").browser_key() == ""
