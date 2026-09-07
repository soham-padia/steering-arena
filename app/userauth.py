"""Supabase session verification for the public demo.

/generate requires ANY verified account — the account is a rate-limiting identity, not
a permission. There is nothing here that grants more than that: the /admin/* allowlist
gate was removed on 2026-09-07 along with the endpoints it protected.

Verification asks Supabase rather than validating the JWT locally: no key material to
manage, nothing to get wrong about signatures or expiry, and a revoked session stops
working at once.
"""

from __future__ import annotations

import hashlib
import logging

_log = logging.getLogger("steering_arena")


class AuthError(Exception):
    """Message is safe to show the client."""


def bearer(request) -> str:
    return (request.headers.get("authorization") or "").removeprefix("Bearer ").strip()


def verify_token(token: str, settings) -> dict:
    """{'id', 'email'} for a valid Supabase session, else AuthError. Fails closed on a
    missing token, missing config, an unreachable Supabase, or a rejected token."""
    if not token:
        raise AuthError("Sign in to continue.")
    if not settings.supabase_url or not settings.browser_key():
        raise AuthError("Sign-in is not configured on this server.")

    import httpx

    try:
        r = httpx.get(
            f"{settings.supabase_url.rstrip('/')}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}", "apikey": settings.browser_key()},
            timeout=10.0,
        )
    except Exception as exc:  # noqa: BLE001 — a network failure must never authorize
        _log.warning("token verification could not reach Supabase: %r", exc)
        raise AuthError("Could not verify your session — try again.") from exc

    if r.status_code != 200:
        raise AuthError("Your session has expired — sign in again.")

    data = r.json()
    uid = (data.get("id") or "").strip()
    if not uid:
        raise AuthError("Your session has expired — sign in again.")
    return {"id": uid, "email": (data.get("email") or "").strip().lower()}


def user_hash(user_id: str, salt: str) -> str:
    """Salted hash of the Supabase user id. Rate limiting needs a stable per-person key;
    it does not need to know who the person is, and the public feed shows no author."""
    return hashlib.sha256(f"{salt}:user:{user_id}".encode()).hexdigest()


