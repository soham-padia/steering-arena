"""Supabase session verification shared by the public demo and the admin view.

/generate requires ANY verified account; /admin/* additionally requires the email to be
on ADMIN_EMAILS. Verification asks Supabase rather than validating the JWT locally: no
key material to manage, nothing to get wrong about signatures or expiry, and a revoked
session stops working at once.
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
        raise AuthError("Sign in to generate.")
    if not settings.supabase_url or not settings.supabase_anon_key:
        raise AuthError("Sign-in is not configured on this server.")

    import httpx

    try:
        r = httpx.get(
            f"{settings.supabase_url.rstrip('/')}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}", "apikey": settings.supabase_anon_key},
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


def admin_emails(settings) -> set[str]:
    return {e.strip().lower() for e in (settings.admin_emails or "").split(",") if e.strip()}


def require_admin(token: str, settings) -> str:
    """Verified email that is also on the admin allowlist. A valid account is not
    authorization — anyone can make one."""
    allow = admin_emails(settings)
    if not allow:
        raise AuthError("Admin access is not configured.")
    user = verify_token(token, settings)
    if user["email"] not in allow:
        _log.warning("admin access denied for %s", user["email"] or "(no email)")
        raise AuthError("That account is not on the admin list.")
    return user["email"]
