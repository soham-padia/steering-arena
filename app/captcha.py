"""Cloudflare Turnstile verification for /submit (security audit H1).

CAPTCHA is the load-bearing guard against rotating-IP bots draining the NDIF
quota — IP rate limits alone don't stop distributed abuse. It's optional: when
`turnstile_secret` is unset (dev/demo), verification is disabled and returns True.
"""

from __future__ import annotations

VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


def verify(token: str, secret: str, ip: str | None = None) -> bool:
    if not secret:
        return True  # CAPTCHA disabled (no secret configured)
    if not token:
        return False  # enabled but the client sent no token
    import httpx

    data = {"secret": secret, "response": token}
    if ip:
        data["remoteip"] = ip
    try:
        r = httpx.post(VERIFY_URL, data=data, timeout=10.0)
        return bool(r.json().get("success"))
    except Exception:  # noqa: BLE001 — a verification failure must fail closed
        return False
