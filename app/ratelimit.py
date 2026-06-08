"""IP hashing + rate limiting. Limits are load-bearing: they cap the maintainer's
NDIF quota (PROJECT_SPEC.md §9), not just abuse. Counts come from the submissions
table's timestamps, so limits survive a Space restart (state lives in Supabase)."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from app.errors import RateLimited


def hash_ip(ip: str, salt: str) -> str:
    """Salted SHA-256 of a client IP. Raw IPs are never stored."""
    return hashlib.sha256(f"{salt}:{ip}".encode("utf-8")).hexdigest()


def check_rate_limits(db, ip_hash: str, settings, now: datetime | None = None) -> None:
    """Raise RateLimited if the per-IP (minute/day) or global daily cap is hit."""
    now = now or datetime.now(timezone.utc)

    minute_ago = (now - timedelta(seconds=60)).isoformat()
    day_ago = (now - timedelta(days=1)).isoformat()

    if db.count_ip_since(ip_hash, minute_ago) >= settings.rate_per_min:
        raise RateLimited("Too many submissions — please wait a minute and try again.")
    if db.count_ip_since(ip_hash, day_ago) >= settings.rate_per_day:
        raise RateLimited("Daily submission limit reached for your connection.")
    if db.count_global_since(day_ago) >= settings.global_per_day:
        raise RateLimited("The arena is at capacity for today — please come back tomorrow.")
