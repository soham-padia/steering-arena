"""The /submit pipeline as a pure-ish service function with injected dependencies
(db, tokenizer, scorer), so it's fully unit-testable without Supabase or NDIF.
See PROJECT_SPEC.md §7."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Callable

from app.errors import DuplicateError, ScoringUnavailable, ValidationError
from app.ratelimit import check_rate_limits

HANDLE_RE = re.compile(r"^[\w \-.]{1,32}$")  # letters, digits, _, space, -, .
MAX_SEQUENCE_CHARS = 6000  # ~1000 tokens of headroom (token budget is enforced separately)


def normalize_key(sequence: str) -> str:
    """Dedup/cache key: lowercased, whitespace-collapsed."""
    return re.sub(r"\s+", " ", sequence.strip().lower())


def validate_handle(handle: str) -> str:
    handle = (handle or "").strip()
    if not HANDLE_RE.match(handle):
        raise ValidationError("Handle must be 1–32 chars: letters, digits, spaces, _ - . only.")
    return handle


def validate_sequence(sequence: str) -> str:
    if sequence is None:
        raise ValidationError("Sequence is required.")
    # Reject control chars (except normal whitespace); require valid text.
    if any(ord(c) < 32 and c not in "\t\n\r" for c in sequence):
        raise ValidationError("Sequence contains invalid control characters.")
    stripped = sequence.strip()
    if not stripped:
        raise ValidationError("Sequence cannot be empty.")
    if len(sequence) > MAX_SEQUENCE_CHARS:
        raise ValidationError(f"Sequence too long (max {MAX_SEQUENCE_CHARS} characters).")
    return sequence


def process_submission(
    *,
    handle: str,
    sequence: str,
    ip_hash: str,
    db,
    count_tokens: Callable[[str], int],
    score_fn: Callable[[str], float],
    settings,
    now: datetime | None = None,
) -> dict:
    """Validate → token-budget → dedup/cache → rate-limit → score → insert → rank.

    `score_fn` should already be wrapped by the ScoringGate (concurrency bound).
    Raises the typed errors in app.errors; main.py maps them to HTTP responses.
    """
    now = now or datetime.now(timezone.utc)

    season = db.get_active_season()
    if not season:
        raise ScoringUnavailable("No active season is open right now.")

    handle = validate_handle(handle)
    sequence = validate_sequence(sequence)

    token_count = count_tokens(sequence)
    budget = season.get("token_budget", settings.token_budget)
    if token_count > budget:
        raise ValidationError(f"Sequence is {token_count} tokens; the budget is {budget}.")

    key = normalize_key(sequence)

    # Dedup == cache: one canonical row per (season, sequence). A repeat is rejected
    # (never re-scored on NDIF), and we hand back the existing score/rank.
    existing = db.find_submission(season["id"], key)
    if existing:
        rank = db.rank_for(season["id"], existing["score"])
        raise DuplicateError(
            f"That sequence is already on the board (submitted by "
            f"{existing.get('user_handle', 'someone')}).",
            existing=existing,
            rank=rank,
        )

    # Rate limits guard the NDIF quota; checked only for genuinely new sequences.
    check_rate_limits(db, ip_hash, settings, now)

    # score_fn returns either a plain float (legacy/tests) or a scoring.ScoreResult
    # (score=ranked value, shift=raw steering shift, specificity=closed-form z).
    res = score_fn(sequence)
    if isinstance(res, tuple):  # ScoreResult (NamedTuple)
        score, shift_raw, specificity = float(res[0]), float(res[1]), res[2]
        specificity = float(specificity) if specificity is not None else None
    else:
        score, shift_raw, specificity = float(res), float(res), None

    inserted = db.insert_submission({
        "season_id": season["id"],
        "user_handle": handle,
        "sequence_text": sequence,
        "norm_key": key,
        "token_count": token_count,
        "score": score,
        "shift_raw": shift_raw,
        "specificity": specificity,
        "ip_hash": ip_hash,
    })
    # One signed score, two boards: pro-human ranks by most-positive projection,
    # anti-human by most-negative (the geometric opposite).
    pro_rank = db.rank_for(season["id"], score, higher_is_better=True)
    anti_rank = db.rank_for(season["id"], score, higher_is_better=False)

    return {
        "score": score,
        "specificity": specificity,
        "rank": pro_rank,  # back-compat alias for pro_rank
        "pro_rank": pro_rank,
        "anti_rank": anti_rank,
        "token_count": token_count,
        "id": inserted.get("id"),
    }
