"""FastAPI app: JSON API + static frontend.

Endpoints: /health, /season, /leaderboard, /submit (PROJECT_SPEC.md §7).
No model is loaded at startup — the scorer (NDIF via NNsight) is built lazily on
the first /submit, so /health, /season, and /leaderboard stay light and always up.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import captcha, scoring
from app.config import settings
from app.errors import DuplicateError, SubmitError
from app.queue import ScoringGate
from app.ratelimit import hash_ip
from app.submission import process_submission

app = FastAPI(title="Steering Arena", version="0.3.0")

_log = logging.getLogger("steering_arena")
if settings.allowed_origin == "*":
    _log.warning("ALLOWED_ORIGIN is '*' — lock it to the Space origin before going public (audit M1).")
# Production heuristic: a real origin is set. If so, CAPTCHA must be on, or /submit
# has no bot gate and the NDIF quota is exposed (audit H1, fail-open).
if settings.allowed_origin != "*" and not settings.turnstile_secret:
    _log.warning(
        "PROD origin set but TURNSTILE_SECRET is empty — /submit has NO CAPTCHA; "
        "the NDIF quota is exposed to bots. Set TURNSTILE_SECRET + TURNSTILE_SITEKEY."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.allowed_origin == "*" else [settings.allowed_origin],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache(request, call_next):
    """Revalidate the app shell each load so users get new JS/HTML without a hard refresh."""
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-cache")
    return response

_gate = ScoringGate(settings.score_concurrency)


# ── Lazy singletons ──────────────────────────────────────────

_db = None
_scorer = None  # (count_tokens, score_fn) tuple


def get_db():
    """Supabase when configured, else an in-memory store seeded with a stub season."""
    global _db
    if _db is not None:
        return _db

    backend = os.getenv("DB_BACKEND", "auto").lower()
    use_supabase = backend == "supabase" or (
        backend == "auto" and settings.supabase_url and settings.supabase_service_key
    )
    if use_supabase:
        from app.db import SupabaseDatabase

        _db = SupabaseDatabase(settings.supabase_url, settings.supabase_service_key)
    else:
        from app.db import InMemoryDatabase

        _db = InMemoryDatabase()
        _db.add_season(
            id=settings.season_id or 1,
            name=settings.season_name,
            model_id=settings.model_id,
            layer=settings.layer,
            d_version=settings.d_version,
            scoring_mode=settings.scoring_mode,
            token_budget=settings.token_budget,
            active=True,
        )
    return _db


def get_scorer():
    """Build the NDIF/local scorer + tokenizer once. Raises ScoringUnavailable on
    any failure (missing deps, NDIF down, d/model dim mismatch)."""
    global _scorer
    if _scorer is not None:
        return _scorer

    from app.errors import ScoringUnavailable

    try:
        from app.ndif_client import ResidualReader

        d, meta = scoring.load_direction(settings.d_file)
        probes = scoring.load_probes(settings.probe_set)
        reader = ResidualReader.from_settings(settings)

        hidden = reader.hidden_size
        if hidden is not None and hidden != d.shape[0]:
            raise ScoringUnavailable(
                f"Direction dim {d.shape[0]} != model hidden {hidden} — "
                f"regenerate the placeholder at --dim {hidden} or ship a matching d."
            )

        def count_tokens(text: str) -> int:
            return len(reader.tokenizer(text, add_special_tokens=False)["input_ids"])

        if settings.scoring_mode in (scoring.STEERING_SHIFT, scoring.SPECIFICITY_Z):
            # One batched forward per submission; unit-normalized probe baselines
            # precomputed once. Raw shift AND the specificity z come from the same
            # forward (the z is closed-form — pure numpy on the same matrices).
            def batch_fn(texts):
                return reader.batch_last_resids(texts, settings.layer)

            base_units = scoring.baseline_unit_rows(probes, batch_fn)

            def score_fn(seq: str) -> scoring.ScoreResult:
                shift, z = scoring.shift_and_specificity(
                    seq, probes, batch_fn, base_units, d, eps=settings.specificity_eps,
                )
                ranked = z if settings.scoring_mode == scoring.SPECIFICITY_Z else shift
                return scoring.ScoreResult(ranked, shift, z if settings.specificity_enabled else None)
        else:
            def score_fn(seq: str) -> float:
                return scoring.score(
                    seq, probes, lambda t: reader.last_token_resid(t, settings.layer),
                    d, mode=settings.scoring_mode,
                )

        _scorer = (count_tokens, score_fn)
    except ScoringUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 — log detail server-side, return a generic message
        _log.exception("scorer build failed")
        raise ScoringUnavailable("Scoring is temporarily unavailable — please try again shortly.") from exc
    return _scorer


def _client_ip(request: Request) -> str:
    """Resolve the client IP for rate limiting, resistant to X-Forwarded-For spoofing.

    A client can prepend fake entries to XFF, but cannot control the entry the
    outermost *trusted* proxy inserts. With `trusted_proxy_hops = N`, that entry
    is the Nth from the right. Falls back to the direct peer if XFF is absent or
    too short.
    """
    hops = settings.trusted_proxy_hops
    xff = request.headers.get("x-forwarded-for")
    if xff and hops >= 1:
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if len(parts) >= hops:
            return parts[-hops]
    return request.client.host if request.client else "unknown"


# ── Routes ───────────────────────────────────────────────────

class SubmitIn(BaseModel):
    handle: str
    sequence: str
    turnstile_token: str = ""  # Cloudflare Turnstile token (when CAPTCHA is enabled)


_tokenizer = None


def _get_tokenizer():
    """Lazy, cached HF tokenizer for the season model (no NDIF / no model weights —
    just the tokenizer, so the live token counter matches server enforcement)."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(settings.model_id)
    return _tokenizer


@app.get("/tokenize")
def tokenize(text: str = "") -> dict:
    """Real token count for the live counter (same tokenizer the budget is enforced with)."""
    try:
        n = len(_get_tokenizer()(text, add_special_tokens=False)["input_ids"]) if text else 0
    except Exception:  # noqa: BLE001
        return {"tokens": None}
    return {"tokens": n, "budget": settings.token_budget}


@app.get("/seed-pairs.jsonl")
def seed_pairs():
    """Download the contrastive seed pairs used to extract the direction (reproducibility)."""
    p = Path(__file__).resolve().parent.parent / "data" / "seed_pairs.jsonl"
    return FileResponse(p, media_type="application/x-ndjson", filename="steering_arena_seed_pairs.jsonl")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "season": settings.season_id, "model": settings.model_id}


@app.get("/season")
def season() -> dict:
    s = None
    try:
        s = get_db().get_active_season()
    except Exception:  # noqa: BLE001 — DB hiccup shouldn't 500 a read; fall back to config
        s = None
    if s:
        return {
            "id": s["id"], "name": s["name"], "model_id": s["model_id"],
            "layer": s["layer"], "d_version": s["d_version"],
            "token_budget": s.get("token_budget", settings.token_budget),
            "scoring_mode": s.get("scoring_mode", settings.scoring_mode),
            "captcha_sitekey": settings.turnstile_sitekey,
        }
    return {
        "id": settings.season_id, "name": settings.season_name, "model_id": settings.model_id,
        "layer": settings.layer, "d_version": settings.d_version,
        "token_budget": settings.token_budget, "scoring_mode": settings.scoring_mode,
        "captcha_sitekey": settings.turnstile_sitekey,
    }


@app.get("/leaderboard")
def leaderboard(season: int | None = None, limit: int = 50, board: str = "pro") -> dict:
    limit = max(1, min(limit, settings.leaderboard_max))
    ascending = board.lower() == "anti"  # anti-human board ranks by most-negative score
    try:
        db = get_db()
        season_id = season if season is not None else (db.get_active_season() or {}).get("id")
        rows = db.leaderboard(season_id, limit, ascending=ascending) if season_id is not None else []
    except Exception:  # noqa: BLE001 — a read shouldn't 500; show an empty board instead
        logging.getLogger("steering_arena").exception("leaderboard read failed")
        return {"season": season, "board": board, "entries": []}
    entries = [
        {
            "rank": i + 1,
            "handle": r.get("user_handle"),
            "sequence": r.get("sequence_text"),
            "score": r.get("score"),
            "specificity": r.get("specificity"),
            "at": r.get("created_at"),
        }
        for i, r in enumerate(rows)
    ]
    return {"season": season_id, "board": board, "entries": entries}


@app.post("/submit")
def submit(body: SubmitIn, request: Request):
    db = get_db()
    ip = _client_ip(request)
    # CAPTCHA gate first — guards the NDIF quota against bot floods (audit H1).
    if not captcha.verify(body.turnstile_token, settings.turnstile_secret, ip):
        return JSONResponse(status_code=400, content={"error": "CAPTCHA check failed — please retry."})
    ip_hash = hash_ip(ip, settings.ip_hash_salt)
    try:
        count_tokens, score_fn = get_scorer()
        result = process_submission(
            handle=body.handle,
            sequence=body.sequence,
            ip_hash=ip_hash,
            db=db,
            count_tokens=count_tokens,
            score_fn=lambda s: _gate.run(score_fn, s),
            settings=settings,
        )
        return result
    except DuplicateError as e:
        return JSONResponse(
            status_code=e.http_status,
            content={"error": e.message, "score": e.existing.get("score"), "rank": e.rank},
        )
    except SubmitError as e:
        return JSONResponse(status_code=e.http_status, content={"error": e.message})
    except Exception:  # noqa: BLE001 — e.g. NDIF deployment down mid-score; a raw 500
        # would surface as a misleading "Network error" in the frontend.
        _log.exception("scoring failed mid-submission (model backend?)")
        return JSONResponse(
            status_code=503,
            content={"error": "Scoring backend is temporarily unavailable (the model "
                              "deployment may be restarting). Your sequence was NOT "
                              "used up — please try again in a few minutes."},
        )


# Serve the static frontend if present (Phase 3 fills web/). Mounted last.
_web_dir = Path(__file__).resolve().parent.parent / "web"
if _web_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_web_dir), html=True), name="web")
