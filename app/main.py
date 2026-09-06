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

from app import captcha, generation, scoring, userauth
from app.config import settings
from app.errors import DuplicateError, RateLimited, SubmitError, ValidationError
from app.queue import ScoringGate
from app.ratelimit import check_generation_limits, hash_ip
from app.submission import process_submission, validate_handle

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
_scorer = None  # (season_id, count_tokens, score_fn) — cached PER SEASON, so flipping
                # `active` in the DB swaps the scorer with no redeploy and no restart.


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


def _season_bands(season):
    """(band1, band2, d1_path, d2_path, probe_set) for the ACTIVE SEASON ROW.

    The scored function follows the DATABASE, not the environment. Env vars are only a
    local override for scripts; on the Space nothing needs setting to switch seasons.

    This exists because the alternative has a corruption window in both directions. If the
    band lived in env, then between deploying the new vars and flipping `active` the app
    would score Season 3's metric and write those rows under Season 2's id -- silently
    poisoning the LIVE board -- and doing it in the other order poisons the new one. Read
    the band off the same row that decides where rows are written and neither can happen.
    """
    layers = str(season.get("layers") or "").strip() if season else ""
    if not layers:
        return [], [], None, None, None
    band1 = [int(x) for x in layers.split(",") if x.strip()]
    # Season 3 ships two files per d_version, distinguished by `role`.
    ver = season["d_version"]
    d1 = settings.score1_d_file or f"data/directions/d_{ver}_score1.npz"
    d2 = settings.score2_d_file or f"data/directions/d_{ver}_score2.npz"
    _, _, band2, _ = scoring.load_banded_direction(d2)
    probe = f"data/probes/{season['probe_set_id']}.json" if season.get("probe_set_id") else None
    return band1, band2, d1, d2, probe


def get_scorer():
    """Build the NDIF/local scorer + tokenizer for the ACTIVE season.

    Cached per season id, so flipping `active` in the database swaps the scorer on the
    next submission with no redeploy and no restart. Raises ScoringUnavailable on any
    failure (missing deps, NDIF down, d/model dim mismatch).
    """
    global _scorer

    from app.errors import ScoringUnavailable

    try:
        season = get_db().get_active_season()
    except Exception:  # noqa: BLE001 — fall back to env-configured behaviour
        season = None
    season_id = (season or {}).get("id")
    if _scorer is not None and _scorer[0] == season_id:
        return _scorer[1], _scorer[2]

    try:
        from app.ndif_client import ResidualReader

        band1, band2, d1_file, d2_file, probe_file = _season_bands(season)
        banded = bool(band1)
        d, meta = scoring.load_direction(settings.d_file)
        probes = scoring.load_probes(probe_file or settings.probe_set)
        reader = ResidualReader.from_settings(settings)

        hidden = reader.hidden_size
        if hidden is not None and hidden != d.shape[0]:
            raise ScoringUnavailable(
                f"Direction dim {d.shape[0]} != model hidden {hidden} — "
                f"regenerate the placeholder at --dim {hidden} or ship a matching d."
            )

        def count_tokens(text: str) -> int:
            return len(reader.tokenizer(text, add_special_tokens=False)["input_ids"])

        if banded:
            # ── Season 3: one multi-layer forward, two scores ──
            # score1 RANKS (banded mean over a shared d_bar); score2 is informational
            # (per-layer min over a wider band). read_layers() is the UNION of the two
            # bands, so BOTH come out of a single remote call — score2 costs no quota.
            d1, per1, f_band1, meta1 = scoring.load_banded_direction(d1_file)
            d2, per2, f_band2, meta2 = scoring.load_banded_direction(d2_file)
            # The season row and the d file must agree, or the board records numbers under
            # a config that did not produce them. Fail CLOSED (503) rather than score wrong:
            # a wrong score is indistinguishable from a right one after the fact.
            if f_band1 != band1:
                raise ScoringUnavailable(
                    f"Band mismatch: season row says {band1} but {d1_file} says {f_band1}. "
                    f"Run scripts/check_season_matches_d.py."
                )
            if meta1.get("d_version") != season["d_version"]:
                raise ScoringUnavailable(
                    f"d_version mismatch: season row says {season['d_version']!r} but "
                    f"{d1_file} says {meta1.get('d_version')!r}."
                )
            band2 = f_band2
            read = sorted(set(band1) | set(band2))
            _pos = {L: i for i, L in enumerate(read)}

            def _make_reader():
                """One remote call per distinct text batch, sliced per band.

                banded_shift asks for its OWN band, so a naive passthrough would issue a
                separate forward for score1 and score2 — two calls where one suffices, and
                the union-band design exists precisely to avoid that. Both scores request
                the SAME texts, so caching the last batch collapses them to one call.
                Cache is per-scorer and holds one entry: submissions are scored one at a
                time behind the queue, so there is nothing to grow.
                """
                last = {"texts": None, "mat": None}

                def fn(texts, layers):
                    key = tuple(texts)
                    if last["texts"] != key:
                        last["mat"] = reader.batch_last_resids_layers(list(texts), read)
                        last["texts"] = key
                    return last["mat"][[_pos[L] for L in layers]]
                return fn

            batch_layers_fn = _make_reader()

            base1 = scoring.banded_baseline(probes, batch_layers_fn, band1)
            base2 = scoring.banded_baseline(probes, batch_layers_fn, band2)

            def score_fn(seq: str) -> scoring.ScoreResult:
                # Specificity survives the move to a band: score1 uses ONE direction
                # (d_bar) over several layers, and the closed form is the same functional
                # over L*P rows instead of P. It is the anti-token-soup measure, so the
                # season built to resist Goodharting is the last one that should lose it.
                # It is NOT defined for score2 (a min is not linear in the direction).
                s1, z = scoring.banded_shift_and_specificity(
                    seq, probes, batch_layers_fn, band1, base1, d1,
                    eps=settings.specificity_eps)
                s2 = scoring.banded_shift(seq, probes, batch_layers_fn, band2, base2, d2,
                                          per_layer=per2, aggregate=scoring.PER_LAYER_MIN)
                return scoring.ScoreResult(
                    s1, s1, z if settings.specificity_enabled else None, s2)
        elif settings.scoring_mode in (scoring.STEERING_SHIFT, scoring.SPECIFICITY_Z):
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

        _scorer = (season_id, count_tokens, score_fn)
    except ScoringUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 — log detail server-side, return a generic message
        _log.exception("scorer build failed")
        raise ScoringUnavailable("Scoring is temporarily unavailable — please try again shortly.") from exc
    return _scorer[1], _scorer[2]


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
    # Research-use consent. Defaults False so programmatic/omitting callers (who never
    # saw the notice) are NOT silently consented; the frontend sends an explicit value
    # (the opt-out checkbox is ticked by default in the UI).
    consent: bool = False


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
    """Liveness, plus which season is actually live.

    Reads the season from the DATABASE, with settings only as a fallback. It used to
    report settings.season_id, which has never matched a real row — the env says 2 while
    the DB ids are 1/3/4/5 — so /health has been advertising a season that does not exist
    and anything scripting against it was reading a fiction. Taking it from the same row
    that decides scoring also means a season flip needs no Space redeploy to stay honest.
    """
    s = None
    try:
        s = get_db().get_active_season()
    except Exception:  # noqa: BLE001 — health must not 500 on a DB hiccup
        s = None
    return {
        "status": "ok",
        "season": s["id"] if s else settings.season_id,
        "season_name": s["name"] if s else settings.season_name,
        "model": s["model_id"] if s else settings.model_id,
    }


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
            # `layer` is the single representative layer and stays for back-compat;
            # `layers` is the authoritative band on a multi-layer season and is null on
            # Seasons 1/2, which really were single-layer. The banner renders whichever
            # is present.
            "layer": s["layer"], "layers": s.get("layers"),
            "d_version": s["d_version"],
            "token_budget": s.get("token_budget", settings.token_budget),
            "scoring_mode": s.get("scoring_mode", settings.scoring_mode),
            "captcha_sitekey": settings.turnstile_sitekey,
        }
    return {
        "id": settings.season_id, "name": settings.season_name, "model_id": settings.model_id,
        "layer": settings.layer, "layers": settings.score1_layers or None,
        "d_version": settings.d_version,
        "token_budget": settings.token_budget, "scoring_mode": settings.scoring_mode,
        "captcha_sitekey": settings.turnstile_sitekey,
    }


@app.get("/seasons")
def seasons() -> dict:
    """Every season, newest first — what the season switcher lists.

    /season returns only the ACTIVE one, so before this there was no way for a client to
    discover that Season 2 exists or to learn what config an archived board was scored
    under. /leaderboard already accepted ?season=<id>; this is the missing half.

    Read-only and public: a season row is already fully described on the rules pages.
    Must stay registered ABOVE the StaticFiles mount at "/", which is a catch-all.
    """
    try:
        rows = get_db().client.table("seasons").select("*").order("id", desc=True).execute().data
    except Exception:  # noqa: BLE001 — a read shouldn't 500; the switcher just won't populate
        logging.getLogger("steering_arena").exception("seasons read failed")
        return {"seasons": []}
    return {"seasons": [
        {"id": s["id"], "name": s["name"], "model_id": s["model_id"],
         "layer": s["layer"], "layers": s.get("layers"), "d_version": s["d_version"],
         "scoring_mode": s.get("scoring_mode"), "token_budget": s.get("token_budget"),
         "active": bool(s.get("active"))}
        for s in rows
    ]}


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
            # null on Season 1/2 rows, which predate the banded scorer — the UI must
            # render that as "not scored", never as 0.
            "score_alt": r.get("score_alt"),
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
            research_consent=bool(body.consent),
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


# ── Live prefix demo (/generate) ─────────────────────────────────────────────
# Public text generation on the maintainer's NDIF key. Every guard here is
# load-bearing (spec §2 constraint 6): CAPTCHA, durable per-IP + global daily caps
# in generation_events, the same concurrency gate as scoring, an enum of frozen
# prefixes (never client text), a length-capped prompt, and a bounded output.

class GenerateIn(BaseModel):
    prompt: str = ""
    arm: str = "base"
    handle: str = ""       # shown publicly on the feed; the email never is
    turnstile_token: str = ""
    consent: bool = True   # gates the PUBLISHED dataset only; the row is kept either way


_generator = None


def get_generator():
    """ResidualReader for generation, built once, lazily."""
    global _generator
    if _generator is None:
        from app.ndif_client import ResidualReader
        _generator = ResidualReader.from_settings(settings)
    return _generator


@app.get("/generate/arms")
def generate_arms() -> dict:
    """The prefixes the demo will accept, for the UI to render."""
    return {"enabled": settings.generation_enabled,
            "max_new_tokens": settings.generate_max_new,
            "prompt_max_chars": settings.generate_prompt_max_chars,
            "model_id": settings.model_id,
            "captcha_sitekey": settings.turnstile_sitekey,
            "logging": settings.generation_logging,
            "consent_version": settings.consent_version,
            "arms": generation.public_arms()}


@app.get("/generations.jsonl")
def generations_dataset():
    """The recorded generations behind /behavior.html — 50 prompts x 5 arms."""
    p = Path(__file__).resolve().parent.parent / "data" / "generations" / "steering_arena_generations.jsonl"
    if not p.exists():
        return JSONResponse(status_code=404, content={"error": "dataset not built yet"})
    return FileResponse(p, media_type="application/x-ndjson",
                        filename="steering_arena_generations.jsonl")


@app.post("/generate")
def generate_text(body: GenerateIn, request: Request):
    if not settings.generation_enabled:
        return JSONResponse(status_code=503, content={"error": "The generation demo is off right now."})
    ip = _client_ip(request)
    if not captcha.verify(body.turnstile_token, settings.turnstile_secret, ip):
        return JSONResponse(status_code=400, content={"error": "CAPTCHA check failed — please retry."})

    # Sign-in is OPTIONAL (settings.generate_require_auth). At low traffic an account is
    # a bigger barrier to the people you want than to the abuse you don't. A signed-in
    # request still gets the stronger rate-limit key (the account survives a network
    # change; an IP does not), so signing in is rewarded rather than required.
    ip_hash = hash_ip(ip, settings.ip_hash_salt)
    token = userauth.bearer(request)
    u_hash, limit_key, limit_by = None, ip_hash, "ip_hash"
    if token or settings.generate_require_auth:
        try:
            user = userauth.verify_token(token, settings)
            u_hash = userauth.user_hash(user["id"], settings.ip_hash_salt)
            limit_key, limit_by = u_hash, "user_hash"
        except userauth.AuthError as e:
            if settings.generate_require_auth:
                return JSONResponse(status_code=401, content={"error": str(e)})
            # token present but bad, and auth is not required: fall back to the IP key
            _log.info("ignoring an invalid session on an open /generate: %s", e)

    db = get_db()

    # Validate the cheap things first: a bad arm or prompt must not cost a rate-limit
    # slot, a DB round trip, or a place in the NDIF queue.
    try:
        prompt = generation.clean_prompt(body.prompt, settings.generate_prompt_max_chars)
        if body.arm not in {a["arm"] for a in generation.public_arms()}:
            raise generation.GenerationError("Unknown prefix.")
        # Same handle rules as the leaderboard, so there is one policy, not two.
        handle = validate_handle(body.handle) if (body.handle or "").strip() else "anonymous"
    except generation.GenerationError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except ValidationError as e:
        return JSONResponse(status_code=400, content={"error": e.message})

    # Fail CLOSED if the durable counters are unreachable: without them there is no cap
    # on how much of the maintainer's NDIF quota this endpoint can spend.
    try:
        check_generation_limits(db, limit_key, settings, by=limit_by)
    except RateLimited as e:
        return JSONResponse(status_code=429, content={"error": e.message})
    except Exception:  # noqa: BLE001 — missing table, Supabase down, network
        _log.exception("generation limit check failed — refusing to generate uncapped")
        return JSONResponse(status_code=503, content={
            "error": "The generation demo is unavailable right now. The recorded "
                     "generations are still downloadable."})

    try:
        cont, cached = _gate.run(generation.generate, get_generator(), prompt,
                                 body.arm, settings.generate_max_new)
    except generation.GenerationError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception:  # noqa: BLE001 — NDIF down / deployment restarting
        _log.exception("generation failed")
        return JSONResponse(status_code=503, content={
            "error": "The model backend is busy or restarting — try again in a minute."})

    # Durable counter, plus the text when logging is on (migration 0006). ip_hash stays
    # salted and is never exported; research_consent gates publication, not storage.
    try:
        db.log_generation(
            ip_hash, body.arm, hash_ip(prompt, settings.ip_hash_salt), cached,
            user_hash=u_hash, handle=handle,
            prompt=prompt if settings.generation_logging else None,
            continuation=cont if settings.generation_logging else None,
            research_consent=bool(body.consent),
            consent_version=settings.consent_version if settings.generation_logging else None,
        )
    except Exception:  # noqa: BLE001 — never fail a served response on bookkeeping
        _log.exception("generation_events insert failed")

    return {"arm": body.arm, "prompt": prompt, "continuation": cont, "cached": cached,
            "handle": handle, "model_id": settings.model_id,
            "max_new_tokens": settings.generate_max_new}


# ── The generation feed ──────────────────────────────────────────────────────
# Reading is public; writing needs an account (see /generate). Both go through the
# service key here, so generation_events stays RLS-on with no policies and only the
# allow-listed columns below can ever leave the server.

PUBLIC_FIELDS = ["created_at", "arm", "handle", "prompt", "continuation"]
# Columns introduced by later migrations; dropped from a query if the schema predates them.
OPTIONAL_COLUMNS = {"hidden", "handle"}
ADMIN_FIELDS = PUBLIC_FIELDS + ["cached", "research_consent", "consent_version", "hidden"]


def _feed(fields, *, limit, arm, consented=False, include_hidden=False):
    db = get_db()
    client = getattr(db, "client", None)
    if client is None:
        raise RuntimeError("no database configured")
    def build(cols, with_hidden_filter):
        q = client.table("generation_events").select(", ".join(cols)).not_.is_("prompt", "null")
        if with_hidden_filter:
            q = q.eq("hidden", False)
        if arm:
            q = q.eq("arm", arm)
        if consented:
            q = q.eq("research_consent", True)
        return q.order("created_at", desc=True).limit(max(1, min(limit, 500)))

    try:
        rows = build(fields, not include_hidden).execute().data or []
    except Exception:
        # A migration adding these columns may not have run yet (0007 `hidden`,
        # 0008 `handle`). Retry without them rather than serving an error: nothing can
        # be hidden or attributed in a schema that has no such columns, so the
        # unfiltered feed is the correct answer there, not merely a convenient one.
        cols = [c for c in fields if c not in OPTIONAL_COLUMNS]
        rows = build(cols, False).execute().data or []
        fields = cols
    # Allow-list on the way out as well: user_hash, ip_hash and prompt_hash never leave.
    return [{k: r.get(k) for k in fields} for r in rows]


@app.get("/generations/recent")
def generations_recent(limit: int = 50, arm: str = ""):
    """Public feed of what people have generated. No identities: the author of a
    generation is never exposed, only the arm, the prompt and the model's answer."""
    try:
        rows = _feed(PUBLIC_FIELDS, limit=limit, arm=arm)
    except Exception:  # noqa: BLE001
        _log.exception("public feed query failed")
        return JSONResponse(status_code=503, content={"error": "Could not read the feed."})
    return {"count": len(rows), "rows": rows}


@app.get("/admin/config")
def admin_config() -> dict:
    """What the sign-in flow needs. The anon key is public by design; the service key is
    never exposed client-side."""
    return {"enabled": bool(settings.browser_key()),
            "admin_enabled": bool(userauth.admin_emails(settings) and settings.browser_key()),
            "supabase_url": settings.supabase_url,
            "anon_key": settings.browser_key()}


@app.get("/admin/generations")
def admin_generations(request: Request, limit: int = 200, arm: str = "",
                      consented: bool = False):
    try:
        email = userauth.require_admin(userauth.bearer(request), settings)
    except userauth.AuthError as e:
        return JSONResponse(status_code=401, content={"error": str(e)})
    try:
        rows = _feed(ADMIN_FIELDS, limit=limit, arm=arm, consented=consented,
                     include_hidden=True)
    except Exception:  # noqa: BLE001
        _log.exception("admin log query failed")
        return JSONResponse(status_code=503, content={"error": "Could not read the log."})
    _log.info("admin log read by %s (%d rows)", email, len(rows))
    return {"email": email, "count": len(rows), "rows": rows}


class HideIn(BaseModel):
    created_at: str = ""
    hidden: bool = True


@app.post("/admin/hide")
def admin_hide(body: HideIn, request: Request):
    """Take a generation out of the public feed without destroying the record."""
    try:
        email = userauth.require_admin(userauth.bearer(request), settings)
    except userauth.AuthError as e:
        return JSONResponse(status_code=401, content={"error": str(e)})
    if not body.created_at:
        return JSONResponse(status_code=400, content={"error": "Which row?"})
    db = get_db()
    client = getattr(db, "client", None)
    if client is None:
        return JSONResponse(status_code=503, content={"error": "No database configured."})
    try:
        client.table("generation_events").update({"hidden": body.hidden}).eq(
            "created_at", body.created_at).execute()
    except Exception:  # noqa: BLE001
        _log.exception("admin hide failed")
        return JSONResponse(status_code=503, content={"error": "Could not update that row."})
    _log.info("admin %s set hidden=%s on %s", email, body.hidden, body.created_at)
    return {"ok": True, "hidden": body.hidden}


# Serve the static frontend if present (Phase 3 fills web/). Mounted last.
_web_dir = Path(__file__).resolve().parent.parent / "web"
if _web_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_web_dir), html=True), name="web")
