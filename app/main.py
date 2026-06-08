"""FastAPI app: serves the JSON API and (later) the static frontend.

Phase 0 scaffold — only `/health` and a stub `/season` exist. Scoring,
submissions, and the leaderboard arrive in Phases 1–2. No model is loaded
here; live scoring will run on NDIF (see PROJECT_SPEC.md §3, §5).
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings

app = FastAPI(title="Steering Arena", version="0.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.allowed_origin == "*" else [settings.allowed_origin],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Liveness probe — also pinged by the keepalive cron (Phase 4)."""
    return {"status": "ok", "season": settings.season_id, "model": settings.model_id}


@app.get("/season")
def season() -> dict:
    """Active season config the frontend needs (stub until Phase 5)."""
    return {
        "id": settings.season_id,
        "name": settings.season_name,
        "model_id": settings.model_id,
        "layer": settings.layer,
        "d_version": settings.d_version,
        "token_budget": settings.token_budget,
        "scoring_mode": settings.scoring_mode,
    }


# Serve the static frontend if present (Phase 3 fills web/). Mounted last so
# API routes above take precedence over the catch-all.
_web_dir = Path(__file__).resolve().parent.parent / "web"
if _web_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_web_dir), html=True), name="web")
