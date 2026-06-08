"""Central config: season settings + secrets, read from env / .env.

Defaults mirror PROJECT_SPEC.md §12. Anything here can be overridden by an
environment variable of the same name (case-insensitive), or by the active
season row in Supabase once that exists. Secrets are server-side only.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # `model_*` fields would collide with pydantic's protected namespace; disable it.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # ── Season / scoring (stub values until a real season is opened) ──
    season_id: int = 0
    season_name: str = "Season 0 — scaffold"
    d_version: str = "v0-stub"
    model_id: str = "OLMo-3-32B"
    model_build: str = ""
    layer: int = 16
    token_budget: int = 10
    scoring_mode: str = "cosine_steering_shift"
    probe_set: str = "data/probes/season1.json"
    d_file: str = "data/directions/d_v1.npz"
    prepend_bos: bool = True

    # ── NDIF call / queue ──
    ndif_api_key: str = ""
    ndif_timeout_s: int = 60
    score_concurrency: int = 2
    hf_token: str = ""

    # ── Rate limiting / leaderboard (caps NDIF quota) ──
    rate_per_min: int = 30
    rate_per_day: int = 500
    global_per_day: int = 5000
    leaderboard_max: int = 200

    # ── Supabase (secrets) ──
    supabase_url: str = ""
    supabase_service_key: str = ""
    ip_hash_salt: str = ""

    # ── Server / deploy ──
    port: int = 7860
    allowed_origin: str = "*"
    # Number of trusted reverse-proxy hops in front of the app. The real client
    # IP is the X-Forwarded-For entry inserted by the outermost trusted proxy
    # (counted from the right) — the leftmost entries are client-spoofable. On an
    # HF Space this is typically 1; confirm against the platform's proxy chain.
    trusted_proxy_hops: int = 1


settings = Settings()
