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

    # ── Season / scoring ──
    # Season 2: OLMo-3-32B, layer 24, logistic d (confound-audited + best causal
    # steerer; see data/directions/d_olmo3_L24_logistic.confound_audit.json). The live
    # scorer reads model_id/layer/d_file/probe_set from here (or env); the Supabase
    # active-season row drives display + leaderboard partitioning. NOTE: a Space env
    # var of the same name overrides these — to switch the live season, update (or
    # remove) any stale Season-1 env vars on the Space too.
    season_id: int = 2
    season_name: str = "Season 2"
    d_version: str = "olmo3_L24_logistic"
    model_id: str = "allenai/Olmo-3-1125-32B"  # exact NDIF-hosted id
    model_build: str = ""
    layer: int = 24
    token_budget: int = 1000
    scoring_mode: str = "cosine_steering_shift"
    probe_set: str = "data/probes/season2.json"
    d_file: str = "data/directions/d_olmo3_L24_logistic.npz"
    prepend_bos: bool = True

    # ── NDIF call / queue ──
    ndif_api_key: str = ""
    ndif_timeout_s: int = 60
    score_concurrency: int = 2
    hf_token: str = ""

    # ── Rate limiting / leaderboard (caps NDIF quota) ──
    rate_per_min: int = 8
    rate_per_day: int = 100
    global_per_day: int = 2000
    leaderboard_max: int = 1000

    # ── Supabase (secrets) ──
    supabase_url: str = ""
    supabase_service_key: str = ""
    ip_hash_salt: str = ""

    # ── CAPTCHA (Cloudflare Turnstile) — optional; active only when secret is set ──
    turnstile_secret: str = ""   # server-side verification key
    turnstile_sitekey: str = ""  # public key sent to the browser

    # ── Server / deploy ──
    port: int = 7860
    allowed_origin: str = "*"
    # Number of trusted reverse-proxy hops in front of the app. The real client
    # IP is the X-Forwarded-For entry inserted by the outermost trusted proxy
    # (counted from the right) — the leftmost entries are client-spoofable. On an
    # HF Space this is typically 1; confirm against the platform's proxy chain.
    trusted_proxy_hops: int = 1


settings = Settings()
