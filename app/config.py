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

    # ── Season 3: banded multi-layer scoring ──
    # Empty `score1_layers` keeps the single-layer Season 2 path; set it (and
    # scoring_mode = "banded_mean_multilayer") to switch the scorer to a band. The DB
    # season row is authoritative for display — seasons.layers — and
    # scripts/check_season_matches_d.py asserts the row and the d files agree before a
    # season may open.
    #
    # score1 RANKS the board (banded mean over one shared d_bar; the only steerable
    # aggregate). score2 is INFORMATIONAL (per-layer min over a wider band — the sharper
    # test of whether a sequence holds up at every depth). The two bands union to six
    # layers, which one batch_last_resids_layers call reads, so score2 is free.
    score1_layers: str = ""                     # e.g. "19,23,27,31"
    score1_d_file: str = "data/directions/d_olmo3_s3_score1.npz"
    score2_layers: str = ""                     # e.g. "15,23,31,39"
    score2_d_file: str = "data/directions/d_olmo3_s3_score2.npz"

    @staticmethod
    def _parse_layers(spec: str) -> list[int]:
        return [int(x) for x in spec.split(",") if x.strip()]

    def band1(self) -> list[int]:
        return self._parse_layers(self.score1_layers)

    def band2(self) -> list[int]:
        return self._parse_layers(self.score2_layers)

    def banded(self) -> bool:
        """True when the active season scores across a band rather than one layer."""
        return bool(self.band1())

    def read_layers(self) -> list[int]:
        """Every layer one forward pass must return: the union of both bands, sorted."""
        return sorted(set(self.band1()) | set(self.band2()))

    # ── Direction-specificity metric (non-ranking column this season; see
    #    db/migrations/0003_specificity.sql + app/scoring.py closed-form notes) ──
    specificity_enabled: bool = True
    specificity_eps: float = 1e-4                       # denominator floor (frozen per season)
    specificity_null: str = "isotropic_closed_form"     # method id, recorded for provenance

    # ── Research-data consent (data governance; see db/migrations/0004_consent.sql) ──
    # Frozen id of the consent notice text currently shown (web/consent.html). Bump when
    # the notice materially changes, so each consented row records what was agreed to.
    consent_version: str = "v1-2026-06"

    # ── NDIF call / queue ──
    ndif_api_key: str = ""
    ndif_timeout_s: int = 60
    score_concurrency: int = 2
    hf_token: str = ""

    # ── Live generation demo (/generate) — spends NDIF quota per uncached call ──
    # Every limit here exists to bound that spend (spec §2 constraint 6). generate_*
    # counters are separate from the submission counters so a burst of demo traffic can
    # never eat the scoring budget the leaderboard needs.
    generation_enabled: bool = True
    # Require a signed-in account to generate. Off by default: at low traffic an account
    # is a bigger barrier to the people you want than to the abuse you don't. Flip to
    # true if the demo starts drawing bots. When off, rate limits fall back to the
    # connection (ip_hash) instead of the account (user_hash).
    generate_require_auth: bool = False
    generation_logging: bool = True      # record prompt + output (see 0006); UI says so
    generate_max_new: int = 40           # matches the offline eval's budget
    generate_prompt_max_chars: int = 240
    generate_per_min: int = 4
    generate_per_day: int = 40
    generate_global_per_day: int = 600

    # ── Rate limiting / leaderboard (caps NDIF quota) ──
    rate_per_min: int = 8
    rate_per_day: int = 100
    global_per_day: int = 2000
    leaderboard_max: int = 1000

    # ── Supabase (secrets) ──
    supabase_url: str = ""
    supabase_service_key: str = ""
    # Browser-safe Supabase key, used ONLY for sign-in. Supabase renamed this concept
    # (anon -> publishable, service_role -> secret) and both formats still work, so
    # accept either name and let browser_key() pick. Whatever lands here is served to
    # clients by /admin/config, so a SECRET key must never be put in it.
    supabase_publishable_key: str = ""
    supabase_anon_key: str = ""

    def browser_key(self) -> str:
        return self.supabase_publishable_key or self.supabase_anon_key
    # Comma-separated emails allowed to read the demo log at /admin.html. Empty = the
    # admin view is off entirely (fail closed).
    admin_emails: str = ""
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
