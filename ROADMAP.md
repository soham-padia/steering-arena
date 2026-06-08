# Build Roadmap

Sequenced execution plan for the Pro-Human Activation-Steering Competition. Expands `PROJECT_SPEC.md` §13 into granular steps with dependencies, the skill/agent to use, and a hard **Definition of Done (DoD)** gate per phase. Do not advance past a phase until its DoD is green.

## Two parallel tracks

The build splits into two independent tracks that converge at launch:

- **Track A — Web app** (Phases 0→1→2→3→4): the FastAPI scorer, DB, frontend, deploy. Runs against a **placeholder `d`** so it never blocks on research.
- **Track B — Offline research** (Phases R1→R2): clean seed pairs → extract & validate the real `d`. Pure offline (maintainer machine / NDIF), owned by the `direction-research` agent.

They converge at **Phase 5 (Launch Season 1)**: swap the placeholder `d` for the validated real one and open the season.

```
A:  P0 ── P1 ── P2 ── P3 ── P4 ─┐
                                ├──► P5 Launch ──► P6 Stretch
B:        R1 ──── R2 ──────────┘
```

Track B can start as soon as `data/seed_pairs.jsonl` exists (early in/after P0) and proceed in parallel with A1–A4. Delegate Track B to the **direction-research** agent.

---

## Phase 0 — Scaffold

**Goal:** a bootable skeleton with the full repo tree and pinned deps.

Steps:
1. `git init` (not yet a repo). Confirm `.gitignore` is in effect.
2. Create the repo tree (spec §10): `app/`, `web/`, `data/{directions,probes}/`, `scripts/`, `tests/`, `.github/workflows/`.
3. `requirements.txt` with **pinned** versions: `torch` (CPU), `transformers`, `fastapi`, `uvicorn`, `supabase`, `numpy`, `slowapi`, `pytest`.
4. `app/config.py` reading season config + env (spec §12 defaults).
5. `.env.example` (template; real `.env` is gitignored).
6. `README.md` with HF Space frontmatter (`sdk: docker`, `app_port: 7860`).
7. Stub `app/main.py` with a `/health` route returning a stub season.

**DoD:** `uvicorn app.main:app` boots locally; `GET /health` returns `{"status":"ok", ...}`.

---

## Phase 1 — Scoring core

**Goal:** deterministic scoring through NDIF against a placeholder direction.

Steps:
1. `app/ndif_client.py` — `nnsight.LanguageModel(remote=True)` for the NDIF model + tokenizer; NDIF key from env; **no local weights**. (Optionally a small local model behind a flag for offline dev when NDIF is slow.)
2. `app/scoring.py` — cosine steering-shift (spec §5.1) via a remote forward pass: for each probe `p`, `cos(R_L(seq⊕p)[-1], d) − cos(R_L(p)[-1], d)`, averaged. Add the `cosine_self` fallback (§5.2) behind `SCORING_MODE`.
3. Ship a **placeholder** `data/directions/d_v1.npz` (random vector at the NDIF model's hidden dim, clearly marked placeholder) and `data/probes/season1.json` (~16 frozen neutral prompts).
4. CLI entry: `python -m app.scoring "be honest and own mistakes"` prints a score (hits NDIF).
5. `tests/test_scoring_determinism.py` — same sequence twice + after reload, assert within ~1e-4; cover empty-ish and max-token-budget inputs.

**Skill/agent:** run **verify-determinism** against this code; delegate test execution to **test-runner**.

**DoD:** CLI prints a score via NDIF; determinism test green within tolerance.

---

## Phase 2 — API + DB

**Goal:** a working, abuse-resistant submit/leaderboard backend.

Steps:
1. Supabase free project; apply the schema (spec §6) as a committed migration SQL file (`seasons`, `submissions`, indexes incl. `unique(season_id, norm_key)`; the `(season_id, norm_key)` row doubles as the score cache).
2. `app/db.py` — Supabase client + queries (server-side service key from env), including score-cache lookup/insert.
3. `app/ratelimit.py` — per-`ip_hash` limits (30/min, 500/day) **plus a global daily ceiling** (NDIF-quota guard); salted SHA-256 of IP, never raw IP.
4. `app/queue.py` — bounded NDIF concurrency (semaphore/worker pool) so bursts don't exceed NDIF limits.
5. Implement `/season`, `/leaderboard?season=&limit=`, `/submit` (spec §7): validate handle + sequence → tokenize + token-budget check → `norm_key` dedup → rate-limit → **cache hit OR enqueue + canonical NDIF score** → insert → compute rank. Friendly error strings incl. 502/503 when NDIF is unavailable.
6. CORS locked to the Space origin; input hardening (char cap before tokenize, strip control chars, reject non-UTF-8).
7. `tests/test_submit_flow.py` — happy path + duplicate rejection + over-budget rejection + cache hit (no NDIF re-call) + rate-limit 429.

**DoD:** against a real free Supabase project, can submit a sequence and read it back ranked; dedup, token-budget, cache-hit, and rate-limit all work.

---

## Phase 3 — Frontend

**Goal:** a clean single-page UI; end-to-end play.

Steps:
1. `web/index.html` + `app.js` + `styles.css` — **plain HTML/CSS/JS + Alpine.js (CDN, no build step)**, served as static assets by FastAPI.
2. Season banner (model, layer, `d_version`, token budget, scoring mode, rules link).
3. Submission form: handle + sequence + live token counter (server authoritative); show score + rank + friendly errors.
4. Leaderboard table (rank, handle, sequence in monospace, score, time) with refresh; highlight top entries.
5. Rules / how-it-works section (optimization league; `d` + model public; token budget; opaque sequences allowed; dedup rule).
6. **No browser storage**; mobile-friendly; single accent color.
7. **Stored-XSS guard (security audit H2):** the backend stores `user_handle` and `sequence_text` verbatim (any printable chars). Render **all** user-supplied content via Alpine `x-text` / `textContent` — **never** `x-html` / `innerHTML`. Treat the API response as untrusted.

**Skill:** optionally use the `frontend-design` skill for polish.

**DoD:** submit end-to-end and see yourself ranked; mobile-friendly; errors render as friendly strings; user content is escaped on render (no `x-html`).

---

## Phase 4 — Deploy

**Goal:** public, warm, free Space.

Steps:
1. `Dockerfile` — app only, **no model download**, port 7860.
2. README frontmatter wired (`sdk: docker`, `app_port: 7860`).
3. Space secrets set: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `IP_HASH_SALT`, **`NDIF_API_KEY`** (server-side only).
4. `.github/workflows/keepalive.yml` — cron `curl /health` every 6 h.
5. **CAPTCHA on `/submit` (audit H1, must-fix):** add Cloudflare Turnstile or hCaptcha (free) verification before scoring. IP-keyed limits alone don't stop a rotating-IP bot from draining a full day's NDIF quota (`global_per_day`). Consider lowering `global_per_day` until it's in.
6. **Lock CORS (audit M1):** set `ALLOWED_ORIGIN=https://<user>-<space>.hf.space` (not `*`).
7. **Confirm `TRUSTED_PROXY_HOPS`** matches the HF Space proxy chain (default 1) so IP rate-limiting isn't spoofable.
8. **Verify the Supabase key (audit L3):** confirm `SUPABASE_SERVICE_KEY` is the `service_role`/`sb_secret_` key (server-side, RLS-bypassing) and that RLS is enabled on both tables.
9. *(optional, audit M3)* light per-IP throttle or short cache on `/season` + `/leaderboard` reads.

**Security gate:** run the `security-auditor` over the whole tree; it must return PASS / PASS WITH NOTES (no BLOCK) before going public.

**DoD:** public Space URL works; survives a cold start; cron keeps it warm; a real submission scores through NDIF; `NDIF_API_KEY` never reaches the client; CAPTCHA active; CORS locked. Cost ledger (spec §15) still all-zero.

---

## Track B — Offline direction research (parallel with P1–P4)

Owned by the **direction-research** agent. Starts once `data/seed_pairs.jsonl` exists.

### Phase R1 — Clean seed pairs
**Skill:** **clean-seed-pairs**. Fix the five confounds (length/structure, register, caricature, domain skew, encoding leaks); produce a cleaning report feeding `confounds_removed` metadata.
**DoD:** encoding-leak scan zero hits; `chosen`/`rejected` length distributions overlap; non-workplace scenarios added per axis.

### Phase R2 — Extract & validate `d`
**Skill:** **extract-direction**. Per-pair diffs, last-token, estimate `d`, orthogonalize confounds, layer sweep on the **served NDIF model (OLMo-3-32B), via NNsight `remote=True`** — the same model live scoring uses. Run all validation gates.
**DoD (all required):** good held-out separation; `cos(d, length_dir)`, `cos(d, sentiment_dir)` ≈ 0; **causal steering check passes**; per-axis coherence acceptable (or the lack of it reported as a finding). Commit the real `d_<version>.npz` with full metadata (incl. the NDIF model id/build).

---

## Phase 5 — Launch Season 1 (tracks converge)

**Goal:** open the real season.

Steps:
1. Confirm R2 produced a validated, committed real `d`.
2. **Skill: new-season.** Bump `d_version`; insert the season row matching the `d` file metadata exactly (incl. the pinned NDIF model id/build); set exactly one `active=true`; freeze + reference the committed probe set; leave any placeholder season's scores untouched (or drop the placeholder season cleanly before any public traffic).
3. **Skill: verify-determinism** end-to-end against the live NDIF `(model, build, layer, d)`; tick the full §14 checklist (incl. NDIF key server-side only, score cache, season-break policy).

**DoD:** `GET /season` returns the real active Season 1; live NDIF model/build/layer match the `d` file; determinism checklist fully green; leaderboard isolated by `season_id`.

---

## Phase 6 — Stretch (optional, post-launch)

Independent add-ons, pick by interest (spec §13/§6 Phase 6):
- Per-axis leaderboards (15 directions) + an anti-human board per axis (push away from `d`).
- Weight-class tiers: a second season on a lighter NDIF model (e.g. `OLMo-2-7B` / `OLMo-3-7B`) for faster/cheaper scoring (needs a fresh `d` via Track B → new season).
- Fluency sub-league (perplexity-gated "human-readable" board).
- SAE-feature direction flagship (if an SAE exists for the model/layer); surface which features a sequence lit up.
- Hidden-oracle exploration league (hide `d`, expose only a rate-limited scoring oracle, rotate axes).

---

## Cross-cutting gates (enforce at every relevant phase)

- **Determinism (NDIF-bounded)** (P1, P5): forward-pass only / no sampling / fixed BOS; same input → same score within ~1e-4 while NDIF serves the same build; documented season-break policy if NDIF re-serves the model.
- **Season isolation** (P2, P5, P6): every leaderboard/rank query filters by `season_id`; a changed scored-function (incl. NDIF model build) = a new season, never a silent mix.
- **$0 ledger** (P2, P4, P6): every component on a genuinely free indefinite tier (NDIF via Northeastern research access); if anything would cost money, request university compute — never pay.
- **Server authority** (P2): client scores are display hints only; the server's NDIF score is canonical.
- **NDIF key + quota protection** (P1, P2, P4): `NDIF_API_KEY` server-side only, never client-side; rate limits + global daily ceiling + scoring queue + `(season_id, norm_key)` cache always active.
- **Same model for extract + score** (R2, P5): `d` is extracted on the exact NDIF model live scoring uses — no cross-size mismatch.
