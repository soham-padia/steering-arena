---
title: Steering Arena
emoji: 🧭
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Steering Arena

A public leaderboard for **activation steering**. Players submit short token
sequences; the server scores each by **how much it pushes a frozen large
language model's internal activations along a fixed "pro-human" direction
`d`**, then ranks it. The served model (**OLMo-3-32B**) runs on
[NDIF](https://ndif.us) via [NNsight](https://nnsight.net) — the server is the
canonical scoring oracle.

- **Build spec (source of truth):** [`PROJECT_SPEC.md`](PROJECT_SPEC.md)
- **Phased plan:** [`ROADMAP.md`](ROADMAP.md)

## Stack (all free)

| Layer | Choice |
|---|---|
| API + UI host | Hugging Face Space (Docker, CPU) — no model loaded locally |
| Model + scoring | OLMo-3-32B on NDIF, via NNsight |
| Database | Supabase (Postgres) |
| Frontend | plain HTML/CSS/JS + Alpine.js (no build step) |

## Local dev

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in secrets (Supabase, NDIF, IP salt)
uvicorn app.main:app --reload --port 7860
# → http://localhost:7860/health
```

Run tests with `pytest`.

> Secrets live in `.env` locally (gitignored) and as **Space secrets** in
> production: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `IP_HASH_SALT`,
> `NDIF_API_KEY`. Never expose them client-side.
