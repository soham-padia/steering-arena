# Deploying to the Hugging Face Space

The Space (`sohampadianeu/steering-arena`) is a **deployment target, not a mirror**. It
holds ONE squashed commit of the application tree. The development history lives on GitHub.

## Why it is not a mirror

The HF pre-receive hook rejects plain binary blobs and **walks the entire pushed history**,
not just the tip tree. This repo's history contains binaries that were committed without
LFS for a stretch (`036f66d` removed the LFS rules; `c340e6e` put them back). Once those
blobs exist in a commit range, converting them at the tip does not help — the push is
still refused, listing the same files:

```
remote: Your push was rejected because it contains binary files.
remote:   - data/directions/d_olmo3_s3_score1.npz (ref: refs/heads/main)
```

The alternatives were rewriting upstream history (`git lfs migrate import --everything`,
which means force-pushing GitHub and invalidating other clones) or deploying a single
commit. The Space does not need 20+ commits of research history, so it gets one.

## Deploy

From a clean tree on the commit you want live (normally `main`):

```bash
git checkout --orphan space-deploy      # keeps the working tree, drops history
git add -A                              # .gitattributes turns binaries into LFS pointers
git commit -m "Steering Arena — deploy (squashed from <repo> @ $(git rev-parse --short main))"
git push --force space space-deploy:main
git checkout main && git branch -D space-deploy
```

The branch is disposable — create, push, delete. Do not keep it around; it only drifts.

### Credentials

`~/.gitconfig` wires the GitHub CLI in as the credential helper for github.com only, so a
push to huggingface.co finds nothing and fails with `could not read Username`. Use an
askpass shim so the token never lands in argv, git config, or shell history:

```bash
cat > /tmp/hf_askpass.sh <<'EOF'
#!/bin/sh
case "$1" in *[Uu]sername*) echo "hf" ;; *) printf '%s' "$HF_TOKEN" ;; esac
EOF
chmod +x /tmp/hf_askpass.sh
GIT_ASKPASS=/tmp/hf_askpass.sh GIT_TERMINAL_PROMPT=0 git push --force space space-deploy:main
rm -f /tmp/hf_askpass.sh
```

`HF_TOKEN` comes from `~/.hf_token`, sourced by `~/startload.sh`.

## What the Space needs, and what it does NOT

**Needs:** `app/`, `web/`, `data/` (probe sets + direction files the scorer loads),
`Dockerfile`, `requirements.txt`, and `README.md` — whose YAML frontmatter (`sdk: docker`,
`app_port: 7860`) is what makes it a Docker Space. Losing that frontmatter breaks the build.

**Does NOT need env vars for a season change.** The scorer, `/health` and `/season` all read
the active season row from Supabase. Opening a season is a SQL statement, not a redeploy —
see `db/migrations/0009_season3.sql`. The Space variables `LAYER`, `D_FILE`, `D_VERSION`,
`PROBE_SET`, `SEASON_ID`, `SEASON_NAME` are legacy fallbacks that the database overrides.

## After deploying

```bash
curl -s https://sohampadianeu-steering-arena.hf.space/health
```

Confirm it reports the season you expect. `season_name` in the response means the new code
is live; its absence means you are still looking at a pre-2026-09 build. Rebuild takes about
a minute (`RUNNING_BUILDING` → `RUNNING_APP_STARTING` → `RUNNING`).

## The ordering trap, for the record

Do not open a season in the database before the deployed code can score it. Between the two,
`/submit` resolves the active season from the DB but scores with whatever the deployed build
does, so rows land under the new season carrying the old metric — silently, with no error,
and indistinguishable afterward from correct rows. This happened once during the Season 3
build and was caught before any submission arrived. Deploy first, flip second.
