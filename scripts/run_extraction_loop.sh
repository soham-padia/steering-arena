#!/usr/bin/env bash
# Resumable, self-continuing extraction of the real direction.
#
# Each run checkpoints every forward pass to disk (data/cache/acts/<model>/), so a
# failure (NDIF eviction, crash, restart) loses nothing — the next run skips what's
# already cached and only re-fetches the missing forwards. This wrapper just keeps
# rerunning until the direction file is produced, with backoff between attempts.
#
#   bash scripts/run_extraction_loop.sh
# Override via env: MODEL, OUT, ATTEMPTS, SLEEP.
set -u
cd "$(dirname "$0")/.."

MODEL="${MODEL:-allenai/Olmo-3.1-32B-Instruct}"
OUT="${OUT:-data/directions/d_olmo3_v1.npz}"
ATTEMPTS="${ATTEMPTS:-200}"
SLEEP="${SLEEP:-120}"
PY="${PY:-.venv/bin/python}"

for i in $(seq 1 "$ATTEMPTS"); do
  echo "=== extraction attempt $i/$ATTEMPTS ($(date +%H:%M:%S)) ==="
  if "$PY" scripts/extract_direction.py --backend ndif --model-id "$MODEL" \
        --out "$OUT" --retry 8 --retry-wait 45; then
    echo "=== EXTRACTION COMPLETE -> $OUT ==="
    "$PY" scripts/validate_direction.py --d "$OUT" --no-steer || true
    echo "review/validate, then open Season 1 (new-season skill)."
    exit 0
  fi
  echo "attempt $i failed (NDIF likely evicting); resuming from cache in ${SLEEP}s…"
  sleep "$SLEEP"
done
echo "exhausted $ATTEMPTS attempts; cache is preserved — rerun later to resume."
exit 1
