"""What scale is the board score on? Measures cos(R, d) before and after each prefix, and
what the injection does on that same axis.

STATUS: COMPLETE (27 Aug). Results in data/analysis/cosine_scale.md. The 26 Aug attempt
died in NDIF congestion (accepted, never dispatched); the retry ran in 105s.

HEADLINE: base cos(R,d) = 0.0067, so the model's neutral state is essentially orthogonal to
d. pro_top moves it +0.0355. The +1 injection moves it +0.7068 — 19.9x further — and
produces HALF the behaviour. Per unit of cosine shift the prefix delivers ~35x more.

WHY IT MATTERS: the board score is a COSINE SHIFT, so +0.108 for pro_top has no intuitive
scale until you know the baseline cosine. The comparison this produces is the interesting
one — an injection of alpha*d_hat onto a residual of norm ~30 should push cos(R,d) to
roughly 0.7, i.e. a shift 5-6x LARGER than the top token string's 0.108, while producing
half the behaviour (compile_check.md). That would be the same Goodhart pattern as the
leaderboard itself, with the steering vector as the entrant gaming the metric.

CHEAPEST USEFUL VERSION: only the base-residual call is strictly needed. Given base, the
injection's cosine is exact arithmetic (base + alpha*d_hat, no model call), and the prefix
side is already approximated by the published board scores. Dropping the three prefixed
arms cuts the NDIF work by 60%.

    python -u scripts/cosine_scale.py > /tmp/cos.log 2>&1 &

Run it EXACTLY like that. Do not pipe it through `tail` or `head`: they buffer to EOF, so a
working job becomes indistinguishable from a hung one (this cost ~20 minutes on 26 Aug).
Every print here flushes and every NDIF call is announced BEFORE it starts, so silence in
the log localises the hang instead of being ambiguous.
"""
import sys
import time

sys.path.insert(0, '/Users/sohampadia/workspace/Nikhil/research/competition')
import numpy as np

def say(*a):
    print(*a, flush=True)

say("[boot] importing app modules")
from app.config import settings
from app.ndif_client import ResidualReader
from app.scoring import compose
from scripts.behavioral_eval import _layer_norm, _load_d, _load_prompts
from scripts.extract_direction import with_retry
from scripts.prefix_gallery import load_gallery

say("[boot] loading d + prompts")
d, layer = _load_d()
prompts = _load_prompts(50)
g = load_gallery()["arms"]

say(f"[boot] building reader for {settings.model_id}")
r = ResidualReader.build(settings.model_id, "ndif", ndif_key=settings.ndif_api_key,
                         prepend_bos=settings.prepend_bos)

t0 = time.time()
say("[ndif 1/5] layer norm (8 prompts)")
rnorm = _layer_norm(r, layer)
alpha = 1.0 * rnorm
say(f"           ||R|| = {rnorm:.2f}, alpha = {alpha:.2f}  ({time.time()-t0:.0f}s)")

say("[ndif 2/5] base residuals (50 prompts)")
base = np.asarray(with_retry(r.batch_last_resids, prompts, layer, attempts=4, wait=20.0),
                  dtype=np.float64)
say(f"           done ({time.time()-t0:.0f}s)")

cos = lambda M: (M @ d) / np.linalg.norm(M, axis=1)
b = cos(base).mean()

rows = [("base (no prefix)", b, None)]
for n, arm in enumerate(("pro_top", "pro_coherent", "anti_top"), start=3):
    say(f"[ndif {n}/5] {arm} prefixed residuals (50 prompts)")
    texts = [compose(g[arm]["sequence"], p) for p in prompts]
    M = np.asarray(with_retry(r.batch_last_resids, texts, layer, attempts=4, wait=20.0),
                   dtype=np.float64)
    c = cos(M).mean()
    rows.append((f"{arm} prefix", c, c - b))
    say(f"           cos = {c:.4f}  shift = {c-b:+.4f}  ({time.time()-t0:.0f}s)")

# the injection is computed exactly, no model call needed
for mult in (0.5, 1.0):
    c = cos(base + (mult * alpha) * d).mean()
    rows.append((f"+{mult:g} x d injection", c, c - b))

say("\n" + f"{'condition':>24} {'cos(R,d)':>10} {'shift vs base':>14}")
for name, c, sh in rows:
    say(f"{name:>24} {c:>10.4f} {('—' if sh is None else f'{sh:+.4f}'):>14}")
