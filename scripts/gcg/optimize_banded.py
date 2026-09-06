"""GCG against the Season 3 BANDED objective.

Adapted from Jesse Li's steering-arena-optim `optimize_prompt.py` (MIT) -- see README.md.
The GCG loop, the simulated-annealing acceptance, the escalating hyperparameter schedule,
the replica sharding and the checkpoint format are his and are kept intact. What changed:

  --role score1|score2   selects the objective and its direction file
  banded objective       one forward hook per band layer, aggregated (mean, or per-layer min)
  truncate to max(band)  every band layer must survive truncation
  board_score            every iteration also scores the RE-TOKENISED prefix -- what the
                         leaderboard would actually give -- and `best` is chosen on that
  provenance             band / aggregate / d_version / d_tag in every record, not `layer`

WHY score2 FIRST. Rescoring the 618 Season 2 entries showed the ranked Score 1 correlates
with the old single-layer metric at rho=0.95 with the same optimised string still at rank 1
-- so beating it would mostly re-confirm what is known. Score 2 (per-layer min over a spread
band) is the one that separates, and nobody has ever searched against it. A failure to beat
Score 2 is as much a result as beating it.

    python scripts/gcg/optimize_banded.py --role score2 --n-controlled-tokens 32
    python scripts/gcg/optimize_banded.py --role score2 --smoke      # gpt2, CPU-ok
"""

import argparse


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=["score1", "score2"], default="score2",
                    help="score2 (per-layer min, the hard one) or score1 (banded mean, ranks the board)")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test on a small model, run locally to test pipeline")
    ap.add_argument("--replicas", type=int, default=None,
                    help="Force number of data-parallel model replicas (default: auto)")
    ap.add_argument("--gpus-per-replica", type=int, default=None,
                    help="Force GPUs per replica (default: auto from model size vs card)")
    ap.add_argument("--n-controlled-tokens", type=int, default=32,
                    help="Number of tokens in the optimized prompt prefix")
    ap.add_argument("--cand-chunk", type=int, default=4,
                    help="Candidates scored per forward pass. Main memory/speed knob.")
    ap.add_argument("--max-iters", type=int, default=0, help="0 = run until killed")
    ap.add_argument("--resume-from", type=str, default=None,
                    help="Path to an existing run dir to continue (default: fresh run)")
    ap.add_argument("--out-root", type=str,
                    default="/work/neu/p2026_0037_neu/steering-arena/gcg",
                    help="Run dirs live on /work, never /home (which is nearly full)")
    return ap.parse_args(argv)


args = parse_args()

import datetime as dt  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch as t  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gcg_utils import (  # noqa: E402
    MEAN, MIN, build_replicas, compute_scores_batch, d_tag,
    load_banded_direction, load_prompt_suffixes, plan_replica_placement,
    roundtrip_ok, truncate_to_layer,
)

SMOKE = args.smoke
N_CONTROLLED_TOKENS = args.n_controlled_tokens
CAND_CHUNK = args.cand_chunk
GPU_MEM_FRACTION = 0.7  # usable fraction of each card (headroom for activations)

# ── objective ────────────────────────────────────────────────────────────────
# The two roles differ in band, direction file and aggregate. Both DROP the per-probe
# baseline that app/scoring.py subtracts: it is constant w.r.t. the prefix, so argmax is
# identical and this is the cheaper quantity. Same argument upstream relies on.
D_FILE = REPO_ROOT / "data" / "directions" / f"d_olmo3_s3_{args.role}.npz"
AGGREGATE = MEAN if args.role == "score1" else MIN
PROBES_FILE = REPO_ROOT / "data" / "probes" / "season3.json"

d_bar, per_layer, BAND, D_META = load_banded_direction(D_FILE)
DIRS = t.tensor(d_bar[None] if AGGREGATE == MEAN else per_layer, dtype=t.float32)
DIRS = DIRS / DIRS.norm(dim=-1, keepdim=True)
D_TAG = d_tag(d_bar if AGGREGATE == MEAN else per_layer)
MODEL_NAME = "gpt2" if SMOKE else D_META["model_id"]

print(f"role={args.role}  aggregate={AGGREGATE}  band={BAND}")
print(f"direction {D_FILE.name}  d_version={D_META.get('d_version')!r}  tag={D_TAG}")

suffixes = load_prompt_suffixes(PROBES_FILE)
print(f"suffixes: {len(suffixes)} from {PROBES_FILE.name}")

# ── model ────────────────────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

device = t.device("cuda" if t.cuda.is_available() else "cpu")
dtype = t.bfloat16 if device.type == "cuda" else t.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Upstream optimize_prompt.py:139-140. I dropped both while adapting and the smoke run
# caught it. `padding_side = "right"` is NOT cosmetic: compute_scores_batch computes
# gather_pos = n_sfx_tokens + ctrl_seq - 1, which is only the last REAL token under right
# padding. With left padding it would silently read a pad position instead, and nothing
# would error. (Our app/ndif_client.py uses the opposite convention — left pad, read [-1] —
# so do not copy that habit here.)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   # gpt2 has none; OLMo-3 has <|pad|> already
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=dtype, device_map="auto" if device.type == "cuda" else None)
if device.type != "cuda":
    model = model.to(device)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

if SMOKE:  # gpt2 has 12 layers; remap the band into range so the pipeline is exercised
    BAND = [min(L, model.config.n_layer - 3) for L in range(2, 2 + len(BAND))]
    H = model.config.n_embd
    DIRS = t.nn.functional.normalize(t.randn(DIRS.shape[0], H, generator=t.Generator().manual_seed(0)), dim=-1)
    print(f"[smoke] band remapped to {BAND}, random dirs at H={H}")

# Every band layer must survive truncation -- hence max(BAND), not BAND[0].
truncate_to_layer(model, max(BAND))
trunk = model.base_model
D_VOCAB = model.get_input_embeddings().weight.shape[0]
print(f"model {MODEL_NAME} truncated to {len(getattr(trunk, 'layers', getattr(trunk, 'h', [])))} blocks")

groups = plan_replica_placement(model, args.replicas, args.gpus_per_replica, GPU_MEM_FRACTION)
if len(groups) > 1:
    replicas = build_replicas(model, groups, GPU_MEM_FRACTION)
    print(f"{len(replicas)} replicas on groups {groups}")
else:
    replicas = [model]

# sfx -> suffix tokens. The leading space matches the board's f"{seq} {probe}" composition.
sfx_enc = tokenizer([" " + s for s in suffixes], padding=True, return_tensors="pt",
                    add_special_tokens=False)
n_sfx_tokens = sfx_enc["attention_mask"].sum(axis=1)
sfx_tokens = sfx_enc["input_ids"]

replica_consts = [{"in_dev": r.get_input_embeddings().weight.device,
                   "sfx_embed": r.get_input_embeddings()(sfx_tokens.to(r.get_input_embeddings().weight.device))}
                  for r in replicas]
sfx_embed = replica_consts[0]["sfx_embed"]

ctrl_token_ids = t.full((N_CONTROLLED_TOKENS,), tokenizer("!")["input_ids"][0], device=device)


def compute_score_gradient(ctrl_token_ids):
    """Current banded score and its gradient w.r.t. the control tokens.

    Unchanged from upstream except for the banded call: builds a one-hot of the prefix,
    scores it, and backprops to get the gradient of the score w.r.t. that one-hot (the
    GCG signal). For the per-layer min the gradient flows to the ARGMIN layer, which is
    exactly the wanted signal -- improve whichever depth you are currently worst at.
    """
    onehot = t.zeros((N_CONTROLLED_TOKENS, D_VOCAB), dtype=dtype, device=device)
    onehot[t.arange(N_CONTROLLED_TOKENS), ctrl_token_ids] = 1
    onehot.requires_grad = True
    ctrl_embed = (onehot @ model.get_input_embeddings().weight)[None]
    scores = compute_scores_batch(trunk, ctrl_embed, sfx_embed, sfx_enc["attention_mask"],
                                  n_sfx_tokens, DIRS, BAND, AGGREGATE)
    mean_score = scores[0]
    mean_score.backward()  # Maximize score => pick top k gradients
    return mean_score.item(), onehot.grad


def score_candidates(candidates):
    """Banded score per candidate, sharded across replicas and gathered (upstream's)."""
    def score_shard(idx_cids):
        r, cids = idx_cids
        rep, c = replicas[r], replica_consts[r]
        with t.inference_mode():
            ce = rep.get_input_embeddings()(cids.to(c["in_dev"]))
            return compute_scores_batch(rep.base_model, ce, c["sfx_embed"],
                                        sfx_enc["attention_mask"], n_sfx_tokens,
                                        DIRS, BAND, AGGREGATE, CAND_CHUNK)

    if len(replicas) == 1:
        return score_shard((0, candidates))
    shards = t.tensor_split(candidates, len(replicas))
    with ThreadPoolExecutor(max_workers=len(replicas)) as ex:
        parts = list(ex.map(score_shard, enumerate(shards)))
    return t.cat(parts)


def decode_prompt(ids):
    return tokenizer.decode(ids.tolist(), skip_special_tokens=True)


def board_score(ids):
    """Score what the LEADERBOARD would actually see for this prefix.

    THE POINT OF THIS FUNCTION. GCG optimises token IDS; the board is given a STRING and
    re-tokenises it. For adversarial sequences encode(decode(ids)) != ids -- in the smoke
    run, EVERY candidate failed the round-trip from iteration 0, because the standard all-"!"
    init already merges under BPE. So a run's reported score can be for a token sequence the
    board never evaluates. That is the leading explanation for the 0.005-0.031 gap between
    upstream's reported scores and its own board entries (_communication/004).

    My first attempt rejected non-round-tripping candidates. That was wrong: it rejects
    essentially everything, so the guard silently disabled itself and best.json was never
    written. Scoring the RE-TOKENISED form instead is the honest fix -- the number recorded
    is one the board can reproduce.

    The gradient step still runs on the raw ids. That is fine: in GCG the gradient only
    PROPOSES candidates; scoring decides between them. Only the recorded best must be true.
    """
    ids_rt = tokenizer(decode_prompt(ids), add_special_tokens=False)["input_ids"]
    if not ids_rt:
        return float("-inf"), ids_rt
    with t.inference_mode():
        ce = model.get_input_embeddings()(t.tensor([ids_rt], device=device))
        sc = compute_scores_batch(trunk, ce, sfx_embed, sfx_enc["attention_mask"],
                                  n_sfx_tokens, DIRS, BAND, AGGREGATE)
    return float(sc[0]), ids_rt


def save_json_atomic(path: Path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


if args.resume_from is not None:
    run_dir = Path(args.resume_from)
    assert run_dir.is_dir(), f"resume-from is not a directory: {run_dir}"
else:
    run_id = f"{args.role}-{dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}"
    run_dir = Path(args.out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
LATEST, BEST, HISTORY = run_dir / "latest.json", run_dir / "best.json", run_dir / "history.jsonl"
print(f"run dir: {run_dir}")

if args.resume_from is not None:
    state = json.loads(LATEST.read_text())
    assert len(state["ctrl_token_ids"]) == N_CONTROLLED_TOKENS, "checkpoint token-count mismatch"
    assert state.get("d_tag") == D_TAG, (
        f"checkpoint was optimised against direction {state.get('d_tag')}, not {D_TAG} — "
        f"resuming would mix two objectives in one run")
    ctrl_token_ids = t.tensor(state["ctrl_token_ids"], device=device)
    iter_idx = state["iter"] + 1
    best_score = json.loads(BEST.read_text())["score"] if BEST.exists() else float("-inf")
    print(f"resumed at iter {iter_idx} (best_score={best_score})")
else:
    iter_idx, best_score = 0, float("-inf")

n_rejected_roundtrip = 0

while args.max_iters == 0 or iter_idx < args.max_iters:
    iter_start = time.perf_counter()

    # Upstream's escalating schedule: widen the search as the prefix settles.
    if iter_idx < N_CONTROLLED_TOKENS * 4:
        N_TOPK_REPL, BATCH_SIZE_OPTIM, T_SA = 8, 16, 0.012
    elif iter_idx < N_CONTROLLED_TOKENS * 8:
        N_TOPK_REPL, BATCH_SIZE_OPTIM, T_SA = 16, 64, 0.006
    elif iter_idx < N_CONTROLLED_TOKENS * 12:
        N_TOPK_REPL, BATCH_SIZE_OPTIM, T_SA = 32, 256, 0.003
    else:
        N_TOPK_REPL, BATCH_SIZE_OPTIM, T_SA = 64, 1024, 0.003

    model.zero_grad(set_to_none=True)
    ids_scored = ctrl_token_ids
    prompt_str = decode_prompt(ids_scored)
    rt_ok = roundtrip_ok(tokenizer, ids_scored)

    score_curr, score_grad = compute_score_gradient(ctrl_token_ids)

    # GCG maximises here (upstream note): the paper minimises a loss, so no negation.
    topk_vals, topk_idxs = score_grad.topk(N_TOPK_REPL, axis=-1)

    with t.inference_mode():
        repl_seq_idx = t.randint(0, N_CONTROLLED_TOKENS, (BATCH_SIZE_OPTIM,), device="cpu")
        repl_topk_idxidx = t.randint(0, N_TOPK_REPL, (BATCH_SIZE_OPTIM,), device="cpu")
        candidates = ctrl_token_ids[None].repeat(BATCH_SIZE_OPTIM, 1)
        candidates[t.arange(BATCH_SIZE_OPTIM), repl_seq_idx] = topk_idxs[repl_seq_idx, repl_topk_idxidx]
        cand_scores = score_candidates(candidates)
        assert t.isfinite(cand_scores).all(), "Candidate score is not finite (nan?)"
        best_candidate = candidates[t.argmax(cand_scores)]

    # What the board would give for the CURRENT prefix, i.e. its re-tokenised form.
    # Recorded alongside the optimiser's own number; `best` is chosen on this one, because
    # this is the only score the leaderboard can reproduce. See board_score().
    bscore, ids_rt = board_score(ids_scored)
    if not rt_ok:
        n_rejected_roundtrip += 1

    iter_time = time.perf_counter() - iter_start
    print(f"{iter_idx=}, score {score_curr:.5f}, board {bscore:.5f}, rt_ok={rt_ok}, "
          f"drift={score_curr - bscore:+.5f}, {iter_time:.2f}s")
    print(f"prompt: {prompt_str!r}")

    record = {
        "iter": iter_idx, "score": score_curr, "ctrl_token_ids": ids_scored.tolist(),
        "prompt": prompt_str, "iter_time_s": iter_time, "model_id": MODEL_NAME,
        # provenance: `layer` alone cannot describe a banded run
        "role": args.role, "band": BAND, "aggregate": AGGREGATE,
        "d_version": D_META.get("d_version"), "d_tag": D_TAG,
        # `score` is the optimiser's, on the raw ids. `board_score` is what the
        # leaderboard would give, on the re-tokenised string. They differ exactly when the
        # prefix does not round-trip, which for adversarial sequences is most of the time.
        "board_score": bscore, "ctrl_token_ids_retokenised": ids_rt,
        "roundtrip_ok": rt_ok, "n_not_roundtrip": n_rejected_roundtrip,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    save_json_atomic(LATEST, record)
    with open(HISTORY, "a") as f:
        f.write(json.dumps(record) + "\n")
    # `best` is chosen on the BOARD score, not the optimiser's. best.json is what would be
    # submitted, so the number in it has to be one the leaderboard can reproduce. Requiring
    # rt_ok instead would never fire: adversarial prefixes almost never round-trip.
    if bscore > best_score:
        best_score = bscore
        save_json_atomic(BEST, record)

    best_cand_score = t.max(cand_scores).item()
    if best_cand_score > score_curr or t.rand((), device=device) < t.exp(
            t.tensor((best_cand_score - score_curr) / T_SA)):
        ctrl_token_ids = best_candidate

    iter_idx += 1
    gc.collect()
    if device.type == "cuda":
        t.cuda.empty_cache()
