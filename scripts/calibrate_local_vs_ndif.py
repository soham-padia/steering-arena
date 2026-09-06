"""Local-vs-NDIF calibration for the Steering Arena scorer.

Imports the math from scripts/score_local.py verbatim (load_direction, load_probes,
cosine, compose) so the only thing that differs from the published local reference is
the driver loop: load the model once, score N submissions, compare against the
canonical NDIF score stored in the production DB.

What matters is reported twice over:
  - absolute gap  -> does a local score mean the same thing as a board score?
  - Spearman rho  -> does GCG search against the local objective transfer to the board?
Rank order is the load-bearing one; _communication/001 says the server stays canonical.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, urllib.request
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[0]
sys.path.insert(0, "/home/padia_so_neu/steering-arena/scripts")
from score_local import load_direction, load_probes, cosine, compose  # noqa: E402


def fetch_reference(n: int, season_id: int):
    env = dict(re.findall(r'^([A-Z_]+)=([^\s#]*)',
                          open('/home/padia_so_neu/steering-arena/.env').read(), re.M))
    url, key = env['SUPABASE_URL'], env['SUPABASE_SERVICE_KEY']
    req = urllib.request.Request(
        f"{url}/rest/v1/submissions?select=id,sequence_text,score,token_count"
        f"&season_id=eq.{season_id}&order=score.desc&limit=2000",
        headers={"apikey": key, "Authorization": f"Bearer {key}"})
    rows = json.load(urllib.request.urlopen(req, timeout=60))
    rows = [r for r in rows if r["sequence_text"].strip()]
    # Stratify across the score range so the fit isn't dominated by the crowded middle.
    idx = np.linspace(0, len(rows) - 1, num=min(n, len(rows))).round().astype(int)
    return [rows[i] for i in sorted(set(idx.tolist()))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--season-id", type=int, default=4, help="DB id of Season 2 (not .env SEASON_ID)")
    ap.add_argument("--model", default="allenai/Olmo-3-1125-32B")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--d", default="/home/padia_so_neu/steering-arena/data/directions/d_olmo3_L24_logistic.npz")
    ap.add_argument("--probes", default="/home/padia_so_neu/steering-arena/data/probes/season2.json")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="sdpa", help="sdpa | eager | flash_attention_2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ref = fetch_reference(a.n, a.season_id)
    d, probes = load_direction(a.d), load_probes(a.probes)
    print(f"transformers {transformers.__version__} | torch {torch.__version__} | attn={a.attn}")
    print(f"{len(ref)} reference submissions | {len(probes)} probes | layer {a.layer}", flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=getattr(torch, a.dtype), device_map="auto",
        output_hidden_states=True, attn_implementation=a.attn)
    model.eval()
    print(f"model loaded in {time.time()-t0:.0f}s onto {model.device}", flush=True)

    @torch.no_grad()
    def resid_last(text: str) -> np.ndarray:
        inp = tok(text, return_tensors="pt").to(model.device)
        hs = model(**inp).hidden_states
        return hs[a.layer + 1][0, -1, :].float().cpu().numpy()

    base = {p: cosine(resid_last(p), d) for p in probes}
    rows = []
    for i, r in enumerate(ref):
        s = float(np.mean([cosine(resid_last(compose(r["sequence_text"], p)), d) - base[p]
                           for p in probes]))
        rows.append({"id": r["id"], "ndif": r["score"], "local": s,
                     "gap": s - r["score"], "tokens": r["token_count"],
                     "text": r["sequence_text"][:120]})
        if i % 10 == 0:
            print(f"  [{i+1}/{len(ref)}] ndif={r['score']:+.6f} local={s:+.6f} "
                  f"gap={s-r['score']:+.2e}", flush=True)

    nd = np.array([x["ndif"] for x in rows]); lo = np.array([x["local"] for x in rows])
    gap = lo - nd
    def rank(v): return np.argsort(np.argsort(v))
    spearman = float(np.corrcoef(rank(nd), rank(lo))[0, 1])
    pearson = float(np.corrcoef(nd, lo)[0, 1])
    summary = {
        "transformers": transformers.__version__, "torch": torch.__version__,
        "attn": a.attn, "dtype": a.dtype, "layer": a.layer, "n": len(rows),
        "ndif_score_range": [float(nd.min()), float(nd.max())],
        "gap_mean": float(gap.mean()), "gap_std": float(gap.std()),
        "gap_abs_max": float(np.abs(gap).max()), "gap_abs_median": float(np.median(np.abs(gap))),
        "spearman_rho": spearman, "pearson_r": pearson,
        "memo_reported_gap": 7e-2, "score_scale": 3e-2,
    }
    print("\n=== SUMMARY ===")
    for k, v in summary.items(): print(f"  {k}: {v}")
    print(f"\n  gap |max| {summary['gap_abs_max']:.3e} vs memo's reported ~7e-2 "
          f"on scores of ~3e-2")
    print(f"  Spearman rho = {spearman:.5f}  <- decides whether local GCG transfers")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
