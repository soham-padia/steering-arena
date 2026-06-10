"""Cross-model transfer report (Track 2) — artifact vs. semantic discriminator.

Semantic steering sequences should partially transfer across models; tokenizer-specific
artifacts can't (different vocabularies). Scores a sample of Season-2 entries on each
model in the registry using THAT model's own validated direction and the same probe
texts, then reports per-entry values + Spearman rank correlations between models and a
transfers/artifact classification.

    python scripts/transfer_report.py                      # all registry models with a d on disk
    python scripts/transfer_report.py --models llama8b     # subset
    python scripts/transfer_report.py --top 5 --mid 2      # smoke

Per-(model, sequence) results are cached in data/cache/transfer/ → fully resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import scoring  # noqa: E402
from app.config import settings  # noqa: E402
from app.ndif_client import ResidualReader  # noqa: E402
from scripts.extract_direction import with_retry  # noqa: E402

# Each entry: short name → (model_id, d_file). Layer comes from the d file's meta.
REGISTRY = {
    "olmo32b": ("allenai/Olmo-3-1125-32B", "data/directions/d_olmo3_L24_logistic.npz"),
    "llama8b": ("meta-llama/Llama-3.1-8B", "data/directions/d_llama_v1.npz"),
    "llama70b": ("meta-llama/Llama-3.1-70B", "data/directions/d_llama70b_v1.npz"),
}

CACHE_DIR = Path("data/cache/transfer")
ANALYSIS_DIR = Path("data/analysis")
SEASON_ID = 4  # Season 2


def _sample_entries(c, top, mid):
    """top-N pro + top-N anti + M mid-board Season-2 sequences (deduped, ordered)."""
    sel = "sequence_text, user_handle, score"
    pro = c.table("submissions").select(sel).eq("season_id", SEASON_ID).order("score", desc=True).limit(top).execute().data
    anti = c.table("submissions").select(sel).eq("season_id", SEASON_ID).order("score", desc=False).limit(top).execute().data
    n_all = c.table("submissions").select("id", count="exact", head=True).eq("season_id", SEASON_ID).execute().count
    mid_rows = (c.table("submissions").select(sel).eq("season_id", SEASON_ID)
                .order("score", desc=True).range(n_all // 2, n_all // 2 + mid - 1).execute().data)
    seen, out = set(), []
    for grp, rows in (("pro", pro), ("anti", anti), ("mid", mid_rows)):
        for r in rows:
            if r["sequence_text"] in seen:
                continue
            seen.add(r["sequence_text"])
            out.append({"sequence": r["sequence_text"], "handle": r["user_handle"],
                        "olmo_score_stored": r["score"], "group": grp})
    return out


def _score_on_model(name, model_id, d_file, sequences, probes):
    """(shift, z) per sequence on one model with its own d; cached per (model, seq)."""
    data = np.load(d_file, allow_pickle=True)
    meta = json.loads(str(data["meta"]))
    d = np.asarray(data["d"], dtype=np.float64)
    layer = int(meta["layer"])
    print(f"\n[{name}] {model_id} L{layer} ({Path(d_file).name})", flush=True)

    reader = ResidualReader.build(model_id, "ndif", ndif_key=settings.ndif_api_key,
                                  prepend_bos=settings.prepend_bos)

    def batch_fn(texts):
        return with_retry(reader.batch_last_resids, texts, layer, attempts=4, wait=20.0)

    base_units = scoring.baseline_unit_rows(probes, batch_fn)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for i, seq in enumerate(sequences, 1):
        key = hashlib.sha256(f"{model_id}\x00L{layer}\x00{Path(d_file).name}\x00{seq}".encode()).hexdigest()
        fp = CACHE_DIR / f"{key}.json"
        if fp.exists():
            out[seq] = tuple(json.loads(fp.read_text()))
            continue
        shift, z = scoring.shift_and_specificity(seq, probes, batch_fn, base_units, d)
        fp.write_text(json.dumps([shift, z]))
        out[seq] = (shift, z)
        print(f"  [{i}/{len(sequences)}] shift={shift:+.5f} z={z:+.2f}  {seq[:45]!r}", flush=True)
    return out


def _spearman(x, y):
    """Spearman rank correlation (scipy if present, else rank-Pearson fallback)."""
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(x, y)
        return float(rho), float(p)
    except Exception:  # noqa: BLE001
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        rho = float(np.corrcoef(rx, ry)[0, 1])
        return rho, float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma subset of registry (default: all with d on disk)")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--mid", type=int, default=20)
    args = ap.parse_args()

    wanted = [m for m in args.models.split(",") if m.strip()] or list(REGISTRY)
    models = {}
    for name in wanted:
        model_id, d_file = REGISTRY[name]
        if not Path(d_file).exists():
            print(f"[skip] {name}: {d_file} not on disk (extract it first)")
            continue
        models[name] = (model_id, d_file)
    if "olmo32b" not in models:
        raise SystemExit("olmo32b (the live model) is required as the reference.")

    from supabase import create_client
    c = create_client(settings.supabase_url, settings.supabase_service_key)
    entries = _sample_entries(c, args.top, args.mid)
    sequences = [e["sequence"] for e in entries]
    probes = scoring.load_probes(settings.probe_set)
    print(f"{len(entries)} entries ({args.top} pro + {args.top} anti + {args.mid} mid) "
          f"× {len(models)} models, {len(probes)} probes")

    results = {name: _score_on_model(name, mid_, df, sequences, probes)
               for name, (mid_, df) in models.items()}

    for e in entries:
        for name in models:
            shift, z = results[name][e["sequence"]]
            e[f"{name}_shift"] = round(shift, 6)
            e[f"{name}_z"] = round(z, 3)

    # Pairwise Spearman on shifts (rank agreement = semantic transfer signal).
    names = list(models)
    corr = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            xa = [results[a][s][0] for s in sequences]
            xb = [results[b][s][0] for s in sequences]
            rho, p = _spearman(xa, xb)
            corr[f"{a}~{b}"] = {"spearman_rho": round(rho, 3), "p": None if np.isnan(p) else round(p, 5)}

    # Classification: sign of the OLMo shift reproduced on every other model = transfers.
    others = [n for n in names if n != "olmo32b"]
    for e in entries:
        ref = e["olmo32b_shift"]
        if others and abs(ref) > 1e-4:
            e["transfers"] = all(np.sign(e[f"{n}_shift"]) == np.sign(ref) for n in others)
        else:
            e["transfers"] = None
    tr = [e for e in entries if e["transfers"] is not None]
    frac = sum(e["transfers"] for e in tr) / len(tr) if tr else None

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    report = {"models": {n: {"model_id": m, "d_file": d} for n, (m, d) in models.items()},
              "n_entries": len(entries), "spearman": corr,
              "frac_sign_transfers": round(frac, 3) if frac is not None else None,
              "entries": entries}
    out = ANALYSIS_DIR / "transfer_report.json"
    out.write_text(json.dumps(report, indent=2))

    print("\n=== TRANSFER REPORT ===")
    for k, v in corr.items():
        print(f"  Spearman {k}: rho={v['spearman_rho']}  p={v['p']}")
    if frac is not None:
        print(f"  sign-transfer rate (OLMo → others): {frac:.0%} of {len(tr)}")
    print(f"  report → {out}")


if __name__ == "__main__":
    main()
