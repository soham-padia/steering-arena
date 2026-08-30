"""Does the winning prefix push toward pro-human at EVERY depth, or only at layer 24?

Everything in this project is measured at layer 24: that is where `d` was fit and where
the leaderboard scores. `pro_top` was found by GCG optimising against *that specific
readout*. So the mechanistic question `compile_check.md` leaves open in its Limits:

    is the activation shift a property of the model's state, or a property of one probe?

Two projections are taken from the SAME forward passes, because they cost the same:

  (A) EVERY sampled layer, projected onto d_olmo3_L24_logistic.
      Gives a curve. CAVEAT: d_L24 was fit in layer 24's basis, so away from 24 this
      partly measures basis drift rather than "pro-humanness". It is the shape, not the
      level, that is informative.

  (B) The five layers with a layer-NATIVE direction (L16/24/32/40/48), each projected
      onto its OWN direction. Five points. This is the principled version and it is the
      one that answers the question.

Reported as COSINE, not raw displacement: the residual norm grows ~4x from L16 to L48
(19.2 -> 80.5), so raw projections are not comparable across depth. Per-layer residual
norms are recorded so a reader can convert back.

Cost: one batched forward per (layer, arm) via ResidualReader.batch_last_resids, which
does all 50 texts in ONE pass. 9 layers x 3 arms = 27 NDIF calls, checkpointed to
data/cache/layer_sweep/ so a rerun only fetches what is missing.

    python -u scripts/layer_sweep_prefix.py > /tmp/lsweep.log 2>&1 &
    python scripts/layer_sweep_prefix.py --plot-only    # redraw the figure from JSON

PREDICTION, recorded before the first forward pass (see the .md for the full text):
  P1  A-curve ramps into 24, peaks at/near 24, decays after but not to zero.
  P2  B-curve: pro_top's L24 point is the max; the other four are each < half of it.
  P3  peakiness(pro_top) > peakiness(pro_coherent), where peakiness = shift(L24) /
      max(shift at the other four native layers). P3 is the Goodhart test: if the two
      are similar the L24 peak belongs to the layer, not to the token search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402
from app.ndif_client import ResidualReader  # noqa: E402
from app.scoring import compose  # noqa: E402
from scripts.behavioral_eval import _load_prompts  # noqa: E402
from scripts.extract_direction import unit, with_retry  # noqa: E402
from scripts.prefix_gallery import load_gallery  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "data" / "analysis"
OUT_JSON = AN / "layer_sweep_prefix.json"
FIGDIR = AN / "figures"
CACHE = ROOT / "data" / "cache" / "layer_sweep"
DIRDIR = ROOT / "data" / "directions"

NATIVE_LAYERS = (16, 24, 32, 40, 48)
SCORED_LAYER = 24

# plot_mechanism.py conventions
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURFACE = "#fcfcfb"


# ── directions ──────────────────────────────────────────────────────────────

def load_dir(layer: int) -> np.ndarray:
    """Unit-normalised layer-native logistic direction."""
    fp = DIRDIR / f"d_olmo3_L{layer}_logistic.npz"
    data = np.load(fp, allow_pickle=True)
    meta = json.loads(str(data["meta"]))
    assert int(meta["layer"]) == layer, f"{fp} claims layer {meta['layer']}, not {layer}"
    return unit(np.asarray(data["d"], dtype=np.float64))


# ── batched, cached residual read ───────────────────────────────────────────

def cached_batch(reader, texts, layer, model_id, attempts=4, wait=20.0):
    """(n, hidden) last-token residuals for a batch, with a disk checkpoint.

    Keyed by (model, layer, the exact text list) so base and each arm get their own
    file and a rerun re-fetches nothing it already has. Atomic publish.
    """
    h = hashlib.sha256()
    h.update(f"{model_id}\x00L{layer}\x00n={len(texts)}".encode())
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    fp = CACHE / f"{h.hexdigest()}.npy"
    if fp.exists():
        return np.load(fp), True
    arr = np.asarray(with_retry(reader.batch_last_resids, texts, layer,
                                attempts=attempts, wait=wait), dtype=np.float32)
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_name(fp.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    tmp.replace(fp)
    return arr, False


def cosines(mat: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Per-row cos(R, d) in float64. d must already be unit."""
    m = np.asarray(mat, dtype=np.float64)
    n = np.linalg.norm(m, axis=1)
    return (m @ d) / np.maximum(n, 1e-12)


# ── measure ─────────────────────────────────────────────────────────────────

def measure(args):
    prompts = _load_prompts(args.limit)
    gallery = load_gallery()["arms"]
    arms = [a for a in args.arms.split(",") if a.strip()]
    for a in arms:
        assert a in gallery, f"unknown arm {a}"

    print(f"building reader for {settings.model_id} (NDIF)", flush=True)
    reader = ResidualReader.build(settings.model_id, "ndif",
                                  ndif_key=settings.ndif_api_key,
                                  prepend_bos=settings.prepend_bos)
    n_layers = reader.num_layers
    print(f"model reports num_hidden_layers = {n_layers}", flush=True)

    if args.layers:
        layers = sorted({int(x) for x in args.layers.split(",")})
    else:
        cand = [0, 8, 16, 24, 32, 40, 48, 56, n_layers - 1]
        layers = sorted({L for L in cand if 0 <= L < n_layers})
    assert set(NATIVE_LAYERS) <= set(layers), "sweep must include all five native layers"
    print(f"layers: {layers}", flush=True)
    print(f"arms:   base + {arms}", flush=True)
    print(f"{len(prompts)} prompts; cache {CACHE}", flush=True)

    d24 = load_dir(SCORED_LAYER)
    native = {L: load_dir(L) for L in NATIVE_LAYERS}
    print(f"cos(d_L24, d_L16/32/40/48) = "
          + ", ".join(f"L{L}:{float(d24 @ native[L]):+.4f}"
                      for L in NATIVE_LAYERS if L != SCORED_LAYER), flush=True)

    texts = {"base": list(prompts)}
    for a in arms:
        pfx = gallery[a]["sequence"]
        texts[a] = [compose(pfx, p) if pfx else p for p in prompts]

    store = {
        "model_id": settings.model_id,
        "n_layers": n_layers,
        "layers": layers,
        "native_layers": list(NATIVE_LAYERS),
        "scored_layer": SCORED_LAYER,
        "n_prompts": len(prompts),
        "arms": {a: {"score": gallery[a]["score"], "sequence": gallery[a]["sequence"]}
                 for a in arms},
        "d_cos_with_d24": {str(L): float(d24 @ native[L]) for L in NATIVE_LAYERS},
        "per_layer": {},
        "nonfinite": [],
    }

    calls = 0
    for L in layers:
        rec = {"layer": L, "arms": {}}
        acts = {}
        for name in ["base"] + arms:
            print(f"[NDIF] layer {L:>2}  arm {name:<14} "
                  f"{len(texts[name])} texts, one batched forward ...", flush=True)
            mat, hit = cached_batch(reader, texts[name], L, settings.model_id,
                                    attempts=args.retry, wait=args.retry_wait)
            calls += 0 if hit else 1
            bad = int((~np.isfinite(mat)).sum())
            if bad:
                store["nonfinite"].append({"layer": L, "arm": name, "count": bad})
            print(f"       {'CACHE HIT ' if hit else 'fetched   '} shape={mat.shape} "
                  f"finite={bad == 0} ||R||={float(np.linalg.norm(mat.astype(np.float64), axis=1).mean()):.2f}",
                  flush=True)
            acts[name] = mat

        base_norm = np.linalg.norm(acts["base"].astype(np.float64), axis=1)
        rec["r_norm_base"] = float(base_norm.mean())

        # (A) every layer onto d_L24, and (B) native where it exists
        probes = {"d24": d24}
        if L in native:
            probes["native"] = native[L]

        base_cos = {k: cosines(acts["base"], v) for k, v in probes.items()}
        for k, c in base_cos.items():
            rec[f"base_cos_{k}"] = float(c.mean())
            rec[f"base_cos_{k}_sd"] = float(c.std(ddof=1))

        for name in arms:
            a = {"r_norm": float(np.linalg.norm(acts[name].astype(np.float64), axis=1).mean())}
            for k, v in probes.items():
                c = cosines(acts[name], v)
                shift = c - base_cos[k]
                m, sd = float(shift.mean()), float(shift.std(ddof=1))
                a[f"cos_{k}"] = float(c.mean())
                a[f"shift_{k}"] = m
                a[f"shift_{k}_sd"] = sd
                a[f"shift_{k}_t"] = m / (sd / np.sqrt(len(shift))) if sd else float("inf")
                # shift measured in units of the base cosine's own spread at this
                # layer, so depths with intrinsically wider cosine scales are comparable
                bsd = rec[f"base_cos_{k}_sd"]
                a[f"shift_{k}_z"] = m / bsd if bsd else float("nan")
            rec["arms"][name] = a
        store["per_layer"][str(L)] = rec

        line = f"  L{L:>2} |R|={rec['r_norm_base']:6.2f} base_cos24={rec['base_cos_d24']:+.4f}"
        for name in arms:
            line += f"  {name}:A={rec['arms'][name]['shift_d24']:+.4f}"
            if "shift_native" in rec["arms"][name]:
                line += f"/B={rec['arms'][name]['shift_native']:+.4f}"
        print(line, flush=True)

    store["ndif_calls_made"] = calls
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(store, indent=2))
    print(f"\n-> {OUT_JSON}  ({calls} new NDIF calls)", flush=True)
    return store


# ── report ──────────────────────────────────────────────────────────────────

def summarise(store):
    """Sanity gates + the two peakiness ratios, printed and returned."""
    L24 = store["per_layer"][str(SCORED_LAYER)]
    arms = list(store["arms"])
    out = {"gates": {}, "peakiness": {}}

    g = out["gates"]
    g["base_cos_L24_on_d24"] = L24["base_cos_d24"]
    g["base_cos_L24_expected"] = 0.0067
    g["base_cos_L24_ok"] = abs(L24["base_cos_d24"] - 0.0067) < 0.002
    if "pro_top" in arms:
        v = L24["arms"]["pro_top"]["shift_d24"]
        g["pro_top_shift_L24"] = v
        g["pro_top_shift_L24_expected"] = 0.0355
        g["pro_top_shift_L24_ok"] = abs(v - 0.0355) < 0.002
    g["all_residuals_finite"] = not store["nonfinite"]

    for a in arms:
        nat = {L: store["per_layer"][str(L)]["arms"][a]["shift_native"]
               for L in store["native_layers"]}
        others = {L: v for L, v in nat.items() if L != SCORED_LAYER}
        mx = max(others.values())
        vals = list(nat.values())
        out["peakiness"][a] = {
            "native_shifts": {str(L): v for L, v in nat.items()},
            "L24": nat[SCORED_LAYER],
            "max_other_native": mx,
            "argmax_over_all_native": max(nat, key=nat.get),
            "argmax_among_the_other_four": max(others, key=others.get),
            "ratio_L24_over_max_other": nat[SCORED_LAYER] / mx if mx else float("inf"),
            "mean_other_native": float(np.mean(list(others.values()))),
            "ratio_L24_over_mean_other": (nat[SCORED_LAYER] / float(np.mean(list(others.values())))
                                          if np.mean(list(others.values())) else float("inf")),
            # flatness across the five native layers: sd / |mean|. A depth-general
            # effect is flat (small CV); a one-readout effect is not.
            "mean_all_native": float(np.mean(vals)),
            "sd_all_native": float(np.std(vals, ddof=1)),
            "cv_all_native": float(np.std(vals, ddof=1) / abs(np.mean(vals))),
        }
    # at how many native layers does each arm beat the other?
    if len(arms) == 2:
        x, y = arms
        out["head_to_head_native"] = {
            str(L): {x: store["per_layer"][str(L)]["arms"][x]["shift_native"],
                     y: store["per_layer"][str(L)]["arms"][y]["shift_native"],
                     "winner": (x if store["per_layer"][str(L)]["arms"][x]["shift_native"]
                                > store["per_layer"][str(L)]["arms"][y]["shift_native"] else y)}
            for L in store["native_layers"]}
    return out


# ── figure ──────────────────────────────────────────────────────────────────

def plot(store):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "steering-arena-layer-sweep"
    import matplotlib.pyplot as plt

    layers = store["layers"]
    arms = list(store["arms"])
    colour = {"pro_top": BLUE, "pro_coherent": ORANGE}
    for i, a in enumerate(arms):
        colour.setdefault(a, [AQUA, MUTED][i % 2])

    fig, ax = plt.subplots(figsize=(6.5, 4.1), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.axvline(SCORED_LAYER, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.annotate("layer 24:\nwhere d was fit\nand the board scores",
                xy=(SCORED_LAYER, 0), xytext=(SCORED_LAYER + 1.4, 0),
                fontsize=7, color=MUTED, va="bottom", ha="left", linespacing=1.3)

    for a in arms:
        ys = [store["per_layer"][str(L)]["arms"][a]["shift_d24"] for L in layers]
        ax.plot(layers, ys, color=colour[a], lw=1.2, marker="o", ms=2.6,
                markerfacecolor=SURFACE, markeredgewidth=0.8, zorder=3,
                label=f"(A) {a} · projected on d_L24 at every depth")
        nl = store["native_layers"]
        yn = [store["per_layer"][str(L)]["arms"][a]["shift_native"] for L in nl]
        ax.plot(nl, yn, color=colour[a], lw=1.0, ls=(0, (2, 2)), zorder=4)
        ax.scatter(nl, yn, s=52, marker="D", color=colour[a], edgecolor=SURFACE,
                   linewidth=1.0, zorder=5,
                   label=f"(B) {a} · projected on that layer's OWN d")

    ax.set_xticks(layers)
    ax.set_xlabel("layer", fontsize=8.5, color=INK2)
    ax.set_ylabel("cos shift vs no prefix", fontsize=8.5, color=INK2)
    ax.set_title("The GCG prefix aligns with d only at the layer it was optimised against.\n"
                 "A readable instruction aligns with it at every depth.",
                 fontsize=9.5, color=INK, pad=8, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=7.5)
    ax.grid(True, axis="y", color="#e6e5e1", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK2, handletextpad=0.5,
              labelspacing=0.32, loc="best")
    fig.tight_layout(pad=0.5)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIGDIR / f"layer_sweep.{ext}", dpi=300, facecolor=SURFACE,
                    metadata={"Date": None} if ext == "svg" else None)
    print(f"wrote {FIGDIR/'layer_sweep.png'} (6.5in @ 300dpi)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="pro_top,pro_coherent")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--layers", default="", help="comma list; default samples the depth")
    ap.add_argument("--retry", type=int, default=4)
    ap.add_argument("--retry-wait", type=float, default=20.0)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    store = json.loads(OUT_JSON.read_text()) if args.plot_only else measure(args)
    s = summarise(store)
    print("\nSANITY GATES")
    for k, v in s["gates"].items():
        print(f"  {k:>28} = {v}")
    print("\nPEAKINESS  shift(L24) / max(other four native layers)")
    for a, v in s["peakiness"].items():
        print(f"  {a:>14}  L24={v['L24']:+.4f}  max_other={v['max_other_native']:+.4f} "
              f"(L{v['argmax_among_the_other_four']})  ratio={v['ratio_L24_over_max_other']:.2f}"
              f"  CV over all five={v['cv_all_native']:.2f}")
    store["summary"] = s
    OUT_JSON.write_text(json.dumps(store, indent=2))
    plot(store)


if __name__ == "__main__":
    main()
