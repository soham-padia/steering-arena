"""Scoring core — cosine steering-shift (PROJECT_SPEC.md §5).

The activation-reading step is injected as a callable `get_resid(text) -> vector`,
so the scoring math is pure, testable, and independent of NDIF / torch. The live
wiring (NDIF via NNsight) lives in `app.ndif_client` and is imported lazily by the
CLI only — importing this module stays light.

    score(seq) = mean over probes p of [ cos(R_L(seq ⊕ p)[-1], d) − cos(R_L(p)[-1], d) ]

Cosine (not raw projection) makes activation-norm inflation worthless (§5.3).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

import numpy as np

ResidFn = Callable[[str], np.ndarray]

STEERING_SHIFT = "cosine_steering_shift"
SELF = "cosine_self"
SPECIFICITY_Z = "steering_shift_specificity_z"  # future season mode: rank by z instead of raw shift

# Floor (not additive eps) for the specificity denominator. Binds only when the
# pooled per-probe movement is essentially zero (per-probe cosine change > 0.99997),
# i.e. "the sequence moved nothing" → z := shift/eps stays bounded by |z| <= sqrt(H).
SPEC_EPS_DEFAULT = 1e-4


class ScoreResult(NamedTuple):
    """What the live scorer returns per submission.

    score       — the RANKED value for the active season (raw shift today; z if the
                  season's scoring_mode is SPECIFICITY_Z).
    shift       — raw cosine steering-shift (always computed; equals `score` today).
    specificity — closed-form direction-specificity z (None when disabled).
    score_alt   — Season 3's second, INFORMATIONAL score: the per-layer-min shift over a
                  wider band. None on single-layer seasons, which is why the DB column is
                  nullable — a Season 1/2 row means "not scored", not "scored zero".
                  It does NOT bound `score` and is not bounded by it; see
                  tests/test_banded_scoring.py::test_min_shift_is_NOT_bounded_by_mean_shift.

    Defaulted, so every existing 3-field construction and `res[0..2]` index still works.
    """

    score: float
    shift: float
    specificity: float | None
    score_alt: float | None = None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        raise ValueError("cosine of a zero vector is undefined")
    return float(np.dot(a, b) / (na * nb))


def compose(seq: str, probe: str) -> str:
    """How a candidate sequence is prepended to a probe. Fixed for determinism."""
    return f"{seq} {probe}"


def steering_shift_score(seq: str, probes: Sequence[str], get_resid: ResidFn, d: np.ndarray) -> float:
    shifts = []
    for p in probes:
        with_seq = cosine(get_resid(compose(seq, p)), d)
        base = cosine(get_resid(p), d)
        shifts.append(with_seq - base)
    return float(np.mean(shifts))


def self_score(seq: str, get_resid: ResidFn, d: np.ndarray) -> float:
    return cosine(get_resid(seq), d)


# ── Batched steering-shift (one forward over all probes; live-scoring path) ──

def _cosines_to_rows(mat: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.array([cosine(mat[i], d) for i in range(mat.shape[0])])


def baseline_cosines(probes, batch_resid_fn, d: np.ndarray) -> np.ndarray:
    """cos(R_L(p), d) for each probe p — constant per (season, probes, d); precompute once."""
    return _cosines_to_rows(batch_resid_fn(list(probes)), d)


def steering_shift_batched(seq, probes, batch_resid_fn, baseline_cos, d: np.ndarray) -> float:
    """Same metric as steering_shift_score, but all `seq ⊕ p` activations come from one
    batched forward (`batch_resid_fn`), and the per-probe baselines are precomputed."""
    mat = batch_resid_fn([compose(seq, p) for p in probes])
    seq_cos = _cosines_to_rows(mat, d)
    return float(np.mean(seq_cos - np.asarray(baseline_cos, dtype=np.float64)))


# ── Direction-specificity (closed-form null; PROJECT validity Track 1) ──────
#
# The score is linear in the direction: per-probe shift along any unit u is
# δ_i·u with δ_i = m̂_i − b̂_i (unit-normalized composed/baseline rows). For
# isotropic random unit directions r in R^H the null is EXACTLY mean 0 with
# pooled std ‖Δ‖_F/√(P·H) — so "z versus K random directions" has a closed
# form at K→∞, no RNG, no stored null vectors:
#
#   z = shift_d / max(‖Δ‖_F/√(P·H), eps)  =  √H · cos(δ̄, d) · coherence
#
# coherence = ‖δ̄‖ / RMS_i‖δ_i‖ ∈ [0,1] penalizes sequences that fling each
# probe's activation in a different direction (token-soup artifacts), while
# rewarding consistent movement along d. Bounded |z| ≤ √H (≈71.55 at H=5120).

def unit_rows(mat: np.ndarray) -> np.ndarray:
    """Rows normalized to unit length, float64. Raises on a zero row (matches cosine())."""
    m = np.asarray(mat, dtype=np.float64)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("cannot unit-normalize a zero activation row")
    return m / norms


def baseline_unit_rows(probes, batch_resid_fn) -> np.ndarray:
    """(P, H) unit-normalized baseline activations — ONE batched forward, precompute
    per (season, probes). Supersedes baseline_cosines for the live path (the d-cosines
    are just `baseline_unit_rows(...) @ d`)."""
    return unit_rows(batch_resid_fn(list(probes)))


def shift_and_specificity(
    seq: str,
    probes: Sequence[str],
    batch_resid_fn,
    base_units: np.ndarray,
    d: np.ndarray,
    *,
    eps: float = SPEC_EPS_DEFAULT,
) -> tuple[float, float]:
    """(raw steering shift, closed-form specificity z) from one batched forward.

    `shift` is numerically identical to steering_shift_batched (parity-tested):
    mean_i(cos(m_i,d) − cos(b_i,d)) = mean_i((m̂_i − b̂_i)·d) = δ̄·d for unit d.
    """
    d64 = np.asarray(d, dtype=np.float64)
    d64 = d64 / np.linalg.norm(d64)  # cosine() normalizes d too — keep exact parity for any d
    mat = unit_rows(batch_resid_fn([compose(seq, p) for p in probes]))
    delta = mat - np.asarray(base_units, dtype=np.float64)          # (P, H)
    # inputs are finite; some BLAS builds emit spurious overflow warnings on @ (see scripts)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        shift = float(delta.mean(axis=0) @ d64)
        sigma_null = float(np.linalg.norm(delta)) / float(np.sqrt(delta.size))  # ‖Δ‖_F/√(P·H)
    z = shift / max(sigma_null, eps)
    return shift, float(z)


# ── Banded (multi-layer) steering-shift — Season 3 ──────────────────────────
#
# Season 2 scored one layer. Season 3 scores a BAND, because a sequence that moves d at
# a single depth is not the same thing as one that moves it throughout — REVISIONS §6
# found the Season 2 winner went +0.10769 at L24 and NEGATIVE under a banded objective.
#
# Two aggregates, and they are deliberately different (see data/analysis/season3_directions.json):
#
#   BANDED_MEAN     one shared d_bar, score = mean over the band of the per-layer shift.
#                   Ranks the board. The only aggregate that stays STEERABLE, which
#                   matters because a causal check gates whatever ranks.
#   PER_LAYER_MIN   one direction per layer, score = shift of the WEAKEST layer.
#                   Informational. Never steered, so the steerability constraint that
#                   forced the mean does not apply here — and a min is the sharper test
#                   of "does this hold up at every depth, or only where it was optimised".
#
# Both are the same shape as the single-layer metric: a cosine SHIFT against a
# precomputed per-probe baseline, so magnitude inflation is still worthless.

BANDED_MEAN = "banded_mean"
PER_LAYER_MIN = "per_layer_min"
BANDED_MULTILAYER = "banded_mean_multilayer"   # the season's scoring_mode string


def banded_baseline(probes, batch_resid_layers_fn, band: Sequence[int]) -> np.ndarray:
    """(L, P, H) unit-normalized baseline activations, ONE batched multi-layer forward.

    Constant per (season, probes, band) — precompute once, exactly as the single-layer
    path precomputes baseline_unit_rows.
    """
    mat = np.asarray(batch_resid_layers_fn(list(probes), list(band)), dtype=np.float64)
    return np.stack([unit_rows(mat[i]) for i in range(mat.shape[0])])


def _band_cosines(units: np.ndarray, d: np.ndarray, per_layer: np.ndarray | None,
                  aggregate: str) -> np.ndarray:
    """(L, P, H) unit rows → (P,) aggregated cosine, one value per probe."""
    if aggregate == BANDED_MEAN:
        d64 = np.asarray(d, dtype=np.float64)
        d64 = d64 / np.linalg.norm(d64)
        return np.mean([units[i] @ d64 for i in range(units.shape[0])], axis=0)
    if aggregate == PER_LAYER_MIN:
        if per_layer is None:
            raise ValueError(f"{PER_LAYER_MIN} needs per-layer directions")
        p = np.asarray(per_layer, dtype=np.float64)
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        return np.min(np.stack([units[i] @ p[i] for i in range(units.shape[0])]), axis=0)
    raise ValueError(f"unknown band aggregate: {aggregate}")


def banded_shift(
    seq: str,
    probes: Sequence[str],
    batch_resid_layers_fn,
    band: Sequence[int],
    base_units: np.ndarray,
    d: np.ndarray,
    per_layer: np.ndarray | None = None,
    aggregate: str = BANDED_MEAN,
) -> float:
    """Mean over probes of [aggregate_cos(seq ⊕ p) − aggregate_cos(p)].

    Reduces EXACTLY to the single-layer steering shift when band has one layer and
    aggregate is BANDED_MEAN — tests/test_scoring_determinism.py pins that equivalence,
    so the Season 2 metric is a special case of this one rather than a parallel code path.
    """
    mat = np.asarray(batch_resid_layers_fn([compose(seq, p) for p in probes], list(band)),
                     dtype=np.float64)
    units = np.stack([unit_rows(mat[i]) for i in range(mat.shape[0])])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        seq_cos = _band_cosines(units, d, per_layer, aggregate)
        base_cos = _band_cosines(np.asarray(base_units, dtype=np.float64), d, per_layer, aggregate)
    return float(np.mean(seq_cos - base_cos))


def banded_shift_and_specificity(
    seq: str,
    probes: Sequence[str],
    batch_resid_layers_fn,
    band: Sequence[int],
    base_units: np.ndarray,
    d: np.ndarray,
    *,
    eps: float = SPEC_EPS_DEFAULT,
) -> tuple[float, float]:
    """(banded-mean shift, closed-form specificity z) for a BAND, from one batched read.

    Specificity generalises to the banded mean EXACTLY, with no new theory. The single-
    layer derivation above needs the score to be linear in the direction and takes a mean
    over P probe rows; the banded mean is the same functional over L*P rows, because

        mean_L mean_i (m̂_{L,i} − b̂_{L,i})·d̄  =  mean over the flattened (L*P, H) stack.

    So the null is the same with P → L·P:  σ = ‖Δ‖_F / √(L·P·H), and the bound is still
    |z| ≤ √H, since ‖δ̄‖ ≤ ‖Δ‖_F/√(LP) gives |z| ≤ √(L·P·H)/√(LP) = √H.

    This is defined ONLY for BANDED_MEAN. It does NOT extend to PER_LAYER_MIN (score2):
    the closed form requires linearity in the direction, and a min over per-layer
    directions is not linear. That is a real limit, not an implementation gap.
    """
    d64 = np.asarray(d, dtype=np.float64)
    d64 = d64 / np.linalg.norm(d64)
    mat = np.asarray(batch_resid_layers_fn([compose(seq, p) for p in probes], list(band)),
                     dtype=np.float64)
    units = np.stack([unit_rows(mat[i]) for i in range(mat.shape[0])])      # (L, P, H)
    delta = units - np.asarray(base_units, dtype=np.float64)                # (L, P, H)
    flat = delta.reshape(-1, delta.shape[-1])                               # (L*P, H)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        shift = float(flat.mean(axis=0) @ d64)
        sigma_null = float(np.linalg.norm(flat)) / float(np.sqrt(flat.size))
    return shift, float(shift / max(sigma_null, eps))


def load_banded_direction(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[int], dict]:
    """(d_bar, per_layer, band, meta) from a banded d_<version>.npz.

    Season 3's files carry `d`, `band`, `per_layer` and `meta`; `meta.aggregate` says
    which of the two aggregates the file is for, and `meta.role` distinguishes the two
    files that share a d_version.
    """
    data = np.load(path, allow_pickle=True)
    d = np.asarray(data["d"], dtype=np.float32)
    per = np.asarray(data["per_layer"], dtype=np.float32) if "per_layer" in data else None
    band = [int(x) for x in data["band"]] if "band" in data else []
    meta = json.loads(str(data["meta"])) if "meta" in data else {}
    return d, per, band, meta


def score(
    seq: str,
    probes: Sequence[str],
    get_resid: ResidFn,
    d: np.ndarray,
    mode: str = STEERING_SHIFT,
) -> float:
    if mode == STEERING_SHIFT:
        return steering_shift_score(seq, probes, get_resid, d)
    if mode == SELF:
        return self_score(seq, get_resid, d)
    raise ValueError(f"unknown scoring mode: {mode}")


# ── I/O helpers ──────────────────────────────────────────────

def load_direction(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load `d` (float32) and its metadata from a d_<version>.npz file."""
    data = np.load(path, allow_pickle=True)
    d = np.asarray(data["d"], dtype=np.float32)
    meta = {}
    if "meta" in data:
        meta = json.loads(str(data["meta"]))
    return d, meta


def load_probes(path: str | Path) -> list[str]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj["prompts"] if isinstance(obj, dict) else list(obj)


# ── CLI: python -m app.scoring "<sequence>" ──────────────────

def _main(argv: list[str] | None = None) -> None:
    import argparse

    from app.config import settings

    ap = argparse.ArgumentParser(description="Score a steering sequence against the active direction.")
    ap.add_argument("sequence", help="the candidate token sequence")
    ap.add_argument("--mode", default=settings.scoring_mode, choices=[STEERING_SHIFT, SELF])
    args = ap.parse_args(argv)

    d, meta = load_direction(settings.d_file)
    probes = load_probes(settings.probe_set)
    if meta.get("placeholder"):
        print("[warning] using a PLACEHOLDER direction — scores are not meaningful yet.")

    # Heavy import deferred to here (needs nnsight/torch + NDIF creds or a local model).
    from app.ndif_client import ResidualReader

    reader = ResidualReader.from_settings(settings)
    hidden = reader.hidden_size
    if hidden is not None and hidden != d.shape[0]:
        raise SystemExit(
            f"direction dim {d.shape[0]} != model hidden size {hidden}. "
            f"Regenerate the placeholder at --dim {hidden} (scripts/make_placeholder_direction.py) "
            f"or ship a real d extracted on this model."
        )

    if args.mode == STEERING_SHIFT:
        # Batched path (one forward) + the specificity z alongside the raw shift.
        def batch_fn(texts):
            return reader.batch_last_resids(texts, settings.layer)

        base_units = baseline_unit_rows(probes, batch_fn)
        shift, z = shift_and_specificity(args.sequence, probes, batch_fn, base_units, d,
                                         eps=settings.specificity_eps)
        print(f"shift={shift:.6f}  specificity_z={z:.2f}")
    else:
        s = score(args.sequence, probes, lambda t: reader.last_token_resid(t, settings.layer), d, mode=args.mode)
        print(f"{s:.6f}")


if __name__ == "__main__":
    _main()
