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
from typing import Callable, Sequence

import numpy as np

ResidFn = Callable[[str], np.ndarray]

STEERING_SHIFT = "cosine_steering_shift"
SELF = "cosine_self"


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

    s = score(args.sequence, probes, lambda t: reader.last_token_resid(t, settings.layer), d, mode=args.mode)
    print(f"{s:.6f}")


if __name__ == "__main__":
    _main()
