"""Phase 1 — scoring-core determinism + correctness.

These test the *scoring math* purely, with a deterministic fake activation
reader (no torch, no NDIF, no network). Model-level (NDIF-bounded) reproducibility
is a separate, manual check — see PROJECT_SPEC.md §5.4 and the verify-determinism skill.
"""

import hashlib

import numpy as np

from app import scoring


def fake_resid(dim: int = 64):
    """Deterministic text → vector map (stable across calls and reloads)."""

    def f(text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(dim).astype(np.float32)

    return f


PROBES = ["alpha probe", "beta probe", "gamma probe"]
D = np.random.default_rng(0).standard_normal(64).astype(np.float32)


def test_cosine_basic():
    v = np.array([1.0, 2.0, 3.0])
    assert scoring.cosine(v, v) == 1.0
    assert abs(scoring.cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0]))) < 1e-12


def test_steering_shift_deterministic():
    g = fake_resid()
    a = scoring.score("be honest", PROBES, g, D, mode=scoring.STEERING_SHIFT)
    b = scoring.score("be honest", PROBES, g, D, mode=scoring.STEERING_SHIFT)
    assert a == b  # pure function → exactly equal


def test_self_score_deterministic():
    g = fake_resid()
    a = scoring.score("be honest", PROBES, g, D, mode=scoring.SELF)
    b = scoring.score("be honest", PROBES, g, D, mode=scoring.SELF)
    assert a == b


def test_batched_equals_per_probe():
    g = fake_resid()
    batch_fn = lambda texts: np.stack([g(t) for t in texts])  # noqa: E731
    base_cos = scoring.baseline_cosines(PROBES, batch_fn, D)
    batched = scoring.steering_shift_batched("be honest", PROBES, batch_fn, base_cos, D)
    per_probe = scoring.steering_shift_score("be honest", PROBES, g, D)
    assert abs(batched - per_probe) < 1e-12


def test_steering_shift_matches_manual():
    g = fake_resid()
    manual = float(
        np.mean(
            [scoring.cosine(g(scoring.compose("seq", p)), D) - scoring.cosine(g(p), D) for p in PROBES]
        )
    )
    assert abs(scoring.score("seq", PROBES, g, D) - manual) < 1e-12


def test_reload_stability():
    """Re-import the module and confirm the same inputs give the same score."""
    import importlib

    g = fake_resid()
    before = scoring.score("steady", PROBES, g, D)
    importlib.reload(scoring)
    after = scoring.score("steady", PROBES, g, D)
    assert abs(before - after) <= 1e-4


def test_load_direction_and_probes(tmp_path):
    import json

    d = np.arange(8, dtype=np.float32)
    meta = {"d_version": "vtest", "placeholder": True}
    p = tmp_path / "d.npz"
    np.savez(p, d=d, meta=np.array(json.dumps(meta)))
    loaded, m = scoring.load_direction(p)
    assert loaded.shape == (8,) and m["d_version"] == "vtest"

    probes = scoring.load_probes("data/probes/season1.json")
    assert len(probes) == 16 and all(isinstance(x, str) for x in probes)
