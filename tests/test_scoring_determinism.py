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


# ── Direction-specificity (closed-form z; Track 1) ───────────

DIM = 64
D_UNIT = D / np.linalg.norm(D)


def _base_units(n_probes=4, seed=7):
    """Random unit-row baseline matrix (n_probes, DIM)."""
    rng = np.random.default_rng(seed)
    b = rng.standard_normal((n_probes, DIM))
    return b / np.linalg.norm(b, axis=1, keepdims=True)


def _spec_from_mats(base, composed, d):
    """Call shift_and_specificity with canned matrices via a fake batch fn."""
    probes = [f"p{i}" for i in range(base.shape[0])]
    batch_fn = lambda texts: composed  # noqa: E731 — return the canned composed matrix
    return scoring.shift_and_specificity("seq", probes, batch_fn, base, d)


def test_specificity_shift_parity_with_batched():
    """The new `shift` must equal steering_shift_batched to float precision."""
    g = fake_resid()
    batch_fn = lambda texts: np.stack([g(t) for t in texts])  # noqa: E731
    base_cos = scoring.baseline_cosines(PROBES, batch_fn, D)
    base_units = scoring.baseline_unit_rows(PROBES, batch_fn)
    old = scoring.steering_shift_batched("be honest", PROBES, batch_fn, base_cos, D)
    shift, _z = scoring.shift_and_specificity("be honest", PROBES, batch_fn, base_units, D)
    assert abs(shift - old) < 1e-12


def test_specificity_aligned_beats_artifact_at_equal_shift():
    """The core claim: equal raw shift, but movement ALONG d z-scores far above
    isotropic per-probe 'token-soup' movement."""
    rng = np.random.default_rng(42)
    base = _base_units(n_probes=8)
    a = 0.05

    # Aligned: every probe nudged along d by the same amount.
    aligned = base + a * D_UNIT
    aligned /= np.linalg.norm(aligned, axis=1, keepdims=True)
    shift_a, z_a = _spec_from_mats(base, aligned, D)

    # Artifact: same d-component PLUS a big per-probe random orthogonal kick.
    kicks = rng.standard_normal((8, DIM))
    kicks -= np.outer(kicks @ D_UNIT, D_UNIT)            # orthogonal to d
    kicks /= np.linalg.norm(kicks, axis=1, keepdims=True)
    artifact = base + a * D_UNIT + 0.5 * kicks
    artifact /= np.linalg.norm(artifact, axis=1, keepdims=True)
    shift_b, z_b = _spec_from_mats(base, artifact, D)

    # Shifts are comparable (artifact's renormalization perturbs it a bit)…
    assert abs(shift_b) > 0.25 * abs(shift_a)
    # …but specificity separates them decisively.
    assert z_a > 5 * z_b


def test_specificity_degenerate_no_movement():
    base = _base_units()
    shift, z = _spec_from_mats(base, base.copy(), D)
    # Renormalization round-off leaves ~1e-18 residue; the eps floor keeps z tame.
    assert abs(shift) < 1e-12 and abs(z) < 1e-8 and np.isfinite(z)


def test_specificity_bounded_by_sqrt_dim():
    rng = np.random.default_rng(3)
    base = _base_units()
    for _ in range(20):
        comp = base + 0.3 * rng.standard_normal(base.shape)
        comp /= np.linalg.norm(comp, axis=1, keepdims=True)
        _s, z = _spec_from_mats(base, comp, D)
        assert abs(z) <= np.sqrt(DIM) + 1e-9
    # Fully-aligned movement approaches the bound.
    comp = base + 0.2 * D_UNIT
    comp /= np.linalg.norm(comp, axis=1, keepdims=True)
    _s, z_max = _spec_from_mats(base, comp, D)
    assert z_max > 0.8 * np.sqrt(DIM)


def test_specificity_matches_monte_carlo_null():
    """The closed form IS the K→∞ random-direction z: check against seeded MC."""
    rng = np.random.default_rng(11)
    base = _base_units(n_probes=6)
    comp = base + 0.05 * D_UNIT + 0.1 * rng.standard_normal(base.shape)
    comp /= np.linalg.norm(comp, axis=1, keepdims=True)
    shift, z = _spec_from_mats(base, comp, D)

    delta = comp.astype(np.float64) - base.astype(np.float64)
    K = 4096
    R = rng.standard_normal((K, DIM))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    null = (delta @ R.T).ravel()                      # pooled per-probe shifts, all dirs
    sigma_mc = float(np.sqrt(np.mean(null**2)))
    z_mc = shift / sigma_mc
    assert abs(sigma_mc - np.linalg.norm(delta) / np.sqrt(delta.size)) / sigma_mc < 0.05
    assert abs(z_mc - z) / abs(z) < 0.05


def test_specificity_deterministic_and_reload_stable():
    import importlib

    g = fake_resid()
    batch_fn = lambda texts: np.stack([g(t) for t in texts])  # noqa: E731
    base_units = scoring.baseline_unit_rows(PROBES, batch_fn)
    a = scoring.shift_and_specificity("steady", PROBES, batch_fn, base_units, D)
    b = scoring.shift_and_specificity("steady", PROBES, batch_fn, base_units, D)
    assert a == b
    importlib.reload(scoring)
    base_units2 = scoring.baseline_unit_rows(PROBES, batch_fn)
    c = scoring.shift_and_specificity("steady", PROBES, batch_fn, base_units2, D)
    assert abs(a[0] - c[0]) <= 1e-9 and abs(a[1] - c[1]) <= 1e-6


def test_unit_rows_rejects_zero_row():
    import pytest

    with pytest.raises(ValueError):
        scoring.unit_rows(np.array([[1.0, 0.0], [0.0, 0.0]]))


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
