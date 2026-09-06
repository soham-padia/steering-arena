"""Season 3 banded scoring: the reduction property, the min aggregate, determinism.

The point of the first test is that the Season 2 metric is a SPECIAL CASE of the Season 3
one, not a parallel implementation. A one-layer band under BANDED_MEAN must reproduce
steering_shift_batched to float64 exactness. If those ever diverge, the two seasons stop
being comparable in construction and the banded path has silently become a different
metric.

No model, no NDIF, no GPU: app.scoring takes its activations through an injected callable,
so the math is testable with a stub.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from app import scoring


H = 32
PROBES = ["alpha probe", "beta probe", "gamma probe"]


def _rng_resid(seed=0):
    """Deterministic fake activations: text+layer -> a fixed pseudo-random vector.

    Seeded with hashlib, NOT the builtin hash(): hash() is salted per process
    (PYTHONHASHSEED), so a fixture built on it produces different activations on every
    run and yields tests that pass alone and fail in a suite. Found exactly that way.
    """
    def resid(texts, layers):
        out = np.zeros((len(layers), len(texts), H))
        for i, L in enumerate(layers):
            for j, t in enumerate(texts):
                digest = hashlib.sha256(f"{seed}\x00{L}\x00{t}".encode()).digest()[:4]
                out[i, j] = np.random.default_rng(int.from_bytes(digest, "big")).normal(size=H)
        return out
    return resid


def _single(resid_layers, layer):
    """Adapt the multi-layer stub to the single-layer batch signature."""
    return lambda texts: resid_layers(texts, [layer])[0]


def test_one_layer_band_reduces_to_single_layer_metric():
    resid = _rng_resid()
    d = np.random.default_rng(7).normal(size=H)
    layer = 24

    banded_base = scoring.banded_baseline(PROBES, resid, [layer])
    banded = scoring.banded_shift("seq under test", PROBES, resid, [layer],
                                  banded_base, d, aggregate=scoring.BANDED_MEAN)

    batch_fn = _single(resid, layer)
    single_base = scoring.baseline_unit_rows(PROBES, batch_fn)
    single, _ = scoring.shift_and_specificity("seq under test", PROBES, batch_fn,
                                              single_base, d)

    assert banded == pytest.approx(single, abs=1e-12), (
        "a one-layer banded mean must BE the single-layer steering shift")


def test_banded_mean_is_the_mean_of_per_layer_shifts():
    resid = _rng_resid()
    d = np.random.default_rng(3).normal(size=H)
    band = [19, 23, 27, 31]

    banded = scoring.banded_shift("x", PROBES, resid, band,
                                  scoring.banded_baseline(PROBES, resid, band), d,
                                  aggregate=scoring.BANDED_MEAN)
    per_layer = []
    for L in band:
        fn = _single(resid, L)
        per_layer.append(scoring.shift_and_specificity(
            "x", PROBES, fn, scoring.baseline_unit_rows(PROBES, fn), d)[0])

    assert banded == pytest.approx(float(np.mean(per_layer)), abs=1e-12)


def test_per_layer_min_really_takes_the_weakest_layer():
    """PER_LAYER_MIN must return, per probe, the minimum over the band — that is the whole
    reason it is the diagnostic: a sequence cannot buy a good number at one depth."""
    resid = _rng_resid()
    band = [15, 23, 31, 39]
    per = np.random.default_rng(11).normal(size=(len(band), H))
    units = scoring.banded_baseline(PROBES, resid, band)          # (L, P, H)

    got = scoring._band_cosines(units, per.mean(0), per, scoring.PER_LAYER_MIN)
    pn = per / np.linalg.norm(per, axis=1, keepdims=True)
    each = np.stack([units[i] @ pn[i] for i in range(len(band))])  # (L, P)

    assert got == pytest.approx(each.min(axis=0), abs=1e-12)
    assert np.all(got <= each + 1e-12), "the min must be <= every layer it minimises over"


def test_min_shift_is_NOT_bounded_by_mean_shift():
    """Documented because it is counterintuitive and easy to assume the opposite.

    min <= mean holds for the COSINES. It does NOT survive taking a shift: Score 1 and
    Score 2 are each a difference of two aggregates, and min(A) - min(B) can exceed
    mean(A) - mean(B) because the argmin of A and of B need not be the same layer. The
    two aggregates also use different directions (a shared d_bar vs one per layer), so
    they are not commensurable to begin with.

    Consequence for the board: a row where Score 2 > Score 1 is NOT a bug, and the UI
    must not present the pair as if one bounds the other.
    """
    resid = _rng_resid()
    band = [15, 23, 31, 39]
    per = np.random.default_rng(11).normal(size=(len(band), H))
    base = scoring.banded_baseline(PROBES, resid, band)
    kw = dict(per_layer=per, )
    mn = scoring.banded_shift("x", PROBES, resid, band, base, per.mean(0),
                              aggregate=scoring.PER_LAYER_MIN, **kw)
    mean = scoring.banded_shift("x", PROBES, resid, band, base, per.mean(0),
                                aggregate=scoring.BANDED_MEAN, **kw)
    assert np.isfinite(mn) and np.isfinite(mean)
    # Asserting a DIRECTION here would be asserting a property of this fixture, not of
    # the metric — the sign depends on where each aggregate's argmin falls. The contract
    # is only that they are genuinely different quantities and neither bounds the other.
    assert mn != mean


def test_per_layer_min_requires_per_layer_directions():
    resid = _rng_resid()
    band = [15, 23]
    with pytest.raises(ValueError, match="per-layer"):
        scoring.banded_shift("x", PROBES, resid, band,
                             scoring.banded_baseline(PROBES, resid, band),
                             np.ones(H), per_layer=None, aggregate=scoring.PER_LAYER_MIN)


def test_unknown_aggregate_raises():
    resid = _rng_resid()
    with pytest.raises(ValueError, match="aggregate"):
        scoring.banded_shift("x", PROBES, resid, [23],
                             scoring.banded_baseline(PROBES, resid, [23]),
                             np.ones(H), aggregate="median_of_vibes")


def test_banded_score_is_deterministic_across_calls():
    resid = _rng_resid()
    d = np.random.default_rng(5).normal(size=H)
    band = [19, 23, 27, 31]
    base = scoring.banded_baseline(PROBES, resid, band)
    a = scoring.banded_shift("same input", PROBES, resid, band, base, d)
    b = scoring.banded_shift("same input", PROBES, resid, band, base, d)
    assert a == b, "scoring is the correctness contract; it must be bit-identical"


def test_banded_specificity_reduces_to_single_layer_z():
    """Specificity is NOT lost by going multi-layer, and this is why.

    score1 scores a band against ONE shared direction, so the closed-form z is the same
    functional over L*P rows as over P. A one-layer band must therefore reproduce
    shift_and_specificity's z exactly.
    """
    resid = _rng_resid()
    d = np.random.default_rng(21).normal(size=H)
    layer = 24

    b_shift, b_z = scoring.banded_shift_and_specificity(
        "seq", PROBES, resid, [layer], scoring.banded_baseline(PROBES, resid, [layer]), d)

    fn = _single(resid, layer)
    s_shift, s_z = scoring.shift_and_specificity(
        "seq", PROBES, fn, scoring.baseline_unit_rows(PROBES, fn), d)

    assert b_shift == pytest.approx(s_shift, abs=1e-12)
    assert b_z == pytest.approx(s_z, abs=1e-9)


def test_banded_specificity_respects_the_sqrt_H_bound():
    """|z| <= sqrt(H) is what makes z comparable across seasons; the band must not break
    it. The bound survives because ‖δ̄‖ <= ‖Δ‖_F/sqrt(L*P)."""
    resid = _rng_resid()
    band = [19, 23, 27, 31]
    base = scoring.banded_baseline(PROBES, resid, band)
    for seed in range(6):
        d = np.random.default_rng(seed).normal(size=H)
        _, z = scoring.banded_shift_and_specificity("seq", PROBES, resid, band, base, d)
        assert abs(z) <= np.sqrt(H) + 1e-9, f"z={z} exceeds sqrt(H)={np.sqrt(H)}"


def test_banded_specificity_shift_matches_banded_shift():
    """The two entry points must agree on the shift, or the ranked number depends on
    which function the caller happened to use."""
    resid = _rng_resid()
    d = np.random.default_rng(4).normal(size=H)
    band = [19, 23, 27, 31]
    base = scoring.banded_baseline(PROBES, resid, band)
    a = scoring.banded_shift("seq", PROBES, resid, band, base, d,
                             aggregate=scoring.BANDED_MEAN)
    b, _ = scoring.banded_shift_and_specificity("seq", PROBES, resid, band, base, d)
    assert a == pytest.approx(b, abs=1e-12)


def test_both_scores_come_from_one_batched_read():
    """The union-band design exists so the informational Score 2 is FREE.

    banded_shift asks for its own band, so a naive passthrough issues one forward per
    band. main.py wraps the reader in a one-entry cache keyed on the text batch; this
    reproduces that wrapper and asserts both scores cost a single underlying call. If
    this regresses, every submission silently doubles its NDIF spend.
    """
    band1, band2 = [19, 23, 27, 31], [15, 23, 31, 39]
    read = sorted(set(band1) | set(band2))
    pos = {L: i for i, L in enumerate(read)}
    raw = _rng_resid()
    calls = []

    last = {"texts": None, "mat": None}

    def fn(texts, layers):
        key = tuple(texts)
        if last["texts"] != key:
            calls.append(key)
            last["mat"] = raw(list(texts), read)
            last["texts"] = key
        return last["mat"][[pos[L] for L in layers]]

    d1 = np.random.default_rng(1).normal(size=H)
    per2 = np.random.default_rng(2).normal(size=(len(band2), H))
    base1 = scoring.banded_baseline(PROBES, fn, band1)
    base2 = scoring.banded_baseline(PROBES, fn, band2)
    assert len(calls) == 1, "the two baselines share one probe batch"

    calls.clear()
    scoring.banded_shift("seq", PROBES, fn, band1, base1, d1, aggregate=scoring.BANDED_MEAN)
    scoring.banded_shift("seq", PROBES, fn, band2, base2, per2.mean(0),
                         per_layer=per2, aggregate=scoring.PER_LAYER_MIN)
    assert len(calls) == 1, f"both scores must share ONE forward, got {len(calls)}"


def test_config_band_parsing_and_read_union():
    """read_layers() is what one forward pass must fetch; it is the UNION of both bands,
    which is why the informational second score costs no extra NDIF call."""
    from app.config import Settings

    s = Settings(score1_layers="19,23,27,31", score2_layers="15,23,31,39")
    assert s.band1() == [19, 23, 27, 31]
    assert s.band2() == [15, 23, 31, 39]
    assert s.banded() is True
    assert s.read_layers() == [15, 19, 23, 27, 31, 39]   # 6 layers, one call

    off = Settings(score1_layers="", score2_layers="")
    assert off.banded() is False, "empty band must keep the Season 2 single-layer path"
    assert off.read_layers() == []
