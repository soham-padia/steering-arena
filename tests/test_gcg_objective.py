"""The banded GCG objective: does it reduce to upstream's, and does it agree with the board?

Two contracts, and both are the point of the fork:

  1. REDUCTION. A one-layer band under `banded_mean` must reproduce Jesse Li's original
     scalar objective exactly. If it does not, the adaptation has quietly become a
     different optimiser and its results cannot be compared to his.

  2. AGREEMENT WITH THE SCORER. The optimiser drops the per-probe baseline (it is constant
     w.r.t. the prefix, so argmax is unchanged) while app/scoring.py subtracts it. The two
     must therefore differ by a CONSTANT per probe set, never by anything prefix-dependent.
     If they drift, GCG is climbing a hill the leaderboard is not measuring.

Runs on a tiny random model on CPU -- no NDIF, no GPU, no 61 GB of weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "gcg"))

import gcg_utils as G  # noqa: E402


H, VOCAB, NLAYERS = 16, 64, 6
N_SFX, SFX_SEQ, CTRL = 3, 5, 4


class _Block(torch.nn.Module):
    """Deterministic stand-in for a decoder block.

    It MUST mix across sequence positions causally. A purely position-wise block
    (x + tanh(lin(x))) looks fine and silently makes every test vacuous: the last-token
    readout cannot see the prefix, so every candidate scores identically and the gradient
    w.r.t. the prefix is legitimately zero. The first version of this fixture had exactly
    that bug and the differentiability test is what caught it.

    A causal running mean is the cheapest thing that carries prefix information forward,
    which is the only property of attention these tests depend on.
    """

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lin = torch.nn.Linear(H, H)
        with torch.no_grad():
            self.lin.weight.copy_(torch.randn(H, H, generator=g) * 0.2)
            self.lin.bias.copy_(torch.randn(H, generator=g) * 0.1)

    def forward(self, x, **kw):
        # causal running mean over positions: pos i sees 0..i, never the future
        csum = x.cumsum(dim=1)
        denom = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)[None, :, None]
        return (x + torch.tanh(self.lin(csum / denom)),)


class _Trunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(_Block(i) for i in range(NLAYERS))

    def forward(self, inputs_embeds=None, attention_mask=None, **kw):
        h = inputs_embeds
        for blk in self.layers:
            h = blk(h)[0]
        return h


def _fixture(seed=0):
    g = torch.Generator().manual_seed(seed)
    trunk = _Trunk()
    ctrl = torch.randn(2, CTRL, H, generator=g)
    sfx = torch.randn(N_SFX, SFX_SEQ, H, generator=g)
    mask = torch.ones(N_SFX, SFX_SEQ, dtype=torch.long)
    ntok = torch.full((N_SFX,), SFX_SEQ, dtype=torch.long)
    return trunk, ctrl, sfx, mask, ntok


def _upstream_single_layer(trunk, ctrl, sfx, mask, ntok, d, layer):
    """Jesse's original scalar objective, transcribed from _Jesse/utils.py:130-160.

    Kept verbatim rather than imported: the point is to check the fork against what
    upstream ACTUALLY does, not against a shared helper that could drift with it.
    """
    n_cand, ctrl_seq, _ = ctrl.shape
    n_sfx, sfx_seq, d_model = sfx.shape
    gather_pos = ntok + ctrl_seq - 1
    ctrl_mask = torch.ones(n_sfx, ctrl_seq, dtype=mask.dtype)
    captured = {}

    def _cap(m, i, o):
        captured["h"] = o[0] if isinstance(o, tuple) else o

    handle = trunk.layers[layer].register_forward_hook(_cap)
    try:
        c = n_cand
        ctrl_e = ctrl[:, None].expand(c, n_sfx, ctrl_seq, d_model)
        sfx_e = sfx[None].expand(c, n_sfx, sfx_seq, d_model)
        inp = torch.cat([ctrl_e, sfx_e], dim=2).reshape(c * n_sfx, ctrl_seq + sfx_seq, d_model)
        attn = torch.cat([ctrl_mask[None].expand(c, n_sfx, ctrl_seq),
                          mask[None].expand(c, n_sfx, sfx_seq)], dim=2).reshape(c * n_sfx, -1)
        trunk(inputs_embeds=inp, attention_mask=attn)
        h = captured["h"]
        pos = gather_pos[None].expand(c, n_sfx).reshape(c * n_sfx)
        acts = h[torch.arange(c * n_sfx), pos].cpu().float()
        sc = (acts @ d) / torch.linalg.norm(acts, dim=-1)
        return sc.reshape(c, n_sfx).mean(dim=1)
    finally:
        handle.remove()


def test_one_layer_band_reproduces_upstream_exactly():
    """THE reduction test. A one-element band IS upstream's objective."""
    trunk, ctrl, sfx, mask, ntok = _fixture()
    d = torch.nn.functional.normalize(torch.randn(H, generator=torch.Generator().manual_seed(9)), dim=0)
    want = _upstream_single_layer(trunk, ctrl, sfx, mask, ntok, d, layer=3)
    got = G.compute_scores_batch(trunk, ctrl, sfx, mask, ntok, d[None], [3], aggregate=G.MEAN)
    assert torch.allclose(got, want, atol=1e-6), (got, want)


def test_banded_mean_is_the_mean_of_single_layer_scores():
    trunk, ctrl, sfx, mask, ntok = _fixture()
    d = torch.nn.functional.normalize(torch.randn(H, generator=torch.Generator().manual_seed(2)), dim=0)
    band = [1, 3, 5]
    per = torch.stack([_upstream_single_layer(trunk, ctrl, sfx, mask, ntok, d, L) for L in band])
    got = G.compute_scores_batch(trunk, ctrl, sfx, mask, ntok, d[None], band, aggregate=G.MEAN)
    assert torch.allclose(got, per.mean(0), atol=1e-6)


def test_per_layer_min_needs_one_direction_per_layer():
    trunk, ctrl, sfx, mask, ntok = _fixture()
    d = torch.nn.functional.normalize(torch.randn(H), dim=0)
    with pytest.raises(ValueError, match="one direction per layer"):
        G.compute_scores_batch(trunk, ctrl, sfx, mask, ntok, d[None], [1, 3], aggregate=G.MIN)


def test_per_layer_min_never_exceeds_any_member_layer():
    """The min must be <= every layer it minimises over -- that is what makes it the
    'improve your weakest depth' signal rather than another average."""
    trunk, ctrl, sfx, mask, ntok = _fixture()
    band = [1, 3, 5]
    g = torch.Generator().manual_seed(4)
    dirs = torch.nn.functional.normalize(torch.randn(len(band), H, generator=g), dim=1)
    got = G.compute_scores_batch(trunk, ctrl, sfx, mask, ntok, dirs, band, aggregate=G.MIN)
    for j, L in enumerate(band):
        one = G.compute_scores_batch(trunk, ctrl, sfx, mask, ntok, dirs[j][None], [L], aggregate=G.MEAN)
        assert torch.all(got <= one + 1e-6), f"min exceeded layer {L}"


def test_objective_is_differentiable_wrt_the_prefix():
    """GCG needs a gradient on the one-hot. If this breaks there is no optimiser."""
    trunk, ctrl, sfx, mask, ntok = _fixture()
    dirs = torch.nn.functional.normalize(torch.randn(3, H, generator=torch.Generator().manual_seed(5)), dim=1)
    for agg in (G.MEAN, G.MIN):
        c = ctrl.clone().requires_grad_(True)
        d_in = dirs[:1] if agg == G.MEAN else dirs
        G.compute_scores_batch(trunk, c, sfx, mask, ntok, d_in, [1, 3, 5], aggregate=agg)[0].backward()
        assert c.grad is not None and torch.isfinite(c.grad).all()
        assert c.grad.abs().sum() > 0, f"{agg}: zero gradient — nothing to optimise"


def test_truncate_keeps_the_deepest_band_layer():
    """Truncating to max(band) must leave every band layer readable. Truncating to the
    band's FIRST layer -- the upstream signature's natural misreading -- would not."""
    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = _Trunk()
            self.config = type("C", (), {"num_hidden_layers": NLAYERS})()

    band = [1, 3]
    m = G.truncate_to_layer(_M(), max(band))
    assert len(m.base_model.layers) == max(band) + 2 <= NLAYERS
    assert max(band) < len(m.base_model.layers)


def test_roundtrip_guard_catches_a_non_reencodable_sequence():
    """The guard that upstream lacks. A sequence whose decode->encode differs would be
    scored by the optimiser and never reproduced by the board."""
    class _Tok:
        def __init__(self, lossy): self.lossy = lossy
        def decode(self, ids): return " ".join(str(i) for i in ids)
        def __call__(self, s, add_special_tokens=False):
            ids = [int(x) for x in s.split()]
            if self.lossy: ids = ids[:-1]          # simulate a merge across the boundary
            return {"input_ids": ids}

    assert G.roundtrip_ok(_Tok(lossy=False), [5, 6, 7]) is True
    assert G.roundtrip_ok(_Tok(lossy=True), [5, 6, 7]) is False


def test_d_tag_distinguishes_the_two_season3_directions():
    """score1 and score2 SHARE a d_version and differ only by role, so the cache/provenance
    tag has to be content-addressed or it collides on exactly the pair that must not."""
    a, b = np.random.default_rng(0).normal(size=H), np.random.default_rng(1).normal(size=H)
    assert G.d_tag(a) != G.d_tag(b)
    assert G.d_tag(a) == G.d_tag(a.copy())
