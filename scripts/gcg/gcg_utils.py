"""GCG helpers, adapted from Jesse Li's steering-arena-optim (MIT) for a BANDED objective.

Upstream: https://github.com/jesse-li-agent-projects/steering-arena-optim  (see README.md
and LICENSE.upstream). Structure, comments and behaviour are kept as close to the original
as the change allows; the diff is deliberately small.

WHAT IS DIFFERENT FROM UPSTREAM
  compute_scores_batch  takes `layers: list[int]` and registers one forward hook per band
                        layer, then aggregates across them (mean, or per-layer min).
  truncate_to_layer     keeps max(band) + 2 blocks instead of layer + 2.
  roundtrip_ok          new. GCG optimises token IDS; the board scores a STRING.

WHAT IS UNCHANGED, AND WHY IT MATTERS
  The per-probe baseline cos(R_L(probe), d) is NOT subtracted, exactly as upstream. It is
  constant with respect to the prefix, so argmax is identical and the cheaper quantity is
  optimised. app/scoring.py applies the baseline when reporting an actual score. This holds
  for the banded mean and for the per-layer min alike, which is why the port is small.
"""

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch as t
from jaxtyping import Float, Int

MEAN = "banded_mean"
MIN = "per_layer_min"
SOFTMIN = "per_layer_softmin"   # SEARCH-ONLY surrogate for MIN; never a reported score

# Temperature for SOFTMIN. -> MIN as T->0, -> mean as T->inf. 0.02 is ~10% of the observed
# per-layer cosine spread on the seed pairs, so it is harsh enough to still punish a weak
# layer while spreading gradient across all of them.
SOFTMIN_T = 0.02


def _aggregate(stacked, aggregate: str):
    """(n_layers, N) per-layer cosines -> (N,) aggregated score.

    MEAN     the ranked Score 1 objective.
    MIN      the true Score 2 objective. Exact, but its gradient reaches only the ARGMIN
             layer, so a GCG step gets signal about one depth out of four.
    SOFTMIN  a smooth surrogate for MIN, for the SEARCH ONLY:

                 softmin_T(x) = -T * log( mean_L exp(-x_L / T) )

             Gradient flows to every layer, weighted toward the weakest -- a strictly
             better optimisation signal than a hard min. It is NOT a scoring mode: the
             board computes MIN, so anything reported must use MIN.

    A geometric mean was considered and REJECTED. The per-layer cosines are signed and, on
    the seed pairs, 99.6% of texts have the SAME sign at all four band layers (49.6% all
    positive, 50.0% all negative). A 4-way product is therefore positive 99.6% of the time,
    half of it from four negatives multiplying -- so the geometric mean of a text that is
    anti-human at every layer comes out strongly POSITIVE (measured: per-layer
    [-0.196,-0.175,-0.154,-0.141] -> GM +0.165 against an arithmetic mean of -0.167). It
    destroys the sign, and because the layers are correlated that is half the corpus, not
    an edge case.
    """
    if aggregate == MEAN:
        return stacked.mean(dim=0)
    if aggregate == MIN:
        return stacked.min(dim=0).values
    if aggregate == SOFTMIN:
        import torch as _t
        return -SOFTMIN_T * _t.logsumexp(-stacked / SOFTMIN_T
                                         - _t.log(_t.tensor(float(stacked.shape[0]))), dim=0)
    raise ValueError(f"unknown aggregate: {aggregate}")


# %%
def cos_sim(a: np.ndarray, b: np.ndarray):
    """
    Compute cosine similarity of two vectors.
    """
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        raise ValueError("cosine of a zero vector is undefined")
    return np.dot(a, b) / (na * nb)


def compose(seq: str, probe: str) -> str:
    """How a candidate sequence is prepended to a probe. Fixed for determinism."""
    return f"{seq} {probe}"


def load_banded_direction(path) -> tuple[np.ndarray, np.ndarray, list[int], dict]:
    """(d_bar, per_layer, band, meta) from a Season 3 d_<version>.npz.

    Upstream's load_direction returned only `d`; the Season 3 files additionally carry
    `band` (the layers this direction was fitted for) and `per_layer` (one direction per
    band layer, which the per-layer-min objective needs).
    """
    data = np.load(path, allow_pickle=True)
    d = np.asarray(data["d"], dtype=np.float32)
    per = np.asarray(data["per_layer"], dtype=np.float32) if "per_layer" in data else None
    band = [int(x) for x in data["band"]] if "band" in data else []
    meta = json.loads(str(data["meta"])) if "meta" in data else {}
    return d, per, band, meta


def load_prompt_suffixes(path) -> list[str]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj["prompts"] if isinstance(obj, dict) else list(obj)


def d_tag(d: np.ndarray) -> str:
    """Short content hash of a direction, recorded in every checkpoint.

    Content-addressed rather than named after d_version, because Season 3 ships two
    directions that SHARE a d_version and differ only by role -- a name-based tag would
    collide between exactly the two vectors that must stay distinguishable.
    """
    import hashlib

    return hashlib.sha256(np.asarray(d, dtype=np.float64).tobytes()).hexdigest()[:12]


def roundtrip_ok(tokenizer, ids) -> bool:
    """Does this token sequence survive decode -> encode unchanged?

    NEW, and the single most important addition. GCG optimises token IDS while the
    leaderboard scores a STRING, so the real pipeline is

        optimise ids -> decode -> submit text -> board re-tokenises

    and for adversarial sequences encode(decode(ids)) != ids in general. When it differs,
    the optimiser is reporting a score for a token sequence the board will never evaluate.
    This is the leading explanation for the 0.005-0.031 gap between upstream's reported
    scores and its own board entries (_communication/004), and it was never tested there.
    """
    seq = [int(x) for x in ids]
    return tokenizer(tokenizer.decode(seq), add_special_tokens=False)["input_ids"] == seq


def _decoder_blocks_attr(trunk) -> str:
    """Name of the attribute holding trunk's decoder blocks: `.layers` (Llama/OLMo-style) or `.h` (GPT-2)."""
    if hasattr(trunk, "layers"):
        return "layers"
    if hasattr(trunk, "h"):
        return "h"
    raise ValueError(f"cannot locate decoder blocks on {type(trunk).__name__}")


def truncate_to_layer(model, layer: int):
    """
    Monkey-patches model in-place, dropping transformer blocks above `layer` so the forward stops right after
    producing hidden_states[layer + 1].

    The probe only reads one layer's residual stream, so running the remaining
    blocks (and the final norm / lm_head, which are skipped by calling
    `model.base_model` for the forward) is wasted work. Mutates and returns
    `model`. Works for Llama/OLMo-style trunks (`.layers`) and GPT-2 (`.h`).

    We keep `layer + 2` blocks, not `layer + 1`: the model applies the final
    norm to the *last* entry of `hidden_states`, so we keep one extra raw block
    output so that `hidden_states[layer + 1]` stays the pre-norm block output the
    probe reads (identical to the untruncated model).

    ADAPTED: for a band, pass max(band) -- every band layer must survive.
    """
    trunk = model.base_model
    attr = _decoder_blocks_attr(trunk)
    blocks = getattr(trunk, attr)
    assert layer + 1 <= len(blocks), f"layer {layer} exceeds model depth {len(blocks)}"
    keep = min(layer + 2, len(blocks))
    setattr(trunk, attr, blocks[:keep])
    for name in ("num_hidden_layers", "n_layer"):
        if hasattr(model.config, name):
            setattr(model.config, name, keep)
    return model


def compute_scores_batch(
    trunk,
    ctrl_embed: Float[t.Tensor, "n_cand ctrl_seq d_model"],
    sfx_embed: Float[t.Tensor, "n_sfx sfx_seq d_model"],
    sfx_mask: Int[t.Tensor, "n_sfx sfx_seq"],
    n_sfx_tokens: Int[t.Tensor, "n_sfx"],
    dirs: Float[t.Tensor, "n_layers d_model"],
    layers: list[int],
    aggregate: str = MEAN,
    chunk: int | None = None,
) -> Float[t.Tensor, "n_cand"]:
    """Mean-over-suffix BANDED probe score for each of n_cand candidate prefixes.

    Each candidate prefix is scored against all n_sfx suffixes in one (chunked) batched forward. All candidates share the same fixed
    suffixes; only the control-token prefix varies.

    This function is differentiable w.r.t. ctrl_embed, so gradients can flow back through it.

    ADAPTED FROM UPSTREAM: `layer: int` -> `layers: list[int]` plus `dirs` and `aggregate`.
    One forward hook per band layer instead of one, then

        MEAN  mean over L of cos(R_L, dirs[0])      -- one shared direction (Score 1)
        MIN   min  over L of cos(R_L, dirs[j])      -- a direction per layer  (Score 2)

    `min` is subdifferentiable: the gradient flows to the argmin layer, which is exactly
    the GCG signal wanted -- improve whichever depth you are currently worst at.

    :param dirs: (1, d_model) for MEAN, or (len(layers), d_model) for MIN. Unit rows,
        float32, CPU (probe scoring is done on CPU by convention, as upstream).
    :param layers: band layers, ascending. Each reads hidden_states[L + 1].
    :param chunk: candidates per forward pass (memory knob); None = all at once
    :returns: mean-over-suffix score per candidate; float32, CPU.
    """
    n_cand, ctrl_seq, _ = ctrl_embed.shape
    n_sfx, sfx_seq, d_model = sfx_embed.shape
    seq = ctrl_seq + sfx_seq

    if aggregate in (MIN, SOFTMIN) and dirs.shape[0] != len(layers):
        raise ValueError(f"{MIN} needs one direction per layer: "
                         f"{dirs.shape[0]} dirs for {len(layers)} layers")

    # Right padding => last real token sits at ctrl_seq + n_sfx - 1.
    gather_pos = n_sfx_tokens + ctrl_seq - 1  # (n_sfx,)
    ctrl_mask = t.ones(n_sfx, ctrl_seq, dtype=sfx_mask.dtype)

    if chunk is None:
        chunk = n_cand

    # hidden_states[layer + 1] is the output of decoder block `layer` (0-indexed;
    # hidden_states[0] is the embedding output). output_hidden_states=True would
    # materialize and hold *every* block's output at once (all ~layer+2 kept
    # blocks) even though only these are read; forward hooks capture just the
    # tensors we need instead. One hook per band layer.
    blocks = getattr(trunk, _decoder_blocks_attr(trunk))
    captured: dict[int, t.Tensor] = {}

    def _make_capture(L):
        def _capture(module, inputs, output):
            captured[L] = output[0] if isinstance(output, tuple) else output
        return _capture

    handles = [blocks[L].register_forward_hook(_make_capture(L)) for L in layers]
    try:
        score_chunks = []
        for ce in t.split(ctrl_embed, chunk):
            c = ce.shape[0]  # chunk size; may be smaller than `chunk` for last iter

            ctrl_e = ce[:, None].expand(c, n_sfx, ctrl_seq, d_model)
            sfx_e = sfx_embed[None].expand(c, n_sfx, sfx_seq, d_model)
            inp = t.cat([ctrl_e, sfx_e], dim=2).reshape(c * n_sfx, seq, d_model)

            # Build attention mask
            cm = ctrl_mask[None].expand(c, n_sfx, ctrl_seq)
            sm = sfx_mask[None].expand(c, n_sfx, sfx_seq)
            attn = t.cat([cm, sm], dim=2).reshape(c * n_sfx, seq)

            trunk(inputs_embeds=inp, attention_mask=attn)

            pos = gather_pos[None].expand(c, n_sfx).reshape(c * n_sfx)
            per_layer_cos = []
            for j, L in enumerate(layers):
                h = captured[L]  # (c*n_sfx, seq, d_model)
                # recall dirs is on CPU and float32
                acts = h[t.arange(c * n_sfx), pos].cpu().float()
                norm = t.linalg.norm(acts, dim=-1)
                dvec = dirs[0] if aggregate == MEAN else dirs[j]  # MIN/SOFTMIN: one per layer
                per_layer_cos.append((acts @ dvec) / norm)  # (c*n_sfx,)

            stacked = t.stack(per_layer_cos)  # (n_layers, c*n_sfx)
            sc = _aggregate(stacked, aggregate)
            score_chunks.append(sc.reshape(c, n_sfx).mean(dim=1))
    finally:
        for h in handles:
            h.remove()

    return t.cat(score_chunks)


def plan_replica_placement(
    model, n_replicas=None, gpus_per_replica=None, mem_fraction=0.9
):
    """Decide GPU groups for data-parallel replicas from runtime hardware.

    Detects device count and per-card capacity so the same code adapts to
    2/4/8 GPUs of any size; all knobs are overridable.

    :param model: the (already truncated) model, used to measure its footprint
    :param n_replicas: force replica count; None => n_gpus // gpus_per_replica
    :param gpus_per_replica: force GPUs per replica; None => from capacity
    :param mem_fraction: usable fraction of each card (headroom for activations)
    :returns: list of GPU-index groups, one per replica, e.g. [[0, 1], [2, 3]].
        Empty list => no CUDA => caller should use the single-model path.
    """
    n_gpus = t.cuda.device_count()
    if n_gpus == 0:
        return []
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # min() is conservative if the cards are heterogeneous.
    budget = (
        min(t.cuda.get_device_properties(i).total_memory for i in range(n_gpus))
        * mem_fraction
    )
    if gpus_per_replica is None:
        gpus_per_replica = max(1, math.ceil(model_bytes / budget))
    if gpus_per_replica > n_gpus:
        raise RuntimeError(
            f"model needs ~{gpus_per_replica} GPUs but only {n_gpus} present"
        )
    if n_replicas is None:
        n_replicas = n_gpus // gpus_per_replica
    return [
        list(range(r * gpus_per_replica, (r + 1) * gpus_per_replica))
        for r in range(n_replicas)
    ]


def build_replicas(model, groups, mem_fraction=0.9):
    """Clone a (truncated) model onto each GPU group for data parallelism.

    Loads/truncates happen once by the caller; this pays only the clone cost.
    Each replica is dispatched (pipelined) across the GPUs of its group.

    :param model: the truncated model to clone (moved to CPU as the source)
    :param groups: GPU-index groups from plan_replica_placement
    :param mem_fraction: usable fraction of each card for the device map
    :returns: list of dispatched replica models, one per group
    """
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.hooks import remove_hook_from_module

    # The source model still carries accelerate's dispatch hooks from its
    # initial device_map="auto" load; those hooks pin each submodule's
    # execution device. Detach them first, or copy.deepcopy below propagates
    # that stale per-submodule placement into every replica, and the fresh
    # dispatch_model call for each replica layers new hooks on top of the old
    # ones instead of replacing them -- leaving some submodules on their
    # original devices while others follow the new device map.
    remove_hook_from_module(model, recurse=True)
    model = model.to("cpu")  # truncated weights now in host RAM
    t.cuda.empty_cache()
    replicas = []
    for grp in groups:
        rep = copy.deepcopy(model)
        max_mem = {
            g: int(t.cuda.get_device_properties(g).total_memory * mem_fraction)
            for g in grp
        }
        dev_map = infer_auto_device_map(
            rep,
            max_memory=max_mem,
            no_split_module_classes=model._no_split_modules,
        )
        replicas.append(dispatch_model(rep, device_map=dev_map))
    del model
    return replicas
