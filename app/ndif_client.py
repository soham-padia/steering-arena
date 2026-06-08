"""Reads the layer-L last-token residual for a text — via NDIF (remote) or a
local model for offline dev. Heavy deps (nnsight, torch) are imported lazily so
that importing `app.scoring` stays light and test-friendly.

Backend is chosen by env `SCORING_BACKEND`:
  - "ndif"  (default): NNsight LanguageModel(remote=True) against NDIF, using
    the maintainer's server-side key. This is the seasoned/live path.
  - "local": a small local model (env `LOCAL_MODEL_ID`, default pythia-70m) for
    fast offline development without spending NDIF quota.

NOTE: the residual module path (`_layer_module`) is best-effort for decoder LMs
that expose `.model.layers[i]`. Confirm it against the served model's NNsight
envoy structure before opening a real season; override via env `RESID_LAYER`.
"""

from __future__ import annotations

# NOTE: do NOT import `os` (or other non-whitelisted modules) at module scope —
# NDIF's remote sandbox inspects the trace's module and rejects e.g. `os`
# ("Module os is not whitelisted"). Import such modules locally where needed.
import numpy as np


class ResidualReader:
    def __init__(self, model, *, prepend_bos: bool = True, remote: bool = True, backend: str = "ndif"):
        self.model = model
        self.prepend_bos = prepend_bos
        self.remote = remote
        self.backend = backend

    @classmethod
    def build(cls, model_id: str, backend: str = "ndif", *, ndif_key: str = "", prepend_bos: bool = True) -> "ResidualReader":
        """Explicit constructor (used by the offline extraction/validation scripts)."""
        import nnsight
        from nnsight import LanguageModel

        backend = backend.lower()
        if backend == "ndif":
            if ndif_key:
                # In-memory only (set_default_api_key writes the key to disk).
                nnsight.CONFIG.API.APIKEY = ndif_key
            return cls(LanguageModel(model_id), prepend_bos=prepend_bos, remote=True, backend=backend)
        return cls(LanguageModel(model_id, device_map="cpu"), prepend_bos=prepend_bos, remote=False, backend=backend)

    @classmethod
    def from_settings(cls, settings) -> "ResidualReader":
        import os

        backend = os.getenv("SCORING_BACKEND", "ndif").lower()
        if backend == "ndif":
            return cls.build(settings.model_id, "ndif", ndif_key=settings.ndif_api_key, prepend_bos=settings.prepend_bos)
        local_id = os.getenv("LOCAL_MODEL_ID", "EleutherAI/pythia-70m")
        return cls.build(local_id, "local", prepend_bos=settings.prepend_bos)

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def hidden_size(self) -> int | None:
        try:
            return int(self.model.config.hidden_size)
        except Exception:
            return None

    def _layer_module(self, layer: int):
        # Common decoder-LM layout; override target via RESID_LAYER if needed.
        return self.model.model.layers[layer]

    def last_token_resid(self, text: str, layer: int) -> np.ndarray:
        """Layer-`layer` residual stream at the last token of `text`, as float32 numpy.

        transformers>=5: a decoder layer's `.output` is the hidden tensor itself,
        shape (batch, seq, hidden) — so the last token of batch 0 is output[0, -1, :].
        """
        import torch

        with self.model.trace(text, remote=self.remote):
            saved = self._layer_module(layer).output[0, -1, :].save()

        vec = saved.value if hasattr(saved, "value") else saved
        return np.asarray(vec.detach().to(torch.float32).cpu().numpy(), dtype=np.float32)

    def batch_last_resids(self, texts: list[str], layer: int) -> np.ndarray:
        """Last-token residual for a BATCH of texts in ONE forward pass → (n, hidden).

        Left-pads so the last real token sits at index -1 for every row (HF derives
        position_ids/attention from the mask, so this matches the unbatched value).
        One `.save()` of the batched layer output — remote-safe, and turns N forwards
        into 1 (the live-scoring speed/quota win)."""
        import torch

        tok = self.model.tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        with self.model.trace(list(texts), remote=self.remote):
            saved = self._layer_module(layer).output[:, -1, :].save()  # (n, hidden)
        v = saved.value if hasattr(saved, "value") else saved
        return np.asarray(v.detach().to(torch.float32).cpu().numpy(), dtype=np.float32)

    @property
    def num_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    def last_resids_all_layers(self, text: str) -> np.ndarray:
        """Last-token residual at EVERY layer in one forward pass → (num_layers, hidden).
        Used by the offline layer sweep so extraction costs one forward per text."""
        import torch

        n = self.num_layers
        saved = []
        with self.model.trace(text, remote=self.remote):
            for i in range(n):
                saved.append(self._layer_module(i).output[0, -1, :].save())
        out = []
        for s in saved:
            v = s.value if hasattr(s, "value") else s
            out.append(np.asarray(v.detach().to(torch.float32).cpu().numpy(), dtype=np.float32))
        return np.stack(out)
