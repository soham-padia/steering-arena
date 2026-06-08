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

import os

import numpy as np


class ResidualReader:
    def __init__(self, model, *, prepend_bos: bool = True, remote: bool = True, backend: str = "ndif"):
        self.model = model
        self.prepend_bos = prepend_bos
        self.remote = remote
        self.backend = backend

    @classmethod
    def from_settings(cls, settings) -> "ResidualReader":
        import nnsight
        from nnsight import LanguageModel

        backend = os.getenv("SCORING_BACKEND", "ndif").lower()
        if backend == "ndif":
            if settings.ndif_api_key:
                # Set the key IN MEMORY only. (set_default_api_key writes the key
                # to a .config file on disk in the nnsight install dir — needless
                # persistence of the crown-jewel secret; avoid it.)
                nnsight.CONFIG.API.APIKEY = settings.ndif_api_key
            model = LanguageModel(settings.model_id)
            return cls(model, prepend_bos=settings.prepend_bos, remote=True, backend=backend)

        local_id = os.getenv("LOCAL_MODEL_ID", "EleutherAI/pythia-70m")
        model = LanguageModel(local_id, device_map="cpu")
        return cls(model, prepend_bos=settings.prepend_bos, remote=False, backend=backend)

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
        """Layer-`layer` residual stream at the last token of `text`, as float32 numpy."""
        import torch

        with self.model.trace(text, remote=self.remote):
            hidden = self._layer_module(layer).output[0]  # (batch, seq, hidden)
            saved = hidden[0, -1, :].save()

        vec = saved.value if hasattr(saved, "value") else saved
        return np.asarray(vec.detach().to(torch.float32).cpu().numpy(), dtype=np.float32)
