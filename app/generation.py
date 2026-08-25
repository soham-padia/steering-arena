"""Live prefix demo: continue a user's sentence with and without a leaderboard prefix.

This is the only place in the app that GENERATES text rather than reading activations,
and every uncached call spends the maintainer's NDIF quota, so the guard rails are part
of the module rather than the caller's problem:

  • only the frozen prefixes in data/analysis/site_prefixes.json can be prepended —
    the arm is an enum, never free text from the client
  • the user's prompt is length-capped and control-stripped before it reaches NDIF
  • output is capped at settings.generate_max_new tokens
  • identical (arm, prompt) pairs are served from an in-process cache, so a refresh
    loop costs nothing

Rate limiting and the durable per-IP counters live in app/ratelimit.py + the
generation_events table; this module assumes the caller has already passed them.
"""

from __future__ import annotations

import json
import re
import threading
import unicodedata
from collections import OrderedDict
from pathlib import Path

from app.scoring import compose

PREFIX_FILE = Path(__file__).resolve().parent.parent / "data" / "analysis" / "site_prefixes.json"
_CACHE_MAX = 512
_cache: OrderedDict[tuple[str, str], str] = OrderedDict()
_cache_lock = threading.Lock()
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class GenerationError(Exception):
    """Bad request from the client (message is safe to show)."""


def load_prefixes() -> dict:
    """{arm: {sequence, label, kind, score}} — frozen on disk, never client-supplied."""
    if not PREFIX_FILE.exists():
        return {}
    return json.loads(PREFIX_FILE.read_text())["arms"]


def public_arms() -> list[dict]:
    """What the frontend may offer. The raw prefix text is public (the board is public),
    but it is sent for display only — the server never accepts one back."""
    out = []
    for arm, a in load_prefixes().items():
        out.append({"arm": arm, "label": a["label"], "kind": a.get("kind", ""),
                    "score": a.get("score"), "sequence": a["sequence"]})
    return out


def clean_prompt(raw: str, max_chars: int) -> str:
    """Normalize + bound what a stranger typed before it becomes an NDIF forward pass."""
    if not isinstance(raw, str):
        raise GenerationError("Prompt must be text.")
    text = unicodedata.normalize("NFC", raw).replace("\r", " ").replace("\n", " ")
    text = _CONTROL.sub("", text).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        raise GenerationError("Type a sentence for the model to continue.")
    if len(text) > max_chars:
        raise GenerationError(f"Keep it under {max_chars} characters.")
    return text


def continuation_of(text: str, model_input: str, prompt: str) -> str:
    """Strip the model input, leaving only what the model added. Cuts at the FIRST
    occurrence of the prompt: the input ends with it, so a later copy is generated text
    and must be kept (see scripts/prefix_behavior_eval)."""
    if text.startswith(model_input):
        return text[len(model_input):].strip()
    i = text.find(prompt)
    return text[i + len(prompt):].strip() if i >= 0 else text.strip()


def _cache_get(key):
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    return None


def _cache_put(key, value):
    with _cache_lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)


def generate(reader, prompt: str, arm: str, max_new: int) -> tuple[str, bool]:
    """(continuation, was_cached). `arm` must name a frozen prefix; 'base' means none."""
    prefixes = load_prefixes()
    if arm not in prefixes:
        raise GenerationError("Unknown prefix.")
    prefix = prefixes[arm]["sequence"]

    key = (arm, prompt)
    hit = _cache_get(key)
    if hit is not None:
        return hit, True

    model_input = compose(prefix, prompt) if prefix else prompt
    with reader.model.generate(model_input, max_new_tokens=max_new, remote=reader.remote):
        out = reader.model.generator.output.save()
    seq = out.value if hasattr(out, "value") else out
    text = reader.model.tokenizer.decode(seq[0], skip_special_tokens=True)
    cont = continuation_of(text, model_input, prompt)
    _cache_put(key, cont)
    return cont, False
