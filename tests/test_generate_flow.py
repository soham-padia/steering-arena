"""/generate — the public live-generation demo. No NDIF, no Supabase.

The point of these tests is the guard rails, not the text: the demo spends the
maintainer's NDIF quota on every uncached call, so validation, the prefix enum, the
cache, and the durable per-IP/global caps are the contract.
"""

from types import SimpleNamespace

import pytest

from app import db as db_mod
from app import generation
from app.errors import RateLimited
from app.ratelimit import check_generation_limits


def settings(**over):
    base = dict(generate_per_min=4, generate_per_day=40, generate_global_per_day=600)
    base.update(over)
    return SimpleNamespace(**base)


class FakeModel:
    """Stands in for the NNsight model: records what it was asked to continue."""

    def __init__(self):
        self.calls = []
        # seq[0] is already the decoded string in this fake, so decode is identity
        self.tokenizer = SimpleNamespace(decode=lambda toks, skip_special_tokens=True: toks)

    def generate(self, text, max_new_tokens=0, remote=False):
        self.calls.append((text, max_new_tokens))

        class Ctx:
            def __enter__(s):
                return s

            def __exit__(s, *a):
                return False

        self.generator = SimpleNamespace(output=SimpleNamespace(save=lambda: [text + " CONT"]))
        return Ctx()


def reader():
    return SimpleNamespace(model=FakeModel(), remote=False)


# ── prompt hygiene ───────────────────────────────────────────

def test_clean_prompt_collapses_and_strips_control_chars():
    assert generation.clean_prompt("  When my\nfriend\x07  lied, I  ", 240) == "When my friend lied, I"


@pytest.mark.parametrize("bad", ["", "   ", "\n\n"])
def test_empty_prompt_rejected(bad):
    with pytest.raises(generation.GenerationError):
        generation.clean_prompt(bad, 240)


def test_overlong_prompt_rejected():
    with pytest.raises(generation.GenerationError):
        generation.clean_prompt("x" * 241, 240)


# ── the prefix is an enum, never client text ─────────────────

def test_unknown_arm_rejected():
    with pytest.raises(generation.GenerationError):
        generation.generate(reader(), "When I saw her, I", "'; drop table --", 8)


def test_client_cannot_smuggle_its_own_prefix():
    """Only arms present in site_prefixes.json may be prepended."""
    arms = {a["arm"] for a in generation.public_arms()}
    assert "base" in arms
    for hostile in ("pro_top ", "PRO_TOP", "../../etc/passwd", ""):
        assert hostile not in arms or hostile == ""


# ── generation + cache ───────────────────────────────────────

def test_base_arm_sends_prompt_unprefixed_and_strips_input():
    r = reader()
    cont, cached = generation.generate(r, "When I found it, I unique-a", "base", 8)
    assert cont == "CONT" and cached is False
    assert r.model.calls[0] == ("When I found it, I unique-a", 8)


def test_prefixed_arm_prepends_the_frozen_sequence():
    r = reader()
    arm = next(a for a in generation.public_arms() if a["arm"] != "base")
    generation.generate(r, "When I found it, I unique-b", arm["arm"], 8)
    sent = r.model.calls[0][0]
    assert sent.startswith(arm["sequence"]) and sent.endswith("When I found it, I unique-b")


def test_repeat_is_served_from_cache_without_touching_the_model():
    r = reader()
    generation.generate(r, "When I found it, I unique-c", "base", 8)
    cont, cached = generation.generate(r, "When I found it, I unique-c", "base", 8)
    assert cached is True and cont == "CONT"
    assert len(r.model.calls) == 1  # second call never reached NDIF


# ── quota caps (durable, DB-backed) ──────────────────────────

def make_db():
    return db_mod.InMemoryDatabase()


def test_per_minute_cap():
    db, s = make_db(), settings(generate_per_min=2)
    for _ in range(2):
        db.log_generation("ip", "base", "h", False)
    with pytest.raises(RateLimited):
        check_generation_limits(db, "ip", s)


def test_per_day_cap():
    db, s = make_db(), settings(generate_per_min=99, generate_per_day=3)
    for _ in range(3):
        db.log_generation("ip", "base", "h", False)
    with pytest.raises(RateLimited):
        check_generation_limits(db, "ip", s)


def test_global_cap_blocks_a_fresh_ip():
    db, s = make_db(), settings(generate_per_min=99, generate_per_day=99, generate_global_per_day=2)
    for i in range(2):
        db.log_generation(f"someone-{i}", "base", "h", False)
    with pytest.raises(RateLimited):
        check_generation_limits(db, "brand-new-ip", s)


def test_other_ips_do_not_consume_your_personal_budget():
    db, s = make_db(), settings(generate_per_min=2)
    for i in range(5):
        db.log_generation(f"stranger-{i}", "base", "h", False)
    check_generation_limits(db, "me", s)  # must not raise


def test_generation_events_never_store_the_prompt():
    db = make_db()
    db.log_generation("ip", "base", "hash-not-text", False)
    assert "prompt" not in db.generations[0]
    assert db.generations[0]["prompt_hash"] == "hash-not-text"
