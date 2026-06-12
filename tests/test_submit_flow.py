"""Phase 2 — /submit pipeline + HTTP layer, with in-memory DB and fake
tokenizer/scorer (no Supabase, no NDIF)."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import db as db_mod
from app.errors import DuplicateError, RateLimited, ValidationError
from app.submission import MAX_SEQUENCE_CHARS, normalize_key, process_submission

# Fakes: 1 token per whitespace word; score = sequence length.
TOK = lambda s: len(s.split())  # noqa: E731
SCORE = lambda s: float(len(s))  # noqa: E731


def make_db(token_budget=10):
    db = db_mod.InMemoryDatabase()
    db.add_season(id=1, name="t", model_id="m", layer=0, d_version="v", token_budget=token_budget)
    return db


def settings(**over):
    base = dict(rate_per_min=30, rate_per_day=500, global_per_day=5000, token_budget=10)
    base.update(over)
    return SimpleNamespace(**base)


def submit(db, handle, seq, ip="iphash", s=None):
    return process_submission(
        handle=handle, sequence=seq, ip_hash=ip, db=db,
        count_tokens=TOK, score_fn=SCORE, settings=s or settings(),
    )


# ── unit: pipeline ───────────────────────────────────────────

def test_normalize_key():
    assert normalize_key("  Be   Honest ") == "be honest"


def test_happy_path_inserts_and_ranks():
    db = make_db()
    r = submit(db, "alice", "be honest")
    assert r["token_count"] == 2 and r["rank"] == 1
    assert len(db.submissions) == 1


def test_scoreresult_carries_specificity():
    """A ScoreResult-returning score_fn lands shift_raw + specificity in the row
    and the response; the plain-float fakes elsewhere keep working (back-compat)."""
    from app.scoring import ScoreResult

    db = make_db()
    r = process_submission(
        handle="alice", sequence="be honest", ip_hash="ip", db=db,
        count_tokens=TOK, score_fn=lambda s: ScoreResult(2.0, 2.0, 14.5),
        settings=settings(),
    )
    assert r["score"] == 2.0 and r["specificity"] == 14.5
    row = db.submissions[0]
    assert row["shift_raw"] == 2.0 and row["specificity"] == 14.5


def test_ranking_orders_by_score():
    db = make_db()
    submit(db, "alice", "aa")            # score 2
    r2 = submit(db, "bob", "aaaaaaaa")   # score 8 → rank 1
    assert r2["rank"] == 1
    assert db.rank_for(1, 2.0) == 2      # the shorter one dropped to rank 2


def test_dual_rank_pro_and_anti():
    db = make_db()
    submit(db, "a", "aa")              # score 2
    r2 = submit(db, "b", "aaaaaaaa")   # score 8
    # pro = most positive tops; anti = most negative tops (geometric opposite)
    assert r2["pro_rank"] == 1 and r2["anti_rank"] == 2
    assert db.rank_for(1, 2.0, higher_is_better=False) == 1   # the low score tops anti
    assert db.rank_for(1, 8.0, higher_is_better=False) == 2


def test_duplicate_rejected_with_existing():
    db = make_db()
    submit(db, "alice", "Be Honest")
    with pytest.raises(DuplicateError) as ei:
        submit(db, "mallory", "be   honest")  # same normalized key
    assert ei.value.rank == 1 and ei.value.existing["user_handle"] == "alice"
    assert len(db.submissions) == 1          # no second row, no re-score


def test_over_budget_rejected():
    db = make_db(token_budget=3)
    with pytest.raises(ValidationError):
        submit(db, "alice", "one two three four", s=settings(token_budget=3))
    assert len(db.submissions) == 0


@pytest.mark.parametrize("handle", ["", "x" * 33, "bad/handle", "no\tbad"])
def test_bad_handle_rejected(handle):
    with pytest.raises(ValidationError):
        submit(make_db(), handle, "ok seq")


@pytest.mark.parametrize("seq", ["", "   ", "x" * (MAX_SEQUENCE_CHARS + 1)])
def test_bad_sequence_rejected(seq):
    with pytest.raises(ValidationError):
        submit(make_db(), "alice", seq)


def test_per_ip_rate_limit():
    db = make_db()
    s = settings(rate_per_min=2)
    submit(db, "a", "one", s=s)
    submit(db, "a", "two", s=s)
    with pytest.raises(RateLimited):
        submit(db, "a", "three", s=s)


def test_global_daily_ceiling():
    db = make_db()
    s = settings(global_per_day=2)
    submit(db, "a", "one", ip="ip1", s=s)
    submit(db, "b", "two", ip="ip2", s=s)
    with pytest.raises(RateLimited):  # different IP, but global cap hit
        submit(db, "c", "three", ip="ip3", s=s)


# ── HTTP layer ───────────────────────────────────────────────

@pytest.fixture()
def client(monkeypatch):
    import app.main as main

    monkeypatch.setenv("DB_BACKEND", "memory")
    main._db = None
    main._scorer = None
    monkeypatch.setattr(main, "get_scorer", lambda: (TOK, SCORE))
    return TestClient(main.app)


def test_http_submit_and_leaderboard(client):
    r = client.post("/submit", json={"handle": "alice", "sequence": "be honest now"})
    assert r.status_code == 200, r.text
    assert r.json()["token_count"] == 3

    lb = client.get("/leaderboard").json()
    assert lb["entries"][0]["handle"] == "alice"


def test_http_duplicate_returns_409(client):
    client.post("/submit", json={"handle": "alice", "sequence": "be honest"})
    r = client.post("/submit", json={"handle": "bob", "sequence": "be   honest"})
    assert r.status_code == 409
    assert "score" in r.json()


def test_http_backend_failure_returns_503_not_500(client, monkeypatch):
    """An exception mid-score (e.g. NDIF deployment down) must surface as a clean
    503 with a JSON error — a raw 500 renders as a misleading 'Network error'."""
    import app.main as main

    def boom(_seq):
        raise RuntimeError("Error submitting request to model deployment.")

    monkeypatch.setattr(main, "get_scorer", lambda: (TOK, boom))
    r = client.post("/submit", json={"handle": "alice", "sequence": "be honest"})
    assert r.status_code == 503
    assert "temporarily unavailable" in r.json()["error"]


def test_http_over_budget_returns_400(client):
    import app.main as main
    seq = " ".join(["w"] * (main.settings.token_budget + 5))  # fake tokens just over the budget
    r = client.post("/submit", json={"handle": "alice", "sequence": seq})
    assert r.status_code == 400
    assert "budget" in r.json()["error"].lower()


def test_http_pro_and_anti_boards_order_oppositely(client):
    client.post("/submit", json={"handle": "lo", "sequence": "aa"})        # score 2
    client.post("/submit", json={"handle": "hi", "sequence": "aaaaaaaa"})  # score 8
    pro = client.get("/leaderboard?board=pro").json()
    anti = client.get("/leaderboard?board=anti").json()
    assert pro["entries"][0]["handle"] == "hi"    # highest score tops pro
    assert anti["entries"][0]["handle"] == "lo"   # lowest score tops anti
    assert anti["board"] == "anti"
