"""CAPTCHA gating (security audit H1). No network: covers the disabled and
missing-token branches, plus the /submit 400 when enabled without a token."""

import pytest
from fastapi.testclient import TestClient

from app import captcha


def test_disabled_when_no_secret():
    assert captcha.verify("", "") is True
    assert captcha.verify("anything", "") is True


def test_enabled_rejects_missing_token():
    # secret set but no token → fail closed, without any network call
    assert captcha.verify("", "a-secret") is False


@pytest.fixture()
def client(monkeypatch):
    import app.main as main

    monkeypatch.setenv("DB_BACKEND", "memory")
    main._db = None
    main._scorer = None
    monkeypatch.setattr(main, "get_scorer", lambda: ((lambda s: len(s.split())), (lambda s: float(len(s)))))
    monkeypatch.setattr(main.settings, "turnstile_secret", "test-secret")
    return TestClient(main.app)


def test_submit_blocked_without_captcha_when_enabled(client):
    r = client.post("/submit", json={"handle": "alice", "sequence": "be honest"})
    assert r.status_code == 400
    assert "captcha" in r.json()["error"].lower()
