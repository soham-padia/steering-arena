"""Phase 0 smoke test: the app boots and /health + /season respond."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model" in body and "season" in body


def test_season_shape():
    r = client.get("/season")
    assert r.status_code == 200
    body = r.json()
    for key in ("id", "name", "model_id", "layer", "d_version", "token_budget", "scoring_mode"):
        assert key in body
