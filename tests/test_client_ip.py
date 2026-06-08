"""X-Forwarded-For handling must resist client spoofing (security fix)."""

from types import SimpleNamespace

import app.main as main


def _req(xff=None, peer="10.0.0.9"):
    headers = {"x-forwarded-for": xff} if xff is not None else {}
    return SimpleNamespace(headers=headers, client=SimpleNamespace(host=peer))


def test_uses_proxy_inserted_entry_not_spoofed(monkeypatch):
    # Attacker prepends a fake IP; the trusted proxy (1 hop) appends the real one.
    monkeypatch.setattr(main.settings, "trusted_proxy_hops", 1)
    ip = main._client_ip(_req(xff="1.2.3.4, 203.0.113.7"))
    assert ip == "203.0.113.7"  # rightmost (proxy-inserted), not the spoofed leftmost


def test_two_trusted_hops(monkeypatch):
    monkeypatch.setattr(main.settings, "trusted_proxy_hops", 2)
    ip = main._client_ip(_req(xff="spoof, 203.0.113.7, 10.0.0.1"))
    assert ip == "203.0.113.7"


def test_falls_back_to_peer_without_xff(monkeypatch):
    monkeypatch.setattr(main.settings, "trusted_proxy_hops", 1)
    assert main._client_ip(_req(xff=None, peer="10.0.0.9")) == "10.0.0.9"


def test_falls_back_when_xff_shorter_than_hops(monkeypatch):
    monkeypatch.setattr(main.settings, "trusted_proxy_hops", 2)
    assert main._client_ip(_req(xff="1.2.3.4", peer="10.0.0.9")) == "10.0.0.9"
