import os
import sys

from fastapi.testclient import TestClient
import pytest


def _fresh_import_gateway(monkeypatch):
    if os.getenv("CI", "").lower() == "true":
        import mcp  # noqa: F401
    else:
        pytest.importorskip("mcp")
    monkeypatch.setenv("API_KEY", "test_api_key")
    # Avoid relying on any real upstream API in unit tests.
    monkeypatch.setenv("INFERENCE_BASE_URL", "http://example.invalid")

    for name in list(sys.modules.keys()):
        if name == "mcp_gateway" or name.startswith("mcp_gateway."):
            sys.modules.pop(name, None)

    import mcp_gateway.main  # noqa: E402

    return mcp_gateway.main.app


def test_gateway_health(monkeypatch):
    app = _fresh_import_gateway(monkeypatch)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_gateway_requires_key(monkeypatch):
    app = _fresh_import_gateway(monkeypatch)
    client = TestClient(app)
    # /mcp is mounted but gateway middleware runs for all paths when API_KEY is set.
    r = client.get("/mcp/sse/")
    assert r.status_code == 401

    r2 = client.get("/mcp/sse/", headers={"X-API-Key": "test_api_key"})
    # SSE endpoint will likely be 405/200 depending on transport impl, but not unauthorized.
    assert r2.status_code != 401
