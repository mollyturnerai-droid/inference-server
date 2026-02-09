import os
import sys
import importlib

import pytest
from fastapi.testclient import TestClient


def _fresh_import_app(monkeypatch, *, tmp_path):
    # In CI we expect full deps; locally, allow running a reduced env without hard failing.
    if os.getenv("CI", "").lower() == "true":
        import slowapi  # noqa: F401
    else:
        pytest.importorskip("slowapi")
    # Ensure settings are read from env (not a developer .env) and that recon doesn't do network in CI.
    monkeypatch.setenv("API_KEY", "test_api_key")
    monkeypatch.setenv("RECON_ENABLED", "false")
    monkeypatch.setenv("RECON_ON_STARTUP", "false")
    monkeypatch.setenv("ENABLE_GPU", "false")
    monkeypatch.setenv("ENABLE_MCP", "false")

    db_path = tmp_path / "test.db"
    storage_path = tmp_path / "storage"
    cache_path = tmp_path / "model_cache"
    catalog_path = tmp_path / "catalog.json"

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("STORAGE_PATH", storage_path.as_posix())
    monkeypatch.setenv("MODEL_CACHE_DIR", cache_path.as_posix())
    monkeypatch.setenv("CATALOG_PATH", catalog_path.as_posix())
    monkeypatch.setenv("API_BASE_URL", "http://testserver")

    # Drop cached imports so the Settings singleton + SQLAlchemy engine are rebuilt with our env vars.
    for name in list(sys.modules.keys()):
        if name == "app" or name.startswith("app."):
            sys.modules.pop(name, None)

    import app.main  # noqa: E402

    return app.main.app


@pytest.fixture()
def client(monkeypatch, tmp_path):
    app = _fresh_import_app(monkeypatch, tmp_path=tmp_path)
    return TestClient(app)


@pytest.fixture()
def client_mcp_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_MCP", "true")
    app = _fresh_import_app(monkeypatch, tmp_path=tmp_path)
    return TestClient(app)


@pytest.fixture()
def api_key():
    return "test_api_key"
