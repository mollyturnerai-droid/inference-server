def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_models_requires_auth(client):
    # /v1/* endpoints are protected by API key middleware.
    r = client.get("/v1/models/")
    assert r.status_code == 401


def test_models_with_master_key(client, api_key):
    r = client.get("/v1/models/", headers={"X-API-Key": api_key})
    assert r.status_code == 200
    data = r.json()
    assert "models" in data


def test_system_status_with_master_key(client, api_key):
    r = client.get("/v1/system/status", headers={"X-API-Key": api_key})
    assert r.status_code == 200
    data = r.json()
    assert "services" in data
    assert "database" in data["services"]
