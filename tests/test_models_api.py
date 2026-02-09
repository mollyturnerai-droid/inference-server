"""Tests for the models API endpoints."""


def test_create_and_get_model(client, api_key):
    """Create a model via the API, then retrieve it."""
    payload = {
        "name": "test-model",
        "description": "A test model",
        "model_type": "text-generation",
        "version": "1.0.0",
        "model_path": "test/model-path",
        "hardware": "cpu",
    }
    r = client.post("/v1/models/", json=payload, headers={"X-API-Key": api_key})
    assert r.status_code == 200
    data = r.json()
    model_id = data["id"]
    assert data["name"] == "test-model"
    assert data["model_type"] == "text-generation"

    # Retrieve it
    r2 = client.get(f"/v1/models/{model_id}", headers={"X-API-Key": api_key})
    assert r2.status_code == 200
    assert r2.json()["id"] == model_id


def test_list_models(client, api_key):
    """List models endpoint returns a list."""
    r = client.get("/v1/models/", headers={"X-API-Key": api_key})
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert isinstance(data["models"], list)


def test_delete_model(client, api_key):
    """Create and then delete a model."""
    payload = {
        "name": "to-delete",
        "model_type": "text-generation",
        "model_path": "test/delete-me",
    }
    r = client.post("/v1/models/", json=payload, headers={"X-API-Key": api_key})
    assert r.status_code == 200
    model_id = r.json()["id"]

    # Delete it
    r2 = client.delete(f"/v1/models/{model_id}", headers={"X-API-Key": api_key})
    assert r2.status_code == 200

    # Confirm gone
    r3 = client.get(f"/v1/models/{model_id}", headers={"X-API-Key": api_key})
    assert r3.status_code == 404


def test_unmount_model(client, api_key):
    """Unmount endpoint should remove a model like delete."""
    payload = {
        "name": "to-unmount",
        "model_type": "text-generation",
        "model_path": "test/unmount-me",
    }
    r = client.post("/v1/models/", json=payload, headers={"X-API-Key": api_key})
    assert r.status_code == 200
    model_id = r.json()["id"]

    # Unmount it
    r2 = client.post(f"/v1/models/{model_id}/unmount", headers={"X-API-Key": api_key})
    assert r2.status_code == 200

    # Confirm gone
    r3 = client.get(f"/v1/models/{model_id}", headers={"X-API-Key": api_key})
    assert r3.status_code == 404


def test_get_nonexistent_model(client, api_key):
    r = client.get("/v1/models/nonexistent-id", headers={"X-API-Key": api_key})
    assert r.status_code == 404


def test_delete_nonexistent_model(client, api_key):
    r = client.delete("/v1/models/nonexistent-id", headers={"X-API-Key": api_key})
    assert r.status_code == 404
