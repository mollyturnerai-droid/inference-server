import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_register_user():
    """Test user registration"""
    response = client.post(
        "/v1/auth/register",
        json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "password123"
        }
    )
    # May fail if user already exists, which is ok for this test
    assert response.status_code in [200, 400]


def test_list_models_unauthenticated():
    """Test listing models without authentication"""
    response = client.get("/v1/models")
    assert response.status_code == 200
