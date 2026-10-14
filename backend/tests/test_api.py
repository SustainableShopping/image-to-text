import pytest
from fastapi.testclient import TestClient

from image_to_text import SERVICE_VERSION
from image_to_text.api.app import create_app
from image_to_text.config import Settings

ALLOWED = "https://allowed.example"


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(Settings(cors_origins=(ALLOWED,))))


def test_health_reports_ok_and_version(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "version": SERVICE_VERSION}


def test_cors_allows_configured_origin(client):
    r = client.get("/health", headers={"Origin": ALLOWED})
    assert r.headers["access-control-allow-origin"] == ALLOWED


def test_cors_ignores_other_origins(client):
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers
