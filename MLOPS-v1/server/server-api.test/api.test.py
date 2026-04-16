"""
API endpoint tests for the FastAPI server.

Run with:
    pytest server/server-api.test/api.test.py -v
"""

import sys
import os

import pytest
from fastapi.testclient import TestClient

# Make `server/app.py` importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import app

client = TestClient(app)

VALID_PAYLOAD = {
    "est_diameter_min":   0.12,
    "est_diameter_max":   0.27,
    "relative_velocity":  48000.0,
    "absolute_magnitude": 22.1,
    "miss_distance":      14_500_000.0,
}

ARTIFACTS_AVAILABLE = os.path.exists(
    os.path.join(os.path.dirname(__file__), "..", "..", "model", "artifacts", "classifier.joblib")
)
requires_artifacts = pytest.mark.skipif(
    not ARTIFACTS_AVAILABLE,
    reason="model/artifacts/ not found — run the training notebooks first",
)

# ── Health endpoints ──────────────────────────────────────────────────────────

def test_root_returns_200():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert "message" in body


def test_items_endpoint():
    resp = client.get("/items/42")
    assert resp.status_code == 200
    body = resp.json()
    assert body["item_id"] == 42
    assert body["category"] == "ML-Model"


# ── Predict endpoint — input validation ───────────────────────────────────────

def test_predict_missing_field_returns_422():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "miss_distance"}
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 422


def test_predict_wrong_type_returns_422():
    payload = {**VALID_PAYLOAD, "relative_velocity": "fast"}
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 422


def test_predict_empty_body_returns_422():
    resp = client.post("/api/predict", json={})
    assert resp.status_code == 422


# ── Predict endpoint — inference (only if artifacts present) ──────────────────

@requires_artifacts
def test_predict_returns_expected_shape():
    resp = client.post("/api/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["hazardous"], bool)
    assert 0.0 <= body["hazardous_probability"] <= 1.0
    assert body["miss_distance_km"] > 0


@requires_artifacts
def test_predict_probability_consistent_with_label():
    """If hazardous is True, probability should be >= 0.5."""
    resp = client.post("/api/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    if body["hazardous"]:
        assert body["hazardous_probability"] >= 0.5
    else:
        assert body["hazardous_probability"] < 0.5


@requires_artifacts
def test_predict_deterministic():
    """Same input must yield identical output (no randomness at inference)."""
    r1 = client.post("/api/predict", json=VALID_PAYLOAD).json()
    r2 = client.post("/api/predict", json=VALID_PAYLOAD).json()
    assert r1 == r2


# ── Predict endpoint — service-unavailable path ───────────────────────────────

def test_predict_503_when_models_not_loaded(monkeypatch):
    """Simulate missing artifacts by clearing the in-memory models dict."""
    import app as server_app
    original = dict(server_app.models)
    server_app.models.clear()
    try:
        resp = client.post("/api/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 503
        assert "Models not loaded" in resp.json()["detail"]
    finally:
        server_app.models.update(original)
