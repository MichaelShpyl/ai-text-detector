"""
API smoke tests. These run against a live container in CI - the workflow
builds the image, runs it, waits for /health to return 200, then runs these.

To run locally:  pytest tests/test_api.py --api-url http://localhost:5000
"""
import os

import pytest
import requests

API_URL = os.environ.get("API_URL", "http://localhost:5000")
HUMAN_SAMPLE = (
    "I think the best part of going to the beach is actually the drive home "
    "afterwards, when you're sunburnt and exhausted and someone in the car is "
    "always trying to find one good radio station that isn't ads."
)
AI_SAMPLE = (
    "There are several compelling reasons why one should consider adopting a "
    "Mediterranean diet. Firstly, it has been associated with numerous health "
    "benefits, including reduced risk of cardiovascular disease and improved "
    "cognitive function. Secondly, it emphasises whole foods and limits processed items."
)


@pytest.mark.smoke
def test_health_endpoint():
    r = requests.get(f"{API_URL}/health", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_version" in body


@pytest.mark.smoke
def test_predict_returns_valid_schema():
    r = requests.post(f"{API_URL}/predict", json={"text": HUMAN_SAMPLE}, timeout=30)
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("AI", "human")
    assert 0.5 <= body["confidence"] <= 1.0
    assert "model_version" in body


@pytest.mark.smoke
def test_predict_rejects_short_input():
    r = requests.post(f"{API_URL}/predict", json={"text": "too short"}, timeout=10)
    assert r.status_code == 400


@pytest.mark.smoke
def test_metrics_endpoint_prometheus_format():
    r = requests.get(f"{API_URL}/metrics", timeout=10)
    assert r.status_code == 200
    assert "ai_text_detector_predictions_total" in r.text
    assert "ai_text_detector_confidence_mean" in r.text
