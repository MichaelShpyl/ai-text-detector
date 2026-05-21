"""
Model regression test. Runs against a fixed hold-out set of curated
human and AI samples. The Continuous Training workflow uses this as a
gate - if accuracy on this set drops below ACCURACY_FLOOR, the workflow
fails and the new model is NOT promoted to Production.
"""
import json
import os
from pathlib import Path

import pytest
import requests

API_URL = os.environ.get("API_URL", "http://localhost:5000")
ACCURACY_FLOOR = float(os.environ.get("ACCURACY_FLOOR", "0.85"))

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "regression_set.jsonl"


def load_fixtures():
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Fixture file not found: {FIXTURE_PATH}")
    return [json.loads(line) for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()]


@pytest.mark.regression
def test_model_accuracy_above_floor():
    fixtures = load_fixtures()
    assert len(fixtures) >= 10, "Need at least 10 fixtures for a meaningful regression check"

    correct = 0
    for ex in fixtures:
        r = requests.post(f"{API_URL}/predict", json={"text": ex["text"]}, timeout=30)
        assert r.status_code == 200
        predicted = r.json()["label"].lower()
        expected = "ai" if ex["label"] == 1 else "human"
        if predicted == expected:
            correct += 1

    accuracy = correct / len(fixtures)
    print(f"Regression accuracy: {accuracy:.3f} ({correct}/{len(fixtures)})")
    assert accuracy >= ACCURACY_FLOOR, (
        f"Model accuracy {accuracy:.3f} below floor {ACCURACY_FLOOR}. "
        "Block promotion of this model version."
    )
