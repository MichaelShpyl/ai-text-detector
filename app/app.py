"""
Model Deployment - Stage 4 of the Padlet pipeline.

Flask serving layer for the AI-text detector. Pulls the latest registered
model from the MLflow Model Registry at startup (Production stage if
available, otherwise the latest version). Exposes:

  GET  /              - simple web form for manual testing
  POST /predict       - JSON API: {text} -> {label, confidence, model_version}
  POST /explain       - JSON API: attention-weight visualisation (Stage 9)
  GET  /metrics       - Prometheus-format metrics (Stage 6: monitoring)
  GET  /health        - liveness probe with model version
"""
import os
import time
from collections import deque

import mlflow
import numpy as np
from flask import Flask, jsonify, render_template, request

from training.preprocess import chunk_text, clean_text, is_valid_length

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5555")
REGISTERED_MODEL_NAME = os.environ.get("MODEL_NAME", "ai-text-detector")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "None")  # "Production" once promoted

app = Flask(__name__)

# Rolling buffers for the /metrics endpoint (Padlet Stage 6 - monitoring)
CONFIDENCE_BUFFER = deque(maxlen=1000)
INPUT_LENGTH_BUFFER = deque(maxlen=1000)
PREDICTION_COUNTS = {"human": 0, "ai": 0}
SERVICE_START_TS = time.time()


def load_model():
    """Load the latest registered model from MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if MODEL_STAGE and MODEL_STAGE != "None":
        uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
    else:
        uri = f"models:/{REGISTERED_MODEL_NAME}/latest"
    print(f"Loading model from {uri}")
    pipe = mlflow.transformers.load_model(uri)
    # Resolve the version that was actually loaded
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    version = max(int(v.version) for v in versions) if versions else "unknown"
    return pipe, str(version)


PIPELINE, MODEL_VERSION = load_model()


def predict_text(text: str):
    """Run prediction on a (possibly long) input text by chunking and averaging."""
    chunks = chunk_text(text)
    results = PIPELINE(chunks, truncation=True, max_length=512, return_all_scores=True)
    # Aggregate: mean probability across chunks
    ai_probs = []
    for chunk_scores in results:
        ai_prob = next(s["score"] for s in chunk_scores if s["label"] in ("LABEL_1", "AI"))
        ai_probs.append(ai_prob)
    mean_ai = float(np.mean(ai_probs))
    label = "AI" if mean_ai >= 0.5 else "human"
    confidence = mean_ai if label == "AI" else 1 - mean_ai
    return label, confidence


@app.route("/")
def home():
    return render_template("index.html", model_version=MODEL_VERSION)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or request.form
    text = payload.get("text", "")
    cleaned = clean_text(text)

    if not is_valid_length(cleaned):
        return jsonify({"error": "input text too short (min 20 chars)"}), 400

    label, confidence = predict_text(cleaned)

    # Update monitoring buffers
    CONFIDENCE_BUFFER.append(confidence)
    INPUT_LENGTH_BUFFER.append(len(cleaned))
    PREDICTION_COUNTS[label.lower()] = PREDICTION_COUNTS.get(label.lower(), 0) + 1

    response = {
        "label": label,
        "confidence": round(confidence, 4),
        "model_version": MODEL_VERSION,
    }
    if request.is_json:
        return jsonify(response)
    return render_template("result.html", **response)


@app.route("/explain", methods=["POST"])
def explain():
    """Return per-token attention weights from the final layer
    (Stage 9 - Explainability, added in response to peer feedback)."""
    from explainability.attention_viz import compute_attention

    text = (request.get_json(silent=True) or {}).get("text", "")
    if not is_valid_length(clean_text(text)):
        return jsonify({"error": "input text too short"}), 400
    tokens, weights = compute_attention(PIPELINE, clean_text(text))
    return jsonify({"tokens": tokens, "weights": weights, "model_version": MODEL_VERSION})


@app.route("/metrics")
def metrics():
    """Prometheus-style text metrics (Padlet Stage 6 - continuous monitoring)."""
    conf = list(CONFIDENCE_BUFFER) or [0]
    lens = list(INPUT_LENGTH_BUFFER) or [0]
    lines = [
        f"# HELP ai_text_detector_predictions_total Total predictions served",
        f"# TYPE ai_text_detector_predictions_total counter",
        f'ai_text_detector_predictions_total{{label="human"}} {PREDICTION_COUNTS.get("human", 0)}',
        f'ai_text_detector_predictions_total{{label="ai"}} {PREDICTION_COUNTS.get("ai", 0)}',
        f"# HELP ai_text_detector_confidence_mean Rolling mean prediction confidence",
        f"# TYPE ai_text_detector_confidence_mean gauge",
        f"ai_text_detector_confidence_mean {float(np.mean(conf)):.4f}",
        f"# HELP ai_text_detector_input_length_mean Rolling mean input length (chars)",
        f"# TYPE ai_text_detector_input_length_mean gauge",
        f"ai_text_detector_input_length_mean {float(np.mean(lens)):.2f}",
        f"# HELP ai_text_detector_uptime_seconds Seconds since service start",
        f"# TYPE ai_text_detector_uptime_seconds gauge",
        f"ai_text_detector_uptime_seconds {time.time() - SERVICE_START_TS:.0f}",
    ]
    return "\n".join(lines) + "\n", 200, {"Content-Type": "text/plain; version=0.0.4"}


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_version": MODEL_VERSION,
        "model_name": REGISTERED_MODEL_NAME,
        "uptime_seconds": int(time.time() - SERVICE_START_TS),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
