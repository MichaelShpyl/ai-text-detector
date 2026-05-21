"""
Continuous monitoring, the back half of Padlet stage 6.

I capture a baseline of prediction confidence and input length at training
time (save_baseline, called from train.py), and then a small cron job on the
VM calls run_monitor() every so often. It reads the live /metrics gauges off
the running app, compares them to the baseline, and if either has moved far
enough it fires a repository_dispatch event that kicks off the retrain
workflow (05-retrain-scheduled.yaml).

The thresholds are deliberately blunt. I want this to catch a real shift in
the kind of text people are sending, not to be a clever statistical test.
"""
import json
import os
import statistics
from pathlib import Path

CONFIDENCE_GAUGE = "ai_text_detector_confidence_mean"
LENGTH_GAUGE = "ai_text_detector_input_length_mean"


def save_baseline(confidence_scores, input_lengths, output_path):
    """Write the mean and std of confidence and input length to a JSON file.

    I call this at the end of train.py once the model is registered, using the
    validation-set predictions. The /metrics endpoint reports the same two
    quantities at serving time, so check_drift can compare like with like.
    """
    def _stats(values):
        values = list(values)
        if not values:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        mean = statistics.fmean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        return {"mean": mean, "std": std, "n": len(values)}

    baseline = {
        "confidence": _stats(confidence_scores),
        "input_length": _stats(input_lengths),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"Wrote drift baseline to {out}: {baseline}")
    return baseline


def _parse_gauge(metrics_text, name):
    """Pull a single unlabelled gauge value out of Prometheus text."""
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 2 and parts[0] == name:
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


def check_drift(current_metrics_url, baseline_path, confidence_threshold=0.15, length_threshold=50):
    """Fetch /metrics and compare the live gauges to the saved baseline.

    Returns True if either the rolling confidence mean or the rolling input
    length mean has moved further from the baseline than its threshold.
    """
    import requests

    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    resp = requests.get(current_metrics_url, timeout=10)
    resp.raise_for_status()
    text = resp.text

    cur_conf = _parse_gauge(text, CONFIDENCE_GAUGE)
    cur_len = _parse_gauge(text, LENGTH_GAUGE)
    if cur_conf is None or cur_len is None:
        print("Could not read both gauges from /metrics, skipping drift check")
        return False

    conf_drift = abs(cur_conf - baseline["confidence"]["mean"])
    len_drift = abs(cur_len - baseline["input_length"]["mean"])
    drifted = conf_drift > confidence_threshold or len_drift > length_threshold

    print(
        f"confidence: live={cur_conf:.3f} baseline={baseline['confidence']['mean']:.3f} "
        f"delta={conf_drift:.3f} (limit {confidence_threshold})"
    )
    print(
        f"input length: live={cur_len:.1f} baseline={baseline['input_length']['mean']:.1f} "
        f"delta={len_drift:.1f} (limit {length_threshold})"
    )
    print("DRIFT" if drifted else "no drift")
    return drifted


def fire_drift_dispatch(github_token, repo):
    """Fire a repository_dispatch of type drift-detected at the given repo.

    This is what the 05 retrain workflow listens for. The token comes in from
    the GITHUB_TOKEN env var via run_monitor, it is never written down here.
    """
    import requests

    url = f"https://api.github.com/repos/{repo}/dispatches"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github+json",
    }
    resp = requests.post(url, headers=headers, json={"event_type": "drift-detected"}, timeout=10)
    resp.raise_for_status()
    print(f"Fired drift-detected dispatch at {repo} (status {resp.status_code})")
    return resp.status_code


def run_monitor():
    """Entry point for the VM cron job.

    Reads its config from the environment so I can drop it into a crontab
    without editing the file:
      METRICS_URL    where the live /metrics is (default the NodePort on localhost)
      BASELINE_PATH  the baseline json written by train.py
      GITHUB_REPO    owner/name to dispatch at
      GITHUB_TOKEN   a PAT with repo scope
    """
    metrics_url = os.environ.get("METRICS_URL", "http://localhost:30080/metrics")
    baseline_path = os.environ.get("BASELINE_PATH", "data/baseline_stats.json")
    repo = os.environ.get("GITHUB_REPO", "MichaelShpyl/ai-text-detector")
    token = os.environ.get("GITHUB_TOKEN")

    if check_drift(metrics_url, baseline_path):
        if not token:
            print("Drift detected but GITHUB_TOKEN is not set, not firing dispatch")
            return False
        fire_drift_dispatch(token, repo)
        return True
    return False


if __name__ == "__main__":
    run_monitor()
