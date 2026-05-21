# The nine stages, and where each one lives

This maps the pipeline design I posted on the Padlet onto the actual code, and it covers the three stages that got added after the Padlet. If you're marking this against the original design, this is the page to read next to the code.

The first six are the stages from my Padlet post. Stages 7 and 8 were asked for in the assessment brief. Stage 9 came out of Shane McDevitt's feedback on the Padlet.

## From the Padlet

### 1. Business problem understanding
The problem is detecting AI-generated text in student submissions, as the front end of the academic-integrity workflow in my dissertation. The reasoning is in the README and the writeup. There's no code for this one as such, it's the framing for everything else.

### 2. Data acquisition and management
`training/ingest.py` pulls HC3 from the HuggingFace Hub, flattens the question/answer rows into (text, label) examples, balances the two classes by down-sampling the bigger one, and writes seed-pinned train/val/test splits to `data/processed/`. Workflow `01-data-ingestion.yaml` runs it on demand or on a weekly schedule, and uploads the splits as an artefact so the training run can pick them up.

### 3. ML training and testing
`training/train.py` fine-tunes `distilbert-base-uncased` on those splits, logs params and metrics (accuracy, F1, AUROC) to MLflow, evaluates on the held-out test split, and registers the result as `ai-text-detector`. Workflow `02-train-register.yaml` runs this on the self-hosted runner. I'm following the general shape of the week 7 ML model workflow tutorial here.

### 4. Model deployment
`app/app.py` is the Flask service, `app/Dockerfile` containerises it, and `k8s/deployment.yaml` plus `k8s/service.yaml` put it on the Kind cluster behind a NodePort. The model is pulled from MLflow when the pod starts, not baked into the image. That's the deliberate change from the reference repo and there's more on it in the README and the architecture doc.

### 5. Continuous integration
`tests/` has three layers. Unit tests on the preprocessing (`test_preprocess.py`), API smoke tests against a running container (`test_api.py`), and a model-regression check (`test_model_regression.py`) that the retrain workflow leans on as a gate. `03-ci-tests.yaml` runs lint and the unit tests on every push.

### 6. Continuous monitoring
The Flask app exposes `/metrics` in Prometheus format with rolling prediction-confidence and input-length signals, plus `/health` for liveness with the model version on it. The `monitoring/` module reads those signals and works out whether things have drifted from the training-time baseline. On the Padlet I flagged drift as the main risk here, because the LLMs people actually use keep changing, so text that looked obviously AI last year won't look the same next year.

## Added after the Padlet

### 7. Continuous delivery (asked for in the brief)
ArgoCD watches `k8s/` with automated sync, prune and self-heal turned on. `argoCD/application.yaml` is the Application resource. A manifest change on `main` rolls the cluster without me touching kubectl, and `04-build-push.yaml` is what produces that manifest change.

### 8. Continuous training (asked for in the brief)
`05-retrain-scheduled.yaml`. This is the one I put the most thought into, so it gets its own detailed section in the assessment document. Short version: it's event-driven rather than a blind weekly cron, and promotion to Production is gated by two checks instead of one. The triggers and the gates are written up in stage 8 of the assessment doc.

### 9. Explainability (from Shane's feedback)
Shane McDevitt commented on my Padlet that explainability was missing, and that it really matters when the output could affect someone's grade. `explainability/attention_viz.py` and the `/explain` endpoint return the per-token attention weights from DistilBERT's last layer, so a lecturer can see which words pushed the decision one way. That gives them something concrete to point at if they end up discussing a flag with a student in an oral.

## Quick map

| # | stage | from | code | workflow |
|---|---|---|---|---|
| 1 | business problem | Padlet | README, docs | - |
| 2 | data acquisition | Padlet | training/ingest.py | 01 |
| 3 | training and testing | Padlet | training/train.py | 02 |
| 4 | deployment | Padlet | app/, k8s/ | 04 |
| 5 | continuous integration | Padlet | tests/ | 03 |
| 6 | continuous monitoring | Padlet | monitoring/, /metrics | runtime |
| 7 | continuous delivery | brief | argoCD/ | 04 |
| 8 | continuous training | brief | 05 workflow | 05 |
| 9 | explainability | Shane's feedback | explainability/ | runtime |
