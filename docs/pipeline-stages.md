# The nine stages, and where each one lives

This is me mapping the pipeline I designed on the Padlet onto the code I actually wrote, plus the three stages I added afterwards. If you're marking this against my original design, this is the page to read next to the code.

The first six are the stages from my Padlet post. Stages 7 and 8 were asked for in the assessment brief. Stage 9 came out of Shane McDevitt's feedback on my Padlet.

## From my Padlet

### 1. Business problem understanding
My problem is detecting AI-generated text in student submissions, as the front end of the academic-integrity workflow in my dissertation. I lay the reasoning out in the README and the writeup. There's no code for this one as such, it's the framing for everything else I did.

### 2. Data acquisition and management
In `training/ingest.py` I pull HC3 from the HuggingFace Hub, flatten the question/answer rows into (text, label) examples, balance the two classes by down-sampling the bigger one, and write seed-pinned train/val/test splits to `data/processed/`. My workflow `01-data-ingestion.yaml` runs that on demand or on a weekly schedule, and uploads the splits as an artefact so the training run can pick them up.

### 3. ML training and testing
In `training/train.py` I fine-tune `distilbert-base-uncased` on those splits, log params and metrics (accuracy, F1, AUROC) to MLflow, evaluate on the held-out test split, and register the result as `ai-text-detector`. My workflow `02-train-register.yaml` runs this on the self-hosted runner. I followed the general shape of the week 7 ML model workflow tutorial here.

### 4. Model deployment
`app/app.py` is my Flask service, `app/Dockerfile` containerises it, and `k8s/deployment.yaml` plus `k8s/service.yaml` put it on the Kind cluster behind a NodePort. I pull the model from MLflow when the pod starts instead of baking it into the image. That's the deliberate change I made from the reference repo, and I go into it more in the README and the architecture doc.

### 5. Continuous integration
My `tests/` folder has three layers. Unit tests on the preprocessing (`test_preprocess.py`), API smoke tests against a running container (`test_api.py`), and a model-regression check (`test_model_regression.py`) that my retrain workflow leans on as a gate. `03-ci-tests.yaml` runs lint and the unit tests on every push.

### 6. Continuous monitoring
My Flask app exposes `/metrics` in Prometheus format with rolling prediction-confidence and input-length signals, plus `/health` for liveness with the model version on it. My `monitoring/` module reads those signals and works out whether things have drifted from the training-time baseline. On the Padlet I flagged drift as the main risk here, because the LLMs people actually use keep changing, so text that looked obviously AI to me last year won't look the same next year.

## What I added after the Padlet

### 7. Continuous delivery (asked for in the brief)
I have ArgoCD watching `k8s/` with automated sync, prune and self-heal turned on. `argoCD/application.yaml` is the Application resource. A manifest change on `main` rolls the cluster without me touching kubectl, and my `04-build-push.yaml` workflow is what produces that manifest change.

### 8. Continuous training (asked for in the brief)
`05-retrain-scheduled.yaml`. This is the one I put the most thought into, so I give it its own detailed section in the assessment document. Short version: I made it event-driven rather than a blind weekly cron, and I gate promotion to Production behind two checks instead of one. I write the triggers and the gates up in stage 8 of the writeup.

### 9. Explainability (from Shane's feedback)
Shane McDevitt commented on my Padlet that I'd left explainability out, and that it really matters when the output could affect someone's grade. So I added `explainability/attention_viz.py` and the `/explain` endpoint, which return the per-token attention weights from DistilBERT's last layer. That way a lecturer can see which words pushed my model's decision one way, which gives them something concrete to point at if they end up talking a flag through with a student.

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
