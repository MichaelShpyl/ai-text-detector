# Demo runbook

This is the order I walk through things in the live demo. It's about 5 to 7 minutes if I don't ramble. Everything should already be up before I start (VM running, MLflow up, cluster up, ArgoCD synced, runner online), so this is presentation, not setup.

Rough timings in brackets.

## 1. Problem and context (30s)
Open the README. Say what the thing does in one line: it takes text and decides human or AI, with a confidence. Then the why: it's stage one of my dissertation pipeline on checking academic integrity, and the whole repo is the implementation of the six-stage pipeline I designed on the Padlet, plus three more stages on top.

## 2. The nine stages (45s)
Open `docs/pipeline-stages.md`. Point at the table. Six stages from the Padlet, two from the brief (continuous delivery, continuous training), and explainability which came from Shane's feedback on my Padlet. This is the map between the design and the code.

## 3. MLflow (60s)
Open the MLflow UI on `:5555`. Show the `ai-text-detector` registered model, the versions, and that one is in the Production stage. Click into a run and show the logged metrics (accuracy, F1, AUROC) and the hyperparameters. Make the point that the model lives here, not in the image.

## 4. GitHub Actions (90s)
Open the Actions tab. Walk the five workflows quickly, one per stage: ingestion (01), train and register on the self-hosted runner (02), CI tests on every push (03), build and push the SHA-tagged image (04), and continuous training (05). Spend the time on 05: explain the four triggers and the two-gate promotion, because that's the part worth marks. Do not promote to Production unless it beats the current one.

## 5. ArgoCD, the live roll (60s)
Open the ArgoCD UI. Show the application synced and healthy. Then the money shot: bump `replicas` in `k8s/deployment.yaml` from 2 to 3, commit and push to main. Switch back to ArgoCD and watch it detect the change and roll the cluster. This is GitOps doing its thing with no kubectl from me.

## 6. A live prediction (60s)
Open the web form on `:30080`. Paste a clearly human sample, submit, show the verdict and confidence. Then paste an obviously AI sample and show it flips. Then hit `/explain` on one of them and talk through the attention weights, which is the explainability stage and the answer to "why did it decide that".

## 7. Monitoring (30s)
Hit `/metrics`. Point at the rolling confidence mean and input-length gauges. Say that the drift detector reads these, compares them to the baseline saved at training time, and fires the retrain trigger if they move too far. That closes the loop back to stage 8.

## 8. Q&A buffer (30s+)
Leave a bit of room. The questions I expect are in my own prep notes.

## If something breaks mid-demo
- App not responding on 30080: check the pods with `kubectl get pods`, and that the service is a NodePort on 30080.
- `/predict` errors but `/health` is fine: no model promoted to Production in MLflow.
- ArgoCD not syncing: check the Application is pointed at the right repo and branch, and that the manifest commit actually landed on main.
