# GCP setup checklist

These are the commands I run to stand the whole thing up on a fresh GCP VM. I pulled them from the module tutorials (install Docker on GCP, week 5 Kind, week 8 self-hosted runners, week 9 ArgoCD), so they should match what we did in class. This is the quick command reference. I work through them in order.

The VM image is a RHEL family one (Rocky/CentOS), which is why the Docker install uses `dnf`/`yum`.

## 1. The VM

Compute Engine, create an instance:

- machine type: `n1-standard-4` (smaller than this and DistilBERT training crawls)
- boot disk: Rocky Linux 9, 30 GB or more
- region: whatever is closest, I use `europe-west1`
- firewall: allow HTTP, and add a rule opening these TCP ports: `22, 30080, 5555, 8080`

Ports: 22 is SSH, 30080 is the app NodePort, 5555 is the MLflow UI, 8080 is for port-forwarding the ArgoCD UI.

## 2. Docker and docker-compose (install-docker-on-gcp-vm tutorial)

SSH in, then:

```bash
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo systemctl enable docker
sudo systemctl start docker

# let my user run docker without sudo
sudo groupadd -f docker
sudo usermod -aG docker $USER
newgrp docker

docker run hello-world          # sanity check
docker compose version
```

## 3. kubectl and kind (week 9 ArgoCD prereqs, week 5 Kind)

```bash
[ $(uname -m) = x86_64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.31.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

sudo dnf install -y kubectl
kubectl version --client
kind version
```

## 4. Self-hosted GitHub runner (week 8 self-hosted runners)

On GitHub: repo Settings, Actions, Runners, New self-hosted runner, Linux. It gives you a download + config block. Run it on the VM, accept the defaults, then start it:

```bash
# (the download/config block GitHub gives you goes here first)
./run.sh
```

The runner needs to stay up for the training and retrain workflows, which target `runs-on: self-hosted`. If I want it to survive a reboot I install it as a service with `sudo ./svc.sh install && sudo ./svc.sh start`.

## 5. MLflow

```bash
git clone https://github.com/MichaelShpyl/ai-text-detector.git
cd ai-text-detector
docker compose up -d mlflow
curl -s localhost:5555 | head        # should respond
```

MLflow UI is then on `http://<vm-external-ip>:5555`.

## 6. Kind cluster (week 5 tutorial 2)

```bash
kind create cluster --name ai-detector --config k8s/kind-config.yaml
kubectl get nodes -A                 # 1 control-plane + 2 workers
```

## 7. ArgoCD (week 9 ArgoCD)

```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml --server-side
kubectl wait --for=condition=available --timeout=600s -n argocd deployment/argocd-server

# admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d

# point ArgoCD at this repo's k8s/ folder
kubectl apply -n argocd -f argoCD/application.yaml
kubectl get applications -n argocd
```

To see the ArgoCD UI: `kubectl port-forward svc/argocd-server -n argocd 8080:443`, then open `http://localhost:8080` (user `admin`, password from above).

## 8. Smoke test the running app

```bash
curl http://<vm-external-ip>:30080/health
curl -X POST http://<vm-external-ip>:30080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "Paste a longer passage here so it clears the 20 character minimum."}'
```

`/health` should return 200 with a `model_version`. `/predict` should return a label and a confidence. If `/health` is up but `/predict` errors about the model, that usually means I haven't promoted a model to Production in MLflow yet.
