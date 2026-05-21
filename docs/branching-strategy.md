# Branching strategy

I'm using GitHub Flow with a `develop` branch sitting in the middle. It's a single-developer project, so I didn't want anything heavier like GitFlow with its release and hotfix ceremony. That would just be overhead here with nothing to show for it.

## Branches

- `main` is what gets deployed. ArgoCD watches it, and every commit on it should be deployable.
- `develop` is where changes get integrated and tested together before they go up to `main`.
- `feature/<short-name>` for individual pieces of work, branched off `develop`. These are short-lived.
- `hotfix/<short-name>` off `main` if something is broken in production and can't wait for the normal flow. It merges back into both `main` and `develop`.

For an assessment-sized project most of the work really happens on `feature/*` and `develop`, and `main` moves in batches rather than on every commit.

## What happens when I push

CI (`03-ci-tests.yaml`) runs on every push and pull request, across all of those branches. So I get lint and unit-test feedback straight away no matter where I'm working.

The build-and-deploy workflow (`04-build-push.yaml`) only runs on `main`, and only when something under `app/`, `training/preprocess.py` or `explainability/` actually changed. There's no point rebuilding and pushing an image for a docs-only commit.

A change's normal life looks like this:

1. branch `feature/x` off `develop`
2. push, CI runs
3. open a PR into `develop`, CI has to be green to merge
4. when a batch on `develop` looks solid, PR `develop` into `main`
5. the merge to `main` triggers build-push, which updates the k8s manifest, and ArgoCD rolls the cluster

## Per-branch behaviour

| branch | CI tests | build + push | retrain (05) |
|---|---|---|---|
| `feature/*` | yes | no | no |
| `develop` | yes | no | no |
| `main` | yes | yes, on app changes | yes, on its own triggers |
| `hotfix/*` | yes | yes, on merge | no |

## Why this and not something heavier

The detector is one deployable thing. There aren't other teams hanging off a release cadence. The only future consumer is the stage-two question generator from my dissertation, and that isn't built yet. So long-lived release branches would be admin with no payoff.

There's also a nicer reason it works. The model is versioned separately, in MLflow, so a code release and a model release are not the same event. I don't need any branch machinery to track model versions, the registry does that. That decoupling of code and model lifecycles is one of the MLOps practices Kreuzberger et al. (2023) describe, and it's the thing that lets the branching stay this simple.
