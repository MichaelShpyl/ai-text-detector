"""
ML Training & Testing - Stage 3 of the Padlet pipeline.

Fine-tunes distilbert-base-uncased on the HC3 splits produced by ingest.py,
logs metrics/params/artifacts to MLflow, and registers the trained model in
the MLflow Model Registry under the name 'ai-text-detector'.

The Continuous Training workflow (05-retrain-scheduled.yaml) re-runs this on
schedule and only promotes the new version to 'Production' stage if it beats
the incumbent on the holdout set.
"""
import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.transformers
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert-base-uncased"
REGISTERED_MODEL_NAME = "ai-text-detector"
DATA_DIR = Path("data/processed")


def load_split(name: str) -> Dataset:
    path = DATA_DIR / f"{name}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return Dataset.from_list(rows)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "auroc": roc_auc_score(labels, probs[:, 1]),
    }


def main(epochs: int, batch_size: int, lr: float, mlflow_uri: str, max_train: int = 0, max_eval: int = 0):
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("ai-text-detector")

    # Cap CPU threads so training doesn't starve the self-hosted runner's
    # heartbeat (that shows up as "runner lost communication" in Actions).
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

    print("Loading data splits...")
    train_ds = load_split("train")
    val_ds = load_split("val")
    test_ds = load_split("test")
    # On a CPU-only runner the full HC3 takes hours, so allow capping the split
    # sizes. 0 means use everything (for example if a GPU is available).
    if max_train:
        train_ds = train_ds.select(range(min(max_train, len(train_ds))))
    if max_eval:
        val_ds = val_ds.select(range(min(max_eval, len(val_ds))))
        test_ds = test_ds.select(range(min(max_eval, len(test_ds))))
    print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    print(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_tok = train_ds.map(tokenize, batched=True)
    val_tok = val_ds.map(tokenize, batched=True)
    test_tok = test_ds.map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",  # we handle MLflow ourselves
    )

    with mlflow.start_run() as run:
        mlflow.log_params({
            "base_model": MODEL_NAME,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "test_size": len(test_ds),
        })

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
        )

        trainer.train()

        print("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_tok)
        mlflow.log_metrics({f"test_{k.replace('eval_', '')}": v for k, v in test_metrics.items()})
        print(f"  test metrics: {test_metrics}")

        print("Logging and registering model to MLflow...")
        components = {"model": trainer.model, "tokenizer": tokenizer}
        mlflow.transformers.log_model(
            transformers_model=components,
            artifact_path="model",
            task="text-classification",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        print(f"Run ID: {run.info.run_id}")

        # Capture a drift baseline from the validation set. monitoring/
        # drift_detector.py compares the live /metrics gauges against this
        # later to decide whether to trigger a retrain.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from monitoring.drift_detector import save_baseline
        from training.preprocess import clean_text

        val_logits = trainer.predict(val_tok).predictions
        val_probs = np.exp(val_logits - val_logits.max(axis=1, keepdims=True))
        val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
        confidences = val_probs.max(axis=1).tolist()
        input_lengths = [len(clean_text(t)) for t in val_ds["text"]]
        save_baseline(confidences, input_lengths, "data/baseline_stats.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5555"))
    parser.add_argument("--max-train", type=int, default=0, help="cap train examples (0 = all); keep small on a CPU runner")
    parser.add_argument("--max-eval", type=int, default=0, help="cap val/test examples (0 = all)")
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.lr, args.mlflow_uri, args.max_train, args.max_eval)
