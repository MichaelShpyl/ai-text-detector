"""
Data acquisition & management - Stage 2 of the Padlet pipeline.

Pulls the HC3 (Human ChatGPT Comparison Corpus) dataset from the HuggingFace
Hub, deduplicates, balances classes, and writes deterministic train/val/test
splits to ./data/processed/.
"""
import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset

SEED = 42
OUTPUT_DIR = Path("data/processed")
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, val, test


def build_examples(hc3_split):
    """Convert HC3 rows (which have lists of human and chatgpt answers per question)
    into a flat list of (text, label) examples."""
    examples = []
    for row in hc3_split:
        for human_ans in row["human_answers"]:
            if human_ans and human_ans.strip():
                examples.append({"text": human_ans.strip(), "label": 0})
        for ai_ans in row["chatgpt_answers"]:
            if ai_ans and ai_ans.strip():
                examples.append({"text": ai_ans.strip(), "label": 1})
    return examples


def balance_and_split(examples):
    """Balance classes (down-sample majority) and split deterministically."""
    random.seed(SEED)
    by_label = {0: [], 1: []}
    for ex in examples:
        by_label[ex["label"]].append(ex)
    min_count = min(len(by_label[0]), len(by_label[1]))
    balanced = random.sample(by_label[0], min_count) + random.sample(by_label[1], min_count)
    random.shuffle(balanced)

    n = len(balanced)
    train_end = int(n * SPLIT_RATIOS[0])
    val_end = train_end + int(n * SPLIT_RATIOS[1])
    return {
        "train": balanced[:train_end],
        "val": balanced[train_end:val_end],
        "test": balanced[val_end:],
    }


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(subset: str):
    print(f"Loading HC3 subset '{subset}' from HuggingFace Hub...")
    ds = load_dataset("Hello-SimpleAI/HC3", subset)
    examples = build_examples(ds["train"])
    print(f"Built {len(examples)} raw examples")

    splits = balance_and_split(examples)
    for split_name, rows in splits.items():
        path = OUTPUT_DIR / f"{split_name}.jsonl"
        write_jsonl(path, rows)
        n_human = sum(1 for r in rows if r["label"] == 0)
        n_ai = sum(1 for r in rows if r["label"] == 1)
        print(f"  {split_name}: {len(rows)} examples ({n_human} human / {n_ai} AI) -> {path}")

    meta = {
        "source": f"Hello-SimpleAI/HC3 [{subset}]",
        "seed": SEED,
        "split_ratios": SPLIT_RATIOS,
        "counts": {k: len(v) for k, v in splits.items()},
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote metadata to {OUTPUT_DIR / 'metadata.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="all", help="HC3 subset (all, finance, medicine, open_qa, reddit_eli5, wiki_csai)")
    args = parser.parse_args()
    main(args.subset)
