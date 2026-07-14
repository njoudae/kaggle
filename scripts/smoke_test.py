from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import StanceDataset, columns_from_config, labels_from_config
from src.metrics import compute_metrics
from src.inference import validate_submission


class TinyTokenizer:
    def __call__(self, target, text, truncation=True, padding="max_length", max_length=128, return_tensors="pt"):
        del target, text, truncation, padding
        return {
            "input_ids": torch.ones((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight StanceEval2026 smoke checks.")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_config(parse_args().config)
    columns = columns_from_config(config)
    label2id, id2label = labels_from_config(config)
    df = pd.DataFrame(
        {
            columns.id_col: ["a", "b", "c"],
            columns.target_col: ["Target", "Target", "Other"],
            columns.text_col: ["sample one", "sample two", "sample three"],
            columns.label_col: ["Against", "Favor", "None"],
            "label": [label2id["Against"], label2id["Favor"], label2id["None"]],
        }
    )
    dataset = StanceDataset(df, TinyTokenizer(), columns, max_length=8, labeled=True)
    sample = dataset[0]
    assert sample["input_ids"].shape[0] == 8
    assert int(sample["labels"]) == 0

    metrics = compute_metrics([0, 1, 2], [0, 1, 1])
    assert set(["F1_Favor", "F1_Against", "F1_None", "Favg2", "Favg3", "Accuracy"]).issubset(metrics)

    submission = pd.DataFrame({columns.id_col: ["a", "b", "c"], columns.label_col: ["Against", "Favor", "None"]})
    validate_submission(df, submission, columns, set(id2label.values()))
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
