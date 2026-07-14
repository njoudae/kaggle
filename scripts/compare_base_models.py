from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import columns_from_config, labels_from_config, load_labeled_dataframe
from src.trainer import train_experiment
from src.utils import copy_tree, ensure_dir, get_device, remove_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare StanceEval2026 base pretrained models.")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    columns = columns_from_config(config)
    label2id, id2label = labels_from_config(config)
    data_dir = Path(config["paths"]["data_dir"])
    output_dir = ensure_dir(Path(config["paths"]["output_dir"]) / "base_models")
    results_dir = ensure_dir(config["paths"]["results_dir"])

    train_df = load_labeled_dataframe(
        data_dir / config["paths"]["train_file"],
        columns,
        label2id,
        max_samples=config["training"].get("max_train_samples"),
    )
    dev_df = load_labeled_dataframe(
        data_dir / config["paths"]["dev_file"],
        columns,
        label2id,
        max_samples=config["training"].get("max_dev_samples"),
    )

    rows: list[dict] = []
    device = get_device()
    for model_cfg in config["models"]:
        if not bool(model_cfg.get("enabled", True)):
            continue
        experiment_config = {"architecture": "sequence_classification", "freeze_encoder": False}
        row = train_experiment(
            experiment_name=model_cfg["name"],
            model_name=model_cfg["name"],
            model_id=model_cfg["hf_id"],
            experiment_config=experiment_config,
            train_df=train_df,
            dev_df=dev_df,
            columns=columns,
            label2id=label2id,
            id2label=id2label,
            training_config=config["training"],
            output_dir=output_dir,
            device=device,
        )
        rows.append(row)

    if not rows:
        raise RuntimeError("No enabled base models were configured.")

    comparison = pd.DataFrame(rows).sort_values("dev_Favg2", ascending=False).reset_index(drop=True)
    winner = comparison.iloc[0]
    best_base_dir = Path(config["paths"]["output_dir"]) / "best_base_model"
    copy_tree(winner["checkpoint_path"], best_base_dir)

    for _, row in comparison.iloc[1:].iterrows():
        checkpoint = Path(str(row["checkpoint_path"]))
        remove_dir(checkpoint)
        comparison.loc[comparison["experiment_name"] == row["experiment_name"], "checkpoint_path"] = ""

    comparison.loc[0, "checkpoint_path"] = str(best_base_dir)
    comparison.to_csv(results_dir / "base_model_comparison.csv", index=False)
    print(comparison.to_string(index=False))
    print(f"Best base model: {winner['model_name']} ({winner['hf_id']})")
    print(f"Best base checkpoint: {best_base_dir}")


if __name__ == "__main__":
    main()
