from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data import columns_from_config, labels_from_config, load_labeled_dataframe
from src.trainer import train_experiment
from src.utils import copy_tree, ensure_dir, get_device, json_dump, remove_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run StanceEval2026 CLS4 experiments.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--base-model-id", default=None, help="Override the best base Hugging Face model id.")
    return parser.parse_args()


def choose_base_model(config: dict, override: str | None) -> tuple[str, str]:
    if override:
        for model_cfg in config["models"]:
            if str(model_cfg.get("hf_id")) == override:
                return str(model_cfg["name"]), override
        return "selected_model", override
    comparison_path = Path(config["paths"]["results_dir"]) / "base_model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        if not comparison.empty:
            row = comparison.sort_values("dev_Favg2", ascending=False).iloc[0]
            return str(row["model_name"]), str(row["hf_id"])
    for model_cfg in config["models"]:
        if bool(model_cfg.get("enabled", True)):
            return str(model_cfg["name"]), str(model_cfg["hf_id"])
    raise RuntimeError("No base model is available.")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    columns = columns_from_config(config)
    label2id, id2label = labels_from_config(config)
    data_dir = Path(config["paths"]["data_dir"])
    output_dir = ensure_dir(Path(config["paths"]["output_dir"]) / "experiments")
    results_dir = ensure_dir(config["paths"]["results_dir"])
    base_model_name, base_model_id = choose_base_model(config, args.base_model_id)

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
    for experiment_name, experiment_config in config["experiments"].items():
        if not bool(experiment_config.get("enabled", False)):
            continue
        row = train_experiment(
            experiment_name=experiment_name,
            model_name=base_model_name,
            model_id=base_model_id,
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
        raise RuntimeError("No enabled CLS experiments were configured.")

    comparison = pd.DataFrame(rows).sort_values("dev_Favg2", ascending=False).reset_index(drop=True)
    winner = comparison.iloc[0]
    best_model_dir = Path(config["paths"]["best_model_dir"])
    copy_tree(winner["checkpoint_path"], best_model_dir)
    json_dump({"best_experiment": winner.to_dict(), "base_model_id": base_model_id}, best_model_dir / "selection.json")

    for _, row in comparison.iloc[1:].iterrows():
        checkpoint = Path(str(row["checkpoint_path"]))
        remove_dir(checkpoint)
        comparison.loc[comparison["experiment_name"] == row["experiment_name"], "checkpoint_path"] = ""

    comparison.loc[0, "checkpoint_path"] = str(best_model_dir)
    comparison.to_csv(results_dir / "experiment_comparison.csv", index=False)
    print(comparison.to_string(index=False))
    print(f"Best experiment: {winner['experiment_name']}")
    print(f"Best checkpoint: {best_model_dir}")


if __name__ == "__main__":
    main()
