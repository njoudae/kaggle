from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import columns_from_config, labels_from_config, load_labeled_dataframe
from src.pipeline_logging import PipelineLogger
from src.trainer import train_experiment
from src.utils import ensure_dir, get_device, json_load, set_seed
from src.validation import require_config, resolve_existing_path, validate_labeled_dataset


EXPERIMENTS = {
    "cls4_equal_loss": {
        "architecture": "cls4",
        "fusion": "equal_logits",
        "loss_mode": "equal_mean",
        "freeze_encoder": False,
        "result_file": "cls4_equal_loss_results.csv",
    },
    "cls4_frozen": {
        "architecture": "cls4",
        "fusion": "equal_logits",
        "loss_mode": "equal_mean",
        "freeze_encoder": True,
        "result_file": "cls4_frozen_results.csv",
    },
    "weighted_logits": {
        "architecture": "cls4",
        "fusion": "weighted_logits",
        "loss_mode": "heads_plus_fused",
        "lambda_fused": 1.0,
        "freeze_encoder": False,
        "result_file": "weighted_logits_results.csv",
    },
    "learnable_loss": {
        "architecture": "cls4",
        "fusion": "equal_logits",
        "loss_mode": "learnable_loss",
        "beta_entropy": 0.0,
        "freeze_encoder": False,
        "result_file": "learnable_loss_results.csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a selected-layer CLS4 experiment.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--base-model-id", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--selected-layers", default=None, help="JSON file. Defaults to results/selected_layers.json.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def select_backbone(config: dict, results_dir: Path, override_id: str | None, override_name: str | None) -> tuple[str, str]:
    if override_id:
        return override_name or "selected_backbone", override_id
    comparison_path = results_dir / "base_model_comparison.csv"
    if comparison_path.exists():
        row = pd.read_csv(comparison_path).sort_values("dev_Favg2", ascending=False).iloc[0]
        return str(row["model_name"]), str(row["hf_id"])
    for model_cfg in config["models"]:
        if bool(model_cfg.get("enabled", True)):
            return str(model_cfg["name"]), str(model_cfg["hf_id"])
    raise RuntimeError("No backbone found. Run base model comparison first or pass --base-model-id.")


def main() -> None:
    args = parse_args()
    logger = PipelineLogger(total_steps=7, verbose=args.verbose)
    try:
        with logger.stage("Loading configuration", "Config"):
            require_config(args.config)
            config = load_config(args.config)
            set_seed(int(config["seed"]))
            columns = columns_from_config(config)
            label2id, id2label = labels_from_config(config)
            results_dir = ensure_dir(config["paths"]["results_dir"])
            model_name, model_id = select_backbone(config, results_dir, args.base_model_id, args.model_name)
            print(f"Selected backbone: {model_name} ({model_id})")
            selected_path = Path(args.selected_layers or (results_dir / "selected_layers.json"))
            selected_layers = [int(layer) for layer in json_load(selected_path)]
            if len(selected_layers) != 4:
                raise ValueError(f"Expected exactly four selected layers, got {selected_layers}")
            print("Selected layers:", selected_layers)

        with logger.stage("Reading datasets", "Dataset"):
            data_dir = Path(config["paths"]["data_dir"])
            train_path = resolve_existing_path(data_dir, config["paths"]["train_file"], config["paths"].get("fallback_train_file"), "train")
            dev_path = resolve_existing_path(data_dir, config["paths"]["dev_file"], config["paths"].get("fallback_dev_file"), "dev")
            train_df = load_labeled_dataframe(train_path, columns, label2id, max_samples=config["training"].get("max_train_samples"))
            dev_df = load_labeled_dataframe(dev_path, columns, label2id, max_samples=config["training"].get("max_dev_samples"))
            validate_labeled_dataset(train_df, columns, set(label2id), "train")
            validate_labeled_dataset(dev_df, columns, set(label2id), "dev")

        with logger.stage("Preparing experiment config", "Config"):
            exp = dict(EXPERIMENTS[args.experiment])
            result_file = exp.pop("result_file")
            exp["selected_layers"] = selected_layers
            output_dir = ensure_dir(Path(config["paths"]["output_dir"]) / args.experiment)

        with logger.stage("Training selected-layer experiment", "Training"):
            row = train_experiment(
                experiment_name=args.experiment,
                model_name=model_name,
                model_id=model_id,
                experiment_config=exp,
                train_df=train_df,
                dev_df=dev_df,
                columns=columns,
                label2id=label2id,
                id2label=id2label,
                training_config=config["training"],
                output_dir=output_dir,
                device=get_device(),
                verbose=args.verbose,
                dry_run=args.dry_run,
            )
            row["selected_layers"] = selected_layers

        with logger.stage("Saving result CSV", "Validation"):
            pd.DataFrame([row]).to_csv(results_dir / result_file, index=False)
            print(pd.DataFrame([row]).to_string(index=False))

        with logger.stage("Saving learned weights if present", "Validation"):
            history_path = Path(row["checkpoint_path"]).parent / "training_history.csv" if row.get("checkpoint_path") else None
            if history_path and history_path.exists():
                history = pd.read_csv(history_path)
                if "learned_fusion_weights" in history.columns:
                    history[["epoch", "learned_fusion_weights"]].to_csv(results_dir / "weighted_logits_alpha_history.csv", index=False)
                if "learned_loss_weights" in history.columns:
                    history[["epoch", "learned_loss_weights"]].to_csv(results_dir / "learnable_loss_lambda_history.csv", index=False)
    finally:
        logger.status.print_summary()


if __name__ == "__main__":
    main()
