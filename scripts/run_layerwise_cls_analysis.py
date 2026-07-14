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
from src.utils import ensure_dir, get_device, json_dump, set_seed
from src.validation import require_config, resolve_existing_path, validate_labeled_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run layer-wise CLS analysis for one backbone.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--base-model-id", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=12)
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

        with logger.stage("Reading datasets", "Dataset"):
            data_dir = Path(config["paths"]["data_dir"])
            train_path = resolve_existing_path(data_dir, config["paths"]["train_file"], config["paths"].get("fallback_train_file"), "train")
            dev_path = resolve_existing_path(data_dir, config["paths"]["dev_file"], config["paths"].get("fallback_dev_file"), "dev")
            train_df = load_labeled_dataframe(train_path, columns, label2id, max_samples=config["training"].get("max_train_samples"))
            dev_df = load_labeled_dataframe(dev_path, columns, label2id, max_samples=config["training"].get("max_dev_samples"))
            validate_labeled_dataset(train_df, columns, set(label2id), "train")
            validate_labeled_dataset(dev_df, columns, set(label2id), "dev")

        with logger.stage("Preparing output folders", "Config"):
            output_dir = ensure_dir(Path(config["paths"]["output_dir"]) / "layerwise_cls")

        rows: list[dict] = []
        device = get_device()
        with logger.stage("Training one CLS head per layer", "Training"):
            for layer in range(1, args.num_layers + 1):
                experiment_name = f"layer_{layer:02d}_cls"
                row = train_experiment(
                    experiment_name=experiment_name,
                    model_name=model_name,
                    model_id=model_id,
                    experiment_config={
                        "architecture": "layerwise_cls",
                        "layer_index": layer,
                        "freeze_encoder": False,
                    },
                    train_df=train_df,
                    dev_df=dev_df,
                    columns=columns,
                    label2id=label2id,
                    id2label=id2label,
                    training_config=config["training"],
                    output_dir=output_dir,
                    device=device,
                    verbose=args.verbose,
                    dry_run=args.dry_run,
                )
                row["layer"] = layer
                rows.append(row)

        with logger.stage("Saving layerwise results", "Validation"):
            results = pd.DataFrame(rows)
            if "dev_Favg2" not in results.columns:
                raise RuntimeError("Layerwise results missing dev_Favg2.")
            results = results.sort_values("dev_Favg2", ascending=False).reset_index(drop=True)
            results.to_csv(results_dir / "layerwise_results.csv", index=False)
            ranking = results[["layer", "dev_Favg2"]].copy()
            ranking.insert(0, "Rank", range(1, len(ranking) + 1))
            ranking = ranking.rename(columns={"layer": "Layer", "dev_Favg2": "Dev Favg2"})
            ranking.to_csv(results_dir / "layer_ranking.csv", index=False)
            selected_layers = [int(layer) for layer in results.head(args.top_k)["layer"].tolist()]
            json_dump(selected_layers, results_dir / "selected_layers.json")
            print("Selected layers:", selected_layers)

        with logger.stage("Plotting layer performance", "Validation"):
            import matplotlib.pyplot as plt

            ordered = results.sort_values("layer")
            plt.figure(figsize=(8, 4.5))
            plt.plot(ordered["layer"], ordered["dev_Favg2"], marker="o")
            plt.xlabel("Layer")
            plt.ylabel("Dev Favg2")
            plt.title("Layer-wise CLS Performance")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(results_dir / "layer_performance.png", dpi=180)
            plt.close()

        with logger.stage("Printing final summary", "Validation"):
            print(results[["layer", "best_epoch", "dev_Favg2", "dev_accuracy", "dev_F_favor", "dev_F_against", "dev_F_none"]].to_string(index=False))
    finally:
        logger.status.print_summary()


if __name__ == "__main__":
    main()
