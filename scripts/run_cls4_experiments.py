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
from src.utils import copy_tree, ensure_dir, get_device, json_dump, remove_dir, set_seed
from src.validation import ensure_output_dirs, require_config, require_files, resolve_existing_path, validate_labeled_dataset, validate_unlabeled_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run StanceEval2026 CLS4 experiments.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--base-model-id", default=None, help="Override the best base Hugging Face model id.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed pipeline actions and tracebacks.")
    parser.add_argument("--dry-run", action="store_true", help="Verify the pipeline without training.")
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
    logger = PipelineLogger(total_steps=8, verbose=args.verbose)
    try:
        with logger.stage("Loading configuration", "Config"):
            require_config(args.config)
            config = load_config(args.config)
            set_seed(int(config["seed"]))
            columns = columns_from_config(config)
            label2id, id2label = labels_from_config(config)

        with logger.stage("Validating paths", "Dataset"):
            data_dir = Path(config["paths"]["data_dir"])
            train_path = resolve_existing_path(
                data_dir,
                config["paths"]["train_file"],
                config["paths"].get("fallback_train_file", "MawqifV2/Track 1/train.csv"),
                "train",
            )
            dev_path = resolve_existing_path(
                data_dir,
                config["paths"]["dev_file"],
                config["paths"].get("fallback_dev_file", "MawqifV2/Track 1/dev.csv"),
                "dev",
            )
            test_seen = data_dir / config["paths"].get("test_seen_file", "")
            test_unseen = data_dir / config["paths"].get("test_unseen_file", "")
            fallback_seen = data_dir / config["paths"].get("fallback_test_seen_file", "")
            fallback_unseen = data_dir / config["paths"].get("fallback_test_unseen_file", "")
            require_files([train_path, dev_path])
            validate_unlabeled_file(test_seen if test_seen.exists() else fallback_seen, columns)
            validate_unlabeled_file(test_unseen if test_unseen.exists() else fallback_unseen, columns)

        with logger.stage("Preparing output folders", "Config"):
            output_dir = ensure_dir(Path(config["paths"]["output_dir"]) / "experiments")
            results_dir = ensure_dir(config["paths"]["results_dir"])
            ensure_output_dirs([output_dir, results_dir, config["paths"]["best_model_dir"]])

        with logger.stage("Reading datasets", "Dataset"):
            train_df = load_labeled_dataframe(
                train_path,
                columns,
                label2id,
                max_samples=config["training"].get("max_train_samples"),
            )
            dev_df = load_labeled_dataframe(
                dev_path,
                columns,
                label2id,
                max_samples=config["training"].get("max_dev_samples"),
            )

        with logger.stage("Validating dataset labels", "Dataset"):
            validate_labeled_dataset(train_df, columns, set(label2id), "train")
            validate_labeled_dataset(dev_df, columns, set(label2id), "dev")

        with logger.stage("Selecting model", "Model"):
            base_model_name, base_model_id = choose_base_model(config, args.base_model_id)
            print(f"Selected model: {base_model_name} ({base_model_id})")

        rows: list[dict] = []
        device = get_device()
        with logger.stage("Running enabled experiment(s)", "Training"):
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
                    verbose=args.verbose,
                    dry_run=args.dry_run,
                )
                rows.append(row)

        with logger.stage("Saving experiment comparison", "Validation"):
            if not rows:
                raise RuntimeError("No enabled experiments were configured.")
            comparison = pd.DataFrame(rows).sort_values("dev_Favg2", ascending=False).reset_index(drop=True)
            winner = comparison.iloc[0]
            if not args.dry_run:
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
            print(f"Best checkpoint: {winner['checkpoint_path']}")
            logger.status.mark("Checkpoint", True if args.dry_run else bool(winner["checkpoint_path"]))
            logger.status.mark("Submission", True, "generated by generate_submissions.py")
    finally:
        logger.status.print_summary()


if __name__ == "__main__":
    main()
