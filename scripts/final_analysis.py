from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.pipeline_logging import PipelineLogger
from src.utils import copy_tree, ensure_dir, json_dump, json_load
from src.validation import require_config


RESULT_FILES = [
    "base_model_comparison.csv",
    "layerwise_results.csv",
    "cls4_equal_loss_results.csv",
    "cls4_frozen_results.csv",
    "weighted_logits_results.csv",
    "learnable_loss_results.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect final StanceEval2026 experiment analysis.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _read_experiment_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source_file"] = path.name
    return df


def _metric_column(df: pd.DataFrame) -> str:
    if "dev_Favg2" in df.columns:
        return "dev_Favg2"
    if "Dev Favg2" in df.columns:
        return "Dev Favg2"
    raise ValueError("No Dev Favg2 metric column found.")


def main() -> None:
    args = parse_args()
    logger = PipelineLogger(total_steps=6, verbose=args.verbose)
    try:
        with logger.stage("Loading configuration", "Config"):
            require_config(args.config)
            config = load_config(args.config)
            results_dir = ensure_dir(config["paths"]["results_dir"])
            final_dir = ensure_dir(results_dir / "final_analysis")

        with logger.stage("Collecting result CSVs", "Validation"):
            frames: list[pd.DataFrame] = []
            for name in RESULT_FILES:
                path = results_dir / name
                if path.exists():
                    frames.append(_read_experiment_csv(path))
                    print(f"Loaded: {path}")
            if not frames:
                raise FileNotFoundError(f"No result CSVs found in {results_dir}")

        with logger.stage("Building comparison table", "Validation"):
            normalized: list[pd.DataFrame] = []
            for frame in frames:
                metric = _metric_column(frame)
                work = frame.copy()
                if metric != "dev_Favg2":
                    work["dev_Favg2"] = work[metric]
                if "experiment_name" not in work.columns:
                    work["experiment_name"] = work.get("source_file", "unknown")
                normalized.append(work)
            comparison = pd.concat(normalized, ignore_index=True, sort=False)
            comparison = comparison.sort_values("dev_Favg2", ascending=False).reset_index(drop=True)
            comparison.to_csv(results_dir / "comparison.csv", index=False)
            comparison.to_csv(final_dir / "comparison.csv", index=False)

        with logger.stage("Selecting best experiment", "Checkpoint"):
            best = comparison.iloc[0].to_dict()
            best_checkpoint = str(best.get("checkpoint_path", ""))
            if best_checkpoint and Path(best_checkpoint).exists():
                source = Path(best_checkpoint).resolve()
                target = Path(config["paths"]["best_model_dir"]).resolve()
                if source != target:
                    copy_tree(source, target)
                best["final_checkpoint_path"] = str(target)
            json_dump(best, results_dir / "best_experiment.json")
            print("Best experiment:", best.get("experiment_name"))
            print("Best checkpoint:", best.get("final_checkpoint_path", best_checkpoint))

        with logger.stage("Loading selected layers and backbone", "Validation"):
            selected_layers = []
            selected_path = results_dir / "selected_layers.json"
            if selected_path.exists():
                selected_layers = json_load(selected_path)
            best_backbone = None
            base_path = results_dir / "base_model_comparison.csv"
            if base_path.exists():
                base = pd.read_csv(base_path).sort_values("dev_Favg2", ascending=False).iloc[0]
                best_backbone = {"model_name": base["model_name"], "hf_id": base["hf_id"]}
            json_dump({"best_backbone": best_backbone, "selected_layers": selected_layers}, final_dir / "summary.json")

        with logger.stage("Generating plots", "Validation"):
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 4.5))
            labels = comparison["experiment_name"].astype(str).tolist()
            plt.bar(range(len(comparison)), comparison["dev_Favg2"].astype(float))
            plt.xticks(range(len(comparison)), labels, rotation=45, ha="right")
            plt.ylabel("Dev Favg2")
            plt.title("Experiment Comparison")
            plt.tight_layout()
            plt.savefig(final_dir / "experiment_comparison.png", dpi=180)
            plt.close()

            layer_path = results_dir / "layerwise_results.csv"
            if layer_path.exists():
                layer = pd.read_csv(layer_path).sort_values("layer")
                plt.figure(figsize=(8, 4.5))
                plt.plot(layer["layer"], layer["dev_Favg2"], marker="o")
                plt.xlabel("Layer")
                plt.ylabel("Dev Favg2")
                plt.title("Layer Ranking")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(final_dir / "layer_ranking.png", dpi=180)
                plt.close()

            for name, column, out in [
                ("weighted_logits_alpha_history.csv", "learned_fusion_weights", "fusion_weights.png"),
                ("learnable_loss_lambda_history.csv", "learned_loss_weights", "loss_weights.png"),
            ]:
                path = results_dir / name
                if path.exists():
                    hist = pd.read_csv(path)
                    values = hist[column].map(ast.literal_eval)
                    weights = pd.DataFrame(values.tolist())
                    weights["epoch"] = hist["epoch"]
                    plt.figure(figsize=(8, 4.5))
                    for col in [item for item in weights.columns if item != "epoch"]:
                        plt.plot(weights["epoch"], weights[col], marker="o", label=f"w{col}")
                    plt.xlabel("Epoch")
                    plt.ylabel("Weight")
                    plt.title(out.replace("_", " ").replace(".png", "").title())
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(final_dir / out, dpi=180)
                    plt.close()
    finally:
        logger.status.print_summary()


if __name__ == "__main__":
    main()
