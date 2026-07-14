from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import columns_from_config, labels_from_config, load_unlabeled_dataframe
from src.inference import load_trained_model, predict_unlabeled, validate_submission, zip_submissions
from src.pipeline_logging import PipelineLogger, print_gpu_diagnostics, print_model_device_and_memory
from src.utils import ensure_dir, get_device, set_seed
from src.validation import ensure_output_dirs, require_config, validate_unlabeled_file

import pandas as pd


def resolve_test_path(data_dir: Path, preferred: str, fallback: str | None, label: str) -> Path:
    preferred_path = data_dir / preferred
    if preferred_path.exists():
        return preferred_path
    if fallback:
        fallback_path = data_dir / fallback
        if fallback_path.exists():
            print(f"Official {label} test file not found: {preferred_path}")
            print(f"Using fallback {label} file from the current repository: {fallback_path}")
            return fallback_path
    raise FileNotFoundError(
        f"Could not find {label} test file. Tried {preferred_path}"
        + (f" and fallback {data_dir / fallback}" if fallback else "")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate StanceEval2026 submission files.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory. Defaults to paths.best_model_dir.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed pipeline actions and tracebacks.")
    parser.add_argument("--dry-run", action="store_true", help="Verify submission pipeline without writing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = PipelineLogger(total_steps=10, verbose=args.verbose)
    try:
        with logger.stage("Loading configuration", "Config"):
            require_config(args.config)
            config = load_config(args.config)
            set_seed(int(config["seed"]))
            columns = columns_from_config(config)
            _, id2label = labels_from_config(config)

        with logger.stage("GPU diagnostics", "GPU"):
            device = get_device()
            print_gpu_diagnostics()

        with logger.stage("Resolving input and output paths", "Config"):
            data_dir = Path(config["paths"]["data_dir"])
            results_dir = ensure_dir(config["paths"]["results_dir"])
            checkpoint_dir = Path(args.checkpoint or config["paths"]["best_model_dir"])
            ensure_output_dirs([results_dir])
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            training = config["training"]
            seen_test_path = resolve_test_path(
                data_dir,
                config["paths"]["test_seen_file"],
                config["paths"].get("fallback_test_seen_file"),
                "seen",
            )
            unseen_test_path = resolve_test_path(
                data_dir,
                config["paths"]["test_unseen_file"],
                config["paths"].get("fallback_test_unseen_file"),
                "unseen",
            )

        with logger.stage("Validating test files", "Dataset"):
            validate_unlabeled_file(seen_test_path, columns)
            validate_unlabeled_file(unseen_test_path, columns)

        with logger.stage("Loading checkpoint model", "Model"):
            model, tokenizer, _ = load_trained_model(checkpoint_dir, device)
            print_model_device_and_memory(model)

        with logger.stage("Reading test datasets", "Dataset"):
            seen_df = load_unlabeled_dataframe(seen_test_path, columns)
            unseen_df = load_unlabeled_dataframe(unseen_test_path, columns)

        with logger.stage("Predicting seen test set", "Validation"):
            seen_preds = predict_unlabeled(
                model,
                tokenizer,
                seen_df,
                columns,
                int(training["max_length"]),
                int(training["eval_batch_size"]),
                device,
                int(training.get("num_workers", 0) or 0),
            )

        with logger.stage("Predicting unseen test set", "Validation"):
            unseen_preds = predict_unlabeled(
                model,
                tokenizer,
                unseen_df,
                columns,
                int(training["max_length"]),
                int(training["eval_batch_size"]),
                device,
                int(training.get("num_workers", 0) or 0),
            )

        with logger.stage("Verifying submissions", "Submission"):
            seen_submission = pd.DataFrame(
                {columns.id_col: seen_df[columns.id_col].astype(str), columns.label_col: [id2label[int(pred)] for pred in seen_preds]}
            )
            unseen_submission = pd.DataFrame(
                {columns.id_col: unseen_df[columns.id_col].astype(str), columns.label_col: [id2label[int(pred)] for pred in unseen_preds]}
            )
            validate_submission(seen_df, seen_submission, columns, set(id2label.values()))
            validate_submission(unseen_df, unseen_submission, columns, set(id2label.values()))

        with logger.stage("Writing submission files", "Submission"):
            if args.dry_run:
                print("Dry-run enabled: submission files were verified but not written.")
            else:
                seen = results_dir / config["submissions"]["seen_name"]
                unseen = results_dir / config["submissions"]["unseen_name"]
                seen_submission.to_csv(seen, index=False, encoding="utf-8")
                unseen_submission.to_csv(unseen, index=False, encoding="utf-8")
                print(f"Saved: {seen}")
                print(f"Saved: {unseen}")
                if bool(config["submissions"].get("create_zip", True)):
                    zip_path = zip_submissions(results_dir / config["submissions"]["zip_name"], [seen, unseen])
                    print(f"Saved: {zip_path}")
    finally:
        logger.status.print_summary()


if __name__ == "__main__":
    main()
