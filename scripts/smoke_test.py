from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import StanceDataset, columns_from_config, labels_from_config, load_labeled_dataframe
from src.pipeline_logging import PipelineLogger, print_gpu_diagnostics, print_model_device_and_memory
from src.utils import ensure_dir, get_device, set_seed
from src.validation import (
    describe_model_cache,
    ensure_output_dirs,
    require_config,
    require_files,
    resolve_existing_path,
    validate_labeled_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the real StanceEval2026 training pipeline.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--base-model-id", default=None)
    parser.add_argument("--verbose", action="store_true", help="Print detailed pipeline actions and tracebacks.")
    parser.add_argument("--dry-run", action="store_true", help="Alias for smoke mode; no full training is performed.")
    return parser.parse_args()


def select_model_id(config: dict, override: str | None) -> str:
    if override:
        return override
    for model_cfg in config["models"]:
        if bool(model_cfg.get("enabled", True)):
            return str(model_cfg["hf_id"])
    raise RuntimeError("No enabled model found in config.")


def main() -> None:
    args = parse_args()
    logger = PipelineLogger(total_steps=12, verbose=args.verbose)
    checkpoint_dir: Path | None = None
    printed_summary = False
    try:
        with logger.stage("Loading configuration", "Config"):
            require_config(args.config)
            config = load_config(args.config)
            set_seed(int(config["seed"]))
            columns = columns_from_config(config)
            label2id, id2label = labels_from_config(config)
            model_id = select_model_id(config, args.base_model_id)

        with logger.stage("Validating files and folders", "Config"):
            data_dir = Path(config["paths"]["data_dir"])
            train_path = resolve_existing_path(
                data_dir,
                config["paths"]["train_file"],
                config["paths"].get("fallback_train_file", "MawqifV2/Track 1/train.csv"),
                "train",
            )
            require_files([train_path])
            smoke_dir = ensure_dir(Path(config["paths"]["output_dir"]) / "smoke_test")
            ensure_output_dirs([smoke_dir])
            checkpoint_dir = smoke_dir / "checkpoint"

        with logger.stage("GPU diagnostics", "GPU"):
            device = get_device()
            print_gpu_diagnostics()
            if bool(config["training"].get("require_gpu", True)) and device.type != "cuda":
                raise RuntimeError("CUDA GPU is required for smoke test. Enable GPU in Kaggle settings.")

        with logger.stage("Reading 16 training samples", "Dataset"):
            df = load_labeled_dataframe(train_path, columns, label2id, max_samples=16)
            validate_labeled_dataset(df, columns, set(label2id), "smoke_train")
            if len(df) < 1:
                raise RuntimeError("Smoke dataset is empty.")
            print(f"Smoke rows: {len(df)}")

        with logger.stage("Loading tokenizer", "Tokenizer"):
            describe_model_cache(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)

        with logger.stage("Loading pretrained model", "Model"):
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                num_labels=len(label2id),
                label2id=label2id,
                id2label=id2label,
                output_hidden_states=False,
            )

        with logger.stage("Moving model to GPU", "GPU"):
            model.to(device)
            print_model_device_and_memory(model)

        with logger.stage("Tokenizing samples", "Tokenization"):
            dataset = StanceDataset(df, tokenizer, columns, max_length=int(config["training"]["max_length"]), labeled=True)
            batch = next(iter(DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=False)))
            print("Batch tensors:", {key: tuple(value.shape) for key, value in batch.items()})

        with logger.stage("Running one forward pass", "Training"):
            model.train()
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            if output.loss is None:
                raise RuntimeError("Forward pass did not produce a loss.")
            print(f"Smoke loss: {float(output.loss.item()):.6f}")

        with logger.stage("Running one backward pass", "Training"):
            output.loss.backward()
            grad_found = any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
            if not grad_found:
                raise RuntimeError("Backward pass completed but no gradients were produced.")

        with logger.stage("Saving temporary checkpoint", "Checkpoint"):
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

        with logger.stage("Reloading checkpoint and running inference", "Checkpoint"):
            reloaded = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
            reloaded.eval()
            inference_batch = {key: value[:1].to(device) for key, value in batch.items() if key != "labels"}
            with torch.no_grad():
                logits = reloaded(**inference_batch).logits
            pred = torch.argmax(logits, dim=-1).detach().cpu().tolist()
            if not pred or any(item not in id2label for item in pred):
                raise RuntimeError(f"Reloaded checkpoint produced invalid prediction(s): {pred}")
            print("Checkpoint inference prediction:", [id2label[item] for item in pred])

        logger.status.mark("Validation", True, "one forward/backward cycle verified")
        logger.status.mark("Submission", True, "not applicable to smoke test")
        logger.status.print_summary()
        printed_summary = True
        print("PIPELINE VERIFIED")
    finally:
        if logger.status.items and not printed_summary:
            logger.status.print_summary()


if __name__ == "__main__":
    main()
