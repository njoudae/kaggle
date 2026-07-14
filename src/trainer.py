from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import ColumnConfig, StanceDataset
from .metrics import compute_metrics, per_target_metrics, prefixed_metrics, save_reports
from .models import CLS4Classifier, build_model, save_custom_model
from .utils import cleanup_cuda, ensure_dir, json_dump, remove_dir, select_amp_dtype


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _output_loss_and_logits(output: Any) -> tuple[torch.Tensor | None, torch.Tensor]:
    return output.loss, output.logits


def _weights_from_output(output: Any) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {}
    fusion = getattr(output, "fusion_weights", None)
    loss_weights = getattr(output, "loss_weights", None)
    if fusion is not None:
        data["fusion_weights"] = [float(item) for item in fusion]
    if loss_weights is not None:
        data["loss_weights"] = [float(item) for item in loss_weights]
    return data


@torch.no_grad()
def predict_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    labels: list[int] = []
    preds: list[int] = []
    for batch in loader:
        if "labels" in batch:
            labels.extend(batch["labels"].cpu().numpy().tolist())
        batch = _move_batch(batch, device)
        label_values = batch.pop("labels", None)
        output = model(labels=label_values, **batch) if label_values is not None else model(**batch)
        _, logits = _output_loss_and_logits(output)
        preds.extend(torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist())
    return labels, preds


def _save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    checkpoint_dir: Path,
    metadata: dict[str, Any],
) -> None:
    remove_dir(checkpoint_dir)
    ensure_dir(checkpoint_dir)
    if isinstance(model, CLS4Classifier):
        save_custom_model(model, checkpoint_dir, metadata)
    else:
        model.save_pretrained(checkpoint_dir)
        json_dump({**metadata, "model_type": "sequence_classification"}, checkpoint_dir / "custom_model_metadata.json")
    tokenizer.save_pretrained(checkpoint_dir)
    json_dump(metadata["label2id"], checkpoint_dir / "label2id.json")
    json_dump(metadata["id2label"], checkpoint_dir / "id2label.json")


def train_experiment(
    *,
    experiment_name: str,
    model_name: str,
    model_id: str,
    experiment_config: dict[str, Any],
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    columns: ColumnConfig,
    label2id: dict[str, int],
    id2label: dict[int, str],
    training_config: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    run_dir = ensure_dir(Path(output_dir) / experiment_name)
    checkpoint_dir = run_dir / "best_checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    max_length = int(training_config["max_length"])
    train_ds = StanceDataset(train_df, tokenizer, columns, max_length=max_length, labeled=True)
    dev_ds = StanceDataset(dev_df, tokenizer, columns, max_length=max_length, labeled=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config.get("num_workers", 0) or 0),
        pin_memory=bool(training_config.get("pin_memory", False)) and device.type == "cuda",
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=int(training_config["eval_batch_size"]),
        shuffle=False,
        num_workers=int(training_config.get("num_workers", 0) or 0),
        pin_memory=bool(training_config.get("pin_memory", False)) and device.type == "cuda",
    )

    model = build_model(model_id, label2id, id2label, experiment_config).to(device)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
        eps=float(training_config.get("adam_epsilon", 1e-8)),
    )
    grad_accum = int(training_config.get("gradient_accumulation_steps", 1) or 1)
    epochs = int(training_config["epochs"])
    total_steps = math.ceil(len(train_loader) / grad_accum) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=max(total_steps, 1))
    amp_dtype = select_amp_dtype(str(training_config.get("mixed_precision", "auto")))
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    history: list[dict[str, Any]] = []
    best_metric = -1.0
    best_epoch = 0
    bad_epochs = 0
    early_cfg = training_config.get("early_stopping", {}) or {}
    state_path = run_dir / "last_training_state.pt"

    metadata = {
        "model_type": "cls4" if experiment_config.get("architecture") == "cls4" else "sequence_classification",
        "base_model_id": model_id,
        "model_name": model_name,
        "experiment_name": experiment_name,
        "experiment_config": experiment_config,
        "label2id": label2id,
        "id2label": {str(key): value for key, value in id2label.items()},
    }
    json_dump(metadata, run_dir / "experiment_config.json")

    start_epoch = 1
    if bool(training_config.get("resume_from_checkpoint", False)) and state_path.exists():
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state.get("history", [])
        best_metric = float(state.get("best_metric", best_metric))
        best_epoch = int(state.get("best_epoch", best_epoch))
        bad_epochs = int(state.get("bad_epochs", bad_epochs))
        start_epoch = int(state["epoch"]) + 1

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"{experiment_name} epoch {epoch}/{epochs}", leave=False)
        last_weights: dict[str, list[float]] = {}

        for step, batch in enumerate(progress, start=1):
            batch = _move_batch(batch, device)
            labels = batch.pop("labels")
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                output = model(labels=labels, **batch)
                loss, _ = _output_loss_and_logits(output)
                if loss is None:
                    raise RuntimeError("Model did not return a training loss.")
                loss = loss / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            last_weights.update(_weights_from_output(output))
            total_loss += float(loss.item()) * grad_accum

            if step % grad_accum == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_config["gradient_clip_norm"]))
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(loss=f"{float(loss.item()) * grad_accum:.4f}")

        dev_labels, dev_preds = predict_loader(model, dev_loader, device)
        raw_metrics = compute_metrics(dev_labels, dev_preds)
        metrics = prefixed_metrics(raw_metrics, prefix="dev")
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_loader), 1),
            **metrics,
            **{f"learned_{key}": value for key, value in last_weights.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
        json_dump(history, run_dir / "training_history.json")

        if metrics["dev_Favg2"] > best_metric:
            best_metric = metrics["dev_Favg2"]
            best_epoch = epoch
            bad_epochs = 0
            _save_checkpoint(model, tokenizer, checkpoint_dir, metadata)
            best_df = dev_df.copy()
            best_df["pred"] = dev_preds
            best_df["pred_stance"] = [id2label[pred] for pred in dev_preds]
            best_df.to_csv(run_dir / "dev_predictions.csv", index=False, encoding="utf-8")
            pd.DataFrame([raw_metrics]).to_csv(run_dir / "overall_metrics.csv", index=False)
            json_dump(raw_metrics, run_dir / "overall_metrics.json")
            per_target_metrics(best_df, "label", "pred", columns.target_col).to_csv(run_dir / "per_target_metrics.csv", index=False)
            save_reports(dev_labels, dev_preds, id2label, run_dir)
            json_dump({"best_epoch": best_epoch, **raw_metrics, **last_weights}, checkpoint_dir / "best_metrics.json")
        else:
            bad_epochs += 1

        if bool(early_cfg.get("enabled", False)) and bad_epochs >= int(early_cfg.get("patience", 3)):
            break

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
            },
            state_path,
        )

    summary = {
        "model_name": model_name,
        "hf_id": model_id,
        "experiment_name": experiment_name,
        "best_epoch": best_epoch,
        "checkpoint_path": str(checkpoint_dir),
    }
    if (run_dir / "overall_metrics.json").exists():
        import json

        with (run_dir / "overall_metrics.json").open("r", encoding="utf-8") as handle:
            best_metrics = json.load(handle)
        summary.update(
            {
                "dev_F_favor": best_metrics["F1_Favor"],
                "dev_F_against": best_metrics["F1_Against"],
                "dev_F_none": best_metrics["F1_None"],
                "dev_Favg2": best_metrics["Favg2"],
                "dev_Favg3": best_metrics["Favg3"],
                "dev_accuracy": best_metrics["Accuracy"],
            }
        )
    pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)
    if state_path.exists():
        state_path.unlink()
    del model
    cleanup_cuda()
    return summary
