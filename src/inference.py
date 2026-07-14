from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import ColumnConfig, StanceDataset, load_unlabeled_dataframe
from .models import build_model
from .trainer import predict_loader
from .utils import ensure_dir, json_load


def load_trained_model(checkpoint_dir: str | Path, device: torch.device) -> tuple[Any, Any, dict[str, Any]]:
    source = Path(checkpoint_dir)
    metadata = json_load(source / "custom_model_metadata.json")
    tokenizer = AutoTokenizer.from_pretrained(source)
    if metadata.get("model_type") == "cls4":
        label2id = {key: int(value) for key, value in metadata["label2id"].items()}
        id2label = {int(key): value for key, value in metadata["id2label"].items()}
        model = build_model(metadata["base_model_id"], label2id, id2label, metadata["experiment_config"])
        state = torch.load(source / "pytorch_model.bin", map_location=device)
        model.load_state_dict(state)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(source)
    model.to(device)
    model.eval()
    return model, tokenizer, metadata


def predict_unlabeled(
    model: torch.nn.Module,
    tokenizer: Any,
    df: pd.DataFrame,
    columns: ColumnConfig,
    max_length: int,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
) -> list[int]:
    dataset = StanceDataset(df, tokenizer, columns, max_length=max_length, labeled=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    _, preds = predict_loader(model, loader, device)
    return preds


def validate_submission(test_df: pd.DataFrame, submission: pd.DataFrame, columns: ColumnConfig, valid_labels: set[str]) -> None:
    expected_columns = [columns.id_col, columns.label_col]
    if list(submission.columns) != expected_columns:
        raise ValueError(f"Submission columns must be exactly {expected_columns}, got {list(submission.columns)}")
    if len(submission) != len(test_df):
        raise ValueError(f"Submission length mismatch: expected {len(test_df)}, got {len(submission)}")
    if submission[columns.id_col].astype(str).tolist() != test_df[columns.id_col].astype(str).tolist():
        raise ValueError("Submission ids do not match test ids in the original order.")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing values.")
    invalid = sorted(set(submission[columns.label_col].astype(str)) - valid_labels)
    if invalid:
        raise ValueError(f"Submission contains invalid stance labels: {invalid}")


def generate_submission_file(
    *,
    test_path: str | Path,
    output_path: str | Path,
    checkpoint_dir: str | Path,
    columns: ColumnConfig,
    id2label: dict[int, str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
) -> Path:
    model, tokenizer, _ = load_trained_model(checkpoint_dir, device)
    test_df = load_unlabeled_dataframe(test_path, columns)
    preds = predict_unlabeled(model, tokenizer, test_df, columns, max_length, batch_size, device, num_workers)
    submission = pd.DataFrame(
        {
            columns.id_col: test_df[columns.id_col].astype(str),
            columns.label_col: [id2label[int(pred)] for pred in preds],
        }
    )
    validate_submission(test_df, submission, columns, set(id2label.values()))
    target = Path(output_path)
    ensure_dir(target.parent)
    submission.to_csv(target, index=False, encoding="utf-8")
    return target


def zip_submissions(zip_path: str | Path, files: list[str | Path]) -> Path:
    target = Path(zip_path)
    ensure_dir(target.parent)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files:
            source = Path(file_path)
            archive.write(source, arcname=source.name)
    return target
