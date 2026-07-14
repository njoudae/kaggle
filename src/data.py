from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .preprocessing import preprocess_text


@dataclass(frozen=True)
class ColumnConfig:
    id_col: str
    id_aliases: tuple[str, ...]
    target_col: str
    text_col: str
    label_col: str


def columns_from_config(config: dict[str, Any]) -> ColumnConfig:
    columns = config["columns"]
    aliases = tuple(columns.get("id_aliases") or [columns["id"]])
    return ColumnConfig(
        id_col=columns["id"],
        id_aliases=aliases,
        target_col=columns["target"],
        text_col=columns["text"],
        label_col=columns["stance"],
    )


def labels_from_config(config: dict[str, Any]) -> tuple[dict[str, int], dict[int, str]]:
    label2id = {str(key): int(value) for key, value in config["labels"].items()}
    id2label = {value: key for key, value in label2id.items()}
    return label2id, id2label


def _resolve_existing_column(df: pd.DataFrame, primary: str, aliases: tuple[str, ...]) -> str:
    candidates = (primary, *aliases)
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Missing required column. Tried: {sorted(set(candidates))}")


def _read_csv(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"CSV file not found: {source}")
    df = pd.read_csv(source, keep_default_na=False, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()
    return df


def load_labeled_dataframe(
    path: str | Path,
    columns: ColumnConfig,
    label2id: dict[str, int],
    max_samples: int | None = None,
) -> pd.DataFrame:
    df = _read_csv(path)
    id_source = _resolve_existing_column(df, columns.id_col, columns.id_aliases)
    required = [id_source, columns.target_col, columns.text_col, columns.label_col]
    for name in required:
        if name not in df.columns:
            raise ValueError(f"Missing column in {path}: {name}")

    work = df.copy()
    if id_source != columns.id_col:
        work[columns.id_col] = work[id_source]
    for name in [columns.id_col, columns.target_col, columns.text_col, columns.label_col]:
        work[name] = work[name].astype(str).str.strip()

    before = len(work)
    work = work[work[columns.label_col] != ""].copy()
    if len(work) != before:
        warnings.warn(f"Dropped {before - len(work)} labeled rows with empty stance from {path}.")

    unknown = sorted(set(work[columns.label_col]) - set(label2id))
    if unknown:
        raise ValueError(f"Unknown stance labels in {path}: {unknown}")

    missing_text = (work[columns.text_col] == "").sum()
    missing_target = (work[columns.target_col] == "").sum()
    if missing_text or missing_target:
        warnings.warn(
            f"{path} has {missing_text} empty text values and {missing_target} empty target values; "
            "they are kept as empty strings."
        )

    tqdm.pandas(desc=f"Preprocessing text: {Path(path).name}")
    work[columns.text_col] = work[columns.text_col].progress_map(preprocess_text)
    tqdm.pandas(desc=f"Preprocessing target: {Path(path).name}")
    work[columns.target_col] = work[columns.target_col].progress_map(preprocess_text)
    work["label"] = work[columns.label_col].map(label2id).astype(int)

    if max_samples:
        work = work.head(int(max_samples)).copy()
    return work.reset_index(drop=True)


def load_unlabeled_dataframe(
    path: str | Path,
    columns: ColumnConfig,
    max_samples: int | None = None,
) -> pd.DataFrame:
    df = _read_csv(path)
    id_source = _resolve_existing_column(df, columns.id_col, columns.id_aliases)
    required = [id_source, columns.target_col, columns.text_col]
    for name in required:
        if name not in df.columns:
            raise ValueError(f"Missing column in {path}: {name}")

    work = df.copy()
    if id_source != columns.id_col:
        work[columns.id_col] = work[id_source]
    for name in [columns.id_col, columns.target_col, columns.text_col]:
        work[name] = work[name].astype(str)

    missing_text = (work[columns.text_col].str.strip() == "").sum()
    missing_target = (work[columns.target_col].str.strip() == "").sum()
    if missing_text or missing_target:
        warnings.warn(
            f"{path} has {missing_text} empty text values and {missing_target} empty target values; "
            "test rows are preserved and empty values become empty strings."
        )

    tqdm.pandas(desc=f"Preprocessing text: {Path(path).name}")
    work[columns.text_col] = work[columns.text_col].progress_map(preprocess_text)
    tqdm.pandas(desc=f"Preprocessing target: {Path(path).name}")
    work[columns.target_col] = work[columns.target_col].progress_map(preprocess_text)
    if max_samples:
        work = work.head(int(max_samples)).copy()
    return work.reset_index(drop=True)


class StanceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        columns: ColumnConfig,
        max_length: int,
        labeled: bool,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.columns = columns
        self.max_length = max_length
        self.labeled = labeled

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row[self.columns.target_col],
            row[self.columns.text_col],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        if self.labeled:
            item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item
