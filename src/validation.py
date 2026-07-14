from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from transformers.utils import cached_file

from .data import ColumnConfig
from .utils import ensure_dir


def require_config(path: str | Path) -> Path:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    return config_path


def require_files(paths: list[str | Path]) -> None:
    missing = [str(Path(path)) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required file(s): {missing}")


def resolve_existing_path(
    data_dir: str | Path,
    preferred: str | Path,
    fallback: str | Path | None = None,
    label: str = "file",
) -> Path:
    data_root = Path(data_dir)
    preferred_path = data_root / preferred
    if preferred_path.exists():
        return preferred_path
    if fallback:
        fallback_path = data_root / fallback
        if fallback_path.exists():
            print(f"Configured {label} file not found: {preferred_path}")
            print(f"Using fallback {label} file: {fallback_path}")
            return fallback_path
    raise FileNotFoundError(
        f"Missing {label} file. Tried {preferred_path}"
        + (f" and fallback {data_root / fallback}" if fallback else "")
    )


def ensure_output_dirs(paths: list[str | Path]) -> None:
    for path in paths:
        ensure_dir(path)


def validate_labeled_dataset(df: pd.DataFrame, columns: ColumnConfig, valid_labels: set[str], name: str) -> None:
    if df.empty:
        raise ValueError(f"{name} dataset is empty.")
    required = [columns.id_col, columns.target_col, columns.text_col, columns.label_col, "label"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name} dataset is missing columns: {missing}")
    invalid = sorted(set(df[columns.label_col].astype(str)) - valid_labels)
    if invalid:
        raise ValueError(f"{name} dataset has invalid labels: {invalid}")
    if df["label"].isna().any():
        raise ValueError(f"{name} dataset has unmapped numeric labels.")


def validate_unlabeled_file(path: str | Path, columns: ColumnConfig) -> None:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Test file does not exist: {source}")
    header = pd.read_csv(source, nrows=0, encoding="utf-8-sig")
    id_candidates = {columns.id_col, *columns.id_aliases}
    required = [columns.target_col, columns.text_col]
    missing = [column for column in required if column not in header.columns]
    if not id_candidates.intersection(set(header.columns)):
        missing.append(f"{columns.id_col} or aliases {list(columns.id_aliases)}")
    if missing:
        raise ValueError(f"Test file {source} is missing columns: {missing}")


def check_model_cache(model_id: str) -> bool:
    try:
        cached_file(model_id, "config.json", local_files_only=True)
        return True
    except Exception:
        return False


def describe_model_cache(model_id: str) -> None:
    if check_model_cache(model_id):
        print("Model found in cache.")
    else:
        print("Downloading pretrained model...")
