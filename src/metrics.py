from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    f_against = f1_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)
    f_favor = f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)
    f_none = f1_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)
    return {
        "F1_Favor": float(f_favor),
        "F1_Against": float(f_against),
        "F1_None": float(f_none),
        "Favg2": float((f_favor + f_against) / 2.0),
        "Favg3": float((f_favor + f_against + f_none) / 3.0),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
    }


def prefixed_metrics(metrics: dict[str, float], prefix: str = "dev") -> dict[str, float]:
    mapping = {
        "F1_Favor": f"{prefix}_F_favor",
        "F1_Against": f"{prefix}_F_against",
        "F1_None": f"{prefix}_F_none",
        "Favg2": f"{prefix}_Favg2",
        "Favg3": f"{prefix}_Favg3",
        "Accuracy": f"{prefix}_accuracy",
    }
    return {mapping[key]: value for key, value in metrics.items()}


def per_target_metrics(
    df: pd.DataFrame,
    label_col: str,
    pred_col: str,
    target_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in sorted(df[target_col].astype(str).unique()):
        subset = df[df[target_col].astype(str) == target]
        metrics = compute_metrics(subset[label_col].astype(int).tolist(), subset[pred_col].astype(int).tolist())
        rows.append({"target": target, **metrics})
    return pd.DataFrame(rows)


def save_reports(
    y_true: list[int],
    y_pred: list[int],
    id2label: dict[int, str],
    output_dir: str | Path,
) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    labels = sorted(id2label)
    names = [id2label[item] for item in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(target / "classification_report.csv")
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(matrix, index=names, columns=names).to_csv(target / "confusion_matrix.csv")
