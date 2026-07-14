from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def json_dump(data: Any, path: str | Path) -> None:
    output = Path(path)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def json_load(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def remove_dir(path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        shutil.rmtree(target)


def copy_tree(src: str | Path, dst: str | Path) -> None:
    source = Path(src)
    target = Path(dst)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def select_amp_dtype(mode: str) -> torch.dtype | None:
    if mode == "off":
        return None
    if not torch.cuda.is_available():
        return None
    if mode == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    if mode == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None


def path_from_parts(*parts: str | Path) -> Path:
    return Path(os.path.join(*(str(part) for part in parts)))
