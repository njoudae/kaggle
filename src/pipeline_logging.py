from __future__ import annotations

import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import torch


@dataclass
class PipelineStatus:
    items: dict[str, tuple[bool, str]] = field(default_factory=dict)

    def mark(self, name: str, ok: bool, detail: str = "") -> None:
        self.items[name] = (ok, detail)

    def print_summary(self) -> None:
        print("\nPipeline Status:")
        for name, (ok, detail) in self.items.items():
            symbol = "✔" if ok else "✘"
            suffix = f" - {detail}" if detail else ""
            print(f"{symbol} {name}{suffix}")


class PipelineLogger:
    def __init__(self, total_steps: int, verbose: bool = False) -> None:
        self.total_steps = total_steps
        self.verbose = verbose
        self.current_step = 0
        self.status = PipelineStatus()

    def info(self, message: str) -> None:
        if self.verbose:
            print(f"    {message}")

    @contextmanager
    def stage(self, name: str, status_name: str | None = None) -> Iterator[float]:
        self.current_step += 1
        total = max(self.total_steps, self.current_step)
        label = status_name or name
        print(f"[{self.current_step}/{total}] {name}...")
        started = time.perf_counter()
        try:
            yield started
        except Exception as exc:
            elapsed = time.perf_counter() - started
            total = max(self.total_steps, self.current_step)
            print(f"[{self.current_step}/{total}] {name}: FAIL ({elapsed:.2f}s)")
            print(f"Stage failed: {name}")
            print(f"Error: {type(exc).__name__}: {exc}")
            if self.verbose:
                traceback.print_exc()
            self.status.mark(label, False, f"{type(exc).__name__}: {exc}")
            raise
        else:
            elapsed = time.perf_counter() - started
            total = max(self.total_steps, self.current_step)
            print(f"[{self.current_step}/{total}] {name}: SUCCESS ({elapsed:.2f}s)")
            self.status.mark(label, True)


def gpu_diagnostics() -> dict[str, str | int | bool]:
    data: dict[str, str | int | bool] = {
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        data.update(
            {
                "gpu_name": torch.cuda.get_device_name(index),
                "gpu_memory_total_mb": int(props.total_memory / 1024**2),
                "gpu_memory_allocated_mb": int(torch.cuda.memory_allocated(index) / 1024**2),
                "gpu_memory_reserved_mb": int(torch.cuda.memory_reserved(index) / 1024**2),
            }
        )
    return data


def print_gpu_diagnostics() -> None:
    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:
        transformers_version = "not importable"

    data = gpu_diagnostics()
    print("CUDA available:", data["cuda_available"])
    print("PyTorch version:", data["pytorch_version"])
    print("Transformers version:", transformers_version)
    if data["cuda_available"]:
        print("GPU name:", data["gpu_name"])
        print("GPU memory total MB:", data["gpu_memory_total_mb"])
        print("GPU memory allocated MB:", data["gpu_memory_allocated_mb"])
        print("GPU memory reserved MB:", data["gpu_memory_reserved_mb"])
    else:
        print("GPU name: CPU only")
        print("GPU memory: unavailable")


def print_model_device_and_memory(model: torch.nn.Module) -> None:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("unknown")
    print("model.device:", device)
    if torch.cuda.is_available():
        print("torch.cuda.memory_allocated():", torch.cuda.memory_allocated())
        print("torch.cuda.memory_reserved():", torch.cuda.memory_reserved())


def print_epoch_gpu_memory(epoch: int) -> None:
    if torch.cuda.is_available():
        print(
            f"Epoch {epoch} GPU memory | "
            f"allocated={torch.cuda.memory_allocated()} reserved={torch.cuda.memory_reserved()}"
        )
