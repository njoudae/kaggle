from __future__ import annotations

import torch
import torch.nn.functional as F


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return -torch.sum(probs * torch.log(probs + 1e-8))


def mean_cross_entropy(logits: list[torch.Tensor], labels: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    losses = [F.cross_entropy(item, labels) for item in logits]
    return torch.stack(losses).mean(), losses
