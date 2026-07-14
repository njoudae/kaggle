from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

from .losses import entropy_from_probs, mean_cross_entropy


@dataclass
class ModelOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    head_logits: list[torch.Tensor] | None = None
    fusion_weights: list[float] | None = None
    loss_weights: list[float] | None = None


def build_sequence_classifier(
    model_id: str,
    label2id: dict[str, int],
    id2label: dict[int, str],
) -> nn.Module:
    return AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        output_hidden_states=False,
    )


class CLS4Classifier(nn.Module):
    def __init__(
        self,
        model_id: str,
        num_labels: int = 3,
        dropout: float | None = None,
        fusion: str = "equal_logits",
        loss_mode: str = "equal_mean",
        lambda_fused: float = 1.0,
        beta_entropy: float = 0.0,
        freeze_encoder: bool = False,
        unfreeze_last_n_layers: int = 0,
    ) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_id, output_hidden_states=True)
        self.encoder = AutoModel.from_pretrained(model_id, config=self.config)
        hidden_size = int(self.config.hidden_size)
        dropout_prob = float(dropout if dropout is not None else getattr(self.config, "hidden_dropout_prob", 0.1))
        self.dropout = nn.Dropout(dropout_prob)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, num_labels) for _ in range(4)])
        self.fusion = fusion
        self.loss_mode = loss_mode
        self.lambda_fused = float(lambda_fused)
        self.beta_entropy = float(beta_entropy)
        self.fusion_weights = nn.Parameter(torch.zeros(4))
        self.loss_weight_logits = nn.Parameter(torch.zeros(4))

        if freeze_encoder:
            self.freeze_encoder()
        if unfreeze_last_n_layers:
            self.unfreeze_last_layers(unfreeze_last_n_layers)

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_last_layers(self, n_layers: int) -> None:
        if n_layers <= 0:
            return
        module = getattr(self.encoder, "encoder", None)
        layers = getattr(module, "layer", None)
        if layers is None:
            return
        for layer in layers[-n_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def aggregate_logits(self, head_logits: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.fusion == "weighted_logits":
            alpha = torch.softmax(self.fusion_weights, dim=0)
            final_logits = sum(alpha[index] * logits for index, logits in enumerate(head_logits))
            return final_logits, alpha
        final_logits = torch.stack(head_logits, dim=0).mean(dim=0)
        return final_logits, None

    def forward(self, labels: torch.Tensor | None = None, **batch: torch.Tensor) -> ModelOutput:
        outputs = self.encoder(**batch, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) < 4:
            raise RuntimeError("The encoder did not return at least four hidden states.")

        cls_layers = [hidden_states[-4][:, 0, :], hidden_states[-3][:, 0, :], hidden_states[-2][:, 0, :], hidden_states[-1][:, 0, :]]
        head_logits = [head(self.dropout(cls_value)) for head, cls_value in zip(self.heads, cls_layers)]
        final_logits, alpha = self.aggregate_logits(head_logits)

        loss = None
        loss_weights = None
        if labels is not None:
            head_loss, per_head = mean_cross_entropy(head_logits, labels)
            if self.loss_mode == "heads_plus_fused":
                fused_loss = F.cross_entropy(final_logits, labels)
                loss = head_loss + self.lambda_fused * fused_loss
            elif self.loss_mode == "learnable_loss":
                loss_weights_tensor = torch.softmax(self.loss_weight_logits, dim=0)
                weighted = sum(loss_weights_tensor[index] * item for index, item in enumerate(per_head))
                loss = weighted - self.beta_entropy * entropy_from_probs(loss_weights_tensor)
                loss_weights = loss_weights_tensor.detach().cpu().tolist()
            else:
                loss = head_loss

        return ModelOutput(
            loss=loss,
            logits=final_logits,
            head_logits=head_logits,
            fusion_weights=alpha.detach().cpu().tolist() if alpha is not None else None,
            loss_weights=loss_weights,
        )


def build_model(
    model_id: str,
    label2id: dict[str, int],
    id2label: dict[int, str],
    experiment: dict[str, Any],
) -> nn.Module:
    architecture = experiment.get("architecture", "sequence_classification")
    if architecture == "sequence_classification":
        return build_sequence_classifier(model_id, label2id, id2label)
    if architecture != "cls4":
        raise ValueError(f"Unsupported architecture: {architecture}")
    return CLS4Classifier(
        model_id=model_id,
        num_labels=len(label2id),
        fusion=experiment.get("fusion", "equal_logits"),
        loss_mode=experiment.get("loss_mode", "equal_mean"),
        lambda_fused=float(experiment.get("lambda_fused", 1.0)),
        beta_entropy=float(experiment.get("beta_entropy", 0.0)),
        freeze_encoder=bool(experiment.get("freeze_encoder", False)),
        unfreeze_last_n_layers=int(experiment.get("unfreeze_last_n_layers", 0) or 0),
    )


def save_custom_model(model: CLS4Classifier, path: str | Path, metadata: dict[str, Any]) -> None:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target / "pytorch_model.bin")
    model.config.save_pretrained(target)
    with (target / "custom_model_metadata.json").open("w", encoding="utf-8") as handle:
        import json

        json.dump(metadata, handle, ensure_ascii=False, indent=2)
