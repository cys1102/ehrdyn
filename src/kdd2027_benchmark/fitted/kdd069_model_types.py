from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class TransitionOutput:
    mean: torch.Tensor
    log_scale: torch.Tensor
    auxiliary_loss: torch.Tensor


class GaussianResidualHead(nn.Module):
    def __init__(self, hidden_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.mean = nn.Linear(hidden_dim, feature_dim)
        self.log_scale = nn.Linear(hidden_dim, feature_dim)

    def forward(self, hidden: torch.Tensor, current: torch.Tensor) -> TransitionOutput:
        mean = current + self.mean(hidden)
        log_scale = torch.clamp(self.log_scale(hidden), min=-5.0, max=2.0)
        return TransitionOutput(mean, log_scale, hidden.new_zeros(()))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
