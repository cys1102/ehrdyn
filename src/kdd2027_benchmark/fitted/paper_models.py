"""Minimal exact model surface ported from KDD152V2A.

The class bodies below are copied without scientific changes from
``kdd_benchmark_discovery/kdd152v2a_repaired_interfaces.py`` at the frozen
world-ehr KDD262 source commit. The data-materialization code in that module is
intentionally not imported because the public constructor already supplies the
same restricted role interfaces.
"""
from __future__ import annotations

import torch
from torch import nn


class GRUDBackbone(nn.Module):
    """Shared recurrent backbone for the point/Gaussian ablation."""

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.input = nn.Linear(feature_dim * 3 + action_dim, hidden_dim)
        self.recurrent = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        deltas: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                values,
                masks,
                torch.log1p(torch.clamp(deltas, min=0.0)),
                actions,
            ],
            dim=-1,
        )
        encoded = torch.tanh(self.input(x))
        hidden, _ = self.recurrent(encoded)
        return hidden


class DeterministicGRUDPoint(nn.Module):
    loss_name = "masked_observed_cell_point_MSE"
    output_contract = "direct_predictive_mean_only"

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = GRUDBackbone(feature_dim, action_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, feature_dim)

    def forward(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        deltas: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.mean_head(self.backbone(values, masks, deltas, actions))

    def loss(
        self,
        target: torch.Tensor,
        mean: torch.Tensor,
        observed: torch.Tensor,
    ) -> torch.Tensor:
        weight = observed.to(mean.dtype)
        return (torch.square(target - mean) * weight).sum() / torch.clamp(
            weight.sum(), min=1.0
        )


class SingleGaussianGRUD(nn.Module):
    loss_name = "masked_Gaussian_negative_log_likelihood"
    output_contract = "direct_predictive_mean_and_log_scale"

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = GRUDBackbone(feature_dim, action_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, feature_dim)
        self.log_scale_head = nn.Linear(hidden_dim, feature_dim)

    def forward(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        deltas: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(values, masks, deltas, actions)
        mean = self.mean_head(hidden)
        log_scale = torch.clamp(
            self.log_scale_head(hidden), min=-7.0, max=5.0
        )
        return mean, log_scale

    def loss(
        self,
        target: torch.Tensor,
        mean: torch.Tensor,
        log_scale: torch.Tensor,
        observed: torch.Tensor,
    ) -> torch.Tensor:
        weight = observed.to(mean.dtype)
        scale = torch.exp(log_scale)
        nll = log_scale + 0.5 * torch.square((target - mean) / scale)
        return (nll * weight).sum() / torch.clamp(weight.sum(), min=1.0)
