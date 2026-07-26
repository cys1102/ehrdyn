from __future__ import annotations

import torch
from torch import nn

from .kdd069_model_types import GaussianResidualHead, TransitionOutput


class GRUSequenceTransition(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(feature_dim * 2 + action_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = GaussianResidualHead(hidden_dim, feature_dim)

    def forward(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        deltas: torch.Tensor,
        actions: torch.Tensor,
    ) -> TransitionOutput:
        del deltas
        embedded = torch.nn.functional.gelu(self.input(torch.cat([values, masks, actions], dim=-1)))
        hidden, _state = self.gru(embedded)
        return self.head(hidden, values)


class GRUDTransition(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.feature_decay = nn.Linear(feature_dim, feature_dim)
        self.hidden_decay = nn.Linear(feature_dim, hidden_dim)
        self.cell = nn.GRUCell(feature_dim * 2 + action_dim, hidden_dim)
        self.head = GaussianResidualHead(hidden_dim, feature_dim)

    def forward(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        deltas: torch.Tensor,
        actions: torch.Tensor,
    ) -> TransitionOutput:
        hidden = values.new_zeros((values.shape[0], self.cell.hidden_size))
        states: list[torch.Tensor] = []
        for step in range(values.shape[1]):
            feature_decay = torch.exp(-torch.relu(self.feature_decay(deltas[:, step])))
            hidden_decay = torch.exp(-torch.relu(self.hidden_decay(deltas[:, step])))
            decayed = masks[:, step] * values[:, step] + (1.0 - masks[:, step]) * feature_decay * values[:, step]
            hidden = self.cell(torch.cat([decayed, masks[:, step], actions[:, step]], dim=-1), hidden * hidden_decay)
            states.append(hidden)
        return self.head(torch.stack(states, dim=1), values)


class CausalTransformerTransition(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int, heads: int = 4) -> None:
        super().__init__()
        self.input = nn.Linear(feature_dim * 3 + action_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.05,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.position = nn.Parameter(torch.zeros(1, 18, hidden_dim))
        self.head = GaussianResidualHead(hidden_dim, feature_dim)

    def forward(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        deltas: torch.Tensor,
        actions: torch.Tensor,
    ) -> TransitionOutput:
        steps = values.shape[1]
        embedded = self.input(torch.cat([values, masks, deltas, actions], dim=-1)) + self.position[:, :steps]
        causal = torch.triu(torch.ones((steps, steps), dtype=torch.bool, device=values.device), diagonal=1)
        hidden = self.encoder(embedded, mask=causal)
        return self.head(hidden, values)


class SepsisAgentStyleTransition(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        static_dim = min(5, feature_dim)
        self.static_dim = static_dim
        self.static_embed = nn.Linear(static_dim, 16)
        self.action_embed = nn.Linear(action_dim, 16)
        self.gru = nn.GRU(
            feature_dim * 2 + 32,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.pre = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2))
        self.head = GaussianResidualHead(hidden_dim, feature_dim)

    def forward(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        deltas: torch.Tensor,
        actions: torch.Tensor,
    ) -> TransitionOutput:
        del deltas
        steps = values.shape[1]
        static = self.static_embed(values[:, :1, : self.static_dim]).expand(-1, steps, -1)
        action = self.action_embed(actions)
        hidden, _state = self.gru(torch.cat([values, masks, static, action], dim=-1))
        return self.head(self.pre(hidden), values)
