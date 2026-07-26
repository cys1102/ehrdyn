from __future__ import annotations

import torch
from torch import nn

from .kdd069_model_types import GaussianResidualHead, TransitionOutput


class DreamerV1GaussianRSSM(nn.Module):
    latent_family = "gaussian"

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(feature_dim * 3, hidden_dim), nn.SiLU())
        self.recurrent = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        self.prior = nn.Linear(hidden_dim, latent_dim * 2)
        self.posterior = nn.Linear(hidden_dim * 2, latent_dim * 2)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.SiLU())
        self.head = GaussianResidualHead(hidden_dim, feature_dim)

    def forward(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        deltas: torch.Tensor,
        actions: torch.Tensor,
    ) -> TransitionOutput:
        encoded = self.encoder(torch.cat([values, masks, deltas], dim=-1))
        hidden = values.new_zeros((values.shape[0], self.recurrent.hidden_size))
        means: list[torch.Tensor] = []
        log_scales: list[torch.Tensor] = []
        divergences: list[torch.Tensor] = []
        for step in range(values.shape[1]):
            post_mean, post_log_std = _gaussian_stats(self.posterior(torch.cat([hidden, encoded[:, step]], dim=-1)))
            latent = _gaussian_sample(post_mean, post_log_std, self.training)
            hidden = self.recurrent(torch.cat([latent, actions[:, step]], dim=-1), hidden)
            prior_mean, prior_log_std = _gaussian_stats(self.prior(hidden))
            decoded = self.head(self.decoder(torch.cat([hidden, prior_mean], dim=-1)), values[:, step])
            means.append(decoded.mean)
            log_scales.append(decoded.log_scale)
            if step + 1 < values.shape[1]:
                next_mean, next_log_std = _gaussian_stats(
                    self.posterior(torch.cat([hidden, encoded[:, step + 1]], dim=-1))
                )
                divergences.append(_gaussian_kl(next_mean, next_log_std, prior_mean, prior_log_std))
        auxiliary = torch.stack(divergences).mean() if divergences else values.new_zeros(())
        return TransitionOutput(torch.stack(means, dim=1), torch.stack(log_scales, dim=1), auxiliary)

    def rollout(
        self,
        initial_values: torch.Tensor,
        initial_masks: torch.Tensor,
        initial_deltas: torch.Tensor,
        actions: torch.Tensor,
    ) -> TransitionOutput:
        encoded = self.encoder(torch.cat([initial_values, initial_masks, initial_deltas], dim=-1))
        hidden = initial_values.new_zeros((initial_values.shape[0], self.recurrent.hidden_size))
        post_mean, _post_log_std = _gaussian_stats(self.posterior(torch.cat([hidden, encoded], dim=-1)))
        latent = post_mean
        current = initial_values
        means: list[torch.Tensor] = []
        log_scales: list[torch.Tensor] = []
        for step in range(actions.shape[1]):
            hidden = self.recurrent(torch.cat([latent, actions[:, step]], dim=-1), hidden)
            prior_mean, _prior_log_std = _gaussian_stats(self.prior(hidden))
            decoded = self.head(self.decoder(torch.cat([hidden, prior_mean], dim=-1)), current)
            current = decoded.mean
            latent = prior_mean
            means.append(current)
            log_scales.append(decoded.log_scale)
        return TransitionOutput(torch.stack(means, dim=1), torch.stack(log_scales, dim=1), current.new_zeros(()))


class DreamerV3CategoricalRSSM(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int,
        groups: int,
        categories: int,
    ) -> None:
        super().__init__()
        self.groups = groups
        self.categories = categories
        latent_dim = groups * categories
        self.latent_family = f"categorical_{groups}x{categories}"
        self.encoder = nn.Sequential(nn.Linear(feature_dim * 3, hidden_dim), nn.GELU())
        self.recurrent = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        self.prior_logits = nn.Linear(hidden_dim, latent_dim)
        self.posterior_logits = nn.Linear(hidden_dim * 2, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.GELU())
        self.head = GaussianResidualHead(hidden_dim, feature_dim)

    def forward(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        deltas: torch.Tensor,
        actions: torch.Tensor,
    ) -> TransitionOutput:
        encoded = self.encoder(torch.cat([values, masks, deltas], dim=-1))
        hidden = values.new_zeros((values.shape[0], self.recurrent.hidden_size))
        means: list[torch.Tensor] = []
        log_scales: list[torch.Tensor] = []
        divergences: list[torch.Tensor] = []
        for step in range(values.shape[1]):
            posterior = self.posterior_logits(torch.cat([hidden, encoded[:, step]], dim=-1))
            latent = self._categorical_latent(posterior)
            hidden = self.recurrent(torch.cat([latent, actions[:, step]], dim=-1), hidden)
            prior = self.prior_logits(hidden)
            prior_prob = self._probabilities(prior).flatten(start_dim=1)
            decoded = self.head(self.decoder(torch.cat([hidden, prior_prob], dim=-1)), values[:, step])
            means.append(decoded.mean)
            log_scales.append(decoded.log_scale)
            if step + 1 < values.shape[1]:
                next_posterior = self.posterior_logits(torch.cat([hidden, encoded[:, step + 1]], dim=-1))
                divergences.append(self._categorical_kl(next_posterior, prior))
        auxiliary = torch.stack(divergences).mean() if divergences else values.new_zeros(())
        return TransitionOutput(torch.stack(means, dim=1), torch.stack(log_scales, dim=1), auxiliary)

    def rollout(
        self,
        initial_values: torch.Tensor,
        initial_masks: torch.Tensor,
        initial_deltas: torch.Tensor,
        actions: torch.Tensor,
    ) -> TransitionOutput:
        encoded = self.encoder(torch.cat([initial_values, initial_masks, initial_deltas], dim=-1))
        hidden = initial_values.new_zeros((initial_values.shape[0], self.recurrent.hidden_size))
        posterior = self.posterior_logits(torch.cat([hidden, encoded], dim=-1))
        latent = self._probabilities(posterior).flatten(start_dim=1)
        current = initial_values
        means: list[torch.Tensor] = []
        log_scales: list[torch.Tensor] = []
        for step in range(actions.shape[1]):
            hidden = self.recurrent(torch.cat([latent, actions[:, step]], dim=-1), hidden)
            latent = self._probabilities(self.prior_logits(hidden)).flatten(start_dim=1)
            decoded = self.head(self.decoder(torch.cat([hidden, latent], dim=-1)), current)
            current = decoded.mean
            means.append(current)
            log_scales.append(decoded.log_scale)
        return TransitionOutput(torch.stack(means, dim=1), torch.stack(log_scales, dim=1), current.new_zeros(()))

    def _probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits.view(-1, self.groups, self.categories), dim=-1)

    def _categorical_latent(self, logits: torch.Tensor) -> torch.Tensor:
        shaped = logits.view(-1, self.groups, self.categories)
        if self.training:
            return torch.nn.functional.gumbel_softmax(shaped, tau=1.0, hard=True, dim=-1).flatten(start_dim=1)
        return torch.softmax(shaped, dim=-1).flatten(start_dim=1)

    def _categorical_kl(self, posterior: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        post = self._probabilities(posterior)
        prior_prob = self._probabilities(prior)
        return torch.sum(post * (torch.log(post + 1.0e-8) - torch.log(prior_prob + 1.0e-8)), dim=-1).mean()


def _gaussian_stats(stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean, raw = torch.chunk(stats, 2, dim=-1)
    return mean, torch.clamp(raw, min=-5.0, max=2.0)


def _gaussian_sample(mean: torch.Tensor, log_std: torch.Tensor, sample: bool) -> torch.Tensor:
    return mean + torch.randn_like(mean) * torch.exp(log_std) if sample else mean


def _gaussian_kl(
    post_mean: torch.Tensor,
    post_log_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_log_std: torch.Tensor,
) -> torch.Tensor:
    variance_ratio = torch.exp(2.0 * (post_log_std - prior_log_std))
    mean_term = torch.square(post_mean - prior_mean) * torch.exp(-2.0 * prior_log_std)
    return 0.5 * torch.mean(variance_ratio + mean_term - 1.0 + 2.0 * (prior_log_std - post_log_std))
