"""Compact, version-specific Dreamer and XQL adapters for KDD263.

These are low-dimensional adaptations for the frozen KDD simulator interfaces.
They are not copies of the official Dreamer or XQL repositories and are not
intended to reproduce the papers' image-domain results.
"""
from __future__ import annotations

import hashlib
import io
import math
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ArrayPolicy = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], np.ndarray
]


def symlog(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.log1p(torch.abs(value))


def symexp(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.expm1(torch.abs(value))


def twohot_targets(
    value: torch.Tensor, low: float, high: float, bins: int
) -> torch.Tensor:
    """Linearly interpolate a scalar target between adjacent fixed bins."""
    if bins < 2 or not low < high:
        raise ValueError("invalid two-hot support")
    clipped = torch.clamp(value, min=float(low), max=float(high))
    position = (clipped - float(low)) * (bins - 1) / (float(high) - float(low))
    lower = torch.floor(position).to(dtype=torch.long)
    upper = torch.clamp(lower + 1, max=bins - 1)
    upper_weight = position - lower.to(dtype=position.dtype)
    lower_weight = 1.0 - upper_weight
    target = torch.zeros((*value.shape, bins), dtype=value.dtype, device=value.device)
    target.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    target.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return target


def twohot_value(
    logits: torch.Tensor, low: float, high: float, bins: int
) -> torch.Tensor:
    support = torch.linspace(
        float(low), float(high), bins, dtype=logits.dtype, device=logits.device
    )
    encoded = torch.sum(torch.softmax(logits, dim=-1) * support, dim=-1)
    return symexp(encoded)


def state_dict_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        local = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(local.shape)).encode())
        digest.update(str(local.dtype).encode())
        digest.update(local.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_bytes(module: nn.Module, metadata: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save({"state_dict": module.state_dict(), "metadata": metadata}, buffer)
    return buffer.getvalue()


def array_sha256(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        local = np.ascontiguousarray(value)
        digest.update(local.dtype.str.encode())
        digest.update(str(local.shape).encode())
        digest.update(local.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class EpisodeDataset:
    features: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    valid: np.ndarray
    behavior_probability: np.ndarray

    def validate(self) -> None:
        if self.features.ndim != 3:
            raise ValueError("features must have episode x time x channel shape")
        episodes, horizon, _ = self.features.shape
        if self.actions.shape != (episodes, horizon):
            raise ValueError("action shape mismatch")
        if self.rewards.shape != (episodes, horizon):
            raise ValueError("reward shape mismatch")
        if self.done.shape != (episodes, horizon):
            raise ValueError("termination shape mismatch")
        if self.valid.shape != (episodes, horizon):
            raise ValueError("valid shape mismatch")
        if self.behavior_probability.shape[:2] != (episodes, horizon):
            raise ValueError("behavior probability shape mismatch")
        if not np.isfinite(self.features[self.valid]).all():
            raise ValueError("nonfinite valid feature")
        if not np.isfinite(self.rewards[self.valid]).all():
            raise ValueError("nonfinite valid reward")
        local_probability = self.behavior_probability[self.valid]
        if (
            not np.isfinite(local_probability).all()
            or np.any(local_probability < 0.0)
            or not np.allclose(local_probability.sum(axis=1), 1.0, atol=1.0e-5)
        ):
            raise ValueError("invalid behavior probabilities")
        if not np.all(self.valid[:, 0]):
            raise ValueError("every generated episode must contain an initial step")

    @property
    def action_count(self) -> int:
        return int(self.behavior_probability.shape[-1])

    @property
    def horizon(self) -> int:
        return int(self.features.shape[1])

    @property
    def support(self) -> np.ndarray:
        return np.any(self.behavior_probability[self.valid] > 1.0e-8, axis=0)


@dataclass(frozen=True, slots=True)
class FeatureAdapter:
    observation_dim: int
    action_count: int
    horizon: int
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls,
        raw_features: np.ndarray,
        valid: np.ndarray,
        observation_dim: int,
        action_count: int,
        horizon: int,
        minimum_scale: float = 1.0e-6,
    ) -> "FeatureAdapter":
        rows = np.asarray(raw_features, dtype=np.float32)[np.asarray(valid, dtype=bool)]
        mean = rows.mean(axis=0).astype(np.float32)
        scale = rows.std(axis=0).astype(np.float32)
        scale = np.where(scale > minimum_scale, scale, 1.0).astype(np.float32)
        return cls(observation_dim, action_count, horizon, mean, scale)

    @property
    def feature_dim(self) -> int:
        return int(len(self.mean))

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        return ((np.asarray(raw, dtype=np.float32) - self.mean) / self.scale).astype(
            np.float32
        )

    def raw_callback_features(
        self,
        observation: np.ndarray,
        mask: np.ndarray,
        recency: np.ndarray,
        previous: np.ndarray,
        step: int,
    ) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        recency = np.asarray(recency, dtype=np.float32)
        previous = np.asarray(previous, dtype=np.int64).copy()
        if step == 0:
            previous.fill(0)
        time = np.full(
            (len(observation), 1),
            float(step) / max(self.horizon - 1, 1),
            dtype=np.float32,
        )
        return np.concatenate(
            [
                observation,
                mask,
                recency / max(self.horizon, 1),
                np.eye(self.action_count, dtype=np.float32)[previous],
                time,
            ],
            axis=1,
        )


def _episode_features(
    observation: np.ndarray,
    mask: np.ndarray,
    recency: np.ndarray,
    actions: np.ndarray,
    action_count: int,
) -> np.ndarray:
    episodes, horizon, _ = observation.shape
    previous = np.zeros_like(actions, dtype=np.int64)
    previous[:, 1:] = np.asarray(actions[:, :-1], dtype=np.int64)
    time = np.broadcast_to(
        np.arange(horizon, dtype=np.float32)[None, :, None] / max(horizon - 1, 1),
        (episodes, horizon, 1),
    )
    return np.concatenate(
        [
            np.asarray(observation, dtype=np.float32),
            np.asarray(mask, dtype=np.float32),
            np.asarray(recency, dtype=np.float32) / max(horizon, 1),
            np.eye(action_count, dtype=np.float32)[previous],
            time,
        ],
        axis=-1,
    ).astype(np.float32)


def episode_dataset_from_controlled(data: Any) -> tuple[EpisodeDataset, int]:
    action_count = int(data.behavior_probability.shape[-1])
    raw = _episode_features(
        data.observed,
        data.masks,
        data.deltas,
        data.actions,
        action_count,
    )
    output = EpisodeDataset(
        raw,
        np.asarray(data.actions, dtype=np.int64),
        np.asarray(data.rewards, dtype=np.float32),
        np.asarray(data.done, dtype=bool),
        np.asarray(data.valid, dtype=bool),
        np.asarray(data.behavior_probability, dtype=np.float32),
    )
    output.validate()
    return output, int(data.observed.shape[-1])


def episode_dataset_from_fitted(batch: Any) -> tuple[EpisodeDataset, int]:
    valid = np.asarray(batch.valid_steps, dtype=bool)
    observation = np.where(
        valid[..., None], np.asarray(batch.observations), 0.0
    ).astype(np.float32)
    mask = np.where(valid[..., None], np.asarray(batch.masks), False)
    recency = np.where(
        valid[..., None], np.asarray(batch.recency), 0.0
    ).astype(np.float32)
    actions = np.where(valid, np.asarray(batch.actions), 0).astype(np.int64)
    reward = np.where(valid, np.asarray(batch.rewards), 0.0).astype(np.float32)
    probability = np.where(
        valid[..., None], np.asarray(batch.action_probabilities), 0.0
    ).astype(np.float32)
    probability[~valid, 0] = 1.0
    action_count = int(probability.shape[-1])
    raw = _episode_features(
        observation, mask, recency, actions, action_count
    )
    output = EpisodeDataset(
        raw,
        actions,
        reward,
        np.asarray(batch.terminations, dtype=bool),
        valid,
        probability,
    )
    output.validate()
    return output, int(observation.shape[-1])


def normalized_dataset(
    dataset: EpisodeDataset, adapter: FeatureAdapter
) -> EpisodeDataset:
    output = replace(dataset, features=adapter.normalize(dataset.features))
    output.validate()
    return output


def _mlp(
    input_dim: int, hidden_dim: int, output_dim: int, layers: int = 2
) -> nn.Sequential:
    modules: list[nn.Module] = []
    current = input_dim
    for _ in range(layers):
        modules.extend([nn.Linear(current, hidden_dim), nn.SiLU()])
        current = hidden_dim
    modules.append(nn.Linear(current, output_dim))
    return nn.Sequential(*modules)


@dataclass(frozen=True, slots=True)
class DreamerSpec:
    version: str
    latent_kind: str
    free_nats: float
    unimix: float = 0.0
    kl_balance: float | None = None
    dynamics_kl_scale: float = 1.0
    representation_kl_scale: float = 0.0
    scalar_heads: bool = True
    minimum_std: float = 0.1
    twohot_bins: int = 51
    twohot_low: float = -1.0
    twohot_high: float = 1.0


def dreamer_spec(version: str, config: dict[str, Any]) -> DreamerSpec:
    if version == "v1":
        local = config["v1"]
        return DreamerSpec(
            version,
            "continuous",
            float(local["free_nats"]),
            scalar_heads=True,
            minimum_std=float(local["minimum_std"]),
        )
    if version == "v2":
        local = config["v2"]
        return DreamerSpec(
            version,
            "categorical",
            float(local["free_nats"]),
            unimix=float(local["unimix"]),
            kl_balance=float(local["kl_balance"]),
            scalar_heads=True,
        )
    if version == "v3":
        local = config["v3"]
        return DreamerSpec(
            version,
            "categorical",
            float(local["free_nats"]),
            unimix=float(local["unimix"]),
            dynamics_kl_scale=float(local["dynamics_kl_scale"]),
            representation_kl_scale=float(local["representation_kl_scale"]),
            scalar_heads=False,
            twohot_bins=int(local["twohot_bins"]),
            twohot_low=float(local["twohot_low"]),
            twohot_high=float(local["twohot_high"]),
        )
    raise ValueError(f"unsupported Dreamer version: {version}")


class CompactDreamer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_count: int,
        support: np.ndarray,
        config: dict[str, Any],
        version: str,
    ) -> None:
        super().__init__()
        self.spec = dreamer_spec(version, config)
        self.feature_dim = int(feature_dim)
        self.action_count = int(action_count)
        self.deter_dim = int(config["deterministic_dim"])
        self.encoder_dim = int(config["encoder_dim"])
        self.groups = int(config["categorical_groups"])
        self.classes = int(config["categorical_classes"])
        if self.spec.latent_kind == "continuous":
            self.stoch_dim = int(config["continuous_stochastic_dim"])
            prior_output = posterior_output = 2 * self.stoch_dim
        else:
            self.stoch_dim = self.groups * self.classes
            prior_output = posterior_output = self.stoch_dim
        hidden = int(config["hidden_dim"])
        self.encoder = _mlp(self.feature_dim, hidden, self.encoder_dim)
        self.recurrent = nn.GRUCell(self.stoch_dim + self.action_count, self.deter_dim)
        self.prior_head = _mlp(self.deter_dim, hidden, prior_output, layers=1)
        self.posterior_head = _mlp(
            self.deter_dim + self.encoder_dim, hidden, posterior_output, layers=1
        )
        latent_feature = self.deter_dim + self.stoch_dim
        self.decoder = _mlp(latent_feature, hidden, self.feature_dim)
        head_output = 1 if self.spec.scalar_heads else self.spec.twohot_bins
        self.reward_head = _mlp(latent_feature, hidden, head_output)
        self.continuation_head = _mlp(latent_feature, hidden, 1)
        self.actor = _mlp(latent_feature, hidden, self.action_count)
        self.critic = _mlp(latent_feature, hidden, head_output)
        local_support = torch.as_tensor(np.asarray(support, dtype=bool))
        if local_support.shape != (self.action_count,) or not local_support.any():
            raise ValueError("invalid static action support")
        self.register_buffer("support", local_support)

    @property
    def version(self) -> str:
        return self.spec.version

    def world_parameters(self) -> list[nn.Parameter]:
        modules = (
            self.encoder,
            self.recurrent,
            self.prior_head,
            self.posterior_head,
            self.decoder,
            self.reward_head,
            self.continuation_head,
        )
        return [parameter for module in modules for parameter in module.parameters()]

    def _categorical_probability(self, logits: torch.Tensor) -> torch.Tensor:
        shape = (*logits.shape[:-1], self.groups, self.classes)
        probability = torch.softmax(logits.reshape(shape), dim=-1)
        if self.spec.unimix:
            probability = (
                1.0 - self.spec.unimix
            ) * probability + self.spec.unimix / self.classes
        return probability

    def _distribution(
        self, raw: torch.Tensor, sample: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.spec.latent_kind == "continuous":
            mean, raw_std = raw.chunk(2, dim=-1)
            std = F.softplus(raw_std) + self.spec.minimum_std
            state = mean + std * torch.randn_like(std) if sample else mean
            return state, {"mean": mean, "std": std}
        probability = self._categorical_probability(raw)
        if sample:
            flat = probability.reshape(-1, self.classes)
            index = torch.multinomial(flat, 1).reshape(*probability.shape[:-1])
        else:
            index = torch.argmax(probability, dim=-1)
        hard = F.one_hot(index, self.classes).to(dtype=probability.dtype)
        state = hard + probability - probability.detach() if sample else hard
        return state.flatten(start_dim=-2), {"probability": probability}

    def prior(
        self, deterministic: torch.Tensor, sample: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self._distribution(self.prior_head(deterministic), sample)

    def posterior(
        self, deterministic: torch.Tensor, encoded: torch.Tensor, sample: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw = self.posterior_head(torch.cat([deterministic, encoded], dim=-1))
        return self._distribution(raw, sample)

    def _base_kl(
        self,
        posterior: dict[str, torch.Tensor],
        prior: dict[str, torch.Tensor],
        detach_posterior: bool,
        detach_prior: bool,
    ) -> torch.Tensor:
        if self.spec.latent_kind == "continuous":
            q_mean = posterior["mean"].detach() if detach_posterior else posterior["mean"]
            q_std = posterior["std"].detach() if detach_posterior else posterior["std"]
            p_mean = prior["mean"].detach() if detach_prior else prior["mean"]
            p_std = prior["std"].detach() if detach_prior else prior["std"]
            value = (
                torch.log(p_std / q_std)
                + (q_std.square() + (q_mean - p_mean).square())
                / (2.0 * p_std.square())
                - 0.5
            )
            return value.sum(dim=-1)
        q = posterior["probability"]
        p = prior["probability"]
        if detach_posterior:
            q = q.detach()
        if detach_prior:
            p = p.detach()
        return torch.sum(
            q * (torch.log(torch.clamp(q, min=1.0e-8)) - torch.log(torch.clamp(p, min=1.0e-8))),
            dim=(-1, -2),
        )

    def kl_loss(
        self,
        posterior: dict[str, torch.Tensor],
        prior: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.spec.version == "v1":
            return torch.clamp(
                self._base_kl(posterior, prior, False, False),
                min=self.spec.free_nats,
            )
        dynamics = torch.clamp(
            self._base_kl(posterior, prior, True, False),
            min=self.spec.free_nats,
        )
        representation = torch.clamp(
            self._base_kl(posterior, prior, False, True),
            min=self.spec.free_nats,
        )
        if self.spec.version == "v2":
            balance = float(self.spec.kl_balance)
            return balance * dynamics + (1.0 - balance) * representation
        return (
            self.spec.dynamics_kl_scale * dynamics
            + self.spec.representation_kl_scale * representation
        )

    def latent_feature(
        self, deterministic: torch.Tensor, stochastic: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([deterministic, stochastic], dim=-1)

    def reward_prediction(self, feature: torch.Tensor) -> torch.Tensor:
        output = self.reward_head(feature)
        if self.spec.scalar_heads:
            return output.squeeze(-1)
        return twohot_value(
            output,
            self.spec.twohot_low,
            self.spec.twohot_high,
            self.spec.twohot_bins,
        )

    def value_prediction(self, feature: torch.Tensor) -> torch.Tensor:
        output = self.critic(feature)
        if self.spec.scalar_heads:
            return output.squeeze(-1)
        return twohot_value(
            output,
            self.spec.twohot_low,
            self.spec.twohot_high,
            self.spec.twohot_bins,
        )

    def scalar_or_twohot_loss(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.spec.scalar_heads:
            return torch.square(output.squeeze(-1) - target)
        encoded = symlog(target)
        target_probability = twohot_targets(
            encoded,
            self.spec.twohot_low,
            self.spec.twohot_high,
            self.spec.twohot_bins,
        )
        return -torch.sum(target_probability * F.log_softmax(output, dim=-1), dim=-1)

    def actor_probability(self, feature: torch.Tensor) -> torch.Tensor:
        logits = self.actor(feature)
        logits = torch.where(
            self.support.to(device=logits.device)[None],
            logits,
            torch.full_like(logits, -1.0e9),
        )
        return torch.softmax(logits, dim=-1)

    def _initial_posterior(
        self, first_feature: torch.Tensor, sample: bool
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch = len(first_feature)
        deterministic = torch.zeros(
            (batch, self.deter_dim), dtype=first_feature.dtype, device=first_feature.device
        )
        _, prior = self.prior(deterministic, sample)
        encoded = self.encoder(first_feature)
        stochastic, posterior = self.posterior(deterministic, encoded, sample)
        return deterministic, stochastic, posterior, prior

    def world_loss(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        done: torch.Tensor,
        valid: torch.Tensor,
        config: dict[str, Any],
        sample: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, horizon, _ = features.shape
        deterministic, stochastic, posterior, prior = self._initial_posterior(
            features[:, 0], sample
        )
        initial_mask = valid[:, 0].to(dtype=features.dtype)
        latent = self.latent_feature(deterministic, stochastic)
        observation_sum = (
            torch.square(self.decoder(latent) - features[:, 0]).mean(dim=-1)
            * initial_mask
        ).sum()
        observation_count = initial_mask.sum()
        kl_sum = (self.kl_loss(posterior, prior) * initial_mask).sum()
        kl_count = initial_mask.sum()
        reward_sum = torch.zeros((), dtype=features.dtype, device=features.device)
        continuation_sum = torch.zeros_like(reward_sum)
        transition_count = torch.zeros_like(reward_sum)
        for step in range(horizon):
            action = F.one_hot(
                actions[:, step], num_classes=self.action_count
            ).to(dtype=features.dtype)
            next_deterministic = self.recurrent(
                torch.cat([stochastic, action], dim=-1), deterministic
            )
            next_prior_state, next_prior = self.prior(next_deterministic, sample)
            next_prior_feature = self.latent_feature(
                next_deterministic, next_prior_state
            )
            local_mask = valid[:, step].to(dtype=features.dtype)
            reward_loss = self.scalar_or_twohot_loss(
                self.reward_head(next_prior_feature), rewards[:, step]
            )
            continuation_target = (~done[:, step]).to(dtype=features.dtype)
            continuation_loss = F.binary_cross_entropy_with_logits(
                self.continuation_head(next_prior_feature).squeeze(-1),
                continuation_target,
                reduction="none",
            )
            reward_sum = reward_sum + (reward_loss * local_mask).sum()
            continuation_sum = continuation_sum + (
                continuation_loss * local_mask
            ).sum()
            transition_count = transition_count + local_mask.sum()
            if step + 1 < horizon:
                next_encoded = self.encoder(features[:, step + 1])
                next_stochastic, next_posterior = self.posterior(
                    next_deterministic, next_encoded, sample
                )
                next_mask = valid[:, step + 1].to(dtype=features.dtype)
                next_feature = self.latent_feature(
                    next_deterministic, next_stochastic
                )
                observation_sum = observation_sum + (
                    torch.square(
                        self.decoder(next_feature) - features[:, step + 1]
                    ).mean(dim=-1)
                    * next_mask
                ).sum()
                observation_count = observation_count + next_mask.sum()
                kl_sum = kl_sum + (
                    self.kl_loss(next_posterior, next_prior) * next_mask
                ).sum()
                kl_count = kl_count + next_mask.sum()
                stochastic = next_stochastic
            else:
                stochastic = next_prior_state
            deterministic = next_deterministic
        observation_loss = observation_sum / torch.clamp(observation_count, min=1.0)
        reward_loss = reward_sum / torch.clamp(transition_count, min=1.0)
        continuation_loss = continuation_sum / torch.clamp(
            transition_count, min=1.0
        )
        kl_loss = kl_sum / torch.clamp(kl_count, min=1.0)
        total = (
            float(config["observation_loss_scale"]) * observation_loss
            + float(config["reward_loss_scale"]) * reward_loss
            + float(config["continuation_loss_scale"]) * continuation_loss
            + float(config["kl_loss_scale"]) * kl_loss
        )
        metrics = {
            "world_total_loss": total,
            "observation_loss": observation_loss,
            "reward_loss": reward_loss,
            "continuation_loss": continuation_loss,
            "kl_loss": kl_loss,
        }
        return total, metrics

    def posterior_states(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        deterministic, stochastic, _, _ = self._initial_posterior(
            features[:, 0], sample=False
        )
        deterministic_rows = [deterministic]
        stochastic_rows = [stochastic]
        for step in range(1, features.shape[1]):
            previous_action = F.one_hot(
                actions[:, step - 1], num_classes=self.action_count
            ).to(dtype=features.dtype)
            deterministic = self.recurrent(
                torch.cat([stochastic, previous_action], dim=-1), deterministic
            )
            encoded = self.encoder(features[:, step])
            stochastic, _ = self.posterior(deterministic, encoded, sample=False)
            deterministic_rows.append(deterministic)
            stochastic_rows.append(stochastic)
        return torch.stack(deterministic_rows, dim=1), torch.stack(
            stochastic_rows, dim=1
        )

    def imagined_rollout(
        self,
        deterministic: torch.Tensor,
        stochastic: torch.Tensor,
        horizon: int,
        discount: float,
        lambda_: float,
        entropy_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        discounts: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for _ in range(horizon):
            current = self.latent_feature(deterministic, stochastic)
            probability = self.actor_probability(current)
            index = torch.multinomial(probability, 1).squeeze(-1)
            hard = F.one_hot(index, self.action_count).to(dtype=probability.dtype)
            action = hard + probability - probability.detach()
            deterministic = self.recurrent(
                torch.cat([stochastic, action], dim=-1), deterministic
            )
            stochastic, _ = self.prior(deterministic, sample=True)
            next_feature = self.latent_feature(deterministic, stochastic)
            features.append(next_feature)
            rewards.append(self.reward_prediction(next_feature))
            continuation = torch.sigmoid(
                self.continuation_head(next_feature).squeeze(-1)
            )
            discounts.append(float(discount) * continuation)
            entropies.append(
                -torch.sum(
                    probability * torch.log(torch.clamp(probability, min=1.0e-8)),
                    dim=-1,
                )
            )
        feature_tensor = torch.stack(features, dim=1)
        reward_tensor = torch.stack(rewards, dim=1)
        discount_tensor = torch.stack(discounts, dim=1)
        entropy_tensor = torch.stack(entropies, dim=1)
        next_values = self.value_prediction(feature_tensor)
        bootstrap = next_values[:, -1]
        returns: list[torch.Tensor] = []
        running = bootstrap
        for step in range(horizon - 1, -1, -1):
            bootstrap_value = next_values[:, step]
            running = reward_tensor[:, step] + discount_tensor[:, step] * (
                (1.0 - float(lambda_)) * bootstrap_value
                + float(lambda_) * running
            )
            returns.append(running)
        return_tensor = torch.stack(list(reversed(returns)), dim=1)
        if self.spec.version == "v3":
            low = torch.quantile(return_tensor.detach(), 0.05)
            high = torch.quantile(return_tensor.detach(), 0.95)
            scale = torch.clamp(high - low, min=1.0)
            actor_objective = return_tensor / scale
        else:
            actor_objective = return_tensor
        actor_loss = -torch.mean(actor_objective + entropy_scale * entropy_tensor)
        return actor_loss, feature_tensor, return_tensor, entropy_tensor


def _tensor_dataset(
    dataset: EpisodeDataset, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(dataset.features).to(device),
        torch.from_numpy(dataset.actions).to(device),
        torch.from_numpy(dataset.rewards).to(device),
        torch.from_numpy(dataset.done).to(device),
        torch.from_numpy(dataset.valid).to(device),
    )


def _metric_float(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in metrics.items()}


def _set_requires_grad(parameters: list[nn.Parameter], enabled: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad_(enabled)


def train_compact_dreamer(
    train: EpisodeDataset,
    validation: EpisodeDataset,
    config: dict[str, Any],
    version: str,
    seed: int,
    world_epochs: int,
    actor_epochs: int,
    episode_batch_size: int,
    discount: float,
    device: torch.device,
) -> tuple[CompactDreamer, dict[str, Any]]:
    train.validate()
    validation.validate()
    if train.action_count != validation.action_count:
        raise ValueError("train/validation action-count drift")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    model = CompactDreamer(
        train.features.shape[-1],
        train.action_count,
        train.support,
        config,
        version,
    ).to(device)
    world_optimizer = torch.optim.AdamW(
        model.world_parameters(), lr=float(config["world_learning_rate"])
    )
    train_tensors = _tensor_dataset(train, device)
    validation_tensors = _tensor_dataset(validation, device)
    generator = np.random.default_rng(seed + 101)
    world_trace: list[dict[str, float]] = []
    for _epoch in range(int(world_epochs)):
        order = generator.permutation(len(train.features))
        epoch_metrics: list[dict[str, float]] = []
        for start in range(0, len(order), int(episode_batch_size)):
            index = torch.as_tensor(
                order[start : start + int(episode_batch_size)],
                dtype=torch.long,
                device=device,
            )
            batch = tuple(value[index] for value in train_tensors)
            world_optimizer.zero_grad(set_to_none=True)
            loss, metrics = model.world_loss(*batch, config, sample=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.world_parameters(), float(config["gradient_clip_norm"])
            )
            world_optimizer.step()
            epoch_metrics.append(_metric_float(metrics))
        world_trace.append(
            {
                key: float(np.mean([row[key] for row in epoch_metrics]))
                for key in epoch_metrics[0]
            }
        )
    model.eval()
    with torch.inference_mode():
        _, validation_metrics_t = model.world_loss(
            *validation_tensors, config, sample=False
        )
    validation_metrics = _metric_float(validation_metrics_t)

    world_parameters = model.world_parameters()
    actor_parameters = list(model.actor.parameters())
    critic_parameters = list(model.critic.parameters())
    actor_optimizer = torch.optim.AdamW(
        actor_parameters, lr=float(config["actor_learning_rate"])
    )
    critic_optimizer = torch.optim.AdamW(
        critic_parameters, lr=float(config["critic_learning_rate"])
    )
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    entropies: list[float] = []
    features, actions, _rewards, _done, valid = train_tensors
    for _epoch in range(int(actor_epochs)):
        _set_requires_grad(world_parameters, False)
        _set_requires_grad(actor_parameters, True)
        _set_requires_grad(critic_parameters, False)
        with torch.no_grad():
            deterministic, stochastic = model.posterior_states(features, actions)
            flat_valid = valid.reshape(-1)
            starts_d = deterministic.reshape(-1, model.deter_dim)[flat_valid]
            starts_s = stochastic.reshape(-1, model.stoch_dim)[flat_valid]
            count = min(512, len(starts_d))
            chosen = torch.as_tensor(
                generator.choice(len(starts_d), size=count, replace=False),
                dtype=torch.long,
                device=device,
            )
            starts_d = starts_d[chosen]
            starts_s = starts_s[chosen]
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss, imagined_features, imagined_returns, entropy = (
            model.imagined_rollout(
                starts_d,
                starts_s,
                int(config["imagination_horizon"]),
                float(discount),
                float(config["lambda"]),
                float(config["entropy_scale"]),
            )
        )
        critic_feature = imagined_features.detach()
        critic_target = imagined_returns.detach()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            actor_parameters, float(config["gradient_clip_norm"])
        )
        actor_optimizer.step()

        _set_requires_grad(actor_parameters, False)
        _set_requires_grad(critic_parameters, True)
        critic_optimizer.zero_grad(set_to_none=True)
        critic_output = model.critic(critic_feature)
        critic_loss = model.scalar_or_twohot_loss(
            critic_output, critic_target
        ).mean()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            critic_parameters, float(config["gradient_clip_norm"])
        )
        critic_optimizer.step()
        actor_losses.append(float(actor_loss.detach().cpu()))
        critic_losses.append(float(critic_loss.detach().cpu()))
        entropies.append(float(entropy.mean().detach().cpu()))
    _set_requires_grad(world_parameters, True)
    _set_requires_grad(actor_parameters, True)
    _set_requires_grad(critic_parameters, True)
    model.eval()
    metadata: dict[str, Any] = {
        "method": f"dreamer_{version}_compact",
        "version": version,
        "latent_kind": model.spec.latent_kind,
        "world_epochs": int(world_epochs),
        "actor_epochs": int(actor_epochs),
        "training_seed": int(seed),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "model_state_sha256": state_dict_sha256(model),
        "runtime_seconds": time.perf_counter() - started,
        "final_train_world_loss": world_trace[-1]["world_total_loss"],
        "final_train_observation_loss": world_trace[-1]["observation_loss"],
        "final_train_reward_loss": world_trace[-1]["reward_loss"],
        "final_train_continuation_loss": world_trace[-1]["continuation_loss"],
        "final_train_kl_loss": world_trace[-1]["kl_loss"],
        "validation_world_loss": validation_metrics["world_total_loss"],
        "validation_observation_loss": validation_metrics["observation_loss"],
        "validation_reward_loss": validation_metrics["reward_loss"],
        "validation_continuation_loss": validation_metrics["continuation_loss"],
        "validation_kl_loss": validation_metrics["kl_loss"],
        "final_actor_loss": actor_losses[-1],
        "final_critic_loss": critic_losses[-1],
        "final_imagination_entropy": entropies[-1],
        "convergence_status": "fixed_pilot_budget_complete",
    }
    return model, metadata


class DreamerPolicyAdapter:
    def __init__(
        self,
        model: CompactDreamer,
        feature_adapter: FeatureAdapter,
        device: torch.device,
        evaluation_mode: str = "sample",
    ) -> None:
        if evaluation_mode not in {"sample", "mode"}:
            raise ValueError("Dreamer evaluation mode must be sample or mode")
        self.model = model
        self.adapter = feature_adapter
        self.device = device
        self.evaluation_mode = evaluation_mode
        self.deterministic: torch.Tensor | None = None
        self.stochastic: torch.Tensor | None = None

    def __call__(
        self,
        observation: np.ndarray,
        mask: np.ndarray,
        recency: np.ndarray,
        previous: np.ndarray,
        step: int,
    ) -> np.ndarray:
        raw = self.adapter.raw_callback_features(
            observation, mask, recency, previous, step
        )
        x = torch.from_numpy(self.adapter.normalize(raw)).to(self.device)
        batch = len(x)
        with torch.inference_mode():
            if (
                step == 0
                or self.deterministic is None
                or len(self.deterministic) != batch
            ):
                deterministic = torch.zeros(
                    (batch, self.model.deter_dim),
                    dtype=x.dtype,
                    device=self.device,
                )
            else:
                previous_tensor = torch.as_tensor(
                    previous, dtype=torch.long, device=self.device
                )
                action = F.one_hot(
                    previous_tensor, num_classes=self.model.action_count
                ).to(dtype=x.dtype)
                deterministic = self.model.recurrent(
                    torch.cat([self.stochastic, action], dim=-1),
                    self.deterministic,
                )
            encoded = self.model.encoder(x)
            stochastic, _ = self.model.posterior(
                deterministic, encoded, sample=False
            )
            probability = self.model.actor_probability(
                self.model.latent_feature(deterministic, stochastic)
            )
            if self.evaluation_mode == "mode":
                chosen = torch.argmax(probability, dim=-1)
                probability = F.one_hot(
                    chosen, num_classes=self.model.action_count
                ).to(dtype=probability.dtype)
            self.deterministic = deterministic
            self.stochastic = stochastic
        output = probability.cpu().numpy().astype(np.float64)
        output /= output.sum(axis=1, keepdims=True)
        return output


def dreamer_policy_probability(
    model: CompactDreamer,
    dataset: EpisodeDataset,
    device: torch.device,
    evaluation_mode: str = "sample",
) -> np.ndarray:
    if evaluation_mode not in {"sample", "mode"}:
        raise ValueError("Dreamer evaluation mode must be sample or mode")
    features, actions, _rewards, _done, _valid = _tensor_dataset(dataset, device)
    with torch.inference_mode():
        deterministic, stochastic = model.posterior_states(features, actions)
        latent = model.latent_feature(deterministic, stochastic)
        probability = model.actor_probability(latent)
        if evaluation_mode == "mode":
            probability = F.one_hot(
                torch.argmax(probability, dim=-1),
                num_classes=model.action_count,
            ).to(dtype=probability.dtype)
    return probability.cpu().numpy()


class DiscreteXQL(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_count: int,
        support: np.ndarray,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.action_count = int(action_count)
        self.q1 = _mlp(feature_dim, hidden_dim, action_count)
        self.q2 = _mlp(feature_dim, hidden_dim, action_count)
        self.target_q1 = _mlp(feature_dim, hidden_dim, action_count)
        self.target_q2 = _mlp(feature_dim, hidden_dim, action_count)
        self.value = _mlp(feature_dim, hidden_dim, 1)
        self.policy = _mlp(feature_dim, hidden_dim, action_count)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        for parameter in self.target_q1.parameters():
            parameter.requires_grad_(False)
        for parameter in self.target_q2.parameters():
            parameter.requires_grad_(False)
        local_support = torch.as_tensor(np.asarray(support, dtype=bool))
        if local_support.shape != (action_count,) or not local_support.any():
            raise ValueError("invalid XQL action support")
        self.register_buffer("support", local_support)

    def policy_probability(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.policy(features)
        logits = torch.where(
            self.support.to(device=logits.device)[None],
            logits,
            torch.full_like(logits, -1.0e9),
        )
        return torch.softmax(logits, dim=-1)


def _flat_transitions(dataset: EpisodeDataset) -> tuple[np.ndarray, ...]:
    features = dataset.features
    next_features = features.copy()
    next_features[:, :-1] = features[:, 1:]
    keep = dataset.valid
    return (
        features[keep].astype(np.float32),
        dataset.actions[keep].astype(np.int64),
        dataset.rewards[keep].astype(np.float32),
        next_features[keep].astype(np.float32),
        dataset.done[keep].astype(np.float32),
    )


def train_discrete_xql(
    train: EpisodeDataset,
    validation: EpisodeDataset,
    config: dict[str, Any],
    seed: int,
    epochs: int,
    discount: float,
    device: torch.device,
) -> tuple[DiscreteXQL, dict[str, Any]]:
    train.validate()
    validation.validate()
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.perf_counter()
    model = DiscreteXQL(
        train.features.shape[-1],
        train.action_count,
        train.support,
        int(config["hidden_dim"]),
    ).to(device)
    trainable = [
        *model.q1.parameters(),
        *model.q2.parameters(),
        *model.value.parameters(),
        *model.policy.parameters(),
    ]
    optimizer = torch.optim.AdamW(trainable, lr=float(config["learning_rate"]))
    x, action, reward, next_x, done = _flat_transitions(train)
    tensors = tuple(
        torch.from_numpy(value).to(device)
        for value in (x, action, reward, next_x, done)
    )
    generator = np.random.default_rng(seed + 211)
    batch_size = min(256, len(x))
    trace: list[dict[str, float]] = []
    for _epoch in range(int(epochs)):
        order = generator.permutation(len(x))
        local: list[dict[str, float]] = []
        for start in range(0, len(order), batch_size):
            index = torch.as_tensor(
                order[start : start + batch_size],
                dtype=torch.long,
                device=device,
            )
            bx, ba, br, bnext, bdone = (value[index] for value in tensors)
            with torch.no_grad():
                target = br + float(discount) * (1.0 - bdone) * model.value(
                    bnext
                ).squeeze(-1)
                target_q = torch.minimum(
                    model.target_q1(bx), model.target_q2(bx)
                ).gather(1, ba[:, None]).squeeze(1)
            q1 = model.q1(bx).gather(1, ba[:, None]).squeeze(1)
            q2 = model.q2(bx).gather(1, ba[:, None]).squeeze(1)
            q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            value = model.value(bx).squeeze(-1)
            normalized = torch.clamp(
                (target_q - value) / float(config["beta"]), min=-10.0, max=10.0
            )
            value_loss = torch.mean(torch.exp(normalized) - normalized - 1.0)
            with torch.no_grad():
                advantage = target_q - value
                weight = torch.clamp(
                    torch.exp(
                        advantage / float(config["advantage_temperature"])
                    ),
                    max=float(config["maximum_weight"]),
                )
            probability = model.policy_probability(bx)
            policy_loss = -torch.mean(
                weight
                * torch.log(
                    torch.clamp(
                        probability.gather(1, ba[:, None]).squeeze(1),
                        min=1.0e-8,
                    )
                )
            )
            loss = q_loss + value_loss + policy_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable, float(config["gradient_clip_norm"])
            )
            optimizer.step()
            tau = float(config["target_update_tau"])
            with torch.no_grad():
                for target_parameter, parameter in zip(
                    model.target_q1.parameters(), model.q1.parameters()
                ):
                    target_parameter.lerp_(parameter, tau)
                for target_parameter, parameter in zip(
                    model.target_q2.parameters(), model.q2.parameters()
                ):
                    target_parameter.lerp_(parameter, tau)
            local.append(
                {
                    "q_loss": float(q_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                }
            )
        trace.append(
            {
                key: float(np.mean([row[key] for row in local]))
                for key in local[0]
            }
        )
    validation_x, validation_action, validation_reward, validation_next, validation_done = (
        _flat_transitions(validation)
    )
    with torch.inference_mode():
        vx = torch.from_numpy(validation_x).to(device)
        va = torch.from_numpy(validation_action).to(device)
        vr = torch.from_numpy(validation_reward).to(device)
        vn = torch.from_numpy(validation_next).to(device)
        vd = torch.from_numpy(validation_done).to(device)
        prediction = torch.minimum(model.q1(vx), model.q2(vx)).gather(
            1, va[:, None]
        ).squeeze(1)
        target = vr + float(discount) * (1.0 - vd) * model.value(vn).squeeze(-1)
        validation_bellman = F.mse_loss(prediction, target)
        validation_nll = -torch.log(
            torch.clamp(
                model.policy_probability(vx).gather(1, va[:, None]).squeeze(1),
                min=1.0e-8,
            )
        ).mean()
    model.eval()
    metadata = {
        "method": "discrete_xql",
        "training_seed": int(seed),
        "epochs": int(epochs),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "model_state_sha256": state_dict_sha256(model),
        "runtime_seconds": time.perf_counter() - started,
        "final_q_loss": trace[-1]["q_loss"],
        "final_value_loss": trace[-1]["value_loss"],
        "final_policy_loss": trace[-1]["policy_loss"],
        "validation_bellman_loss": float(validation_bellman.cpu()),
        "validation_policy_nll": float(validation_nll.cpu()),
        "convergence_status": "fixed_pilot_budget_complete",
    }
    return model, metadata


class XQLPolicyAdapter:
    def __init__(
        self,
        model: DiscreteXQL,
        feature_adapter: FeatureAdapter,
        device: torch.device,
    ) -> None:
        self.model = model
        self.adapter = feature_adapter
        self.device = device

    def __call__(
        self,
        observation: np.ndarray,
        mask: np.ndarray,
        recency: np.ndarray,
        previous: np.ndarray,
        step: int,
    ) -> np.ndarray:
        raw = self.adapter.raw_callback_features(
            observation, mask, recency, previous, step
        )
        x = torch.from_numpy(self.adapter.normalize(raw)).to(self.device)
        with torch.inference_mode():
            probability = self.model.policy_probability(x)
        output = probability.cpu().numpy().astype(np.float64)
        output /= output.sum(axis=1, keepdims=True)
        return output


def xql_policy_probability(
    model: DiscreteXQL, dataset: EpisodeDataset, device: torch.device
) -> np.ndarray:
    x = torch.from_numpy(dataset.features).to(device)
    with torch.inference_mode():
        probability = model.policy_probability(x)
    return probability.cpu().numpy()


def probability_diagnostics(
    probability: np.ndarray,
    dataset: EpisodeDataset,
) -> dict[str, Any]:
    local = np.asarray(probability)[dataset.valid]
    support = dataset.support
    outside = ~support
    unsupported_mass = float(local[:, outside].sum() / max(len(local), 1))
    entropy = -np.sum(local * np.log(np.clip(local, 1.0e-12, 1.0)), axis=1)
    denominator = math.log(max(int(support.sum()), 2))
    return {
        "probability_sha256": array_sha256(local.astype(np.float32)),
        "probability_finite": bool(np.isfinite(local).all()),
        "maximum_probability_sum_error": float(
            np.max(np.abs(local.sum(axis=1) - 1.0))
        ),
        "unsupported_mass": unsupported_mass,
        "mean_normalized_entropy": float(np.mean(entropy) / denominator),
        "greedy_distinct_actions": int(np.unique(np.argmax(local, axis=1)).size),
    }
