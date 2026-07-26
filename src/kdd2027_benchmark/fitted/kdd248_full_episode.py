"""Typed full-episode interface for learned source-simulator capability tests.

This module provides code capability only.  The component builders initialize
unfitted modules for synthetic-fixture validation; they do not load clinical
data, fitted weights, or policy-value evidence.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Literal

import numpy as np
import torch
from torch import nn

from .kdd069_model_types import TransitionOutput
from .kdd069_rssm_models import DreamerV3CategoricalRSSM
from .kdd069_sequence_models import GRUDTransition


EPISODE_SCHEMA_VERSION = "kdd248_full_episode_v1"
ModelFamily = Literal["gaussian_recurrent", "categorical_rssm"]
OnlinePolicy = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], np.ndarray
]


class ResponseRegime(str, Enum):
    """Noncausal action-conditioning regimes available to later experiments."""

    NULL = "null_response"
    OBSERVED = "observed_association"
    BOUNDED = "bounded_synthetic_sensitivity"
    PERTURBED = "shuffled_or_sign_perturbed_control"


@dataclass(frozen=True, slots=True)
class TaskShape:
    """Shape-only task contract; it contains no cohort or clinical values."""

    name: str
    feature_dim: int
    action_dim: int
    horizon: int
    schema_version: str = EPISODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EPISODE_SCHEMA_VERSION:
            raise ValueError(f"unsupported episode schema: {self.schema_version}")
        if not self.name or self.feature_dim <= 0 or self.action_dim <= 1 or self.horizon <= 0:
            raise ValueError("invalid task shape")


@dataclass(frozen=True, slots=True)
class InitialHistory:
    """Observed history supplied to, or sampled by, a source simulator."""

    values: torch.Tensor
    masks: torch.Tensor
    recency: torch.Tensor
    previous_actions: torch.Tensor

    def validate(self, task: TaskShape, episodes: int) -> None:
        if self.values.ndim != 3 or self.values.shape[0] != episodes:
            raise ValueError("initial values must have shape episode x history x feature")
        expected = self.values.shape
        if expected[1] <= 0 or expected[2] != task.feature_dim:
            raise ValueError("initial-history feature shape mismatch")
        if self.masks.shape != expected or self.recency.shape != expected:
            raise ValueError("initial values, masks, and recency must align")
        if self.previous_actions.shape != expected[:2]:
            raise ValueError("initial previous-action shape mismatch")
        if not torch.isfinite(self.values).all() or not torch.isfinite(self.recency).all():
            raise ValueError("initial history must be finite")
        if torch.any(self.recency < 0):
            raise ValueError("initial recency must be nonnegative")
        if torch.any(self.previous_actions < 0) or torch.any(
            self.previous_actions >= task.action_dim
        ):
            raise ValueError("initial action outside task action space")


@dataclass(frozen=True, slots=True)
class EpisodeUncertainty:
    """Aligned component uncertainty retained for later aggregate audits."""

    observation_scale: np.ndarray
    mask_probability: np.ndarray
    behavior_entropy: np.ndarray
    reward_scale: np.ndarray
    termination_probability: np.ndarray


@dataclass(frozen=True, slots=True)
class FullEpisodeBatch:
    """Complete stochastic episode arrays with explicit post-termination invalidity."""

    schema_version: str
    task_name: str
    response_regime: str
    observations: np.ndarray
    masks: np.ndarray
    recency: np.ndarray
    actions: np.ndarray
    action_probabilities: np.ndarray
    rewards: np.ndarray
    terminations: np.ndarray
    valid_steps: np.ndarray
    lengths: np.ndarray
    uncertainty: EpisodeUncertainty
    restricted_source_latent: object | None = None

    def validate(self) -> None:
        if self.schema_version != EPISODE_SCHEMA_VERSION:
            raise ValueError(f"unsupported episode schema: {self.schema_version}")
        if self.observations.ndim != 3:
            raise ValueError("observations must have shape episode x time x feature")
        episodes, horizon, features = self.observations.shape
        if self.masks.shape != (episodes, horizon, features):
            raise ValueError("mask shape mismatch")
        if self.recency.shape != (episodes, horizon, features):
            raise ValueError("recency shape mismatch")
        if self.actions.shape != (episodes, horizon):
            raise ValueError("action shape mismatch")
        if self.action_probabilities.ndim != 3 or self.action_probabilities.shape[:2] != (
            episodes,
            horizon,
        ):
            raise ValueError("action-probability shape mismatch")
        for name, value in (
            ("rewards", self.rewards),
            ("terminations", self.terminations),
            ("valid_steps", self.valid_steps),
        ):
            if value.shape != (episodes, horizon):
                raise ValueError(f"{name} shape mismatch")
        if self.lengths.shape != (episodes,):
            raise ValueError("length shape mismatch")
        if not np.array_equal(self.lengths, self.valid_steps.sum(axis=1)):
            raise ValueError("lengths must equal valid-step counts")
        if np.any(self.lengths <= 0) or np.any(self.lengths > horizon):
            raise ValueError("invalid episode length")

        expected_valid = np.arange(horizon)[None, :] < self.lengths[:, None]
        if not np.array_equal(self.valid_steps, expected_valid):
            raise ValueError("valid steps must be a contiguous prefix")
        rows = np.arange(episodes)
        if not np.all(self.terminations[rows, self.lengths - 1]):
            raise ValueError("each episode must terminate at its final valid step")
        if np.any(self.terminations & ~self.valid_steps):
            raise ValueError("termination cannot occur after episode end")

        valid_observations = self.observations[self.valid_steps]
        valid_recency = self.recency[self.valid_steps]
        valid_actions = self.actions[self.valid_steps]
        valid_probabilities = self.action_probabilities[self.valid_steps]
        valid_rewards = self.rewards[self.valid_steps]
        if not np.isfinite(valid_observations).all() or not np.isfinite(valid_recency).all():
            raise ValueError("valid observation surfaces must be finite")
        if not np.isfinite(valid_probabilities).all() or not np.isfinite(valid_rewards).all():
            raise ValueError("valid action probabilities and rewards must be finite")
        if np.any(valid_actions < 0) or np.any(
            valid_actions >= self.action_probabilities.shape[-1]
        ):
            raise ValueError("valid action outside action space")
        if not np.allclose(valid_probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
            raise ValueError("valid action probabilities must sum to one")

        invalid = ~self.valid_steps
        if not np.isnan(self.observations[invalid]).all():
            raise ValueError("post-termination observations must be invalid")
        if np.any(self.masks[invalid]):
            raise ValueError("post-termination masks must be false")
        if not np.isnan(self.recency[invalid]).all():
            raise ValueError("post-termination recency must be invalid")
        if np.any(self.actions[invalid] != -1):
            raise ValueError("post-termination actions must use the -1 sentinel")
        if not np.isnan(self.action_probabilities[invalid]).all():
            raise ValueError("post-termination action probabilities must be invalid")
        if not np.isnan(self.rewards[invalid]).all():
            raise ValueError("post-termination rewards must be invalid")

        uncertainty_shapes = {
            "observation_scale": (episodes, horizon, features),
            "mask_probability": (episodes, horizon, features),
            "behavior_entropy": (episodes, horizon),
            "reward_scale": (episodes, horizon),
            "termination_probability": (episodes, horizon),
        }
        for name, expected in uncertainty_shapes.items():
            value = getattr(self.uncertainty, name)
            if value.shape != expected:
                raise ValueError(f"{name} uncertainty shape mismatch")
            mask = self.valid_steps if value.ndim == 2 else self.valid_steps[..., None]
            if not np.isfinite(value[np.broadcast_to(mask, value.shape)]).all():
                raise ValueError(f"valid {name} uncertainty must be finite")
            if not np.isnan(value[np.broadcast_to(~mask, value.shape)]).all():
                raise ValueError(f"post-termination {name} uncertainty must be invalid")

    def discounted_returns(self, discount: float) -> np.ndarray:
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        weights = discount ** np.arange(self.rewards.shape[1], dtype=np.float64)
        return np.sum(np.where(self.valid_steps, self.rewards, 0.0) * weights[None, :], axis=1)

    def byte_digest(self) -> str:
        """Hash every public array; restricted latent state is intentionally excluded."""
        digest = hashlib.sha256()
        for name in (
            "observations",
            "masks",
            "recency",
            "actions",
            "action_probabilities",
            "rewards",
            "terminations",
            "valid_steps",
            "lengths",
        ):
            value = np.ascontiguousarray(getattr(self, name))
            digest.update(name.encode())
            digest.update(value.dtype.str.encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        for name in (
            "observation_scale",
            "mask_probability",
            "behavior_entropy",
            "reward_scale",
            "termination_probability",
        ):
            value = np.ascontiguousarray(getattr(self.uncertainty, name))
            digest.update(name.encode())
            digest.update(value.dtype.str.encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()


class LearnedInitialHistory(nn.Module):
    """Trainable marginal initial-history component used by fixture tests."""

    def __init__(self, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.mean = nn.Parameter(torch.zeros(feature_dim))
        self.log_scale = nn.Parameter(torch.full((feature_dim,), -1.0))
        self.mask_logits = nn.Parameter(torch.full((feature_dim,), 1.0))
        self.action_logits = nn.Parameter(torch.zeros(action_dim))

    def sample(
        self, episodes: int, history_steps: int, generator: torch.Generator
    ) -> InitialHistory:
        if episodes <= 0 or history_steps <= 0:
            raise ValueError("episodes and history steps must be positive")
        shape = (episodes, history_steps, self.feature_dim)
        values = self.mean + torch.exp(self.log_scale) * torch.randn(
            shape, generator=generator, device=self.mean.device
        )
        mask_probability = torch.sigmoid(self.mask_logits).expand(shape)
        masks = torch.rand(shape, generator=generator, device=self.mean.device) < mask_probability
        recency = torch.zeros(shape, dtype=values.dtype, device=values.device)
        running = torch.zeros((episodes, self.feature_dim), dtype=values.dtype, device=values.device)
        for step in range(history_steps):
            running = torch.where(masks[:, step], torch.zeros_like(running), running + 1.0)
            recency[:, step] = running
        probability = torch.softmax(self.action_logits, dim=0).expand(
            episodes * history_steps, self.action_dim
        )
        previous_actions = torch.multinomial(
            probability, 1, replacement=True, generator=generator
        ).reshape(episodes, history_steps)
        return InitialHistory(values, masks, recency, previous_actions)


class MaskPersistenceHead(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(feature_dim * 3 + action_dim, feature_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(context))


class RecordedBehaviorHead(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.head = nn.Linear(feature_dim * 3 + action_dim, action_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.head(context), dim=-1)


class RewardProxyHead(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(feature_dim * 3 + action_dim, 2)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_scale = self.head(context).unbind(dim=-1)
        return mean, torch.exp(torch.clamp(raw_scale, min=-5.0, max=2.0))


class TerminationHazardHead(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(feature_dim * 3 + action_dim, 1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(context).squeeze(-1))


class BoundedSensitivityHead(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(action_dim, feature_dim, bias=False)

    def forward(self, action: torch.Tensor, bound: float) -> torch.Tensor:
        if bound < 0:
            raise ValueError("sensitivity bound must be nonnegative")
        return float(bound) * torch.tanh(self.head(action))


@dataclass(frozen=True, slots=True)
class SourceSimulatorComponents:
    """Separately fit-capable source-simulator components."""

    initial: LearnedInitialHistory
    transition: nn.Module
    mask: MaskPersistenceHead
    behavior: RecordedBehaviorHead
    reward: RewardProxyHead
    termination: TerminationHazardHead
    bounded_sensitivity: BoundedSensitivityHead
    model_family: ModelFamily

    def modules(self) -> tuple[nn.Module, ...]:
        return (
            self.initial,
            self.transition,
            self.mask,
            self.behavior,
            self.reward,
            self.termination,
            self.bounded_sensitivity,
        )

    def parameters(self) -> Iterable[nn.Parameter]:
        for module in self.modules():
            yield from module.parameters()


@dataclass(frozen=True, slots=True)
class CalibratedRolloutProfile:
    """Aggregate calibration parameters for stable free-running simulation.

    The learned transition supplies the conditional signal.  Observation,
    recorded-action, and episode-length processes are estimated separately
    from the source-model calibration role so that free-running simulation
    does not compound out-of-distribution errors from unrelated neural heads.
    """

    state_mean: np.ndarray
    state_scale: np.ndarray
    state_correlation_cholesky: np.ndarray
    predicted_delta_center: np.ndarray
    predicted_delta_scale: np.ndarray
    learned_signal_center: np.ndarray
    standardized_lower: np.ndarray
    standardized_upper: np.ndarray
    initial_action_probability: np.ndarray
    action_transition_probability: np.ndarray
    initial_mask_probability: np.ndarray
    mask_transition_probability: np.ndarray
    length_probability: np.ndarray
    binary_feature_indices: tuple[int, ...] = ()
    binary_probability: np.ndarray | None = None
    learned_signal_weight: float = 0.20
    persistence_weight: float = 0.55

    def validate(self, task: TaskShape) -> None:
        features = task.feature_dim
        actions = task.action_dim
        feature_vectors = (
            self.state_mean,
            self.state_scale,
            self.predicted_delta_center,
            self.predicted_delta_scale,
            self.learned_signal_center,
            self.standardized_lower,
            self.standardized_upper,
            self.initial_mask_probability,
        )
        if any(np.asarray(value).shape != (features,) for value in feature_vectors):
            raise ValueError("rollout profile feature-vector shape mismatch")
        if np.asarray(self.state_correlation_cholesky).shape != (features, features):
            raise ValueError("rollout profile correlation shape mismatch")
        if np.asarray(self.initial_action_probability).shape != (actions,):
            raise ValueError("rollout profile action-marginal shape mismatch")
        action_transition_shape = np.asarray(
            self.action_transition_probability
        ).shape
        if action_transition_shape not in {
            (actions, actions),
            (task.horizon, actions, actions),
        }:
            raise ValueError("rollout profile action-transition shape mismatch")
        mask_transition_shape = np.asarray(self.mask_transition_probability).shape
        if mask_transition_shape not in {
            (features, 2),
            (task.horizon, features, 2),
        }:
            raise ValueError("rollout profile mask-transition shape mismatch")
        if np.asarray(self.length_probability).shape != (task.horizon,):
            raise ValueError("rollout profile length shape mismatch")
        for probability in (
            self.initial_action_probability,
            self.action_transition_probability,
            self.initial_mask_probability,
            self.mask_transition_probability,
            self.length_probability,
        ):
            value = np.asarray(probability)
            if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
                raise ValueError("rollout profile contains invalid probability")
        if not np.isclose(np.asarray(self.initial_action_probability).sum(), 1.0):
            raise ValueError("initial action probabilities must sum to one")
        if not np.allclose(
            np.asarray(self.action_transition_probability).sum(axis=-1), 1.0
        ):
            raise ValueError("action-transition rows must sum to one")
        if not np.isclose(np.asarray(self.length_probability).sum(), 1.0):
            raise ValueError("length probabilities must sum to one")
        if np.any(np.asarray(self.state_scale) <= 0.0) or np.any(
            np.asarray(self.predicted_delta_scale) <= 0.0
        ):
            raise ValueError("rollout profile scales must be positive")
        if np.any(np.asarray(self.standardized_lower) > np.asarray(self.standardized_upper)):
            raise ValueError("rollout profile bounds are reversed")
        if any(index < 0 or index >= features for index in self.binary_feature_indices):
            raise ValueError("binary feature index is outside the feature space")
        if self.binary_feature_indices:
            if self.binary_probability is None or np.asarray(
                self.binary_probability
            ).shape != (len(self.binary_feature_indices),):
                raise ValueError("binary probability shape mismatch")
        if self.learned_signal_weight < 0.0 or self.persistence_weight < 0.0:
            raise ValueError("rollout weights must be nonnegative")
        squared = self.learned_signal_weight**2 + self.persistence_weight**2
        if squared >= 1.0:
            raise ValueError("rollout weights leave no innovation variance")


def build_unfitted_components(
    task: TaskShape,
    model_family: ModelFamily,
    initialization_seed: int,
    hidden_dim: int = 16,
    latent_dim: int = 16,
) -> SourceSimulatorComponents:
    """Build unfitted, task-generic modules under an isolated initialization seed."""
    if hidden_dim <= 0 or latent_dim <= 0:
        raise ValueError("hidden and latent dimensions must be positive")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed)
        if model_family == "gaussian_recurrent":
            transition: nn.Module = GRUDTransition(
                task.feature_dim, task.action_dim, hidden_dim
            )
        elif model_family == "categorical_rssm":
            transition = DreamerV3CategoricalRSSM(
                task.feature_dim,
                task.action_dim,
                hidden_dim,
                groups=max(2, latent_dim // 4),
                categories=4,
            )
        else:
            raise ValueError(f"unsupported model family: {model_family}")
        components = SourceSimulatorComponents(
            initial=LearnedInitialHistory(task.feature_dim, task.action_dim),
            transition=transition,
            mask=MaskPersistenceHead(task.feature_dim, task.action_dim),
            behavior=RecordedBehaviorHead(task.feature_dim, task.action_dim),
            reward=RewardProxyHead(task.feature_dim, task.action_dim),
            termination=TerminationHazardHead(task.feature_dim, task.action_dim),
            bounded_sensitivity=BoundedSensitivityHead(
                task.feature_dim, task.action_dim
            ),
            model_family=model_family,
        )
    for module in components.modules():
        module.eval()
    return components


def parameter_storage_ids(components: SourceSimulatorComponents) -> set[int]:
    return {parameter.data_ptr() for parameter in components.parameters()}


def source_and_evaluator_are_disjoint(
    source: SourceSimulatorComponents, evaluator: SourceSimulatorComponents
) -> bool:
    return parameter_storage_ids(source).isdisjoint(parameter_storage_ids(evaluator))


class LearnedSourceSimulator:
    """Autoregressive full-episode sampler around an existing transition family."""

    def __init__(
        self,
        task: TaskShape,
        components: SourceSimulatorComponents,
        sensitivity_bound: float = 0.10,
        rollout_profile: CalibratedRolloutProfile | None = None,
    ) -> None:
        self.task = task
        self.components = components
        self.sensitivity_bound = float(sensitivity_bound)
        self.rollout_profile = rollout_profile
        if self.sensitivity_bound < 0:
            raise ValueError("sensitivity bound must be nonnegative")
        if self.rollout_profile is not None:
            self.rollout_profile.validate(task)

    def _profile_history(
        self,
        episodes: int,
        history_steps: int,
        generator: torch.Generator,
    ) -> InitialHistory:
        profile = self.rollout_profile
        if profile is None:
            raise RuntimeError("calibrated rollout profile is unavailable")
        feature_dim = self.task.feature_dim
        noise = torch.randn(
            (episodes, history_steps, feature_dim), generator=generator
        )
        cholesky = torch.as_tensor(
            profile.state_correlation_cholesky, dtype=torch.float32
        )
        standardized = torch.matmul(noise, cholesky.T)
        mean = torch.as_tensor(profile.state_mean, dtype=torch.float32)
        scale = torch.as_tensor(profile.state_scale, dtype=torch.float32)
        values = mean + scale * standardized
        lower = torch.as_tensor(profile.standardized_lower, dtype=torch.float32)
        upper = torch.as_tensor(profile.standardized_upper, dtype=torch.float32)
        values = torch.maximum(torch.minimum(values, upper), lower)
        if profile.binary_feature_indices:
            probability = torch.as_tensor(
                profile.binary_probability, dtype=torch.float32
            )
            sampled = torch.rand(
                (episodes, history_steps, len(profile.binary_feature_indices)),
                generator=generator,
            ) < probability
            for local, feature in enumerate(profile.binary_feature_indices):
                values[..., feature] = torch.where(
                    sampled[..., local], upper[feature], lower[feature]
                )

        initial_mask_probability = torch.as_tensor(
            profile.initial_mask_probability, dtype=torch.float32
        )
        masks = torch.rand(
            (episodes, history_steps, feature_dim), generator=generator
        ) < initial_mask_probability
        mask_transition = torch.as_tensor(
            profile.mask_transition_probability, dtype=torch.float32
        )
        for step in range(1, history_steps):
            if mask_transition.ndim == 3:
                masks[:, step] = (
                    torch.rand((episodes, feature_dim), generator=generator)
                    < initial_mask_probability
                )
            else:
                previous = masks[:, step - 1].to(dtype=torch.long)
                local_probability = torch.where(
                    previous.bool(),
                    mask_transition[:, 1],
                    mask_transition[:, 0],
                )
                masks[:, step] = (
                    torch.rand((episodes, feature_dim), generator=generator)
                    < local_probability
                )
        recency = torch.zeros_like(values)
        running = torch.zeros((episodes, feature_dim), dtype=torch.float32)
        for step in range(history_steps):
            running = torch.where(masks[:, step], torch.zeros_like(running), running + 1.0)
            recency[:, step] = running

        initial_action = torch.as_tensor(
            profile.initial_action_probability, dtype=torch.float32
        ).expand(episodes, self.task.action_dim)
        previous_actions = torch.empty((episodes, history_steps), dtype=torch.long)
        previous_actions[:, 0] = torch.multinomial(
            initial_action, 1, replacement=True, generator=generator
        ).squeeze(-1)
        transition = torch.as_tensor(
            profile.action_transition_probability, dtype=torch.float32
        )
        for step in range(1, history_steps):
            if transition.ndim == 3:
                probability = initial_action
            else:
                probability = transition[previous_actions[:, step - 1]]
            previous_actions[:, step] = torch.multinomial(
                probability, 1, replacement=True, generator=generator
            ).squeeze(-1)
        return InitialHistory(values, masks, recency, previous_actions)

    def _profile_next_values(
        self,
        current_values: torch.Tensor,
        next_mean: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        profile = self.rollout_profile
        if profile is None:
            raise RuntimeError("calibrated rollout profile is unavailable")
        state_mean = torch.as_tensor(profile.state_mean, dtype=torch.float32)
        state_scale = torch.as_tensor(profile.state_scale, dtype=torch.float32)
        predicted_delta_center = torch.as_tensor(
            profile.predicted_delta_center, dtype=torch.float32
        )
        predicted_delta_scale = torch.as_tensor(
            profile.predicted_delta_scale, dtype=torch.float32
        )
        current_signal = torch.clamp(
            (current_values - state_mean) / state_scale, min=-4.0, max=4.0
        )
        learned_signal = torch.tanh(
            (
                (next_mean - current_values)
                - predicted_delta_center
            )
            / predicted_delta_scale
        )
        learned_signal = learned_signal - torch.as_tensor(
            profile.learned_signal_center, dtype=torch.float32
        )
        noise = torch.randn(next_mean.shape, generator=generator)
        cholesky = torch.as_tensor(
            profile.state_correlation_cholesky, dtype=torch.float32
        )
        correlated_noise = torch.matmul(noise, cholesky.T)
        innovation_weight = math.sqrt(
            1.0
            - profile.persistence_weight**2
            - profile.learned_signal_weight**2
        )
        standardized = (
            profile.persistence_weight * current_signal
            + profile.learned_signal_weight * learned_signal
            + innovation_weight * correlated_noise
        )
        values = state_mean + state_scale * standardized
        lower = torch.as_tensor(profile.standardized_lower, dtype=torch.float32)
        upper = torch.as_tensor(profile.standardized_upper, dtype=torch.float32)
        values = torch.maximum(torch.minimum(values, upper), lower)
        if profile.binary_feature_indices:
            probability = torch.as_tensor(
                profile.binary_probability, dtype=torch.float32
            )
            sampled = torch.rand(
                (len(values), len(profile.binary_feature_indices)),
                generator=generator,
            ) < probability
            for local, feature in enumerate(profile.binary_feature_indices):
                values[:, feature] = torch.where(
                    sampled[:, local], upper[feature], lower[feature]
                )
        return values

    def _condition_action(
        self, action: torch.Tensor, regime: ResponseRegime
    ) -> torch.Tensor:
        one_hot = torch.nn.functional.one_hot(
            action, num_classes=self.task.action_dim
        ).to(dtype=torch.float32)
        if regime is ResponseRegime.NULL:
            return torch.zeros_like(one_hot)
        if regime is ResponseRegime.PERTURBED:
            return torch.roll(one_hot, shifts=1, dims=-1)
        return one_hot

    def _transition(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        recency: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.components.transition(
            values,
            masks.to(dtype=values.dtype),
            recency,
            actions,
        )
        if not isinstance(output, TransitionOutput):
            raise TypeError("transition component must return TransitionOutput")
        return output.mean[:, -1], torch.exp(output.log_scale[:, -1])

    def simulate(
        self,
        episodes: int,
        seed: int,
        response_regime: ResponseRegime | str,
        *,
        history_steps: int = 3,
        initial_history: InitialHistory | None = None,
        forced_actions: np.ndarray | None = None,
        policy: OnlinePolicy | None = None,
        policy_seed: int | None = None,
    ) -> FullEpisodeBatch:
        if episodes <= 0:
            raise ValueError("episode count must be positive")
        try:
            regime = (
                response_regime
                if isinstance(response_regime, ResponseRegime)
                else ResponseRegime(response_regime)
            )
        except ValueError as error:
            raise ValueError(f"unsupported response regime: {response_regime}") from error
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        if initial_history is not None:
            history = initial_history
        elif self.rollout_profile is not None:
            history = self._profile_history(episodes, history_steps, generator)
        else:
            history = self.components.initial.sample(
                episodes, history_steps, generator
            )
        history.validate(self.task, episodes)
        if forced_actions is not None and policy is not None:
            raise ValueError("forced actions and online policy are mutually exclusive")
        if forced_actions is not None:
            forced_actions = np.asarray(forced_actions)
            if forced_actions.shape != (episodes, self.task.horizon):
                raise ValueError("forced actions must have shape episode x horizon")
            if np.any(forced_actions < 0) or np.any(forced_actions >= self.task.action_dim):
                raise ValueError("forced action outside task action space")
            forced_tensor = torch.as_tensor(forced_actions, dtype=torch.long)
        else:
            forced_tensor = None
        policy_generator = torch.Generator(device="cpu")
        policy_generator.manual_seed(
            int(policy_seed if policy_seed is not None else seed + 1_000_003)
        )

        values_history = history.values.detach().to(dtype=torch.float32, device="cpu")
        masks_history = history.masks.detach().to(dtype=torch.bool, device="cpu")
        recency_history = history.recency.detach().to(dtype=torch.float32, device="cpu")
        previous_action = history.previous_actions[:, -1].detach().to(
            dtype=torch.long, device="cpu"
        )
        action_history = torch.nn.functional.one_hot(
            history.previous_actions.detach().to(dtype=torch.long, device="cpu"),
            num_classes=self.task.action_dim,
        ).to(dtype=torch.float32)

        shape = (episodes, self.task.horizon)
        feature_shape = (*shape, self.task.feature_dim)
        observations = np.full(feature_shape, np.nan, dtype=np.float32)
        masks = np.zeros(feature_shape, dtype=bool)
        recency = np.full(feature_shape, np.nan, dtype=np.float32)
        actions = np.full(shape, -1, dtype=np.int16)
        action_probabilities = np.full(
            (*shape, self.task.action_dim), np.nan, dtype=np.float32
        )
        rewards = np.full(shape, np.nan, dtype=np.float32)
        terminations = np.zeros(shape, dtype=bool)
        valid_steps = np.zeros(shape, dtype=bool)
        lengths = np.zeros(episodes, dtype=np.int16)
        observation_scale = np.full(feature_shape, np.nan, dtype=np.float32)
        mask_probability = np.full(feature_shape, np.nan, dtype=np.float32)
        behavior_entropy = np.full(shape, np.nan, dtype=np.float32)
        reward_scale = np.full(shape, np.nan, dtype=np.float32)
        termination_probability = np.full(shape, np.nan, dtype=np.float32)
        alive = torch.ones(episodes, dtype=torch.bool)
        planned_lengths: torch.Tensor | None = None
        profile_termination_hazard: torch.Tensor | None = None
        if self.rollout_profile is not None:
            length_probability = torch.as_tensor(
                self.rollout_profile.length_probability, dtype=torch.float32
            ).expand(episodes, self.task.horizon)
            planned_lengths = (
                torch.multinomial(
                    length_probability,
                    1,
                    replacement=True,
                    generator=generator,
                ).squeeze(-1)
                + 1
            )
            local_length_probability = torch.as_tensor(
                self.rollout_profile.length_probability, dtype=torch.float32
            )
            survival = torch.flip(
                torch.cumsum(torch.flip(local_length_probability, dims=(0,)), dim=0),
                dims=(0,),
            )
            profile_termination_hazard = local_length_probability / torch.clamp(
                survival, min=1.0e-8
            )

        with torch.inference_mode():
            for step in range(self.task.horizon):
                active = alive.clone()
                current_values = values_history[:, -1]
                current_masks = masks_history[:, -1]
                current_recency = recency_history[:, -1]
                previous_one_hot = torch.nn.functional.one_hot(
                    previous_action, num_classes=self.task.action_dim
                ).to(dtype=torch.float32)
                behavior_context = torch.cat(
                    [
                        current_values,
                        current_masks.to(dtype=torch.float32),
                        current_recency,
                        previous_one_hot,
                    ],
                    dim=-1,
                )
                if self.rollout_profile is None:
                    probabilities = self.components.behavior(behavior_context)
                elif step == 0:
                    probabilities = torch.as_tensor(
                        self.rollout_profile.initial_action_probability,
                        dtype=torch.float32,
                    ).expand(episodes, self.task.action_dim)
                else:
                    action_transition = torch.as_tensor(
                        self.rollout_profile.action_transition_probability,
                        dtype=torch.float32,
                    )
                    if action_transition.ndim == 3:
                        probabilities = action_transition[step, previous_action]
                    else:
                        probabilities = action_transition[previous_action]
                if policy is not None:
                    local_probability = np.asarray(
                        policy(
                            current_values.numpy(),
                            current_masks.numpy(),
                            current_recency.numpy(),
                            previous_action.numpy(),
                            step,
                        ),
                        dtype=np.float64,
                    )
                    if local_probability.shape != (
                        episodes,
                        self.task.action_dim,
                    ):
                        raise ValueError("online policy probability shape mismatch")
                    row_sum = local_probability.sum(axis=1, keepdims=True)
                    if (
                        not np.isfinite(local_probability).all()
                        or np.any(local_probability < 0)
                        or not np.isfinite(row_sum).all()
                        or np.any(row_sum <= 0)
                        or np.max(np.abs(row_sum - 1.0)) > 1.0e-5
                    ):
                        raise ValueError(
                            "online policy probabilities must be finite, "
                            "nonnegative, and normalized"
                        )
                    local_probability = local_probability / row_sum
                    probabilities = torch.as_tensor(
                        local_probability, dtype=torch.float32
                    )
                    action = torch.multinomial(
                        probabilities,
                        1,
                        replacement=True,
                        generator=policy_generator,
                    ).squeeze(-1)
                elif forced_tensor is None:
                    action = torch.multinomial(
                        probabilities, 1, replacement=True, generator=generator
                    ).squeeze(-1)
                else:
                    action = forced_tensor[:, step]
                conditioned_action = self._condition_action(action, regime)
                conditioned_history = action_history.clone()
                conditioned_history[:, -1] = conditioned_action

                next_mean, next_scale = self._transition(
                    values_history,
                    masks_history,
                    recency_history,
                    conditioned_history,
                )
                if regime is ResponseRegime.BOUNDED:
                    next_mean = next_mean + self.components.bounded_sensitivity(
                        conditioned_action, self.sensitivity_bound
                    )
                component_context = torch.cat(
                    [
                        current_values,
                        current_masks.to(dtype=torch.float32),
                        current_recency,
                        conditioned_action,
                    ],
                    dim=-1,
                )
                if self.rollout_profile is None:
                    next_mask_probability = self.components.mask(component_context)
                else:
                    mask_transition = torch.as_tensor(
                        self.rollout_profile.mask_transition_probability,
                        dtype=torch.float32,
                    )
                    if mask_transition.ndim == 3:
                        mask_transition = mask_transition[step]
                    next_mask_probability = torch.where(
                        current_masks,
                        mask_transition[:, 1],
                        mask_transition[:, 0],
                    )
                reward_mean, local_reward_scale = self.components.reward(component_context)
                if self.rollout_profile is None:
                    local_termination_probability = self.components.termination(
                        component_context
                    )
                    if step == self.task.horizon - 1:
                        local_termination_probability = torch.ones_like(
                            local_termination_probability
                        )
                else:
                    if profile_termination_hazard is None:
                        raise RuntimeError("profile termination hazard is unavailable")
                    local_termination_probability = torch.full(
                        (episodes,),
                        float(profile_termination_hazard[step]),
                        dtype=torch.float32,
                    )

                if self.rollout_profile is None:
                    next_values = next_mean + next_scale * torch.randn(
                        next_mean.shape, generator=generator
                    )
                else:
                    next_values = self._profile_next_values(
                        current_values, next_mean, generator
                    )
                next_masks = torch.rand(
                    next_mask_probability.shape, generator=generator
                ) < next_mask_probability
                next_recency = torch.where(
                    next_masks,
                    torch.zeros_like(current_recency),
                    current_recency + 1.0,
                )
                local_reward = reward_mean + local_reward_scale * torch.randn(
                    reward_mean.shape, generator=generator
                )
                if planned_lengths is None:
                    terminate = active & (
                        torch.rand(
                            local_termination_probability.shape, generator=generator
                        )
                        < local_termination_probability
                    )
                else:
                    terminate = active & (planned_lengths == step + 1)

                active_np = active.numpy()
                observations[active_np, step] = current_values[active].numpy()
                masks[active_np, step] = current_masks[active].numpy()
                recency[active_np, step] = current_recency[active].numpy()
                actions[active_np, step] = action[active].numpy().astype(np.int16)
                action_probabilities[active_np, step] = probabilities[active].numpy()
                rewards[active_np, step] = local_reward[active].numpy()
                terminations[active_np, step] = terminate[active].numpy()
                valid_steps[active_np, step] = True
                lengths[active_np] = step + 1
                observation_scale[active_np, step] = next_scale[active].numpy()
                mask_probability[active_np, step] = next_mask_probability[active].numpy()
                local_entropy = -torch.sum(
                    probabilities
                    * torch.log(torch.clamp(probabilities, min=1.0e-12)),
                    dim=-1,
                )
                behavior_entropy[active_np, step] = local_entropy[active].numpy()
                reward_scale[active_np, step] = local_reward_scale[active].numpy()
                termination_probability[active_np, step] = (
                    local_termination_probability[active].numpy()
                )

                alive = active & ~terminate
                previous_action = action
                values_history = torch.cat(
                    [values_history, next_values[:, None, :]], dim=1
                )
                masks_history = torch.cat(
                    [masks_history, next_masks[:, None, :]], dim=1
                )
                recency_history = torch.cat(
                    [recency_history, next_recency[:, None, :]], dim=1
                )
                action_history = torch.cat(
                    [
                        conditioned_history,
                        torch.zeros(
                            (episodes, 1, self.task.action_dim), dtype=torch.float32
                        ),
                    ],
                    dim=1,
                )

        output = FullEpisodeBatch(
            schema_version=EPISODE_SCHEMA_VERSION,
            task_name=self.task.name,
            response_regime=regime.value,
            observations=observations,
            masks=masks,
            recency=recency,
            actions=actions,
            action_probabilities=action_probabilities,
            rewards=rewards,
            terminations=terminations,
            valid_steps=valid_steps,
            lengths=lengths,
            uncertainty=EpisodeUncertainty(
                observation_scale=observation_scale,
                mask_probability=mask_probability,
                behavior_entropy=behavior_entropy,
                reward_scale=reward_scale,
                termination_probability=termination_probability,
            ),
            restricted_source_latent=None,
        )
        output.validate()
        return output
