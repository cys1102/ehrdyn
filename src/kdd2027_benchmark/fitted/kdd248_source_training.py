"""Training utilities for the protected KDD248 EHR-anchored source simulator.

The functions in this module operate on already materialized, patient-disjoint
role interfaces. They never write row-level arrays. Checkpoints are written
only to the caller-supplied restricted directory.
"""
from __future__ import annotations

import copy
import hashlib
import io
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from .kdd069_model_types import TransitionOutput, parameter_count
from .kdd248_full_episode import (
    CalibratedRolloutProfile,
    SourceSimulatorComponents,
    TaskShape,
    build_unfitted_components,
)


@dataclass(slots=True)
class PreparedRole:
    values: np.ndarray
    masks: np.ndarray
    deltas: np.ndarray
    actions: np.ndarray
    action_one_hot: np.ndarray
    previous_action_one_hot: np.ndarray
    targets: np.ndarray
    target_masks: np.ndarray
    valid: np.ndarray
    termination: np.ndarray
    rewards: np.ndarray
    reward_masks: np.ndarray
    preprocessing_mean: np.ndarray
    preprocessing_scale: np.ndarray

    @property
    def episodes(self) -> int:
        return int(self.values.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.values.shape[1])


@dataclass(slots=True)
class FitResult:
    components: SourceSimulatorComponents
    selected_epoch: int
    epochs_run: int
    validation_curve: list[float]
    convergence_status: str
    runtime_seconds: float
    checkpoint_sha256: str
    logical_checkpoint_id: str
    parameter_count: int
    calibration_receipt: dict[str, float]


def prepare_role(interface: Any, reward_name: str, action_dim: int) -> PreparedRole:
    if reward_name not in interface.reward_names:
        raise RuntimeError(
            f"frozen reward channel unavailable for {interface.cohort}: {reward_name}"
        )
    valid = np.asarray(interface.valid_steps, dtype=bool)
    classes = np.asarray(interface.actions, dtype=np.int64)
    if np.any(classes[valid] < 0) or np.any(classes[valid] >= action_dim):
        raise RuntimeError("action outside frozen task action space")
    actions = np.zeros((*classes.shape, action_dim), dtype=np.float32)
    positions = np.argwhere(valid)
    actions[positions[:, 0], positions[:, 1], classes[valid]] = 1.0
    previous = np.zeros_like(actions)
    previous[:, 1:] = actions[:, :-1]
    mean = np.asarray(interface.preprocessing_mean, dtype=np.float32)
    scale = np.asarray(interface.preprocessing_scale, dtype=np.float32)
    targets = np.nan_to_num(
        (np.asarray(interface.targets, dtype=np.float32) - mean[None, None])
        / scale[None, None],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    channel = interface.reward_names.index(reward_name)
    return PreparedRole(
        values=np.asarray(interface.imputed_history, dtype=np.float32),
        masks=np.asarray(interface.masks, dtype=bool),
        deltas=np.asarray(interface.deltas, dtype=np.float32),
        actions=classes.astype(np.int16),
        action_one_hot=actions,
        previous_action_one_hot=previous,
        targets=targets,
        target_masks=np.asarray(interface.target_masks, dtype=bool),
        valid=valid,
        termination=np.asarray(interface.termination, dtype=np.float32),
        rewards=np.asarray(interface.rewards[..., channel], dtype=np.float32),
        reward_masks=np.asarray(interface.reward_masks[..., channel], dtype=bool),
        preprocessing_mean=mean,
        preprocessing_scale=scale,
    )


def _probability_logit(value: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(value, 1.0e-5, 1.0 - 1.0e-5)
    return np.log(clipped) - np.log1p(-clipped)


def _fixed_sign(task: str, feature: int, action: int, seed: int) -> float:
    payload = f"{task}|{feature}|{action}|{seed}".encode("utf-8")
    return -1.0 if hashlib.sha256(payload).digest()[-1] & 1 == 0 else 1.0


def initialize_closed_form_components(
    task: TaskShape,
    components: SourceSimulatorComponents,
    train: PreparedRole,
    perturbation_seed: int,
) -> None:
    valid_feature = np.broadcast_to(train.valid[..., None], train.values.shape)
    values = train.values[valid_feature].reshape(-1, task.feature_dim)
    masks = train.masks[valid_feature].reshape(-1, task.feature_dim)
    if values.size == 0:
        raise RuntimeError("source-model train role has no valid state")
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), 1.0e-3)
    mask_probability = np.clip(masks.mean(axis=0), 1.0e-4, 1.0 - 1.0e-4)
    counts = np.bincount(train.actions[train.valid], minlength=task.action_dim).astype(float)
    frequency = np.clip(counts / max(counts.sum(), 1.0), 1.0e-8, None)
    frequency /= frequency.sum()
    with torch.no_grad():
        components.initial.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        components.initial.log_scale.copy_(
            torch.as_tensor(np.log(scale), dtype=torch.float32)
        )
        components.initial.mask_logits.copy_(
            torch.as_tensor(_probability_logit(mask_probability), dtype=torch.float32)
        )
        components.initial.action_logits.copy_(
            torch.as_tensor(np.log(frequency), dtype=torch.float32)
        )
        fixed = np.empty((task.feature_dim, task.action_dim), dtype=np.float32)
        for feature in range(task.feature_dim):
            for action in range(task.action_dim):
                fixed[feature, action] = 7.0 * _fixed_sign(
                    task.name, feature, action, perturbation_seed
                )
        components.bounded_sensitivity.head.weight.copy_(torch.from_numpy(fixed))
    components.initial.requires_grad_(False)
    components.bounded_sensitivity.requires_grad_(False)


def _to_device(
    role: PreparedRole, indices: np.ndarray, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "values": torch.from_numpy(role.values[indices]).to(device),
        "masks": torch.from_numpy(role.masks[indices].astype(np.float32)).to(device),
        "deltas": torch.from_numpy(role.deltas[indices]).to(device),
        "actions": torch.from_numpy(role.action_one_hot[indices]).to(device),
        "previous_actions": torch.from_numpy(
            role.previous_action_one_hot[indices]
        ).to(device),
        "classes": torch.from_numpy(role.actions[indices].astype(np.int64)).to(device),
        "targets": torch.from_numpy(role.targets[indices]).to(device),
        "target_masks": torch.from_numpy(
            role.target_masks[indices].astype(np.float32)
        ).to(device),
        "valid": torch.from_numpy(role.valid[indices].astype(np.float32)).to(device),
        "termination": torch.from_numpy(role.termination[indices]).to(device),
        "rewards": torch.from_numpy(role.rewards[indices]).to(device),
        "reward_masks": torch.from_numpy(
            role.reward_masks[indices].astype(np.float32)
        ).to(device),
    }


def component_losses(
    components: SourceSimulatorComponents,
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    output = components.transition(
        batch["values"], batch["masks"], batch["deltas"], batch["actions"]
    )
    if not isinstance(output, TransitionOutput):
        raise TypeError("transition component must return TransitionOutput")
    observed = batch["target_masks"] * batch["valid"].unsqueeze(-1)
    scale = torch.exp(torch.clamp(output.log_scale, min=-5.0, max=2.0))
    transition_cell = output.log_scale + 0.5 * torch.square(
        (batch["targets"] - output.mean) / scale
    )
    transition = (transition_cell * observed).sum() / torch.clamp(
        observed.sum(), min=1.0
    )

    context = torch.cat(
        [batch["values"], batch["masks"], batch["deltas"], batch["actions"]],
        dim=-1,
    )
    behavior_context = torch.cat(
        [
            batch["values"],
            batch["masks"],
            batch["deltas"],
            batch["previous_actions"],
        ],
        dim=-1,
    )
    valid = batch["valid"]
    mask_probability = torch.clamp(components.mask(context), 1.0e-6, 1.0 - 1.0e-6)
    mask_cell = torch.nn.functional.binary_cross_entropy(
        mask_probability, batch["target_masks"], reduction="none"
    )
    missingness = (mask_cell * valid.unsqueeze(-1)).sum() / torch.clamp(
        valid.sum() * batch["target_masks"].shape[-1], min=1.0
    )

    behavior_probability = torch.clamp(
        components.behavior(behavior_context), 1.0e-8, 1.0
    )
    selected = torch.gather(
        behavior_probability,
        dim=-1,
        index=torch.clamp(batch["classes"], min=0).unsqueeze(-1),
    ).squeeze(-1)
    behavior = (-torch.log(selected) * valid).sum() / torch.clamp(valid.sum(), min=1.0)

    reward_mean, reward_scale = components.reward(context)
    reward_cell = torch.log(reward_scale) + 0.5 * torch.square(
        (batch["rewards"] - reward_mean) / reward_scale
    )
    reward_weight = batch["reward_masks"] * valid
    reward = (reward_cell * reward_weight).sum() / torch.clamp(
        reward_weight.sum(), min=1.0
    )

    termination_probability = torch.clamp(
        components.termination(context), 1.0e-6, 1.0 - 1.0e-6
    )
    termination_cell = torch.nn.functional.binary_cross_entropy(
        termination_probability, batch["termination"], reduction="none"
    )
    termination = (termination_cell * valid).sum() / torch.clamp(
        valid.sum(), min=1.0
    )
    total = (
        float(weights["transition"]) * transition
        + float(weights["native_latent_auxiliary"]) * output.auxiliary_loss
        + float(weights["missingness"]) * missingness
        + float(weights["recorded_behavior"]) * behavior
        + float(weights["reward_proxy"]) * reward
        + float(weights["termination"]) * termination
    )
    return {
        "total": total,
        "transition": transition,
        "native_latent_auxiliary": output.auxiliary_loss,
        "missingness": missingness,
        "recorded_behavior": behavior,
        "reward_proxy": reward,
        "termination": termination,
    }


def _validation_loss(
    components: SourceSimulatorComponents,
    role: PreparedRole,
    batch_size: int,
    device: torch.device,
    weights: dict[str, float],
) -> float:
    components_eval(components)
    total = 0.0
    episodes = 0
    with torch.inference_mode():
        for start in range(0, role.episodes, batch_size):
            indices = np.arange(start, min(start + batch_size, role.episodes))
            loss = component_losses(components, _to_device(role, indices, device), weights)
            total += float(loss["total"].detach().cpu()) * len(indices)
            episodes += len(indices)
    return total / max(episodes, 1)


def components_train(components: SourceSimulatorComponents) -> None:
    for module in components.modules():
        module.train()
    components.initial.eval()
    components.bounded_sensitivity.eval()


def components_eval(components: SourceSimulatorComponents) -> None:
    for module in components.modules():
        module.eval()


def _trainable_parameters(
    components: SourceSimulatorComponents,
) -> Iterable[nn.Parameter]:
    for module in (
        components.transition,
        components.mask,
        components.behavior,
        components.reward,
        components.termination,
    ):
        yield from module.parameters()


def _state_dict(components: SourceSimulatorComponents) -> dict[str, Any]:
    return {
        "initial": copy.deepcopy(components.initial.state_dict()),
        "transition": copy.deepcopy(components.transition.state_dict()),
        "mask": copy.deepcopy(components.mask.state_dict()),
        "behavior": copy.deepcopy(components.behavior.state_dict()),
        "reward": copy.deepcopy(components.reward.state_dict()),
        "termination": copy.deepcopy(components.termination.state_dict()),
        "bounded_sensitivity": copy.deepcopy(
            components.bounded_sensitivity.state_dict()
        ),
    }


def _load_state_dict(
    components: SourceSimulatorComponents, state: dict[str, Any]
) -> None:
    for name in (
        "initial",
        "transition",
        "mask",
        "behavior",
        "reward",
        "termination",
        "bounded_sensitivity",
    ):
        getattr(components, name).load_state_dict(state[name])


def predict_components(
    components: SourceSimulatorComponents,
    role: PreparedRole,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    output: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "mean",
            "scale",
            "mask_probability",
            "behavior_probability",
            "reward_mean",
            "reward_scale",
            "termination_probability",
        )
    }
    components_eval(components)
    with torch.inference_mode():
        for start in range(0, role.episodes, batch_size):
            indices = np.arange(start, min(start + batch_size, role.episodes))
            batch = _to_device(role, indices, device)
            transition = components.transition(
                batch["values"], batch["masks"], batch["deltas"], batch["actions"]
            )
            context = torch.cat(
                [
                    batch["values"],
                    batch["masks"],
                    batch["deltas"],
                    batch["actions"],
                ],
                dim=-1,
            )
            behavior_context = torch.cat(
                [
                    batch["values"],
                    batch["masks"],
                    batch["deltas"],
                    batch["previous_actions"],
                ],
                dim=-1,
            )
            reward_mean, reward_scale = components.reward(context)
            values = {
                "mean": transition.mean,
                "scale": torch.exp(transition.log_scale),
                "mask_probability": components.mask(context),
                "behavior_probability": components.behavior(behavior_context),
                "reward_mean": reward_mean,
                "reward_scale": reward_scale,
                "termination_probability": components.termination(context),
            }
            for name, tensor in values.items():
                output[name].append(tensor.detach().cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in output.items()}


def apply_fixed_calibration(
    components: SourceSimulatorComponents,
    calibration: PreparedRole,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Fit fixed moment-matching bias/scale calibrators on the nested role."""
    predicted = predict_components(components, calibration, batch_size, device)
    valid = calibration.valid
    observed = calibration.target_masks & valid[..., None]
    z2 = np.square(
        (calibration.targets - predicted["mean"])
        / np.maximum(predicted["scale"], 1.0e-5)
    )
    transition_multiplier = float(
        np.sqrt(np.mean(z2[observed])) if observed.any() else 1.0
    )
    transition_multiplier = float(np.clip(transition_multiplier, 0.25, 4.0))

    target_mask_rate = calibration.target_masks[valid].mean(axis=0)
    predicted_mask_rate = predicted["mask_probability"][valid].mean(axis=0)
    mask_shift = np.clip(
        _probability_logit(target_mask_rate)
        - _probability_logit(predicted_mask_rate),
        -2.0,
        2.0,
    )

    action_counts = np.bincount(
        calibration.actions[valid],
        minlength=predicted["behavior_probability"].shape[-1],
    ).astype(float)
    action_frequency = np.clip(
        action_counts / max(action_counts.sum(), 1.0), 1.0e-6, None
    )
    predicted_frequency = np.clip(
        predicted["behavior_probability"][valid].mean(axis=0), 1.0e-6, None
    )
    behavior_shift = np.clip(
        np.log(action_frequency) - np.log(predicted_frequency), -2.0, 2.0
    )

    termination_rate = float(calibration.termination[valid].mean())
    predicted_termination = float(
        predicted["termination_probability"][valid].mean()
    )
    termination_shift = float(
        np.clip(
            _probability_logit(termination_rate)
            - _probability_logit(predicted_termination),
            -2.0,
            2.0,
        )
    )

    reward_weight = calibration.reward_masks & valid
    reward_multiplier = 1.0
    if reward_weight.any():
        reward_z2 = np.square(
            (calibration.rewards - predicted["reward_mean"])
            / np.maximum(predicted["reward_scale"], 1.0e-5)
        )
        reward_multiplier = float(
            np.clip(np.sqrt(np.mean(reward_z2[reward_weight])), 0.25, 4.0)
        )

    with torch.no_grad():
        transition_head = components.transition.head.log_scale
        transition_head.bias.add_(math.log(transition_multiplier))
        components.mask.head.bias.add_(
            torch.as_tensor(
                mask_shift,
                dtype=components.mask.head.bias.dtype,
                device=components.mask.head.bias.device,
            )
        )
        components.behavior.head.bias.add_(
            torch.as_tensor(
                behavior_shift,
                dtype=components.behavior.head.bias.dtype,
                device=components.behavior.head.bias.device,
            )
        )
        components.termination.head.bias.add_(termination_shift)
        components.reward.head.bias[1].add_(math.log(reward_multiplier))
    return {
        "transition_scale_multiplier": transition_multiplier,
        "mask_bias_shift_max_abs": float(np.max(np.abs(mask_shift))),
        "behavior_bias_shift_max_abs": float(np.max(np.abs(behavior_shift))),
        "termination_bias_shift": termination_shift,
        "reward_scale_multiplier": reward_multiplier,
    }


def _nearest_correlation(matrix: np.ndarray) -> np.ndarray:
    """Return a deterministic positive-definite correlation approximation."""
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    projected = (eigenvectors * np.maximum(eigenvalues, 1.0e-4)) @ eigenvectors.T
    diagonal = np.sqrt(np.maximum(np.diag(projected), 1.0e-8))
    correlation = projected / diagonal[:, None] / diagonal[None, :]
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def build_calibrated_rollout_profile(
    *,
    components: SourceSimulatorComponents,
    calibration: PreparedRole,
    task: TaskShape,
    feature_names: list[str],
    variable_ranges: dict[str, list[float]],
    batch_size: int,
    device: torch.device,
    learned_signal_weight: float,
    persistence_weight: float,
) -> CalibratedRolloutProfile:
    """Fit aggregate free-rollout nuisance processes on calibration data only."""
    if len(feature_names) != task.feature_dim:
        raise ValueError("feature inventory does not match task shape")
    predictions = predict_components(components, calibration, batch_size, device)
    valid = calibration.valid
    observed = calibration.target_masks & valid[..., None]
    flat_target = calibration.targets.reshape(-1, task.feature_dim)
    flat_observed = observed.reshape(-1, task.feature_dim)
    flat_prediction = predictions["mean"].reshape(-1, task.feature_dim)

    state_mean = np.zeros(task.feature_dim, dtype=np.float64)
    state_scale = np.ones(task.feature_dim, dtype=np.float64)
    predicted_delta_center = np.zeros(task.feature_dim, dtype=np.float64)
    predicted_delta_scale = np.ones(task.feature_dim, dtype=np.float64)
    learned_signal_center = np.zeros(task.feature_dim, dtype=np.float64)
    for feature in range(task.feature_dim):
        keep = flat_observed[:, feature]
        if keep.any():
            target = flat_target[keep, feature].astype(np.float64)
            prediction = flat_prediction[keep, feature].astype(np.float64)
            current = calibration.values.reshape(
                -1, task.feature_dim
            )[keep, feature].astype(np.float64)
            predicted_delta = prediction - current
            state_mean[feature] = float(target.mean())
            state_scale[feature] = max(float(target.std(ddof=0)), 0.05)
            predicted_delta_center[feature] = float(predicted_delta.mean())
            predicted_delta_scale[feature] = max(
                float(predicted_delta.std(ddof=0)), 0.05
            )
            learned_signal_center[feature] = float(
                np.tanh(
                    (
                        predicted_delta
                        - predicted_delta_center[feature]
                    )
                    / predicted_delta_scale[feature]
                ).mean()
            )

    correlation = np.eye(task.feature_dim, dtype=np.float64)
    for left in range(task.feature_dim):
        for right in range(left + 1, task.feature_dim):
            keep = flat_observed[:, left] & flat_observed[:, right]
            if int(keep.sum()) < 50:
                continue
            left_value = flat_target[keep, left]
            right_value = flat_target[keep, right]
            left_value = left_value - left_value.mean()
            right_value = right_value - right_value.mean()
            denominator = float(
                np.sqrt(
                    np.mean(np.square(left_value))
                    * np.mean(np.square(right_value))
                )
            )
            if denominator <= 1.0e-8:
                continue
            value = float(np.mean(left_value * right_value) / denominator)
            if np.isfinite(value):
                correlation[left, right] = value
                correlation[right, left] = value
    correlation = _nearest_correlation(correlation)
    correlation_cholesky = np.linalg.cholesky(
        correlation + np.eye(task.feature_dim) * 1.0e-6
    )

    initial_action_keep = valid[:, 0]
    action_counts = np.bincount(
        calibration.actions[initial_action_keep, 0],
        minlength=task.action_dim,
    ).astype(np.float64)
    initial_action_probability = (action_counts + 0.5) / (
        action_counts.sum() + 0.5 * task.action_dim
    )
    action_transition = np.empty(
        (task.horizon, task.action_dim, task.action_dim),
        dtype=np.float64,
    )
    last_action_marginal = initial_action_probability
    for step in range(task.horizon):
        if step < calibration.horizon:
            keep = valid[:, step]
            counts = np.bincount(
                calibration.actions[keep, step],
                minlength=task.action_dim,
            ).astype(np.float64)
            marginal = (counts + 0.5) / (
                counts.sum() + 0.5 * task.action_dim
            )
            last_action_marginal = marginal
        else:
            marginal = last_action_marginal
        action_transition[step] = np.broadcast_to(
            marginal, action_transition[step].shape
        )

    initial_mask_probability = np.clip(
        calibration.masks[initial_action_keep, 0].mean(axis=0),
        1.0e-4,
        1.0 - 1.0e-4,
    )
    mask_transition = np.ones(
        (task.horizon, task.feature_dim, 2, 2), dtype=np.float64
    )
    for step in range(1, calibration.horizon):
        keep = valid[:, step - 1] & valid[:, step]
        previous = calibration.masks[keep, step - 1].astype(int)
        current = calibration.masks[keep, step].astype(int)
        for previous_value in (0, 1):
            local = previous == previous_value
            mask_transition[step - 1, :, previous_value, 1] += (
                local & (current == 1)
            ).sum(axis=0)
            mask_transition[step - 1, :, previous_value, 0] += (
                local & (current == 0)
            ).sum(axis=0)
    if task.horizon > 1:
        mask_transition[-1] = mask_transition[-2]
    mask_probability = (
        mask_transition[..., 1] / mask_transition.sum(axis=-1)
    )

    lengths = np.clip(valid.sum(axis=1).astype(int), 1, task.horizon)
    length_counts = np.bincount(lengths, minlength=task.horizon + 1)[1:].astype(
        np.float64
    )
    length_probability = (length_counts + 0.5) / (
        length_counts.sum() + 0.5 * task.horizon
    )

    preprocessing_mean = calibration.preprocessing_mean.astype(np.float64)
    preprocessing_scale = calibration.preprocessing_scale.astype(np.float64)
    standardized_lower = np.empty(task.feature_dim, dtype=np.float64)
    standardized_upper = np.empty(task.feature_dim, dtype=np.float64)
    for feature, name in enumerate(feature_names):
        lower, upper = variable_ranges[name]
        standardized_lower[feature] = (
            float(lower) - preprocessing_mean[feature]
        ) / preprocessing_scale[feature]
        standardized_upper[feature] = (
            float(upper) - preprocessing_mean[feature]
        ) / preprocessing_scale[feature]

    binary_indices = tuple(
        index
        for index, name in enumerate(feature_names)
        if name == "mechanical_ventilation"
    )
    binary_probability = np.empty(len(binary_indices), dtype=np.float64)
    for local, feature in enumerate(binary_indices):
        keep = flat_observed[:, feature]
        physical = (
            flat_target[keep, feature] * preprocessing_scale[feature]
            + preprocessing_mean[feature]
        )
        binary_probability[local] = (
            float((physical >= 0.5).mean()) if len(physical) else 0.0
        )

    profile = CalibratedRolloutProfile(
        state_mean=state_mean.astype(np.float32),
        state_scale=state_scale.astype(np.float32),
        state_correlation_cholesky=correlation_cholesky.astype(np.float32),
        predicted_delta_center=predicted_delta_center.astype(np.float32),
        predicted_delta_scale=predicted_delta_scale.astype(np.float32),
        learned_signal_center=learned_signal_center.astype(np.float32),
        standardized_lower=standardized_lower.astype(np.float32),
        standardized_upper=standardized_upper.astype(np.float32),
        initial_action_probability=initial_action_probability.astype(np.float32),
        action_transition_probability=action_transition.astype(np.float32),
        initial_mask_probability=initial_mask_probability.astype(np.float32),
        mask_transition_probability=mask_probability.astype(np.float32),
        length_probability=length_probability.astype(np.float32),
        binary_feature_indices=binary_indices,
        binary_probability=binary_probability.astype(np.float32),
        learned_signal_weight=float(learned_signal_weight),
        persistence_weight=float(persistence_weight),
    )
    profile.validate(task)
    return profile


def calibrated_rollout_profile_payload(
    profile: CalibratedRolloutProfile,
) -> dict[str, Any]:
    return {
        "state_mean": profile.state_mean,
        "state_scale": profile.state_scale,
        "state_correlation_cholesky": profile.state_correlation_cholesky,
        "predicted_delta_center": profile.predicted_delta_center,
        "predicted_delta_scale": profile.predicted_delta_scale,
        "learned_signal_center": profile.learned_signal_center,
        "standardized_lower": profile.standardized_lower,
        "standardized_upper": profile.standardized_upper,
        "initial_action_probability": profile.initial_action_probability,
        "action_transition_probability": profile.action_transition_probability,
        "initial_mask_probability": profile.initial_mask_probability,
        "mask_transition_probability": profile.mask_transition_probability,
        "length_probability": profile.length_probability,
        "binary_feature_indices": profile.binary_feature_indices,
        "binary_probability": profile.binary_probability,
        "learned_signal_weight": profile.learned_signal_weight,
        "persistence_weight": profile.persistence_weight,
    }


def fit_source_components(
    *,
    task: TaskShape,
    model_family: str,
    seed: int,
    train: PreparedRole,
    calibration: PreparedRole,
    validation: PreparedRole,
    training: dict[str, Any],
    loss_weights: dict[str, float],
    perturbation_seed: int,
    checkpoint_root: Path,
) -> FitResult:
    device = torch.device(training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("frozen CUDA device is unavailable")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    components = build_unfitted_components(
        task,
        model_family,  # type: ignore[arg-type]
        seed,
        hidden_dim=int(training["hidden_dim"]),
        latent_dim=int(training["latent_dim"]),
    )
    initialize_closed_form_components(task, components, train, perturbation_seed)
    for module in components.modules():
        module.to(device)
    parameters = list(_trainable_parameters(components))
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    maximum = int(training["maximum_epochs"])
    minimum = int(training["minimum_epochs"])
    patience = int(training["early_stopping_patience"])
    relative_threshold = float(training["relative_improvement_threshold"])
    batch_size = int(training["batch_size"])
    best_state: dict[str, Any] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    best_torch_rng_state: torch.Tensor | None = None
    best_numpy_rng_state: tuple[Any, ...] | None = None
    best_score = math.inf
    best_epoch = 0
    validation_curve: list[float] = []
    stale = 0
    started = time.perf_counter()
    epochs_run = 0

    for epoch in range(1, maximum + 1):
        components_train(components)
        order = np.random.default_rng(seed * 1000 + epoch).permutation(train.episodes)
        for start in range(0, train.episodes, batch_size):
            indices = order[start : start + batch_size]
            batch = _to_device(train, indices, device)
            optimizer.zero_grad(set_to_none=True)
            losses = component_losses(components, batch, loss_weights)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("nonfinite joint source-model training loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, float(training["gradient_clip_norm"])
            )
            optimizer.step()
        score = _validation_loss(
            components, validation, batch_size, device, loss_weights
        )
        if not np.isfinite(score):
            raise RuntimeError("nonfinite source-model validation loss")
        validation_curve.append(score)
        epochs_run = epoch
        if score < best_score:
            materially_better = (
                not np.isfinite(best_score)
                or (best_score - score) / max(abs(best_score), 1.0e-8)
                >= relative_threshold
            )
            best_score = score
            best_epoch = epoch
            best_state = _state_dict(components)
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_torch_rng_state = torch.get_rng_state().clone()
            best_numpy_rng_state = copy.deepcopy(np.random.get_state())
            stale = 0 if materially_better else stale + 1
        else:
            stale += 1
        if epoch >= minimum and stale >= patience:
            break
    if (
        best_state is None
        or best_optimizer_state is None
        or best_torch_rng_state is None
        or best_numpy_rng_state is None
    ):
        raise RuntimeError("no finite source-model checkpoint")
    _load_state_dict(components, best_state)
    calibration_receipt = apply_fixed_calibration(
        components, calibration, batch_size, device
    )
    window = min(int(training["convergence_window"]), len(validation_curve))
    if epochs_run < maximum:
        convergence_status = "converged_plateau"
    else:
        start_score = validation_curve[-window]
        end_score = min(validation_curve[-window:])
        late_relative = (start_score - end_score) / max(abs(start_score), 1.0e-8)
        convergence_status = (
            "cap_reached_but_qualified"
            if late_relative < relative_threshold
            else "convergence_incomplete"
        )

    logical = f"{task.name}__{model_family}__seed{seed}"
    payload = {
        "schema": training["checkpoint_schema_version"],
        "logical_checkpoint_id": logical,
        "task": task.name,
        "model_family": model_family,
        "seed": seed,
        "selected_epoch": best_epoch,
        "epochs_run": epochs_run,
        "validation_curve": validation_curve,
        "convergence_status": convergence_status,
        "components_state": _state_dict(components),
        "optimizer_state": best_optimizer_state,
        "torch_rng_state": best_torch_rng_state,
        "numpy_rng_state": best_numpy_rng_state,
        "data_order_rule": "numpy_default_rng(seed_times_1000_plus_epoch)",
        "preprocessing_mean": train.preprocessing_mean,
        "preprocessing_scale": train.preprocessing_scale,
        "calibration_receipt": calibration_receipt,
        "task_shape": {
            "feature_dim": task.feature_dim,
            "action_dim": task.action_dim,
            "horizon": task.horizon,
        },
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    content = stream.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (checkpoint_root / f"{logical}.pt").write_bytes(content)
    components_eval(components)
    return FitResult(
        components=components,
        selected_epoch=best_epoch,
        epochs_run=epochs_run,
        validation_curve=validation_curve,
        convergence_status=convergence_status,
        runtime_seconds=time.perf_counter() - started,
        checkpoint_sha256=digest,
        logical_checkpoint_id=logical,
        parameter_count=parameter_count(components.transition)
        + sum(parameter_count(module) for module in components.modules()[2:6]),
        calibration_receipt=calibration_receipt,
    )


def load_source_components(
    checkpoint: Path,
    task: TaskShape,
    model_family: str,
    seed: int,
    hidden_dim: int,
    latent_dim: int,
    device: torch.device,
) -> tuple[SourceSimulatorComponents, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = f"{task.name}__{model_family}__seed{seed}"
    if payload["logical_checkpoint_id"] != expected:
        raise RuntimeError("protected checkpoint logical identity mismatch")
    components = build_unfitted_components(
        task,
        model_family,  # type: ignore[arg-type]
        seed,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )
    _load_state_dict(components, payload["components_state"])
    for module in components.modules():
        module.to(device)
    components_eval(components)
    return components, payload


def recursive_predictions(
    components: SourceSimulatorComponents,
    role: PreparedRole,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    means: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    components_eval(components)
    with torch.inference_mode():
        for start in range(0, role.episodes, batch_size):
            indices = np.arange(start, min(start + batch_size, role.episodes))
            action = torch.from_numpy(role.action_one_hot[indices]).to(device)
            values = torch.from_numpy(role.values[indices, :1]).to(device)
            masks = torch.from_numpy(
                role.masks[indices, :1].astype(np.float32)
            ).to(device)
            deltas = torch.from_numpy(role.deltas[indices, :1]).to(device)
            local_mean: list[torch.Tensor] = []
            local_scale: list[torch.Tensor] = []
            for step in range(role.horizon):
                result = components.transition(
                    values, masks, deltas, action[:, : step + 1]
                )
                current_mean = result.mean[:, -1]
                current_scale = torch.exp(result.log_scale[:, -1])
                local_mean.append(current_mean)
                local_scale.append(current_scale)
                values = torch.cat([values, current_mean[:, None]], dim=1)
                masks = torch.cat(
                    [masks, torch.zeros_like(current_mean[:, None])], dim=1
                )
                deltas = torch.cat([deltas, deltas[:, -1:] + 1.0], dim=1)
            means.append(torch.stack(local_mean, dim=1).cpu().numpy())
            scales.append(torch.stack(local_scale, dim=1).cpu().numpy())
    return np.concatenate(means), np.concatenate(scales)


def gaussian_metric_summary(
    truth: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    observed_truth = truth[mask]
    observed_mean = mean[mask]
    observed_scale = np.maximum(scale[mask], 1.0e-5)
    if observed_truth.size == 0:
        return {
            "rmse": math.nan,
            "mae": math.nan,
            "nll": math.nan,
            "mace": math.nan,
            "sharpness": math.nan,
            "observed_cells": 0,
        }
    error = observed_truth - observed_mean
    z = error / observed_scale
    deviations = []
    for level, quantile in (
        (0.50, 0.67448975),
        (0.80, 1.28155157),
        (0.90, 1.64485363),
        (0.95, 1.95996398),
    ):
        deviations.append(abs(float((np.abs(z) <= quantile).mean()) - level))
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "nll": float(
            np.mean(
                np.log(observed_scale)
                + 0.5 * np.square(z)
                + 0.5 * math.log(2.0 * math.pi)
            )
        ),
        "mace": float(np.mean(deviations)),
        "sharpness": float(np.mean(observed_scale)),
        "observed_cells": int(observed_truth.size),
    }


def uncertainty_error_summary(
    truth: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    from scipy.stats import spearmanr

    error = np.abs(truth[mask] - mean[mask])
    uncertainty = scale[mask]
    if error.size < 3:
        return {
            "spearman_error_scale": math.nan,
            "risk_coverage_auc": math.nan,
            "cells": int(error.size),
        }
    correlation = float(spearmanr(error, uncertainty).statistic)
    order = np.argsort(uncertainty)
    coverage = np.linspace(0.1, 1.0, 10)
    risks = []
    for fraction in coverage:
        keep = order[: max(1, int(round(len(order) * fraction)))]
        risks.append(float(np.sqrt(np.mean(np.square(error[keep])))))
    return {
        "spearman_error_scale": correlation,
        "risk_coverage_auc": float(np.trapezoid(risks, coverage) / 0.9),
        "cells": int(error.size),
    }
