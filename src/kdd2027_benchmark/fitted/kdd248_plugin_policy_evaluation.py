"""Separate-data policy and evaluator utilities for a plug-in simulator.

The policies in this pilot use only time and the previously recorded action.
This deliberately small state contract makes source/evaluator separation
auditable while still allowing a generated-data-trained policy to differ from
fixed and behavior controls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class MarkovPolicy:
    name: str
    initial_probability: np.ndarray
    transition_probability: np.ndarray
    fit_role: str

    @property
    def horizon(self) -> int:
        return int(self.transition_probability.shape[0])

    @property
    def action_count(self) -> int:
        return int(self.transition_probability.shape[-1])

    def validate(self) -> None:
        if self.initial_probability.shape != (self.action_count,):
            raise ValueError("policy initial probability shape mismatch")
        if self.transition_probability.shape != (
            self.horizon,
            self.action_count,
            self.action_count,
        ):
            raise ValueError("policy transition probability shape mismatch")
        if not np.isfinite(self.initial_probability).all() or not np.isfinite(
            self.transition_probability
        ).all():
            raise ValueError("policy probabilities must be finite")
        if np.any(self.initial_probability < 0.0) or np.any(
            self.transition_probability < 0.0
        ):
            raise ValueError("policy probabilities must be nonnegative")
        if not np.isclose(self.initial_probability.sum(), 1.0) or not np.allclose(
            self.transition_probability.sum(axis=-1), 1.0
        ):
            raise ValueError("policy probabilities must be normalized")


def fixed_policy(
    name: str,
    action: int,
    action_count: int,
    horizon: int,
) -> MarkovPolicy:
    probability = np.zeros((horizon, action_count, action_count), dtype=float)
    probability[..., action] = 1.0
    initial = np.zeros(action_count, dtype=float)
    initial[action] = 1.0
    policy = MarkovPolicy(name, initial, probability, "fixed_control")
    policy.validate()
    return policy


def random_policy(action_count: int, horizon: int) -> MarkovPolicy:
    initial = np.full(action_count, 1.0 / action_count)
    transition = np.full(
        (horizon, action_count, action_count), 1.0 / action_count
    )
    policy = MarkovPolicy(
        "supported_random", initial, transition, "fixed_control"
    )
    policy.validate()
    return policy


def behavior_policy(
    initial_probability: np.ndarray,
    transition_probability: np.ndarray,
    horizon: int,
) -> MarkovPolicy:
    source = np.asarray(transition_probability, dtype=float)
    if source.ndim == 2:
        transition = np.broadcast_to(
            source[None],
            (horizon, *source.shape),
        ).copy()
    elif (
        source.ndim == 3
        and source.shape[0] == horizon
        and source.shape[1] == source.shape[2]
    ):
        transition = source.copy()
    else:
        raise ValueError("behavior transition shape mismatch")
    transition[0] = np.broadcast_to(
        np.asarray(initial_probability, dtype=float),
        transition[0].shape,
    )
    policy = MarkovPolicy(
        "fitted_behavior",
        np.asarray(initial_probability, dtype=float).copy(),
        transition,
        "source_model_calibration_control",
    )
    policy.validate()
    return policy


def previous_actions(batch: Any, fallback: int = 0) -> np.ndarray:
    previous = np.full_like(batch.actions, fallback, dtype=np.int64)
    previous[:, 1:] = np.maximum(batch.actions[:, :-1], 0)
    return previous


def fit_markov_q_policy(
    batch: Any,
    *,
    initial_probability: np.ndarray,
    action_count: int,
    discount: float,
    temperature: float,
    behavior_transition: np.ndarray,
    behavior_blend: float,
    name: str,
) -> MarkovPolicy:
    if temperature <= 0.0 or not 0.0 <= behavior_blend <= 1.0:
        raise ValueError("invalid policy temperature or behavior blend")
    horizon = batch.actions.shape[1]
    previous = previous_actions(batch)
    reward_sum = np.zeros((horizon, action_count, action_count), dtype=float)
    reward_count = np.zeros_like(reward_sum)
    termination_sum = np.zeros_like(reward_sum)
    for step in range(horizon):
        keep = batch.valid_steps[:, step]
        state = previous[keep, step]
        action = batch.actions[keep, step].astype(int)
        np.add.at(reward_sum[step], (state, action), batch.rewards[keep, step])
        np.add.at(reward_count[step], (state, action), 1.0)
        np.add.at(
            termination_sum[step],
            (state, action),
            batch.terminations[keep, step].astype(float),
        )
    global_reward = float(np.nanmean(batch.rewards[batch.valid_steps]))
    reward = np.divide(
        reward_sum,
        reward_count,
        out=np.full_like(reward_sum, global_reward),
        where=reward_count > 0,
    )
    termination = np.divide(
        termination_sum,
        reward_count,
        out=np.ones_like(termination_sum),
        where=reward_count > 0,
    )
    q_value = np.zeros_like(reward)
    value = np.zeros((horizon + 1, action_count), dtype=float)
    raw_policy = np.zeros_like(reward)
    for step in range(horizon - 1, -1, -1):
        q_value[step] = reward[step] + discount * (
            1.0 - termination[step]
        ) * value[step + 1][None, :]
        centered = q_value[step] - q_value[step].max(axis=1, keepdims=True)
        local = np.exp(np.clip(centered / temperature, -30.0, 0.0))
        local /= local.sum(axis=1, keepdims=True)
        behavior_source = np.asarray(behavior_transition)
        behavior_step = (
            behavior_source[step]
            if behavior_source.ndim == 3
            else behavior_source
        )
        raw_policy[step] = (
            (1.0 - behavior_blend) * local
            + behavior_blend * behavior_step
        )
        value[step] = np.sum(raw_policy[step] * q_value[step], axis=1)
    initial = raw_policy[0].T @ np.asarray(initial_probability, dtype=float)
    initial /= initial.sum()
    raw_policy[0] = np.broadcast_to(initial, raw_policy[0].shape)
    policy = MarkovPolicy(
        name,
        initial,
        raw_policy,
        "policy_train",
    )
    policy.validate()
    return policy


def sample_policy_actions(
    policy: MarkovPolicy,
    episodes: int,
    seed: int,
) -> np.ndarray:
    policy.validate()
    rng = np.random.default_rng(seed)
    action = np.empty((episodes, policy.horizon), dtype=np.int64)
    initial_probability = np.asarray(
        policy.initial_probability, dtype=np.float64
    )
    initial_probability /= initial_probability.sum()
    action[:, 0] = rng.choice(
        policy.action_count,
        size=episodes,
        p=initial_probability,
    )
    for step in range(1, policy.horizon):
        probability = np.asarray(
            policy.transition_probability[
            step, action[:, step - 1]
            ],
            dtype=np.float64,
        )
        probability /= probability.sum(axis=1, keepdims=True)
        uniforms = rng.random(episodes)
        cumulative = np.cumsum(probability, axis=1)
        cumulative[:, -1] = 1.0
        action[:, step] = (uniforms[:, None] > cumulative).sum(axis=1)
    return action


def target_probabilities(batch: Any, policy: MarkovPolicy) -> np.ndarray:
    policy.validate()
    episodes, horizon = batch.actions.shape
    if horizon != policy.horizon:
        raise ValueError("policy and logged horizon differ")
    output = np.empty(
        (episodes, horizon, policy.action_count), dtype=np.float64
    )
    output[:, 0] = policy.initial_probability
    for step in range(1, horizon):
        previous = np.maximum(batch.actions[:, step - 1], 0).astype(int)
        output[:, step] = policy.transition_probability[step, previous]
    return output


def importance_estimates(
    batch: Any,
    target_probability: np.ndarray,
    discount: float,
    clip: float | None,
) -> tuple[dict[str, float], dict[str, float]]:
    valid = batch.valid_steps
    chosen_target = np.take_along_axis(
        target_probability, np.maximum(batch.actions, 0)[..., None], axis=2
    )[..., 0]
    chosen_behavior = np.take_along_axis(
        batch.action_probabilities,
        np.maximum(batch.actions, 0)[..., None],
        axis=2,
    )[..., 0]
    ratio = np.divide(
        chosen_target,
        chosen_behavior,
        out=np.zeros_like(chosen_target),
        where=chosen_behavior > 0,
    )
    if clip is not None:
        ratio = np.minimum(ratio, float(clip))
    ratio[~valid] = 1.0
    cumulative = np.cumprod(ratio, axis=1)
    weights = discount ** np.arange(batch.actions.shape[1])
    safe_rewards = np.where(valid, batch.rewards, 0.0)
    returns = np.sum(safe_rewards * weights[None], axis=1)
    final_index = np.maximum(valid.sum(axis=1) - 1, 0)
    final = cumulative[np.arange(len(final_index)), final_index]
    denominator = max(float(final.sum()), 1.0e-12)
    at_risk = cumulative * valid
    step_denominator = at_risk.sum(axis=0)
    estimates = {
        "IS": float(np.mean(final * returns)),
        "WIS": float(np.sum(final * returns) / denominator),
        "CWPDIS": float(
            np.sum(
                np.divide(
                    np.sum(at_risk * safe_rewards, axis=0),
                    step_denominator,
                    out=np.zeros(batch.actions.shape[1]),
                    where=step_denominator > 0,
                )
                * weights
            )
        ),
    }
    diagnostics = {
        "ess": float(
            denominator**2 / max(float(np.square(final).sum()), 1.0e-12)
        ),
        "finite_fraction": float(
            np.mean([np.isfinite(value) for value in estimates.values()])
        ),
        "maximum_ratio": float(np.max(ratio[valid])),
    }
    return estimates, diagnostics


def fit_markov_evaluator(
    batch: Any,
    policy: MarkovPolicy,
    discount: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit an independent tabular FQE using time and previous action."""
    action_count = policy.action_count
    horizon = policy.horizon
    previous = previous_actions(batch)
    reward_sum = np.zeros((horizon, action_count, action_count), dtype=float)
    count = np.zeros_like(reward_sum)
    termination_sum = np.zeros_like(reward_sum)
    for step in range(horizon):
        keep = batch.valid_steps[:, step]
        state = previous[keep, step]
        action = batch.actions[keep, step].astype(int)
        np.add.at(reward_sum[step], (state, action), batch.rewards[keep, step])
        np.add.at(count[step], (state, action), 1.0)
        np.add.at(
            termination_sum[step],
            (state, action),
            batch.terminations[keep, step].astype(float),
        )
    global_reward = float(np.nanmean(batch.rewards[batch.valid_steps]))
    reward = np.divide(
        reward_sum,
        count,
        out=np.full_like(reward_sum, global_reward),
        where=count > 0,
    )
    termination = np.divide(
        termination_sum,
        count,
        out=np.ones_like(termination_sum),
        where=count > 0,
    )
    q_value = np.zeros_like(reward)
    value = np.zeros((horizon + 1, action_count), dtype=float)
    for step in range(horizon - 1, -1, -1):
        q_value[step] = reward[step] + discount * (
            1.0 - termination[step]
        ) * value[step + 1][None, :]
        value[step] = np.sum(
            policy.transition_probability[step] * q_value[step], axis=1
        )
    estimate = float(value[0, 0])
    return estimate, q_value, value


def doubly_robust_estimate(
    batch: Any,
    policy: MarkovPolicy,
    q_value: np.ndarray,
    value: np.ndarray,
    discount: float,
    clip: float | None,
) -> float:
    target = target_probabilities(batch, policy)
    chosen_target = np.take_along_axis(
        target, np.maximum(batch.actions, 0)[..., None], axis=2
    )[..., 0]
    chosen_behavior = np.take_along_axis(
        batch.action_probabilities,
        np.maximum(batch.actions, 0)[..., None],
        axis=2,
    )[..., 0]
    ratio = np.divide(
        chosen_target,
        chosen_behavior,
        out=np.zeros_like(chosen_target),
        where=chosen_behavior > 0,
    )
    if clip is not None:
        ratio = np.minimum(ratio, clip)
    ratio[~batch.valid_steps] = 1.0
    cumulative = np.cumprod(ratio, axis=1)
    previous = previous_actions(batch)
    result = np.full(
        batch.actions.shape[0],
        float(value[0, 0]),
    )
    for step in range(policy.horizon):
        keep = batch.valid_steps[:, step]
        state = previous[:, step]
        action = np.maximum(batch.actions[:, step], 0).astype(int)
        next_value = np.zeros(len(result))
        continuation = keep & ~batch.terminations[:, step]
        next_state = action
        next_value[continuation] = value[step + 1, next_state[continuation]]
        residual = (
            np.where(keep, batch.rewards[:, step], 0.0)
            + discount * next_value
            - q_value[step, state, action]
        )
        result += (
            discount**step * cumulative[:, step] * residual * keep
        )
    return float(np.mean(result))
