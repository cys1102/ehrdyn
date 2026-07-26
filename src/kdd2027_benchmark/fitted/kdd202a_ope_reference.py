"""Executable aggregate-safe reference contract for the frozen KDD172 OPE inventory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ESTIMATORS = (
    "IS", "WIS", "PDIS", "WPDIS", "CWPDIS", "DR", "WDR", "FQE",
    "support_restricted_WPDIS",
)
TIE_TOLERANCE = 1e-12
DENOMINATOR_FLOOR = 1e-12


@dataclass(frozen=True)
class OPEInputs:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    target_policy: np.ndarray
    denominator_full: np.ndarray
    q: np.ndarray
    v: np.ndarray
    discount: float
    support: np.ndarray
    absorbing_state: int

    def validate(self) -> None:
        n, h = self.actions.shape
        if self.states.shape != (n, h + 1) or self.rewards.shape != (n, h):
            raise ValueError("trajectory shape mismatch")
        if self.denominator_full.shape[:2] != (n, h):
            raise ValueError("denominator shape mismatch")
        if self.target_policy.shape[0] < h or self.q.shape[0] < h or self.v.shape[0] < h + 1:
            raise ValueError("horizon mismatch")
        if not np.isfinite(self.rewards).all():
            raise ValueError("nonfinite reward")


def reference_estimates(inputs: OPEInputs, clip: float | None = None) -> tuple[dict[str, float], dict[str, float]]:
    """Match the frozen KDD172 point-estimator ordering and zero-weight rules."""
    inputs.validate()
    states, actions, rewards = inputs.states, inputs.actions, inputs.rewards
    n, horizon = actions.shape
    target_probability = np.empty((n, horizon), dtype=np.float64)
    for t in range(horizon):
        target_probability[:, t] = inputs.target_policy[t, states[:, t], actions[:, t]]
    denominator = np.take_along_axis(inputs.denominator_full, actions[:, :, None], axis=2)[:, :, 0]
    ratios = np.divide(target_probability, denominator, out=np.zeros_like(target_probability), where=denominator > 0)
    if clip is not None:
        if clip <= 0:
            raise ValueError("clip must be positive")
        ratios = np.minimum(ratios, float(clip))
    cumulative = np.cumprod(ratios, axis=1)
    discounts = inputs.discount ** np.arange(horizon)
    returns = rewards @ discounts
    final = cumulative[:, -1]
    final_sum = float(final.sum())
    estimates: dict[str, float] = {
        "IS": float(np.mean(final * returns)),
        "WIS": float(np.sum(final * returns) / max(final_sum, DENOMINATOR_FLOOR)),
        "PDIS": float(np.mean(np.sum(cumulative * rewards * discounts, axis=1))),
    }
    step_denominator = cumulative.sum(axis=0)
    estimates["WPDIS"] = float(np.sum(np.divide(
        np.sum(cumulative * rewards, axis=0), step_denominator,
        out=np.zeros(horizon), where=step_denominator > 0,
    ) * discounts))
    at_risk = states[:, :-1] != inputs.absorbing_state
    censored_weight = cumulative * at_risk
    censored_denominator = censored_weight.sum(axis=0)
    estimates["CWPDIS"] = float(np.sum(np.divide(
        np.sum(censored_weight * rewards, axis=0), censored_denominator,
        out=np.zeros(horizon), where=censored_denominator > 0,
    ) * discounts))
    dr_contribution = inputs.v[0, states[:, 0]].copy()
    for t in range(horizon):
        residual = (
            rewards[:, t]
            + inputs.discount * inputs.v[t + 1, states[:, t + 1]]
            - inputs.q[t, states[:, t], actions[:, t]]
        )
        dr_contribution += discounts[t] * cumulative[:, t] * residual
    estimates["DR"] = float(dr_contribution.mean())
    wdr = float(inputs.v[0, states[:, 0]].mean())
    for t in range(horizon):
        residual = (
            rewards[:, t]
            + inputs.discount * inputs.v[t + 1, states[:, t + 1]]
            - inputs.q[t, states[:, t], actions[:, t]]
        )
        denominator_t = float(cumulative[:, t].sum())
        wdr += float(discounts[t] * np.sum(cumulative[:, t] * residual) / max(denominator_t, DENOMINATOR_FLOOR))
    estimates["WDR"] = wdr
    estimates["FQE"] = float(inputs.v[0, states[:, 0]].mean())
    support_broadcast = np.broadcast_to(inputs.support, inputs.target_policy.shape)
    supported = not bool(np.any(inputs.target_policy[~support_broadcast] > TIE_TOLERANCE))
    estimates["support_restricted_WPDIS"] = estimates["WPDIS"] if supported else float("nan")
    diagnostics = {
        "ess": float(final_sum**2 / max(float(np.square(final).sum()), DENOMINATOR_FLOOR)),
        "unsupported_target_mass": float(inputs.target_policy[~support_broadcast].sum()),
        "finite_fraction": float(np.mean([np.isfinite(value) for value in estimates.values()])),
    }
    return estimates, diagnostics


def full_refit_bootstrap(
    data: dict[str, np.ndarray],
    target_policy: np.ndarray,
    support: np.ndarray,
    absorbing_state: int,
    discount: float,
    replicate_indices: list[np.ndarray],
    denominator_fit: Callable[[dict[str, np.ndarray]], np.ndarray],
    qv_fit: Callable[[dict[str, np.ndarray]], tuple[np.ndarray, np.ndarray]],
    fqe_fit: Callable[[dict[str, np.ndarray]], tuple[np.ndarray, np.ndarray]],
    clip: float | None = None,
) -> dict[str, np.ndarray]:
    """Refit every declared nuisance inside every supplied bootstrap replicate."""
    samples = {name: np.full(len(replicate_indices), np.nan) for name in ESTIMATORS}
    for replicate, indices in enumerate(replicate_indices):
        local = {name: np.asarray(value)[indices].copy() for name, value in data.items()}
        denominator = denominator_fit(local)
        q, v = qv_fit(local)
        fqe_q, fqe_v = fqe_fit(local)
        inputs = OPEInputs(
            local["states"], local["actions"], local["rewards"], target_policy,
            denominator, q, v, discount, support, absorbing_state,
        )
        estimates, _ = reference_estimates(inputs, clip)
        fqe_inputs = OPEInputs(
            local["states"], local["actions"], local["rewards"], target_policy,
            denominator, fqe_q, fqe_v, discount, support, absorbing_state,
        )
        estimates["FQE"] = reference_estimates(fqe_inputs, clip)[0]["FQE"]
        for name in ESTIMATORS:
            samples[name][replicate] = estimates[name]
    return samples


def rank_pairwise_and_sign(
    truth: np.ndarray, estimate: np.ndarray, behavior_index: int = 0,
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    finite = np.isfinite(estimate)
    pairs = [(i, j) for i in range(len(truth)) for j in range(i + 1, len(truth))
             if finite[i] and finite[j] and abs(truth[i] - truth[j]) > TIE_TOLERANCE]
    pairwise = float(np.mean([np.sign(truth[i] - truth[j]) == np.sign(estimate[i] - estimate[j])
                              for i, j in pairs])) if pairs else float("nan")
    nonbehavior = np.arange(len(truth)) != behavior_index
    nonzero = nonbehavior & (np.abs(truth - truth[behavior_index]) > TIE_TOLERANCE) & finite
    sign = float(np.mean(np.sign(estimate[nonzero] - estimate[behavior_index])
                         == np.sign(truth[nonzero] - truth[behavior_index]))) if np.any(nonzero) else float("nan")
    return {"pairwise_order_recovery": pairwise, "behavior_relative_sign_recovery": sign}
