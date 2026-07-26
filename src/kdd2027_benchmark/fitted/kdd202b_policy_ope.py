"""Observed-history OPE utilities for the frozen KDD202B constructed-POMDP contract."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .kdd202a_ope_reference import DENOMINATOR_FLOOR, TIE_TOLERANCE


Policy = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]
ESTIMATORS = ("IS", "WIS", "CWPDIS", "DR", "WDR", "FQE")


@dataclass(frozen=True)
class LoggedOPEData:
    observed: np.ndarray
    masks: np.ndarray
    recency: np.ndarray
    previous_actions: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    valid: np.ndarray
    exact_behavior: np.ndarray
    support: np.ndarray
    discount: float
    absorbing_state: int

    @property
    def episodes(self) -> int:
        return int(self.actions.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[1])

    @property
    def action_count(self) -> int:
        return int(self.exact_behavior.shape[2])

    def subset(self, indices: np.ndarray) -> "LoggedOPEData":
        return LoggedOPEData(
            self.observed[indices], self.masks[indices], self.recency[indices],
            self.previous_actions[indices], self.states[indices], self.actions[indices],
            self.rewards[indices], self.valid[indices], self.exact_behavior[indices],
            self.support, self.discount, self.absorbing_state,
        )


def collect_observed_history_dataset(env: Any, episodes: int, seed: int, behavior_family: str) -> LoggedOPEData:
    from .kdd199_repaired_direct_evaluator import collect_repaired_dataset

    data = collect_repaired_dataset(env, episodes, seed, behavior_family)
    streams = env._streams(episodes, seed)
    marginal = np.asarray(env.contract.target_action_frequency, dtype=float)[env.supported]
    marginal /= marginal.sum()
    initial_previous = env.supported[env._draw(
        np.broadcast_to(marginal, (episodes, len(env.supported))), streams["prior_u"]
    )]
    previous = np.empty_like(data.actions)
    previous[:, 0] = initial_previous
    previous[:, 1:] = data.actions[:, :-1]
    context = np.zeros_like(data.actions, dtype=np.int32)
    for step in range(env.horizon):
        context[:, step] = env.behavior.context_bin(
            data.observed[:, step], data.masks[:, step], data.deltas[:, step],
            previous[:, step], step,
        )
    observed_states = int(context.max()) + 1
    absorbing = observed_states
    states = np.full((episodes, env.horizon + 1), absorbing, dtype=np.int32)
    states[:, :-1] = context
    for step in range(env.horizon - 1):
        carry = data.valid[:, step] & ~data.done[:, step]
        states[carry, step + 1] = context[carry, step + 1]
    support = np.zeros(env.contract.action_count, dtype=bool)
    support[env.supported] = True
    return LoggedOPEData(
        data.observed, data.masks, data.deltas, previous, states, data.actions,
        data.rewards, data.valid, data.behavior_probability, support,
        float(env.discount), absorbing,
    )


def target_probabilities(data: LoggedOPEData, policy: Policy) -> np.ndarray:
    probabilities = np.zeros((data.episodes, data.horizon, data.action_count), dtype=np.float64)
    fallback = int(np.flatnonzero(data.support)[0])
    for step in range(data.horizon):
        local = np.asarray(policy(
            data.observed[:, step], data.masks[:, step], data.recency[:, step],
            data.previous_actions[:, step], step,
        ), dtype=float)
        if local.shape != (data.episodes, data.action_count):
            raise ValueError("target policy shape mismatch")
        if not np.isfinite(local).all() or np.any(local < -1e-12):
            raise ValueError("target policy finite/nonnegative failure")
        if np.max(np.abs(local.sum(axis=1) - 1.0)) > 1e-8:
            raise ValueError("target policy normalization failure")
        local[:, ~data.support] = 0.0
        normalizer = local.sum(axis=1, keepdims=True)
        if np.any(normalizer <= 0):
            raise ValueError("target policy support failure")
        local /= normalizer
        invalid = ~data.valid[:, step]
        local[invalid] = 0.0
        local[invalid, fallback] = 1.0
        probabilities[:, step] = local
    return probabilities


def denominator_probabilities(
    data: LoggedOPEData,
    name: str,
    folds: int = 5,
    pseudocount: float = 0.5,
    fold_seed: int = 0,
) -> np.ndarray:
    if name == "exact_behavior":
        return np.asarray(data.exact_behavior, dtype=np.float64).copy()
    if name == "misspecified_behavior":
        transformed = np.power(np.asarray(data.exact_behavior, dtype=float), 0.55)
        transformed[:, :, ~data.support] = 0.0
        return transformed / transformed.sum(axis=2, keepdims=True)
    if name != "crossfit_stronger":
        raise ValueError(f"unknown denominator {name}")
    n, horizon = data.actions.shape
    rng = np.random.default_rng(fold_seed)
    assignment = np.arange(n) % folds
    rng.shuffle(assignment)
    output = np.zeros((n, horizon, data.action_count), dtype=np.float64)
    states_n = data.absorbing_state
    supported = np.flatnonzero(data.support)
    for fold in range(folds):
        fit = assignment != fold
        predict = assignment == fold
        for step in range(horizon):
            counts = np.zeros((states_n, data.action_count), dtype=float)
            counts[:, supported] = pseudocount
            rows = fit & data.valid[:, step]
            np.add.at(counts, (data.states[rows, step], data.actions[rows, step]), 1.0)
            counts[:, ~data.support] = 0.0
            counts /= counts.sum(axis=1, keepdims=True)
            output[predict, step] = counts[data.states[predict, step]]
    invalid = ~data.valid
    output[invalid] = 0.0
    output[invalid, supported[0]] = 1.0
    return output


def _state_policy(data: LoggedOPEData, target: np.ndarray) -> np.ndarray:
    states_n = data.absorbing_state + 1
    policy = np.zeros((data.horizon, states_n, data.action_count), dtype=float)
    fallback = np.zeros(data.action_count, dtype=float)
    fallback[data.support] = 1.0 / data.support.sum()
    for step in range(data.horizon):
        for state in range(data.absorbing_state):
            rows = data.valid[:, step] & (data.states[:, step] == state)
            policy[step, state] = target[rows, step].mean(axis=0) if np.any(rows) else fallback
        policy[step, data.absorbing_state, int(np.flatnonzero(data.support)[0])] = 1.0
    return policy


def fit_dynamics(
    data: LoggedOPEData,
    pseudocount: float,
) -> tuple[np.ndarray, np.ndarray]:
    states_n = data.absorbing_state + 1
    horizon, actions_n = data.horizon, data.action_count
    rewards = np.zeros((horizon, states_n, actions_n), dtype=float)
    transitions = np.zeros((horizon, states_n, actions_n, states_n), dtype=float)
    for step in range(horizon):
        reward_sum = np.zeros((states_n, actions_n), dtype=float)
        count = np.zeros((states_n, actions_n), dtype=float)
        transition = np.zeros((states_n, actions_n, states_n), dtype=float)
        rows = np.flatnonzero(data.valid[:, step])
        s = data.states[rows, step]
        a = data.actions[rows, step]
        ns = data.states[rows, step + 1]
        np.add.at(reward_sum, (s, a), data.rewards[rows, step])
        np.add.at(count, (s, a), 1.0)
        np.add.at(transition, (s, a, ns), 1.0)
        reward_mean = np.divide(reward_sum, count, out=np.zeros_like(reward_sum), where=count > 0)
        transition[:, :, data.absorbing_state] += pseudocount
        transition /= transition.sum(axis=2, keepdims=True)
        rewards[step] = reward_mean
        transitions[step] = transition
    return rewards, transitions


def qv_from_dynamics(
    data: LoggedOPEData,
    target: np.ndarray,
    dynamics: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    states_n = data.absorbing_state + 1
    horizon, actions_n = data.horizon, data.action_count
    rewards, transitions = dynamics
    pi = _state_policy(data, target)
    q = np.zeros((horizon, states_n, actions_n), dtype=float)
    v = np.zeros((horizon + 1, states_n), dtype=float)
    for step in range(horizon - 1, -1, -1):
        q[step] = rewards[step] + data.discount * np.einsum(
            "sak,k->sa", transitions[step], v[step + 1]
        )
        q[step, data.absorbing_state] = 0.0
        v[step] = np.sum(pi[step] * q[step], axis=1)
    return q, v


def fit_qv(
    data: LoggedOPEData,
    target: np.ndarray,
    pseudocount: float,
) -> tuple[np.ndarray, np.ndarray]:
    return qv_from_dynamics(data, target, fit_dynamics(data, pseudocount))


def estimates(
    data: LoggedOPEData,
    target: np.ndarray,
    denominator: np.ndarray,
    clip: float | None,
    pseudocount: float,
    dynamics: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    chosen_target = np.take_along_axis(target, data.actions[:, :, None], axis=2)[:, :, 0]
    chosen_denominator = np.take_along_axis(denominator, data.actions[:, :, None], axis=2)[:, :, 0]
    ratio = np.divide(chosen_target, chosen_denominator, out=np.zeros_like(chosen_target), where=chosen_denominator > 0)
    if clip is not None:
        ratio = np.minimum(ratio, float(clip))
    ratio[~data.valid] = 1.0
    cumulative = np.cumprod(ratio, axis=1)
    discounts = data.discount ** np.arange(data.horizon)
    returns = np.sum(data.rewards * data.valid * discounts, axis=1)
    final_index = np.maximum(data.valid.sum(axis=1) - 1, 0)
    final = cumulative[np.arange(data.episodes), final_index]
    final_sum = float(final.sum())
    result = {
        "IS": float(np.mean(final * returns)),
        "WIS": float(np.sum(final * returns) / max(final_sum, DENOMINATOR_FLOOR)),
    }
    at_risk_weight = cumulative * data.valid
    denominator_t = at_risk_weight.sum(axis=0)
    result["CWPDIS"] = float(np.sum(np.divide(
        np.sum(at_risk_weight * data.rewards, axis=0), denominator_t,
        out=np.zeros(data.horizon), where=denominator_t > 0,
    ) * discounts))
    shared_dynamics = dynamics if dynamics is not None else fit_dynamics(data, pseudocount)
    q, v = qv_from_dynamics(data, target, shared_dynamics)
    dr = v[0, data.states[:, 0]].copy()
    wdr = float(v[0, data.states[:, 0]].mean())
    for step in range(data.horizon):
        residual = data.rewards[:, step] + data.discount * v[step + 1, data.states[:, step + 1]] - q[step, data.states[:, step], data.actions[:, step]]
        residual *= data.valid[:, step]
        dr += discounts[step] * cumulative[:, step] * residual
        total = float((cumulative[:, step] * data.valid[:, step]).sum())
        wdr += discounts[step] * float(np.sum(cumulative[:, step] * residual)) / max(total, DENOMINATOR_FLOOR)
    result["DR"] = float(dr.mean())
    result["WDR"] = float(wdr)
    _, fqe_v = qv_from_dynamics(data, target, shared_dynamics)
    result["FQE"] = float(fqe_v[0, data.states[:, 0]].mean())
    unsupported = np.broadcast_to(~data.support, target.shape)
    diagnostics = {
        "ess": float(final_sum**2 / max(float(np.square(final).sum()), DENOMINATOR_FLOOR)),
        "unsupported_mass": float(target[unsupported].sum() / max(int(data.valid.sum()), 1)),
        "finite_fraction": float(np.mean([np.isfinite(result[name]) for name in ESTIMATORS])),
    }
    return result, diagnostics


def method_estimates(
    data: LoggedOPEData,
    target_members: list[np.ndarray],
    denominator_name: str,
    clip: float | None,
    folds: int,
    denominator_pseudocount: float,
    nuisance_pseudocount: float,
    fold_seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    denominator = denominator_probabilities(data, denominator_name, folds, denominator_pseudocount, fold_seed)
    member_results = []
    member_diagnostics = []
    for target in target_members:
        local, diagnostic = estimates(data, target, denominator, clip, nuisance_pseudocount)
        member_results.append(local)
        member_diagnostics.append(diagnostic)
    result = {name: float(np.mean([row[name] for row in member_results])) for name in ESTIMATORS}
    diagnostics = {
        "ess": float(np.median([row["ess"] for row in member_diagnostics])),
        "unsupported_mass": float(np.mean([row["unsupported_mass"] for row in member_diagnostics])),
        "finite_fraction": float(np.mean([row["finite_fraction"] for row in member_diagnostics])),
    }
    return result, diagnostics


def bootstrap_method_estimates(
    data: LoggedOPEData,
    target_members: list[np.ndarray],
    replicates: int,
    seed: int,
    denominator_name: str,
    clip: float | None,
    folds: int,
    denominator_pseudocount: float,
    nuisance_pseudocount: float,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    output = {name: np.full(replicates, np.nan) for name in ESTIMATORS}
    for replicate in range(replicates):
        indices = rng.integers(0, data.episodes, data.episodes)
        local = data.subset(indices)
        targets = [target[indices] for target in target_members]
        result, _ = method_estimates(
            local, targets, denominator_name, clip, folds,
            denominator_pseudocount, nuisance_pseudocount, seed + replicate + 1,
        )
        for estimator in ESTIMATORS:
            output[estimator][replicate] = result[estimator]
    return output


def point_policy_groups(
    data: LoggedOPEData,
    policy_groups: dict[str, list[np.ndarray]],
    denominator_name: str,
    clip: float | None,
    folds: int,
    denominator_pseudocount: float,
    nuisance_pseudocount: float,
    fold_seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Evaluate all method groups while fitting one shared denominator."""
    denominator = denominator_probabilities(
        data, denominator_name, folds, denominator_pseudocount, fold_seed
    )
    dynamics = fit_dynamics(data, nuisance_pseudocount)
    results: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    for method, members in policy_groups.items():
        local_results, local_diagnostics = [], []
        for target in members:
            result, diagnostic = estimates(
                data, target, denominator, clip, nuisance_pseudocount, dynamics
            )
            local_results.append(result)
            local_diagnostics.append(diagnostic)
        results[method] = {
            estimator: float(np.mean([row[estimator] for row in local_results]))
            for estimator in ESTIMATORS
        }
        diagnostics[method] = {
            "ess": float(np.median([row["ess"] for row in local_diagnostics])),
            "unsupported_mass": float(np.mean([row["unsupported_mass"] for row in local_diagnostics])),
            "finite_fraction": float(np.mean([row["finite_fraction"] for row in local_diagnostics])),
        }
    return results, diagnostics


def bootstrap_policy_groups(
    data: LoggedOPEData,
    policy_groups: dict[str, list[np.ndarray]],
    replicates: int,
    seed: int,
    denominator_name: str,
    clip: float | None,
    folds: int,
    denominator_pseudocount: float,
    nuisance_pseudocount: float,
    workers: int = 1,
) -> dict[str, dict[str, np.ndarray]]:
    """Full-refit episode bootstrap with one denominator refit per replicate."""
    rng = np.random.default_rng(seed)
    replicate_indices = [
        rng.integers(0, data.episodes, data.episodes) for _ in range(replicates)
    ]
    if workers > 1 and replicates > 1:
        chunks = [
            chunk.tolist()
            for chunk in np.array_split(
                np.arange(replicates), min(workers, replicates)
            )
            if len(chunk)
        ]
        arguments = [(
            data,
            policy_groups,
            [(replicate, replicate_indices[replicate]) for replicate in chunk],
            seed,
            denominator_name,
            clip,
            folds,
            denominator_pseudocount,
            nuisance_pseudocount,
        ) for chunk in chunks]
        with ProcessPoolExecutor(max_workers=len(arguments)) as pool:
            parts = list(pool.map(_bootstrap_chunk, arguments))
        output = {
            method: {name: np.full(replicates, np.nan) for name in ESTIMATORS}
            for method in policy_groups
        }
        for part in parts:
            for replicate, method_rows in part:
                for method, result in method_rows.items():
                    for estimator in ESTIMATORS:
                        output[method][estimator][replicate] = result[estimator]
        return output
    output = {
        method: {name: np.full(replicates, np.nan) for name in ESTIMATORS}
        for method in policy_groups
    }
    for replicate in range(replicates):
        indices = replicate_indices[replicate]
        local = data.subset(indices)
        denominator = denominator_probabilities(
            local, denominator_name, folds, denominator_pseudocount,
            seed + replicate + 1,
        )
        dynamics = fit_dynamics(local, nuisance_pseudocount)
        for method, members in policy_groups.items():
            member_results = []
            for target in members:
                result, _ = estimates(
                    local, target[indices], denominator, clip, nuisance_pseudocount,
                    dynamics,
                )
                member_results.append(result)
            for estimator in ESTIMATORS:
                output[method][estimator][replicate] = float(np.mean(
                    [row[estimator] for row in member_results]
                ))
    return output


def _bootstrap_chunk(
    arguments: tuple[Any, ...],
) -> list[tuple[int, dict[str, dict[str, float]]]]:
    (
        data,
        policy_groups,
        indexed_replicates,
        seed,
        denominator_name,
        clip,
        folds,
        denominator_pseudocount,
        nuisance_pseudocount,
    ) = arguments
    rows = []
    for replicate, indices in indexed_replicates:
        local = data.subset(indices)
        denominator = denominator_probabilities(
            local,
            denominator_name,
            folds,
            denominator_pseudocount,
            seed + replicate + 1,
        )
        dynamics = fit_dynamics(local, nuisance_pseudocount)
        method_rows: dict[str, dict[str, float]] = {}
        for method, members in policy_groups.items():
            member_results = [
                estimates(
                    local,
                    target[indices],
                    denominator,
                    clip,
                    nuisance_pseudocount,
                    dynamics,
                )[0]
                for target in members
            ]
            method_rows[method] = {
                estimator: float(np.mean([row[estimator] for row in member_results]))
                for estimator in ESTIMATORS
            }
        rows.append((replicate, method_rows))
    return rows


def spearman_and_pairwise(truth: np.ndarray, estimate: np.ndarray) -> tuple[float, float]:
    truth = np.asarray(truth, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    finite = np.isfinite(truth) & np.isfinite(estimate)
    if finite.sum() < 2:
        return float("nan"), float("nan")
    order_truth = np.argsort(np.argsort(truth[finite], kind="mergesort"), kind="mergesort").astype(float)
    order_estimate = np.argsort(np.argsort(estimate[finite], kind="mergesort"), kind="mergesort").astype(float)
    spearman = float(np.corrcoef(order_truth, order_estimate)[0, 1])
    tv, ev = truth[finite], estimate[finite]
    pairs = [(i, j) for i in range(len(tv)) for j in range(i + 1, len(tv)) if abs(tv[i] - tv[j]) > TIE_TOLERANCE]
    pairwise = float(np.mean([np.sign(tv[i] - tv[j]) == np.sign(ev[i] - ev[j]) for i, j in pairs])) if pairs else float("nan")
    return spearman, pairwise
