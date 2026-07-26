from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Callable

import numpy as np
import torch

from .fitted import kdd164_metric_selection as wm
from .fitted.kdd155v3_model_free import one_hot
from .full_pomdp_core import HistoryComparator, R2Environment
from .fitted.kdd166_pomdp_benchmark import HGBFit, SyntheticData, hgb_predict
from .kdd267_planner_primitives import (
    Policy,
    _candidate_sequences,
    _terminal_expectation,
    _true_sequence_scores,
)


def _representative_observation(environment: R2Environment, contexts: np.ndarray) -> np.ndarray:
    feature_count = environment.contract.feature_dim
    signs = np.where(np.arange(feature_count) % 2 == 0, 1.0, -1.0)
    return (
        environment.observation_loading * signs[None] * (2.0 * contexts[:, 0, None] / 4.0 - 1.0)
        + float(environment.generator["subtype_observation_shift"]) * (contexts[:, 1, None] - 1.0)
    ).astype(np.float32)


def _component_predictions(
    environment: R2Environment,
    component: str,
    fit: wm.Fit | HGBFit | None,
    contexts: np.ndarray,
    sequences: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    context_count, candidates, horizon = sequences.shape
    repeated = np.repeat(contexts, candidates, axis=0)
    actions = sequences.reshape(context_count * candidates, horizon)
    initial = _representative_observation(environment, repeated)
    if component == "persistence_locf":
        return np.repeat(initial[:, None], horizon, axis=1)
    observed = np.repeat(initial[:, None], horizon, axis=1)
    masks = np.ones_like(observed, dtype=bool)
    deltas = np.zeros_like(observed, dtype=np.float32)
    probability = np.zeros((len(observed), horizon, environment.contract.action_count), np.float32)
    probability[np.arange(len(observed))[:, None], np.arange(horizon)[None], actions] = 1.0
    dummy = SyntheticData(
        observed, masks, deltas, actions, observed.copy(), probability,
        np.zeros((len(observed), horizon), np.float32),
        np.zeros((len(observed), horizon), bool),
        np.ones((len(observed), horizon), bool),
        np.repeat(repeated[:, 1, None], horizon, axis=1).astype(np.int16),
    )
    if component == "hgb_residual":
        if not isinstance(fit, HGBFit):
            raise TypeError("HGB component requires HGBFit")
        current = initial.copy()
        output: list[np.ndarray] = []
        for step in range(horizon):
            local_probability = probability[:, step : step + 1]
            local = SyntheticData(
                current[:, None], np.ones_like(current[:, None], bool),
                np.zeros_like(current[:, None], np.float32), actions[:, step : step + 1],
                current[:, None], local_probability, np.zeros((len(current), 1), np.float32),
                np.zeros((len(current), 1), bool), np.ones((len(current), 1), bool),
                repeated[:, 1, None].astype(np.int16),
            )
            current = hgb_predict(fit, local)[:, 0]
            output.append(current)
        return np.stack(output, axis=1)
    if not isinstance(fit, wm.Fit):
        raise TypeError("learned component requires Fit")
    prediction, _ = wm.recursive_predictions(fit, dummy, device, batch=512)
    return prediction


def _component_sequence_scores(
    environment: R2Environment,
    component: str,
    fit: wm.Fit | HGBFit | None,
    contexts: np.ndarray,
    sequences: np.ndarray,
    remaining_after_plan: int,
    device: torch.device,
) -> np.ndarray:
    context_count, candidates, horizon = sequences.shape
    prediction = _component_predictions(environment, component, fit, contexts, sequences, device)
    repeated = np.repeat(contexts, candidates, axis=0)
    subtype = repeated[:, 1]
    feature_count = environment.contract.feature_dim
    signs = np.where(np.arange(feature_count) % 2 == 0, 1.0, -1.0)
    signed = np.sum(
        (prediction - float(environment.generator["subtype_observation_shift"]) * (subtype[:, None, None] - 1.0))
        * signs[None, None], axis=2,
    ) / feature_count
    inferred = np.clip(
        np.rint((signed / max(environment.observation_loading, 1e-8) + 1.0) * 2.0), 0, 4
    ).astype(int)
    pending = repeated[:, 2].astype(np.int16)
    actions = sequences.reshape(context_count * candidates, horizon)
    score = np.zeros(len(repeated))
    for step in range(horizon):
        action = actions[:, step]
        immediate = np.asarray([
            environment.immediate_reward(int(state), int(z), int(old), int(new))
            for state, z, old, new in zip(inferred[:, step], subtype, pending, action, strict=True)
        ])
        hazard = environment.contract.termination_hazards[min(step, environment.horizon - 2)]
        score += (environment.discount ** step) * (
            immediate + hazard * _terminal_expectation(environment, inferred[:, step])
        )
        pending = action
    if remaining_after_plan > 0:
        repeat = np.asarray([
            environment.immediate_reward(int(state), int(z), int(action), int(action))
            for state, z, action in zip(inferred[:, -1], subtype, pending, strict=True)
        ])
        score += (environment.discount ** horizon) * (
            remaining_after_plan * repeat + _terminal_expectation(environment, inferred[:, -1])
        )
    return score.reshape(context_count, candidates)


def build_repaired_sequence_cem(
    environment: R2Environment,
    comparator: HistoryComparator,
    planning: dict[str, Any],
    device: torch.device,
    component: str,
    fit: wm.Fit | HGBFit | None,
) -> tuple[Policy, dict[str, Any]]:
    """Compile the online-equivalent H4 receding-horizon policy over every belief cell."""
    supported = environment.supported
    horizon = int(planning["horizon"])
    candidates = int(planning["candidates"])
    iterations = int(planning["cem_iterations"])
    elites = int(planning["elites"])
    smoothing = float(planning["smoothing"])
    if (horizon, candidates, iterations, elites, smoothing) != (4, 64, 3, 8, 0.2):
        raise RuntimeError("primary H4 sequence-CEM contract drift")
    if not planning["support_mask_every_step"] or not planning["execute_first_action_only"]:
        raise RuntimeError("primary support or execution contract drift")
    contexts = np.asarray([
        (state, subtype, previous)
        for state in range(environment.states)
        for subtype in range(environment.subtypes)
        for previous in supported
    ], dtype=np.int16)
    local = np.full((len(contexts), horizon, len(supported)), 1.0 / len(supported))
    rng = np.random.default_rng(int(planning["planner_seed"]) + 1009 * environment.seed)
    uniforms = rng.random((iterations, len(contexts), candidates, horizon))
    distribution_deltas: list[float] = []
    multi_action_fractions: list[float] = []
    scores = np.zeros((len(contexts), candidates))
    started = time.perf_counter()
    for iteration in range(iterations):
        sequences = _candidate_sequences(local, uniforms[iteration], supported)
        multi_action_fractions.append(float(np.mean(np.ptp(sequences, axis=2) > 0)))
        if component == "true_model":
            scores = _true_sequence_scores(environment, contexts, sequences, environment.horizon - horizon)
        else:
            scores = _component_sequence_scores(
                environment, component, fit, contexts, sequences,
                environment.horizon - horizon, device,
            )
        elite_index = np.argpartition(scores, -elites, axis=1)[:, -elites:]
        elite_sequences = np.take_along_axis(sequences, elite_index[:, :, None], axis=1)
        empirical = np.stack([np.mean(elite_sequences == action, axis=1) for action in supported], axis=-1)
        updated = smoothing * local + (1.0 - smoothing) * empirical
        updated /= updated.sum(axis=-1, keepdims=True)
        distribution_deltas.append(float(np.mean(np.abs(updated - local))))
        local = updated
    table = supported[np.argmax(local[:, 0], axis=1)].reshape(
        environment.states, environment.subtypes, len(supported)
    )
    previous_lookup = {int(value): index for index, value in enumerate(supported)}
    belief: np.ndarray | None = None
    subtype_belief: np.ndarray | None = None
    metadata: dict[str, Any] = {}

    def policy(observation: np.ndarray, mask: np.ndarray, recency: np.ndarray,
               previous: np.ndarray, time_index: int) -> np.ndarray:
        nonlocal belief, subtype_belief
        if belief is None or len(belief) != len(observation) or time_index == 0:
            belief = np.full(len(observation), 2.0)
            subtype_belief = np.full(len(observation), comparator.subtype_assumption, dtype=float)
        belief, subtype_belief = comparator.update(
            belief, subtype_belief, observation, mask, recency, previous
        )
        state = np.clip(np.rint(belief), 0, environment.states - 1).astype(int)
        subtype = np.clip(np.rint(subtype_belief), 0, environment.subtypes - 1).astype(int)
        prior = np.asarray([previous_lookup[int(value)] for value in previous])
        metadata["replan_policy_calls"] += 1
        metadata["executed_first_actions"] += int(len(observation))
        return one_hot(table[state, subtype, prior], environment.contract.action_count)

    weights = (
        environment.initial_state_probability[:, None, None]
        * environment.subtype_prevalence[None, :, None]
        * (np.asarray(environment.contract.target_action_frequency)[supported]
           / np.asarray(environment.contract.target_action_frequency)[supported].sum())[None, None, :]
    )
    context_values = np.max(scores, axis=1).reshape(environment.states, environment.subtypes, len(supported))
    metadata.update({
        "component_model": component,
        "planner": "H4_support_only_sequence_categorical_CEM",
        "planner_horizon": horizon,
        "sequence_length": horizon,
        "candidates_per_iteration": candidates,
        "cem_iterations": iterations,
        "categorical_distribution_update_count": len(distribution_deltas),
        "elites": elites,
        "smoothing": smoothing,
        "support_mask_every_step": True,
        "uncertainty_penalty": 0.0,
        "execute_first_action_only": True,
        "receding_horizon_replan": True,
        "compiled_online_equivalent_belief_cells": len(contexts),
        "candidate_sequences_evaluated": len(contexts) * candidates * iterations,
        "model_query_count": len(contexts) * candidates * iterations * horizon,
        "distribution_update_mean_l1": float(np.mean(distribution_deltas)),
        "distribution_update_min_l1": float(np.min(distribution_deltas)),
        "multi_action_sequence_fraction": float(np.mean(multi_action_fractions)),
        "multi_action_sequences_present": bool(max(multi_action_fractions) > 0),
        "predicted_value": float(np.sum(weights * context_values)),
        "action_table_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
        "distinct_compiled_actions": int(np.unique(table).size),
        "planner_wall_time_seconds": time.perf_counter() - started,
        "replan_policy_calls": 0,
        "executed_first_actions": 0,
        "_action_table": table,
    })
    return policy, metadata


def profile_equal_bootstrap(
    values: list[dict[str, Any]],
    value_key: str,
    seed: int,
    replicates: int,
) -> dict[str, float | int]:
    profiles = sorted({str(row["profile"]) for row in values})
    grouped = {
        profile: np.asarray([float(row[value_key]) for row in values if row["profile"] == profile])
        for profile in profiles
    }
    if len(profiles) != 5 or any(len(local) != 8 for local in grouped.values()):
        raise RuntimeError("profile-equal bootstrap requires five profiles x eight environments")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for index in range(replicates):
        draws[index] = np.mean([
            np.mean(local[rng.integers(0, len(local), len(local))]) for local in grouped.values()
        ])
    flat = np.concatenate(list(grouped.values()))
    return {
        "profile_equal_mean": float(np.mean([local.mean() for local in grouped.values()])),
        "environment_median": float(np.median(flat)),
        "environment_sd": float(np.std(flat, ddof=1)),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
    }


def pairwise_action_agreement(tables: list[np.ndarray]) -> float:
    if len(tables) < 2:
        return math.nan
    values = [float(np.mean(tables[i] == tables[j])) for i in range(len(tables)) for j in range(i + 1, len(tables))]
    return float(np.mean(values))
