#!/usr/bin/env python3
"""Matched policy and OPE atlas inside frozen EHR-fitted simulators."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor

from . import kdd164_metric_selection as wm
from .kdd155v3_model_free import PolicyData, one_hot
from .kdd166_pomdp_benchmark import (
    HGBFit,
    SyntheticData,
    fit_hgb,
    hgb_predict,
    probability_policy_from_model,
)
from .kdd194_staged_training import (
    fit_world_model_staged,
    train_model_free_staged,
)
from .kdd202b_policy_ope import (
    ESTIMATORS,
    LoggedOPEData,
    bootstrap_policy_groups,
    point_policy_groups,
    spearman_and_pairwise,
    target_probabilities,
)
from .kdd229_ope_precision import weight_tail_diagnostics, wilson_interval
from .kdd248_full_episode import (
    CalibratedRolloutProfile,
    LearnedSourceSimulator,
    OnlinePolicy,
    ResponseRegime,
    TaskShape,
)
from .kdd248_plugin_policy_evaluation import (
    MarkovPolicy,
    behavior_policy,
    fit_markov_q_policy,
)
from .kdd248_source_training import load_source_components
from .source_helpers import (
    decision_token,
    flat_tree_sha256,
    sha256,
    task_shape,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/kdd262_matched_fitted_simulator_ope_v1.json"
FULL_DECISION = "complete_kdd262_matched_fitted_simulator_policy_ope_atlas"
PILOT_DECISION = "complete_kdd262_pilot_ready_for_full"
STOP_DECISION = "stop_kdd262_identity_inventory_execution_or_privacy_failure"


@dataclass(slots=True)
class ContextEncoder:
    mean: np.ndarray
    scale: np.ndarray
    centers: np.ndarray

    def assign(self, observation: np.ndarray) -> np.ndarray:
        local = (np.asarray(observation, dtype=np.float64) - self.mean) / self.scale
        distance = np.sum(
            np.square(local[:, None, :] - self.centers[None, :, :]), axis=2
        )
        return np.argmin(distance, axis=1).astype(np.int32)

    @property
    def cluster_count(self) -> int:
        return int(len(self.centers))

    def digest(self) -> str:
        return array_hash(self.mean, self.scale, self.centers)


@dataclass(slots=True)
class PolicyGroup:
    name: str
    family: str
    members: list[OnlinePolicy]
    identities: list[str]


@dataclass(slots=True)
class FrozenTask:
    name: str
    shape: TaskShape
    simulator: LearnedSourceSimulator
    profile: CalibratedRolloutProfile
    checkpoint_sha256: str
    profile_sha256: str


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        materialized = [{"status": "structural_na_no_rows"}]
    fields: list[str] = []
    for row in materialized:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fields}
            for row in materialized
        )


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        local = np.ascontiguousarray(value)
        digest.update(local.dtype.str.encode())
        digest.update(str(local.shape).encode())
        digest.update(local.tobytes())
    return digest.hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
    ).hexdigest()


def policy_hash(policy: MarkovPolicy) -> str:
    return array_hash(policy.initial_probability, policy.transition_probability)


def _effective_config(config: dict[str, Any], pilot: bool) -> dict[str, Any]:
    output = json.loads(json.dumps(config))
    if not pilot:
        return output
    override = output["pilot_overrides"]
    for role, episodes in override["role_episodes"].items():
        output["roles"][role]["episodes"] = episodes
    for key in (
        "staged_caps",
        "minimum_epochs",
        "patience",
        "late_window",
        "policy_checkpoint_window",
        "relative_validation_improvement_threshold",
        "mean_validation_policy_jsd_threshold",
    ):
        output["training"][key] = override[key]
    for key in ("prototype_count", "candidates", "cem_iterations", "elites"):
        output["planner"][key] = override[key]
    for key in (
        "independent_datasets_per_task",
        "episodes_per_dataset",
        "bootstrap_replicates",
        "bootstrap_workers",
    ):
        output["ope"][key] = override[key]
    output["tasks"] = override["tasks"]
    return output


def _load_frozen_tasks(
    config: dict[str, Any],
    checkpoint_root: Path,
    profile_root: Path,
) -> tuple[list[FrozenTask], list[dict[str, Any]]]:
    e4r = ROOT / config["e4r"]["path"]
    observed_decision = decision_token(e4r / "decision.md")
    observed_tree = flat_tree_sha256(e4r)
    receipts = [
        {
            "gate": "e4r_decision",
            "expected": config["e4r"]["decision"],
            "observed": observed_decision,
            "status": (
                "pass"
                if observed_decision == config["e4r"]["decision"]
                else "fail"
            ),
        },
        {
            "gate": "e4r_tree",
            "expected": config["e4r"]["result_tree_sha256"],
            "observed": observed_tree,
            "status": (
                "pass"
                if observed_tree == config["e4r"]["result_tree_sha256"]
                else "fail"
            ),
        },
    ]
    if any(row["status"] != "pass" for row in receipts):
        raise RuntimeError("KDD248E4R5 identity failure")
    selected = pd.read_csv(e4r / "selected_source_simulator_manifest.csv").set_index(
        "task"
    )
    profiles = pd.read_csv(e4r / "calibrated_profile_hash_manifest.csv").set_index(
        "task"
    )
    contract = json.loads(
        (ROOT / config["e3c_contract_config"]).read_text(encoding="utf-8")
    )
    requested = config.get("tasks", contract["tasks"])
    tasks: list[FrozenTask] = []
    for task in contract["tasks"]:
        if task not in requested:
            continue
        source = selected.loc[task]
        family = str(source["source_family"])
        seed = int(source["initialization_seed"])
        shape = task_shape(config, task)
        checkpoint = checkpoint_root / f"{task}__{family}__seed{seed}.pt"
        profile_path = profile_root / f"{task}__calibrated_profile.pt"
        checkpoint_digest = sha256(checkpoint) if checkpoint.is_file() else "missing"
        profile_digest = sha256(profile_path) if profile_path.is_file() else "missing"
        expected_checkpoint = str(source["checkpoint_sha256"])
        expected_profile = str(profiles.loc[task, "profile_sha256"])
        receipts.extend(
            [
                {
                    "gate": "checkpoint_sha256",
                    "task": task,
                    "expected": expected_checkpoint,
                    "observed": checkpoint_digest,
                    "status": (
                        "pass"
                        if checkpoint_digest == expected_checkpoint
                        else "fail"
                    ),
                },
                {
                    "gate": "calibrated_profile_sha256",
                    "task": task,
                    "expected": expected_profile,
                    "observed": profile_digest,
                    "status": (
                        "pass" if profile_digest == expected_profile else "fail"
                    ),
                },
            ]
        )
        if checkpoint_digest != expected_checkpoint or profile_digest != expected_profile:
            raise RuntimeError(f"protected source identity failure for {task}")
        components, _ = load_source_components(
            checkpoint,
            shape,
            family,
            seed,
            int(contract["hidden_dim"]),
            int(contract["latent_dim"]),
            torch.device("cpu"),
        )
        profile_payload = torch.load(
            profile_path, map_location="cpu", weights_only=False
        )
        profile_payload["binary_feature_indices"] = tuple(
            profile_payload["binary_feature_indices"]
        )
        profile = CalibratedRolloutProfile(**profile_payload)
        profile.validate(shape)
        tasks.append(
            FrozenTask(
                task,
                shape,
                LearnedSourceSimulator(
                    shape,
                    components,
                    sensitivity_bound=float(contract["sensitivity_bound"]),
                    rollout_profile=profile,
                ),
                profile,
                checkpoint_digest,
                profile_digest,
            )
        )
    return tasks, receipts


def _filled_episode_arrays(batch: Any) -> tuple[np.ndarray, ...]:
    valid = np.asarray(batch.valid_steps, dtype=bool)
    observation = np.where(
        valid[..., None], batch.observations, 0.0
    ).astype(np.float32)
    mask = np.where(valid[..., None], batch.masks, False)
    recency = np.where(valid[..., None], batch.recency, 0.0).astype(np.float32)
    action = np.where(valid, batch.actions, 0).astype(np.int64)
    reward = np.where(valid, batch.rewards, 0.0).astype(np.float32)
    behavior = np.where(
        valid[..., None], batch.action_probabilities, 0.0
    ).astype(np.float32)
    behavior[~valid, 0] = 1.0
    previous = np.zeros_like(action)
    previous[:, 1:] = action[:, :-1]
    return observation, mask, recency, action, reward, behavior, previous, valid


def policy_data_from_batch(batch: Any, action_count: int) -> PolicyData:
    (
        observation,
        mask,
        recency,
        action,
        reward,
        behavior,
        previous,
        valid,
    ) = _filled_episode_arrays(batch)
    episodes, horizon, _ = observation.shape
    time = np.broadcast_to(
        np.arange(horizon, dtype=np.float32)[None, :, None]
        / max(horizon - 1, 1),
        (episodes, horizon, 1),
    )
    state = np.concatenate(
        [
            observation,
            mask.astype(np.float32),
            recency / max(horizon, 1),
            np.eye(action_count, dtype=np.float32)[previous],
            time,
        ],
        axis=-1,
    )
    next_observation = observation.copy()
    next_mask = mask.copy()
    next_recency = recency.copy()
    next_observation[:, :-1] = observation[:, 1:]
    next_mask[:, :-1] = mask[:, 1:]
    next_recency[:, :-1] = recency[:, 1:]
    next_state = np.concatenate(
        [
            next_observation,
            next_mask.astype(np.float32),
            next_recency / max(horizon, 1),
            np.eye(action_count, dtype=np.float32)[action],
            time,
        ],
        axis=-1,
    )
    keep = valid.reshape(-1)
    episode = np.repeat(np.arange(episodes), horizon)[keep]
    step = np.tile(np.arange(horizon), episodes)[keep]
    local_behavior = behavior.reshape(-1, action_count)[keep]
    return PolicyData(
        state.reshape(-1, state.shape[-1])[keep].astype(np.float32),
        action.reshape(-1)[keep].astype(np.int64),
        reward.reshape(-1)[keep].astype(np.float32),
        np.ones(int(keep.sum()), dtype=bool),
        next_state.reshape(-1, next_state.shape[-1])[keep].astype(np.float32),
        np.asarray(batch.terminations).reshape(-1)[keep].astype(bool),
        local_behavior > 1.0e-8,
        local_behavior.astype(np.float32),
        observation[..., 0].reshape(-1)[keep].astype(np.float32),
        episode.astype(np.int64),
        step.astype(np.int64),
    )


def transition_data_from_batch(batch: Any, action_count: int) -> SyntheticData:
    (
        observation,
        mask,
        recency,
        action,
        reward,
        behavior,
        _previous,
        valid,
    ) = _filled_episode_arrays(batch)
    nonterminal = valid & ~np.asarray(batch.terminations, dtype=bool)
    has_next = np.zeros_like(nonterminal)
    has_next[:, :-1] = valid[:, :-1] & valid[:, 1:]
    keep = nonterminal & has_next
    episode, step = np.nonzero(keep)
    if len(episode) < 100:
        raise RuntimeError("insufficient nonterminal source-simulator transitions")
    current = observation[episode, step][:, None, :]
    current_mask = mask[episode, step][:, None, :]
    current_recency = recency[episode, step][:, None, :]
    current_action = action[episode, step][:, None]
    next_state = observation[episode, step + 1][:, None, :]
    probability = behavior[episode, step][:, None, :]
    local_reward = reward[episode, step][:, None]
    count = len(episode)
    return SyntheticData(
        current.astype(np.float32),
        current_mask.astype(bool),
        current_recency.astype(np.float32),
        current_action.astype(np.int16),
        next_state.astype(np.float32),
        probability.astype(np.float32),
        local_reward.astype(np.float32),
        np.zeros((count, 1), dtype=bool),
        np.ones((count, 1), dtype=bool),
        np.zeros((count, 1), dtype=np.int16),
    )


def fit_context_encoder(batch: Any, cluster_count: int, seed: int) -> ContextEncoder:
    observation = batch.observations[batch.valid_steps].astype(np.float64)
    mean = observation.mean(axis=0)
    scale = observation.std(axis=0)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    standardized = (observation - mean) / scale
    clusters = min(int(cluster_count), len(standardized))
    model = KMeans(n_clusters=clusters, random_state=seed, n_init=10)
    model.fit(standardized)
    return ContextEncoder(mean, scale, model.cluster_centers_)


def logged_data_from_batch(
    batch: Any,
    encoder: ContextEncoder,
    support: np.ndarray,
    discount: float,
) -> LoggedOPEData:
    (
        observation,
        mask,
        recency,
        action,
        reward,
        behavior,
        previous,
        valid,
    ) = _filled_episode_arrays(batch)
    episodes, horizon = action.shape
    absorbing = encoder.cluster_count
    states = np.full((episodes, horizon + 1), absorbing, dtype=np.int32)
    for step in range(horizon):
        local = encoder.assign(observation[:, step])
        states[valid[:, step], step] = local[valid[:, step]]
    # Cross-fitted behavior probabilities index every current-state slot before
    # invalid rows are replaced. Give those padding slots a valid context index
    # while retaining the absorbing index in the next-state column.
    current_states = states[:, :-1]
    current_states[~valid] = 0
    return LoggedOPEData(
        observation,
        mask,
        recency,
        previous,
        states,
        action,
        reward,
        valid,
        behavior,
        support.astype(bool),
        float(discount),
        absorbing,
    )


def behavior_callback(
    profile: CalibratedRolloutProfile, shape: TaskShape
) -> OnlinePolicy:
    initial = np.asarray(profile.initial_action_probability, dtype=np.float64)
    transition = np.asarray(
        profile.action_transition_probability, dtype=np.float64
    )

    def policy(
        observation: np.ndarray,
        mask: np.ndarray,
        recency: np.ndarray,
        previous: np.ndarray,
        step: int,
    ) -> np.ndarray:
        del mask, recency
        if step == 0:
            return np.broadcast_to(initial, (len(observation), shape.action_dim)).copy()
        local_transition = transition[step] if transition.ndim == 3 else transition
        return local_transition[np.asarray(previous, dtype=int)].copy()

    return policy


def fixed_callback(action: int, action_count: int) -> OnlinePolicy:
    def policy(
        observation: np.ndarray,
        mask: np.ndarray,
        recency: np.ndarray,
        previous: np.ndarray,
        step: int,
    ) -> np.ndarray:
        del mask, recency, previous, step
        return one_hot(
            np.full(len(observation), action, dtype=np.int64), action_count
        )

    return policy


def random_callback(support: np.ndarray) -> OnlinePolicy:
    probability = support.astype(np.float64)
    probability /= probability.sum()

    def policy(
        observation: np.ndarray,
        mask: np.ndarray,
        recency: np.ndarray,
        previous: np.ndarray,
        step: int,
    ) -> np.ndarray:
        del mask, recency, previous, step
        return np.broadcast_to(probability, (len(observation), len(probability))).copy()

    return policy


def markov_callback(policy: MarkovPolicy) -> OnlinePolicy:
    def callback(
        observation: np.ndarray,
        mask: np.ndarray,
        recency: np.ndarray,
        previous: np.ndarray,
        step: int,
    ) -> np.ndarray:
        del mask, recency
        if step == 0:
            return np.broadcast_to(
                policy.initial_probability,
                (len(observation), policy.action_count),
            ).copy()
        transition = policy.transition_probability
        local = transition[step] if transition.ndim == 3 else transition
        return local[np.asarray(previous, dtype=int)].copy()

    return callback


def initial_state_safe(policy: OnlinePolicy) -> OnlinePolicy:
    def callback(
        observation: np.ndarray,
        mask: np.ndarray,
        recency: np.ndarray,
        previous: np.ndarray,
        step: int,
    ) -> np.ndarray:
        if step == 0:
            previous = np.zeros_like(previous)
        return policy(observation, mask, recency, previous, step)

    return callback


def probability_safe(policy: OnlinePolicy) -> OnlinePolicy:
    """Remove only floating-point drift from an otherwise valid policy."""

    def callback(
        observation: np.ndarray,
        mask: np.ndarray,
        recency: np.ndarray,
        previous: np.ndarray,
        step: int,
    ) -> np.ndarray:
        probability = np.asarray(
            policy(observation, mask, recency, previous, step),
            dtype=np.float64,
        )
        if (
            not np.isfinite(probability).all()
            or probability.ndim != 2
            or np.any(probability < -1.0e-8)
        ):
            raise ValueError("policy returned invalid probabilities")
        probability = np.maximum(probability, 0.0)
        normalizer = probability.sum(axis=1, keepdims=True)
        if np.any(normalizer <= 0):
            raise ValueError("policy returned zero probability mass")
        return probability / normalizer

    return callback


def fit_reward_model(
    data: SyntheticData, action_count: int, max_iter: int, seed: int
) -> HistGradientBoostingRegressor:
    observation = data.observed[:, 0]
    mask = data.masks[:, 0].astype(np.float32)
    recency = data.deltas[:, 0]
    action = data.actions[:, 0]
    x = np.concatenate(
        [observation, mask, recency, np.eye(action_count)[action]], axis=1
    )
    model = HistGradientBoostingRegressor(
        max_iter=int(max_iter),
        max_depth=4,
        learning_rate=0.08,
        random_state=seed,
    )
    model.fit(x, data.rewards[:, 0])
    return model


def _one_step_data(
    observation: np.ndarray,
    mask: np.ndarray,
    recency: np.ndarray,
    action: np.ndarray,
    action_count: int,
) -> SyntheticData:
    count, features = observation.shape
    probability = one_hot(action, action_count)[:, None, :].astype(np.float32)
    return SyntheticData(
        observation[:, None].astype(np.float32),
        mask[:, None].astype(bool),
        recency[:, None].astype(np.float32),
        action[:, None].astype(np.int16),
        observation[:, None].astype(np.float32),
        probability,
        np.zeros((count, 1), dtype=np.float32),
        np.zeros((count, 1), dtype=bool),
        np.ones((count, 1), dtype=bool),
        np.zeros((count, 1), dtype=np.int16),
    )


def component_prediction(
    component: str,
    fit: wm.Fit | HGBFit | None,
    observation: np.ndarray,
    mask: np.ndarray,
    recency: np.ndarray,
    action: np.ndarray,
    action_count: int,
    device: torch.device,
) -> np.ndarray:
    if component == "persistence_locf":
        return observation.copy()
    data = _one_step_data(
        observation, mask, recency, action, action_count
    )
    if component == "hgb_residual":
        if not isinstance(fit, HGBFit):
            raise TypeError("HGB planner requires an HGB fit")
        return hgb_predict(fit, data)[:, 0]
    if not isinstance(fit, wm.Fit):
        raise TypeError("neural planner requires a frozen world-model fit")
    mean, _ = wm.native_predictions(fit, data, device, batch=1024)
    return mean[:, 0]


def reward_prediction(
    model: HistGradientBoostingRegressor,
    observation: np.ndarray,
    mask: np.ndarray,
    recency: np.ndarray,
    action: np.ndarray,
    action_count: int,
) -> np.ndarray:
    x = np.concatenate(
        [
            observation,
            mask.astype(np.float32),
            recency,
            np.eye(action_count)[action],
        ],
        axis=1,
    )
    return np.asarray(model.predict(x), dtype=np.float64)


def _prototype_rows(
    batch: Any, encoder: ContextEncoder
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observation = batch.observations[batch.valid_steps]
    mask = batch.masks[batch.valid_steps]
    recency = batch.recency[batch.valid_steps]
    assignment = encoder.assign(observation)
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    recencies: list[np.ndarray] = []
    for cluster in range(encoder.cluster_count):
        local = assignment == cluster
        if not local.any():
            observations.append(encoder.mean.copy())
            masks.append(np.ones_like(encoder.mean, dtype=bool))
            recencies.append(np.zeros_like(encoder.mean))
        else:
            observations.append(observation[local].mean(axis=0))
            masks.append(mask[local].mean(axis=0) >= 0.5)
            recencies.append(np.median(recency[local], axis=0))
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(masks, dtype=bool),
        np.asarray(recencies, dtype=np.float32),
    )


def compile_h4_policy(
    component: str,
    fit: wm.Fit | HGBFit | None,
    reward_model: HistGradientBoostingRegressor,
    encoder: ContextEncoder,
    prototypes: tuple[np.ndarray, np.ndarray, np.ndarray],
    support: np.ndarray,
    planner: dict[str, Any],
    discount: float,
    device: torch.device,
    seed_offset: int,
) -> tuple[OnlinePolicy, dict[str, Any]]:
    supported = np.flatnonzero(support)
    horizon = int(planner["horizon"])
    candidates = int(planner["candidates"])
    iterations = int(planner["cem_iterations"])
    elites = int(planner["elites"])
    smoothing = float(planner["smoothing"])
    if horizon != 4 or not planner["support_mask_every_step"]:
        raise RuntimeError("four-step support-only planner contract drift")
    observation, mask, recency = prototypes
    contexts = len(observation)
    local = np.full(
        (contexts, horizon, len(supported)),
        1.0 / len(supported),
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(planner["planner_seed"]) + seed_offset)
    uniforms = rng.random((iterations, contexts, candidates, horizon))
    distribution_deltas: list[float] = []
    started = time.perf_counter()
    for iteration in range(iterations):
        cumulative = np.cumsum(local, axis=2)
        indices = np.sum(
            uniforms[iteration][..., None] > cumulative[:, None, :, :],
            axis=3,
        )
        indices = np.clip(indices, 0, len(supported) - 1)
        sequences = supported[indices]
        current = np.repeat(observation[:, None, :], candidates, axis=1).reshape(
            contexts * candidates, -1
        )
        current_mask = np.repeat(mask[:, None, :], candidates, axis=1).reshape(
            contexts * candidates, -1
        )
        current_recency = np.repeat(
            recency[:, None, :], candidates, axis=1
        ).reshape(contexts * candidates, -1)
        score = np.zeros(contexts * candidates, dtype=np.float64)
        for step in range(horizon):
            action = sequences[:, :, step].reshape(-1)
            score += (discount**step) * reward_prediction(
                reward_model,
                current,
                current_mask,
                current_recency,
                action,
                len(support),
            )
            current = component_prediction(
                component,
                fit,
                current,
                current_mask,
                current_recency,
                action,
                len(support),
                device,
            )
            current_recency = current_recency + 1.0
        score = score.reshape(contexts, candidates)
        elite_index = np.argpartition(score, -elites, axis=1)[:, -elites:]
        elite = np.take_along_axis(
            sequences, elite_index[:, :, None], axis=1
        )
        empirical = np.stack(
            [np.mean(elite == action, axis=1) for action in supported], axis=-1
        )
        updated = smoothing * local + (1.0 - smoothing) * empirical
        updated /= updated.sum(axis=2, keepdims=True)
        distribution_deltas.append(float(np.mean(np.abs(updated - local))))
        local = updated
    action_table = supported[np.argmax(local[:, 0], axis=1)]

    def policy(
        local_observation: np.ndarray,
        local_mask: np.ndarray,
        local_recency: np.ndarray,
        previous: np.ndarray,
        step: int,
    ) -> np.ndarray:
        del local_mask, local_recency, previous, step
        cluster = encoder.assign(local_observation)
        return one_hot(action_table[cluster], len(support))

    metadata = {
        "planner": planner["name"],
        "planner_horizon": horizon,
        "candidates": candidates,
        "cem_iterations": iterations,
        "elites": elites,
        "smoothing": smoothing,
        "prototype_count": contexts,
        "support_count": len(supported),
        "action_table_sha256": array_hash(action_table),
        "distinct_compiled_actions": int(np.unique(action_table).size),
        "distribution_update_mean_l1": float(np.mean(distribution_deltas)),
        "planner_runtime_seconds": time.perf_counter() - started,
    }
    return policy, metadata


def _greedy_validation_policy(
    component: str,
    model: torch.nn.Module,
    validation: SyntheticData,
    reward_model: HistGradientBoostingRegressor,
    support: np.ndarray,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    count = min(256, len(validation.observed))
    observation = validation.observed[:count, 0]
    mask = validation.masks[:count, 0]
    recency = validation.deltas[:count, 0]
    fit = wm.Fit(component, seed, (model,), 0, 0, 0.0, "")
    scores = np.full((count, len(support)), -np.inf, dtype=np.float64)
    for action in np.flatnonzero(support):
        actions = np.full(count, action, dtype=np.int64)
        predicted = component_prediction(
            component,
            fit,
            observation,
            mask,
            recency,
            actions,
            len(support),
            device,
        )
        scores[:, action] = reward_prediction(
            reward_model,
            predicted,
            mask,
            recency + 1.0,
            actions,
            len(support),
        )
    return one_hot(np.argmax(scores, axis=1), len(support))


def train_policy_groups(
    task: FrozenTask,
    config: dict[str, Any],
    device: torch.device,
    task_index: int,
) -> tuple[
    list[PolicyGroup],
    ContextEncoder,
    np.ndarray,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    stride = int(config["task_seed_stride"]) * task_index
    roles = config["roles"]
    policy_train_batch = task.simulator.simulate(
        int(roles["policy_train"]["episodes"]),
        int(roles["policy_train"]["seed"]) + stride,
        ResponseRegime.OBSERVED,
    )
    policy_validation_batch = task.simulator.simulate(
        int(roles["policy_validation"]["episodes"]),
        int(roles["policy_validation"]["seed"]) + stride,
        ResponseRegime.OBSERVED,
    )
    context_batch = task.simulator.simulate(
        int(roles["context_fit"]["episodes"]),
        int(roles["context_fit"]["seed"]) + stride,
        ResponseRegime.OBSERVED,
    )
    encoder = fit_context_encoder(
        context_batch,
        int(config["planner"]["prototype_count"]),
        int(config["planner"]["planner_seed"]) + task_index,
    )
    initial = np.asarray(task.profile.initial_action_probability, dtype=float)
    transition = np.asarray(
        task.profile.action_transition_probability, dtype=float
    )
    support = initial > float(config["planner"]["support_probability_floor"])
    if transition.ndim == 3:
        support |= np.any(
            transition > float(config["planner"]["support_probability_floor"]),
            axis=(0, 1),
        )
    else:
        support |= np.any(
            transition > float(config["planner"]["support_probability_floor"]),
            axis=0,
        )
    if support.sum() < 2:
        raise RuntimeError("frozen source behavior exposes fewer than two actions")
    training_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    groups: dict[str, PolicyGroup] = {}

    def add_group(
        name: str,
        family: str,
        members: list[OnlinePolicy],
        identities: list[str],
    ) -> None:
        groups[name] = PolicyGroup(
            name,
            family,
            [probability_safe(member) for member in members],
            identities,
        )
        identity_rows.append(
            {
                "task": task.name,
                "method": name,
                "family": family,
                "member_count": len(members),
                "member_identities": ";".join(identities),
                "status": "frozen_before_direct_reference",
            }
        )

    add_group(
        "fitted_behavior",
        "control",
        [behavior_callback(task.profile, task.shape)],
        [stable_hash([task.name, "fitted_behavior", initial.tolist()])],
    )
    add_group(
        "supported_random",
        "control",
        [random_callback(support)],
        [stable_hash([task.name, "supported_random", support.tolist()])],
    )
    add_group(
        "fixed_minimum",
        "control",
        [fixed_callback(int(np.flatnonzero(support)[0]), task.shape.action_dim)],
        [stable_hash([task.name, "minimum", int(np.flatnonzero(support)[0])])],
    )
    add_group(
        "fixed_maximum",
        "control",
        [fixed_callback(int(np.flatnonzero(support)[-1]), task.shape.action_dim)],
        [stable_hash([task.name, "maximum", int(np.flatnonzero(support)[-1])])],
    )

    policy_train = policy_data_from_batch(
        policy_train_batch, task.shape.action_dim
    )
    policy_validation = policy_data_from_batch(
        policy_validation_batch, task.shape.action_dim
    )
    for method in config["model_free_methods"]:
        members: list[OnlinePolicy] = []
        identities: list[str] = []
        for seed in config["training_seeds"]:
            _probability, metadata, model = train_model_free_staged(
                method,
                policy_train,
                policy_validation,
                task.shape.action_dim,
                config["training"],
                int(seed),
                device,
            )
            model = model.to("cpu")
            callback = probability_policy_from_model(
                "soft_spibb"
                if method == "spibb_style_count_blend_adapter"
                else method,
                model,
                policy_train,
                task.shape.action_dim,
                config["training"],
                torch.device("cpu"),
                metadata,
            )
            members.append(initial_state_safe(callback))
            identities.append(str(metadata["model_state_sha256"]))
            training_rows.append(
                {
                    "task": task.name,
                    "family": "model_free",
                    "method": method,
                    "training_seed": seed,
                    **metadata,
                    "status": "complete",
                }
            )
        add_group(method, "model_free", members, identities)

    transition_train = transition_data_from_batch(
        policy_train_batch, task.shape.action_dim
    )
    transition_validation = transition_data_from_batch(
        policy_validation_batch, task.shape.action_dim
    )
    reward_model = fit_reward_model(
        transition_train,
        task.shape.action_dim,
        int(config["planner"]["reward_model_max_iter"]),
        int(config["planner"]["planner_seed"]) + task_index,
    )
    prototypes = _prototype_rows(policy_train_batch, encoder)
    reward_hash = object_hash(reward_model)
    training_rows.append(
        {
            "task": task.name,
            "family": "shared_planning_component",
            "method": "generated_data_reward_regressor",
            "training_seed": int(config["planner"]["planner_seed"]) + task_index,
            "convergence_status": "fixed_iteration_budget",
            "model_state_sha256": reward_hash,
            "status": "complete",
        }
    )

    persistence, persistence_meta = compile_h4_policy(
        "persistence_locf",
        None,
        reward_model,
        encoder,
        prototypes,
        support,
        config["planner"],
        float(config["discount"]),
        torch.device("cpu"),
        task_index * 1000,
    )
    add_group(
        "persistence_locf_plus_h4_support_only",
        "model_based",
        [persistence],
        [persistence_meta["action_table_sha256"]],
    )
    training_rows.append(
        {
            "task": task.name,
            "family": "model_based",
            "method": "persistence_locf",
            "training_seed": "deterministic",
            "convergence_status": "not_applicable_control",
            **persistence_meta,
            "status": "complete",
        }
    )

    hgb_members: list[OnlinePolicy] = []
    hgb_identities: list[str] = []
    for seed_index, seed in enumerate(config["training_seeds"]):
        fit = fit_hgb(transition_train, int(seed))
        policy, metadata = compile_h4_policy(
            "hgb_residual",
            fit,
            reward_model,
            encoder,
            prototypes,
            support,
            config["planner"],
            float(config["discount"]),
            torch.device("cpu"),
            task_index * 1000 + 100 + seed_index,
        )
        hgb_members.append(policy)
        hgb_identities.append(fit.fingerprint)
        training_rows.append(
            {
                "task": task.name,
                "family": "model_based",
                "method": "hgb_residual",
                "training_seed": seed,
                "convergence_status": "fixed_iteration_budget",
                "model_state_sha256": fit.fingerprint,
                "runtime_seconds": fit.training_seconds,
                **metadata,
                "status": "complete",
            }
        )
    add_group(
        "hgb_residual_plus_h4_support_only",
        "model_based",
        hgb_members,
        hgb_identities,
    )

    component_fits: dict[str, list[tuple[wm.Fit, dict[str, Any]]]] = {
        method: []
        for method in (
            "deterministic_grud_point",
            "causal_transformer",
            "categorical_rssm",
            "single_gaussian_grud",
        )
    }
    for component in component_fits:
        members: list[OnlinePolicy] = []
        identities: list[str] = []
        for seed_index, seed in enumerate(config["training_seeds"]):
            callback = lambda model, component=component, seed=seed: _greedy_validation_policy(
                component,
                model,
                transition_validation,
                reward_model,
                support,
                int(seed),
                device,
            )
            fit, metadata = fit_world_model_staged(
                component,
                transition_train,
                transition_validation,
                int(seed),
                config["training"],
                device,
                callback,
            )
            component_fits[component].append((fit, metadata))
            planner_policy, planner_metadata = compile_h4_policy(
                component,
                fit,
                reward_model,
                encoder,
                prototypes,
                support,
                config["planner"],
                float(config["discount"]),
                device,
                task_index * 1000 + 200 + 10 * list(component_fits).index(component) + seed_index,
            )
            members.append(planner_policy)
            identities.append(fit.fingerprint)
            training_rows.append(
                {
                    "task": task.name,
                    "family": "model_based",
                    "method": component,
                    "training_seed": seed,
                    **metadata,
                    **planner_metadata,
                    "status": "complete",
                }
            )
        add_group(
            f"{component}_plus_h4_support_only",
            "model_based",
            members,
            identities,
        )

    ensemble_fit = wm.ensemble_fit(
        [fit for fit, _ in component_fits["single_gaussian_grud"]]
    )
    ensemble_policy, ensemble_meta = compile_h4_policy(
        "matched_gaussian_ensemble",
        ensemble_fit,
        reward_model,
        encoder,
        prototypes,
        support,
        config["planner"],
        float(config["discount"]),
        device,
        task_index * 1000 + 900,
    )
    add_group(
        "matched_gaussian_ensemble_plus_h4_support_only",
        "model_based",
        [ensemble_policy],
        [ensemble_fit.fingerprint],
    )
    training_rows.append(
        {
            "task": task.name,
            "family": "model_based",
            "method": "matched_gaussian_ensemble",
            "training_seed": "3408;3411;3414",
            "convergence_status": (
                "convergence_incomplete"
                if any(
                    metadata["convergence_status"] == "convergence_incomplete"
                    for _, metadata in component_fits["single_gaussian_grud"]
                )
                else "converged_or_plateaued"
            ),
            "model_state_sha256": ensemble_fit.fingerprint,
            **ensemble_meta,
            "status": "complete",
        }
    )

    first_probability = np.asarray(
        task.profile.initial_action_probability, dtype=float
    )
    behavior_transition = np.asarray(
        task.profile.action_transition_probability, dtype=float
    )
    candidates: list[tuple[float, MarkovPolicy, float]] = []
    for temperature in config["markov_q"]["temperatures"]:
        fitted = fit_markov_q_policy(
            policy_train_batch,
            initial_probability=first_probability,
            action_count=task.shape.action_dim,
            discount=float(config["discount"]),
            temperature=float(temperature),
            behavior_transition=behavior_transition,
            behavior_blend=float(config["markov_q"]["fitted_behavior_blend"]),
            name=f"fitted_q_temperature_{temperature}",
        )
        validation = task.simulator.simulate(
            int(config["roles"]["policy_validation"]["episodes"]),
            int(config["roles"]["policy_validation"]["seed"]) + stride + 7000,
            ResponseRegime.OBSERVED,
            policy=markov_callback(fitted),
            policy_seed=int(config["roles"]["policy_validation"]["seed"])
            + stride
            + 8000,
        )
        candidates.append(
            (
                float(temperature),
                fitted,
                float(
                    validation.discounted_returns(float(config["discount"])).mean()
                ),
            )
        )
    selected_temperature, fitted, selected_return = max(
        candidates, key=lambda row: row[2]
    )
    fitted = MarkovPolicy(
        "fitted_markov_q",
        fitted.initial_probability,
        fitted.transition_probability,
        fitted.fit_role,
    )
    conservative = fit_markov_q_policy(
        policy_train_batch,
        initial_probability=first_probability,
        action_count=task.shape.action_dim,
        discount=float(config["discount"]),
        temperature=selected_temperature,
        behavior_transition=behavior_transition,
        behavior_blend=float(
            config["markov_q"]["conservative_behavior_blend"]
        ),
        name="conservative_markov_q",
    )
    for local_policy in (fitted, conservative):
        add_group(
            local_policy.name,
            "fitted_specific",
            [markov_callback(local_policy)],
            [policy_hash(local_policy)],
        )
        training_rows.append(
            {
                "task": task.name,
                "family": "fitted_specific",
                "method": local_policy.name,
                "training_seed": "deterministic_generated_data",
                "selected_temperature": selected_temperature,
                "selected_validation_return": (
                    selected_return
                    if local_policy.name == "fitted_markov_q"
                    else ""
                ),
                "convergence_status": "not_applicable_tabular_fit",
                "model_state_sha256": policy_hash(local_policy),
                "status": "complete",
            }
        )

    expected = config["common_methods"] + config["fitted_specific_methods"]
    if list(groups) != expected:
        raise RuntimeError(
            f"policy inventory mismatch: {list(groups)} versus {expected}"
        )
    return [groups[name] for name in expected], encoder, support, training_rows, identity_rows


def evaluate_policy_group(
    task: FrozenTask,
    group: PolicyGroup,
    episodes: int,
    exogenous_seed: int,
    policy_seed_base: int,
    discount: float,
) -> np.ndarray:
    member_returns = []
    for member_index, policy in enumerate(group.members):
        batch = task.simulator.simulate(
            episodes,
            exogenous_seed,
            ResponseRegime.OBSERVED,
            policy=policy,
            policy_seed=policy_seed_base + member_index,
        )
        member_returns.append(batch.discounted_returns(discount))
    return np.mean(member_returns, axis=0)


def direct_references(
    task: FrozenTask,
    groups: list[PolicyGroup],
    config: dict[str, Any],
    task_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    consistency: list[dict[str, Any]] = []
    references: dict[str, float] = {}
    role_a = config["roles"]["direct_reference_a"]
    role_b = config["roles"]["direct_reference_b"]
    stride = int(config["task_seed_stride"]) * task_index
    policy_stride = int(config["policy_seed_stride"])
    for method_index, group in enumerate(groups):
        first = evaluate_policy_group(
            task,
            group,
            int(role_a["episodes"]),
            int(role_a["seed"]) + stride,
            int(role_a["seed"]) + stride + method_index * policy_stride,
            float(config["discount"]),
        )
        second = evaluate_policy_group(
            task,
            group,
            int(role_b["episodes"]),
            int(role_b["seed"]) + stride,
            int(role_b["seed"]) + stride + method_index * policy_stride,
            float(config["discount"]),
        )
        first_mean = float(first.mean())
        second_mean = float(second.mean())
        first_se = float(first.std(ddof=1) / math.sqrt(len(first)))
        second_se = float(second.std(ddof=1) / math.sqrt(len(second)))
        denominator = math.sqrt(first_se**2 + second_se**2)
        standardized = (
            abs(first_mean - second_mean) / denominator
            if denominator > 0
            else (0.0 if first_mean == second_mean else math.inf)
        )
        combined = np.concatenate([first, second])
        reference = float(combined.mean())
        reference_se = float(combined.std(ddof=1) / math.sqrt(len(combined)))
        references[group.name] = reference
        rows.append(
            {
                "task": task.name,
                "method": group.name,
                "family": group.family,
                "member_count": len(group.members),
                "reference_value": reference,
                "reference_standard_error": reference_se,
                "episode_count": len(combined),
                "reference_provenance": "frozen_observational_source_simulator_monte_carlo",
                "status": "complete",
            }
        )
        consistency.append(
            {
                "task": task.name,
                "method": group.name,
                "replica_a_mean": first_mean,
                "replica_a_standard_error": first_se,
                "replica_b_mean": second_mean,
                "replica_b_standard_error": second_se,
                "standardized_difference": standardized,
                "limit": config["direct_reference_gate"][
                    "maximum_standardized_replica_difference"
                ],
                "status": (
                    "pass"
                    if np.isfinite(standardized)
                    and standardized
                    <= config["direct_reference_gate"][
                        "maximum_standardized_replica_difference"
                    ]
                    else "fail"
                ),
            }
        )
    normalizer = max(
        abs(references["fixed_maximum"] - references["fixed_minimum"]),
        float(config["normalization"]["minimum_range"]),
    )
    lower = min(references["fixed_minimum"], references["fixed_maximum"])
    for row in rows:
        row["control_range_normalizer"] = normalizer
        row["control_range_normalized_return"] = (
            float(row["reference_value"]) - lower
        ) / normalizer
    return rows, consistency, normalizer, references


def _dataset_checkpoint(
    output: Path, task: str, dataset_index: int
) -> Path:
    return output / "_checkpoints" / f"{task}__dataset_{dataset_index:03d}.json"


def _privacy_rows(output: Path) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"/(?:home|tmp|data)/|subject_id|stay_id|hadm_id|patient_id|anchor_time|charttime",
        re.IGNORECASE,
    )
    rows = []
    for path in output.iterdir():
        if not path.is_file() or path.suffix not in {".csv", ".md", ".json"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = pattern.findall(text)
        rows.append(
            {
                "artifact": path.name,
                "prohibited_token_count": len(hits),
                "row_level_clinical_data": "no",
                "status": "pass" if not hits else "fail",
            }
        )
    return rows


def evaluate_ope_datasets(
    task: FrozenTask,
    groups: list[PolicyGroup],
    encoder: ContextEncoder,
    support: np.ndarray,
    references: dict[str, float],
    normalizer: float,
    config: dict[str, Any],
    output: Path,
    task_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ope = config["ope"]
    dataset_count = int(ope["independent_datasets_per_task"])
    checkpoint_dir = output / "_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for dataset_index in range(dataset_count):
        checkpoint = _dataset_checkpoint(output, task.name, dataset_index)
        if checkpoint.exists():
            continue
        seed = (
            int(ope["dataset_seed_base"])
            + task_index * int(ope["dataset_task_stride"])
            + dataset_index
        )
        batch = task.simulator.simulate(
            int(ope["episodes_per_dataset"]),
            seed,
            ResponseRegime.OBSERVED,
        )
        data = logged_data_from_batch(
            batch, encoder, support, float(config["discount"])
        )
        policy_groups = {
            group.name: [
                target_probabilities(data, member) for member in group.members
            ]
            for group in groups
        }
        point, diagnostics = point_policy_groups(
            data,
            policy_groups,
            str(ope["denominator"]),
            ope["clip"],
            int(ope["crossfit_folds"]),
            float(ope["denominator_pseudocount"]),
            float(ope["nuisance_pseudocount"]),
            seed,
        )
        bootstrap_seed = (
            int(ope["bootstrap_seed_base"])
            + task_index * int(ope["dataset_task_stride"])
            + dataset_index
        )
        boot = bootstrap_policy_groups(
            data,
            policy_groups,
            int(ope["bootstrap_replicates"]),
            bootstrap_seed,
            str(ope["denominator"]),
            ope["clip"],
            int(ope["crossfit_folds"]),
            float(ope["denominator_pseudocount"]),
            float(ope["nuisance_pseudocount"]),
            int(ope["bootstrap_workers"]),
        )
        tails = weight_tail_diagnostics(
            data,
            policy_groups,
            str(ope["denominator"]),
            ope["clip"],
            int(ope["crossfit_folds"]),
            float(ope["denominator_pseudocount"]),
            seed,
        )
        records: list[dict[str, Any]] = []
        behavior_truth = references["fitted_behavior"]
        for group in groups:
            truth = references[group.name]
            for estimator in ESTIMATORS:
                samples = boot[group.name][estimator]
                finite_samples = np.isfinite(samples)
                low, high = (
                    np.quantile(samples[finite_samples], [0.05, 0.95])
                    if finite_samples.any()
                    else (math.nan, math.nan)
                )
                estimate = float(point[group.name][estimator])
                behavior_estimate = float(point["fitted_behavior"][estimator])
                opportunity = (
                    group.name != "fitted_behavior"
                    and abs(truth - behavior_truth) > float(ope["tie_tolerance"])
                )
                if opportunity:
                    sign_error: float | str = float(
                        np.sign(estimate - behavior_estimate)
                        != np.sign(truth - behavior_truth)
                    )
                    behavior_samples = boot["fitted_behavior"][estimator]
                    difference = samples - behavior_samples
                    finite_difference = difference[np.isfinite(difference)]
                    diff_low = (
                        float(np.quantile(finite_difference, 0.05))
                        if len(finite_difference)
                        else math.nan
                    )
                    false_improvement: float | str = float(
                        np.isfinite(diff_low)
                        and diff_low > 0
                        and truth <= behavior_truth
                    )
                else:
                    sign_error = ""
                    false_improvement = ""
                records.append(
                    {
                        "task": task.name,
                        "dataset_index": dataset_index,
                        "method": group.name,
                        "family": group.family,
                        "estimator": estimator,
                        "reference_value": truth,
                        "control_range_normalizer": normalizer,
                        "estimate": estimate,
                        "normalized_error": (estimate - truth) / normalizer,
                        "normalized_absolute_error": abs(estimate - truth)
                        / normalizer,
                        "interval_low": float(low),
                        "interval_high": float(high),
                        "coverage": float(
                            np.isfinite(low)
                            and np.isfinite(high)
                            and low <= truth <= high
                        ),
                        "interval_width": (
                            float(high - low)
                            if np.isfinite(low) and np.isfinite(high)
                            else math.nan
                        ),
                        "sign_error": sign_error,
                        "false_improvement": false_improvement,
                        "ess": diagnostics[group.name]["ess"],
                        "ess_fraction": diagnostics[group.name]["ess"]
                        / data.episodes,
                        "unsupported_mass": diagnostics[group.name][
                            "unsupported_mass"
                        ],
                        "point_finite": bool(np.isfinite(estimate)),
                        "finite": bool(
                            np.isfinite(estimate)
                            and np.isfinite(low)
                            and np.isfinite(high)
                            and finite_samples.all()
                        ),
                        **tails[group.name],
                    }
                )
        truth_vector = np.asarray(
            [references[group.name] for group in groups], dtype=float
        )
        behavior_index = [group.name for group in groups].index(
            "fitted_behavior"
        )
        rank_rows: list[dict[str, Any]] = []
        for estimator in ESTIMATORS:
            estimate_vector = np.asarray(
                [point[group.name][estimator] for group in groups], dtype=float
            )
            spearman, pairwise = spearman_and_pairwise(
                truth_vector, estimate_vector
            )
            opportunity = (
                np.abs(truth_vector - truth_vector[behavior_index])
                > float(ope["tie_tolerance"])
            )
            opportunity[behavior_index] = False
            sign_recovery = (
                float(
                    np.mean(
                        np.sign(
                            estimate_vector[opportunity]
                            - estimate_vector[behavior_index]
                        )
                        == np.sign(
                            truth_vector[opportunity]
                            - truth_vector[behavior_index]
                        )
                    )
                )
                if opportunity.any()
                else math.nan
            )
            false_rows = [
                row
                for row in records
                if row["estimator"] == estimator
                and row["false_improvement"] != ""
            ]
            rank_rows.append(
                {
                    "task": task.name,
                    "dataset_index": dataset_index,
                    "estimator": estimator,
                    "spearman_rank_recovery": spearman,
                    "pairwise_order_recovery": pairwise,
                    "behavior_relative_sign_recovery": sign_recovery,
                    "false_improvement_rate": (
                        float(
                            np.mean(
                                [
                                    float(row["false_improvement"])
                                    for row in false_rows
                                ]
                            )
                        )
                        if false_rows
                        else math.nan
                    ),
                }
            )
        receipt = {
            "task": task.name,
            "dataset_index": dataset_index,
            "dataset_seed_hash": stable_hash(
                [task.name, dataset_index, seed]
            ),
            "episode_count": data.episodes,
            "policy_count": len(groups),
            "policy_member_count": sum(len(group.members) for group in groups),
            "estimator_count": len(ESTIMATORS),
            "bootstrap_replicates": int(ope["bootstrap_replicates"]),
            "denominator_refit_count": 1
            + int(ope["bootstrap_replicates"]),
            "nuisance_refit_count": 1 + int(ope["bootstrap_replicates"]),
            "truth_used_for_fitting": "no",
            "status": "pass",
        }
        checkpoint.write_text(
            json.dumps(
                {"records": records, "rank_rows": rank_rows, "receipt": receipt},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=True,
            ),
            encoding="utf-8",
        )
        print(
            f"KDD262 OPE {task.name} dataset "
            f"{dataset_index + 1}/{dataset_count}",
            flush=True,
        )
    records: list[dict[str, Any]] = []
    ranks: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for dataset_index in range(dataset_count):
        payload = json.loads(
            _dataset_checkpoint(output, task.name, dataset_index).read_text(
                encoding="utf-8"
            )
        )
        records.extend(payload["records"])
        ranks.extend(payload["rank_rows"])
        receipts.append(payload["receipt"])
    return records, ranks, receipts


def summarize_results(
    direct_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    direct = pd.DataFrame(direct_rows)
    frame = pd.DataFrame(records)
    ranks = pd.DataFrame(rank_rows)
    direct_summary: list[dict[str, Any]] = []
    for method, local in direct.groupby("method", sort=False):
        direct_summary.append(
            {
                "method": method,
                "family": local["family"].iloc[0],
                "task_count": len(local),
                "task_equal_normalized_return": float(
                    local["control_range_normalized_return"].mean()
                ),
                "minimum_task_normalized_return": float(
                    local["control_range_normalized_return"].min()
                ),
                "maximum_task_normalized_return": float(
                    local["control_range_normalized_return"].max()
                ),
            }
        )
    method_rows: list[dict[str, Any]] = []
    for (method, estimator), local in frame.groupby(
        ["method", "estimator"], sort=False
    ):
        task_means = local.groupby("task", sort=False).mean(numeric_only=True)
        successes = int(local["coverage"].sum())
        low, high = wilson_interval(successes, len(local))
        method_rows.append(
            {
                "method": method,
                "family": local["family"].iloc[0],
                "estimator": estimator,
                "task_count": local["task"].nunique(),
                "dataset_count": len(local),
                "task_equal_normalized_mae": float(
                    task_means["normalized_absolute_error"].mean()
                ),
                "empirical_90_interval_coverage": float(
                    local["coverage"].mean()
                ),
                "coverage_deviation": abs(float(local["coverage"].mean()) - 0.9),
                "coverage_wilson_lower": low,
                "coverage_wilson_upper": high,
                "median_ess_fraction": float(local["ess_fraction"].median()),
                "finite_fraction": float(local["finite"].mean()),
                "false_improvement_rate": (
                    float(
                        pd.to_numeric(
                            local["false_improvement"], errors="coerce"
                        ).mean()
                    )
                ),
            }
        )
    estimator_rows: list[dict[str, Any]] = []
    for estimator, local in frame.groupby("estimator", sort=False):
        task_means = local.groupby("task", sort=False).mean(numeric_only=True)
        rank_local = ranks[ranks["estimator"].eq(estimator)]
        estimator_rows.append(
            {
                "estimator": estimator,
                "task_equal_normalized_mae": float(
                    task_means["normalized_absolute_error"].mean()
                ),
                "empirical_90_interval_coverage": float(
                    local["coverage"].mean()
                ),
                "mean_spearman_rank_recovery": float(
                    rank_local["spearman_rank_recovery"].mean()
                ),
                "mean_pairwise_order_recovery": float(
                    rank_local["pairwise_order_recovery"].mean()
                ),
                "mean_behavior_relative_sign_recovery": float(
                    rank_local["behavior_relative_sign_recovery"].mean()
                ),
                "false_improvement_rate": float(
                    rank_local["false_improvement_rate"].mean()
                ),
                "median_ess": float(local["ess"].median()),
                "finite_fraction": float(local["finite"].mean()),
            }
        )
    return direct_summary, method_rows, estimator_rows


def run(args: argparse.Namespace) -> str:
    started = time.time()
    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    config = _effective_config(raw_config, args.pilot)
    if args.output.exists() and not args.resume:
        raise FileExistsError("KDD262 output must be additive")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        tasks, identity_rows = _load_frozen_tasks(
            config, args.checkpoint_root, args.profile_root
        )
        expected_estimators = list(config["ope"]["estimators"])
        if expected_estimators != list(ESTIMATORS):
            raise RuntimeError("six-estimator inventory drift")
        freeze = {
            "experiment_id": config["experiment_id"],
            "pilot": bool(args.pilot),
            "config_sha256": sha256(args.config),
            "runner_sha256": sha256(Path(__file__)),
            "source_e4r_tree_sha256": config["e4r"]["result_tree_sha256"],
            "task_names": [task.name for task in tasks],
            "common_methods": config["common_methods"],
            "fitted_specific_methods": config["fitted_specific_methods"],
            "controlled_only_not_applicable": config[
                "controlled_only_not_applicable"
            ],
            "estimators": expected_estimators,
            "roles": config["roles"],
            "training_seeds": config["training_seeds"],
            "training": config["training"],
            "planner": config["planner"],
            "ope": config["ope"],
            "normalization": config["normalization"],
            "claim_boundary": config["claim_boundary"],
            "protected_paths_exported": False,
            "raw_clinical_access": False,
            "frozen_before_execution": True,
        }
        (args.output / "frozen_kdd262_contract.md").write_text(
            "# KDD262 frozen contract\n\n```json\n"
            + json.dumps(freeze, indent=2, sort_keys=True)
            + "\n```\n",
            encoding="utf-8",
        )
        write_csv(args.output / "source_identity_receipt.csv", identity_rows)
        training_rows: list[dict[str, Any]] = []
        policy_identity_rows: list[dict[str, Any]] = []
        direct_rows: list[dict[str, Any]] = []
        consistency_rows: list[dict[str, Any]] = []
        all_records: list[dict[str, Any]] = []
        all_ranks: list[dict[str, Any]] = []
        all_receipts: list[dict[str, Any]] = []
        for task_index, task in enumerate(tasks):
            groups, encoder, support, local_training, local_identity = (
                train_policy_groups(task, config, args.device, task_index)
            )
            training_rows.extend(local_training)
            policy_identity_rows.extend(local_identity)
            local_direct, local_consistency, normalizer, references = (
                direct_references(task, groups, config, task_index)
            )
            direct_rows.extend(local_direct)
            consistency_rows.extend(local_consistency)
            if any(row["status"] != "pass" for row in local_consistency):
                raise RuntimeError(
                    f"direct-reference replica consistency failure for {task.name}"
                )
            records, ranks, receipts = evaluate_ope_datasets(
                task,
                groups,
                encoder,
                support,
                references,
                normalizer,
                config,
                args.output,
                task_index,
            )
            all_records.extend(records)
            all_ranks.extend(ranks)
            all_receipts.extend(receipts)
            print(f"KDD262 task complete: {task.name}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        expected_rows = (
            len(tasks)
            * int(config["ope"]["independent_datasets_per_task"])
            * (
                len(config["common_methods"])
                + len(config["fitted_specific_methods"])
            )
            * len(ESTIMATORS)
        )
        if len(all_records) != expected_rows:
            raise RuntimeError(
                f"primary row inventory {len(all_records)}/{expected_rows}"
            )
        write_csv(
            args.output / "policy_training_and_qualification.csv", training_rows
        )
        write_csv(
            args.output / "policy_identity_and_inventory.csv",
            policy_identity_rows,
        )
        write_csv(
            args.output / "source_simulator_reference_values.csv", direct_rows
        )
        write_csv(
            args.output / "direct_replica_consistency.csv", consistency_rows
        )
        write_csv(
            args.output / "repeated_dataset_policy_estimator_rows.csv",
            all_records,
        )
        write_csv(
            args.output / "dataset_and_refit_receipt.csv", all_receipts
        )
        write_csv(
            args.output / "dataset_rank_recovery.csv", all_ranks
        )
        direct_summary, method_summary, estimator_summary = summarize_results(
            direct_rows, all_records, all_ranks
        )
        write_csv(
            args.output / "method_direct_return_summary.csv", direct_summary
        )
        write_csv(
            args.output / "policy_by_estimator_summary.csv", method_summary
        )
        write_csv(
            args.output / "estimator_accuracy_coverage_ordering_summary.csv",
            estimator_summary,
        )
        write_csv(
            args.output / "controlled_only_not_applicable.csv",
            [
                {
                    "method": method,
                    "status": "not_applicable",
                    "reason": (
                        "requires independently specified true dynamics or "
                        "privileged latent state unavailable in the observational "
                        "fitted-simulator contract"
                    ),
                }
                for method in config["controlled_only_not_applicable"]
            ],
        )
        incomplete = [
            row
            for row in training_rows
            if row.get("convergence_status") == "convergence_incomplete"
        ]
        nonfinite = [
            row for row in all_records if not bool(row.get("finite", False))
        ]
        write_csv(
            args.output / "failure_ledger.csv",
            [
                {
                    "stage": "optimization_qualification",
                    "task": row.get("task", ""),
                    "method": row.get("method", ""),
                    "training_seed": row.get("training_seed", ""),
                    "status": "retained_convergence_incomplete",
                }
                for row in incomplete
            ]
            + [
                {
                    "stage": "policy_estimator_dataset",
                    "task": row["task"],
                    "dataset_index": row["dataset_index"],
                    "method": row["method"],
                    "estimator": row["estimator"],
                    "status": "nonfinite_retained",
                }
                for row in nonfinite
            ]
            or [{"stage": "all", "status": "pass"}],
        )
        privacy = _privacy_rows(args.output)
        write_csv(args.output / "privacy_scan.csv", privacy)
        if any(row["status"] != "pass" for row in privacy):
            raise RuntimeError("privacy scan failure")
        decision = PILOT_DECISION if args.pilot else FULL_DECISION
        (args.output / "decision.md").write_text(
            f"# KDD262 decision\n\n`{decision}`\n", encoding="utf-8"
        )
        (args.output / "result_audit.md").write_text(
            "# KDD262 result audit\n\n"
            f"- Decision: `{decision}`.\n"
            f"- Tasks: {len(tasks)}.\n"
            f"- Policies per task: "
            f"{len(config['common_methods']) + len(config['fitted_specific_methods'])}.\n"
            f"- Common Figure-4-compatible policies per task: "
            f"{len(config['common_methods'])}.\n"
            f"- Estimators: {len(ESTIMATORS)}.\n"
            f"- Independent datasets per task: "
            f"{config['ope']['independent_datasets_per_task']}.\n"
            f"- Complete policy-estimator-dataset rows: {len(all_records):,}.\n"
            f"- Finite interval rows: "
            f"{sum(bool(row['finite']) for row in all_records):,}/{len(all_records):,}.\n"
            f"- Optimization-incomplete training rows retained: {len(incomplete)}.\n"
            f"- Controlled-only privileged references marked not applicable: "
            f"{len(config['controlled_only_not_applicable'])}.\n"
            f"- Runtime seconds: {time.time() - started:.2f}.\n"
            "- The reference is Monte Carlo return in a frozen observational "
            "fitted simulator, not clinical policy value or causal evidence.\n",
            encoding="utf-8",
        )
        return decision
    except Exception as error:
        traceback.print_exc()
        write_csv(
            args.output / "failure_ledger.csv",
            [
                {
                    "stage": "terminal",
                    "status": "stop",
                    "reason": repr(error),
                }
            ],
        )
        (args.output / "decision.md").write_text(
            f"# KDD262 decision\n\n`{STOP_DECISION}`\n", encoding="utf-8"
        )
        (args.output / "result_audit.md").write_text(
            "# KDD262 result audit\n\n"
            f"- Decision: `{STOP_DECISION}`.\n"
            f"- Failure: `{repr(error)}`.\n",
            encoding="utf-8",
        )
        return STOP_DECISION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    decision = run(args)
    return 0 if decision in {PILOT_DECISION, FULL_DECISION} else 1


if __name__ == "__main__":
    raise SystemExit(main())
