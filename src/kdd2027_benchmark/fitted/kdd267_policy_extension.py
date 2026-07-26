"""KDD267 fitted-policy extension for the four KDD264 candidate adapters."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..kdd267_candidates import (
    DreamerPolicyAdapter,
    FeatureAdapter,
    XQLPolicyAdapter,
    dreamer_policy_probability,
    episode_dataset_from_fitted,
    normalized_dataset,
    probability_diagnostics,
    train_compact_dreamer,
    train_discrete_xql,
    xql_policy_probability,
)
from ..kdd267_inventory import (
    CANDIDATE_METHODS,
    DREAMER_PRIMARY_EVALUATION_MODE,
    DREAMER_V2_DEVELOPMENT_SENSITIVITY_MODE,
    SHARED_METHODS,
    TRAINING_SEEDS,
    validate_shared_inventory,
)
from .kdd248_full_episode import ResponseRegime
from .run_kdd262_matched_fitted_simulator_ope import (
    FrozenTask,
    PolicyGroup,
    probability_safe,
)


ROOT = Path(__file__).resolve().parent
SUCCESSOR_CONFIG = ROOT / "configs/kdd267_21_method_successor_v1.json"
CANDIDATE_CONFIG = ROOT / "configs/kdd267_candidate_adapter_base_v1.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _version(method: str) -> str:
    if method.startswith("dreamer_v") and method.endswith("_compact"):
        return method.removeprefix("dreamer_").removesuffix("_compact")
    raise ValueError(f"not a compact Dreamer method: {method}")


def _family(method: str) -> str:
    return "model_free" if method == "discrete_xql" else "world_model_policy"


def _support(task: FrozenTask, probability_floor: float) -> np.ndarray:
    initial = np.asarray(task.profile.initial_action_probability, dtype=float)
    transition = np.asarray(
        task.profile.action_transition_probability, dtype=float
    )
    support = initial > probability_floor
    if transition.ndim == 3:
        support |= np.any(transition > probability_floor, axis=(0, 1))
    else:
        support |= np.any(transition > probability_floor, axis=0)
    return support


def extend_fitted_policy_groups(
    task: FrozenTask,
    predecessor_groups: list[PolicyGroup],
    matched_config: dict[str, Any],
    task_index: int,
    device: torch.device,
    *,
    pilot: bool,
) -> tuple[list[PolicyGroup], list[dict[str, Any]], dict[str, Any]]:
    """Train the four additions and return the exact shared 21-method order."""
    successor = _load(SUCCESSOR_CONFIG)
    base = _load(CANDIDATE_CONFIG)
    validate_shared_inventory(successor["shared_method_ids"])
    if tuple(successor["training_seeds"]) != TRAINING_SEEDS:
        raise RuntimeError("KDD267 candidate training-seed identity drift")
    if (
        successor["dreamer_primary_evaluation_mode"]
        != DREAMER_PRIMARY_EVALUATION_MODE
    ):
        raise RuntimeError("KDD267 Dreamer primary evaluation mode drift")

    stride = task_index * int(matched_config["task_seed_stride"])
    roles = matched_config["roles"]
    train_batch = task.simulator.simulate(
        int(roles["policy_train"]["episodes"]),
        int(roles["policy_train"]["seed"]) + stride,
        ResponseRegime.OBSERVED,
    )
    validation_batch = task.simulator.simulate(
        int(roles["policy_validation"]["episodes"]),
        int(roles["policy_validation"]["seed"]) + stride,
        ResponseRegime.OBSERVED,
    )
    train_raw, observation_dim = episode_dataset_from_fitted(train_batch)
    validation_raw, _ = episode_dataset_from_fitted(validation_batch)
    support = _support(
        task, float(matched_config["planner"]["support_probability_floor"])
    )
    if not np.array_equal(train_raw.support, support):
        raise RuntimeError(f"KDD267 training/profile support mismatch: {task.name}")
    adapter = FeatureAdapter.fit(
        train_raw.features,
        train_raw.valid,
        observation_dim,
        train_raw.action_count,
        train_raw.horizon,
        minimum_scale=float(base["feature_adapter"]["minimum_scale"]),
    )
    train = normalized_dataset(train_raw, adapter)
    validation = normalized_dataset(validation_raw, adapter)
    budget = dict(successor["training"])
    if pilot:
        budget.update(successor["pilot_overrides"])

    members: dict[str, list[Any]] = {method: [] for method in CANDIDATE_METHODS}
    identities: dict[str, list[str]] = {
        method: [] for method in CANDIDATE_METHODS
    }
    rows: list[dict[str, Any]] = []
    v2_sample_probability: np.ndarray | None = None
    v2_mode_probability: np.ndarray | None = None
    for seed in TRAINING_SEEDS:
        for method in CANDIDATE_METHODS:
            if method == "discrete_xql":
                model, metadata = train_discrete_xql(
                    train,
                    validation,
                    base["xql"],
                    seed,
                    int(budget["xql_epochs"]),
                    float(matched_config["discount"]),
                    device,
                )
                callback = XQLPolicyAdapter(model, adapter, device)
                probability = xql_policy_probability(model, validation, device)
            else:
                version = _version(method)
                model, metadata = train_compact_dreamer(
                    train,
                    validation,
                    base["dreamer"],
                    version,
                    seed,
                    int(budget["dreamer_world_epochs"]),
                    int(budget["dreamer_actor_epochs"]),
                    int(budget["episode_batch_size"]),
                    float(matched_config["discount"]),
                    device,
                )
                callback = DreamerPolicyAdapter(
                    model,
                    adapter,
                    device,
                    evaluation_mode=DREAMER_PRIMARY_EVALUATION_MODE,
                )
                probability = dreamer_policy_probability(
                    model,
                    validation,
                    device,
                    evaluation_mode=DREAMER_PRIMARY_EVALUATION_MODE,
                )
                if method == "dreamer_v2_compact" and seed == TRAINING_SEEDS[0]:
                    v2_sample_probability = probability
                    v2_mode_probability = dreamer_policy_probability(
                        model,
                        validation,
                        device,
                        evaluation_mode=DREAMER_V2_DEVELOPMENT_SENSITIVITY_MODE,
                    )
            diagnostic = probability_diagnostics(probability, validation)
            identity = str(metadata["model_state_sha256"])
            members[method].append(probability_safe(callback))
            identities[method].append(identity)
            reproducible = {
                key: value
                for key, value in metadata.items()
                if key != "runtime_seconds"
            }
            rows.append(
                {
                    "surface": "fitted",
                    "task": task.name,
                    "method": method,
                    "family": _family(method),
                    "training_seed": seed,
                    "evaluation_mode": (
                        DREAMER_PRIMARY_EVALUATION_MODE
                        if method.startswith("dreamer_")
                        else "probability"
                    ),
                    **reproducible,
                    **diagnostic,
                    "source_adapter_sha256": successor["candidate_source"][
                        "source_sha256"
                    ],
                    "status": "complete",
                }
            )

    additions = {
        method: PolicyGroup(
            method,
            _family(method),
            members[method],
            identities[method],
        )
        for method in CANDIDATE_METHODS
    }
    predecessor = {
        group.name: group
        for group in predecessor_groups
        if group.name in SHARED_METHODS
    }
    combined = {**predecessor, **additions}
    if set(combined) != set(SHARED_METHODS):
        raise RuntimeError("KDD267 fitted method coverage mismatch")
    ordered = [combined[method] for method in SHARED_METHODS]
    validate_shared_inventory([group.name for group in ordered])
    if v2_sample_probability is None or v2_mode_probability is None:
        raise RuntimeError("KDD267 Dreamer V2 sensitivity was not exercised")
    valid = validation.valid
    sample_local = v2_sample_probability[valid]
    mode_local = v2_mode_probability[valid]
    sensitivity = {
        "method": "dreamer_v2_compact",
        "primary_mode": DREAMER_PRIMARY_EVALUATION_MODE,
        "development_sensitivity_mode": (
            DREAMER_V2_DEVELOPMENT_SENSITIVITY_MODE
        ),
        "primary_contains_non_one_hot_probability": bool(
            np.any(
                (sample_local > 1.0e-8)
                & (sample_local < 1.0 - 1.0e-8)
            )
        ),
        "mode_is_one_hot": bool(
            np.allclose(mode_local.sum(axis=1), 1.0)
            and np.all(
                np.logical_or(
                    np.isclose(mode_local, 0.0), np.isclose(mode_local, 1.0)
                )
            )
        ),
        "mode_in_shared_inventory": False,
        "status": "pass",
    }
    return ordered, rows, sensitivity
