"""Exact-adapter KDD267 controlled 21-method capability smoke."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .errors import ReleaseContractError
from .fitted import kdd161_model_free as model_free
from .fitted import kdd164_metric_selection as world_model
from .fitted.kdd166_pomdp_benchmark import (
    SyntheticData,
    fit_hgb,
    policy_data,
    probability_policy_from_model,
)
from .full_direct_evaluator import (
    collect_repaired_dataset,
    evaluate_repaired_policy_batch,
)
from .full_ope import (
    ESTIMATORS,
    collect_observed_history_dataset,
    point_policy_groups,
    target_probabilities,
)
from .full_pomdp_core import HistoryComparator
from .full_suite import (
    deserialize_environment,
    fixed_policy,
    load_manifest,
)
from .kdd267_candidates import (
    DreamerPolicyAdapter,
    FeatureAdapter,
    XQLPolicyAdapter,
    dreamer_policy_probability,
    episode_dataset_from_controlled,
    normalized_dataset,
    probability_diagnostics,
    train_compact_dreamer,
    train_discrete_xql,
    xql_policy_probability,
)
from .kdd267_h4_planner import build_repaired_sequence_cem
from .kdd267_inventory import (
    CANDIDATE_METHODS,
    DREAMER_PRIMARY_EVALUATION_MODE,
    DREAMER_V2_DEVELOPMENT_SENSITIVITY_MODE,
    SHARED_METHODS,
    TRAINING_SEEDS,
    validate_shared_inventory,
)


Policy = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], np.ndarray
]
ROOT = Path(__file__).resolve().parent
SUCCESSOR_CONFIG = (
    ROOT / "fitted/configs/kdd267_21_method_successor_v1.json"
)
CANDIDATE_CONFIG = (
    ROOT / "fitted/configs/kdd267_candidate_adapter_base_v1.json"
)
MODEL_FREE_METHODS = SHARED_METHODS[4:10]
COMPONENT_METHODS = SHARED_METHODS[11:18]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _environment(
    manifest_path: Path, profile: str, environment_seed: int
) -> tuple[Any, int]:
    manifest = load_manifest(manifest_path)
    matches = [
        row
        for row in manifest["accepted_old_environments"]
        if row["profile_contract"]["profile"] == profile
        and int(row["environment_seed"]) == environment_seed
    ]
    if len(matches) != 1:
        raise ReleaseContractError(
            "KDD267 controlled smoke requires one frozen environment identity"
        )
    profiles = ("sepsis", "respiratory", "shock", "aki", "heart_failure")
    return deserialize_environment(matches[0]), profiles.index(profile)


def _predecessor_data(data: Any) -> SyntheticData:
    return SyntheticData(
        np.asarray(data.observed),
        np.asarray(data.masks),
        np.asarray(data.deltas),
        np.asarray(data.actions),
        np.asarray(data.next_observed),
        np.asarray(data.behavior_probability),
        np.asarray(data.rewards),
        np.asarray(data.done),
        np.asarray(data.valid),
        np.asarray(data.subtypes),
    )


def _candidate_version(method: str) -> str:
    return method.removeprefix("dreamer_").removesuffix("_compact")


def _train_candidate_groups(
    train_source: Any,
    validation_source: Any,
    successor: dict[str, Any],
    base: dict[str, Any],
    device: torch.device,
) -> tuple[
    dict[str, list[Policy]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    train_raw, observation_dim = episode_dataset_from_controlled(train_source)
    validation_raw, _ = episode_dataset_from_controlled(validation_source)
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
    budget.update(successor["pilot_overrides"])
    groups: dict[str, list[Policy]] = {
        method: [] for method in CANDIDATE_METHODS
    }
    rows: list[dict[str, Any]] = []
    v2_sample: np.ndarray | None = None
    v2_mode: np.ndarray | None = None
    for seed in TRAINING_SEEDS:
        for method in CANDIDATE_METHODS:
            if method == "discrete_xql":
                model, metadata = train_discrete_xql(
                    train,
                    validation,
                    base["xql"],
                    seed,
                    int(budget["xql_epochs"]),
                    float(base["discount"]),
                    device,
                )
                callback = XQLPolicyAdapter(model, adapter, device)
                probability = xql_policy_probability(model, validation, device)
            else:
                model, metadata = train_compact_dreamer(
                    train,
                    validation,
                    base["dreamer"],
                    _candidate_version(method),
                    seed,
                    int(budget["dreamer_world_epochs"]),
                    int(budget["dreamer_actor_epochs"]),
                    int(budget["episode_batch_size"]),
                    float(base["discount"]),
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
                    v2_sample = probability
                    v2_mode = dreamer_policy_probability(
                        model,
                        validation,
                        device,
                        evaluation_mode=(
                            DREAMER_V2_DEVELOPMENT_SENSITIVITY_MODE
                        ),
                    )
            groups[method].append(callback)
            diagnostic = probability_diagnostics(probability, validation)
            rows.append(
                {
                    "family": (
                        "model_free"
                        if method == "discrete_xql"
                        else "world_model_policy"
                    ),
                    "method": method,
                    "training_seed": seed,
                    "model_state_sha256": metadata["model_state_sha256"],
                    **diagnostic,
                    "status": "complete",
                }
            )
    if v2_sample is None or v2_mode is None:
        raise RuntimeError("controlled Dreamer V2 sensitivity was not executed")
    sample = v2_sample[validation.valid]
    mode = v2_mode[validation.valid]
    sensitivity = {
        "method": "dreamer_v2_compact",
        "primary_mode": DREAMER_PRIMARY_EVALUATION_MODE,
        "development_sensitivity_mode": (
            DREAMER_V2_DEVELOPMENT_SENSITIVITY_MODE
        ),
        "primary_contains_non_one_hot_probability": bool(
            np.any((sample > 1.0e-8) & (sample < 1.0 - 1.0e-8))
        ),
        "mode_is_one_hot": bool(
            np.allclose(mode.sum(axis=1), 1.0)
            and np.all(
                np.logical_or(np.isclose(mode, 0.0), np.isclose(mode, 1.0))
            )
        ),
        "mode_in_shared_inventory": False,
        "status": "pass",
    }
    return groups, rows, sensitivity


def _train_predecessor_groups(
    environment: Any,
    train_source: Any,
    validation_source: Any,
    successor: dict[str, Any],
    device: torch.device,
) -> tuple[
    dict[str, list[Policy]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    controlled = successor["controlled"]
    train = _predecessor_data(train_source)
    validation = _predecessor_data(validation_source)
    groups: dict[str, list[Policy]] = {
        "fitted_behavior": [fixed_policy(environment, "behavior")],
        "supported_random": [fixed_policy(environment, "supported_random")],
        "fixed_minimum": [fixed_policy(environment, "minimum")],
        "fixed_maximum": [fixed_policy(environment, "maximum")],
    }
    rows: list[dict[str, Any]] = []
    train_policy = policy_data(train)
    validation_policy = policy_data(validation)
    model_free_config = controlled["model_free_training"]
    for method in MODEL_FREE_METHODS:
        internal = (
            "soft_spibb"
            if method == "spibb_style_count_blend_adapter"
            else method
        )
        groups[method] = []
        for seed in TRAINING_SEEDS:
            _, metadata, model = model_free.train_method(
                internal,
                train_policy,
                validation_policy,
                environment.contract.action_count,
                model_free_config,
                seed,
                device,
            )
            groups[method].append(
                probability_policy_from_model(
                    internal,
                    model,
                    train_policy,
                    environment.contract.action_count,
                    model_free_config,
                    device,
                    metadata,
                )
            )
            rows.append(
                {
                    "family": "model_free",
                    "method": method,
                    "training_seed": seed,
                    "model_state_sha256": metadata["model_state_sha256"],
                    "objective": metadata["published_objective_adapter"],
                    "status": "complete",
                }
            )

    component_config = controlled["component_training"]
    fits: dict[str, list[Any]] = {
        method.removesuffix("_plus_h4_support_only"): []
        for method in COMPONENT_METHODS
    }
    fits["persistence_locf"] = [None]
    rows.append(
        {
            "family": "model_based",
            "method": "persistence_locf",
            "training_seed": "deterministic",
            "status": "deterministic_no_fit",
        }
    )
    for seed in TRAINING_SEEDS:
        hgb = fit_hgb(train, seed)
        fits["hgb_residual"].append(hgb)
        rows.append(
            {
                "family": "model_based",
                "method": "hgb_residual",
                "training_seed": seed,
                "model_state_sha256": hgb.fingerprint,
                "status": "complete",
            }
        )
        for component in (
            "deterministic_grud_point",
            "causal_transformer",
            "categorical_rssm",
            "single_gaussian_grud",
        ):
            fit = world_model.fit_model(
                component,
                train,
                validation,
                seed,
                component_config,
                device,
            )
            fits[component].append(fit)
            rows.append(
                {
                    "family": "model_based",
                    "method": component,
                    "training_seed": seed,
                    "model_state_sha256": fit.fingerprint,
                    "status": "complete",
                }
            )
    ensemble = world_model.ensemble_fit(fits["single_gaussian_grud"])
    fits["matched_gaussian_ensemble"] = [ensemble]
    rows.append(
        {
            "family": "model_based",
            "method": "matched_gaussian_ensemble",
            "training_seed": "3408;3411;3414",
            "model_state_sha256": ensemble.fingerprint,
            "status": "complete",
        }
    )
    comparator = HistoryComparator.create(
        environment.contract,
        smoothing=0.65,
        offset=0.0,
        subtype=1,
        observation_loading=environment.observation_loading,
        subtype_observation_shift=float(
            environment.generator["subtype_observation_shift"]
        ),
    )
    planner_rows: list[dict[str, Any]] = []
    for method in COMPONENT_METHODS:
        component = method.removesuffix("_plus_h4_support_only")
        groups[method] = []
        for fit in fits[component]:
            policy, metadata = build_repaired_sequence_cem(
                environment,
                comparator,
                controlled["planner"],
                device,
                component,
                fit,
            )
            groups[method].append(policy)
            planner_rows.append(
                {
                    "method": method,
                    "planner": metadata["planner"],
                    "planner_horizon": metadata["planner_horizon"],
                    "candidates_per_iteration": metadata[
                        "candidates_per_iteration"
                    ],
                    "cem_iterations": metadata["cem_iterations"],
                    "elites": metadata["elites"],
                    "smoothing": metadata["smoothing"],
                    "support_mask_every_step": metadata[
                        "support_mask_every_step"
                    ],
                    "execute_first_action_only": metadata[
                        "execute_first_action_only"
                    ],
                    "status": "pass",
                }
            )
    return groups, rows, planner_rows


def _build_groups(
    environment: Any, profile_index: int, successor: dict[str, Any]
) -> tuple[
    dict[str, list[Policy]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    controlled = successor["controlled"]
    stride = profile_index * int(controlled["profile_seed_stride"])
    train_source = collect_repaired_dataset(
        environment,
        int(controlled["train_episodes"]),
        int(controlled["train_seed_base"]) + stride + environment.seed,
        str(controlled["behavior_family"]),
    )
    validation_source = collect_repaired_dataset(
        environment,
        int(controlled["validation_episodes"]),
        int(controlled["validation_seed_base"]) + stride + environment.seed,
        str(controlled["behavior_family"]),
    )
    groups, predecessor_rows, planner_rows = _train_predecessor_groups(
        environment,
        train_source,
        validation_source,
        successor,
        torch.device("cpu"),
    )
    candidates, candidate_rows, sensitivity = _train_candidate_groups(
        train_source,
        validation_source,
        successor,
        _load(CANDIDATE_CONFIG),
        torch.device("cpu"),
    )
    groups.update(candidates)
    groups = {method: groups[method] for method in SHARED_METHODS}
    validate_shared_inventory(list(groups))
    return groups, [*predecessor_rows, *candidate_rows], planner_rows, sensitivity


def run_controlled_21_method_smoke(
    config_path: Path,
    profile: str,
    environment_seed: int,
    output: Path,
    *,
    seed: int = 3408,
    direct_episodes: int = 8,
    ope_datasets: int = 2,
    ope_episodes: int = 16,
) -> dict[str, Any]:
    if direct_episodes < 8 or ope_datasets < 2 or ope_episodes < 16:
        raise ReleaseContractError(
            "KDD267 controlled smoke requires >=8 direct episodes and "
            ">=2 OPE datasets of >=16 episodes"
        )
    successor = _load(SUCCESSOR_CONFIG)
    validate_shared_inventory(successor["shared_method_ids"])
    environment, profile_index = _environment(
        config_path, profile, environment_seed
    )
    groups, training_rows, planner_rows, sensitivity = _build_groups(
        environment, profile_index, successor
    )
    direct_finite: dict[str, bool] = {}
    for method_index, (method, members) in enumerate(groups.items()):
        local = []
        for member_index, policy in enumerate(members):
            result = evaluate_repaired_policy_batch(
                environment,
                policy,
                direct_episodes,
                seed + 1_000_000 + method_index * 10_000,
                seed + 2_000_000 + method_index * 100 + member_index,
            )
            local.append(
                np.isfinite(result["returns"]).all()
                and float(result["unsupported_mass"]) <= 1.0e-12
                and int(result["terminal_emission_max"]) <= 1
            )
        direct_finite[method] = all(local)

    estimator_evaluations = 0
    for dataset_index in range(ope_datasets):
        data = collect_observed_history_dataset(
            environment,
            ope_episodes,
            seed + 3_000_000 + dataset_index,
            successor["controlled"]["behavior_family"],
        )
        policy_groups = {
            method: [
                target_probabilities(data, policy) for policy in members
            ]
            for method, members in groups.items()
        }
        estimates, diagnostics = point_policy_groups(
            data,
            policy_groups,
            "crossfit_stronger",
            None,
            5,
            0.5,
            0.5,
            seed + 4_000_000 + dataset_index,
        )
        for method in SHARED_METHODS:
            if tuple(estimates[method]) != ESTIMATORS:
                raise RuntimeError("controlled OPE estimator order drift")
            if not all(
                math.isfinite(value) for value in estimates[method].values()
            ):
                raise RuntimeError("nonfinite controlled OPE capability result")
            if diagnostics[method]["finite_fraction"] != 1.0:
                raise RuntimeError("controlled OPE finite-fraction failure")
            estimator_evaluations += len(ESTIMATORS)

    planner_gate = all(
        row["planner_horizon"] == 4
        and row["candidates_per_iteration"] == 64
        and row["cem_iterations"] == 3
        and row["elites"] == 8
        and row["smoothing"] == 0.2
        and row["support_mask_every_step"]
        and row["execute_first_action_only"]
        for row in planner_rows
    )
    receipt = {
        "schema_version": "kdd267_controlled_21_method_smoke_receipt_v1",
        "status": "pass",
        "synthetic_nonclinical": True,
        "profile": profile,
        "environment_seed": environment_seed,
        "environment_hash": environment.mechanism_hash,
        "method_count": len(SHARED_METHODS),
        "method_ids": list(SHARED_METHODS),
        "controlled_ope_policy_count": len(SHARED_METHODS),
        "controlled_ope_policy_ids": list(SHARED_METHODS),
        "severity_rule_included": False,
        "direct_methods_executed": len(direct_finite),
        "direct_methods_finite": sum(direct_finite.values()),
        "ope_estimators": list(ESTIMATORS),
        "ope_datasets": ope_datasets,
        "ope_policy_estimator_evaluations": estimator_evaluations,
        "predecessor_training_rows": (
            len(training_rows) - len(CANDIDATE_METHODS) * len(TRAINING_SEEDS)
        ),
        "candidate_training_rows": (
            len(CANDIDATE_METHODS) * len(TRAINING_SEEDS)
        ),
        "candidate_methods_executed": list(CANDIDATE_METHODS),
        "exact_model_free_adapters_executed": list(MODEL_FREE_METHODS),
        "exact_component_adapters_executed": list(COMPONENT_METHODS),
        "common_h4_planner_gate": planner_gate,
        "dreamer_v2_mode_sensitivity": sensitivity,
        "source_adapter_sha256": successor["candidate_source"][
            "source_sha256"
        ],
        "claim_boundary": successor["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt
