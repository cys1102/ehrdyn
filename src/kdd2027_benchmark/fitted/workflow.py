"""Paper-bound fitted-simulator workflow.

The credentialed path consumes only the private role archives produced by
``construct-ehr``.  The public smoke path uses synthetic arrays and executes
the same source-fit, calibrated rollout, matched-policy, and six-estimator
code paths at the frozen pilot scale.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .interface import load_restricted_role
from .kdd248_full_episode import (
    CalibratedRolloutProfile,
    LearnedSourceSimulator,
    ResponseRegime,
    TaskShape,
)
from .kdd248_source_training import (
    build_calibrated_rollout_profile,
    calibrated_rollout_profile_payload,
    fit_source_components,
    load_source_components,
    prepare_role,
)
from .run_kdd262_matched_fitted_simulator_ope import (
    FrozenTask,
    _effective_config,
    direct_references,
    evaluate_ope_datasets,
    train_policy_groups,
)
from .kdd267_policy_extension import extend_fitted_policy_groups
from ..kdd267_inventory import (
    OPE_ESTIMATORS,
    SHARED_METHODS,
    TRAINING_SEEDS,
    validate_shared_inventory,
)


ROOT = Path(__file__).resolve().parent
WORKFLOW_CONFIG = ROOT / "configs/kdd263_paper_fitted_workflow_v1.json"
MATCHED_CONFIG = ROOT / "configs/kdd262_matched_fitted_simulator_ope_v1.json"
CONTRACT_CONFIG = ROOT / "configs/kdd248e3c_contract_repair_v1.json"
SUCCESSOR_CONFIG = ROOT / "configs/kdd267_21_method_successor_v1.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task(config: dict[str, Any], name: str) -> TaskShape:
    feature, actions, horizon = config["task_shapes"][name]
    return TaskShape(name, int(feature), int(actions), int(horizon))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile_from_path(path: Path) -> CalibratedRolloutProfile:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["binary_feature_indices"] = tuple(payload["binary_feature_indices"])
    return CalibratedRolloutProfile(**payload)


def validate_constructor_roles(constructor_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    receipt_path = constructor_root / "aggregate_receipt.json"
    role_receipt_path = constructor_root / "simulator_role_receipt.json"
    if not receipt_path.is_file() or not role_receipt_path.is_file():
        raise RuntimeError("constructor aggregate and simulator-role receipts are required")
    receipt = _load(receipt_path)
    role_receipt = _load(role_receipt_path)
    expected_tasks = list(config["tasks"])
    observed_tasks = [str(row["task_id"]) for row in receipt.get("tasks", [])]
    if observed_tasks != expected_tasks:
        raise RuntimeError("constructor task inventory does not match paper contract")
    if role_receipt.get("private_arrays_written") is not True:
        raise RuntimeError("fitted workflow requires restricted role archives")
    tasks = role_receipt.get("tasks", [])
    if [str(row.get("task_id")) for row in tasks] != expected_tasks:
        raise RuntimeError("simulator role receipt task inventory mismatch")
    if any(int(row["subject_overlap"]["intersection_subjects"]) != 0 for row in tasks):
        raise RuntimeError("simulator train/calibration subject overlap detected")
    role_counts: dict[str, dict[str, int]] = {}
    for task in expected_tasks:
        role_counts[task] = {}
        for role in ("source_model_train", "source_model_calibration"):
            interface = load_restricted_role(constructor_root, task, role)
            expected = next(
                item for item in next(row for row in tasks if row["task_id"] == task)["roles"]
                if item["role"] == role
            )
            if int(expected["episodes"]) <= 0:
                raise RuntimeError(f"required fitted role is empty: {task}/{role}")
            if interface.values.shape[1] > int(config["task_shapes"][task][2]):
                raise RuntimeError("restricted role horizon exceeds frozen task horizon")
            if interface.values.shape[2] != int(config["task_shapes"][task][0]):
                raise RuntimeError("restricted role feature dimension mismatch")
            if interface.actions.max(initial=-1) >= int(config["task_shapes"][task][1]):
                raise RuntimeError("restricted role action exceeds frozen action space")
            role_counts[task][role] = int(expected["episodes"])
    return {
        "tasks": len(expected_tasks),
        "role_counts": role_counts,
        "feature_order": "canonical_safe_33",
        "private_outputs_written": True,
        "row_level_fields_exported": False,
    }


def _synthetic_role(task: TaskShape, episodes: int, seed: int, role: str) -> Any:
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 1.0, (episodes, task.horizon, task.feature_dim)).astype(np.float32)
    masks = rng.random(values.shape) > 0.08
    valid = np.ones((episodes, task.horizon), dtype=bool)
    if task.horizon > 2:
        lengths = rng.integers(max(2, task.horizon // 2), task.horizon + 1, episodes)
        for index, length in enumerate(lengths):
            valid[index, int(length) :] = False
    actions = rng.integers(0, task.action_dim, (episodes, task.horizon), dtype=np.int16)
    actions[~valid] = -1
    targets = (values + rng.normal(0.0, 0.15, values.shape)).astype(np.float32)
    target_masks = masks & valid[..., None]
    termination = np.zeros_like(valid)
    for index in range(episodes):
        termination[index, max(0, int(valid[index].sum()) - 1)] = True
    rewards = np.zeros((episodes, task.horizon, 1), dtype=np.float32)
    reward_masks = np.zeros_like(rewards, dtype=bool)
    for index in range(episodes):
        terminal = max(0, int(valid[index].sum()) - 1)
        rewards[index, terminal, 0] = 1.0 if values[index, terminal, 0] > 0 else -1.0
        reward_masks[index, terminal, 0] = True
    return SimpleNamespace(
        cohort=task.name,
        role=role,
        values=values,
        masks=masks,
        deltas=np.broadcast_to(np.arange(task.horizon, dtype=np.float32)[None, :, None], values.shape).copy(),
        imputed_history=values.copy(),
        actions=actions,
        rewards=rewards,
        reward_masks=reward_masks,
        termination=termination,
        continuation=valid & ~termination,
        valid_steps=valid,
        targets=targets,
        target_masks=target_masks,
        preprocessing_mean=np.zeros(task.feature_dim, dtype=np.float32),
        preprocessing_scale=np.ones(task.feature_dim, dtype=np.float32),
        reward_names=("terminal_discharge_origin_90d_proxy",),
    )


def _fit_one(
    task: TaskShape,
    train_interface: Any,
    calibration_interface: Any,
    validation_interface: Any,
    source_config: dict[str, Any],
    seed: int,
    checkpoint_root: Path,
    device: str,
) -> Any:
    training = dict(source_config["training"])
    training.update(
        {
            "device": device,
            "hidden_dim": int(source_config["hidden_dim"]),
            "latent_dim": int(source_config["latent_dim"]),
        }
    )
    return fit_source_components(
        task=task,
        model_family="gaussian_recurrent",
        seed=seed,
        train=prepare_role(train_interface, train_interface.reward_names[0], task.action_dim),
        calibration=prepare_role(calibration_interface, calibration_interface.reward_names[0], task.action_dim),
        validation=prepare_role(validation_interface, validation_interface.reward_names[0], task.action_dim),
        training=training,
        loss_weights=source_config["loss_weights"],
        perturbation_seed=int(source_config["perturbation_seed"]),
        checkpoint_root=checkpoint_root,
    )


def _fit_profile(
    task: TaskShape,
    fit: Any,
    calibration_interface: Any,
    contract: dict[str, Any],
    rollout: dict[str, Any],
    device: str,
) -> CalibratedRolloutProfile:
    prepared = prepare_role(
        calibration_interface, calibration_interface.reward_names[0], task.action_dim
    )
    return build_calibrated_rollout_profile(
        components=fit.components,
        calibration=prepared,
        task=task,
        feature_names=list(contract["feature_names"]),
        variable_ranges=contract["variable_range_definition"],
        batch_size=int(contract["training"]["batch_size"]),
        device=torch.device(device),
        learned_signal_weight=float(rollout["learned_signal_weight"]),
        persistence_weight=float(rollout["persistence_weight"]),
    )


def run_synthetic_smoke(output: Path) -> dict[str, Any]:
    workflow = _load(WORKFLOW_CONFIG)
    matched = _load(MATCHED_CONFIG)
    contract = _load(CONTRACT_CONFIG)
    successor = _load(SUCCESSOR_CONFIG)
    validate_shared_inventory(successor["shared_method_ids"])
    source = workflow["source_model"]
    training = dict(source["training"])
    training.update({"maximum_epochs": 2, "minimum_epochs": 1, "early_stopping_patience": 1, "convergence_window": 1, "batch_size": 16})
    source = dict(source)
    source["training"] = training
    source["hidden_dim"] = 8
    source["latent_dim"] = 8
    matched = _effective_config(matched, pilot=True)
    device = "cpu"
    with tempfile.TemporaryDirectory(prefix="ehrdyn-kdd263-smoke-") as directory:
        restricted = Path(directory)
        tasks: list[FrozenTask] = []
        for task_index, name in enumerate(matched["tasks"]):
            task = _task(workflow, name)
            train = _synthetic_role(task, 48, 26000 + task_index, "source_model_train")
            calibration = _synthetic_role(task, 32, 27000 + task_index, "source_model_calibration")
            validation = _synthetic_role(task, 32, 28000 + task_index, "validation")
            fit = _fit_one(task, train, calibration, validation, source, 3408 + task_index, restricted / "checkpoints", device)
            profile = _fit_profile(task, fit, calibration, contract, workflow["rollout_profile"], device)
            simulator = LearnedSourceSimulator(task, fit.components, rollout_profile=profile)
            tasks.append(FrozenTask(name, task, simulator, profile, fit.checkpoint_sha256, "synthetic"))
        policy_count = 0
        estimator_count = 0
        direct_count = 0
        candidate_training_rows = 0
        sensitivity_rows: list[dict[str, Any]] = []
        for task_index, task in enumerate(tasks):
            groups, encoder, support, _, _ = train_policy_groups(task, matched, torch.device(device), task_index)
            groups, candidates, sensitivity = extend_fitted_policy_groups(
                task,
                groups,
                matched,
                task_index,
                torch.device(device),
                pilot=True,
            )
            candidate_training_rows += len(candidates)
            sensitivity_rows.append(sensitivity)
            policy_count += len(groups)
            direct, consistency, normalizer, references = direct_references(task, groups, matched, task_index)
            if any(row["status"] != "pass" for row in consistency):
                raise RuntimeError("synthetic direct-reference consistency failure")
            local_output = restricted / f"ope_{task.name}"
            local_output.mkdir(parents=True, exist_ok=True)
            records, _, _ = evaluate_ope_datasets(task, groups, encoder, support, references, normalizer, matched, local_output, task_index)
            direct_count += len(direct)
            estimator_count += len({row["estimator"] for row in records})
    receipt = {
        "schema_version": "kdd267_fitted_smoke_receipt_v1",
        "status": "pass",
        "synthetic": True,
        "tasks": len(tasks),
        "matched_methods_per_task": len(SHARED_METHODS),
        "method_ids": list(SHARED_METHODS),
        "additional_fitted_methods_per_task": 0,
        "severity_rule_included": False,
        "training_seeds": list(TRAINING_SEEDS),
        "ope_estimators": len(OPE_ESTIMATORS),
        "estimator_ids": list(OPE_ESTIMATORS),
        "policy_groups": policy_count,
        "direct_reference_rows": direct_count,
        "distinct_estimators_observed": estimator_count,
        "candidate_training_rows": candidate_training_rows,
        "dreamer_v2_mode_sensitivity": sensitivity_rows,
        "claim_boundary": workflow["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def train_credentialed(constructor_root: Path, restricted_root: Path, output: Path, device: str, pilot: bool) -> dict[str, Any]:
    workflow = _load(WORKFLOW_CONFIG)
    contract = _load(CONTRACT_CONFIG)
    source = dict(workflow["source_model"])
    if pilot:
        source["training"] = dict(source["training"])
        source["training"].update({"maximum_epochs": 2, "minimum_epochs": 1, "early_stopping_patience": 1, "convergence_window": 1, "batch_size": 16})
    validation = validate_constructor_roles(constructor_root, workflow)
    restricted_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = restricted_root / "checkpoints"
    profile_root = restricted_root / "profiles"
    rows = []
    for task_name in workflow["tasks"]:
        task = _task(workflow, task_name)
        train = load_restricted_role(constructor_root, task_name, "source_model_train")
        calibration = load_restricted_role(constructor_root, task_name, "source_model_calibration")
        validation_role = load_restricted_role(constructor_root, task_name, "validation")
        fits = []
        for seed in source["candidate_seeds"]:
            fit = _fit_one(task, train, calibration, validation_role, source, int(seed), checkpoint_root, device)
            fits.append(fit)
            rows.append({"task": task_name, "seed": int(seed), "checkpoint_sha256": fit.checkpoint_sha256, "selected_epoch": fit.selected_epoch, "validation_score": min(fit.validation_curve), "status": fit.convergence_status})
        selected_seed = int(source["frozen_selected_seed"][task_name])
        selected = next(fit for fit in fits if fit.logical_checkpoint_id.endswith(f"seed{selected_seed}"))
        profile = _fit_profile(task, selected, calibration, contract, workflow["rollout_profile"], device)
        profile_path = profile_root / f"{task_name}__calibrated_profile.pt"
        torch.save(calibrated_rollout_profile_payload(profile), profile_path)
        rows.append({"task": task_name, "selected_seed": selected_seed, "profile_sha256": _sha256(profile_path), "status": "selected"})
    receipt = {
        "schema_version": "kdd267_fitted_training_receipt_v1",
        "status": "pass",
        "validation": validation,
        "tasks": len(workflow["tasks"]),
        "checkpoint_rows": rows,
        "matched_method_ids": list(SHARED_METHODS),
        "training_seeds": list(TRAINING_SEEDS),
        "severity_rule_included": False,
        "private_outputs_written": True,
        "claim_boundary": workflow["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def evaluate_credentialed(constructor_root: Path, restricted_root: Path, output: Path, device: str, pilot: bool) -> dict[str, Any]:
    workflow = _load(WORKFLOW_CONFIG)
    matched = _effective_config(_load(MATCHED_CONFIG), pilot=pilot)
    source = workflow["source_model"]
    tasks: list[FrozenTask] = []
    for name in matched["tasks"]:
        task = _task(workflow, name)
        selected_seed = int(source["frozen_selected_seed"][name])
        checkpoint = restricted_root / "checkpoints" / f"{name}__gaussian_recurrent__seed{selected_seed}.pt"
        profile_path = restricted_root / "profiles" / f"{name}__calibrated_profile.pt"
        if not checkpoint.is_file() or not profile_path.is_file():
            raise RuntimeError("selected fitted checkpoint/profile is unavailable")
        training = dict(source["training"])
        training.update({"hidden_dim": int(source["hidden_dim"]), "latent_dim": int(source["latent_dim"])})
        components, _ = load_source_components(checkpoint, task, "gaussian_recurrent", selected_seed, int(source["hidden_dim"]), int(source["latent_dim"]), torch.device(device))
        profile = _profile_from_path(profile_path)
        tasks.append(FrozenTask(name, task, LearnedSourceSimulator(task, components, rollout_profile=profile), profile, _sha256(checkpoint), _sha256(profile_path)))
    output.mkdir(parents=True, exist_ok=True)
    direct_count = 0
    ope_count = 0
    candidate_training_rows = 0
    sensitivity_rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        groups, encoder, support, _, _ = train_policy_groups(task, matched, torch.device(device), task_index)
        groups, candidates, sensitivity = extend_fitted_policy_groups(
            task,
            groups,
            matched,
            task_index,
            torch.device(device),
            pilot=pilot,
        )
        candidate_training_rows += len(candidates)
        sensitivity_rows.append(sensitivity)
        direct, consistency, normalizer, references = direct_references(task, groups, matched, task_index)
        if any(row["status"] != "pass" for row in consistency):
            raise RuntimeError(f"direct-reference consistency failure for {task.name}")
        records, _, _ = evaluate_ope_datasets(task, groups, encoder, support, references, normalizer, matched, output / task.name, task_index)
        direct_count += len(direct)
        ope_count += len(records)
    receipt = {
        "schema_version": "kdd267_fitted_evaluation_receipt_v1",
        "status": "pass",
        "tasks": len(tasks),
        "matched_methods_per_task": len(SHARED_METHODS),
        "method_ids": list(SHARED_METHODS),
        "severity_rule_included": False,
        "ope_estimators": len(OPE_ESTIMATORS),
        "direct_reference_rows": direct_count,
        "policy_estimator_rows": ope_count,
        "candidate_training_rows": candidate_training_rows,
        "dreamer_v2_mode_sensitivity": sensitivity_rows,
        "private_outputs_written": True,
        "claim_boundary": workflow["claim_boundary"],
    }
    (output / "aggregate_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt
