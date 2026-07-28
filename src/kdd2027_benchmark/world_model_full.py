"""Full 40-environment validation for an external recursive world-model entrant."""
from __future__ import annotations

import csv
import hashlib
import math
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from .entrant_runtime import IsolatedEntrant, load_entrant
from .full_direct_evaluator import collect_repaired_dataset, evaluate_repaired_policy_batch
from .full_ope import ESTIMATORS, collect_observed_history_dataset, point_policy_groups
from .full_suite import environments
from .world_model_entrant import (
    COVERAGE_LEVELS,
    PROTOCOL_VERSION,
    RETENTION_GRID,
    frozen_h4_contract,
    validate_policy_output,
    validate_prediction,
    validate_recursive_request,
)


FORECAST_SEED_BASE = 2_355_100_000
TRAIN_REWARD_SEED_BASE = 2_355_200_000
DIRECT_ENV_SEED_BASE = 2_355_300_000
DIRECT_POLICY_SEED_BASE = 2_355_400_000
OPE_DATASET_SEED_BASE = 2_355_500_000


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _history(data: Any, step: int = 0) -> dict[str, Any]:
    previous = data.actions[:, max(step - 1, 0)] if step else data.actions[:, 0]
    return {
        "observations": data.observed[:, step].tolist(),
        "masks": data.masks[:, step].astype(int).tolist(),
        "recency": data.deltas[:, step].tolist(),
        "previous_actions": previous.astype(int).tolist(),
    }


def _metric_row(mean: np.ndarray, scale: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    residual = truth - mean
    selected = mask
    count = int(selected.sum())
    if count == 0:
        raise RuntimeError("forecast_metric_has_no_observed_cells")
    squared = residual[selected] ** 2
    absolute = np.abs(residual[selected])
    z = residual / scale
    normal = NormalDist()
    phi = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    cdf = np.vectorize(normal.cdf)(z)
    crps = scale * (z * (2 * cdf - 1) + 2 * phi - 1 / math.sqrt(math.pi))
    nll = np.log(scale * math.sqrt(2 * math.pi)) + 0.5 * z**2
    output: dict[str, float] = {
        "observed_cells": count,
        "rmse": float(np.sqrt(np.mean(squared))),
        "mae": float(np.mean(absolute)),
        "nll": float(np.mean(nll[selected])),
        "crps": float(np.mean(crps[selected])),
    }
    calibration_errors = []
    for level in COVERAGE_LEVELS:
        quantile = normal.inv_cdf((1 + level) / 2)
        empirical = float(np.mean((np.abs(residual) <= quantile * scale)[selected]))
        label = str(int(level * 100))
        output[f"coverage_{label}"] = empirical
        output[f"width_{label}"] = float(np.mean((2 * quantile * scale)[selected]))
        calibration_errors.append(abs(empirical - level))
    output["mace"] = float(np.mean(calibration_errors))
    uncertainty = np.mean(scale, axis=2).reshape(-1)
    row_error = np.mean(residual**2, axis=2).reshape(-1)
    valid_rows = selected.any(axis=2).reshape(-1)
    order = np.argsort(uncertainty[valid_rows], kind="stable")
    losses = row_error[valid_rows][order]
    risks = [float(np.mean(losses[: max(1, int(len(losses) * fraction))])) for fraction in RETENTION_GRID]
    output["risk_coverage_area"] = float(np.trapezoid(risks, RETENTION_GRID))
    return output


def _learned_reward_return(environment: Any, chosen_action: int, episodes: int, seed: int) -> float:
    data = collect_repaired_dataset(environment, episodes, seed, "ehr_matched")
    total = 0.0
    for step in range(environment.horizon):
        rows = data.valid[:, step] & (data.actions[:, step] == chosen_action)
        fallback = data.valid[:, step]
        selected = rows if np.any(rows) else fallback
        total += (environment.discount**step) * float(np.mean(data.rewards[selected, step]))
    return total


def _constant_policy(probability: np.ndarray):
    def policy(observation: np.ndarray, mask: np.ndarray, recency: np.ndarray,
               previous: np.ndarray, step: int) -> np.ndarray:
        return np.broadcast_to(probability, (len(observation), len(probability))).copy()
    return policy


def _ope_one_environment(arguments: tuple[Any, ...]) -> list[dict[str, Any]]:
    (
        entrant_id,
        environment,
        probability,
        direct_return,
        profile_index,
        environment_index,
        datasets,
        episodes,
    ) = arguments
    policy = _constant_policy(np.asarray(probability, dtype=float))
    by_estimator: dict[str, list[float]] = {name: [] for name in ESTIMATORS}
    ess: list[float] = []
    unsupported: list[float] = []
    finite: list[float] = []
    seeds = []
    for dataset_index in range(datasets):
        seed = OPE_DATASET_SEED_BASE + profile_index * 1_000_000 + environment_index * 10_000 + dataset_index
        seeds.append(seed)
        data = collect_observed_history_dataset(environment, episodes, seed, "ehr_matched")
        target = np.broadcast_to(probability, (data.episodes, data.horizon, data.action_count)).copy()
        invalid = ~data.valid
        fallback = int(np.flatnonzero(data.support)[0])
        target[invalid] = 0.0
        target[invalid, fallback] = 1.0
        estimates, diagnostics = point_policy_groups(
            data,
            {"external_recurrent_gaussian_h4": [target]},
            "crossfit_stronger",
            None,
            5,
            0.5,
            0.5,
            seed + 7_000,
        )
        local = estimates["external_recurrent_gaussian_h4"]
        diagnostic = diagnostics["external_recurrent_gaussian_h4"]
        for estimator in ESTIMATORS:
            by_estimator[estimator].append(float(local[estimator]))
        ess.append(float(diagnostic["ess"]))
        unsupported.append(float(diagnostic["unsupported_mass"]))
        finite.append(float(diagnostic["finite_fraction"]))
    seed_digest = hashlib.sha256(",".join(map(str, seeds)).encode()).hexdigest()
    rows = []
    for estimator in ESTIMATORS:
        values = np.asarray(by_estimator[estimator], dtype=float)
        error = values - direct_return
        rows.append({
            "entrant_id": entrant_id,
            "profile": environment.contract.profile,
            "environment_seed": environment.seed,
            "estimator": estimator,
            "denominator": "crossfit_stronger",
            "clipping": "none",
            "dataset_count": datasets,
            "episodes_per_dataset": episodes,
            "dataset_seed_digest": seed_digest,
            "direct_return_reference": direct_return,
            "mean_ope_estimate": float(np.mean(values)),
            "bias": float(np.mean(error)),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "finite_fraction": float(np.mean(np.isfinite(values))),
            "median_ess": float(np.median(ess)),
            "mean_unsupported_mass": float(np.mean(unsupported)),
            "mean_diagnostic_finite_fraction": float(np.mean(finite)),
        })
    return rows


def run_world_model_full(
    manifest: Path,
    declaration: Path,
    output: Path,
    forecast_episodes: int = 32,
    direct_episodes: int = 512,
    ope_datasets: int = 64,
    ope_episodes: int = 256,
    workers: int = 1,
    profiles: tuple[str, ...] | None = None,
    environment_seeds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    entrant_id = str(
        load_entrant(
            declaration,
            schema_name="world_model_entrant",
        )["entrant_id"]
    )
    selected = [
        environment for environment in environments(manifest)
        if (profiles is None or environment.contract.profile in profiles)
        and (environment_seeds is None or environment.seed in environment_seeds)
    ]
    forecast_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    ope_arguments = []
    profile_order = ("sepsis", "respiratory", "shock", "aki", "heart_failure")
    for environment_index, environment in enumerate(selected):
        profile_index = profile_order.index(environment.contract.profile)
        data = collect_repaired_dataset(
            environment,
            forecast_episodes,
            FORECAST_SEED_BASE + profile_index * 10_000 + environment.seed,
            "ehr_matched",
        )
        metadata = {
            "profile": environment.contract.profile,
            "environment_seed": environment.seed,
            "feature_count": environment.contract.feature_dim,
            "action_count": environment.contract.action_count,
            "supported_actions": environment.supported.astype(int).tolist(),
        }
        with IsolatedEntrant(declaration, declaration_schema="world_model_entrant", protocol_version=PROTOCOL_VERSION) as entrant:
            initialized = entrant.request("initialize", metadata, 235_600 + environment_index)["result"]
            fitted = entrant.request(
                "fit_or_load",
                {"roles": ["train", "validation"], "sealed_final_opened": False},
                235_700 + environment_index,
            )["result"]
            history = _history(data)
            request = history | {
                "schema_version": "kdd235a.request.v1",
                "action_sequences": data.actions.astype(int).tolist(),
                "horizon": environment.horizon,
            }
            batch, horizon = validate_recursive_request(
                request,
                environment.contract.feature_dim,
                environment.contract.action_count,
                metadata["supported_actions"],
            )
            result = entrant.request("predict_rollout", request, 235_800 + environment_index)["result"]
            checked = validate_prediction(
                result,
                entrant.declaration,
                batch,
                horizon,
                environment.contract.feature_dim,
            )
            if checked["scale"] is None:
                raise RuntimeError("kdd235b_requires_native_probabilistic_entrant")
            truth = data.next_observed
            valid = data.valid[:, :, None] & data.masks
            for horizon_index in range(environment.horizon):
                metrics = _metric_row(
                    checked["mean"][:, : horizon_index + 1],
                    checked["scale"][:, : horizon_index + 1],
                    truth[:, : horizon_index + 1],
                    valid[:, : horizon_index + 1],
                )
                forecast_rows.append({
                    "entrant_id": entrant.declaration["entrant_id"],
                    "profile": environment.contract.profile,
                    "environment_seed": environment.seed,
                    "horizon": horizon_index + 1,
                    "elapsed_hours": 4 * (horizon_index + 1),
                    "evaluation": "common_origin_recursive",
                    **metrics,
                })
            representative = {key: [value[0]] for key, value in history.items()}
            policy_result = entrant.request(
                "predict_policy",
                representative | {
                    "profile": environment.contract.profile,
                    "action_count": environment.contract.action_count,
                    "supported_actions": metadata["supported_actions"],
                    "step": 0,
                },
                235_900 + environment_index,
            )["result"]
            probability = validate_policy_output(
                policy_result,
                1,
                environment.contract.action_count,
                metadata["supported_actions"],
            )[0]
            checkpoint_hash = hashlib.sha256(
                f"{initialized['checkpoint_id']}|{fitted['checkpoint_id']}|{environment.contract.profile}|{environment.seed}".encode()
            ).hexdigest()
            checkpoint_rows.append({
                "profile": environment.contract.profile,
                "environment_seed": environment.seed,
                "entrant_id": entrant.declaration["entrant_id"],
                "checkpoint_id": fitted["checkpoint_id"],
                "checkpoint_hash": checkpoint_hash,
                "training_role": "public_constructed_train",
                "selection_role": "public_constructed_validation",
                "final_opened_during_fit": False,
            })
        policy = _constant_policy(probability)
        direct = evaluate_repaired_policy_batch(
            environment,
            policy,
            direct_episodes,
            DIRECT_ENV_SEED_BASE + profile_index * 10_000 + environment.seed,
            DIRECT_POLICY_SEED_BASE + profile_index * 10_000 + environment.seed,
        )
        direct_return = float(np.mean(direct["returns"]))
        direct_se = float(np.std(direct["returns"], ddof=1) / math.sqrt(direct_episodes))
        learned_return = _learned_reward_return(
            environment,
            int(np.argmax(probability)),
            max(256, forecast_episodes),
            TRAIN_REWARD_SEED_BASE + profile_index * 10_000 + environment.seed,
        )
        direct_rows.append({
            "entrant_id": entrant_id,
            "profile": environment.contract.profile,
            "environment_seed": environment.seed,
            "planner": frozen_h4_contract()["name"],
            "evaluation_horizon": "full_episode",
            "direct_episode_count": direct_episodes,
            "direct_return": direct_return,
            "direct_return_standard_error": direct_se,
            "learned_model_predicted_return": learned_return,
            "absolute_direct_return_gap": abs(learned_return - direct_return),
            "chosen_action": int(np.argmax(probability)),
            "policy_probability_sum_error": abs(float(np.sum(probability)) - 1.0),
            "unsupported_mass": float(direct["unsupported_mass"]),
            "terminal_emission_max": int(direct["terminal_emission_max"]),
        })
        ope_arguments.append((
            entrant_id,
            environment,
            probability,
            direct_return,
            profile_index,
            environment_index,
            ope_datasets,
            ope_episodes,
        ))
    if workers > 1 and len(ope_arguments) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_ope_one_environment, ope_arguments))
    else:
        parts = [_ope_one_environment(arguments) for arguments in ope_arguments]
    ope_rows = [row for part in parts for row in part]
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "checkpoint_inventory.csv", checkpoint_rows)
    _write_csv(output / "forecast_horizon_metrics.csv", forecast_rows)
    _write_csv(output / "direct_return_summary.csv", direct_rows)
    _write_csv(output / "repeated_ope_summary.csv", ope_rows)
    receipt = {
        "schema_version": "kdd235b.full.v1",
        "environment_count": len(selected),
        "checkpoint_rows": len(checkpoint_rows),
        "forecast_horizon_rows": len(forecast_rows),
        "direct_return_rows": len(direct_rows),
        "ope_summary_rows": len(ope_rows),
        "datasets_per_environment": ope_datasets,
        "episodes_per_dataset": ope_episodes,
        "wall_time_seconds": time.monotonic() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "claim_boundary": "constructed_benchmark_interface_usability_only",
    }
    from .canonical import write_canonical_json
    write_canonical_json(output / "receipt.json", receipt)
    return receipt
