"""Aggregate statistics for the frozen KDD229 repeated-dataset OPE run."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .kdd202b_policy_ope import LoggedOPEData, denominator_probabilities


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        return math.nan, math.nan
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - half, center + half


def weight_tail_diagnostics(
    data: LoggedOPEData,
    policy_groups: dict[str, list[np.ndarray]],
    denominator_name: str,
    clip: float | None,
    folds: int,
    pseudocount: float,
    fold_seed: int,
) -> dict[str, dict[str, float]]:
    denominator = denominator_probabilities(data, denominator_name, folds, pseudocount, fold_seed)
    chosen_denominator = np.take_along_axis(
        denominator, data.actions[:, :, None], axis=2
    )[:, :, 0]
    output: dict[str, dict[str, float]] = {}
    for method, members in policy_groups.items():
        member_rows = []
        for target in members:
            chosen_target = np.take_along_axis(
                target, data.actions[:, :, None], axis=2
            )[:, :, 0]
            ratio = np.divide(
                chosen_target,
                chosen_denominator,
                out=np.zeros_like(chosen_target),
                where=chosen_denominator > 0,
            )
            if clip is not None:
                ratio = np.minimum(ratio, float(clip))
            ratio[~data.valid] = 1.0
            cumulative = np.cumprod(ratio, axis=1)
            final_index = np.maximum(data.valid.sum(axis=1) - 1, 0)
            weights = cumulative[np.arange(data.episodes), final_index]
            member_rows.append({
                "weight_median": float(np.quantile(weights, 0.50)),
                "weight_p95": float(np.quantile(weights, 0.95)),
                "weight_p99": float(np.quantile(weights, 0.99)),
                "weight_max": float(np.max(weights)),
            })
        output[method] = {
            key: float(np.median([row[key] for row in member_rows]))
            for key in member_rows[0]
        }
    return output


def profile_stratified_paired_bootstrap(
    rows: list[dict[str, Any]],
    value_key: str,
    replicates: int,
    seed: int,
) -> tuple[float, float, float, float]:
    profiles = sorted({str(row["profile"]) for row in rows})
    grouped = {profile: [row for row in rows if row["profile"] == profile] for profile in profiles}
    if not profiles or any(not local for local in grouped.values()):
        return math.nan, math.nan, math.nan, math.nan
    values = np.asarray([float(row[value_key]) for row in rows], dtype=float)
    mean = float(np.mean([np.mean([float(row[value_key]) for row in grouped[p]]) for p in profiles]))
    median = float(np.median(values))
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    arrays = {p: np.asarray([float(row[value_key]) for row in grouped[p]], dtype=float) for p in profiles}
    for index in range(replicates):
        draws[index] = np.mean([
            np.mean(local[rng.integers(0, len(local), len(local))])
            for local in arrays.values()
        ])
    low, high = np.quantile(draws, [0.025, 0.975])
    return mean, median, float(low), float(high)


def seed_for_dataset(config: dict[str, Any], profile_index: int, environment_seed: int, dataset_index: int) -> int:
    logged = config["logged_data"]
    return (
        int(logged["dataset_seed_base"])
        + profile_index * int(logged["profile_stride"])
        + environment_seed * int(logged["environment_stride"])
        + dataset_index
    )


def expected_inventory(environments: int, datasets: int, policies: int, estimators: int) -> int:
    return environments * datasets * policies * estimators
