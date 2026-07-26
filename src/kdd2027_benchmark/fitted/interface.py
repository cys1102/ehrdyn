"""Restricted constructor-array adapter for the frozen source training code."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


REWARD_CHANNELS = {
    "sepsis": "terminal_discharge_origin_90d_proxy",
    "respiratory_support": "resp_meddreamer_spo2_mbp",
    "shock": "shock_next_mbp_component",
    "aki": "terminal_discharge_origin_90d_proxy",
    "heart_failure": "terminal_discharge_origin_90d_proxy",
}


@dataclass(slots=True)
class RestrictedRoleInterface:
    cohort: str
    role: str
    values: np.ndarray
    masks: np.ndarray
    deltas: np.ndarray
    imputed_history: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    reward_masks: np.ndarray
    termination: np.ndarray
    continuation: np.ndarray
    valid_steps: np.ndarray
    targets: np.ndarray
    target_masks: np.ndarray
    episode_order: np.ndarray
    preprocessing_mean: np.ndarray
    preprocessing_scale: np.ndarray
    reward_names: tuple[str, ...]


def load_restricted_role(
    constructor_root: Path,
    task: str,
    role: str,
) -> RestrictedRoleInterface:
    path = constructor_root / "private_arrays" / f"{task}.{role}.restricted.npz"
    if not path.is_file():
        raise RuntimeError(
            f"required restricted role archive is unavailable: {task}/{role}"
        )
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "state_values",
            "state_masks",
            "log_recency",
            "imputed_history",
            "action_index",
            "reward",
            "reward_mask",
            "termination",
            "continuation",
            "valid_step",
            "target_values",
            "observed_target_mask",
            "episode_order",
            "preprocessing_mean",
            "preprocessing_scale",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise RuntimeError(
                f"restricted role archive fields are unavailable: {','.join(missing)}"
            )
        return RestrictedRoleInterface(
            cohort=task,
            role=role,
            values=np.asarray(archive["state_values"], dtype=np.float32),
            masks=np.asarray(archive["state_masks"], dtype=bool),
            deltas=np.asarray(archive["log_recency"], dtype=np.float32),
            imputed_history=np.asarray(
                archive["imputed_history"], dtype=np.float32
            ),
            actions=np.asarray(archive["action_index"], dtype=np.int16),
            rewards=np.asarray(archive["reward"], dtype=np.float32),
            reward_masks=np.asarray(archive["reward_mask"], dtype=bool),
            termination=np.asarray(archive["termination"], dtype=bool),
            continuation=np.asarray(archive["continuation"], dtype=bool),
            valid_steps=np.asarray(archive["valid_step"], dtype=bool),
            targets=np.asarray(archive["target_values"], dtype=np.float32),
            target_masks=np.asarray(
                archive["observed_target_mask"], dtype=bool
            ),
            episode_order=np.asarray(archive["episode_order"], dtype=np.int16),
            preprocessing_mean=np.asarray(
                archive["preprocessing_mean"], dtype=np.float32
            ),
            preprocessing_scale=np.asarray(
                archive["preprocessing_scale"], dtype=np.float32
            ),
            reward_names=(REWARD_CHANNELS[task],),
        )
