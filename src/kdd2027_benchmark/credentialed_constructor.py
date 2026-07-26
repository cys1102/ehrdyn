from __future__ import annotations

import os
import tempfile
import importlib
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .canonical import write_canonical_json
from .errors import ReleaseContractError
from .schema import schema_path
from .current_five_task.contracts import (
    FEATURE_INDEX,
    ContractError,
    ROLE_ORDER,
    SAFE_FEATURE_INDICES,
    TABLES,
)
core = importlib.import_module("kdd2027_benchmark.current_five_task.reconstruct")


MIMIC_ROOT_ENV = "AUTHORIZED_MIMICIV_3_1_ROOT"


@contextmanager
def _single_encoding_view(root: Path) -> Iterator[tuple[Path, dict[str, int]]]:
    with tempfile.TemporaryDirectory(prefix=".ehrdyn-source-view-") as directory:
        view = Path(directory) / "3.1"
        counts = {
            "required_tables": len(TABLES),
            "duplicate_encodings_observed": 0,
            "compressed_sources_selected": 0,
            "plain_sources_selected": 0,
        }
        for table in TABLES:
            plain = root / f"{table.relative}.csv"
            compressed = root / f"{table.relative}.csv.gz"
            available = [path for path in (plain, compressed) if path.is_file()]
            if not available:
                raise ReleaseContractError(
                    f"Required MIMIC-IV table is unavailable: {table.relative}"
                )
            if len(available) == 2:
                counts["duplicate_encodings_observed"] += 1
            selected = compressed if compressed.is_file() else plain
            suffix = ".csv.gz" if selected == compressed else ".csv"
            counts[
                "compressed_sources_selected"
                if selected == compressed
                else "plain_sources_selected"
            ] += 1
            destination = view / f"{table.relative}{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(selected)
        yield view, counts


def construct_from_authorized_mimic(
    output: Path,
    *,
    write_private_arrays: bool = True,
) -> dict[str, Any]:
    raw_root = os.environ.get(MIMIC_ROOT_ENV)
    if not raw_root:
        raise ReleaseContractError(
            f"{MIMIC_ROOT_ENV} must name an authorized MIMIC-IV v3.1 directory"
        )
    mimic_root = Path(raw_root)
    if mimic_root.name != "3.1":
        raise ReleaseContractError(
            f"{MIMIC_ROOT_ENV} must name the explicit MIMIC-IV 3.1 directory"
        )
    if output.exists():
        raise ReleaseContractError("Credentialed output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)

    with _single_encoding_view(mimic_root) as (source_view, encoding_counts):
        with tempfile.TemporaryDirectory(
            prefix=".ehrdyn-private-export-",
            dir=output.parent,
        ) as directory:
            private_export = Path(directory) / "private_arrays"
            original_role_surface = core._role_surface

            def export_role_surface(
                task: str,
                role: str,
                candidates: Any,
                transitions: Any,
                actions: np.ndarray,
                arrays: dict[str, np.ndarray],
                train_mean: np.ndarray,
                train_scale: np.ndarray,
                filled: np.ndarray,
                full_recency: np.ndarray,
            ) -> dict[str, Any]:
                if write_private_arrays and role in ROLE_ORDER:
                    private_export.mkdir(parents=True, exist_ok=True, mode=0o700)
                    _write_role_arrays(
                        task,
                        role,
                        candidates,
                        transitions,
                        actions,
                        arrays,
                        train_mean,
                        train_scale,
                        filled,
                        full_recency,
                        private_export,
                    )
                return original_role_surface(
                    task,
                    role,
                    candidates,
                    transitions,
                    actions,
                    arrays,
                    train_mean,
                    train_scale,
                    filled,
                    full_recency,
                )

            core._role_surface = export_role_surface
            try:
                try:
                    receipt = core.reconstruct(
                        source_view,
                        output,
                        schema_path("credentialed_aggregate_receipt"),
                    )
                except ContractError as error:
                    raise ReleaseContractError(str(error)) from error
            finally:
                core._role_surface = original_role_surface
                if output.exists():
                    os.chmod(output, 0o700)
                    for path in output.iterdir():
                        if path.is_file():
                            os.chmod(path, 0o600)
            if write_private_arrays:
                shutil.move(str(private_export), str(output / "private_arrays"))
    write_canonical_json(output / "source_encoding_view.json", encoding_counts)
    os.chmod(output / "source_encoding_view.json", 0o600)
    return receipt


def _write_role_arrays(
    task: str,
    role: str,
    candidates: Any,
    transitions: Any,
    actions: np.ndarray,
    arrays: dict[str, np.ndarray],
    train_mean: np.ndarray,
    train_scale: np.ndarray,
    filled: np.ndarray,
    full_recency: np.ndarray,
    destination: Path,
) -> None:
    select = transitions["role"].eq(role).to_numpy(bool)
    local_transitions = transitions.loc[select].reset_index(drop=True)
    local_actions = np.asarray(actions)[select]
    episode_vector = local_transitions["episode_idx"].to_numpy(int)
    if len(episode_vector):
        starts = np.r_[0, np.flatnonzero(episode_vector[1:] != episode_vector[:-1]) + 1]
        ends = np.r_[starts[1:], len(episode_vector)]
        episode_ids = episode_vector[starts]
        if len(np.unique(episode_ids)) != len(episode_ids):
            raise ReleaseContractError(
                "Episode transitions are not contiguous in canonical order"
            )
    else:
        starts = ends = episode_ids = np.empty(0, dtype=int)
    lengths = (ends - starts).astype(int)
    max_steps = int(lengths.max()) if len(lengths) else 0
    shape = (len(episode_ids), max_steps, len(SAFE_FEATURE_INDICES))
    prefix = f"export-{task}-{role}"
    state_values = core._working_array(
        f"{prefix}-state-values", shape, np.float32, np.nan
    )
    state_masks = core._working_array(
        f"{prefix}-state-masks", shape, bool, False
    )
    recency = core._working_array(f"{prefix}-recency", shape, np.float32, 0.0)
    raw_imputed = core._working_array(
        f"{prefix}-raw-imputed", shape, np.float32, np.nan
    )
    normalized = core._working_array(
        f"{prefix}-normalized", shape, np.float32, 0.0
    )
    target_values = core._working_array(
        f"{prefix}-target-values", shape, np.float32, np.nan
    )
    target_masks = core._working_array(
        f"{prefix}-target-masks", shape, bool, False
    )
    padded_actions = core._working_array(
        f"{prefix}-actions", (len(episode_ids), max_steps), np.int16, -1
    )
    valid = core._working_array(
        f"{prefix}-valid", (len(episode_ids), max_steps), bool, False
    )
    terminal = core._working_array(
        f"{prefix}-terminal", (len(episode_ids), max_steps), bool, False
    )
    order = core._working_array(
        f"{prefix}-order", (len(episode_ids), max_steps), np.int16, -1
    )
    reward = core._working_array(
        f"{prefix}-reward", (len(episode_ids), max_steps, 1), np.float32, 0.0
    )
    reward_mask = core._working_array(
        f"{prefix}-reward-mask", (len(episode_ids), max_steps, 1), bool, False
    )
    candidate_index = candidates.set_index("episode_idx")
    state_vector = local_transitions["state_idx"].to_numpy(int)
    target_vector = local_transitions["target_idx"].to_numpy(int)
    relative_vector = local_transitions["relative_transition"].to_numpy(np.int16)
    for row_index, (episode_id, start, end) in enumerate(
        zip(episode_ids, starts, ends, strict=True)
    ):
        length = int(end - start)
        episode = episode_vector[start:end]
        state = state_vector[start:end]
        target = target_vector[start:end]
        state_values[row_index, :length] = arrays["values"][episode, state][
            :, SAFE_FEATURE_INDICES
        ]
        state_masks[row_index, :length] = arrays["masks"][episode, state][
            :, SAFE_FEATURE_INDICES
        ]
        target_values[row_index, :length] = arrays["values"][episode, target][
            :, SAFE_FEATURE_INDICES
        ]
        target_masks[row_index, :length] = arrays["masks"][episode, target][
            :, SAFE_FEATURE_INDICES
        ]
        local_imputed = filled[episode, state]
        raw_imputed[row_index, :length] = local_imputed
        local_imputed = np.where(np.isfinite(local_imputed), local_imputed, train_mean)
        recency[row_index, :length] = np.nan_to_num(
            full_recency[episode, state],
            nan=18.0,
            posinf=18.0,
            neginf=0.0,
        )
        normalized[row_index, :length] = np.nan_to_num(
            (local_imputed - train_mean) / train_scale,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        padded_actions[row_index, :length] = local_actions[start:end]
        valid[row_index, :length] = True
        terminal[row_index, length - 1] = True
        order[row_index, :length] = relative_vector[start:end]
        if task in {"sepsis", "aki", "heart_failure"}:
            reward[row_index, length - 1, 0] = (
                -1.0
                if float(candidate_index.loc[int(episode_id), "mortality_90d"]) > 0.5
                else 1.0
            )
            reward_mask[row_index, length - 1, 0] = True
        elif task == "shock":
            index = list(SAFE_FEATURE_INDICES).index(FEATURE_INDEX["mbp"])
            reward_mask[row_index, :length, 0] = target_masks[
                row_index, :length, index
            ]
            reward[row_index, :length, 0] = np.clip(
                (target_values[row_index, :length, index] - 65.0) / 25.0,
                -1.0,
                1.0,
            )
        else:
            safe = list(SAFE_FEATURE_INDICES)
            spo2 = safe.index(FEATURE_INDEX["spo2"])
            mbp = safe.index(FEATURE_INDEX["mbp"])
            reward_mask[row_index, :length, 0] = (
                target_masks[row_index, :length, spo2]
                & target_masks[row_index, :length, mbp]
            )
            reward[row_index, :length, 0] = np.where(
                (target_values[row_index, :length, spo2] >= 94)
                & (target_values[row_index, :length, spo2] <= 98),
                1.0,
                -0.5,
            )
            reward[row_index, :length, 0] += np.where(
                (target_values[row_index, :length, mbp] >= 70)
                & (target_values[row_index, :length, mbp] <= 80),
                1.0,
                -0.5,
            )
    reward[~reward_mask] = 0.0
    continuation = core._working_array(
        f"{prefix}-continuation", valid.shape, bool, False
    )
    continuation[...] = valid & ~terminal
    transition_order = (
        local_transitions[
            ["relative_transition", "state_idx", "action_idx", "target_idx"]
        ].to_numpy(np.int16)
        if len(local_transitions)
        else np.empty((0, 4), np.int16)
    )
    path = destination / f"{task}.{role}.restricted.npz"
    np.savez_compressed(
        path,
        state_values=np.asarray(state_values),
        state_masks=np.asarray(state_masks),
        log_recency=np.asarray(recency),
        raw_imputed_history=np.asarray(raw_imputed),
        imputed_history=np.asarray(normalized),
        action_index=np.asarray(padded_actions),
        reward=np.asarray(reward),
        reward_mask=np.asarray(reward_mask),
        termination=np.asarray(terminal),
        continuation=np.asarray(continuation),
        valid_step=np.asarray(valid),
        target_values=np.asarray(target_values),
        observed_target_mask=np.asarray(target_masks),
        episode_order=np.asarray(order),
        transition_order=transition_order,
        preprocessing_mean=train_mean.astype(np.float32),
        preprocessing_scale=train_scale.astype(np.float32),
        feature_index=np.asarray(SAFE_FEATURE_INDICES, dtype=np.int16),
    )
    os.chmod(path, 0o600)
    core._release_working_arrays(
        (
            state_values,
            state_masks,
            recency,
            raw_imputed,
            normalized,
            target_values,
            target_masks,
            padded_actions,
            valid,
            terminal,
            order,
            reward,
            reward_mask,
            continuation,
        )
    )
