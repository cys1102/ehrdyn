from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from .canonical import load_strict_json
from .errors import ReleaseContractError


PAPER_TASK_IDS = (
    "sepsis",
    "respiratory_support",
    "shock",
    "aki",
    "heart_failure",
)
PAPER_ACTION_COUNTS = (25, 25, 25, 4, 2)
PAPER_MAXIMUM_RECURSIVE_HORIZONS = (11, 10, 11, 11, 10)


def validate_paper_task_manifest(path: Path) -> dict[str, Any]:
    value = load_strict_json(path)
    if not isinstance(value, dict):
        raise ReleaseContractError("Paper task manifest must be an object")
    manifest = cast(dict[str, Any], value)
    if manifest.get("schema_version") != "canonical-v2-five-task-manifest-v1":
        raise ReleaseContractError("Paper task manifest schema drift")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(PAPER_TASK_IDS):
        raise ReleaseContractError("Paper task manifest must contain five tasks")
    task_ids = tuple(row.get("task_id") for row in tasks if isinstance(row, dict))
    action_counts = tuple(
        row.get("action_count") for row in tasks if isinstance(row, dict)
    )
    horizons = tuple(
        row.get("maximum_recursive_horizon")
        for row in tasks
        if isinstance(row, dict)
    )
    if task_ids != PAPER_TASK_IDS:
        raise ReleaseContractError("Paper task order or identity drift")
    if action_counts != PAPER_ACTION_COUNTS:
        raise ReleaseContractError("Paper task action-count drift")
    if horizons != PAPER_MAXIMUM_RECURSIVE_HORIZONS:
        raise ReleaseContractError("Paper task recursive-horizon drift")
    surface = manifest.get("scientific_surface_sha256")
    if not isinstance(surface, str) or len(surface) != 64:
        raise ReleaseContractError("Paper scientific-surface identity is invalid")
    return {
        "product": "EHRDyn",
        "paper_task_count": len(PAPER_TASK_IDS),
        "task_ids": list(PAPER_TASK_IDS),
        "action_counts": list(PAPER_ACTION_COUNTS),
        "maximum_recursive_horizons": list(
            PAPER_MAXIMUM_RECURSIVE_HORIZONS
        ),
        "scientific_surface_sha256": surface,
        "legacy_seven_config_surface": "compatibility_only",
        "pass": True,
    }
