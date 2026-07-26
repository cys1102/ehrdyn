"""Small provenance-preserving helpers from the frozen source-fit runner."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .kdd248_full_episode import TaskShape


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flat_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def decision_token(path: Path) -> str:
    tokens = set(
        re.findall(
            r"(?:ready|complete|stop)_[a-z0-9_]+",
            path.read_text(encoding="utf-8"),
        )
    )
    if len(tokens) != 1:
        raise RuntimeError("decision receipt must contain exactly one token")
    return next(iter(tokens))


def task_shape(cfg: dict[str, Any], external_task: str) -> TaskShape:
    feature_dim, action_dim, horizon = cfg["task_shapes"][external_task]
    return TaskShape(external_task, feature_dim, action_dim, horizon)
