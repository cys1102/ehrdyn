from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

from .errors import ReleaseContractError


PACKAGE_DISTRIBUTION = "ehrdyn-icu"
RESOURCE_PREFIX = ("share", "ehrdyn")
PACKAGE_SCHEMA_DIRECTORY = Path(__file__).resolve().parent / "package_schemas"
EDITABLE_SOURCE_PATHS = {
    (
        "configs",
        "full_benchmark",
        "kdd198_v2_generator_contract.json",
    ): Path("configs/full_benchmark/kdd198_v2_generator_contract.json"),
    (
        "fixtures",
        "kdd245v2r",
        "gaussian.json",
    ): Path("fixtures/kdd245v2r/gaussian.json"),
    (
        "paper",
        "canonical_v2_five_task_manifest.json",
    ): Path("expected/canonical_v2_five_task_manifest.json"),
    (
        "world_model_entrant_example",
        "point.json",
    ): Path("world_model_entrant_example/point.json"),
    (
        "world_model_entrant_example",
        "gaussian.json",
    ): Path("world_model_entrant_example/gaussian.json"),
    (
        "world_model_entrant_example",
        "ensemble.json",
    ): Path("world_model_entrant_example/ensemble.json"),
}


def installed_resource_path(*parts: str) -> Path:
    """Resolve a file installed by this distribution without guessing a repo root."""
    if not parts or any(not part or part in {".", ".."} for part in parts):
        raise ReleaseContractError("Invalid installed resource path")
    try:
        package = distribution(PACKAGE_DISTRIBUTION)
    except PackageNotFoundError as error:
        raise ReleaseContractError(
            "EHRDyn package resources require an installed distribution"
        ) from error
    target = (*RESOURCE_PREFIX, *parts)
    matches: list[Path] = []
    for entry in package.files or ():
        entry_parts = PurePosixPath(str(entry)).parts
        if len(entry_parts) >= len(target) and entry_parts[-len(target) :] == target:
            matches.append(Path(package.locate_file(entry)).resolve())
    if len(matches) == 1 and matches[0].is_file():
        return matches[0]
    direct_url_text = package.read_text("direct_url.json")
    source_relative = EDITABLE_SOURCE_PATHS.get(tuple(parts))
    if direct_url_text is not None and source_relative is not None:
        direct_url = json.loads(direct_url_text)
        if direct_url.get("dir_info", {}).get("editable") is True:
            parsed = urlparse(str(direct_url.get("url", "")))
            if parsed.scheme == "file":
                source_root = Path(unquote(parsed.path)).resolve()
                candidate = source_root / source_relative
                if candidate.is_file():
                    return candidate
    rendered = "/".join(parts)
    raise ReleaseContractError(
        f"Installed EHRDyn resource is unavailable: {rendered}"
    )


def controlled_manifest_path() -> Path:
    return installed_resource_path(
        "configs", "full_benchmark", "kdd198_v2_generator_contract.json"
    )


def ehr_component_example_path() -> Path:
    return installed_resource_path(
        "fixtures", "kdd245v2r", "gaussian.json"
    )


def paper_task_manifest_path() -> Path:
    return installed_resource_path(
        "paper", "canonical_v2_five_task_manifest.json"
    )


def world_model_example_paths() -> tuple[Path, ...]:
    return tuple(
        installed_resource_path("world_model_entrant_example", name)
        for name in ("point.json", "gaussian.json", "ensemble.json")
    )
