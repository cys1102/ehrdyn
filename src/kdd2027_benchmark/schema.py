from __future__ import annotations

import math
from pathlib import Path
from typing import cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from .canonical import NonFiniteConstant, load_strict_json
from .errors import ReleaseContractError


SCHEMA_FILES = {
    "aggregate_metrics": "aggregate_metrics.schema.json",
    "credentialed_aggregate_receipt": "credentialed_aggregate_receipt.schema.json",
    "credentialed_controlled_stop_receipt": "credentialed_controlled_stop_receipt.schema.json",
    "ehr_component_result": "ehr_component_result.schema.json",
    "ehr_component_submission": "ehr_component_submission.schema.json",
    "entrant_protocol": "entrant_protocol.schema.json",
    "world_model_entrant": "entrant_api_and_component_source.schema.json",
    "world_model_request": "recursive_entrant_request.schema.json",
    "world_model_prediction": "recursive_prediction_output.schema.json",
    "world_model_policy": "planner_adapter_and_policy_output.schema.json",
    "leaderboard_submission": "leaderboard_submission.schema.json",
    "stage_resource_instrumentation": "stage_resource_instrumentation.schema.json",
    "transition_submission": "transition_submission.schema.json",
    "kdd267_method_inventory_receipt": "kdd267_method_inventory_receipt.schema.json",
    "kdd267_fitted_smoke_receipt": "kdd267_fitted_smoke_receipt.schema.json",
    "kdd267_controlled_smoke_receipt": "kdd267_controlled_smoke_receipt.schema.json",
}


def validate_json_document(path: Path, schema_name: str) -> dict[str, object]:
    return validate_json_file(path, schema_path(schema_name))


def schema_path(name: str) -> Path:
    filename = SCHEMA_FILES[name]
    root = Path(__file__).resolve().parents[2]
    path = root / "schemas" / filename
    if not path.is_file():
        path = Path(__file__).resolve().parent / "package_schemas" / filename
    if not path.is_file():
        raise ReleaseContractError(f"Released schema is unavailable: {filename}")
    return path


def validate_schema_file(path: Path) -> dict[str, object]:
    schema = load_strict_json(path)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise ReleaseContractError(_format_error(error, "Schema")) from error
    return {"schema": path.name, "draft": "2020-12", "valid": True}


def validate_schema_directory(path: Path) -> list[dict[str, object]]:
    files = sorted(path.glob("*.schema.json"))
    if set(item.name for item in files) != set(SCHEMA_FILES.values()):
        raise ReleaseContractError("Released schema inventory does not match the frozen schema registry")
    return [validate_schema_file(item) for item in files]


def validate_json_file(path: Path, released_schema: Path) -> dict[str, object]:
    instance = load_strict_json(path)
    validate_instance(instance, released_schema)
    return cast(dict[str, object], instance)


def validate_instance(instance: object, released_schema: Path) -> None:
    _reject_nonfinite(instance, ())
    schema = load_strict_json(released_schema)
    try:
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(instance),
            key=lambda error: (tuple(str(item) for item in error.absolute_path), str(error.validator)),
        )
    except SchemaError as error:
        raise ReleaseContractError(_format_error(error, "Schema")) from error
    if errors:
        raise ReleaseContractError(_format_error(errors[0], "Instance"))


def _reject_nonfinite(value: object, path: tuple[object, ...]) -> None:
    if isinstance(value, NonFiniteConstant):
        raise ReleaseContractError(
            f"Instance validation failed at {_pointer(path)} [number]: nonfinite constant {value.token}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        pointer = _pointer(path)
        raise ReleaseContractError(f"Instance validation failed at {pointer} [number]: value must be finite")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, path + (index,))
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, path + (key,))


def _format_error(error: ValidationError | SchemaError, label: str) -> str:
    return f"{label} validation failed at {_pointer(tuple(error.absolute_path))} [{error.validator}]: {error.message}"


def _pointer(parts: tuple[object, ...]) -> str:
    if not parts:
        return "/"
    escaped = [str(part).replace("~", "~0").replace("/", "~1") for part in parts]
    return "/" + "/".join(escaped)
