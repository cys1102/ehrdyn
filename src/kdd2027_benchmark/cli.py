from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from . import BENCHMARK_VERSION
from .baseline import BASELINES, run_baseline
from .canonical import canonical_bytes, write_canonical_json
from .config import validate_config_directory, validate_task_config
from .errors import ReleaseContractError
from .evaluator import evaluate_fixture, evaluate_predictions
from .ehr_component_scorer import score_submission
from .fixture import generate_fixture
from .manifest import validate_paper_manifests
from .privacy import scan_release, verify_checksums
from .report import write_aggregate_report
from .split import deterministic_split
from .schema import validate_schema_directory
from .submission import validate_submission
from .public_bundle import rebuild_public_bundle
from .public_ope import run_public_ope_smoke
from .public_pomdp import run_public_pomdp_smoke
from .transition_entrant import validate_transition_submission
from .world_model_smoke import run_world_model_smoke
from .world_model_full import run_world_model_full
from .full_suite import (
    generate_full_suite,
    run_component_forecasting,
    run_direct_returns,
    run_full_ope,
    summarize_ope,
    validate_entrant_conformance,
)

@dataclass(slots=True)
class CliArgs(argparse.Namespace):
    command: str = ""
    output: Path = Path()
    episodes: int = 8
    seed: int = 3408
    config: Path | None = None
    config_dir: Path | None = None
    fixture: Path = Path()
    task_config: Path = Path()
    baseline: str = ""
    output_fixture: Path = Path()
    input_dir: Path = Path()
    entity_key: str = ""
    root: Path = Path()
    submission: Path = Path()
    predictions: Path = Path()
    episode_key_column: str = "local_episode_key"
    task_manifest: Path = Path()
    contract_manifest: Path = Path()
    evidence: Path = Path()
    profile: str = "aki"
    environment_seed: int = 21201
    datasets: int = 4
    bootstrap: int = 8
    bundle: Path = Path()
    schema_dir: Path = Path()
    entrant: Path = Path()
    manifest: Path = Path()
    input: Path = Path()
    contrasts: Path = Path()
    direct_returns: Path = Path()
    workers: int = 1
    cache_dir: Path | None = None
    entrants: list[Path] | None = None
    forecast_episodes: int = 32
    direct_episodes: int = 512
    ope_datasets: int = 64
    ope_episodes: int = 256
    aggregate_only: bool = False
    fitted_mode: str = "dry-run"
    constructor_root: Path = Path()
    restricted_root: Path = Path()
    device: str = "cpu"
    pilot: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KDD 2027 frozen benchmark clean-room utilities.")
    _ = parser.add_argument("--version", action="version", version=BENCHMARK_VERSION)
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("generate-fixture", help="Generate a deterministic synthetic fixture.")
    _ = fixture.add_argument("--output", type=Path, required=True)
    _ = fixture.add_argument("--episodes", type=int, default=8)
    _ = fixture.add_argument("--seed", type=int, default=3408)
    validate = commands.add_parser("validate-config", help="Validate one task config or the frozen config directory.")
    _ = validate.add_argument("--config", type=Path)
    _ = validate.add_argument("--config-dir", type=Path)
    evaluate = commands.add_parser("evaluate", help="Evaluate schema-compatible synthetic predictions.")
    _ = evaluate.add_argument("--fixture", type=Path, required=True)
    _ = evaluate.add_argument("--task-config", type=Path, required=True)
    _ = evaluate.add_argument("--output", type=Path, required=True)
    evaluate_local = commands.add_parser(
        "evaluate-local",
        help="Aggregate a credentialed local prediction file without exporting row keys.",
    )
    _ = evaluate_local.add_argument("--predictions", type=Path, required=True)
    _ = evaluate_local.add_argument("--episode-key-column", default="local_episode_key")
    _ = evaluate_local.add_argument("--task-config", type=Path, required=True)
    _ = evaluate_local.add_argument("--output", type=Path, required=True)
    baseline = commands.add_parser("baseline", help="Run a deterministic synthetic forecasting baseline.")
    _ = baseline.add_argument("--fixture", type=Path, required=True)
    _ = baseline.add_argument("--task-config", type=Path, required=True)
    _ = baseline.add_argument("--baseline", choices=sorted(BASELINES), required=True)
    _ = baseline.add_argument("--output-fixture", type=Path, required=True)
    _ = baseline.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("aggregate-report", help="Render aggregate metric JSON files as Markdown.")
    _ = report.add_argument("--input-dir", type=Path, required=True)
    _ = report.add_argument("--output", type=Path, required=True)
    split = commands.add_parser("split", help="Assign an opaque local entity key to a deterministic split.")
    _ = split.add_argument("--entity-key", required=True)
    scan = commands.add_parser("scan-release", help="Fail closed on restricted release artifacts.")
    _ = scan.add_argument("--root", type=Path, required=True)
    checksums = commands.add_parser("verify-checksums", help="Verify the frozen public artifact manifest.")
    _ = checksums.add_argument("--root", type=Path, required=True)
    schemas = commands.add_parser("validate-schemas", help="Validate every released JSON Schema as Draft 2020-12.")
    _ = schemas.add_argument("--schema-dir", type=Path, required=True)
    submission = commands.add_parser("validate-submission", help="Validate an aggregate leaderboard submission.")
    _ = submission.add_argument("--submission", type=Path, required=True)
    _ = submission.add_argument("--config-dir", type=Path, required=True)
    manifest = commands.add_parser(
        "validate-manifest",
        help="Validate paper task and contract mappings against public evidence.",
    )
    _ = manifest.add_argument("--task-manifest", type=Path, required=True)
    _ = manifest.add_argument("--contract-manifest", type=Path, required=True)
    _ = manifest.add_argument("--evidence", type=Path, required=True)
    pomdp = commands.add_parser("pomdp-smoke", help="Run the repaired public constructed-POMDP entrant smoke.")
    _ = pomdp.add_argument("--config", type=Path, required=True)
    _ = pomdp.add_argument("--profile", required=True)
    _ = pomdp.add_argument("--environment-seed", type=int, default=21201)
    _ = pomdp.add_argument("--episodes", type=int, default=64)
    _ = pomdp.add_argument("--seed", type=int, default=3408)
    _ = pomdp.add_argument("--output", type=Path, required=True)
    ope = commands.add_parser("ope-smoke", help="Run the bounded KDD202B-compatible repeated-dataset OPE smoke.")
    _ = ope.add_argument("--config", type=Path, required=True)
    _ = ope.add_argument("--profile", required=True)
    _ = ope.add_argument("--environment-seed", type=int, default=21201)
    _ = ope.add_argument("--datasets", type=int, default=4)
    _ = ope.add_argument("--episodes", type=int, default=64)
    _ = ope.add_argument("--bootstrap", type=int, default=8)
    _ = ope.add_argument("--seed", type=int, default=3411)
    _ = ope.add_argument("--output", type=Path, required=True)
    transition = commands.add_parser("validate-transition-submission", help="Validate aggregate transition rows and task/version hashes.")
    _ = transition.add_argument("--submission", type=Path, required=True)
    _ = transition.add_argument("--config-dir", type=Path, required=True)
    rebuild = commands.add_parser("rebuild-public-bundle", help="Deterministically materialize public manuscript tables and figures.")
    _ = rebuild.add_argument("--bundle", type=Path, required=True)
    _ = rebuild.add_argument("--output", type=Path, required=True)
    full = commands.add_parser("generate-full-suite", help="Verify and enumerate the authoritative 40-environment/320-dataset suite.")
    _ = full.add_argument("--manifest", type=Path, required=True)
    _ = full.add_argument("--output", type=Path, required=True)
    _ = full.add_argument("--cache-dir", type=Path)
    entrant = commands.add_parser("validate-entrant", help="Validate and sandbox-probe a KDD215 entrant.")
    _ = entrant.add_argument("--entrant", type=Path, required=True)
    _ = entrant.add_argument("--manifest", type=Path, required=True)
    train = commands.add_parser("train-entrant", help="Validate the entrant training boundary and public role contract.")
    _ = train.add_argument("--entrant", type=Path, required=True)
    _ = train.add_argument("--manifest", type=Path, required=True)
    _ = train.add_argument("--output", type=Path, required=True)
    transition_full = commands.add_parser("evaluate-transition", help="Evaluate a component entrant on all 40 environments.")
    _ = transition_full.add_argument("--entrant", type=Path, required=True)
    _ = transition_full.add_argument("--manifest", type=Path, required=True)
    _ = transition_full.add_argument("--output", type=Path, required=True)
    direct = commands.add_parser("evaluate-policy-return", help="Evaluate a policy entrant by paired full-suite direct return.")
    _ = direct.add_argument("--entrant", type=Path, required=True)
    _ = direct.add_argument("--manifest", type=Path, required=True)
    _ = direct.add_argument("--output", type=Path, required=True)
    _ = direct.add_argument("--contrasts", type=Path, required=True)
    full_ope = commands.add_parser("evaluate-policy-ope", help="Run the frozen 320-dataset repeated-OPE protocol.")
    _ = full_ope.add_argument("--entrant", type=Path, required=True)
    _ = full_ope.add_argument("--manifest", type=Path, required=True)
    _ = full_ope.add_argument("--direct-returns", type=Path, required=True)
    _ = full_ope.add_argument("--workers", type=int, default=1)
    _ = full_ope.add_argument("--output", type=Path, required=True)
    summarize = commands.add_parser("summarize-submission", help="Summarize full-suite entrant OPE results.")
    _ = summarize.add_argument("--input", type=Path, required=True)
    _ = summarize.add_argument("--output", type=Path, required=True)
    world_model = commands.add_parser("evaluate-world-model-smoke", help="Run the KDD235A recursive world-model entrant smoke.")
    _ = world_model.add_argument("--manifest", type=Path, required=True)
    _ = world_model.add_argument("--entrant", dest="entrants", action="append", type=Path, required=True)
    _ = world_model.add_argument("--episodes", type=int, default=8)
    _ = world_model.add_argument("--output", type=Path, required=True)
    world_model_full = commands.add_parser(
        "evaluate-world-model-full",
        help="Run the KDD235B recursive entrant over all 40 constructed environments.",
    )
    _ = world_model_full.add_argument("--manifest", type=Path, required=True)
    _ = world_model_full.add_argument("--entrant", type=Path, required=True)
    _ = world_model_full.add_argument("--output", type=Path, required=True)
    _ = world_model_full.add_argument("--forecast-episodes", type=int, default=32)
    _ = world_model_full.add_argument("--direct-episodes", type=int, default=512)
    _ = world_model_full.add_argument("--ope-datasets", type=int, default=64)
    _ = world_model_full.add_argument("--ope-episodes", type=int, default=256)
    _ = world_model_full.add_argument("--workers", type=int, default=1)
    ehr_components = commands.add_parser(
        "score-ehr-components",
        help="Score local canonical-v2 EHR component predictions and write aggregate-only metrics.",
    )
    _ = ehr_components.add_argument("--submission", type=Path, required=True)
    _ = ehr_components.add_argument("--output", type=Path, required=True)
    constructor = commands.add_parser(
        "construct-ehr",
        help="Build the five credentialed MIMIC-IV interfaces in a private local directory.",
    )
    _ = constructor.add_argument("--output", type=Path, required=True)
    _ = constructor.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Write only the aggregate receipt and omit restricted modeling arrays.",
    )
    fitted = commands.add_parser(
        "fitted-simulator",
        help="Train/calibrate/evaluate the paper-bound EHR-fitted simulator in caller-owned paths.",
    )
    _ = fitted.add_argument("--mode", dest="fitted_mode", choices=("dry-run", "train", "evaluate", "all"), default="dry-run")
    _ = fitted.add_argument("--constructor-root", type=Path, required=True)
    _ = fitted.add_argument("--restricted-root", type=Path)
    _ = fitted.add_argument("--output", type=Path, required=True)
    _ = fitted.add_argument("--device", default="cpu")
    _ = fitted.add_argument("--pilot", action="store_true")
    smoke = commands.add_parser(
        "fitted-synthetic-smoke",
        help="Run the complete fitted/matched/OPE workflow on synthetic fixtures only.",
    )
    _ = smoke.add_argument("--output", type=Path, required=True)
    return parser


def parse_args() -> CliArgs:
    return build_parser().parse_args(namespace=CliArgs())


def main() -> int:
    args = parse_args()
    try:
        return _dispatch(args)
    except ReleaseContractError as error:
        print(f"KDD2027 contract error: {error}")
        return 2


def _dispatch(args: CliArgs) -> int:
    if args.command == "generate-fixture":
        _print_json({"rows": generate_fixture(args.output, args.episodes, args.seed), "synthetic": True})
    elif args.command == "validate-config":
        if bool(args.config) == bool(args.config_dir):
            raise ReleaseContractError("Choose exactly one of --config or --config-dir")
        count = 1 if args.config is not None else len(validate_config_directory(_required_path(args.config_dir)))
        if args.config is not None:
            _ = validate_task_config(args.config)
        _print_json({"valid_configs": count, "benchmark_version": BENCHMARK_VERSION})
    elif args.command == "evaluate":
        config = validate_task_config(args.task_config)
        _write_json(args.output, evaluate_fixture(args.fixture, str(config["task_id"])))
    elif args.command == "evaluate-local":
        config = validate_task_config(args.task_config)
        _write_json(
            args.output,
            evaluate_predictions(args.predictions, str(config["task_id"]), args.episode_key_column),
        )
    elif args.command == "baseline":
        config = validate_task_config(args.task_config)
        _write_json(args.output, run_baseline(args.fixture, args.output_fixture, str(config["task_id"]), args.baseline))
    elif args.command == "aggregate-report":
        _print_json({"aggregate_records": write_aggregate_report(args.input_dir, args.output)})
    elif args.command == "split":
        _print_json({"split": deterministic_split(args.entity_key)})
    elif args.command == "scan-release":
        _print_json(scan_release(args.root))
    elif args.command == "verify-checksums":
        _print_json(verify_checksums(args.root))
    elif args.command == "validate-schemas":
        _print_json({"schemas": validate_schema_directory(args.schema_dir), "pass": True})
    elif args.command == "validate-submission":
        _print_json(validate_submission(args.submission, _required_path(args.config_dir)))
    elif args.command == "validate-manifest":
        _print_json(validate_paper_manifests(args.task_manifest, args.contract_manifest, args.evidence))
    elif args.command == "pomdp-smoke":
        _write_json(args.output, run_public_pomdp_smoke(_required_path(args.config), args.profile, args.environment_seed, args.episodes, args.seed))
    elif args.command == "ope-smoke":
        _write_json(args.output, run_public_ope_smoke(_required_path(args.config), args.profile, args.environment_seed, args.datasets, args.episodes, args.bootstrap, args.seed))
    elif args.command == "validate-transition-submission":
        _print_json(validate_transition_submission(args.submission, _required_path(args.config_dir)))
    elif args.command == "rebuild-public-bundle":
        _print_json(rebuild_public_bundle(args.bundle, args.output))
    elif args.command == "generate-full-suite":
        _print_json(generate_full_suite(args.manifest, args.output, args.cache_dir))
    elif args.command == "validate-entrant":
        _print_json(validate_entrant_conformance(args.entrant, args.manifest))
    elif args.command == "train-entrant":
        receipt = validate_entrant_conformance(args.entrant, args.manifest)
        receipt.update({"training_data_role": "public_constructed_train_only", "checkpoint_selection_role": "public_constructed_validation_only", "final_role_opened": False})
        _write_json(args.output, receipt)
    elif args.command == "evaluate-transition":
        _print_json({"rows": len(run_component_forecasting(args.entrant, args.manifest, args.output)), "environment_count": 40})
    elif args.command == "evaluate-policy-return":
        _print_json({"rows": len(run_direct_returns(args.entrant, args.manifest, args.output, args.contrasts)), "environment_count": 40})
    elif args.command == "evaluate-policy-ope":
        _print_json({"rows": len(run_full_ope(args.entrant, args.manifest, args.direct_returns, args.output, args.workers)), "dataset_count": 320})
    elif args.command == "summarize-submission":
        _print_json({"rows": len(summarize_ope(args.input, args.output))})
    elif args.command == "evaluate-world-model-smoke":
        _print_json(run_world_model_smoke(args.manifest, args.entrants or [], args.output, args.episodes))
    elif args.command == "evaluate-world-model-full":
        _print_json(run_world_model_full(
            args.manifest,
            args.entrant,
            args.output,
            args.forecast_episodes,
            args.direct_episodes,
            args.ope_datasets,
            args.ope_episodes,
            args.workers,
        ))
    elif args.command == "score-ehr-components":
        _write_json(args.output, score_submission(args.submission))
    elif args.command == "construct-ehr":
        from .credentialed_constructor import construct_from_authorized_mimic

        receipt = construct_from_authorized_mimic(
            args.output,
            write_private_arrays=not args.aggregate_only,
        )
        _print_json(
            {
                "tasks": len(receipt["tasks"]),
                "scientific_surface_sha256": receipt["contracts"][
                    "scientific_surface_sha256"
                ],
                "restricted_output_created": True,
            }
        )
    elif args.command == "fitted-synthetic-smoke":
        from .fitted.workflow import run_synthetic_smoke

        _print_json(run_synthetic_smoke(args.output))
    elif args.command == "fitted-simulator":
        from .fitted.workflow import (
            evaluate_credentialed,
            train_credentialed,
            validate_constructor_roles,
        )

        workflow_config = json.loads(
            (Path(__file__).parent / "fitted" / "configs" / "kdd263_paper_fitted_workflow_v1.json").read_text(encoding="utf-8")
        )
        if args.fitted_mode == "dry-run":
            _print_json(validate_constructor_roles(args.constructor_root, workflow_config))
        else:
            restricted = args.restricted_root
            if restricted is None:
                raise ReleaseContractError("--restricted-root is required for fitted training/evaluation")
            if args.fitted_mode == "train":
                _print_json(train_credentialed(args.constructor_root, restricted, args.output, args.device, args.pilot))
            elif args.fitted_mode == "evaluate":
                _print_json(evaluate_credentialed(args.constructor_root, restricted, args.output, args.device, args.pilot))
            else:
                _ = train_credentialed(args.constructor_root, restricted, args.output / "training_receipt.json", args.device, args.pilot)
                _print_json(evaluate_credentialed(args.constructor_root, restricted, args.output / "evaluation", args.device, args.pilot))
    return 0


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    write_canonical_json(path, value)


def _print_json(value: object) -> None:
    sys.stdout.buffer.write(canonical_bytes(value))


def _required_path(path: Path | None) -> Path:
    if path is None:
        raise ReleaseContractError("Required path was not provided")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
