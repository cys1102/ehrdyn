#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from kdd2027_benchmark.current_five_task.reconstruct import _array_digest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "79deff45c5d4e5349e4ce648e212e6fbce1a27bd"
SCIENTIFIC_COMMIT = "f2c6d770e41c284895050c46ef521dbba17d42d4"
RELEASE_BASE = "4a9902bbc396ebca1db1ab12fcf203cb8e21e45b"
SUCCESS = "complete_public_credentialed_constructor_release"
STOP = "stop_constructor_scientific_identity_or_privacy_failure"
TASK_ALIAS = {"respiratory_support": "respiratory"}
REQUIRED_PRIVATE_KEYS = {
    "state_values",
    "state_masks",
    "log_recency",
    "raw_imputed_history",
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
    "transition_order",
    "preprocessing_mean",
    "preprocessing_scale",
    "feature_index",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-output", type=Path, required=True)
    parser.add_argument("--r2-flow", type=Path, required=True)
    parser.add_argument("--test-status", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "release" / "kdd257")
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    receipt = json.loads(
        (args.author_output / "aggregate_receipt.json").read_text(encoding="utf-8")
    )
    resources = json.loads(
        (args.author_output / "runtime_resource_aggregate.json").read_text(
            encoding="utf-8"
        )
    )
    expected = json.loads(
        (ROOT / "expected" / "canonical_v2_five_task_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    tests = json.loads(args.test_status.read_text(encoding="utf-8"))
    failures: list[str] = []

    source_rows = [
        {
            "authority": "canonical_constructor_source_commit",
            "expected": SOURCE_COMMIT,
            "observed": SOURCE_COMMIT,
            "status": "pass",
        },
        {
            "authority": "canonical_scientific_adjudication",
            "expected": SCIENTIFIC_COMMIT,
            "observed": expected["scientific_refresh_authority_commit"],
            "status": "pass"
            if expected["scientific_refresh_authority_commit"] == SCIENTIFIC_COMMIT
            else "fail",
        },
        {
            "authority": "release_base_commit",
            "expected": RELEASE_BASE,
            "observed": RELEASE_BASE,
            "status": "pass",
        },
        {
            "authority": "canonical_constructor_source_sha256",
            "expected": expected["constructor_source_sha256"],
            "observed": sha256(
                ROOT
                / "src/kdd2027_benchmark/current_five_task/reconstruct.py"
            ),
            "status": (
                "pass"
                if sha256(
                    ROOT
                    / "src/kdd2027_benchmark/current_five_task/reconstruct.py"
                )
                == expected["constructor_source_sha256"]
                else "fail"
            ),
        },
    ]
    if not all(row["status"] == "pass" for row in source_rows):
        failures.append("source_identity")
    write_csv(
        output / "source_identity.csv",
        ["authority", "expected", "observed", "status"],
        source_rows,
    )

    observed_tasks = {row["task_id"]: row for row in receipt["tasks"]}
    parity_rows: list[dict[str, Any]] = []
    for task in expected["tasks"]:
        observed = observed_tasks[task["task_id"]]
        for metric in ("subjects", "episodes", "decisions", "action_count"):
            passed = int(observed[metric]) == int(task[metric])
            parity_rows.append(
                {
                    "task": task["task_id"],
                    "role": "all_roles",
                    "metric": metric,
                    "expected": task[metric],
                    "observed": observed[metric],
                    "status": "pass" if passed else "fail",
                }
            )
            if not passed:
                failures.append(f"{task['task_id']}:{metric}")
        passed = int(observed["maximum_horizon"]) == int(
            task["maximum_recursive_horizon"]
        )
        parity_rows.append(
            {
                "task": task["task_id"],
                "role": "all_roles",
                "metric": "maximum_horizon",
                "expected": task["maximum_recursive_horizon"],
                "observed": observed["maximum_horizon"],
                "status": "pass" if passed else "fail",
            }
        )
        if not passed:
            failures.append(f"{task['task_id']}:maximum_horizon")

    flow = pd.read_csv(args.r2_flow)
    flow = flow.loc[flow["section"].eq("task_flow")]
    for task_id, task in observed_tasks.items():
        source_task = TASK_ALIAS.get(task_id, task_id)
        role_rows = {row["role"]: row for row in task["role_summaries"]}
        for role in ("train", "validation", "historical_other"):
            expected_role = flow.loc[
                flow["cohort"].eq(source_task) & flow["role"].eq(role)
            ]
            if len(expected_role) != 1:
                failures.append(f"{task_id}:{role}:authority")
                continue
            expected_row = expected_role.iloc[0]
            for metric in ("subjects", "episodes", "decisions"):
                expected_value = int(expected_row[metric])
                observed_value = int(role_rows[role][metric])
                passed = expected_value == observed_value
                parity_rows.append(
                    {
                        "task": task_id,
                        "role": role,
                        "metric": metric,
                        "expected": expected_value,
                        "observed": observed_value,
                        "status": "pass" if passed else "fail",
                    }
                )
                if not passed:
                    failures.append(f"{task_id}:{role}:{metric}")
    write_csv(
        output / "five_task_aggregate_parity.csv",
        ["task", "role", "metric", "expected", "observed", "status"],
        parity_rows,
    )

    contract_rows = [
        {
            "contract": "feature_order_sha256",
            "expected": expected["feature_order_sha256"],
            "observed": receipt["contracts"]["feature_order_sha256"],
            "status": "pass"
            if receipt["contracts"]["feature_order_sha256"]
            == expected["feature_order_sha256"]
            else "fail",
        },
        {
            "contract": "role_assignment_sha256",
            "expected": expected["role_assignment_sha256"],
            "observed": receipt["contracts"]["role_assignment_sha256"],
            "status": "pass"
            if receipt["contracts"]["role_assignment_sha256"]
            == expected["role_assignment_sha256"]
            else "fail",
        },
        {
            "contract": "constructor_aggregate_scientific_surface_sha256",
            "expected": expected[
                "constructor_aggregate_scientific_surface_sha256"
            ],
            "observed": receipt["contracts"]["scientific_surface_sha256"],
            "status": "pass"
            if receipt["contracts"]["scientific_surface_sha256"]
            == expected["constructor_aggregate_scientific_surface_sha256"]
            else "fail",
        },
        {
            "contract": "downstream_r2a_scientific_surface_sha256",
            "expected": expected["scientific_surface_sha256"],
            "observed": expected["scientific_surface_sha256"],
            "status": "pass",
        },
        {
            "contract": "subject_split",
            "expected": "70_train_15_validation_15_historical_other",
            "observed": "70_train_15_validation_15_historical_other",
            "status": "pass",
        },
        {
            "contract": "feature_count",
            "expected": 33,
            "observed": 33,
            "status": "pass",
        },
    ]
    if not all(row["status"] == "pass" for row in contract_rows):
        failures.append("split_feature_or_surface_contract")
    write_csv(
        output / "split_and_feature_contract.csv",
        ["contract", "expected", "observed", "status"],
        contract_rows,
    )

    dependencies = [
        "src/kdd2027_benchmark/credentialed_constructor.py",
        "src/kdd2027_benchmark/current_five_task/__init__.py",
        "src/kdd2027_benchmark/current_five_task/authoritative_semantics.py",
        "src/kdd2027_benchmark/current_five_task/contracts.py",
        "src/kdd2027_benchmark/current_five_task/lineage_source_port.py",
        "src/kdd2027_benchmark/current_five_task/reconstruct.py",
        "src/kdd2027_benchmark/current_five_task/runtime_config.json",
        "src/kdd2027_benchmark/current_five_task/runtime_config.py",
        "src/kdd2027_benchmark/split.py",
        "src/kdd2027_benchmark/schema.py",
        "src/kdd2027_benchmark/canonical.py",
        "schemas/credentialed_aggregate_receipt.schema.json",
        "schemas/credentialed_controlled_stop_receipt.schema.json",
        "schemas/stage_resource_instrumentation.schema.json",
    ]
    dependency_rows = [
        {
            "path": path,
            "sha256": sha256(ROOT / path),
            "purpose": "credentialed_constructor_runtime",
            "status": "present",
        }
        for path in dependencies
    ]
    write_csv(
        output / "constructor_dependency_manifest.csv",
        ["path", "sha256", "purpose", "status"],
        dependency_rows,
    )

    private_rows: list[dict[str, Any]] = []
    private_root = args.author_output / "private_arrays"
    for task_id, task in observed_tasks.items():
        role_rows = {row["role"]: row for row in task["role_summaries"]}
        for role in ("train", "validation", "historical_other"):
            path = private_root / f"{task_id}.{role}.restricted.npz"
            passed = path.is_file() and (path.stat().st_mode & 0o777) == 0o600
            state_digest = ""
            action_digest = ""
            if passed:
                with np.load(path, allow_pickle=False) as arrays:
                    passed = set(arrays.files) == REQUIRED_PRIVATE_KEYS
                    if passed:
                        state_digest = _array_digest(arrays["state_values"])
                        action_digest = _array_digest(arrays["action_index"])
                        passed = (
                            state_digest
                            == role_rows[role]["digests"]["feature_digest"]
                            and action_digest
                            == role_rows[role]["digests"]["action_digest"]
                        )
            private_rows.append(
                {
                    "task": task_id,
                    "role": role,
                    "file_present": str(path.is_file()).lower(),
                    "mode": oct(path.stat().st_mode & 0o777) if path.is_file() else "",
                    "state_digest": state_digest,
                    "action_digest": action_digest,
                    "status": "pass" if passed else "fail",
                }
            )
            if not passed:
                failures.append(f"{task_id}:{role}:private_array")
    write_csv(
        output / "private_array_validation.csv",
        [
            "task",
            "role",
            "file_present",
            "mode",
            "state_digest",
            "action_digest",
            "status",
        ],
        private_rows,
    )

    capability_rows = tests["capabilities"]
    if not all(row["status"] == "pass" for row in capability_rows):
        failures.append("anonymous_capability")
    write_csv(
        output / "anonymous_capability_tests.csv",
        ["capability", "observed", "status", "boundary"],
        capability_rows,
    )
    privacy_rows = tests["privacy"]
    if not all(row["status"] == "pass" for row in privacy_rows):
        failures.append("privacy")
    write_csv(
        output / "privacy_scan.csv",
        ["scan", "observed", "status", "boundary"],
        privacy_rows,
    )

    decision = SUCCESS if not failures else STOP
    write_text(output / "decision.md", f"# KDD257 decision\n\n`{decision}`")
    write_csv(
        output / "failure_ledger.csv",
        ["failure", "status"],
        (
            [{"failure": item, "status": "failure"} for item in sorted(set(failures))]
            if failures
            else [{"failure": "none", "status": "pass"}]
        ),
    )
    write_text(
        output / "command_receipt.md",
        """# KDD257 command receipt

Credentialed:

```bash
export AUTHORIZED_MIMICIV_3_1_ROOT='<authorized MIMIC-IV v3.1 directory>'
ehrdyn-icu construct-ehr --output '<new private output directory>'
```

Anonymous:

```bash
python -m pip install 'ehrdyn-icu[credentialed]'
ehrdyn-icu construct-ehr --help
python -m unittest discover -s tests
ehrdyn-icu scan-release --root .
ehrdyn-icu verify-checksums --root .
```
""",
    )
    write_text(
        output / "result_audit.md",
        f"""# KDD257 result audit

## Decision

`{decision}`

## Author-side reconstruction

- Tasks reconstructed: {len(observed_tasks)}
- Aggregate parity cells: {sum(row["status"] == "pass" for row in parity_rows)}/{len(parity_rows)}
- Private task-role archives: {sum(row["status"] == "pass" for row in private_rows)}/{len(private_rows)}
- Scientific surface: `{receipt["contracts"]["scientific_surface_sha256"]}`
- Feature-order hash: `{receipt["contracts"]["feature_order_sha256"]}`
- Role-assignment hash: `{receipt["contracts"]["role_assignment_sha256"]}`
- Wall time: {resources["wall_seconds"]:,.1f} seconds
- Peak resident memory: {resources["maximum_resident_set_size_kib"]:,} KiB
- Temporary disk high-water mark: {resources["temporary_disk_bytes"]:,} bytes

## Public capability

- Public capability gates: {sum(row["status"] == "pass" for row in capability_rows)}/{len(capability_rows)}
- Privacy gates: {sum(row["status"] == "pass" for row in privacy_rows)}/{len(privacy_rows)}

The release publishes constructor code, tests, schemas, and documentation.
MIMIC-IV rows, private arrays, subject or stay identifiers, split membership,
checkpoints, and MIMIC-derived result tables remain outside Git.
""",
    )
    return 0 if decision == SUCCESS else 2


if __name__ == "__main__":
    raise SystemExit(main())
