from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from kdd2027_benchmark.errors import ReleaseContractError
from kdd2027_benchmark.kdd267_inventory import (
    CANDIDATE_METHODS,
    FORBIDDEN_SEVERITY_IDENTIFIERS,
    SHARED_METHODS,
    inventory_receipt,
    validate_shared_inventory,
)


ROOT = Path(__file__).resolve().parents[1]


class KDD267InventoryTests(unittest.TestCase):
    def test_exact_shared_order_and_no_severity_exception(self) -> None:
        receipt = inventory_receipt()
        self.assertEqual(receipt["method_count"], 21)
        self.assertEqual(tuple(receipt["method_ids"]), SHARED_METHODS)
        self.assertEqual(receipt["fitted_method_ids"], receipt["method_ids"])
        self.assertEqual(receipt["controlled_method_ids"], receipt["method_ids"])
        self.assertEqual(
            receipt["controlled_ope_policy_ids"], receipt["method_ids"]
        )
        self.assertFalse(receipt["severity_rule_included"])
        self.assertTrue(
            FORBIDDEN_SEVERITY_IDENTIFIERS.isdisjoint(receipt["method_ids"])
        )

    def test_missing_duplicate_reordered_unknown_and_severity_are_rejected(
        self,
    ) -> None:
        invalid = [
            SHARED_METHODS[:-1],
            (*SHARED_METHODS[:-1], SHARED_METHODS[-2]),
            (SHARED_METHODS[1], SHARED_METHODS[0], *SHARED_METHODS[2:]),
            (*SHARED_METHODS[:-1], "unknown_method"),
            (*SHARED_METHODS[:-1], "observed_history_severity_rule"),
        ]
        for methods in invalid:
            with self.subTest(methods=methods):
                with self.assertRaises(ReleaseContractError):
                    validate_shared_inventory(methods)

    def test_receipt_schema_enforces_order(self) -> None:
        schema = json.loads(
            (
                ROOT
                / "schemas/kdd267_method_inventory_receipt.schema.json"
            ).read_text(encoding="utf-8")
        )
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(inventory_receipt())

    def test_all_kdd267_schemas_are_valid_draft_2020_12(self) -> None:
        for path in sorted((ROOT / "schemas").glob("kdd267_*.schema.json")):
            with self.subTest(path=path.name):
                Draft202012Validator.check_schema(
                    json.loads(path.read_text(encoding="utf-8"))
                )

    def test_candidate_source_is_exact_kdd264_copy(self) -> None:
        path = (
            ROOT / "src/kdd2027_benchmark/kdd267_candidates.py"
        )
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "9df9363764cb47b1d49de87fe9c403baa567430f240565be9a3138d4c5c2739d",
        )
        self.assertEqual(
            hashlib.sha256(
                (
                    ROOT
                    / "src/kdd2027_benchmark/fitted/configs/kdd267_candidate_adapter_base_v1.json"
                ).read_bytes()
            ).hexdigest(),
            "54972e4f149d1dcb5bb1ef02bfae628214b4cca47e54be654b78bd8f2203d8e2",
        )
        self.assertEqual(
            CANDIDATE_METHODS,
            (
                "discrete_xql",
                "dreamer_v1_compact",
                "dreamer_v2_compact",
                "dreamer_v3_compact",
            ),
        )

    def test_controlled_adapter_artifact_hashes_are_frozen(self) -> None:
        successor = json.loads(
            (
                ROOT
                / "src/kdd2027_benchmark/fitted/configs/kdd267_21_method_successor_v1.json"
            ).read_text(encoding="utf-8")
        )
        for name, identity in successor["controlled_sources"].items():
            if name == "repository_commit":
                continue
            with self.subTest(adapter=name):
                artifact = ROOT / "src" / identity["artifact_path"]
                self.assertEqual(
                    hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    identity["artifact_sha256"],
                )

    def test_cli_and_python_inventory_are_identical(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "kdd2027_benchmark.cli", "method-inventory"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(completed.stdout), inventory_receipt())
        self.assertEqual(
            json.loads(
                (
                    ROOT / "release/kdd267/method_inventory.json"
                ).read_text(encoding="utf-8")
            ),
            inventory_receipt(),
        )

    def test_publication_is_bound_to_revised_paper_and_aggregate(self) -> None:
        snapshot = json.loads(
            (
                ROOT / "release/kdd267/paper_snapshot.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            snapshot["submitted_researchwiki_commit"],
            "f6b449e1a8fd906bf7ca8b68bafec2ea9ca1a0e6",
        )
        self.assertFalse(snapshot["required_severity_rule_included"])
        self.assertEqual(snapshot["required_controlled_ope_policy_count"], 21)
        self.assertEqual(
            snapshot["controlled_ope_aggregate_identity_sha256"],
            "b4a0cabc09e7a27cdf9f2b549dfe7ab3483f37cf8d1798a057962f4a00c18647",
        )
        self.assertEqual(
            snapshot["publication_decision"],
            "complete_21_method_paper_bound_public_benchmark_artifact",
        )

    def test_credentialed_fitted_surface_is_structurally_bound(self) -> None:
        successor = json.loads(
            (
                ROOT
                / "src/kdd2027_benchmark/fitted/configs/kdd267_21_method_successor_v1.json"
            ).read_text(encoding="utf-8")
        )
        workflow = json.loads(
            (
                ROOT
                / "src/kdd2027_benchmark/fitted/configs/kdd263_paper_fitted_workflow_v1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(len(workflow["tasks"]), 5)
        self.assertEqual(tuple(successor["shared_method_ids"]), SHARED_METHODS)
        self.assertEqual(successor["training_seeds"], [3408, 3411, 3414])
        self.assertFalse(successor["severity_rule_included"])


@unittest.skipIf(
    importlib.util.find_spec("torch") is None,
    "fitted extra is not installed",
)
class KDD267ControlledSmokeTests(unittest.TestCase):
    def test_all_candidates_execute_on_controlled_surface(self) -> None:
        from kdd2027_benchmark.kdd267_controlled import (
            run_controlled_21_method_smoke,
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            receipt = run_controlled_21_method_smoke(
                ROOT
                / "configs/full_benchmark/kdd198_v2_generator_contract.json",
                "aki",
                171901,
                output,
            )
            self.assertEqual(receipt["status"], "pass")
            self.assertEqual(receipt["method_count"], 21)
            self.assertEqual(receipt["controlled_ope_policy_count"], 21)
            self.assertEqual(
                tuple(receipt["candidate_methods_executed"]),
                CANDIDATE_METHODS,
            )
            self.assertEqual(receipt["direct_methods_finite"], 21)
            self.assertEqual(receipt["predecessor_training_rows"], 35)
            self.assertEqual(
                len(receipt["exact_model_free_adapters_executed"]), 6
            )
            self.assertEqual(
                len(receipt["exact_component_adapters_executed"]), 7
            )
            self.assertTrue(receipt["common_h4_planner_gate"])
            self.assertEqual(
                receipt["ope_policy_estimator_evaluations"],
                2 * 21 * 6,
            )
            self.assertFalse(receipt["severity_rule_included"])
            self.assertTrue(
                receipt["dreamer_v2_mode_sensitivity"]["mode_is_one_hot"]
            )
            self.assertEqual(json.loads(output.read_text()), receipt)
            schema = json.loads(
                (
                    ROOT
                    / "schemas/kdd267_controlled_smoke_receipt.schema.json"
                ).read_text(encoding="utf-8")
            )
            Draft202012Validator(schema).validate(receipt)
            replay_output = Path(directory) / "replay.json"
            replay = run_controlled_21_method_smoke(
                ROOT
                / "configs/full_benchmark/kdd198_v2_generator_contract.json",
                "aki",
                171901,
                replay_output,
            )
            self.assertEqual(replay, receipt)
            self.assertEqual(replay_output.read_bytes(), output.read_bytes())


if __name__ == "__main__":
    unittest.main()
