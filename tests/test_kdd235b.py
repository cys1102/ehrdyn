from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from kdd2027_benchmark.entrant_runtime import IsolatedEntrant
from kdd2027_benchmark.world_model_entrant import PROTOCOL_VERSION, validate_policy_output
from kdd2027_benchmark.world_model_full import run_world_model_full


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/full_benchmark/kdd198_v2_generator_contract.json"
ENTRANT = ROOT / "recursive_world_model_entrant/entrant.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class KDD235BTest(unittest.TestCase):
    def test_external_package_has_no_benchmark_import(self) -> None:
        text = (ENTRANT.parent / "entrant.py").read_text(encoding="utf-8")
        self.assertNotIn("kdd2027_benchmark", text)
        self.assertNotIn("full_pomdp", text)
        self.assertNotIn("private", text.lower())

    def test_h4_policy_is_normalized_and_deterministic(self) -> None:
        metadata = {
            "profile": "aki",
            "environment_seed": 171901,
            "feature_count": 3,
            "action_count": 4,
            "supported_actions": [0, 1, 2, 3],
        }
        history = {
            "observations": [[0.0, 0.0, 0.0]],
            "masks": [[1, 1, 1]],
            "recency": [[0.0, 0.0, 0.0]],
            "previous_actions": [0],
            "profile": "aki",
            "action_count": 4,
            "supported_actions": [0, 1, 2, 3],
            "step": 0,
        }
        values = []
        for _ in range(2):
            with IsolatedEntrant(ENTRANT, declaration_schema="world_model_entrant", protocol_version=PROTOCOL_VERSION) as entrant:
                entrant.request("initialize", metadata, 1)
                values.append(entrant.request("predict_policy", history, 2)["result"])
        self.assertEqual(values[0], values[1])
        validate_policy_output(values[0], 1, 4, [0, 1, 2, 3])

    def test_reduced_full_runner_and_deterministic_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            custom_entrant = root / "custom_entrant"
            shutil.copytree(ENTRANT.parent, custom_entrant)
            declaration = custom_entrant / "entrant.json"
            payload = json.loads(declaration.read_text(encoding="utf-8"))
            payload["entrant_id"] = "custom_identity_regression_v1"
            declaration.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            first = root / "first"
            second = root / "second"
            arguments = dict(
                manifest=MANIFEST,
                declaration=declaration,
                forecast_episodes=4,
                direct_episodes=8,
                ope_datasets=2,
                ope_episodes=16,
                workers=1,
                profiles=("heart_failure",),
                environment_seeds=(171901,),
            )
            one = run_world_model_full(output=first, **arguments)
            two = run_world_model_full(output=second, **arguments)
            self.assertEqual(one["forecast_horizon_rows"], 11)
            self.assertEqual(one["direct_return_rows"], 1)
            self.assertEqual(one["ope_summary_rows"], 6)
            for name in (
                "checkpoint_inventory.csv",
                "forecast_horizon_metrics.csv",
                "direct_return_summary.csv",
                "repeated_ope_summary.csv",
            ):
                self.assertEqual(digest(first / name), digest(second / name), name)
                with (first / name).open(newline="", encoding="utf-8") as handle:
                    identity_rows = list(csv.DictReader(handle))
                self.assertEqual(
                    {row["entrant_id"] for row in identity_rows},
                    {"custom_identity_regression_v1"},
                    name,
                )
            with (first / "repeated_ope_summary.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["estimator"] for row in rows}, {"IS", "WIS", "CWPDIS", "DR", "WDR", "FQE"})


if __name__ == "__main__":
    unittest.main()
