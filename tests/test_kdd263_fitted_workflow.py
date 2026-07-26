from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("fitted extra is not installed")

from kdd2027_benchmark.fitted.workflow import run_synthetic_smoke


class KDD263FittedWorkflowTests(unittest.TestCase):
    def test_complete_synthetic_fitted_matched_ope_route(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = Path(directory) / "receipt.json"
            receipt = run_synthetic_smoke(receipt_path)
            self.assertEqual(receipt["status"], "pass")
            self.assertEqual(
                receipt["schema_version"], "kdd267_fitted_smoke_receipt_v1"
            )
            self.assertEqual(receipt["matched_methods_per_task"], 21)
            self.assertEqual(receipt["ope_estimators"], 6)
            self.assertFalse(receipt["severity_rule_included"])
            self.assertEqual(receipt["candidate_training_rows"], 12)
            self.assertTrue(
                receipt["dreamer_v2_mode_sensitivity"][0]["mode_is_one_hot"]
            )
            self.assertEqual(json.loads(receipt_path.read_text())["tasks"], 1)
            schema = json.loads(
                (
                    Path(__file__).resolve().parents[1]
                    / "schemas/kdd267_fitted_smoke_receipt.schema.json"
                ).read_text()
            )
            Draft202012Validator(schema).validate(receipt)


if __name__ == "__main__":
    unittest.main()
