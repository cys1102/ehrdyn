from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("fitted extra is not installed")

from kdd2027_benchmark.fitted.workflow import run_synthetic_smoke


class KDD263FittedWorkflowTests(unittest.TestCase):
    def test_complete_synthetic_fitted_matched_ope_route(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = Path(directory) / "receipt.json"
            receipt = run_synthetic_smoke(receipt_path)
            self.assertEqual(receipt["status"], "pass")
            self.assertEqual(receipt["matched_methods_per_task"], 17)
            self.assertEqual(receipt["ope_estimators"], 6)
            self.assertEqual(json.loads(receipt_path.read_text())["tasks"], 1)


if __name__ == "__main__":
    unittest.main()
