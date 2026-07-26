from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class KDD263PaperIdentityTests(unittest.TestCase):
    def test_submitted_commit_and_hashes_are_bound(self) -> None:
        snapshot = json.loads(
            (ROOT / "release/kdd263/paper_snapshot.json").read_text()
        )
        self.assertEqual(
            snapshot["submitted_researchwiki_commit"],
            "146a02840570368ac7a3efd106cf10b0d3cbba91",
        )
        self.assertEqual(
            snapshot["publication_decision"],
            "complete_paper_bound_public_benchmark_artifact",
        )

    def test_public_scope_excludes_clinical_artifacts(self) -> None:
        scope = json.loads((ROOT / "release/kdd263/public_scope.json").read_text())
        self.assertIn("trained checkpoints", scope["excluded"])
        self.assertIn("MIMIC-derived result tables", scope["excluded"])


if __name__ == "__main__":
    unittest.main()
