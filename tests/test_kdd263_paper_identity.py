from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from kdd2027_benchmark import PACKAGE_VERSION
from kdd2027_benchmark.cli import (
    DEFAULT_CONTROLLED_MANIFEST,
    DEFAULT_EHR_COMPONENT_EXAMPLE,
    build_parser,
)


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

    def test_public_documentation_matches_ehrdyn_benchmark_identity(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        paper_artifact = (ROOT / "PAPER_ARTIFACT.md").read_text(encoding="utf-8")
        self.assertIn("# EHRDyn", readme)
        self.assertIn("21 policy specifications", readme)
        self.assertIn("six OPE", readme)
        self.assertIn("executable artifact for EHRDyn", paper_artifact)
        self.assertIn("contain exactly 21", paper_artifact)
        for text in (readme, paper_artifact):
            self.assertNotIn("stop_missing_frozen_paper_identity", text)
            self.assertNotIn("missing submitted ResearchWiki commit", text)

    def test_cli_version_and_reader_facing_defaults(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "kdd2027_benchmark.cli", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), f"EHRDyn {PACKAGE_VERSION}")

        parser = build_parser()
        controlled = parser.parse_args(
            [
                "controlled-21-method-smoke",
                "--profile",
                "aki",
                "--output",
                "receipt.json",
            ]
        )
        self.assertEqual(controlled.config, DEFAULT_CONTROLLED_MANIFEST)

        component = parser.parse_args(
            ["score-ehr-components", "--output", "score.json"]
        )
        self.assertEqual(component.submission, DEFAULT_EHR_COMPONENT_EXAMPLE)


if __name__ == "__main__":
    unittest.main()
