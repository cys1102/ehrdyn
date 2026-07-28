from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kdd2027_benchmark.paper_tasks import (
    PAPER_TASK_IDS,
    validate_paper_task_manifest,
)
from kdd2027_benchmark.resources import (
    PACKAGE_SCHEMA_DIRECTORY,
    controlled_manifest_path,
    ehr_component_example_path,
    paper_task_manifest_path,
    world_model_example_paths,
)


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PostSubmissionArtifactBugfixTests(unittest.TestCase):
    def test_installed_resources_match_repository_authorities(self) -> None:
        pairs = [
            (
                controlled_manifest_path(),
                ROOT
                / "configs/full_benchmark/kdd198_v2_generator_contract.json",
            ),
            (
                ehr_component_example_path(),
                ROOT / "fixtures/kdd245v2r/gaussian.json",
            ),
            (
                paper_task_manifest_path(),
                ROOT / "expected/canonical_v2_five_task_manifest.json",
            ),
        ]
        source_examples = ROOT / "world_model_entrant_example"
        pairs.extend(
            (installed, source_examples / installed.name)
            for installed in world_model_example_paths()
        )
        for installed, authority in pairs:
            with self.subTest(resource=authority.name):
                self.assertTrue(installed.is_file())
                self.assertEqual(digest(installed), digest(authority))

    def test_package_schemas_are_the_default_schema_directory(self) -> None:
        self.assertTrue(PACKAGE_SCHEMA_DIRECTORY.is_dir())
        self.assertEqual(
            len(list(PACKAGE_SCHEMA_DIRECTORY.glob("*.schema.json"))),
            16,
        )

    def test_five_task_paper_route_is_default_reader_identity(self) -> None:
        receipt = validate_paper_task_manifest(paper_task_manifest_path())
        self.assertEqual(tuple(receipt["task_ids"]), PAPER_TASK_IDS)
        self.assertEqual(receipt["paper_task_count"], 5)
        self.assertEqual(
            receipt["legacy_seven_config_surface"],
            "compatibility_only",
        )

    def test_default_component_scorer_runs_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "score.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kdd2027_benchmark.cli",
                    "score-ehr-components",
                    "--output",
                    str(output),
                ],
                cwd=directory,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))[
                    "records_scored"
                ],
                2,
            )

    def test_readme_declares_test_extra_and_scope_boundaries(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("python -m pip install '.[test]'", text)
        self.assertIn("capability", text.lower())
        self.assertIn("full numerical", text.lower())
        self.assertIn("built-in", text.lower())
        self.assertIn("compatibility", text.lower())

    def test_cli_no_longer_guesses_repository_root(self) -> None:
        text = (
            ROOT / "src/kdd2027_benchmark/cli.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("REPOSITORY_ROOT", text)
        self.assertNotIn("Path(__file__).resolve().parents[2]", text)

    def test_world_model_full_has_no_reference_entrant_literal(self) -> None:
        text = (
            ROOT / "src/kdd2027_benchmark/world_model_full.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("kdd235b_recurrent_gaussian_v1", text)

    def test_final_paper_identity_is_bound(self) -> None:
        snapshot = json.loads(
            (
                ROOT / "release/v2.1.1/paper_snapshot.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            snapshot["authoritative_researchwiki_commit"],
            "e05c94e696d3af0e6e4f1e3db72bbcbb1cdcee4c",
        )
        self.assertEqual(
            snapshot["authoritative_manuscript_tex_sha256"],
            "3687d1d70d899ed22dd5847a274b8f08ebbffe24e40ff1bb8f8f1bac690ae5af",
        )
        self.assertEqual(
            snapshot["authoritative_appendix_tex_sha256"],
            "ed9d27a85888c88c9afb3a49809e6a55744e2ca260b1227c461c626e2c8256b8",
        )
        self.assertEqual(
            snapshot["authoritative_pdf_sha256"],
            "5ed66459c82a9ed72bf156512053eca908c377c0a21db8818b7e83513e4b89dd",
        )
        self.assertTrue(snapshot["observed_local_researchwiki_worktree_clean"])
        self.assertEqual(
            snapshot["terminal_decision"],
            "complete_paper_bound_v2_1_1_release",
        )


if __name__ == "__main__":
    unittest.main()
