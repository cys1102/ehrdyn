from __future__ import annotations

import hashlib
import json
import tomllib
import unittest
from pathlib import Path

import kdd2027_benchmark
from kdd2027_benchmark.ehr_component_scorer import (
    EHR_COMPONENT_BENCHMARK_VERSION,
    EHR_COMPONENT_EVALUATOR_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]


class KDD245V2MMinimalReleaseTests(unittest.TestCase):
    def test_packaging_version_changes_without_scientific_version_change(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        self.assertEqual(project["version"], "2.0.2")
        self.assertEqual(kdd2027_benchmark.PACKAGE_VERSION, "2.0.2")
        self.assertEqual(EHR_COMPONENT_BENCHMARK_VERSION, "ehrdyn-icu-canonical-v2.0.0")
        self.assertEqual(EHR_COMPONENT_EVALUATOR_VERSION, "ehr-component-scorer-v2.0.0")

    def test_prohibited_release_surfaces_are_absent(self) -> None:
        for path in (
            "public_bundle",
            "clinical_review",
            "credentialed",
            "evidence",
            "sbom",
        ):
            self.assertFalse((ROOT / path).exists(), path)
        for pattern in (
            "**/checkpoint*",
            "**/*.pt",
            "**/*.pth",
            "**/*.ckpt",
            "**/*.npy",
            "**/*.npz",
            "**/*bootstrap*.csv",
            "**/*policy*ope*.csv",
            "**/*real*ehr*.csv",
            "**/*figure-data*",
        ):
            self.assertEqual(list(ROOT.glob(pattern)), [], pattern)

    def test_release_manifest_covers_every_runtime_asset(self) -> None:
        if kdd2027_benchmark.PACKAGE_VERSION != "2.0.1":
            self.skipTest("KDD245V2M manifest is the immutable v2.0.1 receipt")
        manifest_path = ROOT / "release" / "kdd245v2m" / "release_manifest.csv"
        if not manifest_path.exists():
            self.skipTest("manifest is generated after the validation run")
        import csv

        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        self.assertEqual(len({row["path"] for row in rows}), len(rows))
        for row in rows:
            path = ROOT / row["path"]
            self.assertTrue(path.is_file(), row["path"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])

    def test_capability_manifest_is_truthful(self) -> None:
        path = ROOT / "release" / "kdd245v2m" / "capability_manifest.csv"
        if not path.exists():
            self.skipTest("manifest is generated after the validation run")
        import csv

        with path.open(newline="", encoding="utf-8") as handle:
            rows = {row["capability"]: row["status"] for row in csv.DictReader(handle)}
        self.assertEqual(rows["synthetic_ehr_component_scoring"], "Yes")
        self.assertEqual(rows["constructed_environment_entrant"], "Yes")
        self.assertEqual(rows["mimic_result_regeneration"], "No")
        self.assertEqual(rows["mimic_derived_result_bundle"], "No")

    def test_canonical_fixture_identity_is_unchanged(self) -> None:
        expected = {
            "point.json": "0473b96eb1f8b78786078737cfd2e02586d6f38e44bbce5d576fb0b3576b1bfb",
            "gaussian.json": "c944bc873f65e27f283a7b68e620ed940adab5139c9d8a4dfab78e754156862c",
            "ensemble.json": "b02239fc8bbc346f36d4867c3cd1751f1ddfaaff3644d28ab0a64e7c98858288",
        }
        fixture_root = ROOT / "fixtures" / "kdd245v2r"
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((fixture_root / name).read_bytes()).hexdigest(), digest)

    def test_task_contract_identity_is_unchanged(self) -> None:
        manifest = json.loads(
            (ROOT / "expected" / "canonical_v2_five_task_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["scientific_surface_sha256"],
            "848be7b3103f6272c020e4ba1a7c23fe278c51bfd4aaf2d7ad53a171569a5505",
        )


if __name__ == "__main__":
    unittest.main()
