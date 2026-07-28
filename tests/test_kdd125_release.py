from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "decision/kdd125"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class KDD125ReleaseTests(unittest.TestCase):
    def test_release_manifest_matches_every_file(self):
        manifest = json.loads((RELEASE / "release_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["release"], "KDD125")
        for name, expected in manifest["artifacts"].items():
            self.assertEqual(sha256(RELEASE / name), expected, name)

    def test_primary_scale_has_five_cohorts_and_af_receipt(self):
        with (RELEASE / "final_cohort_scale.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        primary = {row["cohort"] for row in rows if not row["primary_status"].startswith("excluded_")}
        self.assertEqual(primary, {"sepsis", "heart_failure", "respiratory", "shock", "aki"})
        af = next(row for row in rows if row["cohort"] == "af_flutter")
        self.assertEqual((af["subjects"], af["episodes"]), ("9963", "11912"))

    def test_ope_atlas_keeps_noncomputable_and_undefined_cells(self):
        for name in ("complete_retrospective_ope_estimates.csv", "ope_reliability_atlas.csv", "ope_evidence_status.csv"):
            with (RELEASE / name).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            statuses = {row["evidence_status"] for row in rows}
            self.assertIn("not_computable_missing_probability", statuses)
            self.assertIn("estimand_not_defined", statuses)

    def test_small_lineage_counts_are_confined_to_historical_audits(self):
        allowed = {"lineage_replacement_audit.csv", "historical_small_lineage_sensitivity.csv"}
        tokens = {"3440", "4276", "3986", "3,440", "4,276", "3,986"}
        for path in RELEASE.rglob("*"):
            if not path.is_file() or path.suffix not in {".csv", ".md", ".tex", ".json"} or path.name in allowed:
                continue
            text = path.read_text(encoding="utf-8")
            self.assertFalse(tokens & set(text.replace("\n", ",").split(",")), path)


if __name__ == "__main__":
    unittest.main()
