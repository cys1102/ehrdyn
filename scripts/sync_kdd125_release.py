#!/usr/bin/env python3
"""Sync the additive KDD125 aggregate reader package into decision/kdd125."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "decision/kdd125"
REQUIRED = {
    "README.md", "manuscript.tex", "manuscript.pdf", "method_cards.md", "limitations.md", "artifact_documentation.md",
    "final_cohort_scale.csv", "lineage_replacement_audit.csv",
    "historical_small_lineage_sensitivity.csv", "reward_lineage_and_scale_audit.csv",
    "complete_model_performance_by_cohort.csv", "complete_uncertainty_by_cohort.csv",
    "complete_action_information.csv", "synthesis_statistics.csv", "related_work_and_evidence_landscape.csv",
    "complete_policy_and_planner_inventory.csv", "known_value_to_ehr_reliability_bridge.csv",
    "complete_retrospective_ope_estimates.csv", "ope_reliability_atlas.csv",
    "ope_evidence_status.csv", "manuscript_number_audit.csv", "artifact_hashes.json",
    "privacy_and_release_validation.md", "decision.md",
    "figures/final_cohort_scale.png", "figures/final_cohort_scale.pdf",
    "figures/transition_leader_counts.png", "figures/transition_leader_counts.pdf",
    "figures/ope_evidence_status.png", "figures/ope_evidence_status.pdf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    missing = sorted(name for name in REQUIRED if not (source / name).is_file())
    if missing:
        raise RuntimeError(f"KDD125 package is incomplete: {missing}")
    for name in sorted(REQUIRED):
        destination = DESTINATION / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
    hashes = {name: sha256(DESTINATION / name) for name in sorted(REQUIRED)}
    (DESTINATION / "release_manifest.json").write_text(
        json.dumps({"release": "KDD125", "additive": True, "artifacts": hashes}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
