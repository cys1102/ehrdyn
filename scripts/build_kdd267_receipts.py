#!/usr/bin/env python3
"""Regenerate aggregate-safe KDD267 privacy and release manifests."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from kdd2027_benchmark.privacy import scan_release


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release/kdd267"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    privacy = scan_release(ROOT)
    _write_csv(
        RELEASE / "aggregate_privacy_scan.csv",
        ["scope", "files_scanned", "findings", "status"],
        [
            {
                "scope": "complete_public_source_tree",
                "files_scanned": privacy["files_scanned"],
                "findings": privacy["findings"],
                "status": "pass" if privacy["pass"] else "fail",
            }
        ],
    )
    manifest_path = RELEASE / "release_asset_manifest.csv"
    paths = [
        path
        for path in sorted(RELEASE.iterdir())
        if path.is_file() and path != manifest_path
    ]
    _write_csv(
        manifest_path,
        ["path", "sha256", "scope", "status"],
        [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
                "scope": "aggregate_safe_kdd267_receipt",
                "status": "pass",
            }
            for path in paths
        ],
    )
    print(
        {
            "privacy_files_scanned": privacy["files_scanned"],
            "privacy_findings": privacy["findings"],
            "release_assets": len(paths),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
