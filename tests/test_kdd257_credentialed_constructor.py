from __future__ import annotations

import json
import os
import gzip
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from kdd2027_benchmark.credentialed_constructor import (
    MIMIC_ROOT_ENV,
    construct_from_authorized_mimic,
)
from kdd2027_benchmark.current_five_task.reconstruct import _array_digest
from kdd2027_benchmark.errors import ReleaseContractError
from tests.test_kdd217ar3a import make_fixture


TASKS = ("sepsis", "respiratory_support", "shock", "aki", "heart_failure")
ROLES = ("train", "validation", "historical_other")
RESTRICTED_KEYS = {
    "state_values",
    "state_masks",
    "log_recency",
    "raw_imputed_history",
    "imputed_history",
    "action_index",
    "reward",
    "reward_mask",
    "termination",
    "continuation",
    "valid_step",
    "target_values",
    "observed_target_mask",
    "episode_order",
    "transition_order",
    "preprocessing_mean",
    "preprocessing_scale",
    "feature_index",
}


class KDD257CredentialedConstructorTests(unittest.TestCase):
    def test_missing_authorized_root_fails_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "private"
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ReleaseContractError):
                    construct_from_authorized_mimic(output)
            self.assertFalse(output.exists())

    def test_private_arrays_match_frozen_role_digests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_fixture(Path(directory))
            output = Path(directory) / "private"
            with patch.dict(os.environ, {MIMIC_ROOT_ENV: str(root)}, clear=True):
                receipt = construct_from_authorized_mimic(output)

            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            private = output / "private_arrays"
            files = sorted(private.glob("*.restricted.npz"))
            self.assertEqual(len(files), len(TASKS) * len(ROLES))
            by_task = {row["task_id"]: row for row in receipt["tasks"]}
            for task in TASKS:
                role_rows = {
                    row["role"]: row for row in by_task[task]["role_summaries"]
                }
                for role in ROLES:
                    path = private / f"{task}.{role}.restricted.npz"
                    self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                    with np.load(path, allow_pickle=False) as arrays:
                        self.assertEqual(set(arrays.files), RESTRICTED_KEYS)
                        self.assertNotIn("subject_id", arrays.files)
                        self.assertNotIn("stay_id", arrays.files)
                        self.assertEqual(
                            _array_digest(arrays["state_values"]),
                            role_rows[role]["digests"]["feature_digest"],
                        )
                        self.assertEqual(
                            _array_digest(arrays["action_index"]),
                            role_rows[role]["digests"]["action_digest"],
                        )

    def test_aggregate_only_preserves_scientific_surface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_fixture(Path(directory))
            private = Path(directory) / "private"
            aggregate = Path(directory) / "aggregate"
            with patch.dict(os.environ, {MIMIC_ROOT_ENV: str(root)}, clear=True):
                private_receipt = construct_from_authorized_mimic(private)
                aggregate_receipt = construct_from_authorized_mimic(
                    aggregate,
                    write_private_arrays=False,
                )
            self.assertFalse((aggregate / "private_arrays").exists())
            self.assertEqual(
                private_receipt["contracts"]["scientific_surface_sha256"],
                aggregate_receipt["contracts"]["scientific_surface_sha256"],
            )
            self.assertEqual(
                json.loads((private / "aggregate_receipt.json").read_text())["tasks"],
                json.loads((aggregate / "aggregate_receipt.json").read_text())["tasks"],
            )

    def test_duplicate_source_encodings_use_one_temporary_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_fixture(Path(directory))
            compressed = root / "hosp" / "patients.csv.gz"
            plain = root / "hosp" / "patients.csv"
            with gzip.open(compressed, "rb") as source, plain.open("wb") as target:
                shutil.copyfileobj(source, target)
            output = Path(directory) / "private"
            with patch.dict(os.environ, {MIMIC_ROOT_ENV: str(root)}, clear=True):
                construct_from_authorized_mimic(
                    output,
                    write_private_arrays=False,
                )
            encoding = json.loads(
                (output / "source_encoding_view.json").read_text(encoding="utf-8")
            )
            self.assertEqual(encoding["duplicate_encodings_observed"], 1)
            self.assertEqual(encoding["compressed_sources_selected"], 14)


if __name__ == "__main__":
    unittest.main()
