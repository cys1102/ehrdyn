from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator, ValidationError

from kdd2027_benchmark.current_five_task.contracts import (
    FEATURE_INDEX,
    ContractError,
    aggregate_receipt,
    apply_kdd201_temporal_repair,
    corrected_chart_value,
    eligible_transition_indices,
    encode_five_levels,
    fit_train_positive_edges,
    joint_codes,
    reward_components,
    subject_role,
    validate_layout,
)
from kdd2027_benchmark.current_five_task.reconstruct import HIGH_VOLUME_TABLE_ORDER, reconstruct


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas" / "credentialed_aggregate_receipt.schema.json"


def _train_subjects(count: int) -> list[int]:
    output: list[int] = []
    subject = 1
    while len(output) < count:
        compact_bucket = (subject * 1103515245 + 12345) % 100
        if subject_role(subject) == "train" and compact_bucket < 70:
            output.append(subject)
        subject += 1
    return output


def _write(root: Path, relative: str, rows: list[dict], columns: list[str]) -> None:
    path = root / f"{relative}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, compression="gzip")


def make_fixture(parent: Path) -> Path:
    root = parent / "3.1"
    subjects = _train_subjects(6)
    task_subjects = dict(zip(("sepsis", "respiratory", "shock", "aki", "heart_failure"), subjects[:5]))
    patients = [{"subject_id": s, "gender": "M" if s % 2 else "F", "anchor_age": 60 + s, "dod": ""} for s in subjects]
    admissions = []
    stays = []
    diagnoses = []
    for index, (task, subject) in enumerate(task_subjects.items(), start=1):
        hadm, stay = 1000 + index, 2000 + index
        admissions.append({"subject_id": subject, "hadm_id": hadm, "admittime": "2019-12-31 00:00:00", "dischtime": "2020-01-05 00:00:00", "deathtime": "", "hospital_expire_flag": 0, "discharge_location": "HOME"})
        stays.append({"subject_id": subject, "hadm_id": hadm, "stay_id": stay, "first_careunit": "MICU", "last_careunit": "MICU", "intime": "2020-01-01 00:00:00", "outtime": "2020-01-04 00:00:00"})
        if task == "heart_failure":
            prior_hadm, prior_stay = 9000 + index, 8000 + index
            admissions.append({"subject_id": subject, "hadm_id": prior_hadm, "admittime": "2019-01-01 00:00:00", "dischtime": "2019-01-05 00:00:00", "deathtime": "", "hospital_expire_flag": 0, "discharge_location": "HOME"})
            stays.append({"subject_id": subject, "hadm_id": prior_hadm, "stay_id": prior_stay, "first_careunit": "MICU", "last_careunit": "MICU", "intime": "2019-01-01 00:00:00", "outtime": "2019-01-04 00:00:00"})
            diagnoses.append({"subject_id": subject, "hadm_id": prior_hadm, "icd_code": "I50", "icd_version": 10})
    _write(root, "hosp/patients", patients, ["subject_id", "gender", "anchor_age", "dod"])
    _write(root, "hosp/admissions", admissions, ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag", "discharge_location"])
    _write(root, "icu/icustays", stays, ["subject_id", "hadm_id", "stay_id", "first_careunit", "last_careunit", "intime", "outtime"])
    _write(root, "hosp/diagnoses_icd", diagnoses, ["subject_id", "hadm_id", "icd_code", "icd_version"])
    _write(root, "hosp/procedures_icd", [], ["subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version"])

    by_task = {task: (1000 + i, 2000 + i, subject) for i, (task, subject) in enumerate(task_subjects.items(), 1)}
    prescriptions = []
    sepsis_hadm, sepsis_stay, sepsis_subject = by_task["sepsis"]
    prescriptions.append({"subject_id": sepsis_subject, "hadm_id": sepsis_hadm, "starttime": "2020-01-02 00:00:00", "stoptime": "2020-01-02 04:00:00", "drug": "Vancomycin", "route": "IV"})
    hf_hadm, hf_stay, hf_subject = by_task["heart_failure"]
    prescriptions.append({"subject_id": hf_subject, "hadm_id": hf_hadm, "starttime": "2020-01-02 00:00:00", "stoptime": "2020-01-02 04:00:00", "drug": "Furosemide", "route": "IV"})
    _write(root, "hosp/prescriptions", prescriptions, ["subject_id", "hadm_id", "starttime", "stoptime", "drug", "route"])
    _write(root, "hosp/microbiologyevents", [{"microevent_id": 1, "subject_id": sepsis_subject, "hadm_id": sepsis_hadm, "micro_specimen_id": 10, "chartdate": "2020-01-01", "charttime": "2020-01-01 20:00:00", "spec_type_desc": "BLOOD CULTURE", "org_itemid": 1, "org_name": "synthetic organism"}], ["microevent_id", "subject_id", "hadm_id", "micro_specimen_id", "chartdate", "charttime", "spec_type_desc", "org_itemid", "org_name"])

    items = [
        {"itemid": 225792, "label": "Invasive Ventilation", "abbreviation": "vent", "linksto": "procedureevents", "category": "Resp", "unitname": ""},
        {"itemid": 220339, "label": "PEEP", "abbreviation": "PEEP", "linksto": "chartevents", "category": "Resp", "unitname": "cmH2O"},
        {"itemid": 225158, "label": "NaCl Fluid", "abbreviation": "fluid", "linksto": "inputevents", "category": "Fluids", "unitname": "mL"},
        {"itemid": 221906, "label": "Norepinephrine", "abbreviation": "norepi", "linksto": "inputevents", "category": "Meds", "unitname": "mcg/kg/min"},
        {"itemid": 999001, "label": "Furosemide", "abbreviation": "lasix", "linksto": "inputevents", "category": "Meds", "unitname": "mg"},
        {"itemid": 999002, "label": "CRRT dialysis", "abbreviation": "crrt", "linksto": "procedureevents", "category": "Renal", "unitname": ""},
    ]
    _write(root, "icu/d_items", items, ["itemid", "label", "abbreviation", "linksto", "category", "unitname"])
    _write(root, "hosp/d_labitems", [{"itemid": 50912, "label": "Creatinine", "fluid": "Blood", "category": "Chemistry"}], ["itemid", "label", "fluid", "category"])

    chart = []
    inputevents = []
    procedureevents = []
    outputevents = []
    counter = 0
    for task, (hadm, stay, subject) in by_task.items():
        for step in range(0, 18):
            time = pd.Timestamp("2020-01-01 00:00:00") + pd.Timedelta(hours=4 * step + 1)
            common = {"subject_id": subject, "hadm_id": hadm, "stay_id": stay, "charttime": str(time)}
            for itemid, value, unit in ((220045, 80 + step, "bpm"), (220050, 110, "mmHg"), (220052, 72 + step % 5, "mmHg"), (220051, 60, "mmHg"), (220277, 96, "%"), (226754, 25 + step % 4 * 10, "%"), (227013, 15, "score")):
                chart.append({**common, "itemid": itemid, "valuenum": value, "valueuom": unit})
            if task == "respiratory":
                chart.append({**common, "itemid": 220339, "valuenum": 2 + step % 4 * 2, "valueuom": "cmH2O"})
            magnitude = 1 + step % 4
            action_rows = [(225158, 100 * magnitude, 0)]
            if task != "sepsis" or step >= 6:
                action_rows.append((221906, magnitude, magnitude / 10))
            for itemid, amount, rate in action_rows:
                counter += 1
                inputevents.append({"subject_id": subject, "hadm_id": hadm, "stay_id": stay, "starttime": str(time), "endtime": str(time + pd.Timedelta(hours=1)), "itemid": itemid, "amount": amount, "amountuom": "mL" if itemid == 225158 else "mg", "rate": rate, "rateuom": "mcg/kg/min" if itemid == 221906 else "mL/hour", "ordercategoryname": "Fluid Bolus" if itemid == 225158 else "Continuous Med", "ordercategorydescription": "Crystalloid Fluid" if itemid == 225158 else "Drug"})
            if task in {"aki", "heart_failure"}:
                inputevents.append({"subject_id": subject, "hadm_id": hadm, "stay_id": stay, "starttime": str(time), "endtime": str(time + pd.Timedelta(hours=1)), "itemid": 999001, "amount": 1, "amountuom": "mg", "rate": 1, "rateuom": "mg/hour", "ordercategoryname": "Medication", "ordercategorydescription": "Drug"})
            outputevents.append({"subject_id": subject, "hadm_id": hadm, "stay_id": stay, "charttime": str(time), "itemid": 226566, "value": 100})
        if task == "respiratory":
            procedureevents.append({"subject_id": subject, "hadm_id": hadm, "stay_id": stay, "starttime": "2020-01-02 00:00:00", "endtime": "2020-01-02 08:00:00", "itemid": 225792})
        if task == "aki":
            procedureevents.append({"subject_id": subject, "hadm_id": hadm, "stay_id": stay, "starttime": "2020-01-02 04:00:00", "endtime": "2020-01-02 08:00:00", "itemid": 999002})
    # Two low measurements create the shock anchor independently of treatment exposure.
    shock_hadm, shock_stay, shock_subject = by_task["shock"]
    for minute in (0, 30):
        chart.append({"subject_id": shock_subject, "hadm_id": shock_hadm, "stay_id": shock_stay, "charttime": f"2020-01-02 00:{minute:02d}:00", "itemid": 220052, "valuenum": 60, "valueuom": "mmHg"})
    _write(root, "icu/chartevents", chart, ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum", "valueuom"])
    _write(root, "icu/inputevents", inputevents, ["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "amount", "amountuom", "rate", "rateuom", "ordercategoryname", "ordercategorydescription"])
    _write(root, "icu/procedureevents", procedureevents, ["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid"])
    _write(root, "icu/outputevents", outputevents, ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value"])
    aki_hadm, _, aki_subject = by_task["aki"]
    labs = [
        {"labevent_id": 1, "subject_id": aki_subject, "hadm_id": aki_hadm, "itemid": 50912, "charttime": "2020-01-01 08:00:00", "valuenum": 1.0, "valueuom": "mg/dL"},
        {"labevent_id": 2, "subject_id": aki_subject, "hadm_id": aki_hadm, "itemid": 50912, "charttime": "2020-01-02 00:00:00", "valuenum": 1.6, "valueuom": "mg/dL"},
    ]
    _write(root, "hosp/labevents", labs, ["labevent_id", "subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"])
    return root


class ContractTests(unittest.TestCase):
    def test_authoritative_low_level_goldens(self) -> None:
        roles = ["train"] * 4 + ["validation"]
        values = np.array([1, 2, 3, 4, 999], dtype=float)
        edges = fit_train_positive_edges(values, np.ones(5, bool), roles)
        self.assertEqual(edges, (1.75, 2.5, 3.25))
        labels = encode_five_levels(np.array([0, 1, 2, 3, 4, np.nan]), np.array([1, 1, 1, 1, 1, 0], bool), edges)
        np.testing.assert_array_equal(labels, np.array([0, 1, 2, 3, 4, -1], np.int16))
        np.testing.assert_array_equal(joint_codes(labels, labels, labels >= 0), np.array([0, 6, 12, 18, 24, -1], np.int16))
        eligible = eligible_transition_indices(pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-04"))
        self.assertEqual(eligible, tuple(range(0, 11)))
        np.testing.assert_array_equal(apply_kdd201_temporal_repair("vasopressor_support", eligible), np.array([False] + [True] * 10))
        np.testing.assert_array_equal(apply_kdd201_temporal_repair("sustained_hypotension", eligible), np.ones(11, bool))

    def test_dbp_and_reward_timing(self) -> None:
        self.assertEqual(corrected_chart_value(220051, 60, "mmHg"), 60)
        self.assertTrue(np.isnan(corrected_chart_value(220051, 250, "mmHg")))
        self.assertTrue(np.isnan(corrected_chart_value(220051, 60, "kPa")))
        shape = (2, 3, len(FEATURE_INDEX)); states = np.zeros(shape); targets = np.zeros(shape); masks = np.ones(shape, bool); valid = np.array([[1, 1, 0], [1, 1, 1]], bool)
        reward, reward_mask, _ = reward_components("sepsis", states, masks, targets, masks, valid, np.array([0, 1]))
        self.assertEqual(int(reward_mask.sum()), 2)
        self.assertEqual(reward[0, 1, 0], 1)
        self.assertEqual(reward[1, 2, 0], -1)
        targets[..., FEATURE_INDEX["mbp"]] = 75
        shock, shock_mask, _ = reward_components("shock", states, masks, targets, masks, valid, None)
        self.assertAlmostEqual(float(shock[0, 0, 0]), 0.4)
        self.assertFalse(shock_mask[0, 2, 0])

    def test_schema_rejects_privileged_and_unknown_fields(self) -> None:
        schema = json.loads(SCHEMA.read_text())
        Draft202012Validator.check_schema(schema)
        task_rows = {}
        digests = {name: "0" * 64 for name in (
            "feature_digest", "mask_digest", "delta_digest", "raw_imputed_history_digest",
            "imputed_history_digest", "action_digest", "reward_digest", "reward_mask_digest",
            "termination_digest", "continuation_digest", "valid_step_digest", "target_digest",
            "observed_target_mask_digest", "episode_order_digest", "preprocessing_mean_digest",
            "preprocessing_scale_digest", "transition_order_digest",
        )}
        for task, k in zip(("sepsis", "respiratory_support", "shock", "aki", "heart_failure"), (25, 25, 25, 4, 2)):
            stages = {"candidate_transitions": 1, "original_target_membership_transitions": 1, "action_observed_transitions": 1, "kdd152_joint_retained_transitions": 1, "missing_action_exclusions": 0, "original_target_exclusions": 0, "kdd201_removed_transitions": 0, "final_transitions": 1}
            stages["role_counts"] = [{"role": role, "candidate_transitions": 1, "original_target_membership_transitions": 1, "action_observed_transitions": 1, "kdd152_joint_retained_transitions": 1, "kdd201_removed_transitions": 0, "final_transitions": 1} for role in ("train", "validation", "historical_other", "all_roles")]
            task_rows[task] = {"subjects": 1, "episodes": 1, "decisions": 1, "action_counts": [1] + [0] * (k - 1), "minimum_horizon": 1, "maximum_horizon": 1, "action_count": k, "cutpoints": [], "cutpoint_hash": "0" * 64, "support_mask_digest": "0" * 64, "transition_stages": stages, "role_summaries": [{"role": role, "subjects": 1, "episodes": 1, "decisions": 1, "action_counts": [1] + [0] * (k - 1), "minimum_horizon": 1, "maximum_horizon": 1, "digests": digests} for role in ("train", "validation", "historical_other", "all_roles")], "reward_contract": "synthetic", "reward_observed_decisions": 1, "terminal_reward_count": 0}
        streaming = [
            {
                "table": table, "rows_read": 1, "rows_retained": 1,
                "chunks_processed": 1, "maximum_retained_rows_per_chunk": 1,
                "effective_chunk_size": 250000, "compression_encoding": "csv_gz",
                "scan_count": 1,
            }
            for table in HIGH_VOLUME_TABLE_ORDER
        ]
        receipt = aggregate_receipt(task_rows, {"source": "1" * 64}, streaming)
        Draft202012Validator(schema).validate(receipt)
        for field in ("subject_id", "stay_id", "timestamp", "trajectory", "split_membership", "free_text", "checkpoint", "raw_path"):
            invalid = copy.deepcopy(receipt); invalid[field] = "forbidden"
            with self.assertRaises(ValidationError): Draft202012Validator(schema).validate(invalid)

    def test_full_synthetic_flat_file_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_fixture(Path(directory))
            paths = validate_layout(root)
            self.assertEqual(len(paths), 14)
            output = Path(directory) / "receipt"
            receipt = reconstruct(root, output, SCHEMA, source_hashes={"fixture": "a" * 64})
            self.assertEqual([row["task_id"] for row in receipt["tasks"]], ["sepsis", "respiratory_support", "shock", "aki", "heart_failure"])
            self.assertEqual([row["action_count"] for row in receipt["tasks"]], [25, 25, 25, 4, 2])
            self.assertTrue(all(row["minimum_horizon"] < row["maximum_horizon"] or row["maximum_horizon"] > 0 for row in receipt["tasks"]))
            for row in receipt["tasks"]:
                if row["task_id"] in {"sepsis", "aki", "heart_failure"}:
                    self.assertEqual(row["terminal_reward_count"], row["episodes"])

    def test_fail_closed_layout_and_duplicate_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_fixture(Path(directory))
            patients = root / "hosp" / "patients.csv.gz"
            frame = pd.read_csv(patients); pd.concat([frame, frame.iloc[[0]]]).to_csv(patients, index=False, compression="gzip")
            with self.assertRaises(ContractError): reconstruct(root, Path(directory) / "out", SCHEMA)
            wrong = Path(directory) / "3.2"; root.rename(wrong)
            with self.assertRaises(ContractError): reconstruct(wrong, Path(directory) / "wrong-out", SCHEMA)


if __name__ == "__main__":
    unittest.main()
