# Credentialed five-task constructor

This command reconstructs the five EHR benchmark interfaces inside an
authorized MIMIC-IV v3.1 environment.

```bash
export AUTHORIZED_MIMICIV_3_1_ROOT='<authorized MIMIC-IV v3.1 directory>'
ehrdyn-icu construct-ehr --output '<new private output directory>'
```

## Fixed contract

- Data version: MIMIC-IV v3.1.
- Time step: four hours.
- Episode window: 24 hours before and 48 hours after the task anchor.
- Roles: train, validation, and historical evaluation.
- Subject split: sepsis and heart failure use the first eight bytes of
  `SHA-256("KDD-RV-SUBJECT-ROLE-v1|" + subject_id)` modulo 10,000.
  Respiratory support, shock, and AKI retain their frozen compact-lineage
  rule, `(subject_id * 1103515245 + 12345) mod 100`. In both cases, the ranges
  assign 70% to train, 15% to validation, and 15% to historical evaluation.
- Preprocessing: past-only carry-forward initialized by train medians, followed
  by train-only mean and standard-deviation scaling.
- Tasks: sepsis, respiratory support, shock, AKI, and heart failure.
- Action counts: 25, 25, 25, 4, and 2, respectively.

The 25-action tasks cross two five-level axes. Sepsis and shock use fluid and
vasopressor exposure. Respiratory support uses directly observed PEEP and
FiO2. Each axis has a no-exposure level and four positive levels determined by
training-role quartiles. AKI crosses binary diuretic and renal-replacement
therapy indicators. Heart failure uses a binary diuretic indicator. Exact
item maps, exposure rules, and train-only cut-point code are in
`src/kdd2027_benchmark/current_five_task/`.

The integer encoding is deterministic. For each 25-action task,
`action = 5 * first_axis_level + second_axis_level`. For AKI,
`action = diuretic + 2 * renal_replacement_therapy`. For heart failure,
`action = diuretic`. The generated aggregate receipt records each fitted
train-only edge and the resulting action histogram.

The 33 state features are ordered as follows:

```text
heart_rate, sbp, mbp, dbp, respiratory_rate, temperature_c, spo2,
shock_index, gcs_proxy, fio2, sirs_proxy, lactate, pao2, paco2, ph,
base_excess, co2_bicarbonate, pao2_fio2, wbc, platelet, bun, creatinine,
ptt, pt, inr, ast, alt, total_bilirubin, magnesium, ionized_calcium,
calcium, urine_output, mechanical_ventilation
```

The exact cohort anchors, item maps, exclusion order, action cut points, reward
definitions, and termination rules are encoded in
`src/kdd2027_benchmark/current_five_task/`.

## Generated files

The new output directory is created with mode `0700`. It contains:

- `aggregate_receipt.json`
- aggregate streaming, resource, exclusion, and respiratory-filter receipts
- `source_encoding_view.json`
- `private_arrays/<task>.<role>.restricted.npz`

Each private array archive contains state values, observation masks, recency,
past-only imputed histories, normalized histories, actions, rewards, reward
masks, termination and continuation flags, valid-step masks, next-state
targets, target-observation masks, episode order, transition order, train-only
preprocessing statistics, and the feature index. The archives do not contain
subject or stay identifiers, but their role membership and row-level
trajectories remain restricted.

`--aggregate-only` skips the private archives and writes only the aggregate
receipts. The command never compares against hidden paper counts during
construction.

## Resource use

The author-side MIMIC-IV v3.1 reconstruction took 2 hours 8 minutes, peaked at
81.3 GiB resident memory, and used 32.3 GiB of temporary disk on the tested
workstation. Runtime varies with storage and available memory. Allocate
additional headroom for the private output archives.

## Interpretation

Exact reconstruction verifies the task and data-processing contract. It does
not establish clinical validity, causal treatment effects, policy value, or
generalization to another hospital.
