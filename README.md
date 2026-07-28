# EHRDyn-ICU: multi-cohort decision evaluation for EHR model components and offline RL

The additive KDD125 paper-facing release candidate is in
[`decision/kdd125/`](decision/kdd125/README.md). It replaces small-lineage
sepsis and heart-failure evidence with their large-lineage results, records
AF/flutter as a scale-gate exclusion, and reports reliability-qualified
retrospective OPE without a real-EHR policy winner. The earlier `decision/`
artifacts remain preserved as historical release-engineering evidence.

```bash
python -m pip install -e .
ehrdyn-icu decision-validate --root .
ehrdyn-icu decision-smoke --output /tmp/ehrdyn-decision-smoke.json
python -m unittest tests.test_decision
```

Anonymous artifact: <https://anonymous.4open.science/r/ehrdyn-icu-65FB>

## Preserved earlier lineages

EHRDyn-ICU contains the historical KDD089 benchmark and an isolated local
release candidate for the five-task KDD-RV successor. The historical contract
identifier remains `KDD2027-E060-4H-v1.0.0+KDD089`; none of its configs,
metrics, or compatibility commands is relabeled. The successor identifier is
`KDD-RV-SUCCESSOR-RC1` and is not yet a public release.

This repository contains software, task definitions, synthetic fixtures, and
aggregate evidence. It does **not** contain MIMIC-IV rows, patient identifiers,
split membership, timestamps, trajectories, model checkpoints, row-level
predictions, or credentials.

## Benchmark Surfaces

The paper uses several intentionally separate evidence surfaces. Their exact
roles and non-pooling rules are defined in [CANONICAL_SURFACES.md](CANONICAL_SURFACES.md).
The offline-policy estimand and estimator conventions are defined in
[OPE_CONTRACT.md](OPE_CONTRACT.md). Clinical task definitions are versioned in
`configs/tasks/` and `task_cards/`; independent clinician review is tracked in
[CLINICAL_REVIEW.md](CLINICAL_REVIEW.md).

`contracts/paper_task_manifest.csv` maps every paper task to one canonical
config and clinical packet. `contracts/paper_contract_manifest.csv` maps all 41
headline contracts to the 533 public leaderboard rows. Rich K25 audits are
separate under `configs/rich_action/`.

Post-freeze audit evidence is in `evidence/audits/`. KDD094 identifies limits in
target-policy provenance and OPE documentation; those limits take precedence
over earlier aggregate summaries.

## Quick Start

Python 3.11 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

ehrdyn-icu validate-config --config-dir configs/tasks
ehrdyn-icu generate-fixture --output /tmp/ehrdyn_fixture.csv
ehrdyn-icu evaluate \
  --fixture /tmp/ehrdyn_fixture.csv \
  --task-config configs/tasks/kdd2027_sepsis_vasopressor_3bin.json \
  --output /tmp/ehrdyn_metrics.json
ehrdyn-icu validate-manifest \
  --task-manifest contracts/paper_task_manifest.csv \
  --contract-manifest contracts/paper_contract_manifest.csv \
  --evidence evidence/core/contract_transition_leaderboard.csv
ehrdyn-icu validate-submission \
  --submission submission/leaderboard_submission_template.json \
  --config-dir configs/tasks
python -m unittest discover -s tests
ehrdyn-icu scan-release --root .
ehrdyn-icu verify-checksums --root .
```

The historical `kdd2027` command remains available as a compatibility alias.

## Successor Release-Candidate Quick Start

The successor commands require NumPy and use only synthetic inputs unless an
authorized user explicitly supplies restricted local files.

```bash
ehrdyn-icu rv-validate-config \
  --config-dir successor/configs/tasks \
  --contract-manifest successor/contracts/contract_manifest.json
ehrdyn-icu rv-generate-fixture \
  --output /tmp/rv_predictions.csv \
  --normalization-output /tmp/rv_normalization.json \
  --contract-output /tmp/rv_evaluation_contract.json
ehrdyn-icu rv-evaluate-fixture \
  --fixture /tmp/rv_predictions.csv \
  --normalization /tmp/rv_normalization.json \
  --evaluation-contract /tmp/rv_evaluation_contract.json \
  --output /tmp/rv_metrics.json
ehrdyn-icu rv-validate-submission \
  --submission /tmp/rv_metrics.json \
  --config-dir successor/configs/tasks
ehrdyn-icu rv-verify-evidence \
  --root . \
  --manifest successor/evidence/evidence_manifest.csv
```

Restricted local evaluation accepts only `sealed_test` rows, uses the exact
train-only normalization receipt, and aggregates paired subject-cluster
inference without writing local keys. See
[`successor/README.md`](successor/README.md) for the row and recursion contract.
It also requires a benchmark-operator evaluation contract that binds the full
target-cell set before prediction scoring; an identically truncated method
subset is rejected.

Source-tree and sdist installations include successor configs, schemas,
aggregate evidence, and credentialed adapter documentation. The wheel is a
code-only evaluator distribution plus its pinned source manifest; it does not
carry the repository-relative evidence bundle. Release archives must be built,
hashed, and scanned separately with `rv-audit-distributions`.

```bash
python -m build
ehrdyn-icu rv-audit-distributions --dist-dir dist
```

## Repository Contents

- `configs/tasks/`: five primary and two extended compact paper contracts.
- `configs/rich_action/`: separate K25/reference/exclusion audit contracts.
- `contracts/`: preprocessing and layered gate definitions.
- `credentialed/`: public SQL, preprocessing, action encoding, and aggregate parity targets for authorized local MIMIC-IV execution.
- `dictionaries/`: feature, action, reward, metric, and baseline provenance.
- `src/`: aggregate evaluator, fixture, split, gate, privacy, and submission code.
- `fixtures/`: schema-compatible synthetic data only.
- `evidence/`: publication-safe aggregate dynamics, temporal, and policy diagnostics.
- `clinical_review/`: seven aggregate-safe external-review packets with blank responses.
- `task_cards/`: human-readable task summaries.
- `schemas/` and `submission/`: leaderboard interfaces.
- `tests/`: clean-room, privacy, checksum, and contract tests.
- `successor/`: separate five-task RV configs, schemas, task cards, source
  provenance, and aggregate-submission interface.
- `decision/`: paper-facing six-task world-model and four-task known-value
  decision-evaluation release, including complete aggregate evidence.
- `src/kdd2027_benchmark/rv/`: installable successor split, recursive rollout,
  evaluator, source verification, fixture, and submission code.
- `src/kdd2027_benchmark/decision/`: portable decision-release validator,
  known-value planner smoke test, and OPE formula checks.

## Credentialed Benchmark Execution

MIMIC-IV remains governed by PhysioNet credentialing and is not redistributed.
See [MIMIC_ACCESS.md](MIMIC_ACCESS.md). Extraction and preprocessing must run in
an authorized environment. `ehrdyn-icu evaluate-local` converts a local
cell-level prediction file to aggregate metrics without exporting row keys.
Only separately privacy-reviewed aggregate outputs may be contributed.

## Evidence Boundary

The dynamics track evaluates one-step and conditional recursive forecasts under
logged future actions. It does not simulate outcomes under a new treatment
policy. Existing policy evidence is quarantined as a frozen aggregate diagnostic
and is not accepted by the public leaderboard validator. See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)
and [ETHICS_AND_MISUSE.md](ETHICS_AND_MISUSE.md).

Unsupported uses include treatment recommendation, causal effect estimation,
counterfactual benefit claims, clinical deployment, and autonomous decisions.

The successor RC does not close independent credentialed reproduction,
clinical adjudication, institutional review, subgroup/fairness audit,
final-manuscript parity, public tagging, release, or DOI gates.

## Citation

Citation metadata are in [CITATION.cff](CITATION.cff). Anonymous repository:
<https://anonymous.4open.science/r/ehrdyn-icu-65FB>.
