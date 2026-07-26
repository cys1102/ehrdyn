# EHRDyn-ICU

EHRDyn-ICU is a frozen benchmark contract for recorded ICU trajectory
forecasting and offline-RL readiness diagnostics.

The KDD267 source implements the version 2.1.0 successor to immutable v2.0.4.
It preserves the earlier paper-bound benchmark and adds one exact shared
21-method inventory for the fitted, controlled, and controlled-OPE surfaces.
The four additions are the discrete XQL and compact Dreamer V1--V3 benchmark
adaptations used by KDD264. They are not official reproductions. Severity
rules are excluded from every KDD267 public method inventory.

The paper-bound artifact includes a
documented local constructor for authorized MIMIC-IV v3.1
users. It contains runtime code, task and configuration contracts, schemas,
tiny synthetic fixtures, public tests, and the constructed-environment entrant
workflow, plus the source-model train/calibration split and paper-fitted
simulator/matched-method route. It does not redistribute MIMIC-IV data, split
membership, model checkpoints, or MIMIC-derived result tables.

The canonical-v2 scientific scorer contract remains version 2.0.0. No cohort,
task, schema, metric, tolerance, expected synthetic output, or API changed in
this successor.

## Installation

Python 3.11, 3.12, or 3.13 is required. Dependencies are frozen in `uv.lock`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .

ehrdyn-icu --help
ehrdyn-icu --version
ehrdyn-icu validate-config --config-dir configs/tasks
ehrdyn-icu validate-schemas --schema-dir schemas
ehrdyn-icu method-inventory
python -m unittest discover -s tests
ehrdyn-icu scan-release --root .
ehrdyn-icu verify-checksums --root .
```

`ehrdyn-icu --version` reports the frozen benchmark contract identifier.
Package metadata reports `2.1.0`.

The paper-bound route and its publication evidence are documented in
[PAPER_ARTIFACT.md](PAPER_ARTIFACT.md). The previous KDD263 release remains
`complete_paper_bound_public_benchmark_artifact`. KDD267 is bound to the
remotely frozen no-severity paper and 21-policy controlled-OPE aggregate in
[`release/kdd267/paper_snapshot.json`](release/kdd267/paper_snapshot.json).

## Credentialed MIMIC-IV construction

Authorized MIMIC-IV v3.1 users can reconstruct the five EHR task interfaces
locally from the official flat files. Keep the data path out of shell history
by setting it through the required environment variable.

```bash
python -m pip install '.[credentialed]'
export AUTHORIZED_MIMICIV_3_1_ROOT='<authorized MIMIC-IV v3.1 directory>'
ehrdyn-icu construct-ehr --output '<new private output directory>'
```

The command validates all 14 required input tables, reconstructs the five
cohorts and deterministic subject roles, applies train-only preprocessing, and
writes role-specific modeling arrays to a restricted local directory. Use
`--aggregate-only` to validate construction without writing the modeling
arrays. See [CREDENTIALED_CONSTRUCTOR.md](CREDENTIALED_CONSTRUCTOR.md) and
[MIMIC_ACCESS.md](MIMIC_ACCESS.md). The author-side reference run took about
2 hours 8 minutes, peaked at 81.3 GiB resident memory, and used 32.3 GiB of
temporary disk.

## Synthetic canonical-v2 scorer

The scorer accepts point, independent-Gaussian, and Gaussian-ensemble
submissions. It reports aggregate forecasting, calibration, interval,
termination, support, ESS, and evaluability diagnostics where defined.

```bash
ehrdyn-icu score-ehr-components \
  --submission fixtures/kdd245v2r/gaussian.json \
  --output build/ehr-component-score.json
```

See [EHR_COMPONENT_SCORER.md](EHR_COMPONENT_SCORER.md),
[SCHEMA_VALIDATION.md](SCHEMA_VALIDATION.md), and
[CANONICAL_SERIALIZATION.md](CANONICAL_SERIALIZATION.md).

## Constructed-environment entrant

The public constructed workflow uses only released synthetic mechanisms and
fixtures. A bounded smoke is:

```bash
ehrdyn-icu evaluate-world-model-smoke \
  --manifest configs/full_benchmark/kdd198_v2_generator_contract.json \
  --entrant world_model_entrant_example/point.json \
  --entrant world_model_entrant_example/gaussian.json \
  --entrant world_model_entrant_example/ensemble.json \
  --output build/world-model-smoke \
  --episodes 8
```

The full 40-environment workflow is documented in
[RECURSIVE_WORLD_MODEL_ENTRANT.md](RECURSIVE_WORLD_MODEL_ENTRANT.md). The
demonstration entrant is an interface example and is not part of a scientific
leaderboard.

## Exact 21-method successor

The public inventory receipt is identical for fitted and controlled workflows:

```bash
ehrdyn-icu method-inventory
ehrdyn-icu fitted-synthetic-smoke \
  --output build/kdd267-fitted-smoke.json
ehrdyn-icu controlled-21-method-smoke \
  --config configs/full_benchmark/kdd198_v2_generator_contract.json \
  --profile aki \
  --environment-seed 171901 \
  --output build/kdd267-controlled-smoke.json
```

Both smokes execute all 21 named methods in the frozen order and all six OPE
estimators. The controlled OPE inventory also contains exactly those 21
methods; there is no severity-rule exception. These are nonclinical capability
checks and do not provide return, rank, or algorithm-family superiority
evidence. See
[the KDD267 quickstart](docs/kdd267/21_method_quickstart.md).

## Included interfaces

- `src/`: benchmark, constructor, scorer, schema, entrant, planner, and evaluator code.
- `configs/`: frozen task and constructed-environment contracts.
- `schemas/`: Draft 2020-12 input/output schemas.
- `fixtures/` and `invalid_entrant_fixtures/`: synthetic positive and negative
  fixtures.
- `dictionaries/`, `task_cards/`, and `submission/`: public contract metadata.
- `tests/`: synthetic-only schema, scorer, entrant, privacy, and replay tests.

## Data and claim boundary

MIMIC-IV remains governed by PhysioNet credentialing and is not redistributed.
The constructor runs only inside an authorized local environment and publishes
no MIMIC-derived scientific results. See [MIMIC_ACCESS.md](MIMIC_ACCESS.md),
[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md), and
[ETHICS_AND_MISUSE.md](ETHICS_AND_MISUSE.md).

Unsupported uses include treatment recommendation, causal effect estimation,
counterfactual benefit claims, clinical deployment, and autonomous decisions.

Citation metadata are in [CITATION.cff](CITATION.cff). The repository is
<https://github.com/cys1102/ehrdyn-icu>.
