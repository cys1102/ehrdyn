# EHRDyn

EHRDyn is a benchmark for evaluating offline reinforcement learning from
electronic health records (EHRs). It aligns five MIMIC-IV task contracts across
credentialed EHR dynamics and support diagnostics, frozen EHR-fitted
observational simulators, and 40 public controlled environments. These
complementary surfaces test transition models, policies, world models, and
off-policy evaluation (OPE) estimators without treating simulator returns as
clinical counterfactual truth.

The shared simulator suite contains 21 policy specifications and six OPE
estimators. The fitted and controlled surfaces use the same method order;
severity rules are not part of either inventory. Discrete XQL and the compact
Dreamer V1--V3 implementations are benchmark adaptations, not official
reproductions of the original systems.

The repository and benchmark display name is EHRDyn. The Python package and
command-line interface retain the compatibility name `ehrdyn-icu`.

## Benchmark surfaces

| Surface | Access | Evaluated object or submission | Evaluation reference | Start here |
| --- | --- | --- | --- | --- |
| EHR dynamics and support | Credentialed MIMIC-IV v3.1 | Transition-prediction submission | Recorded next observations and logged trajectories | [MIMIC access](MIMIC_ACCESS.md), [component scorer](EHR_COMPONENT_SCORER.md) |
| EHR-fitted simulator | Credentialed full workflow; public synthetic smoke | Named policy or candidate transition implementation | Direct return in a frozen fitted observational simulator | [Paper artifact](PAPER_ARTIFACT.md) |
| Controlled policy evaluation | Public | Policy or world-model entrant | Direct return in a frozen synthetic mechanism | [World-model entrant](RECURSIVE_WORLD_MODEL_ENTRANT.md) |
| Simulator OPE | Public controlled workflow; credentialed fitted workflow; public smoke | Six included OPE implementations | Direct simulator return for each fixed policy | [OPE contract](OPE_CONTRACT.md) |

EHRDyn reports a multi-metric profile rather than one composite score. The
current controlled environments are public development assets, not a hidden
test service.

The public controlled workflows and synthetic smokes require no MIMIC access.
Credentialed rows run only in an authorized, caller-owned MIMIC-IV environment
and keep all restricted inputs and outputs local.

The controlled policy and world-model surfaces expose isolated entrant
interfaces. The current OPE surface provides six versioned Python estimator
implementations; it does not yet define an isolated estimator-subprocess
schema.

## Installation

Python 3.11, 3.12, or 3.13 is required. Dependencies are frozen in `uv.lock`.
The base install supports the public CLI, component scorer, controlled
environments, and entrant smokes. Installed defaults are package resources, so
these commands do not depend on the repository being the working directory.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .

ehrdyn-icu --help
ehrdyn-icu --version
ehrdyn-icu validate-paper-tasks
ehrdyn-icu validate-schemas
ehrdyn-icu method-inventory
ehrdyn-icu score-ehr-components \
  --output build/ehr-component-score.json
ehrdyn-icu generate-full-suite \
  --output build/controlled-suite.csv
ehrdyn-icu evaluate-world-model-smoke \
  --output build/world-model-smoke \
  --episodes 8
```

The five-task manifest used by `validate-paper-tasks` is the current
paper-facing taxonomy. The seven JSON files in `configs/tasks/` retain an older
action-abstraction configuration surface for compatibility; they are not the
default paper identity. They can still be checked explicitly:

```bash
ehrdyn-icu validate-config --config-dir configs/tasks
```

Install the test extra before running the complete source-checkout suite. It
includes `pandas` for the synthetic five-task constructor tests and the fitted
stack required by the 21-method regression tests.

```bash
python -m pip install '.[test]'
python -m unittest discover -s tests
ehrdyn-icu scan-release --root .
ehrdyn-icu verify-checksums --root .
git diff --check
```

`ehrdyn-icu --version` and package metadata report `2.1.1`. The
manuscript-facing workflow is documented
in [PAPER_ARTIFACT.md](PAPER_ARTIFACT.md).

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

## EHR-format component scorer

The scorer accepts point, independent-Gaussian, and Gaussian-ensemble
submissions. It reports aggregate forecasting, calibration, interval,
termination, support, ESS, and evaluability diagnostics where defined.

```bash
ehrdyn-icu score-ehr-components \
  --output build/ehr-component-score.json
```

The default submission is an installed synthetic example. An explicit
`--submission` path always overrides it.

See [EHR_COMPONENT_SCORER.md](EHR_COMPONENT_SCORER.md),
[SCHEMA_VALIDATION.md](SCHEMA_VALIDATION.md), and
[CANONICAL_SERIALIZATION.md](CANONICAL_SERIALIZATION.md).

## Controlled-environment entrant

The public constructed workflow uses only released synthetic mechanisms and
fixtures. A bounded smoke is:

```bash
ehrdyn-icu evaluate-world-model-smoke \
  --output build/world-model-smoke \
  --episodes 8
```

Without `--entrant`, this command uses the three installed point, Gaussian, and
ensemble demonstration entrants. One or more explicit `--entrant` arguments
always override those examples.

The full 40-environment workflow is documented in
[RECURSIVE_WORLD_MODEL_ENTRANT.md](RECURSIVE_WORLD_MODEL_ENTRANT.md). The
demonstration entrant is an interface example and is not part of a scientific
leaderboard.

## Shared 21-policy and six-estimator workflow

Install the fitted dependency group, then run the public bounded workflow:

```bash
python -m pip install '.[fitted]'
ehrdyn-icu method-inventory
ehrdyn-icu fitted-synthetic-smoke \
  --output build/ehrdyn-fitted-smoke.json
ehrdyn-icu controlled-21-method-smoke \
  --profile aki \
  --environment-seed 171901 \
  --output build/ehrdyn-controlled-smoke.json
```

Both smokes execute all 21 named methods in the frozen order and all six OPE
estimators. The controlled OPE inventory also contains exactly those 21
methods; there is no severity-rule exception. These are nonclinical capability
checks. They do not regenerate the paper's full numerical 21-method by
40-environment return and OPE aggregates and do not provide return, rank, or
algorithm-family superiority evidence. `generate-full-suite` enumerates the
controlled contracts, while `evaluate-world-model-full` evaluates one external
entrant; neither command reconstructs the bundled paper result tables. See
[the 21-method quickstart](docs/21_method_quickstart.md).

## Included interfaces

- `src/`: benchmark, constructor, scorer, schema, entrant, planner, and evaluator code.
- `configs/`: frozen task and constructed-environment contracts.
- `schemas/`: Draft 2020-12 input/output schemas.
- `fixtures/` and `invalid_entrant_fixtures/`: synthetic positive and negative
  fixtures.
- `dictionaries/`, `task_cards/`, and `submission/`: public contract metadata.
- `tests/`: synthetic-only schema, scorer, entrant, privacy, and replay tests.

Transition, policy, and world-model entrants use isolated subprocess
interfaces. The six OPE estimators are built-in benchmark implementations, not
an open estimator-submission API.

## Data and claim boundary

MIMIC-IV remains governed by PhysioNet credentialing and is not redistributed.
The constructor runs only inside an authorized local environment and publishes
no MIMIC-derived scientific results. See [MIMIC_ACCESS.md](MIMIC_ACCESS.md),
[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md), and
[ETHICS_AND_MISUSE.md](ETHICS_AND_MISUSE.md).

Intended use includes nonclinical evaluation of forecasting, simulator-based
policy construction, direct simulator returns, and OPE recovery. Unsupported
uses include treatment recommendation, causal effect estimation,
counterfactual benefit claims, clinical deployment, and autonomous decisions.

Citation metadata are in [CITATION.cff](CITATION.cff). The repository is
<https://github.com/cys1102/ehrdyn>.
