# Paper Artifact and Benchmark Surfaces

This repository is the executable artifact for EHRDyn. It contains the
credentialed constructor and component scorer, the five EHR-fitted simulator
workflow, 40 controlled environments, a shared 21-policy inventory, and six
OPE estimators.

The paper-facing public identity starts with:

```bash
ehrdyn-icu validate-paper-tasks
ehrdyn-icu method-inventory
```

The first command validates the five submitted EHR task contracts. The seven
files under `configs/tasks/` remain available only as a legacy compatibility
surface and are not the default reader taxonomy.

An authorized MIMIC-IV v3.1 user can construct the five task interfaces,
obtain disjoint `source_model_train` and `source_model_calibration` roles,
train and calibrate the Gaussian recurrent source simulator, freeze the
selected checkpoint, and run the shared policy and OPE workflows. All
MIMIC-derived outputs remain in caller-owned restricted paths.

Public synthetic smoke test:

```bash
python -m pip install '.[fitted]'
ehrdyn-icu method-inventory
ehrdyn-icu fitted-synthetic-smoke --output /tmp/ehrdyn-fitted-smoke.json
ehrdyn-icu controlled-21-method-smoke \
  --profile aki \
  --environment-seed 171901 \
  --output /tmp/ehrdyn-controlled-smoke.json
```

Credentialed route (all outputs are caller-owned and must remain outside Git):

```bash
ehrdyn-icu construct-ehr --output /private/ehrdyn
ehrdyn-icu fitted-simulator --mode dry-run --constructor-root /private/ehrdyn --output /tmp/ehrdyn-dry-run.json
ehrdyn-icu fitted-simulator --mode train --constructor-root /private/ehrdyn --restricted-root /private/ehrdyn-fit --output /tmp/ehrdyn-training.json --device cpu
ehrdyn-icu fitted-simulator --mode evaluate --constructor-root /private/ehrdyn --restricted-root /private/ehrdyn-fit --output /private/ehrdyn-results --device cpu
```

The fitted route never publishes MIMIC rows, identifiers, timestamps, arrays,
checkpoints, trajectories, or result tables. The synthetic smoke path is the
public end-to-end proof of the same source-fit, calibrated rollout, matched
method, and six-estimator workflow.

The fitted, controlled, and controlled-OPE inventories contain exactly 21
methods in one order. Discrete XQL and compact Dreamer V1--V3 are source-backed
benchmark adaptations. Dreamer policies use sampled actor probabilities as the
primary target policy; Dreamer V2 actor mode is a development sensitivity
rather than an additional method. Severity is absent from every inventory.

## What the public smoke establishes

The synthetic smoke path checks executable source fitting, calibrated rollout,
method ordering, policy-probability validity, and availability of all six OPE
estimators. It does not reproduce the MIMIC-derived results, establish
algorithm-family superiority, or validate a clinical policy.

The public 21-method smoke is capability reproducibility, not full numerical
result reproducibility. It uses bounded smoke budgets and writes conformance
receipts rather than the paper's full 21-method by 40-environment return and OPE
tables. `generate-full-suite` validates and enumerates all 40 controlled
contracts, and `evaluate-world-model-full` evaluates one external recursive
entrant. The release does not claim that either command regenerates the paper
figures or scientific aggregate tables.

Transition, policy, and world-model entrants have isolated interfaces. The six
OPE estimators are built-in implementations and are not an isolated
estimator-submission track.

Versioned provenance and audit receipts remain under `release/`. They are
retained for artifact verification but are not part of the reader-facing
benchmark taxonomy.
