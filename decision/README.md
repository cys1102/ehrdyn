# EHRDyn-ICU multi-cohort decision-evaluation benchmark

> KDD125 successor: the additive lineage-consistent release candidate is in
> [`kdd125/`](kdd125/README.md). This document is preserved as the pre-KDD125
> release receipt and is not relabeled.

This directory is the anonymous release surface for the paper
*A Multi-Cohort Decision-Evaluation Benchmark for EHR World-Model
Components and Offline Reinforcement Learning*.

It is intentionally separate from the historical forecasting benchmark and
the five-task RV release candidate at the repository root. Nothing in those
older lineages is relabeled as a decision-benchmark result.

## What is released

- six final cohort targets that each exceed 10,000 subjects and 10,000
  episodes, with subjects and episodes recorded in
  `evidence/primary_cohort_scale_eligibility.csv`;
- an 18-row current transition matrix for respiratory, shock, and AKI across
  all six transition methods; the 18 sepsis, AF/flutter, and heart-failure rows
  from smaller constructions remain available but are labeled superseded and
  excluded from current synthesis;
- an AI-Clinician-aligned K25 sepsis scaffold, labeled as an alignment
  surface rather than an exact reproduction;
- 18 current heterogeneous exact-finite task--mechanism environments over
  respiratory, shock, and AKI--RRT, each with dynamic-programming truth;
- the complete layer-specific inventory of six transition methods, five
  uncertainty-capable methods, 34 policy/planner labels, and nine OPE
  estimators;
- a cross-layer baseline atlas and cohort-specific uncertainty leader table;
- 3,060 current heterogeneous policy--seed rows, 2,160 current
  world-model--planner rows, all corresponding exploitation-gap rows, and the
  earlier adaptive sensitivity; the larger historical ledgers remain packaged
  but heart-failure rows are not current evidence;
- all 2,448 historical monotone policy rows, retained as evaluator
  smoke/sensitivity evidence rather than the primary policy benchmark;
- 15,552 current heterogeneous repeated-dataset coverage rows and 2,592 current
  OPE dispositions over four mechanisms and six fixed target policies; the
  7,776-row null/composite-adaptive grid and prior fixed-dataset diagnostics
  remain separately labeled sensitivities;
- the 16-row EHR-to-known-value model-family bridge and all contract mappings;
- a source-verified representative literature landscape;
- the anonymous manuscript source and compiled PDF;
- a portable contract/evidence validator; and
- a synthetic known-value smoke test for exact planning, common random
  numbers, support masking, null-response invariance, and basic OPE formulas.

The paper-to-code mapping is in [`CODE_MAP.md`](CODE_MAP.md), and the exact
unrestricted versus credentialed boundary is in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Byte-identical
world-model architecture and exact-evaluator snapshots are under
`reference_code/`, with original-source and packaged hashes in
`reference_code/source_manifest.csv`. Three credentialed runners replace only
machine-local default paths with explicit portable placeholders.

No MIMIC-IV rows, identifiers, split membership, exact timestamps,
trajectories, row-level predictions, or checkpoints are included.

The frozen final targets are sepsis 22,437 subjects/27,236 episodes,
respiratory 18,709/21,679, shock 37,015/49,495, AF/flutter 11,820/14,580,
AKI 16,380/21,083, and heart failure 27,611/32,552. Sepsis, AF/flutter, and
heart failure require complete large-lineage reevaluation before their numeric
results can re-enter the primary manuscript; metrics from the smaller
constructions are not transferred.

## Clean-room quick start

Python 3.11 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

ehrdyn-icu decision-validate --root .
ehrdyn-icu decision-smoke --output /tmp/ehrdyn-decision-smoke.json --seed 3408
python -m unittest tests.test_decision
ehrdyn-icu scan-release --root .
ehrdyn-icu verify-checksums --root .
```

The smoke test checks executable invariants only; it does not reproduce paper
numbers or use EHR data. Numerical paper evidence, including every complete
baseline row, is checksum- and row-count-validated against
`decision/evidence/manifest.csv`.

## Evidence layers

The release keeps four layers distinct:

1. EHR P/R/T-component prediction and uncertainty on development roles;
2. retrospective action support, collapse, and observability diagnostics;
3. exact policy returns in heterogeneous constructed known-value environments;
4. repeated-dataset OPE calibration and authorization within known-value
   environments.

The older adaptive-composite and monotone families are explicitly secondary
construction/evaluator sensitivities. They are not used to claim a universal
policy winner.

No task completes the trained-target-policy and post-training overlap gates
needed for retrospective EHR policy-value scoring. The release therefore does
not contain an EHR policy leaderboard.

## Credentialed reconstruction

MIMIC-IV reconstruction must occur in an authorized environment. Public code
and task contracts define the inputs and aggregate outputs, while restricted
arrays remain local. Independent credentialed reconstruction means that a
second MIMIC-credentialed user regenerates aggregate results from a clean
clone; it is a reproducibility check, not external-site, clinical, or
generalization validation. Synthetic clean-room success is not presented as
credentialed numerical parity.
