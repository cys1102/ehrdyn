# Canonical-v2 credentialed EHR component scorer

The v2.0.0 scorer evaluates prediction components locally without transmitting
credentialed data. Its frozen identities are:

- benchmark: `ehrdyn-icu-canonical-v2.0.0`;
- evaluator: `ehr-component-scorer-v2.0.0`;
- feature interface: 33 ordered SAFE forecasting features;
- tasks: sepsis, respiratory support, shock, AKI, and heart failure;
- four-hour decisions with task-bounded recursive horizons of at most 10 or 11
  steps.

## Local command

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .

ehrdyn-icu score-ehr-components \
  --output /tmp/ehrdyn_ehr_component_score.json
```

Without `--submission`, the command uses the released synthetic example. Pass
`--submission <local-document>` to score a local document conforming to
`schemas/ehr_component_submission.schema.json`. The scorer validates the full
33-feature record, task action range, horizon, prediction form, termination
probability, behavior denominator, and target probability. It writes only the
aggregate document defined by `schemas/ehr_component_result.schema.json`.

Point submissions receive Structural N/A for CRPS, intervals, calibration, and
risk-coverage. Independent-Gaussian and Gaussian-ensemble submissions receive
Gaussian CRPS, empirical 50/80/90/95 percent coverage, interval width, MACE,
and risk-coverage area. Ensemble uncertainty combines mean within-member
variance and between-member mean variance.

Termination metrics use the frozen local termination labels and submitted
probabilities. Support, overlap, behavior-denominator NLL, importance-weight
ESS, and unsupported target mass are retrospective diagnostics only. Real-EHR
cells below 100 distinct subjects or 100 episodes suppress those diagnostics.

## Privacy boundary

Do not put identifiers, exact times, free text, split membership, trajectories,
raw rows, credentials, or local paths in a submission intended for sharing.
Run the scorer inside the authorized environment and share only the
schema-valid aggregate result after privacy review.

Planning, direct return, known policy value, causal effect, treatment benefit,
clinical utility, and deployment are not provided by this scorer.
