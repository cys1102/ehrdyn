# KDD267 21-method paper-bound successor

This successor preserves every immutable v2.0.4 capability: the constructor,
scorer, schemas, 40 controlled environments, and paper-bound fitted-simulator
route. An authorized MIMIC-IV v3.1 user constructs the five canonical task
interfaces, obtains disjoint `source_model_train` and
`source_model_calibration` roles, trains and calibrates the Gaussian recurrent
source simulator, freezes the selected checkpoint, and runs the exact shared
21 methods with the six OPE estimators.

Public synthetic smoke test:

```bash
uv pip install -e '.[fitted]'
ehrdyn-icu method-inventory
ehrdyn-icu fitted-synthetic-smoke --output /tmp/kdd267-fitted-smoke.json
ehrdyn-icu controlled-21-method-smoke \
  --config configs/full_benchmark/kdd198_v2_generator_contract.json \
  --profile aki \
  --environment-seed 171901 \
  --output /tmp/kdd267-controlled-smoke.json
```

Credentialed route (all outputs are caller-owned and must remain outside Git):

```bash
ehrdyn-icu construct-ehr --output /private/ehrdyn-kdd267
ehrdyn-icu fitted-simulator --mode dry-run --constructor-root /private/ehrdyn-kdd267 --output /tmp/kdd267-dry-run.json
ehrdyn-icu fitted-simulator --mode train --constructor-root /private/ehrdyn-kdd267 --restricted-root /private/ehrdyn-kdd267-fit --output /tmp/kdd267-training.json --device cpu
ehrdyn-icu fitted-simulator --mode evaluate --constructor-root /private/ehrdyn-kdd267 --restricted-root /private/ehrdyn-kdd267-fit --output /private/ehrdyn-kdd267-results --device cpu
```

The fitted route never publishes MIMIC rows, identifiers, timestamps, arrays,
checkpoints, trajectories, or result tables. The synthetic smoke path is the
public end-to-end proof of the same source-fit, calibrated rollout, matched
method, and six-estimator workflow.

The shared fitted, controlled, and controlled-OPE inventory contains exactly
21 methods in one order. Discrete XQL and compact Dreamer V1--V3 are exact
source ports of the KDD264 benchmark adapters. Dreamer policies use sampled
actor probabilities as primary; Dreamer V2 mode is a named development
sensitivity only. Severity is absent from every public inventory.

The prior KDD263 decision remains
`complete_paper_bound_public_benchmark_artifact`. KDD267 is bound to
ResearchWiki commit
`f6b449e1a8fd906bf7ca8b68bafec2ea9ca1a0e6`, which freezes the revised
21-policy manuscript, appendix, PDF, and controlled-OPE aggregate. The
completed gate is recorded in `release/kdd267/terminal_decision.md`.
