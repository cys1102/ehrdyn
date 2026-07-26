# KDD263 paper-bound artifact

This successor preserves the immutable v2.0.2 constructor, scorer, schemas, and
40 controlled-environment workflow. It adds a paper-bound fitted-simulator
route: an authorized MIMIC-IV v3.1 user constructs the five canonical task
interfaces, obtains disjoint `source_model_train` and
`source_model_calibration` roles, trains and calibrates the Gaussian recurrent
source simulator, freezes the selected checkpoint, and runs the matched 17
methods with the six OPE estimators.

Public synthetic smoke test:

```bash
uv pip install -e '.[fitted]'
ehrdyn-icu fitted-synthetic-smoke --output /tmp/kdd263-fitted-smoke.json
```

Credentialed route (all outputs are caller-owned and must remain outside Git):

```bash
ehrdyn-icu construct-ehr --output /private/ehrdyn-kdd263
ehrdyn-icu fitted-simulator --mode dry-run --constructor-root /private/ehrdyn-kdd263 --output /tmp/kdd263-dry-run.json
ehrdyn-icu fitted-simulator --mode train --constructor-root /private/ehrdyn-kdd263 --restricted-root /private/ehrdyn-kdd263-fit --output /tmp/kdd263-training.json --device cpu
ehrdyn-icu fitted-simulator --mode evaluate --constructor-root /private/ehrdyn-kdd263 --restricted-root /private/ehrdyn-kdd263-fit --output /private/ehrdyn-kdd263-results --device cpu
```

The fitted route never publishes MIMIC rows, identifiers, timestamps, arrays,
checkpoints, trajectories, or result tables. The synthetic smoke path is the
public end-to-end proof of the same source-fit, calibrated rollout, matched
method, and six-estimator workflow.

The publication decision is `complete_paper_bound_public_benchmark_artifact`.
`release/kdd263/paper_snapshot.json` binds the exact manuscript, appendix, and
PDF hashes to the remotely reachable ResearchWiki commit
`146a02840570368ac7a3efd106cf10b0d3cbba91`. The immutable paper-bound artifact
was published as v2.0.3. Version 2.0.4 changes only documentation and package
metadata so that the public status consistently reflects that completed
publication.
