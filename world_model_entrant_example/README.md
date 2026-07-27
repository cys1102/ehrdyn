# Recursive world-model entrant example

This example is entrant-owned code: it imports no benchmark model. It communicates through
`kdd235a.runtime.v1` JSONL and implements initialization, load/fit acknowledgement, one-step
prediction, recursive action-conditioned rollout, and full policy probabilities.

Three declarations demonstrate the supported uncertainty surfaces: `point.json`,
`gaussian.json`, and `ensemble.json`. Point entrants remain eligible for RMSE and planning,
but receive structural NA for CRPS, coverage, interval width, MACE, and risk-coverage area.

The evaluator passes only observable history, masks, recency, previous actions, public task
metadata, supported actions, and deterministic seeds. Entrants never receive latent state,
subtype, future observations, true value, or final returns. Checkpoints remain entrant-owned;
only their declared identity is recorded.

Run the two-profile smoke from a clean install:

```bash
ehrdyn-icu evaluate-world-model-smoke \
  --entrant world_model_entrant_example/point.json \
  --entrant world_model_entrant_example/gaussian.json \
  --output build/ehrdyn-world-model-smoke --episodes 8
```

Arbitrary entrant execution uses the existing fail-closed subprocess sandbox. Linux entrants
require `bubblewrap`; `EHRDYN_ALLOW_UNSANDBOXED_ENTRANT=1` exists only for controlled local
tests and is not the public isolation contract.
