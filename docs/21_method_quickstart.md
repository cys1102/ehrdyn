# Shared 21-Method Quickstart

The canonical inventory is available without MIMIC access:

```bash
ehrdyn-icu method-inventory
```

Its `method_ids`, `fitted_method_ids`, `controlled_method_ids`, and
`controlled_ope_policy_ids` arrays are identical and ordered. Each contains 21
entries. Any missing, duplicate, reordered, unknown, or severity method fails
closed.

Install the fitted extra before executing the XQL and Dreamer adapters:

```bash
python -m pip install '.[fitted]'
ehrdyn-icu fitted-synthetic-smoke \
  --output build/ehrdyn-fitted-smoke.json
ehrdyn-icu controlled-21-method-smoke \
  --profile aki \
  --environment-seed 171901 \
  --output build/ehrdyn-controlled-smoke.json
```

Both commands execute discrete XQL and compact Dreamer V1--V3 with training
seeds 3408, 3411, and 3414. The primary Dreamer interface emits sampled actor
probabilities. Dreamer V2 mode is exercised and reported only as a development
sensitivity; it is not another method.

The controlled smoke executes the released model-free adapters, transition
fits, and HGB/policy helpers. All seven transition-model specifications use the
same sequence-CEM planner: horizon 4, 64 candidates, three iterations, eight
elites, 0.2 smoothing, a support mask at every step, and first-action-only
execution.

The smokes use synthetic or constructed nonclinical data and omit scientific
return and OPE tables. They test execution, ordering, probability validity,
six-estimator coverage, and deterministic source identity. They do not support
algorithm-family superiority, clinical policy value, treatment effects, or
external validation.
