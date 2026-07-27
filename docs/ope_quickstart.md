# OPE quickstart

EHRDyn includes IS, WIS, CWPDIS, DR, WDR, and FQE under the frozen
complete-episode refit/bootstrap contract. The public controlled smoke and
synthetic fitted smoke execute all six implementations against the shared
21-policy inventory:

```bash
python -m pip install '.[fitted]'
ehrdyn-icu fitted-synthetic-smoke \
  --output build/ehrdyn-fitted-smoke.json
ehrdyn-icu controlled-21-method-smoke \
  --profile aki \
  --environment-seed 171901 \
  --output build/ehrdyn-controlled-smoke.json
```

The credentialed fitted route uses the same estimator identities with
caller-owned generated data. The current release provides these six versioned
implementations rather than an isolated external-estimator subprocess
interface.

Interpret fitted-simulator return as model-implied scenario evidence and
controlled-environment return as a synthetic reference. Neither is a clinical
policy value or causal effect.
