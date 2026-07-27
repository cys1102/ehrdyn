# Controlled-environment quickstart

Use `ehrdyn-icu generate-full-suite` with the released manifest and keep generated
results in a caller-owned output directory.

The nonclinical smoke exercises the shared 21-method order and six-estimator
OPE path:

```bash
ehrdyn-icu controlled-21-method-smoke \
  --profile aki \
  --environment-seed 171901 \
  --output build/ehrdyn-controlled-smoke.json
```

The controlled OPE inventory contains the same 21 methods and no severity
policy. The smoke proves executable capability only.
