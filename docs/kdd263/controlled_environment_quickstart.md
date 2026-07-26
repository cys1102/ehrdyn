# Controlled-environment quickstart

The predecessor controlled workflow remains canonical. Use
`ehrdyn-icu generate-full-suite` with the existing manifest and keep generated
results in a caller-owned output directory.

KDD267 additionally exposes a nonclinical smoke for the exact shared
21-method order and six-estimator OPE path:

```bash
ehrdyn-icu controlled-21-method-smoke \
  --config configs/full_benchmark/kdd198_v2_generator_contract.json \
  --profile aki \
  --environment-seed 171901 \
  --output build/kdd267-controlled-smoke.json
```

The controlled OPE inventory contains the same 21 methods and no severity
policy. The smoke proves executable capability only.
