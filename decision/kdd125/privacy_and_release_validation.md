---
title: KDD125 Privacy and Release Validation
created: 2026-07-15
updated: 2026-07-15
type: writing
tags: [kdd, benchmark, reproducibility]
sources: [KDD097, KDD098R, KDD121, KDD122, KDD123-v2, KDD124-v2]
status: active
confidence: high
---

# Privacy and release validation

- Aggregate-only source artifacts: PASS
- Patient membership export: absent
- Raw clinical notes, identifiers, exact patient times, trajectories, tensors, and checkpoints: absent
- Small-lineage primary-row scan: PASS (superseded counts occur only in lineage and historical-sensitivity CSVs)
- KDD098R receipt hashes verified: 24
- KDD121/KDD122/KDD123-v2/KDD124-v2 prerequisite decisions: PASS
- Claim boundary: development release candidate; no causal, treatment-benefit, clinical-utility, deployment, or confirmatory-generalization claim.
