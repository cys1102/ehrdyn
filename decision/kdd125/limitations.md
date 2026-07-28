---
title: KDD125 Limitations
created: 2026-07-15
updated: 2026-07-15
type: writing
tags: [kdd, benchmark, causal-limitation]
sources: [KDD121, KDD122, KDD123-v2, KDD124-v2]
status: active
confidence: high
---

# Limitations

The observation process is not generated. The sepsis terminal outcome is discharge-relative and treats missing recorded death as survival proxy without an explicit right-censoring field. AF/flutter fails the frozen subject gate. Known-value truth does not transfer to EHR data, and retrospective OPE remains sensitive to support, estimator, denominator, and clipping choices. No causal, treatment-benefit, clinical-utility, deployment, or confirmatory-generalization claim is supported.
