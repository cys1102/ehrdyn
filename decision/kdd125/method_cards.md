---
title: KDD125 Method Cards
created: 2026-07-15
updated: 2026-07-15
type: writing
tags: [kdd, benchmark, reproducibility]
sources: [KDD122, KDD123-v2, KDD124-v2]
status: active
confidence: high
---

# Method cards

- H1: one-step exhaustive supported-action search.
- H4: four-step categorical CEM with receding-horizon first-action execution.
- H8: eight-step categorical CEM with receding-horizon first-action execution.

Terminal reward is emitted exactly once; truncated horizons use the frozen terminal value/outcome bootstrap.
