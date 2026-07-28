---
title: KDD125 Artifact Documentation
created: 2026-07-15
updated: 2026-07-15
type: writing
tags: [kdd, benchmark, reproducibility]
sources: [KDD097, KDD098R, KDD121, KDD122, KDD123-v2, KDD124-v2]
status: active
confidence: high
---

# Artifact documentation

All CSV files are aggregate-only. `final_cohort_scale.csv` is the Table 2 source. `complete_model_performance_by_cohort.csv` contains one-step and horizon-resolved logged-action recursive rows. `complete_uncertainty_by_cohort.csv` retains available and unavailable uncertainty rows. `complete_policy_and_planner_inventory.csv` includes model-free policies, fixed controls, and all component-model H1/H4/H8 planners. The three OPE files retain estimates and explicit nonexecution statuses. `manuscript_number_audit.csv` traces generated counts and summaries; `artifact_hashes.json` binds all consumed source artifacts and package outputs.
