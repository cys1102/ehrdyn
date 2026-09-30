# Known Limitations

- MIMIC-IV access is credentialed, and no row-level data, split membership,
  checkpoints, or MIMIC-derived arrays are redistributed.
- The real-EHR analyses use one database and development roles rather than an
  untouched confirmatory test set.
- Recorded actions are interval-level exposures. They may reflect adaptation
  during the interval and do not identify assigned interventions or causal
  effects.
- Real-EHR outcomes do not reveal the value of an unexecuted policy.
  Retrospective OPE on those trajectories is therefore a support and agreement
  diagnostic, not an accuracy benchmark.
- EHR-fitted simulator returns are defined by learned observational dynamics
  and may inherit model error, distribution shift, and unmeasured confounding.
- Fitted sepsis, AKI, and heart-failure rewards default to `terminal_once` in
  v2.2.0: one model-implied -1 or +1 proxy at the final valid transition,
  zero earlier, with discounted returns in [-1, 1]. The submitted paper and
  v2.1.1 used `per_step_head`, repeatedly sampling the terminal-trained Gaussian
  head. Those legacy returns can leave [-1, 1] and depend on episode length.
- Fitted respiratory-support and shock rewards retain `per_step_head`:
  learned Gaussian samples on current values, masks, recency, and conditioned
  action. The physiological formulas define recorded training targets;
  fitted rollout rewards are not computed from the simulated next state.
- Controlled-environment returns are measurable but synthetic. Matching
  selected EHR aggregates does not establish clinical or joint-distribution
  realism.
- The 40 controlled environments are public development assets rather than a
  protected final evaluation service.
- Fixed-budget comparisons concern the released named implementations, not
  method families or asymptotic performance.
- Proxy rewards and compact task contracts require further independent
  clinical review.
- The benchmark does not establish clinical utility, causal benefit,
  deployment readiness, fairness, or cross-site generalization.
