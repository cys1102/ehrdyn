# Off-Policy Evaluation Contract

EHRDyn uses OPE in two distinct roles. On real EHR trajectories, OPE diagnoses
longitudinal support and estimator agreement because the value of an unexecuted
policy is unknown. In the fitted and controlled simulator workflows, direct
policy execution supplies a reference return against which estimator recovery
can be measured.

Neither role estimates a clinical counterfactual value.

## Real-EHR diagnostics

The credentialed EHR workflow fits behavior probabilities on training data and
evaluates support on a separate historical development role. It reports
calibration, trajectory importance weights, effective sample size (ESS),
numerical availability, and agreement among estimators. Estimator disagreement
is a warning; it does not identify the correct policy value.

For logged action `a_it`, evaluation policy `pi_e`, and fitted behavior policy
`pi_b`,

```text
rho_it = pi_e(a_it | h_it) / max(pi_b(a_it | h_it), 1e-12)
w_i,t  = product_{j=0}^t rho_ij
ESS    = (sum_i w_i,H)^2 / sum_i w_i,H^2
```

The full reporting contract identifies the task, action and reward contract,
policy, behavior denominator, horizon, clipping, estimator, bootstrap unit,
ESS, weight concentration, and numerical status.

## Simulator recovery benchmark

The simulator benchmark fixes the same 21 target policies used by the direct
return comparison and evaluates six estimators:

- importance sampling (IS);
- weighted importance sampling (WIS);
- consistent weighted per-decision importance sampling (CWPDIS);
- doubly robust estimation (DR);
- weighted doubly robust estimation (WDR); and
- fitted Q evaluation (FQE).

Each estimator uses logged simulator datasets that are separate from policy
development and direct-return batches. Behavior and nuisance models are refit
within each dataset and bootstrap replicate. The target is the direct
full-episode simulator return of the fixed policy.

Primary metrics are value MAE, empirical coverage of nominal 90% intervals,
policy-ordering recovery, ESS, support, and numerical availability. Fitted and
controlled simulator errors use different natural normalization ranges and
must not be compared as if they shared a clinical scale.

## Interpretation boundary

Low simulator OPE error establishes recovery only for the stated simulator
mechanism and data-generation process. It does not establish retrospective-EHR
accuracy, causal treatment benefit, or suitability for clinical deployment.
No single OPE estimator is designated as a universal benchmark winner.
