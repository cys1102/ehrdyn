# Ethics and Misuse Boundary

## Intended Use

- Reproducible construction of the five credentialed ICU task interfaces.
- Evaluation of transition prediction, calibration, missingness, and
  longitudinal support.
- Nonclinical evaluation of policies and world models by direct execution in
  the released fitted or controlled simulator interfaces.
- Evaluation of OPE estimators against direct simulator-return references.
- Diagnostic comparison of named implementations under the frozen benchmark
  contracts.

## Prohibited or Unsupported Use

- Treatment recommendation or clinical decision support.
- Estimation of causal treatment effects or counterfactual benefit.
- Simulator policy optimization presented as patient benefit or a treatment
  recommendation.
- Autonomous treatment selection, triage, or deployment.
- Re-identification, linkage attacks, membership inference, or reconstruction.
- Redistribution of MIMIC-IV rows, credentials, or restricted derivatives.

Clinical actions are observational exposures affected by severity, clinician
judgment, documentation, and support limitations. Forecasting gains must not be
interpreted as evidence that an action improves an outcome. Fitted-simulator
returns are model-implied observational quantities, and controlled-environment
returns are synthetic quantities. Neither is a clinical counterfactual value.
