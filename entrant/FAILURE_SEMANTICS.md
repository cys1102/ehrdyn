# Entrant failure semantics

- Invalid schema, task hash, benchmark version, action index, probability
  normalization, or bundle hash is a hard failure with exit code 2.
- Unsupported methods or missing probabilistic objects are structural NAs;
  they must not be replaced by zero uncertainty or smoothed deterministic
  actions.
- Nonfinite OPE values, zero denominators, low ESS, and unsupported mass remain
  visible outcomes in full evaluations. They are not silently dropped.
- Public fixture success does not establish parity with credentialed MIMIC
  reconstruction or authorize retrospective policy-value claims.
- A bundle artifact marked as restricted or omitted remains nonreproducible
  from the public bundle. A synthetic substitute is never used for parity.
- The entrant protocol distinguishes `entrant_timeout`, crash/empty response, malformed JSON,
  protocol-version mismatch, action-dimension failure, nonfinite or negative
  probabilities, normalization failure, unsupported mass, nondeterminism, and
  component mean/scale contract failure. These are retained in the aggregate
  failure ledger; no fallback policy is scored under the entrant's name.
