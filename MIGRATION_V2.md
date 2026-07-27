# Migration from v1.3.0 to v2.0.0

Version 1.3.0 remains the immutable constructed-environment recursive entrant
release. Version 2.0.0 adds a separate credentialed-EHR component scorer and
adopts the explicitly versioned canonical-v2 five-task constructor.

The migration freezes new cohort, action, reward, schema, and metric identities
rather than silently reinterpreting results from the earlier version.
Version-specific audit history remains under `release/`.

Mixed v1/v2 submissions are rejected. Every v2 result must declare both
`ehrdyn-icu-canonical-v2.0.0` and `ehr-component-scorer-v2.0.0`.

The major version does not retroactively change v1.3.0 results or expand the
real-EHR claim boundary. Direct simulator return is available only on the
fitted and controlled simulator surfaces and remains nonclinical. The
credentialed EHR surface does not provide known policy values, causal effects,
treatment recommendations, or clinical utility.
