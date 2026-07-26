# KDD267 terminal decision

Decision: `complete_21_method_paper_bound_public_benchmark_artifact`

The remotely reachable ResearchWiki commit
`f6b449e1a8fd906bf7ca8b68bafec2ea9ca1a0e6` binds the final 21-method,
no-severity manuscript, appendix, PDF, and controlled-OPE aggregate.
The exact paper and aggregate hashes are recorded in `paper_snapshot.json`.

The controlled aggregate contains 322,560 finite rows: 21 policies, six
estimators, 64 independently generated datasets, and 40 environments. It was
regenerated from frozen KDD229 and KDD264D aggregate rows without refitting a
policy, simulator, or estimator. The current inventory contains the same 21
methods in the fitted, controlled, and controlled-OPE workflows and excludes
the severity rule.

The KDD267 source, clean-install archive, schemas, checksums, privacy scan,
synthetic fitted workflow, and controlled workflow have passed their release
gates. Immutable v2.0.4 and every earlier release remain unchanged.
