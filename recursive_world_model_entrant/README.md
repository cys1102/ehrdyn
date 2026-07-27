# Full recursive entrant demonstration

This package is deliberately entrant-owned. It imports no EHRDyn transition
class, checkpoint, private configuration, or historical expected result. It
communicates only through the public `kdd235a.runtime.v1` JSONL protocol.

The example is a small recurrent independent-Gaussian predictor. Its policy
surface executes the frozen three-iteration, 64-candidate, eight-elite H4
support-only categorical-CEM contract. The entrant's deliberately simple
planner score is an action-cost proxy; it is not an EHR reward model or a
clinical objective.

The package demonstrates interface usability and is not part of the scientific
leaderboard.
