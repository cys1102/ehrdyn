# Canonical-v2 shock

Four-hour decisions, 33 SAFE forecasting features, K25 fluid-by-vasopressor
actions, at most 11 recursive steps, and the frozen dense reward and
termination contract.

The fitted reward channel `shock_next_mbp_component` maps to `per_step_head`.
Rewards are Gaussian samples from the learned head on current values, masks,
recency, and conditioned action at every valid transition, not the recorded
next-MBP formula evaluated on simulated next states. This dense fitted emission
is unchanged from v2.1.1 and the submitted paper; it is not a terminal task.
