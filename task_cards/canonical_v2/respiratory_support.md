# Canonical-v2 respiratory support

Four-hour decisions, 33 SAFE forecasting features, K25 directly observed
PEEP-by-FiO2 actions, at most 10 recursive steps, and the frozen dense reward
and termination contract. Missing action is not class zero.

The fitted reward channel `resp_meddreamer_spo2_mbp` maps to `per_step_head`.
Rewards are Gaussian samples from the learned head on current values, masks,
recency, and conditioned action at every valid transition, not the recorded
SpO2/MBP formula evaluated on simulated next states. This dense fitted emission
is unchanged from v2.1.1 and the submitted paper; it is not a terminal task.
