# Canonical-v2 sepsis

Four-hour decisions, 33 SAFE forecasting features, K25 fluid-by-vasopressor
actions, at most 11 recursive steps, and the frozen terminal reward and
termination contract. Scoring is retrospective and diagnostic-only.

The recorded reward target is -1 for a recorded death under the frozen
discharge-origin 90-day proxy and +1 otherwise, only at the final valid
transition. In v2.2.0 the fitted simulator maps this channel to `terminal_once`:
one sampled -1 or +1 proxy from the final current-context reward-head mean,
zero earlier, and NaN after termination. Discounted returns lie in [-1, 1].
The submitted paper and v2.1.1 used `per_step_head` at every valid step;
select it explicitly for legacy reproduction or sensitivity analysis.
