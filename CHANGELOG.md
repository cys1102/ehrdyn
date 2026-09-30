# Changelog

## 2.2.0

The EHR-fitted simulator sampled its reward head at every active step, including
for sepsis, AKI, and heart failure. Those heads are trained only on terminal
reward cells. Repeated Gaussian samples summed terminal-outcome proxies,
allowing discounted returns outside [-1, 1] and dependence on episode length.
Release v2.1.1 and the submitted paper used this `per_step_head` behavior.

The packaged reward-channel configuration now selects `terminal_once` for
sepsis, AKI, and heart failure. After the unchanged rollout, the final valid
current context supplies the reward-head mean. The simulator clips
`(1 + mean) / 2` to [0, 1] and draws one -1 or +1 outcome using an independent
NumPy uniform stream seeded by `[seed, 277003]`. Earlier valid rewards are zero;
post-termination rewards remain NaN. Fitted discounted returns under
`terminal_once` lie in [-1, 1] for discounts in (0, 1]. The outcome is a sampled
model-implied proxy, not a newly observed death or a causal treatment effect.

All response regimes use the rollout's `_condition_action` rule for the final
reward context. Observed-association behavior matches the authors' reference
rule exactly. The Gaussian head scale remains a head diagnostic in the batch's
uncertainty arrays; it is not the uncertainty of the binary emitted reward.

Respiratory support and shock retain `per_step_head`. Their fitted rewards are
learned Gaussian samples on current values, masks, recency, and conditioned
action, not the stated physiological formulas evaluated on simulated next
states. Those formulas define the recorded training targets.

Users can explicitly select `per_step_head` for any task through configuration
or `--reward-emission-mode per_step_head`. This preserves the v2.1.1 sampler for
submitted-number reproduction and sensitivity analysis; full numerical
reproduction still requires the original credentialed inputs and run budgets.
Frozen historical release receipts and paper identities remain unchanged.

API change: `LearnedSourceSimulator` now requires the keyword-only argument
`reward_emission_mode` (`"terminal_once"` or `"per_step_head"`). Code that
constructs the simulator directly must pass it; omitting it raises a
`TypeError`. `resolve_reward_emission_mode(config, task_name, override)` returns
the explicit override, then any per-task `reward_emission_overrides` entry, then
the mode configured for the task's reward channel.
