# Fitted simulator quickstart

Run `ehrdyn-icu fitted-simulator --mode train` with a caller-owned restricted
root. The code trains candidate seeds on `source_model_train`, calibrates only
on `source_model_calibration`, selects against `validation`, and leaves
checkpoints and profiles in the restricted root. Run `--mode evaluate` only
against that same local restricted root.

Version 2.2.0 maps each task's configured reward channel to an emission mode:

| Tasks | Reward channel | Default fitted emission |
| --- | --- | --- |
| Sepsis, AKI, heart failure | `terminal_discharge_origin_90d_proxy` | `terminal_once` |
| Respiratory support | `resp_meddreamer_spo2_mbp` | `per_step_head` |
| Shock | `shock_next_mbp_component` | `per_step_head` |

Recorded terminal targets are -1 for a recorded death under the frozen
discharge-origin 90-day proxy and +1 otherwise. The fitted simulator emits a
sampled model-implied proxy, not an observed outcome. In `terminal_once`, it
finishes the unchanged rollout, evaluates the reward-head mean at each
episode's final valid current values, masks, recency, and conditioned action,
and clips `(1 + mean) / 2` to [0, 1]. NumPy uniforms seeded by `[seed, 277003]`
select +1 when the uniform is below that probability and -1 otherwise. Rewards
are zero at earlier valid transitions and NaN after termination. Discounted
returns lie in [-1, 1] for discounts in (0, 1]. Policies with the same exogenous
seed and episode count share terminal uniforms, even when their lengths differ.

Observed-association emission matches the authors' reference exactly. Other
regimes use `_condition_action`, as the rollout does: null response uses zeros,
perturbed control rolls the action one-hot by one, and observed and bounded
regimes use the action one-hot. This remains noncausal scenario analysis.
The batch's `uncertainty.reward_scale` retains the Gaussian head diagnostic;
it does not describe the emitted binary reward's uncertainty.

Dense fitted rewards remain Gaussian samples from the learned head on the
current context. Respiratory-support and shock physiological formulas define
recorded training targets; the fitted simulator does not evaluate those
formulas on simulated next states.

The submitted paper and v2.1.1 used `per_step_head` for all fitted tasks,
including terminal tasks. To select that exact legacy sampling behavior:

```bash
ehrdyn-icu fitted-synthetic-smoke \
  --reward-emission-mode per_step_head \
  --output /tmp/ehrdyn-legacy-fitted-smoke.json

ehrdyn-icu fitted-simulator --mode evaluate \
  --constructor-root /private/ehrdyn \
  --restricted-root /private/ehrdyn-fit \
  --reward-emission-mode per_step_head \
  --output /private/ehrdyn-legacy-results --device cpu
```

The flag overrides every evaluated task and also applies to `--mode all`.
Legacy terminal-task returns can leave [-1, 1] and scale with episode length.
Submitted-number reproduction still requires the original credentialed inputs,
checkpoints, profiles, seeds, and full run budgets.

Python callers can pass `reward_emission_mode="per_step_head"` to
`run_synthetic_smoke` or `evaluate_credentialed`. Direct
`LearnedSourceSimulator` construction requires an explicit
`reward_emission_mode`; no constructor default silently selects a mode.
The packaged paper-workflow and KDD262 JSON configs contain `reward_channels`,
`reward_emission_modes` keyed by channel, and `reward_emission_overrides` keyed
by task. For a KDD262 configuration copy, set
`"reward_emission_overrides": {"sepsis": "per_step_head"}` to override one task,
or map the terminal channel to `per_step_head` to override all three terminal
tasks. Unknown modes fail with `ValueError`. KDD262 freezes the selected modes
in its contract; paper-workflow receipts record them. Policy extensions reuse
the configured simulator for training and validation.
