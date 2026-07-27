# Entrant Protocol

The compatibility schema `kdd215.entrant.v1` defines entrant declarations and
a persistent newline-delimited JSON subprocess protocol
(`kdd215.runtime.v1`). Entrant code is not imported by
the evaluator. On Linux the evaluator requires Bubblewrap, removes network
access, mounts the entrant read-only, provides a private working directory, and
applies CPU, address-space, file-size, descriptor, output, and response-time
limits. Standard output is protocol-only; bounded standard error is retained as
a failure diagnostic.

Each request contains `operation`, deterministic `seed`, and `payload`.
`predict_policy` receives only the current observable-history channels: emitted
values, masks, recency, previous action, step, public task metadata, action
count, and supported-action indices. It returns a complete `probabilities`
matrix. Every row must be finite, nonnegative, normalized within `1e-8`, and put
zero mass on unsupported actions.

`predict_component` receives the same history/action interface. A transition
entrant returns `mean`; probabilistic entrants additionally return strictly
positive `scale`. A `complete_prt` declaration is accepted for planning only
when transition, reward, and termination responses all pass their contracts.
Transition-only entrants are structurally unavailable for H1/H4/H8 planning;
the evaluator never invents reward, termination, or uncertainty outputs.

Seeds are inputs, not access capabilities. Entrants never receive latent state,
latent subtype, generator parameters, final environment seeds as a training
role, true values, final returns, another entrant, or privileged references.
