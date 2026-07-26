"""Exact KDD171 sequence-scoring primitives required by the KDD190 H4 port."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .full_pomdp_core import R2Environment


Policy = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], np.ndarray
]


def _candidate_sequences(
    probabilities: np.ndarray,
    uniforms: np.ndarray,
    supported: np.ndarray,
) -> np.ndarray:
    local = np.sum(
        uniforms[..., None]
        > np.cumsum(probabilities, axis=-1)[:, None, :, :],
        axis=-1,
    )
    return supported[np.minimum(local, len(supported) - 1)].astype(np.int16)


def _terminal_expectation(
    environment: R2Environment, states: np.ndarray
) -> np.ndarray:
    if environment.contract.primary_reward_type != "terminal":
        return np.zeros_like(states, dtype=float)
    death = environment.death_probability(states)
    return (
        death * float(environment.contract.terminal_reward_minimum)
        + (1.0 - death)
        * float(environment.contract.terminal_reward_maximum)
    )


def _true_sequence_scores(
    environment: R2Environment,
    contexts: np.ndarray,
    sequences: np.ndarray,
    remaining_after_plan: int,
) -> np.ndarray:
    context_count, candidates, horizon = sequences.shape
    scores = np.zeros((context_count, candidates))
    for context_index, (
        initial_state,
        subtype,
        initial_pending,
    ) in enumerate(contexts):
        state_probability = np.zeros((candidates, environment.states))
        state_probability[:, initial_state] = 1.0
        pending = np.full(candidates, initial_pending, dtype=np.int16)
        for step in range(horizon):
            action = sequences[context_index, :, step]
            immediate = np.zeros(candidates)
            next_probability = np.zeros_like(state_probability)
            for state in range(environment.states):
                weight = state_probability[:, state]
                immediate += weight * np.asarray(
                    [
                        environment.immediate_reward(
                            state,
                            int(subtype),
                            int(old),
                            int(new),
                        )
                        for old, new in zip(pending, action, strict=True)
                    ]
                )
                next_probability += (
                    weight[:, None]
                    * environment.transition[
                        state, int(subtype), pending, action
                    ]
                )
            terminal = np.sum(
                next_probability
                * _terminal_expectation(
                    environment, np.arange(environment.states)
                )[None],
                axis=1,
            )
            hazard = environment.contract.termination_hazards[
                min(step, environment.horizon - 2)
            ]
            scores[context_index] += environment.discount**step * (
                immediate + hazard * terminal
            )
            state_probability = next_probability
            pending = action
        if remaining_after_plan > 0:
            final_state = np.argmax(state_probability, axis=1)
            repeat = np.asarray(
                [
                    environment.immediate_reward(
                        int(state),
                        int(subtype),
                        int(action),
                        int(action),
                    )
                    for state, action in zip(
                        final_state, pending, strict=True
                    )
                ]
            )
            terminal = _terminal_expectation(environment, final_state)
            bootstrap = remaining_after_plan * repeat + terminal
            scores[context_index] += (
                environment.discount**horizon * bootstrap
            )
    return scores
