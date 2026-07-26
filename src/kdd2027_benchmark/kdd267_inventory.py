"""Frozen KDD267 method inventory shared by every simulator surface."""
from __future__ import annotations

from collections.abc import Sequence

from .errors import ReleaseContractError


SHARED_METHODS: tuple[str, ...] = (
    "fitted_behavior",
    "supported_random",
    "fixed_minimum",
    "fixed_maximum",
    "behavior_cloning",
    "decision_transformer",
    "discrete_bcq",
    "discrete_cql",
    "discrete_iql",
    "spibb_style_count_blend_adapter",
    "discrete_xql",
    "persistence_locf_plus_h4_support_only",
    "hgb_residual_plus_h4_support_only",
    "deterministic_grud_point_plus_h4_support_only",
    "causal_transformer_plus_h4_support_only",
    "categorical_rssm_plus_h4_support_only",
    "single_gaussian_grud_plus_h4_support_only",
    "matched_gaussian_ensemble_plus_h4_support_only",
    "dreamer_v1_compact",
    "dreamer_v2_compact",
    "dreamer_v3_compact",
)

CANDIDATE_METHODS: tuple[str, ...] = (
    "discrete_xql",
    "dreamer_v1_compact",
    "dreamer_v2_compact",
    "dreamer_v3_compact",
)

PREDECESSOR_METHODS: tuple[str, ...] = tuple(
    method for method in SHARED_METHODS if method not in CANDIDATE_METHODS
)

DISPLAY_NAMES: dict[str, str] = {
    "fitted_behavior": "behavior policy",
    "supported_random": "supported random",
    "fixed_minimum": "minimum action",
    "fixed_maximum": "maximum action",
    "behavior_cloning": "behavior cloning",
    "decision_transformer": "Decision Transformer",
    "discrete_bcq": "discrete BCQ",
    "discrete_cql": "discrete CQL",
    "discrete_iql": "discrete IQL",
    "spibb_style_count_blend_adapter": "count-blend adapter",
    "discrete_xql": "discrete XQL adaptation",
    "persistence_locf_plus_h4_support_only": "persistence plus common H4 planner",
    "hgb_residual_plus_h4_support_only": "HGB residual plus common H4 planner",
    "deterministic_grud_point_plus_h4_support_only": "GRU-D plus common H4 planner",
    "causal_transformer_plus_h4_support_only": "Transformer plus common H4 planner",
    "categorical_rssm_plus_h4_support_only": "categorical RSSM plus common H4 planner",
    "single_gaussian_grud_plus_h4_support_only": "single Gaussian transition model plus common H4 planner",
    "matched_gaussian_ensemble_plus_h4_support_only": "Gaussian ensemble plus common H4 planner",
    "dreamer_v1_compact": "compact Dreamer V1 adaptation",
    "dreamer_v2_compact": "compact Dreamer V2 adaptation",
    "dreamer_v3_compact": "compact Dreamer V3 adaptation",
}

FORBIDDEN_SEVERITY_IDENTIFIERS: frozenset[str] = frozenset(
    {
        "severity",
        "severity_rule",
        "observed_history_severity_rule",
        "history_severity_rule",
    }
)

TRAINING_SEEDS: tuple[int, ...] = (3408, 3411, 3414)
OPE_ESTIMATORS: tuple[str, ...] = ("IS", "WIS", "CWPDIS", "DR", "WDR", "FQE")
DREAMER_PRIMARY_EVALUATION_MODE = "sample"
DREAMER_V2_DEVELOPMENT_SENSITIVITY_MODE = "mode"


def validate_shared_inventory(methods: Sequence[str]) -> tuple[str, ...]:
    """Return the frozen inventory or fail closed on any identity drift."""
    observed = tuple(str(method) for method in methods)
    severity = sorted(set(observed).intersection(FORBIDDEN_SEVERITY_IDENTIFIERS))
    if severity:
        raise ReleaseContractError(
            "KDD267 excludes severity rules from every public method inventory: "
            + ", ".join(severity)
        )
    if len(observed) != len(set(observed)):
        raise ReleaseContractError("KDD267 method inventory contains a duplicate")
    if observed != SHARED_METHODS:
        raise ReleaseContractError(
            "KDD267 method inventory must match the frozen 21-method order exactly"
        )
    return observed


def inventory_receipt() -> dict[str, object]:
    methods = validate_shared_inventory(SHARED_METHODS)
    return {
        "schema_version": "kdd267_method_inventory_receipt_v1",
        "method_count": len(methods),
        "method_ids": list(methods),
        "display_names": [DISPLAY_NAMES[method] for method in methods],
        "fitted_method_ids": list(methods),
        "controlled_method_ids": list(methods),
        "controlled_ope_policy_ids": list(methods),
        "training_seeds": list(TRAINING_SEEDS),
        "ope_estimators": list(OPE_ESTIMATORS),
        "dreamer_primary_evaluation_mode": DREAMER_PRIMARY_EVALUATION_MODE,
        "dreamer_v2_development_sensitivity_mode": (
            DREAMER_V2_DEVELOPMENT_SENSITIVITY_MODE
        ),
        "severity_rule_included": False,
        "candidate_description": (
            "Discrete XQL and Dreamer V1-V3 are compact benchmark adaptations, "
            "not official reproductions."
        ),
    }
