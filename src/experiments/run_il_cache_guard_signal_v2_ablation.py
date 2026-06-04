from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.trace_reader import TraceReader
from src.experiments import run_il_cache_guard_signal_v2 as il_guard_v2
from src.experiments import run_il_cache_guard_signal_v2_no_guard as il_no_guard_v2


EXPERIMENT_FAMILY = "guardrails_signal_v2"
DEFAULT_RESULTS_ROOT = "results/guardrails_signal_v2"
DEFAULT_CAP020_RESULTS_ROOT = "results/guardrails_signal_v2_cap020_validation"
DEFAULT_CACHE_SIZE_PERCENTAGES = (0.8, 1.0, 2.0, 3.0, 4.0, 5.0)
DATASETS = ("wikipedia_september_2007", "wiki2018")
SUMMARY_AGGREGATE_KEYS = (
    "total_selected_admissions",
    "selected_admissions_total",
    "total_applied_admissions",
    "applied_admissions_total",
    "admissions_per_1000_requests",
    "aggregate_hit_yield_per_applied_admission",
    "aggregate_pollution_ratio",
    "aggregate_admission_precision",
    "postfill_hit_ratio",
    "postfill_selected_admissions",
    "postfill_applied_admissions",
    "postfill_admissions_per_1000_requests",
    "postfill_hit_yield_per_applied_admission",
    "postfill_pollution_ratio",
    "postfill_admission_precision",
    "quality_suppressed_total",
    "quality_suppressed_next_slot_requests_total",
    "quality_admitted_next_slot_requests_total",
    "quality_suppressed_next_slot_request_rate",
    "quality_admitted_next_slot_request_rate",
    "quality_admitted_to_suppressed_request_ratio",
)


CORE_CLEAN_VARIANTS = (
    "no_guard_legacy",
    "cap_only",
    "precision_only",
    "quality_only",
    "full",
)
CORE_CLEAN_CAP020_VARIANTS = (
    "no_guard_legacy",
    "cap_only_cap020",
    "quality_only_cap020",
    "precision_only_cap020",
    "full_cap020",
)
NO_CAP_STRESS_VARIANTS = (
    "no_guard_legacy",
    "precision_no_cap",
    "quality_no_cap",
    "full_no_cap",
)
CAP_MASKING_GRID_VARIANTS = (
    "quality_only_cap005",
    "quality_only_cap010",
    "quality_only_cap015",
    "quality_only_cap020",
    "quality_only_cap025",
    "quality_only_cap050",
    "quality_only_cap075",
    "quality_only_no_cap",
    "full_cap005",
    "full_cap010",
    "full_cap015",
    "full_cap020",
    "full_cap025",
    "full_cap050",
    "full_cap075",
    "full_no_cap",
)
SCORE_QUALITY_BUDGET_MATCHED_VARIANTS = (
    "precision_only_budget_matched_uniform",
    "precision_only_budget_matched_random",
    "precision_only_budget_matched_permuted_quality",
    "full",
)
SCORE_QUALITY_BUDGET_MATCHED_CAP020_VARIANTS = (
    "precision_only_cap020",
    "full_cap020",
    "precision_only_budget_matched_uniform_cap020",
    "precision_only_budget_matched_random_cap020",
    "precision_only_budget_replay_cap020",
    "precision_only_budget_replay_permuted_cap020",
    "precision_only_budget_matched_permuted_quality_cap020",
)
PRECISION_VALIDITY_VARIANTS = (
    "precision_reward_only_legacy",
    "precision_bidirectional_fixed_target",
    "precision_bidirectional_base_rate_target",
    "precision_bidirectional_base_rate_margin_target",
    "precision_lag_fixed",
    "precision_lag_weighted",
    "precision_lag_max_legacy",
)
PRECISION_FLOOR_VALIDITY_VARIANTS = (
    "precision_bidirectional_no_floor",
    "full_no_alpha_floor",
    "precision_floor_grid_050",
    "precision_floor_grid_070",
    "precision_floor_grid_090",
)
QUALITY_VALIDITY_VARIANTS = (
    "quality_spread_symmetric",
    "quality_spread_safety_only",
    "quality_calibrated_spread_safety",
    "quality_boundary_margin_safety",
    "full_quality_spread_symmetric",
    "full_quality_spread_safety_only",
    "full_quality_calibrated_spread_safety",
    "full_quality_boundary_margin_safety",
)
QUALITY_MODE_CAP020_VARIANTS = (
    "full_quality_spread_safety_cap020",
    "full_quality_boundary_margin_safety_cap020",
    "full_quality_calibrated_spread_safety_cap020",
)
NEGATIVE_CONTROL_VARIANTS = (
    "quality_random_control",
    "quality_inverted_control",
    "quality_permuted_control",
    "precision_random_control",
    "precision_permuted_control",
)
CAP020_VALIDATION_VARIANTS = tuple(
    dict.fromkeys(
        CORE_CLEAN_CAP020_VARIANTS
        + SCORE_QUALITY_BUDGET_MATCHED_CAP020_VARIANTS
        + QUALITY_MODE_CAP020_VARIANTS
    )
)

VARIANT_SETS: Dict[str, Tuple[str, ...]] = {
    "core_clean": CORE_CLEAN_VARIANTS,
    "core_clean_cap020": CORE_CLEAN_CAP020_VARIANTS,
    "no_cap_stress": NO_CAP_STRESS_VARIANTS,
    "cap_masking_grid": CAP_MASKING_GRID_VARIANTS,
    "score_quality_budget_matched": SCORE_QUALITY_BUDGET_MATCHED_VARIANTS,
    "score_quality_budget_matched_cap020": SCORE_QUALITY_BUDGET_MATCHED_CAP020_VARIANTS,
    "precision_validity": PRECISION_VALIDITY_VARIANTS,
    "precision_floor_validity": PRECISION_FLOOR_VALIDITY_VARIANTS,
    "quality_validity": QUALITY_VALIDITY_VARIANTS,
    "quality_mode_cap020": QUALITY_MODE_CAP020_VARIANTS,
    "negative_controls": NEGATIVE_CONTROL_VARIANTS,
    "cap020_validation": CAP020_VALIDATION_VARIANTS,
}
ALL_VARIANTS = tuple(
    dict.fromkeys(
        CORE_CLEAN_VARIANTS
        + CORE_CLEAN_CAP020_VARIANTS
        + NO_CAP_STRESS_VARIANTS
        + CAP_MASKING_GRID_VARIANTS
        + SCORE_QUALITY_BUDGET_MATCHED_VARIANTS
        + SCORE_QUALITY_BUDGET_MATCHED_CAP020_VARIANTS
        + PRECISION_VALIDITY_VARIANTS
        + PRECISION_FLOOR_VALIDITY_VARIANTS
        + QUALITY_VALIDITY_VARIANTS
        + QUALITY_MODE_CAP020_VARIANTS
        + NEGATIVE_CONTROL_VARIANTS
    )
)
VARIANT_SETS["all"] = ALL_VARIANTS


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str
    total_requests: int
    warmup_requests: int
    slot_size: int


@dataclass(frozen=True)
class VariantSpec:
    variant: str
    module: Any
    precision_feedback_enabled: bool
    score_quality_enabled: bool
    purpose: str
    overrides: Dict[str, Any]


def _main_precision_overrides() -> Dict[str, Any]:
    return {
        "PRECISION_CONTROL_MODE": "bidirectional",
        "EFFECTIVE_PRECISION_LAG_MODE": "weighted",
        "ADMISSION_PRECISION_TARGET_MODE": "base_rate_margin",
        "PRECISION_CONTROL_OVERRIDE_MODE": "normal",
    }


def _quality_off_overrides() -> Dict[str, Any]:
    return {
        "SCORE_QUALITY_SIGNAL_MODE": "off",
        "SCORE_QUALITY_CONTROL_ROLE": "safety",
        "SCORE_QUALITY_MIN": 1.0,
        "SCORE_QUALITY_MAX": 1.0,
        "SCORE_QUALITY_MIN_BOOST": 0.0,
        "QUALITY_CONTROL_MODE": "normal",
    }


def _precision_off_overrides() -> Dict[str, Any]:
    return {
        "PRECISION_CONTROL_MODE": "off",
        "PRECISION_CONTROL_OVERRIDE_MODE": "normal",
    }


def _main_quality_overrides() -> Dict[str, Any]:
    return {
        "SCORE_QUALITY_SIGNAL_MODE": "calibrated_spread",
        "SCORE_QUALITY_CONTROL_ROLE": "safety",
        "SCORE_QUALITY_MIN": 0.7,
        "SCORE_QUALITY_MAX": 1.0,
        "SCORE_QUALITY_MIN_BOOST": 0.0,
        "QUALITY_CONTROL_MODE": "normal",
    }


def _base_guard_overrides(cap: float = 0.05) -> Dict[str, Any]:
    return {
        "SCORE_GATE_TOP_PERCENT": float(cap),
        "GUARD_QUALITY_INTERACTION_MODE": "budget_multiplier",
        "QUALITY_CONTROL_MODE": "normal",
        "PRECISION_CONTROL_OVERRIDE_MODE": "normal",
        "BUDGET_MATCH_MODE": "off",
        "BUDGET_MATCH_MULTIPLIER": 1.0,
        "BUDGET_MATCH_TARGET_MULTIPLIER": 1.0,
    }


def _merge(*parts: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for part in parts:
        merged.update(part)
    return merged


def resolve_dataset(name: str) -> DatasetSpec:
    source = {
        "wikipedia_september_2007": il_guard_v2.WIKIPEDIA_SEPTEMBER_2007,
        "wiki2018": il_guard_v2.WIKI2018,
    }.get(name)
    if source is None:
        raise ValueError(f"Unknown dataset: {name}")
    return DatasetSpec(
        name=str(source["name"]),
        path=str(source["path"]),
        total_requests=int(source["num_total_requests"]),
        warmup_requests=int(source["num_warmup_requests"]),
        slot_size=int(source["slot_size"]),
    )


def resolve_datasets(dataset_arg: str) -> List[DatasetSpec]:
    if dataset_arg == "all":
        return [resolve_dataset(name) for name in DATASETS]
    return [resolve_dataset(dataset_arg)]


def resolve_variants(variant_args: Optional[List[str]], variant_set: str) -> List[str]:
    if variant_args:
        requested = variant_args
    else:
        requested = list(VARIANT_SETS[variant_set])
    if "all" in requested:
        return list(ALL_VARIANTS)
    unknown = [variant for variant in requested if variant not in ALL_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variant(s): {', '.join(unknown)}")
    return list(dict.fromkeys(requested))


def learner_label(base_learner: str) -> str:
    return str(base_learner).upper()


CAP020_MODEL_NAME_STEMS = {
    "precision_only_budget_matched_uniform_cap020": "precision_budget_uniform_cap020",
    "precision_only_budget_matched_random_cap020": "precision_budget_random_cap020",
    "precision_only_budget_replay_cap020": "precision_budget_replay_cap020",
    "precision_only_budget_replay_permuted_cap020": "precision_budget_replay_permuted_cap020",
    "precision_only_budget_matched_permuted_quality_cap020": "precision_budget_permuted_quality_cap020",
}


def model_name_for_variant(variant: str, feature_set: str, base_learner: str) -> str:
    stem = CAP020_MODEL_NAME_STEMS.get(variant, variant)
    return f"ilnse_{feature_set}_guardv2_{stem}_{learner_label(base_learner)}"


def get_variant_spec(variant: str) -> VariantSpec:
    if variant == "no_guard_legacy":
        return VariantSpec(
            variant=variant,
            module=il_no_guard_v2,
            precision_feedback_enabled=False,
            score_quality_enabled=False,
            purpose="legacy no-guard reference with no top-percent cap",
            overrides=_merge(
                _base_guard_overrides(1.0),
                _precision_off_overrides(),
                _quality_off_overrides(),
                {"ADMISSION_PRECISION_SENSITIVITY": 0.0},
            ),
        )
    if variant == "cap_only":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=False,
            score_quality_enabled=False,
            purpose="isolates the hard top-percent cap",
            overrides=_merge(_base_guard_overrides(0.05), _precision_off_overrides(), _quality_off_overrides()),
        )
    if variant == "precision_only":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=True,
            score_quality_enabled=False,
            purpose="tests effective-precision feedback above the same hard cap",
            overrides=_merge(_base_guard_overrides(0.05), _main_precision_overrides(), _quality_off_overrides()),
        )
    if variant == "quality_only":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=False,
            score_quality_enabled=True,
            purpose="tests score-quality modulation above the same hard cap",
            overrides=_merge(_base_guard_overrides(0.05), _precision_off_overrides(), _main_quality_overrides()),
        )
    if variant == "full":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=True,
            score_quality_enabled=True,
            purpose="main v2 precision plus calibrated safety quality design",
            overrides=_merge(_base_guard_overrides(0.05), _main_precision_overrides(), _main_quality_overrides()),
        )
    if variant == "cap_only_cap020":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=False,
            score_quality_enabled=False,
            purpose="final-cap clean ablation: hard top-percent cap only at 0.02",
            overrides=_merge(_base_guard_overrides(0.02), _precision_off_overrides(), _quality_off_overrides()),
        )
    if variant == "precision_only_cap020":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=True,
            score_quality_enabled=False,
            purpose="final-cap clean ablation: precision feedback only at cap 0.02",
            overrides=_merge(_base_guard_overrides(0.02), _main_precision_overrides(), _quality_off_overrides()),
        )
    if variant == "precision_only_budget_matched_uniform":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=True,
            score_quality_enabled=False,
            purpose="budget-matched uniform throttle without score-quality timing",
            overrides=_merge(
                _base_guard_overrides(0.05),
                _main_precision_overrides(),
                _quality_off_overrides(),
                {
                    "BUDGET_MATCH_MODE": "uniform",
                    "BUDGET_MATCH_MULTIPLIER": 0.904,
                },
            ),
        )
    if variant == "precision_only_budget_matched_random":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=True,
            score_quality_enabled=False,
            purpose="budget-matched deterministic random slot throttle",
            overrides=_merge(
                _base_guard_overrides(0.05),
                _main_precision_overrides(),
                _quality_off_overrides(),
                {
                    "BUDGET_MATCH_MODE": "random_slot_throttle",
                    "BUDGET_MATCH_TARGET_MULTIPLIER": 0.904,
                },
            ),
        )
    if variant == "precision_only_budget_matched_permuted_quality":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=True,
            score_quality_enabled=True,
            purpose="budget-matched control using permuted score-quality timing",
            overrides=_merge(
                _base_guard_overrides(0.05),
                _main_precision_overrides(),
                _main_quality_overrides(),
                {"QUALITY_CONTROL_MODE": "permuted"},
            ),
        )
    if variant == "precision_only_budget_matched_uniform_cap020":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=True,
            score_quality_enabled=False,
            purpose="cap-0.02 budget control: uniform throttle without score-quality timing",
            overrides=_merge(
                _base_guard_overrides(0.02),
                _main_precision_overrides(),
                _quality_off_overrides(),
                {
                    "BUDGET_MATCH_MODE": "uniform",
                    "BUDGET_MATCH_MULTIPLIER": 1.0,
                },
            ),
        )
    if variant == "precision_only_budget_matched_random_cap020":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=True,
            score_quality_enabled=False,
            purpose="cap-0.02 budget control: deterministic random slot throttle",
            overrides=_merge(
                _base_guard_overrides(0.02),
                _main_precision_overrides(),
                _quality_off_overrides(),
                {
                    "BUDGET_MATCH_MODE": "random_slot_throttle",
                    "BUDGET_MATCH_TARGET_MULTIPLIER": 1.0,
                    "CONTROL_RANDOM_SEED": 42,
                },
            ),
        )
    if variant == "precision_only_budget_replay_cap020":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=True,
            score_quality_enabled=False,
            purpose="cap-0.02 offline diagnostic: precision with exact full budget replay",
            overrides=_merge(
                _base_guard_overrides(0.02),
                _main_precision_overrides(),
                _quality_off_overrides(),
                {
                    "BUDGET_MATCH_MODE": "slot_replay",
                    "BUDGET_REPLAY_PATH": None,
                    "BUDGET_REPLAY_KEY": "budget",
                    "BUDGET_REPLAY_STRICT": True,
                    "BUDGET_REPLAY_ALLOW_CLAMP": True,
                    "BUDGET_REPLAY_PERMUTE_SEED": 42,
                    "OFFLINE_DIAGNOSTIC_CONTROL": True,
                },
            ),
        )
    if variant == "precision_only_budget_replay_permuted_cap020":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=True,
            score_quality_enabled=False,
            purpose="cap-0.02 offline diagnostic: precision with permuted full budget replay",
            overrides=_merge(
                _base_guard_overrides(0.02),
                _main_precision_overrides(),
                _quality_off_overrides(),
                {
                    "BUDGET_MATCH_MODE": "slot_replay_permuted",
                    "BUDGET_REPLAY_PATH": None,
                    "BUDGET_REPLAY_KEY": "budget",
                    "BUDGET_REPLAY_STRICT": True,
                    "BUDGET_REPLAY_ALLOW_CLAMP": True,
                    "BUDGET_REPLAY_PERMUTE_SEED": 42,
                    "OFFLINE_DIAGNOSTIC_CONTROL": True,
                },
            ),
        )
    if variant == "precision_only_budget_matched_permuted_quality_cap020":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=True,
            score_quality_enabled=True,
            purpose="cap-0.02 budget control: precision with permuted score-quality timing",
            overrides=_merge(
                _base_guard_overrides(0.02),
                _main_precision_overrides(),
                _main_quality_overrides(),
                {"QUALITY_CONTROL_MODE": "permuted"},
            ),
        )
    if variant in ("precision_no_cap", "quality_no_cap", "full_no_cap"):
        base = variant.replace("_no_cap", "")
        capless = _base_guard_overrides(1.0)
        if base == "precision":
            return VariantSpec(
                variant=variant,
                module=il_guard_v2,
                precision_feedback_enabled=True,
                score_quality_enabled=False,
                purpose="precision feedback with cap removed",
                overrides=_merge(capless, _main_precision_overrides(), _quality_off_overrides()),
            )
        if base == "quality":
            return VariantSpec(
                variant=variant,
                module=il_guard_v2,
                precision_feedback_enabled=False,
                score_quality_enabled=True,
                purpose="score-quality modulation with cap removed",
                overrides=_merge(capless, _precision_off_overrides(), _main_quality_overrides()),
            )
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=True,
            score_quality_enabled=True,
            purpose="full v2 control with cap removed",
            overrides=_merge(capless, _main_precision_overrides(), _main_quality_overrides()),
        )

    cap_grid = {
        "quality_only_cap005": ("quality", 0.005),
        "quality_only_cap010": ("quality", 0.010),
        "quality_only_cap015": ("quality", 0.015),
        "quality_only_cap020": ("quality", 0.020),
        "quality_only_cap025": ("quality", 0.025),
        "quality_only_cap050": ("quality", 0.05),
        "quality_only_cap075": ("quality", 0.075),
        "quality_only_no_cap": ("quality", 1.0),
        "full_cap005": ("full", 0.005),
        "full_cap010": ("full", 0.010),
        "full_cap015": ("full", 0.015),
        "full_cap020": ("full", 0.020),
        "full_cap025": ("full", 0.025),
        "full_cap050": ("full", 0.05),
        "full_cap075": ("full", 0.075),
    }
    if variant in cap_grid:
        kind, cap = cap_grid[variant]
        if kind == "quality":
            return VariantSpec(
                variant=variant,
                module=il_guard_v2,
                precision_feedback_enabled=False,
                score_quality_enabled=True,
                purpose=f"quality-only masking check at top cap {cap}",
                overrides=_merge(_base_guard_overrides(cap), _precision_off_overrides(), _main_quality_overrides()),
            )
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=True,
            score_quality_enabled=True,
            purpose=f"full-control masking check at top cap {cap}",
            overrides=_merge(_base_guard_overrides(cap), _main_precision_overrides(), _main_quality_overrides()),
        )

    precision_variants: Dict[str, Dict[str, Any]] = {
        "precision_reward_only_legacy": {
            "PRECISION_CONTROL_MODE": "reward_only_legacy",
            "EFFECTIVE_PRECISION_LAG_MODE": "max_legacy",
            "ADMISSION_PRECISION_TARGET_MODE": "fixed",
        },
        "precision_bidirectional_fixed_target": {
            "PRECISION_CONTROL_MODE": "bidirectional",
            "EFFECTIVE_PRECISION_LAG_MODE": "weighted",
            "ADMISSION_PRECISION_TARGET_MODE": "fixed",
        },
        "precision_bidirectional_base_rate_target": {
            "PRECISION_CONTROL_MODE": "bidirectional",
            "EFFECTIVE_PRECISION_LAG_MODE": "weighted",
            "ADMISSION_PRECISION_TARGET_MODE": "base_rate_ema",
        },
        "precision_bidirectional_base_rate_margin_target": {
            "PRECISION_CONTROL_MODE": "bidirectional",
            "EFFECTIVE_PRECISION_LAG_MODE": "weighted",
            "ADMISSION_PRECISION_TARGET_MODE": "base_rate_margin",
        },
        "precision_lag_fixed": {
            "PRECISION_CONTROL_MODE": "bidirectional",
            "EFFECTIVE_PRECISION_LAG_MODE": "fixed",
            "EFFECTIVE_PRECISION_FIXED_LAG": 1,
            "ADMISSION_PRECISION_TARGET_MODE": "base_rate_margin",
        },
        "precision_lag_weighted": {
            "PRECISION_CONTROL_MODE": "bidirectional",
            "EFFECTIVE_PRECISION_LAG_MODE": "weighted",
            "ADMISSION_PRECISION_TARGET_MODE": "base_rate_margin",
        },
        "precision_lag_max_legacy": {
            "PRECISION_CONTROL_MODE": "bidirectional",
            "EFFECTIVE_PRECISION_LAG_MODE": "max_legacy",
            "ADMISSION_PRECISION_TARGET_MODE": "base_rate_margin",
        },
    }
    if variant in precision_variants:
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=True,
            score_quality_enabled=False,
            purpose="precision controller validity ablation",
            overrides=_merge(
                _base_guard_overrides(0.05),
                precision_variants[variant],
                {"PRECISION_CONTROL_OVERRIDE_MODE": "normal"},
                _quality_off_overrides(),
            ),
        )
    if variant == "precision_bidirectional_no_floor":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=True,
            score_quality_enabled=False,
            purpose="precision controller validity with alpha floor disabled",
            overrides=_merge(
                _base_guard_overrides(0.05),
                _main_precision_overrides(),
                _quality_off_overrides(),
                {"DRIFT_ALPHA_FLOOR_MULT": 0.0},
            ),
        )
    if variant == "full_no_alpha_floor":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=True,
            score_quality_enabled=True,
            purpose="full v2 control with alpha floor disabled",
            overrides=_merge(
                _base_guard_overrides(0.05),
                _main_precision_overrides(),
                _main_quality_overrides(),
                {"DRIFT_ALPHA_FLOOR_MULT": 0.0},
            ),
        )
    precision_floor_grid = {
        "precision_floor_grid_050": 0.5,
        "precision_floor_grid_070": 0.7,
        "precision_floor_grid_090": 0.9,
    }
    if variant in precision_floor_grid:
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=True,
            score_quality_enabled=False,
            purpose="precision-only alpha-floor masking grid",
            overrides=_merge(
                _base_guard_overrides(0.05),
                _main_precision_overrides(),
                _quality_off_overrides(),
                {"DRIFT_ALPHA_FLOOR_MULT": precision_floor_grid[variant]},
            ),
        )

    quality_variants: Dict[str, Tuple[str, str, float, float, bool]] = {
        "quality_spread_symmetric": ("spread_prev_ema", "symmetric", 0.7, 1.3, False),
        "quality_spread_safety_only": ("spread_prev_ema", "safety", 0.7, 1.0, False),
        "quality_calibrated_spread_safety": ("calibrated_spread", "safety", 0.7, 1.0, False),
        "quality_boundary_margin_safety": ("boundary_margin", "safety", 0.7, 1.0, False),
        "full_quality_spread_symmetric": ("spread_prev_ema", "symmetric", 0.7, 1.3, True),
        "full_quality_spread_safety_only": ("spread_prev_ema", "safety", 0.7, 1.0, True),
        "full_quality_calibrated_spread_safety": ("calibrated_spread", "safety", 0.7, 1.0, True),
        "full_quality_boundary_margin_safety": ("boundary_margin", "safety", 0.7, 1.0, True),
    }
    cap020_quality_variants: Dict[str, Tuple[str, str, float, float, bool]] = {
        "full_quality_spread_safety_cap020": ("spread_prev_ema", "safety", 0.7, 1.0, True),
        "full_quality_boundary_margin_safety_cap020": ("boundary_margin", "safety", 0.7, 1.0, True),
        "full_quality_calibrated_spread_safety_cap020": ("calibrated_spread", "safety", 0.7, 1.0, True),
    }
    if variant in quality_variants:
        mode, role, q_min, q_max, with_precision = quality_variants[variant]
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=with_precision,
            score_quality_enabled=True,
            purpose="score-quality validity ablation",
            overrides=_merge(
                _base_guard_overrides(0.05),
                _main_precision_overrides() if with_precision else _precision_off_overrides(),
                {
                    "SCORE_QUALITY_SIGNAL_MODE": mode,
                    "SCORE_QUALITY_CONTROL_ROLE": role,
                    "SCORE_QUALITY_MIN": q_min,
                    "SCORE_QUALITY_MAX": q_max,
                    "SCORE_QUALITY_MIN_BOOST": 0.0,
                    "QUALITY_CONTROL_MODE": "normal",
                },
            ),
        )
    if variant in cap020_quality_variants:
        mode, role, q_min, q_max, with_precision = cap020_quality_variants[variant]
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=with_precision,
            score_quality_enabled=True,
            purpose="cap-0.02 score-quality mode sanity ablation",
            overrides=_merge(
                _base_guard_overrides(0.02),
                _main_precision_overrides() if with_precision else _precision_off_overrides(),
                {
                    "SCORE_QUALITY_SIGNAL_MODE": mode,
                    "SCORE_QUALITY_CONTROL_ROLE": role,
                    "SCORE_QUALITY_MIN": q_min,
                    "SCORE_QUALITY_MAX": q_max,
                    "SCORE_QUALITY_MIN_BOOST": 0.0,
                    "QUALITY_CONTROL_MODE": "normal",
                },
            ),
        )

    if variant == "quality_random_control":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=False,
            score_quality_enabled=True,
            purpose="negative control: deterministic random quality multiplier",
            overrides=_merge(
                _base_guard_overrides(0.05),
                _precision_off_overrides(),
                _main_quality_overrides(),
                {"QUALITY_CONTROL_MODE": "random"},
            ),
        )
    if variant == "quality_inverted_control":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=False,
            score_quality_enabled=True,
            purpose="negative control: inverted score-quality direction",
            overrides=_merge(
                _base_guard_overrides(0.05),
                _precision_off_overrides(),
                _main_quality_overrides(),
                {
                    "SCORE_QUALITY_CONTROL_ROLE": "symmetric",
                    "SCORE_QUALITY_MAX": 1.3,
                    "QUALITY_CONTROL_MODE": "inverted",
                },
            ),
        )
    if variant == "quality_permuted_control":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=False,
            score_quality_enabled=True,
            purpose="negative control: lagged empirical permutation of quality multipliers",
            overrides=_merge(
                _base_guard_overrides(0.05),
                _precision_off_overrides(),
                _main_quality_overrides(),
                {"QUALITY_CONTROL_MODE": "permuted"},
            ),
        )
    if variant == "precision_random_control":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=True,
            score_quality_enabled=False,
            purpose="negative control: deterministic random precision multiplier",
            overrides=_merge(
                _base_guard_overrides(0.05),
                _main_precision_overrides(),
                _quality_off_overrides(),
                {"PRECISION_CONTROL_OVERRIDE_MODE": "random"},
            ),
        )
    if variant == "precision_permuted_control":
        return VariantSpec(
            variant=variant,
            module=il_guard_v2,
            precision_feedback_enabled=True,
            score_quality_enabled=False,
            purpose="negative control: lagged empirical permutation of precision multipliers",
            overrides=_merge(
                _base_guard_overrides(0.05),
                _main_precision_overrides(),
                _quality_off_overrides(),
                {"PRECISION_CONTROL_OVERRIDE_MODE": "permuted"},
            ),
        )
    raise ValueError(f"Unknown variant: {variant}")


@contextmanager
def patched_module(module: Any, overrides: Dict[str, Any], disable_progress: bool):
    missing = object()
    saved: Dict[str, Any] = {}
    for name, value in overrides.items():
        saved[name] = getattr(module, name, missing)
        setattr(module, name, value)

    tqdm_saved = None
    if disable_progress:
        tqdm_saved = getattr(module, "tqdm")

        def quiet_tqdm(*args, **kwargs):
            kwargs["disable"] = True
            return tqdm_saved(*args, **kwargs)

        setattr(module, "tqdm", quiet_tqdm)

    try:
        yield
    finally:
        if tqdm_saved is not None:
            setattr(module, "tqdm", tqdm_saved)
        for name, value in saved.items():
            if value is missing:
                delattr(module, name)
            else:
                setattr(module, name, value)


def parse_cache_size_percentages(value: str) -> Tuple[float, ...]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise ValueError("--cache-size-percentages must contain at least one value.")
    percentages = tuple(float(part) for part in parts)
    for pct in percentages:
        if pct <= 0.0:
            raise ValueError("--cache-size-percentages values must be > 0.")
    return percentages


def count_distinct_objects(trace_path: str, total_requests: int) -> int:
    reader = TraceReader(path=trace_path, max_rows=total_requests)
    unique_objects = set()
    for idx, req in enumerate(reader.iter_requests()):
        if idx >= total_requests:
            break
        unique_objects.add(req["object_id"])
    return len(unique_objects)


def compute_capacities(
    trace_path: str,
    total_requests: int,
    percentages: Iterable[float],
    num_unique_objects: Optional[int] = None,
) -> Tuple[List[Dict[str, float]], int]:
    if num_unique_objects is None:
        num_unique_objects = count_distinct_objects(trace_path, total_requests)
    capacities = [
        {
            "cache_size_objects": max(1, int(num_unique_objects * float(percent) / 100.0)),
            "capacity_percent": float(percent),
        }
        for percent in percentages
    ]
    return capacities, num_unique_objects


def same_cache_percentages(left: Iterable[Any], right: Iterable[Any]) -> bool:
    try:
        left_values = [float(value) for value in left]
        right_values = [float(value) for value in right]
    except (TypeError, ValueError):
        return False
    if len(left_values) != len(right_values):
        return False
    return all(abs(a - b) < 1e-9 for a, b in zip(left_values, right_values))


def find_existing_all_size_summary(
    results_root: str,
    dataset: str,
    variant: str,
    feature_set: str,
    base_learner: str,
    cache_size_percentages: Iterable[float],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    dataset_dir = os.path.join(results_root, dataset)
    if not os.path.isdir(dataset_dir):
        return None

    matches: List[Tuple[int, str, Dict[str, Any]]] = []
    for fname in os.listdir(dataset_dir):
        if not fname.endswith("_all_sizes.json"):
            continue
        path = os.path.join(dataset_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("dataset") != dataset:
            continue
        if data.get("variant") != variant:
            continue
        if data.get("feature_set") != feature_set:
            continue
        if str(data.get("base_learner")).lower() != str(base_learner).lower():
            continue
        if not same_cache_percentages(data.get("cache_size_percentages") or [], cache_size_percentages):
            continue
        prefix = fname.split("_", 1)[0]
        run_id = int(prefix) if len(prefix) == 3 and prefix.isdigit() else -1
        matches.append((run_id, path, data))

    if not matches:
        return None
    _, path, data = sorted(matches, key=lambda item: (item[0], item[1]))[-1]
    return path, data


def get_next_run_id(results_root: str, dataset: str, model: str, cache_size: int) -> Tuple[str, str]:
    dataset_dir = os.path.join(results_root, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    existing_ids: List[int] = []
    for fname in os.listdir(dataset_dir):
        if not (
            fname.endswith(f"_{model}_{cache_size}.jsonl")
            or fname.endswith(f"_summary_{model}_{cache_size}.json")
        ):
            continue
        prefix = fname.split("_", 1)[0]
        if len(prefix) == 3 and prefix.isdigit():
            existing_ids.append(int(prefix))
    next_id = 1 if not existing_ids else max(existing_ids) + 1
    return f"{next_id:03d}", dataset_dir


def get_next_group_id(results_root: str, dataset: str, model: str) -> Tuple[str, str]:
    dataset_dir = os.path.join(results_root, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    suffix = f"_summary_{model}_all_sizes.json"
    existing_ids: List[int] = []
    for fname in os.listdir(dataset_dir):
        if not fname.endswith(suffix):
            continue
        prefix = fname.split("_", 1)[0]
        if len(prefix) == 3 and prefix.isdigit():
            existing_ids.append(int(prefix))
    next_id = 1 if not existing_ids else max(existing_ids) + 1
    return f"{next_id:03d}", dataset_dir


def get_next_aggregate_id(results_root: str, label: str) -> Tuple[str, str]:
    aggregate_dir = os.path.join(results_root, "_aggregate")
    os.makedirs(aggregate_dir, exist_ok=True)
    suffix = f"_summary_{label}.json"
    existing_ids: List[int] = []
    for fname in os.listdir(aggregate_dir):
        if not fname.endswith(suffix):
            continue
        prefix = fname.split("_", 1)[0]
        if len(prefix) == 3 and prefix.isdigit():
            existing_ids.append(int(prefix))
    next_id = 1 if not existing_ids else max(existing_ids) + 1
    return f"{next_id:03d}", aggregate_dir


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=True)


def admission_summary_aliases(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "selected_admissions_total": metrics.get("total_selected_admissions"),
        "applied_admissions_total": metrics.get("total_applied_admissions"),
    }


def existing_summary_entry(path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    aggregate_diagnostic_summary = data.get("aggregate_diagnostic_summary") or {}
    return {
        "dataset": data.get("dataset"),
        "variant": data.get("variant"),
        "model": data.get("model"),
        "all_sizes_summary_path": path,
        "avg_hr": data.get("avg_hr"),
        "hr_by_capacity": {
            int(row["cache_size_objects"]): float(row["hit_ratio"])
            for row in data.get("hr_curve") or []
            if isinstance(row, dict)
            and row.get("cache_size_objects") is not None
            and row.get("hit_ratio") is not None
        },
        "config_summary": data.get("config_summary") or {},
        "aggregate_diagnostic_summary": aggregate_diagnostic_summary,
        "diagnostic_summary_by_cache_size": data.get("diagnostic_summary_by_cache_size") or {},
        "run_status": "skipped_existing",
        "skipped_existing": True,
        "total_selected_admissions": data.get("total_selected_admissions"),
        "selected_admissions_total": data.get("selected_admissions_total")
        or data.get("total_selected_admissions"),
        "total_applied_admissions": data.get("total_applied_admissions"),
        "applied_admissions_total": data.get("applied_admissions_total")
        or data.get("total_applied_admissions"),
        "admissions_per_1000_requests": data.get("admissions_per_1000_requests"),
        "aggregate_hit_yield_per_applied_admission": data.get("aggregate_hit_yield_per_applied_admission"),
        "aggregate_pollution_ratio": data.get("aggregate_pollution_ratio"),
        "aggregate_admission_precision": data.get("aggregate_admission_precision"),
    }


def apply_cli_budget_overrides(
    variant: str,
    overrides: Dict[str, Any],
    budget_match_multiplier: Optional[float],
    budget_match_target_multiplier: Optional[float],
    budget_replay_path: Optional[str],
    budget_replay_key: Optional[str],
    budget_replay_strict: Optional[bool],
    budget_replay_allow_clamp: Optional[bool],
    budget_replay_permute_seed: Optional[int],
) -> Dict[str, Any]:
    out = dict(overrides)
    if variant == "precision_only_budget_matched_uniform_cap020":
        if budget_match_multiplier is None:
            raise ValueError(
                "precision_only_budget_matched_uniform_cap020 requires "
                "--budget-match-multiplier from budget_match_ratios_cap020.csv."
            )
        out["BUDGET_MATCH_MULTIPLIER"] = float(budget_match_multiplier)
        out["BUDGET_MATCH_TARGET_MULTIPLIER"] = float(budget_match_multiplier)
    if (
        variant == "precision_only_budget_matched_random_cap020"
    ):
        if budget_match_target_multiplier is None:
            raise ValueError(
                "precision_only_budget_matched_random_cap020 requires "
                "--budget-match-target-multiplier from budget_match_ratios_cap020.csv."
            )
        out["BUDGET_MATCH_MULTIPLIER"] = float(budget_match_target_multiplier)
        out["BUDGET_MATCH_TARGET_MULTIPLIER"] = float(budget_match_target_multiplier)
    if variant in {
        "precision_only_budget_replay_cap020",
        "precision_only_budget_replay_permuted_cap020",
    }:
        if not budget_replay_path:
            raise ValueError(f"{variant} requires --budget-replay-path.")
        out["BUDGET_REPLAY_PATH"] = str(budget_replay_path)
        if budget_replay_key is not None:
            out["BUDGET_REPLAY_KEY"] = str(budget_replay_key)
        if budget_replay_strict is not None:
            out["BUDGET_REPLAY_STRICT"] = bool(budget_replay_strict)
        if budget_replay_allow_clamp is not None:
            out["BUDGET_REPLAY_ALLOW_CLAMP"] = bool(budget_replay_allow_clamp)
        if budget_replay_permute_seed is not None:
            out["BUDGET_REPLAY_PERMUTE_SEED"] = int(budget_replay_permute_seed)
    return out


def code_version() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = proc.stdout.strip()
    return value or None


def source_files_for_variant(spec: VariantSpec) -> List[str]:
    files = [
        "src/experiments/run_il_cache_guard_signal_v2_ablation.py",
        "src/experiments/run_il_cache_guard_signal_v2.py",
    ]
    if spec.module is il_no_guard_v2:
        files[1] = "src/experiments/run_il_cache_guard_signal_v2_no_guard.py"
    return files


def _is_number(value: Any) -> bool:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value_f)


def aggregate_diagnostics(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    values_by_key: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.items():
            if _is_number(value):
                values_by_key.setdefault(key, []).append(float(value))
    return {
        key: (sum(values) / len(values) if values else None)
        for key, values in sorted(values_by_key.items())
    }


def config_summary_for_variant(spec: VariantSpec, module: Any) -> Dict[str, Any]:
    return {
        "experiment_family": EXPERIMENT_FAMILY,
        "variant": spec.variant,
        "purpose": spec.purpose,
        "precision_feedback_enabled": bool(spec.precision_feedback_enabled),
        "score_quality_enabled": bool(spec.score_quality_enabled),
        "score_quality_modulation_enabled": bool(spec.score_quality_enabled),
        "PRECISION_CONTROL_MODE": getattr(module, "PRECISION_CONTROL_MODE"),
        "EFFECTIVE_PRECISION_LAG_MODE": getattr(module, "EFFECTIVE_PRECISION_LAG_MODE"),
        "EFFECTIVE_PRECISION_FIXED_LAG": getattr(module, "EFFECTIVE_PRECISION_FIXED_LAG"),
        "EFFECTIVE_PRECISION_LAG_WEIGHTS": list(getattr(module, "EFFECTIVE_PRECISION_LAG_WEIGHTS")),
        "ADMISSION_PRECISION_TARGET_MODE": getattr(module, "ADMISSION_PRECISION_TARGET_MODE"),
        "ADMISSION_PRECISION_TARGET": getattr(module, "ADMISSION_PRECISION_TARGET"),
        "ADMISSION_PRECISION_MARGIN": getattr(module, "ADMISSION_PRECISION_MARGIN"),
        "ADMISSION_PRECISION_POS_GAIN": getattr(module, "ADMISSION_PRECISION_POS_GAIN"),
        "ADMISSION_PRECISION_NEG_GAIN": getattr(module, "ADMISSION_PRECISION_NEG_GAIN"),
        "ADMISSION_PRECISION_MULT_MIN": getattr(module, "ADMISSION_PRECISION_MULT_MIN"),
        "ADMISSION_PRECISION_MULT_MAX": getattr(module, "ADMISSION_PRECISION_MULT_MAX"),
        "SCORE_QUALITY_SIGNAL_MODE": getattr(module, "SCORE_QUALITY_SIGNAL_MODE"),
        "SCORE_QUALITY_CONTROL_ROLE": getattr(module, "SCORE_QUALITY_CONTROL_ROLE"),
        "SCORE_QUALITY_CALIBRATION_ENABLED": getattr(module, "SCORE_QUALITY_CALIBRATION_ENABLED"),
        "SCORE_QUALITY_CALIBRATION_METRIC": getattr(module, "SCORE_QUALITY_CALIBRATION_METRIC"),
        "SCORE_QUALITY_TARGET_MODE": getattr(module, "SCORE_QUALITY_TARGET_MODE"),
        "SCORE_QUALITY_TARGET_MARGIN": getattr(module, "SCORE_QUALITY_TARGET_MARGIN"),
        "SCORE_QUALITY_MIN": getattr(module, "SCORE_QUALITY_MIN"),
        "SCORE_QUALITY_MAX": getattr(module, "SCORE_QUALITY_MAX"),
        "SCORE_QUALITY_MIN_BOOST": getattr(module, "SCORE_QUALITY_MIN_BOOST"),
        "BUDGET_MATCH_MODE": getattr(module, "BUDGET_MATCH_MODE", "off"),
        "BUDGET_MATCH_MULTIPLIER": getattr(module, "BUDGET_MATCH_MULTIPLIER", 1.0),
        "BUDGET_MATCH_TARGET_MULTIPLIER": getattr(
            module,
            "BUDGET_MATCH_TARGET_MULTIPLIER",
            1.0,
        ),
        "BUDGET_MATCH_RATIO_SOURCE": getattr(module, "BUDGET_MATCH_RATIO_SOURCE", None),
        "BUDGET_REPLAY_PATH": getattr(module, "BUDGET_REPLAY_PATH", None),
        "BUDGET_REPLAY_KEY": getattr(module, "BUDGET_REPLAY_KEY", "budget"),
        "BUDGET_REPLAY_STRICT": getattr(module, "BUDGET_REPLAY_STRICT", True),
        "BUDGET_REPLAY_ALLOW_CLAMP": getattr(module, "BUDGET_REPLAY_ALLOW_CLAMP", False),
        "BUDGET_REPLAY_PERMUTE_SEED": getattr(module, "BUDGET_REPLAY_PERMUTE_SEED", 42),
        "OFFLINE_DIAGNOSTIC_CONTROL": getattr(module, "OFFLINE_DIAGNOSTIC_CONTROL", False),
        "QUALITY_SUPPRESSED_UTILITY_WINDOW_SLOTS": getattr(
            module,
            "QUALITY_SUPPRESSED_UTILITY_WINDOW_SLOTS",
            1,
        ),
        "SCORE_GATE_TOP_PERCENT": getattr(module, "SCORE_GATE_TOP_PERCENT"),
        "QUALITY_CONTROL_MODE": getattr(module, "QUALITY_CONTROL_MODE"),
        "PRECISION_CONTROL_OVERRIDE_MODE": getattr(module, "PRECISION_CONTROL_OVERRIDE_MODE"),
        "FILL_RATIO": getattr(module, "FILL_RATIO"),
        "FILL_RATE": getattr(module, "FILL_RATE"),
        "PRESSURE_MISS_GAMMA": getattr(module, "PRESSURE_MISS_GAMMA"),
        "DRIFT_GAIN": getattr(module, "DRIFT_GAIN"),
        "DRIFT_ALPHA_FLOOR_MULT": getattr(module, "DRIFT_ALPHA_FLOOR_MULT"),
        "DRIFT_ALPHA_MIN": getattr(module, "DRIFT_ALPHA_MIN"),
        "DRIFT_ALPHA_MAX": getattr(module, "DRIFT_ALPHA_MAX"),
        "DRIFT_USE_CAPACITY_SCALE": getattr(module, "DRIFT_USE_CAPACITY_SCALE", True),
        "ADMISSION_CAPACITY_ALPHA": getattr(module, "ADMISSION_CAPACITY_ALPHA"),
        "CONTROL_RANDOM_SEED": getattr(module, "CONTROL_RANDOM_SEED"),
    }


def print_config(
    ds: DatasetSpec,
    variant: str,
    model_name: str,
    total_requests: int,
    warmup_requests: int,
    slot_size: int,
    feature_set: str,
    base_learner: str,
    capacities: List[Dict[str, float]],
    cfg_summary: Dict[str, Any],
) -> None:
    print(f"=== IL Guardrails Signal v2: dataset={ds.name}, variant={variant}, model={model_name} ===")
    print(f"Trace path        : {ds.path}")
    print(f"Total requests    : {total_requests}")
    print(f"Warm-up requests  : {warmup_requests}")
    print(f"Slot size         : {slot_size}")
    print(f"Feature set       : {feature_set}")
    print(f"Base learner      : {base_learner}")
    print(f"Precision enabled : {cfg_summary['precision_feedback_enabled']}")
    print(f"Quality enabled   : {cfg_summary['score_quality_modulation_enabled']}")
    print(f"Precision mode    : {cfg_summary['PRECISION_CONTROL_MODE']}")
    print(f"Quality mode      : {cfg_summary['SCORE_QUALITY_SIGNAL_MODE']}")
    print(f"Top cap percent   : {cfg_summary['SCORE_GATE_TOP_PERCENT']}")
    print(f"Protocol          : one_slot_delayed_admission")
    print(f"Eviction policy   : fixed_lru")
    print(f"Cache capacities  : {[row['cache_size_objects'] for row in capacities]}")


def runtime_for_dataset(
    ds: DatasetSpec,
    max_requests: Optional[int],
    smoke_test: bool,
) -> Tuple[int, int, int]:
    slot_size = ds.slot_size
    total_requests = ds.total_requests if max_requests is None else min(int(max_requests), ds.total_requests)
    warmup_requests = ds.warmup_requests
    if smoke_test:
        total_requests = min(total_requests, 300_000)
        slot_size = min(slot_size, 100_000)
        if total_requests <= slot_size:
            total_requests = slot_size * 3
        warmup_requests = slot_size
    if total_requests <= warmup_requests:
        warmup_requests = max(slot_size, (total_requests // slot_size - 1) * slot_size)
    if warmup_requests <= 0 or warmup_requests >= total_requests:
        raise ValueError("Unable to choose a valid warmup for the requested runtime.")
    if warmup_requests % slot_size != 0:
        raise ValueError("warmup_requests must be a multiple of slot_size.")
    if total_requests <= warmup_requests:
        raise ValueError("total_requests must be greater than warmup_requests.")
    return total_requests, warmup_requests, slot_size


def run_dataset_variant(
    ds: DatasetSpec,
    variant: str,
    feature_set: str,
    base_learner: str,
    results_root: str,
    cache_size_percentages: Tuple[float, ...],
    max_requests: Optional[int],
    disable_progress: bool,
    seed: int,
    smoke_test: bool,
    code_hash: Optional[str],
    skip_existing: bool,
    budget_match_multiplier: Optional[float],
    budget_match_target_multiplier: Optional[float],
    budget_replay_path: Optional[str],
    budget_replay_key: Optional[str],
    budget_replay_strict: Optional[bool],
    budget_replay_allow_clamp: Optional[bool],
    budget_replay_permute_seed: Optional[int],
) -> Dict[str, Any]:
    spec = get_variant_spec(variant)
    module = spec.module
    model_name = model_name_for_variant(variant, feature_set, base_learner)
    total_requests, warmup_requests, slot_size = runtime_for_dataset(ds, max_requests, smoke_test)

    percentages = cache_size_percentages
    if smoke_test and len(percentages) > 2:
        percentages = percentages[:2]

    if skip_existing:
        existing = find_existing_all_size_summary(
            results_root=results_root,
            dataset=ds.name,
            variant=variant,
            feature_set=feature_set,
            base_learner=base_learner,
            cache_size_percentages=percentages,
        )
        if existing is not None:
            path, data = existing
            print(f"[SKIP] {ds.name} {variant} all-size summary already exists -> {path}")
            return existing_summary_entry(path, data)

    capacities, _ = compute_capacities(ds.path, total_requests, percentages)
    overrides = apply_cli_budget_overrides(
        variant,
        spec.overrides,
        budget_match_multiplier=budget_match_multiplier,
        budget_match_target_multiplier=budget_match_target_multiplier,
        budget_replay_path=budget_replay_path,
        budget_replay_key=budget_replay_key,
        budget_replay_strict=budget_replay_strict,
        budget_replay_allow_clamp=budget_replay_allow_clamp,
        budget_replay_permute_seed=budget_replay_permute_seed,
    )
    overrides["CONTROL_RANDOM_SEED"] = int(seed)

    capacity_results: List[Dict[str, Any]] = []
    diagnostic_summary_by_cache_size: Dict[str, Dict[str, Any]] = {}
    with patched_module(module, overrides, disable_progress=disable_progress):
        cfg_summary = config_summary_for_variant(spec, module)
        print_config(
            ds,
            variant,
            model_name,
            total_requests,
            warmup_requests,
            slot_size,
            feature_set,
            base_learner,
            capacities,
            cfg_summary,
        )
        for cap_info in capacities:
            capacity = int(cap_info["cache_size_objects"])
            cap_percent = float(cap_info["capacity_percent"])
            run_id, dataset_dir = get_next_run_id(results_root, ds.name, model_name, capacity)
            slot_log_path = os.path.join(dataset_dir, f"{run_id}_{model_name}_{capacity}.jsonl")
            summary_path = os.path.join(dataset_dir, f"{run_id}_summary_{model_name}_{capacity}.json")

            stats, summary_metrics = module.run_single_capacity(
                trace_path=ds.path,
                total_requests=total_requests,
                warmup_requests=warmup_requests,
                slot_size=slot_size,
                capacity_objects=capacity,
                feature_set=feature_set,
                base_learner=base_learner,
                slot_log_path=slot_log_path,
            )
            summary_metrics = dict(summary_metrics)
            summary_metrics.update(admission_summary_aliases(summary_metrics))
            slots_processed = int(summary_metrics.get("slots_processed", 0))
            warmup_slots = warmup_requests // slot_size
            diagnostic_summary = summary_metrics.get("diagnostic_summary") or {}
            diagnostic_summary_by_cache_size[str(capacity)] = diagnostic_summary
            summary_payload: Dict[str, Any] = {
                "dataset": ds.name,
                "model": model_name,
                "variant": variant,
                "experiment_family": EXPERIMENT_FAMILY,
                "feature_set": feature_set,
                "base_learner": base_learner,
                "trace_path": ds.path,
                "cache_size_objects": capacity,
                "capacity_percent": cap_percent,
                "total_requests": total_requests,
                "warmup_requests": warmup_requests,
                "slot_size": slot_size,
                "slot_log_path": slot_log_path,
                "hit_ratio": stats.hit_ratio,
                "cache_hits": stats.cache_hits,
                "cache_requests": stats.total_requests,
                "num_slots": slots_processed,
                "warmup_slots": warmup_slots,
                "cache_slots": max(0, slots_processed - warmup_slots),
                "protocol": "one_slot_delayed_admission",
                "eviction_policy": "fixed_lru",
                "fixed_lru_eviction": True,
                "one_slot_delayed_apply": True,
                "precision_feedback_enabled": bool(spec.precision_feedback_enabled),
                "score_quality_enabled": bool(spec.score_quality_enabled),
                "final_score_gate_top_percent": getattr(module, "SCORE_GATE_TOP_PERCENT"),
                "budget_match_mode": getattr(module, "BUDGET_MATCH_MODE", "off"),
                "budget_match_multiplier": getattr(module, "BUDGET_MATCH_MULTIPLIER", 1.0),
                "budget_match_target_multiplier": getattr(
                    module,
                    "BUDGET_MATCH_TARGET_MULTIPLIER",
                    1.0,
                ),
                "budget_match_ratio_source": getattr(module, "BUDGET_MATCH_RATIO_SOURCE", None),
                "budget_replay_enabled": getattr(module, "BUDGET_MATCH_MODE", "off")
                in {"slot_replay", "slot_replay_permuted"},
                "budget_replay_permuted": getattr(module, "BUDGET_MATCH_MODE", "off")
                == "slot_replay_permuted",
                "offline_diagnostic_control": bool(
                    getattr(module, "OFFLINE_DIAGNOSTIC_CONTROL", False)
                ),
                "config_summary": cfg_summary,
                "diagnostic_summary": diagnostic_summary,
                "source_files": source_files_for_variant(spec),
                "code_version": code_hash,
                "reproducibility_seed": seed,
            }
            summary_payload.update(summary_metrics)
            save_json(summary_path, summary_payload)

            capacity_results.append(
                {
                    "cache_size_objects": capacity,
                    "capacity_percent": cap_percent,
                    "hit_ratio": stats.hit_ratio,
                    "cache_hits": stats.cache_hits,
                    "cache_requests": stats.total_requests,
                    "summary_path": summary_path,
                    "slot_log_path": slot_log_path,
                    "diagnostic_summary": diagnostic_summary,
                    **{
                        key: summary_metrics.get(key)
                        for key in SUMMARY_AGGREGATE_KEYS
                    },
                }
            )
            print(f"[RUN] {ds.name} {variant} cap={capacity} HR={stats.hit_ratio:.6f} -> {summary_path}")

    hr_curve = [
        {"cache_size_objects": row["cache_size_objects"], "hit_ratio": row["hit_ratio"]}
        for row in capacity_results
    ]
    avg_hr = (
        sum(float(row["hit_ratio"]) for row in capacity_results) / len(capacity_results)
        if capacity_results
        else None
    )
    aggregate_diagnostic_summary = aggregate_diagnostics(diagnostic_summary_by_cache_size.values())
    aggregate_result_summary = aggregate_diagnostics(
        [
            {
                key: row.get(key)
                for key in SUMMARY_AGGREGATE_KEYS
            }
            for row in capacity_results
        ]
    )
    group_id, dataset_dir = get_next_group_id(results_root, ds.name, model_name)
    all_sizes_path = os.path.join(dataset_dir, f"{group_id}_summary_{model_name}_all_sizes.json")
    all_sizes_payload = {
        "dataset": ds.name,
        "model": model_name,
        "variant": variant,
        "experiment_family": EXPERIMENT_FAMILY,
        "feature_set": feature_set,
        "base_learner": base_learner,
        "cache_sizes": [row["cache_size_objects"] for row in capacity_results],
        "cache_size_percentages": [row["capacity_percent"] for row in capacity_results],
        "hr_curve": hr_curve,
        "avg_hr": avg_hr,
        "config_summary": cfg_summary,
        "diagnostic_summary_by_cache_size": diagnostic_summary_by_cache_size,
        "aggregate_diagnostic_summary": aggregate_diagnostic_summary,
        "results": capacity_results,
        "source_files": source_files_for_variant(spec),
        "code_version": code_hash,
    }
    all_sizes_payload.update(aggregate_result_summary)
    save_json(all_sizes_path, all_sizes_payload)
    print(f"[SUMMARY] {all_sizes_path}")

    return {
        "dataset": ds.name,
        "variant": variant,
        "model": model_name,
        "all_sizes_summary_path": all_sizes_path,
        "avg_hr": avg_hr,
        "hr_by_capacity": {
            int(row["cache_size_objects"]): float(row["hit_ratio"])
            for row in capacity_results
        },
        "config_summary": cfg_summary,
        "aggregate_diagnostic_summary": aggregate_diagnostic_summary,
        "diagnostic_summary_by_cache_size": diagnostic_summary_by_cache_size,
        "run_status": "ran",
        "skipped_existing": False,
        **aggregate_result_summary,
    }


def write_master_summary(
    results_root: str,
    entries: List[Dict[str, Any]],
    datasets: List[DatasetSpec],
    variants: List[str],
    feature_set: str,
    base_learner: str,
    variant_set: str,
    code_hash: Optional[str],
) -> Optional[str]:
    if not entries:
        return None
    label = f"{EXPERIMENT_FAMILY}_{variant_set}"
    run_id, aggregate_dir = get_next_aggregate_id(results_root, label)
    path = os.path.join(aggregate_dir, f"{run_id}_summary_{label}.json")
    save_json(
        path,
        {
            "experiment_family": EXPERIMENT_FAMILY,
            "variant_set": variant_set,
            "datasets_executed": [ds.name for ds in datasets],
            "variants_requested": variants,
            "variants_executed": sorted(
                {
                    str(entry.get("variant"))
                    for entry in entries
                    if entry.get("run_status") == "ran" and entry.get("variant")
                }
            ),
            "variants_skipped_existing": sorted(
                {
                    str(entry.get("variant"))
                    for entry in entries
                    if entry.get("run_status") == "skipped_existing" and entry.get("variant")
                }
            ),
            "feature_set": feature_set,
            "base_learner": base_learner,
            "code_version": code_hash,
            "results": entries,
        },
    )
    print(f"[MASTER] {path}")
    return path


def run_experiment(args: argparse.Namespace) -> List[Dict[str, Any]]:
    datasets = resolve_datasets(args.dataset)
    variants = resolve_variants(args.variant, args.variant_set)
    percentages = parse_cache_size_percentages(args.cache_size_percentages)
    code_hash = code_version()

    entries: List[Dict[str, Any]] = []
    for ds in datasets:
        for variant in variants:
            entries.append(
                run_dataset_variant(
                    ds=ds,
                    variant=variant,
                    feature_set=args.feature_set,
                    base_learner=args.base_learner,
                    results_root=args.results_root,
                    cache_size_percentages=percentages,
                    max_requests=args.max_requests,
                    disable_progress=args.disable_progress,
                    seed=args.seed,
                    smoke_test=args.smoke_test,
                    code_hash=code_hash,
                    skip_existing=args.skip_existing,
                    budget_match_multiplier=args.budget_match_multiplier,
                    budget_match_target_multiplier=args.budget_match_target_multiplier,
                    budget_replay_path=args.budget_replay_path,
                    budget_replay_key=args.budget_replay_key,
                    budget_replay_strict=args.budget_replay_strict,
                    budget_replay_allow_clamp=args.budget_replay_allow_clamp,
                    budget_replay_permute_seed=args.budget_replay_permute_seed,
                )
            )
    skipped = [
        f"{entry.get('dataset')}/{entry.get('variant')}"
        for entry in entries
        if entry.get("run_status") == "skipped_existing"
    ]
    ran = [
        f"{entry.get('dataset')}/{entry.get('variant')}"
        for entry in entries
        if entry.get("run_status") == "ran"
    ]
    if skipped:
        print(f"[SKIPPED EXISTING] {', '.join(skipped)}")
    if ran:
        print(f"[RAN] {', '.join(ran)}")
    write_master_summary(
        args.results_root,
        entries,
        datasets,
        variants,
        args.feature_set,
        args.base_learner,
        args.variant_set,
        code_hash,
    )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Guardrails Admission signal v2 ablations.")
    parser.add_argument(
        "--dataset",
        choices=["wikipedia_september_2007", "wiki2018", "all"],
        default="all",
    )
    parser.add_argument("--variant", choices=list(ALL_VARIANTS) + ["all"], nargs="+", default=None)
    parser.add_argument(
        "--variant-set",
        choices=sorted(VARIANT_SETS.keys()),
        default="core_clean",
    )
    parser.add_argument("--feature-set", choices=["A0", "A1", "A2", "A3"], default="A2")
    parser.add_argument("--base-learner", choices=["nb", "svm", "dt"], default="nb")
    parser.add_argument("--cache-size-percentages", default="0.8,1.0,2.0,3.0,4.0,5.0")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--budget-match-multiplier", type=float, default=None)
    parser.add_argument("--budget-match-target-multiplier", type=float, default=None)
    parser.add_argument("--budget-replay-path", default=None)
    parser.add_argument("--budget-replay-key", default=None)
    parser.add_argument("--budget-replay-strict", action="store_true", default=None)
    parser.add_argument("--budget-replay-allow-clamp", action="store_true", default=None)
    parser.add_argument("--budget-replay-permute-seed", type=int, default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.max_requests is not None and args.max_requests < 1:
        raise ValueError("--max-requests must be >= 1.")
    if args.budget_match_multiplier is not None and args.budget_match_multiplier < 0.0:
        raise ValueError("--budget-match-multiplier must be >= 0.")
    if args.budget_match_target_multiplier is not None and args.budget_match_target_multiplier < 0.0:
        raise ValueError("--budget-match-target-multiplier must be >= 0.")
    if args.results_root == DEFAULT_RESULTS_ROOT and args.variant_set in {
        "core_clean_cap020",
        "score_quality_budget_matched_cap020",
        "quality_mode_cap020",
        "cap020_validation",
    }:
        args.results_root = DEFAULT_CAP020_RESULTS_ROOT
    if args.smoke_test:
        args.disable_progress = True
        if args.max_requests is None:
            args.max_requests = 300_000

    run_experiment(args)


if __name__ == "__main__":
    main()
