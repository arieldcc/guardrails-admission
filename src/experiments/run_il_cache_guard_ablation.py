from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.trace_reader import TraceReader
from src.experiments import run_il_cache_guard_no_guard as il_no_guard
from src.experiments import run_il_cache_guard_only as il_full


CORE_VARIANTS: Tuple[str, ...] = ("no-guard", "precision-only", "quality-only", "full")
ALTERNATIVE_VARIANTS: Tuple[str, ...] = (
    "full_quality_safety_only",
    "full_quality_controls_cap",
    "full_quality_controls_cap_conservative",
    "full_quality_controls_cap_balanced",
    "full_boundary_quality_controls_cap",
    "full_quality_conditional",
    "full_quality_penalty_on_ambiguity",
)
VARIANTS: Tuple[str, ...] = CORE_VARIANTS + ALTERNATIVE_VARIANTS
DATASETS: Tuple[str, ...] = (
    "wikipedia_september_2007",
    # "wikipedia_oktober_2007",
    "wiki2018",
)
CACHE_SIZE_PERCENTAGES: Tuple[float, ...] = (0.8, 1.0, 2.0, 3.0, 4.0, 5.0)


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
    model_guard_name: str
    precision_feedback_enabled: bool
    score_quality_enabled: bool
    overrides: Dict[str, Any]


def resolve_dataset(name: str) -> DatasetSpec:
    source = {
        "wikipedia_september_2007": il_full.WIKIPEDIA_SEPTEMBER_2007,
        # "wikipedia_oktober_2007": il_full.WIKIPEDIA_OKTOBER_2007,
        "wiki2018": il_full.WIKI2018,
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


def resolve_variants(variant_arg: Any) -> List[str]:
    if isinstance(variant_arg, str):
        requested = [variant_arg]
    else:
        requested = list(variant_arg)
    if "all" in requested:
        return list(CORE_VARIANTS)
    variants: List[str] = []
    for variant in requested:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        variants.append(variant)
    return variants


def learner_label(base_learner: str) -> str:
    return str(base_learner).upper()


def model_name_for_variant(variant: str, feature_set: str, base_learner: str) -> str:
    guard_name = {
        "no-guard": "guard_no_guard",
        "precision-only": "guard_precision_only",
        "quality-only": "guard_quality_only",
        "full": "guard_full",
        "full_quality_safety_only": "guard_full_quality_safety_only",
        "full_quality_controls_cap": "guard_full_quality_controls_cap",
        "full_quality_controls_cap_conservative": "guard_full_quality_controls_cap_conservative",
        "full_quality_controls_cap_balanced": "guard_full_quality_controls_cap_balanced",
        "full_boundary_quality_controls_cap": "guard_full_boundary_quality_controls_cap",
        "full_quality_conditional": "guard_full_quality_conditional",
        "full_quality_penalty_on_ambiguity": "guard_full_quality_penalty_on_ambiguity",
    }[variant]
    return f"ilnse_{feature_set}_{guard_name}_{learner_label(base_learner)}"


def get_variant_spec(variant: str) -> VariantSpec:
    if variant == "no-guard":
        return VariantSpec(
            variant=variant,
            module=il_no_guard,
            model_guard_name="guard_no_guard",
            precision_feedback_enabled=False,
            score_quality_enabled=False,
            overrides={},
        )
    if variant == "full":
        return VariantSpec(
            variant=variant,
            module=il_full,
            model_guard_name="guard_full",
            precision_feedback_enabled=True,
            score_quality_enabled=True,
            overrides={},
        )
    if variant == "precision-only":
        return VariantSpec(
            variant=variant,
            module=il_full,
            model_guard_name="guard_precision_only",
            precision_feedback_enabled=True,
            score_quality_enabled=False,
            overrides={
                "ADMISSION_PRECISION_TARGET": 0.12,
                "ADMISSION_PRECISION_SENSITIVITY": 1.0,
                "DRIFT_ALPHA_FLOOR_MULT": 0.7,
                "DRIFT_USE_CAPACITY_SCALE": True,
                "SCORE_GATE_TOP_PERCENT": 0.05,
                "FILL_RATIO": 0.9,
                "FILL_RATE": 0.05,
                "PRESSURE_MISS_GAMMA": 0.5,
                "SCORE_QUALITY_MIN": 1.0,
                "SCORE_QUALITY_MAX": 1.0,
                "SCORE_QUALITY_MIN_BOOST": 0.0,
            },
        )
    if variant == "quality-only":
        return VariantSpec(
            variant=variant,
            module=il_full,
            model_guard_name="guard_quality_only",
            precision_feedback_enabled=False,
            score_quality_enabled=True,
            overrides={
                "ADMISSION_PRECISION_TARGET": 0.12,
                "ADMISSION_PRECISION_SENSITIVITY": 0.0,
                "DRIFT_ALPHA_FLOOR_MULT": 0.7,
                "DRIFT_USE_CAPACITY_SCALE": True,
                "SCORE_GATE_TOP_PERCENT": 0.05,
                "FILL_RATIO": 0.9,
                "FILL_RATE": 0.05,
                "PRESSURE_MISS_GAMMA": 0.5,
                "SCORE_SPREAD_Q": 0.9,
                "SCORE_SPREAD_EMA_ALPHA": 0.2,
                "SCORE_SPREAD_EPS": 1e-6,
                "SCORE_QUALITY_MIN": 0.7,
                "SCORE_QUALITY_MAX": 1.3,
                "SCORE_QUALITY_MIN_BOOST": 0.1,
            },
        )
    if variant == "full_quality_safety_only":
        return VariantSpec(
            variant=variant,
            module=il_full,
            model_guard_name="guard_full_quality_safety_only",
            precision_feedback_enabled=True,
            score_quality_enabled=True,
            overrides={
                "ADMISSION_PRECISION_TARGET": 0.12,
                "ADMISSION_PRECISION_SENSITIVITY": 1.0,
                "DRIFT_ALPHA_FLOOR_MULT": 0.7,
                "DRIFT_USE_CAPACITY_SCALE": True,
                "SCORE_GATE_TOP_PERCENT": 0.05,
                "FILL_RATIO": 0.9,
                "FILL_RATE": 0.05,
                "PRESSURE_MISS_GAMMA": 0.5,
                "SCORE_QUALITY_MIN": 0.7,
                "SCORE_QUALITY_MAX": 1.0,
                "SCORE_QUALITY_MIN_BOOST": 0.0,
                "GUARD_QUALITY_INTERACTION_MODE": "budget_multiplier",
            },
        )
    if variant == "full_quality_controls_cap":
        variant = "full_quality_controls_cap_conservative"
    if variant == "full_quality_controls_cap_conservative":
        return VariantSpec(
            variant=variant,
            module=il_full,
            model_guard_name="guard_full_quality_controls_cap_conservative",
            precision_feedback_enabled=True,
            score_quality_enabled=True,
            overrides={
                "ADMISSION_PRECISION_TARGET": 0.12,
                "ADMISSION_PRECISION_SENSITIVITY": 1.0,
                "DRIFT_ALPHA_FLOOR_MULT": 0.7,
                "DRIFT_USE_CAPACITY_SCALE": True,
                "SCORE_GATE_TOP_PERCENT": 0.05,
                "FILL_RATIO": 0.9,
                "FILL_RATE": 0.05,
                "PRESSURE_MISS_GAMMA": 0.5,
                "SCORE_QUALITY_MIN": 0.7,
                "SCORE_QUALITY_MAX": 1.3,
                "SCORE_QUALITY_MIN_BOOST": 0.1,
                "GUARD_QUALITY_INTERACTION_MODE": "cap_modulation",
                "QUALITY_CAP_MIN": 0.025,
                "QUALITY_CAP_MAX": 0.05,
            },
        )
    if variant == "full_quality_controls_cap_balanced":
        return VariantSpec(
            variant=variant,
            module=il_full,
            model_guard_name="guard_full_quality_controls_cap_balanced",
            precision_feedback_enabled=True,
            score_quality_enabled=True,
            overrides={
                "ADMISSION_PRECISION_TARGET": 0.12,
                "ADMISSION_PRECISION_SENSITIVITY": 1.0,
                "DRIFT_ALPHA_FLOOR_MULT": 0.7,
                "DRIFT_USE_CAPACITY_SCALE": True,
                "SCORE_GATE_TOP_PERCENT": 0.05,
                "FILL_RATIO": 0.9,
                "FILL_RATE": 0.05,
                "PRESSURE_MISS_GAMMA": 0.5,
                "SCORE_QUALITY_MIN": 0.7,
                "SCORE_QUALITY_MAX": 1.3,
                "SCORE_QUALITY_MIN_BOOST": 0.1,
                "GUARD_QUALITY_INTERACTION_MODE": "cap_modulation",
                "QUALITY_CAP_MIN": 0.025,
                "QUALITY_CAP_MAX": 0.075,
            },
        )
    if variant == "full_boundary_quality_controls_cap":
        return VariantSpec(
            variant=variant,
            module=il_full,
            model_guard_name="guard_full_boundary_quality_controls_cap",
            precision_feedback_enabled=True,
            score_quality_enabled=True,
            overrides={
                "ADMISSION_PRECISION_TARGET": 0.12,
                "ADMISSION_PRECISION_SENSITIVITY": 1.0,
                "DRIFT_ALPHA_FLOOR_MULT": 0.7,
                "DRIFT_USE_CAPACITY_SCALE": True,
                "SCORE_GATE_TOP_PERCENT": 0.05,
                "FILL_RATIO": 0.9,
                "FILL_RATE": 0.05,
                "PRESSURE_MISS_GAMMA": 0.5,
                "SCORE_QUALITY_MIN": 0.7,
                "SCORE_QUALITY_MAX": 1.3,
                "SCORE_QUALITY_MIN_BOOST": 0.1,
                "GUARD_QUALITY_INTERACTION_MODE": "boundary_cap_modulation",
                "QUALITY_CAP_MIN": 0.025,
                "QUALITY_CAP_MAX": 0.075,
            },
        )
    if variant == "full_quality_conditional":
        return VariantSpec(
            variant=variant,
            module=il_full,
            model_guard_name="guard_full_quality_conditional",
            precision_feedback_enabled=True,
            score_quality_enabled=True,
            overrides={
                "ADMISSION_PRECISION_TARGET": 0.12,
                "ADMISSION_PRECISION_SENSITIVITY": 1.0,
                "DRIFT_ALPHA_FLOOR_MULT": 0.7,
                "DRIFT_USE_CAPACITY_SCALE": True,
                "SCORE_GATE_TOP_PERCENT": 0.05,
                "FILL_RATIO": 0.9,
                "FILL_RATE": 0.05,
                "PRESSURE_MISS_GAMMA": 0.5,
                "SCORE_QUALITY_MIN": 0.7,
                "SCORE_QUALITY_MAX": 1.3,
                "SCORE_QUALITY_MIN_BOOST": 0.1,
                "GUARD_QUALITY_INTERACTION_MODE": "conditional",
            },
        )
    if variant == "full_quality_penalty_on_ambiguity":
        return VariantSpec(
            variant=variant,
            module=il_full,
            model_guard_name="guard_full_quality_penalty_on_ambiguity",
            precision_feedback_enabled=True,
            score_quality_enabled=True,
            overrides={
                "ADMISSION_PRECISION_TARGET": 0.12,
                "ADMISSION_PRECISION_SENSITIVITY": 1.0,
                "DRIFT_ALPHA_FLOOR_MULT": 0.7,
                "DRIFT_USE_CAPACITY_SCALE": True,
                "SCORE_GATE_TOP_PERCENT": 0.05,
                "FILL_RATIO": 0.9,
                "FILL_RATE": 0.05,
                "PRESSURE_MISS_GAMMA": 0.5,
                "SCORE_QUALITY_MIN": 0.7,
                "SCORE_QUALITY_MAX": 1.0,
                "SCORE_QUALITY_MIN_BOOST": 0.0,
                "GUARD_QUALITY_INTERACTION_MODE": "budget_multiplier",
            },
        )
    raise ValueError(f"Unknown variant: {variant}")


@contextmanager
def patched_module(module: Any, overrides: Dict[str, Any], disable_progress: bool):
    saved: Dict[str, Any] = {}
    for name, value in overrides.items():
        saved[name] = getattr(module, name)
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
            setattr(module, name, value)


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
    capacity_percent: Optional[float],
    capacity_objects: Optional[int],
    num_unique_objects: Optional[int] = None,
) -> Tuple[List[Dict[str, Optional[float]]], Optional[int]]:
    if capacity_objects is not None:
        if capacity_objects < 1:
            raise ValueError("capacity_objects must be >= 1.")
        return [{"cache_size_objects": int(capacity_objects), "capacity_percent": None}], num_unique_objects

    if capacity_percent is not None:
        if capacity_percent <= 0:
            raise ValueError("capacity_percent must be > 0.")
        if num_unique_objects is None:
            num_unique_objects = count_distinct_objects(trace_path, total_requests)
        cap = max(1, int(num_unique_objects * float(capacity_percent) / 100.0))
        return [{"cache_size_objects": cap, "capacity_percent": float(capacity_percent)}], num_unique_objects

    if num_unique_objects is None:
        num_unique_objects = count_distinct_objects(trace_path, total_requests)
    capacities = [
        {
            "cache_size_objects": max(1, int(num_unique_objects * float(percent) / 100.0)),
            "capacity_percent": float(percent),
        }
        for percent in CACHE_SIZE_PERCENTAGES
    ]
    return capacities, num_unique_objects


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
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")


def validate_runtime(total_requests: int, warmup_requests: int, slot_size: int) -> None:
    if slot_size <= 0:
        raise ValueError("slot_size must be > 0.")
    if total_requests <= warmup_requests:
        raise ValueError("total_requests must be greater than warmup_requests.")
    if warmup_requests % slot_size != 0:
        raise ValueError("warmup_requests must be a multiple of slot_size.")


def config_summary_for_variant(spec: VariantSpec, module: Any) -> Dict[str, Any]:
    precision_enabled = bool(spec.precision_feedback_enabled)
    quality_enabled = bool(spec.score_quality_enabled)
    return {
        "precision_feedback_enabled": precision_enabled,
        "score_quality_modulation_enabled": quality_enabled,
        "ADMISSION_PRECISION_TARGET": getattr(module, "ADMISSION_PRECISION_TARGET"),
        "ADMISSION_PRECISION_SENSITIVITY": getattr(module, "ADMISSION_PRECISION_SENSITIVITY"),
        "ADMISSION_ALPHA_FLOOR_MULT": getattr(module, "DRIFT_ALPHA_FLOOR_MULT"),
        "ADMISSION_USE_CAPACITY_SCALE": getattr(module, "DRIFT_USE_CAPACITY_SCALE", True),
        "SCORE_GATE_TOP_PERCENT": getattr(module, "SCORE_GATE_TOP_PERCENT"),
        "SCORE_SPREAD_Q": getattr(module, "SCORE_SPREAD_Q"),
        "SCORE_SPREAD_EMA_ALPHA": getattr(module, "SCORE_SPREAD_EMA_ALPHA"),
        "SCORE_SPREAD_EPS": getattr(module, "SCORE_SPREAD_EPS"),
        "SCORE_QUALITY_MIN": getattr(module, "SCORE_QUALITY_MIN"),
        "SCORE_QUALITY_MAX": getattr(module, "SCORE_QUALITY_MAX"),
        "SCORE_QUALITY_MIN_BOOST": getattr(module, "SCORE_QUALITY_MIN_BOOST"),
        "FILL_RATIO": getattr(module, "FILL_RATIO"),
        "FILL_RATE": getattr(module, "FILL_RATE"),
        "PRESSURE_MISS_GAMMA": getattr(module, "PRESSURE_MISS_GAMMA"),
        "GUARD_QUALITY_INTERACTION_MODE": getattr(module, "GUARD_QUALITY_INTERACTION_MODE", "budget_multiplier"),
        "QUALITY_CAP_MIN": getattr(module, "QUALITY_CAP_MIN", None),
        "QUALITY_CAP_MAX": getattr(module, "QUALITY_CAP_MAX", None),
        "DRIFT_GAIN": getattr(module, "DRIFT_GAIN"),
        "DRIFT_ALPHA_MIN": getattr(module, "DRIFT_ALPHA_MIN"),
        "DRIFT_ALPHA_MAX": getattr(module, "DRIFT_ALPHA_MAX"),
    }


def print_paper_config(
    *,
    ds: DatasetSpec,
    variant: str,
    model_name: str,
    trace_path: str,
    total_requests: int,
    warmup_requests: int,
    slot_size: int,
    feature_set: str,
    base_learner: str,
    capacities: List[Dict[str, Optional[float]]],
    num_features: int,
    module: Any,
    cfg_summary: Dict[str, Any],
) -> None:
    print(f"=== IL Guard Ablation: dataset={ds.name}, variant={variant}, model={model_name} ===")
    print(f"Trace path        : {trace_path}")
    print(f"Total requests    : {total_requests}")
    print(f"Warm-up requests  : {warmup_requests}")
    print(f"Slot size         : {slot_size}")
    print(f"IL num_gaps       : {module.IL_NUM_GAPS}")
    # print(f"Feature set       : {feature_set}")
    print(f"Num features      : {num_features}")
    print(f"Base learner      : {base_learner}")
    print(f"Variant           : {variant}")
    print(f"Precision feedback: {cfg_summary['precision_feedback_enabled']}")
    print(f"Score quality     : {cfg_summary['score_quality_modulation_enabled']}")
    print(f"IL top_percent    : {module.IL_POP_TOP_PERCENT}")
    print(f"Label rounding    : {module.IL_LABEL_TOPK_ROUNDING}")
    print(f"Label tie-break   : {module.IL_LABEL_TIE_BREAK}")
    print(f"IL sigmoid (a, b) : ({module.IL_SIGMOID_A}, {module.IL_SIGMOID_B})")
    print(f"Max learners      : {module.IL_MAX_CLASSIFIERS}")
    print("Eviction policy   : fixed_lru")
    print("Protocol          : one_slot_delayed_admission")
    print("Admission policy  : top_capacity_rate")
    print(f"Admission alpha   : {module.ADMISSION_CAPACITY_ALPHA}")
    print(f"Use cap scale     : {cfg_summary['ADMISSION_USE_CAPACITY_SCALE']}")
    print(f"Alpha floor mult  : {cfg_summary['ADMISSION_ALPHA_FLOOR_MULT']}")
    print(f"Alpha range       : {cfg_summary['DRIFT_ALPHA_MIN']}-{cfg_summary['DRIFT_ALPHA_MAX']}")
    print(f"Fill ratio/rate   : {cfg_summary['FILL_RATIO']} / {cfg_summary['FILL_RATE']}")
    print(f"Pressure gamma    : {cfg_summary['PRESSURE_MISS_GAMMA']}")
    print(f"Top cap percent   : {cfg_summary['SCORE_GATE_TOP_PERCENT']}")
    print(f"Precision target  : {cfg_summary['ADMISSION_PRECISION_TARGET']}")
    print(f"Precision sens.   : {cfg_summary['ADMISSION_PRECISION_SENSITIVITY']}")
    print(
        "Quality spread    : "
        f"q={cfg_summary['SCORE_SPREAD_Q']}, "
        f"ema={cfg_summary['SCORE_SPREAD_EMA_ALPHA']}, "
        f"eps={cfg_summary['SCORE_SPREAD_EPS']}"
    )
    print(
        "Quality mult      : "
        f"min={cfg_summary['SCORE_QUALITY_MIN']}, "
        f"max={cfg_summary['SCORE_QUALITY_MAX']}, "
        f"min_boost={cfg_summary['SCORE_QUALITY_MIN_BOOST']}"
    )
    print(f"Quality interaction: {cfg_summary['GUARD_QUALITY_INTERACTION_MODE']}")
    if cfg_summary.get("GUARD_QUALITY_INTERACTION_MODE") in ("cap_modulation", "boundary_cap_modulation"):
        print(f"Quality cap range : {cfg_summary['QUALITY_CAP_MIN']}-{cfg_summary['QUALITY_CAP_MAX']}")
    # print(f"Drift gain        : {cfg_summary['DRIFT_GAIN']}")
    print(f"Cache capacities  : {[row['cache_size_objects'] for row in capacities]}")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value_f = float(value)
        if math.isnan(value_f):
            return default
        return value_f
    except (TypeError, ValueError):
        return default


def _ceil_budget(capacity: int, alpha: float, pressure: float) -> int:
    if capacity <= 0 or alpha <= 0.0 or pressure <= 0.0:
        return 0
    return int(math.ceil(capacity * alpha * pressure))


def _clip_alpha(alpha: float, cfg: Dict[str, Any]) -> float:
    alpha_min = _safe_float(cfg.get("DRIFT_ALPHA_MIN"), 0.005)
    alpha_max = _safe_float(cfg.get("DRIFT_ALPHA_MAX"), 0.2)
    if alpha < alpha_min:
        return alpha_min
    if alpha > alpha_max:
        return alpha_max
    return alpha


def _derive_alpha_base(record: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    capacity_scale = _safe_float(record.get("capacity_scale"), 1.0)
    drift_norm = _safe_float(record.get("drift_norm"), 0.0)
    drift_gain = _safe_float(cfg.get("DRIFT_GAIN"), 0.0)
    alpha = 0.08 * (1.0 + drift_gain * drift_norm) * capacity_scale
    return _clip_alpha(alpha, cfg)


def _derive_precision_adjust(record: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    precision_eff = record.get("admission_precision_eff")
    admission_precision = record.get("admission_precision")
    guard_precision = precision_eff if precision_eff is not None else admission_precision
    if guard_precision is None:
        return 1.0
    p_tar = _safe_float(cfg.get("ADMISSION_PRECISION_TARGET"), 0.12)
    sensitivity = _safe_float(cfg.get("ADMISSION_PRECISION_SENSITIVITY"), 0.0)
    delta = (_safe_float(guard_precision) - p_tar) / max(p_tar, 1e-12)
    if delta > 0.0:
        return 1.0 + sensitivity * delta
    return 1.0


def _binding_reason(
    record: Dict[str, Any],
    cfg: Dict[str, Any],
    budget_raw_before_cap: int,
    budget_after_quality_before_cap: int,
    final_budget: int,
) -> str:
    miss_candidates = _safe_int(record.get("miss_candidates"))
    if miss_candidates <= 0:
        return "no_candidates"
    if bool(record.get("fill_phase")) and final_budget <= _safe_int(record.get("fill_min_budget")):
        return "fill_floor"
    if bool(record.get("score_gate_applied")) or (
        _safe_int(record.get("score_gate_k")) > 0
        and final_budget == _safe_int(record.get("score_gate_k"))
        and budget_after_quality_before_cap > final_budget
    ):
        return "top_percent_cap"
    if final_budget == miss_candidates and budget_after_quality_before_cap > miss_candidates:
        return "candidate_count"
    q = _safe_float(record.get("quality_mult"), 1.0)
    q_min = _safe_float(cfg.get("SCORE_QUALITY_MIN"), 1.0)
    q_max = _safe_float(cfg.get("SCORE_QUALITY_MAX"), 1.0)
    if q_min != 1.0 and abs(q - q_min) <= 1e-12:
        return "quality_min_clip"
    if q_max != 1.0 and abs(q - q_max) <= 1e-12:
        return "quality_max_clip"
    alpha_final = _safe_float(record.get("admission_alpha"))
    alpha_min = _safe_float(cfg.get("DRIFT_ALPHA_MIN"), 0.005)
    alpha_max = _safe_float(cfg.get("DRIFT_ALPHA_MAX"), 0.2)
    if abs(alpha_final - alpha_min) <= 1e-12:
        return "alpha_min"
    if abs(alpha_final - alpha_max) <= 1e-12:
        return "alpha_max"
    if final_budget == budget_raw_before_cap or final_budget == budget_after_quality_before_cap:
        return "none"
    if budget_raw_before_cap != budget_after_quality_before_cap and final_budget == budget_raw_before_cap:
        return "rounding_no_effect"
    return "none"


def _apply_budget_caps(record: Dict[str, Any], raw_budget: int, score_gate_k_override: Optional[int] = None) -> int:
    miss_candidates = _safe_int(record.get("miss_candidates"))
    budget = raw_budget
    if bool(record.get("fill_phase")) and budget < _safe_int(record.get("fill_min_budget")):
        budget = _safe_int(record.get("fill_min_budget"))
    if miss_candidates <= 0:
        return 0
    if budget > miss_candidates:
        budget = miss_candidates
    score_gate_k = _safe_int(score_gate_k_override if score_gate_k_override is not None else record.get("score_gate_k"))
    if not bool(record.get("fill_phase")) and score_gate_k > 0 and budget > score_gate_k:
        budget = score_gate_k
    return budget


def enrich_signal_diagnostics(
    slot_log_path: str,
    diagnostic_path: str,
    *,
    dataset: str,
    variant: str,
    capacity: int,
    capacity_percent: Optional[float],
    cfg_summary: Dict[str, Any],
) -> None:
    records = load_jsonl(slot_log_path)
    enriched: List[Dict[str, Any]] = []
    prev_record: Optional[Dict[str, Any]] = None
    for record in records:
        miss_candidates = _safe_int(record.get("miss_candidates"))
        miss_rate = _safe_float(record.get("miss_rate"))
        pressure_without_quality = 1.0 + _safe_float(cfg_summary.get("PRESSURE_MISS_GAMMA"), 0.5) * miss_rate
        quality_mult = _safe_float(record.get("quality_mult"), 1.0)
        score_spread = _safe_float(record.get("score_spread"), 0.0)
        score_spread_ema = _safe_float(record.get("score_spread_ema"), 0.0)
        normalized_score_spread = score_spread / (score_spread_ema + _safe_float(cfg_summary.get("SCORE_SPREAD_EPS"), 1e-6)) if score_spread_ema > 0.0 else None
        alpha_base = _derive_alpha_base(record, cfg_summary)
        precision_adjust = _derive_precision_adjust(record, cfg_summary)
        alpha_floor = alpha_base * _safe_float(cfg_summary.get("ADMISSION_ALPHA_FLOOR_MULT"), 0.7)
        alpha_before_precision = _clip_alpha(max(alpha_base, alpha_floor), cfg_summary)
        alpha_after_precision = _clip_alpha(max(alpha_base * precision_adjust, alpha_floor), cfg_summary)
        alpha_final = _safe_float(record.get("admission_alpha"), alpha_after_precision)
        interaction_mode = str(cfg_summary.get("GUARD_QUALITY_INTERACTION_MODE") or "budget_multiplier")
        quality_budget_mult = 1.0 if interaction_mode == "cap_modulation" else quality_mult
        raw_after_precision = _ceil_budget(capacity, alpha_after_precision, pressure_without_quality)
        # Isolate q_t by holding the precision-adjusted alpha fixed. This keeps
        # the diagnostic neutral when quality_mult == 1.0, including after
        # alpha floor/min/max clipping.
        raw_after_quality = _ceil_budget(capacity, alpha_after_precision, pressure_without_quality * quality_budget_mult)
        raw_before_precision_clipped = _ceil_budget(capacity, alpha_before_precision, pressure_without_quality)
        base_score_gate_k = max(1, int(math.ceil(_safe_float(cfg_summary.get("SCORE_GATE_TOP_PERCENT"), 0.05) * miss_candidates))) if miss_candidates > 0 else 0
        actual_score_gate_k = _safe_int(record.get("score_gate_k"))
        without_quality_gate_k = base_score_gate_k if interaction_mode == "cap_modulation" else actual_score_gate_k
        with_quality_gate_k = actual_score_gate_k
        final_without_precision = _apply_budget_caps(record, raw_before_precision_clipped, without_quality_gate_k)
        final_without_quality = _apply_budget_caps(record, raw_after_precision, without_quality_gate_k)
        final_with_quality = _apply_budget_caps(record, raw_after_quality, with_quality_gate_k)
        final_budget = _safe_int(record.get("admit_budget"))
        reason = _binding_reason(
            record,
            cfg_summary,
            raw_after_precision,
            raw_after_quality,
            final_budget,
        )
        if (
            reason == "none"
            and raw_after_quality != raw_after_precision
            and final_with_quality == final_without_quality
        ):
            reason = "rounding_no_effect"
        diagnostic = {
            "dataset": dataset,
            "variant": variant,
            "cache_size_objects": int(capacity),
            "capacity_percent": capacity_percent,
            "slot_index": record.get("slot_index"),
            "phase": record.get("phase"),
            "slot_cache_requests": record.get("slot_cache_requests"),
            "slot_cache_hits": record.get("slot_cache_hits"),
            "slot_cache_misses": record.get("slot_cache_misses"),
            "slot_hit_ratio": record.get("slot_hit_ratio"),
            "hit_ratio": record.get("hit_ratio"),
            "miss_candidates": miss_candidates,
            "candidate_ids_count": miss_candidates,
            "admit_budget_final": final_budget,
            "admit_selected": record.get("admit_selected"),
            "admit_applied": record.get("admit_applied"),
            "candidate_cap_U_t": record.get("score_gate_k"),
            "eta_top_t": record.get("eta_top_t"),
            "eta_top_base": record.get("eta_top_base"),
            "eta_top_min": record.get("eta_top_min"),
            "eta_top_max": record.get("eta_top_max"),
            "quality_cap_changed": record.get("quality_cap_changed"),
            "fill_phase": record.get("fill_phase"),
            "fill_min_budget": record.get("fill_min_budget"),
            "capacity_scale_chi_t": record.get("capacity_scale"),
            "alpha_base": alpha_base,
            "alpha_after_precision": alpha_after_precision,
            "alpha_final": alpha_final,
            "precision_eff": record.get("admission_precision_eff"),
            "precision_lag0": record.get("admission_precision_lag0"),
            "precision_lag1": record.get("admission_precision_lag1"),
            "precision_lag2": record.get("admission_precision_lag2"),
            "precision_adjust": precision_adjust,
            "score_spread": record.get("score_spread"),
            "score_spread_ema": record.get("score_spread_ema"),
            "normalized_score_spread": normalized_score_spread,
            "quality_mult": quality_mult,
            "boundary_quality": record.get("boundary_quality"),
            "global_quality": record.get("global_quality"),
            "miss_rate": miss_rate,
            "pressure_mult": record.get("pressure_mult"),
            "budget_raw_before_cap": raw_after_precision,
            "budget_raw_before_quality": raw_after_precision,
            "budget_after_quality_before_cap": raw_after_quality,
            "budget_without_quality_after_cap": final_without_quality,
            "budget_with_quality_after_cap": final_with_quality,
            "budget_after_top_percent_cap": final_with_quality,
            "final_budget_after_rounding": final_with_quality,
            "final_budget": record.get("final_budget", final_budget),
            "precision_budget_before_quality_cap": record.get("precision_budget_before_quality_cap"),
            "final_budget_after_quality_cap": record.get("final_budget_after_quality_cap"),
            "budget_binding_reason": reason,
            "quality_mult_integer_budget_delta": abs(final_with_quality - final_without_quality),
            "precision_integer_budget_delta": abs(final_without_quality - final_without_precision),
            "quality_mult_changes_integer_budget": final_with_quality != final_without_quality,
            "precision_changes_integer_budget": final_without_quality != final_without_precision,
            "quality_min_clip": reason == "quality_min_clip",
            "quality_max_clip": reason == "quality_max_clip",
            "top_percent_cap_binds": reason == "top_percent_cap",
            "alpha_min_binds": reason == "alpha_min",
            "alpha_max_binds": reason == "alpha_max",
            "fill_floor_binds": reason == "fill_floor",
            "admission_precision": record.get("admission_precision"),
            "admit_true_from_applied": record.get("admit_true_from_applied"),
            "admitted_from_previous_slot": record.get("admitted_from_previous_slot"),
            "hits_from_recent_admissions": record.get("hits_from_recent_admissions"),
            "hit_yield": record.get("hit_yield_slot"),
            "pollution_count_slot": record.get("pollution_count_slot"),
            "pollution_total_slot": record.get("pollution_total_slot"),
            "pollution_proxy": record.get("pollution_rate_slot"),
            "pollution_rate": record.get("pollution_rate_slot"),
            "admission_rate_slot": record.get("admission_rate_slot"),
            "insertions_per_request_slot": record.get("insertions_per_request_slot"),
        }
        if prev_record is not None:
            prev_record["next_slot_hit_ratio"] = diagnostic.get("slot_hit_ratio")
            prev_record["next_slot_admission_precision"] = diagnostic.get("admission_precision")
            prev_record["next_slot_hit_yield"] = diagnostic.get("hit_yield")
            prev_record["next_slot_pollution_proxy"] = diagnostic.get("pollution_proxy")
            prev_record["next_slot_pollution_rate"] = diagnostic.get("pollution_rate")
        enriched.append(diagnostic)
        prev_record = diagnostic
    write_jsonl(diagnostic_path, enriched)


def run_dataset_variant(
    ds: DatasetSpec,
    variant: str,
    feature_set: str,
    base_learner: str,
    results_root: str,
    capacity_percent: Optional[float],
    capacity_objects: Optional[int],
    max_requests: Optional[int],
    warmup_requests_override: Optional[int],
    slot_size_override: Optional[int],
    disable_progress: bool,
    seed: int,
    enable_signal_diagnostics: bool,
) -> Dict[str, Any]:
    spec = get_variant_spec(variant)
    module = spec.module
    model_name = model_name_for_variant(variant, feature_set, base_learner)

    total_requests = ds.total_requests if max_requests is None else min(int(max_requests), ds.total_requests)
    warmup_requests = ds.warmup_requests if warmup_requests_override is None else int(warmup_requests_override)
    slot_size = ds.slot_size if slot_size_override is None else int(slot_size_override)
    validate_runtime(total_requests, warmup_requests, slot_size)

    num_unique_objects: Optional[int] = None
    if capacity_objects is None:
        num_unique_objects = count_distinct_objects(ds.path, total_requests)
    capacities, num_unique_objects = compute_capacities(
        trace_path=ds.path,
        total_requests=total_requests,
        capacity_percent=capacity_percent,
        capacity_objects=capacity_objects,
        num_unique_objects=num_unique_objects,
    )

    capacity_results: List[Dict[str, Any]] = []
    with patched_module(module, spec.overrides, disable_progress=disable_progress):
        cfg_summary = config_summary_for_variant(spec, module)
        num_features = module._get_feature_dim(module.IL_NUM_GAPS, feature_set)
        print_paper_config(
            ds=ds,
            variant=variant,
            model_name=model_name,
            trace_path=ds.path,
            total_requests=total_requests,
            warmup_requests=warmup_requests,
            slot_size=slot_size,
            feature_set=feature_set,
            base_learner=base_learner,
            capacities=capacities,
            num_features=num_features,
            module=module,
            cfg_summary=cfg_summary,
        )
        for cap_info in capacities:
            capacity = int(cap_info["cache_size_objects"])
            cap_percent = cap_info["capacity_percent"]
            run_id, dataset_dir = get_next_run_id(results_root, ds.name, model_name, capacity)
            per_slot_path = os.path.join(dataset_dir, f"{run_id}_{model_name}_{capacity}.jsonl")
            diagnostic_path = os.path.join(
                dataset_dir,
                f"{run_id}_{model_name}_{capacity}_signal_diagnostics.jsonl",
            )
            summary_path = os.path.join(dataset_dir, f"{run_id}_summary_{model_name}_{capacity}.json")

            stats, summary_metrics = module.run_single_capacity(
                trace_path=ds.path,
                total_requests=total_requests,
                warmup_requests=warmup_requests,
                slot_size=slot_size,
                capacity_objects=capacity,
                feature_set=feature_set,
                base_learner=base_learner,
                slot_log_path=per_slot_path,
            )

            slots_processed = int(summary_metrics.get("slots_processed", 0))
            warmup_slots = warmup_requests // slot_size
            summary_payload: Dict[str, Any] = {
                "dataset": ds.name,
                "model": model_name,
                "variant": variant,
                "feature_set": feature_set,
                "base_learner": base_learner,
                "num_features": num_features,
                "guardrails_precision_feedback_enabled": spec.precision_feedback_enabled,
                "guardrails_score_quality_enabled": spec.score_quality_enabled,
                "cache_size_objects": capacity,
                "capacity_percent": cap_percent,
                "total_requests": total_requests,
                "warmup_requests": warmup_requests,
                "slot_size": slot_size,
                "slot_log_path": per_slot_path,
                "hit_ratio": stats.hit_ratio,
                "cache_hits": stats.cache_hits,
                "cache_requests": stats.total_requests,
                "num_slots": slots_processed,
                "warmup_slots": warmup_slots,
                "cache_slots": max(0, slots_processed - warmup_slots),
                "pop_top_percent": module.IL_POP_TOP_PERCENT,
                "label_topk_rounding": module.IL_LABEL_TOPK_ROUNDING,
                "label_tie_break": module.IL_LABEL_TIE_BREAK,
                "protocol": "one_slot_delayed_admission",
                "eviction_policy": "fixed_lru",
                "config_summary": cfg_summary,
                "reproducibility_seed": seed,
            }
            summary_payload.update(summary_metrics)
            if enable_signal_diagnostics:
                enrich_signal_diagnostics(
                    per_slot_path,
                    diagnostic_path,
                    dataset=ds.name,
                    variant=variant,
                    capacity=capacity,
                    capacity_percent=cap_percent,
                    cfg_summary=cfg_summary,
                )
                summary_payload["signal_diagnostics_path"] = diagnostic_path
            save_json(summary_path, summary_payload)

            capacity_results.append(
                {
                    "cache_size_objects": capacity,
                    "capacity_percent": cap_percent,
                    "hit_ratio": stats.hit_ratio,
                    "cache_hits": stats.cache_hits,
                    "cache_requests": stats.total_requests,
                    "summary_path": summary_path,
                    "slot_log_path": per_slot_path,
                    "signal_diagnostics_path": diagnostic_path if enable_signal_diagnostics else None,
                }
            )
            print(
                f"[RUN] {ds.name} {variant} cap={capacity} "
                f"HR={stats.hit_ratio:.6f} -> {summary_path}"
            )

    hr_curve = [
        {"cache_size_objects": row["cache_size_objects"], "hit_ratio": row["hit_ratio"]}
        for row in capacity_results
    ]
    avg_hr = (
        sum(float(row["hit_ratio"]) for row in capacity_results) / len(capacity_results)
        if capacity_results
        else None
    )
    group_id, dataset_dir = get_next_group_id(results_root, ds.name, model_name)
    all_sizes_path = os.path.join(dataset_dir, f"{group_id}_summary_{model_name}_all_sizes.json")
    all_sizes_payload = {
        "dataset": ds.name,
        "model": model_name,
        "variant": variant,
        "feature_set": feature_set,
        "base_learner": base_learner,
        "num_features": num_features,
        "guardrails_precision_feedback_enabled": spec.precision_feedback_enabled,
        "guardrails_score_quality_enabled": spec.score_quality_enabled,
        "cache_sizes": [row["cache_size_objects"] for row in capacity_results],
        "cache_size_percentages": [row["capacity_percent"] for row in capacity_results],
        "hr_curve": hr_curve,
        "avg_hr": avg_hr,
        "config_summary": cfg_summary,
        "results": capacity_results,
    }
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
    }


def write_master_summary(
    results_root: str,
    entries: List[Dict[str, Any]],
    datasets: List[DatasetSpec],
    variants: List[str],
    feature_set: str,
    base_learner: str,
) -> Optional[str]:
    if len(entries) <= 1:
        return None
    label = "il_guard_ablation_all_variants_all_datasets"
    run_id, aggregate_dir = get_next_aggregate_id(results_root, label)
    path = os.path.join(aggregate_dir, f"{run_id}_summary_{label}.json")
    save_json(
        path,
        {
            "datasets_executed": [ds.name for ds in datasets],
            "variants_executed": variants,
            "feature_set": feature_set,
            "base_learner": base_learner,
            "results": [
                {
                    "dataset": row["dataset"],
                    "variant": row["variant"],
                    "model": row["model"],
                    "all_sizes_summary_path": row["all_sizes_summary_path"],
                    "avg_hr": row["avg_hr"],
                }
                for row in entries
            ],
        },
    )
    print(f"[MASTER] {path}")
    return path


def write_gap_summary(results_root: str, entries: List[Dict[str, Any]]) -> Optional[str]:
    by_dataset: Dict[str, Dict[str, Dict[int, float]]] = {}
    for entry in entries:
        by_dataset.setdefault(entry["dataset"], {})[entry["variant"]] = entry["hr_by_capacity"]

    gap_rows: List[Dict[str, Any]] = []
    for dataset, variant_map in by_dataset.items():
        required = {"no-guard", "precision-only", "quality-only", "full"}
        if not required.issubset(variant_map):
            continue
        common_caps = set(variant_map["full"])
        for variant in required - {"full"}:
            common_caps &= set(variant_map[variant])
        for cap in sorted(common_caps):
            full_hr = variant_map["full"][cap]
            gap_rows.append(
                {
                    "dataset": dataset,
                    "cache_size_objects": cap,
                    "full": full_hr,
                    "no_guard": variant_map["no-guard"][cap],
                    "precision_only": variant_map["precision-only"][cap],
                    "quality_only": variant_map["quality-only"][cap],
                    "full_minus_no_guard": full_hr - variant_map["no-guard"][cap],
                    "full_minus_precision_only": full_hr - variant_map["precision-only"][cap],
                    "full_minus_quality_only": full_hr - variant_map["quality-only"][cap],
                }
            )

    if not gap_rows:
        print("[WARN] Gap summary skipped: complete no-guard/precision-only/quality-only/full set not present.")
        return None

    def avg(field: str) -> float:
        return sum(float(row[field]) for row in gap_rows) / len(gap_rows)

    label = "il_guard_ablation_gap"
    run_id, aggregate_dir = get_next_aggregate_id(results_root, label)
    path = os.path.join(aggregate_dir, f"{run_id}_summary_{label}.json")
    save_json(
        path,
        {
            "rows": gap_rows,
            "avg_full_minus_no_guard": avg("full_minus_no_guard"),
            "avg_full_minus_precision_only": avg("full_minus_precision_only"),
            "avg_full_minus_quality_only": avg("full_minus_quality_only"),
        },
    )
    print(f"[GAP] {path}")
    return path


def run_experiment(args: argparse.Namespace) -> List[Dict[str, Any]]:
    datasets = resolve_datasets(args.dataset)
    variants = resolve_variants(args.variant)
    capacity_objects = args.capacity_objects
    if capacity_objects is None:
        capacity_objects = args.cache_size_objects
    elif args.cache_size_objects is not None and args.cache_size_objects != capacity_objects:
        raise ValueError("--capacity-objects and --cache-size-objects must match when both are set.")

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
                    capacity_percent=args.capacity_percent,
                    capacity_objects=capacity_objects,
                    max_requests=args.max_requests,
                    warmup_requests_override=args.warmup_requests,
                    slot_size_override=args.slot_size,
                    disable_progress=args.disable_progress,
                    seed=args.seed,
                    enable_signal_diagnostics=args.enable_signal_diagnostics,
                )
            )

    write_master_summary(args.results_root, entries, datasets, variants, args.feature_set, args.base_learner)
    requested_variants = args.variant if isinstance(args.variant, list) else [args.variant]
    if "all" in requested_variants or set(CORE_VARIANTS).issubset(set(variants)):
        write_gap_summary(args.results_root, entries)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IL Guardrails component ablations.")
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS) + ["all"],
        nargs="+",
        default=["all"],
    )
    parser.add_argument(
        "--dataset",
        choices=["wikipedia_september_2007", "wikipedia_oktober_2007", "wiki2018", "all"],
        default="all",
    )
    parser.add_argument("--feature-set", choices=["A1", "A2", "A3"], default="A2")
    parser.add_argument("--base-learner", choices=["nb", "svm", "dt"], default="nb")
    parser.add_argument("--capacity-percent", type=float, default=None)
    parser.add_argument("--capacity-objects", type=int, default=None)
    parser.add_argument("--cache-size-objects", type=int, default=None)
    parser.add_argument("--all-sizes", action="store_true")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--warmup-requests", type=int, default=None)
    parser.add_argument("--slot-size", type=int, default=None)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--enable-signal-diagnostics", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.variant = ["all"]
        args.dataset = "wikipedia_september_2007"
        args.capacity_percent = 0.8
        args.max_requests = 300_000
        args.warmup_requests = 100_000
        args.slot_size = 100_000
        args.disable_progress = True
        args.enable_signal_diagnostics = True

    if args.capacity_percent is not None and args.capacity_percent <= 0:
        raise ValueError("--capacity-percent must be > 0.")
    if args.capacity_objects is not None and args.capacity_objects < 1:
        raise ValueError("--capacity-objects must be >= 1.")
    if args.cache_size_objects is not None and args.cache_size_objects < 1:
        raise ValueError("--cache-size-objects must be >= 1.")
    if args.max_requests is not None and args.max_requests < 1:
        raise ValueError("--max-requests must be >= 1.")

    run_experiment(args)


if __name__ == "__main__":
    main()
