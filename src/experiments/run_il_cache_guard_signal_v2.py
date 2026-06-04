# src/experiments/run_il_cache.py

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import deque
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
 
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.trace_reader import TraceReader
from src.data.feature_table import FeatureTable
from src.ml.learn_nse_opt import (
    LearnNSE,
    GaussianNaiveBayes,
    LinearSVMWrapper,
    DecisionTreeWrapper,
)
from src.cache.cache_simulator import CacheStats
from src.cache.lru import LRUCache

FEATURE_SETS = ("A0", "A1", "A2", "A3")
FEATURE_WINDOWS = {"short": 1, "mid": 7, "long": 30}
POLLUTION_WINDOW_SLOTS = 1

# Dataset configs (inline, no external config dependency)
WIKIPEDIA_SEPTEMBER_2007 = {
    "name": "wikipedia_september_2007",
    "path": "data/raw/wikipedia_september_2007/wiki.1190153705.gz",
    "num_total_requests": 9_000_000,
    "num_warmup_requests": 1_000_000,
    "slot_size": 100_000,
}
WIKIPEDIA_OKTOBER_2007 = {
    "name": "wikipedia_oktober_2007",
    "path": "data/raw/wikipedia_oktober_2007/wiki.1191201596.gz",
    "num_total_requests": 8_000_000,
    "num_warmup_requests": 1_000_000,
    "slot_size": 100_000,
}
WIKI2018 = {
    "name": "wiki2018",
    "path": "data/raw/wiki2018/wiki2018.gz",
    "num_total_requests": 10_000_000,
    "num_warmup_requests": 1_000_000,
    "slot_size": 100_000,
}

# Fixed experiment settings (no external config)
IL_NUM_GAPS = 6
IL_POP_TOP_PERCENT = 0.20
IL_LABEL_TOPK_ROUNDING = "floor"
IL_LABEL_TIE_BREAK = "none"
IL_SIGMOID_A = 0.5
IL_SIGMOID_B = 10.0
IL_MAX_CLASSIFIERS = 20
IL_MISSING_GAP_VALUE = 1e6
CACHE_SIZE_PERCENTAGES = (0.8, 1.0, 2.0, 3.0, 4.0, 5.0)
ADMISSION_CAPACITY_ALPHA = 0.08
DRIFT_ALPHA_MIN = 0.005
DRIFT_ALPHA_MAX = 0.2
DRIFT_SENSITIVITY = 1.0
DRIFT_EMA_ALPHA = 0.3
DRIFT_STATS_ALPHA = 0.1
DRIFT_NORM_EPS = 1e-6
DRIFT_Z_CLIP = 3.0
DRIFT_WEIGHT_JSD = 0.4
DRIFT_WEIGHT_OVERLAP = 0.6
# GUARD-ONLY: disable drift contribution to budget
DRIFT_GAIN = 0.0
DRIFT_ALPHA_FLOOR_MULT = 0.7
# Drift control configuration
# - scaled: alpha = base * (1 + gain * drift_norm^p)
# - piecewise: alpha = high if drift_norm^p >= threshold else low
DRIFT_CONTROL_MODE = "scaled"  # "scaled", "piecewise", "fixed"
DRIFT_NORM_POWER = 1.0
DRIFT_THRESHOLD = 0.6
DRIFT_ALPHA_LOW = 0.04
DRIFT_ALPHA_HIGH = 0.18
DRIFT_USE_CAPACITY_SCALE = True
FINAL_SCORE_GATE_TOP_PERCENT = 0.02
SCORE_GATE_TOP_PERCENT = 0.05
FILL_RATIO = 0.9
FILL_RATE = 0.05
PRESSURE_MISS_GAMMA = 0.5
SCORE_SPREAD_Q = 0.9
SCORE_SPREAD_EMA_ALPHA = 0.2
SCORE_SPREAD_EPS = 1e-6
SCORE_QUALITY_SIGNAL_MODE = "calibrated_spread"
# allowed: "off", "spread_prev_ema", "calibrated_spread", "boundary_margin"
SCORE_QUALITY_CONTROL_ROLE = "safety"
# allowed: "symmetric", "safety"
SCORE_QUALITY_USE_PREV_EMA = True
SCORE_QUALITY_CALIBRATION_ENABLED = True
SCORE_QUALITY_CALIBRATION_METRIC = "score_precK"
SCORE_QUALITY_PRECK_EMA_ALPHA = 0.2
SCORE_QUALITY_TARGET_MODE = "candidate_base_rate_margin"
# allowed: "fixed", "candidate_base_rate", "candidate_base_rate_margin"
SCORE_QUALITY_TARGET_MARGIN = 0.02
SCORE_QUALITY_CALIB_MIN = 0.5
SCORE_QUALITY_CALIB_MAX = 1.2
SCORE_QUALITY_MIN = 0.7
SCORE_QUALITY_MAX = 1.0
SCORE_QUALITY_MIN_BOOST = 0.0
GUARD_QUALITY_INTERACTION_MODE = "budget_multiplier"
QUALITY_CAP_MIN = 0.025
QUALITY_CAP_MAX = 0.05
QUALITY_CAP_BALANCED_MAX = 0.075
PRECISION_CONTROL_MODE = "bidirectional"
# allowed: "off", "reward_only_legacy", "bidirectional"
EFFECTIVE_PRECISION_LAG_MODE = "weighted"
# allowed: "max_legacy", "fixed", "weighted"
EFFECTIVE_PRECISION_FIXED_LAG = 1
EFFECTIVE_PRECISION_LAG_WEIGHTS = (0.5, 0.3, 0.2)
ADMISSION_PRECISION_TARGET_MODE = "base_rate_margin"
# allowed: "fixed", "base_rate_ema", "base_rate_margin"
ADMISSION_PRECISION_TARGET = 0.12
ADMISSION_PRECISION_SENSITIVITY = 1.0
ADMISSION_PRECISION_MARGIN = 0.02
CANDIDATE_POS_RATE_EMA_ALPHA = 0.2
ADMISSION_PRECISION_POS_GAIN = 0.7
ADMISSION_PRECISION_NEG_GAIN = 1.2
ADMISSION_PRECISION_MULT_MIN = 0.5
ADMISSION_PRECISION_MULT_MAX = 1.8
CAPACITY_ALPHA_SCALE_MIN = 0.4
QUALITY_CONTROL_MODE = "normal"
# allowed: "normal", "random", "inverted", "permuted"
PRECISION_CONTROL_OVERRIDE_MODE = "normal"
# allowed: "normal", "random", "permuted"
CONTROL_RANDOM_SEED = 42
BUDGET_MATCH_MODE = "off"
# allowed: "off", "uniform", "random_slot_throttle", "slot_replay", "slot_replay_permuted"
BUDGET_MATCH_MULTIPLIER = 1.0
BUDGET_MATCH_TARGET_MULTIPLIER = 1.0
BUDGET_MATCH_RATIO_SOURCE = None
BUDGET_REPLAY_PATH = None
BUDGET_REPLAY_KEY = "budget"
BUDGET_REPLAY_STRICT = True
BUDGET_REPLAY_ALLOW_CLAMP = False
BUDGET_REPLAY_PERMUTE_SEED = 42
OFFLINE_DIAGNOSTIC_CONTROL = False
QUALITY_SUPPRESSED_UTILITY_WINDOW_SLOTS = 1

# ---------------------------------------------------------------------------
# Utilitas untuk membangun D_t dari statistik slot
# ---------------------------------------------------------------------------

def _compute_topk_k(n_objects: int, top_ratio: float, rounding: str) -> int:
    raw_k = n_objects * top_ratio
    if rounding == "ceil":
        k = int(math.ceil(raw_k))
    elif rounding == "floor":
        k = int(raw_k)
    else:
        raise ValueError(f"Unknown label_topk_rounding: {rounding}")
    if k <= 0:
        k = 1
    if k > n_objects:
        k = n_objects
    return k


def _get_feature_dim(num_gaps: int, feature_set: str) -> int:
    if feature_set == "A0":
        return num_gaps
    if feature_set == "A1":
        return num_gaps
    if feature_set == "A2":
        return num_gaps + 4
    if feature_set == "A3":
        return 4
    raise ValueError(f"Unknown feature_set: {feature_set}")


def _compute_admission_budget(
    miss_requests: int,
    capacity_objects: int,
    alpha: float,
    pressure_mult: float = 1.0,
) -> int:
    # Eq. (6) operational counterpart:
    # - Budget M_t is realized via bounded alpha * pressure in this implementation.
    # - This is a controller-equivalent heuristic, not literal additive update.
    if miss_requests <= 0:
        return 0
    return int(math.ceil(capacity_objects * alpha * pressure_mult))


def _safe_div(num, den, default=0.0):
    try:
        num_f = float(num)
        den_f = float(den)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(num_f) or not math.isfinite(den_f) or den_f == 0.0:
        return default
    return float(num_f / den_f)


_BUDGET_REPLAY_KEY_CANDIDATES = (
    "budget",
    "admit_selected",
    "selected_admissions",
    "selected_admissions_total",
    "final_budget",
    "admit_budget",
    "final_budget_after_quality_cap",
    "final_budget_full_same_cap",
)
_SLOT_INDEX_KEY_CANDIDATES = ("slot_index", "slot", "slot_num")


def _resolve_required_key(
    row: Dict[str, Any],
    candidates: Tuple[str, ...],
    context: str,
) -> str:
    for key in candidates:
        if key in row and row[key] is not None:
            return key
    available = ", ".join(sorted(str(key) for key in row.keys()))
    expected = ", ".join(candidates)
    raise KeyError(
        f"{context}: missing required key. Expected one of [{expected}]. "
        f"Available keys: [{available}]"
    )


def _resolve_required_int(
    row: Dict[str, Any],
    candidates: Tuple[str, ...],
    context: str,
) -> Tuple[int, str]:
    key = _resolve_required_key(row, candidates, context)
    try:
        value = int(row[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: key '{key}' must be integer-like, got {row[key]!r}") from exc
    return value, key


def _read_budget_replay_rows(path: str) -> List[Dict[str, Any]]:
    if not path:
        raise ValueError("BUDGET_REPLAY_PATH is required for slot replay modes.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Budget replay file not found: {path}")
    if path.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_num}: invalid JSONL row") from exc
                if isinstance(row, dict):
                    rows.append(row)
        return rows

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        slots = payload.get("slots")
        if isinstance(slots, list):
            return [row for row in slots if isinstance(row, dict)]
        raise ValueError(f"{path}: replay JSON must contain a list-valued 'slots' field.")
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError(f"{path}: replay file must be a JSON object or list.")


def _load_budget_replay_with_metadata(path: str) -> Tuple[Dict[int, int], Dict[str, Any]]:
    rows = _read_budget_replay_rows(path)
    requested_key = str(BUDGET_REPLAY_KEY or "").strip()
    budget_candidates = (
        (requested_key,) if requested_key else _BUDGET_REPLAY_KEY_CANDIDATES
    )
    replay: Dict[int, int] = {}
    budget_key_used: Optional[str] = None
    for row_idx, row in enumerate(rows):
        if row.get("phase") is not None and row.get("phase") != "cache":
            continue
        slot_index, _ = _resolve_required_int(
            row,
            _SLOT_INDEX_KEY_CANDIDATES,
            f"{path} row {row_idx}",
        )
        budget, key = _resolve_required_int(
            row,
            budget_candidates,
            f"{path} row {row_idx}",
        )
        if budget < 0:
            raise ValueError(f"{path} row {row_idx}: replay budget must be non-negative.")
        if slot_index in replay:
            raise ValueError(f"{path}: duplicate replay slot_index={slot_index}.")
        replay[int(slot_index)] = int(budget)
        if budget_key_used is None:
            budget_key_used = key
        elif budget_key_used != key and BUDGET_REPLAY_STRICT:
            raise ValueError(
                f"{path}: mixed budget keys are not allowed in strict mode "
                f"({budget_key_used!r} and {key!r})."
            )
    if not replay:
        raise ValueError(f"{path}: no cache-phase replay budgets were loaded.")
    return replay, {
        "path": path,
        "budget_key_used": budget_key_used,
        "slots_loaded": len(replay),
        "total_budget": int(sum(replay.values())),
    }


def load_budget_replay(path) -> dict[int, int]:
    replay, _ = _load_budget_replay_with_metadata(str(path))
    return replay


def _permute_budget_replay(
    replay: Dict[int, int],
    seed: int,
) -> Tuple[Dict[int, int], Dict[str, Any]]:
    slots = sorted(replay.keys())
    original_values = [int(replay[slot]) for slot in slots]
    permuted_values = list(original_values)
    rng = random.Random(int(seed))
    rng.shuffle(permuted_values)
    permuted = {slot: int(value) for slot, value in zip(slots, permuted_values)}
    same_slot_count = sum(
        1 for slot in slots if int(replay[slot]) == int(permuted[slot])
    )
    return permuted, {
        "original_total_budget": int(sum(original_values)),
        "permuted_total_budget": int(sum(permuted_values)),
        "histogram_preserved": sorted(original_values) == sorted(permuted_values),
        "same_slot_budget_rate_against_full": (
            float(same_slot_count / len(slots)) if slots else None
        ),
    }


def _clip(x, lo, hi):
    x_f = float(x)
    if x_f < lo:
        return lo
    if x_f > hi:
        return hi
    return x_f


def _is_valid_number(value: Any) -> bool:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value_f)


def _aggregate_effective_precision(precision_by_lag, mode, fixed_lag, lag_weights):
    """
    precision_by_lag maps lag -> precision value.
    mode:
    - "max_legacy": return max valid lag precision.
    - "fixed": return precision_by_lag[fixed_lag] if available else None.
    - "weighted": return weighted average over available lags, normalized over valid weights.
    Return None when no valid value exists.
    """
    valid = {
        int(lag): float(value)
        for lag, value in precision_by_lag.items()
        if _is_valid_number(value)
    }
    if not valid:
        return None
    mode = (mode or "weighted").lower()
    if mode == "max_legacy":
        return float(max(valid.values()))
    if mode == "fixed":
        value = valid.get(int(fixed_lag))
        return float(value) if value is not None else None
    if mode != "weighted":
        raise ValueError(f"Unknown EFFECTIVE_PRECISION_LAG_MODE: {mode}")
    weighted_sum = 0.0
    weight_sum = 0.0
    for lag, weight in enumerate(tuple(lag_weights)):
        if lag not in valid:
            continue
        weight_f = float(weight)
        if weight_f <= 0.0 or not math.isfinite(weight_f):
            continue
        weighted_sum += weight_f * valid[lag]
        weight_sum += weight_f
    if weight_sum <= 0.0:
        return None
    return float(weighted_sum / weight_sum)


def _compute_precision_target(candidate_pos_rate, prev_candidate_pos_rate_ema, mode):
    """
    mode:
    - fixed: ADMISSION_PRECISION_TARGET
    - base_rate_ema: max(ADMISSION_PRECISION_TARGET, prev_candidate_pos_rate_ema)
    - base_rate_margin: max(ADMISSION_PRECISION_TARGET, prev_candidate_pos_rate_ema + ADMISSION_PRECISION_MARGIN)
    Use previous EMA for control. Do not use future information.
    If prev_candidate_pos_rate_ema is None, fall back to fixed target.
    """
    del candidate_pos_rate
    base_target = float(ADMISSION_PRECISION_TARGET)
    mode = (mode or "fixed").lower()
    if mode == "fixed" or not _is_valid_number(prev_candidate_pos_rate_ema):
        return base_target
    prev_rate = float(prev_candidate_pos_rate_ema)
    if mode == "base_rate_ema":
        return float(max(base_target, prev_rate))
    if mode == "base_rate_margin":
        return float(max(base_target, prev_rate + float(ADMISSION_PRECISION_MARGIN)))
    raise ValueError(f"Unknown ADMISSION_PRECISION_TARGET_MODE: {mode}")


def _compute_precision_multiplier(p_eff, p_target, mode):
    """
    mode:
    - off: return 1.0
    - reward_only_legacy:
        if p_eff > p_target:
            return clipped 1 + ADMISSION_PRECISION_SENSITIVITY * (p_eff - p_target) / p_target
        else:
            return 1.0
    - bidirectional:
        if p_eff >= p_target:
            mult = 1 + ADMISSION_PRECISION_POS_GAIN * (p_eff - p_target) / p_target
        else:
            mult = 1 + ADMISSION_PRECISION_NEG_GAIN * (p_eff - p_target) / p_target
        return clipped between ADMISSION_PRECISION_MULT_MIN and ADMISSION_PRECISION_MULT_MAX
    Return 1.0 when p_eff or p_target is invalid.
    """
    mode = (mode or "off").lower()
    if mode == "off":
        return 1.0
    if not _is_valid_number(p_eff) or not _is_valid_number(p_target):
        return 1.0
    p_eff_f = float(p_eff)
    p_target_f = float(p_target)
    if p_target_f <= 0.0:
        return 1.0
    rel_delta = (p_eff_f - p_target_f) / p_target_f
    if mode == "reward_only_legacy":
        if rel_delta <= 0.0:
            return 1.0
        return _clip(
            1.0 + float(ADMISSION_PRECISION_SENSITIVITY) * rel_delta,
            float(ADMISSION_PRECISION_MULT_MIN),
            float(ADMISSION_PRECISION_MULT_MAX),
        )
    if mode == "bidirectional":
        gain = (
            float(ADMISSION_PRECISION_POS_GAIN)
            if rel_delta >= 0.0
            else float(ADMISSION_PRECISION_NEG_GAIN)
        )
        return _clip(
            1.0 + gain * rel_delta,
            float(ADMISSION_PRECISION_MULT_MIN),
            float(ADMISSION_PRECISION_MULT_MAX),
        )
    raise ValueError(f"Unknown PRECISION_CONTROL_MODE: {mode}")


def _compute_score_spread(scores, q=0.9):
    """
    Return Q_q(scores) - median(scores).
    Return 0.0 when scores are empty or invalid.
    """
    if not scores:
        return 0.0
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, float(q)) - np.quantile(arr, 0.5))


def _compute_boundary_margin(scores, m):
    """
    Sort scores descending.
    If m <= 0 or m >= len(scores), return 0.0.
    Return score[m-1] - score[m].
    This measures separability around the admission boundary.
    """
    if not scores:
        return 0.0
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    m_int = int(m)
    if m_int <= 0 or m_int >= int(arr.size):
        return 0.0
    sorted_scores = np.sort(arr)[::-1]
    return float(sorted_scores[m_int - 1] - sorted_scores[m_int])


def _compute_score_quality_multiplier(
    scores,
    pre_quality_budget,
    prev_score_spread_ema,
    prev_boundary_margin_ema,
    prev_score_precK_ema,
    prev_candidate_pos_rate_ema,
    mode,
    role,
):
    """
    Compute a slot-level quality multiplier.
    Do not update any EMA inside this function.

    mode:
    - off: return 1.0
    - spread_prev_ema:
        raw = current_spread / prev_score_spread_ema
    - calibrated_spread:
        spread_ratio = current_spread / prev_score_spread_ema
        calibration = prev_score_precK_ema / target_precK
        raw = spread_ratio * clipped calibration
        target_precK should be based on previous candidate_pos_rate_ema + SCORE_QUALITY_TARGET_MARGIN, with fallback to ADMISSION_PRECISION_TARGET
    - boundary_margin:
        margin_ratio = current_boundary_margin / prev_boundary_margin_ema
        optionally multiply by lagged calibration as above

    role:
    - symmetric: clip raw between SCORE_QUALITY_MIN and SCORE_QUALITY_MAX.
    - safety: clip raw between SCORE_QUALITY_MIN and 1.0.

    Use previous EMAs only for the decision. After the budget decision and logging, update EMAs using current values.
    """
    mode = (mode or "off").lower()
    role = (role or "safety").lower()
    spread = _compute_score_spread(scores, SCORE_SPREAD_Q)
    boundary_margin = _compute_boundary_margin(scores, pre_quality_budget)
    target_precK = float(ADMISSION_PRECISION_TARGET)
    target_mode = (SCORE_QUALITY_TARGET_MODE or "candidate_base_rate_margin").lower()
    if target_mode == "fixed" or not _is_valid_number(prev_candidate_pos_rate_ema):
        target_precK = float(ADMISSION_PRECISION_TARGET)
    elif target_mode == "candidate_base_rate":
        target_precK = max(float(ADMISSION_PRECISION_TARGET), float(prev_candidate_pos_rate_ema))
    elif target_mode == "candidate_base_rate_margin":
        target_precK = max(
            float(ADMISSION_PRECISION_TARGET),
            float(prev_candidate_pos_rate_ema) + float(SCORE_QUALITY_TARGET_MARGIN),
        )
    else:
        raise ValueError(f"Unknown SCORE_QUALITY_TARGET_MODE: {SCORE_QUALITY_TARGET_MODE}")
    calibration = 1.0
    if SCORE_QUALITY_CALIBRATION_ENABLED and _is_valid_number(prev_score_precK_ema):
        calibration = _clip(
            _safe_div(float(prev_score_precK_ema), target_precK, default=1.0),
            float(SCORE_QUALITY_CALIB_MIN),
            float(SCORE_QUALITY_CALIB_MAX),
        )
    raw = 1.0
    if mode == "off":
        raw = 1.0
    elif mode in ("spread_prev_ema", "calibrated_spread"):
        if not _is_valid_number(prev_score_spread_ema) or float(prev_score_spread_ema) <= 0.0:
            raw = 1.0
        else:
            raw = _safe_div(spread, prev_score_spread_ema, default=1.0)
        if mode == "calibrated_spread":
            raw *= calibration
    elif mode == "boundary_margin":
        if not _is_valid_number(prev_boundary_margin_ema) or float(prev_boundary_margin_ema) <= 0.0:
            raw = 1.0
        else:
            raw = _safe_div(boundary_margin, prev_boundary_margin_ema, default=1.0)
        if SCORE_QUALITY_CALIBRATION_ENABLED:
            raw *= calibration
    else:
        raise ValueError(f"Unknown SCORE_QUALITY_SIGNAL_MODE: {mode}")

    if not math.isfinite(float(raw)):
        raw = 1.0
    upper = 1.0 if role == "safety" else float(SCORE_QUALITY_MAX)
    if role not in ("safety", "symmetric"):
        raise ValueError(f"Unknown SCORE_QUALITY_CONTROL_ROLE: {role}")
    mult = _clip(float(raw), float(SCORE_QUALITY_MIN), upper)
    return {
        "score_quality_mult": float(mult),
        "score_quality_mult_raw": float(raw),
        "score_spread": float(spread),
        "boundary_margin": float(boundary_margin),
        "score_quality_target_precK": float(target_precK),
        "score_quality_calibration": float(calibration),
    }


def _apply_budget_bounds(
    raw_budget,
    miss_candidates,
    score_gate_k,
    cache_fill_ratio,
    fill_budget,
    fill_phase=False,
):
    """
    Return final_budget and binding_reason.
    Binding reason is one of:
    - no_candidates
    - raw_budget
    - top_percent_cap
    - candidate_count
    - fill_floor
    - zero_budget
    - other
    Use the exact same semantics as the real admission path.
    """
    del cache_fill_ratio
    miss_candidates = int(miss_candidates)
    budget = int(raw_budget)
    if fill_phase and budget < int(fill_budget):
        budget = int(fill_budget)
        fill_applied = True
    else:
        fill_applied = False
    if miss_candidates <= 0:
        return 0, "no_candidates"
    candidate_applied = False
    if budget > miss_candidates:
        budget = miss_candidates
        candidate_applied = True
    if not fill_phase and int(score_gate_k) > 0 and budget > int(score_gate_k):
        return int(score_gate_k), "top_percent_cap"
    if budget <= 0:
        return 0, "zero_budget"
    if candidate_applied:
        return int(budget), "candidate_count"
    if fill_applied:
        return int(budget), "fill_floor"
    return int(budget), "raw_budget"


def _ema_after(prev_value: Optional[float], current_value: Optional[float], alpha: float) -> Optional[float]:
    if not _is_valid_number(current_value):
        return prev_value
    current_f = float(current_value)
    if prev_value is None or not _is_valid_number(prev_value):
        return current_f
    return float(alpha * current_f + (1.0 - alpha) * float(prev_value))


def _safe_median(values: List[float]) -> Optional[float]:
    clean = [float(v) for v in values if _is_valid_number(v)]
    if not clean:
        return None
    return float(np.median(np.asarray(clean, dtype=float)))


def _safe_mean(values: List[float]) -> Optional[float]:
    clean = [float(v) for v in values if _is_valid_number(v)]
    if not clean:
        return None
    return float(np.mean(np.asarray(clean, dtype=float)))


def _safe_rate(values: List[Any]) -> Optional[float]:
    if not values:
        return None
    return float(sum(1 for value in values if bool(value)) / len(values))


def _safe_pearson(xs: List[Any], ys: List[Any]) -> Optional[float]:
    pairs: List[Tuple[float, float]] = []
    for x, y in zip(xs, ys):
        if not _is_valid_number(x) or not _is_valid_number(y):
            continue
        pairs.append((float(x), float(y)))
    if len(pairs) < 3:
        return None
    x_arr = np.asarray([x for x, _ in pairs], dtype=float)
    y_arr = np.asarray([y for _, y in pairs], dtype=float)
    x_centered = x_arr - float(x_arr.mean())
    y_centered = y_arr - float(y_arr.mean())
    denom = float(np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2)))
    if denom <= 0.0 or not math.isfinite(denom):
        return None
    return float(np.sum(x_centered * y_centered) / denom)


def _summarize_signal_diagnostics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    cache_records = [row for row in records if row.get("phase") == "cache"]
    if not cache_records:
        return {}
    pollution_lag = POLLUTION_WINDOW_SLOTS + 1
    for idx, row in enumerate(cache_records):
        if idx + 1 < len(cache_records):
            nxt = cache_records[idx + 1]
            row["next_slot_hit_yield"] = nxt.get("admission_hit_yield")
            row["next_slot_admission_precision"] = nxt.get("admission_precision")
            row["next_slot_pollution_proxy"] = nxt.get("pollution_proxy")
        else:
            row["next_slot_hit_yield"] = None
            row["next_slot_admission_precision"] = None
            row["next_slot_pollution_proxy"] = None
        row["admission_window_pollution_proxy"] = (
            cache_records[idx + pollution_lag].get("pollution_proxy")
            if idx + pollution_lag < len(cache_records)
            else None
        )
    precision_deltas = [row.get("budget_delta_precision") for row in cache_records]
    quality_deltas = [row.get("budget_delta_quality") for row in cache_records]
    return {
        "precision_action_rate": _safe_rate(
            [row.get("precision_changes_integer_budget") for row in cache_records]
        ),
        "precision_multiplier_action_rate": _safe_rate(
            [float(row.get("precision_multiplier_t") or 1.0) != 1.0 for row in cache_records]
        ),
        "precision_continuous_action_rate": _safe_rate(
            [row.get("precision_changes_continuous_budget") for row in cache_records]
        ),
        "precision_masked_by_integer_rounding_rate": _safe_rate(
            [row.get("precision_masked_by_integer_rounding") for row in cache_records]
        ),
        "precision_mean_budget_delta": _safe_mean(precision_deltas),
        "precision_median_budget_delta": _safe_median(precision_deltas),
        "precision_increase_rate": _safe_rate(
            [float(row.get("budget_delta_precision") or 0) > 0 for row in cache_records]
        ),
        "precision_decrease_rate": _safe_rate(
            [float(row.get("budget_delta_precision") or 0) < 0 for row in cache_records]
        ),
        "quality_action_rate": _safe_rate(
            [row.get("quality_changes_integer_budget") for row in cache_records]
        ),
        "score_quality_multiplier_action_rate": _safe_rate(
            [float(row.get("score_quality_mult") or 1.0) != 1.0 for row in cache_records]
        ),
        "quality_continuous_action_rate_given_precision": _safe_rate(
            [
                row.get("quality_changes_continuous_budget_given_precision")
                for row in cache_records
            ]
        ),
        "quality_masked_by_integer_rounding_rate": _safe_rate(
            [row.get("quality_masked_by_integer_rounding") for row in cache_records]
        ),
        "quality_action_rate_given_precision": _safe_rate(
            [row.get("quality_changes_integer_budget_given_precision") for row in cache_records]
        ),
        "quality_mean_budget_delta": _safe_mean(quality_deltas),
        "quality_median_budget_delta": _safe_median(quality_deltas),
        "budget_delta_precision": _safe_mean(precision_deltas),
        "budget_delta_quality": _safe_mean(quality_deltas),
        "quality_suppression_rate": _safe_rate(
            [float(row.get("score_quality_mult") or 1.0) < 1.0 for row in cache_records]
        ),
        "quality_boost_rate": _safe_rate(
            [float(row.get("score_quality_mult") or 1.0) > 1.0 for row in cache_records]
        ),
        "quality_masked_by_cap_rate": _safe_rate(
            [row.get("quality_masked_by_cap") for row in cache_records]
        ),
        "quality_masked_by_rounding_rate": _safe_rate(
            [row.get("quality_masked_by_rounding") for row in cache_records]
        ),
        "top_percent_cap_binding_rate": _safe_rate(
            [row.get("top_percent_cap_binds") for row in cache_records]
        ),
        "candidate_count_binding_rate": _safe_rate(
            [row.get("candidate_count_binds") for row in cache_records]
        ),
        "fill_floor_binding_rate": _safe_rate(
            [row.get("fill_floor_binds") for row in cache_records]
        ),
        "mean_score_quality_mult": _safe_mean([row.get("score_quality_mult") for row in cache_records]),
        "median_score_quality_mult": _safe_median([row.get("score_quality_mult") for row in cache_records]),
        "mean_precision_multiplier": _safe_mean([row.get("precision_multiplier_t") for row in cache_records]),
        "median_precision_multiplier": _safe_median([row.get("precision_multiplier_t") for row in cache_records]),
        "corr_precision_eff_to_next_slot_hit_yield": _safe_pearson(
            [row.get("precision_eff") for row in cache_records],
            [row.get("next_slot_hit_yield") for row in cache_records],
        ),
        "corr_precision_eff_to_next_slot_admission_precision": _safe_pearson(
            [row.get("precision_eff") for row in cache_records],
            [row.get("next_slot_admission_precision") for row in cache_records],
        ),
        "corr_quality_mult_to_score_precK": _safe_pearson(
            [row.get("score_quality_mult") for row in cache_records],
            [row.get("score_precK") for row in cache_records],
        ),
        "corr_quality_mult_to_score_AP": _safe_pearson(
            [row.get("score_quality_mult") for row in cache_records],
            [row.get("score_AP") for row in cache_records],
        ),
        "corr_quality_mult_to_next_slot_hit_yield": _safe_pearson(
            [row.get("score_quality_mult") for row in cache_records],
            [row.get("next_slot_hit_yield") for row in cache_records],
        ),
        "corr_quality_mult_to_next_slot_pollution_proxy": _safe_pearson(
            [row.get("score_quality_mult") for row in cache_records],
            [row.get("next_slot_pollution_proxy") for row in cache_records],
        ),
        "corr_quality_mult_to_admission_window_pollution_proxy": _safe_pearson(
            [row.get("score_quality_mult") for row in cache_records],
            [row.get("admission_window_pollution_proxy") for row in cache_records],
        ),
        "corr_precision_eff_to_admission_window_pollution_proxy": _safe_pearson(
            [row.get("precision_eff") for row in cache_records],
            [row.get("admission_window_pollution_proxy") for row in cache_records],
        ),
    }


def _select_top_m(candidate_ids: List[str], candidate_scores: List[float], m: int) -> set[str]:
    if m <= 0 or not candidate_ids:
        return set()
    n = len(candidate_ids)
    if m >= n:
        return set(candidate_ids)
    scores = np.asarray(candidate_scores, dtype=float)
    kth = np.partition(scores, -m)[-m]
    greater_idx = np.where(scores > kth)[0]
    if len(greater_idx) >= m:
        top_idx = np.argpartition(scores, -m)[-m:]
        return {candidate_ids[i] for i in top_idx}
    remaining = m - len(greater_idx)
    eq_idx = np.where(scores == kth)[0]
    selected_idx = np.concatenate([greater_idx, eq_idx[:remaining]])
    return {candidate_ids[i] for i in selected_idx}


def _auc_roc(y_true: List[int], y_score: List[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    pos = y_true_arr == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score_arr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score_arr) + 1, dtype=float)

    # average ranks for ties
    s = y_score_arr[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j

    sum_ranks_pos = float(ranks[pos].sum())
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _avg_precision(y_true: List[int], y_score: List[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    n_pos = int((y_true_arr == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score_arr)
    y_sorted = y_true_arr[order]
    tp = np.cumsum(y_sorted == 1)
    denom = np.arange(1, len(y_sorted) + 1)
    precision = tp / denom
    return float((precision[y_sorted == 1]).sum() / n_pos)


def _precision_at_k(y_true: List[int], y_score: List[float], k: int) -> float:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    if k <= 0:
        return float("nan")
    k = min(k, len(y_true_arr))
    idx = np.argsort(-y_score_arr)[:k]
    return float((y_true_arr[idx] == 1).mean())

def _compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.size == 0 or q.size == 0:
        return 0.0
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = np.sum(p * (np.log(p + eps) - np.log(m + eps)))
    kl_qm = np.sum(q * (np.log(q + eps) - np.log(m + eps)))
    jsd = 0.5 * (kl_pm + kl_qm)
    return float(jsd / math.log(2.0))


def _compute_jsd_identity(
    prev_dist: Optional[Dict[str, float]],
    curr_dist: Optional[Dict[str, float]],
) -> float:
    if not prev_dist or not curr_dist:
        return 0.0
    keys = list(set(prev_dist) | set(curr_dist))
    if not keys:
        return 0.0
    p = np.fromiter((prev_dist.get(k, 0.0) for k in keys), dtype=float)
    q = np.fromiter((curr_dist.get(k, 0.0) for k in keys), dtype=float)
    return _compute_jsd(p, q)


def _weighted_jaccard(
    prev_dist: Optional[Dict[str, float]],
    curr_dist: Optional[Dict[str, float]],
) -> float:
    if not prev_dist or not curr_dist:
        return 0.0
    keys = set(prev_dist) | set(curr_dist)
    if not keys:
        return 0.0
    num = 0.0
    den = 0.0
    for k in keys:
        a = prev_dist.get(k, 0.0)
        b = curr_dist.get(k, 0.0)
        num += min(a, b)
        den += max(a, b)
    if den <= 0.0:
        return 0.0
    return float(num / den)


def _sum_history(window: int, slot_index: int, history: deque[Tuple[int, int]]) -> int:
    if slot_index <= 0 or window <= 0:
        return 0
    start_slot = max(1, slot_index - window + 1)
    return sum(count for slot_id, count in history if slot_id >= start_slot)


def _get_history_features(
    obj_id: str,
    slot_index: int,
    freq_history: Dict[str, deque[Tuple[int, int]]],
    cum_counts: Dict[str, int],
    windows: Dict[str, int],
) -> Tuple[int, int, int, int]:
    history = freq_history.get(obj_id)
    if history is None:
        return 0, 0, 0, int(cum_counts.get(obj_id, 0))
    f_short = _sum_history(windows["short"], slot_index, history)
    f_mid = _sum_history(windows["mid"], slot_index, history)
    f_long = _sum_history(windows["long"], slot_index, history)
    f_cum = int(cum_counts.get(obj_id, 0))
    return f_short, f_mid, f_long, f_cum


def _build_feature_vector(
    feature_set: str,
    gaps: List[float],
    history_features: Tuple[int, int, int, int],
) -> List[float]:
    if feature_set == "A0":
        return list(gaps)
    if feature_set == "A1":
        return list(gaps)
    f_short, f_mid, f_long, f_cum = history_features
    if feature_set == "A2":
        return list(gaps) + [float(f_short), float(f_mid), float(f_long), float(f_cum)]
    if feature_set == "A3":
        return [float(f_short), float(f_mid), float(f_long), float(f_cum)]
    raise ValueError(f"Unknown feature_set: {feature_set}")


def _select_top_ids_from_stats(
    slot_stats: Dict[str, Dict],
    top_ratio: float,
    label_topk_rounding: str,
    label_tie_break: str,
) -> Tuple[set[str], Dict[str, Any]]:
    label_info: Dict[str, Any] = {
        "n_objects": 0,
        "k_target": 0,
        "k_actual": 0,
        "pos_ratio": 0.0,
        "top_ratio": float(top_ratio),
        "rounding": label_topk_rounding,
        "tie_break": label_tie_break,
        "freq_at_k": None,
        "tie_count": 0,
        "tie_rate": 0.0,
    }
    if not slot_stats:
        return set(), label_info

    items: List[Tuple[str, Dict]] = list(slot_stats.items())
    items.sort(key=lambda kv: kv[1]["freq"], reverse=True)

    n_objects = len(items)
    k = _compute_topk_k(n_objects, top_ratio, label_topk_rounding)

    kth_freq = items[k - 1][1]["freq"]
    tie_count = sum(1 for _, stats in items if stats["freq"] == kth_freq)
    if label_tie_break == "include_ties":
        top_ids = {obj_id for obj_id, stats in items if stats["freq"] >= kth_freq}
    elif label_tie_break == "none":
        top_ids = {items[i][0] for i in range(k)}
    else:
        raise ValueError(f"Unknown label_tie_break: {label_tie_break}")

    label_info.update(
        {
            "n_objects": int(n_objects),
            "k_target": int(k),
            "k_actual": int(len(top_ids)),
            "pos_ratio": float(len(top_ids) / n_objects) if n_objects > 0 else 0.0,
            "freq_at_k": int(kth_freq),
            "tie_count": int(tie_count),
            "tie_rate": float(tie_count / n_objects) if n_objects > 0 else 0.0,
        }
    )
    return top_ids, label_info


def _update_freq_history(
    slot_index: int,
    slot_stats: Dict[str, Dict[str, Any]],
    freq_history: Dict[str, deque[Tuple[int, int]]],
    cum_counts: Dict[str, int],
    long_window: int,
) -> None:
    cutoff = slot_index - long_window + 1
    for obj_id, stats in slot_stats.items():
        count = int(stats.get("freq", 0))
        if count <= 0:
            continue
        history = freq_history.setdefault(obj_id, deque())
        while history and history[0][0] < cutoff:
            history.popleft()
        history.append((slot_index, count))
        cum_counts[obj_id] = int(cum_counts.get(obj_id, 0)) + count


def build_slot_dataset_from_stats(
    slot_stats: Dict[str, Dict],
    top_ratio: float,
    num_gaps: int,
    missing_gap_value: float,
    feature_set: str,
    history_slot_index: int,
    freq_history: Dict[str, deque[Tuple[int, int]]],
    cum_counts: Dict[str, int],
    label_topk_rounding: str = "floor",
    label_tie_break: str = "none",
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Membangun dataset D_t untuk Learn++.NSE dari statistik per-objek dalam satu slot.

    slot_stats: dict[object_id] -> {"freq": int, "last_gaps": List[float]}
    top_ratio: misalnya 0.20 (top-20% populer)
    num_features: L = jumlah fitur gap (6 di eksperimen utama)

    return: list of dict:
      {
          "x": List[float],  # Gap1..GapL
          "y": int,          # 0/1 (popularitas)
          "freq": int,
          "object_id": str,
      }
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    num_features = _get_feature_dim(num_gaps, feature_set)

    label_info: Dict[str, Any] = {}
    if not slot_stats:
        return [], label_info

    items: List[Tuple[str, Dict]] = list(slot_stats.items())
    # urutkan berdasarkan freq menurun (popularitas per-slot)
    items.sort(key=lambda kv: kv[1]["freq"], reverse=True)

    top_ids, label_info = _select_top_ids_from_stats(
        slot_stats,
        top_ratio,
        label_topk_rounding,
        label_tie_break,
    )

    dataset: List[Dict] = []
    for obj_id, stats in items:
        gaps = stats["last_gaps"]
        if len(gaps) != num_gaps:
            if len(gaps) < num_gaps:
                gaps = list(gaps) + [missing_gap_value] * (num_gaps - len(gaps))
            else:
                gaps = list(gaps[:num_gaps])

        history_feats = _get_history_features(
            obj_id,
            history_slot_index,
            freq_history,
            cum_counts,
            FEATURE_WINDOWS,
        )
        features = _build_feature_vector(
            feature_set,
            gaps,
            history_feats,
        )
        if len(features) != num_features:
            raise ValueError(
                f"feature length mismatch: expected {num_features}, got {len(features)}"
            )
        hist_len = sum(1 for g in gaps if g != missing_gap_value)

        y = 1 if obj_id in top_ids else 0
        dataset.append(
            {
                "x": features,
                "y": y,
                "freq": stats["freq"],
                "object_id": obj_id,
                "hist_len": hist_len,
            }
        )

    return dataset, label_info


# ---------------------------------------------------------------------------
# Satu run IL+LRU untuk satu kapasitas cache
# ---------------------------------------------------------------------------

def run_single_capacity(
    trace_path: str,
    total_requests: int,
    warmup_requests: int,
    slot_size: int,
    capacity_objects: int,
    feature_set: str,
    base_learner: str = "nb",
    slot_log_path: Optional[str] = None,
) -> Tuple[CacheStats, Dict[str, Any]]:
    """
    Menjalankan satu eksperimen IL+LRU untuk satu kapasitas cache.

    - Stream trace sekali:
        * slot demi slot (slot_size dari config, misal 100k)
        * warm-up pada prefix warmup_requests pertama (tanpa cache)
        * caching pada sisa request dengan IL+LRU
    - Learn++.NSE di-update setiap akhir slot dengan label top-20% per-slot.
    """
    # TraceReader otomatis deteksi:
    # - jika trace_path file .gz -> mode raw_gz
    # - jika trace_path direktori .parquet -> mode parquet
    reader = TraceReader(path=trace_path, max_rows=total_requests)
    req_iter = reader.iter_requests()

    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    num_gaps = IL_NUM_GAPS
    num_features = _get_feature_dim(num_gaps, feature_set)

    feature_table = FeatureTable(
        L=num_gaps,
        missing_gap_value=IL_MISSING_GAP_VALUE,
    )
    base_learner = (base_learner or "nb").lower()
    if base_learner == "nb":
        base_factory = GaussianNaiveBayes
    elif base_learner == "svm":
        base_factory = LinearSVMWrapper
    elif base_learner == "dt":
        base_factory = DecisionTreeWrapper
    else:
        raise ValueError(f"Unknown base_learner: {base_learner}")

    il_model = LearnNSE(
        n_features=num_features,
        a=IL_SIGMOID_A,
        b=IL_SIGMOID_B,
        max_learners=IL_MAX_CLASSIFIERS,
        base_learner_factory=base_factory,
    )
    cache = LRUCache(capacity_objects=capacity_objects)
    stats = CacheStats(capacity_objects=capacity_objects)

    label_topk_rounding = IL_LABEL_TOPK_ROUNDING
    label_tie_break = IL_LABEL_TIE_BREAK
    admission_capacity_alpha = float(ADMISSION_CAPACITY_ALPHA)

    total_slots_expected = (total_requests + slot_size - 1) // slot_size
    pbar = tqdm(total=total_slots_expected, desc="Processing slots", unit="slot")

    global_idx = 0          # counter global request (0-based)
    slot_req_count = 0      # jumlah request dalam slot saat ini
    slot_index = 0          # 1-based untuk logging
    slot_stats: Dict[str, Dict] = {}  # per-slot: freq, last_gaps
    prev_ts: Optional[float] = None
    slot_cache_requests = 0
    slot_cache_hits = 0
    slot_cache_misses = 0
    slot_hits_from_admitted = 0
    total_hits_from_admitted = 0
    slot_admit_requests = 0
    slot_reject_requests = 0
    total_admit_requests = 0
    total_reject_requests = 0
    total_miss_requests = 0
    total_miss_candidates = 0
    total_admit_budget = 0
    total_admit_selected = 0
    total_admit_applied = 0
    total_admit_true_popular = 0
    total_pollution_count = 0
    total_pollution_total = 0
    total_update_time_s = 0.0
    total_update_slots = 0
    total_slots_processed = 0
    total_cache_slots = 0
    last_pollution_finalized = 0
    slot_log_file = open(slot_log_path, "w", encoding="utf-8") if slot_log_path else None
    prev_topk_dist: Optional[Dict[str, float]] = None
    drift_ema = 0.0
    drift_mean_ema = 0.0
    drift_var_ema = 0.0
    drift_stats_count = 0
    drift_raw_sum = 0.0
    drift_norm_sum = 0.0
    jsd_sum = 0.0
    overlap_sum = 0.0
    miss_rate_sum = 0.0
    alpha_sum = 0.0
    alpha_min_seen = None
    alpha_max_seen = None
    score_eval_slots = 0
    score_eval_n_sum = 0
    score_eval_pos_sum = 0
    score_auc_sum = 0.0
    score_ap_sum = 0.0
    score_precK_sum = 0.0
    score_auc_count = 0
    score_ap_count = 0
    score_precK_count = 0
    score_acc_sum = 0.0
    score_acc_count = 0
    score_precK_hr_next_n = 0
    score_precK_hr_next_sum_x = 0.0
    score_precK_hr_next_sum_y = 0.0
    score_precK_hr_next_sum_x2 = 0.0
    score_precK_hr_next_sum_y2 = 0.0
    score_precK_hr_next_sum_xy = 0.0
    prev_score_precK = None
    score_gate_k_sum = 0
    score_gate_applied_slots = 0
    fill_phase_slots = 0
    fill_min_budget_sum = 0
    pressure_mult_sum = 0.0
    pressure_mult_count = 0
    score_spread_ema = None
    score_spread_sum = 0.0
    score_spread_count = 0
    quality_mult_sum = 0.0
    quality_mult_count = 0
    precision_eff_sum = 0.0
    precision_eff_count = 0
    precision_multiplier_sum = 0.0
    precision_multiplier_count = 0
    boundary_margin_ema = None
    score_precK_ema = None
    candidate_pos_rate_ema = None
    slot_diagnostic_records: List[Dict[str, Any]] = []
    precision_multiplier_pool: List[float] = []
    quality_multiplier_pool: List[float] = []
    diagnostic_groups_by_future_slot: Dict[int, List[Dict[str, Any]]] = {}
    fill_phase_by_slot: Dict[int, bool] = {}
    postfill_cache_requests = 0
    postfill_cache_hits = 0
    postfill_selected_admissions = 0
    postfill_applied_admissions = 0
    postfill_hits_from_admitted = 0
    postfill_admit_true_popular = 0
    postfill_pollution_count = 0
    postfill_pollution_total = 0
    quality_suppressed_total = 0
    quality_admitted_eval_total = 0
    quality_suppressed_future_eval_total = 0
    quality_admitted_future_eval_total = 0
    quality_suppressed_next_slot_requests_total = 0
    quality_admitted_next_slot_requests_total = 0

    drift_control_mode = (DRIFT_CONTROL_MODE or "scaled").lower()
    if drift_control_mode not in ("scaled", "piecewise", "fixed"):
        raise ValueError(f"Unknown drift_control_mode: {drift_control_mode}")

    configured_budget_match_mode = (BUDGET_MATCH_MODE or "off").lower()
    if configured_budget_match_mode not in (
        "off",
        "uniform",
        "random_slot_throttle",
        "slot_replay",
        "slot_replay_permuted",
    ):
        raise ValueError(f"Unknown BUDGET_MATCH_MODE: {BUDGET_MATCH_MODE}")
    budget_replay_enabled = configured_budget_match_mode in (
        "slot_replay",
        "slot_replay_permuted",
    )
    budget_replay_permuted = configured_budget_match_mode == "slot_replay_permuted"
    budget_replay_schedule: Dict[int, int] = {}
    budget_replay_original_schedule: Dict[int, int] = {}
    budget_replay_metadata: Dict[str, Any] = {}
    budget_replay_permutation_metadata: Dict[str, Any] = {
        "original_total_budget": None,
        "permuted_total_budget": None,
        "histogram_preserved": None,
        "same_slot_budget_rate_against_full": None,
    }
    budget_replay_clamp_count = 0
    budget_replay_evaluated_slots = 0
    budget_replay_exact_slot_match_count = 0
    budget_replay_total_target_budget = 0
    budget_replay_total_realized_selected = 0
    if budget_replay_enabled:
        budget_replay_original_schedule, budget_replay_metadata = _load_budget_replay_with_metadata(
            str(BUDGET_REPLAY_PATH or "")
        )
        budget_replay_schedule = dict(budget_replay_original_schedule)
        if budget_replay_permuted:
            budget_replay_schedule, budget_replay_permutation_metadata = _permute_budget_replay(
                budget_replay_original_schedule,
                int(BUDGET_REPLAY_PERMUTE_SEED),
            )
    elif BUDGET_REPLAY_PATH:
        raise ValueError(
            "BUDGET_REPLAY_PATH was provided, but BUDGET_MATCH_MODE is not a replay mode."
        )

    freq_history: Dict[str, deque[Tuple[int, int]]] = {}
    cum_counts: Dict[str, int] = {}

    applied_history: deque[Tuple[int, set[str]]] = deque(maxlen=3)

    admitted_by_slot: Dict[int, set[str]] = {}
    admitted_hit_by_slot: Dict[int, set[str]] = {}
    active_admit_slots_by_obj: Dict[str, set[int]] = {}
    slot_miss_obj_ids: List[str] = []
    slot_miss_features: List[List[float]] = []
    pending_admit_set: set[str] = set()
    pending_admit_source_slot: Optional[int] = None
    applied_admit_set: set[str] = set()
    applied_admit_source_slot: Optional[int] = None

    def _register_admission(obj_id: str, slot_num: int) -> None:
        admitted_by_slot.setdefault(slot_num, set()).add(obj_id)
        active_admit_slots_by_obj.setdefault(obj_id, set()).add(slot_num)

    def _mark_admit_hit(obj_id: str) -> None:
        slots = active_admit_slots_by_obj.get(obj_id)
        if not slots:
            return
        for slot_num in slots:
            admitted_hit_by_slot.setdefault(slot_num, set()).add(obj_id)

    def _finalize_pollution(slot_num: int) -> Tuple[int, int]:
        admitted = admitted_by_slot.get(slot_num, set())
        hits = admitted_hit_by_slot.get(slot_num, set())
        total = len(admitted)
        polluted = len(admitted - hits)
        return polluted, total

    def _prune_active_slots(expired_slot: int) -> None:
        for obj_id in admitted_by_slot.get(expired_slot, set()):
            active = active_admit_slots_by_obj.get(obj_id)
            if not active:
                continue
            active.discard(expired_slot)
            if not active:
                del active_admit_slots_by_obj[obj_id]

    def _apply_pending_admits(next_slot_num: int) -> None:
        nonlocal pending_admit_set
        nonlocal pending_admit_source_slot
        nonlocal applied_admit_set
        nonlocal applied_admit_source_slot
        nonlocal total_admit_applied
        # Eq. (2) mapping:
        # - Paper A_t          -> pending_admit_set (decided at end of slot t)
        # - Paper S_{t+1}      -> cache state after this boundary insertion
        # - Paper "apply at t+1" -> this function, called before serving next slot
        # In-slot recency updates still follow LRU via cache.access(...).
        if not pending_admit_set:
            applied_admit_set = set()
            applied_admit_source_slot = None
            return
        applied_admit_set = set(pending_admit_set)
        applied_admit_source_slot = pending_admit_source_slot
        pending_admit_set = set()
        pending_admit_source_slot = None
        total_admit_applied += len(applied_admit_set)
        if applied_admit_set:
            applied_history.append((next_slot_num, set(applied_admit_set)))
        for obj_id in applied_admit_set:
            cache.insert(obj_id)
            _register_admission(obj_id, next_slot_num)

    def _write_slot_log(payload: Dict[str, Any]) -> None:
        if slot_log_file is None:
            return
        slot_log_file.write(json.dumps(payload) + "\n")
        slot_log_file.flush()

    def _finalize_slot() -> None:
        nonlocal slot_index
        nonlocal slot_stats
        nonlocal slot_cache_requests
        nonlocal slot_cache_hits
        nonlocal slot_cache_misses
        nonlocal slot_hits_from_admitted
        nonlocal slot_miss_obj_ids
        nonlocal slot_miss_features
        nonlocal slot_admit_requests
        nonlocal slot_reject_requests
        nonlocal pending_admit_set
        nonlocal pending_admit_source_slot
        nonlocal applied_admit_set
        nonlocal applied_admit_source_slot
        nonlocal total_admit_selected
        nonlocal total_admit_requests
        nonlocal total_reject_requests
        nonlocal total_miss_requests
        nonlocal total_miss_candidates
        nonlocal total_admit_budget
        nonlocal total_admit_true_popular
        nonlocal total_pollution_count
        nonlocal total_pollution_total
        nonlocal total_update_time_s
        nonlocal total_update_slots
        nonlocal total_slots_processed
        nonlocal total_cache_slots
        nonlocal last_pollution_finalized
        nonlocal prev_topk_dist
        nonlocal drift_ema
        nonlocal drift_mean_ema
        nonlocal drift_var_ema
        nonlocal drift_stats_count
        nonlocal drift_raw_sum
        nonlocal drift_norm_sum
        nonlocal jsd_sum
        nonlocal overlap_sum
        nonlocal miss_rate_sum
        nonlocal alpha_sum
        nonlocal alpha_min_seen
        nonlocal alpha_max_seen
        nonlocal score_eval_slots
        nonlocal score_eval_n_sum
        nonlocal score_eval_pos_sum
        nonlocal score_auc_sum
        nonlocal score_ap_sum
        nonlocal score_precK_sum
        nonlocal score_auc_count
        nonlocal score_ap_count
        nonlocal score_precK_count
        nonlocal score_acc_sum
        nonlocal score_acc_count
        nonlocal score_precK_hr_next_n
        nonlocal score_precK_hr_next_sum_x
        nonlocal score_precK_hr_next_sum_y
        nonlocal score_precK_hr_next_sum_x2
        nonlocal score_precK_hr_next_sum_y2
        nonlocal score_precK_hr_next_sum_xy
        nonlocal prev_score_precK
        nonlocal score_gate_k_sum
        nonlocal score_gate_applied_slots
        nonlocal fill_phase_slots
        nonlocal fill_min_budget_sum
        nonlocal pressure_mult_sum
        nonlocal pressure_mult_count
        nonlocal score_spread_ema
        nonlocal score_spread_sum
        nonlocal score_spread_count
        nonlocal quality_mult_sum
        nonlocal quality_mult_count
        nonlocal precision_eff_sum
        nonlocal precision_eff_count
        nonlocal precision_multiplier_sum
        nonlocal precision_multiplier_count
        nonlocal boundary_margin_ema
        nonlocal score_precK_ema
        nonlocal candidate_pos_rate_ema
        nonlocal applied_history
        nonlocal slot_diagnostic_records
        nonlocal precision_multiplier_pool
        nonlocal quality_multiplier_pool
        nonlocal diagnostic_groups_by_future_slot
        nonlocal fill_phase_by_slot
        nonlocal postfill_cache_requests
        nonlocal postfill_cache_hits
        nonlocal postfill_selected_admissions
        nonlocal postfill_applied_admissions
        nonlocal postfill_hits_from_admitted
        nonlocal postfill_admit_true_popular
        nonlocal postfill_pollution_count
        nonlocal postfill_pollution_total
        nonlocal quality_suppressed_total
        nonlocal quality_admitted_eval_total
        nonlocal quality_suppressed_future_eval_total
        nonlocal quality_admitted_future_eval_total
        nonlocal quality_suppressed_next_slot_requests_total
        nonlocal quality_admitted_next_slot_requests_total
        nonlocal budget_replay_clamp_count
        nonlocal budget_replay_evaluated_slots
        nonlocal budget_replay_exact_slot_match_count
        nonlocal budget_replay_total_target_budget
        nonlocal budget_replay_total_realized_selected

        if not slot_stats:
            return

        D_t, label_info = build_slot_dataset_from_stats(
            slot_stats,
            IL_POP_TOP_PERCENT,
            num_gaps,
            IL_MISSING_GAP_VALUE,
            feature_set,
            slot_index,
            freq_history,
            cum_counts,
            label_topk_rounding,
            label_tie_break,
        )

        slot_index += 1
        slot_num = slot_index
        total_slots_processed += 1
        warmup_slots = warmup_requests // slot_size
        is_cache_slot = slot_num > warmup_slots

        matured_quality_groups = diagnostic_groups_by_future_slot.pop(slot_num, [])
        quality_admitted_eval_count = 0
        quality_suppressed_eval_count = 0
        quality_admitted_next_slot_requests = 0
        quality_suppressed_next_slot_requests = 0
        for group in matured_quality_groups:
            admitted_group = group.get("admitted_by_full", set())
            suppressed_group = group.get("suppressed_by_quality", set())
            quality_admitted_eval_count += len(admitted_group)
            quality_suppressed_eval_count += len(suppressed_group)
            quality_admitted_next_slot_requests += sum(
                int(slot_stats.get(obj_id, {}).get("freq", 0))
                for obj_id in admitted_group
            )
            quality_suppressed_next_slot_requests += sum(
                int(slot_stats.get(obj_id, {}).get("freq", 0))
                for obj_id in suppressed_group
            )
        quality_future_eval_group_count = len(matured_quality_groups)
        quality_admitted_next_slot_request_rate = (
            float(quality_admitted_next_slot_requests / quality_admitted_eval_count)
            if quality_admitted_eval_count > 0
            else None
        )
        quality_suppressed_next_slot_request_rate = (
            float(quality_suppressed_next_slot_requests / quality_suppressed_eval_count)
            if quality_suppressed_eval_count > 0
            else None
        )
        quality_admitted_to_suppressed_request_ratio = (
            float(
                quality_admitted_next_slot_request_rate
                / quality_suppressed_next_slot_request_rate
            )
            if (
                quality_admitted_next_slot_request_rate is not None
                and quality_suppressed_next_slot_request_rate is not None
                and quality_suppressed_next_slot_request_rate > 0.0
            )
            else None
        )
        quality_admitted_next_slot_requests_total += quality_admitted_next_slot_requests
        quality_suppressed_next_slot_requests_total += quality_suppressed_next_slot_requests
        quality_admitted_future_eval_total += quality_admitted_eval_count
        quality_suppressed_future_eval_total += quality_suppressed_eval_count

        # Eq. (3) mapping:
        # - Paper P_t          -> pos_ids (objects with y=1 in current D_t)
        # - Paper A_{t-l}^{applied} -> apply_set from applied_history
        # - Paper p_t^{(l)}    -> precision_by_lag[lag]
        # - Paper p_t^{eff}    -> precision_eff from the configured lag aggregator.
        # This code uses the implementation-aligned lagged precision proxy.
        pos_ids: set[str] = set()
        admit_true_applied = None
        admit_precision = None
        precision_lag0 = None
        precision_lag1 = None
        precision_lag2 = None
        precision_eff = None
        if applied_admit_set:
            pos_ids = {row["object_id"] for row in D_t if row["y"] == 1}
            admit_true_applied = len(applied_admit_set & pos_ids)
            admit_precision = (
                float(admit_true_applied / len(applied_admit_set))
                if applied_admit_set
                else None
            )
        if not pos_ids:
            pos_ids = {row["object_id"] for row in D_t if row["y"] == 1}
        precision_by_lag: Dict[int, float] = {}
        for apply_slot, apply_set in applied_history:
            if not apply_set:
                continue
            lag = slot_num - apply_slot
            if lag < 0:
                continue
            precision_by_lag[lag] = float(len(apply_set & pos_ids) / len(apply_set))
        precision_lag0 = precision_by_lag.get(0)
        precision_lag1 = precision_by_lag.get(1)
        precision_lag2 = precision_by_lag.get(2)
        precision_eff = _aggregate_effective_precision(
            precision_by_lag,
            EFFECTIVE_PRECISION_LAG_MODE,
            EFFECTIVE_PRECISION_FIXED_LAG,
            EFFECTIVE_PRECISION_LAG_WEIGHTS,
        )
        admit_budget = 0
        admit_selected = 0
        miss_requests = int(slot_cache_misses)
        total_miss_requests += miss_requests
        miss_rate = float(miss_requests / slot_cache_requests) if slot_cache_requests > 0 else 0.0
        jsd_freq = 0.0
        overlap_w = 0.0
        drift_raw = 0.0
        drift_norm = 0.0
        alpha_base = admission_capacity_alpha
        alpha_for_slot = admission_capacity_alpha
        alpha_after_precision_unclipped = admission_capacity_alpha
        precision_target_t = _compute_precision_target(
            None,
            candidate_pos_rate_ema,
            ADMISSION_PRECISION_TARGET_MODE,
        )
        precision_multiplier_t = 1.0
        precision_multiplier_raw = 1.0
        precision_multiplier_normal = 1.0
        capacity_scale = 1.0
        fill_phase = False
        fill_min_budget = 0
        pressure_mult = 1.0
        quality_mult = 1.0
        score_quality_mult_raw = 1.0
        score_quality_action = "none"
        score_spread = 0.0
        score_spread_ema_prev = score_spread_ema
        score_spread_ema_after = score_spread_ema
        boundary_margin = 0.0
        boundary_margin_ema_prev = boundary_margin_ema
        boundary_margin_ema_after = boundary_margin_ema
        score_precK_ema_prev = score_precK_ema
        score_precK_ema_after = score_precK_ema
        candidate_pos_rate = None
        candidate_pos_rate_ema_prev = candidate_pos_rate_ema
        candidate_pos_rate_ema_after = candidate_pos_rate_ema
        score_quality_calibration = 1.0
        score_quality_target_precK = ADMISSION_PRECISION_TARGET
        quality_control_mode = (QUALITY_CONTROL_MODE or "normal").lower()
        precision_control_override_mode = (PRECISION_CONTROL_OVERRIDE_MODE or "normal").lower()
        budget_match_mode = configured_budget_match_mode
        budget_match_multiplier_t = 1.0
        budget_match_action = "none"
        budget_replay_target_budget = None
        budget_replay_effective_budget = None
        budget_replay_clamped = False
        budget_replay_clamp_from = None
        budget_replay_missing_slot = False
        budget_replay_exact_slot_match = None
        quality_suppressed_count = 0
        quality_suppressed_mean_score = None
        quality_admitted_mean_score = None
        quality_suppressed_pos_label_rate_current = None
        quality_admitted_pos_label_rate_current = None

        if is_cache_slot:
            total_cache_slots += 1
            top_k = int(label_info.get("k_actual", 0))
            curr_topk_dist: Optional[Dict[str, float]] = None
            if top_k > 0:
                top_k = min(top_k, len(D_t))
                top_freq_total = sum(int(row["freq"]) for row in D_t[:top_k])
                if top_freq_total > 0:
                    curr_topk_dist = {
                        row["object_id"]: float(row["freq"] / top_freq_total)
                        for row in D_t[:top_k]
                    }
            jsd_freq = _compute_jsd_identity(prev_topk_dist, curr_topk_dist)
            overlap_w = _weighted_jaccard(prev_topk_dist, curr_topk_dist)
            drift_raw = DRIFT_WEIGHT_JSD * jsd_freq + DRIFT_WEIGHT_OVERLAP * (1.0 - overlap_w)
            drift_ema = DRIFT_EMA_ALPHA * drift_raw + (1.0 - DRIFT_EMA_ALPHA) * drift_ema
            if drift_stats_count == 0:
                drift_mean_ema = drift_ema
                drift_var_ema = 0.0
            else:
                drift_mean_ema = (
                    (1.0 - DRIFT_STATS_ALPHA) * drift_mean_ema
                    + DRIFT_STATS_ALPHA * drift_ema
                )
                diff = drift_ema - drift_mean_ema
                drift_var_ema = (
                    (1.0 - DRIFT_STATS_ALPHA) * drift_var_ema
                    + DRIFT_STATS_ALPHA * diff * diff
                )
            drift_stats_count += 1
            drift_std = math.sqrt(drift_var_ema) if drift_var_ema > 0.0 else 0.0
            z = (drift_ema - drift_mean_ema) / (drift_std + DRIFT_NORM_EPS)
            if z < -DRIFT_Z_CLIP:
                z = -DRIFT_Z_CLIP
            elif z > DRIFT_Z_CLIP:
                z = DRIFT_Z_CLIP
            drift_norm = 1.0 / (1.0 + math.exp(-z))
            prev_topk_dist = curr_topk_dist
            capacity_scale = 1.0
            slot_unique = int(label_info.get("n_objects", 0))
            if slot_unique > 0:
                ratio = capacity_objects / slot_unique
                if ratio < CAPACITY_ALPHA_SCALE_MIN:
                    ratio = CAPACITY_ALPHA_SCALE_MIN
                elif ratio > 1.0:
                    ratio = 1.0
                capacity_scale = ratio
            fill_threshold = int(math.ceil(FILL_RATIO * capacity_objects))
            if len(cache) < fill_threshold:
                fill_phase = True
                fill_min_budget = int(math.ceil(FILL_RATE * capacity_objects))

            drift_norm_p = drift_norm ** max(DRIFT_NORM_POWER, 0.0)
            scale = capacity_scale if DRIFT_USE_CAPACITY_SCALE else 1.0
            if drift_control_mode == "fixed":
                alpha_base = admission_capacity_alpha * scale
            elif drift_control_mode == "piecewise":
                alpha_base = (DRIFT_ALPHA_HIGH if drift_norm_p >= DRIFT_THRESHOLD else DRIFT_ALPHA_LOW) * scale
            else:
                alpha_base = admission_capacity_alpha * (1.0 + DRIFT_GAIN * drift_norm_p) * scale
            # Eq. (6) base-intensity mapping:
            # - Paper chi_t / zeta_t -> capacity_scale when DRIFT_USE_CAPACITY_SCALE is on
            # - Paper alpha_t^{base} -> alpha_base after scaling and clipping
            # Drift terms remain in code, but with current config DRIFT_GAIN=0 they do
            # not change the budget-driving formula beyond the bounded alpha range.
            if alpha_base < DRIFT_ALPHA_MIN:
                alpha_base = DRIFT_ALPHA_MIN
            elif alpha_base > DRIFT_ALPHA_MAX:
                alpha_base = DRIFT_ALPHA_MAX
            guard_precision = precision_eff if precision_eff is not None else admit_precision
            precision_target_t = _compute_precision_target(
                None,
                candidate_pos_rate_ema,
                ADMISSION_PRECISION_TARGET_MODE,
            )
            precision_multiplier_raw = _compute_precision_multiplier(
                guard_precision,
                precision_target_t,
                PRECISION_CONTROL_MODE,
            )
            precision_multiplier_normal = precision_multiplier_raw
            precision_multiplier_t = precision_multiplier_raw
            if precision_control_override_mode == "random" and PRECISION_CONTROL_MODE != "off":
                rng = random.Random(int(CONTROL_RANDOM_SEED) + slot_num * 1009 + 17)
                precision_multiplier_t = rng.uniform(
                    float(ADMISSION_PRECISION_MULT_MIN),
                    float(ADMISSION_PRECISION_MULT_MAX),
                )
            elif precision_control_override_mode == "permuted" and PRECISION_CONTROL_MODE != "off":
                if precision_multiplier_pool:
                    rng = random.Random(int(CONTROL_RANDOM_SEED) + slot_num * 1009 + 23)
                    precision_multiplier_t = precision_multiplier_pool[
                        rng.randrange(len(precision_multiplier_pool))
                    ]
                else:
                    precision_multiplier_t = precision_multiplier_raw
            elif precision_control_override_mode not in ("normal", "random", "permuted"):
                raise ValueError(
                    f"Unknown PRECISION_CONTROL_OVERRIDE_MODE: {PRECISION_CONTROL_OVERRIDE_MODE}"
                )
            if PRECISION_CONTROL_MODE != "off" and _is_valid_number(precision_multiplier_normal):
                precision_multiplier_pool.append(float(precision_multiplier_normal))
            alpha_after_precision_unclipped = alpha_base * precision_multiplier_t
            alpha_candidate = alpha_after_precision_unclipped
            alpha_floor = alpha_base * DRIFT_ALPHA_FLOOR_MULT
            # Eq. (6) final-intensity mapping:
            # - Paper p_t^{eff}      -> guard_precision / precision_eff
            # - Paper p_tar          -> ADMISSION_PRECISION_TARGET
            # - Paper xi_p           -> configured positive/negative precision gains
            # - Paper nu             -> DRIFT_ALPHA_FLOOR_MULT
            # - Paper alpha_t        -> alpha_for_slot
            alpha_for_slot = alpha_candidate if alpha_candidate >= alpha_floor else alpha_floor
            if alpha_for_slot < DRIFT_ALPHA_MIN:
                alpha_for_slot = DRIFT_ALPHA_MIN
            elif alpha_for_slot > DRIFT_ALPHA_MAX:
                alpha_for_slot = DRIFT_ALPHA_MAX

        if is_cache_slot:
            drift_raw_sum += drift_raw
            drift_norm_sum += drift_norm
            jsd_sum += jsd_freq
            overlap_sum += overlap_w
            miss_rate_sum += miss_rate
            alpha_sum += alpha_for_slot
            if alpha_min_seen is None or alpha_for_slot < alpha_min_seen:
                alpha_min_seen = alpha_for_slot
            if alpha_max_seen is None or alpha_for_slot > alpha_max_seen:
                alpha_max_seen = alpha_for_slot
        candidate_scores: Dict[str, float] = {}
        candidate_ids: List[str] = []
        # Eq. (1) mapping:
        # - Paper C_t          -> unique miss-object pool built in candidate_ids
        # - Paper x_{t,j}(o)   -> per-miss feature vector in slot_miss_features
        # - Paper f_{t-1}(.)   -> il_model.score_batch(...) using pre-update model
        # - Paper s_t(o)       -> candidate_scores[obj_id] after max-per-object merge
        if slot_miss_features:
            X_miss = np.asarray(slot_miss_features, dtype=float)
            miss_scores = il_model.score_batch(X_miss)
            for obj_id, score in zip(slot_miss_obj_ids, miss_scores):
                prev = candidate_scores.get(obj_id)
                score_val = float(score)
                if prev is None:
                    candidate_scores[obj_id] = score_val
                    candidate_ids.append(obj_id)
                elif score_val > prev:
                    candidate_scores[obj_id] = score_val
        miss_candidates = int(len(candidate_ids))
        total_miss_candidates += miss_candidates

        candidate_score_list = [candidate_scores[obj_id] for obj_id in candidate_ids]
        global_quality = _compute_score_spread(candidate_score_list, SCORE_SPREAD_Q)
        score_spread = global_quality
        score_eval_n = miss_candidates
        score_eval_pos = 0
        score_auc = float("nan")
        score_ap = float("nan")
        score_precK = float("nan")
        score_acc = float("nan")
        y_miss: List[int] = []
        if miss_candidates > 0:
            label_map = {row["object_id"]: int(row["y"]) for row in D_t}
            y_miss = [label_map.get(obj_id, 0) for obj_id in candidate_ids]
            score_eval_pos = int(sum(y_miss))
            candidate_pos_rate = float(score_eval_pos / miss_candidates)
            if score_eval_pos > 0 and score_eval_pos < miss_candidates:
                score_auc = _auc_roc(y_miss, candidate_score_list)
            if score_eval_pos > 0:
                score_ap = _avg_precision(y_miss, candidate_score_list)
                score_precK = _precision_at_k(y_miss, candidate_score_list, score_eval_pos)
            y_pred = [1 if s >= 0.5 else 0 for s in candidate_score_list]
            score_acc = float(
                sum(1 for yp, yt in zip(y_pred, y_miss) if yp == yt) / miss_candidates
            )
            if score_eval_pos > 0:
                score_eval_slots += 1
                score_eval_n_sum += miss_candidates
                score_eval_pos_sum += score_eval_pos
                if score_auc == score_auc:
                    score_auc_sum += score_auc
                    score_auc_count += 1
                if score_ap == score_ap:
                    score_ap_sum += score_ap
                    score_ap_count += 1
                if score_precK == score_precK:
                    score_precK_sum += score_precK
                    score_precK_count += 1
            if score_acc == score_acc:
                score_acc_sum += score_acc
                score_acc_count += 1

        base_pressure_mult = 1.0 + PRESSURE_MISS_GAMMA * miss_rate
        score_gate_k = (
            max(1, int(math.ceil(SCORE_GATE_TOP_PERCENT * miss_candidates)))
            if miss_candidates > 0
            else 0
        )
        raw_budget_precision_pre_quality = _compute_admission_budget(
            slot_cache_misses,
            capacity_objects,
            alpha_for_slot,
            base_pressure_mult,
        )
        precision_budget_before_quality_cap, _ = _apply_budget_bounds(
            raw_budget_precision_pre_quality,
            miss_candidates,
            score_gate_k,
            FILL_RATIO,
            fill_min_budget,
            fill_phase=fill_phase,
        )

        quality_info = _compute_score_quality_multiplier(
            candidate_score_list,
            precision_budget_before_quality_cap,
            score_spread_ema_prev,
            boundary_margin_ema_prev,
            score_precK_ema_prev,
            candidate_pos_rate_ema_prev,
            SCORE_QUALITY_SIGNAL_MODE,
            SCORE_QUALITY_CONTROL_ROLE,
        )
        quality_mult = float(quality_info["score_quality_mult"])
        score_quality_mult_normal = quality_mult
        score_quality_mult_raw = float(quality_info["score_quality_mult_raw"])
        score_spread = float(quality_info["score_spread"])
        boundary_margin = float(quality_info["boundary_margin"])
        boundary_quality = boundary_margin
        score_quality_calibration = float(quality_info["score_quality_calibration"])
        score_quality_target_precK = float(quality_info["score_quality_target_precK"])
        if quality_control_mode == "random" and SCORE_QUALITY_SIGNAL_MODE != "off":
            upper = 1.0 if SCORE_QUALITY_CONTROL_ROLE == "safety" else float(SCORE_QUALITY_MAX)
            rng = random.Random(int(CONTROL_RANDOM_SEED) + slot_num * 7919 + 31)
            score_quality_mult_raw = rng.uniform(float(SCORE_QUALITY_MIN), upper)
            quality_mult = score_quality_mult_raw
        elif quality_control_mode == "permuted" and SCORE_QUALITY_SIGNAL_MODE != "off":
            if quality_multiplier_pool:
                rng = random.Random(int(CONTROL_RANDOM_SEED) + slot_num * 7919 + 37)
                quality_mult = quality_multiplier_pool[rng.randrange(len(quality_multiplier_pool))]
                score_quality_mult_raw = quality_mult
            else:
                quality_mult = score_quality_mult_normal
        elif quality_control_mode == "inverted" and SCORE_QUALITY_SIGNAL_MODE != "off":
            normal_quality_mult = quality_mult
            score_quality_mult_raw = _safe_div(1.0, max(normal_quality_mult, 1e-12), default=1.0)
            upper = 1.0 if SCORE_QUALITY_CONTROL_ROLE == "safety" else float(SCORE_QUALITY_MAX)
            quality_mult = _clip(score_quality_mult_raw, float(SCORE_QUALITY_MIN), upper)
        elif quality_control_mode not in ("normal", "random", "inverted", "permuted"):
            raise ValueError(f"Unknown QUALITY_CONTROL_MODE: {QUALITY_CONTROL_MODE}")
        if SCORE_QUALITY_SIGNAL_MODE != "off" and _is_valid_number(score_quality_mult_normal):
            quality_multiplier_pool.append(float(score_quality_mult_normal))

        score_spread_ema_after = _ema_after(score_spread_ema_prev, score_spread, SCORE_SPREAD_EMA_ALPHA)
        boundary_margin_ema_after = _ema_after(
            boundary_margin_ema_prev,
            boundary_margin,
            SCORE_SPREAD_EMA_ALPHA,
        )
        score_precK_ema_after = _ema_after(
            score_precK_ema_prev,
            score_precK,
            SCORE_QUALITY_PRECK_EMA_ALPHA,
        )
        candidate_pos_rate_ema_after = _ema_after(
            candidate_pos_rate_ema_prev,
            candidate_pos_rate,
            CANDIDATE_POS_RATE_EMA_ALPHA,
        )
        if miss_candidates > 0:
            score_spread_sum += score_spread
            score_spread_count += 1
            quality_mult_sum += quality_mult
            quality_mult_count += 1
        precision_multiplier_sum += precision_multiplier_t
        precision_multiplier_count += 1

        if budget_match_mode == "off":
            budget_match_multiplier_t = 1.0
            budget_match_action = "none"
        elif budget_match_mode == "uniform":
            budget_match_multiplier_t = max(0.0, float(BUDGET_MATCH_MULTIPLIER))
            budget_match_action = "uniform"
        elif budget_match_mode == "random_slot_throttle":
            target = _clip(float(BUDGET_MATCH_TARGET_MULTIPLIER), 0.0, 1.0)
            rng = random.Random(int(CONTROL_RANDOM_SEED) + slot_num * 6151 + 43)
            if rng.random() <= target:
                budget_match_multiplier_t = 1.0
                budget_match_action = "keep"
            else:
                budget_match_multiplier_t = 0.0
                budget_match_action = "throttle_zero"
        elif budget_match_mode in ("slot_replay", "slot_replay_permuted"):
            budget_match_multiplier_t = 1.0
            budget_match_action = budget_match_mode
        else:
            raise ValueError(f"Unknown BUDGET_MATCH_MODE: {BUDGET_MATCH_MODE}")

        if is_cache_slot:
            # Eq. (7) pressure mapping:
            # - Paper m_t        -> miss_rate = miss_requests / slot_cache_requests
            # - Paper gamma_m    -> PRESSURE_MISS_GAMMA
            # - Paper g_t        -> pressure_mult = (1 + gamma_m * m_t) * u_t
            pressure_mult = base_pressure_mult * quality_mult * budget_match_multiplier_t
            pressure_mult_sum += pressure_mult
            pressure_mult_count += 1
            if fill_phase:
                fill_phase_slots += 1
                fill_min_budget_sum += fill_min_budget

        cont_budget_no_signals = capacity_objects * alpha_base * base_pressure_mult
        cont_budget_precision_only = capacity_objects * alpha_for_slot * base_pressure_mult
        cont_budget_quality_only = capacity_objects * alpha_base * base_pressure_mult * quality_mult
        cont_budget_full = (
            capacity_objects
            * alpha_for_slot
            * base_pressure_mult
            * quality_mult
            * budget_match_multiplier_t
        )
        raw_budget_no_signals = _compute_admission_budget(
            slot_cache_misses,
            capacity_objects,
            alpha_base,
            base_pressure_mult,
        )
        raw_budget_precision_only = _compute_admission_budget(
            slot_cache_misses,
            capacity_objects,
            alpha_for_slot,
            base_pressure_mult,
        )
        raw_budget_quality_only = _compute_admission_budget(
            slot_cache_misses,
            capacity_objects,
            alpha_base,
            base_pressure_mult * quality_mult,
        )
        raw_budget_full = _compute_admission_budget(
            slot_cache_misses,
            capacity_objects,
            alpha_for_slot,
            base_pressure_mult * quality_mult * budget_match_multiplier_t,
        )
        quality_changes_continuous_budget_given_precision = (
            abs(cont_budget_full - cont_budget_precision_only) > 1e-12
        )
        precision_changes_continuous_budget = (
            abs(cont_budget_precision_only - cont_budget_no_signals) > 1e-12
        )
        quality_masked_by_integer_rounding = (
            quality_changes_continuous_budget_given_precision
            and raw_budget_full == raw_budget_precision_only
        )
        precision_masked_by_integer_rounding = (
            precision_changes_continuous_budget
            and raw_budget_precision_only == raw_budget_no_signals
        )
        final_budget_no_signals_same_cap, binding_reason_no_signals = _apply_budget_bounds(
            raw_budget_no_signals,
            miss_candidates,
            score_gate_k,
            FILL_RATIO,
            fill_min_budget,
            fill_phase=fill_phase,
        )
        final_budget_precision_only_same_cap, binding_reason_precision_only = _apply_budget_bounds(
            raw_budget_precision_only,
            miss_candidates,
            score_gate_k,
            FILL_RATIO,
            fill_min_budget,
            fill_phase=fill_phase,
        )
        final_budget_quality_only_same_cap, binding_reason_quality_only = _apply_budget_bounds(
            raw_budget_quality_only,
            miss_candidates,
            score_gate_k,
            FILL_RATIO,
            fill_min_budget,
            fill_phase=fill_phase,
        )
        final_budget_full_same_cap, binding_reason_full = _apply_budget_bounds(
            raw_budget_full,
            miss_candidates,
            score_gate_k,
            FILL_RATIO,
            fill_min_budget,
            fill_phase=fill_phase,
        )
        admit_budget = final_budget_full_same_cap
        if is_cache_slot and budget_replay_enabled:
            if slot_num not in budget_replay_schedule:
                budget_replay_missing_slot = True
                if BUDGET_REPLAY_STRICT:
                    raise KeyError(
                        f"Budget replay missing cache slot {slot_num}. "
                        f"Loaded {len(budget_replay_schedule)} replay slots from {BUDGET_REPLAY_PATH!r}."
                    )
                budget_replay_target_budget = int(final_budget_full_same_cap)
                budget_replay_effective_budget = int(final_budget_full_same_cap)
                budget_match_action = "slot_replay_missing_keep_original"
            else:
                budget_replay_target_budget = int(budget_replay_schedule[slot_num])
                budget_replay_effective_budget = int(budget_replay_target_budget)
                if budget_replay_effective_budget > int(miss_candidates):
                    budget_replay_clamped = True
                    budget_replay_clamp_from = int(budget_replay_effective_budget)
                    if not BUDGET_REPLAY_ALLOW_CLAMP:
                        raise ValueError(
                            f"Budget replay slot {slot_num} target "
                            f"{budget_replay_effective_budget} exceeds candidate count "
                            f"{miss_candidates}; set BUDGET_REPLAY_ALLOW_CLAMP=True to clamp."
                        )
                    budget_replay_effective_budget = int(miss_candidates)
                    budget_replay_clamp_count += 1
                    budget_match_action = "slot_replay_clamped"
                admit_budget = max(0, int(budget_replay_effective_budget))
            budget_replay_evaluated_slots += 1
            budget_replay_total_target_budget += int(budget_replay_target_budget or 0)
        score_gate_applied = binding_reason_full == "top_percent_cap"
        eta_top_base = SCORE_GATE_TOP_PERCENT
        eta_top_t = SCORE_GATE_TOP_PERCENT
        if miss_candidates > 0:
            score_gate_k_sum += score_gate_k
            if score_gate_applied:
                score_gate_applied_slots += 1
        budget_delta_precision = (
            final_budget_precision_only_same_cap - final_budget_no_signals_same_cap
        )
        budget_delta_quality = (
            final_budget_quality_only_same_cap - final_budget_no_signals_same_cap
        )
        budget_delta_quality_given_precision = (
            final_budget_full_same_cap - final_budget_precision_only_same_cap
        )
        budget_delta_interaction = (
            final_budget_full_same_cap
            - final_budget_precision_only_same_cap
            - final_budget_quality_only_same_cap
            + final_budget_no_signals_same_cap
        )
        precision_changes_integer_budget = (
            final_budget_precision_only_same_cap != final_budget_no_signals_same_cap
        )
        quality_changes_integer_budget = (
            final_budget_quality_only_same_cap != final_budget_no_signals_same_cap
        )
        quality_changes_integer_budget_given_precision = (
            final_budget_full_same_cap != final_budget_precision_only_same_cap
        )
        top_percent_cap_binds = binding_reason_full == "top_percent_cap"
        candidate_count_binds = binding_reason_full == "candidate_count"
        fill_floor_binds = binding_reason_full == "fill_floor"
        quality_masked_by_cap = (
            quality_mult != 1.0
            and not quality_changes_integer_budget_given_precision
            and binding_reason_full == "top_percent_cap"
        )
        quality_masked_by_rounding = (
            quality_mult != 1.0
            and raw_budget_full != raw_budget_precision_only
            and final_budget_full_same_cap == final_budget_precision_only_same_cap
            and binding_reason_full
            not in ("top_percent_cap", "candidate_count", "fill_floor", "no_candidates")
        )
        if budget_delta_precision > 0:
            precision_action = "increase"
        elif budget_delta_precision < 0:
            precision_action = "decrease"
        else:
            precision_action = "none"
        if budget_delta_quality_given_precision < 0:
            score_quality_action = "suppress"
        elif budget_delta_quality_given_precision > 0:
            score_quality_action = "boost"
        else:
            score_quality_action = "none"
        admitted_by_full_diag: set[str] = set()
        suppressed_by_quality_diag: set[str] = set()
        if (
            is_cache_slot
            and SCORE_QUALITY_SIGNAL_MODE != "off"
            and final_budget_precision_only_same_cap > final_budget_full_same_cap
            and miss_candidates > 0
        ):
            ranked_idx = sorted(
                range(miss_candidates),
                key=lambda idx: (-float(candidate_score_list[idx]), idx),
            )
            m_full_actual = min(max(int(final_budget_full_same_cap), 0), miss_candidates)
            m_precision_counterfactual = min(
                max(int(final_budget_precision_only_same_cap), 0),
                miss_candidates,
            )
            admitted_idx = ranked_idx[:m_full_actual]
            suppressed_idx = ranked_idx[m_full_actual:m_precision_counterfactual]
            admitted_by_full_diag = {candidate_ids[idx] for idx in admitted_idx}
            suppressed_by_quality_diag = {candidate_ids[idx] for idx in suppressed_idx}
            quality_suppressed_count = len(suppressed_idx)
            quality_suppressed_total += quality_suppressed_count
            quality_admitted_eval_total += len(admitted_idx)
            quality_suppressed_mean_score = _safe_mean(
                [candidate_score_list[idx] for idx in suppressed_idx]
            )
            quality_admitted_mean_score = _safe_mean(
                [candidate_score_list[idx] for idx in admitted_idx]
            )
            quality_suppressed_pos_label_rate_current = _safe_mean(
                [y_miss[idx] for idx in suppressed_idx]
            )
            quality_admitted_pos_label_rate_current = _safe_mean(
                [y_miss[idx] for idx in admitted_idx]
            )
            future_slot = slot_num + max(int(QUALITY_SUPPRESSED_UTILITY_WINDOW_SLOTS), 1)
            diagnostic_groups_by_future_slot.setdefault(future_slot, []).append(
                {
                    "source_slot": slot_num,
                    "admitted_by_full": admitted_by_full_diag,
                    "suppressed_by_quality": suppressed_by_quality_diag,
                }
            )
        total_admit_budget += admit_budget
        # Eq. (1) selector mapping:
        # - Paper Top_{M_t}(.) -> _select_top_m(candidate_ids, candidate_score_list, admit_budget)
        # - Paper A_t          -> next_admit_set (stored as pending_admit_set)
        # - No explicit tau_t  -> guard_full uses implicit gating via admit_budget and score_gate_k
        next_admit_set = _select_top_m(candidate_ids, candidate_score_list, admit_budget)
        admit_selected = len(next_admit_set)
        if is_cache_slot and budget_replay_target_budget is not None:
            budget_replay_total_realized_selected += int(admit_selected)
            budget_replay_exact_slot_match = (
                int(admit_selected) == int(budget_replay_target_budget)
            )
            if budget_replay_exact_slot_match:
                budget_replay_exact_slot_match_count += 1
        slot_admit_requests = admit_selected
        slot_reject_requests = max(miss_candidates - admit_selected, 0)
        total_admit_requests += slot_admit_requests
        total_reject_requests += slot_reject_requests
        total_admit_selected += admit_selected
        pending_admit_set = next_admit_set
        pending_admit_source_slot = slot_num

        if admit_true_applied is not None:
            total_admit_true_popular += admit_true_applied
        if precision_eff is not None:
            precision_eff_sum += precision_eff
            precision_eff_count += 1
        if is_cache_slot:
            fill_phase_by_slot[slot_num] = bool(fill_phase)
            if not fill_phase:
                postfill_cache_requests += slot_cache_requests
                postfill_cache_hits += slot_cache_hits
                postfill_selected_admissions += admit_selected
                postfill_applied_admissions += len(applied_admit_set)
                postfill_hits_from_admitted += slot_hits_from_admitted
                if admit_true_applied is not None:
                    postfill_admit_true_popular += admit_true_applied

        update_time_s = 0.0
        update_start = time.perf_counter()
        il_model.update_slot(D_t)
        update_time_s = time.perf_counter() - update_start
        total_update_time_s += update_time_s
        total_update_slots += 1

        warmup_slots = warmup_requests // slot_size
        phase = "warmup" if slot_num <= warmup_slots else "cache"
        current_slot_hr = (
            float(slot_cache_hits / slot_cache_requests)
            if slot_cache_requests > 0
            else None
        )
        if (
            is_cache_slot
            and prev_score_precK is not None
            and current_slot_hr is not None
            and prev_score_precK == prev_score_precK
        ):
            score_precK_hr_next_n += 1
            score_precK_hr_next_sum_x += float(prev_score_precK)
            score_precK_hr_next_sum_y += float(current_slot_hr)
            score_precK_hr_next_sum_x2 += float(prev_score_precK) ** 2
            score_precK_hr_next_sum_y2 += float(current_slot_hr) ** 2
            score_precK_hr_next_sum_xy += float(prev_score_precK) * float(current_slot_hr)
        if is_cache_slot:
            prev_score_precK = score_precK

        expired_for_log = slot_num - POLLUTION_WINDOW_SLOTS
        pollution_count_slot = 0
        pollution_total_slot = 0
        if expired_for_log >= 1:
            pollution_count_slot, pollution_total_slot = _finalize_pollution(expired_for_log)
        pollution_rate_slot = (
            float(pollution_count_slot / pollution_total_slot)
            if pollution_total_slot > 0
            else None
        )
        hit_yield_slot = (
            float(slot_hits_from_admitted / len(applied_admit_set))
            if applied_admit_set
            else None
        )
        admission_hit_yield = (
            float(slot_hits_from_admitted / len(applied_admit_set))
            if applied_admit_set
            else None
        )
        admission_hit_yield_zero_filled = float(
            slot_hits_from_admitted / max(len(applied_admit_set), 1)
        )
        pollution_proxy = pollution_rate_slot

        slot_log = {
            "slot_index": slot_num,
            "phase": phase,
            "slot_cache_requests": slot_cache_requests,
            "slot_cache_hits": slot_cache_hits,
            "slot_cache_misses": slot_cache_misses,
            "slot_hit_ratio": (
                current_slot_hr
            ),
            "hit_ratio": stats.hit_ratio,
            "miss_requests": miss_requests,
            "miss_rate": miss_rate,
            "miss_candidates": miss_candidates,
            # rho_t = |A_t| / max(1, |C_t|) from Section III.
            "admission_rate": (
                float(admit_selected / miss_candidates) if miss_candidates > 0 else 0.0
            ),
            "admission_rate_slot": (
                float(admit_selected / miss_candidates) if miss_candidates > 0 else 0.0
            ),
            "insertions_per_request_slot": (
                float(len(applied_admit_set) / slot_cache_requests)
                if slot_cache_requests > 0
                else 0.0
            ),
            "fill_phase": fill_phase,
            "fill_min_budget": fill_min_budget,
            "pressure_mult": pressure_mult,
            "base_pressure_mult": base_pressure_mult,
            "budget_match_mode": budget_match_mode,
            "budget_match_multiplier_t": budget_match_multiplier_t,
            "budget_match_action": budget_match_action,
            "budget_replay_enabled": budget_replay_enabled,
            "budget_replay_permuted": budget_replay_permuted,
            "budget_replay_path": BUDGET_REPLAY_PATH,
            "budget_replay_key": budget_replay_metadata.get("budget_key_used"),
            "budget_replay_target_budget": budget_replay_target_budget,
            "budget_replay_effective_budget": budget_replay_effective_budget,
            "budget_replay_clamped": budget_replay_clamped,
            "budget_replay_clamp_from": budget_replay_clamp_from,
            "budget_replay_missing_slot": budget_replay_missing_slot,
            "budget_replay_exact_slot_match": budget_replay_exact_slot_match,
            "precision_control_mode": PRECISION_CONTROL_MODE,
            "effective_precision_lag_mode": EFFECTIVE_PRECISION_LAG_MODE,
            "admission_precision_target_mode": ADMISSION_PRECISION_TARGET_MODE,
            "precision_eff": precision_eff,
            "precision_target_t": precision_target_t,
            "precision_multiplier_raw": precision_multiplier_raw,
            "precision_multiplier_normal": precision_multiplier_normal,
            "precision_multiplier_t": precision_multiplier_t,
            "precision_action": precision_action,
            "alpha_base": alpha_base,
            "alpha_after_precision_unclipped": alpha_after_precision_unclipped,
            "alpha_for_slot": alpha_for_slot,
            "score_quality_signal_mode": SCORE_QUALITY_SIGNAL_MODE,
            "score_quality_control_role": SCORE_QUALITY_CONTROL_ROLE,
            "score_spread": score_spread,
            "score_spread_ema": score_spread_ema_after,
            "score_spread_ema_prev": score_spread_ema_prev,
            "score_spread_ema_after": score_spread_ema_after,
            "boundary_margin": boundary_margin,
            "boundary_margin_ema_prev": boundary_margin_ema_prev,
            "boundary_margin_ema_after": boundary_margin_ema_after,
            "score_quality_mult_raw": score_quality_mult_raw,
            "score_quality_mult_normal": score_quality_mult_normal,
            "score_quality_mult": quality_mult,
            "quality_mult": quality_mult,
            "score_quality_action": score_quality_action,
            "score_quality_calibration": score_quality_calibration,
            "score_quality_target_precK": score_quality_target_precK,
            "quality_control_mode": quality_control_mode,
            "precision_control_override_mode": precision_control_override_mode,
            "eta_top_t": eta_top_t,
            "eta_top_base": eta_top_base,
            "eta_top_min": QUALITY_CAP_MIN,
            "eta_top_max": QUALITY_CAP_MAX,
            "quality_cap_changed": abs(eta_top_t - eta_top_base) > 1e-15,
            "boundary_quality": boundary_quality,
            "global_quality": global_quality,
            "score_eval_n": score_eval_n,
            "score_eval_pos": score_eval_pos,
            "score_auc": score_auc,
            "score_ap": score_ap,
            "score_precK": score_precK,
            "score_AUC": score_auc,
            "score_AP": score_ap,
            "score_acc": score_acc,
            "score_precK_ema_prev": score_precK_ema_prev,
            "score_precK_ema_after": score_precK_ema_after,
            "candidate_pos_rate": candidate_pos_rate,
            "candidate_pos_rate_ema_prev": candidate_pos_rate_ema_prev,
            "candidate_pos_rate_ema_after": candidate_pos_rate_ema_after,
            "admission_precision_lag0": precision_lag0,
            "admission_precision_lag1": precision_lag1,
            "admission_precision_lag2": precision_lag2,
            "admission_precision_eff": precision_eff,
            "score_gate_k": score_gate_k,
            "score_gate_applied": score_gate_applied,
            "cont_budget_no_signals": cont_budget_no_signals,
            "cont_budget_precision_only": cont_budget_precision_only,
            "cont_budget_quality_only": cont_budget_quality_only,
            "cont_budget_full": cont_budget_full,
            "raw_budget_no_signals": raw_budget_no_signals,
            "raw_budget_precision_only": raw_budget_precision_only,
            "raw_budget_quality_only": raw_budget_quality_only,
            "raw_budget_full": raw_budget_full,
            "quality_changes_continuous_budget_given_precision": (
                quality_changes_continuous_budget_given_precision
            ),
            "precision_changes_continuous_budget": precision_changes_continuous_budget,
            "quality_masked_by_integer_rounding": quality_masked_by_integer_rounding,
            "precision_masked_by_integer_rounding": precision_masked_by_integer_rounding,
            "final_budget_no_signals_same_cap": final_budget_no_signals_same_cap,
            "final_budget_precision_only_same_cap": final_budget_precision_only_same_cap,
            "final_budget_quality_only_same_cap": final_budget_quality_only_same_cap,
            "final_budget_full_same_cap": final_budget_full_same_cap,
            "budget_delta_precision": budget_delta_precision,
            "budget_delta_quality": budget_delta_quality,
            "budget_delta_quality_given_precision": budget_delta_quality_given_precision,
            "budget_delta_interaction": budget_delta_interaction,
            "precision_changes_integer_budget": precision_changes_integer_budget,
            "quality_changes_integer_budget": quality_changes_integer_budget,
            "quality_changes_integer_budget_given_precision": quality_changes_integer_budget_given_precision,
            "binding_reason_no_signals": binding_reason_no_signals,
            "binding_reason_precision_only": binding_reason_precision_only,
            "binding_reason_quality_only": binding_reason_quality_only,
            "binding_reason_full": binding_reason_full,
            "top_percent_cap_binds": top_percent_cap_binds,
            "quality_masked_by_cap": quality_masked_by_cap,
            "quality_masked_by_rounding": quality_masked_by_rounding,
            "candidate_count_binds": candidate_count_binds,
            "fill_floor_binds": fill_floor_binds,
            "quality_suppressed_count": quality_suppressed_count,
            "quality_admitted_eval_count": len(admitted_by_full_diag),
            "quality_suppressed_mean_score": quality_suppressed_mean_score,
            "quality_admitted_mean_score": quality_admitted_mean_score,
            "quality_suppressed_pos_label_rate_current": (
                quality_suppressed_pos_label_rate_current
            ),
            "quality_admitted_pos_label_rate_current": (
                quality_admitted_pos_label_rate_current
            ),
            "quality_future_eval_group_count": quality_future_eval_group_count,
            "quality_suppressed_next_slot_requests": (
                quality_suppressed_next_slot_requests
            ),
            "quality_admitted_next_slot_requests": quality_admitted_next_slot_requests,
            "quality_suppressed_next_slot_request_rate": (
                quality_suppressed_next_slot_request_rate
            ),
            "quality_admitted_next_slot_request_rate": (
                quality_admitted_next_slot_request_rate
            ),
            "quality_admitted_to_suppressed_request_ratio": (
                quality_admitted_to_suppressed_request_ratio
            ),
            "admit_budget": admit_budget,
            "final_budget": admit_budget,
            "precision_budget_before_quality_cap": precision_budget_before_quality_cap,
            "final_budget_after_quality_cap": admit_budget,
            "admit_selected": admit_selected,
            "admit_applied": len(applied_admit_set),
            "admitted_from_previous_slot": len(applied_admit_set),
            "hits_from_recent_admissions": slot_hits_from_admitted,
            "hit_yield_slot": hit_yield_slot,
            "admission_hit_yield": admission_hit_yield,
            "admission_hit_yield_zero_filled": admission_hit_yield_zero_filled,
            "pollution_count_slot": pollution_count_slot,
            "pollution_total_slot": pollution_total_slot,
            "pollution_rate_slot": pollution_rate_slot,
            "pollution_proxy": pollution_proxy,
            "admit_applied_from_slot": applied_admit_source_slot,
            "admit_true_from_applied": admit_true_applied,
            "admission_precision": admit_precision,
            "jsd_freq": jsd_freq,
            "topk_overlap_w": overlap_w,
            "drift_raw": drift_raw,
            "drift_ema": drift_ema,
            "drift_norm": drift_norm,
            "admission_alpha": alpha_for_slot,
            "capacity_scale": capacity_scale,
            "update_time_s": float(update_time_s),
        }
        _write_slot_log(slot_log)
        slot_diagnostic_records.append(slot_log)
        score_spread_ema = score_spread_ema_after
        boundary_margin_ema = boundary_margin_ema_after
        score_precK_ema = score_precK_ema_after
        candidate_pos_rate_ema = candidate_pos_rate_ema_after

        # update rolling history for next slot
        _update_freq_history(
            slot_num,
            slot_stats,
            freq_history,
            cum_counts,
            FEATURE_WINDOWS["long"],
        )

        # finalize pollution for expired slot
        expired = slot_num - POLLUTION_WINDOW_SLOTS
        if expired >= 1:
            total_pollution_count += pollution_count_slot
            total_pollution_total += pollution_total_slot
            if expired in fill_phase_by_slot and not fill_phase_by_slot[expired]:
                postfill_pollution_count += pollution_count_slot
                postfill_pollution_total += pollution_total_slot
            _prune_active_slots(expired)
            last_pollution_finalized = expired

        # reset per-slot state
        slot_stats = {}
        slot_cache_requests = 0
        slot_cache_hits = 0
        slot_cache_misses = 0
        slot_hits_from_admitted = 0
        slot_admit_requests = 0
        slot_reject_requests = 0
        slot_miss_obj_ids = []
        slot_miss_features = []
        applied_admit_set = set()
        applied_admit_source_slot = None

    try:
        while True:
            try:
                req = next(req_iter)
            except StopIteration:
                # flush slot terakhir jika ada
                if slot_stats:
                    _finalize_slot()
                    pbar.update(1)
                    pbar.set_postfix(
                        hr=f"{stats.hit_ratio:.4f}" if stats.total_requests > 0 else "0.0000"
                    )
                break

            if global_idx >= total_requests:
                break

            obj_id = req["object_id"]

            # gunakan timestamp dataset yang dimonotonkan (non-decreasing)
            ts_raw = req.get("timestamp", None)
            if ts_raw is None:
                ts = (prev_ts + 1.0) if prev_ts is not None else float(global_idx + 1)
            else:
                ts_raw = float(ts_raw)
                if prev_ts is None:
                    ts = ts_raw
                else:
                    ts = ts_raw if ts_raw >= prev_ts else prev_ts
            prev_ts = ts

            # 1) update fitur (gap) & freq (slot_stats)
            gaps = feature_table.update_and_get_gaps(obj_id, ts)

            info = slot_stats.get(obj_id)
            if info is None:
                slot_stats[obj_id] = {
                    "freq": 1,
                    "last_gaps": gaps,
                }
                info = slot_stats[obj_id]
            else:
                info["freq"] += 1
                info["last_gaps"] = gaps

            history_feats = _get_history_features(
                obj_id,
                slot_index,
                freq_history,
                cum_counts,
                FEATURE_WINDOWS,
            )
            features = _build_feature_vector(
                feature_set,
                gaps,
                history_feats,
            )
            # features dipakai untuk keputusan admission saat ini (policy-time)

            # 2) fase caching (setelah warm-up selesai)
            if global_idx >= warmup_requests:
                stats.total_requests += 1
                slot_cache_requests += 1
                if cache.access(obj_id):
                    stats.cache_hits += 1
                    slot_cache_hits += 1
                    if active_admit_slots_by_obj.get(obj_id):
                        slot_hits_from_admitted += 1
                    _mark_admit_hit(obj_id)
                    if obj_id in applied_admit_set:
                        total_hits_from_admitted += 1
                else:
                    slot_cache_misses += 1
                    slot_miss_obj_ids.append(obj_id)
                    slot_miss_features.append(features)

            # 3) update counter & cek boundary slot
            global_idx += 1
            slot_req_count += 1

            if slot_req_count >= slot_size:
                _finalize_slot()

                next_slot_start = slot_index * slot_size
                if next_slot_start < total_requests and next_slot_start >= warmup_requests:
                    _apply_pending_admits(slot_index + 1)
                else:
                    pending_admit_set = set()
                    pending_admit_source_slot = None

                pbar.update(1)
                current_hr = stats.hit_ratio if stats.total_requests > 0 else 0.0
                pbar.set_postfix(hr=f"{current_hr:.4f}")
                slot_req_count = 0
    finally:
        if slot_log_file is not None:
            slot_log_file.close()
        pbar.close()

    # finalize pollution for remaining slots
    for slot_num in range(last_pollution_finalized + 1, slot_index + 1):
        polluted, total = _finalize_pollution(slot_num)
        total_pollution_count += polluted
        total_pollution_total += total
        if slot_num in fill_phase_by_slot and not fill_phase_by_slot[slot_num]:
            postfill_pollution_count += polluted
            postfill_pollution_total += total

    diagnostic_summary = _summarize_signal_diagnostics(slot_diagnostic_records)

    score_precK_hr_next_corr = None
    if score_precK_hr_next_n > 1:
        denom = (
            (score_precK_hr_next_n * score_precK_hr_next_sum_x2)
            - (score_precK_hr_next_sum_x ** 2)
        ) * (
            (score_precK_hr_next_n * score_precK_hr_next_sum_y2)
            - (score_precK_hr_next_sum_y ** 2)
        )
        if denom > 0.0:
            score_precK_hr_next_corr = (
                (score_precK_hr_next_n * score_precK_hr_next_sum_xy)
                - (score_precK_hr_next_sum_x * score_precK_hr_next_sum_y)
            ) / math.sqrt(denom)

    summary_metrics = {
        "feature_set": feature_set,
        "num_features": num_features,
        "slots_processed": total_slots_processed,
        "cache_slots_processed": total_cache_slots,
        "update_slots": total_update_slots,
        "miss_requests_total": total_miss_requests,
        "miss_candidates_total": total_miss_candidates,
        "score_eval_slots": score_eval_slots,
        "score_eval_n_avg": (
            float(score_eval_n_sum / score_eval_slots) if score_eval_slots > 0 else None
        ),
        "score_eval_pos_avg": (
            float(score_eval_pos_sum / score_eval_slots) if score_eval_slots > 0 else None
        ),
        "score_auc_avg": (
            float(score_auc_sum / score_auc_count) if score_auc_count > 0 else None
        ),
        "score_ap_avg": (
            float(score_ap_sum / score_ap_count) if score_ap_count > 0 else None
        ),
        "score_precK_avg": (
            float(score_precK_sum / score_precK_count) if score_precK_count > 0 else None
        ),
        "score_acc_avg": (
            float(score_acc_sum / score_acc_count) if score_acc_count > 0 else None
        ),
        "score_precK_hr_next_corr": score_precK_hr_next_corr,
        "score_precK_hr_next_n": score_precK_hr_next_n,
        "score_gate_k_avg": (
            float(score_gate_k_sum / total_cache_slots) if total_cache_slots > 0 else None
        ),
        "score_gate_applied_slots": score_gate_applied_slots,
        "fill_phase_slots": fill_phase_slots,
        "fill_min_budget_avg": (
            float(fill_min_budget_sum / fill_phase_slots) if fill_phase_slots > 0 else None
        ),
        "pressure_mult_avg": (
            float(pressure_mult_sum / pressure_mult_count)
            if pressure_mult_count > 0
            else None
        ),
        "score_spread_avg": (
            float(score_spread_sum / score_spread_count)
            if score_spread_count > 0
            else None
        ),
        "quality_mult_avg": (
            float(quality_mult_sum / quality_mult_count)
            if quality_mult_count > 0
            else None
        ),
        "admission_precision_eff_avg": (
            float(precision_eff_sum / precision_eff_count)
            if precision_eff_count > 0
            else None
        ),
        "precision_multiplier_avg": (
            float(precision_multiplier_sum / precision_multiplier_count)
            if precision_multiplier_count > 0
            else None
        ),
        "diagnostic_summary": diagnostic_summary,
        "final_score_gate_top_percent": float(SCORE_GATE_TOP_PERCENT),
        "fixed_lru_eviction": True,
        "one_slot_delayed_apply": True,
        "precision_feedback_enabled": (PRECISION_CONTROL_MODE or "off").lower() != "off",
        "score_quality_enabled": (SCORE_QUALITY_SIGNAL_MODE or "off").lower() != "off",
        "budget_match_mode": configured_budget_match_mode,
        "budget_match_multiplier": float(BUDGET_MATCH_MULTIPLIER),
        "budget_match_target_multiplier": float(BUDGET_MATCH_TARGET_MULTIPLIER),
        "budget_match_ratio_source": BUDGET_MATCH_RATIO_SOURCE,
        "budget_replay_enabled": bool(budget_replay_enabled),
        "budget_replay_path": BUDGET_REPLAY_PATH,
        "budget_replay_key_used": budget_replay_metadata.get("budget_key_used"),
        "budget_replay_slots_loaded": int(budget_replay_metadata.get("slots_loaded", 0) or 0),
        "budget_replay_exact_slot_match_rate": (
            float(budget_replay_exact_slot_match_count / budget_replay_evaluated_slots)
            if budget_replay_evaluated_slots > 0
            else None
        ),
        "budget_replay_clamp_count": int(budget_replay_clamp_count),
        "budget_replay_clamp_rate": (
            float(budget_replay_clamp_count / budget_replay_evaluated_slots)
            if budget_replay_evaluated_slots > 0
            else None
        ),
        "budget_replay_total_target_budget": int(budget_replay_total_target_budget),
        "budget_replay_total_realized_selected": int(budget_replay_total_realized_selected),
        "budget_replay_permuted": bool(budget_replay_permuted),
        "budget_replay_permute_seed": (
            int(BUDGET_REPLAY_PERMUTE_SEED) if budget_replay_permuted else None
        ),
        "budget_replay_original_total_budget": (
            budget_replay_permutation_metadata.get("original_total_budget")
            if budget_replay_permuted
            else budget_replay_metadata.get("total_budget")
        ),
        "budget_replay_permuted_total_budget": (
            budget_replay_permutation_metadata.get("permuted_total_budget")
            if budget_replay_permuted
            else None
        ),
        "budget_replay_histogram_preserved": (
            budget_replay_permutation_metadata.get("histogram_preserved")
            if budget_replay_permuted
            else None
        ),
        "budget_replay_same_slot_budget_rate_against_full": (
            budget_replay_permutation_metadata.get("same_slot_budget_rate_against_full")
            if budget_replay_permuted
            else (1.0 if budget_replay_enabled else None)
        ),
        "offline_diagnostic_control": bool(OFFLINE_DIAGNOSTIC_CONTROL),
        "score_auc_count": score_auc_count,
        "score_ap_count": score_ap_count,
        "score_precK_count": score_precK_count,
        "score_acc_count": score_acc_count,
        "admit_budget_total": total_admit_budget,
        "admit_requests_total": total_admit_requests,
        "reject_requests_total": total_reject_requests,
        "admit_selected_total": total_admit_selected,
        "admit_applied_total": total_admit_applied,
        # rho_bar over trace (aggregate form of rho_t).
        "admission_rate_total": (
            float(total_admit_selected / total_miss_candidates)
            if total_miss_candidates > 0
            else None
        ),
        "admit_true_popular_total": total_admit_true_popular,
        "hits_from_admitted_total": total_hits_from_admitted,
        "pollution_total": total_pollution_total,
        "pollution_count": total_pollution_count,
        "admission_precision_total": (
            float(total_admit_true_popular / total_admit_applied)
            if total_admit_applied > 0
            else None
        ),
        "hit_yield_total": (
            float(total_hits_from_admitted / total_admit_applied)
            if total_admit_applied > 0
            else None
        ),
        "pollution_rate_total": (
            float(total_pollution_count / total_pollution_total)
            if total_pollution_total > 0
            else None
        ),
        "total_selected_admissions": total_admit_selected,
        "total_applied_admissions": total_admit_applied,
        "admissions_per_1000_requests": (
            float(total_admit_selected * 1000.0 / stats.total_requests)
            if stats.total_requests > 0
            else None
        ),
        "aggregate_hit_yield_per_applied_admission": (
            float(total_hits_from_admitted / total_admit_applied)
            if total_admit_applied > 0
            else None
        ),
        "aggregate_pollution_ratio": (
            float(total_pollution_count / total_pollution_total)
            if total_pollution_total > 0
            else None
        ),
        "aggregate_admission_precision": (
            float(total_admit_true_popular / total_admit_applied)
            if total_admit_applied > 0
            else None
        ),
        "postfill_hit_ratio": (
            float(postfill_cache_hits / postfill_cache_requests)
            if postfill_cache_requests > 0
            else None
        ),
        "postfill_selected_admissions": postfill_selected_admissions,
        "postfill_applied_admissions": postfill_applied_admissions,
        "postfill_admissions_per_1000_requests": (
            float(postfill_selected_admissions * 1000.0 / postfill_cache_requests)
            if postfill_cache_requests > 0
            else None
        ),
        "postfill_hit_yield_per_applied_admission": (
            float(postfill_hits_from_admitted / postfill_applied_admissions)
            if postfill_applied_admissions > 0
            else None
        ),
        "postfill_pollution_ratio": (
            float(postfill_pollution_count / postfill_pollution_total)
            if postfill_pollution_total > 0
            else None
        ),
        "postfill_admission_precision": (
            float(postfill_admit_true_popular / postfill_applied_admissions)
            if postfill_applied_admissions > 0
            else None
        ),
        "quality_suppressed_total": quality_suppressed_total,
        "quality_suppressed_next_slot_requests_total": (
            quality_suppressed_next_slot_requests_total
        ),
        "quality_admitted_next_slot_requests_total": (
            quality_admitted_next_slot_requests_total
        ),
        "quality_suppressed_next_slot_request_rate": (
            float(
                quality_suppressed_next_slot_requests_total
                / quality_suppressed_future_eval_total
            )
            if quality_suppressed_future_eval_total > 0
            else None
        ),
        "quality_admitted_next_slot_request_rate": (
            float(
                quality_admitted_next_slot_requests_total
                / quality_admitted_future_eval_total
            )
            if quality_admitted_future_eval_total > 0
            else None
        ),
        "quality_admitted_to_suppressed_request_ratio": (
            float(
                (
                    quality_admitted_next_slot_requests_total
                    / quality_admitted_future_eval_total
                )
                / (
                    quality_suppressed_next_slot_requests_total
                    / quality_suppressed_future_eval_total
                )
            )
            if (
                quality_admitted_future_eval_total > 0
                and quality_suppressed_future_eval_total > 0
                and quality_suppressed_next_slot_requests_total > 0
            )
            else None
        ),
        # Eq. (5) evaluation mapping:
        # - Paper rho_t / bar{rho} -> admission_rate_total over trace
        # - Paper Poll_t / bar{Poll} -> pollution_rate_total over trace
        # - Paper HR               -> stats.hit_ratio
        # This block reports the evaluation terms; it does not drive online control.
        "miss_rate_avg": (
            float(miss_rate_sum / total_cache_slots) if total_cache_slots > 0 else None
        ),
        "jsd_freq_avg": (
            float(jsd_sum / total_cache_slots) if total_cache_slots > 0 else None
        ),
        "topk_overlap_w_avg": (
            float(overlap_sum / total_cache_slots) if total_cache_slots > 0 else None
        ),
        "drift_raw_avg": (
            float(drift_raw_sum / total_cache_slots) if total_cache_slots > 0 else None
        ),
        "drift_norm_avg": (
            float(drift_norm_sum / total_cache_slots) if total_cache_slots > 0 else None
        ),
        "admission_alpha_avg": (
            float(alpha_sum / total_cache_slots) if total_cache_slots > 0 else None
        ),
        "admission_alpha_min": alpha_min_seen,
        "admission_alpha_max": alpha_max_seen,
        "avg_update_time_s": (
            float(total_update_time_s / total_update_slots)
            if total_update_slots > 0
            else None
        ),
        "update_time_total_s": float(total_update_time_s),
    }

    return stats, summary_metrics


# ---------------------------------------------------------------------------
# Pre-scan: hitung jumlah objek unik
# ---------------------------------------------------------------------------

def count_distinct_objects(trace_path: str, total_requests: int) -> int:
    """
    Satu pass ringan untuk menghitung jumlah object_id unik
    pada prefix trace sepanjang total_requests.
    """
    reader = TraceReader(path=trace_path, max_rows=total_requests)
    req_iter = reader.iter_requests()

    unique_objects = set()
    count = 0

    for req in req_iter:
        obj_id = req["object_id"]
        unique_objects.add(obj_id)
        count += 1
        if count >= total_requests:
            break

    return len(unique_objects)


# ---------------------------------------------------------------------------
# Utilitas penomoran run dan penulisan JSON
# ---------------------------------------------------------------------------

def get_next_run_id(results_root: str, dataset: str, model: str, cache_size: int) -> Tuple[str, str]:
    """
    Menentukan ID run berikutnya (001, 002, ...) agar tidak overwrite.
    """
    dataset_dir = os.path.join(results_root, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    existing_ids: List[int] = []
    for fname in os.listdir(dataset_dir):
        if not fname.endswith(f"_{model}_{cache_size}.jsonl") and \
           not fname.endswith(f"_summary_{model}_{cache_size}.json"):
            continue
        prefix = fname.split("_", 1)[0]
        if len(prefix) == 3 and prefix.isdigit():
            existing_ids.append(int(prefix))

    next_id = 1 if not existing_ids else max(existing_ids) + 1
    run_id = f"{next_id:03d}"
    return run_id, dataset_dir


def get_next_group_id(results_root: str, dataset: str, model: str) -> Tuple[str, str]:
    """
    Menentukan ID run berikutnya untuk summary lintas cache size.
    """
    dataset_dir = os.path.join(results_root, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    existing_ids: List[int] = []
    suffix = f"_summary_{model}_all_sizes.json"
    for fname in os.listdir(dataset_dir):
        if not fname.endswith(suffix):
            continue
        prefix = fname.split("_", 1)[0]
        if len(prefix) == 3 and prefix.isdigit():
            existing_ids.append(int(prefix))

    next_id = 1 if not existing_ids else max(existing_ids) + 1
    run_id = f"{next_id:03d}"
    return run_id, dataset_dir


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# hitung jumlah oobjek uniq
def get_dynamic_capacities(trace_path: str, total_requests: int, percentages: List[float]) -> List[int]:
    """
    Hitung kapasitas cache dalam jumlah objek berdasarkan persentase.
    """
    n_unique = count_distinct_objects(trace_path, total_requests)
    capacities = [int(n_unique * p / 100) for p in percentages]
    print(f"Jumlah objek unik: {n_unique}")
    print(f"Kapasitas cache dinamis (berdasarkan %): {capacities}")
    return capacities

# ---------------------------------------------------------------------------
# Main: TANPA argparse, semua pakai konstanta lokal
# ---------------------------------------------------------------------------

def run_experiment(
    ds_cfg: Dict[str, Any],
    feature_set: str = "A0",
    model_name: Optional[str] = None,
    base_learner: str = "nb",
    cache_size_percentages: Tuple[float, ...] = CACHE_SIZE_PERCENTAGES,
) -> None:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    if model_name is None:
        model_name = f"ilnse_{feature_set}"
    label_topk_rounding = IL_LABEL_TOPK_ROUNDING
    label_tie_break = IL_LABEL_TIE_BREAK

    trace_path = ds_cfg["path"]                # path ke .gz (seperti di debug)
    slot_size = ds_cfg["slot_size"]            # misal 100000
    warmup_requests = ds_cfg["num_warmup_requests"]  # misal 1_000_000
    # asumsi: total request juga sudah ada di config
    # kalau nama field beda (misal ds_cfg.total_requests), ganti baris ini
    total_requests = ds_cfg["num_total_requests"]

    if warmup_requests % slot_size != 0:
        raise ValueError(
            f"warmup_requests ({warmup_requests}) harus kelipatan slot_size ({slot_size}) "
            "agar sesuai dengan setup eksperimen Xu."
        )

    dataset_name = ds_cfg["name"]     # misal "WIKI2018"
    results_root = "results/guardrails_signal_v2"

    drift_control_mode = (DRIFT_CONTROL_MODE or "scaled").lower()
    piecewise_active = drift_control_mode == "piecewise"
    scaled_active = drift_control_mode == "scaled" and DRIFT_GAIN != 0.0
    drift_signal_active = piecewise_active or scaled_active
    quality_active = (
        SCORE_QUALITY_MIN != 1.0
        or SCORE_QUALITY_MAX != 1.0
        or SCORE_QUALITY_MIN_BOOST != 0.0
    )

    print(f"=== IL-based Edge Cache Experiment ({dataset_name} trace) ===")
    print(f"Trace path        : {trace_path}")
    print(f"Total requests    : {total_requests}")
    print(f"Warm-up requests  : {warmup_requests}")
    print(f"Slot size         : {slot_size}")
    print(f"IL num_gaps       : {IL_NUM_GAPS}")
    print(f"Feature set       : {feature_set}")
    print(f"Num features      : {_get_feature_dim(IL_NUM_GAPS, feature_set)}")
    print(f"Base learner      : {base_learner}")
    if drift_signal_active:
        print(f"Drift control     : {DRIFT_CONTROL_MODE}")
    if drift_signal_active:
        print(f"Drift norm power  : {DRIFT_NORM_POWER}")
    if scaled_active:
        print(f"Drift gain        : {DRIFT_GAIN}")
    if piecewise_active:
        print(f"Drift threshold   : {DRIFT_THRESHOLD}")
        print(f"Drift alpha low   : {DRIFT_ALPHA_LOW}")
        print(f"Drift alpha high  : {DRIFT_ALPHA_HIGH}")
    # print(f"Use cap scale     : {DRIFT_USE_CAPACITY_SCALE}")
    print(f"IL top_percent    : {IL_POP_TOP_PERCENT}")
    # print(f"Label rounding    : {label_topk_rounding}")
    # print(f"Label tie-break   : {label_tie_break}")
    print(f"IL sigmoid (a, b) : ({IL_SIGMOID_A}, {IL_SIGMOID_B})")
    print(f"Max learners      : {IL_MAX_CLASSIFIERS}")
    # print("Admission policy  : top_capacity_rate")
    print(f"Admission alpha   : {ADMISSION_CAPACITY_ALPHA}")
    # print(f"Fill ratio/rate   : {FILL_RATIO} / {FILL_RATE}")
    # print(f"Pressure gamma   : {PRESSURE_MISS_GAMMA}")
    if quality_active:
        print(
            "Quality spread   : q="
            f"{SCORE_SPREAD_Q}, ema={SCORE_SPREAD_EMA_ALPHA}, "
            f"min={SCORE_QUALITY_MIN}, max={SCORE_QUALITY_MAX}, "
            f"min_boost={SCORE_QUALITY_MIN_BOOST}"
        )
    # print(f"Drift alpha range : {DRIFT_ALPHA_MIN}–{DRIFT_ALPHA_MAX}")
    if drift_signal_active:
        print(f"Drift EMA alpha  : {DRIFT_EMA_ALPHA}")
        print(
            f"Drift weights    : jsd={DRIFT_WEIGHT_JSD}, overlap={DRIFT_WEIGHT_OVERLAP}"
        )
    if ADMISSION_PRECISION_SENSITIVITY > 0.0:
        print(f"Precision target  : {ADMISSION_PRECISION_TARGET}")
    print(f"Cache capacities  : {list(cache_size_percentages)}")
    print()

    # Hitung kapasitas dinamis
    capacities_objects = get_dynamic_capacities(trace_path, total_requests, list(cache_size_percentages))

    results: List[CacheStats] = []

    for capacity in capacities_objects:
        print(f"[RUN] Capacity Objects={capacity}")

        run_id, dataset_dir = get_next_run_id(results_root, dataset_name, model_name, capacity)
        per_slot_path = os.path.join(
            dataset_dir,
            f"{run_id}_{model_name}_{capacity}.jsonl",
        )
        summary_path = os.path.join(
            dataset_dir,
            f"{run_id}_summary_{model_name}_{capacity}.json",
        )

        stats, summary_metrics = run_single_capacity(
            trace_path=trace_path,
            total_requests=total_requests,
            warmup_requests=warmup_requests,
            slot_size=slot_size,
            capacity_objects=capacity,
            feature_set=feature_set,
            base_learner=base_learner,
            slot_log_path=per_slot_path,
        )
        results.append(stats)

        slots_processed = int(summary_metrics.get("slots_processed", 0))
        warmup_slots = warmup_requests // slot_size
        cache_slots = max(0, slots_processed - warmup_slots)

        summary_payload = {
            "dataset": dataset_name,
            "model": model_name,
            "experiment_family": "guardrails_signal_v2",
            "feature_set": feature_set,
            "base_learner": base_learner,
            "drift_control_mode": DRIFT_CONTROL_MODE,
            "drift_norm_power": DRIFT_NORM_POWER,
            "drift_threshold": DRIFT_THRESHOLD,
            "drift_alpha_low": DRIFT_ALPHA_LOW,
            "drift_alpha_high": DRIFT_ALPHA_HIGH,
            "drift_use_capacity_scale": DRIFT_USE_CAPACITY_SCALE,
            "cache_size_objects": capacity,
            "total_requests": total_requests,
            "warmup_requests": warmup_requests,
            "slot_size": slot_size,
            "slot_log_path": per_slot_path,
            "hit_ratio": stats.hit_ratio,
            "cache_hits": stats.cache_hits,
            "cache_requests": stats.total_requests,
            "num_slots": slots_processed,
            "warmup_slots": warmup_slots,
            "cache_slots": cache_slots,
            "pop_top_percent": IL_POP_TOP_PERCENT,
            "label_topk_rounding": label_topk_rounding,
            "label_tie_break": label_tie_break,
            "admission_policy": "top_capacity_rate",
            "admission_capacity_alpha": ADMISSION_CAPACITY_ALPHA,
            "drift_alpha_min": DRIFT_ALPHA_MIN,
            "drift_alpha_max": DRIFT_ALPHA_MAX,
            "drift_sensitivity": DRIFT_SENSITIVITY,
            "drift_ema_alpha": DRIFT_EMA_ALPHA,
            "drift_weight_jsd": DRIFT_WEIGHT_JSD,
            "drift_weight_overlap": DRIFT_WEIGHT_OVERLAP,
            "drift_gain": DRIFT_GAIN,
            "drift_alpha_floor_mult": DRIFT_ALPHA_FLOOR_MULT,
            "precision_control_mode": PRECISION_CONTROL_MODE,
            "effective_precision_lag_mode": EFFECTIVE_PRECISION_LAG_MODE,
            "admission_precision_target_mode": ADMISSION_PRECISION_TARGET_MODE,
            "admission_precision_target": ADMISSION_PRECISION_TARGET,
            "admission_precision_sensitivity": ADMISSION_PRECISION_SENSITIVITY,
            "admission_precision_pos_gain": ADMISSION_PRECISION_POS_GAIN,
            "admission_precision_neg_gain": ADMISSION_PRECISION_NEG_GAIN,
            "admission_precision_mult_min": ADMISSION_PRECISION_MULT_MIN,
            "admission_precision_mult_max": ADMISSION_PRECISION_MULT_MAX,
            "score_quality_signal_mode": SCORE_QUALITY_SIGNAL_MODE,
            "score_quality_control_role": SCORE_QUALITY_CONTROL_ROLE,
            "capacity_alpha_scale_min": CAPACITY_ALPHA_SCALE_MIN,
            "feature_windows": FEATURE_WINDOWS,
            "pollution_window_slots": POLLUTION_WINDOW_SLOTS,
            "hit_contribution": {
                "hits_from_admitted": summary_metrics.get("hits_from_admitted_total"),
                "reject_requests_total": summary_metrics.get("reject_requests_total"),
            },
        }
        summary_payload.update(summary_metrics)
        save_json(summary_path, summary_payload)

        print(f"      [LOG] per-slot  -> {per_slot_path}")
        print(f"      [LOG] summary   -> {summary_path}")
        print()

    print("=== Summary ===")
    for capacity, stats in zip(capacities_objects, results):
        print(
            f"cache_size={stats.capacity_objects}, "
            f"hit_ratio={stats.hit_ratio:.4f} "
            f"({stats.cache_hits}/{stats.total_requests})"
        )

    hr_curve = [
        {"cache_size_objects": cap, "hit_ratio": st.hit_ratio}
        for cap, st in zip(capacities_objects, results)
    ]
    avg_hr = float(np.mean([st.hit_ratio for st in results])) if results else None

    group_id, dataset_dir = get_next_group_id(results_root, dataset_name, model_name)
    overall_path = os.path.join(
        dataset_dir,
        f"{group_id}_summary_{model_name}_all_sizes.json",
    )
    overall_payload = {
        "dataset": dataset_name,
        "model": model_name,
        "experiment_family": "guardrails_signal_v2",
        "feature_set": feature_set,
        "base_learner": base_learner,
        "drift_control_mode": DRIFT_CONTROL_MODE,
        "drift_norm_power": DRIFT_NORM_POWER,
        "drift_threshold": DRIFT_THRESHOLD,
        "drift_alpha_low": DRIFT_ALPHA_LOW,
        "drift_alpha_high": DRIFT_ALPHA_HIGH,
        "drift_use_capacity_scale": DRIFT_USE_CAPACITY_SCALE,
        "num_features": _get_feature_dim(IL_NUM_GAPS, feature_set),
        "cache_sizes": capacities_objects,
        "hr_curve": hr_curve,
        "avg_hr": avg_hr,
    }
    save_json(overall_path, overall_payload)


def main() -> None:
    run_experiment(WIKIPEDIA_SEPTEMBER_2007)


if __name__ == "__main__":
    main()
# 1697
