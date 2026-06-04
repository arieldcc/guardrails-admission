# src/experiments/run_il_cache.py

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
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
)
from src.cache.cache_simulator import CacheStats
from src.cache.lru import LRUCache
from src.experiments.overhead_utils import (
    build_run_metadata,
    elapsed_s,
    get_current_rss_mb,
    get_peak_rss_mb,
    now_ns,
    prepare_run_dir,
    summarize_overhead_slots,
    write_jsonl,
    write_error_json,
    write_summary_json,
)
from src.experiments import run_il_cache_guard_signal_v2 as guard_v2

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
ADMISSION_ALPHA_MIN = guard_v2.DRIFT_ALPHA_MIN
ADMISSION_ALPHA_MAX = guard_v2.DRIFT_ALPHA_MAX
ADMISSION_ALPHA_FLOOR_MULT = guard_v2.DRIFT_ALPHA_FLOOR_MULT
ADMISSION_USE_CAPACITY_SCALE = guard_v2.DRIFT_USE_CAPACITY_SCALE
SCORE_GATE_TOP_PERCENT = guard_v2.FINAL_SCORE_GATE_TOP_PERCENT
FILL_RATIO = 0.9
FILL_RATE = 0.05
PRESSURE_MISS_GAMMA = 0.5
SCORE_SPREAD_Q = 0.9
SCORE_SPREAD_EMA_ALPHA = 0.2
SCORE_SPREAD_EPS = 1e-6
SCORE_QUALITY_SIGNAL_MODE = guard_v2.SCORE_QUALITY_SIGNAL_MODE
SCORE_QUALITY_CONTROL_ROLE = guard_v2.SCORE_QUALITY_CONTROL_ROLE
SCORE_QUALITY_MIN = guard_v2.SCORE_QUALITY_MIN
SCORE_QUALITY_MAX = guard_v2.SCORE_QUALITY_MAX
SCORE_QUALITY_MIN_BOOST = guard_v2.SCORE_QUALITY_MIN_BOOST
ADMISSION_PRECISION_TARGET = 0.12
ADMISSION_PRECISION_SENSITIVITY = 1.0
PRECISION_CONTROL_MODE = guard_v2.PRECISION_CONTROL_MODE
EFFECTIVE_PRECISION_LAG_MODE = guard_v2.EFFECTIVE_PRECISION_LAG_MODE
EFFECTIVE_PRECISION_FIXED_LAG = guard_v2.EFFECTIVE_PRECISION_FIXED_LAG
EFFECTIVE_PRECISION_LAG_WEIGHTS = guard_v2.EFFECTIVE_PRECISION_LAG_WEIGHTS
ADMISSION_PRECISION_TARGET_MODE = guard_v2.ADMISSION_PRECISION_TARGET_MODE
ADMISSION_PRECISION_MARGIN = guard_v2.ADMISSION_PRECISION_MARGIN
ADMISSION_PRECISION_POS_GAIN = guard_v2.ADMISSION_PRECISION_POS_GAIN
ADMISSION_PRECISION_NEG_GAIN = guard_v2.ADMISSION_PRECISION_NEG_GAIN
ADMISSION_PRECISION_MULT_MIN = guard_v2.ADMISSION_PRECISION_MULT_MIN
ADMISSION_PRECISION_MULT_MAX = guard_v2.ADMISSION_PRECISION_MULT_MAX
CANDIDATE_POS_RATE_EMA_ALPHA = guard_v2.CANDIDATE_POS_RATE_EMA_ALPHA
SCORE_QUALITY_PRECK_EMA_ALPHA = guard_v2.SCORE_QUALITY_PRECK_EMA_ALPHA
CAPACITY_ALPHA_SCALE_MIN = 0.4
DEFAULT_GUARD_CONFIG = "v2-cap020"
GUARD_CONFIG_CHOICES = ("v2-cap020",)

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
    if miss_requests <= 0:
        return 0
    return int(math.ceil(capacity_objects * alpha * pressure_mult))


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


def _hash_ordered(values: List[str]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(str(value).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _hash_set(values: set[str]) -> str:
    return _hash_ordered(sorted(values))


def _hash_scores(scores: List[float]) -> str:
    h = hashlib.sha256()
    for score in scores:
        h.update(f"{float(score):.12f}".encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


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
    dataset_name: str = "",
    model_name: str = "",
    disable_progress: bool = False,
    impl_mode: str = "reference",
    guard_config: str = DEFAULT_GUARD_CONFIG,
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
    if impl_mode not in ("reference", "optimized"):
        raise ValueError("impl_mode must be 'reference' or 'optimized'.")
    if guard_config not in GUARD_CONFIG_CHOICES:
        raise ValueError(f"guard_config must be one of {GUARD_CONFIG_CHOICES}.")

    num_gaps = IL_NUM_GAPS
    num_features = _get_feature_dim(num_gaps, feature_set)

    feature_table = FeatureTable(
        L=num_gaps,
        missing_gap_value=IL_MISSING_GAP_VALUE,
    )
    base_learner = (base_learner or "nb").lower()
    if base_learner != "nb":
        raise ValueError("run_il_cache_overhead_guard.py supports only base_learner='nb'.")

    il_model = LearnNSE(
        n_features=num_features,
        a=IL_SIGMOID_A,
        b=IL_SIGMOID_B,
        max_learners=IL_MAX_CLASSIFIERS,
        base_learner_factory=GaussianNaiveBayes,
    )
    cache = LRUCache(capacity_objects=capacity_objects)
    stats = CacheStats(capacity_objects=capacity_objects)

    label_topk_rounding = IL_LABEL_TOPK_ROUNDING
    label_tie_break = IL_LABEL_TIE_BREAK
    admission_capacity_alpha = float(ADMISSION_CAPACITY_ALPHA)

    total_slots_expected = (total_requests + slot_size - 1) // slot_size
    pbar = tqdm(
        total=total_slots_expected,
        desc="Processing slots",
        unit="slot",
        disable=disable_progress,
    )

    global_idx = 0          # counter global request (0-based)
    slot_req_count = 0      # jumlah request dalam slot saat ini
    slot_index = 0          # 1-based untuk logging
    slot_stats: Dict[str, Dict] = {}  # per-slot: freq, last_gaps
    prev_ts: Optional[float] = None
    slot_cache_requests = 0
    slot_cache_hits = 0
    slot_cache_misses = 0
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
    total_feat_time_s = 0.0
    total_score_time_s = 0.0
    total_guard_time_s = 0.0
    total_select_time_s = 0.0
    total_slot_time_s = 0.0
    total_update_slots = 0
    max_feature_bytes = 0
    total_slots_processed = 0
    total_cache_slots = 0
    last_pollution_finalized = 0
    slot_records: List[Dict[str, Any]] = []
    miss_rate_sum = 0.0
    alpha_sum = 0.0
    alpha_min_seen = None
    alpha_max_seen = None
    score_gate_k_sum = 0
    score_gate_applied_slots = 0
    fill_phase_slots = 0
    fill_min_budget_sum = 0
    pressure_mult_sum = 0.0
    pressure_mult_count = 0
    score_spread_ema = None
    boundary_margin_ema = None
    score_precK_ema = None
    candidate_pos_rate_ema = None
    score_spread_sum = 0.0
    score_spread_count = 0
    quality_mult_sum = 0.0
    quality_mult_count = 0
    precision_eff_sum = 0.0
    precision_eff_count = 0

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
    current_slot_apply_pending_time_s = 0.0

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
        slot_records.append(payload)

    def _finalize_slot() -> None:
        nonlocal slot_index
        nonlocal slot_stats
        nonlocal slot_cache_requests
        nonlocal slot_cache_hits
        nonlocal slot_cache_misses
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
        nonlocal total_feat_time_s
        nonlocal total_score_time_s
        nonlocal total_guard_time_s
        nonlocal total_select_time_s
        nonlocal total_slot_time_s
        nonlocal total_update_slots
        nonlocal max_feature_bytes
        nonlocal total_slots_processed
        nonlocal total_cache_slots
        nonlocal last_pollution_finalized
        nonlocal miss_rate_sum
        nonlocal alpha_sum
        nonlocal alpha_min_seen
        nonlocal alpha_max_seen
        nonlocal score_gate_k_sum
        nonlocal score_gate_applied_slots
        nonlocal fill_phase_slots
        nonlocal fill_min_budget_sum
        nonlocal pressure_mult_sum
        nonlocal pressure_mult_count
        nonlocal score_spread_ema
        nonlocal boundary_margin_ema
        nonlocal score_precK_ema
        nonlocal candidate_pos_rate_ema
        nonlocal score_spread_sum
        nonlocal score_spread_count
        nonlocal quality_mult_sum
        nonlocal quality_mult_count
        nonlocal precision_eff_sum
        nonlocal precision_eff_count
        nonlocal applied_history
        nonlocal current_slot_apply_pending_time_s

        if not slot_stats:
            return

        feat_start = now_ns()
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
        X_slot = None
        y_slot = None
        if impl_mode == "optimized":
            X_slot, y_slot = il_model.prepare_slot_arrays(D_t)
        feat_time_s = elapsed_s(feat_start)

        slot_index += 1
        slot_num = slot_index
        total_slots_processed += 1
        warmup_slots = warmup_requests // slot_size
        is_cache_slot = slot_num > warmup_slots

        time_budget_s = 0.0
        time_guard_signal_s = 0.0

        guard_signal_start = now_ns()
        # admission precision (applied set berasal dari slot sebelumnya)
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
        if guard_config == "v2-cap020":
            precision_eff = guard_v2._aggregate_effective_precision(
                precision_by_lag,
                EFFECTIVE_PRECISION_LAG_MODE,
                EFFECTIVE_PRECISION_FIXED_LAG,
                EFFECTIVE_PRECISION_LAG_WEIGHTS,
            )
        elif precision_by_lag:
            precision_eff = max(precision_by_lag.values())
        time_guard_signal_s += elapsed_s(guard_signal_start)
        admit_budget = 0
        admit_selected = 0
        miss_requests = int(slot_cache_misses)
        total_miss_requests += miss_requests
        miss_rate = float(miss_requests / slot_cache_requests) if slot_cache_requests > 0 else 0.0
        alpha_for_slot = admission_capacity_alpha
        capacity_scale = 1.0
        fill_phase = False
        fill_min_budget = 0
        pressure_mult = 1.0
        quality_mult = 1.0
        score_quality_mult_raw = 1.0
        score_spread = 0.0
        boundary_margin = 0.0
        candidate_pos_rate = None
        score_precK = float("nan")
        precision_target_t = ADMISSION_PRECISION_TARGET
        precision_multiplier_t = 1.0

        budget_start = now_ns()
        if is_cache_slot:
            total_cache_slots += 1
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

            scale = capacity_scale if ADMISSION_USE_CAPACITY_SCALE else 1.0
            alpha_base = admission_capacity_alpha * scale
            if alpha_base < ADMISSION_ALPHA_MIN:
                alpha_base = ADMISSION_ALPHA_MIN
            elif alpha_base > ADMISSION_ALPHA_MAX:
                alpha_base = ADMISSION_ALPHA_MAX
            time_budget_s += elapsed_s(budget_start)
            guard_signal_start = now_ns()
            precision_adjust = 1.0
            guard_precision = precision_eff if precision_eff is not None else admit_precision
            if guard_config == "v2-cap020":
                precision_target_t = guard_v2._compute_precision_target(
                    None,
                    candidate_pos_rate_ema,
                    ADMISSION_PRECISION_TARGET_MODE,
                )
                precision_multiplier_t = guard_v2._compute_precision_multiplier(
                    guard_precision,
                    precision_target_t,
                    PRECISION_CONTROL_MODE,
                )
                precision_adjust = precision_multiplier_t
            elif guard_precision is not None:
                precision_delta = (
                    (guard_precision - ADMISSION_PRECISION_TARGET)
                    / max(ADMISSION_PRECISION_TARGET, 1e-12)
                )
                if precision_delta > 0.0:
                    precision_adjust = 1.0 + ADMISSION_PRECISION_SENSITIVITY * precision_delta
            alpha_candidate = alpha_base * precision_adjust
            alpha_floor = alpha_base * ADMISSION_ALPHA_FLOOR_MULT
            alpha_for_slot = alpha_candidate if alpha_candidate >= alpha_floor else alpha_floor
            if alpha_for_slot < ADMISSION_ALPHA_MIN:
                alpha_for_slot = ADMISSION_ALPHA_MIN
            elif alpha_for_slot > ADMISSION_ALPHA_MAX:
                alpha_for_slot = ADMISSION_ALPHA_MAX
            time_guard_signal_s += elapsed_s(guard_signal_start)
            budget_start = now_ns()

        if is_cache_slot:
            miss_rate_sum += miss_rate
            alpha_sum += alpha_for_slot
            if alpha_min_seen is None or alpha_for_slot < alpha_min_seen:
                alpha_min_seen = alpha_for_slot
            if alpha_max_seen is None or alpha_for_slot > alpha_max_seen:
                alpha_max_seen = alpha_for_slot
        time_budget_s += elapsed_s(budget_start)
        candidate_scores: Dict[str, float] = {}
        candidate_ids: List[str] = []
        score_start = now_ns()
        slot_feature_bytes = 0
        if slot_miss_features:
            X_miss = np.asarray(slot_miss_features, dtype=float)
            slot_feature_bytes = int(X_miss.nbytes)
            miss_scores = il_model.score_batch(X_miss)
            for obj_id, score in zip(slot_miss_obj_ids, miss_scores):
                prev = candidate_scores.get(obj_id)
                score_val = float(score)
                if prev is None:
                    candidate_scores[obj_id] = score_val
                    candidate_ids.append(obj_id)
                elif score_val > prev:
                    candidate_scores[obj_id] = score_val
        score_time_s = elapsed_s(score_start)
        if slot_feature_bytes > max_feature_bytes:
            max_feature_bytes = slot_feature_bytes
        miss_candidates = int(len(candidate_ids))
        total_miss_candidates += miss_candidates

        guard_signal_start = now_ns()
        candidate_score_list = [candidate_scores[obj_id] for obj_id in candidate_ids]
        if guard_config == "v2-cap020" and miss_candidates > 0:
            label_map = {row["object_id"]: int(row["y"]) for row in D_t}
            y_miss = [label_map.get(obj_id, 0) for obj_id in candidate_ids]
            score_eval_pos = int(sum(y_miss))
            candidate_pos_rate = float(score_eval_pos / miss_candidates)
            if score_eval_pos > 0:
                score_precK = guard_v2._precision_at_k(
                    y_miss,
                    candidate_score_list,
                    score_eval_pos,
                )
            base_pressure_mult = 1.0 + PRESSURE_MISS_GAMMA * miss_rate
            score_gate_k_for_quality = max(
                1,
                int(math.ceil(SCORE_GATE_TOP_PERCENT * miss_candidates)),
            )
            raw_budget_precision_pre_quality = _compute_admission_budget(
                slot_cache_misses,
                capacity_objects,
                alpha_for_slot,
                base_pressure_mult,
            )
            precision_budget_before_quality_cap, _ = guard_v2._apply_budget_bounds(
                raw_budget_precision_pre_quality,
                miss_candidates,
                score_gate_k_for_quality,
                FILL_RATIO,
                fill_min_budget,
                fill_phase=fill_phase,
            )
            quality_info = guard_v2._compute_score_quality_multiplier(
                candidate_score_list,
                precision_budget_before_quality_cap,
                score_spread_ema,
                boundary_margin_ema,
                score_precK_ema,
                candidate_pos_rate_ema,
                SCORE_QUALITY_SIGNAL_MODE,
                SCORE_QUALITY_CONTROL_ROLE,
            )
            quality_mult = float(quality_info["score_quality_mult"])
            score_quality_mult_raw = float(quality_info["score_quality_mult_raw"])
            score_spread = float(quality_info["score_spread"])
            boundary_margin = float(quality_info["boundary_margin"])
            score_spread_ema = guard_v2._ema_after(
                score_spread_ema,
                score_spread,
                SCORE_SPREAD_EMA_ALPHA,
            )
            boundary_margin_ema = guard_v2._ema_after(
                boundary_margin_ema,
                boundary_margin,
                SCORE_SPREAD_EMA_ALPHA,
            )
            score_precK_ema = guard_v2._ema_after(
                score_precK_ema,
                score_precK,
                SCORE_QUALITY_PRECK_EMA_ALPHA,
            )
            candidate_pos_rate_ema = guard_v2._ema_after(
                candidate_pos_rate_ema,
                candidate_pos_rate,
                CANDIDATE_POS_RATE_EMA_ALPHA,
            )
            score_spread_sum += score_spread
            score_spread_count += 1
            quality_mult_sum += quality_mult
            quality_mult_count += 1
        elif miss_candidates > 0:
            scores = np.asarray(candidate_score_list, dtype=float)
            p50, pq = np.quantile(scores, [0.5, SCORE_SPREAD_Q])
            p50 = float(p50)
            pq = float(pq)
            score_spread = pq - p50
            if score_spread_ema is None:
                score_spread_ema = score_spread
            else:
                score_spread_ema = (
                    SCORE_SPREAD_EMA_ALPHA * score_spread
                    + (1.0 - SCORE_SPREAD_EMA_ALPHA) * score_spread_ema
                )
            denom = (score_spread_ema if score_spread_ema is not None else 0.0) + SCORE_SPREAD_EPS
            quality_mult = score_spread / denom
            effective_quality_min = SCORE_QUALITY_MIN
            if capacity_scale < 1.0:
                boost = (1.0 - capacity_scale) * SCORE_QUALITY_MIN_BOOST
                effective_quality_min = min(SCORE_QUALITY_MAX, SCORE_QUALITY_MIN + boost)
            if quality_mult < effective_quality_min:
                quality_mult = effective_quality_min
            elif quality_mult > SCORE_QUALITY_MAX:
                quality_mult = SCORE_QUALITY_MAX
            score_spread_sum += score_spread
            score_spread_count += 1
            quality_mult_sum += quality_mult
            quality_mult_count += 1
        time_guard_signal_s += elapsed_s(guard_signal_start)

        budget_start = now_ns()
        if is_cache_slot:
            pressure_mult = (1.0 + PRESSURE_MISS_GAMMA * miss_rate) * quality_mult
            pressure_mult_sum += pressure_mult
            pressure_mult_count += 1
            if fill_phase:
                fill_phase_slots += 1
                fill_min_budget_sum += fill_min_budget
        admit_budget = _compute_admission_budget(
            slot_cache_misses,
            capacity_objects,
            alpha_for_slot,
            pressure_mult,
        )
        if fill_phase and admit_budget < fill_min_budget:
            admit_budget = fill_min_budget
        if miss_candidates <= 0:
            admit_budget = 0
        elif admit_budget > miss_candidates:
            admit_budget = miss_candidates
        score_gate_k = 0
        score_gate_applied = False
        if miss_candidates > 0:
            score_gate_k = max(1, int(math.ceil(SCORE_GATE_TOP_PERCENT * miss_candidates)))
            score_gate_k_sum += score_gate_k
            if not fill_phase and admit_budget > score_gate_k:
                admit_budget = score_gate_k
                score_gate_applied = True
                score_gate_applied_slots += 1
        time_budget_s += elapsed_s(budget_start)
        total_admit_budget += admit_budget
        select_start = now_ns()
        next_admit_set = _select_top_m(candidate_ids, candidate_score_list, admit_budget)
        select_time_s = elapsed_s(select_start)
        admit_selected = len(next_admit_set)
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

        update_time_s = 0.0
        update_start = now_ns()
        if impl_mode == "optimized":
            assert X_slot is not None and y_slot is not None
            il_model.update_slot_arrays(X_slot, y_slot)
        else:
            il_model.update_slot(D_t)
        update_time_s = elapsed_s(update_start)
        total_update_time_s += update_time_s
        total_feat_time_s += feat_time_s
        total_score_time_s += score_time_s
        total_guard_time_s += time_guard_signal_s
        total_select_time_s += select_time_s
        slot_time_s = (
            feat_time_s
            + score_time_s
            + time_budget_s
            + time_guard_signal_s
            + select_time_s
            + update_time_s
        )
        boundary_total_s = current_slot_apply_pending_time_s + slot_time_s
        total_slot_time_s += slot_time_s
        total_update_slots += 1

        warmup_slots = warmup_requests // slot_size
        phase = "warmup" if slot_num <= warmup_slots else "cache"
        current_slot_hr = (
            float(slot_cache_hits / slot_cache_requests)
            if slot_cache_requests > 0
            else None
        )
        slot_log = {
            "slot_index": slot_num,
            "phase": phase,
            "dataset": dataset_name,
            "model_name": model_name,
            "feature_set": feature_set,
            "capacity_objects": int(capacity_objects),
            "slot_size": int(slot_size),
            "warmup_requests": int(warmup_requests),
            "total_requests": int(total_requests),
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
            "candidate_instances": int(len(slot_miss_obj_ids)),
            "fill_phase": fill_phase,
            "fill_min_budget": fill_min_budget,
            "pressure_mult": pressure_mult,
            "guard_config": guard_config,
            "score_spread": score_spread,
            "score_spread_ema": score_spread_ema,
            "boundary_margin": boundary_margin,
            "boundary_margin_ema": boundary_margin_ema,
            "candidate_pos_rate": candidate_pos_rate,
            "candidate_pos_rate_ema": candidate_pos_rate_ema,
            "score_precK": score_precK,
            "score_precK_ema": score_precK_ema,
            "score_quality_signal_mode": SCORE_QUALITY_SIGNAL_MODE,
            "score_quality_control_role": SCORE_QUALITY_CONTROL_ROLE,
            "score_quality_mult_raw": score_quality_mult_raw,
            "quality_mult": quality_mult,
            "precision_control_mode": PRECISION_CONTROL_MODE,
            "effective_precision_lag_mode": EFFECTIVE_PRECISION_LAG_MODE,
            "admission_precision_target_mode": ADMISSION_PRECISION_TARGET_MODE,
            "admission_precision_target_t": precision_target_t,
            "precision_multiplier": precision_multiplier_t,
            "admission_precision_lag0": precision_lag0,
            "admission_precision_lag1": precision_lag1,
            "admission_precision_lag2": precision_lag2,
            "admission_precision_eff": precision_eff,
            "score_gate_k": score_gate_k,
            "score_gate_applied": score_gate_applied,
            "admit_budget": admit_budget,
            "admit_selected": admit_selected,
            "admit_applied": len(applied_admit_set),
            "admission_rate": (
                float(admit_selected / miss_candidates) if miss_candidates > 0 else 0.0
            ),
            "admit_applied_from_slot": applied_admit_source_slot,
            "admit_true_from_applied": admit_true_applied,
            "admission_precision": admit_precision,
            "admission_alpha": alpha_for_slot,
            "capacity_scale": capacity_scale,
            "candidate_ids_hash": _hash_ordered(candidate_ids),
            "candidate_scores_hash": _hash_scores(candidate_score_list),
            "admit_set_hash": _hash_set(next_admit_set),
            "pending_admit_hash": _hash_set(pending_admit_set),
            "implementation_mode": impl_mode,
            "time_feat_s": float(feat_time_s),
            "time_score_s": float(score_time_s),
            "time_budget_s": float(time_budget_s),
            "time_guard_signal_s": float(time_guard_signal_s),
            "time_guard_s": float(time_guard_signal_s),
            "time_select_s": float(select_time_s),
            "time_update_s": float(update_time_s),
            "time_buffer_add_s": 0.0,
            "time_history_update_s": 0.0,
            "time_rebuild_s": 0.0,
            "time_apply_pending_s": float(current_slot_apply_pending_time_s),
            "time_slot_control_s": float(slot_time_s),
            "time_boundary_total_s": float(boundary_total_s),
            "time_slot_s": float(slot_time_s),
            "feature_bytes": int(slot_feature_bytes),
            "current_rss_mb": get_current_rss_mb(),
            "peak_rss_mb": get_peak_rss_mb(),
            "rebuild_triggered": False,
            "rebuild_phase": None,
        }
        _write_slot_log(slot_log)
        current_slot_apply_pending_time_s = 0.0

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
            polluted, total = _finalize_pollution(expired)
            total_pollution_count += polluted
            total_pollution_total += total
            _prune_active_slots(expired)
            last_pollution_finalized = expired

        # reset per-slot state
        slot_stats = {}
        slot_cache_requests = 0
        slot_cache_hits = 0
        slot_cache_misses = 0
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

            # 2) fase caching (setelah warm-up selesai)
            if global_idx >= warmup_requests:
                stats.total_requests += 1
                slot_cache_requests += 1
                if cache.access(obj_id):
                    stats.cache_hits += 1
                    slot_cache_hits += 1
                    _mark_admit_hit(obj_id)
                    if obj_id in applied_admit_set:
                        total_hits_from_admitted += 1
                else:
                    slot_cache_misses += 1
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
                    slot_miss_obj_ids.append(obj_id)
                    slot_miss_features.append(features)

            # 3) update counter & cek boundary slot
            global_idx += 1
            slot_req_count += 1

            if slot_req_count >= slot_size:
                _finalize_slot()

                next_slot_start = slot_index * slot_size
                if next_slot_start < total_requests and next_slot_start >= warmup_requests:
                    apply_start = now_ns()
                    _apply_pending_admits(slot_index + 1)
                    current_slot_apply_pending_time_s = elapsed_s(apply_start)
                else:
                    pending_admit_set = set()
                    pending_admit_source_slot = None
                    current_slot_apply_pending_time_s = 0.0

                pbar.update(1)
                current_hr = stats.hit_ratio if stats.total_requests > 0 else 0.0
                pbar.set_postfix(hr=f"{current_hr:.4f}")
                slot_req_count = 0
    finally:
        pbar.close()
        if slot_log_path:
            write_jsonl(slot_log_path, slot_records)

    # finalize pollution for remaining slots
    for slot_num in range(last_pollution_finalized + 1, slot_index + 1):
        polluted, total = _finalize_pollution(slot_num)
        total_pollution_count += polluted
        total_pollution_total += total

    overhead_summary = summarize_overhead_slots(slot_records, phase="cache")
    summary_metrics = {
        "feature_set": feature_set,
        "num_features": num_features,
        "guard_config": guard_config,
        "score_gate_top_percent": SCORE_GATE_TOP_PERCENT,
        "precision_control_mode": PRECISION_CONTROL_MODE,
        "effective_precision_lag_mode": EFFECTIVE_PRECISION_LAG_MODE,
        "admission_precision_target_mode": ADMISSION_PRECISION_TARGET_MODE,
        "score_quality_signal_mode": SCORE_QUALITY_SIGNAL_MODE,
        "score_quality_control_role": SCORE_QUALITY_CONTROL_ROLE,
        "slots_processed": total_slots_processed,
        "cache_slots_processed": total_cache_slots,
        "update_slots": total_update_slots,
        "miss_requests_total": total_miss_requests,
        "miss_candidates_total": total_miss_candidates,
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
        "admit_budget_total": total_admit_budget,
        "admit_requests_total": total_admit_requests,
        "reject_requests_total": total_reject_requests,
        "admit_selected_total": total_admit_selected,
        "admit_applied_total": total_admit_applied,
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
        "miss_rate_avg": (
            float(miss_rate_sum / total_cache_slots) if total_cache_slots > 0 else None
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
        "avg_feat_time_s": (
            float(total_feat_time_s / total_update_slots)
            if total_update_slots > 0
            else None
        ),
        "avg_score_time_s": (
            float(total_score_time_s / total_update_slots)
            if total_update_slots > 0
            else None
        ),
        "avg_guard_time_s": (
            float(total_guard_time_s / total_update_slots)
            if total_update_slots > 0
            else None
        ),
        "avg_select_time_s": (
            float(total_select_time_s / total_update_slots)
            if total_update_slots > 0
            else None
        ),
        "avg_slot_time_s": (
            float(total_slot_time_s / total_update_slots)
            if total_update_slots > 0
            else None
        ),
        "update_time_total_s": float(total_update_time_s),
        "feat_time_total_s": float(total_feat_time_s),
        "score_time_total_s": float(total_score_time_s),
        "guard_time_total_s": float(total_guard_time_s),
        "select_time_total_s": float(total_select_time_s),
        "slot_time_total_s": float(total_slot_time_s),
        "max_feature_bytes": int(max_feature_bytes),
        "rss_max_mb": get_peak_rss_mb(),
    }
    summary_metrics.update(overhead_summary)
    for field in (
        "avg_update_time_s",
        "avg_feat_time_s",
        "avg_score_time_s",
        "avg_guard_time_s",
        "avg_select_time_s",
    ):
        if field in summary_metrics:
            summary_metrics[f"diagnostic_{field}_all_slots"] = summary_metrics.pop(field)
    summary_metrics["avg_slot_time_s"] = overhead_summary.get("avg_slot_control_s")
    summary_metrics["slot_time_total_s"] = (
        overhead_summary.get("feat_time_total_s", 0.0)
        + overhead_summary.get("score_time_total_s", 0.0)
        + overhead_summary.get("budget_time_total_s", 0.0)
        + overhead_summary.get("guard_signal_time_total_s", 0.0)
        + overhead_summary.get("select_time_total_s", 0.0)
        + overhead_summary.get("update_time_total_s", 0.0)
    )

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
    cache_size_objects: Optional[int] = None,
    capacity_percent: Optional[float] = None,
    results_root: str = "results/overhead",
    disable_progress: bool = False,
    benchmark_mode: bool = False,
    run_id_override: Optional[str] = None,
    seed: int = 42,
    max_requests: Optional[int] = None,
    smoke_test: bool = False,
    impl_mode: str = "reference",
    warmup_requests_override: Optional[int] = None,
    slot_size_override: Optional[int] = None,
    guard_config: str = DEFAULT_GUARD_CONFIG,
) -> None:
    np.random.seed(seed)
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    if impl_mode not in ("reference", "optimized"):
        raise ValueError("impl_mode must be 'reference' or 'optimized'.")
    if guard_config not in GUARD_CONFIG_CHOICES:
        raise ValueError(f"guard_config must be one of {GUARD_CONFIG_CHOICES}.")
    base_learner = (base_learner or "nb").lower()
    if base_learner != "nb":
        raise ValueError("run_il_cache_overhead_guard.py supports only base_learner='nb'.")
    if model_name is None:
        if guard_config == "v2-cap020":
            model_name = f"ilnse_{feature_set}_guardv2_full_cap020_{base_learner}_overhead"
        else:
            model_name = f"ilnse_{feature_set}_guard_full_{base_learner}_overhead"
        if impl_mode == "optimized":
            model_name += "_optimized"
    label_topk_rounding = IL_LABEL_TOPK_ROUNDING
    label_tie_break = IL_LABEL_TIE_BREAK

    trace_path = ds_cfg["path"]                # path ke .gz (seperti di debug)
    slot_size = ds_cfg["slot_size"]            # misal 100000
    warmup_requests = ds_cfg["num_warmup_requests"]  # misal 1_000_000
    # asumsi: total request juga sudah ada di config
    # kalau nama field beda (misal ds_cfg.total_requests), ganti baris ini
    total_requests = ds_cfg["num_total_requests"]
    if slot_size_override is not None:
        slot_size = int(slot_size_override)
    if warmup_requests_override is not None:
        warmup_requests = int(warmup_requests_override)
    if smoke_test:
        slot_size = 100_000
        warmup_requests = 100_000
        total_requests = min(total_requests, 300_000)
    elif max_requests is not None:
        total_requests = min(total_requests, int(max_requests))
        if total_requests <= warmup_requests:
            raise ValueError("--max-requests must be greater than warmup_requests; use --smoke-test for tiny runs.")

    if warmup_requests % slot_size != 0:
        raise ValueError(
            f"warmup_requests ({warmup_requests}) harus kelipatan slot_size ({slot_size}) "
            "agar sesuai dengan setup eksperimen Xu."
        )

    dataset_name = ds_cfg["name"]     # misal "WIKI2018"

    print(f"=== IL-based Edge Cache Experiment ({dataset_name} trace) ===")
    print(f"Trace path        : {trace_path}")
    print(f"Total requests    : {total_requests}")
    print(f"Warm-up requests  : {warmup_requests}")
    print(f"Slot size         : {slot_size}")
    print(f"IL num_gaps       : {IL_NUM_GAPS}")
    print(f"Feature set       : {feature_set}")
    print(f"Num features      : {_get_feature_dim(IL_NUM_GAPS, feature_set)}")
    print(f"Base learner      : {base_learner}")
    print(f"Implementation    : {impl_mode}")
    print(f"Guard config      : {guard_config}")
    print(f"Use cap scale     : {ADMISSION_USE_CAPACITY_SCALE}")
    print(f"IL top_percent    : {IL_POP_TOP_PERCENT}")
    print(f"Label rounding    : {label_topk_rounding}")
    print(f"Label tie-break   : {label_tie_break}")
    print(f"IL sigmoid (a, b) : ({IL_SIGMOID_A}, {IL_SIGMOID_B})")
    print(f"Max learners      : {IL_MAX_CLASSIFIERS}")
    print("Admission policy  : top_capacity_rate")
    print(f"Admission alpha   : {ADMISSION_CAPACITY_ALPHA}")
    print(f"Score-gate cap    : {SCORE_GATE_TOP_PERCENT}")
    print(f"Fill ratio/rate   : {FILL_RATIO} / {FILL_RATE}")
    print(f"Pressure gamma   : {PRESSURE_MISS_GAMMA}")
    print(
        "Quality signal   : mode="
        f"{SCORE_QUALITY_SIGNAL_MODE}, role={SCORE_QUALITY_CONTROL_ROLE}, q="
        f"{SCORE_SPREAD_Q}, ema={SCORE_SPREAD_EMA_ALPHA}, "
        f"min={SCORE_QUALITY_MIN}, max={SCORE_QUALITY_MAX}, "
        f"min_boost={SCORE_QUALITY_MIN_BOOST}"
    )
    print(f"Admission alpha range: {ADMISSION_ALPHA_MIN}-{ADMISSION_ALPHA_MAX}")
    print(
        "Precision control: "
        f"mode={PRECISION_CONTROL_MODE}, lag={EFFECTIVE_PRECISION_LAG_MODE}, "
        f"target={ADMISSION_PRECISION_TARGET_MODE}"
    )
    print(f"Cache capacities  : {list(cache_size_percentages)}")
    print()

    # Hitung kapasitas dinamis
    if cache_size_objects is not None:
        capacities_objects = [int(cache_size_objects)]
    elif capacity_percent is not None:
        capacities_objects = get_dynamic_capacities(
            trace_path,
            total_requests,
            [float(capacity_percent)],
        )
    else:
        capacities_objects = get_dynamic_capacities(
            trace_path,
            total_requests,
            list(cache_size_percentages),
        )

    results: List[CacheStats] = []

    for capacity in capacities_objects:
        print(f"[RUN] Capacity Objects={capacity}")

        if benchmark_mode:
            run_start = datetime.now(timezone.utc).isoformat()
            run_id, run_dir = prepare_run_dir(
                results_root, dataset_name, model_name, capacity, run_id_override
            )
            per_slot_path = os.path.join(run_dir, "slot_log.jsonl")
            summary_path = os.path.join(run_dir, "overhead_summary.json")
            config_path = os.path.join(run_dir, "config.json")
            metadata_path = os.path.join(run_dir, "run_metadata.json")
        else:
            run_id, dataset_dir = get_next_run_id(results_root, dataset_name, model_name, capacity)
            per_slot_path = os.path.join(
                dataset_dir,
                f"{run_id}_{model_name}_{capacity}.jsonl",
            )
            summary_path = os.path.join(
                dataset_dir,
                f"{run_id}_summary_{model_name}_{capacity}.json",
            )
            config_path = ""
            metadata_path = ""

        try:
            stats, summary_metrics = run_single_capacity(
                trace_path=trace_path,
                total_requests=total_requests,
                warmup_requests=warmup_requests,
                slot_size=slot_size,
                capacity_objects=capacity,
                feature_set=feature_set,
                base_learner=base_learner,
                slot_log_path=per_slot_path,
                dataset_name=dataset_name,
                model_name=model_name,
                disable_progress=disable_progress or benchmark_mode,
                impl_mode=impl_mode,
                guard_config=guard_config,
            )
        except Exception as exc:
            if benchmark_mode:
                write_error_json(
                    os.path.join(run_dir, "error.json"),
                    {
                        "dataset": dataset_name,
                        "model_name": model_name,
                        "feature_set": feature_set,
                        "capacity_objects": capacity,
                        "capacity_percent": capacity_percent,
                        "total_requests": total_requests,
                        "warmup_requests": warmup_requests,
                        "slot_size": slot_size,
                        "seed": seed,
                        "implementation_mode": impl_mode,
                        "guard_config": guard_config,
                    },
                    exc,
                )
            raise
        results.append(stats)

        slots_processed = int(summary_metrics.get("slots_processed", 0))
        warmup_slots = warmup_requests // slot_size
        cache_slots = max(0, slots_processed - warmup_slots)

        summary_payload = {
            "dataset": dataset_name,
            "model": model_name,
            "model_name": model_name,
            "feature_set": feature_set,
            "base_learner": base_learner,
            "logging_profile": "paper_overhead",
            "guard_config": guard_config,
            "implementation_mode": impl_mode,
            "optimized_components": (
                ["slot_array_reuse", "array_based_update_slot"]
                if impl_mode == "optimized"
                else []
            ),
            "admission_use_capacity_scale": ADMISSION_USE_CAPACITY_SCALE,
            "cache_size_objects": capacity,
            "capacity_objects": capacity,
            "capacity_percent": capacity_percent,
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
            "score_gate_top_percent": SCORE_GATE_TOP_PERCENT,
            "admission_capacity_alpha": ADMISSION_CAPACITY_ALPHA,
            "admission_alpha_min": ADMISSION_ALPHA_MIN,
            "admission_alpha_max": ADMISSION_ALPHA_MAX,
            "admission_alpha_floor_mult": ADMISSION_ALPHA_FLOOR_MULT,
            "admission_precision_target": ADMISSION_PRECISION_TARGET,
            "admission_precision_sensitivity": ADMISSION_PRECISION_SENSITIVITY,
            "precision_control_mode": PRECISION_CONTROL_MODE,
            "effective_precision_lag_mode": EFFECTIVE_PRECISION_LAG_MODE,
            "admission_precision_target_mode": ADMISSION_PRECISION_TARGET_MODE,
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
        write_summary_json(summary_path, summary_payload)
        if benchmark_mode:
            config_payload = {
                "dataset": dataset_name,
                "model": model_name,
                "feature_set": feature_set,
                "base_learner": base_learner,
                "cache_size_objects": capacity,
                "capacity_percent": capacity_percent,
                "total_requests": total_requests,
                "warmup_requests": warmup_requests,
                "slot_size": slot_size,
                "seed": seed,
                "implementation_mode": impl_mode,
                "guard_config": guard_config,
                "score_gate_top_percent": SCORE_GATE_TOP_PERCENT,
                "precision_control_mode": PRECISION_CONTROL_MODE,
                "effective_precision_lag_mode": EFFECTIVE_PRECISION_LAG_MODE,
                "admission_precision_target_mode": ADMISSION_PRECISION_TARGET_MODE,
                "score_quality_signal_mode": SCORE_QUALITY_SIGNAL_MODE,
                "score_quality_control_role": SCORE_QUALITY_CONTROL_ROLE,
                "optimized_components": (
                    ["slot_array_reuse", "array_based_update_slot"]
                    if impl_mode == "optimized"
                    else []
                ),
            }
            write_summary_json(config_path, config_payload)
            run_end = datetime.now(timezone.utc).isoformat()
            write_summary_json(
                metadata_path,
                build_run_metadata(
                    {
                        **config_payload,
                        "results_root": results_root,
                        "benchmark_mode": benchmark_mode,
                    },
                    run_start,
                    run_end,
                    PROJECT_ROOT,
                ),
            )

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
        "feature_set": feature_set,
        "base_learner": base_learner,
        "logging_profile": "paper_overhead",
        "guard_config": guard_config,
        "score_gate_top_percent": SCORE_GATE_TOP_PERCENT,
        "admission_use_capacity_scale": ADMISSION_USE_CAPACITY_SCALE,
        "num_features": _get_feature_dim(IL_NUM_GAPS, feature_set),
        "cache_sizes": capacities_objects,
        "hr_curve": hr_curve,
        "avg_hr": avg_hr,
    }
    save_json(overall_path, overall_payload)


def _resolve_dataset(name: str) -> Dict[str, Any]:
    if name == "wikipedia_september_2007":
        return WIKIPEDIA_SEPTEMBER_2007
    if name == "wiki2018":
        return WIKI2018
    raise ValueError(f"Unknown dataset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="IL guard overhead runner (slot-level timing).")
    parser.add_argument("--dataset", choices=["wikipedia_september_2007", "wiki2018"], required=True)
    parser.add_argument("--feature-set", default="A2")
    parser.add_argument("--base-learner", default="nb", choices=["nb"])
    parser.add_argument("--cache-size-objects", type=int, default=None)
    parser.add_argument("--capacity-objects", type=int, default=None)
    parser.add_argument("--capacity-percent", type=float, default=None)
    parser.add_argument("--results-root", default="results/overhead")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--benchmark-mode", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--impl-mode", choices=["reference", "optimized"], default="reference")
    parser.add_argument(
        "--guard-config",
        choices=GUARD_CONFIG_CHOICES,
        default=DEFAULT_GUARD_CONFIG,
        help="Guard policy timed by this overhead runner. Default follows final v2 cap-0.02.",
    )
    parser.add_argument("--warmup-requests", type=int, default=None)
    parser.add_argument("--slot-size", type=int, default=None)
    args = parser.parse_args()

    ds_cfg = _resolve_dataset(args.dataset)
    capacity_objects = args.capacity_objects if args.capacity_objects is not None else args.cache_size_objects
    try:
        run_experiment(
            ds_cfg,
            feature_set=args.feature_set,
            base_learner=args.base_learner,
            cache_size_objects=capacity_objects,
            capacity_percent=args.capacity_percent,
            results_root=args.results_root,
            disable_progress=args.disable_progress,
            benchmark_mode=args.benchmark_mode,
            run_id_override=args.run_id,
            seed=args.seed,
            max_requests=args.max_requests,
            smoke_test=args.smoke_test,
            impl_mode=args.impl_mode,
            guard_config=args.guard_config,
            warmup_requests_override=args.warmup_requests,
            slot_size_override=args.slot_size,
        )
    except Exception as exc:
        error_path = os.path.join(
            args.results_root,
            f"error_il_guard_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
        )
        write_error_json(error_path, vars(args), exc)
        raise


if __name__ == "__main__":
    main()
# 1185
