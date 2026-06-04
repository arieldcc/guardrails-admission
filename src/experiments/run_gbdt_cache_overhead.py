# src/experiments/run_gbdt_cache_overhead.py

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.trace_reader import TraceReader
from src.data.feature_table import FeatureTable
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

FEATURE_SETS = ("A0", "A1", "A2")
FEATURE_WINDOWS = {"short": 1, "mid": 7, "long": 30}

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

IL_NUM_GAPS = 6
IL_POP_TOP_PERCENT = 0.20
IL_LABEL_TOPK_ROUNDING = "floor"
IL_LABEL_TIE_BREAK = "none"
IL_MISSING_GAP_VALUE = 1e6

GDBT_N_ESTIMATORS = 30
GDBT_UPDATE_INTERVAL_REQUESTS = 1_000_000
GDBT_ADMIT_THRESHOLD = 0.5
POLLUTION_WINDOW_SLOTS = 1

CACHE_SIZE_PERCENTAGES = (0.8, 1.0, 2.0, 3.0, 4.0, 5.0)
DEFAULT_FEATURE_SET = "A2"


@dataclass
class ILConfig:
    num_gaps: int = IL_NUM_GAPS
    pop_top_percent: float = IL_POP_TOP_PERCENT
    missing_gap_value: float = IL_MISSING_GAP_VALUE


@dataclass
class GDBTConfig:
    n_estimators: int = GDBT_N_ESTIMATORS
    update_interval_requests: int = GDBT_UPDATE_INTERVAL_REQUESTS
    learning_rate: float = 0.1
    max_depth: int = 3
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42


@dataclass
class GDBTCachePredictorStandard:
    """Literature-style sklearn GBDT baseline with no Guardrails signals.

    A0 is closest to the original Gap1..GapL GBDT baseline. A2 is a standard
    sklearn GBDT scorer evaluated with the same A2 feature representation used
    by IL variants for paper fairness.
    """

    il_config: ILConfig
    gdbt_config: GDBTConfig
    n_features: int
    model: Optional[GradientBoostingClassifier] = None
    _X_buffer: List[List[float]] = None  # type: ignore
    _y_buffer: List[int] = None          # type: ignore
    num_rebuilds: int = 0
    total_training_samples: int = 0
    last_rebuild_info: Dict[str, Any] = None  # type: ignore

    def __post_init__(self) -> None:
        self._X_buffer = []
        self._y_buffer = []
        self.num_rebuilds = 0
        self.total_training_samples = 0
        self.last_rebuild_info = {}

    def add_training_sample(self, x: List[float], y: int) -> None:
        if len(x) != self.n_features:
            raise ValueError(
                f"n_features mismatch: expected {self.n_features}, got {len(x)}"
            )
        self._X_buffer.append(list(x))
        self._y_buffer.append(int(y))

    def add_training_batch(self, dataset: List[Dict[str, Any]]) -> None:
        for item in dataset:
            self.add_training_sample(item["x"], item["y"])

    def get_buffer_size(self) -> int:
        return len(self._X_buffer)

    def should_rebuild(self, requests_since_last_rebuild: int) -> bool:
        return requests_since_last_rebuild >= int(self.gdbt_config.update_interval_requests)

    def _create_model(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            n_estimators=int(self.gdbt_config.n_estimators),
            learning_rate=float(self.gdbt_config.learning_rate),
            max_depth=int(self.gdbt_config.max_depth),
            min_samples_split=int(self.gdbt_config.min_samples_split),
            min_samples_leaf=int(self.gdbt_config.min_samples_leaf),
            random_state=int(self.gdbt_config.random_state),
        )

    def rebuild_model(self) -> Dict[str, Any]:
        rebuild_info: Dict[str, Any] = {
            "rebuild_number": self.num_rebuilds + 1,
            "buffer_size": len(self._X_buffer),
            "success": False,
        }
        if not self._X_buffer:
            rebuild_info["error"] = "Empty buffer, cannot rebuild"
            self.last_rebuild_info = rebuild_info
            return rebuild_info

        try:
            X = np.asarray(self._X_buffer, dtype=float)
            y = np.asarray(self._y_buffer, dtype=int)

            if X.ndim != 2 or X.shape[1] != self.n_features:
                raise ValueError(
                    f"n_features mismatch: expected {self.n_features}, got {X.shape[1]}"
                )

            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                rebuild_info["error"] = f"Only one class in buffer: {unique_classes}"
                self.last_rebuild_info = rebuild_info
                return rebuild_info

            self.model = self._create_model()
            self.model.fit(X, y)

            self.num_rebuilds += 1
            self.total_training_samples += len(self._X_buffer)
            rebuild_info["success"] = True
            y_pred = self.model.predict(X)
            rebuild_info.update(
                {
                    "num_samples": int(len(X)),
                    "num_label_1": int((y == 1).sum()),
                    "num_label_0": int((y == 0).sum()),
                    "train_accuracy": float((y_pred == y).mean()),
                    "n_estimators": int(self.gdbt_config.n_estimators),
                    "n_estimators_actual": getattr(self.model, "n_estimators_", None),
                }
            )
        except Exception as exc:
            rebuild_info["error"] = str(exc)

        self.last_rebuild_info = rebuild_info
        return rebuild_info

    def clear_buffer(self) -> None:
        self._X_buffer = []
        self._y_buffer = []

    def predict(self, x: List[float]) -> int:
        if self.model is None:
            return 1
        x_arr = np.asarray(x, dtype=float).reshape(1, -1)
        if x_arr.shape[1] != self.n_features:
            raise ValueError(
                f"n_features mismatch: expected {self.n_features}, got {x_arr.shape[1]}"
            )
        return int(self.model.predict(x_arr)[0])

    def predict_proba(self, x: List[float]) -> float:
        if self.model is None:
            return 1.0
        x_arr = np.asarray(x, dtype=float).reshape(1, -1)
        if x_arr.shape[1] != self.n_features:
            raise ValueError(
                f"n_features mismatch: expected {self.n_features}, got {x_arr.shape[1]}"
            )
        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(x_arr)[0, 1])
        return float(self.predict(x))

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.shape[1] != self.n_features:
            raise ValueError(
                f"n_features mismatch: expected {self.n_features}, got {X_arr.shape[1]}"
            )
        if self.model is None:
            return np.ones(X_arr.shape[0], dtype=float)
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X_arr)[:, 1], dtype=float)
        return np.asarray(self.model.predict(X_arr), dtype=float)

    def get_feature_importances(self) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        try:
            return getattr(self.model, "feature_importances_", None)
        except Exception:
            return None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_rebuilds": self.num_rebuilds,
            "total_training_samples": self.total_training_samples,
            "current_buffer_size": self.get_buffer_size(),
            "has_model": self.model is not None,
            "last_rebuild_info": self.last_rebuild_info,
            "n_features": self.n_features,
            "pop_top_percent": self.il_config.pop_top_percent,
            "update_interval_requests": self.gdbt_config.update_interval_requests,
        }


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
        return num_gaps + 1
    if feature_set == "A2":
        return num_gaps + 4
    raise ValueError(f"Unknown feature_set: {feature_set}")


def _sum_history(window: int, slot_index: int, history: List[Tuple[int, int]]) -> int:
    if slot_index <= 0 or window <= 0:
        return 0
    start_slot = max(1, slot_index - window + 1)
    return sum(count for slot_id, count in history if slot_id >= start_slot)


def _get_history_features(
    obj_id: str,
    slot_index: int,
    freq_history: Dict[str, List[Tuple[int, int]]],
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
        f_short, _, _, _ = history_features
        return list(gaps) + [float(f_short)]
    f_short, f_mid, f_long, f_cum = history_features
    if feature_set == "A2":
        return list(gaps) + [float(f_short), float(f_mid), float(f_long), float(f_cum)]
    raise ValueError(f"Unknown feature_set: {feature_set}")


def _select_top_ids_from_stats(
    slot_stats: Dict[str, Dict[str, Any]],
    top_ratio: float,
    label_topk_rounding: str,
    label_tie_break: str,
) -> Tuple[set[str], Dict[str, Any]]:
    label_info: Dict[str, Any] = {}
    if not slot_stats:
        return set(), label_info
    items: List[Tuple[str, Dict[str, Any]]] = list(slot_stats.items())
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
            "top_ratio": float(len(top_ids) / n_objects) if n_objects > 0 else 0.0,
            "freq_at_k": int(kth_freq),
            "tie_count": int(tie_count),
            "tie_rate": float(tie_count / n_objects) if n_objects > 0 else 0.0,
        }
    )
    return top_ids, label_info


def _update_freq_history(
    slot_index: int,
    slot_stats: Dict[str, Dict[str, Any]],
    freq_history: Dict[str, List[Tuple[int, int]]],
    cum_counts: Dict[str, int],
    long_window: int,
) -> None:
    cutoff = slot_index - long_window + 1
    for obj_id, stats in slot_stats.items():
        count = int(stats.get("freq", 0))
        if count <= 0:
            continue
        history = freq_history.setdefault(obj_id, [])
        history[:] = [(sid, cnt) for sid, cnt in history if sid >= cutoff]
        history.append((slot_index, count))
        cum_counts[obj_id] = int(cum_counts.get(obj_id, 0)) + count


def build_slot_dataset_from_stats(
    slot_stats: Dict[str, Dict[str, Any]],
    top_ratio: float,
    num_gaps: int,
    missing_gap_value: float,
    feature_set: str,
    history_slot_index: int,
    freq_history: Dict[str, List[Tuple[int, int]]],
    cum_counts: Dict[str, int],
    label_topk_rounding: str,
    label_tie_break: str,
) -> List[Dict[str, Any]]:
    if not slot_stats:
        return []
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    num_features = _get_feature_dim(num_gaps, feature_set)

    items: List[Tuple[str, Dict[str, Any]]] = list(slot_stats.items())
    items.sort(key=lambda kv: kv[1]["freq"], reverse=True)

    top_ids, _ = _select_top_ids_from_stats(
        slot_stats,
        top_ratio,
        label_topk_rounding,
        label_tie_break,
    )

    dataset: List[Dict[str, Any]] = []
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
        features = _build_feature_vector(feature_set, gaps, history_feats)
        if len(features) != num_features:
            raise ValueError(
                f"feature length mismatch: expected {num_features}, got {len(features)}"
            )
        y = 1 if obj_id in top_ids else 0
        dataset.append(
            {
                "x": features,
                "y": y,
                "freq": stats["freq"],
                "object_id": obj_id,
            }
        )
    return dataset


def count_distinct_objects(trace_path: str, total_requests: int) -> int:
    reader = TraceReader(path=trace_path, max_rows=total_requests)
    unique_objects = set()
    count = 0
    for req in reader.iter_requests():
        unique_objects.add(req["object_id"])
        count += 1
        if count >= total_requests:
            break
    return len(unique_objects)


def get_dynamic_capacities(
    trace_path: str,
    total_requests: int,
    percentages: List[float],
) -> List[int]:
    n_unique = count_distinct_objects(trace_path, total_requests)
    capacities = [max(1, int(n_unique * p / 100.0)) for p in percentages]
    print(f"Jumlah objek unik: {n_unique}")
    print(f"Kapasitas cache (objek): {capacities}")
    return capacities


def get_next_run_id(results_root: str, dataset: str, model: str, cache_size: int) -> Tuple[str, str]:
    dataset_dir = os.path.join(results_root, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    existing_ids: List[int] = []
    for fname in os.listdir(dataset_dir):
        if not (
            fname.endswith(f"_{model}_{cache_size}.json")
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


def run_single_capacity(
    trace_path: str,
    total_requests: int,
    warmup_requests: int,
    slot_size: int,
    capacity_objects: int,
    feature_set: str,
    il_cfg: ILConfig,
    gdbt_cfg: GDBTConfig,
    slot_log_path: Optional[str] = None,
    dataset_name: str = "",
    model_name: str = "",
    disable_progress: bool = False,
    prediction_mode: str = "per_candidate",
) -> Tuple[CacheStats, List[Dict[str, Any]], Dict[str, Any]]:
    if prediction_mode not in ("per_candidate", "batch"):
        raise ValueError("prediction_mode must be 'per_candidate' or 'batch'.")
    reader = TraceReader(path=trace_path, max_rows=total_requests)
    req_iter = reader.iter_requests()

    num_gaps = il_cfg.num_gaps
    num_features = _get_feature_dim(num_gaps, feature_set)
    feature_table = FeatureTable(
        L=num_gaps,
        missing_gap_value=il_cfg.missing_gap_value,
    )
    gdbt_model = GDBTCachePredictorStandard(il_cfg, gdbt_cfg, num_features)
    cache = LRUCache(capacity_objects=capacity_objects)
    stats = CacheStats(capacity_objects=capacity_objects)

    total_feat_time_s = 0.0
    total_score_time_s = 0.0
    total_update_time_s = 0.0
    total_slot_time_s = 0.0
    total_slots = 0
    total_cache_slots = 0
    max_feature_bytes = 0
    total_miss_requests = 0
    total_miss_candidates = 0
    total_admit_selected = 0
    total_admit_applied = 0
    total_reject_requests = 0
    total_admit_true_popular = 0
    total_hits_from_admitted = 0
    total_pollution_total = 0
    total_pollution_count = 0
    last_pollution_finalized = 0
    slot_records: List[Dict[str, Any]] = []

    rebuild_logs: List[Dict[str, Any]] = []
    global_idx = 0
    slot_req_count = 0
    slot_index = 0
    slot_stats: Dict[str, Dict[str, Any]] = {}
    slot_cache_requests = 0
    slot_cache_hits = 0
    slot_cache_misses = 0
    slot_miss_obj_ids: List[str] = []
    slot_miss_features: List[List[float]] = []
    prev_ts: Optional[float] = None

    freq_history: Dict[str, List[Tuple[int, int]]] = {}
    cum_counts: Dict[str, int] = {}
    admitted_by_slot: Dict[int, set[str]] = {}
    admitted_hit_by_slot: Dict[int, set[str]] = {}
    active_admit_slots_by_obj: Dict[str, set[int]] = {}
    pending_admit_set: set[str] = set()
    pending_admit_source_slot: Optional[int] = None
    applied_admit_set: set[str] = set()
    applied_admit_source_slot: Optional[int] = None
    current_slot_apply_pending_time_s = 0.0

    eval_requests_planned = max(total_requests - warmup_requests, 0)
    pbar = tqdm(
        total=eval_requests_planned,
        desc=f"GDBT Eval (cap={capacity_objects})",
        unit="req",
        disable=disable_progress,
    )

    warmup_slots = warmup_requests // slot_size if slot_size > 0 else 0
    slots_per_rebuild = (
        gdbt_cfg.update_interval_requests // slot_size if slot_size > 0 else 1
    )
    slots_since_rebuild = 0

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

    def _write_slot_log(payload: Dict[str, Any]) -> None:
        slot_records.append(payload)

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
        for obj_id in applied_admit_set:
            cache.insert(obj_id)
            _register_admission(obj_id, next_slot_num)

    def _finalize_slot() -> None:
        nonlocal slot_index
        nonlocal slot_stats
        nonlocal slot_cache_requests
        nonlocal slot_cache_hits
        nonlocal slot_cache_misses
        nonlocal slot_miss_obj_ids
        nonlocal slot_miss_features
        nonlocal total_feat_time_s
        nonlocal total_score_time_s
        nonlocal total_update_time_s
        nonlocal total_slot_time_s
        nonlocal total_slots
        nonlocal total_cache_slots
        nonlocal max_feature_bytes
        nonlocal total_miss_requests
        nonlocal total_miss_candidates
        nonlocal total_admit_selected
        nonlocal total_reject_requests
        nonlocal total_admit_true_popular
        nonlocal total_hits_from_admitted
        nonlocal total_pollution_total
        nonlocal total_pollution_count
        nonlocal last_pollution_finalized
        nonlocal slots_since_rebuild
        nonlocal pending_admit_set
        nonlocal pending_admit_source_slot
        nonlocal applied_admit_set
        nonlocal applied_admit_source_slot
        nonlocal current_slot_apply_pending_time_s

        if not slot_stats:
            return

        feat_start = now_ns()
        D_t = build_slot_dataset_from_stats(
            slot_stats,
            il_cfg.pop_top_percent,
            num_gaps,
            il_cfg.missing_gap_value,
            feature_set,
            slot_index,
            freq_history,
            cum_counts,
            IL_LABEL_TOPK_ROUNDING,
            IL_LABEL_TIE_BREAK,
        )
        top_ids, label_info = _select_top_ids_from_stats(
            slot_stats,
            il_cfg.pop_top_percent,
            IL_LABEL_TOPK_ROUNDING,
            IL_LABEL_TIE_BREAK,
        )
        feat_time_s = elapsed_s(feat_start)

        slot_index += 1
        slot_num = slot_index
        total_slots += 1
        is_cache_slot = slot_num > warmup_slots

        score_start = now_ns()
        candidate_scores: Dict[str, float] = {}
        candidate_ids: List[str] = []
        slot_feature_bytes = 0
        if slot_miss_features:
            X_miss = np.asarray(slot_miss_features, dtype=float)
            slot_feature_bytes = int(X_miss.nbytes)
            if prediction_mode == "batch":
                miss_scores = gdbt_model.predict_proba_batch(X_miss)
            else:
                miss_scores = [
                    gdbt_model.predict_proba(features)
                    for features in slot_miss_features
                ]
            for obj_id, score in zip(slot_miss_obj_ids, miss_scores):
                score_val = float(score)
                prev = candidate_scores.get(obj_id)
                if prev is None:
                    candidate_scores[obj_id] = score_val
                    candidate_ids.append(obj_id)
                elif score_val > prev:
                    candidate_scores[obj_id] = score_val
        score_time_s = elapsed_s(score_start)

        if slot_feature_bytes > max_feature_bytes:
            max_feature_bytes = slot_feature_bytes

        miss_requests = int(slot_cache_misses)
        miss_candidates = len(candidate_ids)
        admit_selected = 0
        reject_requests = 0
        next_admit_set: set[str] = set()
        admit_true_applied = None
        admit_precision = None
        candidate_score_list = [candidate_scores[obj_id] for obj_id in candidate_ids]
        time_budget_s = 0.0
        time_guard_signal_s = 0.0
        select_time_s = 0.0

        if is_cache_slot:
            total_cache_slots += 1
            total_miss_requests += miss_requests
            total_miss_candidates += miss_candidates
            select_start = now_ns()
            next_admit_set = {
                obj_id for obj_id, score in zip(candidate_ids, candidate_score_list)
                if score >= GDBT_ADMIT_THRESHOLD
            }
            select_time_s = elapsed_s(select_start)
            admit_selected = len(next_admit_set)
            reject_requests = max(miss_candidates - admit_selected, 0)
            total_admit_selected += admit_selected
            total_reject_requests += reject_requests
            if applied_admit_set:
                admit_true_applied = len(applied_admit_set & top_ids)
                total_admit_true_popular += admit_true_applied
                admit_precision = (
                    float(admit_true_applied / len(applied_admit_set))
                    if applied_admit_set
                    else None
                )

        pending_admit_set = next_admit_set
        pending_admit_source_slot = slot_num if next_admit_set else None

        time_buffer_add_s = 0.0
        time_history_update_s = 0.0
        time_rebuild_s = 0.0
        buffer_start = now_ns()
        gdbt_model.add_training_batch(D_t)
        time_buffer_add_s = elapsed_s(buffer_start)
        history_start = now_ns()
        _update_freq_history(
            slot_num,
            slot_stats,
            freq_history,
            cum_counts,
            FEATURE_WINDOWS["long"],
        )
        time_history_update_s = elapsed_s(history_start)

        rebuild_info = None
        if slot_num == warmup_slots:
            rebuild_start = now_ns()
            rebuild_info = gdbt_model.rebuild_model()
            time_rebuild_s = elapsed_s(rebuild_start)
            rebuild_info["phase"] = "warmup_complete"
            rebuild_info["global_request_idx"] = global_idx
            rebuild_logs.append(rebuild_info)
            gdbt_model.clear_buffer()
            slots_since_rebuild = 0
        elif slot_num > warmup_slots and global_idx >= warmup_requests:
            slots_since_rebuild += 1
            if slots_since_rebuild >= max(1, slots_per_rebuild):
                rebuild_start = now_ns()
                rebuild_info = gdbt_model.rebuild_model()
                time_rebuild_s = elapsed_s(rebuild_start)
                rebuild_info["phase"] = "eval_rebuild"
                rebuild_info["global_request_idx"] = global_idx
                rebuild_info["cache_hit_ratio_so_far"] = stats.hit_ratio
                rebuild_logs.append(rebuild_info)
                gdbt_model.clear_buffer()
                slots_since_rebuild = 0
                pbar.set_postfix(
                    hr=f"{stats.hit_ratio:.4f}",
                    rebuilds=gdbt_model.get_stats()["num_rebuilds"],
                )
        update_time_s = time_buffer_add_s + time_history_update_s + time_rebuild_s

        total_feat_time_s += feat_time_s
        total_score_time_s += score_time_s
        total_update_time_s += update_time_s
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

        current_slot_hr = (
            float(slot_cache_hits / slot_cache_requests)
            if slot_cache_requests > 0
            else None
        )
        phase = "warmup" if slot_num <= warmup_slots else "cache"
        selected_score_mean = (
            float(np.mean([candidate_scores[obj_id] for obj_id in next_admit_set]))
            if next_admit_set
            else None
        )
        _write_slot_log(
            {
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
                "slot_hit_ratio": current_slot_hr,
                "hit_ratio": stats.hit_ratio,
                "miss_requests": miss_requests,
                "miss_candidates": miss_candidates,
                "candidate_instances": int(len(slot_miss_obj_ids)),
                "label_k_actual": int(label_info.get("k_actual", 0)),
                "admit_threshold": GDBT_ADMIT_THRESHOLD,
                "gbdt_backend": "sklearn.GradientBoostingClassifier",
                "optimized_backend": False,
                "lightgbm_used": False,
                "prediction_mode": prediction_mode,
                "admit_selected": admit_selected,
                "admit_applied": len(applied_admit_set),
                "admit_applied_from_slot": applied_admit_source_slot,
                "admit_true_from_applied": admit_true_applied,
                "admission_precision": admit_precision,
                "admission_rate": (
                    float(admit_selected / miss_candidates) if miss_candidates > 0 else 0.0
                ),
                "selected_score_mean": selected_score_mean,
                "rebuild_triggered": rebuild_info is not None,
                "rebuild_phase": None if rebuild_info is None else rebuild_info.get("phase"),
                "has_model": gdbt_model.model is not None,
                "time_feat_s": float(feat_time_s),
                "time_score_s": float(score_time_s),
                "time_budget_s": float(time_budget_s),
                "time_guard_signal_s": float(time_guard_signal_s),
                "time_guard_s": 0.0,
                "time_select_s": float(select_time_s),
                "time_update_s": float(update_time_s),
                "time_buffer_add_s": float(time_buffer_add_s),
                "time_history_update_s": float(time_history_update_s),
                "time_rebuild_s": float(time_rebuild_s),
                "time_apply_pending_s": float(current_slot_apply_pending_time_s),
                "time_slot_control_s": float(slot_time_s),
                "time_boundary_total_s": float(boundary_total_s),
                "time_slot_s": float(slot_time_s),
                "feature_bytes": int(slot_feature_bytes),
                "current_rss_mb": get_current_rss_mb(),
                "peak_rss_mb": get_peak_rss_mb(),
                "num_features": int(num_features),
            }
        )
        current_slot_apply_pending_time_s = 0.0

        expired = slot_num - POLLUTION_WINDOW_SLOTS
        if expired >= 1:
            polluted, total = _finalize_pollution(expired)
            total_pollution_count += polluted
            total_pollution_total += total
            _prune_active_slots(expired)
            last_pollution_finalized = expired

        slot_stats = {}
        slot_cache_requests = 0
        slot_cache_hits = 0
        slot_cache_misses = 0
        slot_miss_obj_ids = []
        slot_miss_features = []
        applied_admit_set = set()
        applied_admit_source_slot = None

    try:
        while True:
            try:
                req = next(req_iter)
            except StopIteration:
                if slot_stats:
                    _finalize_slot()
                break

            if global_idx >= total_requests:
                break

            obj_id = req["object_id"]

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

            gaps = feature_table.update_and_get_gaps(obj_id, ts)

            info = slot_stats.get(obj_id)
            if info is None:
                slot_stats[obj_id] = {"freq": 1, "last_gaps": gaps}
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
            features = _build_feature_vector(feature_set, gaps, history_feats)

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
                    slot_miss_obj_ids.append(obj_id)
                    slot_miss_features.append(features)
                pbar.update(1)

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
                slot_req_count = 0

            global_idx += 1
    finally:
        pbar.close()
        if slot_log_path:
            write_jsonl(slot_log_path, slot_records)

    for slot_num in range(last_pollution_finalized + 1, slot_index + 1):
        polluted, total = _finalize_pollution(slot_num)
        total_pollution_count += polluted
        total_pollution_total += total

    final_stats = gdbt_model.get_stats()
    fi = gdbt_model.get_feature_importances()
    final_log = {
        "final_model_stats": final_stats,
        "feature_importances": fi.tolist() if fi is not None else None,
    }
    rebuild_logs.append(final_log)

    overhead_summary = summarize_overhead_slots(slot_records, phase="cache")
    summary_metrics = {
        "slots_processed": int(total_slots),
        "cache_slots_processed": int(total_cache_slots),
        "avg_feat_time_s": float(total_feat_time_s / total_slots) if total_slots > 0 else None,
        "avg_score_time_s": float(total_score_time_s / total_slots) if total_slots > 0 else None,
        "avg_update_time_s": float(total_update_time_s / total_slots) if total_slots > 0 else None,
        "avg_slot_time_s": float(total_slot_time_s / total_slots) if total_slots > 0 else None,
        "feat_time_total_s": float(total_feat_time_s),
        "score_time_total_s": float(total_score_time_s),
        "update_time_total_s": float(total_update_time_s),
        "slot_time_total_s": float(total_slot_time_s),
        "miss_requests_total": int(total_miss_requests),
        "miss_candidates_total": int(total_miss_candidates),
        "admit_selected_total": int(total_admit_selected),
        "admit_applied_total": int(total_admit_applied),
        "reject_requests_total": int(total_reject_requests),
        "admit_true_popular_total": int(total_admit_true_popular),
        "hits_from_admitted_total": int(total_hits_from_admitted),
        "admission_rate_total": (
            float(total_admit_selected / total_miss_candidates)
            if total_miss_candidates > 0
            else None
        ),
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
        "pollution_total": int(total_pollution_total),
        "pollution_count": int(total_pollution_count),
        "pollution_rate_total": (
            float(total_pollution_count / total_pollution_total)
            if total_pollution_total > 0
            else None
        ),
        "admit_threshold": float(GDBT_ADMIT_THRESHOLD),
        "protocol": "one_slot_delayed_admission",
        "pollution_window_slots": int(POLLUTION_WINDOW_SLOTS),
        "max_feature_bytes": int(max_feature_bytes),
        "rss_max_mb": get_peak_rss_mb(),
        "num_features": int(num_features),
    }
    summary_metrics.update(overhead_summary)
    for field in ("avg_feat_time_s", "avg_score_time_s", "avg_update_time_s"):
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

    return stats, rebuild_logs, summary_metrics


def run_experiment(
    ds_cfg: Dict[str, Any],
    feature_set: str = DEFAULT_FEATURE_SET,
    model_name: Optional[str] = None,
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
    prediction_mode: str = "per_candidate",
) -> None:
    np.random.seed(seed)
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    if prediction_mode not in ("per_candidate", "batch"):
        raise ValueError("prediction_mode must be 'per_candidate' or 'batch'.")
    if model_name is None:
        suffix = "" if prediction_mode == "per_candidate" else "_sklearn_batch"
        model_name = f"delayed_gdbt_standard_{feature_set}{suffix}_overhead"

    trace_path = ds_cfg["path"]
    slot_size = ds_cfg["slot_size"]
    warmup_requests = ds_cfg["num_warmup_requests"]
    total_requests = ds_cfg["num_total_requests"]
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
            f"warmup_requests ({warmup_requests}) harus kelipatan slot_size ({slot_size})."
        )

    dataset_name = ds_cfg["name"]
    il_cfg = ILConfig()
    gdbt_cfg = GDBTConfig()

    print(f"=== Delayed-GBDT Edge Cache Overhead ({dataset_name} trace) ===")
    print(f"Trace path          : {trace_path}")
    print(f"Total requests      : {total_requests}")
    print(f"Warm-up requests    : {warmup_requests}")
    print(f"Slot size           : {slot_size}")
    print(f"Feature set         : {feature_set}")
    print(f"Num gaps            : {il_cfg.num_gaps}")
    print(f"Num features        : {_get_feature_dim(il_cfg.num_gaps, feature_set)}")
    print(f"Top percent         : {il_cfg.pop_top_percent}")
    print(f"GDBT n_estimators   : {gdbt_cfg.n_estimators}")
    print(f"GDBT update_interval: {gdbt_cfg.update_interval_requests}")
    print(f"GDBT backend         : sklearn.GradientBoostingClassifier")
    print(f"Optimized backend    : False")
    print(f"Prediction mode      : {prediction_mode}")
    print(f"Admission protocol  : one-slot delayed apply")
    print(f"Admission threshold : {GDBT_ADMIT_THRESHOLD}")
    print(f"Cache size (%)      : {list(cache_size_percentages)}")
    print()

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
        print(f"\n[RUN] Delayed-GBDT+LRU, Capacity={capacity} objects")
        if benchmark_mode:
            run_start = datetime.now(timezone.utc).isoformat()
            run_id, run_dir = prepare_run_dir(
                results_root, dataset_name, model_name, capacity, run_id_override
            )
            per_slot_path = os.path.join(run_dir, "slot_log.jsonl")
            summary_path = os.path.join(run_dir, "overhead_summary.json")
            rebuild_path = os.path.join(run_dir, "rebuilds.json")
            config_path = os.path.join(run_dir, "config.json")
            metadata_path = os.path.join(run_dir, "run_metadata.json")
        else:
            run_id, dataset_dir = get_next_run_id(
                results_root, dataset_name, model_name, capacity
            )
            per_slot_path = os.path.join(
                dataset_dir,
                f"{run_id}_{model_name}_{capacity}.jsonl",
            )
            summary_path = os.path.join(
                dataset_dir,
                f"{run_id}_summary_{model_name}_{capacity}.json",
            )
            rebuild_path = os.path.join(
                dataset_dir,
                f"{run_id}_{model_name}_{capacity}_rebuilds.json",
            )
            config_path = ""
            metadata_path = ""

        try:
            stats, rebuild_logs, summary_metrics = run_single_capacity(
                trace_path=trace_path,
                total_requests=total_requests,
                warmup_requests=warmup_requests,
                slot_size=slot_size,
                capacity_objects=capacity,
                feature_set=feature_set,
                il_cfg=il_cfg,
                gdbt_cfg=gdbt_cfg,
                slot_log_path=per_slot_path,
                dataset_name=dataset_name,
                model_name=model_name,
                disable_progress=disable_progress or benchmark_mode,
                prediction_mode=prediction_mode,
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
                        "prediction_mode": prediction_mode,
                    },
                    exc,
                )
            raise
        results.append(stats)

        write_summary_json(
            rebuild_path,
            {
                "dataset": dataset_name,
                "model": model_name,
                "feature_set": feature_set,
                "cache_size_objects": capacity,
                "rebuilds": rebuild_logs,
            },
        )

        summary_payload = {
            "dataset": dataset_name,
            "model": model_name,
            "model_name": model_name,
            "feature_set": feature_set,
            "cache_size_objects": capacity,
            "capacity_objects": capacity,
            "capacity_percent": capacity_percent,
            "total_requests": total_requests,
            "warmup_requests": warmup_requests,
            "slot_size": slot_size,
            "gdbt_n_estimators": gdbt_cfg.n_estimators,
            "n_estimators": gdbt_cfg.n_estimators,
            "gdbt_update_interval": gdbt_cfg.update_interval_requests,
            "update_interval_requests": gdbt_cfg.update_interval_requests,
            "pop_top_percent": il_cfg.pop_top_percent,
            "protocol": "one_slot_delayed_admission",
            "admission_protocol": "one_slot_delayed_admission",
            "admit_threshold": GDBT_ADMIT_THRESHOLD,
            "gbdt_backend": "sklearn.GradientBoostingClassifier",
            "optimized_backend": False,
            "lightgbm_used": False,
            "prediction_mode": prediction_mode,
            "hit_ratio": stats.hit_ratio,
            "cache_hits": stats.cache_hits,
            "cache_requests": stats.total_requests,
            "num_rebuilds": rebuild_logs[-1]["final_model_stats"]["num_rebuilds"],
        }
        summary_payload.update(summary_metrics)
        write_summary_json(summary_path, summary_payload)
        if benchmark_mode:
            config_payload = {
                "dataset": dataset_name,
                "model": model_name,
                "feature_set": feature_set,
                "cache_size_objects": capacity,
                "capacity_percent": capacity_percent,
                "total_requests": total_requests,
                "warmup_requests": warmup_requests,
                "slot_size": slot_size,
                "seed": seed,
                "gbdt_backend": "sklearn.GradientBoostingClassifier",
                "optimized_backend": False,
                "lightgbm_used": False,
                "admission_protocol": "one_slot_delayed_admission",
                "prediction_mode": prediction_mode,
                "n_estimators": gdbt_cfg.n_estimators,
                "update_interval_requests": gdbt_cfg.update_interval_requests,
                "admit_threshold": GDBT_ADMIT_THRESHOLD,
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

        print(
            f"      Hit ratio = {stats.hit_ratio:.4f} "
            f"({stats.cache_hits}/{stats.total_requests})"
        )
        print(f"      [LOG] per-slot -> {per_slot_path}")
        print(f"      [LOG] rebuilds -> {rebuild_path}")
        print(f"      [LOG] summary  -> {summary_path}")

    print("\n=== Summary ===")
    for capacity, stats in zip(capacities_objects, results):
        print(
            f"Delayed-GBDT+LRU, cache_size={capacity}, "
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
        "num_features": _get_feature_dim(il_cfg.num_gaps, feature_set),
        "protocol": "one_slot_delayed_admission",
        "admission_protocol": "one_slot_delayed_admission",
        "admit_threshold": GDBT_ADMIT_THRESHOLD,
        "gbdt_backend": "sklearn.GradientBoostingClassifier",
        "optimized_backend": False,
        "lightgbm_used": False,
        "prediction_mode": prediction_mode,
        "n_estimators": gdbt_cfg.n_estimators,
        "update_interval_requests": gdbt_cfg.update_interval_requests,
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
    parser = argparse.ArgumentParser(description="GDBT overhead runner (slot-level timing).")
    parser.add_argument("--dataset", choices=["wikipedia_september_2007", "wiki2018"], required=True)
    parser.add_argument("--feature-set", default="A2")
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
    parser.add_argument("--prediction-mode", choices=["per_candidate", "batch"], default="per_candidate")
    args = parser.parse_args()

    ds_cfg = _resolve_dataset(args.dataset)
    capacity_objects = args.capacity_objects if args.capacity_objects is not None else args.cache_size_objects
    try:
        run_experiment(
            ds_cfg,
            feature_set=args.feature_set,
            cache_size_objects=capacity_objects,
            capacity_percent=args.capacity_percent,
            results_root=args.results_root,
            disable_progress=args.disable_progress,
            benchmark_mode=args.benchmark_mode,
            run_id_override=args.run_id,
            seed=args.seed,
            max_requests=args.max_requests,
            smoke_test=args.smoke_test,
            prediction_mode=args.prediction_mode,
        )
    except Exception as exc:
        error_path = os.path.join(
            args.results_root,
            f"error_gbdt_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
        )
        write_error_json(error_path, vars(args), exc)
        raise


if __name__ == "__main__":
    main()
