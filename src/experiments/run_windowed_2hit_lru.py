from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cache.cache_simulator import CacheStats
from src.cache.lru import LRUCache
from src.config.experiment_config import (
    CacheConfig,
    DatasetConfig,
    WIKI2018,
    WIKIPEDIA_OKTOBER_2007,
    WIKIPEDIA_SEPTEMBER_2007,
)
from src.data.trace_reader import TraceReader


POLLUTION_WINDOW_SLOTS = 1
BASELINE_NAME = "Windowed-2Hit-LRU"
BASELINE_TYPE = "rule_based_selective_admission"


@dataclass
class RunResult:
    stats: CacheStats
    summary: Dict[str, Any]
    slot_records: List[Dict[str, Any]]


def resolve_dataset(name: str) -> DatasetConfig:
    if name == "wikipedia_september_2007":
        return WIKIPEDIA_SEPTEMBER_2007
    if name == "wikipedia_oktober_2007":
        return WIKIPEDIA_OKTOBER_2007
    if name == "wiki2018":
        return WIKI2018
    raise ValueError(f"Unknown dataset: {name}")


def count_distinct_objects(trace_path: str, total_requests: int) -> int:
    reader = TraceReader(path=trace_path, max_rows=total_requests)
    unique_objects: set[str] = set()
    for idx, req in enumerate(reader.iter_requests()):
        if idx >= total_requests:
            break
        unique_objects.add(str(req["object_id"]))
    return len(unique_objects)


def get_capacity_from_percent(
    trace_path: str,
    total_requests: int,
    capacity_percent: float,
) -> int:
    num_unique = count_distinct_objects(trace_path, total_requests)
    return max(1, int(num_unique * float(capacity_percent) / 100.0))


def get_next_run_id(results_root: str, dataset: str, model: str, cache_size: int) -> Tuple[str, str]:
    dataset_dir = os.path.join(results_root, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    existing_ids: List[int] = []
    for fname in os.listdir(dataset_dir):
        if not (
            fname.endswith(f"_{model}_{cache_size}.json")
            or fname.endswith(f"_{model}_{cache_size}_slot_log.jsonl")
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
    return f"{next_id:03d}", dataset_dir


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_ratio(num: int, den: int) -> Optional[float]:
    return float(num / den) if den > 0 else None


def _model_name(window_slots: int) -> str:
    return f"windowed_2hit_lru_w{int(window_slots)}slot"


def _iter_trace_objects(trace_path: str, total_requests: int) -> Iterable[str]:
    reader = TraceReader(path=trace_path, max_rows=total_requests)
    for req in reader.iter_requests():
        yield str(req["object_id"])


def run_single_capacity(
    object_ids: Iterable[str],
    trace_path: str,
    total_requests: int,
    warmup_requests: int,
    slot_size: int,
    capacity_objects: int,
    window_slots: int,
    dataset_name: str,
    capacity_percent: Optional[float] = None,
    seed: int = 42,
    disable_progress: bool = False,
) -> RunResult:
    if window_slots < 1:
        raise ValueError("window_slots must be >= 1.")
    if warmup_requests % slot_size != 0:
        raise ValueError("warmup_requests must be a multiple of slot_size.")

    window_requests = int(window_slots * slot_size)
    warmup_slots = warmup_requests // slot_size

    cache = LRUCache(capacity_objects=capacity_objects)
    stats = CacheStats(capacity_objects=capacity_objects)

    last_uncached_request: Dict[str, int] = {}
    last_uncached_queue: Deque[Tuple[int, str]] = deque()
    pending_admit_order: List[str] = []
    pending_admit_set: set[str] = set()
    admitted_by_slot: Dict[int, set[str]] = {}
    admitted_hit_by_slot: Dict[int, set[str]] = {}
    active_admit_slots_by_obj: Dict[str, set[int]] = {}

    slot_records: List[Dict[str, Any]] = []
    global_idx = 0
    slot_req_count = 0
    slot_index = 0
    last_pollution_finalized = 0

    slot_cache_requests = 0
    slot_cache_hits = 0
    slot_cache_misses = 0
    slot_admit_scheduled = 0
    slot_pending_duplicates = 0
    current_slot_admit_applied = 0
    current_slot_skipped_already_cached = 0
    current_slot_evictions = 0

    total_miss_requests = 0
    scheduled_admissions_total = 0
    pending_duplicate_total = 0
    skipped_already_cached_total = 0
    total_admit_applied = 0
    eviction_count = 0
    pollution_count = 0
    pollution_total = 0
    hits_from_admitted_total = 0

    total_slots_expected = (total_requests + slot_size - 1) // slot_size
    pbar = tqdm(
        total=total_slots_expected,
        desc=f"Windowed-2Hit-LRU cap={capacity_objects}",
        unit="slot",
        disable=disable_progress,
    )

    def _reset_window_state() -> None:
        last_uncached_request.clear()
        last_uncached_queue.clear()
        pending_admit_order.clear()
        pending_admit_set.clear()

    def _clean_evicted_state(evicted_id: str) -> None:
        last_uncached_request.pop(evicted_id, None)
        if evicted_id in pending_admit_set:
            pending_admit_set.discard(evicted_id)
            pending_admit_order[:] = [obj for obj in pending_admit_order if obj != evicted_id]
        active_admit_slots_by_obj.pop(evicted_id, None)

    def _apply_pending_admits(next_slot_num: int) -> Tuple[int, int, int]:
        nonlocal total_admit_applied
        nonlocal skipped_already_cached_total
        nonlocal eviction_count

        applied = 0
        skipped = 0
        evictions = 0
        if not pending_admit_order:
            pending_admit_set.clear()
            return applied, skipped, evictions

        for obj_id in list(pending_admit_order):
            if obj_id not in pending_admit_set:
                continue
            if cache.contains(obj_id):
                skipped += 1
                skipped_already_cached_total += 1
                continue
            inserted, evicted_id = cache.insert_with_eviction(obj_id)
            if inserted:
                applied += 1
                total_admit_applied += 1
                admitted_by_slot.setdefault(next_slot_num, set()).add(obj_id)
                active_admit_slots_by_obj.setdefault(obj_id, set()).add(next_slot_num)
            if evicted_id is not None:
                evictions += 1
                eviction_count += 1
                _clean_evicted_state(evicted_id)

        pending_admit_order.clear()
        pending_admit_set.clear()
        return applied, skipped, evictions

    def _mark_admit_hit(obj_id: str) -> None:
        active_slots = active_admit_slots_by_obj.get(obj_id)
        if not active_slots:
            return
        for applied_slot in sorted(active_slots):
            admitted_hit_by_slot.setdefault(applied_slot, set()).add(obj_id)

    def _finalize_pollution(expired_slot: int) -> None:
        nonlocal pollution_count
        nonlocal pollution_total
        nonlocal hits_from_admitted_total
        nonlocal last_pollution_finalized

        admitted = admitted_by_slot.get(expired_slot, set())
        hit_objects = admitted_hit_by_slot.get(expired_slot, set())
        polluted = len(admitted - hit_objects)
        total = len(admitted)
        pollution_count += polluted
        pollution_total += total
        hits_from_admitted_total += len(admitted & hit_objects)

        for obj_id in admitted:
            active = active_admit_slots_by_obj.get(obj_id)
            if not active:
                continue
            active.discard(expired_slot)
            if not active:
                active_admit_slots_by_obj.pop(obj_id, None)
        last_pollution_finalized = expired_slot

    def _prune_stale_uncached(current_idx: int) -> None:
        while last_uncached_queue:
            old_idx, old_obj = last_uncached_queue[0]
            if current_idx - old_idx <= window_requests:
                break
            last_uncached_queue.popleft()
            if last_uncached_request.get(old_obj) == old_idx:
                last_uncached_request.pop(old_obj, None)

    def _finalize_slot() -> None:
        nonlocal slot_index
        nonlocal slot_cache_requests
        nonlocal slot_cache_hits
        nonlocal slot_cache_misses
        nonlocal slot_admit_scheduled
        nonlocal slot_pending_duplicates
        nonlocal current_slot_admit_applied
        nonlocal current_slot_skipped_already_cached
        nonlocal current_slot_evictions

        slot_index += 1
        slot_num = slot_index
        phase = "warmup" if slot_num <= warmup_slots else "cache"
        slot_records.append(
            {
                "slot_index": slot_num,
                "phase": phase,
                "slot_cache_requests": int(slot_cache_requests),
                "slot_cache_hits": int(slot_cache_hits),
                "slot_cache_misses": int(slot_cache_misses),
                "slot_hit_ratio": _safe_ratio(slot_cache_hits, slot_cache_requests),
                "slot_admit_scheduled": int(slot_admit_scheduled),
                "slot_admit_applied": int(current_slot_admit_applied),
                "slot_pending_duplicates": int(slot_pending_duplicates),
                "slot_skipped_already_cached": int(current_slot_skipped_already_cached),
                "slot_evictions": int(current_slot_evictions),
                "cache_size": int(len(cache)),
                "pending_admit_count_end": int(len(pending_admit_set)),
                "tracked_uncached_count_end": int(len(last_uncached_request)),
            }
        )

        expired = slot_num - POLLUTION_WINDOW_SLOTS
        if expired >= 1:
            _finalize_pollution(expired)

        slot_cache_requests = 0
        slot_cache_hits = 0
        slot_cache_misses = 0
        slot_admit_scheduled = 0
        slot_pending_duplicates = 0
        current_slot_admit_applied = 0
        current_slot_skipped_already_cached = 0
        current_slot_evictions = 0

    try:
        for obj_id in object_ids:
            if global_idx >= total_requests:
                break

            is_eval = global_idx >= warmup_requests
            if is_eval:
                stats.total_requests += 1
                slot_cache_requests += 1
                if cache.access(obj_id):
                    stats.cache_hits += 1
                    slot_cache_hits += 1
                    _mark_admit_hit(obj_id)
                else:
                    slot_cache_misses += 1
                    total_miss_requests += 1
                    _prune_stale_uncached(global_idx)

                    if obj_id in pending_admit_set:
                        pending_duplicate_total += 1
                        slot_pending_duplicates += 1
                    else:
                        prev_idx = last_uncached_request.get(obj_id)
                        if prev_idx is not None and global_idx - prev_idx <= window_requests:
                            pending_admit_order.append(obj_id)
                            pending_admit_set.add(obj_id)
                            scheduled_admissions_total += 1
                            slot_admit_scheduled += 1
                            last_uncached_request.pop(obj_id, None)
                        else:
                            last_uncached_request[obj_id] = global_idx
                            last_uncached_queue.append((global_idx, obj_id))

            global_idx += 1
            slot_req_count += 1

            if slot_req_count >= slot_size:
                _finalize_slot()
                next_slot_start = slot_index * slot_size
                if next_slot_start == warmup_requests:
                    _reset_window_state()
                if next_slot_start < total_requests and next_slot_start >= warmup_requests:
                    (
                        current_slot_admit_applied,
                        current_slot_skipped_already_cached,
                        current_slot_evictions,
                    ) = _apply_pending_admits(slot_index + 1)
                else:
                    pending_admit_order.clear()
                    pending_admit_set.clear()
                    current_slot_admit_applied = 0
                    current_slot_skipped_already_cached = 0
                    current_slot_evictions = 0
                slot_req_count = 0
                pbar.update(1)

        if slot_req_count > 0:
            _finalize_slot()
            pbar.update(1)
    finally:
        pbar.close()

    for expired_slot in range(last_pollution_finalized + 1, slot_index + 1):
        _finalize_pollution(expired_slot)

    total_cache_slots = max(0, slot_index - warmup_slots)
    summary = {
        "dataset": dataset_name,
        "model_name": _model_name(window_slots),
        "baseline_name": BASELINE_NAME,
        "baseline_type": BASELINE_TYPE,
        "learner_baseline": False,
        "included_in_overhead_analysis": False,
        "feature_set": "none",
        "uses_features": False,
        "uses_labels": False,
        "uses_scores": False,
        "eviction_policy": "fixed_lru",
        "admission_policy": "windowed_cache_on_second_request",
        "M": 2,
        "delayed_apply": True,
        "window_unit": "requests",
        "window_slots": int(window_slots),
        "window_requests": int(window_requests),
        "trace_path": trace_path,
        "total_requests": int(total_requests),
        "warmup_requests": int(warmup_requests),
        "warmup_behavior": (
            "warm-up requests advance the stream only; cache and Windowed-2Hit state "
            "start empty at evaluation"
        ),
        "slot_size": int(slot_size),
        "capacity_objects": int(capacity_objects),
        "capacity_percent": capacity_percent,
        "cache_hits": int(stats.cache_hits),
        "cache_requests": int(stats.total_requests),
        "hit_ratio": float(stats.hit_ratio),
        "miss_requests_total": int(total_miss_requests),
        "miss_candidates_total": int(scheduled_admissions_total),
        "scheduled_admissions_total": int(scheduled_admissions_total),
        "admit_selected_total": int(scheduled_admissions_total),
        "admit_applied_total": int(total_admit_applied),
        "pending_duplicate_total": int(pending_duplicate_total),
        "skipped_already_cached_total": int(skipped_already_cached_total),
        "eviction_count": int(eviction_count),
        "pollution_count": int(pollution_count),
        "pollution_total": int(pollution_total),
        "pollution_rate_total": _safe_ratio(pollution_count, pollution_total),
        "hits_from_admitted_total": int(hits_from_admitted_total),
        "hit_yield_total": _safe_ratio(hits_from_admitted_total, total_admit_applied),
        "total_slots_processed": int(slot_index),
        "total_cache_slots": int(total_cache_slots),
        "warmup_slots": int(warmup_slots),
        "cache_size_final": int(len(cache)),
        "reproducibility_seed": int(seed),
    }
    return RunResult(stats=stats, summary=summary, slot_records=slot_records)


def run_sequence_for_smoke(
    sequence: Sequence[str],
    capacity_objects: int,
    warmup_requests: int,
    slot_size: int,
    window_slots: int,
) -> RunResult:
    return run_single_capacity(
        object_ids=[str(x) for x in sequence],
        trace_path="<synthetic>",
        total_requests=len(sequence),
        warmup_requests=warmup_requests,
        slot_size=slot_size,
        capacity_objects=capacity_objects,
        window_slots=window_slots,
        dataset_name="synthetic",
        capacity_percent=None,
        seed=42,
        disable_progress=True,
    )


def run_smoke_test() -> None:
    result = run_sequence_for_smoke(
        ["W", "X", "A", "A", "A", "C", "D", "E"],
        capacity_objects=1,
        warmup_requests=2,
        slot_size=2,
        window_slots=1,
    )
    slots = result.slot_records
    assert result.stats.total_requests == 6, "warm-up requests must be excluded"
    assert slots[1]["phase"] == "cache"
    assert slots[1]["slot_admit_scheduled"] == 1, "A second miss within window schedules A"
    assert slots[1]["slot_admit_applied"] == 0, "A is not inserted in the same slot"
    assert slots[2]["slot_admit_applied"] == 1, "A is inserted at next slot boundary"
    assert slots[2]["slot_cache_hits"] >= 1, "A should hit after delayed apply"

    late = run_sequence_for_smoke(
        ["B", "X", "Y", "Z", "B", "C"],
        capacity_objects=2,
        warmup_requests=0,
        slot_size=2,
        window_slots=1,
    )
    assert late.summary["scheduled_admissions_total"] == 0, (
        "B second miss after window_requests must not schedule"
    )

    duplicate = run_sequence_for_smoke(
        ["A", "A", "A", "A"],
        capacity_objects=1,
        warmup_requests=0,
        slot_size=4,
        window_slots=1,
    )
    assert duplicate.summary["scheduled_admissions_total"] == 1
    assert duplicate.summary["pending_duplicate_total"] == 2

    evict = run_sequence_for_smoke(
        ["A", "A", "B", "B", "C", "C", "A", "A"],
        capacity_objects=1,
        warmup_requests=0,
        slot_size=2,
        window_slots=1,
    )
    assert evict.summary["eviction_count"] >= 1

    repeat_a = run_sequence_for_smoke(
        ["W", "X", "A", "A", "A", "C", "D", "E"],
        capacity_objects=1,
        warmup_requests=2,
        slot_size=2,
        window_slots=1,
    )
    assert result.summary == repeat_a.summary
    print("Windowed-2Hit-LRU smoke test passed.")


def run_experiment(
    ds_cfg: DatasetConfig,
    capacity_percent: Optional[float],
    capacity_objects: Optional[int],
    all_sizes: bool,
    results_root: str,
    max_requests: Optional[int],
    warmup_requests_override: Optional[int],
    slot_size_override: Optional[int],
    window_slots: int,
    disable_progress: bool,
    seed: int,
) -> None:
    total_requests = int(max_requests if max_requests is not None else ds_cfg.num_total_requests)
    warmup_requests = int(
        warmup_requests_override
        if warmup_requests_override is not None
        else ds_cfg.num_warmup_requests
    )
    slot_size = int(slot_size_override if slot_size_override is not None else ds_cfg.slot_size)
    model_name = _model_name(window_slots)

    capacities: List[Tuple[int, Optional[float]]] = []
    if capacity_objects is not None:
        capacities = [(int(capacity_objects), capacity_percent)]
    elif capacity_percent is not None:
        cap = get_capacity_from_percent(ds_cfg.path, total_requests, capacity_percent)
        capacities = [(cap, float(capacity_percent))]
    else:
        percentages = list(CacheConfig().cache_size_percentages)
        num_unique = count_distinct_objects(ds_cfg.path, total_requests)
        capacities = [
            (max(1, int(num_unique * float(percent) / 100.0)), float(percent))
            for percent in percentages
        ]

    results: List[Dict[str, Any]] = []
    for cap, percent in capacities:
        run_id, dataset_dir = get_next_run_id(results_root, ds_cfg.name, model_name, cap)
        summary_path = os.path.join(dataset_dir, f"{run_id}_{model_name}_{cap}.json")
        slot_log_path = os.path.join(dataset_dir, f"{run_id}_{model_name}_{cap}_slot_log.jsonl")

        result = run_single_capacity(
            object_ids=_iter_trace_objects(ds_cfg.path, total_requests),
            trace_path=ds_cfg.path,
            total_requests=total_requests,
            warmup_requests=warmup_requests,
            slot_size=slot_size,
            capacity_objects=cap,
            window_slots=window_slots,
            dataset_name=ds_cfg.name,
            capacity_percent=percent,
            seed=seed,
            disable_progress=disable_progress,
        )
        payload = dict(result.summary)
        payload["slot_log_path"] = slot_log_path
        save_json(summary_path, payload)
        write_jsonl(slot_log_path, result.slot_records)
        results.append(payload)
        print(
            f"[{ds_cfg.name}] cap={cap} percent={percent} "
            f"HR={result.stats.hit_ratio:.6f} -> {summary_path}"
        )

    if len(results) > 1 or all_sizes:
        group_id, dataset_dir = get_next_group_id(results_root, ds_cfg.name, model_name)
        overall_path = os.path.join(dataset_dir, f"{group_id}_summary_{model_name}_all_sizes.json")
        save_json(
            overall_path,
            {
                "dataset": ds_cfg.name,
                "model_name": model_name,
                "baseline_name": BASELINE_NAME,
                "baseline_type": BASELINE_TYPE,
                "learner_baseline": False,
                "included_in_overhead_analysis": False,
                "window_slots": int(window_slots),
                "window_requests": int(window_slots * slot_size),
                "cache_sizes": [row["capacity_objects"] for row in results],
                "hr_curve": [
                    {
                        "capacity_objects": row["capacity_objects"],
                        "capacity_percent": row["capacity_percent"],
                        "hit_ratio": row["hit_ratio"],
                    }
                    for row in results
                ],
                "avg_hr": (
                    sum(float(row["hit_ratio"]) for row in results) / len(results)
                    if results
                    else None
                ),
                "runs": results,
            },
        )
        print(f"[SUMMARY] {overall_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Windowed-2Hit-LRU selective-admission baseline.")
    parser.add_argument(
        "--dataset",
        choices=["wikipedia_september_2007", "wikipedia_oktober_2007", "wiki2018"],
        default="wikipedia_september_2007",
    )
    parser.add_argument("--capacity-percent", type=float, default=None)
    parser.add_argument("--capacity-objects", type=int, default=None)
    parser.add_argument("--cache-size-objects", type=int, default=None)
    parser.add_argument("--all-sizes", action="store_true")
    parser.add_argument("--results-root", default="results/windowed_2hit_lru")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--warmup-requests", type=int, default=None)
    parser.add_argument("--slot-size", type=int, default=None)
    parser.add_argument("--window-slots", type=int, default=1)
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    capacity_objects = (
        args.capacity_objects
        if args.capacity_objects is not None
        else args.cache_size_objects
    )
    run_experiment(
        ds_cfg=resolve_dataset(args.dataset),
        capacity_percent=args.capacity_percent,
        capacity_objects=capacity_objects,
        all_sizes=bool(args.all_sizes or (capacity_objects is None and args.capacity_percent is None)),
        results_root=args.results_root,
        max_requests=args.max_requests,
        warmup_requests_override=args.warmup_requests,
        slot_size_override=args.slot_size,
        window_slots=args.window_slots,
        disable_progress=args.disable_progress,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
