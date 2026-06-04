# src/experiments/run_baseline_cache.py

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List, Tuple, Dict, Optional
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.experiment_config import (
    DatasetConfig,
    WIKI2018,
    WIKIPEDIA_OKTOBER_2007,
    WIKIPEDIA_SEPTEMBER_2007,
    CacheConfig,
)
from src.data.trace_reader import TraceReader
from src.cache.cache_simulator import CacheStats
from src.cache.lru import LRUCache
from src.cache.lfuda import LFUDACache
from src.cache.lru2 import LRU2Cache
from src.cache.tinylfu import TinyLFUCache


BASELINE_POLICIES: Tuple[str, ...] = ("LRU", "LRU2", "LFUDA", "TINYLFU")
BASELINE_DATASETS: Tuple[str, ...] = (
    "wikipedia_september_2007",
    "wikipedia_oktober_2007",
    "wiki2018",
)

POLICY_MODEL_NAMES: Dict[str, str] = {
    "LRU": "baseline_lru_delayed",
    "LRU2": "baseline_lru2_delayed",
    "LRU-K": "baseline_lru2_delayed",
    "LRU-K2": "baseline_lru2_delayed",
    "LFUDA": "baseline_lfuda_delayed",
    "TINYLFU": "baseline_tinylfu_delayed",
}


def get_next_run_id(results_root: str, dataset: str, model: str, cache_size: int) -> Tuple[str, str]:
    dataset_dir = os.path.join(results_root, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    existing_ids: List[int] = []
    for fname in os.listdir(dataset_dir):
        if not fname.endswith(f"_{model}_{cache_size}.json") and \
           not fname.endswith(f"_summary_{model}_{cache_size}.json"):
            continue
        prefix = fname.split("_", 1)[0]
        if len(prefix) == 3 and prefix.isdigit():
            existing_ids.append(int(prefix))

    next_id = 1 if not existing_ids else max(existing_ids) + 1
    run_id = f"{next_id:03d}"
    return run_id, dataset_dir


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


def get_next_master_id(results_root: str, label: str) -> str:
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
    return f"{next_id:03d}"


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_cache(policy: str, capacity_objects: int):
    policy = policy.upper()
    if policy == "LRU":
        return LRUCache(capacity_objects)
    elif policy in ("LRU2", "LRU-K", "LRU-K2"):
        return LRU2Cache(capacity_objects)
    elif policy == "LFUDA":
        return LFUDACache(capacity_objects)
    elif policy == "TINYLFU":
        return TinyLFUCache(capacity_objects)
    else:
        raise ValueError(f"Unknown policy: {policy}")


def resolve_dataset(name: str) -> DatasetConfig:
    if name == "wikipedia_september_2007":
        return WIKIPEDIA_SEPTEMBER_2007
    if name == "wikipedia_oktober_2007":
        return WIKIPEDIA_OKTOBER_2007
    if name == "wiki2018":
        return WIKI2018
    raise ValueError(f"Unknown dataset: {name}")


def resolve_datasets(dataset_arg: str) -> List[DatasetConfig]:
    if dataset_arg == "all":
        return [resolve_dataset(name) for name in BASELINE_DATASETS]
    return [resolve_dataset(dataset_arg)]


def resolve_policies(policy_arg: str) -> List[str]:
    if policy_arg.lower() == "all":
        policies = list(BASELINE_POLICIES)
        if len(set(policies)) != len(policies):
            raise ValueError("Duplicate policy found in BASELINE_POLICIES.")
        return policies
    return [policy_arg.upper()]


def model_name_for_policy(policy: str) -> str:
    policy = policy.upper()
    if policy not in POLICY_MODEL_NAMES:
        raise ValueError(f"Unknown policy: {policy}")
    return POLICY_MODEL_NAMES[policy]


def validate_runtime_config(
    total_requests: int,
    warmup_requests: int,
    slot_size: int,
) -> None:
    if slot_size <= 0:
        raise ValueError("slot_size must be > 0.")
    if total_requests <= warmup_requests:
        raise ValueError("total_requests must be greater than warmup_requests.")
    if warmup_requests % slot_size != 0:
        raise ValueError(
            f"warmup_requests ({warmup_requests}) harus kelipatan slot_size ({slot_size})."
        )


def run_single_policy(
    ds_cfg: DatasetConfig,
    policy: str,
    capacity_objects: int,
    total_requests: Optional[int] = None,
    warmup_requests: Optional[int] = None,
    slot_size: Optional[int] = None,
    disable_progress: bool = False,
) -> CacheStats:
    """
    Baseline cache (LRU / LRU2 / LFUDA / TinyLFU) pada raw Wikipedia trace.

    - Dataset & parameter eksperimen diambil dari experiment_config:
        * ds_cfg.path            -> file .gz asli
        * ds_cfg.num_total_requests
        * ds_cfg.num_warmup_requests
    """
    total_requests = ds_cfg.num_total_requests if total_requests is None else total_requests
    warmup_requests = ds_cfg.num_warmup_requests if warmup_requests is None else warmup_requests
    slot_size = ds_cfg.slot_size if slot_size is None else slot_size

    validate_runtime_config(total_requests, warmup_requests, slot_size)
    if capacity_objects < 1:
        raise ValueError("capacity_objects must be >= 1.")

    # TraceReader generik: path bisa .gz atau direktori parquet
    reader = TraceReader(path=ds_cfg.path, max_rows=total_requests)
    req_iter = reader.iter_requests()

    cache = make_cache(policy, capacity_objects)
    stats = CacheStats(capacity_objects=capacity_objects)

    global_idx = 0
    slot_req_count = 0
    slot_index = 0
    slot_miss_order: List[str] = []
    slot_miss_seen: set[str] = set()
    pending_admit_order: List[str] = []
    pending_admit_source_slot: Optional[int] = None

    # Hanya request setelah warm-up yang dihitung ke hit ratio
    eval_requests_planned = max(total_requests - warmup_requests, 0)
    pbar = tqdm(
        total=eval_requests_planned,
        desc=f"Eval {policy} (cap={capacity_objects})",
        unit="req",
        disable=disable_progress,
    )

    def _apply_pending_admits(next_slot_num: int) -> None:
        nonlocal pending_admit_order
        nonlocal pending_admit_source_slot
        if not pending_admit_order:
            pending_admit_source_slot = None
            return
        for obj_id in pending_admit_order:
            cache.insert(obj_id)
        pending_admit_order = []
        pending_admit_source_slot = None

    def _finalize_slot() -> None:
        nonlocal slot_index
        nonlocal slot_miss_order
        nonlocal slot_miss_seen
        nonlocal pending_admit_order
        nonlocal pending_admit_source_slot
        slot_index += 1
        slot_num = slot_index
        if slot_num > (warmup_requests // slot_size):
            pending_admit_order = list(slot_miss_order)
            pending_admit_source_slot = slot_num if pending_admit_order else None
        else:
            pending_admit_order = []
            pending_admit_source_slot = None
        slot_miss_order = []
        slot_miss_seen = set()

    for req in req_iter:
        if global_idx >= total_requests:
            break

        obj_id = req["object_id"]

        if global_idx >= warmup_requests:
            stats.total_requests += 1
            if cache.access(obj_id):
                stats.cache_hits += 1
            else:
                # one-slot delayed admission:
                # - miss unik dalam slot t dikumpulkan sebagai kandidat
                # - insert baru di-apply pada awal slot t+1
                if obj_id not in slot_miss_seen:
                    slot_miss_seen.add(obj_id)
                    slot_miss_order.append(obj_id)
            # progress bar hanya untuk fase eval
            if stats.total_requests <= eval_requests_planned:
                pbar.update(1)

        slot_req_count += 1
        if slot_req_count >= slot_size:
            _finalize_slot()
            next_slot_start = slot_index * slot_size
            if next_slot_start < total_requests and next_slot_start >= warmup_requests:
                _apply_pending_admits(slot_index + 1)
            else:
                pending_admit_order = []
                pending_admit_source_slot = None
            slot_req_count = 0

        global_idx += 1

    if slot_req_count > 0:
        _finalize_slot()

    pbar.close()
    return stats
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

# hitung jumlah objek unik
def get_dynamic_capacities(trace_path: str, total_requests: int, percentages: List[float]) -> List[int]:
    """
    Hitung kapasitas cache dalam jumlah objek berdasarkan persentase.
    """
    n_unique = count_distinct_objects(trace_path, total_requests)
    return [max(1, int(n_unique * p / 100.0)) for p in percentages]


def compute_capacities(
    trace_path: str,
    total_requests: int,
    capacity_percent: Optional[float] = None,
    capacity_objects: Optional[int] = None,
    cache_size_percentages: Optional[List[float]] = None,
    num_unique_objects: Optional[int] = None,
) -> Tuple[List[Dict[str, Optional[float]]], Optional[int]]:
    if capacity_objects is not None:
        if capacity_objects < 1:
            raise ValueError("capacity_objects must be >= 1.")
        return [{"capacity_objects": capacity_objects, "capacity_percent": None}], num_unique_objects

    if capacity_percent is not None:
        if capacity_percent <= 0:
            raise ValueError("capacity_percent must be > 0.")
        if num_unique_objects is None:
            num_unique_objects = count_distinct_objects(trace_path, total_requests)
        capacity = max(1, int(num_unique_objects * capacity_percent / 100.0))
        return [{"capacity_objects": capacity, "capacity_percent": float(capacity_percent)}], num_unique_objects

    percentages = list(cache_size_percentages or CacheConfig().cache_size_percentages)
    if not percentages:
        raise ValueError("cache_size_percentages must not be empty.")
    if any(p <= 0 for p in percentages):
        raise ValueError("All cache size percentages must be > 0.")

    if num_unique_objects is None:
        num_unique_objects = count_distinct_objects(trace_path, total_requests)

    capacities: List[Dict[str, Optional[float]]] = []
    seen_capacities: set[int] = set()
    for percent in sorted(float(p) for p in percentages):
        capacity = max(1, int(num_unique_objects * percent / 100.0))
        if capacity in seen_capacities:
            raise ValueError(
                f"Duplicate capacity_objects={capacity} generated from cache-size percentages."
            )
        seen_capacities.add(capacity)
        capacities.append({"capacity_objects": capacity, "capacity_percent": percent})

    return capacities, num_unique_objects


def _best_result(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(results, key=lambda item: item["hit_ratio"])


def _worst_result(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return min(results, key=lambda item: item["hit_ratio"])


def run_experiment(
    datasets: List[DatasetConfig],
    policies: List[str],
    results_root: str,
    capacity_percent: Optional[float] = None,
    capacity_objects: Optional[int] = None,
    cache_size_percentages: Optional[List[float]] = None,
    max_requests: Optional[int] = None,
    warmup_requests_override: Optional[int] = None,
    slot_size_override: Optional[int] = None,
    disable_progress: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    cache_cfg = CacheConfig()
    percentages = list(cache_size_percentages or cache_cfg.cache_size_percentages)
    master_entries: List[Dict[str, Any]] = []
    datasets_executed: List[str] = []

    for ds_cfg in datasets:
        total_requests = ds_cfg.num_total_requests
        if max_requests is not None:
            total_requests = min(max_requests, ds_cfg.num_total_requests)
        warmup_requests = (
            ds_cfg.num_warmup_requests
            if warmup_requests_override is None
            else warmup_requests_override
        )
        slot_size = ds_cfg.slot_size if slot_size_override is None else slot_size_override
        validate_runtime_config(total_requests, warmup_requests, slot_size)

        dataset_name = ds_cfg.name
        datasets_executed.append(dataset_name)

        num_unique_objects: Optional[int] = None
        if capacity_objects is None:
            print(f"[SCAN] Dataset={dataset_name}: counting distinct objects once")
            num_unique_objects = count_distinct_objects(ds_cfg.path, total_requests)

        capacities, num_unique_objects = compute_capacities(
            trace_path=ds_cfg.path,
            total_requests=total_requests,
            capacity_percent=capacity_percent,
            capacity_objects=capacity_objects,
            cache_size_percentages=percentages,
            num_unique_objects=num_unique_objects,
        )

        print(f"=== Baseline Cache Experiment ({dataset_name}) ===")
        print(f"Trace path       : {ds_cfg.path}")
        print(f"Policies         : {policies}")
        print(f"Protocol         : one-slot delayed admission")
        print(f"Total requests   : {total_requests}")
        print(f"Warm-up requests : {warmup_requests}")
        print(f"Slot size        : {slot_size}")
        print(f"Unique objects   : {num_unique_objects}")
        print(f"Capacities       : {[item['capacity_objects'] for item in capacities]}")
        print()

        for policy in policies:
            model_name = model_name_for_policy(policy)
            capacity_results: List[Dict[str, Any]] = []

            for cap_info in capacities:
                capacity = int(cap_info["capacity_objects"])
                cap_percent = cap_info["capacity_percent"]
                print(f"[RUN] Dataset={dataset_name}, Policy={policy}, Capacity={capacity} objects")

                stats = run_single_policy(
                    ds_cfg=ds_cfg,
                    policy=policy,
                    capacity_objects=capacity,
                    total_requests=total_requests,
                    warmup_requests=warmup_requests,
                    slot_size=slot_size,
                    disable_progress=disable_progress,
                )

                run_id, dataset_dir = get_next_run_id(
                    results_root, dataset_name, model_name, capacity
                )
                summary_path = os.path.join(
                    dataset_dir,
                    f"{run_id}_summary_{model_name}_{capacity}.json",
                )
                payload = {
                    "dataset": dataset_name,
                    "policy": policy,
                    "model_name": model_name,
                    "baseline_type": "standard_cache_baseline",
                    "learner_baseline": False,
                    "protocol": "one_slot_delayed_admission",
                    "cache_size_objects": capacity,
                    "capacity_percent": cap_percent,
                    "total_requests_config": total_requests,
                    "warmup_requests_config": warmup_requests,
                    "slot_size": slot_size,
                    "cache_hits": stats.cache_hits,
                    "cache_requests": stats.total_requests,
                    "hit_ratio": stats.hit_ratio,
                    "results_root": results_root,
                    "run_id": run_id,
                    "reproducibility_seed": seed,
                }
                save_json(summary_path, payload)
                print(
                    f"      HR={stats.hit_ratio:.6f} "
                    f"({stats.cache_hits}/{stats.total_requests})"
                )
                print(f"      [LOG] summary -> {summary_path}")

                capacity_results.append(
                    {
                        "capacity_percent": cap_percent,
                        "cache_size_objects": capacity,
                        "cache_hits": stats.cache_hits,
                        "cache_requests": stats.total_requests,
                        "hit_ratio": stats.hit_ratio,
                        "summary_path": summary_path,
                    }
                )

            group_run_id, dataset_dir = get_next_group_id(results_root, dataset_name, model_name)
            all_sizes_path = os.path.join(
                dataset_dir,
                f"{group_run_id}_summary_{model_name}_all_sizes.json",
            )
            hr_curve = [
                {
                    "capacity_percent": item["capacity_percent"],
                    "cache_size_objects": item["cache_size_objects"],
                    "hit_ratio": item["hit_ratio"],
                }
                for item in capacity_results
            ]
            avg_hr = (
                sum(item["hit_ratio"] for item in capacity_results) / len(capacity_results)
                if capacity_results
                else 0.0
            )
            best = _best_result(capacity_results)
            worst = _worst_result(capacity_results)
            all_sizes_payload = {
                "dataset": dataset_name,
                "policy": policy,
                "model_name": model_name,
                "baseline_type": "standard_cache_baseline",
                "learner_baseline": False,
                "protocol": "one_slot_delayed_admission",
                "trace_path": ds_cfg.path,
                "total_requests_config": total_requests,
                "warmup_requests_config": warmup_requests,
                "slot_size": slot_size,
                "num_unique_objects": num_unique_objects,
                "cache_size_percentages": [item["capacity_percent"] for item in capacity_results],
                "cache_sizes_objects": [item["cache_size_objects"] for item in capacity_results],
                "results": capacity_results,
                "hr_curve": hr_curve,
                "avg_hr": avg_hr,
                "best": best,
                "worst": worst,
                "results_root": results_root,
                "run_id": group_run_id,
                "reproducibility_seed": seed,
            }
            save_json(all_sizes_path, all_sizes_payload)
            print(f"      [LOG] all-size summary -> {all_sizes_path}")
            print()

            master_entries.append(
                {
                    "dataset": dataset_name,
                    "policy": policy,
                    "model_name": model_name,
                    "all_sizes_summary_path": all_sizes_path,
                    "avg_hr": avg_hr,
                    "best": best,
                    "worst": worst,
                }
            )

    policies_executed = list(dict.fromkeys(policies))
    master_summary: Dict[str, Any] = {
        "runner": "run_baseline_cache.py",
        "baseline_type": "standard_cache_baseline",
        "learner_baseline": False,
        "protocol": "one_slot_delayed_admission",
        "datasets_executed": list(dict.fromkeys(datasets_executed)),
        "policies_executed": policies_executed,
        "total_runs": len(master_entries),
        "results": master_entries,
    }

    if len(master_entries) > 1:
        label = "baseline_all_datasets_all_policies"
        master_run_id = get_next_master_id(results_root, label)
        aggregate_dir = os.path.join(results_root, "_aggregate")
        master_path = os.path.join(aggregate_dir, f"{master_run_id}_summary_{label}.json")
        master_summary["master_summary_path"] = master_path
        master_summary["run_id"] = master_run_id
        save_json(master_path, master_summary)
        print(f"[LOG] master summary -> {master_path}")

    return master_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run standard baseline cache policies with one-slot delayed admission."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikipedia_september_2007",
        choices=[
            "wikipedia_september_2007",
            "wikipedia_oktober_2007",
            "wiki2018",
            "all",
        ],
        help="Dataset trace to run, or 'all'.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="all",
        choices=["LRU", "LRU2", "LRU-K", "LRU-K2", "LFUDA", "TINYLFU", "all"],
        help="Cache replacement policy, or 'all' for canonical standard baselines.",
    )
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--capacity-percent", type=float, default=None)
    parser.add_argument("--cache-size-objects", type=int, default=None)
    parser.add_argument("--capacity-objects", type=int, default=None)
    parser.add_argument(
        "--all-sizes",
        action="store_true",
        help="Run all percentages from CacheConfig.cache_size_percentages.",
    )
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--warmup-requests", type=int, default=None)
    parser.add_argument("--slot-size", type=int, default=None)
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    exact_capacity = args.capacity_objects
    if exact_capacity is None:
        exact_capacity = args.cache_size_objects
    elif args.cache_size_objects is not None and args.cache_size_objects != exact_capacity:
        raise ValueError("--capacity-objects and --cache-size-objects must match when both are set.")
    if exact_capacity is not None and exact_capacity < 1:
        raise ValueError("--capacity-objects/--cache-size-objects must be >= 1.")
    if args.capacity_percent is not None and args.capacity_percent <= 0:
        raise ValueError("--capacity-percent must be > 0.")
    if args.max_requests is not None and args.max_requests < 1:
        raise ValueError("--max-requests must be >= 1.")

    datasets = resolve_datasets(args.dataset)
    policies = resolve_policies(args.policy)

    run_experiment(
        datasets=datasets,
        policies=policies,
        results_root=args.results_root,
        capacity_percent=args.capacity_percent,
        capacity_objects=exact_capacity,
        cache_size_percentages=list(CacheConfig().cache_size_percentages),
        max_requests=args.max_requests,
        warmup_requests_override=args.warmup_requests,
        slot_size_override=args.slot_size,
        disable_progress=args.disable_progress,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
