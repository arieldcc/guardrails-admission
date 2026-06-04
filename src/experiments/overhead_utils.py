from __future__ import annotations

import csv
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


TIMING_FIELDS = (
    "time_feat_s",
    "time_score_s",
    "time_budget_s",
    "time_guard_signal_s",
    "time_guard_s",
    "time_select_s",
    "time_update_s",
    "time_buffer_add_s",
    "time_history_update_s",
    "time_rebuild_s",
    "time_apply_pending_s",
    "time_slot_control_s",
    "time_boundary_total_s",
)


def ns_to_s(elapsed_ns: int) -> float:
    return float(elapsed_ns) / 1_000_000_000.0


def now_ns() -> int:
    return time.perf_counter_ns()


def elapsed_s(start_ns: int) -> float:
    return ns_to_s(time.perf_counter_ns() - start_ns)


def safe_sum(values: Iterable[Optional[float]]) -> float:
    return float(sum(float(v) for v in values if v is not None))


def safe_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def percentile(values: Sequence[Optional[float]], q: float) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    q = min(max(float(q), 0.0), 100.0)
    pos = (len(vals) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def get_current_rss_mb() -> float:
    try:
        import psutil  # type: ignore

        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss / (1024.0**2))
    except Exception:
        return get_peak_rss_mb()


def get_peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system().lower() == "darwin":
        return float(usage / (1024.0**2))
    return float(usage / 1024.0)


def load_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | os.PathLike[str], records: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary_json(path: str | os.PathLike[str], summary: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _require_fields(record: Dict[str, Any], fields: Iterable[str]) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(
            f"slot_index={record.get('slot_index')} missing overhead fields: {missing}"
        )


def validate_slot_record(record: Dict[str, Any]) -> None:
    _require_fields(
        record,
        (
            "slot_index",
            "phase",
            "miss_candidates",
            "miss_requests",
            "time_slot_control_s",
            "time_boundary_total_s",
            *TIMING_FIELDS,
        ),
    )
    phase = record["phase"]
    if phase not in ("warmup", "cache"):
        raise ValueError(f"Invalid phase={phase!r} at slot={record.get('slot_index')}")
    for field in TIMING_FIELDS:
        value = float(record.get(field, 0.0))
        if value < -1e-12:
            raise ValueError(f"Negative timing {field}={value} at slot={record.get('slot_index')}")

    guard_signal = float(record.get("time_guard_signal_s", 0.0))
    guard_alias = float(record.get("time_guard_s", 0.0))
    if abs(guard_signal - guard_alias) > 1e-6:
        raise ValueError(
            "time_guard_s/time_guard_signal_s mismatch at slot="
            f"{record.get('slot_index')}: guard={guard_alias}, signal={guard_signal}"
        )

    expected_control = (
        float(record.get("time_feat_s", 0.0))
        + float(record.get("time_score_s", 0.0))
        + float(record.get("time_budget_s", 0.0))
        + guard_signal
        + float(record.get("time_select_s", 0.0))
        + float(record.get("time_update_s", 0.0))
    )
    if abs(float(record["time_slot_control_s"]) - expected_control) > 1e-6:
        raise ValueError(
            "time_slot_control_s mismatch at slot="
            f"{record.get('slot_index')}: got {record['time_slot_control_s']}, "
            f"expected {expected_control}"
        )
    expected_boundary = expected_control + float(record.get("time_apply_pending_s", 0.0))
    if abs(float(record["time_boundary_total_s"]) - expected_boundary) > 1e-6:
        raise ValueError(
            "time_boundary_total_s mismatch at slot="
            f"{record.get('slot_index')}: got {record['time_boundary_total_s']}, "
            f"expected {expected_boundary}"
        )
    if not isinstance(record.get("rebuild_triggered", False), bool):
        raise ValueError(f"rebuild_triggered must be bool at slot={record.get('slot_index')}")
    rebuild_phase = record.get("rebuild_phase")
    if rebuild_phase is not None and not isinstance(rebuild_phase, str):
        raise ValueError(f"rebuild_phase must be string or None at slot={record.get('slot_index')}")

    model_name = str(record.get("model_name", "")).lower()
    is_gbdt = "gbdt" in model_name or "gdbt" in model_name
    is_no_guard = "no_guard" in model_name or "noguard" in model_name
    if (is_gbdt or is_no_guard) and abs(guard_signal) > 1e-9:
        raise ValueError(
            f"{record.get('model_name')} must not report Guardrails signal time "
            f"at slot={record.get('slot_index')}"
        )
    if is_gbdt and abs(float(record.get("time_budget_s", 0.0))) > 1e-9:
        raise ValueError(
            f"{record.get('model_name')} must not report Guardrails/budget time "
            f"at slot={record.get('slot_index')}"
        )


def summarize_overhead_slots(
    slot_records: Sequence[Dict[str, Any]],
    phase: str = "cache",
) -> Dict[str, Any]:
    for record in slot_records:
        validate_slot_record(record)

    selected = [record for record in slot_records if record.get("phase") == phase]
    warmup = [record for record in slot_records if record.get("phase") == "warmup"]

    control = [float(record["time_slot_control_s"]) for record in selected]
    boundary = [float(record["time_boundary_total_s"]) for record in selected]
    miss_candidates = [float(record.get("miss_candidates", 0)) for record in selected]
    miss_requests = [float(record.get("miss_requests", 0)) for record in selected]
    rebuild_slots = [record for record in selected if bool(record.get("rebuild_triggered", False))]
    non_rebuild = [record for record in selected if not bool(record.get("rebuild_triggered", False))]
    non_rebuild_control = [float(record["time_slot_control_s"]) for record in non_rebuild]
    score_total = safe_sum(record.get("time_score_s", 0.0) for record in selected)
    guard_signal_total = safe_sum(
        record.get("time_guard_signal_s", record.get("time_guard_s", 0.0))
        for record in selected
    )
    miss_candidates_total = safe_sum(record.get("miss_candidates", 0.0) for record in selected)

    expected_cache_slots = None
    if slot_records:
        first = slot_records[0]
        total_requests = first.get("total_requests")
        warmup_requests = first.get("warmup_requests")
        slot_size = first.get("slot_size")
        if total_requests is not None and warmup_requests is not None and slot_size:
            expected_cache_slots = int((int(total_requests) - int(warmup_requests)) // int(slot_size))

    summary = {
        "cache_slots_processed": int(len(selected)),
        "warmup_slots_processed": int(len(warmup)),
        "expected_cache_slots": expected_cache_slots,
        "avg_slot_control_s": safe_mean(control),
        "p50_slot_control_s": percentile(control, 50),
        "p95_slot_control_s": percentile(control, 95),
        "p99_slot_control_s": percentile(control, 99),
        "avg_boundary_total_s": safe_mean(boundary),
        "p95_boundary_total_s": percentile(boundary, 95),
        "p99_boundary_total_s": percentile(boundary, 99),
        "feat_time_total_s": safe_sum(record.get("time_feat_s", 0.0) for record in selected),
        "score_time_total_s": score_total,
        "budget_time_total_s": safe_sum(record.get("time_budget_s", 0.0) for record in selected),
        "guard_signal_time_total_s": guard_signal_total,
        "guard_time_total_s": guard_signal_total,
        "select_time_total_s": safe_sum(record.get("time_select_s", 0.0) for record in selected),
        "update_time_total_s": safe_sum(record.get("time_update_s", 0.0) for record in selected),
        "rebuild_time_total_s": safe_sum(record.get("time_rebuild_s", 0.0) for record in selected),
        "buffer_add_time_total_s": safe_sum(record.get("time_buffer_add_s", 0.0) for record in selected),
        "history_update_time_total_s": safe_sum(
            record.get("time_history_update_s", 0.0) for record in selected
        ),
        "apply_pending_time_total_s": safe_sum(
            record.get("time_apply_pending_s", 0.0) for record in selected
        ),
        "score_us_per_candidate": (
            float(1_000_000.0 * score_total / miss_candidates_total)
            if miss_candidates_total > 0
            else None
        ),
        "avg_miss_candidates": safe_mean(miss_candidates),
        "p95_miss_candidates": percentile(miss_candidates, 95),
        "p99_miss_candidates": percentile(miss_candidates, 99),
        "max_miss_candidates": int(max(miss_candidates)) if miss_candidates else None,
        "avg_miss_requests": safe_mean(miss_requests),
        "p95_miss_requests": percentile(miss_requests, 95),
        "p99_miss_requests": percentile(miss_requests, 99),
        "peak_rss_mb": max(
            [float(record.get("peak_rss_mb", 0.0)) for record in slot_records] + [get_peak_rss_mb()]
        ),
        "max_feature_bytes": int(max([int(record.get("feature_bytes", 0)) for record in selected] or [0])),
        "rebuild_slots_count": int(len(rebuild_slots)),
        "rebuild_slots_indices": [int(record["slot_index"]) for record in rebuild_slots],
        "avg_slot_control_s_excluding_rebuild_slots": safe_mean(non_rebuild_control),
        "p95_slot_control_s_excluding_rebuild_slots": percentile(non_rebuild_control, 95),
        "p99_slot_control_s_excluding_rebuild_slots": percentile(non_rebuild_control, 99),
    }
    if (
        expected_cache_slots is not None
        and phase == "cache"
        and summary["cache_slots_processed"] != expected_cache_slots
    ):
        summary["cache_slot_count_warning"] = (
            f"cache_slots_processed={summary['cache_slots_processed']} "
            f"expected={expected_cache_slots}"
        )
    return summary


def prepare_run_dir(
    results_root: str,
    dataset: str,
    model_name: str,
    capacity_objects: int,
    run_id: Optional[str] = None,
) -> tuple[str, str]:
    dataset_dir = Path(results_root) / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        existing: List[int] = []
        suffix = f"_{model_name}_{capacity_objects}"
        for child in dataset_dir.iterdir():
            name = child.name
            if suffix not in name:
                continue
            prefix = name.split("_", 1)[0]
            if len(prefix) == 3 and prefix.isdigit():
                existing.append(int(prefix))
        run_id = f"{(max(existing) + 1) if existing else 1:03d}"
    run_dir = dataset_dir / f"{run_id}_{model_name}_{capacity_objects}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, str(run_dir)


def get_git_commit(project_root: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def build_run_metadata(
    args: Dict[str, Any],
    start_time: str,
    end_time: str,
    project_root: str,
) -> Dict[str, Any]:
    env_keys = (
        "PYTHONHASHSEED",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "environment": {key: os.environ.get(key) for key in env_keys},
        "start_time": start_time,
        "end_time": end_time,
        "git_commit": get_git_commit(project_root),
        "argv": sys.argv,
        "args": args,
    }


def write_error_json(path: str | os.PathLike[str], config: Dict[str, Any], exc: BaseException) -> None:
    write_summary_json(
        path,
        {
            "status": "failed",
            "config": config,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        },
    )


def write_aggregate_csv(path: str | os.PathLike[str], rows: Sequence[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
