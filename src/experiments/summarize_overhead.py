from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

if __package__ is None or __package__ == "":
    import sys

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from src.experiments.overhead_utils import (
    load_jsonl,
    percentile,
    safe_mean,
    summarize_overhead_slots,
    write_aggregate_csv,
    write_summary_json,
)


AGGREGATE_METRICS = (
    "avg_slot_control_s",
    "p95_slot_control_s",
    "p99_slot_control_s",
    "score_us_per_candidate",
    "avg_miss_candidates",
    "p95_miss_candidates",
    "peak_rss_mb",
    "rebuild_time_total_s",
    "rebuild_slots_count",
    "budget_time_total_s",
    "guard_signal_time_total_s",
)

COMPATIBILITY_KEYS = (
    "dataset",
    "model_name",
    "feature_set",
    "capacity_objects",
    "capacity_percent",
    "slot_size",
    "warmup_requests",
    "total_requests",
)


def summarize_one(slot_log_path: str, output_path: str | None = None) -> Dict[str, Any]:
    records = load_jsonl(slot_log_path)
    summary = summarize_overhead_slots(records, phase="cache")
    if records:
        first = records[0]
        summary.update(
            {
                "dataset": first.get("dataset"),
                "model_name": first.get("model_name"),
                "model": first.get("model_name"),
                "feature_set": first.get("feature_set"),
                "capacity_objects": first.get("capacity_objects"),
                "cache_size_objects": first.get("capacity_objects"),
                "capacity_percent": first.get("capacity_percent"),
                "slot_size": first.get("slot_size"),
                "warmup_requests": first.get("warmup_requests"),
                "total_requests": first.get("total_requests"),
            }
        )
    summary["slot_log_path"] = slot_log_path
    if output_path:
        write_summary_json(output_path, summary)
    return summary


def _summary_identity(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset": summary.get("dataset"),
        "model_name": summary.get("model_name") or summary.get("model"),
        "feature_set": summary.get("feature_set"),
        "capacity_objects": summary.get("capacity_objects")
        if summary.get("capacity_objects") is not None
        else summary.get("cache_size_objects"),
        "capacity_percent": summary.get("capacity_percent"),
        "slot_size": summary.get("slot_size"),
        "warmup_requests": summary.get("warmup_requests"),
        "total_requests": summary.get("total_requests"),
    }


def _iqr(values: Sequence[float]) -> Dict[str, float | None]:
    if not values:
        return {"median": None, "q1": None, "q3": None, "iqr": None}
    q1 = percentile(values, 25)
    med = percentile(values, 50)
    q3 = percentile(values, 75)
    return {
        "median": med,
        "q1": q1,
        "q3": q3,
        "iqr": None if q1 is None or q3 is None else float(q3 - q1),
    }


def _load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_compatible(summary_paths: Sequence[str], summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not summaries:
        raise ValueError("No summaries to aggregate.")
    reference = _summary_identity(summaries[0])
    for path, summary in zip(summary_paths[1:], summaries[1:]):
        current = _summary_identity(summary)
        mismatches = {
            key: (reference.get(key), current.get(key))
            for key in COMPATIBILITY_KEYS
            if reference.get(key) != current.get(key)
        }
        if mismatches:
            raise ValueError(
                f"Incompatible overhead summary for aggregation: {path}; "
                f"mismatches={mismatches}"
            )
    return reference


def aggregate_summaries(summary_paths: Sequence[str]) -> Dict[str, Any]:
    summaries = [_load_summary(path) for path in summary_paths]
    identity = _check_compatible(summary_paths, summaries)
    aggregate: Dict[str, Any] = {
        "runs": len(summaries),
        "summary_paths": list(summary_paths),
        "identity": identity,
        "metrics": {},
    }
    for metric in AGGREGATE_METRICS:
        vals = [
            float(summary[metric])
            for summary in summaries
            if isinstance(summary.get(metric), (int, float))
        ]
        aggregate["metrics"][metric] = {
            "mean": safe_mean(vals),
            **_iqr(vals),
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
            "runs": len(vals),
        }
    return aggregate


def flatten_aggregate(aggregate: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric, stats in aggregate.get("metrics", {}).items():
        row = {"metric": metric, "runs": aggregate.get("runs")}
        row.update(stats)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize standardized overhead slot logs.")
    parser.add_argument("--slot-log", default=None, help="Path to a single slot_log.jsonl.")
    parser.add_argument("--summary-out", default=None, help="Output JSON path for a single summary.")
    parser.add_argument("--aggregate-glob", default=None, help="Glob for overhead_summary.json files.")
    parser.add_argument("--aggregate-out-json", "--aggregate-json-out", dest="aggregate_json_out", default=None)
    parser.add_argument("--aggregate-out-csv", "--aggregate-csv-out", dest="aggregate_csv_out", default=None)
    args = parser.parse_args()

    if args.slot_log:
        out = args.summary_out
        if out is None:
            out = str(Path(args.slot_log).with_name("overhead_summary.json"))
        summary = summarize_one(args.slot_log, out)
        print(json.dumps(summary, indent=2))

    if args.aggregate_glob:
        paths = sorted(glob.glob(args.aggregate_glob))
        if not paths:
            raise SystemExit(f"No summary files match: {args.aggregate_glob}")
        aggregate = aggregate_summaries(paths)
        if args.aggregate_json_out:
            write_summary_json(args.aggregate_json_out, aggregate)
        if args.aggregate_csv_out:
            write_aggregate_csv(args.aggregate_csv_out, flatten_aggregate(aggregate))
        print(json.dumps(aggregate, indent=2))

    if not args.slot_log and not args.aggregate_glob:
        parser.error("Provide --slot-log or --aggregate-glob.")


if __name__ == "__main__":
    main()
