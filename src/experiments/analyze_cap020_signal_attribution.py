from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DATASETS = ("wikipedia_september_2007", "wiki2018")
DEFAULT_CACHE_PERCENTAGES = (0.8, 1.0, 2.0, 3.0, 4.0, 5.0)
DEFAULT_SOURCE_RESULTS_ROOT = "results/guardrails_signal_v2_cap020"
DEFAULT_OUTPUT_ROOT = "results/guardrails_signal_v2_cap020_budget_controls"

CONTROL_VARIANTS = (
    "full_cap020",
    "precision_only_cap020",
    "precision_only_budget_matched_uniform_cap020",
    "precision_only_budget_matched_random_cap020",
    "precision_only_budget_replay_cap020",
    "precision_only_budget_replay_permuted_cap020",
    "precision_only_budget_matched_permuted_quality_cap020",
)
SUPPRESSION_VARIANTS = CONTROL_VARIANTS
EFFECTIVE_PRECISION_VARIANTS = (
    "cap_only_cap020",
    "precision_only_cap020",
    "full_cap020",
)
PAIRED_COMPARISONS = (
    ("full_cap020", "precision_only_cap020"),
    ("full_cap020", "precision_only_budget_matched_uniform_cap020"),
    ("full_cap020", "precision_only_budget_matched_random_cap020"),
    ("full_cap020", "precision_only_budget_replay_cap020"),
    ("full_cap020", "precision_only_budget_replay_permuted_cap020"),
    ("precision_only_cap020", "cap_only_cap020"),
)

SELECTED_KEYS = (
    "total_selected_admissions",
    "selected_admissions_total",
    "admit_selected_total",
    "postfill_selected_admissions",
)
HIT_YIELD_KEYS = (
    "aggregate_hit_yield_per_applied_admission",
    "hit_yield_total",
    "postfill_hit_yield_per_applied_admission",
)
POLLUTION_KEYS = (
    "aggregate_pollution_ratio",
    "pollution_rate_total",
    "postfill_pollution_ratio",
)
ADMISSION_PRECISION_KEYS = (
    "aggregate_admission_precision",
    "admission_precision_total",
    "postfill_admission_precision",
)


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_div(num: Any, den: Any) -> Optional[float]:
    num_f = _safe_float(num)
    den_f = _safe_float(den)
    if num_f is None or den_f is None or den_f == 0.0:
        return None
    return float(num_f / den_f)


def _mean(values: Iterable[Any]) -> Optional[float]:
    clean = [_safe_float(value) for value in values]
    nums = [value for value in clean if value is not None]
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def _median(values: Iterable[Any]) -> Optional[float]:
    clean = [_safe_float(value) for value in values]
    nums = [value for value in clean if value is not None]
    if not nums:
        return None
    return float(median(nums))


def _rate(values: Iterable[Any]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return float(sum(1 for value in vals if bool(value)) / len(vals))


def _nested_get(mapping: Dict[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping.get(key)
    config = mapping.get("config_summary")
    if isinstance(config, dict) and key in config:
        return config.get(key)
    diagnostics = mapping.get("diagnostic_summary")
    if isinstance(diagnostics, dict) and key in diagnostics:
        return diagnostics.get(key)
    return None


def _first_present(mapping: Dict[str, Any], candidates: Sequence[str]) -> Tuple[Any, Optional[str]]:
    for key in candidates:
        value = _nested_get(mapping, key)
        if value is not None:
            return value, key
    return None, None


def _required_number(mapping: Dict[str, Any], candidates: Sequence[str], context: str) -> float:
    value, key = _first_present(mapping, candidates)
    out = _safe_float(value)
    if out is None:
        available = ", ".join(sorted(str(k) for k in mapping.keys()))
        expected = ", ".join(candidates)
        raise KeyError(
            f"{context}: missing required numeric field. Expected one of "
            f"[{expected}]. Available top-level keys: [{available}]"
        )
    return out


def _optional_number(mapping: Dict[str, Any], candidates: Sequence[str]) -> Optional[float]:
    value, _ = _first_present(mapping, candidates)
    return _safe_float(value)


def _optional_bool(mapping: Dict[str, Any], candidates: Sequence[str]) -> Optional[bool]:
    value, _ = _first_present(mapping, candidates)
    if value is None:
        return None
    return bool(value)


def _run_id_from_path(path: str) -> int:
    base = os.path.basename(path)
    prefix = base.split("_", 1)[0]
    return int(prefix) if len(prefix) == 3 and prefix.isdigit() else -1


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object.")
    data["_summary_path"] = path
    return data


def _same_percent(a: Any, b: Any) -> bool:
    a_f = _safe_float(a)
    b_f = _safe_float(b)
    return a_f is not None and b_f is not None and abs(a_f - b_f) < 1e-9


def find_summary(
    roots: Sequence[str],
    dataset: str,
    variant: str,
    capacity_percent: float,
    feature_set: str,
    base_learner: str,
) -> Optional[Dict[str, Any]]:
    matches: List[Tuple[int, float, str, Dict[str, Any]]] = []
    for root in roots:
        dataset_dir = os.path.join(root, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        for fname in os.listdir(dataset_dir):
            if not fname.endswith(".json") or "_summary_" not in fname or fname.endswith("_all_sizes.json"):
                continue
            path = os.path.join(dataset_dir, fname)
            try:
                data = _load_json(path)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if data.get("dataset") != dataset:
                continue
            if data.get("variant") != variant:
                continue
            if data.get("feature_set") != feature_set:
                continue
            if str(data.get("base_learner")).lower() != str(base_learner).lower():
                continue
            if not _same_percent(data.get("capacity_percent"), capacity_percent):
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = 0.0
            matches.append((_run_id_from_path(path), mtime, path, data))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[1], item[0], item[2]))
    return matches[-1][3]


def load_slot_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = summary.get("slot_log_path")
    if not path:
        return []
    if not os.path.isabs(path):
        path = os.path.normpath(path)
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    except (OSError, json.JSONDecodeError):
        return []
    return rows


def _cache_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("phase") == "cache"]


def _slot_value(row: Dict[str, Any], candidates: Sequence[str]) -> Optional[float]:
    for key in candidates:
        if key in row and row[key] is not None:
            return _safe_float(row[key])
    return None


def _summary_metric(summary: Dict[str, Any]) -> Dict[str, Optional[float]]:
    total_requests = _required_number(summary, ("total_requests", "cache_requests"), summary["_summary_path"])
    selected = _required_number(summary, SELECTED_KEYS, summary["_summary_path"])
    return {
        "hit_ratio": _required_number(summary, ("hit_ratio", "postfill_hit_ratio"), summary["_summary_path"]),
        "total_selected_admissions": selected,
        "admissions_per_1000_requests": _optional_number(
            summary,
            ("admissions_per_1000_requests", "postfill_admissions_per_1000_requests"),
        )
        or _safe_div(selected * 1000.0, total_requests),
        "hit_yield_per_admission": _optional_number(summary, HIT_YIELD_KEYS),
        "pollution_ratio": _optional_number(summary, POLLUTION_KEYS),
        "admission_precision": _optional_number(summary, ADMISSION_PRECISION_KEYS),
    }


def _diagnostic(summary: Dict[str, Any], key: str) -> Any:
    diagnostics = summary.get("diagnostic_summary")
    if isinstance(diagnostics, dict) and key in diagnostics:
        return diagnostics.get(key)
    return summary.get(key)


def _write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _summaries_by_variant(
    roots: Sequence[str],
    datasets: Sequence[str],
    cache_percentages: Sequence[float],
    variants: Sequence[str],
    feature_set: str,
    base_learner: str,
) -> Dict[Tuple[str, float, str], Dict[str, Any]]:
    out: Dict[Tuple[str, float, str], Dict[str, Any]] = {}
    for dataset in datasets:
        for pct in cache_percentages:
            for variant in variants:
                summary = find_summary(roots, dataset, variant, pct, feature_set, base_learner)
                if summary is not None:
                    out[(dataset, float(pct), variant)] = summary
    return out


def build_suppression_summary(
    summaries: Dict[Tuple[str, float, str], Dict[str, Any]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (dataset, pct, variant), summary in sorted(summaries.items()):
        if variant not in SUPPRESSION_VARIANTS:
            continue
        slots = _cache_rows(load_slot_rows(summary))
        suppressed_total = _optional_number(summary, ("quality_suppressed_total",))
        if suppressed_total is None:
            suppressed_total = _sum_slot(slots, ("quality_suppressed_count", "quality_suppressed_total"))
        admitted_total = _optional_number(summary, ("quality_admitted_total", "quality_admitted_count"))
        if admitted_total is None:
            admitted_total = _sum_slot(
                slots,
                ("quality_admitted_count", "quality_admitted_total", "quality_admitted_eval_count"),
            )
        suppressed_next = _optional_number(summary, ("quality_suppressed_next_slot_requests_total",))
        if suppressed_next is None:
            suppressed_next = _sum_slot(
                slots,
                ("quality_suppressed_next_slot_requests",),
            )
        admitted_next = _optional_number(summary, ("quality_admitted_next_slot_requests_total",))
        if admitted_next is None:
            admitted_next = _sum_slot(slots, ("quality_admitted_next_slot_requests",))
        suppressed_rate = _safe_div(suppressed_next, suppressed_total)
        admitted_rate = _safe_div(admitted_next, admitted_total)
        rows.append(
            {
                "dataset": dataset,
                "capacity_percent": pct,
                "cache_size_objects": summary.get("cache_size_objects"),
                "variant": variant,
                "quality_suppressed_total": suppressed_total,
                "quality_admitted_total": admitted_total,
                "quality_suppressed_next_slot_requests_total": suppressed_next,
                "quality_admitted_next_slot_requests_total": admitted_next,
                "quality_suppressed_next_slot_request_rate": suppressed_rate,
                "quality_admitted_next_slot_request_rate": admitted_rate,
                "quality_admitted_to_suppressed_request_ratio": _safe_div(admitted_rate, suppressed_rate),
                "quality_action_rate": _diagnostic(summary, "quality_action_rate")
                if _diagnostic(summary, "quality_action_rate") is not None
                else _rate(_bool_slot_values(slots, ("quality_changes_integer_budget",))),
                "quality_suppression_rate": _diagnostic(summary, "quality_suppression_rate")
                if _diagnostic(summary, "quality_suppression_rate") is not None
                else _rate(
                    [
                        (_slot_value(row, ("score_quality_mult", "quality_mult")) or 1.0) < 1.0
                        for row in slots
                    ]
                ),
                "quality_masked_by_cap_rate": _diagnostic(summary, "quality_masked_by_cap_rate")
                if _diagnostic(summary, "quality_masked_by_cap_rate") is not None
                else _rate(_bool_slot_values(slots, ("quality_masked_by_cap",))),
                "quality_masked_by_rounding_rate": _diagnostic(summary, "quality_masked_by_rounding_rate")
                if _diagnostic(summary, "quality_masked_by_rounding_rate") is not None
                else _rate(_bool_slot_values(slots, ("quality_masked_by_rounding",))),
                "top_percent_cap_binding_rate": _diagnostic(summary, "top_percent_cap_binding_rate")
                if _diagnostic(summary, "top_percent_cap_binding_rate") is not None
                else _rate(_bool_slot_values(slots, ("top_percent_cap_binds", "score_gate_applied"))),
                "mean_score_quality_mult": _diagnostic(summary, "mean_score_quality_mult")
                if _diagnostic(summary, "mean_score_quality_mult") is not None
                else _mean(_slot_value(row, ("score_quality_mult", "quality_mult")) for row in slots),
                "median_score_quality_mult": _diagnostic(summary, "median_score_quality_mult")
                if _diagnostic(summary, "median_score_quality_mult") is not None
                else _median(_slot_value(row, ("score_quality_mult", "quality_mult")) for row in slots),
            }
        )
    return rows


def _sum_slot(rows: List[Dict[str, Any]], candidates: Sequence[str]) -> float:
    total = 0.0
    for row in rows:
        value = _slot_value(row, candidates)
        if value is not None:
            total += value
    return float(total)


def _bool_slot_values(rows: List[Dict[str, Any]], candidates: Sequence[str]) -> List[bool]:
    values: List[bool] = []
    for row in rows:
        for key in candidates:
            if key in row and row[key] is not None:
                values.append(bool(row[key]))
                break
    return values


def build_effective_precision_summary(
    summaries: Dict[Tuple[str, float, str], Dict[str, Any]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (dataset, pct, variant), summary in sorted(summaries.items()):
        if variant not in EFFECTIVE_PRECISION_VARIANTS:
            continue
        metrics = _summary_metric(summary)
        selected = metrics["total_selected_admissions"]
        rows.append(
            {
                "dataset": dataset,
                "capacity_percent": pct,
                "cache_size_objects": summary.get("cache_size_objects"),
                "variant": variant,
                "hit_ratio": metrics["hit_ratio"],
                "total_selected_admissions": selected,
                "admissions_per_1000_requests": metrics["admissions_per_1000_requests"],
                "hit_yield_per_admission": metrics["hit_yield_per_admission"],
                "pollution_ratio": metrics["pollution_ratio"],
                "admission_precision": metrics["admission_precision"],
                "precision_action_rate": _diagnostic(summary, "precision_action_rate"),
                "precision_multiplier_action_rate": _diagnostic(
                    summary,
                    "precision_multiplier_action_rate",
                ),
                "precision_continuous_action_rate": _diagnostic(
                    summary,
                    "precision_continuous_action_rate",
                ),
                "precision_masked_by_integer_rounding_rate": _diagnostic(
                    summary,
                    "precision_masked_by_integer_rounding_rate",
                ),
                "precision_mean_budget_delta": _diagnostic(summary, "precision_mean_budget_delta"),
                "precision_median_budget_delta": _diagnostic(summary, "precision_median_budget_delta"),
                "mean_precision_multiplier": _diagnostic(summary, "mean_precision_multiplier"),
                "median_precision_multiplier": _diagnostic(summary, "median_precision_multiplier"),
                "corr_precision_eff_to_next_slot_hit_yield": _diagnostic(
                    summary,
                    "corr_precision_eff_to_next_slot_hit_yield",
                ),
                "corr_precision_eff_to_next_slot_admission_precision": _diagnostic(
                    summary,
                    "corr_precision_eff_to_next_slot_admission_precision",
                ),
                "corr_precision_eff_to_admission_window_pollution_proxy": _diagnostic(
                    summary,
                    "corr_precision_eff_to_admission_window_pollution_proxy",
                ),
            }
        )
    return rows


def build_budget_control_summary(
    summaries: Dict[Tuple[str, float, str], Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    full_metrics: Dict[Tuple[str, float], Dict[str, Optional[float]]] = {}
    for key, summary in summaries.items():
        dataset, pct, variant = key
        if variant == "full_cap020":
            full_metrics[(dataset, pct)] = _summary_metric(summary)
    for (dataset, pct, variant), summary in sorted(summaries.items()):
        if variant not in CONTROL_VARIANTS:
            continue
        metrics = _summary_metric(summary)
        full = full_metrics.get((dataset, pct), {})
        full_hr = full.get("hit_ratio")
        full_selected = full.get("total_selected_admissions")
        selected = metrics["total_selected_admissions"]
        rows.append(
            {
                "dataset": dataset,
                "capacity_percent": pct,
                "cache_size_objects": summary.get("cache_size_objects"),
                "variant": variant,
                "hit_ratio": metrics["hit_ratio"],
                "delta_hr_vs_full_cap020": _delta(metrics["hit_ratio"], full_hr),
                "total_selected_admissions": selected,
                "delta_selected_vs_full_cap020": _delta(selected, full_selected),
                "selected_match_ratio_vs_full_cap020": _safe_div(selected, full_selected),
                "admissions_per_1000_requests": metrics["admissions_per_1000_requests"],
                "hit_yield_per_admission": metrics["hit_yield_per_admission"],
                "pollution_ratio": metrics["pollution_ratio"],
                "admission_precision": metrics["admission_precision"],
                "budget_match_mode": _nested_get(summary, "budget_match_mode")
                or _nested_get(summary, "BUDGET_MATCH_MODE"),
                "budget_match_multiplier": _nested_get(summary, "budget_match_multiplier")
                or _nested_get(summary, "BUDGET_MATCH_MULTIPLIER"),
                "budget_replay_enabled": _optional_bool(summary, ("budget_replay_enabled",)),
                "budget_replay_permuted": _optional_bool(summary, ("budget_replay_permuted",)),
                "budget_replay_exact_slot_match_rate": _optional_number(
                    summary,
                    ("budget_replay_exact_slot_match_rate",),
                ),
                "budget_replay_clamp_rate": _optional_number(summary, ("budget_replay_clamp_rate",)),
                "offline_diagnostic_control": _optional_bool(
                    summary,
                    ("offline_diagnostic_control", "OFFLINE_DIAGNOSTIC_CONTROL"),
                ),
            }
        )
    avg_rows = _average_budget_rows(rows)
    return rows, avg_rows


def _delta(value: Any, base: Any) -> Optional[float]:
    value_f = _safe_float(value)
    base_f = _safe_float(base)
    if value_f is None or base_f is None:
        return None
    return float(value_f - base_f)


def _average_budget_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("dataset")), str(row.get("variant")))].append(row)
    out: List[Dict[str, Any]] = []
    metric_map = {
        "avg_hit_ratio": "hit_ratio",
        "avg_delta_hr_vs_full_cap020": "delta_hr_vs_full_cap020",
        "avg_total_selected_admissions": "total_selected_admissions",
        "avg_selected_match_ratio_vs_full_cap020": "selected_match_ratio_vs_full_cap020",
        "avg_hit_yield_per_admission": "hit_yield_per_admission",
        "avg_pollution_ratio": "pollution_ratio",
        "avg_admission_precision": "admission_precision",
        "avg_budget_replay_exact_slot_match_rate": "budget_replay_exact_slot_match_rate",
        "avg_budget_replay_clamp_rate": "budget_replay_clamp_rate",
    }
    for (dataset, variant), group in sorted(groups.items()):
        row = {"dataset": dataset, "variant": variant}
        for out_key, in_key in metric_map.items():
            row[out_key] = _mean(item.get(in_key) for item in group)
        out.append(row)
    return out


def build_paired_bootstrap(
    summaries: Dict[Tuple[str, float, str], Dict[str, Any]],
    seed: int,
    resamples: int,
) -> List[Dict[str, Any]]:
    metrics = {
        "slot_hit_ratio": ("slot_hit_ratio",),
        "admission_hit_yield": (
            "admission_hit_yield",
            "hit_yield_slot",
            "admission_hit_yield_zero_filled",
        ),
        "pollution_proxy": ("pollution_proxy", "pollution_rate_slot"),
        "admission_precision": ("admission_precision",),
        "selected_admissions_per_slot": ("admit_selected", "selected_admissions"),
    }
    rows: List[Dict[str, Any]] = []
    datasets = sorted({dataset for dataset, _, _ in summaries.keys()})
    pcts = sorted({pct for _, pct, _ in summaries.keys()})
    for dataset in datasets:
        for pct in pcts:
            for variant_a, variant_b in PAIRED_COMPARISONS:
                summary_a = summaries.get((dataset, pct, variant_a))
                summary_b = summaries.get((dataset, pct, variant_b))
                if summary_a is None or summary_b is None:
                    continue
                slots_a = _slot_map(_cache_rows(load_slot_rows(summary_a)))
                slots_b = _slot_map(_cache_rows(load_slot_rows(summary_b)))
                common_slots = sorted(set(slots_a) & set(slots_b))
                for metric, candidates in metrics.items():
                    deltas: List[float] = []
                    for slot in common_slots:
                        val_a = _slot_value(slots_a[slot], candidates)
                        val_b = _slot_value(slots_b[slot], candidates)
                        if val_a is None or val_b is None:
                            continue
                        deltas.append(float(val_a - val_b))
                    if not deltas:
                        continue
                    ci_low, ci_high = _bootstrap_ci(deltas, seed, resamples)
                    rows.append(
                        {
                            "dataset": dataset,
                            "capacity_percent": pct,
                            "metric": metric,
                            "variant_a": variant_a,
                            "variant_b": variant_b,
                            "mean_delta": _mean(deltas),
                            "median_delta": _median(deltas),
                            "bootstrap_ci95_low": ci_low,
                            "bootstrap_ci95_high": ci_high,
                            "n_paired_slots": len(deltas),
                            "seed": seed,
                        }
                    )
    return rows


def _slot_map(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        slot = _safe_int(row.get("slot_index"))
        if slot is not None:
            out[slot] = row
    return out


def _bootstrap_ci(values: List[float], seed: int, resamples: int) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(int(seed))
    n = len(values)
    means: List[float] = []
    for _ in range(int(resamples)):
        sample_sum = 0.0
        for _idx in range(n):
            sample_sum += values[rng.randrange(n)]
        means.append(sample_sum / n)
    means.sort()
    low_idx = max(0, int(math.floor(0.025 * (len(means) - 1))))
    high_idx = min(len(means) - 1, int(math.ceil(0.975 * (len(means) - 1))))
    return float(means[low_idx]), float(means[high_idx])


def parse_datasets(value: str) -> List[str]:
    if value == "all":
        return list(DATASETS)
    if value not in DATASETS:
        raise ValueError(f"Unknown dataset: {value}")
    return [value]


def parse_cache_percentages(values: Sequence[str]) -> List[float]:
    if not values:
        return list(DEFAULT_CACHE_PERCENTAGES)
    return [float(value) for value in values]


def run_analysis(args: argparse.Namespace) -> Dict[str, str]:
    datasets = parse_datasets(args.datasets)
    cache_percentages = parse_cache_percentages(args.cache_percentages)
    roots = [args.output_root, args.source_results_root]
    variants = tuple(
        dict.fromkeys(CONTROL_VARIANTS + EFFECTIVE_PRECISION_VARIANTS)
    )
    summaries = _summaries_by_variant(
        roots,
        datasets,
        cache_percentages,
        variants,
        args.feature_set,
        args.base_learner,
    )

    suppression_rows = build_suppression_summary(summaries)
    effective_rows = build_effective_precision_summary(summaries)
    budget_rows, budget_avg_rows = build_budget_control_summary(summaries)
    paired_rows = build_paired_bootstrap(
        summaries,
        seed=args.seed,
        resamples=args.bootstrap_resamples,
    )

    outputs = {
        "score_quality_suppression_summary": os.path.join(
            args.output_root,
            "score_quality_suppression_summary.csv",
        ),
        "effective_precision_summary": os.path.join(
            args.output_root,
            "effective_precision_summary.csv",
        ),
        "cap020_budget_control_summary": os.path.join(
            args.output_root,
            "cap020_budget_control_summary.csv",
        ),
        "cap020_budget_control_avg_by_dataset": os.path.join(
            args.output_root,
            "cap020_budget_control_avg_by_dataset.csv",
        ),
        "paired_slot_bootstrap_ci": os.path.join(
            args.output_root,
            "paired_slot_bootstrap_ci.csv",
        ),
    }
    _write_csv(
        outputs["score_quality_suppression_summary"],
        suppression_rows,
        (
            "dataset",
            "capacity_percent",
            "cache_size_objects",
            "variant",
            "quality_suppressed_total",
            "quality_admitted_total",
            "quality_suppressed_next_slot_requests_total",
            "quality_admitted_next_slot_requests_total",
            "quality_suppressed_next_slot_request_rate",
            "quality_admitted_next_slot_request_rate",
            "quality_admitted_to_suppressed_request_ratio",
            "quality_action_rate",
            "quality_suppression_rate",
            "quality_masked_by_cap_rate",
            "quality_masked_by_rounding_rate",
            "top_percent_cap_binding_rate",
            "mean_score_quality_mult",
            "median_score_quality_mult",
        ),
    )
    _write_csv(
        outputs["effective_precision_summary"],
        effective_rows,
        (
            "dataset",
            "capacity_percent",
            "cache_size_objects",
            "variant",
            "hit_ratio",
            "total_selected_admissions",
            "admissions_per_1000_requests",
            "hit_yield_per_admission",
            "pollution_ratio",
            "admission_precision",
            "precision_action_rate",
            "precision_multiplier_action_rate",
            "precision_continuous_action_rate",
            "precision_masked_by_integer_rounding_rate",
            "precision_mean_budget_delta",
            "precision_median_budget_delta",
            "mean_precision_multiplier",
            "median_precision_multiplier",
            "corr_precision_eff_to_next_slot_hit_yield",
            "corr_precision_eff_to_next_slot_admission_precision",
            "corr_precision_eff_to_admission_window_pollution_proxy",
        ),
    )
    _write_csv(
        outputs["cap020_budget_control_summary"],
        budget_rows,
        (
            "dataset",
            "capacity_percent",
            "cache_size_objects",
            "variant",
            "hit_ratio",
            "delta_hr_vs_full_cap020",
            "total_selected_admissions",
            "delta_selected_vs_full_cap020",
            "selected_match_ratio_vs_full_cap020",
            "admissions_per_1000_requests",
            "hit_yield_per_admission",
            "pollution_ratio",
            "admission_precision",
            "budget_match_mode",
            "budget_match_multiplier",
            "budget_replay_enabled",
            "budget_replay_permuted",
            "budget_replay_exact_slot_match_rate",
            "budget_replay_clamp_rate",
            "offline_diagnostic_control",
        ),
    )
    _write_csv(
        outputs["cap020_budget_control_avg_by_dataset"],
        budget_avg_rows,
        (
            "dataset",
            "variant",
            "avg_hit_ratio",
            "avg_delta_hr_vs_full_cap020",
            "avg_total_selected_admissions",
            "avg_selected_match_ratio_vs_full_cap020",
            "avg_hit_yield_per_admission",
            "avg_pollution_ratio",
            "avg_admission_precision",
            "avg_budget_replay_exact_slot_match_rate",
            "avg_budget_replay_clamp_rate",
        ),
    )
    _write_csv(
        outputs["paired_slot_bootstrap_ci"],
        paired_rows,
        (
            "dataset",
            "capacity_percent",
            "metric",
            "variant_a",
            "variant_b",
            "mean_delta",
            "median_delta",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
            "n_paired_slots",
            "seed",
        ),
    )
    print(
        "[ANALYSIS] rows: "
        f"suppression={len(suppression_rows)}, "
        f"effective_precision={len(effective_rows)}, "
        f"budget_controls={len(budget_rows)}, "
        f"paired={len(paired_rows)}"
    )
    for path in outputs.values():
        print(f"[ANALYSIS] wrote {path}")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze cap020 signal attribution diagnostics.")
    parser.add_argument("--datasets", choices=["all", *DATASETS], default="all")
    parser.add_argument(
        "--cache-percentages",
        nargs="+",
        default=[str(value) for value in DEFAULT_CACHE_PERCENTAGES],
    )
    parser.add_argument("--feature-set", choices=["A0", "A1", "A2", "A3"], default="A2")
    parser.add_argument("--base-learner", choices=["nb", "svm", "dt"], default="nb")
    parser.add_argument("--source-results-root", default=DEFAULT_SOURCE_RESULTS_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.bootstrap_resamples < 1:
        raise ValueError("--bootstrap-resamples must be >= 1.")
    run_analysis(args)


if __name__ == "__main__":
    main()
