from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments import analyze_cap020_signal_attribution as attribution
from src.experiments import run_il_cache_guard_signal_v2_ablation as ablation


DATASETS = ("wikipedia_september_2007", "wiki2018")
DEFAULT_CACHE_PERCENTAGES = (0.8, 1.0, 2.0, 3.0, 4.0, 5.0)
DEFAULT_SOURCE_RESULTS_ROOT = "results/guardrails_signal_v2_cap020"
DEFAULT_OUTPUT_ROOT = "results/guardrails_signal_v2_cap020_budget_controls"

SELECTED_TOTAL_KEYS = (
    "total_selected_admissions",
    "selected_admissions_total",
    "admit_selected_total",
    "postfill_selected_admissions",
)
REPLAY_BUDGET_KEYS = (
    "admit_selected",
    "selected_admissions",
    "final_budget",
    "admit_budget",
    "final_budget_after_quality_cap",
    "final_budget_full_same_cap",
)
SLOT_INDEX_KEYS = ("slot_index", "slot", "slot_num")
CONTROL_VARIANTS = (
    "precision_only_budget_matched_uniform_cap020",
    "precision_only_budget_matched_random_cap020",
    "precision_only_budget_replay_cap020",
    "precision_only_budget_replay_permuted_cap020",
)


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _required_number(mapping: Dict[str, Any], keys: Sequence[str], context: str) -> float:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            value = _safe_float(mapping[key])
            if value is not None:
                return value
        config = mapping.get("config_summary")
        if isinstance(config, dict) and key in config and config[key] is not None:
            value = _safe_float(config[key])
            if value is not None:
                return value
    available = ", ".join(sorted(str(key) for key in mapping.keys()))
    raise KeyError(
        f"{context}: missing required numeric field. Expected one of "
        f"{list(keys)}. Available top-level keys: [{available}]"
    )


def _resolve_required_key(row: Dict[str, Any], keys: Sequence[str], context: str) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            return key
    available = ", ".join(sorted(str(key) for key in row.keys()))
    raise KeyError(
        f"{context}: missing required key. Expected one of {list(keys)}. "
        f"Available keys: [{available}]"
    )


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


def _run_id_from_path(path: str) -> int:
    prefix = os.path.basename(path).split("_", 1)[0]
    return int(prefix) if len(prefix) == 3 and prefix.isdigit() else -1


def find_summary(
    root: str,
    dataset: str,
    variant: str,
    capacity_percent: float,
    feature_set: str,
    base_learner: str,
) -> Optional[Dict[str, Any]]:
    dataset_dir = os.path.join(root, dataset)
    if not os.path.isdir(dataset_dir):
        return None
    matches: List[Tuple[float, int, str, Dict[str, Any]]] = []
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
        matches.append((mtime, _run_id_from_path(path), path, data))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1], item[2]))
    return matches[-1][3]


def _dataset_args(value: str) -> List[str]:
    if value == "all":
        return list(DATASETS)
    if value not in DATASETS:
        raise ValueError(f"Unknown dataset: {value}")
    return [value]


def _pct_label(value: float) -> str:
    return f"{float(value):.1f}"


def compute_budget_ratios(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset in _dataset_args(args.datasets):
        for pct in args.cache_percentages:
            full = find_summary(
                args.source_results_root,
                dataset,
                "full_cap020",
                pct,
                args.feature_set,
                args.base_learner,
            )
            precision = find_summary(
                args.source_results_root,
                dataset,
                "precision_only_cap020",
                pct,
                args.feature_set,
                args.base_learner,
            )
            if full is None:
                raise FileNotFoundError(f"Missing full_cap020 summary for {dataset} pct={pct}.")
            if precision is None:
                raise FileNotFoundError(
                    f"Missing precision_only_cap020 summary for {dataset} pct={pct}."
                )
            full_selected = _required_number(
                full,
                SELECTED_TOTAL_KEYS,
                str(full.get("_summary_path")),
            )
            precision_selected = _required_number(
                precision,
                SELECTED_TOTAL_KEYS,
                str(precision.get("_summary_path")),
            )
            if precision_selected == 0.0:
                raise ZeroDivisionError(
                    f"precision_only selected admissions is zero for {dataset} pct={pct}."
                )
            rows.append(
                {
                    "dataset": dataset,
                    "capacity_percent": float(pct),
                    "cache_size_objects": int(full.get("cache_size_objects")),
                    "full_total_selected_admissions": int(full_selected),
                    "precision_only_total_selected_admissions": int(precision_selected),
                    "ratio_full_over_precision": float(full_selected / precision_selected),
                    "source_full_summary_path": full.get("_summary_path"),
                    "source_precision_summary_path": precision.get("_summary_path"),
                    "_full_summary": full,
                }
            )
    return rows


def write_ratio_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = (
        "dataset",
        "capacity_percent",
        "cache_size_objects",
        "full_total_selected_admissions",
        "precision_only_total_selected_admissions",
        "ratio_full_over_precision",
        "source_full_summary_path",
        "source_precision_summary_path",
    )
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def load_cache_slot_budgets(slot_log_path: str) -> Tuple[List[Dict[str, Any]], str]:
    slots: List[Dict[str, Any]] = []
    key_used: Optional[str] = None
    with open(slot_log_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("phase") != "cache":
                continue
            slot_key = _resolve_required_key(row, SLOT_INDEX_KEYS, f"{slot_log_path}:{line_num}")
            budget_key = _resolve_required_key(row, REPLAY_BUDGET_KEYS, f"{slot_log_path}:{line_num}")
            if key_used is None:
                key_used = budget_key
            elif key_used != budget_key:
                raise ValueError(
                    f"{slot_log_path}: mixed replay budget keys {key_used!r} and {budget_key!r}."
                )
            budget = int(row[budget_key])
            if budget < 0:
                raise ValueError(f"{slot_log_path}:{line_num}: negative replay budget {budget}.")
            slots.append(
                {
                    "slot_index": int(row[slot_key]),
                    "phase": "cache",
                    "budget": budget,
                }
            )
    if not slots:
        raise ValueError(f"{slot_log_path}: no cache-phase slot budgets found.")
    return slots, str(key_used)


def generate_replay_files(
    row: Dict[str, Any],
    output_root: str,
    seed: int,
    overwrite: bool,
    dry_run: bool,
) -> Tuple[str, str]:
    full_summary = row["_full_summary"]
    dataset = row["dataset"]
    pct = float(row["capacity_percent"])
    pct_label = _pct_label(pct)
    replay_dir = os.path.join(output_root, "replay_budgets")
    exact_path = os.path.join(
        replay_dir,
        f"{dataset}_{pct_label}_full_cap020_budget_replay.json",
    )
    permuted_path = os.path.join(
        replay_dir,
        f"{dataset}_{pct_label}_full_cap020_budget_replay_permuted_seed{int(seed)}.json",
    )
    if dry_run:
        print(f"[DRY-RUN] would generate {exact_path}")
        print(f"[DRY-RUN] would generate {permuted_path}")
        return exact_path, permuted_path

    os.makedirs(replay_dir, exist_ok=True)
    slot_log_path = str(full_summary.get("slot_log_path") or "")
    if not slot_log_path:
        raise KeyError(f"{full_summary.get('_summary_path')}: missing slot_log_path.")
    slots, budget_key_used = load_cache_slot_budgets(slot_log_path)
    exact_payload = {
        "dataset": dataset,
        "capacity_percent": pct,
        "cache_size_objects": int(row["cache_size_objects"]),
        "source_slot_log_path": slot_log_path,
        "source_variant": "full_cap020",
        "replay_budget_key_used": budget_key_used,
        "slots": slots,
    }
    if overwrite or not os.path.exists(exact_path):
        with open(exact_path, "w", encoding="utf-8") as f:
            json.dump(exact_payload, f, ensure_ascii=False, indent=2)

    ordered_slots = sorted(slots, key=lambda item: int(item["slot_index"]))
    budgets = [int(item["budget"]) for item in ordered_slots]
    permuted_budgets = list(budgets)
    rng = random.Random(int(seed))
    rng.shuffle(permuted_budgets)
    same_slot_count = sum(1 for a, b in zip(budgets, permuted_budgets) if a == b)
    permuted_slots = [
        {
            "slot_index": int(slot["slot_index"]),
            "phase": "cache",
            "budget": int(budget),
            "source_budget": int(slot["budget"]),
        }
        for slot, budget in zip(ordered_slots, permuted_budgets)
    ]
    permuted_payload = {
        "dataset": dataset,
        "capacity_percent": pct,
        "cache_size_objects": int(row["cache_size_objects"]),
        "source_slot_log_path": slot_log_path,
        "source_variant": "full_cap020",
        "source_replay_path": exact_path,
        "replay_budget_key_used": budget_key_used,
        "budget_replay_permuted": True,
        "budget_replay_permute_seed": int(seed),
        "budget_replay_original_total_budget": int(sum(budgets)),
        "budget_replay_permuted_total_budget": int(sum(permuted_budgets)),
        "budget_replay_histogram_preserved": sorted(budgets) == sorted(permuted_budgets),
        "budget_replay_same_slot_budget_rate_against_full": (
            float(same_slot_count / len(budgets)) if budgets else None
        ),
        "slots": permuted_slots,
    }
    if overwrite or not os.path.exists(permuted_path):
        with open(permuted_path, "w", encoding="utf-8") as f:
            json.dump(permuted_payload, f, ensure_ascii=False, indent=2)
    if sum(budgets) != sum(permuted_budgets):
        raise AssertionError("Permuted replay schedule changed total budget.")
    if sorted(budgets) != sorted(permuted_budgets):
        raise AssertionError("Permuted replay schedule changed budget multiset.")
    return exact_path, permuted_path


def existing_output_summary(
    args: argparse.Namespace,
    dataset: str,
    variant: str,
    pct: float,
) -> Optional[Dict[str, Any]]:
    return find_summary(
        args.output_root,
        dataset,
        variant,
        pct,
        args.feature_set,
        args.base_learner,
    )


def control_overrides(
    variant: str,
    row: Dict[str, Any],
    ratio_csv_path: str,
    exact_replay_path: str,
    seed: int,
) -> Dict[str, Any]:
    ratio = float(row["ratio_full_over_precision"])
    spec = ablation.get_variant_spec(variant)
    overrides = dict(spec.overrides)
    overrides["CONTROL_RANDOM_SEED"] = int(seed)
    overrides["BUDGET_MATCH_RATIO_SOURCE"] = ratio_csv_path
    overrides["BUDGET_MATCH_MULTIPLIER"] = ratio
    overrides["BUDGET_MATCH_TARGET_MULTIPLIER"] = ratio
    if variant == "precision_only_budget_matched_uniform_cap020":
        overrides["BUDGET_MATCH_MODE"] = "uniform"
    elif variant == "precision_only_budget_matched_random_cap020":
        overrides["BUDGET_MATCH_MODE"] = "random_slot_throttle"
    elif variant == "precision_only_budget_replay_cap020":
        overrides["BUDGET_MATCH_MODE"] = "slot_replay"
        overrides["BUDGET_REPLAY_PATH"] = exact_replay_path
        overrides["BUDGET_REPLAY_KEY"] = "budget"
        overrides["BUDGET_REPLAY_STRICT"] = True
        overrides["BUDGET_REPLAY_ALLOW_CLAMP"] = True
        overrides["OFFLINE_DIAGNOSTIC_CONTROL"] = True
    elif variant == "precision_only_budget_replay_permuted_cap020":
        overrides["BUDGET_MATCH_MODE"] = "slot_replay_permuted"
        overrides["BUDGET_REPLAY_PATH"] = exact_replay_path
        overrides["BUDGET_REPLAY_KEY"] = "budget"
        overrides["BUDGET_REPLAY_STRICT"] = True
        overrides["BUDGET_REPLAY_ALLOW_CLAMP"] = True
        overrides["BUDGET_REPLAY_PERMUTE_SEED"] = int(seed)
        overrides["OFFLINE_DIAGNOSTIC_CONTROL"] = True
    else:
        raise ValueError(f"Unknown control variant: {variant}")
    return overrides


def run_control(
    args: argparse.Namespace,
    row: Dict[str, Any],
    variant: str,
    ratio_csv_path: str,
    exact_replay_path: str,
) -> Optional[str]:
    dataset = str(row["dataset"])
    pct = float(row["capacity_percent"])
    capacity = int(row["cache_size_objects"])
    existing = existing_output_summary(args, dataset, variant, pct)
    if existing is not None and not args.overwrite:
        print(f"[SKIP] {dataset} {variant} pct={pct} -> {existing.get('_summary_path')}")
        return str(existing.get("_summary_path"))
    if args.dry_run:
        action = "rerun" if existing is not None and args.overwrite else "run"
        print(f"[DRY-RUN] would {action} {dataset} {variant} pct={pct} cap={capacity}")
        return None

    ds = ablation.resolve_dataset(dataset)
    spec = ablation.get_variant_spec(variant)
    module = spec.module
    model_name = ablation.model_name_for_variant(variant, args.feature_set, args.base_learner)
    overrides = control_overrides(
        variant,
        row,
        ratio_csv_path,
        exact_replay_path,
        args.seed,
    )
    with ablation.patched_module(module, overrides, disable_progress=True):
        cfg_summary = ablation.config_summary_for_variant(spec, module)
        run_id, dataset_dir = ablation.get_next_run_id(args.output_root, dataset, model_name, capacity)
        slot_log_path = os.path.join(dataset_dir, f"{run_id}_{model_name}_{capacity}.jsonl")
        summary_path = os.path.join(
            dataset_dir,
            f"{run_id}_summary_{model_name}_{capacity}.json",
        )
        stats, summary_metrics = module.run_single_capacity(
            trace_path=ds.path,
            total_requests=ds.total_requests,
            warmup_requests=ds.warmup_requests,
            slot_size=ds.slot_size,
            capacity_objects=capacity,
            feature_set=args.feature_set,
            base_learner=args.base_learner,
            slot_log_path=slot_log_path,
        )
        summary_metrics = dict(summary_metrics)
        summary_metrics.update(ablation.admission_summary_aliases(summary_metrics))
        slots_processed = int(summary_metrics.get("slots_processed", 0))
        warmup_slots = ds.warmup_requests // ds.slot_size
        summary_payload: Dict[str, Any] = {
            "dataset": dataset,
            "model": model_name,
            "variant": variant,
            "experiment_family": ablation.EXPERIMENT_FAMILY,
            "feature_set": args.feature_set,
            "base_learner": args.base_learner,
            "trace_path": ds.path,
            "total_requests": ds.total_requests,
            "warmup_requests": ds.warmup_requests,
            "slot_size": ds.slot_size,
            "cache_size_objects": capacity,
            "capacity_percent": pct,
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
            "precision_feedback_enabled": True,
            "score_quality_enabled": False,
            "final_score_gate_top_percent": 0.02,
            "budget_match_mode": overrides.get("BUDGET_MATCH_MODE"),
            "budget_match_multiplier": overrides.get("BUDGET_MATCH_MULTIPLIER"),
            "budget_match_target_multiplier": overrides.get("BUDGET_MATCH_TARGET_MULTIPLIER"),
            "budget_match_ratio_source": ratio_csv_path,
            "budget_replay_enabled": overrides.get("BUDGET_MATCH_MODE")
            in {"slot_replay", "slot_replay_permuted"},
            "budget_replay_permuted": overrides.get("BUDGET_MATCH_MODE") == "slot_replay_permuted",
            "offline_diagnostic_control": bool(overrides.get("OFFLINE_DIAGNOSTIC_CONTROL", False)),
            "config_summary": cfg_summary,
            "diagnostic_summary": summary_metrics.get("diagnostic_summary") or {},
            "source_files": ablation.source_files_for_variant(spec),
            "code_version": ablation.code_version(),
            "reproducibility_seed": int(args.seed),
        }
        summary_payload.update(summary_metrics)
        ablation.save_json(summary_path, summary_payload)
        print(f"[RUN] {dataset} {variant} pct={pct} HR={stats.hit_ratio:.6f} -> {summary_path}")
        return summary_path


def run_controls(args: argparse.Namespace, ratio_rows: List[Dict[str, Any]], ratio_csv_path: str) -> None:
    for row in ratio_rows:
        exact_replay_path, _ = generate_replay_files(
            row,
            args.output_root,
            args.seed,
            args.overwrite,
            args.dry_run,
        )
        for variant in CONTROL_VARIANTS:
            run_control(args, row, variant, ratio_csv_path, exact_replay_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cap020 budget-matched and replay controls.")
    parser.add_argument("--datasets", choices=["all", *DATASETS], default="all")
    parser.add_argument(
        "--cache-percentages",
        nargs="+",
        type=float,
        default=list(DEFAULT_CACHE_PERCENTAGES),
    )
    parser.add_argument("--feature-set", choices=["A0", "A1", "A2", "A3"], default="A2")
    parser.add_argument("--base-learner", choices=["nb", "svm", "dt"], default="nb")
    parser.add_argument("--source-results-root", default=DEFAULT_SOURCE_RESULTS_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    ratio_rows = compute_budget_ratios(args)
    ratio_csv_path = os.path.join(args.output_root, "budget_match_ratios_cap020.csv")
    if args.dry_run:
        print(f"[DRY-RUN] computed {len(ratio_rows)} budget-match ratios")
        print(f"[DRY-RUN] would write {ratio_csv_path}")
    else:
        write_ratio_csv(ratio_csv_path, ratio_rows)
        print(f"[RATIOS] wrote {ratio_csv_path}")
    run_controls(args, ratio_rows, ratio_csv_path)
    if not args.dry_run:
        analysis_args = argparse.Namespace(
            datasets=args.datasets,
            cache_percentages=[str(value) for value in args.cache_percentages],
            feature_set=args.feature_set,
            base_learner=args.base_learner,
            source_results_root=args.source_results_root,
            output_root=args.output_root,
            seed=args.seed,
            bootstrap_resamples=2000,
        )
        attribution.run_analysis(analysis_args)


if __name__ == "__main__":
    main()
