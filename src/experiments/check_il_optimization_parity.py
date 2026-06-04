from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


RUN_LEVEL_FIELDS = (
    "hit_ratio",
    "cache_hits",
    "cache_requests",
    "miss_requests_total",
    "miss_candidates_total",
    "admit_selected_total",
    "admit_applied_total",
    "admit_true_popular_total",
    "hits_from_admitted_total",
    "pollution_count",
    "pollution_rate_total",
    "hit_yield_total",
)

SLOT_FIELDS = (
    "slot_index",
    "phase",
    "miss_requests",
    "miss_candidates",
    "admit_selected",
    "admit_applied",
    "slot_cache_hits",
    "slot_cache_requests",
    "slot_hit_ratio",
    "candidate_ids_hash",
    "candidate_scores_hash",
    "admit_set_hash",
    "pending_admit_hash",
)

POLICIES = {
    "il-no-guard": {
        "script": "src/experiments/run_il_cache_overhead_no_guard.py",
        "model_ref": "ilnse_A2_guard_no_guard_nb_overhead",
        "model_opt": "ilnse_A2_guard_no_guard_nb_overhead_optimized",
    },
    "il-guard": {
        "script": "src/experiments/run_il_cache_overhead_guard.py",
        "model_ref": "ilnse_A2_guard_full_nb_overhead",
        "model_opt": "ilnse_A2_guard_full_nb_overhead_optimized",
    },
}


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _run_policy(args: argparse.Namespace, policy: str, impl_mode: str, results_root: str) -> str:
    spec = POLICIES[policy]
    cmd = [
        sys.executable,
        spec["script"],
        "--dataset",
        args.dataset,
        "--feature-set",
        args.feature_set,
        "--base-learner",
        args.base_learner,
        "--capacity-percent",
        str(args.capacity_percent),
        "--max-requests",
        str(args.max_requests),
        "--warmup-requests",
        str(args.warmup_requests),
        "--slot-size",
        str(args.slot_size),
        "--results-root",
        results_root,
        "--benchmark-mode",
        "--disable-progress",
        "--run-id",
        f"parity_{policy.replace('-', '_')}_{impl_mode}",
        "--seed",
        str(args.seed),
        "--impl-mode",
        impl_mode,
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    model_name = spec["model_opt"] if impl_mode == "optimized" else spec["model_ref"]
    pattern = os.path.join(
        results_root,
        args.dataset,
        f"parity_{policy.replace('-', '_')}_{impl_mode}_{model_name}_*",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise RuntimeError(f"No run directory matches: {pattern}")
    return matches[-1]


def _equal_value(a: Any, b: Any, field: str) -> bool:
    if field in ("hit_ratio", "slot_hit_ratio", "pollution_rate_total", "hit_yield_total"):
        if a is None or b is None:
            return a is b
        return abs(float(a) - float(b)) <= 1e-12
    return a == b


def _compare_run(ref_dir: str, opt_dir: str) -> Dict[str, Any]:
    ref_summary = _load_json(os.path.join(ref_dir, "overhead_summary.json"))
    opt_summary = _load_json(os.path.join(opt_dir, "overhead_summary.json"))
    ref_slots = _load_jsonl(os.path.join(ref_dir, "slot_log.jsonl"))
    opt_slots = _load_jsonl(os.path.join(opt_dir, "slot_log.jsonl"))

    failures: List[Dict[str, Any]] = []
    for field in RUN_LEVEL_FIELDS:
        if not _equal_value(ref_summary.get(field), opt_summary.get(field), field):
            failures.append(
                {
                    "level": "run",
                    "field": field,
                    "reference": ref_summary.get(field),
                    "optimized": opt_summary.get(field),
                }
            )

    if len(ref_slots) != len(opt_slots):
        failures.append(
            {
                "level": "slot_log",
                "field": "num_slots",
                "reference": len(ref_slots),
                "optimized": len(opt_slots),
            }
        )
    else:
        for ref, opt in zip(ref_slots, opt_slots):
            for field in SLOT_FIELDS:
                if not _equal_value(ref.get(field), opt.get(field), field):
                    failures.append(
                        {
                            "level": "slot",
                            "slot_index": ref.get("slot_index"),
                            "field": field,
                            "reference": ref.get(field),
                            "optimized": opt.get(field),
                        }
                    )
                    break
            if failures and failures[-1].get("level") == "slot":
                break

    return {
        "reference_dir": ref_dir,
        "optimized_dir": opt_dir,
        "passed": not failures,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check IL reference/optimized parity.")
    parser.add_argument("--dataset", default="wikipedia_september_2007")
    parser.add_argument("--feature-set", default="A2")
    parser.add_argument("--base-learner", default="nb")
    parser.add_argument("--capacity-percent", type=float, default=0.8)
    parser.add_argument("--max-requests", type=int, default=500_000)
    parser.add_argument("--warmup-requests", type=int, default=100_000)
    parser.add_argument("--slot-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-root", default=None)
    args = parser.parse_args()

    results_root = args.results_root or tempfile.mkdtemp(prefix="il_parity_", dir="/private/tmp")
    Path(results_root).mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "results_root": results_root,
        "config": vars(args),
        "policies": {},
        "passed": True,
    }
    for policy in ("il-no-guard", "il-guard"):
        ref_dir = _run_policy(args, policy, "reference", results_root)
        opt_dir = _run_policy(args, policy, "optimized", results_root)
        comparison = _compare_run(ref_dir, opt_dir)
        results["policies"][policy] = comparison
        if not comparison["passed"]:
            results["passed"] = False

    out_path = os.path.join(results_root, "parity_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if not results["passed"]:
        failure_path = os.path.join(results_root, "parity_failure.json")
        with open(failure_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))
        raise SystemExit(1)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
