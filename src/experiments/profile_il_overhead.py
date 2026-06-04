from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import tempfile
from pathlib import Path
from typing import Any, Dict

import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.run_il_cache_overhead_guard import (
    WIKIPEDIA_SEPTEMBER_2007,
    WIKI2018,
    run_experiment as run_guard,
)
from src.experiments.run_il_cache_overhead_no_guard import run_experiment as run_no_guard


def _dataset(name: str) -> Dict[str, Any]:
    if name == "wikipedia_september_2007":
        return WIKIPEDIA_SEPTEMBER_2007
    if name == "wiki2018":
        return WIKI2018
    raise ValueError(f"Unknown dataset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile IL overhead runner. Do not use for paper timings.")
    parser.add_argument("--dataset", default="wikipedia_september_2007")
    parser.add_argument("--policy", choices=["il-no-guard", "il-guard"], default="il-guard")
    parser.add_argument("--feature-set", default="A2")
    parser.add_argument("--base-learner", default="nb")
    parser.add_argument("--capacity-percent", type=float, default=0.8)
    parser.add_argument("--impl-mode", choices=["reference", "optimized"], default="reference")
    parser.add_argument("--max-requests", type=int, default=300_000)
    parser.add_argument("--warmup-requests", type=int, default=100_000)
    parser.add_argument("--slot-size", type=int, default=100_000)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--profile-out", default="profile_summary.json")
    args = parser.parse_args()

    results_root = args.results_root or tempfile.mkdtemp(prefix="il_profile_", dir="/private/tmp")
    runner = run_guard if args.policy == "il-guard" else run_no_guard

    profiler = cProfile.Profile()
    profiler.enable()
    runner(
        _dataset(args.dataset),
        feature_set=args.feature_set,
        base_learner=args.base_learner,
        capacity_percent=args.capacity_percent,
        results_root=results_root,
        disable_progress=True,
        benchmark_mode=True,
        run_id_override=f"profile_{args.impl_mode}",
        max_requests=args.max_requests,
        impl_mode=args.impl_mode,
        warmup_requests_override=args.warmup_requests,
        slot_size_override=args.slot_size,
    )
    profiler.disable()

    stats_path = Path(results_root) / "profile.pstats"
    profiler.dump_stats(str(stats_path))
    stats = pstats.Stats(profiler)

    def rows(sort_key: str) -> list[dict[str, Any]]:
        stats.sort_stats(sort_key)
        output = []
        for func, stat in list(stats.stats.items())[:40]:
            cc, nc, tt, ct, callers = stat
            output.append(
                {
                    "file": func[0],
                    "line": func[1],
                    "function": func[2],
                    "call_count": nc,
                    "primitive_call_count": cc,
                    "self_time_s": tt,
                    "cumulative_time_s": ct,
                }
            )
        return output

    summary = {
        "warning": "Profiling mode is diagnostic only; do not use for paper overhead numbers.",
        "config": vars(args),
        "results_root": results_root,
        "profile_stats_path": str(stats_path),
        "top_cumulative_functions": rows("cumulative"),
        "top_self_time_functions": rows("time"),
    }
    out_path = Path(args.profile_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
