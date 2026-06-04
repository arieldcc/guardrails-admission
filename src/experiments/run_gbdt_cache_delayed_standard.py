from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.run_gbdt_cache_overhead import _resolve_dataset, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standard sklearn Delayed-GBDT cache simulation."
    )
    parser.add_argument("--dataset", choices=["wikipedia_september_2007", "wiki2018"], required=True)
    parser.add_argument("--feature-set", default="A2")
    parser.add_argument("--cache-size-objects", type=int, default=None)
    parser.add_argument("--capacity-objects", type=int, default=None)
    parser.add_argument("--capacity-percent", type=float, default=None)
    parser.add_argument("--results-root", default="results/overhead")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--prediction-mode", choices=["per_candidate", "batch"], default="per_candidate")
    args = parser.parse_args()

    capacity_objects = (
        args.capacity_objects
        if args.capacity_objects is not None
        else args.cache_size_objects
    )
    run_experiment(
        _resolve_dataset(args.dataset),
        feature_set=args.feature_set,
        cache_size_objects=capacity_objects,
        capacity_percent=args.capacity_percent,
        results_root=args.results_root,
        disable_progress=args.disable_progress,
        benchmark_mode=False,
        seed=args.seed,
        max_requests=args.max_requests,
        smoke_test=args.smoke_test,
        prediction_mode=args.prediction_mode,
    )


if __name__ == "__main__":
    main()
