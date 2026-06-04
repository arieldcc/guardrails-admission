#!/usr/bin/env bash
set -euo pipefail

REPEATS=5
RESULTS_ROOT="results/overhead_benchmark"
DATASET_FILTER="all"
POLICY_FILTER="all"
CAPACITY_PERCENT="0.8"
IMPL_MODE="optimized"
GUARD_CONFIG="v2-cap020"
GBDT_PREDICTION_MODE="batch"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage: scripts/run_overhead_benchmark.sh [options]

Options:
  --repeats N              Number of repeated runs per configuration (default: 5)
  --results-root PATH      Output root (default: results/overhead_benchmark)
  --dataset NAME           all, wikipedia_september_2007, or wiki2018 (default: all)
  --policy NAME            all, gbdt, il-no-guard, or il-guard (default: all)
  --capacity-percent PCT   Cache size percentage (default: 0.8)
  --impl-mode MODE         reference or optimized for IL policies (default: optimized)
  --guard-config MODE      v2-cap020 for il-guard (default: v2-cap020)
  --gbdt-prediction-mode MODE
                           per_candidate or batch for GBDT (default: batch)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repeats)
      REPEATS="$2"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --dataset)
      DATASET_FILTER="$2"
      shift 2
      ;;
    --policy)
      POLICY_FILTER="$2"
      shift 2
      ;;
    --capacity-percent)
      CAPACITY_PERCENT="$2"
      shift 2
      ;;
    --impl-mode)
      IMPL_MODE="$2"
      shift 2
      ;;
    --guard-config)
      GUARD_CONFIG="$2"
      shift 2
      ;;
    --gbdt-prediction-mode)
      GBDT_PREDICTION_MODE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${IMPL_MODE}" != "reference" && "${IMPL_MODE}" != "optimized" ]]; then
  echo "Invalid --impl-mode: ${IMPL_MODE}" >&2
  exit 2
fi

if [[ "${GBDT_PREDICTION_MODE}" != "per_candidate" && "${GBDT_PREDICTION_MODE}" != "batch" ]]; then
  echo "Invalid --gbdt-prediction-mode: ${GBDT_PREDICTION_MODE}" >&2
  exit 2
fi

if [[ "${GUARD_CONFIG}" != "v2-cap020" ]]; then
  echo "Invalid --guard-config: ${GUARD_CONFIG}" >&2
  exit 2
fi

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

dataset_enabled() {
  [[ "${DATASET_FILTER}" == "all" || "${DATASET_FILTER}" == "$1" ]]
}

policy_enabled() {
  [[ "${POLICY_FILTER}" == "all" || "${POLICY_FILTER}" == "$1" ]]
}

model_name_for_policy() {
  case "$1" in
    gbdt)
      if [[ "${GBDT_PREDICTION_MODE}" == "batch" ]]; then
        echo "delayed_gdbt_standard_A2_sklearn_batch_overhead"
      else
        echo "delayed_gdbt_standard_A2_overhead"
      fi
      ;;
    il-no-guard)
      if [[ "${IMPL_MODE}" == "optimized" ]]; then
        echo "ilnse_A2_guard_no_guard_nb_overhead_optimized"
      else
        echo "ilnse_A2_guard_no_guard_nb_overhead"
      fi
      ;;
    il-guard)
      if [[ "${IMPL_MODE}" == "optimized" ]]; then
        echo "ilnse_A2_guardv2_full_cap020_nb_overhead_optimized"
      else
        echo "ilnse_A2_guardv2_full_cap020_nb_overhead"
      fi
      ;;
    *) echo "Unknown policy: $1" >&2; exit 2 ;;
  esac
}

script_for_policy() {
  case "$1" in
    gbdt) echo "src/experiments/run_gbdt_cache_overhead.py" ;;
    il-no-guard) echo "src/experiments/run_il_cache_overhead_no_guard.py" ;;
    il-guard) echo "src/experiments/run_il_cache_overhead_guard.py" ;;
    *) echo "Unknown policy: $1" >&2; exit 2 ;;
  esac
}

run_one() {
  local repeat="$1"
  local dataset="$2"
  local policy="$3"
  local script
  local run_id
  script="$(script_for_policy "${policy}")"
  run_id="$(printf 'r%02d' "${repeat}")"

  "${PYTHON_BIN}" "${script}" \
    --dataset "${dataset}" \
    --feature-set A2 \
    --capacity-percent "${CAPACITY_PERCENT}" \
    --results-root "${RESULTS_ROOT}" \
    --disable-progress \
    --benchmark-mode \
    --run-id "${run_id}" \
    --seed 42 \
    $(if [[ "${policy}" == il-* ]]; then printf '%s' "--impl-mode ${IMPL_MODE}"; fi) \
    $(if [[ "${policy}" == "il-guard" ]]; then printf '%s' "--guard-config ${GUARD_CONFIG}"; fi) \
    $(if [[ "${policy}" == "gbdt" ]]; then printf '%s' "--prediction-mode ${GBDT_PREDICTION_MODE}"; fi)
}

aggregate_one() {
  local dataset="$1"
  local policy="$2"
  local model_name
  local caps=""
  local dir
  local cap
  model_name="$(model_name_for_policy "${policy}")"

  for dir in "${RESULTS_ROOT}/${dataset}"/r*_"${model_name}"_*; do
    [[ -d "${dir}" ]] || continue
    cap="${dir##*_}"
    case " ${caps} " in
      *" ${cap} "*) ;;
      *) caps="${caps} ${cap}" ;;
    esac
  done

  for cap in ${caps}; do
    local aggregate_dir="${RESULTS_ROOT}/${dataset}/aggregate_${model_name}_${cap}"
    mkdir -p "${aggregate_dir}"
    "${PYTHON_BIN}" -m src.experiments.summarize_overhead \
      --aggregate-glob "${RESULTS_ROOT}/${dataset}/r*_${model_name}_${cap}/overhead_summary.json" \
      --aggregate-out-json "${aggregate_dir}/aggregate_summary.json" \
      --aggregate-out-csv "${aggregate_dir}/aggregate_summary.csv" >/dev/null
  done
}

DATASETS=(wikipedia_september_2007 wiki2018)
POLICIES=(gbdt il-no-guard il-guard)

echo "Benchmark configuration:"
echo "  repeats              : ${REPEATS}"
echo "  results root         : ${RESULTS_ROOT}"
echo "  dataset filter       : ${DATASET_FILTER}"
echo "  policy filter        : ${POLICY_FILTER}"
echo "  capacity percent     : ${CAPACITY_PERCENT}"
echo "  IL implementation    : ${IMPL_MODE}"
echo "  Guard config         : ${GUARD_CONFIG}"
echo "  GBDT prediction mode : ${GBDT_PREDICTION_MODE}"

for repeat in $(seq 1 "${REPEATS}"); do
  for dataset in "${DATASETS[@]}"; do
    dataset_enabled "${dataset}" || continue
    for policy in "${POLICIES[@]}"; do
      policy_enabled "${policy}" || continue
      run_one "${repeat}" "${dataset}" "${policy}"
    done
  done
done

for dataset in "${DATASETS[@]}"; do
  dataset_enabled "${dataset}" || continue
  for policy in "${POLICIES[@]}"; do
    policy_enabled "${policy}" || continue
    aggregate_one "${dataset}" "${policy}"
  done
done

echo "Benchmark outputs written under ${RESULTS_ROOT}"
