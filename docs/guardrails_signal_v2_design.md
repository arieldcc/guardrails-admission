# Guardrails Signal v2 Design

This experiment family isolates slot-level admission budget controls for IL-based cache admission under fixed LRU eviction. It preserves the legacy one-slot delayed apply protocol: slot `t` chooses an admission set from miss candidates, and that set is inserted before serving slot `t+1`. Object ranking is unchanged; the IL score is still the only per-object ranking rule.

Historical v2 outputs default under `results/guardrails_signal_v2/`. The final cap-0.02 validation outputs use `results/guardrails_signal_v2_cap020_validation/`. The old guard, no-guard, overhead, data, learner, and cache files are not modified.

## Guarded Simulator

`src/experiments/run_il_cache_guard_signal_v2.py` is copied from the old guard-only simulator and changes only the budget controller family.

Effective precision is now configurable:

- `off`: no precision feedback.
- `reward_only_legacy`: the old one-sided release behavior, where precision above target can increase alpha and poor precision does not suppress alpha.
- `bidirectional`: the v2 default, where precision above target increases alpha and precision below target suppresses alpha, clipped by `ADMISSION_PRECISION_MULT_MIN/MAX`.

The effective precision proxy can use the legacy max over lags, a fixed lag, or a weighted lag average. Precision targets can be fixed, based on the previous candidate positive-rate EMA, or based on that EMA plus a margin.

Score quality is now computed with previous EMAs only:

- `off`: no score-quality multiplier.
- `spread_prev_ema`: current spread over previous spread EMA.
- `calibrated_spread`: spread ratio multiplied by lagged score precision calibration.
- `boundary_margin`: separability around the pre-quality budget boundary.

The default role is `safety`, so quality can suppress admission under ambiguity but cannot boost above `1.0`. `symmetric` is retained for ablations.

The final selected score-gate metadata is `FINAL_SCORE_GATE_TOP_PERCENT = 0.02`. Historical variants keep their explicit caps; the final Guardrails configuration is represented by explicit cap-0.02 variants rather than by rewriting old result definitions.

## Counterfactual Budgets

Each cache slot logs four raw budget paths and the same bounded final budgets:

- no signals
- precision only
- quality only
- full precision plus quality

The bounded paths apply the same candidate-count, top-percent cap, fill-floor, and zero-candidate semantics as the real admission path. Diagnostics record whether precision or quality changed the integer budget and whether the effect was masked by the top-percent cap, integer rounding, candidate count, or fill floor.

The simulator also logs continuous pre-rounding budget values. These show when a signal changed the continuous budget but did not survive integer rounding.

## Budget-Matched Controls

Budget-matched controls test whether score-quality timing matters beyond a generic budget reduction:

- `precision_only_budget_matched_uniform`: precision-only plus a constant budget multiplier.
- `precision_only_budget_matched_random`: precision-only plus deterministic random slot-level throttling with the same expected multiplier.
- `precision_only_budget_matched_permuted_quality`: precision plus the empirical score-quality multiplier distribution with lagged, permuted timing.
- `precision_only_budget_matched_uniform_cap020`: cap-0.02 precision-only plus a configurable uniform multiplier, defaulting to `1.0` for first-pass measurement.
- `precision_only_budget_matched_random_cap020`: cap-0.02 precision-only plus deterministic random slot throttling, default target multiplier `1.0` and seed `42`.
- `precision_only_budget_matched_permuted_quality_cap020`: cap-0.02 precision plus permuted score-quality timing.

These controls do not alter object ranking. Uniform and random controls do not use score-quality values at all.

## Suppressed-Candidate Utility

When quality reduces the full budget relative to the precision-only counterfactual, the simulator records the IL-ranked candidates that quality suppressed. It logs their current-slot score and label rates, then evaluates future requests for the admitted and suppressed groups over `QUALITY_SUPPRESSED_UTILITY_WINDOW_SLOTS`.

Aggregate summaries include selected/applied admissions, admissions per 1000 requests, hit-yield per applied admission, pollution ratio, admission precision, post-fill versions of those metrics, and rejected-candidate future utility ratios.

## Ablation Runner

`src/experiments/run_il_cache_guard_signal_v2_ablation.py` defines clean variant groups:

- `core_clean`: no-guard legacy, cap-only, precision-only, quality-only, full.
- `core_clean_cap020`: no-guard legacy, cap-only, quality-only, precision-only, and full at the final `0.02` cap.
- `no_cap_stress`: removes the hard cap to check whether signals are hidden by cap binding.
- `cap_masking_grid`: sweeps top-percent caps for quality-only and full controls.
- `score_quality_budget_matched`: compares full score-quality against uniform, random, and permuted budget-matched controls.
- `score_quality_budget_matched_cap020`: compares `full_cap020` against cap-0.02 precision-only, uniform, random, and permuted-quality controls.
- `quality_mode_cap020`: sanity-checks spread, boundary-margin, and calibrated-spread safety modes at cap `0.02`.
- `cap020_validation`: targeted final validation union of `core_clean_cap020`, `score_quality_budget_matched_cap020`, and `quality_mode_cap020`.
- `precision_validity`: compares legacy reward-only precision, bidirectional targets, and lag aggregators.
- `quality_validity`: compares spread, calibrated spread, boundary margin, safety-only, and symmetric roles.
- `negative_controls`: random quality, inverted quality, permuted quality, random precision, and permuted precision controls.

Each all-size summary includes `experiment_family`, `config_summary`, per-cache diagnostic summaries, aggregate diagnostic summaries, source files, a git hash when available, and admission summary fields for budget matching: `selected_admissions_total`, `applied_admissions_total`, and `admissions_per_1000_requests`.

The ablation runner supports `--skip-existing` to avoid rerunning a dataset/variant when an all-size summary already exists in the selected result root with the same feature set, base learner, and cache-size percentage list. It prints both skipped and executed dataset/variant pairs.
