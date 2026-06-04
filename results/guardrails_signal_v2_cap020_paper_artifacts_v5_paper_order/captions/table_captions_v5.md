# Table Captions v5

## Table_I_experimental_setup_paper
Experimental setup for trace-driven replay. All experiments use object-level trace replay with fixed LRU eviction and one-slot delayed admission. Guardrails controls admission only and does not modify the eviction policy.

## Table_II_system_level_average_hr_paper
System-level average hit-ratio summary. The table complements Fig. 3 by reporting averages, best and worst values, IL-guard-relative deltas, and rank across cache sizes.

## Table_III_admission_diagnostics_across_baselines_paper
Admission diagnostics summary. Lower insertion pressure and pollution proxy are better; higher hit-yield per admission and HR are better. Proxy-derived diagnostics are used only for contextual comparison.

## Table_IV_effective_precision_attribution_paper
Effective-precision attribution summary. The causal contrast for Fig. 5 is Precision-only versus Cap-only; the IL-guard row is retained as full-controller context.

## Table_V_score_quality_timing_summary_paper
Score-quality timing summary. The table reports selected-admission volume, match ratio versus IL-guard, average HR, ΔHR versus IL-guard, hit-yield per admission, and pollution proxy so Fig. 6 can focus on timing behavior.

## Table_S1_selective_suppression_diagnostics_paper
Selective suppression diagnostics for score-quality under cap 0.020. These diagnostics are appendix-only and support interpretation of timing behavior and cap masking.

## Table_S2_precision_correlation_diagnostics_paper
Precision correlation diagnostics for the effective-precision signal under cap 0.020. These appendix diagnostics are secondary checks and are not used as standalone proof of HR gains.

## Table_S3_quality_only_cap020_ablation
Quality-only cap 0.020 ablation summary. Quality-only isolates score-quality without effective-precision; weak HR movement here should be reported rather than hidden.

## Table_S4_cap_selection_summary_paper
Cap-selection summary. Cap 0.020 is retained as a robust near-best final operating point; the table does not claim global optimality. The Wiki-CDN-2018 decision is a unified robust-cap decision, while the pressure-reduction decision applies only where shown by the table.
