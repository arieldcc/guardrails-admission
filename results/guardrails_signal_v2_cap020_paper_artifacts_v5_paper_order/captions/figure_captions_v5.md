# Figure Captions v5

## Fig3_system_hit_ratio_comparison_paper
System-level hit-ratio comparison under post-warm-up trace-driven replay. IL-guard is compared with IL-no-guard, Delayed-GBDT, Windowed-2Hit-LRU, and Delayed-TinyLFU on Wikipedia-2007 web and Wiki-CDN-2018 using the same cache-size sweep and one-slot delayed-apply protocol.

## Fig4_admission_diagnostics_across_baselines_paper
Admission-side diagnostics across baselines. Panels summarize insertion pressure, pollution proxy, and hit-yield per admission over the same post-warm-up replay interval as Fig. 3. For learner-based admission variants, diagnostics are computed from recorded admission events. For baselines without explicit admission-event outcomes, proxy quantities are derived consistently from replay summaries and are used as contextual indicators. The intended conclusion is the best hit-ratio/admission-efficiency trade-off, not universal dominance on every diagnostic.

## Fig5_effective_precision_feedback_delta_paper
Effective-precision feedback attribution relative to a cap-only controller. Positive ΔHR and Δhit-yield together with reduced admission pressure or pollution indicate that lag-robust precision feedback improves admission efficiency rather than merely increasing insertions.

## Fig6_score_quality_timing_attribution_paper
Score-quality timing attribution under volume-preserving replay. Exact schedule replay reuses the realized slot-level admission budget of IL-guard, whereas permuted schedule replay preserves selected-admission volume but removes slot alignment. The supported claim is timing refinement, not direct HR improvement over the precision-only controller.

## FigS1_paired_bootstrap_support_paper
Paired slot-level bootstrap support. Panel (a) supports the score-quality timing comparison from the replay analysis, and panel (b) supports the effective-precision comparison against cap-only. Error bars show 95% bootstrap confidence intervals; this figure is supporting evidence after the mechanism figures, not the primary attribution by itself.

## FigS2_cap_sensitivity_selection_paper
Cap-sensitivity analysis for selecting the final top-percent cap. Cap 0.020 is marked as the final setting and is interpreted as a robust near-best operating point across workloads, not as a globally optimal cap. For Wiki-CDN-2018 it serves as a unified robust cap; pressure reduction is not asserted there relative to the best-HR cap.

## FigS3_selective_suppression_diagnostics_paper
Selective suppression diagnostics for score-quality. These appendix diagnostics show where score-quality actions are observable and where they are masked by the tight cap; they are not used as the main evidence for average-HR gains.

## FigS4_quality_only_cap020_ablation
Quality-only cap 0.020 appendix ablation. The figure shows what happens when score-quality is active without effective-precision feedback. Where quality-only is weak, the interpretation is explicit: score-quality alone is not the primary HR-driving component; its validated role is timing refinement within the final controller.
