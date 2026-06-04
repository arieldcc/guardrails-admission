# Claim-Driven Overhead Narrative

To isolate the marginal cost of the admission-stabilization layer, IL-no-guard is used as the primary overhead reference because it preserves the same IL admission pipeline without the Guardrails controller. The reported metric is slot-level control-path time under the one-slot delayed-apply protocol, not online request-serving latency.

Across three repeated runs per configuration, IL-guard shows bounded but workload-dependent measured control-path overhead relative to IL-no-guard. The observed mean slot-control delta ranges are +3.62 [+0.77, +13.25] for Wiki-CDN-2018 and -2.66 [-14.32, +6.68] for Wikipedia-2007 web. Negative relative values are interpreted as workload-dependent measured outcomes rather than evidence that Guardrails is intrinsically cheaper.

The component-level breakdown indicates that the guard/control component does not dominate the measured IL-guard pipeline. Across capacities, the guard/control share ranges are 5.02 [4.73, 5.23] for Wiki-CDN-2018 and 5.36 [4.63, 5.74] for Wikipedia-2007 web, computed over measured components only. This supports the interpretation that the additional stabilization logic contributes a limited share of the measured control path.

Tail behavior and memory are reported alongside the mean because average overhead alone can hide high-percentile control-path costs. The main table reports P95 and P99 slot-control deltas relative to IL-no-guard and peak RSS deltas where available. Delayed-GBDT is retained as a learned-baseline cost comparison, but its update/rebuild-dominated cost is interpreted separately from the marginal Guardrails controller cost.

All overhead values are descriptive over three repeated runs per dataset-method-capacity configuration; no statistical-significance claim is made.
