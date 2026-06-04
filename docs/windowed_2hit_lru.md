# Windowed-2Hit-LRU

Windowed-2Hit-LRU is a standard rule-based selective-admission baseline. It is
not a learner baseline and is not included in learner-overhead analysis.

The baseline adapts the window-based cache-on-second-request principle from
Carlsson and Eager, "Worst-case Bounds and Optimized Cache on Mth Request Cache
Insertion Policies under Elastic Conditions" (Performance Evaluation, 2018).
The adapted setting uses `M = 2`: an uncached object is admissible only if it has
another uncached request within a recent request window.

The original `W = T = R` cost-equivalence setting cannot be directly
instantiated in this repository because the experiments use fixed-size
hit-rate replay, not elastic storage/bandwidth cost. The primary adapted
setting therefore anchors the window to the Guardrails slot granularity:

```text
window_slots = 1
window_requests = 100,000 requests under the main configuration
```

Eviction remains fixed LRU for fairness with the Guardrails experiments.
Admissions use the same one-slot delayed-apply protocol: objects satisfying the
Windowed-2Hit rule in slot `t` are applied at the beginning of slot `t + 1`.

This baseline does not use features, top-20% labels, model scores, training,
future information, Guardrails feedback, or score-quality modulation.
