# Block 1H Checkpoint: Pareto Evaluation Protocol

**Date:** 2026-05-10
**Blocks covered:** 1H0, 1H1, 1H2
**Status:** Checkpoint — scalar utility not viable; Pareto reporting adopted.

---

## 1. Accepted Findings

1. **Raw composite accuracy is majority-class dominated.**
   - All policies score raw_comp >= 0.6933 because "always false" scores 0.7000 on a 25/75 pos/neg split for 4 of 5 queries.
   - Block 1H0 confirmed: all_false and iom_prior both score raw_comp=0.7000 with macro_bal=0.5000.

2. **macro_query_balanced_accuracy is now the primary task metric.**
   - Defined as mean over 5 queries of 0.5*(positive_recall + negative_recall).
   - Neutralizes the majority-class artifact. all_false = 0.5000, oracle = 1.0000.

3. **C13 has weak but real macro_bal gain over C0b.**
   - C13 macro_bal = 0.5578 vs C0b macro_bal = 0.5444, delta = +0.0133.
   - C13 macro_bal is slightly higher than C0b (+0.0133), but this gain is not driven by improved positive recall.
   - C13 positive recall sum is lower than C0b (0.733 vs 1.033).
   - C13 appears to trade positive recall for higher negative recall.
   - Therefore, C13 has a weak asymmetric signal, not successful positive-affordance discovery.

4. **C13 is not cost-effective under current cost scale.**
   - C13 ncost = 0.9667, C0b ncost = 0.7533. Cost delta = +0.2133.
   - C13 bal_ub25 = 0.3161 vs C0b bal_ub25 = 0.3561, delta = -0.0400.
   - C13 would need beta <= 0.0598 (23.9% of current 0.25) to break even vs all_false.

5. **stop_on_skip is a metric exploit.**
   - macro_bal = 0.4933 (below chance), ncost = 0.0533 (near-zero).
   - bal_ub25 = 0.4800, highest of any interactive policy — but achieved by doing almost nothing.
   - Dominated by all_false and iom_prior on Pareto criteria (higher macro_bal AND lower cost).
   - Violates "no fallback to chance-level" constraint.

6. **Current scalar utility (bal_ub25) is not policy-comparison-ready.**
   - No interactive policy beats all_false at beta=0.25.
   - The signal-to-cost ratio (~0.24 for C13) is far below 1.0.
   - A single beta cannot simultaneously make interaction positive-sum AND discriminate between policies.

---

## 2. Invalidated Claims

| Invalidated Claim | Rationale |
|-------------------|-----------|
| "C13 wins based on raw composite accuracy" | C13 raw-comp gain is not sufficient evidence of policy success. Balanced metrics show only a small macro_bal gain, while positive recall decreases. |
| "stop_on_skip is a valid policy" | macro_bal below 0.5, dominated by all_false and iom_prior, achieves high bal_ub25 through cost avoidance not prediction quality. |
| "distance_filter fixes C13" | filter macro_bal = 0.5533 < C13 macro_bal = 0.5578 at same cost. filter is Pareto-dominated by C13. |
| "beta=0.25 supports interactive policy ranking" | Would need beta <= 0.0625 for C13 to beat C0b, or beta <= 0.0598 to beat all_false. Current beta overshoots by ~4x. |

---

## 3. Pareto Evaluation Protocol

### Per-Policy Reporting Standard

Every policy report must include:

| Metric | Definition |
|--------|-----------|
| `raw_comp` | Raw composite task accuracy (majority-biased, secondary only) |
| `macro_query_balanced_accuracy` | Mean[0.5*(pos_recall + neg_recall)] over 5 queries — **primary** |
| `normalized_cost` | total_cost / max_possible_cost |
| `probe_count` | Number of probe actions executed |
| `positive_recall_per_query` | Per-query positive recall (5 values) |
| `negative_recall_per_query` | Per-query negative recall (5 values) |
| `balanced_utility_beta0.25` | macro_bal - 0.25*ncost — **secondary only, not for ranking** |

### Primary Comparison Rule: Pareto Dominance

Policy A **dominates** policy B if and only if:

```
macro_bal_A >= macro_bal_B
normalized_cost_A <= normalized_cost_B
at least one inequality is strict
```

If neither dominates the other, report the trade-off explicitly rather than forcing a scalar ranking.

### Reporting Template

```
Policy X vs Policy Y:
  macro_bal:  X=0.xxx  Y=0.xxx  delta=+/-0.xxx
  ncost:      X=0.xxx  Y=0.xxx  delta=+/-0.xxx
  Result: [X dominates Y | Y dominates X | trade-off — no dominance]
  If trade-off: X offers [+0.xxx macro_bal] at cost of [+0.xxx ncost]
```

---

## 4. Policy Status Table

All values from Block 1H1 (seed=101, budget=1.5, C3_P060_O040).

| Policy | macro_bal | ncost | bal_ub25 | Dominated By | Status |
|--------|-----------|-------|----------|-------------|--------|
| `all_false` | 0.5000 | 0.0000 | 0.5000 | oracle | baseline — duplicate zero-cost chance baseline (same point as iom_prior) |
| `iom_prior` | 0.5000 | 0.0000 | 0.5000 | oracle | baseline — duplicate zero-cost chance baseline (same point as all_false) |
| `c0b` | 0.5444 | 0.7533 | 0.3561 | oracle | Pareto-optimal (observation-only frontier) |
| `c13` | 0.5578 | 0.9667 | 0.3161 | oracle | Pareto-optimal (probe frontier; dominates filter/aware) |
| `c13_distance_filter` | 0.5533 | 0.9667 | 0.3117 | c13, oracle | dominated — same cost, lower macro_bal than C13 |
| `c13_distance_aware_selection` | 0.5533 | 0.9667 | 0.3117 | c13, oracle | dominated — identical to distance_filter |
| `stop_on_skip` | 0.4933 | 0.0533 | 0.4800 | all_false, iom_prior, oracle | dominated — metric exploit; macro_bal below chance |
| `oracle_unconstrained` | 1.0000 | 0.0000 | 1.0000 | none | undominated upper bound |

### Pareto Frontier (interactive policies only)

```
ncost
1.0 |                                    * c13 (0.5578, 0.9667)
    |                       * distance_filter (0.5533, 0.9667) [dominated]
0.8 |
    |               * c0b (0.5444, 0.7533)
0.6 |
    |
0.4 |
    |
0.2 |
    |   * stop_on_skip (0.4933, 0.0533) [dominated]
0.0 | * all_false / iom_prior (0.5000, 0.0000)
    +---------------------------------------------------- macro_bal
    0.49   0.50   0.51   0.52   0.53   0.54   0.55   0.56
```

### Per-Query Recall Summary

| Policy | need_planks pos/neg | need_stone pos/neg | need_food pos/neg | need_tool pos/neg | need_fuel pos/neg |
|--------|--------------------|--------------------|--------------------|--------------------|--------------------|
| all_false | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 |
| c0b | 0.13 / 0.96 | 0.13 / 0.96 | 0.33 / 0.93 | 0.00 / 0.93 | 0.43 / 0.63 |
| c13 | 0.20 / 1.00 | 0.20 / 0.98 | 0.13 / 1.00 | 0.07 / 1.00 | 0.13 / 0.87 |
| distance_filter | 0.13 / 0.98 | 0.13 / 0.98 | 0.20 / 1.00 | 0.13 / 0.98 | 0.23 / 0.77 |
| stop_on_skip | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 0.93 |
| oracle | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

---

## 5. Recommended Next Research Direction

**Primary recommendation: C — Report macro_bal/cost Pareto without scalar utility.**

Rationale:
- Options A (environment redesign) and B (cost redesign) are premature. We have only established that the *scalar utility* doesn't work — not whether the environment or cost scale is fundamentally wrong.
- Option C allows continued policy evaluation using the Pareto protocol defined above. This is immediately actionable and doesn't require environment changes.
- Option D (revisit C13 later) can be done in parallel — the Pareto framework works for comparing any policies.

**Secondary: A subset of A — investigate why macro_bal gain ceiling is ~0.058.**
- The oracle gap (1.000 - 0.558 = 0.442) shows there is signal the IOM+VOI mechanism could theoretically capture.
- C13 only probes 22/60 objects (37%) and gets 11/22 effective probes (50%).
- Understanding what limits probe effectiveness may reveal whether environment redesign is needed.

---

## 6. Final Checkpoint Decision

| Flag | Value | Rationale |
|------|-------|-----------|
| `policy_comparison_ready` | **false** | No interactive policy beats all_false under scalar utility; Pareto comparison is the fallback |
| `scalar_utility_ready` | **false** | beta=0.25 overshoots achievable gain by ~4x; no single beta works |
| `balanced_metric_ready` | **true** | macro_query_balanced_accuracy correctly neutralizes majority-class bias |
| `pareto_reporting_ready` | **true** | Pareto dominance is well-defined, immediately computable, and identifies dominated policies |

### Decision

**All future Mini-MC v0 policy evaluation SHALL use Pareto (macro_bal, ncost) reporting as the primary comparison framework.** scalar utility (bal_ub25) is demoted to secondary-only. No single beta should be used to rank policies.

This protocol applies to:
- All existing Block 1F/1G/1H policy comparisons
- Any future policy variants
- Cross-condition comparisons

---

*Generated by Block 1H3 from data in Blocks 1H0, 1H1, and 1H2.*
