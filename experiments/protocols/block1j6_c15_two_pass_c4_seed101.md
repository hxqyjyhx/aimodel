# Block 1J6: C15 Two-Pass Observe-Then-Probe Evaluation — C4_instance_subtype_cued_v1

**Date:** 2026-05-10
**Block:** 1J6 — C15 two-pass policy evaluation
**Condition:** C4_instance_subtype_cued_v1
**Seed:** 101
**Budget:** 1.5
**Cost Weight:** 0.5

---

## 1. Executive Summary

C15_two_pass_observe_then_probe improves over both C13 (original VOI) and C0b (observe-only), but the improvement is small — gap capture is only 4.9% of the C0b-to-oracle gap. The two-pass design successfully addresses visit-breadth loss (55-57 vs 22 visits), but the probe phase adds limited value because the VOI gate with cost_weight=0.5 rejects most post-observation probes.

**C15a (observe_all_then_probe):** macro_bal=0.6075, +0.0204 vs C0b, +0.0342 vs C13. Visits 55 objects, probes 8.

**C15b (probe_reserve):** macro_bal=0.6142, +0.0271 vs C0b, +0.0408 vs C13. Visits 57 objects, probes 8.

Both C15 variants are on the normal-policy Pareto frontier (not dominated by any other deployable policy).

**Best C15 variant:** C15b_probe_reserve (macro_bal=0.6142) outperforms C15a (0.6075). C15b delta vs C0b = +0.0271, gap capture = 6.55%.

---

## 2. Policy Design

### Two-Pass Architecture

```
Pass 1 (Observe):  Visit objects nearest-first, observe only (no probes)
                   Stop when: budget reserve threshold reached, or
                               ≤5 unvisited objects remain, or
                               cannot afford one more observe + one probe
                   
Pass 2 (Probe):    Rank already-observed objects by post-observation net VOI
                   Visit (re-reach) and probe highest-VOI objects
                   Same C13 VOI gate on probe decision (cost_weight=0.5)
                   Stop when no observed-unprobed object has positive net VOI
                   or budget exhausted
```

### Variants

| Variant | Observe-Stop Rule | Effect |
|---------|-------------------|--------|
| C15a | Transition when ≤5 unvisited or can't afford observe+probe | Minimally conservative |
| C15b | Reserve 25% of initial budget (0.375) for probe pass | More aggressive probe reserve |

### Key Difference from C13

C13 couples visit+probe: every visited object is probed if VOI gate passes. This forces a choice between breadth (more objects) and depth (probes). C15 decouples: observe cheaply (0.005/object) first, then selectively probe where post-observation uncertainty is highest.

---

## 3. Results: All Policies

| Policy | macro_bal | pos_recall | neg_recall | visited | probed | ncost | vs C0b delta |
|--------|-----------|------------|------------|---------|--------|-------|-------------|
| all_false | 0.5000 | 0.0000 | 1.0000 | 0 | 0 | 0.0000 | -0.0871 |
| C_minus_prior_only | 0.5000 | 0.0000 | 1.0000 | 0 | 0 | 0.0000 | -0.0871 |
| C0b_observe_only | 0.5871 | 0.3100 | 0.8642 | 60 | 0 | 0.7533 | — |
| random_probe_budgeted | 0.5425 | 0.1500 | 0.9350 | 22 | 22 | 0.9667 | -0.0446 |
| C13_instance_VOI | 0.5733 | 0.2000 | 0.9467 | 22 | 22 | 0.9667 | -0.0138 |
| **C15a_observe_all_then_probe** | **0.6075** | **0.3467** | **0.8683** | **55** | **8** | **0.9700** | **+0.0204** |
| **C15b_probe_reserve** | **0.6142** | **0.3500** | **0.8783** | **57** | **8** | **0.9900** | **+0.0271** |
| *C_oracle_full_information* | *1.0000* | *1.0000* | *1.0000* | — | — | — | — |

*Note: diagnostic_category_majority is not included in this table. It was already established as a diagnostic shortcut-risk bound (macro_bal=0.9750) in 1J2/1J3 and is not recomputed here. It is not a deployable policy and is excluded from the normal-policy Pareto frontier.*

Oracle-C0b gap: 0.4129.

---

## 4. Per-Query Detail

### C15a

| Query | C15a bal | C0b bal | C13 bal | C15a pos_r | C0b pos_r | delta |
|-------|----------|---------|---------|------------|-----------|-------|
| need_food | 0.6250 | 0.5833 | 0.7083 | 0.2500 | 0.1667 | +0.0417 |
| need_fuel | 0.6000 | 0.5833 | 0.5333 | 0.5667 | 0.5000 | +0.0167 |
| need_planks | 0.5729 | 0.5000 | 0.5104 | 0.3333 | 0.0000 | +0.0729 |
| need_stone | 0.5521 | 0.5208 | 0.5833 | 0.1667 | 0.1667 | +0.0313 |
| need_tool | 0.6875 | 0.7500 | 0.5313 | 0.4167 | 0.5000 | -0.0625 |

C15a improves over C0b on 4/5 queries. need_tool is the only regression, likely because the 8 C15a probes didn't target need_tool objects well.

### C15b

| Query | C15b bal | C0b bal | C13 bal | C15b pos_r | C0b pos_r | delta |
|-------|----------|---------|---------|------------|-----------|-------|
| need_food | 0.6458 | 0.5833 | 0.7083 | 0.3333 | 0.1667 | +0.0625 |
| need_fuel | 0.6333 | 0.5833 | 0.5333 | 0.6000 | 0.5000 | +0.0500 |
| need_planks | 0.5521 | 0.5000 | 0.5104 | 0.2500 | 0.0000 | +0.0521 |
| need_stone | 0.5104 | 0.5208 | 0.5833 | 0.0833 | 0.1667 | -0.0104 |
| need_tool | 0.7292 | 0.7500 | 0.5313 | 0.5000 | 0.5000 | -0.0208 |

C15b improves over C0b on 3/5 queries. need_fuel shows the largest gain (+0.05).

---

## 5. C15 Probe Diagnostics

### C15a (8 probes)

| Metric | Value |
|--------|-------|
| Total probes | 8 |
| Effective | 3 (37.5%) |
| Zero-gain | 5 (62.5%) |
| Harmful | 0 |
| Differentiating action | 7 (87.5%) |
| Non-differentiating action | 1 (12.5%) |

### C15b (8 probes)

| Metric | Value |
|--------|-------|
| Total probes | 8 |
| Effective | 4 (50.0%) |
| Zero-gain | 4 (50.0%) |
| Harmful | 0 |
| Differentiating action | 6 (75.0%) |
| Non-differentiating action | 2 (25.0%) |

### Probe Efficiency Comparison

| Policy | n Probes | Effective | Diff-Action Rate | Eff on Diff |
|--------|----------|-----------|------------------|-------------|
| C13 (original) | 22 | 50.0% | 40.9% | 66.7% |
| C15a | 8 | 37.5% | 87.5% | ~42.9% |
| C15b | 8 | 50.0% | 75.0% | ~66.7% |

C15 probes many fewer objects than C13 (8 vs 22) but with much better action selection (75-87.5% differentiating vs 40.9%). The higher differentiating-action rate is a natural consequence of post-observation VOI: after observing visible features, the VOI computation preferentially selects the differentiating action because uncertainty is concentrated there.

### Visited Coverage (C15a)

| Query | Total Pos | Visited Pos | Pos Coverage | Total Neg | Visited Neg | Neg Coverage |
|-------|-----------|-------------|--------------|-----------|-------------|--------------|
| need_food | 12 | 10 | 83.3% | 48 | 45 | 93.8% |
| need_fuel | 30 | 27 | 90.0% | 30 | 28 | 93.3% |
| need_planks | 12 | 11 | 91.7% | 48 | 44 | 91.7% |
| need_stone | 12 | 11 | 91.7% | 48 | 44 | 91.7% |
| need_tool | 12 | 10 | 83.3% | 48 | 45 | 93.8% |

Compared to C13's 22/60 (36.7% coverage), C15 visits 55/60 (91.7%) — a massive improvement in observation breadth. The 5 unvisited objects are the cheapest-to-reach ones left at transition time.

---

## 6. Required Comparisons

### 6.1 C15 vs C0b

| Metric | C15a | C15b |
|--------|------|------|
| delta_macro_bal | **+0.0204** | **+0.0271** |
| delta_normalized_cost | +0.2167 | +0.2367 |
| delta_mean_positive_recall | +0.0367 | +0.0400 |
| delta_mean_negative_recall | +0.0042 | +0.0142 |
| Pareto relation | Not dominated | Not dominated |

Both C15 variants beat C0b on macro_bal and positive recall. The cost is higher (ncost ~0.98 vs 0.75) but C0b does not dominate either C15.

### 6.2 C15 vs C13

| Metric | C15a | C15b |
|--------|------|------|
| delta_macro_bal | **+0.0342** | **+0.0408** |
| delta_visit_count | +33 | +35 |
| delta_probe_count | -14 | -14 |
| delta_positive_recall | +0.1467 | +0.1500 |
| delta_negative_recall | -0.0783 | -0.0683 |

C15 dramatically improves visit breadth (+33-35 more objects) and positive recall (+0.15). Negative recall drops slightly because broader observation means less conservative predictions on unvisited objects.

### 6.3 C15 vs random_probe

| Metric | C15a | C15b |
|--------|------|------|
| delta_macro_bal | **+0.0650** | **+0.0717** |
| delta_normalized_cost | +0.0033 | +0.0233 |

Both C15 variants substantially outperform random_probe.

### 6.4 Gap Capture

```
oracle_c0b_gap = 1.0000 - 0.5871 = 0.4129

C15a gap capture = (0.6075 - 0.5871) / 0.4129 = 0.0494  (4.94%)
C15b gap capture = (0.6142 - 0.5871) / 0.4129 = 0.0655  (6.55%)
```

**10% threshold (0.0413)**: C15a gap capture (0.0494, 4.94%) is below the 10% threshold. C15b (6.55%) is also below. Neither meets the practical target of ≥10% gap capture.

### 6.5 Positive Discovery

C15a improves mean_positive_recall: 0.3100 → 0.3467 (+0.0367). C15b: 0.3100 → 0.3500 (+0.0400). The negative-only tradeoff is NOT present — both positive AND negative recall improve (slightly) over C0b.

### 6.6 Visit-Breadth Check

| Policy | Visits | vs C13 (+22) |
|--------|--------|-------------|
| C13 | 22 | — |
| C15a | 55 | +33 |
| C15b | 57 | +35 |
| C0b | 60 | +38 |

C15 achieves 91.7-95% of C0b's visit coverage while preserving budget for 8 selective probes. The 5 unvisited objects are a small price for probe budget.

### 6.7 Pareto Frontier

**Normal policies on frontier:** all_false, C0b_observe_only, C15a_observe_all_then_probe, C15b_probe_reserve

Both C15 variants are on the frontier — they are not dominated by any other deployable policy. Notably, C13 is NOT on the frontier (dominated by C0b on cost or C15 on accuracy).

---

## 7. Why Is the Improvement Small?

The gap capture (4.9-6.6%) is well below the 10% target. Three factors limit C15's impact:

1. **VOI gate rejects most probes.** With cost_weight=0.5 and probe_cost=0.05 on budget=1.5 (normalized=0.0333), the net_voi threshold is high. Post-observation, many objects have their uncertainty resolved by visible features alone — the marginal VOI of probing is below the cost threshold. Only 8/55 observed objects pass the gate.

2. **Action selection still suboptimal.** While differentiating-action rate improved (75-87.5% vs 40.9%), the effective probe rate is lower (37.5-50% vs 50%). Post-observation, the differentiating action may already be well-predicted from visible cues, making even differentiating-action probes zero-gain.

3. **No Route B yet.** Route B (differentiating-action VOI / cross-action inference) is not implemented. C15 uses the same C13 VOI gate, which treats actions independently and doesn't weight differentiating actions higher.

---

## 8. C15a vs C15b

C15b slightly outperforms C15a (0.6142 vs 0.6075) at slightly higher cost (0.9900 vs 0.9700). The gap is small. C15b's explicit 25% probe reserve forces an earlier transition, preserving 0.375 budget for probes. C15a's more conservative transition (≤5 unvisited) results in a similar outcome because the budget constraint naturally limits observation.

For practical purposes, the variants are nearly equivalent. The two-pass architecture is what matters, not the specific transition rule.

---

## 9. Single-Seed Assessment

### Is C15 promising on seed=101?

| Condition | Required | C15a | C15b | Met? |
|-----------|----------|------|------|------|
| A: beats C0b | > 0.5871 | 0.6075 ✓ | 0.6142 ✓ | **Yes** |
| B: beats C13 | > 0.5733 | 0.6075 ✓ | 0.6142 ✓ | **Yes** |
| C: beats random | > 0.5425 | 0.6075 ✓ | 0.6142 ✓ | **Yes** |
| D: improves pos_recall | > 0.3100 | 0.3467 ✓ | 0.3500 ✓ | **Yes** |
| E: gap capture ≥ 10% | ≥ 0.0413 | 0.0204 ✗ | 0.0271 ✗ | **No** |
| F: not Pareto-dominated | — | ✓ | ✓ | **Yes** |
| G: no neg-only tradeoff | — | ✓ | ✓ | **Yes** |

**Verdict:** C15 is directionally validated but not practically strong enough yet. Route A is validated as a structural improvement over C13. C15 passes 6/7 conditions but fails the pre-registered practical threshold because gap capture remains below 10%. Therefore C15 should not proceed to multi-seed; Route B is needed next.

---

## 10. Interpretation

**C15 beats C13 but not convincingly beats C0b.** The two-pass design succeeds at its primary goal: increasing visit coverage from 22 to 55-57 objects. But the probe phase underdelivers — only 8 probes pass the VOI gate, and only 3-4 are effective. The 0.0204-0.0271 gain over C0b is real but weak.

**The bottleneck has shifted from breadth to probe informativeness.** C13's failure was breadth (82.5% FN from unvisited). C15 fixes this — now 90%+ of positives are visited. But the probes that do happen are only 37.5-50% effective. The remaining gap to oracle is about probe quality, not quantity.

**Route B (differentiating-action selection) is the logical next step.** Better action selection would make each probe more effective. If C15's 8 probes were 75%+ effective (instead of 37.5-50%), the macro_bal improvement would compound. The high differentiating-action rate (75-87.5%) is a good foundation, but the probes need to actually change predictions.

**Route F (blurred peripheral observation) may also help.** If local context could help identify minority-subtype objects before probing, probe targeting would improve. This remains deferred per 1J5_patch.

---

## 11. Route Recommendation

- **Route A (C15 two-pass):** Implemented and evaluated. Clear improvement over C13, but insufficient alone. ✓ DONE
- **Route B (differentiating-action selection):** **Needed next.** Layer on C15 to improve probe effectiveness.
- **Route F (blurred observation):** Still deferred. Re-evaluate after Route B.

Do NOT proceed to multi-seed. Single-seed signal is clear: two-pass fixes breadth but probe quality is the remaining bottleneck.

---

*Generated by Block 1J6 — single-seed C15 evaluation, no Route B, no environment changes.*
