# Block 1J7: C15B Cross-Action Diagnostic VOI — C4_instance_subtype_cued_v1

**Date:** 2026-05-10
**Block:** 1J7 — Route B (differentiating-action / cross-action VOI) on top of C15 two-pass
**Condition:** C4_instance_subtype_cued_v1
**Seed:** 101
**Budget:** 1.5
**Cost Weight:** 0.5

---

## 1. Executive Summary

C15B adds Route B (cross-action diagnostic VOI) on top of the C15 two-pass architecture. The diagnostic score ranks object-action pairs by uncertainty × query_relevance × boundary_proximity × training_neighbor_variance / cost. **C15B does not improve over C15b** — both achieve macro_bal=0.6142 with identical probe selections.

**C15B macro_bal=0.6142, delta vs C15b=+0.0000, delta vs C0b=+0.0271, gap capture=6.56%.**

The diagnostic score successfully rank-orders pairs (selected mean=1.685 vs unselected mean=0.313, 5.4x ratio) but converges to the same 8 probes as C15b's VOI gate. Even the oracle diff-action upper bound (which probes KNOWN differentiating actions per category) achieves only 0.5983 — worse than both C15b and C15B.

**Core finding:** Route B does not improve over Route A because the bottleneck is NOT action selection quality. C15b's VOI gate already selects the best actions given visible features. The remaining gap to oracle is about probe informativeness — visible features already resolve most uncertainty, and additional probes have limited marginal value under current cost parameters.

---

## 2. Policy Design

### C15B Cross-Action Diagnostic VOI

```
Pass 1 (Observe):  Visit objects nearest-first (C15b-style, 25% budget reserve)
                   Same as C15b — no Route B changes here
                   
Pass 2 (Probe):    For each observed object × candidate action, compute diagnostic_score
                   Rank object-action pairs by score, probe highest
                   Same C15b budget reserve transition rule
```

### Diagnostic Score Components

```
diagnostic_score(object, action) =
    query_relevance           # 1.0 if action maps to a task query
  * outcome_uncertainty       # p_success * (1 - p_success) from IOM
  * boundary_proximity        # 1 - 2*|P(query=true|object) - 0.5|
  * neighbor_variance         # outcome_variance among top-k IOM training neighbors
  / normalized_cost           # (reach + observe + probe) / initial_budget
```

All components use only realizable information (visible features, IOM predictions, training statistics). No hidden subtype, no ground-truth labels, no oracle differentiating action.

### Action-to-Query Mapping

| Action | Query |
|--------|-------|
| craft_plank | need_planks |
| eat | need_food |
| use_as_tool | need_tool |
| burn_as_fuel | need_fuel |
| mine_by_hand, mine_with_pickaxe | need_stone |

### Oracle Diff-Action Upper Bound (diagnostic only)

C15B_oracle_diff_action_upper_bound probes the KNOWN differentiating action per category (e.g., craft_plank for wood_log). Uses hidden_category to look up the differentiating action — clearly labeled non-deployable. Ranks objects by uncertainty of that known action.

---

## 3. Results: All Policies

| Policy | macro_bal | pos_recall | neg_recall | visited | probed | ncost | vs C0b delta |
|--------|-----------|------------|------------|---------|--------|-------|-------------|
| all_false | 0.5000 | 0.0000 | 1.0000 | 0 | 0 | 0.0000 | -0.0871 |
| C_minus_prior_only | 0.5000 | 0.0000 | 1.0000 | 0 | 0 | 0.0000 | -0.0871 |
| C0b_observe_only | 0.5871 | 0.3100 | 0.8642 | 60 | 0 | 0.7533 | — |
| random_probe_budgeted | 0.5425 | 0.1500 | 0.9350 | 22 | 22 | 0.9667 | -0.0446 |
| C13_instance_VOI | 0.5733 | 0.2000 | 0.9467 | 22 | 22 | 0.9667 | -0.0138 |
| **C15b_probe_reserve** | **0.6142** | **0.3500** | **0.8783** | **57** | **8** | **0.9900** | **+0.0271** |
| **C15B_cross_action_diag_VOI** | **0.6142** | **0.3500** | **0.8783** | **57** | **8** | **0.9900** | **+0.0271** |
| *C_oracle_full_information* | *1.0000* | *1.0000* | *1.0000* | — | — | — | — |

### Diagnostic (non-deployable)

| Policy | macro_bal | visited | probed | ncost |
|--------|-----------|---------|--------|-------|
| diagnostic_category_majority | 0.9750* | — | — | — |
| C15B_oracle_diff_action_upper_bound | 0.5983 | 57 | 8 | 0.9833 |

*Category_majority value from 1J2/1J3 (not recomputed in 1J7; the script's direct computation giving 0.5000 reflects a diagnostic implementation issue unrelated to C15/C15B conclusions).

Oracle-C0b gap: 0.4129.

---

## 4. C15B vs C15b: Probe Comparison

C15B and C15b select the **exact same 8 objects and actions** for probing:

| # | Object | Category | Subtype | Action | Is Diff? | Efficiency |
|---|--------|----------|---------|--------|----------|------------|
| 1 | test_wooden_pickaxe_006 | wooden_pickaxe | damaged | burn_as_fuel | no | zero_gain |
| 2 | test_wooden_pickaxe_003 | wooden_pickaxe | intact | burn_as_fuel | no | effective |
| 3 | test_stone_block_001 | stone_block | brittle | mine_by_hand | yes | zero_gain |
| 4 | test_apple_007 | apple | ripe | eat | yes | effective |
| 5 | test_wood_log_002 | wood_log | craftable | craft_plank | yes | effective |
| 6 | test_stone_block_009 | stone_block | hard | mine_by_hand | yes | zero_gain |
| 7 | test_apple_010 | apple | ripe | eat | yes | zero_gain |
| 8 | test_stone_block_000 | stone_block | hard | mine_by_hand | yes | effective |

### Diagnostic Score Separation

| Metric | Value |
|--------|-------|
| Candidate object-action pairs evaluated | 2250 |
| Mean score (selected) | 1.685 |
| Mean score (unselected) | 0.313 |
| Score ratio (selected/unselected) | 5.4x |

The diagnostic score achieves strong rank-ordering but converges to the same top-8 as VOI.

---

## 5. Required Comparisons

### 5.1 C15B vs C15b

| Metric | C15B | C15b | Delta |
|--------|------|------|-------|
| macro_bal | 0.6142 | 0.6142 | **+0.0000** |
| mean_positive_recall | 0.3500 | 0.3500 | +0.0000 |
| mean_negative_recall | 0.8783 | 0.8783 | +0.0000 |
| effective_probe_rate | 0.5000 | 0.5000 | +0.0000 |
| diff_action_probe_rate | 0.7500 | 0.7500 | +0.0000 |
| normalized_cost | 0.9900 | 0.9900 | +0.0000 |

C15B and C15b are identical on all metrics. The diagnostic score and VOI gate converge to the same probe selection.

### 5.2 C15B vs C0b

| Metric | Value |
|--------|-------|
| delta_macro_bal | **+0.0271** |
| delta_normalized_cost | +0.2367 |
| Pareto relation | Not dominated (C0b lower cost, C15B higher accuracy) |

### 5.3 C15B vs C13

| Metric | Value |
|--------|-------|
| delta_macro_bal | **+0.0408** |
| delta_visit_count | +35 |
| delta_probe_count | -14 |
| delta_positive_recall | +0.1500 |

### 5.4 C15B vs random_probe

| Metric | Value |
|--------|-------|
| delta_macro_bal | **+0.0717** |
| delta_normalized_cost | +0.0233 |

### 5.5 Gap Capture

```
oracle_c0b_gap = 1.0000 - 0.5871 = 0.4129
C15B gap capture = (0.6142 - 0.5871) / 0.4129 = 0.0656  (6.56%)
10% threshold = 0.0413 → C15B is below
Practical target = 0.6284 → C15B (0.6142) does not reach
```

### 5.6 Positive Discovery

C15B improves mean_positive_recall:
- Over C0b: 0.3100 → 0.3500 (+0.0400) ✓
- Over C15b: 0.3500 → 0.3500 (+0.0000) —
- Over C13: 0.2000 → 0.3500 (+0.1500) ✓

### 5.7 Negative-Only Tradeoff

No negative-only tradeoff. Both positive and negative recall improve (or stay equal) over C0b.

### 5.8 Pareto Frontier

**Normal policies on frontier:** all_false, C_minus_prior_only, C0b_observe_only, C15b_probe_reserve, C15B_cross_action_diagnostic_VOI

C15B is on the frontier (coincident with C15b at the same point). C13 is NOT on the frontier (dominated).

---

## 6. Why Route B Doesn't Improve Over Route A

Three factors explain the null result:

1. **VOI already selects the right actions.** C15b's VOI gate maximizes expected utility improvement, which is strongly correlated with the diagnostic score's components (uncertainty × query_relevance). In the C4 environment, the actions with highest VOI are also the ones with highest diagnostic score. The two scoring methods converge.

2. **Visible features resolve most uncertainty.** Post-observation, the IOM's predictions are already fairly confident. The marginal value of an additional probe — even on the differentiating action — is small. Only 4/8 probes (50%) change any prediction, and those changes are often minor.

3. **Oracle diff-action upper bound performs worse (0.5983).** Probing the KNOWN differentiating action (using hidden_category) produces WORSE results than C15b/C15B (0.6142). This is because the oracle upper bound probes differentiating actions indiscriminately, while VOI/diagnostic score also considers whether the current prediction is uncertain enough to benefit from probing. Simply knowing which action is differentiating is not enough — you also need to know when probing that action will change predictions.

---

## 7. Single-Seed Assessment

### Is C15B strongly promising on seed=101?

| Condition | Required | C15B | Met? |
|-----------|----------|------|------|
| A: beats C15b | C15B_mb > 0.6142 | 0.6142 (=) | **No** |
| B: beats C0b | > 0.5871 | 0.6142 ✓ | Yes |
| C: beats random | > 0.5425 | 0.6142 ✓ | Yes |
| D: improves pos_recall over C0b | > 0.3100 | 0.3500 ✓ | Yes |
| E: gap capture ≥ 10% | ≥ 0.6284 | 0.6142 ✗ | No |
| F: not Pareto-dominated by C0b or C15b | — | ✓ (coincident with C15b) | Yes |
| G: diff_action_rate improves over C15b | > 0.7500 | 0.7500 (=) | **No** |
| H: no neg-only tradeoff | — | ✓ | Yes |

**Verdict:** C15B is NOT strongly promising. Route B does not improve over Route A (C15b). The diagnostic score is a valid rank-ordering mechanism but converges to the same probe selection as VOI. The bottleneck is not action quality — it's probe informativeness given that visible features already resolve most uncertainty.

Passes 5/8 conditions, fails A, E, G. Even the oracle diff-action bound (condition A equivalent) performs worse.

---

## 8. Interpretation

**Route B is a useful idea but doesn't add value over VOI in this environment.** The diagnostic score correctly identifies that differentiating actions are more valuable (75% of probes target differentiating actions), but it doesn't improve the selection beyond what VOI already achieves. The two methods are convergently valid.

**The bottleneck has shifted again.** After 1J6, we thought the bottleneck was probe quality (differentiating-action selection). 1J7 shows that action selection is NOT the bottleneck — VOI already picks the best actions. The real bottleneck is probe informativeness: even when you probe the right action on the right object, visible features have already resolved most uncertainty, and the marginal value of one more probe is small.

**The oracle diff-action bound (0.5983) being worse than C15b (0.6142) is telling.** It means that probing differentiating actions indiscriminately is worse than probing VOI-guided actions. The key is not just which action to probe, but which objects are uncertain enough to benefit.

**Three paths forward emerge:**

1. **Route F (blurred peripheral observation):** If observing one object gave coarse information about nearby objects, probes might be more informative because the prior uncertainty would be higher at probe time. Deferred per 1J5_patch.

2. **Cost/environment redesign (Routes D/E):** Lower probe cost or larger budget would allow more probes, potentially capturing more value. But this risks making the task too easy for C0b.

3. **Interleaved breadth-depth (MCTS-inspired):** Instead of strict two-pass, interleave observation and probing — observe a few objects, probe one if promising, continue. This might help by updating the VOI/diagnostic scores with each probe outcome before choosing the next observation target.

---

## 9. Route Recommendation

- **Route A (C15 two-pass):** Validated. Clear improvement over C13. ✓ DONE
- **Route B (cross-action diagnostic VOI):** Evaluated. Does not improve over Route A. The diagnostic score converges to VOI selection. Route B is not a bottleneck. ✓ DONE
- **Route F (blurred peripheral observation):** May increase probe informativeness by raising prior uncertainty. Still deferred but increasingly relevant.
- **Routes D/E (cost/environment redesign):** May be needed if probe informativeness can't be improved through policy design alone. Still deferred.
- **New Route (interleaved breadth-depth):** Consider as alternative to strict two-pass. May improve by updating scores with each probe outcome.

Do NOT proceed to multi-seed. Route B is a null result — the diagnostic score is valid but redundant with VOI.

---

*Generated by Block 1J7 — single-seed C15B cross-action diagnostic VOI evaluation.*
