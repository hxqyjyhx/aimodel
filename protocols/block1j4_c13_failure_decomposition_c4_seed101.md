# Block 1J4: C13 Failure Decomposition — C4_instance_subtype_cued_v1

**Date:** 2026-05-10
**Block:** 1J4 — C13 failure decomposition
**Condition:** C4_instance_subtype_cued_v1
**Seed:** 101
**Budget:** 1.5
**C13 Cost Weight:** 0.5

---

## 1. Executive Summary

C13_instance_VOI (CW=0.5) loses to C0b_observe_only (macro_bal 0.5733 vs 0.5871) despite spending more budget (ncost 0.9667 vs 0.7533). This block decomposes the failure into 8 diagnostic dimensions.

**Primary finding: C13 performs BETTER than C0b on objects it actually visits (macro_bal 0.7163 vs 0.5570 on the C13-visited subset), but this advantage cannot compensate for the 38 unvisited objects.** The failure is overwhelmingly a visit-breadth problem — 82.5% of C13's false negatives come from objects it never reaches. Secondary failures include poor differentiating-action selection (only 40.9% of probes target the subtype-differentiating action), poor minority-subtype targeting, and the persistent negative-only tradeoff.

**Final diagnosis:** `visit_breadth_loss + poor_differentiating_action_selection + poor_minority_subtype_targeting + persistent_negative_only_tradeoff + environment_category_shortcut_risk`

---

## 2. C13 vs C0b Failure Recap

| Metric | C13 | C0b | Delta |
|--------|-----|-----|-------|
| macro_bal | 0.5733 | 0.5871 | **-0.0138** |
| normalized_cost | 0.9667 | 0.7533 | +0.2133 |
| mean_pos_recall | 0.2000 | 0.3100 | -0.1100 |
| mean_neg_recall | 0.9467 | 0.8642 | +0.0825 |
| visits | 22 | 60 | -38 |
| probes | 22 | 0 | +22 |

C13 is Pareto-dominated by C0b: lower accuracy at higher cost. The question is **why**.

---

## 3. Diagnostic 1: Visit-Breadth Loss

### Summary

| Metric | Value |
|--------|-------|
| C13 visit count | 22 / 60 |
| C0b visit count | 60 / 60 |
| Total positives across 5 queries | 78 |
| C13 visited positives | 26 (33.3%) |
| C13 max possible pos_recall (from visits) | 0.3333 |
| C0b actual pos_recall | 0.3100 |
| pos_recall loss from missed visits | -0.0233 (negligible at aggregate) |

C13's visit ceiling (0.3333 max pos_recall) is barely above C0b's actual pos_recall (0.3100). Even with perfect predictions on visited objects, C13 would only achieve roughly C0b-level positive recall.

### Per-Query Coverage

| Query | Total Pos | C13 Visited Pos | Pos Coverage | C13 Visited Neg | Neg Coverage |
|-------|-----------|-----------------|--------------|-----------------|--------------|
| need_food | 12 | 9 | **75.0%** | 13 | 27.1% |
| need_fuel | 30 | 6 | **20.0%** | 16 | 53.3% |
| need_planks | 12 | 4 | **33.3%** | 18 | 37.5% |
| need_stone | 12 | 5 | **41.7%** | 17 | 35.4% |
| need_tool | 12 | 2 | **16.7%** | 20 | 41.7% |

**need_fuel is the hardest hit**: 24/30 positive objects are never visited. need_tool also suffers severely with only 2/12 positives visited.

### How much positive recall loss is due to not visiting?

C13 visits 26/78 total positive objects across queries. Even with perfect predictions, C13's pos_recall ceiling is **0.3333**. Since C0b already achieves 0.3100 by visiting all 60 objects, C13 cannot meaningfully exceed C0b on positive recall through visit-breadth alone — the observation-only signal on the 38 unvisited objects provides more value than the probe information on the 22 visited objects.

**Conclusion: visit_breadth_loss is confirmed as the primary failure mechanism.** C13 sacrifices too many observations for too few probes.

---

## 4. Diagnostic 2: Visited-Object Prediction Comparison

Restricted to the 22 objects C13 visited:

| Metric | C13 (visited) | C0b (same subset) | random (same subset) |
|--------|---------------|-------------------|----------------------|
| macro_bal | **0.7163** | 0.5570 | 0.6522 |
| mean_pos_recall | **0.5578** | 0.3011 | 0.4457 |

### Per-Query Detail (C13-visited subset)

| Query | C13 bal | C0b bal | C13 pos_r | C0b pos_r | C13 better? |
|-------|---------|---------|-----------|-----------|-------------|
| need_food | 0.7778 | 0.6111 | 0.5556 | 0.2222 | **Yes (+0.1667)** |
| need_fuel | 0.8229 | 0.6979 | 0.8333 | 0.8333 | **Yes (+0.1250)** |
| need_planks | 0.5556 | 0.4306 | 0.5000 | 0.2500 | **Yes (+0.1250)** |
| need_stone | 0.7000 | 0.5706 | 0.4000 | 0.2000 | **Yes (+0.1294)** |
| need_tool | 0.7250 | 0.4750 | 0.5000 | 0.0000 | **Yes (+0.2500)** |

**C13 outperforms C0b on every single query when restricted to visited objects.** The probing actually works — C13's predictions are substantially better than C0b's on the same objects. C0b's advantage comes entirely from seeing all 60 objects.

**Conclusion: posterior_prediction_bias is NOT supported.** C13 does not have worse predictions on visited objects. The failure is breadth, not quality.

---

## 5. Diagnostic 3: Probe Action Relevance

### Aggregate

| Metric | Count | Rate |
|--------|-------|------|
| Total probes | 22 | 1.000 |
| Differentiating action probes | 9 | **0.4091** |
| Non-differentiating action probes | 13 | **0.5909** |

### Efficiency by Probe Type

| Probe Type | n | Effective | Zero-Gain | Harmful | Eff Rate |
|------------|---|-----------|-----------|---------|----------|
| Differentiating action | 9 | 6 | 3 | 0 | **0.6667** |
| Non-differentiating action | 13 | 5 | 8 | 0 | **0.3846** |

**59.1% of C13's probes target non-differentiating actions.** These probes test actions that are identical between subtypes for the object's category (e.g., probing `burn_as_fuel` on an apple, which always fails regardless of subtype). Among non-differentiating probes, 61.5% are zero-gain — they provide no accuracy improvement because the outcome was already predictable.

When C13 does probe the differentiating action (41% of probes), it achieves 66.7% effectiveness — much better than the non-differentiating baseline. However, 33.3% of differentiating-action probes are still zero-gain, typically because the object is the majority subtype and observation already pushed the prediction in the correct direction.

### Why Does C13 Choose Non-Differentiating Actions?

C13's VOI gate evaluates: which action, if probed, maximizes expected utility gain? For many objects, the predicted probability for the differentiating feature may already be confidently near 0.5 or observation may not strongly differentiate subtypes. In these cases, probing a non-differentiating action with a confident prediction (e.g., `burn_as_fuel` on apple, which always fails) may show a small VOI signal because:
1. The current confidence for that action is already high (e.g., 0.95)
2. A probe that confirms the expected outcome slightly increases confidence
3. The VOI computation sees a small positive gain

This is a structural limitation: VOI over individual actions doesn't know which actions are diagnostic for subtype differentiation. It treats all actions as equally informative about their own outcomes, not about cross-action inference.

**Conclusion: poor_differentiating_action_selection is confirmed.** Over half of C13's probe budget is spent on actions that cannot distinguish subtypes.

---

## 6. Diagnostic 4: Minority-Subtype Targeting

### Aggregate

| Metric | Majority | Minority | Ratio |
|--------|----------|----------|-------|
| Total objects | 48 | 12 | 4:1 |
| C13 visited | 20 | 2 | 10:1 |
| C13 probed | 20 | 2 | 10:1 |
| Visit rate | 41.7% | 16.7% | 2.5:1 |
| Probe rate (of visited) | 100% | 100% | 1:1 |
| **Minority enrichment** | — | — | **1.00** |

C13 visits majority-subtype objects at 2.5x the rate of minority-subtype objects. This is expected given the 4:1 ratio in the population, but the minority visit rate (16.7%) is alarmingly low — only 2/12 minority objects are ever seen. Since C13 probes every visited object (both rates are 100%), the minority enrichment ratio is exactly 1.0: **no preferential targeting of minority objects at all.**

### Per-Category

| Category | n Maj | n Min | Vis Maj | Vis Min | Maj Visit% | Min Visit% | Enrichment |
|----------|-------|-------|---------|---------|------------|------------|------------|
| wood_log | 12 | 3 | 4 | **0** | 33.3% | **0.0%** | 0.00 |
| stone_block | 12 | 3 | 5 | 1 | 41.7% | 33.3% | 1.00 |
| apple | 12 | 3 | 9 | 1 | 75.0% | 33.3% | 1.00 |
| wooden_pickaxe | 12 | 3 | 2 | **0** | 16.7% | **0.0%** | 0.00 |

**wood_log and wooden_pickaxe minorities are never visited at all.** The 3 rotten wood_logs and 3 damaged pickaxes — the objects where probing should matter most — are completely missed.

### Why Doesn't C13 Target Minorities?

C13's `select_next_object` uses prior/global VOI (without visible features) to rank unvisited objects. The prior VOI depends on the entropy of the global outcome distribution, which is identical for all objects of the same category before observation. The differences in prior VOI scores come only from reach cost (Manhattan distance). Since minority/majority objects of the same category are randomly positioned, there is no VOI-driven preference for minority objects at the object-selection stage.

After observation, visible features do provide subtype cues, but by then the object has already been visited. The visit decision was made with only prior information.

**Conclusion: poor_minority_subtype_targeting is confirmed.** C13's object selection has no mechanism to prefer minority-subtype objects.

---

## 7. Diagnostic 5: Probe Value by Subtype

| Subtype | n Probes | Effective | Zero-Gain | Harmful | Eff Rate |
|---------|----------|-----------|-----------|---------|----------|
| Majority | 20 | 11 | 9 | 0 | **0.5500** |
| Minority | 2 | 0 | 2 | 0 | **0.0000** |

Minority-subtype probes are **less effective** than majority-subtype probes (0% vs 55% effective). This is counterintuitive — minority objects are where probing should be most valuable — but the sample size is tiny (only 2 minority probes).

The two minority probes are:
1. `test_apple_011` (unripe_or_spoiled, eat probe): outcome=fail, zero-gain — the probe confirmed what was already predicted (the observation may have already indicated unripe)
2. `test_stone_block_008` (brittle, use_as_tool probe): outcome=fail, zero-gain — probing `use_as_tool` on stone_block is always fail (non-differentiating action for this category)

Neither of these two minority probes targets the differentiating action for the object's category. The `eat` probe on the unripe apple IS the differentiating action for apple, but it was zero-gain (the prediction was already correct before probing).

**With only 2 minority probes out of 22 total, and 0 effective, the probe strategy fails to deliver value where it's most needed.**

---

## 8. Diagnostic 6: Positive Recall Loss Decomposition

### Aggregate (across all 5 queries)

| Category | Count | % of FN |
|----------|-------|---------|
| **A: Unvisited positive** | 52 | **82.5%** |
| B: Visited, not probed, wrong | 0 | 0.0% |
| C: Probed, but still wrong | 2 | 3.2% |
| D: Probe changed prediction to false | 9 | 14.3% |
| **Total FN** | 63 | 100% |

### Per-Query

| Query | Total FN | A (unvisited) | A% | B (no probe, wrong) | C (probed, wrong) | D (probe→false) |
|-------|----------|---------------|-----|---------------------|-------------------|-----------------|
| need_food | 7 | 3 | 42.9% | 0 | 0 | 4 (57.1%) |
| need_fuel | 25 | 24 | **96.0%** | 0 | 0 | 1 (4.0%) |
| need_planks | 10 | 8 | **80.0%** | 0 | 1 (10.0%) | 1 (10.0%) |
| need_stone | 10 | 7 | **70.0%** | 0 | 0 | 3 (30.0%) |
| need_tool | 11 | 10 | **90.9%** | 0 | 1 (9.1%) | 0 (0.0%) |

**82.5% of C13's false negatives are simply objects it never visited.** This is the dominant failure mode. For need_fuel, 96% of missed positives are unvisited objects.

Category D (probe changed prediction to false) accounts for 14.3% of FNs. This is the negative-only tradeoff in action: on some objects, the probe outcome makes C13 more conservative, flipping a correct positive prediction to a false negative. This occurs most notably on need_food (4/7 FNs from this category).

**Conclusion: breadth loss dominates.** The theoretical upper bound on C13 positive recall, even with perfect predictions, is 0.333 (26 visited / 78 total positives). C13's actual positive recall of 0.200 reflects both the breadth ceiling and some prediction errors on visited objects.

---

## 9. Diagnostic 7: C13 vs Random Probe Explanation

Both C13 and random_probe visit exactly the same 22 objects (overlap = 22, meaning identical object selection). Both use nearest-first selection under the same budget constraint. The +0.0308 macro_bal advantage comes entirely from better **action selection**:

| Metric | C13 | Random | Advantage |
|--------|-----|--------|-----------|
| Visit count | 22 | 22 | identical |
| Object overlap | 22 | 22 | identical |
| Differentiating action rate | **40.9%** | 22.7% | +18.2pp |
| Effective probe rate | **50.0%** | 31.8% | +18.2pp |
| macro_bal | **0.5733** | 0.5425 | +0.0308 |

C13's VOI gate selects the differentiating action nearly twice as often as random (40.9% vs 22.7% = 1.8x). With 6 candidate actions and only 1 differentiating per category, random achieves exactly the expected rate (1/6 ≈ 16.7% if uniform; the 22.7% reflects some action-level bias from the `rng.choice` distribution).

### Per-Query Comparison

| Query | C13 bal | Random bal | C13 Better? | Pos Coverage |
|-------|---------|------------|-------------|--------------|
| need_food | 0.7083 | 0.6667 | Yes (+0.0417) | 75.0% |
| need_fuel | 0.5333 | 0.4833 | Yes (+0.0500) | 20.0% |
| need_planks | 0.5104 | 0.5000 | Yes (+0.0104) | 33.3% |
| need_stone | 0.5833 | 0.5313 | Yes (+0.0521) | 41.7% |
| need_tool | 0.5313 | 0.5313 | No (tie) | 16.7% |

C13 beats random on 4/5 queries, ties on 1. The largest advantages are on need_fuel (+0.05) and need_stone (+0.052). need_tool is a tie because only 2 positive objects are visited — both policies make the same prediction on these objects.

**C13's weak advantage (+0.0308) is real but insufficient.** Better action selection provides a small but consistent improvement over random probing, but the fundamental bottleneck — visiting too few objects — is shared by both policies.

---

## 10. Diagnostic 8: Category Shortcut Risk Note

| Metric | Value |
|--------|-------|
| category_majority macro_bal | 0.9750 |
| C_minus_prior_only macro_bal | 0.5000 |
| Category shortcut risk | **Strong** |
| Agent exploits shortcut? | **No** |

category_majority = 0.9750 shows that C4 has a strong latent category-level shortcut. However, C_minus_prior_only = 0.5000 confirms the current zero-interaction agent does NOT exploit it. The shortcut is latent, not active.

**For future environment work:** Consider designs where category-majority does not trivially solve the task (e.g., balanced subtypes, cross-category affordance patterns). This block does not modify the environment.

---

## 11. Final Diagnosis

### Diagnosis Summary

| Diagnosis | Supported? | Evidence |
|-----------|------------|----------|
| **visit_breadth_loss** | **YES** | 82.5% of FNs from unvisited objects. C13 visits only 22/60 objects. C13 outperforms C0b on visited subset (0.7163 vs 0.5570). |
| **poor_differentiating_action_selection** | **YES** | Only 40.9% of probes target the differentiating action. 59.1% probe non-differentiating actions, 61.5% of which are zero-gain. |
| **poor_minority_subtype_targeting** | **YES** | Only 2/12 minority objects visited vs 20/48 majority. No enrichment (ratio = 1.0). wood_log and wooden_pickaxe minorities completely missed. |
| posterior_prediction_bias | **NO** | C13 outperforms C0b on every query when restricted to visited objects. Probing improves predictions; the problem is breadth, not quality. |
| **persistent_negative_only_tradeoff** | **YES** | C13 worsens positive recall (-0.11) while improving negative recall (+0.0825). 14.3% of FNs from probes that changed prediction to false. |
| **environment_category_shortcut_risk** | **YES** | category_majority = 0.975, but C_minus = 0.500 confirms agent does not exploit it. Latent risk only. |

### Main Diagnosis

```
visit_breadth_loss + poor_differentiating_action_selection
+ poor_minority_subtype_targeting + persistent_negative_only_tradeoff
+ environment_category_shortcut_risk
```

### The Failure Cascade

1. **Budget forces a visit-probe tradeoff.** At budget 1.5, each visit+probe costs ~0.066 (reach ~0.011 + observe 0.005 + probe 0.05). C13 can afford only 22 objects.

2. **Object selection is category-blind.** C13's prior VOI cannot distinguish majority from minority objects before observation. Minority objects are visited at a lower rate (16.7%) than majority (41.7%) simply because there are fewer of them.

3. **Action selection wastes 59% of probes.** C13's VOI gate prioritizes actions with the highest expected utility gain for that specific action, not actions that would maximally reduce subtype uncertainty across all actions. Non-differentiating actions often show small VOI signals because confirming an already-confident prediction technically reduces entropy.

4. **Missed positives dominate.** 82.5% of false negatives are objects C13 never sees. Even perfect predictions on the 22 visited objects cannot overcome the 38 unvisited ones.

5. **Probing on visited objects helps but can't compensate.** C13 achieves 0.7163 macro_bal on visited objects (vs C0b's 0.5570 on the same subset), but the 38 unvisited objects — all predicted using prior/observation-only — drag the overall macro_bal down to 0.5733.

### What Would Need to Change (for reference, not for implementation now)

1. **Better object selection**: A mechanism to identify likely-minority-subtype objects from prior/global information (e.g., category-level uncertainty, cluster outlier detection) before visiting.

2. **Better action selection**: A VOI computation that considers cross-action inference — how probing action A changes beliefs about action B. Or explicit knowledge of which actions are subtype-differentiating per category.

3. **Higher budget or cheaper probes**: At budget 3.0 (2x current), C13 could visit ~44 objects, substantially reducing the breadth gap. Or reducing probe_cost from 0.05 to 0.02 would allow more probes per object.

4. **Two-pass strategy**: First pass to observe all objects (like C0b), then a second pass to selectively probe objects where observation leaves high uncertainty. This would require a different budget structure.

These are design considerations for future blocks, not changes to make now.

---

*Generated by Block 1J4 — C13 failure decomposition, no fixes, no environment changes.*
