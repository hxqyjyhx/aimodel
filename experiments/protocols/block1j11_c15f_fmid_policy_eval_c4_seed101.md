# Block 1J11 — C15F Policy Evaluation using F_mid Peripheral Observation Interface

**Date:** 2026-05-11
**Block:** 1J11 — C15F policy evaluation
**Condition:** C4_instance_subtype_cued_v1
**Seed:** 101
**Budget:** 1.5
**Route F Variant:** F_mid (d1=2, d2=4)
**Primary Variant:** C15F_V2_context_guided_probe

---

## 1. Executive Summary

1J11 evaluates whether the audited F_mid peripheral observation interface improves C15F policy performance over the original C15b two-pass architecture.

**C15F_V2 (context_guided_probe) achieves 0.6142 — identical to C15b (0.6142).** The F_mid context diversity bonus does not change probe selection because public-visible-cue diversity scores are near-uniform across objects (mean_selected=0.9957 vs mean_all=0.9963). Under C4_instance_subtype_cued_v1, the public visible features don't produce enough between-region variation for the peripheral histogram diversity score to differentiate objects.

**Route F is safe but not yet useful.** The F_mid interface adds no leakable information (confirmed in 1J10) and doesn't harm performance, but also doesn't improve object selection under this policy design. C15F_V2 is on the Pareto frontier (not dominated). Both C15b and C15F_V2 achieve 100% effective probe rate under the same utility-delta definition, and probe the same objects with identical actions and outcomes.

**Do not multi-seed.** Consider interleaved breadth-depth or object-addressable Route F redesign.

---

## 2. Why 1J11 Was Allowed After 1J10

| 1J10 Finding | Value | Gate |
|-------------|-------|------|
| Leakage audit (F_mid) | PASS (0 hard fails) | ✓ |
| Marginal-value audit (F_mid) | PASS | ✓ |
| C0b_RouteF_deployable == C0b_original | True (0.5871) | ✓ (expected for aggregate-only) |
| Public category-like | 0.5000 (≤ 0.60) | ✓ |
| Peripheral UB near oracle | False | ✓ |
| local_prevalence_predictor_removed | True | ✓ |

1J10 confirmed that F_mid peripheral signals do not leak hidden information and that C0b_RouteF predictions remain identical to original C0b. F_mid is safe enough to evaluate as an observation interface for policy targeting.

---

## 3. C15F Implementation Details

### 3.1 Variant Architecture

Two C15F variants were implemented atop the C15b two-pass architecture:

| Variant | Pass 1 | Pass 2 | F_mid Use |
|---------|--------|--------|-----------|
| **C15F_V2** (context_guided_probe) | Same nearest-first observe as C15b | VOI + context diversity bonus for probe ranking | Context guides probe prioritization among visited objects |
| **C15F_V1** (context_guided_observe) | Diversity-weighted observe (prefer high-diversity regions) | Same VOI-based probe as C15b | Context guides observation target selection |

### 3.2 C15F_V2 Design (Primary)

- **Pass 1**: Identical to C15b — nearest-first broad observe, transition to probe phase at 25% budget reserve
- **Pass 2**: For each visited object, compute VOI as in C15b. Apply F_mid context diversity bonus: `adjusted_voi = voi * (1 + 0.15 * context_diversity_score)`. Select highest adjusted-VOI object.
- **Peripheral signal**: Computed after each focal observation using positions and public visible features only
- **Context diversity score**: Entropy of public-visible-cue histogram (near+mid layers) combined with local object density. Range [0, 1]. Uses ONLY public visible features.

### 3.3 C15F_V1 Design (Secondary)

- **Pass 1**: For each unvisited object, estimate regional diversity from already-visited positions' peripheral signals. Score = `0.5 * diversity / (1 + reach_cost)`. Prefer high-diversity, low-cost objects.
- **Pass 2**: Standard C15b VOI-based probe selection (no context bonus).

### 3.4 Forbidden-Information Compliance

| Check | Status |
|-------|--------|
| No hidden_subtype access | ✓ |
| No true affordance access | ✓ |
| No hidden category access | ✓ |
| No peripheral → object-level prediction | ✓ |
| No local_prevalence + global_prior blending | ✓ |
| No best-signal oracle access | ✓ |
| No classifier-based peripheral prediction | ✓ |
| Only public visible features + geometry used | ✓ |
| Within 1J10 audited F_mid constraints | ✓ |

---

## 4. Baseline Results

| Policy | macro_bal | pos_recall | neg_recall | visited | probed | ncost | Type |
|--------|-----------|------------|------------|---------|--------|-------|------|
| all_false | 0.5000 | 0.0000 | 1.0000 | 0 | 0 | 0.0000 | floor |
| C_minus_prior_only | 0.5000 | — | — | 0 | 0 | 0.0000 | no_interaction |
| C0b_original | 0.5871 | 0.3100 | 0.8642 | 60 | 0 | 0.7533 | observe_only |
| C0b_RouteF_Fmid | 0.5871 | 0.3100 | 0.8642 | 60 | 0 | 0.7533 | observe_only (identical to C0b) |
| random_probe_original | 0.5750 | — | — | — | 14 | — | random_probe |
| random_probe_RouteF_Fmid | 0.5750 | — | — | — | 14 | — | random_probe (identical) |
| C13_instance_VOI_original | 0.5733 | — | — | 22 | 22 | — | instance_VOI |
| C15b_probe_reserve | 0.6142 | 0.3600 | 0.8683 | 57 | 8 | 0.9900 | two_pass_VOI |
| C15B_cross_action_diag | 0.6142 | 0.3600 | 0.8683 | 57 | 8 | 0.9900 | same as C15b (Route B null) |
| C_oracle | 1.0000 | 1.0000 | 1.0000 | — | — | — | oracle |

Oracle-C0b gap: **0.4129**. C15b captures 6.56%.

---

## 5. C15F Results

### 5.1 C15F_V2 (context_guided_probe) — PRIMARY

| Metric | Value |
|--------|-------|
| macro_query_balanced_accuracy | **0.6142** |
| mean_positive_recall | 0.3600 |
| mean_negative_recall | 0.8683 |
| visit_count | 57 |
| probe_count | 8 |
| normalized_cost | 0.9900 |
| effective_probe_rate | **1.00** (8/8) |
| zero_gain_probes | 0 |
| harmful_probes | 0 |
| majority_probed | 6 |
| minority_probed | 2 |
| context_score_mean_selected | 0.9957 |
| context_score_mean_unselected | 0.9963 |
| F_mid_context_used | true |
| n_focal_with_peripheral | 57 |

### 5.2 C15F_V1 (context_guided_observe)

| Metric | Value |
|--------|-------|
| macro_query_balanced_accuracy | **0.5925** |
| mean_positive_recall | 0.3333 |
| mean_negative_recall | 0.8517 |
| visit_count | 57 |
| probe_count | 5 |
| normalized_cost | 0.9600 |
| effective_probe_rate | 1.00 (5/5) |
| majority_probed | 3 |
| minority_probed | 2 |

C15F_V1 performs worse than C15b because diversity-guided observe deprioritizes some diagnostically valuable objects in favor of public-cue-diverse regions. Under C4, public-cue diversity does not correlate with diagnostic value.

### 5.3 Per-Query Breakdown (C15F_V2 vs C15b)

| Query | C15b Bal Acc | C15F_V2 Bal Acc | Delta |
|-------|-------------|-----------------|-------|
| need_food | 0.6250 | 0.6250 | 0 |
| need_fuel | 0.6333 | 0.6333 | 0 |
| need_planks | 0.5729 | 0.5729 | 0 |
| need_stone | 0.5521 | 0.5521 | 0 |
| need_tool | 0.6875 | 0.6875 | 0 |

All per-query metrics identical to C15b. Context bonus doesn't change object ranking.

---

## 6. Key Comparisons

### 6.1 C15F vs C15b

| Metric | Delta |
|--------|-------|
| delta_macro_bal | **0.0000** |
| delta_mean_positive_recall | 0.0000 |
| delta_mean_negative_recall | 0.0000 |
| delta_visit_count | 0 |
| delta_probe_count | 0 |
| delta_effective_probe_rate | **0.00** (1.00 vs 1.00) |
| delta_normalized_cost | 0.0000 |
| probe_efficiency_definition_same | True |
| probes_identical_oids | True |
| probes_identical_actions | True |
| probes_identical_outcomes | True |

C15F_V2 and C15b achieve identical accuracy with identical probe efficiency — both score 8/8 effective under the same `_compute_utility` delta > 0.001 method. They probe the exact same 8 objects with the same actions and outcomes. The previously reported C15b eff_rate=0.50 from 1J8 was a definitional artifact — when the same efficiency method is applied to both policies' `_probe_tracking`, they produce identical results.

### 6.2 C15F vs C0b

| Metric | Delta |
|--------|-------|
| delta_macro_bal | **+0.0271** |
| delta_normalized_cost | +0.2367 |
| Pareto relation | tradeoff (higher accuracy, higher cost) |

### 6.3 C15F vs C0b_RouteF_Fmid

C15F adds +0.0271 over C0b_RouteF (same as over C0b). Policy use of F_mid adds value over observe-only Route F interface, but this is the same value C15b already provides without Route F.

### 6.4 C15F vs Random Probe Route F

C15F: +0.0392 macro_bal, -0.0033 normalized_cost. C15F weakly dominates random probe.

### 6.5 Gap Capture

| Metric | C15b | C15F_V2 |
|--------|------|---------|
| macro_bal | 0.6142 | 0.6142 |
| gap_capture_vs_C0b | 6.56% | 6.56% |
| 10% threshold (0.6284) | no | no |

---

## 7. Positive-Recall and Tradeoff Analysis

| Metric | C0b | C15b | C15F_V2 |
|--------|-----|------|---------|
| mean_positive_recall | 0.3100 | 0.3600 | 0.3600 |
| mean_negative_recall | 0.8642 | 0.8683 | 0.8683 |

C15F_V2 improves positive recall over C0b (+0.0500) but is identical to C15b. No negative-only tradeoff detected — the gains (over C0b) are balanced.

---

## 8. Object-Selection / Minority-Targeting Analysis

| Metric | C15b | C15F_V2 |
|--------|------|---------|
| majority_probed | 6 | 6 |
| minority_probed | 2 | 2 |
| minority_probe_rate | 25% | 25% |

C15F_V2 probes the same 2 minority-subtype objects as C15b. Context diversity does not help surface minority objects — they are not in systematically higher or lower diversity regions.

### Context Score Analysis

Context diversity scores are near-uniform:
- mean_selected: 0.9957
- mean_unselected: 0.9963
- Difference: -0.0006

The near-zero difference confirms that F_mid peripheral histograms don't differentiate objects under C4_instance_subtype_cued_v1. The visible-feature distribution is too homogeneous across the 7x9 grid for aggregate histograms to provide useful between-region variation.

---

## 9. Interpretation

```
C15F improves over C15b:    False (0.0000 delta)
C15F improves over C0b:     True  (+0.0271 delta)
C15F hits 10% threshold:    False (0.6142 < 0.6284)
C15F improves pos_recall:   True  (over C0b only, not over C15b)
C15F not Pareto-dominated:  True  (on frontier)
C15F not neg-only tradeoff: True
C15F strongly promising:    False
Candidate for multi-seed:   False
```

**Interpretation: Route F interface is safe but not yet useful under this policy.**

C15F_V2 equals C15b on all accuracy metrics. The F_mid context diversity bonus is effectively a no-op because public-visible-cue histograms are near-uniform across the grid under C4_instance_subtype_cued_v1. Both policies achieve 100% effective probe rate under the same utility-delta definition — the previously reported C15b 50% rate was a definitional artifact from 1J8 (see Section 11).

C15F_V1 (diversity-guided observe) performs worse (0.5925), confirming that public-cue diversity does not correlate with diagnostic value under C4.

---

## 10. Final Recommendation

**Do not multi-seed.** The F_mid peripheral signal does not differentiate objects under C4_instance_subtype_cued_v1. Route F is not harmful — it's safe to use — but it doesn't help under the current aggregate-only signal design.

### Possible Next Directions

1. **Object-addressable Route F redesign**: If peripheral signals were per-object (e.g., blurred individual features rather than aggregate histograms), they might differentiate objects. This would require a new leakage audit.

2. **Interleaved breadth-depth**: Instead of two-pass (observe all then probe), interleave observation and probing. Peripheral context might help decide when to switch from breadth to depth.

3. **Different cue condition**: A condition with more spatially clustered visible features (e.g., C5 or a new condition) might produce more between-region variation in peripheral histograms.

4. **Accept C15b as ceiling for aggregate-only Route F**: C15b at 0.6142 may be near the ceiling for single-seed VOI-based probing under C4 with budget=1.5. The oracle object selection (0.6442 from 1J8) shows 0.0300 of remaining headroom, but capturing it requires per-object diagnostic information that aggregate peripheral signals don't provide.

---

## 11. Probe-Efficiency Consistency Check

### 11.1 Motivation

The original 1J8 report claimed C15b had eff_rate=0.50 (4/8 effective probes), while 1J11's C15F_V2 reported eff_rate=1.00 (8/8). Since both policies probed the same 8 objects with the same macro_bal=0.6142, this discrepancy was suspicious — either the probes/actions/outcomes differed, or the efficiency definitions differed.

### 11.2 Method

A standalone `compute_probe_efficiency_from_tracking()` function was applied to both policies' `_probe_tracking` lists, using the identical method as C15F_V2's `get_probe_efficiency()`:

```python
pre_u = _compute_utility(pre_probs)
post_u = _compute_utility(post_probs)
delta = post_u - pre_u
if delta > 0.001:    effective
elif delta < -0.001: harmful
else:                zero_gain
```

### 11.3 Results

| Check | Result |
|-------|--------|
| C15b effective_probe_count | **8** |
| C15b zero_gain_probe_count | 0 |
| C15b harmful_probe_count | 0 |
| C15b effective_probe_rate | **1.0000** |
| C15F_V2 effective_probe_count | 8 |
| C15F_V2 effective_probe_rate | 1.0000 |
| probe_efficiency_definition_same | **True** |
| probes_identical_oids | **True** |
| probes_identical_actions_by_oid | **True** (all 8) |
| probes_identical_outcomes_by_oid | **True** (all 8) |
| probe_efficiency_comparison_consistent | **True** |

C15b probed_oids: `['test_wooden_pickaxe_006', 'test_wooden_pickaxe_003', 'test_stone_block_001', 'test_apple_007', 'test_wood_log_002', 'test_stone_block_009', 'test_apple_010', 'test_stone_block_000']`

C15F_V2 probed_oids: identical set and order.

### 11.4 Conclusion

**The C15b eff_rate=0.50 from 1J8 was a definitional artifact.** When the same `_compute_utility` delta > 0.001 method is applied to both policies' probe tracking, both achieve 100% effective probe rate with identical probes, actions, and outcomes.

The robust conclusion is that C15F_V2 matches C15b on all accuracy metrics. The probe-efficiency comparison requires caution unless the C15b and C15F efficiency definitions are verified identical — which this check confirms. The apparent 50% improvement in effective probe rate was not real; both policies are equally probe-efficient under the same metric.

---

*Generated by Block 1J11 — C15F policy evaluation using audited F_mid peripheral observation interface.*
