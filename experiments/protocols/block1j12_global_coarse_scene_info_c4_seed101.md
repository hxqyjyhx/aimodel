# Block 1J12 — Global Coarse Scene Information Diagnostic

**Date:** 2026-05-11
**Block:** 1J12 — Global coarse scene information diagnostic
**Condition:** C4_instance_subtype_cued_v1
**Seed:** 101
**Budget:** 1.5
**Scene Bonus Weight:** 0.15

---

## 1. Executive Summary

1J12 tests whether giving the agent coarse object-addressable information about all objects before acting improves object selection or probe targeting. The motivation comes from 1J11's finding that aggregate (non-object-addressable) peripheral observation via Route F_mid was safe but not useful.

**C13G (0.5796) improves over C13 (0.5733) by +0.0063, but C15G (0.5992) underperforms C15b (0.6142) by -0.0150.** The global coarse public-visible map helps C13's single-pass object selection but misleads C15b's two-pass probe ranking when used as a multiplicative VOI bonus.

**Global coarse public-visible information is weakly useful for single-pass selection but not sufficient to improve two-pass VOI-based probing.** The scene priority signal does not correlate with diagnostic value under C4_instance_subtype_cued_v1 — selected objects have slightly lower scene priority (0.5322) than unselected ones (0.5471).

**Do not proceed to Route F2.** Object-addressable coarse scene information is not enough. Consider interleaved breadth-depth or probe informativeness redesign.

---

## 2. Motivation

| Prior Finding | Implication |
|--------------|-------------|
| 1J11: F_mid aggregate peripheral is safe but not useful | Aggregate histograms don't differentiate objects under C4 |
| 1J11: Context diversity scores near-uniform (0.9957 vs 0.9963) | Public visible features too homogeneous for regional differentiation |
| 1J11: Route F aggregate-only signals can't rank objects | Need object-addressable information |

**Hypothesis:** If the remaining bottleneck is lack of scene-level object information, then a global coarse public-visible map (object-addressable) should improve probe target selection. If even global coarse information does not help, then Route F2 / object-addressable peripheral observation may not be sufficient.

---

## 3. Global Coarse Scene Map Design

### 3.1 Map Contents

For every object, the scene map exposes:

| Field | Source | Type |
|-------|--------|------|
| object_id | object identity | string |
| position (row, col) | grid position | tuple |
| distance_from_start | Manhattan distance from agent start | int |
| visible_feature_vector | public visible features per object | dict[str,bool] |
| visible_feature_count | count of present features | int |
| region_id | 3x3 region assignment | string |
| region_object_count | objects in same region | int |
| region_feature_prevalence | per-feature prevalence in region | dict[str,float] |
| public_cue_rarity_score | how rare this object's feature pattern is globally | float [0,1] |
| region_cue_diversity | entropy of feature prevalence within region | float [0,1] |

### 3.2 Scene Priority Formula

```
scene_priority = 0.4 * rarity + 0.3 * diversity + 0.3 * (1 - distance_normalized)
```

Higher values indicate objects with rarer public cue patterns, in diverse regions, closer to start.

### 3.3 Scene Map Statistics

| Metric | Value |
|--------|-------|
| Objects | 60 |
| Regions | 9 (3x3) |
| Rarity score range | [0.3729, 0.5271] |
| Rarity score mean | 0.4572 |
| Diversity score range | [0.7307, 0.8616] |
| Diversity score mean | 0.8125 |

The narrow rarity range (0.15 spread) and compressed diversity range (0.13 spread) indicate limited between-object differentiation from public visible features alone.

### 3.4 Forbidden-Information Compliance

| Check | Status |
|-------|--------|
| No hidden_subtype access | ✓ |
| No true affordance access | ✓ |
| No hidden category access | ✓ |
| No query label access | ✓ |
| Scene map uses only public visible features + geometry | ✓ |
| Oracle UB labeled non-deployable | ✓ |
| No per-object hidden-state prediction | ✓ |

---

## 4. Policy Variants

### 4.1 C13G_GlobalCoarseMapPolicy

Extends C13_InstanceVOIPolicy. In `select_next_object`, adds scene priority bonus to VOI-based object score:

```
adjusted_voi = best_action_voi + scene_bonus_weight * scene_priority(oid)
```

Selects object with highest adjusted VOI. Probe decision unchanged (standard VOI gate).

### 4.2 C15G_GlobalCoarseMapPolicy

Extends C15b two-pass architecture. In `_select_probe_target` (Pass 2), multiplies VOI by scene priority:

```
adjusted_voi = best_action_voi * (1.0 + scene_bonus_weight * scene_priority(oid))
```

Pass 1 is identical to C15b (nearest-first observe). Pass 2 uses scene-weighted VOI ranking.

### 4.3 Oracle Object Selection Upper Bound (Non-Deployable)

Greedy oracle selection: rank objects by oracle-prediction utility gain over global prior, select top objects within budget. Labeled `diagnostic_non_deployable`.

---

## 5. Baseline Results

| Policy | macro_bal | pos_recall | neg_recall | visited | probed | ncost | Type |
|--------|-----------|------------|------------|---------|--------|-------|------|
| all_false | 0.5000 | 0.0000 | 1.0000 | 0 | 0 | 0.0000 | floor |
| C_minus_prior_only | 0.5000 | — | — | 0 | 0 | 0.0000 | no_interaction |
| C0b_original | 0.5871 | 0.3100 | 0.8642 | 60 | 0 | 0.7533 | observe_only |
| C13_instance_VOI_original | 0.5733 | — | — | 22 | 22 | 0.9733 | instance_VOI |
| C15b_probe_reserve_original | 0.6142 | 0.3600 | 0.8683 | 57 | 8 | 0.9900 | two_pass_VOI |
| C15F_Fmid (from 1J11) | 0.6142 | 0.3600 | 0.8683 | 57 | 8 | 0.9900 | C15F_aggregate_peripheral |
| C_oracle_full_information | 1.0000 | 1.0000 | 1.0000 | — | — | — | oracle |

---

## 6. C13G and C15G Results

### 6.1 C13G (global coarse map + C13 base)

| Metric | Value |
|--------|-------|
| macro_query_balanced_accuracy | **0.5796** |
| mean_positive_recall | 0.3067 |
| mean_negative_recall | 0.8525 |
| visit_count | 20 |
| probe_count | 20 |
| normalized_cost | 0.9800 |
| effective_probe_count | 20 |
| effective_probe_rate | 1.00 |
| majority_probed | 17 |
| minority_probed | 3 |

### 6.2 C15G (global coarse map + C15b base)

| Metric | Value |
|--------|-------|
| macro_query_balanced_accuracy | **0.5992** |
| mean_positive_recall | 0.3500 |
| mean_negative_recall | 0.8483 |
| visit_count | 57 |
| probe_count | 8 |
| normalized_cost | 0.9967 |
| effective_probe_count | 8 |
| effective_probe_rate | 1.00 |
| majority_probed | 3 |
| minority_probed | 5 |
| scene_priority_mean_selected | 0.5322 |
| scene_priority_mean_unselected | 0.5471 |

### 6.3 Oracle Object Selection Upper Bound

| Metric | Value |
|--------|-------|
| macro_query_balanced_accuracy | **0.6292** |
| n_selected | 15 |
| normalized_cost | 0.9900 |
| label | diagnostic_non_deployable |

---

## 7. Key Comparisons

### 7.1 C13G vs C13

| Metric | Delta |
|--------|-------|
| delta_macro_bal | **+0.0063** |
| delta_visit_count | -2 |
| delta_probe_count | -2 |
| delta_positive_recall | +0.0067 |
| delta_minority_probed | +3 → 3 (vs C13's 0) |

C13G probes 3 minority objects (C13 probes 0), suggesting the scene map's rarity signal helps surface underrepresented objects. But the overall gain is small.

### 7.2 C15G vs C15b

| Metric | Delta |
|--------|-------|
| delta_macro_bal | **-0.0150** |
| delta_positive_recall | -0.0100 |
| delta_minority_probed | +3 (5 vs 2) |
| delta_effective_probe_rate | 0.0000 |

C15G probes 5 minority objects (vs C15b's 2) but loses 0.0150 macro_bal. The scene priority shifts probe selection toward minority objects, but those objects have lower diagnostic value. The multiplicative VOI bonus degrades the pure VOI ranking.

### 7.3 C15G vs C15F_Fmid

| Metric | Delta |
|--------|-------|
| delta_macro_bal | **-0.0150** |
| global_coarse_more_useful_than_aggregate_Fmid | False |

Object-addressable global coarse information is NOT more useful than aggregate F_mid under this policy design. Both are identical/slightly worse than C15b.

### 7.4 Object Overlap

| Set | Count | Objects |
|-----|-------|---------|
| C15b probed | 8 | (same 8 as 1J11) |
| C15G probed | 8 | different set |
| Shared C15b/C15G | varies | partial overlap |
| C13G probed | 20 | broader coverage |

### 7.5 Gap Capture

| Metric | Value |
|--------|-------|
| C0b base | 0.5871 |
| Oracle ceiling | 1.0000 |
| Oracle-C0b gap | 0.4129 |
| C15b gap capture | 6.56% |
| C15G gap capture | 2.92% |
| C15G hits 10% threshold (0.6284) | **No** |
| Oracle UB | 0.6292 |

The oracle upper bound (0.6292) confirms there is ~0.015 headroom above C15b, but the scene map does not capture it.

---

## 8. Scene Priority Analysis

### 8.1 Selection vs Non-Selection

C15G's selected objects have LOWER mean scene priority (0.5322) than unselected visited objects (0.5471). This negative correlation indicates the scene priority formula does not align with diagnostic value under C4.

### 8.2 Why Scene Priority Fails

The public_cue_rarity_score is based on how rare an object's visible feature pattern is globally. Under C4_instance_subtype_cued_v1:
- Rarity range is narrow (0.37-0.53) — objects don't differ much in feature pattern rarity
- Rare feature patterns are NOT more diagnostically informative — they may indicate unusual but irrelevant cue combinations
- Regional diversity is uniformly high (0.73-0.86) — all regions have similar cue entropy

The scene priority bonus effectively adds noise to the VOI ranking, degrading it.

### 8.3 Minority Surfacing vs Diagnostic Value

C15G probes 5 minority objects (vs C15b's 2), a real increase in minority targeting. But this comes at the cost of 0.0150 macro_bal, suggesting the additionally surfaced minority objects are not the diagnostically valuable ones.

---

## 9. Interpretation

```
C13G improves over C13:          True  (+0.0063 delta)
C15G improves over C15b:         False (-0.0150 delta)
C15G improves over C15F_Fmid:    False (-0.0150 delta)
C15G hits 10% threshold:         False (0.5992 < 0.6284)
C15G improves pos_recall:        False (-0.0100)
C15G improves minority targeting: True  (5 vs 2)
global_coarse_info_useful:       True  (C13G > C13)
route_f2_supported:              False
candidate_ready_for_multiseed:   False
```

**Interpretation: C13 can use scene information somewhat, but breadth-preserving structure remains important. Global coarse map helps C13's single-pass selection but C15b's two-pass architecture already captures similar information.**

The scene priority signal is weakly useful for initial object ranking (helps C13) but degrades probe decision quality when applied as a bonus to VOI (hurts C15G). The scene map helps surface minority objects but the surfaced objects are not diagnostically valuable.

---

## 10. Final Recommendation

**Do not proceed to Route F2 (object-addressable peripheral observation).** Object-addressable coarse public-visible scene information is not sufficient to improve over C15b under C4_instance_subtype_cued_v1.

### Detailed Findings

1. **C13G (+0.0063 over C13):** Weak positive signal. Scene map helps object selection slightly but C13 remains well below C15b.

2. **C15G (-0.0150 vs C15b):** Scene priority bonus degrades probe ranking. Multiplicative VOI bonus is not the right formula — it overrides diagnostic VOI with irrelevant public-cue rarity.

3. **Oracle UB = 0.6292:** Confirms only ~0.015 headroom exists above C15b with current probes/actions/budget. The ceiling is genuinely tight.

4. **Minority surfacing vs diagnostic value:** Scene map increases minority targeting (+3) but at accuracy cost — minority objects surfaced are not diagnostically valuable.

### Possible Next Directions

1. **Interleaved breadth-depth**: Instead of two-pass (observe all then probe), interleave observation and probing. The key question is when to stop observing and start probing — not which objects to prefer a priori.

2. **Probe informativeness redesign**: The current 5 main candidate actions may not be equally informative for all object subtypes. If some actions are uninformative for particular objects, a per-object action selection policy could help.

3. **Accept C15b as near-ceiling**: C15b at 0.6142 may be close to the maximum achievable with VOI-based probing under C4 with budget=1.5. The remaining headroom (0.015 to oracle UB, 0.386 to full oracle) is mostly in object-specific hidden information that no public-visible signal can recover.

4. **Different cue condition**: C4's public visible features may be inherently limited. A condition with richer public cues (e.g., C3 or a new design) might show more benefit from scene-level information.

---

*Generated by Block 1J12 — Global coarse scene information diagnostic.*
