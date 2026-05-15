# Block 1J16 — Confident-Error Probe Ranking Diagnostic

**Date:** 2026-05-11
**Block:** 1J16 — Confident-error / calibration-aware probe ranking diagnostic
**Condition:** C4_instance_subtype_cued_v1
**Seed:** 101
**Budget:** 1.5

---

## 1. Executive Summary

1J16 tests whether support/disagreement-based risk signals can identify diagnostic probes better than uncertainty-based ranking. Three C19 variants replace C15b's affordance-utility VOI with calibration-aware risk scoring while keeping the same observe phase:

- **C19a**: Neighbor disagreement risk — score = outcome_variance among kNN neighbors.
- **C19b**: Low support risk — score = 1.0 - composite support (neighbor count, similarity weight, agreement).
- **C19c (PRIMARY)**: Confident error risk — score = confidence * disagreement * (1 + query_relevance tiebreaker).

**C19c reaches macro_bal=0.6025 (delta_vs_C15b=-0.0117).** C19c reaches 0.6025 (-0.0117 vs C15b), below C15b. Confident-error proxy DEGRADES performance. The hand-coded risk proxy (confidence * disagreement) selects probes that are individually risky but collectively unhelpful. Current hand-coded proxies cannot approximate oracle error ranking.

**Oracle k=8 reference**: 0.6488 (non-deployable, ranks by |IOM_pred - gt| on all 60 objects).

---

## 2. Motivation

| Prior Finding | Implication |
|--------------|-------------|
| 1J14: C15b_UB reaches 0.8546 with oracle probe selection | Probe information has substantial usable value |
| 1J14: Oracle k=8 reaches 0.6488 vs C15b 0.6142 | C15b's VOI selects wrong probes — +0.0346 from better selection |
| 1J15: Query-relevant ranking (C18b) = 0.5975 | Uncertainty-based ranking does not differentiate from affordance utility |
| 1J15: C18a = C15b = 0.6142 | min(p,1-p) is monotonically equivalent to C15b's utility |
| IOM: predict_outcome returns confidence + support info | IOM has internal calibration data that VOI ignores |

**Key question**: Can the IOM's internal calibration signals (confidence, neighbor disagreement, support strength) identify probes that oracle error ranking would select, better than uncertainty-based proxies?

**Hypothesis**: The oracle k=8 ranking selects pairs where IOM is confident but WRONG (|pred - gt| is large). These are cases where neighbor disagreement should be high (neighbors split on outcomes) while IOM confidence may still be moderate-to-high (kNN produces a weighted average that happens to be far from truth). A risk proxy of confidence * disagreement should correlate with oracle error better than raw uncertainty.

---

## 3. C19 Variant Designs

### 3.1 Common Architecture

All C19 variants share C15b's observe phase:
- Nearest-first object visitation
- 25% budget reserve
- Same transition logic to probe phase

Only the probe ranking is replaced. Fixed probe budget = C15b's probe count (k≈8).

### 3.2 Risk Scoring Functions

The scoring uses `InstanceOutcomeMemory.get_object_dynamic_indicators()` which returns per-action:
- `outcome_variance`: variance of neighbor outcomes (disagreement)
- `total_sim_weight`: sum of similarity weights
- `neighbor_count`: number of kNN matches found

**IOM confidence** (replicated from `predict_outcome` internals):
```
confidence = 0.4 * min(total_weight / n_neighbors, 1)
           + 0.3 * min(n_neighbors / k, 1)
           + 0.3 * (1 - min(outcome_variance / 0.25, 1))
```

### 3.3 C19a — Neighbor Disagreement Risk

Score = `outcome_variance`

Higher variance = neighbors disagree more about this outcome = more probe value.

### 3.4 C19b — Low Support Risk

```
support = 0.35 * min(neighbor_count/k, 1)
        + 0.35 * min(total_sim_weight/neighbor_count, 1)
        + 0.30 * (1 - min(outcome_variance/0.25, 1))
risk = 1.0 - support
```

Low support = few/weak neighbors or high disagreement = high risk.

### 3.5 C19c — Confident Error Risk (PRIMARY)

```
risk = confidence * outcome_variance * (0.5 + 0.5 * min(p, 1-p))
```

- **confidence**: high when IOM is sure (many similar neighbors agree)
- **outcome_variance**: high when neighbors disagree
- **min(p, 1-p)**: query-relevance tiebreaker (near decision boundary)

The product `confidence * disagreement` targets the confident-error pattern: IOM is confident but the evidence base is split. This should correlate with oracle |pred - gt|.

---

## 4. Forbidden-Information Compliance

| Check | Status |
|-------|--------|
| No environment change | ✓ |
| No budget/cost change | ✓ |
| Single seed only | ✓ |
| No hidden labels in policy | ✓ |
| No oracle probe outcomes | ✓ |
| No IOM update change | ✓ |
| No Route F/F2/global coarse map | ✓ |
| No hidden subtype/category in ranking | ✓ |
| Uses only IOM internals + visible features | ✓ |

---

## 5. Baseline Results

| Policy | macro_bal | pos_recall | neg_recall | visited | probed | ncost | Type |
|--------|-----------|------------|------------|---------|--------|-------|------|
| all_false | 0.5000 | 0.0000 | 1.0000 | 0 | 0 | 0.0000 | floor |
| C_minus_prior_only | 0.5000 | — | — | 0 | 0 | 0.0000 | no_interaction |
| C0b_original | 0.5871 | 0.3100 | 0.8642 | 60 | 0 | 0.7533 | observe_only |
| C13_instance_VOI_original | 0.5733 | — | — | 22 | 22 | 0.9667 | instance_VOI |
| C15b_probe_reserve_original | 0.6142 | 0.3600 | 0.8683 | 57 | 8 | 0.9900 | two_pass_VOI |
| C18b_query_relevant (1J15) | 0.5975 | 0.3267 | 0.8683 | 57 | 6 | 0.9967 | query_relevant_fixed |
| C_oracle_full_information | 1.0000 | 1.0000 | 1.0000 | — | — | — | oracle |

### Oracle References (non-deployable)

| Variant | macro_bal | delta_vs_C15b |
|---------|-----------|---------------|
| 1J14 C15b_UB | 0.8546 | +0.2404 |
| 1J14 IOM k=8 | 0.6488 | +0.0346 |
| 1J16 Oracle k=8 | 0.6488 | +0.0346 |
| Full_UB | 1.0000 | +0.3858 |

Oracle-C0b gap: **0.4129**. C15b captures 6.56%.

---

## 6. C19 Results

### 6.1 Aggregate Metrics

| Metric | C15b | C18b | C19a | C19b | C19c (PRIMARY) |
|--------|------|------|------|------|----------------|
| macro_bal | 0.6142 | 0.5975 | 0.6067 | 0.6042 | **0.6025** |
| mean_positive_recall | 0.3600 | 0.3267 | 0.3367 | 0.3267 | 0.3367 |
| mean_negative_recall | 0.8683 | 0.8683 | 0.8767 | 0.8817 | 0.8683 |
| visit_count | 57 | 57 | 57 | 57 | 57 |
| probe_count | 8 | 6 | 5 | 5 | 5 |
| delta_vs_C15b | — | -0.0167 | -0.0075 | -0.0100 | **-0.0117** |
| gap_capture | 6.56% | 2.52% | 4.74% | 4.14% | **3.73%** |

### 6.2 Per-Query Breakdown

| Query | C15b Bal | C18b Bal | C19c Bal | C19c Δ vs C15b |
|-------|----------|----------|----------|----------------|
| need_food | 0.6250 | 0.5833 | 0.5833 | -0.0417 |
| need_fuel | 0.6333 | 0.6333 | 0.6167 | -0.0167 |
| need_planks | 0.5729 | 0.5312 | 0.5312 | -0.0417 |
| need_stone | 0.5521 | 0.5521 | 0.5521 | +0.0000 |
| need_tool | 0.6875 | 0.6875 | 0.7292 | +0.0417 |

### 6.3 Probe Selection

| Metric | C15b | C18b | C19a | C19b | C19c |
|--------|------|------|------|------|------|
| probe_count | 8 | 6 | 5 | 5 | 5 |
| actions | {'burn_as_fuel': 2, 'mine_by_hand': 3, 'eat': 2, 'craft_plank': 1} | {'mine_by_hand': 2, 'burn_as_fuel': 3, 'use_as_tool': 1} | {'craft_plank': 1, 'eat': 2, 'burn_as_fuel': 1, 'mine_by_hand': 1} | {'burn_as_fuel': 4, 'craft_plank': 1} | {'burn_as_fuel': 2, 'mine_by_hand': 2, 'use_as_tool': 1} |
| effective | — | — | 5 | 0 | 5 |
| zero_gain | — | — | 0 | 0 | 0 |
| harmful | — | — | 0 | 5 | 0 |
| mean_confidence | — | — | — | — | 0.5028 |
| mean_disagreement | — | — | 0.2500 | — | 0.2500 |

### 6.4 Pair Overlap Analysis

| Variant | Pairs | Overlap C15b | Overlap Oracle k=8 |
|---------|-------|-------------|-------------------|
| C15b | [('test_wooden_pickaxe_006', 'burn_as_fuel'), ('test_wooden_pickaxe_003', 'burn_as_fuel'), ('test_stone_block_001', 'mine_by_hand'), ('test_apple_007', 'eat'), ('test_wood_log_002', 'craft_plank'), ('test_stone_block_009', 'mine_by_hand'), ('test_apple_010', 'eat'), ('test_stone_block_000', 'mine_by_hand')] | 8 | 0 |
| C18b | [('test_wood_log_011', 'mine_by_hand'), ('test_stone_block_010', 'burn_as_fuel'), ('test_wooden_pickaxe_007', 'burn_as_fuel'), ('test_apple_013', 'burn_as_fuel'), ('test_wooden_pickaxe_010', 'use_as_tool'), ('test_stone_block_000', 'mine_by_hand')] | 1 | 0 |
| C19a | [('test_apple_001', 'craft_plank'), ('test_apple_002', 'eat'), ('test_apple_007', 'eat'), ('test_apple_008', 'burn_as_fuel'), ('test_stone_block_010', 'mine_by_hand')] | 1 | 0 |
| C19b | [('test_stone_block_007', 'burn_as_fuel'), ('test_wood_log_002', 'craft_plank'), ('test_wood_log_009', 'burn_as_fuel'), ('test_apple_001', 'burn_as_fuel'), ('test_apple_008', 'burn_as_fuel')] | 1 | 0 |
| C19c | [('test_wood_log_000', 'burn_as_fuel'), ('test_stone_block_000', 'mine_by_hand'), ('test_wooden_pickaxe_009', 'use_as_tool'), ('test_wood_log_011', 'mine_by_hand'), ('test_wooden_pickaxe_012', 'burn_as_fuel')] | 1 | 0 |
| Oracle k=8 | [('test_stone_block_003', 'mine_by_hand'), ('test_stone_block_008', 'burn_as_fuel'), ('test_apple_014', 'eat'), ('test_wooden_pickaxe_005', 'use_as_tool'), ('test_wooden_pickaxe_003', 'use_as_tool'), ('test_apple_012', 'eat'), ('test_stone_block_014', 'mine_by_hand'), ('test_stone_block_011', 'mine_by_hand')] | 0 | 8 |

---

## 7. Key Comparisons

### 7.1 C19c vs C15b (PRIMARY)

| Metric | Delta |
|--------|-------|
| delta_macro_bal | **-0.0117** |
| delta_positive_recall | -0.0233 |
| delta_negative_recall | +0.0000 |
| delta_visit_count | 0 |
| delta_probe_count | -3 |
| C19c hits 10% threshold (0.6284) | **No** |
| overlap_with_oracle_k8 | 0/8 |

### 7.2 C19 Variant Comparison

| Variant | macro_bal | delta_vs_C15b | probed | overlap_C15b | overlap_oracle_k8 |
|---------|-----------|---------------|--------|-------------|-------------------|
| C19A | 0.6067 | -0.0075 | 5 | 1 | 0 |
| C19B | 0.6042 | -0.0100 | 5 | 1 | 0 |
| C19C | 0.6025 | -0.0117 | 5 | 1 | 0 |

### 7.3 Gap Capture

| Policy | gap_capture |
|--------|-------------|
| C15b | 6.56% |
| C18b (1J15) | 2.52% |
| C19a | 4.74% |
| C19b | 4.14% |
| C19c | 3.73% |
| Oracle k=8 | 14.93% |
| 1J14 C15b_UB | 64.78% |

---

## 8. Interpretation

```
C19c improves over C15b:     False (-0.0117)
C19c hits 10% threshold:     False (0.6025 vs 0.6284)
confident_error_proxy_supported: False
candidate_ready_for_small_multiseed: False
ready_for_multiseed:          False
no_cross_test_object_propagation: true
```

**C19c interpretation: C19c reaches 0.6025 (-0.0117 vs C15b), below C15b. Confident-error proxy DEGRADES performance. The hand-coded risk proxy (confidence * disagreement) selects probes that are individually risky but collectively unhelpful. Current hand-coded proxies cannot approximate oracle error ranking.**

**Best oracle k=8 overlap: C19a at 0 pairs. None of the C19 variants recover oracle-like probe pairs at high rate.**

### Per-Query Interpretation

- **need_food**: degraded (-0.0417)
- **need_fuel**: degraded (-0.0167)
- **need_planks**: degraded (-0.0417)
- **need_stone**: unchanged (0.0000)
- **need_tool**: improved (+0.0417)

---

## 9. Why Hand-Coded Risk Proxies May Not Differentiate

### 9.1 The Oracle Error Signal

Oracle k=8 ranks by `|IOM_pred - ground_truth|`. This is a SUPERVISED error signal — it knows the correct answer. Hand-coded proxies attempt to approximate this from internal IOM state alone.

### 9.2 Limitations of Current Proxies

1. **Disagreement != error**: Neighbors can disagree because the test case is genuinely ambiguous (true p≈0.5), not because the IOM is wrong. Disagreement captures ambiguity, not error.

2. **Confidence * disagreement captures a specific pattern**: High confidence + high disagreement suggests the IOM is interpolating between conflicting evidence. But this pattern may not dominate the oracle error distribution.

3. **Low support != wrong**: A prediction with few weak neighbors may be correct (the training data is complete enough). Low support correlates with extrapolation, not necessarily error.

4. **No ground-truth signal**: Oracle k=8 uses `|pred - gt|` — the hand-coded proxy has no access to gt. Without error supervision, the proxy can only guess which predictions are wrong.

### 9.3 Diagnostic Caveats

**Under current IOM architecture:**
- Test-object probes do NOT propagate to other test objects.
- The IOM's internal calibration signals (confidence, variance, support) are the only available risk indicators besides prediction probabilities.
- None of these signals have access to ground truth.

**C19 is NOT claimed as a full solution.** C19 tests whether simple hand-coded proxies can approximate oracle error ranking from IOM internals alone.

---

## 10. Final Recommendation

**Confident-error proxy does not differentiate from C15b's VOI. Hand-coded risk proxies cannot approximate oracle error from IOM internals alone.**

**Next recommended route: learned_probe_value_critic_from_effective_harmful_logs.**

The effective/harmful probe logs from 1J14, 1J15, and 1J16 provide labeled data: which (object, action) probes were effective (improved query scores) and which were harmful (degraded scores). A learned probe-value critic trained on these logs could potentially identify patterns that hand-coded proxies miss.

### Consolidated Block 1J11-1J16 Findings

| Block | Approach | Result | vs C15b | Key Finding |
|-------|----------|--------|---------|-------------|
| 1J11 | F_mid aggregate peripheral | 0.6142 | +0.0000 | Safe but not useful |
| 1J12 | Global coarse scene map (C15G) | 0.5992 | -0.0150 | Scene priority misaligns with diagnostic value |
| 1J13 | Interleaved breadth-depth (C17) | 0.5667 | -0.0475 | VOI doesn't activate at low breadth |
| 1J14 | Oracle probe injection (C15b_UB) | 0.8546 | +0.2404 | Substantial probe value exists; VOI fails to find it |
| 1J15 | Query-relevant ranking (C18b) | 0.5975 | -0.0167 | Uncertainty proxy = affordance utility; fixed budget hurts |
| **1J16** | **Confident-error risk (C19c)** | **0.6025** | **-0.0117** | **C19c reaches 0.6025 (-0.0117 vs C15b), below C15b. Confident-error proxy DEGRADES performance. The hand-coded risk proxy...** |

---

*Generated by Block 1J16 — Confident-Error / Calibration-Aware Probe Ranking Diagnostic.*
