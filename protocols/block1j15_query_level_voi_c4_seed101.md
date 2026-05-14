# Block 1J15 — Query-Relevant Probe Ranking Diagnostic

**Date:** 2026-05-11
**Block:** 1J15 — Query-relevant probe ranking diagnostic
**Condition:** C4_instance_subtype_cued_v1
**Seed:** 101
**Budget:** 1.5

---

## 1. Executive Summary

1J15 tests whether query-relevant probe ranking (preferring object-action pairs near query decision boundaries) improves macro_bal over C15b's affordance-utility VOI. Three variants isolate ranking quality from stop-rule/cost-calibration issues:

- **C18a (ablation)**: Local query-confidence VOI, VOI-gated with cost penalty.
- **C18b (PRIMARY)**: Query-relevant ranking, fixed probe budget (= C15b's probe count, k≈8). No cost threshold — isolates probe selection quality.
- **C18c**: Same ranking as C18b, but VOI-gated with cost penalty. Tests whether the deployable stop rule works.

**Architecture finding**: The current IOM has no cross-test-object propagation (`_direct_outcomes` only affects the probed object; test objects are not in `_train_oids`). Global recomputation is equivalent to local direct-effect scoring. Therefore C18 tests query-relevant direct probe ranking, not propagation-based global VOI.

**C18b reaches macro_bal=0.5975 (delta_vs_C15b=-0.0167).** C18b reaches 0.5975 (-0.0167 vs C15b), below C15b. C15b–C18b (object, action) pair overlap: 1/6 = 16.7%. Overlap pairs: [('test_stone_block_000', 'mine_by_hand')]. Unique to C18b: [('test_apple_013', 'burn_as_fuel'), ('test_stone_block_010', 'burn_as_fuel'), ('test_wood_log_011', 'mine_by_hand'), ('test_wooden_pickaxe_007', 'burn_as_fuel'), ('test_wooden_pickaxe_010', 'use_as_tool')]. Unique to C15b: [('test_apple_007', 'eat'), ('test_apple_010', 'eat'), ('test_stone_block_001', 'mine_by_hand'), ('test_stone_block_009', 'mine_by_hand'), ('test_wood_log_002', 'craft_plank'), ('test_wooden_pickaxe_003', 'burn_as_fuel'), ('test_wooden_pickaxe_006', 'burn_as_fuel')]. Query-relevant ranking DEGRADES performance. Under current IOM architecture, test-object probes do not propagate to other objects, and local query-confidence gain min(p,1-p) is nearly monotonic with C15b's affordance-utility gain. The fixed budget forces probes that the VOI cost gate would have skipped, resulting in budget wasted on low-value probes. The missing ingredient is likely confident-error detection / calibration-aware probe value, not simple uncertainty. Do not claim C18 is full query-level VOI.

---

## 2. Motivation

| Prior Finding | Implication |
|--------------|-------------|
| 1J14: C15b_UB reaches 0.8546 with oracle probe selection | Probe information has substantial usable value |
| 1J14: Oracle k=8 (same budget) reaches 0.6488 vs C15b 0.6142 | C15b's VOI selects wrong probes — +0.0346 from better selection |
| 1J14: C15b VOI = affordance utility; oracle ranking = \|pred - gt\| | Different ranking functions produce different probe sets |
| IOM: _direct_outcomes only affects probed object | No cross-object propagation; global = local |

**Key question**: Can a query-relevant ranking (preferring probes near query decision boundaries) improve over C15b's affordance-utility ranking, even with the same probe budget?

---

## 3. C18 Variant Designs

### 3.1 Common Architecture

All C18 variants share C15b's observe phase:
- Nearest-first object visitation
- 25% budget reserve
- Same transition logic to probe phase

### 3.2 Query-Relevant Ranking Score

For candidate (object, action), score = `min(p, 1-p)` where p = current IOM prediction for the feature that this action probes.

This is the expected gain in query decision confidence from resolving this (object, feature):
- Current query confidence = max(p, 1-p)
- After probing (resolved to 0 or 1): confidence = 1.0
- Expected gain = 1.0 - max(p, 1-p) = min(p, 1-p)

Action-to-query mapping:
| Action | Query |
|--------|-------|
| craft_plank | need_planks |
| eat | need_food |
| use_as_tool | need_tool |
| burn_as_fuel | need_fuel |
| mine_by_hand | need_stone |
| mine_with_pickaxe | need_stone |

### 3.3 C18a — Local Query Confidence VOI (Ablation)

- Score: query-relevant gain / 6 (scaled like C15b's per-feature utility)
- Stop rule: positive net VOI with cost penalty
- Purpose: test whether query-confidence framing alone differs from C15b

### 3.4 C18b — Query-Relevant Fixed Probe Budget (PRIMARY)

- Score: query-relevant gain (no scaling)
- Stop rule: probe exactly k objects (k = C15b's observed probe count = 8)
- No cost threshold — isolates probe selection quality
- Purpose: if C18b > C15b with same probe count, the ranking is better

### 3.5 C18c — Query-Relevant Net VOI

- Score: same as C18b
- Stop rule: positive net VOI with cost penalty (like C15b)
- Purpose: if C18b > C15b but C18c ≤ C15b, the ranking works but the stop rule / cost calibration blocks it

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
| No ground truth in ranking | ✓ (uses only current IOM predictions) |

---

## 5. Baseline Results

| Policy | macro_bal | pos_recall | neg_recall | visited | probed | ncost | Type |
|--------|-----------|------------|------------|---------|--------|-------|------|
| all_false | 0.5000 | 0.0000 | 1.0000 | 0 | 0 | 0.0000 | floor |
| C_minus_prior_only | 0.5000 | — | — | 0 | 0 | 0.0000 | no_interaction |
| C0b_original | 0.5871 | 0.3100 | 0.8642 | 60 | 0 | 0.7533 | observe_only |
| C13_instance_VOI_original | 0.5733 | — | — | 22 | 22 | 0.9667 | instance_VOI |
| C15b_probe_reserve_original | 0.6142 | 0.3600 | 0.8683 | 57 | 8 | 0.9900 | two_pass_VOI |
| C_oracle_full_information | 1.0000 | 1.0000 | 1.0000 | — | — | — | oracle |

### 1J14 Oracle UB Reference (non-deployable)

| Variant | macro_bal | delta_vs_C15b |
|---------|-----------|---------------|
| C15b_UB | 0.8546 | +0.2404 |
| IOM k=8 | 0.6488 | +0.0346 |
| C13_full_UB | 0.6867 | +0.0725 |
| Full_UB | 1.0000 | +0.3858 |

Oracle-C0b gap: **0.4129**. C15b captures 6.56%.

---

## 6. C18 Results

### 6.1 Aggregate Metrics

| Metric | C15b | C18a (ablation) | C18b (primary) | C18c (netVOI) |
|--------|------|-----------------|----------------|---------------|
| macro_bal | 0.6142 | 0.6142 | **0.5975** | 0.6042 |
| mean_positive_recall | 0.3600 | 0.3600 | 0.3267 | 0.3333 |
| mean_negative_recall | 0.8683 | 0.8683 | 0.8683 | 0.8750 |
| visit_count | 57 | 57 | 57 | 57 |
| probe_count | 8 | 8 | 6 | 7 |
| total_cost | 1.4850 | 1.4850 | 1.4950 | 1.4600 |
| delta_vs_C15b | — | +0.0000 | **-0.0167** | -0.0100 |
| gap_capture | 6.56% | 6.56% | **2.52%** | 4.14% |
| runtime_s | — | 2.1 | 0.5 | 1.9 |

### 6.2 Per-Query Breakdown

| Query | C15b Bal | C18b Bal | C18c Bal | C18b Δ vs C15b |
|-------|----------|----------|----------|----------------|
| need_food | 0.6250 | 0.5833 | 0.6250 | -0.0417 |
| need_fuel | 0.6333 | 0.6333 | 0.6667 | +0.0000 |
| need_planks | 0.5729 | 0.5312 | 0.5312 | -0.0417 |
| need_stone | 0.5521 | 0.5521 | 0.5104 | +0.0000 |
| need_tool | 0.6875 | 0.6875 | 0.6875 | +0.0000 |

### 6.3 Probe Selection

| Metric | C15b | C18a | C18b | C18c |
|--------|------|------|------|------|
| probe_count | 8 | 8 | 6 | 7 |
| actions | {'burn_as_fuel': 2, 'mine_by_hand': 3, 'eat': 2, 'craft_plank': 1} | {'burn_as_fuel': 2, 'mine_by_hand': 3, 'eat': 2, 'craft_plank': 1} | {'mine_by_hand': 2, 'burn_as_fuel': 3, 'use_as_tool': 1} | {'burn_as_fuel': 5, 'eat': 1, 'mine_by_hand': 1} |
| effective | — | 0 | 0 | 0 |
| zero_gain | — | 0 | 0 | 0 |
| harmful | — | 8 | 6 | 7 |

### 6.4 C15b vs C18b (Object, Action) Pair Overlap

| Metric | Value |
|--------|-------|
| C15b probe pairs | [('test_apple_007', 'eat'), ('test_apple_010', 'eat'), ('test_stone_block_000', 'mine_by_hand'), ('test_stone_block_001', 'mine_by_hand'), ('test_stone_block_009', 'mine_by_hand'), ('test_wood_log_002', 'craft_plank'), ('test_wooden_pickaxe_003', 'burn_as_fuel'), ('test_wooden_pickaxe_006', 'burn_as_fuel')] |
| C18b probe pairs | [('test_apple_013', 'burn_as_fuel'), ('test_stone_block_000', 'mine_by_hand'), ('test_stone_block_010', 'burn_as_fuel'), ('test_wood_log_011', 'mine_by_hand'), ('test_wooden_pickaxe_007', 'burn_as_fuel'), ('test_wooden_pickaxe_010', 'use_as_tool')] |
| overlap pairs | [('test_stone_block_000', 'mine_by_hand')] |
| overlap count | 1 / 6 = 16.7% |
| unique to C18b | [('test_apple_013', 'burn_as_fuel'), ('test_stone_block_010', 'burn_as_fuel'), ('test_wood_log_011', 'mine_by_hand'), ('test_wooden_pickaxe_007', 'burn_as_fuel'), ('test_wooden_pickaxe_010', 'use_as_tool')] (5) |
| unique to C15b | [('test_apple_007', 'eat'), ('test_apple_010', 'eat'), ('test_stone_block_001', 'mine_by_hand'), ('test_stone_block_009', 'mine_by_hand'), ('test_wood_log_002', 'craft_plank'), ('test_wooden_pickaxe_003', 'burn_as_fuel'), ('test_wooden_pickaxe_006', 'burn_as_fuel')] (7) |
| ranking differs due to cost penalty removal | **Yes** |

---

## 7. Key Comparisons

### 7.1 C18b vs C15b (PRIMARY)

| Metric | Delta |
|--------|-------|
| delta_macro_bal | **-0.0167** |
| delta_positive_recall | -0.0333 |
| delta_negative_recall | +0.0000 |
| delta_visit_count | 0 |
| delta_probe_count | -2 |
| C18b hits 10% threshold (0.6284) | **No** |

### 7.2 C18b vs C18c (Fixed Budget vs Net VOI Stop Rule)

| Metric | C18b | C18c | Delta |
|--------|------|------|-------|
| macro_bal | 0.5975 | 0.6042 | -0.0067 |
| probe_count | 6 | 7 | -1 |

**C18c (net VOI, 7 probes) outperforms C18b (fixed budget, 6 probes) by +0.0067. The net VOI stop rule effectively filters out low-value probes.**

### 7.3 C18a vs C15b (Ablation)

| Metric | Delta |
|--------|-------|
| delta_macro_bal | **+0.0000** |
| C18a equivalent to C15b | **Yes** |

### 7.4 Gap Capture

| Policy | gap_capture |
|--------|-------------|
| C15b | 6.56% |
| C18a | 6.56% |
| C18b | 2.52% |
| C18c | 4.14% |
| 1J14 C15b_UB | 64.78% |
| 1J14 IOM k=8 | 14.93% |

---

## 8. Interpretation

```
C18b improves over C15b:     False (-0.0167)
C18b hits 10% threshold:     False (0.5975 vs 0.6284)
query_relevant_ranking_supported: False
candidate_ready_for_small_multiseed: False
ready_for_multiseed:          False
no_cross_test_object_propagation: true
c18_global_recompute_equivalent_to_local: true
```

**C18b interpretation: C18b reaches 0.5975 (-0.0167 vs C15b), below C15b. C15b–C18b (object, action) pair overlap: 1/6 = 16.7%. Overlap pairs: [('test_stone_block_000', 'mine_by_hand')]. Unique to C18b: [('test_apple_013', 'burn_as_fuel'), ('test_stone_block_010', 'burn_as_fuel'), ('test_wood_log_011', 'mine_by_hand'), ('test_wooden_pickaxe_007', 'burn_as_fuel'), ('test_wooden_pickaxe_010', 'use_as_tool')]. Unique to C15b: [('test_apple_007', 'eat'), ('test_apple_010', 'eat'), ('test_stone_block_001', 'mine_by_hand'), ('test_stone_block_009', 'mine_by_hand'), ('test_wood_log_002', 'craft_plank'), ('test_wooden_pickaxe_003', 'burn_as_fuel'), ('test_wooden_pickaxe_006', 'burn_as_fuel')]. Query-relevant ranking DEGRADES performance. Under current IOM architecture, test-object probes do not propagate to other objects, and local query-confidence gain min(p,1-p) is nearly monotonic with C15b's affordance-utility gain. The fixed budget forces probes that the VOI cost gate would have skipped, resulting in budget wasted on low-value probes. The missing ingredient is likely confident-error detection / calibration-aware probe value, not simple uncertainty. Do not claim C18 is full query-level VOI.**

**C18a ablation: C18a at 0.6142 (delta vs C15b: +0.0000). See per-query breakdown for differences.**

**Stop rule: C18c (net VOI, 7 probes) outperforms C18b (fixed budget, 6 probes) by +0.0067. The net VOI stop rule effectively filters out low-value probes.**

### Per-Query Interpretation

- **need_food**: degraded (-0.0417)
- **need_fuel**: unchanged (0.0000)
- **need_planks**: degraded (-0.0417)
- **need_stone**: unchanged (0.0000)
- **need_tool**: unchanged (0.0000)

---

## 9. Why Query-Relevant Ranking May Not Differentiate

### 9.1 Monotonic Equivalence Under Current IOM

For single-feature queries (need_planks, need_food, need_tool, need_fuel), the query decision depends on a single feature crossing the 0.5 threshold. The query-relevant gain `min(p, 1-p)` and C15b's per-feature utility gain `(1 - max(p, 1-p)) / 6` are **monotonically equivalent** — they produce the same per-action ranking within an object.

The only structural differences between C15b and C18b are:
1. **Per-object aggregation**: Both use max over actions → same best action per object
2. **Cost penalty**: C18b ignores reach cost in ranking; C15b penalizes distant objects
3. **Stop rule**: C18b probes exactly k objects; C15b stops when no net VOI > 0

### 9.2 Diagnostic Caveats

**Under current IOM architecture:**
- Test-object probes do NOT propagate to other test objects (`_direct_outcomes` only affects the probed object; test objects are not in `_train_oids`).
- Local query-confidence gain using `min(p, 1-p)` is nearly monotonic with C15b's affordance-utility gain.
- Therefore C18a/C18c mainly test cost/stop calibration, not a fundamentally new query-level value model.

**C18 is NOT claimed as full query-level VOI.** The C18 variants test query-relevant direct probe ranking, not propagation-based global VOI.

### 9.3 Pair Overlap Findings

| Metric | Value |
|--------|-------|
| C15b probe pairs | [('test_apple_007', 'eat'), ('test_apple_010', 'eat'), ('test_stone_block_000', 'mine_by_hand'), ('test_stone_block_001', 'mine_by_hand'), ('test_stone_block_009', 'mine_by_hand'), ('test_wood_log_002', 'craft_plank'), ('test_wooden_pickaxe_003', 'burn_as_fuel'), ('test_wooden_pickaxe_006', 'burn_as_fuel')] |
| C18b probe pairs | [('test_apple_013', 'burn_as_fuel'), ('test_stone_block_000', 'mine_by_hand'), ('test_stone_block_010', 'burn_as_fuel'), ('test_wood_log_011', 'mine_by_hand'), ('test_wooden_pickaxe_007', 'burn_as_fuel'), ('test_wooden_pickaxe_010', 'use_as_tool')] |
| Overlap | 1/6 = 16.7% |
| Ranking differs due to cost penalty removal | **Yes — cost penalty removal changes selected pairs substantially** |

### 9.4 Interpreting C18b Results

- **If C18b improves over C15b**: improvement likely comes from fixed probe budget / reduced cost penalty, not necessarily query-level reasoning.
- **If C18b equals C15b**: query-confidence ranking is equivalent to affordance-utility ranking under current IOM.
- **If C18b fails to approach 1J14 oracle k=8 (0.6488)**: the missing ingredient is likely confident-error detection / calibration-aware probe value, not simple uncertainty.

**Current result: C18b = 0.5975 vs 1J14 oracle k=8 = 0.6488 → gap = 0.0513.** C18b is substantially below oracle k=8. Confident-error detection / calibration-aware probe value is the missing ingredient.

---

## 10. Final Recommendation

**Query-relevant ranking does not differentiate from C15b's VOI. The ranking functions are monotonically equivalent for the current IOM architecture.**

### Consolidated Block 1J11-1J15 Findings

| Block | Approach | Result | vs C15b | Key Finding |
|-------|----------|--------|---------|-------------|
| 1J11 | F_mid aggregate peripheral | 0.6142 | +0.0000 | Safe but not useful |
| 1J12 | Global coarse scene map (C15G) | 0.5992 | -0.0150 | Scene priority misaligns with diagnostic value |
| 1J13 | Interleaved breadth-depth (C17) | 0.5667 | -0.0475 | VOI doesn't activate at low breadth |
| 1J14 | Oracle probe injection (C15b_UB) | 0.8546 | +0.2404 | Substantial probe value exists; VOI fails to find it |
| **1J15** | **Query-relevant ranking (C18b)** | **0.5975** | **-0.0167** | **C18b reaches 0.5975 (-0.0167 vs C15b), below C15b. C15b–C18b (object, action) pair overlap: 1/6 = 16.7%. Overlap pairs: ...** |

---

*Generated by Block 1J15 — Query-Relevant Probe Ranking Diagnostic.*
