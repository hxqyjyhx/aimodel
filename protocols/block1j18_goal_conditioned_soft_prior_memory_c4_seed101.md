# Block 1J18 — Goal-Conditioned Soft Probe Prior Memory

## 1. Objective

Add a minimal goal-conditioned soft prior memory layer:
visible attributes should raise or lower probe priority under a given goal,
without hard exclusion. Never use hidden subtype/category as input.
Never use oracle outcomes as deployment input.

## 2. Setup

| Parameter | Value |
|-----------|-------|
| Condition | C4_instance_subtype_cued_v1 |
| Seed | 101 |
| Budget | 1.5 |
| Cost Weight | 0.5 |
| Probe Cost | 0.05 |
| Observe Cost | 0.005 |
| Reach Cost/unit | 0.01 |
| IOM k | 10 |
| Exploration Floor | 0.05 |
| Type Families | 4 (wood-like, stone-like, apple-like, tool-like) |
| Prior Sanity Checks | 5/5 passed |

## 3. Default Goal Prior Table

Visible type families are detected from characteristic features:

| Family | Characteristic Features |
|--------|------------------------|
| wood-like | brownish, has_bark_texture, has_wood_grain, long_shape, rough_texture |
| stone-like | block_like, grayish, has_crystal_flecks, has_granular_surface, heavy_weight |
| apple-like | greenish, has_peel_texture, has_stem_remnant, light_weight, round_small, smooth_texture |
| tool-like | elongated_with_handle, has_grip_area, has_shaft_shape, long_shape, movable |

### Prior Values (type_family x goal x action)

| Family | Query | Action | Prior |
|--------|-------|--------|-------|
| apple-like | need_food | eat | 0.85 |
| apple-like | need_fuel | burn_as_fuel | 0.15 |
| apple-like | need_planks | craft_plank | 0.10 |
| apple-like | need_stone | mine_by_hand | 0.15 |
| apple-like | need_stone | mine_with_pickaxe | 0.15 |
| apple-like | need_tool | use_as_tool | 0.10 |
| stone-like | need_food | eat | 0.15 |
| stone-like | need_fuel | burn_as_fuel | 0.10 |
| stone-like | need_planks | craft_plank | 0.10 |
| stone-like | need_stone | mine_by_hand | 0.85 |
| stone-like | need_stone | mine_with_pickaxe | 0.85 |
| stone-like | need_tool | use_as_tool | 0.10 |
| tool-like | need_food | eat | 0.10 |
| tool-like | need_fuel | burn_as_fuel | 0.60 |
| tool-like | need_planks | craft_plank | 0.15 |
| tool-like | need_stone | mine_by_hand | 0.25 |
| tool-like | need_stone | mine_with_pickaxe | 0.25 |
| tool-like | need_tool | use_as_tool | 0.85 |
| wood-like | need_food | eat | 0.15 |
| wood-like | need_fuel | burn_as_fuel | 0.85 |
| wood-like | need_planks | craft_plank | 0.85 |
| wood-like | need_stone | mine_by_hand | 0.25 |
| wood-like | need_stone | mine_with_pickaxe | 0.25 |
| wood-like | need_tool | use_as_tool | 0.15 |

**Min prior: 0.10** (never zero — no hard exclusion)

## 4. C15b Reference

| Metric | Value |
|--------|-------|
| Macro BAcc | 0.6142 |
| Visited | 57 |
| Probed | 8 |
| Probe pairs | [('test_wooden_pickaxe_006', 'burn_as_fuel'), ('test_wooden_pickaxe_003', 'burn_as_fuel'), ('test_stone_block_001', 'mine_by_hand'), ('test_apple_007', 'eat'), ('test_wood_log_002', 'craft_plank'), ('test_stone_block_009', 'mine_by_hand'), ('test_apple_010', 'eat'), ('test_stone_block_000', 'mine_by_hand')] |

## 5. Variant Results

### 5.1 Aggregate Metrics

| Policy | Macro BAcc | Pos Rec | Neg Rec | Visited | Probed | Delta vs C15b |
|--------|-----------|---------|---------|---------|--------|---------------|
| C0b (observe only) | 0.5871 | — | — | — | 0 | — |
| C15b (VOI reserve) | 0.6142 | 0.3600 | 0.8683 | 57 | 8 | 0 |
| C18b (query-rel) | 0.5975 | — | — | — | — | -0.0167 |
| C19a (best from 1J16) | 0.6067 | — | — | — | 5 | -0.0075 |
| crossfit MLP (1J17p) | 0.5942 | — | — | — | — | -0.0200 |
| **soft_prior_only** | **0.6275** | 0.3867 | 0.8683 | 57 | 8 | **+0.0133** |
| **c15b_soft_prior_mul** | **0.6142** | 0.3533 | 0.8750 | 57 | 7 | **+0.0000** |
| **c15b_soft_prior_memory** | **0.6129** | 0.3533 | 0.8725 | 57 | 7 | **-0.0012** |
| Oracle k=8 | 0.6488 | — | — | — | 8 | +0.0346 |
| Oracle (full) | 1.0000 | — | — | — | — | — |

### 5.2 Probe Selection

| Metric | soft_prior_only | c15b_soft_prior_mul | c15b_soft_prior_memory |
|--------|----------------|---------------------|------------------------|
| Probe count | 8 | 7 | 7 |
| Effective | 8 | 7 | 7 |
| Zero gain | 0 | 0 | 0 |
| Harmful | 0 | 0 | 0 |
| Avg sel prior | 0.3469 | 0.3679 | 0.3679 |
| Avg unsel prior | 0.3488 | 0.3458 | 0.3458 |
| Low prior selected | 0 | 0 | 0 |
| High prior selected | 8 | 7 | 7 |
| Selected actions | {'eat': 2, 'mine_by_hand': 3, 'use_as_tool': 1, 'craft_plank': 2} | {'mine_by_hand': 3, 'eat': 1, 'craft_plank': 2, 'burn_as_fuel': 1} | {'mine_by_hand': 3, 'eat': 1, 'craft_plank': 3} |

### 5.3 Overlap Analysis

| Comparison | soft_prior_only | c15b_soft_prior_mul | c15b_soft_prior_memory |
|------------|----------------|---------------------|------------------------|
| Overlap C15b | 3 | 5 | 5 |
| Overlap oracle k=8 | 1 | 0 | 0 |

| Reference | Pairs |
|-----------|-------|
| C15b | [('test_wooden_pickaxe_006', 'burn_as_fuel'), ('test_wooden_pickaxe_003', 'burn_as_fuel'), ('test_stone_block_001', 'mine_by_hand'), ('test_apple_007', 'eat'), ('test_wood_log_002', 'craft_plank'), ('test_stone_block_009', 'mine_by_hand'), ('test_apple_010', 'eat'), ('test_stone_block_000', 'mine_by_hand')] |
| Oracle k=8 | [('test_stone_block_003', 'mine_by_hand'), ('test_stone_block_008', 'burn_as_fuel'), ('test_apple_014', 'eat'), ('test_wooden_pickaxe_005', 'use_as_tool'), ('test_wooden_pickaxe_003', 'use_as_tool'), ('test_apple_012', 'eat'), ('test_stone_block_014', 'mine_by_hand'), ('test_stone_block_011', 'mine_by_hand')] |
| soft_prior_only | [('test_wooden_pickaxe_006', 'eat'), ('test_stone_block_006', 'mine_by_hand'), ('test_stone_block_001', 'mine_by_hand'), ('test_wooden_pickaxe_003', 'use_as_tool'), ('test_wood_log_005', 'craft_plank'), ('test_apple_007', 'eat'), ('test_wood_log_002', 'craft_plank'), ('test_wood_log_014', 'mine_by_hand')] |
| c15b_soft_prior_mul | [('test_stone_block_001', 'mine_by_hand'), ('test_apple_007', 'eat'), ('test_wood_log_002', 'craft_plank'), ('test_stone_block_009', 'mine_by_hand'), ('test_stone_block_000', 'mine_by_hand'), ('test_wood_log_012', 'craft_plank'), ('test_stone_block_007', 'burn_as_fuel')] |
| c15b_soft_prior_memory | [('test_stone_block_001', 'mine_by_hand'), ('test_apple_007', 'eat'), ('test_wood_log_002', 'craft_plank'), ('test_stone_block_009', 'mine_by_hand'), ('test_stone_block_000', 'mine_by_hand'), ('test_wood_log_012', 'craft_plank'), ('test_stone_block_007', 'craft_plank')] |

### 5.4 Per-Query Breakdown

| Query | C15b BAcc | soft_prior_only | c15b_soft_prior_mul | c15b_soft_prior_memory |
|-------|----------|----------------|---------------------|------------------------|
| need_food | 0.6250 | 0.6250 (+0.0000) | 0.6250 (+0.0000) | 0.6250 (+0.0000) |
| need_fuel | 0.6333 | 0.6167 (-0.0167) | 0.6333 (+0.0000) | 0.6167 (-0.0167) |
| need_planks | 0.5729 | 0.6146 (+0.0417) | 0.5729 (+0.0000) | 0.5833 (+0.0104) |
| need_stone | 0.5521 | 0.5521 (+0.0000) | 0.5521 (+0.0000) | 0.5521 (+0.0000) |
| need_tool | 0.6875 | 0.7292 (+0.0417) | 0.6875 (+0.0000) | 0.6875 (+0.0000) |

## 6. Gap Capture

| Policy | Gap Capture % |
|--------|--------------|
| C15b | 6.6% |
| soft_prior_only | 9.8% |
| c15b_soft_prior_mul | 6.6% |
| c15b_soft_prior_memory | 6.3% |
| Oracle k=8 | 14.9% |

## 7. Pattern Priority Memory

After variant 2 probes, the pattern_priority_memory contains:

| Pattern | Support | Eff Rate | Harm Rate | Adj | Conf |
|---------|---------|----------|-----------|-----|------|
| apple-like|need_food|eat | 3 | 1.00 | 0.00 | 1.105 | 0.500 |
| stone-like|need_stone|mine_by_hand | 6 | 1.00 | 0.00 | 1.140 | 0.667 |
| tool-like|need_tool|use_as_tool | 1 | 1.00 | 0.00 | 1.052 | 0.250 |
| wood-like|need_fuel|burn_as_fuel | 1 | 1.00 | 0.00 | 1.052 | 0.250 |
| wood-like|need_planks|craft_plank | 4 | 1.00 | 0.00 | 1.120 | 0.571 |

## 8. Design Compliance

| Check | Status |
|-------|--------|
| No environment change | [OK] |
| No budget/cost change | [OK] |
| Single seed only | [OK] |
| No hidden subtype/category in deployment | [OK] |
| No oracle outcomes in deployment | [OK] |
| No hard exclusion (min prior = 0.10 > 0) | [OK] |
| Exploration floor > 0 | [OK] |
| No same-style learned critic | [OK] |
| C15b observe phase unchanged | [OK] |

## 9. Interpretation

Best soft prior variant (soft_prior_only) reaches 0.6275 (++0.0133 vs C15b), improving over C15b but not reaching 10% threshold (0.6284). Soft goal prior shows partial signal.

### Tiered Rules

- **If c15b_soft_prior_mul > C15b**: soft goal prior is useful as a probe-ranking modifier.
  -> **Result**: NOT SUPPORTED (delta=+0.0000)
- **If macro is similar to C15b but probes are better targeted**:
  keep prior as a coarse candidate-priority layer.
  -> **Result**: effect_on_C15b=improves
- **If soft prior hurts C15b**:
  reduce prior weight and use it only as diagnostic.
  -> **Result**: hurts=False
- **If soft_prior_only performs poorly**:
  acceptable; prior is not expected to replace C15b, only to guide it.
  -> **Result**: soft_prior_only_macro_bal=0.6275 (beats C0b=True)

## 10. Summary

```
[block_done]
block_id=1J18
c15b_macro_bal=0.6142
best_soft_prior_variant=soft_prior_only
best_soft_prior_macro_bal=0.6275
best_soft_prior_delta_vs_c15b=+0.0133
best_soft_prior_probe_count=8
best_soft_prior_overlap_with_c15b=3
best_soft_prior_overlap_with_oracle_k8=1
best_soft_prior_effective_probe_count=8
best_soft_prior_harmful_probe_count=0
hard_exclusion_used=false
soft_goal_prior_supported=true
memory_layer_keep=false
next_recommended_route=tune_soft_prior_weights_and_retry
candidate_ready_for_small_multiseed=false
ready_for_multiseed=false
```
