# Block 1J19c -- Prior-Violation Negative-Exception Feature-Risk Memory

## 1. Objective

Test whether prior-violation-based feature-risk memory reduces repeated high-prior negative outcomes across episodes, and whether learned risk features match injected deviation features.

## 2. Setup

| Parameter | Value |
|-----------|-------|
| Condition | C4_instance_subtype_cued_v1 |
| Seed | 101 |
| Budget | 1.5 |
| Episodes | 5 |
| Prior Violation Threshold | 0.6 |
| Min Violation Support | 2 |
| Risk Alpha | 5.0 |
| Max Risk | 2.0 |
| Threshold Sensitivity Tested | false |

## 3. Memory Semantics

- Only ELIGIBLE prior events (soft_prior >= threshold) update counts
- violation = eligible AND outcome == failure
- nonviolation = eligible AND outcome == success
- Low-prior events excluded from both numerator and denominator
- Risk weight = log(P(feature|violation) / P(feature|nonviolation)), clipped to [0, max_risk]
- Beta(1,1) pseudo-counts + shrinkage (alpha=5.0)

## 4. Baselines

| Policy | Macro BAcc |
|--------|------------|
| C0b | 0.5871 |
| C15b | 0.5975 |

## 5. Mode: Original Safe

Explore fraction: 0.0  |  Objects: 60

### Per-Episode

| Ep | A bal | B bal | C bal | A pv | B pv | C pv | B eff | B zero | B harm | B riskAdj |
|----|-------|-------|-------|------|------|------|-------|--------|--------|-----------|
| 1 | 0.5887 | 0.5887 | 0.5763 | 3 | 3 | 5 | 7 | 0 | 0 | 0.0000 |
| 2 | 0.5921 | 0.5754 | 0.5983 | 3 | 1 | 5 | 1 | 6 | 0 | 0.0000 |
| 3 | 0.5929 | 0.5929 | 0.5975 | 4 | 4 | 3 | 6 | 1 | 0 | 0.0000 |
| 4 | 0.6046 | 0.5892 | 0.5962 | 2 | 3 | 1 | 4 | 3 | 0 | 0.0000 |
| 5 | 0.5958 | 0.5708 | 0.5825 | 4 | 0 | 0 | 0 | 7 | 0 | 0.0000 |

### Aggregated

| Metric | A | B | C |
|--------|---|---|---|
| Mean BAcc | 0.5948 | 0.5834 | 0.5902 |
| Total Prior Violations | 16 | 11 | 14 |

### Prior Violation Diagnostics

| Metric | Value |
|--------|-------|
| Prior violations (B) | 11 |
| Repeated prior violations (B) | 1 |
| B reduces violations vs A | true |
| B beats C on violations | true |
| Eligible updates | 35 |
| Ineligible updates | 0 |

### Memory Diagnostics

| Metric | Value |
|--------|-------|
| Risk weights nonzero | 57 |
| Max risk weight | 0.4581 |
| Top risk features overlap deceptive | true |

### Top Risk Features

| Feature | Goal | Action | Risk | VWith | NVWith | VWithout | NVWithout | ActV |
|---------|------|--------|------|-------|--------|----------|-----------|------|
| elongated_with_handle | need_food | eat | 0.4581 | 2 | 1 | 2 | 4 | 2 |
| has_wood_grain | need_food | eat | 0.4581 | 2 | 1 | 2 | 4 | 2 |
| has_grip_area | need_food | eat | 0.4581 | 2 | 1 | 2 | 4 | 2 |
| on_left_side | need_food | eat | 0.4581 | 2 | 1 | 2 | 4 | 2 |
| has_granular_surface | need_food | eat | 0.4581 | 2 | 1 | 2 | 4 | 2 |
| heavy_weight | need_fuel | burn_as_fuel | 0.3891 | 4 | 1 | 1 | 2 | 3 |
| long_shape | need_fuel | burn_as_fuel | 0.3891 | 4 | 1 | 1 | 2 | 3 |
| movable | need_food | eat | 0.3143 | 3 | 2 | 1 | 3 | 2 |
| smooth_texture | need_food | eat | 0.3143 | 3 | 2 | 1 | 3 | 2 |
| solid | need_stone | mine_by_hand | 0.3041 | 3 | 1 | 1 | 2 | 2 |
| has_peel_texture | need_planks | craft_plank | 0.3041 | 3 | 1 | 1 | 2 | 2 |
| has_stem_remnant | need_planks | craft_plank | 0.3041 | 3 | 1 | 1 | 2 | 2 |
| recently_seen | need_planks | craft_plank | 0.3041 | 3 | 1 | 1 | 2 | 2 |
| round_small | need_planks | craft_plank | 0.3041 | 3 | 1 | 1 | 2 | 2 |
| near_table | need_tool | use_as_tool | 0.3041 | 3 | 1 | 1 | 2 | 2 |
| elongated_with_handle | need_planks | craft_plank | 0.3041 | 3 | 1 | 1 | 2 | 2 |
| long_shape | need_tool | use_as_tool | 0.3041 | 3 | 1 | 1 | 2 | 2 |
| has_bark_texture | need_planks | craft_plank | 0.3041 | 3 | 1 | 1 | 2 | 2 |
| has_stem_remnant | need_fuel | burn_as_fuel | 0.2612 | 3 | 1 | 2 | 2 | 3 |
| recently_seen | need_stone | mine_by_hand | 0.1520 | 2 | 1 | 2 | 2 | 2 |
## 5. Mode: Deceptive High-Prior

Explore fraction: 0.0  |  Objects: 60

### Per-Episode

| Ep | A bal | B bal | C bal | A pv | B pv | C pv | B eff | B zero | B harm | B riskAdj |
|----|-------|-------|-------|------|------|------|-------|--------|--------|-----------|
| 1 | 0.5613 | 0.5613 | 0.5488 | 5 | 5 | 5 | 6 | 1 | 0 | 0.0000 |
| 2 | 0.5842 | 0.5787 | 0.5722 | 2 | 3 | 6 | 7 | 0 | 0 | 0.0000 |
| 3 | 0.5758 | 0.5611 | 0.5457 | 5 | 3 | 3 | 6 | 1 | 0 | 0.0000 |
| 4 | 0.5862 | 0.5417 | 0.5541 | 4 | 2 | 2 | 2 | 5 | 0 | 0.0000 |
| 5 | 0.5880 | 0.5622 | 0.5601 | 4 | 2 | 2 | 6 | 0 | 0 | 0.0601 |

### Aggregated

| Metric | A | B | C |
|--------|---|---|---|
| Mean BAcc | 0.5791 | 0.5610 | 0.5562 |
| Total Prior Violations | 20 | 15 | 18 |

### Prior Violation Diagnostics

| Metric | Value |
|--------|-------|
| Prior violations (B) | 15 |
| Repeated prior violations (B) | 2 |
| B reduces violations vs A | true |
| B beats C on violations | true |
| Eligible updates | 34 |
| Ineligible updates | 0 |

### Memory Diagnostics

| Metric | Value |
|--------|-------|
| Risk weights nonzero | 72 |
| Max risk weight | 1.1026 |
| Top risk features overlap deceptive | true |

### Top Risk Features

| Feature | Goal | Action | Risk | VWith | NVWith | VWithout | NVWithout | ActV |
|---------|------|--------|------|-------|--------|----------|-----------|------|
| grayish | need_tool | use_as_tool | 1.1026 | 3 | 1 | 1 | 7 | 2 |
| elongated_with_handle | need_food | eat | 0.9150 | 4 | 1 | 1 | 5 | 3 |
| greenish | need_tool | use_as_tool | 0.8531 | 2 | 1 | 2 | 7 | 2 |
| has_stem_remnant | need_tool | use_as_tool | 0.8531 | 2 | 1 | 2 | 7 | 2 |
| has_granular_surface | need_tool | use_as_tool | 0.8531 | 2 | 1 | 2 | 7 | 2 |
| has_bark_texture | need_fuel | burn_as_fuel | 0.6609 | 3 | 1 | 1 | 4 | 2 |
| greenish | need_stone | mine_with_pickaxe | 0.5992 | 2 | 1 | 2 | 5 | 2 |
| has_stem_remnant | need_stone | mine_with_pickaxe | 0.5992 | 2 | 1 | 2 | 5 | 2 |
| recently_seen | need_stone | mine_by_hand | 0.5816 | 4 | 1 | 1 | 3 | 3 |
| greenish | need_food | eat | 0.5107 | 4 | 2 | 1 | 4 | 3 |
| heavy_weight | need_food | eat | 0.5107 | 4 | 2 | 1 | 4 | 3 |
| rough_texture | need_food | eat | 0.5107 | 4 | 2 | 1 | 4 | 3 |
| has_stem_remnant | need_fuel | burn_as_fuel | 0.4581 | 2 | 1 | 2 | 4 | 2 |
| light_weight | need_fuel | burn_as_fuel | 0.4581 | 2 | 1 | 2 | 4 | 2 |
| has_wood_grain | need_fuel | burn_as_fuel | 0.4581 | 2 | 1 | 2 | 4 | 2 |
| has_crystal_flecks | need_fuel | burn_as_fuel | 0.4581 | 2 | 1 | 2 | 4 | 2 |
| has_granular_surface | need_fuel | burn_as_fuel | 0.4581 | 2 | 1 | 2 | 4 | 2 |
| on_left_side | need_stone | mine_with_pickaxe | 0.4423 | 3 | 2 | 1 | 4 | 2 |
| has_granular_surface | need_stone | mine_by_hand | 0.4377 | 3 | 1 | 2 | 3 | 3 |
| has_peel_texture | need_tool | use_as_tool | 0.4266 | 1 | 1 | 3 | 7 | 2 |
## 5. Mode: Epsilon-Boundary Explore

Explore fraction: 0.3  |  Objects: 60

### Per-Episode

| Ep | A bal | B bal | C bal | A pv | B pv | C pv | B eff | B zero | B harm | B riskAdj |
|----|-------|-------|-------|------|------|------|-------|--------|--------|-----------|
| 1 | 0.6071 | 0.6008 | 0.6175 | 5 | 7 | 3 | 7 | 0 | 0 | 0.0000 |
| 2 | 0.5992 | 0.5763 | 0.5858 | 3 | 3 | 3 | 3 | 2 | 0 | 0.0000 |
| 3 | 0.6096 | 0.5913 | 0.5867 | 2 | 2 | 2 | 3 | 3 | 0 | 0.0636 |
| 4 | 0.6000 | 0.5742 | 0.5875 | 4 | 2 | 4 | 5 | 1 | 0 | 0.0426 |
| 5 | 0.5962 | 0.5858 | 0.5892 | 5 | 1 | 0 | 3 | 3 | 0 | 0.2702 |

### Aggregated

| Metric | A | B | C |
|--------|---|---|---|
| Mean BAcc | 0.6024 | 0.5857 | 0.5933 |
| Total Prior Violations | 19 | 15 | 12 |

### Prior Violation Diagnostics

| Metric | Value |
|--------|-------|
| Prior violations (B) | 15 |
| Repeated prior violations (B) | 1 |
| B reduces violations vs A | true |
| B beats C on violations | false |
| Eligible updates | 30 |
| Ineligible updates | 0 |

### Memory Diagnostics

| Metric | Value |
|--------|-------|
| Risk weights nonzero | 42 |
| Max risk weight | 0.7562 |
| Top risk features overlap deceptive | true |

### Top Risk Features

| Feature | Goal | Action | Risk | VWith | NVWith | VWithout | NVWithout | ActV |
|---------|------|--------|------|-------|--------|----------|-----------|------|
| has_stem_remnant | need_fuel | burn_as_fuel | 0.7562 | 4 | 1 | 1 | 4 | 3 |
| has_peel_texture | need_fuel | burn_as_fuel | 0.5992 | 3 | 1 | 2 | 4 | 3 |
| rough_texture | need_fuel | burn_as_fuel | 0.5992 | 3 | 1 | 2 | 4 | 3 |
| smooth_texture | need_food | eat | 0.4581 | 5 | 1 | 1 | 2 | 4 |
| has_peel_texture | need_planks | craft_plank | 0.3891 | 4 | 1 | 1 | 2 | 3 |
| has_stem_remnant | need_planks | craft_plank | 0.3891 | 4 | 1 | 1 | 2 | 3 |
| movable | need_planks | craft_plank | 0.3891 | 4 | 1 | 1 | 2 | 3 |
| on_left_side | need_planks | craft_plank | 0.3891 | 4 | 1 | 1 | 2 | 3 |
| has_shaft_shape | need_fuel | burn_as_fuel | 0.3781 | 2 | 1 | 3 | 4 | 3 |
| has_crystal_flecks | need_fuel | burn_as_fuel | 0.3781 | 2 | 1 | 3 | 4 | 3 |
| heavy_weight | need_food | eat | 0.3466 | 4 | 1 | 2 | 2 | 4 |
| on_left_side | need_food | eat | 0.3466 | 4 | 1 | 2 | 2 | 4 |
| light_weight | need_planks | craft_plank | 0.2612 | 3 | 1 | 2 | 2 | 3 |
| near_table | need_planks | craft_plank | 0.2612 | 3 | 1 | 2 | 2 | 3 |
| has_stem_remnant | need_tool | use_as_tool | 0.2612 | 3 | 1 | 2 | 2 | 3 |
| block_like | need_planks | craft_plank | 0.2612 | 3 | 1 | 2 | 2 | 3 |
| brownish | need_planks | craft_plank | 0.2612 | 3 | 1 | 2 | 2 | 3 |
| long_shape | need_planks | craft_plank | 0.2612 | 3 | 1 | 2 | 2 | 3 |
| heavy_weight | need_tool | use_as_tool | 0.2612 | 3 | 1 | 2 | 2 | 3 |
| recently_seen | need_fuel | burn_as_fuel | 0.2212 | 3 | 2 | 2 | 3 | 3 |
## 6. Cross-Mode Interpretation

| Criterion | Value |
|-----------|-------|
| Mode 1 prior violations | 11 |
| Mode 2 prior violations | 15 |
| Mode 3 prior violations | 15 |
| Deceptive visible deviation valid | true |
| Top risk features include deceptive | true |
| B reduces repeated prior violations | true |
| B beats permuted control | true |
| Feature-risk memory supported | true |
| Hard exclusion used | false |
| Ready for full 1J19 | true |
| Next route | run_small_multiseed_1j19c |

## 7. Summary

```
[block_done]
block_id=1J19c
prior_violation_threshold=0.6
mode_original_prior_violation_count=11
mode_deceptive_prior_violation_count=15
top_risk_features_include_deceptive_features=true
B_reduces_repeated_prior_violations=true
B_beats_permuted_control=true
feature_risk_memory_supported=true
hard_exclusion_used=false
ready_for_full_1j19=true
next_recommended_route=run_small_multiseed_1j19c
elapsed=39.0s
```