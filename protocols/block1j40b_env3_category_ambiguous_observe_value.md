# Block 1J40b-env3: Category-Ambiguous Observe-Value Environment

- **Implementation Status**: PASS
- **Seeds**: [101, 103, 107, 109, 113]
- **Total elapsed**: 23.8s
- **Total objects**: 180

## 1. Checkpoint Summary

env3 creates category ambiguity in pre-observe features by grouping categories into ambient groups that share identical ambient feature signatures. Post-observe reveals core features (identify category) and diagnostic features (action-relevant evidence). Base affordance profiles were modified so no single action succeeds for both categories within an ambient group, creating genuine pre-observe uncertainty.

## 2. Env3a Generation Design

### Ambient Groups

- **Group A** (wood_log, stone_block): brownish, rough_texture, long_shape, heavy_weight
- **Group B** (apple, wooden_pickaxe): greenish, smooth_texture, round_small, light_weight

Within each group, all objects share identical ambient features. The agent cannot distinguish between categories pre-observe.

### Feature Layers

| Layer | Visibility | Purpose |
|-------|-----------|--------|
| Ambient | no_observe | Shared within ambient group, creates ambiguity |
| Core | one_observe+ | Identifies category uniquely |
| Diagnostic | one_observe+ | Action-relevant evidence, some features shared across categories |
| Hidden | repeated_observe | Hints at variant bonus for helps objects |

### Core Features per Category

- **wood_log**: has_bark_texture, fibrous, flammable, porous_surface
- **stone_block**: has_crystal_flecks, has_granular_surface, block_like, cold_to_touch, scratch_resistant
- **apple**: has_stem_remnant, has_peel_texture, fruity_scent
- **wooden_pickaxe**: has_grip_area, has_shaft_shape, elongated_with_handle, movable, has_metal_head, jointed

### Diagnostic Features per Category

- **wood_log**: has_fibrous_inside, has_burnable_fibers, is_hollow, has_rough_surface
- **stone_block**: has_hard_surface, has_crystal_fragments, is_solid, is_fragile
- **apple**: has_edible_smell, is_soft_inside, is_fragile, has_fibrous_inside
- **wooden_pickaxe**: has_grip_shape, has_tool_edge, is_balanced, has_leverage

## 3. Difference from env2

### Affordance Profile Changes

| Category | Action | Change | Reason |
|----------|--------|--------|--------|
| wood_log | mine_with_pickaxe | success -> fail | Removes common safe action within ambient group |
| apple | mine_with_pickaxe | success -> fail | Removes common safe action within ambient group |

### Feature System Changes

- Ambient features no longer identify category (shared within groups)
- Core + diagnostic features replace env2's single visible feature set
- Diagnostic features are action-relevant (has_edible_smell, etc.) not visual variants
- Profile-rotation is simplified to bonus-action-only (no variant visible features)
- Hidden features are distinct from diagnostic features

## 4. Pre-observe Ambiguity Audit

- oracle mean true return: 0.4500
- pre mean true return (randomized tie): 0.1711
- pre mean true return (fixed tie-break): 0.1700
- pre/oracle gap (randomized): 0.2789
- pre/oracle gap (fixed): 0.2800

### Per-seed Pre vs Oracle

| Seed | Pre (rand) | Oracle | Gap |
|------|-----------|--------|-----|
| 101 | 0.1694 | 0.4500 | 0.2806 |
| 103 | 0.1583 | 0.4500 | 0.2917 |
| 107 | 0.1889 | 0.4500 | 0.2611 |
| 109 | 0.1806 | 0.4500 | 0.2694 |
| 113 | 0.1583 | 0.4500 | 0.2917 |

## 5. Category Probes by Visibility Level

| Level | Accuracy | Mean Seed | Max Seed | N Features |
|-------|----------|-----------|----------|------------|
| ambient | 0.5000 | 0.5000 | 0.5000 | 8 |
| core | 1.0000 | 1.0000 | 1.0000 | 18 |
| diagnostic | 1.0000 | 1.0000 | 1.0000 | 14 |
| core_plus_diagnostic | 1.0000 | 1.0000 | 1.0000 | 32 |
| all_visible | 1.0000 | 1.0000 | 1.0000 | 40 |

## 6. Post-observe Source Ablations

| Condition | Mean Return | Gain vs Pre | Oracle Gap |
|-----------|-------------|-------------|------------|
| pre_ambient_only | 0.1711 | +0.0000 | 0.2789 |
| post_core_only | 0.4500 | +0.2789 | 0.0000 |
| post_diagnostic_only | 0.4500 | +0.2789 | 0.0000 |
| post_full | 0.4500 | +0.2789 | 0.0000 |

### Per-seed Breakdown

| Seed | Pre | Core | Diagnostic | Full | Oracle |
|------|-----|------|------------|------|--------|
| 101 | 0.1694 | 0.4500 | 0.4500 | 0.4500 | 0.4500 |
| 103 | 0.1583 | 0.4500 | 0.4500 | 0.4500 | 0.4500 |
| 107 | 0.1889 | 0.4500 | 0.4500 | 0.4500 | 0.4500 |
| 109 | 0.1806 | 0.4500 | 0.4500 | 0.4500 | 0.4500 |
| 113 | 0.1583 | 0.4500 | 0.4500 | 0.4500 | 0.4500 |

## 7. Role Leakage Audits

- full_visible_role_probe_accuracy: 0.3333
- mean_seed_role_probe: 0.3333
- max_seed_role_probe: 0.3333
- single_feature_high_count: 0
- feature_pairs_exclusive: 0
- direct_role_marker_detected: False
- hidden_timing_audit_passed: True
- recursive_leakage_check_passed: True

## 8. Pre-observe Tie Diagnostics

### Group A (wood_log, stone_block)

| Action | Expected Return | wood_log | stone_block |
|--------|----------------|-------|-------|
| burn_as_fuel | +0.15 | +0.45 | -0.15 |
| craft_plank | +0.15 | +0.45 | -0.15 |
| eat | -0.15 | -0.15 | -0.15 |
| mine_by_hand | +0.15 | +0.45 | -0.15 |
| mine_with_pickaxe | +0.15 | -0.15 | +0.45 |
| use_as_tool | -0.15 | -0.15 | -0.15 |

- Best action(s): ['burn_as_fuel', 'craft_plank', 'mine_by_hand', 'mine_with_pickaxe']
- Tie count: 4
- Fixed tie-break (alphabetical): burn_as_fuel
- Randomized-tie pre return: mean=0.1700, std=0.1453

### Group B (apple, wooden_pickaxe)

| Action | Expected Return | apple | wooden_pickaxe |
|--------|----------------|-------|-------|
| burn_as_fuel | -0.15 | -0.15 | -0.15 |
| craft_plank | -0.15 | -0.15 | -0.15 |
| eat | +0.15 | +0.45 | -0.15 |
| mine_by_hand | +0.15 | +0.45 | -0.15 |
| mine_with_pickaxe | -0.15 | -0.15 | -0.15 |
| use_as_tool | +0.15 | -0.15 | +0.45 |

- Best action(s): ['eat', 'mine_by_hand', 'use_as_tool']
- Tie count: 3
- Fixed tie-break (alphabetical): eat
- Randomized-tie pre return: mean=0.1722, std=0.1143

## 9. Distinguish Metrics

- mean_all_try_return_pre: 0.0250
- mean_all_try_return_post: 0.0583
- best_action_selected_return_pre: 0.1711
- best_action_selected_return_post: 0.4500

**Note**: mean-all-try-return is NOT the same as best-action-selected return. The env2 metric used mean-all-try-return. env3 reports both explicitly.

## 10. Possible Observe Gain Summary

- pre mean true return: 0.1711
- post mean true return: 0.4500
- oracle mean true return: 0.4500
- pre/oracle gap: 0.2789
- post/oracle gap: 0.0000
- post-pre gain: +0.2789
- positive gain rate: 0.9667
- best action change rate: 1.0000

## 11. Acceptance Checks

| Check | Result |
|-------|--------|
| recursive_leakage_check_passed | **PASS** |
| direct_role_marker_detected = false | **PASS** |
| full_visible_role_probe_accuracy <= 0.45 | **PASS** |
| mean_seed_role_probe_accuracy <= 0.45 | **PASS** |
| max_seed_role_probe_accuracy <= 0.55 | **PASS** |
| single_feature_role_probe_clean | **PASS** |
| feature_pairs_exclusive_to_one_role = 0 | **PASS** |
| hidden_timing_audit_passed | **PASS** |
| old C4 preserved | **PASS** |
| old C5 preserved | **PASS** |
| no_seed_crashes | **PASS** |
| schedule_label_removed_all_seeds | **PASS** |
| category_probe_ambient < 1.0 | **PASS** |
| category_probe_ambient <= 0.70 | **PASS** |
| category_probe_core = 1.0 | **PASS** |
| pre_oracle_gap > 0 | **PASS** |
| post_mean > pre_mean | **PASS** |
| positive_gain_rate > 0 | **PASS** |
| post_oracle_gap < pre_oracle_gap | **PASS** |
| post_pre_gain >= +0.03 | **PASS** |
| positive_gain_rate >= 0.25 | **PASS** |

**Overall**: PASS


```
[block_done]
block_id=1J40b-env3
implementation_status=pass
category_probe_ambient=0.5000
category_probe_core=1.0000
pre_oracle_gap_randomized=0.2789
post_pre_gain=0.2789
positive_gain_rate=0.9667
full_role_probe_accuracy=0.3333
max_seed_role_probe=0.3333
hidden_timing_audit_passed=true
recursive_leakage_check_passed=true
failure_reasons=none
```
