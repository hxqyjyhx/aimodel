# Block 1J40b-env2: Role-Deconfounded Observe-Value Environment

- **New Condition**: C6_observe_try_counterfactual_v0
- **Seeds**: [101, 103, 107, 109, 113]
- **Total elapsed**: 2.3s
- **Implementation Status**: PASS

## Summary

Deconfounds visible features from group_role using deterministic profile rotation.
No direct role-marker features. Every variant visible feature appears across all 3 roles.

- total_objects (all seeds): 180
- total_events (all seeds): 1320
- observe_events: 240
- try_events: 1080

## Design: Deterministic Profile Rotation

3 visual profiles (A/B/C) per category. Each profile has 4 of 6 variant features.
Within each (seed, category), all 3 roles share the same profile.
Profiles vary across seeds and categories for cross-seed feature variation.

### Seed-Category-Profile Assignment

| Seed | wood_log | stone_block | apple | wooden_pickaxe |
|------|----------|-------------|-------|---------------|
| 101 | A | B | C | A |
| 103 | B | C | A | B |
| 107 | C | A | B | C |
| 109 | A | C | B | A |
| 113 | B | A | C | B |

## Feature-Role Balance

- features_in_all_roles: 24
- features_in_2_roles: 0
- features_in_1_role: 0
- max_role_gap_per_feature: 0.0
- direct_role_marker_detected: False

### P(role | feature)

| Feature | helps | neutral | wasteful | max_gap |
|---------|-------|---------|----------|--------|
| banded | 0.333 | 0.333 | 0.333 | 0.000 |
| chalky | 0.333 | 0.333 | 0.333 | 0.000 |
| cracked_ends | 0.333 | 0.333 | 0.333 | 0.000 |
| dark_colored | 0.333 | 0.333 | 0.333 | 0.000 |
| glassy | 0.333 | 0.333 | 0.333 | 0.000 |
| glossy_surface | 0.333 | 0.333 | 0.333 | 0.000 |
| golden_flesh | 0.333 | 0.333 | 0.333 | 0.000 |
| has_knot_hole | 0.333 | 0.333 | 0.333 | 0.000 |
| has_wood_grain | 0.333 | 0.333 | 0.333 | 0.000 |
| light_colored | 0.333 | 0.333 | 0.333 | 0.000 |
| mossy_surface | 0.333 | 0.333 | 0.333 | 0.000 |
| notched_shaft | 0.333 | 0.333 | 0.333 | 0.000 |
| pitted | 0.333 | 0.333 | 0.333 | 0.000 |
| polished_head | 0.333 | 0.333 | 0.333 | 0.000 |
| red_blush | 0.333 | 0.333 | 0.333 | 0.000 |
| reinforced_joint | 0.333 | 0.333 | 0.333 | 0.000 |
| russet_skin | 0.333 | 0.333 | 0.333 | 0.000 |
| speckled | 0.333 | 0.333 | 0.333 | 0.000 |
| spotted | 0.333 | 0.333 | 0.333 | 0.000 |
| striped | 0.333 | 0.333 | 0.333 | 0.000 |
| tapered_end | 0.333 | 0.333 | 0.333 | 0.000 |
| veined | 0.333 | 0.333 | 0.333 | 0.000 |
| weathered_handle | 0.333 | 0.333 | 0.333 | 0.000 |
| wrapped_grip | 0.333 | 0.333 | 0.333 | 0.000 |

## Role Probe

- full_feature_probe_accuracy: 0.3333
- mean_seed_accuracy: 0.3333
- std_seed_accuracy: 0.0000
- max_seed_accuracy: 0.3333
- chance_baseline: 0.333
- threshold: 0.45
- feature_pairs_exclusive: 0

## Expected Observe Value by Subtype

| Subtype | No-Obs | One-Obs | Obs Adv |
|---------|--------|---------|--------|
| apple|profile_A|observe_helps | 0.150 | 0.250 | +0.100 |
| apple|profile_A|observe_neutral | 0.150 | 0.150 | +0.000 |
| apple|profile_A|observe_wasteful | 0.150 | 0.150 | +0.000 |
| apple|profile_B|observe_helps | 0.150 | 0.250 | +0.100 |
| apple|profile_B|observe_neutral | 0.150 | 0.150 | +0.000 |
| apple|profile_B|observe_wasteful | 0.150 | 0.150 | +0.000 |
| apple|profile_C|observe_helps | 0.150 | 0.250 | +0.100 |
| apple|profile_C|observe_neutral | 0.150 | 0.150 | +0.000 |
| apple|profile_C|observe_wasteful | 0.150 | 0.150 | +0.000 |
| stone_block|profile_A|observe_helps | -0.050 | 0.050 | +0.100 |
| stone_block|profile_A|observe_neutral | -0.050 | -0.050 | +0.000 |
| stone_block|profile_A|observe_wasteful | -0.050 | -0.050 | +0.000 |
| stone_block|profile_B|observe_helps | -0.050 | 0.050 | +0.100 |
| stone_block|profile_B|observe_neutral | -0.050 | -0.050 | +0.000 |
| stone_block|profile_B|observe_wasteful | -0.050 | -0.050 | +0.000 |
| stone_block|profile_C|observe_helps | -0.050 | 0.050 | +0.100 |
| stone_block|profile_C|observe_neutral | -0.050 | -0.050 | +0.000 |
| stone_block|profile_C|observe_wasteful | -0.050 | -0.050 | +0.000 |
| wood_log|profile_A|observe_helps | 0.250 | 0.350 | +0.100 |
| wood_log|profile_A|observe_neutral | 0.250 | 0.250 | +0.000 |
| wood_log|profile_A|observe_wasteful | 0.250 | 0.250 | +0.000 |
| wood_log|profile_B|observe_helps | 0.250 | 0.350 | +0.100 |
| wood_log|profile_B|observe_neutral | 0.250 | 0.250 | +0.000 |
| wood_log|profile_B|observe_wasteful | 0.250 | 0.250 | +0.000 |
| wood_log|profile_C|observe_helps | 0.250 | 0.350 | +0.100 |
| wood_log|profile_C|observe_neutral | 0.250 | 0.250 | +0.000 |
| wood_log|profile_C|observe_wasteful | 0.250 | 0.250 | +0.000 |
| wooden_pickaxe|profile_A|observe_helps | -0.050 | 0.050 | +0.100 |
| wooden_pickaxe|profile_A|observe_neutral | -0.050 | -0.050 | +0.000 |
| wooden_pickaxe|profile_A|observe_wasteful | -0.050 | -0.050 | +0.000 |
| wooden_pickaxe|profile_B|observe_helps | -0.050 | 0.050 | +0.100 |
| wooden_pickaxe|profile_B|observe_neutral | -0.050 | -0.050 | +0.000 |
| wooden_pickaxe|profile_B|observe_wasteful | -0.050 | -0.050 | +0.000 |
| wooden_pickaxe|profile_C|observe_helps | -0.050 | 0.050 | +0.100 |
| wooden_pickaxe|profile_C|observe_neutral | -0.050 | -0.050 | +0.000 |
| wooden_pickaxe|profile_C|observe_wasteful | -0.050 | -0.050 | +0.000 |

## Post-Observe Improvement

- pre-observe best-action mean success: 1.0000
- post-observe best-action mean success: 1.0000
- improvement: +0.0000
- pre-observe mean try return: 0.0750
- post-observe mean try return: 0.1083

## Hidden Feature Timing

- no_observe hidden uses: 0
- one_observe hidden uses: 0
- repeated_observe hidden before reveal: 0
- hidden_timing_audit_passed: True

## Matched Advantage (Audit Only)

- observe_helps_advantage: 0.1
- observe_neutral_advantage: 0.0
- observe_wasteful_advantage: 0.0
- role_sanity_pass: True

## Acceptance Checks

| Check | Result |
|-------|--------|
| recursive_leakage_check_passed | **PASS** |
| direct_role_marker_detected = false | **PASS** |
| visible_feature_role_balance_passed | **PASS** |
| full_visible_feature_role_probe_accuracy <= 0.45 | **PASS** |
| mean_role_probe_accuracy <= 0.45 | **PASS** |
| max_seed_role_probe_accuracy <= 0.55 | **PASS** |
| feature_pairs_exclusive_to_one_role = 0 | **PASS** |
| hidden_timing_audit_passed | **PASS** |
| matched_advantage_role_sanity_passed | **PASS** |
| expected_observe_advantage_positive | **PASS** |
| post_observe_improvement_detected | **PASS** |
| post_observe_improvement_source_clean | **PASS** |
| delayed_credit_overattribution_risk = false | **PASS** |
| old C4 preserved | **PASS** |
| old C5 preserved | **PASS** |
| policy_decisions_changed = false | **PASS** |
| no_seed_crashes | **PASS** |
| no_observe_try_count > 0 | **PASS** |
| one_observe_try_count > 0 | **PASS** |
| repeated_observe_try_count > 0 | **PASS** |
| **overall_acceptance** | **PASS** |


```
[block_done]
block_id=1J40b-env2
role_probe_accuracy=0.3333
mean_role_probe_accuracy=0.3333
max_seed_role_probe_accuracy=0.3333
role_probe_threshold=0.45
direct_role_marker_detected=false
max_role_gap_per_feature=0.0
feature_pairs_exclusive_count=0
visible_feature_role_balance_passed=true
observe_helps_advantage=0.1
observe_neutral_advantage=0.0
observe_wasteful_advantage=0.0
expected_observe_advantage_positive=true
post_observe_improvement_detected=true
post_observe_source_clean=true
hidden_timing_audit_passed=true
recursive_leakage_check_passed=true
old_c4_preserved=true
old_c5_preserved=true
policy_decisions_changed=false
implementation_status=pass
failure_reason=none
```
