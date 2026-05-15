# Block 1J40b-env1-fix2: Delayed Credit Advantage Correction

- **New Condition**: C6_observe_try_counterfactual_v0
- **Total elapsed**: 0.0s
- **Implementation Status**: PASS

## Summary

Fixes delayed observe credit: value = matched advantage over no-observe baseline,
not raw accumulated later try rewards.

- total_objects: 36
- total_events: 264
- observe_events: 48
- try_events: 216

## Leakage Fix (Preserved)

- schedule_label_removed_from_agent_visible: True
- recursive_leakage_check_passed: True
- leakage_violations_found: 0

## Observe-Depth Schedules (Audit-Only)

- no_observe_try_count: 72
- one_observe_try_count: 72
- repeated_observe_try_count: 72
- matched_schedule_group_count: 12

### Try Success Rate by Observe Depth

| Depth | Try Events | Success Rate |
|-------|-----------|-------------|
| no_observe | 72 | 0.375 |
| one_observe | 72 | 0.4306 |
| repeated_observe | 72 | 0.4306 |

### Try Success Rate by Depth and Role

| Depth | Role | N | Success Rate |
|-------|------|---|-------------|
| no_observe | observe_helps | 24 | 0.375 |
| no_observe | observe_neutral | 24 | 0.375 |
| no_observe | observe_wasteful | 24 | 0.375 |
| one_observe | observe_helps | 24 | 0.5417 |
| one_observe | observe_neutral | 24 | 0.375 |
| one_observe | observe_wasteful | 24 | 0.375 |
| repeated_observe | observe_helps | 24 | 0.5417 |
| repeated_observe | observe_neutral | 24 | 0.375 |
| repeated_observe | observe_wasteful | 24 | 0.375 |

## Matched Observe Advantage (Corrected)

Advantage = try return of observed objects - try return of matched no-observe baseline.

### Per-Group Matched Advantage

| Group | Role | No-Obs Baseline | One-Obs Return | One-Obs Adv | Rep-Obs Return | Rep-Obs Adv |
|-------|------|----------------|---------------|------------|---------------|------------|
| apple_g0 | observe_helps | 0.15 | 0.25 | 0.1 | 0.25 | 0.1 |
| apple_g1 | observe_neutral | 0.15 | 0.15 | 0.0 | 0.15 | 0.0 |
| apple_g2 | observe_wasteful | 0.15 | 0.15 | 0.0 | 0.15 | 0.0 |
| stone_block_g0 | observe_helps | -0.05 | 0.05 | 0.1 | 0.05 | 0.1 |
| stone_block_g1 | observe_neutral | -0.05 | -0.05 | 0.0 | -0.05 | 0.0 |
| stone_block_g2 | observe_wasteful | -0.05 | -0.05 | 0.0 | -0.05 | 0.0 |
| wood_log_g0 | observe_helps | 0.25 | 0.35 | 0.1 | 0.35 | 0.1 |
| wood_log_g1 | observe_neutral | 0.25 | 0.25 | 0.0 | 0.25 | 0.0 |
| wood_log_g2 | observe_wasteful | 0.25 | 0.25 | 0.0 | 0.25 | 0.0 |
| wooden_pickaxe_g0 | observe_helps | -0.05 | 0.05 | 0.1 | 0.05 | 0.1 |
| wooden_pickaxe_g1 | observe_neutral | -0.05 | -0.05 | 0.0 | -0.05 | 0.0 |
| wooden_pickaxe_g2 | observe_wasteful | -0.05 | -0.05 | 0.0 | -0.05 | 0.0 |

### Role Mean Matched Advantage

- observe_helps_advantage: 0.1
- observe_neutral_advantage: 0.0
- observe_wasteful_advantage: 0.0

## Delayed Credit Records (Raw + Matched Baseline)

- delayed_credit_records: 36
- mean_immediate: 0.25
- mean_matched_advantage: 0.0333
- mean_delayed_per_try: 0.1083
- delayed_benefit_detectable: True

### Role Breakdown

| Role | Count | Immediate | Raw Delayed | Per-Try | Matched Adv |
|------|-------|-----------|-------------|---------|------------|
| observe_helps | 12 | 0.25 | 1.05 | 0.175 | 0.1 |
| observe_neutral | 12 | 0.25 | 0.45 | 0.075 | 0.0 |
| observe_wasteful | 12 | 0.25 | 0.45 | 0.075 | 0.0 |

## Overattribution Risk

- delayed_credit_overattribution_risk: False
- note: Matched advantage is consistent with per-try returns. No overattribution detected.
- raw_per_try vs matched_adv gap: 0.075
- neutral mean_matched_adv: 0.0
- wasteful mean_matched_adv: 0.0

## Comparable Action Sets

- total_situations: 36
- complete_action_set_count: 36
- partial_action_set_count: 0
- comparable_observe_try_set_count: 24

## Value Scale Audit

| Value Type | Min | Max | Mean |
|-----------|-----|-----|------|
| observe_immediate | 0.225 | 0.305 | 0.25 |
| delayed_return_raw | -0.3 | 2.1 | 0.65 |
| delayed_return_discounted | -0.0926 | 1.5788 | 0.5406 |
| delayed_return_normalized_per_try | -0.05 | 0.35 | 0.1083 |
| matched_advantage | 0.0 | 0.1 | 0.0333 |
| try_value | -0.15 | 0.45 | 0.0972 |

## Role Sanity (Threshold-Based)

- observe_helps_advantage >= 0.05: 0.1 -> PASS
- abs(observe_neutral_advantage) <= 0.05: 0.0 -> PASS
- observe_wasteful_advantage <= 0.05: 0.0 -> PASS
- role_sanity_pass: True

## Acceptance Checks

| Check | Result |
|-------|--------|
| schedule_label_removed_from_agent_visible | **PASS** |
| recursive_leakage_check_passed | **PASS** |
| no_observe_try_count > 0 | **PASS** |
| one_observe_try_count > 0 | **PASS** |
| repeated_observe_try_count > 0 | **PASS** |
| matched_schedule_group_count > 0 | **PASS** |
| matched_advantage_available | **PASS** |
| comparable_observe_try_set_count > 0 | **PASS** |
| delayed_credit_available = true | **PASS** |
| observe_helps_advantage >= +0.05 | **PASS** |
| abs(observe_neutral_advantage) <= 0.05 | **PASS** |
| observe_wasteful_advantage <= 0.05 | **PASS** |
| role_sanity (all three thresholds) | **PASS** |
| delayed_credit_overattribution_risk = false | **PASS** |
| old C4 preserved | **PASS** |
| old C5 preserved | **PASS** |
| no policy decisions change | **PASS** |
| no_seed_crashes | **PASS** |
| **overall_acceptance** | **PASS** |


```
[block_done]
block_id=1J40b-env1-fix2
observe_helps_advantage=0.1
observe_neutral_advantage=0.0
observe_wasteful_advantage=0.0
matched_advantage_available=true
delayed_credit_overattribution_risk=false
recursive_leakage_check_passed=true
schedule_label_removed_from_agent_visible=true
old_c4_preserved=true
old_c5_preserved=true
policy_decisions_changed=false
implementation_status=pass
failure_reason=none
```
