# Block 1J40b-env1-fix: Leakage-Fixed C6 Observe-Try Counterfactual Patch

- **New Condition**: C6_observe_try_counterfactual_v0
- **Total elapsed**: 0.0s
- **Implementation Status**: PASS

## Leakage Fix

- schedule_label_removed_from_agent_visible: True
- recursive_leakage_check_passed: True
- leakage_violations_found: 0

### Forbidden keys checked recursively: ['deceptive_flag', 'depth_schedule', 'full_object_state', 'group_id', 'group_role', 'hidden_subtype', 'mixed_source_type', 'oracle_outcome', 'preferred_explanation', 'prior_violation', 'schedule', 'schedule_type', 'true_family']

## Observe-Depth from Event History

Schedule labels (`no_observe`/`one_observe`/`repeated_observe`) are audit-only.
Agent-visible action_params contain only:
- `affordance`: try action name
- `observe_depth`: count of prior observes for this object (derived from event history)

## Event Summary

- total_objects: 36
- total_events: 264
- observe_events: 48
- try_events: 216

## Observe-Depth Schedules (Audit-Only)

- no_observe_try_count: 72
- one_observe_try_count: 72
- repeated_observe_try_count: 72
- matched_schedule_group_count: 12

### Try Success Rate by Observe Depth

| Depth | Try Events | Success Rate |
|-------|-----------|-------------|
| no_observe | 72 | 0.3889 |
| one_observe | 72 | 0.3889 |
| repeated_observe | 72 | 0.3889 |

## Comparable Action Sets

- total_situations: 36
- complete_action_set_count: 36
- partial_action_set_count: 0
- comparable_observe_try_set_count: 24

## Detailed Delayed Observe Credit

### Overall

- delayed_credit_records: 36
- delayed_observe_benefit_detectable: True
- mean_immediate_observe_value: 0.25
- mean_combined_observe_value_raw: 0.75
- mean_delayed_per_try: 0.0833

### Observe Advantage vs Matched No-Observe

- mean_advantage_vs_no_observe: 0.0
- mean_advantage_vs_one_observe: 0.0

### Role Breakdown

| Role | Count | Immediate | Delayed Raw | Delayed Discounted | Delayed Per-Try | Combined Raw |
|------|-------|-----------|-------------|-------------------|----------------|-------------|
| observe_helps | 12 | 0.25 | 0.6 | 0.4686 | 0.1 | 0.85 |
| observe_neutral | 12 | 0.25 | 0.45 | 0.4138 | 0.075 | 0.7 |
| observe_wasteful | 12 | 0.25 | 0.45 | 0.4138 | 0.075 | 0.7 |

### Role Advantage (Delayed Credit)

- observe_helps_advantage: 0.6
- observe_neutral_advantage: 0.45
- observe_wasteful_advantage: 0.45
- role_sanity_pass: True

## Overattribution Risk

- delayed_credit_overattribution_risk: False
- mean_delayed_per_try: 0.0833 vs try_max: 0.45
- note: Delayed per-try return is within single-try bounds. No overattribution.

## Value Scale Audit

| Value Type | Min | Max | Mean |
|-----------|-----|-----|------|
| observe_immediate | 0.225 | 0.305 | 0.25 |
| delayed_return_raw | -0.3 | 1.5 | 0.5 |
| delayed_return_discounted | -0.0926 | 1.1851 | 0.432 |
| delayed_return_normalized_per_try | -0.05 | 0.25 | 0.0833 |
| observe_combined_raw | -0.075 | 1.805 | 0.75 |
| observe_combined_discounted | 0.1324 | 1.4901 | 0.682 |
| try_value | -0.15 | 0.45 | 0.0833 |

- combined_observe_can_exceed_try (discounted): True

## Acceptance Checks

| Check | Result |
|-------|--------|
| schedule_label_removed_from_agent_visible | **PASS** |
| recursive_leakage_check_passed | **PASS** |
| no_observe_try_count > 0 | **PASS** |
| one_observe_try_count > 0 | **PASS** |
| repeated_observe_try_count > 0 | **PASS** |
| matched_schedule_group_count > 0 | **PASS** |
| comparable_observe_try_set_count > 0 | **PASS** |
| delayed_credit_available = true | **PASS** |
| delayed_credit_overattribution_risk reported | **PASS** |
| role_sanity_check | **PASS** |
| old C4 preserved | **PASS** |
| old C5 preserved | **PASS** |
| no policy decisions change | **PASS** |
| no_seed_crashes | **PASS** |
| **overall_acceptance** | **PASS** |


```
[block_done]
block_id=1J40b-env1-fix
schedule_label_removed_from_agent_visible=true
recursive_leakage_check_passed=true
no_observe_try_count=72
one_observe_try_count=72
repeated_observe_try_count=72
matched_schedule_group_count=12
complete_action_set_count=36
observe_helps_advantage=0.6
observe_neutral_advantage=0.45
observe_wasteful_advantage=0.45
delayed_credit_overattribution_risk=false
old_c4_preserved=true
old_c5_preserved=true
policy_decisions_changed=false
implementation_status=pass
failure_reason=none
```
