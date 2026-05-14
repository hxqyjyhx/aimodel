# Block 1J40b-env1: Observe-Try Counterfactual Environment Patch

- **New Condition**: C6_observe_try_counterfactual_v0
- **Total elapsed**: 0.0s
- **Implementation Status**: PASS

## Summary

- total_objects: 36
- total_events: 264
- observe_events: 48
- try_events: 216

## Observe-Depth Schedules

- no_observe_try_count: 72
- one_observe_try_count: 72
- repeated_observe_try_count: 72
- observe_then_try_count: 144
- matched_schedule_group_count: 12

### Per-Schedule Object Counts

| Depth Schedule | Unique Objects |
|---------------|---------------|
| no_observe | 12 |
| one_observe | 12 |
| repeated_observe | 12 |

### Try Success Rate by Observe Depth

| Depth | Try Events | Success Rate |
|-------|-----------|-------------|
| no_observe | 72 | 0.3889 |
| one_observe | 72 | 0.3889 |
| repeated_observe | 72 | 0.3889 |

### Try Success Rate by Schedule and Role

| Schedule | Role | Count | Success Rate |
|----------|------|-------|-------------|
| no_observe | observe_helps | 24 | 0.4167 |
| no_observe | observe_neutral | 24 | 0.375 |
| no_observe | observe_wasteful | 24 | 0.375 |
| one_observe | observe_helps | 24 | 0.4167 |
| one_observe | observe_neutral | 24 | 0.375 |
| one_observe | observe_wasteful | 24 | 0.375 |
| repeated_observe | observe_helps | 24 | 0.4167 |
| repeated_observe | observe_neutral | 24 | 0.375 |
| repeated_observe | observe_wasteful | 24 | 0.375 |

## Comparable Action Sets

- total_situations: 36
- complete_action_set_count: 36
- partial_action_set_count: 0
- comparable_observe_try_set_count: 24

### Action Set Size Distribution

| Actions per Situation | Count |
|----------------------|-------|
| 6 | 36 |

## Delayed Observe Credit

- delayed_credit_available: True
- delayed_credit_records: 48
- mean_immediate_observe_value: 0.26
- mean_delayed_return: 0.3625
- mean_combined_observe_value: 0.6225
- delayed_observe_benefit_detectable: True

## Value Scale Audit

| Value Type | Min | Max | Mean |
|-----------|-----|-----|------|
| observe_immediate | 0.225 | 0.305 | 0.26 |
| observe_delayed | -0.45 | 1.5 | 0.3625 |
| observe_combined | -0.165 | 1.805 | 0.6225 |
| try_value | -0.15 | 0.45 | 0.0833 |

- combined_observe_can_exceed_try: True

## Group Role Distribution

| Role | Description | Object Count |
|------|-------------|-------------|
| observe_helps | Hidden features reveal affordance changes | 12 |
| observe_neutral | Same affordance regardless of observe depth | 12 |
| observe_wasteful | Observe cost exceeds benefit | 12 |

## Audit Checks

| Check | Result |
|-------|--------|
| audit_label_leakage_detected | PASS |
| old_c4_preserved | PASS |
| old_c5_preserved | PASS |
| policy_decisions_changed | PASS |
| no_seed_crashes | PASS |

## Acceptance Checks

| Check | Result |
|-------|--------|
| no_observe_try_count > 0 | **PASS** |
| one_observe_try_count > 0 | **PASS** |
| repeated_observe_try_count > 0 | **PASS** |
| matched_schedule_group_count > 0 | **PASS** |
| comparable_observe_try_set_count > 0 | **PASS** |
| delayed_credit_available = true | **PASS** |
| audit labels not agent-visible | **PASS** |
| old C4 preserved | **PASS** |
| old C5 preserved | **PASS** |
| no policy decisions change | **PASS** |
| no_seed_crashes | **PASS** |
| **overall_acceptance** | **PASS** |


```
[block_done]
block_id=1J40b-env1
new_condition=C6_observe_try_counterfactual_v0
no_observe_try_count=72
one_observe_try_count=72
repeated_observe_try_count=72
matched_schedule_group_count=12
comparable_observe_try_set_count=24
complete_action_set_count=36
partial_action_set_count=0
delayed_credit_available=true
delayed_observe_benefit_detectable=true
audit_label_leakage_detected=false
old_c4_preserved=true
old_c5_preserved=true
policy_decisions_changed=false
implementation_status=pass
failure_reason=none
```
