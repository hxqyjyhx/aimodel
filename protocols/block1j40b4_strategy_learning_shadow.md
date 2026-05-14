# Block 1J40b-4: Shadow Strategy Learner on env2 (Role-Deconfounded)

- **Condition**: C6_observe_try_counterfactual_v0
- **Environment**: env2_role_deconfounded
- **Seeds**: [101, 103, 107, 109, 113]
- **Total elapsed**: 15.3s
- **Implementation Status**: FAIL

## Summary

- total_records: 1320
- observe_records: 240
- try_records: 1080
- feature_space_size: 86

## Shadow Learner Design

- Per-action ridge regression on try events
- Feature space: visible features + hidden diagnostic features + state + numeric
- NO RoleSignal features (deconfounded from env2)
- Observe decision: observe if max(post_Q) - max(pre_Q) > observe_cost
- Pre-observe features: 6 ambient features only
- Post-observe features: all visible + hidden diagnostic features

## Input Fields Used by Learner

- Visible feature flags: all core + variant visible features
- Hidden diagnostic feature flags (only after observe)
- State feature flags: fresh, wet, damaged, clean, hot, open
- Numeric: prior_observe_count, n_features_known, n_states_known, episode_id, episode_progress

## Forbidden Fields NOT Used

- group_role, group_id, depth_schedule, schedule_type
- RoleSignal features (variable_texture, uniform_texture, worn_texture)
- All FORBIDDEN_AGENT_VISIBLE keys

## Leakage Checks

- recursive_leakage_check_passed: True
- audit_label_used_as_input: False
- leakage_violations: 0
- full_visible_role_probe_accuracy: 0.3333 (threshold 0.45)
- mean_loso_seed_probe: 0.3333
- max_seed_probe: 0.3333 (threshold 0.55)

## Baseline Comparison (Episode-Heldout)

| Baseline | Mean Return | Action Match Rate | Observe Rate |
|----------|------------|------------------|---------------|
| no_observe_pre_only | 0.45 | 0.4286 | 0.0 |
| random_observe | 0.45 | 0.4286 | 0.4857 |
| always_observe | 0.445 | 0.4286 | 1.0 |
| oracle_post_observe | 0.45 | 1.0 | 1.0 |
| **shadow_learner** | **0.4464** | **0.4286** | **0.7143** |

- delta_return (shadow - pre): -0.0036
- zero_gain_observe_rate: 1.0
- harmful_observe_rate: 1.0
- observe_collapse: False

### Model Quality

- heldout MSE: 0.043209
- heldout MAE: 0.102751
- MSE global baseline: 0.114702
- MSE per-action baseline: 0.07145
- ridge vs per-action delta: 0.028241
- ridge vs global delta: 0.071494

## Per-Seed Results (LOSO)

| Seed | MSE | Pre Ret | Post Ret | Shadow Ret | Oracle Ret | Delta | Obs Rate |
|------|-----|---------|----------|------------|------------|-------|----------|
| 101 | 0.018659 | 0.45 | 0.445 | 0.445 | 0.45 | -0.005 | 1.0 |
| 103 | 0.018633 | 0.45 | 0.445 | 0.4463 | 0.45 | -0.0037 | 0.75 |
| 107 | 0.021974 | 0.45 | 0.445 | 0.4463 | 0.45 | -0.0037 | 0.75 |
| 109 | 0.018182 | 0.45 | 0.445 | 0.4463 | 0.45 | -0.0037 | 0.75 |
| 113 | 0.018206 | 0.45 | 0.445 | 0.445 | 0.45 | -0.005 | 1.0 |

- avg MSE: 0.019131
- avg pre return: 0.45
- avg post return: 0.445
- avg shadow return: 0.4458
- avg oracle return: 0.45
- avg delta return: -0.0042
- avg observe rate: 0.85
- seeds with positive delta: 0/5

## Acceptance Checks

| Check | Result |
|-------|--------|
| recursive_leakage_check_passed | **PASS** |
| audit_label_used_as_input | **PASS** |
| full_visible_role_probe_accuracy <= 0.45 | **PASS** |
| max_seed_role_probe_accuracy <= 0.55 | **PASS** |
| shadow_post > pre (aggregate) | **FAIL** |
| shadow > random (aggregate) | **FAIL** |
| shadow > no_observe_pre (aggregate) | **FAIL** |
| delta_return > 0 in most seeds | **FAIL** |
| zero_gain_observe_rate reported | **PASS** |
| observe_policy_not_collapsed | **PASS** |
| hidden_timing_clean | **PASS** |
| no_direct_role_marker | **PASS** |
| no_role_label_as_input | **PASS** |
| policy_decisions_changed = false | **PASS** |
| environment_changed = false | **PASS** |
| no_seed_crashes | **PASS** |
| **overall_acceptance** | **FAIL** |


```
[block_done]
block_id=1J40b-4
condition=C6_observe_try_counterfactual_v0
environment=env2_role_deconfounded
total_records_used=1320
observe_records_used=240
try_records_used=1080
heldout_mse=0.043209
heldout_mae=0.102751
pre_mean_return=0.45
random_mean_return=0.45
post_mean_return=0.445
oracle_mean_return=0.45
shadow_mean_return=0.4464
delta_return=-0.0036
shadow_obs_rate=0.7143
zero_gain_observe_rate=1.0
harmful_observe_rate=1.0
observe_collapse=false
full_visible_role_probe_accuracy=0.3333333333333333
max_seed_role_probe=0.3333333333333333
audit_label_used_as_input=false
recursive_leakage_check_passed=true
implementation_status=fail
failure_reason=shadow_post > pre (aggregate); shadow > random (aggregate); shadow > no_observe_pre (aggregate); delta_return > 0 in most seeds
```
