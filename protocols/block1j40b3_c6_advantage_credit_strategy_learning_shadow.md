# Block 1J40b-3: C6 Advantage-Credit Strategy Learning Shadow

- **Condition**: C6_observe_try_counterfactual_v0
- **Seeds**: [101, 103, 107, 109, 113]
- **Total elapsed**: 1.7s
- **Implementation Status**: PASS

## Summary

- total_records: 1320
- observe_records: 240
- try_records: 1080
- switch_records: 0
- proxy_records_used: 0

## Data

Uses corrected C6 environment from 1J40b-env1-fix2:
- observe net_value = immediate + matched_advantage (per-try adv * n_later_tries)
- Role-signaling visible features allow model to distinguish group_role
- Situation feature space: 54 features

## Leakage Check

- recursive_leakage_check_passed: True
- audit_label_used_as_input: False
- leakage_violations: 0

## Episode-Heldout Evaluation

- n_train: 1065
- n_test: 255
- MSE: 0.026648
- MAE: 0.094173
- MSE global: 0.106396
- MSE per-action: 0.07615
- MSE action-type-only: 0.076182
- ridge vs per-action delta MSE: 0.049502
- ridge vs global delta MSE: 0.079748
- ridge vs ATO delta MSE: 0.049534

### Per-Action MSE

| Action | MSE |
|--------|-----|
| observe | 0.079524 |
| try_burn_as_fuel | 0.06649 |
| try_craft_plank | 0.0 |
| try_eat | 0.025388 |
| try_mine_by_hand | 4e-06 |
| try_mine_with_pickaxe | 1.9e-05 |
| try_use_as_tool | 5e-06 |

## Shadow Policy

- overall_observe_rate: 0.3725
- overall_try_rate: 0.6275
- n_observe_chosen: 95
- n_try_chosen: 160
- mean_predicted_observe_Q: 0.3988
- mean_predicted_best_try_Q: 0.4608

## Role-Based Shadow Analysis

| Role | Observe Rate | N Chosen | Mean Q Observe | Mean Q Best Try |
|------|-------------|----------|---------------|----------------|
| observe_helps | 0.8636 | 95/110 | 0.5565 | 0.4745 |
| observe_neutral | 0.0 | 0/80 | 0.2706 | 0.4504 |
| observe_wasteful | 0.0 | 0/65 | 0.2899 | 0.4505 |

- observe_rate_on_helps > neutral: True
- observe_rate_on_helps > wasteful: True

## LOSO Evaluation

| Heldout Seed | MSE | Observe Rate |
|-------------|-----|-------------|
| 101 | 0.013296 | 0.3636 |
| 103 | 0.013296 | 0.3636 |
| 107 | 0.013296 | 0.3636 |
| 109 | 0.013296 | 0.3636 |
| 113 | 0.013296 | 0.3636 |

- avg MSE: 0.013296
- avg observe_rate: 0.3636

## Top Action Match

- available: True
- rate: 0.5
- n_complete_sets: 10

## Acceptance Checks

| Check | Result |
|-------|--------|
| recursive_leakage_check_passed | **PASS** |
| audit_label_used_as_input | **PASS** |
| heldout_evaluation_exists | **PASS** |
| baseline_comparison_exists | **PASS** |
| top_action_match_available | **PASS** |
| observe_rate_on_helps > neutral | **PASS** |
| observe_rate_on_helps > wasteful | **PASS** |
| overall_observe_rate in (0.05, 0.95) | **PASS** |
| policy_decisions_changed = false | **PASS** |
| environment_changed = false | **PASS** |
| switch_q_values_created = false | **PASS** |
| proxy_records_used = 0 | **PASS** |
| all_records_have_source_ids | **PASS** |
| no_seed_crashes | **PASS** |
| **overall_acceptance** | **PASS** |


```
[block_done]
block_id=1J40b-3
condition=C6_observe_try_counterfactual_v0
total_records_used=1320
observe_records_used=240
try_records_used=1080
switch_records_used=0
proxy_records_used=0
heldout_mse=0.026648
heldout_mae=0.094173
ridge_vs_per_action_baseline_delta=0.049502
top_action_match_available=true
top_action_match_rate=0.5
overall_shadow_observe_rate=0.3725
observe_choice_rate_on_observe_helps=0.8636
observe_choice_rate_on_observe_neutral=0.0
observe_choice_rate_on_observe_wasteful=0.0
audit_label_used_as_input=false
recursive_leakage_check_passed=true
policy_decisions_changed=false
switch_q_values_created=false
implementation_status=pass
failure_reason=none
```
