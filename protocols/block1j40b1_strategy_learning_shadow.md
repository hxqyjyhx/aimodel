# Block 1J40b-1: Strategy Learning Shadow

- **Condition**: C5_mixed_source_identifiability_v0
- **Seeds**: [101, 103, 107, 109, 113]
- **Total elapsed**: 17.1s
- **Model**: Per-action ridge regression (alpha=1.0)

## Data Summary

- total_records_used: 2625
- observe_records_used: 450
- try_records_used: 2175
- switch_records_used: 0
- proxy_records_used: 0
- switch_q_values_created: false

## Value Scale Report

### Observe
- count: 350, mean: 0.1047, min: -0.005, max: 0.325
### Try
- count: 1715, mean: 0.1274, min: -0.15, max: 0.45
- unique_values: [-0.15, 0.45]
- scales_comparable: True
- note: Observe and try net_values are computed from different reward functions. Observe: info_gain - 0.005. Try: success(0.5)/failure(-0.1) - 0.05. Q values are comparable as utilities on the same scale (net budget change), but observe values are bounded above by feature novelty while try values are dominated by binary success/failure.

## Episode-Heldout Evaluation

- train_count: 2065, test_count: 560

### Ridge Regression
- mse: 0.057558, mae: 0.18809

### Baselines
| Baseline | MSE | MAE | Ridge Delta MSE | Ridge Delta MAE |
|----------|-----|-----|-----------------|----------------|
| global_mean | 0.073339 | 0.254931 | 0.015781 | 0.066841 |
| per_action_mean | 0.061636 | 0.202027 | 0.004078 | 0.013937 |
| action_type_only_ridge | 0.061609 | 0.202224 | 0.004051 | 0.014134 |

### Per-Action Metrics (Episode-Heldout)

| Action | Count | MSE | MAE | Mean True | Mean Pred | Rank Rho |
|--------|-------|-----|-----|-----------|-----------|----------|
| observe | 100 | 0.000614 | 0.019725 | 0.0829 | 0.082 | 0.888 |
| try_burn_as_fuel | 60 | 0.098017 | 0.293664 | 0.09 | 0.0679 | 0.0894 |
| try_craft_plank | 100 | 0.078271 | 0.240903 | 0.018 | 0.025 | 0.1655 |
| try_eat | 100 | 0.071807 | 0.235717 | 0.012 | 0.0301 | 0.1366 |
| try_mine_by_hand | 60 | 0.052419 | 0.191715 | 0.22 | 0.2348 | 0.6483 |
| try_mine_with_pickaxe | 60 | 0.037925 | 0.155584 | 0.37 | 0.3276 | 0.3312 |
| try_use_as_tool | 80 | 0.073272 | 0.215476 | 0.0225 | -0.0302 | 0.2422 |

### Per-Action Train/Test Counts

| Action | Train | Test |
|--------|-------|------|
| observe | 350 | 100 |
| try_burn_as_fuel | 240 | 60 |
| try_craft_plank | 350 | 100 |
| try_eat | 350 | 100 |
| try_mine_by_hand | 240 | 60 |
| try_mine_with_pickaxe | 240 | 60 |
| try_use_as_tool | 295 | 80 |

### Action Value Margin
- mean: 0.0797, min: 0.0001, max: 0.3198

### Top-Action Match
- available: False
- reason: Each situation has only 1 observed action; cannot determine true argmax over all 7 actions

### Shadow Policy Analysis
- shadow_observe_choice_rate: 0.0
- shadow_try_choice_rate: 1.0
- counterfactual_choice_rate: 0.8857
- avg_predicted_value_of_chosen: 0.3378
- avg_actual_value_of_chosen: 0.3656
- counterfactual_value_available: True

| Action | Shadow Choice Count |
|--------|--------------------|
| observe | 0 |
| try_burn_as_fuel | 1 |
| try_craft_plank | 25 |
| try_eat | 15 |
| try_mine_by_hand | 109 |
| try_mine_with_pickaxe | 410 |
| try_use_as_tool | 0 |

## Leave-One-Seed-Out Evaluation

- folds: 5, total_test: 2625

### Ridge Regression
- mse: 0.051866, mae: 0.173863

### Baselines
| Baseline | MSE | MAE | Ridge Delta MSE | Ridge Delta MAE |
|----------|-----|-----|-----------------|----------------|
| global_mean | 0.07496 | 0.257631 | 0.023095 | 0.083768 |
| per_action_mean | 0.056851 | 0.196564 | 0.004985 | 0.022702 |
| action_type_only_ridge | 0.05685 | 0.19679 | 0.004984 | 0.022927 |

### Per-Action Metrics (LOSO)

| Action | Count | MSE | MAE | Mean True | Mean Pred |
|--------|-------|-----|-----|-----------|-----------|
| observe | 450 | 0.000636 | 0.019782 | 0.0999 | 0.0999 |
| try_burn_as_fuel | 300 | 0.086637 | 0.280735 | 0.136 | 0.1355 |
| try_craft_plank | 450 | 0.070191 | 0.226975 | 0.0193 | 0.0175 |
| try_eat | 450 | 0.069964 | 0.225113 | 0.0127 | 0.0092 |
| try_mine_by_hand | 300 | 0.048941 | 0.167361 | 0.306 | 0.3055 |
| try_mine_with_pickaxe | 300 | 0.029117 | 0.105956 | 0.392 | 0.3922 |
| try_use_as_tool | 375 | 0.062355 | 0.207554 | -0.0012 | 0.0024 |

## Acceptance Checks

| Check | Result |
|-------|--------|
| hidden_feature_leakage_detected=false | **PASS** |
| oracle_leakage_detected=false | **PASS** |
| audit_label_used_as_input=false | **PASS** |
| preferred_explanation_used_as_hard_label=false | **PASS** |
| policy_decisions_changed=false | **PASS** |
| environment_changed=false | **PASS** |
| switch_proxy_created=false | **PASS** |
| switch_q_values_created=false | **PASS** |
| proxy_records_used=0 | **PASS** |
| all_records_have_source_ids=true | **PASS** |
| episode_heldout_available | **PASS** |
| loso_heldout_available | **PASS** |
| baseline_comparison_available | **PASS** |
| no_seed_crashes | **PASS** |
| **overall_acceptance** | **PASS** |


```
[block_done]
block_id=1J40b-1
condition=C5_mixed_source_identifiability_v0
seeds=101,103,107,109,113
total_records_used=2625
observe_records_used=450
try_records_used=2175
switch_records_used=0
proxy_records_used=0
switch_q_values_created=false
heldout_mse=0.057558
heldout_mae=0.18809
top_action_match_rate_available=false
shadow_observe_choice_rate=0.0
shadow_try_choice_rate=1.0
hidden_feature_leakage_detected_any=false
oracle_leakage_detected_any=false
audit_label_used_as_input=false
preferred_explanation_used_as_hard_label=false
policy_decisions_changed=false
implementation_status=pass
failure_reason=none
```
