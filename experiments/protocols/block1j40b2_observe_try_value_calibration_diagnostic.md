# Block 1J40b-2: Observe-vs-Try Value Calibration Diagnostic

- **Condition**: C5_mixed_source_identifiability_v0
- **Seeds**: [101, 103, 107, 109, 113]
- **Total elapsed**: 30.0s
- **Status**: PASS

## 1. Action Distribution Check

- observe_record_count: 450
- try_record_count: 2175
- observe/try ratio: 0.2069
- is_action_imbalanced: True
- note: Observe = 450 (17.1%), Try = 2175 (82.9%). Observe is a minority class with 20.7% ratio.

### Per-Try-Action Counts

| Action | Count |
|--------|-------|
| try_burn_as_fuel | 300 |
| try_craft_plank | 450 |
| try_eat | 450 |
| try_mine_by_hand | 300 |
| try_mine_with_pickaxe | 300 |
| try_use_as_tool | 375 |

## 2. Value Scale Check

### Observe Net Value
- count: 450, mean: 0.1012, std: 0.0763, min: -0.005, max: 0.325

### Try Net Value
- count: 2175, mean: 0.1214, std: 0.2986, min: -0.15, max: 0.45

### Per-Try-Action Net Value

| Action | Count | Mean | Std | Min | Max |
|--------|-------|------|-----|-----|-----|
| try_burn_as_fuel | 300 | 0.136 | 0.2997 | -0.15 | 0.45 |
| try_craft_plank | 450 | 0.0193 | 0.27 | -0.15 | 0.45 |
| try_eat | 450 | 0.0127 | 0.2667 | -0.15 | 0.45 |
| try_mine_by_hand | 300 | 0.306 | 0.2562 | -0.15 | 0.45 |
| try_mine_with_pickaxe | 300 | 0.392 | 0.1773 | -0.15 | 0.45 |
| try_use_as_tool | 375 | -0.0012 | 0.2591 | -0.15 | 0.45 |

- fraction_try_above_max_observe: 0.4524
- fraction_observe_above_mean_try: 0.3489
- note: Try max (0.45 = success) substantially exceeds observe max (0.325). Only 34.9% of observe values exceed the mean try value (0.1214). This creates a structural bias toward try choices in argmax Q.

## 3. Model Prediction Check

### Predicted Q Statistics

| Action | Mean Q | Std Q | Min Q | Max Q |
|--------|--------|-------|-------|-------|
| observe | 0.0304 | 0.048 | -0.0713 | 0.2365 |
| try_burn_as_fuel | 0.0615 | 0.1527 | -0.516 | 0.3389 |
| try_craft_plank | 0.0592 | 0.12 | -0.3058 | 0.4045 |
| try_eat | 0.0789 | 0.1241 | -0.2467 | 0.5085 |
| try_mine_by_hand | 0.2754 | 0.133 | -0.1497 | 0.4653 |
| try_mine_with_pickaxe | 0.3523 | 0.069 | 0.1598 | 0.5735 |
| try_use_as_tool | -0.013 | 0.102 | -0.3457 | 0.2218 |

### Observe Margin to Best Try
- count: 560
- mean: -0.3319
- min: -0.4826
- max: -0.136
- observe_wins: 0 (0.0)
- note: Observe Q beats best try Q in only 0/560 (0.0%) test situations. Mean observe margin to best try: -0.3319. Observe is systematically predicted lower than try.

## 4. Delayed Observe Value Check

- observe_then_try_count: 2175
- post_observe_try_success_rate: 0.4524
- post_observe_net_value_mean: 0.1214
- no_recent_observe_try_count: 0
- no_recent_observe_try_success_rate: None
- no_recent_observe_net_value_mean: None

### Gradient Check: Prior Observes vs Try Outcome

- low_observe_1_2_try_count: 1098
- low_observe_1_2_success_rate: 0.4636
- low_observe_1_2_mean_net_value: 0.1281
- high_observe_3plus_try_count: 1077
- high_observe_3plus_success_rate: 0.441
- high_observe_3plus_mean_net_value: 0.1146
- delayed_observe_benefit_detected: False
- note: No zero-observe tries exist for comparison. Gradient check (3+ vs 1-2 prior observes): delta success rate: -0.0226, delta mean net_value: -0.0135. Low-observe (1-2) tries: n=1098, success=0.4636, mean_nv=0.1281. High-observe (3+) tries: n=1077, success=0.441, mean_nv=0.1146.

## 5. Explanation-Feature Ablation

### Episode-Heldout

| Variant | MSE | MAE | Shadow Observe Rate |
|---------|-----|-----|--------------------|
| action_type_only | 0.061594 | 0.201793 | 0.0 |
| situation_without_explanation | 0.056284 | 0.181713 | 0.0 |
| situation_with_explanation | 0.055253 | 0.179996 | 0.0 |
| observe_specific_features | 0.062964 | 0.197768 | 0.0 |

### LOSO

| Variant | MSE | MAE | Shadow Observe Rate |
|---------|-----|-----|--------------------|
| action_type_only | 0.056856 | 0.196623 | 0.0 |
| situation_without_explanation | 0.050761 | 0.171358 | 0.0 |
| situation_with_explanation | 0.048209 | 0.166984 | 0.0 |
| observe_specific_features | 0.054261 | 0.182162 | 0.0 |

- explanation_features_help_observe_prediction: False
- note: Observe MSE without explanation: 0.000537, with explanation: 0.000537. Explanation features do not help observe prediction.

## 6. Reweighting Diagnostic

| Weighting Scheme | MSE | MAE | Shadow Observe Rate |
|------------------|-----|-----|--------------------|
| no_reweighting | 0.056262 | 0.181686 | 0.0 |
| balanced_observe_try | 0.056171 | 0.18162 | 0.0 |
| per_action_balanced | 0.056265 | 0.181693 | 0.0 |

## 7. Complete Action-Set Availability

- total_situations: 203
- complete_action_set_count: 0
- partial_action_set_count: 203
- top_action_match_available: False
- note: Each situation has on average 1.0 action observations. Only 0 situations have all 7 actions observed. Top-action match is NOT available.

### Action Set Size Distribution

| Actions per Situation | Count |
|----------------------|-------|
| 1 | 203 |

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
| no_seed_crashes | **PASS** |
| **overall_acceptance** | **PASS** |


```
[block_done]
block_id=1J40b-2
condition=C5_mixed_source_identifiability_v0
observe_record_count=450
try_record_count=2175
shadow_observe_rate_original=0.0
shadow_observe_rate_balanced=0.0
observe_mean_net_value=0.1012
try_mean_net_value=0.1214
post_observe_try_success_rate=0.4524
no_recent_observe_try_success_rate=None
delayed_observe_benefit_detected=false
explanation_features_help_observe_prediction=false
complete_action_set_count=0
top_action_match_available=false
hidden_feature_leakage_detected_any=false
oracle_leakage_detected_any=false
policy_decisions_changed=false
switch_q_values_created=false
proxy_records_used=0
implementation_status=pass
failure_reason=none
```
