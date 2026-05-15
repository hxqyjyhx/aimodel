# Block 1J40a-2: Conservative Mixed Explanation Calibration

- **Condition**: C5_mixed_source_identifiability_v0
- **Seeds**: [101, 103, 107, 109, 113]
- **Total elapsed**: 28.4s
- **Configuration**: conf_threshold=0.1, margin_threshold=0.04

## Calibration Sweep Summary

- Configs tested: 25
- Selection rule: prefer safety (lower over_split + noise_overfit) over macro accuracy

### Best Safety Config
- conf_threshold=0.1, margin_threshold=0.04
- macro_accuracy=0.395, state_recall=0.7571
- over_split_rate=0.0, noise_overfit_rate=0.4915, unresolved_rate=0.224

### Best Accuracy Config
- conf_threshold=0.1, margin_threshold=0.0
- macro_accuracy=0.4104, state_recall=0.7571
- over_split_rate=0.3889, noise_overfit_rate=0.4915, unresolved_rate=0.1818

Safety and accuracy configs DIFFER ¡ª using SAFETY config.

### Selected (Final): conf_threshold=0.1, margin_threshold=0.04

| conf | margin | macro_acc | state_recall | gap_recall | split_recall | noise_recall | unresolved | over_split | noise_overfit |
|------|--------|-----------|--------------|------------|-------------|-------------|------------|------------|---------------|
| 0.10 | 0.00 | 0.4104 | 0.7571 | 0.4026 | 0.1692 | 0.3125 | 0.1818 | 0.3889 | 0.4915 |
| 0.10 | 0.04 | 0.3950 | 0.7571 | 0.4026 | 0.1077 | 0.3125 | 0.2240 | 0.0000 | 0.4915 |
| 0.10 | 0.08 | 0.3636 | 0.7429 | 0.3377 | 0.0615 | 0.3125 | 0.2792 | 0.0000 | 0.4737 |
| 0.10 | 0.12 | 0.3138 | 0.6571 | 0.2857 | 0.0000 | 0.3125 | 0.4188 | 0.0000 | 0.4444 |
| 0.10 | 0.16 | 0.2807 | 0.6286 | 0.1818 | 0.0000 | 0.3125 | 0.5422 | 0.0000 | 0.4340 |
| 0.15 | 0.00 | 0.3681 | 0.7571 | 0.4026 | 0.0000 | 0.3125 | 0.2175 | 1.0000 | 0.4915 |
| 0.15 | 0.04 | 0.3681 | 0.7571 | 0.4026 | 0.0000 | 0.3125 | 0.2468 | 0.0000 | 0.4915 |
| 0.15 | 0.08 | 0.3483 | 0.7429 | 0.3377 | 0.0000 | 0.3125 | 0.2922 | 0.0000 | 0.4737 |
| 0.15 | 0.12 | 0.3138 | 0.6571 | 0.2857 | 0.0000 | 0.3125 | 0.4188 | 0.0000 | 0.4444 |
| 0.15 | 0.16 | 0.2807 | 0.6286 | 0.1818 | 0.0000 | 0.3125 | 0.5422 | 0.0000 | 0.4340 |
| 0.20 | 0.00 | 0.3291 | 0.7571 | 0.2468 | 0.0000 | 0.3125 | 0.3214 | 1.0000 | 0.4915 |
| 0.20 | 0.04 | 0.3291 | 0.7571 | 0.2468 | 0.0000 | 0.3125 | 0.3344 | 0.0000 | 0.4915 |
| 0.20 | 0.08 | 0.3190 | 0.7429 | 0.2208 | 0.0000 | 0.3125 | 0.3701 | 0.0000 | 0.4737 |
| 0.20 | 0.12 | 0.2911 | 0.6571 | 0.1948 | 0.0000 | 0.3125 | 0.4643 | 0.0000 | 0.4444 |
| 0.20 | 0.16 | 0.2807 | 0.6286 | 0.1818 | 0.0000 | 0.3125 | 0.5552 | 0.0000 | 0.4340 |
| 0.25 | 0.00 | 0.3291 | 0.7571 | 0.2468 | 0.0000 | 0.3125 | 0.3409 | 0.0000 | 0.4915 |
| 0.25 | 0.04 | 0.3291 | 0.7571 | 0.2468 | 0.0000 | 0.3125 | 0.3442 | 0.0000 | 0.4915 |
| 0.25 | 0.08 | 0.3190 | 0.7429 | 0.2208 | 0.0000 | 0.3125 | 0.3734 | 0.0000 | 0.4737 |
| 0.25 | 0.12 | 0.2911 | 0.6571 | 0.1948 | 0.0000 | 0.3125 | 0.4643 | 0.0000 | 0.4444 |
| 0.25 | 0.16 | 0.2807 | 0.6286 | 0.1818 | 0.0000 | 0.3125 | 0.5552 | 0.0000 | 0.4340 |
| 0.30 | 0.00 | 0.3148 | 0.7000 | 0.2468 | 0.0000 | 0.3125 | 0.4253 | 0.0000 | 0.4915 |
| 0.30 | 0.04 | 0.3148 | 0.7000 | 0.2468 | 0.0000 | 0.3125 | 0.4286 | 0.0000 | 0.4915 |
| 0.30 | 0.08 | 0.3083 | 0.7000 | 0.2208 | 0.0000 | 0.3125 | 0.4416 | 0.0000 | 0.4737 |
| 0.30 | 0.12 | 0.2840 | 0.6286 | 0.1948 | 0.0000 | 0.3125 | 0.4870 | 0.0000 | 0.4444 |
| 0.30 | 0.16 | 0.2736 | 0.6000 | 0.1818 | 0.0000 | 0.3125 | 0.5714 | 0.0000 | 0.4340 |

## Per-Seed Results

| Seed | Total Mixed | Gap | State | Split | Noise | Unresolved | Macro Acc | OverSplit | NoiseFit |
|------|-------------|-----|-------|-------|-------|------------|-----------|-----------|----------|
| 101 | 59 | 13 | 24 | 0 | 11 | 11 | 0.3552 | 0.0 | 0.5455 |
| 103 | 68 | 13 | 33 | 0 | 12 | 10 | 0.4023 | 0.0 | 0.4167 |
| 107 | 56 | 4 | 27 | 0 | 13 | 12 | 0.3708 | 0.0 | 0.3846 |
| 109 | 55 | 6 | 19 | 3 | 10 | 17 | 0.4581 | 0.0 | 0.7 |
| 113 | 70 | 10 | 24 | 4 | 13 | 19 | 0.3826 | 0.0 | 0.4615 |

## Aggregate Confusion Matrix

| Audit \ Predicted | gap | state | split | noise | unresolved |
|--------------------|-----|-------|-------|-------|------------|
| observation_gap | 31 | 17 | 0 | 17 | 12 |
| state_condition | 0 | 53 | 0 | 6 | 11 |
| identity_split | 7 | 21 | 7 | 6 | 24 |
| irreducible_noise | 8 | 36 | 0 | 30 | 22 |

## Per-Class Recall

- **observation_gap**: 0.4026 (31/77)
- **state_condition**: 0.7571 (53/70)
- **identity_split**: 0.1077 (7/65)
- **irreducible_noise**: 0.3125 (30/96)

## Aggregate Metrics

- total_mixed_cases: 308
- observation_gap_pred_count: 46
- state_condition_pred_count: 127
- identity_split_pred_count: 7
- irreducible_noise_pred_count: 59
- unresolved_pred_count: 69
- macro_accuracy: 0.395
- over_split_rate: 0.0
- noise_overfit_rate: 0.4915
- unresolved_rate: 0.224
- heldout_prediction_available: True
- all_records_have_source_ids: True

## Safety Metrics (vs 1J40a-1 baseline)

| Metric | 1J40a-1 | 1J40a-2 | Target | Status |
|--------|---------|---------|--------|--------|
| over_split_rate | 0.6125 | 0.0000 | <0.61 | PASS |
| noise_overfit_rate | 0.5185 | 0.4915 | <0.52 | PASS |
| unresolved_rate | 0.0479 | 0.2240 | >0.05 | PASS |
| state_condition_recall | 0.8929 | 0.7571 | >=0.75 | PASS |
| over_split <= 0.35 (preferred) | - | 0.0000 | <=0.35 | PASS |
| noise_overfit <= 0.35 (preferred) | - | 0.4915 | <=0.35 | -- |
| unresolved 0.10-0.35 (preferred) | - | 0.2240 | [0.10,0.35] | PASS |

## Acceptance Checks

| Check | Result |
|-------|--------|
| policy_decisions_changed=false | **PASS** |
| candidate_split_performed=false | **PASS** |
| state_condition_created=false | **PASS** |
| hidden_feature_leakage_detected_any=false | **PASS** |
| oracle_leakage_detected_any=false | **PASS** |
| all_records_have_source_ids=true | **PASS** |
| heldout_prediction_available=true | **PASS** |
| over_split_rate_decreased | **PASS** |
| noise_overfit_rate_decreased | **PASS** |
| unresolved_rate_increased | **PASS** |
| state_condition_recall>=0.75 | **PASS** |
| no_seed_crashes | **PASS** |
| **overall_acceptance** | **PASS** |


```
[block_done]
block_id=1J40a-2
condition=C5_mixed_source_identifiability_v0
seeds=101,103,107,109,113
macro_accuracy=0.395
state_condition_recall=0.7571
observation_gap_recall=0.4026
identity_split_recall=0.1077
irreducible_noise_recall=0.3125
unresolved_rate=0.224
over_split_rate=0.0
noise_overfit_rate=0.4915
hidden_feature_leakage_detected_any=false
oracle_leakage_detected_any=false
policy_decisions_changed=false
candidate_split_performed=false
state_condition_created=false
implementation_status=pass
failure_reason=none
```
