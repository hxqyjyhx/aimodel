# Block 1J40a-1: Mixed Candidate Explanation Shadow

- **Condition**: C5_mixed_source_identifiability_v0
- **Seeds**: [101, 103, 107, 109, 113]
- **Total elapsed**: 25.3s

## Per-Seed Results

| Seed | Total Mixed | Gap | State | Split | Noise | Unresolved | Macro Acc | OverSplit | NoiseFit |
|------|-------------|-----|-------|-------|-------|------------|-----------|-----------|----------|
| 101 | 65 | 18 | 19 | 20 | 5 | 3 | 0.5214 | 0.75 | 0.0 |
| 103 | 63 | 9 | 32 | 15 | 6 | 1 | 0.4862 | 0.4 | 0.6667 |
| 107 | 57 | 9 | 32 | 9 | 6 | 1 | 0.4219 | 1.0 | 0.5 |
| 109 | 63 | 7 | 24 | 21 | 4 | 7 | 0.4944 | 0.4762 | 0.5 |
| 113 | 65 | 12 | 29 | 15 | 6 | 3 | 0.4489 | 0.6 | 0.8333 |

## Aggregate Confusion Matrix

| Audit \ Predicted | gap | state | split | noise | unresolved |
|--------------------|-----|-------|-------|-------|------------|
| observation_gap | 46 | 38 | 18 | 8 | 0 |
| state_condition | 0 | 50 | 6 | 0 | 0 |
| identity_split | 6 | 30 | 31 | 6 | 15 |
| irreducible_noise | 3 | 18 | 25 | 13 | 0 |

## Per-Class Recall

- **observation_gap**: 0.4182 (46/110)
- **state_condition**: 0.8929 (50/56)
- **identity_split**: 0.4247 (31/73)
- **irreducible_noise**: 0.2203 (13/59)

## Aggregate Metrics

- total_mixed_cases: 313
- observation_gap_pred_count: 55
- state_condition_pred_count: 136
- identity_split_pred_count: 80
- irreducible_noise_pred_count: 27
- unresolved_pred_count: 15
- macro_accuracy: 0.489
- over_split_rate: 0.6125
- noise_overfit_rate: 0.5185
- unresolved_rate: 0.0479
- heldout_prediction_available: True
- all_records_have_source_ids: True

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
| unresolved_allowed | **PASS** |
| no_seed_crashes | **PASS** |
| **overall_acceptance** | **PASS** |


```
[block_done]
block_id=1J40a-1
condition=C5_mixed_source_identifiability_v0
seeds=101,103,107,109,113
total_mixed_cases=313
observation_gap_pred_count=55
state_condition_pred_count=136
identity_split_pred_count=80
irreducible_noise_pred_count=27
unresolved_pred_count=15
macro_accuracy=0.489
over_split_rate=0.6125
noise_overfit_rate=0.5185
heldout_prediction_available=true
all_records_have_source_ids=true
hidden_feature_leakage_detected_any=false
oracle_leakage_detected_any=false
policy_decisions_changed=false
candidate_split_performed=false
state_condition_created=false
implementation_status=pass
failure_reason=none
```
