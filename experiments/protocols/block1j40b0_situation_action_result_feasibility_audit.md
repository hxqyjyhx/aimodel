# Block 1J40b-0: Situation-Action-Result Learning Feasibility Audit

- **Condition**: C5_mixed_source_identifiability_v0
- **Seeds**: [101, 103, 107, 109, 113]
- **Total elapsed**: 26.1s
- **Total events processed**: 2625
- **Total objects**: 300
- **Total OCM candidates**: 120

## Action Event Availability

- observe_event_count: 2625 (estimated, see per-seed)
- switch_learning_available: False
- unique_try_action_types: ['burn_as_fuel', 'craft_plank', 'eat', 'mine_by_hand', 'mine_with_pickaxe', 'use_as_tool']

## Per-Seed Results

| Seed | Total Recs | Observe | Try | Switch | Switch Avail | Valid NV | Source IDs | Expl Feat | Proxy | Hidden Leak | Oracle Leak | Can Proceed |
|------|-----------|---------|-----|--------|-------------|----------|------------|-----------|-------|-------------|-------------|-------------|
| 101 | 525 | 90 | 435 | 0 | False | 525 | 525 | 435 | 0 | False | False | True |
| 103 | 525 | 90 | 435 | 0 | False | 525 | 525 | 435 | 0 | False | False | True |
| 107 | 525 | 90 | 435 | 0 | False | 525 | 525 | 435 | 0 | False | False | True |
| 109 | 525 | 90 | 435 | 0 | False | 525 | 525 | 435 | 0 | False | False | True |
| 113 | 525 | 90 | 435 | 0 | False | 525 | 525 | 435 | 0 | False | False | True |

## Learning Record Details

- total_learning_records: 2625
- observe_learning_records: 450
- try_learning_records: 2175
- switch_learning_records: 0
- records_with_valid_net_value: 2625
- records_with_source_ids: 2625
- records_using_soft_explanation_features: 2175
- proxy_records_created: 0
- soft_explanation_features_used: True

## Net Value Statistics

### Observe
- mean: 0.0987, min: -0.005, max: 0.325

### Try
- mean: 0.1214, min: -0.15, max: 0.45

## Situation Features Used

- candidate_confidence: 2625
- candidate_support_count: 2625
- candidate_stability: 2625
- candidate_positive_feature_count: 2625
- candidate_mixed_rate: 2625
- candidate_contradiction_count: 2625
- observed_feature_coverage: 2625
- observed_state_coverage: 2625
- recent_observe_count: 2625
- recent_try_count: 2625
- total_object_observes: 2625
- total_object_tries: 2625
- best_current_try_value: 2625
- top_action_value_margin: 2625
- episode_id: 2625
- episode_progress: 2625

## Audit Checks

| Check | Result |
|-------|--------|
| total_learning_records > 0 | **PASS** |
| observe_learning_records > 0 | **PASS** |
| try_learning_records > 0 | **PASS** |
| records_with_valid_net_value == total | **PASS** |
| records_with_source_ids == total | **PASS** |
| proxy_records_created == 0 | **PASS** |
| hidden_feature_leakage_detected=false | **PASS** |
| oracle_leakage_detected=false | **PASS** |
| policy_decisions_changed=false | **PASS** |
| no_forbidden_in_situation_features | **PASS** |
| switch_learning_available (optional) | **PASS** |
| no_seed_crashes | **PASS** |
| **can_proceed_to_1J40b1** | **YES** |
| **overall_audit** | **PASS** |


```
[block_done]
block_id=1J40b-0
condition=C5_mixed_source_identifiability_v0
total_learning_records=2625
observe_learning_records=450
try_learning_records=2175
switch_learning_records=0
switch_learning_available=false
records_with_valid_net_value=2625
records_with_source_ids=2625
proxy_records_created=0
soft_explanation_features_used=true
hidden_feature_leakage_detected=false
oracle_leakage_detected=false
policy_decisions_changed=false
can_proceed_to_1J40b1=true
implementation_status=pass
failure_reason=none
```
