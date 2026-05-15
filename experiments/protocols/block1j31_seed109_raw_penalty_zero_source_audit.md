# Block 1J31 -- Seed 109 Raw-Penalty-Zero Source Audit

## 1. Objective

Explain WHY most missed offline-avoidable prior violations have
raw_signed_penalty = 0 in online signed memory at seed 109.

Main question: for each raw-zero missed PV, is the zero penalty caused by:
1. unobserved_relevant_feature
2. no_online_memory_key
3. below_support_threshold
4. zero_weight_after_estimation
5. family_action_mismatch
6. feature_not_in_memory_universe
7. feature_not_attached_to_candidate
8. evidence_cancellation
9. other

## 2. Setup

| Parameter | Value |
|-----------|-------|
| Condition | C4_instance_subtype_cued_v1 |
| Seed | 109 |
| Budget | 1.5 |
| Episodes | 5 |
| Risk Alpha | 5.0 |
| Max Risk | 2.0 |
| Min Violation Support | 2 |
| Scales | [1.0, 3.0] |

## 3. Baselines

- A_no_memory: total_pv=29, macro_bal=0.5678
- Offline_signed_sum_upper_bound: total_pv=6, macro_bal=0.5887
- B_signed_sum_scale1: total_pv=27
- Offline-avoidable PVs (A - Offline): 20
- Missed by online (in A AND online PVs): 16

## 4. Part 1: Raw-Zero Missed-Case Table

Total raw-zero cases: 12 / 16 missed PVs

| Object | Action | Family | Relevant Dev Feats | Observed Dev | Online Keys Present | Online Raw | Offline Raw | Zero Reason |
|--------|--------|--------|--------------------|--------------|--------------------|------------|-------------|-------------|
| test_apple_001 | craft_plank | wood-like | [] | [] | 0 | 0.0 | 0.0 | feature_not_in_memory_universe |
| test_apple_004 | eat | apple-like | [] | [] | 0 | 0.0 | 0.0 | feature_not_in_memory_universe |
| test_apple_005 | use_as_tool | tool-like | [] | [] | 0 | 0.0 | 0.0 | feature_not_in_memory_universe |
| test_stone_block_002 | eat | apple-like | [] | [] | 0 | 0.0 | 0.0 | feature_not_in_memory_universe |
| test_stone_block_003 | use_as_tool | tool-like | [] | [] | 0 | 0.0 | 0.0 | feature_not_in_memory_universe |
| test_stone_block_004 | craft_plank | wood-like | [] | [] | 0 | 0.0 | 0.0 | feature_not_in_memory_universe |
| test_stone_block_006 | eat | apple-like | [] | [] | 0 | 0.0 | 0.0 | feature_not_in_memory_universe |
| test_stone_block_008 | craft_plank | wood-like | [] | [] | 0 | 0.0 | 0.0 | feature_not_in_memory_universe |
| test_stone_block_012 | craft_plank | wood-like | [] | [] | 0 | 0.0 | 0.0 | feature_not_in_memory_universe |
| test_wood_log_001 | craft_plank | wood-like | [] | [] | 0 | 0.0 | 0.0 | feature_not_in_memory_universe |
| test_wooden_pickaxe_002 | eat | apple-like | [] | [] | 0 | 0.0 | 0.0 | feature_not_in_memory_universe |
| test_wooden_pickaxe_005 | eat | apple-like | [] | [] | 0 | 0.0 | 0.0 | feature_not_in_memory_universe |

## 5. Part 2: Aggregate Zero-Penalty Source Counts

- total_raw_zero_missed_cases: 12
- count_unobserved_relevant_feature: 0
- count_no_online_memory_key: 0
- count_below_support_threshold: 0
- count_zero_weight_after_estimation: 0
- count_family_action_mismatch: 0
- count_feature_not_in_memory_universe: 12
- count_feature_not_attached_to_candidate: 0
- count_evidence_cancellation: 0
- count_other: 0

## 6. Part 3: Online vs Offline Key Comparison

- offline has key, online missing: 0
- online has key but below support: 0
- both have key, weight diff large: 0
- relevant feature unobserved online: 0
- mean abs online-offline weight diff: 0.399682


## 7. Part 4: Observation Contribution Check

- all relevant observed: 0
- some relevant missing: 0
- all relevant missing: 12
- observation_bottleneck_candidate: False

## 8. Part 5: Memory Support Contribution Check

- raw-zero cases with relevant features observed: 0
- count_no_online_memory_key: 0
- count_below_support_threshold: 0
- count_zero_weight_after_estimation: 0
- count_evidence_cancellation: 0
- memory_support_bottleneck_candidate: False

## 9. Part 6: Boolean Flags

### raw_penalty_zero_source_identified: True
  Rule: every raw-zero missed case has a zero_penalty_reason assigned
  - total_raw_zero: 12
  - all_have_reason: True

### observation_bottleneck_candidate: False
  Rule: count_unobserved_relevant_feature >= 50% of total_raw_zero_missed_cases
  - count_unobserved: 0
  - total_raw_zero: 12

### memory_support_bottleneck_candidate: False
  Rule: no_online_memory_key + below_support_threshold >= 50% of raw-zero cases with relevant features observed
  - no_key: 0
  - below_support: 0
  - total_with_relevant_observed: 0

### family_action_mismatch_detected: False
  Rule: count_family_action_mismatch > 0
  - count: 0

### feature_universe_issue_detected: True
  Rule: count_feature_not_in_memory_universe > 0
  - count: 12

### feature_attachment_issue_detected: False
  Rule: count_feature_not_attached_to_candidate > 0
  - count: 0

### evidence_cancellation_detected: False
  Rule: count_evidence_cancellation > 0
  - count: 0

### offline_has_signal_online_missing: False
  Rule: num_cases_where_offline_has_relevant_key_but_online_does_not > 0
  - count: 0

### hidden_feature_leakage_detected: False
  Rule: features used by policy are subset of observed features

### no_oracle_leakage_confirmed: True
  Rule: all online variants use no offline/oracle evidence
  - all_online_variants_clean: True

### hard_exclusion_used: False
  Rule: hard_exclusion is always false for all variants

### deceptive_family_preservation_valid: True
  Rule: all deceptive objects preserve expected family identity
  - total_deceptive: 13
  - preserved: 13

### implementation_status: pass
  Rule: pass if audit completes and no leakage is detected; partial if missing or leak

### failure_reason: none
  Rule: derived from raw-penalty-zero source audit results

## 10. Summary

```
[block_done]
block_id=1J31
seed=109
total_raw_zero_missed_cases=12
count_unobserved_relevant_feature=0
count_no_online_memory_key=0
count_below_support_threshold=0
count_zero_weight_after_estimation=0
count_family_action_mismatch=0
count_feature_not_in_memory_universe=12
count_feature_not_attached_to_candidate=0
count_evidence_cancellation=0
count_other=0
observation_bottleneck_candidate=False
memory_support_bottleneck_candidate=False
offline_has_signal_online_missing=False
feature_universe_issue_detected=True
feature_attachment_issue_detected=False
hidden_feature_leakage_detected=False
no_oracle_leakage_confirmed=True
implementation_status=pass
failure_reason=none
elapsed=36.3s
```