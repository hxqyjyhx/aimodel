# Block 1J23 -- Deviation Risk Aggregation Ablation

## 1. Objective

Test whether the failure of 1J21/1J22 family-conditioned deviation memory is caused by the sum-all-positive-weights risk aggregation rule.

## 2. Setup

| Parameter | Value |
|-----------|-------|
| Condition | C4_instance_subtype_cued_v1 |
| Seed | 101 |
| Budget | 1.5 |
| Episodes | 5 |
| Prior Violation Threshold | 0.6 |
| Min Violation Support | 2 |
| Risk Alpha | 5.0 |
| Max Risk | 2.0 |

## 3. Baselines

| Policy | Macro BAcc |
|--------|------------|
| C15b | 0.5975 |
| A_no_memory | 0.5937 |

## 4. Part 1: Aggregation Formula Comparison

| Variant | Total PV | Dec PV | Norm PV | Macro BAcc | Delta vs A | Eff Probes | Avoided Eff | Avoided PV | Avoided NV | Helpful | Harmful | Hard Excl | Aggregation Formula | Oracle |
|---------|----------|--------|---------|------------|------------|------------|-------------|------------|------------|---------|---------|-----------|---------------------|--------|
| B_sum_positive_current | 22 | 0 | 22 | 0.5846 | -0.0091 | 35 | 0 | 9 | 6 | 9 | 6 | False | sum(max(0, w_i) for each present dev_feat i) | False |
| B_max_positive | 16 | 0 | 16 | 0.5889 | -0.0048 | 35 | 0 | 9 | 5 | 9 | 5 | False | max(max(0, w_i) for each present dev_feat i) | False |
| B_top2_positive_capped | 21 | 0 | 21 | 0.5957 | 0.002 | 35 | 0 | 9 | 6 | 9 | 6 | False | min(max_risk, sum(top_2_positive_weights)) | False |
| B_mean_positive | 19 | 0 | 19 | 0.5819 | -0.0118 | 35 | 0 | 12 | 5 | 12 | 5 | False | mean(max(0, w_i) for each present dev_feat i) | False |
| B_signed_sum_capped | 2 | 0 | 2 | 0.5904 | -0.0033 | 29 | 6 | 14 | 5 | 14 | 5 | False | clip(sum(signed_w_i), -max_risk, +max_risk) | False |
| B_signed_max_abs | 2 | 0 | 2 | 0.5820 | -0.0117 | 28 | 7 | 14 | 7 | 14 | 7 | False | clip(argmax(|w_i|).signed_weight, -max_risk, +max_risk) | False |
| C_max_positive_permuted | 25 | 3 | 22 | 0.5788 | -0.0149 | 35 | 0 | 7 | 6 | 7 | 6 | False | max(max(0, w_i) for each present dev_feat i) [PERMUTED withi | False |
| C_signed_sum_permuted | 3 | 0 | 3 | 0.5852 | -0.0085 | 27 | 8 | 15 | 8 | 15 | 8 | False | clip(sum(signed_w_i), -max_risk, +max_risk) [PERMUTED within | False |
| Family_only_control | 10 | 3 | 7 | 0.5736 | -0.0201 | 21 | 14 | 11 | 8 | 11 | 8 | False | sum(risk_weight(goal, action, family)) across families | False |
| Oracle_dev_feature_rule | 18 | 0 | 18 | 0.5897 | -0.004 | 35 | 0 | 8 | 6 | 8 | 6 | False | 2.0 if best_dev_feature_per(action,family) present else 0.0 | True |

## 5. Part 2: Candidate-Level Aggregation Audit

Total candidates audited: 70

### Top 30 Candidates by Sum Positive Penalty

| OID | Action | Family | Dev Feats | SumPos | MaxPos | Top2 | MeanPos | SignedSum | SignedMaxAbs | PV | NV | Utility | Sel by A | Sel by SumPos | Sel by MaxPos | Sel by SignedSum |
|-----|--------|--------|-----------|--------|--------|------|---------|-----------|-------------|----|----|---------|----------|--------------|--------------|-----------------|
| test_stone_block_002 | mine_with_pickaxe | stone-like | ['hollow_sound', 'brittle_surface'] | 3.0000 | 1.5000 | 2.0000 | 1.5000 | 2.0000 | 1.8119 | True | False | unknown |  |  |  |  |
| test_stone_block_009 | mine_with_pickaxe | stone-like | ['hollow_sound', 'brittle_surface'] | 3.0000 | 1.5000 | 2.0000 | 1.5000 | 2.0000 | 1.8119 | True | False | unknown |  |  |  |  |
| test_wood_log_000 | burn_as_fuel | wood-like | ['treated_surface', 'damp_texture'] | 2.4215 | 1.2108 | 2.0000 | 1.2108 | 2.0000 | 1.2108 | True | False | unknown |  |  |  |  |
| test_wood_log_001 | burn_as_fuel | wood-like | ['treated_surface', 'damp_texture'] | 2.4215 | 1.2108 | 2.0000 | 1.2108 | 2.0000 | 1.2108 | True | False | unknown |  |  |  |  |
| test_wood_log_012 | burn_as_fuel | wood-like | ['treated_surface', 'damp_texture'] | 2.4215 | 1.2108 | 2.0000 | 1.2108 | 2.0000 | 1.2108 | True | False | unknown |  |  |  |  |
| test_apple_004 | eat | apple-like | ['damp_texture', 'brittle_surface'] | 1.9286 | 0.9643 | 1.9286 | 0.9643 | 1.9286 | 0.9643 | True | False | effective | Y |  |  |  |
| test_apple_006 | eat | apple-like | ['damp_texture', 'brittle_surface'] | 1.9286 | 0.9643 | 1.9286 | 0.9643 | 1.9286 | 0.9643 | True | False | effective | Y |  |  |  |
| test_apple_010 | eat | apple-like | ['damp_texture', 'brittle_surface'] | 1.9286 | 0.9643 | 1.9286 | 0.9643 | 1.9286 | 0.9643 | True | False | effective | Y |  |  |  |
| test_apple_014 | eat | apple-like | ['damp_texture', 'brittle_surface'] | 1.9286 | 0.9643 | 1.9286 | 0.9643 | 1.9286 | 0.9643 | True | False | effective | Y |  |  |  |
| test_stone_block_014 | mine_by_hand | stone-like | ['hollow_sound', 'brittle_surface'] | 1.2709 | 0.6355 | 1.2709 | 0.6355 | 1.2709 | 0.6355 | True | False | effective | Y |  |  |  |
| test_apple_000 | eat | apple-like | [] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | True | effective | Y |  |  |  |
| test_apple_001 | eat | apple-like | [] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | True | effective | Y | Y | Y |  |
| test_apple_003 | eat | apple-like | [] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | True | effective | Y | Y |  | Y |
| test_apple_005 | eat | apple-like | [] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | True | effective | Y |  | Y | Y |
| test_apple_007 | eat | apple-like | [] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | True | unknown |  |  |  |  |
| test_apple_008 | eat | apple-like | [] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True | False | effective | Y | Y | Y |  |
| test_apple_009 | craft_plank | wood-like | [] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True | False | unknown |  | Y | Y |  |
| test_apple_011 | eat | apple-like | [] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True | False | unknown |  | Y |  |  |
| test_apple_012 | eat | apple-like | [] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | True | effective | Y | Y |  |  |
| test_apple_013 | craft_plank | wood-like | [] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True | False | effective | Y |  |  |  |
| test_stone_block_000 | mine_by_hand | stone-like | [] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True | False | unknown |  | Y | Y |  |
| test_stone_block_001 | mine_by_hand | stone-like | [] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | True | unknown |  | Y | Y |  |
| test_stone_block_001 | mine_with_pickaxe | stone-like | [] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | True | unknown |  |  |  |  |
| test_stone_block_003 | craft_plank | wood-like | [] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True | False | effective | Y | Y | Y |  |
| test_stone_block_004 | mine_by_hand | stone-like | [] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True | False | unknown |  |  |  |  |
| test_stone_block_004 | mine_with_pickaxe | stone-like | [] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | True | unknown |  |  |  |  |
| test_stone_block_005 | mine_by_hand | stone-like | [] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True | False | unknown |  | Y | Y |  |
| test_stone_block_005 | mine_with_pickaxe | stone-like | [] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | True | unknown |  |  |  |  |
| test_stone_block_006 | mine_by_hand | stone-like | [] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True | False | effective | Y | Y | Y |  |
| test_stone_block_006 | mine_with_pickaxe | stone-like | [] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | True | unknown |  |  |  |  |

## 6. Part 3: Helpful vs Harmful Avoidance

| Variant | Avoided PV | Avoided NV | Avoided Eff | Avoided TP | Helpful | Harmful | Net |
|---------|------------|------------|-------------|------------|---------|---------|-----|
| B_sum_positive_current | 9 | 6 | 0 | 9 | 9 | 6 | 3 |
| B_max_positive | 9 | 5 | 0 | 9 | 9 | 5 | 4 |
| B_top2_positive_capped | 9 | 6 | 0 | 9 | 9 | 6 | 3 |
| B_mean_positive | 12 | 5 | 0 | 12 | 12 | 5 | 7 |
| B_signed_sum_capped | 14 | 5 | 6 | 14 | 14 | 5 | 9 |
| B_signed_max_abs | 14 | 7 | 7 | 14 | 14 | 7 | 7 |
| C_max_positive_permuted | 7 | 6 | 0 | 7 | 7 | 6 | 1 |
| C_signed_sum_permuted | 15 | 8 | 8 | 15 | 15 | 8 | 7 |
| Family_only_control | 11 | 8 | 14 | 11 | 11 | 8 | 3 |
| Oracle_dev_feature_rule | 8 | 6 | 0 | 8 | 8 | 6 | 2 |

## 7. Part 4: Positive vs Negative Signal Audit

- num_positive_weight_keys: 16
- num_negative_weight_keys: 8
- total_abs_positive_weight: 12.9110
- total_abs_negative_weight: 4.5214
- candidates_with_protective_signal: 5
- candidates_where_negative_changed_selection: 5

### Top 10 Positive Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_stone | mine_with_pickaxe | stone-like | hollow_sound | 1.8119 |
| need_stone | mine_with_pickaxe | stone-like | brittle_surface | 1.8119 |
| need_fuel | burn_as_fuel | wood-like | treated_surface | 1.2108 |
| need_fuel | burn_as_fuel | wood-like | damp_texture | 1.2108 |
| need_food | eat | apple-like | damp_texture | 0.9643 |
| need_food | eat | apple-like | brittle_surface | 0.9643 |
| need_stone | mine_with_pickaxe | stone-like | treated_surface | 0.7722 |
| need_stone | mine_with_pickaxe | stone-like | damp_texture | 0.7722 |
| need_stone | mine_by_hand | stone-like | hollow_sound | 0.6355 |
| need_stone | mine_by_hand | stone-like | brittle_surface | 0.6355 |

### Top 10 Negative Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_planks | craft_plank | wood-like | treated_surface | -1.2108 |
| need_planks | craft_plank | wood-like | damp_texture | -1.2108 |
| need_food | eat | apple-like | treated_surface | -0.4380 |
| need_food | eat | apple-like | hollow_sound | -0.4380 |
| need_stone | mine_by_hand | stone-like | treated_surface | -0.4042 |
| need_stone | mine_by_hand | stone-like | damp_texture | -0.4042 |
| need_fuel | burn_as_fuel | wood-like | hollow_sound | -0.2077 |
| need_fuel | burn_as_fuel | wood-like | brittle_surface | -0.2077 |

## 8. Part 5: Matching Permuted Controls

| Real Variant | Permuted Variant | Real PV | Permuted PV | Real BAcc | Permuted BAcc | Real Beats Permuted |
|-------------|-----------------|---------|-------------|-----------|---------------|---------------------|
| B_max_positive | C_max_positive_permuted | 16 | 25 | 0.5889 | 0.5788 | True |
| B_signed_sum_capped | C_signed_sum_permuted | 2 | 3 | 0.5904 | 0.5852 | True |

## 9. Part 6: Boolean Flag Details

### current_sum_reproduced_1j22: True
  Rule: B_sum_positive_current PV >= A PV (same as 1J22 finding)
  - sum_positive_pv: 22
  - A_pv: 22

### max_positive_beats_current_sum: True
  Rule: B_max_positive_total_pv < B_sum_positive_current_total_pv
  - max_positive_pv: 16
  - sum_positive_pv: 22

### max_positive_beats_A: True
  Rule: B_max_positive_total_pv < A_total_pv
  - max_positive_pv: 16
  - A_pv: 22

### top2_positive_beats_current_sum: True
  Rule: B_top2_positive_capped_total_pv < B_sum_positive_current_total_pv
  - top2_pv: 21
  - sum_positive_pv: 22

### signed_sum_beats_current_sum: True
  Rule: B_signed_sum_capped_total_pv < B_sum_positive_current_total_pv
  - signed_sum_pv: 2
  - sum_positive_pv: 22

### signed_sum_beats_A: True
  Rule: B_signed_sum_capped_total_pv < A_total_pv
  - signed_sum_pv: 2
  - A_pv: 22

### signed_variant_uses_protective_signal: True
  Rule: signed_sum or signed_max_abs produces different result than sum_positive_only
  - signed_sum_pv: 2
  - sum_positive_pv: 22
  - signed_max_abs_pv: 2

### any_dev_aggregation_beats_A: True
  Rule: at least one non-oracle dev aggregation variant has PV < A_total_pv
  - B_sum_positive_current: 22
  - B_max_positive: 16
  - B_top2_positive_capped: 21
  - B_mean_positive: 19
  - B_signed_sum_capped: 2
  - B_signed_max_abs: 2

### any_dev_aggregation_beats_matching_permuted: True
  Rule: at least one real variant beats its permuted control
  - max_positive_vs_permuted: 16 vs 25
  - signed_sum_vs_permuted: 2 vs 3

### any_dev_aggregation_beats_oracle_dev_feature: True
  Rule: at least one dev aggregation variant has PV < oracle_dev_feature PV
  - oracle_dev_feature_pv: 18

### family_only_still_dominates_all_dev_aggregators: False
  Rule: family_only PV <= every dev aggregation variant PV
  - family_only_pv: 10
  - dev_aggregator_pvs: {'B_sum_positive_current': 22, 'B_max_positive': 16, 'B_top2_positive_capped': 21, 'B_mean_positive': 19, 'B_signed_sum_capped': 2, 'B_signed_max_abs': 2}

### counterproductive_weighting_reduced: True
  Rule: at least one variant has helpful >= harmful avoidances
  - B_sum_positive_current: helpful=9, harmful=6
  - B_max_positive: helpful=9, harmful=5
  - B_top2_positive_capped: helpful=9, harmful=6
  - B_mean_positive: helpful=12, harmful=5
  - B_signed_sum_capped: helpful=14, harmful=5
  - B_signed_max_abs: helpful=14, harmful=7

### hard_exclusion_used: False
  Rule: hard_exclusion is always false for all variants

### deceptive_family_preservation_valid: True
  Rule: all deceptive objects preserve expected family identity
  - total_deceptive: 13
  - preserved: 13

### implementation_status: pass
  Rule: pass if any dev aggregation beats A; partial otherwise
  - any_dev_aggregation_beats_A: True

### failure_reason: none
  Rule: derived from aggregation comparison results

## 10. Summary

```
[block_done]
block_id=1J23
A_total_pv=22
B_sum_positive_current_total_pv=22
B_max_positive_total_pv=16
B_top2_positive_capped_total_pv=21
B_mean_positive_total_pv=19
B_signed_sum_capped_total_pv=2
B_signed_max_abs_total_pv=2
Oracle_dev_feature_rule_total_pv=18
Family_only_control_total_pv=10
best_non_oracle_dev_aggregation_name=B_signed_sum_capped
best_non_oracle_dev_aggregation_total_pv=2
best_non_oracle_dev_aggregation_macro_delta_vs_A=-0.0033
current_sum_reproduced_1j22=True
max_positive_beats_current_sum=True
signed_sum_beats_current_sum=True
any_dev_aggregation_beats_A=True
any_dev_aggregation_beats_matching_permuted=True
family_only_still_dominates_all_dev_aggregators=False
counterproductive_weighting_reduced=True
hard_exclusion_used=False
deceptive_family_preservation_valid=True
implementation_status=pass
failure_reason=none
elapsed=133.2s
```