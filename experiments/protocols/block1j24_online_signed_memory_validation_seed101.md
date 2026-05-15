# Block 1J24 -- Online Signed Risk/Protection Memory Validation

## 1. Objective

Validate whether the strong 1J23 signed aggregation result holds in an online causal setting without offline/oracle evidence leakage.

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

## 4. Part 1: Main Comparison Table

| Variant | Total PV | Dec PV | Norm PV | Macro BAcc | Delta vs A | Eff Probes | Avoided Eff | Avoided PV | Avoided NV | Helpful | Harmful | Hard Excl | Oracle Evid | Future Outcome | True Dec Label |
|---------|----------|--------|---------|------------|------------|------------|-------------|------------|------------|---------|---------|-----------|-------------|----------------|----------------|
| A_no_memory | 22 | 7 | 15 | 0.5937 | 0.0 | 35 | 0 | 0 | 0 | 0 | 0 | False | False | False | False |
| B_sum_positive_online | 19 | 2 | 17 | 0.5893 | -0.0044 | 35 | 0 | 7 | 7 | 7 | 7 | False | False | False | False |
| B_signed_sum_online | 14 | 5 | 9 | 0.5889 | -0.0048 | 33 | 2 | 9 | 4 | 9 | 4 | False | False | False | False |
| B_signed_max_abs_online | 11 | 4 | 7 | 0.5894 | -0.0043 | 30 | 5 | 11 | 5 | 11 | 5 | False | False | False | False |
| C_signed_sum_online_permuted | 5 | 4 | 1 | 0.5818 | -0.0119 | 28 | 7 | 13 | 8 | 13 | 8 | False | False | False | False |
| C_signed_max_abs_online_permuted | 4 | 3 | 1 | 0.5802 | -0.0135 | 27 | 8 | 13 | 8 | 13 | 8 | False | False | False | False |
| Family_only_online_control | 13 | 6 | 7 | 0.5750 | -0.0187 | 18 | 17 | 10 | 10 | 10 | 10 | False | False | False | False |
| Offline_signed_sum_upper_bound | 3 | 0 | 3 | 0.5885 | -0.0052 | 28 | 7 | 14 | 7 | 14 | 7 | False | True | False | False |

## 5. Part 2: Online Learning Curve

| Ep | A PV | SumPos PV | SignedSum PV | SignedMax PV | FamOnly PV | SignedSum BAcc | SignedMax BAcc | PosKeys | NegKeys |
|----|------|-----------|-------------|-------------|------------|----------------|----------------|---------|---------|
| 0 | 5 | 5 | 3 | 3 | 4 | 0.5864 | 0.5992 | 6 | 6 |
| 1 | 5 | 4 | 4 | 4 | 3 | 0.5826 | 0.5759 | 6 | 6 |
| 2 | 4 | 3 | 4 | 2 | 3 | 0.6002 | 0.6004 | 6 | 6 |
| 3 | 4 | 4 | 2 | 1 | 1 | 0.5893 | 0.5860 | 6 | 6 |
| 4 | 4 | 3 | 1 | 1 | 2 | 0.5860 | 0.5853 | 6 | 6 |

## 6. Part 3: Signed Protective Signal Audit

### B_signed_sum_online

- num_positive_weight_keys: 6
- num_negative_weight_keys: 6
- total_abs_positive_weight: 3.0542
- total_abs_negative_weight: 3.1036
- candidates_with_protective_signal: 5
- candidates_where_negative_changed_selection: 5
- protective_signal_helpful_count: 0
- protective_signal_harmful_count: 5

#### Top 10 Positive Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_planks | craft_plank | wood-like | brittle_surface | 0.8513 |
| need_planks | craft_plank | wood-like | hollow_sound | 0.8513 |
| need_food | eat | apple-like | damp_texture | 0.6072 |
| need_food | eat | apple-like | brittle_surface | 0.6072 |
| need_stone | mine_by_hand | stone-like | brittle_surface | 0.0687 |
| need_stone | mine_by_hand | stone-like | hollow_sound | 0.0687 |

#### Top 10 Negative Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_planks | craft_plank | wood-like | damp_texture | -0.8806 |
| need_planks | craft_plank | wood-like | treated_surface | -0.8806 |
| need_stone | mine_by_hand | stone-like | damp_texture | -0.5722 |
| need_stone | mine_by_hand | stone-like | treated_surface | -0.5722 |
| need_food | eat | apple-like | treated_surface | -0.0991 |
| need_food | eat | apple-like | hollow_sound | -0.0991 |

### B_signed_max_abs_online

- num_positive_weight_keys: 8
- num_negative_weight_keys: 8
- total_abs_positive_weight: 3.2259
- total_abs_negative_weight: 3.0119
- candidates_with_protective_signal: 5
- candidates_where_negative_changed_selection: 5
- protective_signal_helpful_count: 0
- protective_signal_harmful_count: 5

#### Top 10 Positive Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_planks | craft_plank | wood-like | brittle_surface | 1.1181 |
| need_planks | craft_plank | wood-like | hollow_sound | 1.1181 |
| need_food | eat | apple-like | damp_texture | 0.2980 |
| need_food | eat | apple-like | brittle_surface | 0.2980 |
| need_stone | mine_with_pickaxe | stone-like | brittle_surface | 0.1158 |
| need_stone | mine_with_pickaxe | stone-like | hollow_sound | 0.1158 |
| need_stone | mine_by_hand | stone-like | brittle_surface | 0.0810 |
| need_stone | mine_by_hand | stone-like | hollow_sound | 0.0810 |

#### Top 10 Negative Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_planks | craft_plank | wood-like | damp_texture | -0.9745 |
| need_planks | craft_plank | wood-like | treated_surface | -0.9745 |
| need_stone | mine_by_hand | stone-like | damp_texture | -0.2270 |
| need_stone | mine_by_hand | stone-like | treated_surface | -0.2270 |
| need_stone | mine_with_pickaxe | stone-like | damp_texture | -0.1980 |
| need_stone | mine_with_pickaxe | stone-like | treated_surface | -0.1980 |
| need_food | eat | apple-like | treated_surface | -0.1064 |
| need_food | eat | apple-like | hollow_sound | -0.1064 |

## 7. Part 4: Matching Permuted Controls

| Real Variant | Permuted Variant | Real PV | Permuted PV | Real BAcc | Permuted BAcc | Real Beats Permuted |
|-------------|-----------------|---------|-------------|-----------|---------------|---------------------|
| B_signed_sum_online | C_signed_sum_online_permuted | 14 | 5 | 0.5889 | 0.5818 | False |
| B_signed_max_abs_online | C_signed_max_abs_online_permuted | 11 | 4 | 0.5894 | 0.5802 | False |

## 8. Part 5: Offline Upper-Bound Comparison

- B_signed_sum_online_total_pv: 14
- Offline_signed_sum_upper_bound_total_pv: 3
- online_gap_to_offline_upper_bound: 11
- online_risk_weights_nonzero: 12
- offline_risk_weights_nonzero: 24
- online_negative_weight_keys: 6
- offline_negative_weight_keys: 8
- overlap_top_10_online_offline_keys: 10

## 9. Part 6: Boolean Flag Details

### no_oracle_leakage_confirmed: True
  Rule: all online variants use only observed probe outcomes, no oracle access
  - all_online_variants: True

### signed_sum_online_beats_A: True
  Rule: B_signed_sum_online_total_pv < A_total_pv
  - signed_sum_pv: 14
  - A_pv: 22

### signed_sum_online_beats_positive_sum_online: True
  Rule: B_signed_sum_online_total_pv < B_sum_positive_online_total_pv
  - signed_sum_pv: 14
  - sum_positive_pv: 19

### signed_sum_online_beats_permuted: False
  Rule: B_signed_sum_online_total_pv < C_signed_sum_online_permuted_total_pv
  - signed_sum_pv: 14
  - permuted_pv: 5

### signed_max_abs_online_beats_A: True
  Rule: B_signed_max_abs_online_total_pv < A_total_pv
  - signed_max_pv: 11
  - A_pv: 22

### signed_max_abs_online_beats_permuted: False
  Rule: B_signed_max_abs_online_total_pv < C_signed_max_abs_online_permuted_total_pv
  - signed_max_pv: 11
  - permuted_pv: 4

### signed_online_beats_family_only_online: True
  Rule: at least one signed online variant beats family_only_online
  - signed_sum_pv: 14
  - signed_max_pv: 11
  - family_only_pv: 13

### protective_signal_used: True
  Rule: online memory has at least one negative weight key
  - num_negative_keys: 6

### protective_signal_net_helpful: False
  Rule: protective_signal_helpful_count >= protective_signal_harmful_count
  - helpful: 0
  - harmful: 5

### online_signed_approaches_offline_upper_bound: False
  Rule: online PV - offline PV <= 5
  - online_pv: 14
  - offline_pv: 3
  - gap: 11

### hard_exclusion_used: False
  Rule: hard_exclusion is always false for all variants

### deceptive_family_preservation_valid: True
  Rule: all deceptive objects preserve expected family identity
  - total_deceptive: 13
  - preserved: 13

### implementation_status: partial
  Rule: pass if signed online beats A and permuted; partial otherwise
  - signed_sum_beats_A: True
  - signed_sum_beats_permuted: False

### failure_reason: signed_beats_A_but_not_permuted
  Rule: derived from online signed validation results

## 10. Summary

```
[block_done]
block_id=1J24
A_total_pv=22
B_sum_positive_online_total_pv=19
B_signed_sum_online_total_pv=14
B_signed_max_abs_online_total_pv=11
C_signed_sum_online_permuted_total_pv=5
C_signed_max_abs_online_permuted_total_pv=4
Family_only_online_control_total_pv=13
Offline_signed_sum_upper_bound_total_pv=3
B_signed_sum_online_macro_delta_vs_A=-0.0048
B_signed_max_abs_online_macro_delta_vs_A=-0.0043
no_oracle_leakage_confirmed=True
signed_sum_online_beats_A=True
signed_sum_online_beats_positive_sum_online=True
signed_sum_online_beats_permuted=False
signed_max_abs_online_beats_A=True
signed_max_abs_online_beats_permuted=False
signed_online_beats_family_only_online=True
protective_signal_used=True
protective_signal_net_helpful=False
online_signed_approaches_offline_upper_bound=False
hard_exclusion_used=False
deceptive_family_preservation_valid=True
implementation_status=partial
failure_reason=signed_beats_A_but_not_permuted
elapsed=61.1s
```