# Block 1J25 -- Small Multiseed Online Signed Memory Robustness

## 1. Objective

Test whether the 1J24b online signed memory result is stable across multiple seeds.

## 2. Setup

- Seeds: [101, 103, 107]
- Permutation repeats per seed: 10
- Valid seeds: 3/3

## 3. Per-Seed Results

| Seed | A PV | SumPos PV | SignedSum PV | SignedMax PV | FamOnly PV | Offline PV | SS Perm Median | SM Perm Median | SS %ile | SM %ile | SS Macro D | SM Macro D |
|------|------|-----------|-------------|-------------|------------|------------|----------------|----------------|---------|---------|------------|------------|
| 101 | 23 | 21 | 9 | 9 | 15 | 1 | 23 | 23 | 0.000 | 0.000 | -0.0053 | -0.0053 |
| 103 | 19 | 18 | 18 | 19 | 17 | 10 | 18 | 18 | 0.900 | 1.000 | -0.0120 | -0.0128 |
| 107 | 18 | 19 | 12 | 12 | 17 | 5 | 26 | 26 | 0.000 | 0.000 | 0.0003 | 0.0003 |

## 4. Aggregate Metrics

| Metric | Value |
|--------|-------|
| mean_A_total_pv | 20.0 |
| mean_B_sum_positive_online_total_pv | 19.3333 |
| mean_B_signed_sum_online_total_pv | 13.0 |
| mean_B_signed_max_abs_online_total_pv | 13.3333 |
| mean_Family_only_online_control_total_pv | 16.3333 |
| mean_Offline_signed_sum_upper_bound_total_pv | 5.3333 |
| mean_signed_sum_pv_reduction_vs_A | -7.0 |
| mean_signed_max_abs_pv_reduction_vs_A | -6.6667000000000005 |
| mean_signed_sum_macro_delta_vs_A | -0.0057 |
| mean_signed_max_abs_macro_delta_vs_A | -0.0059 |
| seeds_where_signed_sum_beats_A | 3/3 |
| seeds_where_signed_max_abs_beats_A | 2/3 |
| seeds_where_signed_sum_beats_positive_sum | 2/3 |
| seeds_where_signed_sum_beats_family_only | 2/3 |
| seeds_where_signed_sum_beats_permuted_median | 2/3 |
| seeds_where_signed_max_abs_beats_permuted_median | 2/3 |
| seeds_with_no_oracle_leakage_confirmed | 3/3 |
| seeds_with_shared_positions_used | 3/3 |
| seeds_with_hard_exclusion_false | 3/3 |
| seeds_with_deceptive_family_preservation_valid | 3/3 |

## 5. Boolean Flags

### all_seeds_completed: True
  Rule: all 3 seeds completed successfully
  - n_valid: 3
  - n_expected: 3

### all_seeds_shared_positions_used: True
  Rule: all valid seeds used shared episode positions
  - count: 3
  - total: 3

### all_seeds_online_stepwise_permuted: True
  Rule: all valid seeds used online_stepwise permuted control

### all_seeds_no_final_memory_permutation: True
  Rule: no seed used frozen-final-memory permutation

### all_seeds_no_oracle_leakage: True
  Rule: all valid seeds confirmed no oracle leakage
  - count: 3
  - total: 3

### all_seeds_hard_exclusion_false: True
  Rule: no seed used hard exclusion
  - count: 3
  - total: 3

### all_seeds_family_preservation_valid: True
  Rule: all valid seeds preserved deceptive family identity
  - count: 3
  - total: 3

### signed_sum_beats_A_all_seeds: True
  Rule: B_signed_sum_online PV < A PV in all seeds
  - count: 3
  - total: 3

### signed_max_abs_beats_A_all_seeds: False
  Rule: B_signed_max_abs_online PV < A PV in all seeds
  - count: 2
  - total: 3

### signed_sum_beats_positive_sum_all_seeds: False
  Rule: B_signed_sum_online PV < sum_positive_online PV in all seeds
  - count: 2
  - total: 3

### signed_sum_beats_family_only_majority: True
  Rule: signed_sum beats family_only in majority of seeds
  - count: 2
  - total: 3

### signed_sum_beats_permuted_median_all_seeds: False
  Rule: signed_sum PV < permuted median PV in all seeds
  - count: 2
  - total: 3

### signed_max_abs_beats_permuted_median_all_seeds: False
  Rule: signed_max_abs PV < permuted median PV in all seeds
  - count: 2
  - total: 3

### signed_sum_macro_tradeoff_acceptable: True
  Rule: mean_macro_delta >= -0.01 and no seed < -0.02
  - mean_delta: -0.0057
  - min_delta: -0.012

### signed_max_abs_macro_tradeoff_acceptable: True
  Rule: mean_macro_delta >= -0.01 and no seed < -0.02
  - mean_delta: -0.0059
  - min_delta: -0.0128

### robustness_criteria_met: False
  Rule: all_seeds_completed and no_leakage and online_stepwise and no_hard_excl and family_ok and at least one signed variant passes all criteria
  - signed_sum_robust: False
  - signed_max_abs_robust: False

### implementation_status: partial
  Rule: pass if all criteria met, partial otherwise

### failure_reason: robustness_criteria_not_met
  Rule: derived from multiseed validation results

## 6. Summary

```
[block_done]
block_id=1J25
seeds=[101, 103, 107]
n_permutation_repeats=10
mean_A_total_pv=20.0
mean_B_sum_positive_online_total_pv=19.3333
mean_B_signed_sum_online_total_pv=13.0
mean_B_signed_max_abs_online_total_pv=13.3333
mean_Family_only_online_control_total_pv=16.3333
mean_Offline_signed_sum_upper_bound_total_pv=5.3333
seeds_where_signed_sum_beats_A=3/3
seeds_where_signed_max_abs_beats_A=2/3
seeds_where_signed_sum_beats_permuted_median=2/3
seeds_where_signed_max_abs_beats_permuted_median=2/3
mean_signed_sum_macro_delta_vs_A=-0.0057
mean_signed_max_abs_macro_delta_vs_A=-0.0059
all_seeds_no_oracle_leakage=True
all_seeds_shared_positions_used=True
all_seeds_online_stepwise_permuted=True
all_seeds_no_final_memory_permutation=True
signed_sum_macro_tradeoff_acceptable=True
signed_max_abs_macro_tradeoff_acceptable=True
robustness_criteria_met=False
implementation_status=partial
failure_reason=robustness_criteria_not_met
elapsed=187.6s
```