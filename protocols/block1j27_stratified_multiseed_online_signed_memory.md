# Block 1J27 -- Stratified Multiseed Online Signed Memory Robustness

## 1. Objective

Test whether the 1J24b online signed memory result is stable across multiple seeds with ceiling stratification.

## 2. Setup

- Seeds: [101, 103, 107, 109, 113]
- Permutation repeats per seed: 20
- Valid seeds: 5/5

## 3. Per-Seed Results

| Seed | A PV | SumPos PV | SignedSum PV | SignedMax PV | FamOnly PV | Offline PV | SS Perm Median | SM Perm Median | SS %ile | SM %ile | SS Macro D | SM Macro D | Reduc vs A | Improv Room | Gap to Off | Beats A | Beats Perm | Ties Perm |
|------|------|-----------|-------------|-------------|------------|------------|----------------|----------------|---------|---------|------------|------------|------------|-------------|------------|---------|-----------|-----------|
| 101 | 23 | 21 | 9 | 9 | 15 | 1 | 21 | 21 | 0.050 | 0.050 | -0.0053 | -0.0053 | 14 | 22 | 8 | Y | Y | N |
| 103 | 19 | 18 | 18 | 19 | 17 | 10 | 18 | 18 | 0.950 | 1.000 | -0.0120 | -0.0128 | 1 | 9 | 8 | Y | N | Y |
| 107 | 18 | 19 | 12 | 12 | 17 | 5 | 26 | 26 | 0.050 | 0.050 | 0.0003 | 0.0003 | 6 | 13 | 7 | Y | Y | N |
| 109 | 29 | 28 | 27 | 27 | 24 | 6 | 26 | 26 | 1.000 | 1.000 | 0.0060 | 0.0060 | 2 | 23 | 21 | Y | N | N |
| 113 | 26 | 23 | 8 | 8 | 24 | 3 | 10 | 9 | 0.150 | 0.150 | -0.0050 | -0.0050 | 18 | 23 | 5 | Y | Y | N |

## 4. Seed Stratification

| Seed | Ceiling | Offline Improv Room | Online Gap to Offline | Mapping Advantage | Gap Large |
|------|---------|---------------------|----------------------|-------------------|-----------|
| 101 | high | 22 | 8 | present | Y |
| 103 | medium | 9 | 8 | absent | Y |
| 107 | high | 13 | 7 | present | Y |
| 109 | high | 23 | 21 | absent | Y |
| 113 | high | 23 | 5 | present | Y |

## 5. Aggregate Metrics

| Metric | Value |
|--------|-------|
| n_valid_seeds | 5 |
| mean_A_total_pv | 23.0 |
| mean_B_sum_positive_online_total_pv | 21.8 |
| mean_B_signed_sum_online_total_pv | 14.8 |
| mean_B_signed_max_abs_online_total_pv | 15.0 |
| mean_Family_only_online_control_total_pv | 19.4 |
| mean_Offline_signed_sum_upper_bound_total_pv | 5.0 |
| mean_signed_sum_pv_reduction_vs_A | 8.2 |
| mean_signed_max_abs_pv_reduction_vs_A | 8.0 |
| mean_offline_improvement_room | 18.0 |
| mean_online_gap_to_offline | 9.8 |
| mean_signed_sum_macro_delta_vs_A | -0.0032 |
| mean_signed_max_abs_macro_delta_vs_A | -0.0034 |
| seeds_where_signed_sum_beats_A | 5/5 |
| seeds_where_signed_max_abs_beats_A | 4/5 |
| seeds_where_signed_sum_beats_positive_sum | 4/5 |
| seeds_where_signed_sum_beats_family_only | 3/5 |
| seeds_where_signed_sum_beats_permuted_median | 3/5 |
| seeds_where_signed_sum_ties_permuted_median | 1/5 |
| seeds_where_signed_sum_loses_to_permuted_median | 1/5 |
| seeds_where_signed_max_abs_beats_permuted_median | 3/5 |
| high_ceiling_seed_count | 4 |
| medium_ceiling_seed_count | 1 |
| low_ceiling_seed_count | 0 |
| high_ceiling_seeds_where_signed_sum_beats_permuted_median | 3/4 |
| medium_ceiling_seeds_where_signed_sum_beats_permuted_median | 0/1 |
| low_ceiling_seeds_where_signed_sum_beats_permuted_median | N/A |
| seeds_with_no_oracle_leakage_confirmed | 5/5 |
| seeds_with_shared_positions_used | 5/5 |
| seeds_with_hard_exclusion_false | 5/5 |
| seeds_with_deceptive_family_preservation_valid | 5/5 |

## 6. Boolean Flags

### all_seeds_completed: True
  Rule: all 5 seeds completed successfully
  - n_valid: 5
  - n_expected: 5

### all_seeds_shared_positions_used: True
  Rule: all valid seeds used shared episode positions
  - count: 5
  - total: 5

### all_seeds_online_stepwise_permuted: True
  Rule: all valid seeds used online_stepwise permuted control

### all_seeds_no_final_memory_permutation: True
  Rule: no seed used frozen-final-memory permutation

### all_seeds_no_oracle_leakage: True
  Rule: all valid seeds confirmed no oracle leakage
  - count: 5
  - total: 5

### all_seeds_hard_exclusion_false: True
  Rule: no seed used hard exclusion
  - count: 5
  - total: 5

### all_seeds_family_preservation_valid: True
  Rule: all valid seeds preserved deceptive family identity
  - count: 5
  - total: 5

### signed_sum_beats_A_all_seeds: True
  Rule: B_signed_sum_online PV < A PV in all seeds
  - count: 5
  - total: 5

### signed_sum_beats_permuted_median_majority: True
  Rule: signed_sum beats permuted median in at least 3/5 seeds
  - count: 3
  - total: 5
  - threshold: 3

### signed_sum_beats_permuted_median_high_ceiling_seeds: False
  Rule: among high_ceiling seeds, signed_sum beats permuted median in all such seeds
  - high_ceiling_seeds: [101, 107, 109, 113]
  - n_high_ceiling: 4
  - n_beat_perm: 3

### signed_sum_macro_tradeoff_acceptable: True
  Rule: mean_macro_delta >= -0.01 and no seed < -0.02
  - mean_delta: -0.0032
  - min_delta: -0.012

### weak_seed_pattern_detected: True
  Rule: at least one seed has mapping_advantage_absent=true and online_learning_gap_large=true
  - seeds_with_pattern: [103, 109]

### online_offline_gap_detected: True
  Rule: mean_online_gap_to_offline >= 5
  - mean_online_gap_to_offline: 9.8

### stratified_robustness_criteria_met: True
  Rule: all_seeds_completed and no_leakage and online_stepwise and no_hard_excl and family_ok and signed_sum_beats_A_all and beats_permuted_median_majority and macro_tradeoff_acceptable
  - all_completed: True
  - no_leakage: True
  - online_stepwise: True
  - no_hard_excl: True
  - family_ok: True
  - ss_beats_A_all: True
  - ss_beats_perm_majority: True
  - ss_macro_tradeoff_ok: True

### implementation_status: pass
  Rule: pass if all stratified criteria met, partial otherwise

### failure_reason: none
  Rule: derived from stratified multiseed validation results

## 7. Summary

```
[block_done]
block_id=1J27
seeds=[101, 103, 107, 109, 113]
n_permutation_repeats=20
n_valid_seeds=5
mean_A_total_pv=23.0
mean_B_signed_sum_online_total_pv=14.8
mean_Offline_signed_sum_upper_bound_total_pv=5.0
mean_signed_sum_pv_reduction_vs_A=8.2
mean_offline_improvement_room=18.0
mean_online_gap_to_offline=9.8
seeds_where_signed_sum_beats_A=5/5
seeds_where_signed_sum_beats_permuted_median=3/5
seeds_where_signed_sum_ties_permuted_median=1/5
high_ceiling_seed_count=4
medium_ceiling_seed_count=1
low_ceiling_seed_count=0
high_ceiling_seeds_where_signed_sum_beats_permuted_median=3/4
signed_sum_macro_tradeoff_acceptable=True
weak_seed_pattern_detected=True
online_offline_gap_detected=True
stratified_robustness_criteria_met=True
implementation_status=pass
failure_reason=none
elapsed=478.6s
```