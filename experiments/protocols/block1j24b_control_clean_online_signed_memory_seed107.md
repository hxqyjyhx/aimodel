# Block 1J24b -- Control-Clean Online Signed Memory Validation

## 1. Objective

Validate whether the strong 1J23 signed aggregation result holds in an online causal setting without offline/oracle evidence leakage.

## 2. Setup

| Parameter | Value |
|-----------|-------|
| Condition | C4_instance_subtype_cued_v1 |
| Seed | 107 |
| Budget | 1.5 |
| Episodes | 5 |
| Prior Violation Threshold | 0.6 |
| Min Violation Support | 2 |
| Risk Alpha | 5.0 |
| Max Risk | 2.0 |

## 3. Baselines

| Policy | Macro BAcc |
|--------|------------|
| C15b | 0.6142 |
| A_no_memory | 0.6208 |

## 4. Part 1: Main Comparison Table

| Variant | Total PV | Dec PV | Norm PV | Macro BAcc | Delta vs A | Eff Probes | Avoided Eff | Avoided PV | Avoided NV | Helpful | Harmful | Hard Excl | Oracle Evid | True Dec Label | Future Outcome | Same-Ep Future |
|---------|----------|--------|---------|------------|------------|------------|-------------|------------|------------|---------|---------|-----------|-------------|----------------|----------------|----------------|
| A_no_memory | 18 | 5 | 13 | 0.6208 | 0.0 | 34 | 0 | 0 | 0 | 0 | 0 | False | False | False | False | False |
| B_sum_positive_online | 19 | 4 | 15 | 0.6212 | 0.0004 | 34 | 0 | 0 | 1 | 0 | 1 | False | False | False | False | False |
| B_signed_sum_online | 12 | 7 | 5 | 0.6211 | 0.0003 | 29 | 5 | 6 | 10 | 6 | 10 | False | False | False | False | False |
| B_signed_max_abs_online | 12 | 7 | 5 | 0.6211 | 0.0003 | 29 | 5 | 6 | 10 | 6 | 10 | False | False | False | False | False |
| C_signed_sum_online_permuted | 26 | 21 | 5 | 0.6088 | -0.012 | 30 | 4 | 8 | 10 | 8 | 10 | False | False | False | False | False |
| C_signed_max_abs_online_permuted | 26 | 21 | 5 | 0.6088 | -0.012 | 30 | 4 | 8 | 10 | 8 | 10 | False | False | False | False | False |
| Family_only_online_control | 17 | 8 | 9 | 0.6018 | -0.019 | 27 | 7 | 6 | 11 | 6 | 11 | False | False | False | False | False |
| Offline_signed_sum_upper_bound | 5 | 5 | 0 | 0.6195 | -0.0013 | 25 | 9 | 14 | 11 | 14 | 11 | False | True | False | False | False |

## 5. Part 2: Online Learning Curve

| Ep | A PV | SumPos PV | SignedSum PV | SignedMax PV | FamOnly PV | SignedSum BAcc | SignedMax BAcc | PosKeys | NegKeys |
|----|------|-----------|-------------|-------------|------------|----------------|----------------|---------|---------|
| 0 | 4 | 4 | 4 | 4 | 4 | 0.6256 | 0.6256 | 4 | 10 |
| 1 | 6 | 6 | 5 | 5 | 5 | 0.6037 | 0.6037 | 4 | 10 |
| 2 | 3 | 4 | 1 | 1 | 2 | 0.6300 | 0.6300 | 4 | 10 |
| 3 | 3 | 3 | 1 | 1 | 3 | 0.6343 | 0.6343 | 4 | 10 |
| 4 | 2 | 2 | 1 | 1 | 3 | 0.6119 | 0.6119 | 4 | 10 |

## 6. Part 3: Signed Protective Signal Audit

### B_signed_sum_online

- num_positive_weight_keys: 4
- num_negative_weight_keys: 10
- total_abs_positive_weight: 1.9297
- total_abs_negative_weight: 1.8728
- candidates_with_protective_signal: 5
- candidates_where_negative_changed_selection: 5
- protective_signal_helpful_count: 1
- protective_signal_harmful_count: 4

#### Top 10 Positive Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_planks | craft_plank | wood-like | brittle_surface | 0.6608 |
| need_planks | craft_plank | wood-like | hollow_sound | 0.6608 |
| need_food | eat | apple-like | brittle_surface | 0.3041 |
| need_food | eat | apple-like | damp_texture | 0.3041 |

#### Top 10 Negative Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_planks | craft_plank | wood-like | damp_texture | -0.2344 |
| need_planks | craft_plank | wood-like | treated_surface | -0.2344 |
| need_stone | mine_by_hand | stone-like | damp_texture | -0.1980 |
| need_stone | mine_by_hand | stone-like | treated_surface | -0.1980 |
| need_tool | use_as_tool | tool-like | brittle_surface | -0.1980 |
| need_tool | use_as_tool | tool-like | damp_texture | -0.1980 |
| need_tool | use_as_tool | tool-like | hollow_sound | -0.1980 |
| need_tool | use_as_tool | tool-like | treated_surface | -0.1980 |
| need_food | eat | apple-like | hollow_sound | -0.1079 |
| need_food | eat | apple-like | treated_surface | -0.1079 |

### B_signed_max_abs_online

- num_positive_weight_keys: 4
- num_negative_weight_keys: 10
- total_abs_positive_weight: 1.9297
- total_abs_negative_weight: 1.8728
- candidates_with_protective_signal: 5
- candidates_where_negative_changed_selection: 5
- protective_signal_helpful_count: 1
- protective_signal_harmful_count: 4

#### Top 10 Positive Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_planks | craft_plank | wood-like | brittle_surface | 0.6608 |
| need_planks | craft_plank | wood-like | hollow_sound | 0.6608 |
| need_food | eat | apple-like | brittle_surface | 0.3041 |
| need_food | eat | apple-like | damp_texture | 0.3041 |

#### Top 10 Negative Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_planks | craft_plank | wood-like | damp_texture | -0.2344 |
| need_planks | craft_plank | wood-like | treated_surface | -0.2344 |
| need_stone | mine_by_hand | stone-like | damp_texture | -0.1980 |
| need_stone | mine_by_hand | stone-like | treated_surface | -0.1980 |
| need_tool | use_as_tool | tool-like | brittle_surface | -0.1980 |
| need_tool | use_as_tool | tool-like | damp_texture | -0.1980 |
| need_tool | use_as_tool | tool-like | hollow_sound | -0.1980 |
| need_tool | use_as_tool | tool-like | treated_surface | -0.1980 |
| need_food | eat | apple-like | hollow_sound | -0.1079 |
| need_food | eat | apple-like | treated_surface | -0.1079 |

## 7. Part 4: Permutation Distribution (Fix 2: online_stepwise)

- permuted_control_mode: online_stepwise
- final_memory_permutation_used: False
- deterministic_permutation_seed_used: True
- n_permutation_repeats: 20

### Signed Sum Permuted Distribution

- min_pv: 12, max_pv: 26, mean_pv: 23.6, median_pv: 26, std_pv: 4.31
- real_pv: 12, real_percentile: 0.05
- num_better_than_real: 0, num_equal: 1, num_worse: 19
- all_permuted_pvs: [12, 17, 17, 18, 18, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26]

### Signed Max Abs Permuted Distribution

- min_pv: 12, max_pv: 26, mean_pv: 23.6, median_pv: 26, std_pv: 4.31
- real_pv: 12, real_percentile: 0.05
- num_better_than_real: 0, num_equal: 1, num_worse: 19
- all_permuted_pvs: [12, 17, 17, 18, 18, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26]

### Single Repeat (r00) Comparison

| Real Variant | Real PV | Permuted r00 PV | Real BAcc | Permuted r00 BAcc | Real Beats Permuted r00 |
|-------------|---------|-----------------|-----------|-------------------|------------------------|
| B_signed_sum_online | 12 | 26 | 0.6211 | 0.6088 | True |
| B_signed_max_abs_online | 12 | 26 | 0.6211 | 0.6088 | True |

## 8. Part 5: Offline Upper-Bound Comparison

- B_signed_sum_online_total_pv: 12
- Offline_signed_sum_upper_bound_total_pv: 5
- online_gap_to_offline_upper_bound: 7
- online_risk_weights_nonzero: 14
- offline_risk_weights_nonzero: 28
- online_negative_weight_keys: 10
- offline_negative_weight_keys: 6
- overlap_top_10_online_offline_keys: 8

## 9. Part 6: Boolean Flag Details

### shared_positions_used: True
  Rule: all variants use the same precomputed episode positions
  - position_seed_by_episode: ['a8035c7ec493', 'f1841da6cc8c', 'f0f04138762e', 'fd998b95acd4', '643a5c7faf21']
  - position_hash_by_episode: {0: 'a8035c7ec493', 1: 'f1841da6cc8c', 2: 'f0f04138762e', 3: 'fd998b95acd4', 4: '643a5c7faf21'}

### permuted_control_mode_is_online_stepwise: True
  Rule: C variants learn online and permute at decision time, not frozen final memory
  - permuted_control_mode: online_stepwise

### final_memory_permutation_used: False
  Rule: no variant uses final-memory post-hoc permutation

### deterministic_permutation_seed_used: True
  Rule: hashlib.md5 used instead of Python hash()

### no_oracle_leakage_confirmed: True
  Rule: all online variants have offline_oracle_evidence_used=false, true_deceptive_label_used=false, future_outcome_used=false, same_episode_future_probe_used=false
  - all_online_variants_clean: True

### signed_sum_online_beats_A: True
  Rule: B_signed_sum_online_total_pv < A_total_pv
  - signed_sum_pv: 12
  - A_pv: 18

### signed_sum_online_beats_positive_sum_online: True
  Rule: B_signed_sum_online_total_pv < B_sum_positive_online_total_pv
  - signed_sum_pv: 12
  - sum_positive_pv: 19

### signed_sum_online_beats_permuted_median: True
  Rule: B_signed_sum_online_total_pv < median(20 permuted repeats total_pv)
  - signed_sum_pv: 12
  - permuted_median: 26
  - permuted_all: [12, 17, 17, 18, 18, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26]

### signed_sum_online_beats_80pct_of_permuted: True
  Rule: real PV better than >= 80% of permuted repeats
  - num_worse: 19
  - num_better: 0

### signed_max_abs_online_beats_A: True
  Rule: B_signed_max_abs_online_total_pv < A_total_pv
  - signed_max_pv: 12
  - A_pv: 18

### signed_max_abs_online_beats_permuted_median: True
  Rule: B_signed_max_abs_online_total_pv < median(20 permuted repeats total_pv)
  - signed_max_pv: 12
  - permuted_median: 26

### signed_max_abs_online_beats_80pct_of_permuted: True
  Rule: real PV better than >= 80% of permuted repeats
  - num_worse: 19
  - num_better: 0

### protective_signal_used: True
  Rule: online memory has at least one negative weight key
  - num_negative_keys: 10

### protective_signal_net_helpful: False
  Rule: protective_signal_helpful_count >= protective_signal_harmful_count
  - helpful: 1
  - harmful: 4

### online_signed_approaches_offline_upper_bound: False
  Rule: online PV - offline PV <= 5
  - online_pv: 12
  - offline_pv: 5
  - gap: 7

### hard_exclusion_used: False
  Rule: hard_exclusion is always false for all variants

### deceptive_family_preservation_valid: True
  Rule: all deceptive objects preserve expected family identity
  - total_deceptive: 13
  - preserved: 13

### implementation_status: pass
  Rule: pass if signed online beats A and permuted_median; partial otherwise
  - signed_sum_beats_A: True
  - signed_sum_beats_permuted_median: True

### failure_reason: none
  Rule: derived from online signed validation results

## 10. Summary

```
[block_done]
block_id=1J24b
A_total_pv=18
B_sum_positive_online_total_pv=19
B_signed_sum_online_total_pv=12
B_signed_max_abs_online_total_pv=12
Family_only_online_control_total_pv=17
Offline_signed_sum_upper_bound_total_pv=5
signed_sum_permuted_median_pv=26
signed_sum_permuted_min_pv=12
signed_sum_permuted_max_pv=26
signed_sum_real_percentile_against_permuted=0.05
signed_max_abs_permuted_median_pv=26
signed_max_abs_permuted_min_pv=12
signed_max_abs_permuted_max_pv=26
signed_max_abs_real_percentile_against_permuted=0.05
shared_positions_used=True
permuted_control_mode=online_stepwise
final_memory_permutation_used=False
deterministic_permutation_seed_used=True
no_oracle_leakage_confirmed=True
signed_sum_online_beats_A=True
signed_sum_online_beats_positive_sum_online=True
signed_sum_online_beats_permuted_median=True
signed_max_abs_online_beats_A=True
signed_max_abs_online_beats_permuted_median=True
protective_signal_used=True
protective_signal_net_helpful=False
hard_exclusion_used=False
deceptive_family_preservation_valid=True
implementation_status=pass
failure_reason=none
elapsed=94.0s
```