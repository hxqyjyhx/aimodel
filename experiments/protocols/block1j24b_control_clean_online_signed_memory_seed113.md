# Block 1J24b -- Control-Clean Online Signed Memory Validation

## 1. Objective

Validate whether the strong 1J23 signed aggregation result holds in an online causal setting without offline/oracle evidence leakage.

## 2. Setup

| Parameter | Value |
|-----------|-------|
| Condition | C4_instance_subtype_cued_v1 |
| Seed | 113 |
| Budget | 1.5 |
| Episodes | 5 |
| Prior Violation Threshold | 0.6 |
| Min Violation Support | 2 |
| Risk Alpha | 5.0 |
| Max Risk | 2.0 |

## 3. Baselines

| Policy | Macro BAcc |
|--------|------------|
| C15b | 0.6175 |
| A_no_memory | 0.6557 |

## 4. Part 1: Main Comparison Table

| Variant | Total PV | Dec PV | Norm PV | Macro BAcc | Delta vs A | Eff Probes | Avoided Eff | Avoided PV | Avoided NV | Helpful | Harmful | Hard Excl | Oracle Evid | True Dec Label | Future Outcome | Same-Ep Future |
|---------|----------|--------|---------|------------|------------|------------|-------------|------------|------------|---------|---------|-----------|-------------|----------------|----------------|----------------|
| A_no_memory | 26 | 8 | 18 | 0.6557 | 0.0 | 40 | 0 | 0 | 0 | 0 | 0 | False | False | False | False | False |
| B_sum_positive_online | 23 | 5 | 18 | 0.6599 | 0.0042 | 38 | 2 | 3 | 1 | 3 | 1 | False | False | False | False | False |
| B_signed_sum_online | 8 | 2 | 6 | 0.6507 | -0.005 | 31 | 9 | 10 | 9 | 10 | 9 | False | False | False | False | False |
| B_signed_max_abs_online | 8 | 2 | 6 | 0.6507 | -0.005 | 31 | 9 | 10 | 9 | 10 | 9 | False | False | False | False | False |
| C_signed_sum_online_permuted | 8 | 2 | 6 | 0.6507 | -0.005 | 31 | 9 | 11 | 9 | 11 | 9 | False | False | False | False | False |
| C_signed_max_abs_online_permuted | 8 | 2 | 6 | 0.6507 | -0.005 | 31 | 9 | 11 | 9 | 11 | 9 | False | False | False | False | False |
| Family_only_online_control | 24 | 14 | 10 | 0.6622 | 0.0065 | 35 | 5 | 8 | 5 | 8 | 5 | False | False | False | False | False |
| Offline_signed_sum_upper_bound | 3 | 0 | 3 | 0.6561 | 0.0004 | 30 | 10 | 14 | 10 | 14 | 10 | False | True | False | False | False |

## 5. Part 2: Online Learning Curve

| Ep | A PV | SumPos PV | SignedSum PV | SignedMax PV | FamOnly PV | SignedSum BAcc | SignedMax BAcc | PosKeys | NegKeys |
|----|------|-----------|-------------|-------------|------------|----------------|----------------|---------|---------|
| 0 | 6 | 6 | 5 | 5 | 6 | 0.6512 | 0.6512 | 4 | 8 |
| 1 | 5 | 5 | 0 | 0 | 4 | 0.6424 | 0.6424 | 4 | 8 |
| 2 | 6 | 4 | 1 | 1 | 5 | 0.6547 | 0.6547 | 4 | 8 |
| 3 | 6 | 5 | 1 | 1 | 4 | 0.6587 | 0.6587 | 4 | 8 |
| 4 | 3 | 3 | 1 | 1 | 5 | 0.6464 | 0.6464 | 4 | 8 |

## 6. Part 3: Signed Protective Signal Audit

### B_signed_sum_online

- num_positive_weight_keys: 4
- num_negative_weight_keys: 8
- total_abs_positive_weight: 2.5571
- total_abs_negative_weight: 4.0478
- candidates_with_protective_signal: 5
- candidates_where_negative_changed_selection: 5
- protective_signal_helpful_count: 0
- protective_signal_harmful_count: 5

#### Top 10 Positive Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_planks | craft_plank | wood-like | hollow_sound | 1.1627 |
| need_planks | craft_plank | wood-like | brittle_surface | 1.1627 |
| need_stone | mine_by_hand | stone-like | hollow_sound | 0.1158 |
| need_stone | mine_by_hand | stone-like | brittle_surface | 0.1158 |

#### Top 10 Negative Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_planks | craft_plank | wood-like | damp_texture | -1.4298 |
| need_planks | craft_plank | wood-like | treated_surface | -1.4298 |
| need_tool | use_as_tool | tool-like | damp_texture | -0.1980 |
| need_tool | use_as_tool | tool-like | hollow_sound | -0.1980 |
| need_tool | use_as_tool | tool-like | treated_surface | -0.1980 |
| need_tool | use_as_tool | tool-like | brittle_surface | -0.1980 |
| need_stone | mine_by_hand | stone-like | damp_texture | -0.1980 |
| need_stone | mine_by_hand | stone-like | treated_surface | -0.1980 |

### B_signed_max_abs_online

- num_positive_weight_keys: 4
- num_negative_weight_keys: 8
- total_abs_positive_weight: 2.5571
- total_abs_negative_weight: 4.0478
- candidates_with_protective_signal: 5
- candidates_where_negative_changed_selection: 5
- protective_signal_helpful_count: 0
- protective_signal_harmful_count: 5

#### Top 10 Positive Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_planks | craft_plank | wood-like | hollow_sound | 1.1627 |
| need_planks | craft_plank | wood-like | brittle_surface | 1.1627 |
| need_stone | mine_by_hand | stone-like | hollow_sound | 0.1158 |
| need_stone | mine_by_hand | stone-like | brittle_surface | 0.1158 |

#### Top 10 Negative Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_planks | craft_plank | wood-like | damp_texture | -1.4298 |
| need_planks | craft_plank | wood-like | treated_surface | -1.4298 |
| need_tool | use_as_tool | tool-like | damp_texture | -0.1980 |
| need_tool | use_as_tool | tool-like | hollow_sound | -0.1980 |
| need_tool | use_as_tool | tool-like | treated_surface | -0.1980 |
| need_tool | use_as_tool | tool-like | brittle_surface | -0.1980 |
| need_stone | mine_by_hand | stone-like | damp_texture | -0.1980 |
| need_stone | mine_by_hand | stone-like | treated_surface | -0.1980 |

## 7. Part 4: Permutation Distribution (Fix 2: online_stepwise)

- permuted_control_mode: online_stepwise
- final_memory_permutation_used: False
- deterministic_permutation_seed_used: True
- n_permutation_repeats: 20

### Signed Sum Permuted Distribution

- min_pv: 8, max_pv: 22, mean_pv: 15.0, median_pv: 10, std_pv: 6.18
- real_pv: 8, real_percentile: 0.15
- num_better_than_real: 0, num_equal: 3, num_worse: 17
- all_permuted_pvs: [8, 8, 8, 10, 10, 10, 10, 10, 10, 10, 10, 20, 22, 22, 22, 22, 22, 22, 22, 22]

### Signed Max Abs Permuted Distribution

- min_pv: 8, max_pv: 22, mean_pv: 14.6, median_pv: 9, std_pv: 6.51
- real_pv: 8, real_percentile: 0.15
- num_better_than_real: 0, num_equal: 3, num_worse: 17
- all_permuted_pvs: [8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 20, 22, 22, 22, 22, 22, 22, 22, 22]

### Single Repeat (r00) Comparison

| Real Variant | Real PV | Permuted r00 PV | Real BAcc | Permuted r00 BAcc | Real Beats Permuted r00 |
|-------------|---------|-----------------|-----------|-------------------|------------------------|
| B_signed_sum_online | 8 | 8 | 0.6507 | 0.6507 | False |
| B_signed_max_abs_online | 8 | 8 | 0.6507 | 0.6507 | False |

## 8. Part 5: Offline Upper-Bound Comparison

- B_signed_sum_online_total_pv: 8
- Offline_signed_sum_upper_bound_total_pv: 3
- online_gap_to_offline_upper_bound: 5
- online_risk_weights_nonzero: 12
- offline_risk_weights_nonzero: 22
- online_negative_weight_keys: 8
- offline_negative_weight_keys: 6
- overlap_top_10_online_offline_keys: 4

## 9. Part 6: Boolean Flag Details

### shared_positions_used: True
  Rule: all variants use the same precomputed episode positions
  - position_seed_by_episode: ['5ef8128f7f8f', 'f0d47af37417', '27d825b5374f', '7ec773525ce7', '4023d0f1c805']
  - position_hash_by_episode: {0: '5ef8128f7f8f', 1: 'f0d47af37417', 2: '27d825b5374f', 3: '7ec773525ce7', 4: '4023d0f1c805'}

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
  - signed_sum_pv: 8
  - A_pv: 26

### signed_sum_online_beats_positive_sum_online: True
  Rule: B_signed_sum_online_total_pv < B_sum_positive_online_total_pv
  - signed_sum_pv: 8
  - sum_positive_pv: 23

### signed_sum_online_beats_permuted_median: True
  Rule: B_signed_sum_online_total_pv < median(20 permuted repeats total_pv)
  - signed_sum_pv: 8
  - permuted_median: 10
  - permuted_all: [8, 8, 8, 10, 10, 10, 10, 10, 10, 10, 10, 20, 22, 22, 22, 22, 22, 22, 22, 22]

### signed_sum_online_beats_80pct_of_permuted: True
  Rule: real PV better than >= 80% of permuted repeats
  - num_worse: 17
  - num_better: 0

### signed_max_abs_online_beats_A: True
  Rule: B_signed_max_abs_online_total_pv < A_total_pv
  - signed_max_pv: 8
  - A_pv: 26

### signed_max_abs_online_beats_permuted_median: True
  Rule: B_signed_max_abs_online_total_pv < median(20 permuted repeats total_pv)
  - signed_max_pv: 8
  - permuted_median: 9

### signed_max_abs_online_beats_80pct_of_permuted: True
  Rule: real PV better than >= 80% of permuted repeats
  - num_worse: 17
  - num_better: 0

### protective_signal_used: True
  Rule: online memory has at least one negative weight key
  - num_negative_keys: 8

### protective_signal_net_helpful: False
  Rule: protective_signal_helpful_count >= protective_signal_harmful_count
  - helpful: 0
  - harmful: 5

### online_signed_approaches_offline_upper_bound: True
  Rule: online PV - offline PV <= 5
  - online_pv: 8
  - offline_pv: 3
  - gap: 5

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
A_total_pv=26
B_sum_positive_online_total_pv=23
B_signed_sum_online_total_pv=8
B_signed_max_abs_online_total_pv=8
Family_only_online_control_total_pv=24
Offline_signed_sum_upper_bound_total_pv=3
signed_sum_permuted_median_pv=10
signed_sum_permuted_min_pv=8
signed_sum_permuted_max_pv=22
signed_sum_real_percentile_against_permuted=0.15
signed_max_abs_permuted_median_pv=9
signed_max_abs_permuted_min_pv=8
signed_max_abs_permuted_max_pv=22
signed_max_abs_real_percentile_against_permuted=0.15
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
elapsed=97.5s
```