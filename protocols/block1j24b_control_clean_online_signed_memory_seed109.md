# Block 1J24b -- Control-Clean Online Signed Memory Validation

## 1. Objective

Validate whether the strong 1J23 signed aggregation result holds in an online causal setting without offline/oracle evidence leakage.

## 2. Setup

| Parameter | Value |
|-----------|-------|
| Condition | C4_instance_subtype_cued_v1 |
| Seed | 109 |
| Budget | 1.5 |
| Episodes | 5 |
| Prior Violation Threshold | 0.6 |
| Min Violation Support | 2 |
| Risk Alpha | 5.0 |
| Max Risk | 2.0 |

## 3. Baselines

| Policy | Macro BAcc |
|--------|------------|
| C15b | 0.5779 |
| A_no_memory | 0.5678 |

## 4. Part 1: Main Comparison Table

| Variant | Total PV | Dec PV | Norm PV | Macro BAcc | Delta vs A | Eff Probes | Avoided Eff | Avoided PV | Avoided NV | Helpful | Harmful | Hard Excl | Oracle Evid | True Dec Label | Future Outcome | Same-Ep Future |
|---------|----------|--------|---------|------------|------------|------------|-------------|------------|------------|---------|---------|-----------|-------------|----------------|----------------|----------------|
| A_no_memory | 29 | 8 | 21 | 0.5678 | 0.0 | 39 | 0 | 0 | 0 | 0 | 0 | False | False | False | False | False |
| B_sum_positive_online | 28 | 7 | 21 | 0.5703 | 0.0025 | 38 | 1 | 4 | 0 | 4 | 0 | False | False | False | False | False |
| B_signed_sum_online | 27 | 6 | 21 | 0.5738 | 0.006 | 37 | 2 | 5 | 2 | 5 | 2 | False | False | False | False | False |
| B_signed_max_abs_online | 27 | 6 | 21 | 0.5738 | 0.006 | 37 | 2 | 5 | 2 | 5 | 2 | False | False | False | False | False |
| C_signed_sum_online_permuted | 26 | 23 | 3 | 0.5609 | -0.0069 | 31 | 8 | 16 | 5 | 16 | 5 | False | False | False | False | False |
| C_signed_max_abs_online_permuted | 26 | 23 | 3 | 0.5609 | -0.0069 | 31 | 8 | 16 | 5 | 16 | 5 | False | False | False | False | False |
| Family_only_online_control | 24 | 5 | 19 | 0.5700 | 0.0022 | 31 | 8 | 14 | 3 | 14 | 3 | False | False | False | False | False |
| Offline_signed_sum_upper_bound | 6 | 5 | 1 | 0.5887 | 0.0209 | 29 | 10 | 20 | 7 | 20 | 7 | False | True | False | False | False |

## 5. Part 2: Online Learning Curve

| Ep | A PV | SumPos PV | SignedSum PV | SignedMax PV | FamOnly PV | SignedSum BAcc | SignedMax BAcc | PosKeys | NegKeys |
|----|------|-----------|-------------|-------------|------------|----------------|----------------|---------|---------|
| 0 | 5 | 5 | 5 | 5 | 5 | 0.5644 | 0.5644 | 4 | 12 |
| 1 | 6 | 6 | 6 | 6 | 6 | 0.5685 | 0.5685 | 4 | 12 |
| 2 | 4 | 4 | 5 | 5 | 3 | 0.5839 | 0.5839 | 4 | 12 |
| 3 | 6 | 5 | 5 | 5 | 6 | 0.5834 | 0.5834 | 4 | 12 |
| 4 | 8 | 8 | 6 | 6 | 4 | 0.5688 | 0.5688 | 4 | 12 |

## 6. Part 3: Signed Protective Signal Audit

### B_signed_sum_online

- num_positive_weight_keys: 4
- num_negative_weight_keys: 12
- total_abs_positive_weight: 0.9389
- total_abs_negative_weight: 4.1827
- candidates_with_protective_signal: 3
- candidates_where_negative_changed_selection: 3
- protective_signal_helpful_count: 1
- protective_signal_harmful_count: 2

#### Top 10 Positive Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_food | eat | apple-like | damp_texture | 0.3174 |
| need_food | eat | apple-like | brittle_surface | 0.3174 |
| need_stone | mine_by_hand | stone-like | hollow_sound | 0.1520 |
| need_stone | mine_by_hand | stone-like | brittle_surface | 0.1520 |

#### Top 10 Negative Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_food | eat | apple-like | hollow_sound | -0.7041 |
| need_food | eat | apple-like | treated_surface | -0.7041 |
| need_planks | craft_plank | wood-like | damp_texture | -0.6885 |
| need_planks | craft_plank | wood-like | treated_surface | -0.6885 |
| need_planks | craft_plank | wood-like | hollow_sound | -0.3752 |
| need_planks | craft_plank | wood-like | brittle_surface | -0.3752 |
| need_tool | use_as_tool | tool-like | hollow_sound | -0.1079 |
| need_tool | use_as_tool | tool-like | damp_texture | -0.1079 |
| need_tool | use_as_tool | tool-like | treated_surface | -0.1079 |
| need_tool | use_as_tool | tool-like | brittle_surface | -0.1079 |

### B_signed_max_abs_online

- num_positive_weight_keys: 4
- num_negative_weight_keys: 12
- total_abs_positive_weight: 0.9389
- total_abs_negative_weight: 4.1827
- candidates_with_protective_signal: 3
- candidates_where_negative_changed_selection: 3
- protective_signal_helpful_count: 1
- protective_signal_harmful_count: 2

#### Top 10 Positive Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_food | eat | apple-like | damp_texture | 0.3174 |
| need_food | eat | apple-like | brittle_surface | 0.3174 |
| need_stone | mine_by_hand | stone-like | hollow_sound | 0.1520 |
| need_stone | mine_by_hand | stone-like | brittle_surface | 0.1520 |

#### Top 10 Negative Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_food | eat | apple-like | hollow_sound | -0.7041 |
| need_food | eat | apple-like | treated_surface | -0.7041 |
| need_planks | craft_plank | wood-like | damp_texture | -0.6885 |
| need_planks | craft_plank | wood-like | treated_surface | -0.6885 |
| need_planks | craft_plank | wood-like | hollow_sound | -0.3752 |
| need_planks | craft_plank | wood-like | brittle_surface | -0.3752 |
| need_tool | use_as_tool | tool-like | hollow_sound | -0.1079 |
| need_tool | use_as_tool | tool-like | damp_texture | -0.1079 |
| need_tool | use_as_tool | tool-like | treated_surface | -0.1079 |
| need_tool | use_as_tool | tool-like | brittle_surface | -0.1079 |

## 7. Part 4: Permutation Distribution (Fix 2: online_stepwise)

- permuted_control_mode: online_stepwise
- final_memory_permutation_used: False
- deterministic_permutation_seed_used: True
- n_permutation_repeats: 20

### Signed Sum Permuted Distribution

- min_pv: 24, max_pv: 27, mean_pv: 25.65, median_pv: 26, std_pv: 0.85
- real_pv: 27, real_percentile: 1.0
- num_better_than_real: 19, num_equal: 1, num_worse: 0
- all_permuted_pvs: [24, 24, 24, 24, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27]

### Signed Max Abs Permuted Distribution

- min_pv: 24, max_pv: 27, mean_pv: 25.65, median_pv: 26, std_pv: 0.85
- real_pv: 27, real_percentile: 1.0
- num_better_than_real: 19, num_equal: 1, num_worse: 0
- all_permuted_pvs: [24, 24, 24, 24, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27]

### Single Repeat (r00) Comparison

| Real Variant | Real PV | Permuted r00 PV | Real BAcc | Permuted r00 BAcc | Real Beats Permuted r00 |
|-------------|---------|-----------------|-----------|-------------------|------------------------|
| B_signed_sum_online | 27 | 26 | 0.5738 | 0.5609 | False |
| B_signed_max_abs_online | 27 | 26 | 0.5738 | 0.5609 | False |

## 8. Part 5: Offline Upper-Bound Comparison

- B_signed_sum_online_total_pv: 27
- Offline_signed_sum_upper_bound_total_pv: 6
- online_gap_to_offline_upper_bound: 21
- online_risk_weights_nonzero: 16
- offline_risk_weights_nonzero: 28
- online_negative_weight_keys: 12
- offline_negative_weight_keys: 10
- overlap_top_10_online_offline_keys: 10

## 9. Part 6: Boolean Flag Details

### shared_positions_used: True
  Rule: all variants use the same precomputed episode positions
  - position_seed_by_episode: ['92ca73f4cec0', '60cd592ff501', '1592523d1bc0', '102076764c05', '506b8f987598']
  - position_hash_by_episode: {0: '92ca73f4cec0', 1: '60cd592ff501', 2: '1592523d1bc0', 3: '102076764c05', 4: '506b8f987598'}

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
  - signed_sum_pv: 27
  - A_pv: 29

### signed_sum_online_beats_positive_sum_online: True
  Rule: B_signed_sum_online_total_pv < B_sum_positive_online_total_pv
  - signed_sum_pv: 27
  - sum_positive_pv: 28

### signed_sum_online_beats_permuted_median: False
  Rule: B_signed_sum_online_total_pv < median(20 permuted repeats total_pv)
  - signed_sum_pv: 27
  - permuted_median: 26
  - permuted_all: [24, 24, 24, 24, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27]

### signed_sum_online_beats_80pct_of_permuted: False
  Rule: real PV better than >= 80% of permuted repeats
  - num_worse: 0
  - num_better: 19

### signed_max_abs_online_beats_A: True
  Rule: B_signed_max_abs_online_total_pv < A_total_pv
  - signed_max_pv: 27
  - A_pv: 29

### signed_max_abs_online_beats_permuted_median: False
  Rule: B_signed_max_abs_online_total_pv < median(20 permuted repeats total_pv)
  - signed_max_pv: 27
  - permuted_median: 26

### signed_max_abs_online_beats_80pct_of_permuted: False
  Rule: real PV better than >= 80% of permuted repeats
  - num_worse: 0
  - num_better: 19

### protective_signal_used: True
  Rule: online memory has at least one negative weight key
  - num_negative_keys: 12

### protective_signal_net_helpful: False
  Rule: protective_signal_helpful_count >= protective_signal_harmful_count
  - helpful: 1
  - harmful: 2

### online_signed_approaches_offline_upper_bound: False
  Rule: online PV - offline PV <= 5
  - online_pv: 27
  - offline_pv: 6
  - gap: 21

### hard_exclusion_used: False
  Rule: hard_exclusion is always false for all variants

### deceptive_family_preservation_valid: True
  Rule: all deceptive objects preserve expected family identity
  - total_deceptive: 13
  - preserved: 13

### implementation_status: partial
  Rule: pass if signed online beats A and permuted_median; partial otherwise
  - signed_sum_beats_A: True
  - signed_sum_beats_permuted_median: False

### failure_reason: signed_beats_A_but_not_permuted_median
  Rule: derived from online signed validation results

## 10. Summary

```
[block_done]
block_id=1J24b
A_total_pv=29
B_sum_positive_online_total_pv=28
B_signed_sum_online_total_pv=27
B_signed_max_abs_online_total_pv=27
Family_only_online_control_total_pv=24
Offline_signed_sum_upper_bound_total_pv=6
signed_sum_permuted_median_pv=26
signed_sum_permuted_min_pv=24
signed_sum_permuted_max_pv=27
signed_sum_real_percentile_against_permuted=1.0
signed_max_abs_permuted_median_pv=26
signed_max_abs_permuted_min_pv=24
signed_max_abs_permuted_max_pv=27
signed_max_abs_real_percentile_against_permuted=1.0
shared_positions_used=True
permuted_control_mode=online_stepwise
final_memory_permutation_used=False
deterministic_permutation_seed_used=True
no_oracle_leakage_confirmed=True
signed_sum_online_beats_A=True
signed_sum_online_beats_positive_sum_online=True
signed_sum_online_beats_permuted_median=False
signed_max_abs_online_beats_A=True
signed_max_abs_online_beats_permuted_median=False
protective_signal_used=True
protective_signal_net_helpful=False
hard_exclusion_used=False
deceptive_family_preservation_valid=True
implementation_status=partial
failure_reason=signed_beats_A_but_not_permuted_median
elapsed=96.3s
```