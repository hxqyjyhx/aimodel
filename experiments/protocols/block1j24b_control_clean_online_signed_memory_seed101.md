# Block 1J24b -- Control-Clean Online Signed Memory Validation

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
| A_no_memory | 0.5864 |

## 4. Part 1: Main Comparison Table

| Variant | Total PV | Dec PV | Norm PV | Macro BAcc | Delta vs A | Eff Probes | Avoided Eff | Avoided PV | Avoided NV | Helpful | Harmful | Hard Excl | Oracle Evid | True Dec Label | Future Outcome | Same-Ep Future |
|---------|----------|--------|---------|------------|------------|------------|-------------|------------|------------|---------|---------|-----------|-------------|----------------|----------------|----------------|
| A_no_memory | 23 | 5 | 18 | 0.5864 | 0.0 | 35 | 0 | 0 | 0 | 0 | 0 | False | False | False | False | False |
| B_sum_positive_online | 21 | 4 | 17 | 0.5871 | 0.0007 | 34 | 1 | 1 | 2 | 1 | 2 | False | False | False | False | False |
| B_signed_sum_online | 9 | 3 | 6 | 0.5811 | -0.0053 | 29 | 6 | 5 | 7 | 5 | 7 | False | False | False | False | False |
| B_signed_max_abs_online | 9 | 3 | 6 | 0.5811 | -0.0053 | 29 | 6 | 5 | 7 | 5 | 7 | False | False | False | False | False |
| C_signed_sum_online_permuted | 27 | 24 | 3 | 0.5655 | -0.0209 | 31 | 4 | 9 | 9 | 9 | 9 | False | False | False | False | False |
| C_signed_max_abs_online_permuted | 27 | 24 | 3 | 0.5655 | -0.0209 | 31 | 4 | 9 | 9 | 9 | 9 | False | False | False | False | False |
| Family_only_online_control | 15 | 8 | 7 | 0.5627 | -0.0237 | 15 | 20 | 7 | 9 | 7 | 9 | False | False | False | False | False |
| Offline_signed_sum_upper_bound | 1 | 0 | 1 | 0.5843 | -0.0021 | 27 | 8 | 13 | 9 | 13 | 9 | False | True | False | False | False |

## 5. Part 2: Online Learning Curve

| Ep | A PV | SumPos PV | SignedSum PV | SignedMax PV | FamOnly PV | SignedSum BAcc | SignedMax BAcc | PosKeys | NegKeys |
|----|------|-----------|-------------|-------------|------------|----------------|----------------|---------|---------|
| 0 | 5 | 5 | 5 | 5 | 6 | 0.5847 | 0.5847 | 4 | 6 |
| 1 | 5 | 5 | 4 | 4 | 4 | 0.5722 | 0.5722 | 4 | 6 |
| 2 | 4 | 4 | 0 | 0 | 2 | 0.5860 | 0.5860 | 4 | 6 |
| 3 | 5 | 5 | 0 | 0 | 2 | 0.5805 | 0.5805 | 4 | 6 |
| 4 | 4 | 2 | 0 | 0 | 1 | 0.5820 | 0.5820 | 4 | 6 |

## 6. Part 3: Signed Protective Signal Audit

### B_signed_sum_online

- num_positive_weight_keys: 4
- num_negative_weight_keys: 6
- total_abs_positive_weight: 2.8792
- total_abs_negative_weight: 3.3283
- candidates_with_protective_signal: 5
- candidates_where_negative_changed_selection: 5
- protective_signal_helpful_count: 0
- protective_signal_harmful_count: 5

#### Top 10 Positive Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_planks | craft_plank | wood-like | hollow_sound | 1.2875 |
| need_planks | craft_plank | wood-like | brittle_surface | 1.2875 |
| need_food | eat | apple-like | damp_texture | 0.1520 |
| need_food | eat | apple-like | brittle_surface | 0.1520 |

#### Top 10 Negative Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_planks | craft_plank | wood-like | treated_surface | -1.0680 |
| need_planks | craft_plank | wood-like | damp_texture | -1.0680 |
| need_stone | mine_by_hand | stone-like | treated_surface | -0.4883 |
| need_stone | mine_by_hand | stone-like | damp_texture | -0.4883 |
| need_food | eat | apple-like | hollow_sound | -0.1079 |
| need_food | eat | apple-like | treated_surface | -0.1079 |

### B_signed_max_abs_online

- num_positive_weight_keys: 4
- num_negative_weight_keys: 6
- total_abs_positive_weight: 2.8792
- total_abs_negative_weight: 3.3283
- candidates_with_protective_signal: 5
- candidates_where_negative_changed_selection: 5
- protective_signal_helpful_count: 0
- protective_signal_harmful_count: 5

#### Top 10 Positive Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_planks | craft_plank | wood-like | hollow_sound | 1.2875 |
| need_planks | craft_plank | wood-like | brittle_surface | 1.2875 |
| need_food | eat | apple-like | damp_texture | 0.1520 |
| need_food | eat | apple-like | brittle_surface | 0.1520 |

#### Top 10 Negative Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_planks | craft_plank | wood-like | treated_surface | -1.0680 |
| need_planks | craft_plank | wood-like | damp_texture | -1.0680 |
| need_stone | mine_by_hand | stone-like | treated_surface | -0.4883 |
| need_stone | mine_by_hand | stone-like | damp_texture | -0.4883 |
| need_food | eat | apple-like | hollow_sound | -0.1079 |
| need_food | eat | apple-like | treated_surface | -0.1079 |

## 7. Part 4: Permutation Distribution (Fix 2: online_stepwise)

- permuted_control_mode: online_stepwise
- final_memory_permutation_used: False
- deterministic_permutation_seed_used: True
- n_permutation_repeats: 20

### Signed Sum Permuted Distribution

- min_pv: 9, max_pv: 27, mean_pv: 20.65, median_pv: 21, std_pv: 3.94
- real_pv: 9, real_percentile: 0.05
- num_better_than_real: 0, num_equal: 1, num_worse: 19
- all_permuted_pvs: [9, 18, 18, 18, 18, 18, 18, 18, 20, 20, 23, 23, 23, 23, 23, 23, 23, 23, 27, 27]

### Signed Max Abs Permuted Distribution

- min_pv: 9, max_pv: 27, mean_pv: 21.35, median_pv: 21, std_pv: 3.57
- real_pv: 9, real_percentile: 0.05
- num_better_than_real: 0, num_equal: 1, num_worse: 19
- all_permuted_pvs: [9, 20, 20, 20, 20, 20, 20, 20, 20, 20, 23, 23, 23, 23, 23, 23, 23, 23, 27, 27]

### Single Repeat (r00) Comparison

| Real Variant | Real PV | Permuted r00 PV | Real BAcc | Permuted r00 BAcc | Real Beats Permuted r00 |
|-------------|---------|-----------------|-----------|-------------------|------------------------|
| B_signed_sum_online | 9 | 27 | 0.5811 | 0.5655 | True |
| B_signed_max_abs_online | 9 | 27 | 0.5811 | 0.5655 | True |

## 8. Part 5: Offline Upper-Bound Comparison

- B_signed_sum_online_total_pv: 9
- Offline_signed_sum_upper_bound_total_pv: 1
- online_gap_to_offline_upper_bound: 8
- online_risk_weights_nonzero: 10
- offline_risk_weights_nonzero: 24
- online_negative_weight_keys: 6
- offline_negative_weight_keys: 8
- overlap_top_10_online_offline_keys: 8

## 9. Part 6: Boolean Flag Details

### shared_positions_used: True
  Rule: all variants use the same precomputed episode positions
  - position_seed_by_episode: ['00524a7a7d20', '9446507cb097', 'bb64109758fa', 'd766d1ac932b', '33e7883cb032']
  - position_hash_by_episode: {0: '00524a7a7d20', 1: '9446507cb097', 2: 'bb64109758fa', 3: 'd766d1ac932b', 4: '33e7883cb032'}

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
  - signed_sum_pv: 9
  - A_pv: 23

### signed_sum_online_beats_positive_sum_online: True
  Rule: B_signed_sum_online_total_pv < B_sum_positive_online_total_pv
  - signed_sum_pv: 9
  - sum_positive_pv: 21

### signed_sum_online_beats_permuted_median: True
  Rule: B_signed_sum_online_total_pv < median(20 permuted repeats total_pv)
  - signed_sum_pv: 9
  - permuted_median: 21
  - permuted_all: [9, 18, 18, 18, 18, 18, 18, 18, 20, 20, 23, 23, 23, 23, 23, 23, 23, 23, 27, 27]

### signed_sum_online_beats_80pct_of_permuted: True
  Rule: real PV better than >= 80% of permuted repeats
  - num_worse: 19
  - num_better: 0

### signed_max_abs_online_beats_A: True
  Rule: B_signed_max_abs_online_total_pv < A_total_pv
  - signed_max_pv: 9
  - A_pv: 23

### signed_max_abs_online_beats_permuted_median: True
  Rule: B_signed_max_abs_online_total_pv < median(20 permuted repeats total_pv)
  - signed_max_pv: 9
  - permuted_median: 21

### signed_max_abs_online_beats_80pct_of_permuted: True
  Rule: real PV better than >= 80% of permuted repeats
  - num_worse: 19
  - num_better: 0

### protective_signal_used: True
  Rule: online memory has at least one negative weight key
  - num_negative_keys: 6

### protective_signal_net_helpful: False
  Rule: protective_signal_helpful_count >= protective_signal_harmful_count
  - helpful: 0
  - harmful: 5

### online_signed_approaches_offline_upper_bound: False
  Rule: online PV - offline PV <= 5
  - online_pv: 9
  - offline_pv: 1
  - gap: 8

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
A_total_pv=23
B_sum_positive_online_total_pv=21
B_signed_sum_online_total_pv=9
B_signed_max_abs_online_total_pv=9
Family_only_online_control_total_pv=15
Offline_signed_sum_upper_bound_total_pv=1
signed_sum_permuted_median_pv=21
signed_sum_permuted_min_pv=9
signed_sum_permuted_max_pv=27
signed_sum_real_percentile_against_permuted=0.05
signed_max_abs_permuted_median_pv=21
signed_max_abs_permuted_min_pv=9
signed_max_abs_permuted_max_pv=27
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
elapsed=95.9s
```