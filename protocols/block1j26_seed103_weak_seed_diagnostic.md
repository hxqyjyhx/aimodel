# Block 1J26 -- Seed 103 Weak-Seed Diagnostic for Online Signed Memory

## 1. Objective

Validate whether the strong 1J23 signed aggregation result holds in an online causal setting without offline/oracle evidence leakage.

**This is a deep-dive diagnostic on seed 103**, identified as a weak seed
in the 1J25 multiseed robustness test (signed_sum PV=18 ties with permuted
median=18, vs seeds 101/107 where signed_sum strongly beats permuted).

## 2. Setup

| Parameter | Value |
|-----------|-------|
| Condition | C4_instance_subtype_cued_v1 |
| Seed | 103 |
| Budget | 1.5 |
| Episodes | 5 |
| Prior Violation Threshold | 0.6 |
| Min Violation Support | 2 |
| Risk Alpha | 5.0 |
| Max Risk | 2.0 |

## 3. Baselines

| Policy | Macro BAcc |
|--------|------------|
| C15b | 0.6025 |
| A_no_memory | 0.6337 |

## 4. Part 1: Main Comparison Table

| Variant | Total PV | Dec PV | Norm PV | Macro BAcc | Delta vs A | Eff Probes | Avoided Eff | Avoided PV | Avoided NV | Helpful | Harmful | Hard Excl | Oracle Evid | True Dec Label | Future Outcome | Same-Ep Future |
|---------|----------|--------|---------|------------|------------|------------|-------------|------------|------------|---------|---------|-----------|-------------|----------------|----------------|----------------|
| A_no_memory | 19 | 6 | 13 | 0.6337 | 0.0 | 35 | 0 | 0 | 0 | 0 | 0 | False | False | False | False | False |
| B_sum_positive_online | 18 | 4 | 14 | 0.6341 | 0.0004 | 34 | 1 | 1 | 0 | 1 | 0 | False | False | False | False | False |
| B_signed_sum_online | 18 | 8 | 10 | 0.6217 | -0.012 | 31 | 4 | 6 | 6 | 6 | 6 | False | False | False | False | False |
| B_signed_max_abs_online | 19 | 9 | 10 | 0.6209 | -0.0128 | 31 | 4 | 8 | 8 | 8 | 8 | False | False | False | False | False |
| C_signed_sum_online_permuted | 18 | 8 | 10 | 0.6217 | -0.012 | 31 | 4 | 6 | 6 | 6 | 6 | False | False | False | False | False |
| C_signed_max_abs_online_permuted | 19 | 9 | 10 | 0.6209 | -0.0128 | 31 | 4 | 8 | 8 | 8 | 8 | False | False | False | False | False |
| Family_only_online_control | 17 | 8 | 9 | 0.6286 | -0.0051 | 29 | 6 | 4 | 4 | 4 | 4 | False | False | False | False | False |
| Offline_signed_sum_upper_bound | 10 | 9 | 1 | 0.6331 | -0.0006 | 27 | 8 | 15 | 9 | 15 | 9 | False | True | False | False | False |

## 5. Part 2: Online Learning Curve

| Ep | A PV | SumPos PV | SignedSum PV | SignedMax PV | FamOnly PV | SignedSum BAcc | SignedMax BAcc | PosKeys | NegKeys |
|----|------|-----------|-------------|-------------|------------|----------------|----------------|---------|---------|
| 0 | 4 | 4 | 4 | 4 | 4 | 0.6402 | 0.6402 | 6 | 4 |
| 1 | 4 | 4 | 3 | 3 | 3 | 0.6182 | 0.6182 | 6 | 4 |
| 2 | 3 | 3 | 5 | 5 | 3 | 0.6172 | 0.6172 | 6 | 4 |
| 3 | 2 | 2 | 4 | 5 | 2 | 0.6085 | 0.6085 | 6 | 4 |
| 4 | 6 | 5 | 2 | 2 | 5 | 0.6242 | 0.6202 | 6 | 4 |

## 6. Part 3: Signed Protective Signal Audit

### B_signed_sum_online

- num_positive_weight_keys: 6
- num_negative_weight_keys: 4
- total_abs_positive_weight: 1.0967
- total_abs_negative_weight: 1.4045
- candidates_with_protective_signal: 4
- candidates_where_negative_changed_selection: 4
- protective_signal_helpful_count: 1
- protective_signal_harmful_count: 3

#### Top 10 Positive Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_food | eat | apple-like | damp_texture | 0.4266 |
| need_food | eat | apple-like | brittle_surface | 0.4266 |
| need_planks | craft_plank | wood-like | brittle_surface | 0.1064 |
| need_planks | craft_plank | wood-like | hollow_sound | 0.1064 |
| need_stone | mine_by_hand | stone-like | brittle_surface | 0.0155 |
| need_stone | mine_by_hand | stone-like | hollow_sound | 0.0155 |

#### Top 10 Negative Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_stone | mine_by_hand | stone-like | damp_texture | -0.4042 |
| need_stone | mine_by_hand | stone-like | treated_surface | -0.4042 |
| need_planks | craft_plank | wood-like | damp_texture | -0.2980 |
| need_planks | craft_plank | wood-like | treated_surface | -0.2980 |

### B_signed_max_abs_online

- num_positive_weight_keys: 4
- num_negative_weight_keys: 8
- total_abs_positive_weight: 2.0998
- total_abs_negative_weight: 2.2580
- candidates_with_protective_signal: 3
- candidates_where_negative_changed_selection: 3
- protective_signal_helpful_count: 2
- protective_signal_harmful_count: 1

#### Top 10 Positive Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_food | eat | apple-like | damp_texture | 0.7477 |
| need_food | eat | apple-like | brittle_surface | 0.7477 |
| need_food | eat | apple-like | treated_surface | 0.3021 |
| need_food | eat | apple-like | hollow_sound | 0.3021 |

#### Top 10 Negative Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_stone | mine_by_hand | stone-like | damp_texture | -0.5965 |
| need_stone | mine_by_hand | stone-like | treated_surface | -0.5965 |
| need_tool | use_as_tool | tool-like | damp_texture | -0.1980 |
| need_tool | use_as_tool | tool-like | treated_surface | -0.1980 |
| need_tool | use_as_tool | tool-like | brittle_surface | -0.1980 |
| need_tool | use_as_tool | tool-like | hollow_sound | -0.1980 |
| need_stone | mine_by_hand | stone-like | brittle_surface | -0.1364 |
| need_stone | mine_by_hand | stone-like | hollow_sound | -0.1364 |

## 7. Part 4: Permutation Distribution (Fix 2: online_stepwise)

- permuted_control_mode: online_stepwise
- final_memory_permutation_used: False
- deterministic_permutation_seed_used: True
- n_permutation_repeats: 100

### Signed Sum Permuted Distribution

- min_pv: 18, max_pv: 21, mean_pv: 18.14, median_pv: 18, std_pv: 0.62
- real_pv: 18, real_percentile: 0.95
- num_better_than_real: 0, num_equal: 95, num_worse: 5
- all_permuted_pvs: [18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 20, 21, 21, 21, 21]

### Signed Max Abs Permuted Distribution

- min_pv: 18, max_pv: 19, mean_pv: 18.2, median_pv: 18, std_pv: 0.4
- real_pv: 19, real_percentile: 1.0
- num_better_than_real: 80, num_equal: 20, num_worse: 0
- all_permuted_pvs: [18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19]

### Single Repeat (r00) Comparison

| Real Variant | Real PV | Permuted r00 PV | Real BAcc | Permuted r00 BAcc | Real Beats Permuted r00 |
|-------------|---------|-----------------|-----------|-------------------|------------------------|
| B_signed_sum_online | 18 | 18 | 0.6217 | 0.6217 | False |
| B_signed_max_abs_online | 19 | 19 | 0.6209 | 0.6209 | False |

## 8. Part 5: Offline Upper-Bound Comparison

- B_signed_sum_online_total_pv: 18
- Offline_signed_sum_upper_bound_total_pv: 10
- online_gap_to_offline_upper_bound: 8
- online_risk_weights_nonzero: 10
- offline_risk_weights_nonzero: 20
- online_negative_weight_keys: 4
- offline_negative_weight_keys: 6
- overlap_top_10_online_offline_keys: 6

## 9. Part 6: Boolean Flag Details

### shared_positions_used: True
  Rule: all variants use the same precomputed episode positions
  - position_seed_by_episode: ['a811b6842ab9', '36a820643892', 'cd4a30b0ce5b', 'fb2712907830', 'cfdff785aa4e']
  - position_hash_by_episode: {0: 'a811b6842ab9', 1: '36a820643892', 2: 'cd4a30b0ce5b', 3: 'fb2712907830', 4: 'cfdff785aa4e'}

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
  - signed_sum_pv: 18
  - A_pv: 19

### signed_sum_online_beats_positive_sum_online: False
  Rule: B_signed_sum_online_total_pv < B_sum_positive_online_total_pv
  - signed_sum_pv: 18
  - sum_positive_pv: 18

### signed_sum_online_beats_permuted_median: False
  Rule: B_signed_sum_online_total_pv < median(20 permuted repeats total_pv)
  - signed_sum_pv: 18
  - permuted_median: 18
  - permuted_all: [18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 20, 21, 21, 21, 21]

### signed_sum_online_beats_80pct_of_permuted: False
  Rule: real PV better than >= 80% of permuted repeats
  - num_worse: 5
  - num_better: 0

### signed_max_abs_online_beats_A: False
  Rule: B_signed_max_abs_online_total_pv < A_total_pv
  - signed_max_pv: 19
  - A_pv: 19

### signed_max_abs_online_beats_permuted_median: False
  Rule: B_signed_max_abs_online_total_pv < median(20 permuted repeats total_pv)
  - signed_max_pv: 19
  - permuted_median: 18

### signed_max_abs_online_beats_80pct_of_permuted: False
  Rule: real PV better than >= 80% of permuted repeats
  - num_worse: 0
  - num_better: 80

### protective_signal_used: True
  Rule: online memory has at least one negative weight key
  - num_negative_keys: 4

### protective_signal_net_helpful: False
  Rule: protective_signal_helpful_count >= protective_signal_harmful_count
  - helpful: 1
  - harmful: 3

### online_signed_approaches_offline_upper_bound: False
  Rule: online PV - offline PV <= 5
  - online_pv: 18
  - offline_pv: 10
  - gap: 8

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

## 10. Part 7: Weak-Seed Diagnostic

Deep-dive into why seed 103 shows weak signed-memory signal
compared to seeds 101 and 107.

### 10.1 Baseline and Ceiling

| Metric | Value |
|--------|-------|
| A baseline total PV | 19 |
| Offline upper bound total PV | 10 |
| Online-offline gap | 8 |
| Offline norm PV | 1 |
| Offline dec PV | 9 |
| High offline floor (>==8) | True |

### 10.2 Signed Aggregation Divergence

| Metric | Value |
|--------|-------|
| signed_sum total PV | 18 |
| signed_max_abs total PV | 19 |
| PV divergence (max - sum) | 1 |
| signed_sum beats A | True |
| signed_max_abs beats A | False |

### 10.3 Permuted Distribution (signed_sum)

- n_permutation_repeats: 100
- min_pv: 18, max_pv: 21
- mean_pv: 18.14, median_pv: 18
- std_pv: 0.62, range: 3
- real_pv: 18, real_percentile: 0.95
- num_worse: 5, num_equal: 95, num_better: 0
- pct_worse: 5.0%, pct_equal: 95.0%, pct_better: 0.0%
- Tight permuted distribution: True
- Real near permuted median: True

### 10.4 Protective Signal Quality

| Metric | Value |
|--------|-------|
| num_positive_weight_keys | 6 |
| num_negative_weight_keys | 4 |
| total_abs_positive_weight | 1.0967 |
| total_abs_negative_weight | 1.4045 |
| pos/neg weight ratio | 0.7809 |
| protective_signal_helpful | 1 |
| protective_signal_harmful | 3 |
| protective_signal_net_helpful | False |
| candidates_with_signal | 0 |
| candidates_changed_by_negative | 0 |
| pct where negative helped | 0.0% |

### 10.5 Online vs Offline Weight Growth

| Metric | Value |
|--------|-------|
| online_risk_weights_nonzero | 10 |
| offline_risk_weights_nonzero | 20 |
| online/offline nonzero ratio | 0.5 |
| overlap top-10 online/offline keys | 6 |

### 10.6 Learning Curve

- signed_sum PV per episode: [4, 3, 5, 4, 2]
- signed_sum PV episode range: 3
- signed_sum PV final episode: 2
- signed_sum PV best episode: 2
- A PV per episode: [4, 4, 3, 2, 6]

### 10.7 Aggregate Assessment

| Indicator | Value |
|-----------|-------|
| High offline floor (>==8) | True |
| Tight permuted distribution (range<==5) | True |
| Real near permuted median | True |
| Weak seed indicator count (out of 3) | 3 |
| **weak_seed_confirmed** | **True** |

**Interpretation:** A confirmed weak seed has >=2 of these indicators:
1. High offline floor: the oracle itself cannot eliminate many PVs
2. Tight permuted distribution: real signal indistinguishable from noise
3. Real PV near permuted median: signed memory adds no value over random permutation

## 11. Summary

```
[block_done]
block_id=1J26
A_total_pv=19
B_sum_positive_online_total_pv=18
B_signed_sum_online_total_pv=18
B_signed_max_abs_online_total_pv=19
Family_only_online_control_total_pv=17
Offline_signed_sum_upper_bound_total_pv=10
signed_sum_permuted_median_pv=18
signed_sum_permuted_min_pv=18
signed_sum_permuted_max_pv=21
signed_sum_real_percentile_against_permuted=0.95
signed_max_abs_permuted_median_pv=18
signed_max_abs_permuted_min_pv=18
signed_max_abs_permuted_max_pv=19
signed_max_abs_real_percentile_against_permuted=1.0
shared_positions_used=True
permuted_control_mode=online_stepwise
final_memory_permutation_used=False
deterministic_permutation_seed_used=True
no_oracle_leakage_confirmed=True
signed_sum_online_beats_A=True
signed_sum_online_beats_positive_sum_online=False
signed_sum_online_beats_permuted_median=False
signed_max_abs_online_beats_A=False
signed_max_abs_online_beats_permuted_median=False
protective_signal_used=True
protective_signal_net_helpful=False
hard_exclusion_used=False
deceptive_family_preservation_valid=True
implementation_status=partial
failure_reason=signed_beats_A_but_not_permuted_median
n_permutation_repeats=100
offline_upper_bound_pv=10
online_offline_gap=8
signed_sum_vs_signed_max_pv_divergence=1
permuted_distribution_range=3
pct_permuted_worse=5.0
pct_permuted_equal=95.0
positive_to_negative_weight_ratio=0.7809
protective_signal_net_helpful=False
high_offline_floor_indicator=True
tight_permuted_distribution_indicator=True
real_near_permuted_median_indicator=True
weak_seed_indicator_count=3
weak_seed_confirmed=True
elapsed=388.7s
```