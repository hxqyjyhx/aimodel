# Block 1J28 -- Seed 109 Weak-Seed Diagnostic for Online Signed Memory

## 1. Objective

Validate whether the strong 1J23 signed aggregation result holds in an online causal setting without offline/oracle evidence leakage.

**This is a deep-dive diagnostic on seed 109**, identified as a weak seed
in the 1J25 multiseed robustness test (signed_sum PV=18 ties with permuted
median=18, vs seeds 101/107 where signed_sum strongly beats permuted).

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
| need_food | eat | apple-like | brittle_surface | 0.3174 |
| need_food | eat | apple-like | damp_texture | 0.3174 |
| need_stone | mine_by_hand | stone-like | brittle_surface | 0.1520 |
| need_stone | mine_by_hand | stone-like | hollow_sound | 0.1520 |

#### Top 10 Negative Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_food | eat | apple-like | treated_surface | -0.7041 |
| need_food | eat | apple-like | hollow_sound | -0.7041 |
| need_planks | craft_plank | wood-like | treated_surface | -0.6885 |
| need_planks | craft_plank | wood-like | damp_texture | -0.6885 |
| need_planks | craft_plank | wood-like | brittle_surface | -0.3752 |
| need_planks | craft_plank | wood-like | hollow_sound | -0.3752 |
| need_tool | use_as_tool | tool-like | brittle_surface | -0.1079 |
| need_tool | use_as_tool | tool-like | treated_surface | -0.1079 |
| need_tool | use_as_tool | tool-like | hollow_sound | -0.1079 |
| need_tool | use_as_tool | tool-like | damp_texture | -0.1079 |

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
| need_food | eat | apple-like | brittle_surface | 0.3174 |
| need_food | eat | apple-like | damp_texture | 0.3174 |
| need_stone | mine_by_hand | stone-like | brittle_surface | 0.1520 |
| need_stone | mine_by_hand | stone-like | hollow_sound | 0.1520 |

#### Top 10 Negative Keys

| Goal | Action | Family | Dev Feat | Signed Weight |
|------|--------|--------|-----------|---------------|
| need_food | eat | apple-like | treated_surface | -0.7041 |
| need_food | eat | apple-like | hollow_sound | -0.7041 |
| need_planks | craft_plank | wood-like | treated_surface | -0.6885 |
| need_planks | craft_plank | wood-like | damp_texture | -0.6885 |
| need_planks | craft_plank | wood-like | brittle_surface | -0.3752 |
| need_planks | craft_plank | wood-like | hollow_sound | -0.3752 |
| need_tool | use_as_tool | tool-like | brittle_surface | -0.1079 |
| need_tool | use_as_tool | tool-like | treated_surface | -0.1079 |
| need_tool | use_as_tool | tool-like | hollow_sound | -0.1079 |
| need_tool | use_as_tool | tool-like | damp_texture | -0.1079 |

## 7. Part 4: Permutation Distribution (Fix 2: online_stepwise)

- permuted_control_mode: online_stepwise
- final_memory_permutation_used: False
- deterministic_permutation_seed_used: True
- n_permutation_repeats: 100

### Signed Sum Permuted Distribution

- min_pv: 18, max_pv: 27, mean_pv: 25.65, median_pv: 26, std_pv: 1.53
- real_pv: 27, real_percentile: 1.0
- num_better_than_real: 97, num_equal: 3, num_worse: 0
- all_permuted_pvs: [18, 18, 19, 19, 24, 24, 24, 24, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27]

### Signed Max Abs Permuted Distribution

- min_pv: 17, max_pv: 27, mean_pv: 25.63, median_pv: 26, std_pv: 1.63
- real_pv: 27, real_percentile: 1.0
- num_better_than_real: 97, num_equal: 3, num_worse: 0
- all_permuted_pvs: [17, 17, 19, 19, 24, 24, 24, 24, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27]

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

### seed109_real_beats_A: True
  Rule: B_signed_sum_online_total_pv < A_total_pv
  - signed_sum_pv: 27
  - A_pv: 29

### seed109_real_beats_permuted_median_100repeat: False
  Rule: real signed_sum PV < permuted median PV (100 repeats)
  - real_pv: 27
  - permuted_median: 26
  - n_perm_repeats: 100

### seed109_real_beats_80pct_permuted: False
  Rule: real PV beats >= 80% of 100 permuted repeats
  - num_worse: 0
  - num_equal: 3
  - n_total: 100

### seed109_high_ceiling_confirmed: True
  Rule: A_total_pv - Offline_signed_sum_upper_bound_total_pv >= 12
  - A_pv: 29
  - offline_pv: 6
  - improvement_room: 23

### seed109_online_offline_gap_large: True
  Rule: B_signed_sum_online_total_pv - Offline_signed_sum_upper_bound_total_pv >= 10
  - signed_sum_pv: 27
  - offline_pv: 6
  - gap: 21

### seed109_mapping_advantage_absent: True
  Rule: real_total_pv >= permuted_median_total_pv
  - real_pv: 27
  - permuted_median: 26

### seed109_online_key_coverage_insufficient: False
  Rule: offline_only_pos + offline_only_neg > overlap_top10_pos + overlap_top10_neg
  - offline_only_pos: 6
  - offline_only_neg: 4
  - overlap_top10_pos: 4
  - overlap_top10_neg: 6

### seed109_evidence_arrives_too_late: False
  Rule: first useful key appears after episode 3 or episodes remaining <= 1
  - first_useful_neg_ep: 0
  - first_useful_pos_ep: 0
  - episodes_remaining: 4

### seed109_weight_signal_wrong_or_too_weak: True
  Rule: wrong_sign + too_weak + selection_order > missing_key + below_support
  - wrong_sign: 0
  - too_weak: 0
  - selection_order: 4
  - missing_key: 0
  - below_support: 0

### no_oracle_leakage_confirmed: True
  Rule: all online variants use no offline/oracle evidence
  - all_online_variants_clean: True

### shared_positions_used: True
  Rule: all variants use the same precomputed episode positions
  - position_hash_by_episode: {0: '92ca73f4cec0', 1: '60cd592ff501', 2: '1592523d1bc0', 3: '102076764c05', 4: '506b8f987598'}

### permuted_control_mode_is_online_stepwise: True
  Rule: C variants learn online and permute at decision time
  - permuted_control_mode: online_stepwise

### final_memory_permutation_used: False
  Rule: no variant uses final-memory post-hoc permutation

### hard_exclusion_used: False
  Rule: hard_exclusion is always false for all variants

### deceptive_family_preservation_valid: True
  Rule: all deceptive objects preserve expected family identity
  - total_deceptive: 13
  - preserved: 13

### implementation_status: partial
  Rule: pass if sanity ok and signed_sum beats A and permuted_median; partial/fail otherwise
  - sanity_ok: True
  - ss_beats_A: True
  - ss_beats_perm_median: False

### failure_reason: real_does_not_beat_permuted_median
  Rule: derived from seed 109 high-ceiling gap diagnostic results

## 10. Part 7: High-Ceiling Online/Offline Gap Diagnostic

Diagnose why seed109 has high offline improvement room (23)
but weak online signed_sum improvement (PV=27 vs A=29, reduction of only 2).

### 10.1 Permutation Distribution (100 repeats)

#### Signed Sum

- real_total_pv: 27
- permuted: min=18, max=27, mean=25.65, median=26, std=1.53
- num_better_than_real: 97, num_equal: 3, num_worse: 0
- real_percentile: 1.0
- real_beats_permuted_median: False
- real_beats_80pct: False
- all_permuted_pvs: [18, 18, 19, 19, 24, 24, 24, 24, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27]

#### Signed Max Abs

- real_total_pv: 27
- permuted: min=17, max=27, mean=25.63, median=26, std=1.63
- num_better_than_real: 97, num_equal: 3, num_worse: 0
- real_percentile: 1.0
- real_beats_permuted_median: False
- real_beats_80pct: False

### 10.2 Online vs Offline Key Coverage

| Metric | Value |
|--------|-------|
| online_supported_positive_keys | 4 |
| online_supported_negative_keys | 12 |
| offline_supported_positive_keys | 18 |
| offline_supported_negative_keys | 10 |
| overlap_top_10_positive_keys | 4 |
| overlap_top_10_negative_keys | 6 |
| offline_only_positive_keys | 6 |
| offline_only_negative_keys | 4 |
| online_only_positive_keys | 0 |
| online_only_negative_keys | 4 |
| missing_offline_keys_not_observed | 4 |
| missing_offline_keys_below_support | 0 |
- offline_only_positive_keys: [['need_fuel', 'burn_as_fuel', 'wood-like', 'treated_surface'], ['need_fuel', 'burn_as_fuel', 'wood-like', 'damp_texture'], ['need_fuel', 'burn_as_fuel', 'tool-like', 'brittle_surface'], ['need_stone', 'mine_with_pickaxe', 'stone-like', 'hollow_sound'], ['need_stone', 'mine_with_pickaxe', 'stone-like', 'brittle_surface'], ['need_fuel', 'burn_as_fuel', 'tool-like', 'treated_surface']]
- offline_only_negative_keys: [['need_fuel', 'burn_as_fuel', 'wood-like', 'hollow_sound'], ['need_stone', 'mine_by_hand', 'stone-like', 'treated_surface'], ['need_stone', 'mine_by_hand', 'stone-like', 'damp_texture'], ['need_fuel', 'burn_as_fuel', 'wood-like', 'brittle_surface']]

### 10.3 Evidence Acquisition Timing

| Ep | A PV | SS PV | PosKeys | NegKeys | Reduction vs A |
|----|------|-------|---------|---------|----------------|
| 0 | 5 | 5 | 4 | 12 | 0 |
| 1 | 6 | 6 | 4 | 12 | 0 |
| 2 | 4 | 5 | 4 | 12 | -1 |
| 3 | 6 | 5 | 4 | 12 | 1 |
| 4 | 8 | 6 | 4 | 12 | 2 |

- first_episode_where_useful_negative_key_appears: 0
- first_episode_where_useful_positive_key_appears: 0
- episodes_remaining_after_first_useful_key: 4
- evidence_arrives_too_late: False

### 10.4 Selection-Difference Audit (A vs B_signed_sum)

| Metric | Value |
|--------|-------|
| avoided_prior_violation_count | 8 |
| avoided_nonviolation_count | 5 |
| avoided_effective_probe_count | 7 |
| helpful_avoidance_count | 5 |
| harmful_avoidance_count | 2 |
| net_helpful_minus_harmful | 3 |
| total_audit_entries | 13 |

| OID | Action | Family | Dev Feats | A PV | A NV | Helped/Hurt |
|-----|--------|--------|------------|------|------|-------------|
| test_wooden_pickaxe_014 | use_as_tool | tool-like | [] | False | True | hurt |
| test_wood_log_003 | craft_plank | wood-like | ['treated_surface', 'damp_texture'] | False | True | hurt |
| test_apple_003 | eat | apple-like | ['brittle_surface', 'damp_texture'] | True | False | helped |
| test_apple_002 | craft_plank | wood-like | [] | True | False | helped |
| test_wood_log_013 | use_as_tool | tool-like | [] | True | False | helped |
| test_stone_block_014 | mine_by_hand | stone-like | ['brittle_surface', 'hollow_sound'] | True | False | helped |
| test_stone_block_005 | craft_plank | wood-like | [] | True | False | helped |
| test_stone_block_007 | mine_by_hand | stone-like | [] | True | False | helped |
| test_wood_log_014 | craft_plank | wood-like | [] | False | True | hurt |
| test_stone_block_009 | craft_plank | wood-like | [] | True | False | helped |
| test_stone_block_010 | mine_with_pickaxe | stone-like | ['brittle_surface', 'hollow_sound'] | True | False | helped |
| test_wood_log_007 | craft_plank | wood-like | ['treated_surface', 'damp_texture'] | False | True | hurt |
| test_wood_log_006 | craft_plank | wood-like | ['treated_surface', 'damp_texture'] | False | True | hurt |

### 10.5 Missed Offline-Ceiling Opportunities

| Metric | Value |
|--------|-------|
| missed_offline_pv_avoidance_count | 16 |
| missed_due_to_missing_key | 0 |
| missed_due_to_below_support | 0 |
| missed_due_to_wrong_sign | 0 |
| missed_due_to_too_weak | 0 |
| missed_due_to_selection_order | 4 |
| total_missed | 16 |

| OID | Action | Family | Dev Feats | Reason |
|-----|--------|--------|------------|--------|
| test_stone_block_006 | eat | apple-like | [] | other |
| test_apple_001 | craft_plank | wood-like | [] | other |
| test_apple_011 | eat | apple-like | ['brittle_surface', 'damp_texture'] | selection_order_not_changed |
| test_stone_block_008 | craft_plank | wood-like | [] | other |
| test_wooden_pickaxe_005 | eat | apple-like | [] | other |
| test_stone_block_012 | craft_plank | wood-like | [] | other |
| test_wooden_pickaxe_002 | eat | apple-like | [] | other |
| test_stone_block_004 | craft_plank | wood-like | [] | other |
| test_apple_005 | use_as_tool | tool-like | [] | other |
| test_stone_block_003 | use_as_tool | tool-like | [] | other |
| test_apple_006 | eat | apple-like | ['brittle_surface', 'damp_texture'] | selection_order_not_changed |
| test_wood_log_001 | craft_plank | wood-like | [] | other |
| test_stone_block_002 | eat | apple-like | [] | other |
| test_apple_007 | eat | apple-like | ['brittle_surface', 'damp_texture'] | selection_order_not_changed |
| test_stone_block_013 | mine_by_hand | stone-like | ['brittle_surface', 'hollow_sound'] | selection_order_not_changed |
| test_apple_004 | eat | apple-like | [] | other |

## 11. Summary

```
[block_done]
block_id=1J28
seed=109
n_permutation_repeats=100
A_total_pv=29
B_signed_sum_online_total_pv=27
B_signed_max_abs_online_total_pv=27
Offline_signed_sum_upper_bound_total_pv=6
signed_sum_permuted_median_pv=26
signed_sum_permuted_mean_pv=25.65
signed_sum_permuted_min_pv=18
signed_sum_permuted_max_pv=27
signed_sum_num_permuted_better_than_real=97
signed_sum_num_permuted_equal_to_real=3
signed_sum_num_permuted_worse_than_real=0
signed_sum_real_percentile_against_permuted=1.0
seed109_real_beats_A=True
seed109_real_beats_permuted_median_100repeat=False
seed109_real_beats_80pct_permuted=False
seed109_high_ceiling_confirmed=True
seed109_online_offline_gap_large=True
seed109_mapping_advantage_absent=True
seed109_online_key_coverage_insufficient=False
seed109_evidence_arrives_too_late=False
seed109_weight_signal_wrong_or_too_weak=True
no_oracle_leakage_confirmed=True
shared_positions_used=True
permuted_control_mode=online_stepwise
final_memory_permutation_used=False
hard_exclusion_used=False
deceptive_family_preservation_valid=True
implementation_status=partial
failure_reason=real_does_not_beat_permuted_median
elapsed=623.8s
```