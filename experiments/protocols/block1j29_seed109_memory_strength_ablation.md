# Block 1J29 -- Seed 109 Memory-Strength Ablation for Online Signed Memory

## 1. Objective

Diagnose whether seed109 fails because the online signed memory signal is
directionally correct but too weak to change selection order, by sweeping
the risk-penalty strength scale across [0.5, 1.0, 1.5, 2.0, 3.0].

## 2. Setup

| Parameter | Value |
|-----------|-------|
| Condition | C4_instance_subtype_cued_v1 |
| Seed | 109 |
| Budget | 1.5 |
| Episodes | 5 |
| Risk Alpha | 5.0 |
| Max Risk | 2.0 |
| Scales | [0.5, 1.0, 1.5, 2.0, 3.0] |
| Perm Repeats per Scale | 20 |
| Scale Applied After Clipping | True |

## 3. Baselines (not scaled)

- A_no_memory: total_pv=29, macro_bal=0.5678
- Family_only_online_control: total_pv=24, macro_bal=0.5700
- Offline_signed_sum_upper_bound: total_pv=6, macro_bal=0.5887

## 4. Part 1: Scale Comparison Table

| Scale | Variant | PV | Dec PV | Norm PV | Macro BAcc | Delta vs A | Eff Probes | Avoided Eff | Avoided PV | Avoided NV | Helpful | Harmful | Net | Hard Excl |
|-------|---------|-----|--------|---------|------------|------------|------------|-------------|------------|------------|---------|---------|-----|-----------|
| 0.5 | B_signed_sum_online_scale_0.5 | 27 | 6 | 21 | 0.5738 | 0.006 | 37 | 2 | 5 | 2 | 5 | 2 | 3 | False |
| 0.5 | B_signed_max_abs_online_scale_0.5 | 27 | 6 | 21 | 0.5738 | 0.006 | 37 | 2 | 5 | 2 | 5 | 2 | 3 | False |
| 1.0 | B_signed_sum_online_scale_1.0 | 27 | 6 | 21 | 0.5738 | 0.006 | 37 | 2 | 5 | 2 | 5 | 2 | 3 | False |
| 1.0 | B_signed_max_abs_online_scale_1.0 | 27 | 6 | 21 | 0.5738 | 0.006 | 37 | 2 | 5 | 2 | 5 | 2 | 3 | False |
| 1.5 | B_signed_sum_online_scale_1.5 | 27 | 6 | 21 | 0.5738 | 0.006 | 37 | 2 | 5 | 2 | 5 | 2 | 3 | False |
| 1.5 | B_signed_max_abs_online_scale_1.5 | 27 | 6 | 21 | 0.5738 | 0.006 | 37 | 2 | 5 | 2 | 5 | 2 | 3 | False |
| 2.0 | B_signed_sum_online_scale_2.0 | 27 | 6 | 21 | 0.5738 | 0.006 | 37 | 2 | 5 | 2 | 5 | 2 | 3 | False |
| 2.0 | B_signed_max_abs_online_scale_2.0 | 27 | 6 | 21 | 0.5738 | 0.006 | 37 | 2 | 5 | 2 | 5 | 2 | 3 | False |
| 3.0 | B_signed_sum_online_scale_3.0 | 27 | 6 | 21 | 0.5738 | 0.006 | 37 | 2 | 5 | 2 | 5 | 2 | 3 | False |
| 3.0 | B_signed_max_abs_online_scale_3.0 | 27 | 6 | 21 | 0.5738 | 0.006 | 37 | 2 | 5 | 2 | 5 | 2 | 3 | False |

## 5. Part 2: Permuted Control Comparison by Scale

### Scale 0.5

**signed_sum**: real_pv=27, median=26, mean=25.65, min=24, max=27, beats_median=False, beats_80pct=False, better=19, equal=1, worse=0

**signed_max_abs**: real_pv=27, median=26, mean=25.65, min=24, max=27, beats_median=False, beats_80pct=False, better=19, equal=1, worse=0

### Scale 1.0

**signed_sum**: real_pv=27, median=26, mean=25.65, min=24, max=27, beats_median=False, beats_80pct=False, better=19, equal=1, worse=0

**signed_max_abs**: real_pv=27, median=26, mean=25.65, min=24, max=27, beats_median=False, beats_80pct=False, better=19, equal=1, worse=0

### Scale 1.5

**signed_sum**: real_pv=27, median=26, mean=25.65, min=24, max=27, beats_median=False, beats_80pct=False, better=19, equal=1, worse=0

**signed_max_abs**: real_pv=27, median=26, mean=25.65, min=24, max=27, beats_median=False, beats_80pct=False, better=19, equal=1, worse=0

### Scale 2.0

**signed_sum**: real_pv=27, median=26, mean=25.65, min=24, max=27, beats_median=False, beats_80pct=False, better=19, equal=1, worse=0

**signed_max_abs**: real_pv=27, median=26, mean=25.65, min=24, max=27, beats_median=False, beats_80pct=False, better=19, equal=1, worse=0

### Scale 3.0

**signed_sum**: real_pv=27, median=26, mean=25.65, min=24, max=27, beats_median=False, beats_80pct=False, better=19, equal=1, worse=0

**signed_max_abs**: real_pv=27, median=26, mean=25.65, min=24, max=27, beats_median=False, beats_80pct=False, better=19, equal=1, worse=0


## 6. Part 3: Online-Offline Gap by Scale

Offline_signed_sum_upper_bound_total_pv = 6

| Scale | SS PV | SM PV | SS Gap | SM Gap | SS Gap Reduction vs 1.0 | SM Gap Reduction vs 1.0 |
|-------|-------|-------|--------|--------|--------------------------|--------------------------|
| 0.5 | 27 | 27 | 21 | 21 | 0 | 0 |
| 1.0 | 27 | 27 | 21 | 21 | 0 | 0 |
| 1.5 | 27 | 27 | 21 | 21 | 0 | 0 |
| 2.0 | 27 | 27 | 21 | 21 | 0 | 0 |
| 3.0 | 27 | 27 | 21 | 21 | 0 | 0 |

## 7. Part 4: Selection-Order Rescue Audit

PVs not avoided at scale=1.0 (total=17): [('test_apple_001', 'craft_plank'), ('test_apple_004', 'eat'), ('test_apple_005', 'use_as_tool'), ('test_apple_006', 'eat'), ('test_apple_007', 'eat'), ('test_apple_011', 'eat'), ('test_stone_block_002', 'eat'), ('test_stone_block_003', 'use_as_tool'), ('test_stone_block_004', 'craft_plank'), ('test_stone_block_006', 'eat'), ('test_stone_block_008', 'craft_plank'), ('test_stone_block_012', 'craft_plank'), ('test_stone_block_013', 'mine_by_hand'), ('test_wood_log_000', 'craft_plank'), ('test_wood_log_001', 'craft_plank'), ('test_wooden_pickaxe_002', 'eat'), ('test_wooden_pickaxe_005', 'eat')]

| Scale | Rescued | Still Missed |
|-------|---------|-------------|
| 0.5 | 0 | 17 |
| 1.0 | 0 | 17 |
| 1.5 | 0 | 17 |
| 2.0 | 0 | 17 |
| 3.0 | 0 | 17 |

## 8. Part 5: Over-Conservatism Audit

| Scale | Variant | Avoided NV | Harmful | Macro Delta | Over-Conservative |
|-------|---------|------------|---------|-------------|-------------------|
| 0.5 | B_signed_sum_online_scale_0.5 | 2 | 2 | 0.006 | False |
| 0.5 | B_signed_max_abs_online_scale_0.5 | 2 | 2 | 0.006 | False |
| 1.0 | B_signed_sum_online_scale_1.0 | 2 | 2 | 0.006 | False |
| 1.0 | B_signed_max_abs_online_scale_1.0 | 2 | 2 | 0.006 | False |
| 1.5 | B_signed_sum_online_scale_1.5 | 2 | 2 | 0.006 | False |
| 1.5 | B_signed_max_abs_online_scale_1.5 | 2 | 2 | 0.006 | False |
| 2.0 | B_signed_sum_online_scale_2.0 | 2 | 2 | 0.006 | False |
| 2.0 | B_signed_max_abs_online_scale_2.0 | 2 | 2 | 0.006 | False |
| 3.0 | B_signed_sum_online_scale_3.0 | 2 | 2 | 0.006 | False |
| 3.0 | B_signed_max_abs_online_scale_3.0 | 2 | 2 | 0.006 | False |

## 9. Part 6: Boolean Flags

### all_scales_completed: True
  Rule: All 5 scales completed successfully
  - n_scales: 5
  - scales: [0.5, 1.0, 1.5, 2.0, 3.0]

### no_oracle_leakage_confirmed: True
  Rule: all online variants use no offline/oracle evidence
  - all_online_variants_clean: True

### shared_positions_used: True
  Rule: all variants use the same precomputed episode positions

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

### any_scale_beats_A: True
  Rule: any signed_sum or signed_max_abs scale has PV < A_total_pv
  - A_total_pv: 29

### any_scale_beats_permuted_median: False
  Rule: any scale has real PV < matching permuted median PV

### any_scale_beats_family_only: False
  Rule: any scale has real PV < Family_only_online_control_total_pv
  - family_only_pv: 24

### any_scale_reduces_online_offline_gap: False
  Rule: any scale has gap_to_offline smaller than scale=1.0
  - scale1_gap: 21

### any_scale_rescues_selection_order_cases: False
  Rule: rescued_by_scale_count > 0 for any scale

### any_scale_macro_tradeoff_acceptable: True
  Rule: any scale has macro_delta_vs_A >= -0.01 and no hard_exclusion

### best_scale_identified: False
  Rule: at least one scale improves PV vs scale=1.0 without macro_delta_vs_A < -0.02
  - best_ss_scale: 0.5
  - best_ss_pv: 27
  - best_sm_scale: 0.5
  - best_sm_pv: 27

### best_scale_over_conservative: False
  Rule: best PV scale has macro_delta_vs_A < -0.02 or harmful > helpful
  - best_ss_macro_delta: 0.006

### implementation_status: pass
  Rule: pass if any scale beats A with acceptable macro tradeoff; partial otherwise
  - any_beats_A: True
  - any_macro_ok: True
  - final_memory_permutation_used: False

### failure_reason: none
  Rule: derived from memory-strength ablation results

## 10. Summary

```
[block_done]
block_id=1J29
seed=109
scales=[0.5, 1.0, 1.5, 2.0, 3.0]
n_permutation_repeats=20
A_total_pv=29
Family_only_online_control_total_pv=24
Offline_signed_sum_upper_bound_total_pv=6
scale1_signed_sum_total_pv=27
scale1_signed_max_abs_total_pv=27
best_signed_sum_scale=0.5
best_signed_sum_total_pv=27
best_signed_sum_macro_delta_vs_A=0.006
best_signed_sum_permuted_median_pv=26
best_signed_sum_beats_permuted_median=False
best_signed_max_abs_scale=0.5
best_signed_max_abs_total_pv=27
best_signed_max_abs_macro_delta_vs_A=0.006
best_signed_max_abs_permuted_median_pv=26
best_signed_max_abs_beats_permuted_median=False
any_scale_beats_A=True
any_scale_beats_permuted_median=False
any_scale_beats_family_only=False
any_scale_reduces_online_offline_gap=False
any_scale_rescues_selection_order_cases=False
any_scale_macro_tradeoff_acceptable=True
best_scale_identified=False
best_scale_over_conservative=False
no_oracle_leakage_confirmed=True
shared_positions_used=True
permuted_control_mode=online_stepwise
final_memory_permutation_used=False
hard_exclusion_used=False
deceptive_family_preservation_valid=True
implementation_status=pass
failure_reason=none
elapsed=373.2s
```