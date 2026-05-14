# Block 1J21 -- Evidence Bottleneck and Offline Upper-Bound Diagnostic

## 1. Objective

Diagnose whether the weak 1J20 B_family_dev result is caused by insufficient online evidence collection or by family-conditioned deviation keys being uninformative even with enough evidence.

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

## 4. Part 1: Evidence Funnel Aggregates

| Metric | Value |
|--------|-------|
| num_keys_total | 96 |
| num_keys_online_above_support_threshold | 12 |
| num_keys_offline_above_support_threshold | 8 |
| num_keys_online_zero_evidence | 80 |
| num_keys_offline_zero_evidence | 86 |

### Top Keys by Offline Violations

| Goal | Action | Family | Dev Feature | Off Eligible | Off PV | Off NPV | Online Selected | Online PV | Online Above | Offline Above |
|------|--------|--------|-------------|-------------|--------|---------|-----------------|-----------|--------------|---------------|
| need_food | eat | apple-like | brittle_surface | 5 | 5 | 0 | 11 | 6 | True | True |
| need_food | eat | apple-like | damp_texture | 5 | 5 | 0 | 11 | 6 | True | True |
| need_fuel | burn_as_fuel | wood-like | damp_texture | 5 | 5 | 0 | 0 | 0 | False | True |
| need_fuel | burn_as_fuel | wood-like | treated_surface | 5 | 5 | 0 | 0 | 0 | False | True |
| need_stone | mine_by_hand | stone-like | brittle_surface | 3 | 3 | 0 | 9 | 8 | True | True |
| need_stone | mine_by_hand | stone-like | hollow_sound | 3 | 3 | 0 | 9 | 8 | True | True |
| need_stone | mine_with_pickaxe | stone-like | brittle_surface | 3 | 3 | 0 | 0 | 0 | False | True |
| need_stone | mine_with_pickaxe | stone-like | hollow_sound | 3 | 3 | 0 | 0 | 0 | False | True |
| need_food | eat | apple-like | hollow_sound | 0 | 0 | 0 | 11 | 6 | True | False |
| need_food | eat | apple-like | treated_surface | 0 | 0 | 0 | 11 | 6 | True | False |
| need_food | eat | stone-like | brittle_surface | 0 | 0 | 0 | 0 | 0 | False | False |
| need_food | eat | stone-like | damp_texture | 0 | 0 | 0 | 0 | 0 | False | False |
| need_food | eat | stone-like | hollow_sound | 0 | 0 | 0 | 0 | 0 | False | False |
| need_food | eat | stone-like | treated_surface | 0 | 0 | 0 | 0 | 0 | False | False |
| need_food | eat | tool-like | brittle_surface | 0 | 0 | 0 | 0 | 0 | False | False |
| need_food | eat | tool-like | damp_texture | 0 | 0 | 0 | 0 | 0 | False | False |
| need_food | eat | tool-like | hollow_sound | 0 | 0 | 0 | 0 | 0 | False | False |
| need_food | eat | tool-like | treated_surface | 0 | 0 | 0 | 0 | 0 | False | False |
| need_food | eat | wood-like | brittle_surface | 0 | 0 | 0 | 0 | 0 | False | False |
| need_food | eat | wood-like | damp_texture | 0 | 0 | 0 | 0 | 0 | False | False |

## 5. Part 2: Bottleneck Reason Summary

| Goal | Action | Family | Dev Keys | No Obj | Low Prior | Not Selected | No Fail | Min Thresh | Off Eligible | Off PV |
|------|--------|--------|----------|--------|-----------|--------------|---------|------------|-------------|--------|
| need_food | eat | apple-like | 4 | 2 | 0 | 0 | 0 | 0 | 10 | 10 |
| need_food | eat | stone-like | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| need_food | eat | tool-like | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| need_food | eat | wood-like | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| need_fuel | burn_as_fuel | apple-like | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| need_fuel | burn_as_fuel | stone-like | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| need_fuel | burn_as_fuel | tool-like | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| need_fuel | burn_as_fuel | wood-like | 4 | 2 | 0 | 2 | 0 | 0 | 10 | 10 |
| need_planks | craft_plank | apple-like | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| need_planks | craft_plank | stone-like | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| need_planks | craft_plank | tool-like | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| need_planks | craft_plank | wood-like | 4 | 2 | 0 | 0 | 0 | 0 | 10 | 0 |
| need_stone | mine_by_hand | apple-like | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| need_stone | mine_by_hand | stone-like | 4 | 2 | 0 | 0 | 0 | 0 | 6 | 6 |
| need_stone | mine_by_hand | tool-like | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| need_stone | mine_by_hand | wood-like | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| need_stone | mine_with_pickaxe | apple-like | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| need_stone | mine_with_pickaxe | stone-like | 4 | 2 | 0 | 2 | 0 | 0 | 6 | 6 |
| need_stone | mine_with_pickaxe | tool-like | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| need_stone | mine_with_pickaxe | wood-like | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| need_tool | use_as_tool | apple-like | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| need_tool | use_as_tool | stone-like | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| need_tool | use_as_tool | tool-like | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| need_tool | use_as_tool | wood-like | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |

## 6. Part 3: Main Comparison Table

| Variant | Total PV | Deceptive PV | Normal PV | Mean BAcc | Delta vs A | Eff Probes | Avoided Eff | Nonzero W | Keys Above | Oracle |
|---------|----------|-------------|-----------|-----------|------------|------------|-------------|-----------|------------|--------|
| A_no_memory | 22 | 7 | 15 | 0.5937 | 0.0 | 35 | 0 | 0 | 0 | False |
| B_family_dev_online | 21 | 5 | 16 | 0.5899 | -0.0038 | 35 | 6 | 2 | 12 | False |
| B_family_dev_offline_upper | 22 | 0 | 22 | 0.5846 | -0.0091 | 35 | 33 | 16 | 24 | True |
| C_family_dev_offline_permuted | 16 | 1 | 15 | 0.588 | -0.0057 | 35 | 32 | 16 | 24 | True |
| C_family_only_offline | 11 | 4 | 7 | 0.5707 | -0.023 | 18 | 35 | 14 | 24 | True |

## 7. Part 4: Offline-vs-Online Risk Comparison

| Metric | Online | Offline |
|--------|--------|---------|
| risk_weights_nonzero | 2 | 16 |
| total_abs_risk_mass | 1.722549 | 12.28711 |
| overlap_top_10_keys | 2 | - |
| offline_only_supported_keys | - | 14 |
| online_only_supported_keys | 0 | - |

### Top 10 Online Risk Keys

| Key | Risk Weight |
|-----|-------------|
| ['need_food', 'eat', 'apple-like', 'brittle_surface'] | 0.861275 |
| ['need_food', 'eat', 'apple-like', 'damp_texture'] | 0.861275 |

### Top 10 Offline Risk Keys

| Key | Risk Weight |
|-----|-------------|
| ['need_stone', 'mine_with_pickaxe', 'stone-like', 'brittle_surface'] | 1.5 |
| ['need_stone', 'mine_with_pickaxe', 'stone-like', 'hollow_sound'] | 1.5 |
| ['need_fuel', 'burn_as_fuel', 'wood-like', 'treated_surface'] | 1.210771 |
| ['need_fuel', 'burn_as_fuel', 'wood-like', 'damp_texture'] | 1.210771 |
| ['need_food', 'eat', 'apple-like', 'brittle_surface'] | 0.964286 |
| ['need_food', 'eat', 'apple-like', 'damp_texture'] | 0.964286 |
| ['need_stone', 'mine_with_pickaxe', 'stone-like', 'treated_surface'] | 0.772215 |
| ['need_stone', 'mine_with_pickaxe', 'stone-like', 'damp_texture'] | 0.772215 |
| ['need_stone', 'mine_by_hand', 'stone-like', 'brittle_surface'] | 0.635473 |
| ['need_stone', 'mine_by_hand', 'stone-like', 'hollow_sound'] | 0.635473 |

## 8. Part 5: Boolean Flag Details

### online_evidence_bottleneck_detected: False
  Rule: num_keys_online_above_support_threshold < 3
  - num_keys_online_above_support_threshold: 12
  - num_keys_total: 96
  - num_keys_online_zero_evidence: 80

### offline_support_sufficient: True
  Rule: num_keys_offline_above_support_threshold >= 3
  - num_keys_offline_above_support_threshold: 8
  - num_keys_total: 96

### offline_family_dev_beats_A: False
  Rule: B_family_dev_offline_upper total_prior_violations < A_no_memory total_prior_violations
  - A_total_pv: 22
  - offline_total_pv: 22

### offline_family_dev_beats_permuted: False
  Rule: B_family_dev_offline_upper total_prior_violations < C_family_dev_offline_permuted total_prior_violations
  - offline_total_pv: 22
  - permuted_total_pv: 16

### offline_family_dev_beats_family_only: False
  Rule: B_family_dev_offline_upper total_prior_violations < C_family_only_offline total_prior_violations
  - offline_total_pv: 22
  - family_only_total_pv: 11

### family_only_still_dominates_offline: True
  Rule: C_family_only_offline total_prior_violations <= B_family_dev_offline_upper total_prior_violations
  - family_only_total_pv: 11
  - offline_total_pv: 22

### hard_exclusion_used: False
  Rule: hard_exclusion is always false for all variants

### deceptive_family_preservation_valid: True
  Rule: all deceptive objects preserve expected family identity
  - total_deceptive_objects: 13
  - preserved_family_count: 13
  - violated_family_count: 0

### implementation_status: partial
  Rule: pass if offline support sufficient AND beats A AND beats permuted; partial otherwise
  - num_keys_offline_above_support_threshold: 8
  - offline_beats_A: False
  - offline_beats_permuted: False

### failure_reason: offline_support_exists_but_no_benefit
  Rule: derived from combination of support, bottleneck, and comparison metrics

## 9. Summary

```
[block_done]
block_id=1J21
A_total_pv=22
B_family_dev_online_total_pv=21
B_family_dev_offline_upper_total_pv=22
C_family_dev_offline_permuted_total_pv=16
C_family_only_offline_total_pv=11
B_family_dev_online_macro_delta_vs_A=-0.0038
B_family_dev_offline_macro_delta_vs_A=-0.0091
num_keys_online_above_support_threshold=12
num_keys_offline_above_support_threshold=8
online_evidence_bottleneck_detected=False
offline_support_sufficient=True
offline_family_dev_beats_A=False
offline_family_dev_beats_permuted=False
offline_family_dev_beats_family_only=False
family_only_still_dominates_offline=True
deceptive_family_preservation_valid=True
hard_exclusion_used=False
implementation_status=partial
failure_reason=offline_support_exists_but_no_benefit
elapsed=51.4s
```