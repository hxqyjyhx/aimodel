# Block 1J22 -- Deviation Feature Informativeness and Risk-Weight Audit

## 1. Objective

Audit why 1J21 offline family-conditioned deviation memory failed even with oracle evidence. Check implementation correctness, feature informativeness, and counterproductive weighting.

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
| A_no_memory | 0.5937 |
| B_family_dev_offline_upper | 0.5846 |

## 4. Part 1: Implementation Sanity Audit

### 1a. Risk-Weight Sign Check

Risk sign errors: 0 -> risk_sign_error_detected=False

| Goal | Action | Family | Dev Feat | V_With | NV_With | V_Without | NV_Without | P(F|V) | P(F|NV) | Raw LR | Clipped | Shrink | Final W | Sign OK | Above Sup |
|------|--------|--------|-----------|--------|---------|-----------|------------|-------|---------|--------|---------|--------|---------|---------|-----------|
| need_stone | mine_with_pickaxe | stone-like | hollow_sound | 3 | 0 | 0 | 12 | 0.8000 | 0.0714 | 2.4159 | 2.0000 | 0.7500 | 1.5000 | True | True |
| need_stone | mine_with_pickaxe | stone-like | brittle_surface | 3 | 0 | 0 | 12 | 0.8000 | 0.0714 | 2.4159 | 2.0000 | 0.7500 | 1.5000 | True | True |
| need_fuel | burn_as_fuel | wood-like | damp_texture | 5 | 0 | 6 | 8 | 0.4615 | 0.1000 | 1.5294 | 1.5294 | 0.7917 | 1.2108 | True | True |
| need_fuel | burn_as_fuel | wood-like | treated_surface | 5 | 0 | 6 | 8 | 0.4615 | 0.1000 | 1.5294 | 1.5294 | 0.7917 | 1.2108 | True | True |
| need_fuel | burn_as_fuel | tool-like | hollow_sound | 0 | 0 | 0 | 8 | 0.5000 | 0.1000 | 1.6094 | 1.6094 | 0.6154 | 0.9904 | True | False |
| need_fuel | burn_as_fuel | tool-like | damp_texture | 0 | 0 | 0 | 8 | 0.5000 | 0.1000 | 1.6094 | 1.6094 | 0.6154 | 0.9904 | True | False |
| need_fuel | burn_as_fuel | tool-like | brittle_surface | 0 | 0 | 0 | 8 | 0.5000 | 0.1000 | 1.6094 | 1.6094 | 0.6154 | 0.9904 | True | False |
| need_fuel | burn_as_fuel | tool-like | treated_surface | 0 | 0 | 0 | 8 | 0.5000 | 0.1000 | 1.6094 | 1.6094 | 0.6154 | 0.9904 | True | False |
| need_food | eat | apple-like | damp_texture | 5 | 0 | 7 | 6 | 0.4286 | 0.1250 | 1.2321 | 1.2321 | 0.7826 | 0.9643 | True | True |
| need_food | eat | apple-like | brittle_surface | 5 | 0 | 7 | 6 | 0.4286 | 0.1250 | 1.2321 | 1.2321 | 0.7826 | 0.9643 | True | True |
| need_stone | mine_with_pickaxe | stone-like | damp_texture | 0 | 0 | 3 | 12 | 0.2000 | 0.0714 | 1.0296 | 1.0296 | 0.7500 | 0.7722 | True | True |
| need_stone | mine_with_pickaxe | stone-like | treated_surface | 0 | 0 | 3 | 12 | 0.2000 | 0.0714 | 1.0296 | 1.0296 | 0.7500 | 0.7722 | True | True |
| need_stone | mine_by_hand | stone-like | hollow_sound | 3 | 0 | 7 | 5 | 0.3333 | 0.1429 | 0.8473 | 0.8473 | 0.7500 | 0.6355 | True | True |
| need_stone | mine_by_hand | stone-like | brittle_surface | 3 | 0 | 7 | 5 | 0.3333 | 0.1429 | 0.8473 | 0.8473 | 0.7500 | 0.6355 | True | True |
| need_tool | use_as_tool | tool-like | hollow_sound | 0 | 0 | 2 | 6 | 0.2500 | 0.1250 | 0.6931 | 0.6931 | 0.6154 | 0.4266 | True | True |
| need_tool | use_as_tool | tool-like | damp_texture | 0 | 0 | 2 | 6 | 0.2500 | 0.1250 | 0.6931 | 0.6931 | 0.6154 | 0.4266 | True | True |
| need_tool | use_as_tool | tool-like | brittle_surface | 0 | 0 | 2 | 6 | 0.2500 | 0.1250 | 0.6931 | 0.6931 | 0.6154 | 0.4266 | True | True |
| need_tool | use_as_tool | tool-like | treated_surface | 0 | 0 | 2 | 6 | 0.2500 | 0.1250 | 0.6931 | 0.6931 | 0.6154 | 0.4266 | True | True |
| need_planks | craft_plank | wood-like | hollow_sound | 0 | 0 | 8 | 11 | 0.1000 | 0.0769 | 0.2624 | 0.2624 | 0.7917 | 0.2077 | True | True |
| need_planks | craft_plank | wood-like | brittle_surface | 0 | 0 | 8 | 11 | 0.1000 | 0.0769 | 0.2624 | 0.2624 | 0.7917 | 0.2077 | True | True |
| need_planks | craft_plank | wood-like | damp_texture | 0 | 5 | 8 | 6 | 0.1000 | 0.4615 | -1.5294 | 0.0000 | 0.7917 | 0.0000 | True | True |
| need_planks | craft_plank | wood-like | treated_surface | 0 | 5 | 8 | 6 | 0.1000 | 0.4615 | -1.5294 | 0.0000 | 0.7917 | 0.0000 | True | True |
| need_fuel | burn_as_fuel | wood-like | hollow_sound | 0 | 0 | 11 | 8 | 0.0769 | 0.1000 | -0.2624 | 0.0000 | 0.7917 | 0.0000 | True | True |
| need_fuel | burn_as_fuel | wood-like | brittle_surface | 0 | 0 | 11 | 8 | 0.0769 | 0.1000 | -0.2624 | 0.0000 | 0.7917 | 0.0000 | True | True |
| need_food | eat | apple-like | hollow_sound | 0 | 0 | 12 | 6 | 0.0714 | 0.1250 | -0.5596 | 0.0000 | 0.7826 | 0.0000 | True | True |
| need_food | eat | apple-like | treated_surface | 0 | 0 | 12 | 6 | 0.0714 | 0.1250 | -0.5596 | 0.0000 | 0.7826 | 0.0000 | True | True |
| need_stone | mine_by_hand | stone-like | damp_texture | 0 | 0 | 10 | 5 | 0.0833 | 0.1429 | -0.5390 | 0.0000 | 0.7500 | 0.0000 | True | True |
| need_stone | mine_by_hand | stone-like | treated_surface | 0 | 0 | 10 | 5 | 0.0833 | 0.1429 | -0.5390 | 0.0000 | 0.7500 | 0.0000 | True | True |

### 1b. Penalty-Direction Check

Penalty direction errors: 0 -> penalty_direction_error_detected=False

### 1c. Support-Definition Check

- Online support event unit: episode_repeated_probe_events
- Offline support event unit: unique_object_action_candidates
- Online total eligible events: 408
- Offline total eligible events: 42
- Online keys above support: 24
- Offline keys above support: 8
- support_count_inconsistency_detected=True

## 5. Part 2: Deviation Feature Informativeness Table

deviation_features_informative=True
Informative keys (|delta|>0.1 & above support): 24

### Top 20 Keys by |Risk Rate Delta|

| Goal | Action | Family | Dev Feat | N_With | N_Without | V_With | NV_With | V_Without | NV_Without | Rate_With | Rate_Without | Delta | LogOdds | MI | Above Sup | Final W |
|------|--------|--------|-----------|--------|-----------|--------|---------|-----------|------------|-----------|--------------|-------|---------|----|-----------|---------|
| need_stone | mine_with_pickaxe | stone-like | hollow_sound | 3 | 12 | 3 | 0 | 0 | 12 | 1.0000 | 0.0000 | 1.0000 | 4.9698 | 0.3044 | True | 1.5000 |
| need_stone | mine_with_pickaxe | stone-like | brittle_surface | 3 | 12 | 3 | 0 | 0 | 12 | 1.0000 | 0.0000 | 1.0000 | 4.9698 | 0.3044 | True | 1.5000 |
| need_food | eat | apple-like | hollow_sound | 0 | 18 | 0 | 0 | 12 | 6 | 0.0000 | 0.6667 | -0.6667 | -0.6931 | 0.0029 | True | 0.0000 |
| need_food | eat | apple-like | treated_surface | 0 | 18 | 0 | 0 | 12 | 6 | 0.0000 | 0.6667 | -0.6667 | -0.6931 | 0.0029 | True | 0.0000 |
| need_stone | mine_by_hand | stone-like | damp_texture | 0 | 15 | 0 | 0 | 10 | 5 | 0.0000 | 0.6667 | -0.6667 | -0.6931 | 0.0034 | True | 0.0000 |
| need_stone | mine_by_hand | stone-like | treated_surface | 0 | 15 | 0 | 0 | 10 | 5 | 0.0000 | 0.6667 | -0.6667 | -0.6931 | 0.0034 | True | 0.0000 |
| need_fuel | burn_as_fuel | wood-like | hollow_sound | 0 | 19 | 0 | 0 | 11 | 8 | 0.0000 | 0.5789 | -0.5789 | -0.3185 | 0.0006 | True | 0.0000 |
| need_fuel | burn_as_fuel | wood-like | brittle_surface | 0 | 19 | 0 | 0 | 11 | 8 | 0.0000 | 0.5789 | -0.5789 | -0.3185 | 0.0006 | True | 0.0000 |
| need_planks | craft_plank | wood-like | damp_texture | 5 | 14 | 0 | 5 | 8 | 6 | 0.0000 | 0.5714 | -0.5714 | -2.5903 | 0.1087 | True | 0.0000 |
| need_planks | craft_plank | wood-like | treated_surface | 5 | 14 | 0 | 5 | 8 | 6 | 0.0000 | 0.5714 | -0.5714 | -2.5903 | 0.1087 | True | 0.0000 |
| need_fuel | burn_as_fuel | wood-like | damp_texture | 5 | 14 | 5 | 0 | 6 | 8 | 1.0000 | 0.4286 | 0.5714 | 2.5903 | 0.1087 | True | 1.2108 |
| need_fuel | burn_as_fuel | wood-like | treated_surface | 5 | 14 | 5 | 0 | 6 | 8 | 1.0000 | 0.4286 | 0.5714 | 2.5903 | 0.1087 | True | 1.2108 |
| need_food | eat | apple-like | damp_texture | 5 | 13 | 5 | 0 | 7 | 6 | 1.0000 | 0.5385 | 0.4615 | 2.1484 | 0.0727 | True | 0.9643 |
| need_food | eat | apple-like | brittle_surface | 5 | 13 | 5 | 0 | 7 | 6 | 1.0000 | 0.5385 | 0.4615 | 2.1484 | 0.0727 | True | 0.9643 |
| need_planks | craft_plank | wood-like | hollow_sound | 0 | 19 | 0 | 0 | 8 | 11 | 0.0000 | 0.4211 | -0.4211 | 0.3185 | 0.0006 | True | 0.2077 |
| need_planks | craft_plank | wood-like | brittle_surface | 0 | 19 | 0 | 0 | 8 | 11 | 0.0000 | 0.4211 | -0.4211 | 0.3185 | 0.0006 | True | 0.2077 |
| need_stone | mine_by_hand | stone-like | hollow_sound | 3 | 12 | 3 | 0 | 7 | 5 | 1.0000 | 0.5833 | 0.4167 | 1.4553 | 0.0320 | True | 0.6355 |
| need_stone | mine_by_hand | stone-like | brittle_surface | 3 | 12 | 3 | 0 | 7 | 5 | 1.0000 | 0.5833 | 0.4167 | 1.4553 | 0.0320 | True | 0.6355 |
| need_tool | use_as_tool | tool-like | hollow_sound | 0 | 8 | 0 | 0 | 2 | 6 | 0.0000 | 0.2500 | -0.2500 | 1.0986 | 0.0140 | True | 0.4266 |
| need_tool | use_as_tool | tool-like | damp_texture | 0 | 8 | 0 | 0 | 2 | 6 | 0.0000 | 0.2500 | -0.2500 | 1.0986 | 0.0140 | True | 0.4266 |

## 6. Part 3: Best Possible Deviation-Rule Diagnostic

| Rule | Total PV | Deceptive PV | Normal PV | Mean BAcc | Delta vs A | Eff Probes | Avoided Eff | Avoided PV |
|------|----------|-------------|-----------|-----------|------------|------------|-------------|------------|
| oracle_dev_feature | 19 | 0 | 19 | 0.5819 | -0.0118 | 35 | 34 | 12 |
| oracle_deceptive_object | 0 | 0 | 0 | 0.5877 | -0.006 | 27 | 8 | 15 |
| family_only | 8 | 0 | 8 | 0.5758 | -0.0179 | 27 | 35 | 12 |

oracle_dev_rule_beats_A=True
oracle_dev_rule_beats_family_only=False
oracle_deceptive_rule_beats_A=True

## 7. Part 4: Counterproductive-Weight Audit

Avoided probes: 15 (helpful=9, harmful=6)
New probes selected: 15
counterproductive_weighting_detected=True

### Avoided Candidates (top 15 by risk penalty)

| OID | Action | Family | Dev Feats | Risk Penalty | PV | NV | Utility | Helped | Hurt |
|-----|--------|--------|-----------|-------------|----|----|---------|--------|------|
| test_apple_004 | eat | apple-like | ['damp_texture', 'brittle_surface'] | 1.9286 | True | False | effective | True | False |
| test_apple_006 | eat | apple-like | ['damp_texture', 'brittle_surface'] | 1.9286 | True | False | effective | True | False |
| test_apple_010 | eat | apple-like | ['damp_texture', 'brittle_surface'] | 1.9286 | True | False | effective | True | False |
| test_apple_014 | eat | apple-like | ['damp_texture', 'brittle_surface'] | 1.9286 | True | False | effective | True | False |
| test_stone_block_014 | mine_by_hand | stone-like | ['hollow_sound', 'brittle_surface'] | 1.2709 | True | False | effective | True | False |
| test_apple_000 | eat | apple-like | [] | 0.0000 | False | True | effective | False | True |
| test_apple_005 | eat | apple-like | [] | 0.0000 | False | True | effective | False | True |
| test_apple_013 | craft_plank | wood-like | [] | 0.0000 | True | False | effective | True | False |
| test_stone_block_011 | mine_by_hand | stone-like | [] | 0.0000 | True | False | effective | True | False |
| test_stone_block_012 | mine_by_hand | stone-like | [] | 0.0000 | True | False | effective | True | False |
| test_stone_block_013 | mine_by_hand | stone-like | [] | 0.0000 | True | False | effective | True | False |
| test_wood_log_003 | craft_plank | wood-like | [] | 0.0000 | False | True | effective | False | True |
| test_wood_log_012 | craft_plank | wood-like | ['damp_texture', 'treated_surface'] | 0.0000 | False | True | effective | False | True |
| test_wood_log_013 | mine_by_hand | stone-like | [] | 0.0000 | False | True | effective | False | True |
| test_wooden_pickaxe_004 | use_as_tool | tool-like | [] | 0.0000 | False | True | effective | False | True |

## 8. Part 5: Boolean Flag Details

### risk_sign_error_detected: False
  Rule: true if any key has p_feat|v > p_feat|nv but risk<=0, or p_feat|v < p_feat|nv but risk>0
  - risk_sign_errors: 0
  - total_keys: 28

### penalty_direction_error_detected: False
  Rule: true if any positive risk penalty increases final selection priority
  - penalty_direction_errors: 0
  - total_checked: 35

### support_count_inconsistency_detected: True
  Rule: true if online/offline support counts differ under intended event unit
  - online_total: 408
  - offline_total: 42
  - ratio: 9.71

### deviation_features_informative: True
  Rule: true if at least one dev feature has |risk_rate_delta| > 0.1 with above-threshold support
  - informative_keys_count: 24
  - total_keys: 28

### oracle_dev_rule_beats_A: True
  Rule: oracle_dev_rule_total_pv < A_total_pv
  - oracle_dev_total_pv: 19
  - A_total_pv: 22

### oracle_dev_rule_beats_family_only: False
  Rule: oracle_dev_rule_total_pv < family_only_total_pv
  - oracle_dev_total_pv: 19
  - family_only_total_pv: 8

### oracle_deceptive_rule_beats_A: True
  Rule: oracle_deceptive_total_pv < A_total_pv
  - oracle_deceptive_total_pv: 0
  - A_total_pv: 22

### family_only_conservative_win_detected: True
  Rule: family_only_total_pv <= B_family_dev_offline_upper_total_pv
  - family_only_total_pv: 8
  - offline_total_pv: 22

### counterproductive_weighting_detected: True
  Rule: harmful_avoidances > helpful_avoidances OR B_offline doesn't reduce PV vs A
  - harmful_avoidances: 6
  - helpful_avoidances: 9
  - offline_pv: 22
  - A_pv: 22

### hard_exclusion_used: False
  Rule: hard_exclusion is always false for all variants

### deceptive_family_preservation_valid: True
  Rule: all deceptive objects preserve expected family identity
  - total_deceptive: 13
  - preserved: 13

### implementation_status: partial
  Rule: pass if no implementation bugs AND deviation features informative AND oracle beats A
  - risk_sign_error: False
  - penalty_direction_error: False
  - deviation_features_informative: True
  - oracle_dev_beats_A: True

### failure_reason: deviation_features_informative_but_family_only_dominates
  Rule: derived from implementation audit and informativeness results

## 9. Summary

```
[block_done]
block_id=1J22
risk_sign_error_detected=False
penalty_direction_error_detected=False
support_count_inconsistency_detected=True
deviation_features_informative=True
oracle_dev_rule_total_pv=19
oracle_deceptive_rule_total_pv=0
family_only_rule_total_pv=8
A_total_pv=22
B_family_dev_offline_upper_total_pv=22
counterproductive_weighting_detected=True
family_only_conservative_win_detected=True
hard_exclusion_used=False
deceptive_family_preservation_valid=True
implementation_status=partial
failure_reason=deviation_features_informative_but_family_only_dominates
elapsed=41.8s
```