# Block 1J33 ¡ª Dev+Family+Action Fallback Fix

- **Seed**: 107
- **Condition**: C4_instance_subtype_cued_v1
- **Episodes per variant**: 5
- **Elapsed**: 10.2s

## Comparison Table

| Variant | Total PV | Deceptive PV | Normal PV | Macro Bal | Delta vs A | Eff Probes | Avoided Eff | Avoided PV | Avoided NV | Helpful | Harmful |
|---------|----------|-------------|-----------|-----------|------------|------------|-------------|------------|------------|---------|---------|
| A_no_memory | 15 | 4 | 11 | 0.6098 | +0.0000 | 25 | 0 | 0 | 0 | 0 | 0 |
| B_signed_sum_current | 17 | 6 | 11 | 0.6066 | -0.0032 | 25 | 0 | 1 | 2 | 1 | 2 |
| Family_only | 13 | 6 | 7 | 0.5982 | -0.0116 | 12 | 13 | 5 | 8 | 5 | 8 |
| B_dev_plus_family_action_fallback | 13 | 7 | 6 | 0.5980 | -0.0118 | 14 | 11 | 4 | 8 | 4 | 8 |
| Offline_current | 5 | 5 | 0 | 0.6175 | +0.0077 | 24 | 1 | 12 | 8 | 12 | 8 |

## Raw-Zero Case Diagnostics

- Raw-zero cases (B_signed_sum PV with no dev_feat, penalty=0): **11**
- Raw-zero cases rescued by fallback: **4**
- Raw-zero cases still missed: **7**
- Family_only avoided cases matched: **4**
- Rank changed count (symmetric diff of probe sets): **28**
- Selected probe set changed: **19**

## Sanity Checks

- `hidden_feature_leakage_detected`: **False**
- `no_oracle_leakage_confirmed`: **True**
- `implementation_status`: pass
- `failure_reason`: none

## Raw-Zero Case Details

| Object | Action | Family | Dev Feats | Risk Penalty | Avoided by FB | Avoided by FO |
|--------|--------|--------|-----------|-------------|---------------|---------------|
| test_wooden_pickaxe_001 | craft_plank | wood-like | [] | 0.000000 | False | False |
| test_stone_block_002 | mine_by_hand | stone-like | [] | 0.000000 | False | False |
| test_stone_block_005 | use_as_tool | tool-like | [] | 0.000000 | False | False |
| test_wooden_pickaxe_004 | use_as_tool | tool-like | [] | 0.000000 | False | False |
| test_stone_block_001 | craft_plank | wood-like | [] | 0.000000 | False | False |
| test_apple_014 | craft_plank | wood-like | [] | 0.000000 | True | True |
| test_stone_block_011 | eat | apple-like | [] | 0.000000 | True | True |
| test_apple_014 | craft_plank | wood-like | [] | 0.000000 | True | True |
| test_stone_block_005 | use_as_tool | tool-like | [] | 0.000000 | False | False |
| test_stone_block_010 | mine_by_hand | stone-like | [] | 0.000000 | True | True |
| test_wooden_pickaxe_004 | use_as_tool | tool-like | [] | 0.000000 | False | False |

## Rescued Cases

- **test_apple_014** / craft_plank (wood-like): was raw-zero, now avoided by fallback
- **test_stone_block_011** / eat (apple-like): was raw-zero, now avoided by fallback
- **test_apple_014** / craft_plank (wood-like): was raw-zero, now avoided by fallback
- **test_stone_block_010** / mine_by_hand (stone-like): was raw-zero, now avoided by fallback


```
[block_done]
block_id=1J33
seed_mode=seed109
A_total_pv=15
B_signed_sum_total_pv=17
Family_only_total_pv=13
B_dev_plus_family_action_fallback_total_pv=13
Offline_total_pv=5
raw_zero_missed_cases_remaining=7
raw_zero_cases_rescued=4
rank_changed_count=28
hidden_feature_leakage_detected=False
no_oracle_leakage_confirmed=True
implementation_status=pass
failure_reason=none
elapsed=10.2s
```