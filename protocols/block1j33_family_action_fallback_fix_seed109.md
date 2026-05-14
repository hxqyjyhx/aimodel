# Block 1J33 ¡ª Dev+Family+Action Fallback Fix

- **Seed**: 109
- **Condition**: C4_instance_subtype_cued_v1
- **Episodes per variant**: 5
- **Elapsed**: 10.3s

## Comparison Table

| Variant | Total PV | Deceptive PV | Normal PV | Macro Bal | Delta vs A | Eff Probes | Avoided Eff | Avoided PV | Avoided NV | Helpful | Harmful |
|---------|----------|-------------|-----------|-----------|------------|------------|-------------|------------|------------|---------|---------|
| A_no_memory | 18 | 5 | 13 | 0.5645 | +0.0000 | 25 | 0 | 0 | 0 | 0 | 0 |
| B_signed_sum_current | 16 | 5 | 11 | 0.5678 | +0.0033 | 25 | 0 | 3 | 1 | 3 | 1 |
| Family_only | 13 | 6 | 7 | 0.5637 | -0.0008 | 20 | 5 | 10 | 3 | 10 | 3 |
| B_dev_plus_family_action_fallback | 11 | 5 | 6 | 0.5737 | +0.0092 | 25 | 0 | 10 | 4 | 10 | 4 |
| Offline_current | 5 | 5 | 0 | 0.5870 | +0.0225 | 25 | 0 | 14 | 7 | 14 | 7 |

## Raw-Zero Case Diagnostics

- Raw-zero cases (B_signed_sum PV with no dev_feat, penalty=0): **11**
- Raw-zero cases rescued by fallback: **10**
- Raw-zero cases still missed: **1**
- Family_only avoided cases matched: **7**
- Rank changed count (symmetric diff of probe sets): **34**
- Selected probe set changed: **21**

## Sanity Checks

- `hidden_feature_leakage_detected`: **False**
- `no_oracle_leakage_confirmed`: **True**
- `implementation_status`: pass
- `failure_reason`: none

## Raw-Zero Case Details

| Object | Action | Family | Dev Feats | Risk Penalty | Avoided by FB | Avoided by FO |
|--------|--------|--------|-----------|-------------|---------------|---------------|
| test_stone_block_002 | eat | apple-like | [] | 0.000000 | False | False |
| test_wooden_pickaxe_002 | eat | apple-like | [] | 0.000000 | True | True |
| test_apple_004 | eat | apple-like | [] | 0.000000 | True | True |
| test_apple_005 | use_as_tool | tool-like | [] | 0.000000 | True | False |
| test_stone_block_004 | craft_plank | wood-like | [] | 0.000000 | True | False |
| test_apple_001 | craft_plank | wood-like | [] | 0.000000 | True | True |
| test_wooden_pickaxe_002 | eat | apple-like | [] | 0.000000 | True | True |
| test_stone_block_009 | craft_plank | wood-like | [] | 0.000000 | True | True |
| test_wood_log_001 | craft_plank | wood-like | [] | 0.000000 | True | True |
| test_stone_block_012 | craft_plank | wood-like | [] | 0.000000 | True | True |
| test_stone_block_004 | craft_plank | wood-like | [] | 0.000000 | True | False |

## Rescued Cases

- **test_wooden_pickaxe_002** / eat (apple-like): was raw-zero, now avoided by fallback
- **test_apple_004** / eat (apple-like): was raw-zero, now avoided by fallback
- **test_apple_005** / use_as_tool (tool-like): was raw-zero, now avoided by fallback
- **test_stone_block_004** / craft_plank (wood-like): was raw-zero, now avoided by fallback
- **test_apple_001** / craft_plank (wood-like): was raw-zero, now avoided by fallback
- **test_wooden_pickaxe_002** / eat (apple-like): was raw-zero, now avoided by fallback
- **test_stone_block_009** / craft_plank (wood-like): was raw-zero, now avoided by fallback
- **test_wood_log_001** / craft_plank (wood-like): was raw-zero, now avoided by fallback
- **test_stone_block_012** / craft_plank (wood-like): was raw-zero, now avoided by fallback
- **test_stone_block_004** / craft_plank (wood-like): was raw-zero, now avoided by fallback


```
[block_done]
block_id=1J33
seed_mode=seed109
A_total_pv=18
B_signed_sum_total_pv=16
Family_only_total_pv=13
B_dev_plus_family_action_fallback_total_pv=11
Offline_total_pv=5
raw_zero_missed_cases_remaining=1
raw_zero_cases_rescued=10
rank_changed_count=34
hidden_feature_leakage_detected=False
no_oracle_leakage_confirmed=True
implementation_status=pass
failure_reason=none
elapsed=10.3s
```