# Block 1J33 ¡ª Dev+Family+Action Fallback Fix

- **Seed**: 103
- **Condition**: C4_instance_subtype_cued_v1
- **Episodes per variant**: 5
- **Elapsed**: 9.6s

## Comparison Table

| Variant | Total PV | Deceptive PV | Normal PV | Macro Bal | Delta vs A | Eff Probes | Avoided Eff | Avoided PV | Avoided NV | Helpful | Harmful |
|---------|----------|-------------|-----------|-----------|------------|------------|-------------|------------|------------|---------|---------|
| A_no_memory | 13 | 5 | 8 | 0.6176 | +0.0000 | 20 | 0 | 0 | 0 | 0 | 0 |
| B_signed_sum_current | 12 | 6 | 6 | 0.6155 | -0.0021 | 20 | 0 | 5 | 3 | 5 | 3 |
| Family_only | 12 | 6 | 6 | 0.6089 | -0.0087 | 13 | 7 | 5 | 6 | 5 | 6 |
| B_dev_plus_family_action_fallback | 13 | 7 | 6 | 0.6068 | -0.0108 | 16 | 4 | 5 | 6 | 5 | 6 |
| Offline_current | 7 | 7 | 0 | 0.6248 | +0.0072 | 20 | 0 | 12 | 5 | 12 | 5 |

## Raw-Zero Case Diagnostics

- Raw-zero cases (B_signed_sum PV with no dev_feat, penalty=0): **6**
- Raw-zero cases rescued by fallback: **1**
- Raw-zero cases still missed: **5**
- Family_only avoided cases matched: **2**
- Rank changed count (symmetric diff of probe sets): **12**
- Selected probe set changed: **10**

## Sanity Checks

- `hidden_feature_leakage_detected`: **False**
- `no_oracle_leakage_confirmed`: **True**
- `implementation_status`: pass
- `failure_reason`: none

## Raw-Zero Case Details

| Object | Action | Family | Dev Feats | Risk Penalty | Avoided by FB | Avoided by FO |
|--------|--------|--------|-----------|-------------|---------------|---------------|
| test_wood_log_001 | use_as_tool | tool-like | [] | 0.000000 | False | False |
| test_stone_block_002 | craft_plank | wood-like | [] | 0.000000 | False | False |
| test_stone_block_011 | mine_by_hand | stone-like | [] | 0.000000 | False | False |
| test_stone_block_000 | mine_by_hand | stone-like | [] | 0.000000 | False | False |
| test_stone_block_006 | mine_by_hand | stone-like | [] | 0.000000 | True | True |
| test_wooden_pickaxe_014 | craft_plank | wood-like | [] | 0.000000 | False | True |

## Rescued Cases

- **test_stone_block_006** / mine_by_hand (stone-like): was raw-zero, now avoided by fallback


```
[block_done]
block_id=1J33
seed_mode=seed109
A_total_pv=13
B_signed_sum_total_pv=12
Family_only_total_pv=12
B_dev_plus_family_action_fallback_total_pv=13
Offline_total_pv=7
raw_zero_missed_cases_remaining=5
raw_zero_cases_rescued=1
rank_changed_count=12
hidden_feature_leakage_detected=False
no_oracle_leakage_confirmed=True
implementation_status=pass
failure_reason=none
elapsed=9.6s
```