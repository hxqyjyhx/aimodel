# Block 1J33 ¡ª Dev+Family+Action Fallback Fix

- **Seed**: 101
- **Condition**: C4_instance_subtype_cued_v1
- **Episodes per variant**: 5
- **Elapsed**: 10.1s

## Comparison Table

| Variant | Total PV | Deceptive PV | Normal PV | Macro Bal | Delta vs A | Eff Probes | Avoided Eff | Avoided PV | Avoided NV | Helpful | Harmful |
|---------|----------|-------------|-----------|-----------|------------|------------|-------------|------------|------------|---------|---------|
| A_no_memory | 13 | 4 | 9 | 0.5699 | +0.0000 | 20 | 0 | 0 | 0 | 0 | 0 |
| B_signed_sum_current | 9 | 4 | 5 | 0.5723 | +0.0024 | 20 | 0 | 2 | 3 | 2 | 3 |
| Family_only | 12 | 6 | 6 | 0.5645 | -0.0054 | 15 | 5 | 2 | 2 | 2 | 2 |
| B_dev_plus_family_action_fallback | 6 | 2 | 4 | 0.5730 | +0.0031 | 18 | 2 | 3 | 2 | 3 | 2 |
| Offline_current | 0 | 0 | 0 | 0.5787 | +0.0088 | 20 | 0 | 9 | 5 | 9 | 5 |

## Raw-Zero Case Diagnostics

- Raw-zero cases (B_signed_sum PV with no dev_feat, penalty=0): **5**
- Raw-zero cases rescued by fallback: **1**
- Raw-zero cases still missed: **4**
- Family_only avoided cases matched: **1**
- Rank changed count (symmetric diff of probe sets): **6**
- Selected probe set changed: **6**

## Sanity Checks

- `hidden_feature_leakage_detected`: **False**
- `no_oracle_leakage_confirmed`: **True**
- `implementation_status`: pass
- `failure_reason`: none

## Raw-Zero Case Details

| Object | Action | Family | Dev Feats | Risk Penalty | Avoided by FB | Avoided by FO |
|--------|--------|--------|-----------|-------------|---------------|---------------|
| test_stone_block_000 | mine_by_hand | stone-like | [] | 0.000000 | False | False |
| test_wooden_pickaxe_013 | use_as_tool | tool-like | [] | 0.000000 | False | False |
| test_stone_block_005 | mine_by_hand | stone-like | [] | 0.000000 | True | True |
| test_wood_log_011 | craft_plank | wood-like | [] | 0.000000 | False | False |
| test_stone_block_010 | craft_plank | wood-like | [] | 0.000000 | False | False |

## Rescued Cases

- **test_stone_block_005** / mine_by_hand (stone-like): was raw-zero, now avoided by fallback


```
[block_done]
block_id=1J33
seed_mode=seed109
A_total_pv=13
B_signed_sum_total_pv=9
Family_only_total_pv=12
B_dev_plus_family_action_fallback_total_pv=6
Offline_total_pv=0
raw_zero_missed_cases_remaining=4
raw_zero_cases_rescued=1
rank_changed_count=6
hidden_feature_leakage_detected=False
no_oracle_leakage_confirmed=True
implementation_status=pass
failure_reason=none
elapsed=10.1s
```