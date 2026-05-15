# Block 1J33 ¡ª Dev+Family+Action Fallback Fix

- **Seed**: 113
- **Condition**: C4_instance_subtype_cued_v1
- **Episodes per variant**: 5
- **Elapsed**: 10.1s

## Comparison Table

| Variant | Total PV | Deceptive PV | Normal PV | Macro Bal | Delta vs A | Eff Probes | Avoided Eff | Avoided PV | Avoided NV | Helpful | Harmful |
|---------|----------|-------------|-----------|-----------|------------|------------|-------------|------------|------------|---------|---------|
| A_no_memory | 16 | 6 | 10 | 0.6427 | +0.0000 | 25 | 0 | 0 | 0 | 0 | 0 |
| B_signed_sum_current | 5 | 1 | 4 | 0.6473 | +0.0046 | 25 | 0 | 8 | 6 | 8 | 6 |
| Family_only | 14 | 8 | 6 | 0.6391 | -0.0036 | 15 | 10 | 4 | 5 | 4 | 5 |
| B_dev_plus_family_action_fallback | 5 | 1 | 4 | 0.6473 | +0.0046 | 25 | 0 | 8 | 6 | 8 | 6 |
| Offline_current | 0 | 0 | 0 | 0.6505 | +0.0078 | 25 | 0 | 13 | 7 | 13 | 7 |

## Raw-Zero Case Diagnostics

- Raw-zero cases (B_signed_sum PV with no dev_feat, penalty=0): **4**
- Raw-zero cases rescued by fallback: **0**
- Raw-zero cases still missed: **4**
- Family_only avoided cases matched: **0**
- Rank changed count (symmetric diff of probe sets): **0**
- Selected probe set changed: **0**

## Sanity Checks

- `hidden_feature_leakage_detected`: **False**
- `no_oracle_leakage_confirmed`: **True**
- `implementation_status`: pass
- `failure_reason`: none

## Raw-Zero Case Details

| Object | Action | Family | Dev Feats | Risk Penalty | Avoided by FB | Avoided by FO |
|--------|--------|--------|-----------|-------------|---------------|---------------|
| test_wooden_pickaxe_012 | use_as_tool | tool-like | [] | 0.000000 | False | False |
| test_stone_block_009 | craft_plank | wood-like | [] | 0.000000 | False | False |
| test_apple_006 | use_as_tool | tool-like | [] | 0.000000 | False | False |
| test_wood_log_013 | craft_plank | wood-like | [] | 0.000000 | False | False |


```
[block_done]
block_id=1J33
seed_mode=seed109
A_total_pv=16
B_signed_sum_total_pv=5
Family_only_total_pv=14
B_dev_plus_family_action_fallback_total_pv=5
Offline_total_pv=0
raw_zero_missed_cases_remaining=4
raw_zero_cases_rescued=0
rank_changed_count=0
hidden_feature_leakage_detected=False
no_oracle_leakage_confirmed=True
implementation_status=pass
failure_reason=none
elapsed=10.1s
```