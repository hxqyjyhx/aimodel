# Block 1J32 -- Seed 109 Cross-Family Representation Audit

## 1. Purpose

Audit why cross-family deceptive objects with no injected deviation features
are not represented by signed memory keys. Analyzes the 12 raw-zero missed
offline-avoidable PV cases from 1J31.

## 2. Source Files

- `runs/block1j31_seed109_raw_zero_monitor.json`
- `runs/block1j31_seed109_raw_penalty_zero_source_audit.json`

## 3. 1J31 Reference Findings

- A_total_pv: 29
- Family_only_total_pv: 24
- B_signed_sum_total_pv: 27
- Offline_total_pv: 6
- Missed offline-avoidable PVs: 16
- Raw-zero missed PVs: 12
- Dominant raw-zero reason: feature_not_in_memory_universe

## 4. Raw-Zero Cross-Family Case Table

| Object | Action | Inferred Family | Dominant Cue | 2nd Cue | Conflict | Dev Feats | Type Feats | Rep Gap | FamOnly Avoid |
|--------|--------|----------------|-------------|---------|----------|-----------|------------|---------|--------------|
| test_apple_001 | craft_plank | wood-like | wood-like | stone-like | True | none | block_like;brownish;grayish +4 | cross_family_conflict_not_represented | False |
| test_apple_004 | eat | apple-like | apple-like | tool-like | True | none | block_like;greenish;has_shaft_shape +5 | family_only_signal_not_integrated | True |
| test_apple_005 | use_as_tool | tool-like | tool-like | wood-like | True | none | block_like;elongated_with_handle;has_stem_remnant +4 | cross_family_conflict_not_represented | False |
| test_stone_block_002 | eat | apple-like | apple-like | tool-like | True | none | has_stem_remnant;heavy_weight;light_weight +2 | cross_family_conflict_not_represented | False |
| test_stone_block_003 | use_as_tool | tool-like | tool-like | stone-like | True | none | has_granular_surface;has_grip_area;has_shaft_shape +4 | cross_family_conflict_not_represented | False |
| test_stone_block_004 | craft_plank | wood-like | wood-like | stone-like | True | none | brownish;has_bark_texture;has_crystal_flecks +6 | family_only_signal_not_integrated | True |
| test_stone_block_006 | eat | apple-like | apple-like | wood-like | True | none | greenish;has_granular_surface;has_shaft_shape +4 | family_only_signal_not_integrated | True |
| test_stone_block_008 | craft_plank | wood-like | wood-like | stone-like | True | none | block_like;brownish;greenish +4 | family_only_signal_not_integrated | True |
| test_stone_block_012 | craft_plank | wood-like | wood-like | tool-like | True | none | brownish;has_shaft_shape;has_wood_grain +4 | family_only_signal_not_integrated | True |
| test_wood_log_001 | craft_plank | wood-like | wood-like | stone-like | True | none | block_like;elongated_with_handle;grayish +5 | family_only_signal_not_integrated | True |
| test_wooden_pickaxe_002 | eat | apple-like | apple-like | tool-like | True | none | brownish;greenish;has_grip_area +5 | family_only_signal_not_integrated | True |
| test_wooden_pickaxe_005 | eat | apple-like | apple-like | tool-like | True | none | has_bark_texture;has_granular_surface;has_grip_area +6 | family_only_signal_not_integrated | True |

## 5. Feature Routing Summary

- total_raw_zero_cases: 12
- count_cases_with_no_injected_dev_feats: 12
- count_cases_with_family_cue_conflict: 12
- count_cases_where_type_family_features_present: 12
- count_cases_where_type_family_features_excluded_from_signed_memory: 12
- count_cases_where_family_only_avoided: 8
- count_cases_where_offline_has_signal_without_dev_feat: 0
- count_cases_where_signed_memory_requires_dev_feat: 12
- count_cases_offline_raw_signal_nonzero: 0
- count_cases_offline_raw_signal_zero: 12
- count_cases_offline_avoid_aligned_true: 0
- count_cases_offline_avoid_unknown: 12

## 6. Family Cue Conflict Summary

| Inferred Family | Dominant Cue | Action | Cases | FamOnly Avoided | Offline Avoided | Mean Margin | Mean Conflict |
|----------------|-------------|--------|-------|----------------|----------------|-------------|--------------|
| apple-like | apple-like | eat | 5 | 4 | 5 | 0.2 | 0.6567 |
| tool-like | tool-like | use_as_tool | 2 | 0 | 2 | 1.5 | 0.5625 |
| wood-like | wood-like | craft_plank | 5 | 4 | 5 | 0.0 | 0.7079 |

## 7. Family-Only vs Signed-Memory Comparison

- Raw-zero cases where Family_only avoids but signed memory doesn't: 8
  - test_apple_004 / eat: family_only_uses_different_keys_without_dev_feat_requirement
  - test_stone_block_004 / craft_plank: family_only_uses_different_keys_without_dev_feat_requirement
  - test_stone_block_006 / eat: family_only_uses_different_keys_without_dev_feat_requirement
  - test_stone_block_008 / craft_plank: family_only_uses_different_keys_without_dev_feat_requirement
  - test_stone_block_012 / craft_plank: family_only_uses_different_keys_without_dev_feat_requirement
  - test_wood_log_001 / craft_plank: family_only_uses_different_keys_without_dev_feat_requirement
  - test_wooden_pickaxe_002 / eat: family_only_uses_different_keys_without_dev_feat_requirement
  - test_wooden_pickaxe_005 / eat: family_only_uses_different_keys_without_dev_feat_requirement

## 8. Offline Signal Comparison

- Offline raw signal nonzero: 0
- Offline raw signal zero: 12
- Offline avoid aligned true: 0
- Offline avoid unknown: 12
- Offline has signal without dev_feat: 0
- Method: per-case aligned (not set-difference from trajectory-level PV absence)
- Conclusion: offline oracle also has zero penalty for all raw-zero cases — offline memory equally depends on dev_feat keys.

## 9. Candidate Representation-Channel Diagnostic Table

| Object | Action | Family Action Key | Cue Vector Key | Conflict Key | Boundary Key | Dev Feat Key |
|--------|--------|------------------|---------------|-------------|-------------|-------------|
| test_apple_001 | craft_plank | True | True | True | True | False |
| test_apple_004 | eat | True | True | True | True | False |
| test_apple_005 | use_as_tool | True | True | True | True | False |
| test_stone_block_002 | eat | True | True | True | True | False |
| test_stone_block_003 | use_as_tool | True | True | True | True | False |
| test_stone_block_004 | craft_plank | True | True | True | True | False |
| test_stone_block_006 | eat | True | True | True | True | False |
| test_stone_block_008 | craft_plank | True | True | True | True | False |
| test_stone_block_012 | craft_plank | True | True | True | True | False |
| test_wood_log_001 | craft_plank | True | True | True | True | False |
| test_wooden_pickaxe_002 | eat | True | True | True | True | False |
| test_wooden_pickaxe_005 | eat | True | True | True | True | False |

## 10. Leakage Audit Summary

- hidden_feature_leakage_detected: False
- outcome_leakage_detected: False
- deceptive_flag_used_by_policy: False
- prior_violation_label_used_by_policy: False

## 11. Boolean Flags

### all_12_raw_zero_cases_loaded: True
  Rule: exactly 12 raw-zero cases from 1J31 missed table
  - count: 12

### audit_completed: True
  Rule: all 12 raw-zero cases analyzed
  - cases_loaded: 12

### cross_family_representation_gap_confirmed: True
  Rule: no_dev_feat_dominates AND type_family_features_excluded AND no hidden leakage
  - no_dev_feat_dominates: True
  - type_family_excluded: True
  - no_hidden_leakage: True

### deceptive_flag_used_by_policy: False
  Rule: deceptive flag is audit-only, not accessed by online policy

### family_action_key_missing: True
  Rule: signed memory does not construct keys from (goal, action, family) alone — always requires dev_feat

### family_cue_conflict_detected: True
  Rule: at least one raw-zero case has cross-family cue conflict
  - count: 12

### family_only_signal_available_but_not_integrated: True
  Rule: Family_only avoids at least one raw-zero case that signed memory misses
  - count: 8

### hidden_feature_leakage_detected: False
  Rule: from 1J31 decision_path_audit

### no_dev_feat_dominates_raw_zero_cases: True
  Rule: >= 75% of raw-zero cases have no injected deviation features
  - count_no_dev_feat: 12
  - total: 12

### no_oracle_leakage_confirmed: True
  Rule: audit reads only 1J31 outputs, no new simulation

### outcome_leakage_detected: False
  Rule: audit does not access ground truth outcomes beyond 1J31 data

### prior_violation_label_used_by_policy: False
  Rule: PV label is post-hoc audit metric, not used by online policy at decision time

### signed_memory_requires_dev_feat_confirmed: True
  Rule: signed memory keys are (goal, action, family, dev_feat); dev_feat must be present on object
  - num_raw_zero_without_dev_feat: 12

### type_family_features_excluded_from_signed_memory: True
  Rule: type family features present on objects but not used in signed memory key construction
  - count: 12

## 12. Short Factual Summary

1. Do the 12 raw-zero cases lack dev_feat? **Yes** — 12/12 have no injected deviation features attached.
2. Do type/family cues exist but are excluded from signed memory? **Yes** — 12/12 cases have observable type-family features, but signed memory only uses deviation features for key construction.
3. Does family-only have useful signal not integrated into signed memory? **Yes** — Family_only avoids 8/12 raw-zero cases.
4. Is cross-family representation gap confirmed? **True**
5. What remains unknown: offline oracle also misses most raw-zero cases (0/12 have offline signal without dev_feat), suggesting the offline oracle also relies on dev_feat keys.

## 13. Summary

```
[block_done]
block_id=1J32_fix
seed=109
offline_comparison_method=per_case_aligned
raw_zero_cases_loaded=12
count_cases_with_no_injected_dev_feats=12
count_cases_with_family_cue_conflict=12
count_cases_where_type_family_features_present=12
count_cases_where_type_family_features_excluded_from_signed_memory=12
count_cases_where_family_only_avoided=8
count_cases_offline_raw_signal_nonzero=0
count_cases_offline_raw_signal_zero=12
count_cases_offline_avoid_aligned_true=0
count_cases_offline_avoid_unknown=12
count_cases_where_offline_has_signal_without_dev_feat=0
signed_memory_requires_dev_feat_confirmed=True
no_dev_feat_dominates_raw_zero_cases=True
type_family_features_excluded_from_signed_memory=True
family_only_signal_available_but_not_integrated=True
cross_family_representation_gap_confirmed=True
hidden_feature_leakage_detected=False
no_oracle_leakage_confirmed=True
json_output=runs/block1j32_seed109_cross_family_representation_audit.json
md_output=protocols/block1j32_seed109_cross_family_representation_audit.md
csv_output=protocols/block1j32_seed109_cross_family_representation_audit_table.csv
implementation_status=pass
failure_reason=none
elapsed=0.0s
```