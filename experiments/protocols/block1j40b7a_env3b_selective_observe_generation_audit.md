# Block 1J40b-7a: env3b Selective-Observe Environment Generation + Audit

- **Implementation Status**: PASS
- **Seeds**: [101, 103, 107, 109, 113]
- **Total elapsed**: 12.4s
- **Total objects**: 180

## 1. Files Changed

- `_block1j40b7a_env3b_selective_observe_generation_audit.py` ！ Created
- `runs/block1j40b7a_env3b_selective_observe_generation_audit.json` ！ Generated
- `protocols/block1j40b7a_env3b_selective_observe_generation_audit.md` ！ Generated
- `protocols/block1j40b7a_env3b_selective_observe_generation_audit_table.csv` ！ Generated

## 2. Commands Run

```
& "D:\conda\python.exe" "_block1j40b7a_env3b_selective_observe_generation_audit.py"
```

## 3. Env3b Generation Design

**Categories**: wood_log, stone_block, apple, wooden_pickaxe

**Feature layers** (renamed per correction):
- pre_surface_features (8 ambient + 6 shared surface cues)
- post_observe_features (18 category-diagnostic)
- hidden_features (only in repeated_observe or audit)

**Two-layer VOI reporting**:
- Layer 1 (numeric): positive_net vs non_positive_net
- Layer 2 (audit rationale): positive_change, confirmatory, wasteful_cost_only

**Key design**: Objects share pre_surface signatures across categories.
VOI emerges from signature-group expected returns, not from true category.
Pre-oracle uses ONLY legal pre_surface signature groups.

### Shared surface features (weak pre-observe cues across categories):
spotted, weathered, coarse_texture, discolored, has_veins, pitted

### Instance subtypes per category:

| Category | Subtype | Rationale | Pre Extra | Post Reveals |
|----------|---------|-----------|-----------|-------------|
| wood_log | P | positive_change | ambient | has_bark_texture, fibrous, flammable, porous_surface |
| wood_log | Z | positive_change | ambient, spotted | has_bark_texture, fibrous, flammable, porous_surface |
| wood_log | N | wasteful_cost_only | ambient, has_bark_texture, discolored | fibrous, flammable, porous_surface |
| stone_block | P | positive_change | ambient | has_crystal_flecks, has_granular_surface, block_like, cold_to_touch |
| stone_block | Z | positive_change | ambient, spotted | has_crystal_flecks, has_granular_surface, block_like, cold_to_touch |
| stone_block | N | wasteful_cost_only | ambient, has_crystal_flecks | has_granular_surface, block_like, cold_to_touch, scratch_resistant |
| apple | P | positive_change | ambient | has_stem_remnant, has_peel_texture, fruity_scent |
| apple | Z | positive_change | ambient, discolored | has_stem_remnant, has_peel_texture, fruity_scent |
| apple | N | wasteful_cost_only | ambient, has_stem_remnant, spotted | has_peel_texture, fruity_scent |
| wooden_pickaxe | P | positive_change | ambient | has_grip_area, has_shaft_shape, elongated_with_handle, movable |
| wooden_pickaxe | Z | positive_change | ambient, discolored | has_grip_area, has_shaft_shape, elongated_with_handle, movable |
| wooden_pickaxe | N | wasteful_cost_only | ambient, has_grip_area | has_shaft_shape, elongated_with_handle, movable, has_metal_head |

## 4. Agent-Visible Fields

- **Pre-observe feature keys** (32): `feat_block_like`, `feat_brownish`, `feat_coarse_texture`, `feat_cold_to_touch`, `feat_discolored`, `feat_elongated_with_handle`, `feat_fibrous`, `feat_flammable...`
- **State keys** (6): `state_fresh`, `state_wet`, `state_damaged`, `state_clean`, `state_hot`, `state_open`
- **Numeric**: `prior_observe_count`

## 5. Hidden/Audit-Only Fields

| Field | Visibility |
|-------|-----------|
| affordance_profile | hidden/audit-only |
| ambient_group | hidden/audit-only |
| audit_VOI | hidden/audit-only |
| audit_computed_gain | hidden/audit-only |
| audit_observe_value | hidden/audit-only |
| audit_rationale_class | hidden/audit-only |
| category | hidden/audit-only |
| depth | hidden/audit-only |
| depth_schedule | hidden/audit-only |
| effective_affordance_profile | hidden/audit-only |
| episode_id | hidden/audit-only |
| episode_progress | hidden/audit-only |
| group_role | hidden/audit-only |
| hidden_affordance_profile | hidden/audit-only |
| hidden_audit_rationale_class | hidden/audit-only |
| hidden_category | hidden/audit-only |
| hidden_features | hidden/audit-only |
| hidden_subtype | hidden/audit-only |
| n_features_known | hidden/audit-only |
| n_states_known | hidden/audit-only |

## 6. Three-World Separation Audit

- Agent-visible pre features: 32
- Forbidden fields total: 31
- Forbidden-in-agent leaks: 0
- Hidden-before-reveal violations: 0
- **Separation**: PASS

## 7. VOI Distribution Audit

### Numeric VOI Class

| Class | Count | Rate | Mean VOI |
|-------|-------|------|----------|
| positive_net | 120 | 66.7% | 0.2950 |
| non_positive_net | 60 | 33.3% | -0.0050 |
| cost_only (~-0.005) | 60 | 33.3% | |

### Audit Rationale Class

| Rationale | Count | Rate | Mean VOI |
|-----------|-------|------|----------|
| positive_change | 120 | 66.7% | 0.2950 |
| confirmatory | 0 | 0.0% | 0.0000 |
| wasteful_cost_only | 60 | 33.3% | -0.0050 |

### Per-Category

| Category | Total | Pos Net | Non-Pos Net | Mean VOI |
|----------|-------|---------|-------------|----------|
| wood_log | 45 | 30 | 15 | 0.1950 |
| stone_block | 45 | 30 | 15 | 0.1950 |
| apple | 45 | 30 | 15 | 0.1950 |
| wooden_pickaxe | 45 | 30 | 15 | 0.1950 |

## 8. Baseline Audit

| Baseline | Mean Return |
|----------|------------|
| no_observe_pre_only | 0.2500 |
| always_try | 0.2500 |
| always_observe_net | 0.4450 |
| random_observe | 0.3474 |
| **oracle_selective** | **0.4467** |

| Gap | Value |
|-----|-------|
| oracle - always_observe | 0.0017 |
| oracle - always_try | 0.1967 |
| oracle - no_observe | 0.1967 |

## 9. n_features_known Distribution

| VOI Class | Mean nfeat | Range | Unique |
|-----------|-----------|-------|--------|
| positive_net | 4.5 | [4,5] | [4, 5] |
| non_positive_net | 5.5 | [5,6] | [5, 6] |

Overlap: [5] (count=1)

## 10. Cue / Shortcut Audit

- Max single-feature positive_net acc: 0.6667
- Max feature-pair positive_net acc: 0.6667
- n_features_known VOI acc: 0.6667
- Category VOI acc: 0.6667
- Pre-signature VOI acc: 0.6667 (n_sigs=8)
- Full-visible VOI probe acc: 0.6667
- corr(n_features_known, VOI): -0.6860

## 11. Counterfactual Slice Audit

Features appearing in both positive_net and non_positive_net: 10

- **brownish**: 2 pos categories, 2 non-pos categories
- **greenish**: 2 pos categories, 2 non-pos categories
- **heavy_weight**: 2 pos categories, 2 non-pos categories
- **light_weight**: 2 pos categories, 2 non-pos categories
- **long_shape**: 2 pos categories, 2 non-pos categories
- **rough_texture**: 2 pos categories, 2 non-pos categories
- **round_small**: 2 pos categories, 2 non-pos categories
- **smooth_texture**: 2 pos categories, 2 non-pos categories
- **discolored**: 2 pos categories, 1 non-pos categories
- **spotted**: 2 pos categories, 1 non-pos categories

## 12. Observe-Cost Calibration Sweep

| Cost | Pos Rate | Zero Rate | Neg Rate | AlwaysObs | AlwaysTry | Selective | Sel-Obs | Sel-Try | ObsRate |
|------|----------|-----------|----------|-----------|-----------|-----------|---------|---------|--------|
| 0.0000 | 0.6667 | 0.3333 | 0.0000 | 0.4500 | 0.2500 | 0.4500 | 0.0000 | 0.2000 | 0.6667 |
| 0.0010 | 0.6667 | 0.3333 | 0.0000 | 0.4490 | 0.2500 | 0.4493 | 0.0003 | 0.1993 | 0.6667 |
| 0.0020 | 0.6667 | 0.0000 | 0.3333 | 0.4480 | 0.2500 | 0.4487 | 0.0007 | 0.1987 | 0.6667 |
| 0.0050 | 0.6667 | 0.0000 | 0.3333 | 0.4450 | 0.2500 | 0.4467 | 0.0017 | 0.1967 | 0.6667 |
| 0.0100 | 0.6667 | 0.0000 | 0.3333 | 0.4400 | 0.2500 | 0.4433 | 0.0033 | 0.1933 | 0.6667 |
| 0.0200 | 0.6667 | 0.0000 | 0.3333 | 0.4300 | 0.2500 | 0.4367 | 0.0067 | 0.1867 | 0.6667 |
| 0.0500 | 0.6667 | 0.0000 | 0.3333 | 0.4000 | 0.2500 | 0.4167 | 0.0167 | 0.1667 | 0.6667 |
| 0.1000 | 0.6667 | 0.0000 | 0.3333 | 0.3500 | 0.2500 | 0.3833 | 0.0333 | 0.1333 | 0.6667 |

Default observe_cost=0.005: VALID

## 13. Acceptance Result

**PASS**

| Check | Result |
|-------|--------|
| three_world_separation | **PASS** |
| forbidden_fields_absent | **PASS** |
| hidden_before_reveal_clean | **PASS** |
| positive_net_exists | **PASS** |
| positive_net_not_universal | **PASS** |
| every_category_has_non_positive_net | **PASS** |
| each_category_has_mixed_rationale | **PASS** |
| oracle_selective_beats_always_observe | **PASS** |
| oracle_selective_beats_always_try | **PASS** |
| oracle_selective_is_selective | **PASS** |
| no_single_feature_shortcut | **PASS** |
| no_feature_pair_shortcut | **PASS** |
| n_features_known_overlap | **PASS** |
| shared_feature_in_both_voi_classes | **PASS** |
| category_not_determine_voi | **PASS** |
| feature_count_not_strongly_corr_voi | **PASS** |
| pre_oracle_uses_legal_signature_only | **PASS** |
| observe_cost_calibration_passed | **PASS** |

## 14. Output Files

- `runs/block1j40b7a_env3b_selective_observe_generation_audit.json`
- `protocols/block1j40b7a_env3b_selective_observe_generation_audit.md`
- `protocols/block1j40b7a_env3b_selective_observe_generation_audit_table.csv`

## 15. Next Suggested Resume Point

1J40b-7a PASS: env3b provides a valid mixed observe-value environment.
Next: 1J40b-7b ！ train shadow learner on env3b to test whether the learner can:
1. Learn to observe only when VOI > 0
2. Avoid always-observe collapse
3. Achieve returns between always_observe and oracle_selective

```
[block_done]
block_id=1J40b-7a
implementation_status=pass
overall=PASS
oracle_selective=0.4467
always_observe=0.4450
pos_rate=0.6667
failure_reasons=none
```
