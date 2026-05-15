# Block 1J40b-5: Shadow Strategy Learner on env3a

- **Implementation Status**: PASS
- **Seeds**: [101, 103, 107, 109, 113]
- **Total elapsed**: 6.4s
- **Total test objects**: 180

## 1. Files Changed

- `_block1j40b5_strategy_learning_shadow.py` ！ Created
- `runs/block1j40b5_strategy_learning_shadow.json` ！ Generated
- `protocols/block1j40b5_strategy_learning_shadow.md` ！ Generated
- `protocols/block1j40b5_strategy_learning_shadow_table.csv` ！ Generated

## 2. Learner Design

- **Algorithm**: Per-action ridge regression (alpha=1.0) with feature normalization
- **Training**: LOSO (leave-one-seed-out) across 5 seeds
- **Feature space**: 73 features (40 visible + 12 hidden + 6 state + 1 numeric)
- **Pre-observe vector**: ambient features only + prior_observe_count=0
- **Post-observe vector**: ambient+core+diagnostic + state + prior_observe_count=1
- **Shadow decision**: observe if `max(post_Q_try) - max(pre_Q_try) > OBSERVE_COST`
- **Observe credit**: simple -OBSERVE_COST (no matched advantage)

## 3. Exact Input Fields Used

- **Pre-model features** (8): `feat_brownish, feat_greenish, feat_heavy_weight, feat_light_weight, feat_long_shape, feat_rough_texture, feat_round_small, feat_smooth_texture`...
- **Post-model features** (47 total): 40 visible + 6 state + 1 numeric
- **Hidden features** (12): `hfeat_bruise_markings, hfeat_carvable_interior, hfeat_cleavage_plane, hfeat_edible_core, hfeat_hairline_crack, hfeat_handle_looseness, hfeat_internal_fractures, hfeat_layered_structure, hfeat_moisture_content, hfeat_shaft_rot, hfeat_sweet_flesh, hfeat_wax_coating`
- **Numeric**: `prior_observe_count` only

## 4. Forbidden Fields Confirmed Unused

| Field | Status |
|-------|--------|
| ambient_group | ABSENT |
| audit_computed_gain | ABSENT |
| category | ABSENT |
| depth | ABSENT |
| effective_affordance_profile | ABSENT |
| episode_id | ABSENT |
| episode_progress | ABSENT |
| group_role | ABSENT |
| n_features_known | ABSENT |
| n_states_known | ABSENT |
| object_id | ABSENT |
| oracle_action | ABSENT |
| seed_id | ABSENT |
| true_affordance_profile | ABSENT |

## 5. Baseline Comparison Table

| Baseline | Mean True Return |
|----------|-----------------|
| no_observe_pre_only | 0.1633 |
| random_observe | 0.3100 |
| always_observe (gross) | 0.4500 |
| always_observe (net) | 0.4450 |
| oracle (audit only) | 0.4500 |
| **shadow_learner (gross)** | **0.4500** |
| **shadow_learner (net)** | **0.4450** |

## 6. Gross vs Net Return Table

| Condition | Gross Return | Net Return (after observe cost) |
|-----------|-------------|------|
| pre (no observe) | 0.1633 | 0.1633 |
| always_observe | 0.4500 | 0.4450 |
| shadow_learner | 0.4500 | 0.4450 |
| oracle | 0.4500 | 0.4500 |

## 7. Per-Seed Results

| Seed | Pre | Post Net | Random | Shadow Net | Oracle | Obs Rate |
|------|-----|----------|--------|------------|--------|----------|
| 101 | 0.1833 | 0.4450 | 0.2500 | 0.4450 | 0.4500 | 1.0000 |
| 103 | 0.1500 | 0.4450 | 0.3167 | 0.4450 | 0.4500 | 1.0000 |
| 107 | 0.1500 | 0.4450 | 0.3667 | 0.4450 | 0.4500 | 1.0000 |
| 109 | 0.1833 | 0.4450 | 0.2833 | 0.4450 | 0.4500 | 1.0000 |
| 113 | 0.1500 | 0.4450 | 0.3333 | 0.4450 | 0.4500 | 1.0000 |

## 8. Per-Category Results

| Category | Count | Shadow Net | Oracle |
|----------|-------|------------|--------|
| wood_log | 45 | 0.4450 | 0.4500 |
| stone_block | 45 | 0.4450 | 0.4500 |
| apple | 45 | 0.4450 | 0.4500 |
| wooden_pickaxe | 45 | 0.4450 | 0.4500 |

## 9. Observe-Rate Analysis

- observe_rate: 1.0000
- zero_gain_observe_rate: ~0 (all observes have positive true gain in env3a)
- harmful_observe_rate: 0.0 (post >= pre for all objects in env3a)
- Note: High observe rate is NOT a failure in env3a ！ most objects benefit from observation.

## 10. Leakage Audit

- recursive_leakage_ok: True
- full_role_probe_accuracy: 0.3333 (chance=0.333)
- mean_seed_role_probe: 0.3333
- max_seed_role_probe: 0.3333
- forbidden_learner_fields_found: 0
- hidden_in_pre: False
- hidden_in_post: False

## 11. Feature Audit Summary

- **Pre-model features**: 8 ambient feature indicators
- **Post-model features**: 47 (visible + state + numeric)
- **Hidden features**: 12 (repeated_observe only)
- **Numeric fields**: prior_observe_count only
- **REMOVED from env2 learner**: episode_id, episode_progress, n_features_known, n_states_known
- **Hidden features absent from normal pre/post eval**: true

## 12. Acceptance Checks

| Check | Result |
|-------|--------|
| role_probe_pass | **PASS** |
| leakage_pass | **PASS** |
| shadow_net > pre | **PASS** |
| shadow_net > random | **PASS** |
| shadow-pre gain > 0 | **PASS** |
| harmful_observe_rate_low | **PASS** |
| hidden_features_absent_from_pre | **PASS** |
| hidden_features_absent_from_post | **PASS** |
| forbidden_fields_absent | **PASS** |
| **overall** | **PASS** |

## 13. Commands Run

```
& "D:\conda\python.exe" "_block1j40b5_strategy_learning_shadow.py"
```

## 14. Output Files

- `runs/block1j40b5_strategy_learning_shadow.json`
- `protocols/block1j40b5_strategy_learning_shadow.md`
- `protocols/block1j40b5_strategy_learning_shadow_table.csv`

## 15. Next Suggested Resume Point

If acceptance passes: analyze whether shadow learner exploits category disambiguation or collapses to always-observe.
If shadow 「 always_observe: the learner discovered that env3a post-observe is nearly always beneficial.
If shadow 「 pre: the learner failed to learn from post-observe features ！ debug Q-value estimation.
Next step could be env3b with non-deterministic diagnostic features and detectable helps bonuses.


```
[block_done]
block_id=1J40b-5
implementation_status=pass
shadow_mean_net=0.4450
pre_mean=0.1633
oracle_mean=0.4500
observe_rate=1.0000
role_probe=0.3333
failure_reasons=none
```
