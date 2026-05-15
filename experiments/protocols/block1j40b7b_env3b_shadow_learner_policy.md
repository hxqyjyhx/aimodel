# Block 1J40b-7b: env3b Shadow Learner + Policy Intervention

- **Implementation Status**: PASS
- **Seeds**: [101, 103, 107, 109, 113]
- **Total elapsed**: 2.3s

## 1. Files

- `_block1j40b7b_env3b_shadow_learner_policy.py` ！ Created
- `runs/block1j40b7b_env3b_shadow_learner_policy.json` ！ Generated
- `protocols/block1j40b7b_env3b_shadow_learner_policy.md` ！ Generated
- `protocols/block1j40b7b_env3b_shadow_learner_policy_table.csv` ！ Generated

## 2. Design

**Algorithm**: Per-action ridge regression (alpha=1.0) with feature normalization

**Training**: LOSO (leave-one-seed-out) across 5 seeds

**Feature space**:
- Pre-surface features: 14
- Post-observe features: 28
- Model keys: 35
- Hidden features: absent from model input

**Policy chain** (per object):
1. Build pre-observe legal feature vector (pre_surface_features only, prior_obs=0)
2. Estimate pre_best_try_value and post_best_try_value from learned Q
3. `predicted_observe_net = post_best - pre_best - OBSERVE_COST`
4. If > 0: execute observe, select best post try, score true_return - cost
5. Else: execute best pre try, score true_return

**Observe costs**:
- Primary: 0.03
- Robustness: 0.05

## 3. Primary Results (cost=0.03)

| Metric | Value |
|--------|-------|
| pre_mean | 0.2500 |
| always_try | 0.2500 |
| always_observe_net | 0.4200 |
| random_observe | 0.3433 |
| oracle_selective | 0.4300 |
| **learned_policy_net** | **0.4300** |
| learned - always_observe | +0.0100 |
| learned - always_try | +0.1800 |
| learned - oracle gap | 0.0000 |
| observe_rate | 0.6667 |
| direct_try_rate | 0.3333 |
| pos_net observe rate | 1.0000 |
| nonpos_net observe rate | 0.0000 |
| harmful/cost-only obs rate | 0.0000 |
| mean \|Q - true\| error | 0.0014 |

### Per-Seed (0.03)

| Seed | Pre | Policy Net | Oracle Sel | Obs Rate | Direct | PosObs | NonposObs |
|------|-----|------------|------------|----------|--------|--------|-----------|
| 101 | 0.2500 | 0.4300 | 0.4300 | 0.6667 | 0.3333 | 1.0000 | 0.0000 |
| 103 | 0.2500 | 0.4300 | 0.4300 | 0.6667 | 0.3333 | 1.0000 | 0.0000 |
| 107 | 0.2500 | 0.4300 | 0.4300 | 0.6667 | 0.3333 | 1.0000 | 0.0000 |
| 109 | 0.2500 | 0.4300 | 0.4300 | 0.6667 | 0.3333 | 1.0000 | 0.0000 |
| 113 | 0.2500 | 0.4300 | 0.4300 | 0.6667 | 0.3333 | 1.0000 | 0.0000 |

### Per-Category (0.03)

| Category | Count | Policy Net | Obs Rate |
|----------|-------|------------|----------|
| wood_log | 45 | 0.4300 | 0.6667 |
| stone_block | 45 | 0.4300 | 0.6667 |
| apple | 45 | 0.4300 | 0.6667 |
| wooden_pickaxe | 45 | 0.4300 | 0.6667 |

### Per n_features_known (0.03)

| n_pre_features | Count | Policy Net | Obs Rate |
|----------------|-------|------------|----------|
| 4 | 60 | 0.4200 | 1.0000 |
| 5 | 90 | 0.4300 | 0.6667 |
| 6 | 30 | 0.4500 | 0.0000 |

### Acceptance (0.03)

| Check | Result |
|-------|--------|
| learned_policy_net > always_observe_net | **PASS** |
| learned_policy_net > always_try | **PASS** |
| observe_rate between 0 and 1 | **PASS** |
| pos_obs_rate > nonpos_obs_rate | **PASS** |
| hidden_audit_fields_absent | **PASS** |
| forbidden_fields_absent | **PASS** |
| **overall** | **PASS** |

## 4. Robustness Results (cost=0.05)

| Metric | Value |
|--------|-------|
| pre_mean | 0.2500 |
| always_try | 0.2500 |
| always_observe_net | 0.4000 |
| oracle_selective | 0.4167 |
| **learned_policy_net** | **0.4167** |
| learned - always_observe | +0.0167 |
| learned - always_try | +0.1667 |
| learned - oracle gap | 0.0000 |
| observe_rate | 0.6667 |
| direct_try_rate | 0.3333 |
| pos_net observe rate | 1.0000 |
| nonpos_net observe rate | 0.0000 |
| harmful/cost-only obs rate | 0.0000 |
| mean \|Q - true\| error | 0.0014 |

### Per-Seed (0.05)

| Seed | Pre | Policy Net | Oracle Sel | Obs Rate | Direct | PosObs | NonposObs |
|------|-----|------------|------------|----------|--------|--------|-----------|
| 101 | 0.2500 | 0.4167 | 0.4167 | 0.6667 | 0.3333 | 1.0000 | 0.0000 |
| 103 | 0.2500 | 0.4167 | 0.4167 | 0.6667 | 0.3333 | 1.0000 | 0.0000 |
| 107 | 0.2500 | 0.4167 | 0.4167 | 0.6667 | 0.3333 | 1.0000 | 0.0000 |
| 109 | 0.2500 | 0.4167 | 0.4167 | 0.6667 | 0.3333 | 1.0000 | 0.0000 |
| 113 | 0.2500 | 0.4167 | 0.4167 | 0.6667 | 0.3333 | 1.0000 | 0.0000 |

### Per-Category (0.05)

| Category | Count | Policy Net | Obs Rate |
|----------|-------|------------|----------|
| wood_log | 45 | 0.4167 | 0.6667 |
| stone_block | 45 | 0.4167 | 0.6667 |
| apple | 45 | 0.4167 | 0.6667 |
| wooden_pickaxe | 45 | 0.4167 | 0.6667 |

### Per n_features_known (0.05)

| n_pre_features | Count | Policy Net | Obs Rate |
|----------------|-------|------------|----------|
| 4 | 60 | 0.4000 | 1.0000 |
| 5 | 90 | 0.4167 | 0.6667 |
| 6 | 30 | 0.4500 | 0.0000 |

### Acceptance (0.05)

| Check | Result |
|-------|--------|
| learned_policy_net > always_observe_net | **PASS** |
| learned_policy_net > always_try | **PASS** |
| observe_rate between 0 and 1 | **PASS** |
| pos_obs_rate > nonpos_obs_rate | **PASS** |
| hidden_audit_fields_absent | **PASS** |
| forbidden_fields_absent | **PASS** |
| **overall** | **PASS** |

## 5. Feature Audit

- pre_surface feature count: 14
- post_observe feature count: 28
- model feature keys: 35
- hidden features in pre keys: False
- hidden features in post keys: False
- hidden-before-reveal violations: 0
- forbidden fields in model keys: 0
- Hidden features absent from model input: PASS

## 6. Three-World Separation

- Environment: holds hidden truth (category, subtype, affordance, rationale)
- Agent/Model: only legal pre_surface_features and post_observe_features
- Audit: inspects hidden truth for evaluation only
- Forbidden fields: all confirmed absent from model input

## 7. Output Files

- `runs/block1j40b7b_env3b_shadow_learner_policy.json`
- `protocols/block1j40b7b_env3b_shadow_learner_policy.md`
- `protocols/block1j40b7b_env3b_shadow_learner_policy_table.csv`


```
[block_done]
block_id=1J40b-7b
implementation_status=PASS
primary_status=PASS
robustness_status=PASS
primary_policy_net=0.4300
primary_obs_rate=0.6667
robustness_policy_net=0.4167
robustness_obs_rate=0.6667
```
