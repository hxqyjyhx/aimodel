# Block 1J40b-7d: env3b Abstract Pre-Only VOI Feature Generalization Test

- **Implementation Status**: FAIL
- **Elapsed**: 19.2s
- **Seeds**: [101, 103, 107, 109, 113]

## 1. Files

- `_block1j40b7d_env3b_abstract_voi_feature_generalization.py` ！ Written
- `runs/block1j40b7d_env3b_abstract_voi_feature_generalization.json` ！ Generated
- `protocols/block1j40b7d_env3b_abstract_voi_feature_generalization.md` ！ Generated
- `protocols/block1j40b7d_env3b_abstract_voi_feature_generalization_table.csv` ！ Generated

## 2. Commands Run

```
& "D:\conda\python.exe" "_block1j40b7d_env3b_abstract_voi_feature_generalization.py"
```

## 3. Variant Definitions

- **D0**: raw pre_vec only (7c baseline)
  - use_raw=True, use_abstract=False, exclude_exact_sig=False

- **D1**: raw pre_vec + abstract features
  - use_raw=True, use_abstract=True, exclude_exact_sig=False

- **D2**: abstract features only
  - use_raw=False, use_abstract=True, exclude_exact_sig=False

- **D3**: raw + abstract, minus exact-signature features (10,11)
  - use_raw=True, use_abstract=True, exclude_exact_sig=True

## 4. Abstract Feature List

17 features, all legal (pre_surface_features, training experience, trained Q-function only):

| # | Feature | Source |
|---|---------|--------|
| 1 | n_pre_surface_features | pre_surface_features |
| 2 | n_ambient_features_known | pre_surface_features |
| 3 | n_shared_surface_cues_known | pre_surface_features |
| 4 | n_pre_visible_diagnostic_cues_known | pre_surface_features |
| 5 | has_any_pre_visible_diagnostic_cue | pre_surface_features |
| 6 | has_any_shared_surface_cue | pre_surface_features |
| 7 | pre_try_value_spread | Q-estimates from pre_vec |
| 8 | pre_best_second_margin | Q-estimates from pre_vec |
| 9 | pre_action_value_variance | Q-estimates from pre_vec |
| 10 | training_support_count_for_exact_signature | training signature statistics |
| 11 | training_mean_observe_target_for_exact_signature | training signature statistics |
| 12 | cue_support_mean | training cue/nfeat statistics |
| 13 | cue_support_max | training cue/nfeat statistics |
| 14 | cue_observe_target_mean | training cue/nfeat statistics |
| 15 | cue_observe_target_max | training cue/nfeat statistics |
| 16 | nfeat_group_mean_observe_target | training cue conflict |
| 17 | shared_cue_conflict_sum | training cue conflict |

- Abstract forbidden field leak: False

## 5. Forbidden-Field Audit

- Forbidden fields in model keys: 0
- Abstract forbidden field leak: False
- Hidden-before-reveal violations: 0
- Decision-time post_vec absent: True (two-stage policy)

## 6. Results at observe_cost=0.03

### Test A: Held-out Signatures

| Variant | Status | PolicyNet | AlwaysObs | OracleSel | ObsRate | PosObs | NonposObs | vsAlwaysObs | vsOracle |
|---------|--------|-----------|-----------|-----------|---------|--------|-----------|-------------|----------|
| D0 | **FAIL** | 0.4200 | 0.4200 | 0.4300 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | 0.0100 |
| D1 | **FAIL** | 0.4200 | 0.4200 | 0.4300 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | 0.0100 |
| D2 | **FAIL** | 0.4200 | 0.4200 | 0.4300 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | 0.0100 |
| D3 | **FAIL** | 0.4200 | 0.4200 | 0.4300 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | 0.0100 |

**D0 checks:**
| policy_net > always_observe_net | **FAIL** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **FAIL** |
| pos_obs > nonpos_obs | **FAIL** |
Failures: ['policy_net > always_observe_net', 'obs_rate between 0 and 1', 'pos_obs > nonpos_obs']

**D1 checks:**
| policy_net > always_observe_net | **FAIL** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **FAIL** |
| pos_obs > nonpos_obs | **FAIL** |
Failures: ['policy_net > always_observe_net', 'obs_rate between 0 and 1', 'pos_obs > nonpos_obs']

**D2 checks:**
| policy_net > always_observe_net | **FAIL** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **FAIL** |
| pos_obs > nonpos_obs | **FAIL** |
Failures: ['policy_net > always_observe_net', 'obs_rate between 0 and 1', 'pos_obs > nonpos_obs']

**D3 checks:**
| policy_net > always_observe_net | **FAIL** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **FAIL** |
| pos_obs > nonpos_obs | **FAIL** |
Failures: ['policy_net > always_observe_net', 'obs_rate between 0 and 1', 'pos_obs > nonpos_obs']

### Test B: Held-out Cue Combinations

| Variant | Status | PolicyNet | AlwaysObs | OracleSel | ObsRate | PosObs | NonposObs | vsAlwaysObs | vsOracle |
|---------|--------|-----------|-----------|-----------|---------|--------|-----------|-------------|----------|
| D0 | **FAIL** | 0.4200 | 0.4200 | 0.4300 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | 0.0100 |
| D1 | **FAIL** | 0.4200 | 0.4200 | 0.4300 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | 0.0100 |
| D2 | **FAIL** | 0.4200 | 0.4200 | 0.4300 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | 0.0100 |
| D3 | **FAIL** | 0.4200 | 0.4200 | 0.4300 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | 0.0100 |

**D0 checks:**
| policy_net > always_observe_net | **FAIL** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **FAIL** |
| pos_obs > nonpos_obs | **FAIL** |
Failures: ['policy_net > always_observe_net', 'obs_rate between 0 and 1', 'pos_obs > nonpos_obs']

**D1 checks:**
| policy_net > always_observe_net | **FAIL** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **FAIL** |
| pos_obs > nonpos_obs | **FAIL** |
Failures: ['policy_net > always_observe_net', 'obs_rate between 0 and 1', 'pos_obs > nonpos_obs']

**D2 checks:**
| policy_net > always_observe_net | **FAIL** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **FAIL** |
| pos_obs > nonpos_obs | **FAIL** |
Failures: ['policy_net > always_observe_net', 'obs_rate between 0 and 1', 'pos_obs > nonpos_obs']

**D3 checks:**
| policy_net > always_observe_net | **FAIL** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **FAIL** |
| pos_obs > nonpos_obs | **FAIL** |
Failures: ['policy_net > always_observe_net', 'obs_rate between 0 and 1', 'pos_obs > nonpos_obs']

### Test C: Held-out Seeds (LOSO)

| Variant | Status | PolicyNet | AlwaysObs | OracleSel | ObsRate | PosObs | NonposObs | vsAlwaysObs | vsOracle |
|---------|--------|-----------|-----------|-----------|---------|--------|-----------|-------------|----------|
| D0 | **PASS** | 0.4300 | 0.4200 | 0.4300 | 0.6667 | 1.0000 | 0.0000 | +0.0100 | 0.0000 |
| D1 | **PASS** | 0.4300 | 0.4200 | 0.4300 | 0.6667 | 1.0000 | 0.0000 | +0.0100 | 0.0000 |
| D2 | **PASS** | 0.4300 | 0.4200 | 0.4300 | 0.6667 | 1.0000 | 0.0000 | +0.0100 | 0.0000 |
| D3 | **PASS** | 0.4300 | 0.4200 | 0.4300 | 0.6667 | 1.0000 | 0.0000 | +0.0100 | 0.0000 |

**D0 checks:**
| policy_net > always_observe_net | **PASS** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **PASS** |
| pos_obs > nonpos_obs | **PASS** |

**D1 checks:**
| policy_net > always_observe_net | **PASS** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **PASS** |
| pos_obs > nonpos_obs | **PASS** |

**D2 checks:**
| policy_net > always_observe_net | **PASS** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **PASS** |
| pos_obs > nonpos_obs | **PASS** |

**D3 checks:**
| policy_net > always_observe_net | **PASS** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **PASS** |
| pos_obs > nonpos_obs | **PASS** |

## 6. Results at observe_cost=0.05

### Test A: Held-out Signatures

| Variant | Status | PolicyNet | AlwaysObs | OracleSel | ObsRate | PosObs | NonposObs | vsAlwaysObs | vsOracle |
|---------|--------|-----------|-----------|-----------|---------|--------|-----------|-------------|----------|
| D0 | **FAIL** | 0.4000 | 0.4000 | 0.4167 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | 0.0167 |
| D1 | **FAIL** | 0.4000 | 0.4000 | 0.4167 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | 0.0167 |
| D2 | **FAIL** | 0.4000 | 0.4000 | 0.4167 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | 0.0167 |
| D3 | **FAIL** | 0.4000 | 0.4000 | 0.4167 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | 0.0167 |

**D0 checks:**
| policy_net > always_observe_net | **FAIL** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **FAIL** |
| pos_obs > nonpos_obs | **FAIL** |
Failures: ['policy_net > always_observe_net', 'obs_rate between 0 and 1', 'pos_obs > nonpos_obs']

**D1 checks:**
| policy_net > always_observe_net | **FAIL** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **FAIL** |
| pos_obs > nonpos_obs | **FAIL** |
Failures: ['policy_net > always_observe_net', 'obs_rate between 0 and 1', 'pos_obs > nonpos_obs']

**D2 checks:**
| policy_net > always_observe_net | **FAIL** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **FAIL** |
| pos_obs > nonpos_obs | **FAIL** |
Failures: ['policy_net > always_observe_net', 'obs_rate between 0 and 1', 'pos_obs > nonpos_obs']

**D3 checks:**
| policy_net > always_observe_net | **FAIL** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **FAIL** |
| pos_obs > nonpos_obs | **FAIL** |
Failures: ['policy_net > always_observe_net', 'obs_rate between 0 and 1', 'pos_obs > nonpos_obs']

### Test B: Held-out Cue Combinations

| Variant | Status | PolicyNet | AlwaysObs | OracleSel | ObsRate | PosObs | NonposObs | vsAlwaysObs | vsOracle |
|---------|--------|-----------|-----------|-----------|---------|--------|-----------|-------------|----------|
| D0 | **FAIL** | 0.4000 | 0.4000 | 0.4167 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | 0.0167 |
| D1 | **FAIL** | 0.4000 | 0.4000 | 0.4167 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | 0.0167 |
| D2 | **FAIL** | 0.4000 | 0.4000 | 0.4167 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | 0.0167 |
| D3 | **FAIL** | 0.4000 | 0.4000 | 0.4167 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | 0.0167 |

**D0 checks:**
| policy_net > always_observe_net | **FAIL** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **FAIL** |
| pos_obs > nonpos_obs | **FAIL** |
Failures: ['policy_net > always_observe_net', 'obs_rate between 0 and 1', 'pos_obs > nonpos_obs']

**D1 checks:**
| policy_net > always_observe_net | **FAIL** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **FAIL** |
| pos_obs > nonpos_obs | **FAIL** |
Failures: ['policy_net > always_observe_net', 'obs_rate between 0 and 1', 'pos_obs > nonpos_obs']

**D2 checks:**
| policy_net > always_observe_net | **FAIL** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **FAIL** |
| pos_obs > nonpos_obs | **FAIL** |
Failures: ['policy_net > always_observe_net', 'obs_rate between 0 and 1', 'pos_obs > nonpos_obs']

**D3 checks:**
| policy_net > always_observe_net | **FAIL** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **FAIL** |
| pos_obs > nonpos_obs | **FAIL** |
Failures: ['policy_net > always_observe_net', 'obs_rate between 0 and 1', 'pos_obs > nonpos_obs']

### Test C: Held-out Seeds (LOSO)

| Variant | Status | PolicyNet | AlwaysObs | OracleSel | ObsRate | PosObs | NonposObs | vsAlwaysObs | vsOracle |
|---------|--------|-----------|-----------|-----------|---------|--------|-----------|-------------|----------|
| D0 | **PASS** | 0.4167 | 0.4000 | 0.4167 | 0.6667 | 1.0000 | 0.0000 | +0.0167 | 0.0000 |
| D1 | **PASS** | 0.4167 | 0.4000 | 0.4167 | 0.6667 | 1.0000 | 0.0000 | +0.0167 | 0.0000 |
| D2 | **PASS** | 0.4167 | 0.4000 | 0.4167 | 0.6667 | 1.0000 | 0.0000 | +0.0167 | 0.0000 |
| D3 | **PASS** | 0.4167 | 0.4000 | 0.4167 | 0.6667 | 1.0000 | 0.0000 | +0.0167 | 0.0000 |

**D0 checks:**
| policy_net > always_observe_net | **PASS** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **PASS** |
| pos_obs > nonpos_obs | **PASS** |

**D1 checks:**
| policy_net > always_observe_net | **PASS** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **PASS** |
| pos_obs > nonpos_obs | **PASS** |

**D2 checks:**
| policy_net > always_observe_net | **PASS** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **PASS** |
| pos_obs > nonpos_obs | **PASS** |

**D3 checks:**
| policy_net > always_observe_net | **PASS** |
| policy_net > always_try | **PASS** |
| obs_rate between 0 and 1 | **PASS** |
| pos_obs > nonpos_obs | **PASS** |

## 7. Q-Target vs True-Target Sanity Check

| Test | Variant | Q-Target Mean | True-Target Mean | Correlation |
|------|---------|---------------|------------------|-------------|
| A/A1 | D0 | N/A | N/A | 1.0 |
| A/A1 | D1 | N/A | N/A | 1.0 |
| A/A1 | D2 | N/A | N/A | 1.0 |
| A/A1 | D3 | N/A | N/A | 1.0 |
| B/B1 | D0 | N/A | N/A | 1.0 |
| B/B1 | D1 | N/A | N/A | 1.0 |
| B/B1 | D2 | N/A | N/A | 1.0 |
| B/B1 | D3 | N/A | N/A | 1.0 |
| C/C_seed101 | D0 | N/A | N/A | 1.0 |
| C/C_seed101 | D1 | N/A | N/A | 1.0 |
| C/C_seed101 | D2 | N/A | N/A | 1.0 |
| C/C_seed101 | D3 | N/A | N/A | 1.0 |

## 8. Decision-Time Information Boundary Audit

- Two-stage policy: Stage 1 uses only pre_vec + abstract (pre-computed).
- Stage 2 reveals post_observe_features only after observe chosen.
- Decision logs: post_best_action=None, post_best_value=0.0 when observe not chosen.
- Abstract features: no test post_vec, no audit labels at test time.

## 9. Overall

**FAIL**

| Test | D0 | D1 | D2 | D3 |
|------|----|----|----|----|
| Test A | FAIL | FAIL | FAIL | FAIL |
| Test B | FAIL | FAIL | FAIL | FAIL |
| Test C | PASS | PASS | PASS | PASS |

## 10. Key Findings

- D0 baseline reproduces 7c: Test A=FAIL, Test B=FAIL
- D1 Test A: FAIL, policy_net=0.4200, obs_rate=1.0000, vs_always_obs=+0.0000
- D1 Test B: FAIL, policy_net=0.4200, obs_rate=1.0000, vs_always_obs=+0.0000
- D2 Test A: FAIL, policy_net=0.4200, obs_rate=1.0000, vs_always_obs=+0.0000
- D2 Test B: FAIL, policy_net=0.4200, obs_rate=1.0000, vs_always_obs=+0.0000

```
[block_done]
block_id=1J40b-7d
implementation_status=FAIL
```
