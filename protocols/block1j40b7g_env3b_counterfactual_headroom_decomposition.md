# Block 1J40b-7g: Env3b / Counterfactual Headroom Decomposition

## 1. Checkpoint Summary

- **Status**: DIAGNOSTIC_COMPLETE
- **Elapsed**: 0.23s
- **Diagnostic Flags**: {'weak_oracle_headroom': True, 'always_observe_too_strong': True, 'nonpositive_penalty_too_weak': True, 'pre_cue_predictability_low': True, 'cf_cov1_destroyed_signal': True, 'level_estimate_sign_mismatch': True, 'fallback_problem': False, 'distribution_review_needed': True}

## 2. Files Changed

- `_block1j40b7g_env3b_counterfactual_headroom_decomposition.py`
- `runs/block1j40b7g_env3b_counterfactual_headroom_decomposition.json`
- `protocols/block1j40b7g_env3b_counterfactual_headroom_decomposition.md`
- `protocols/block1j40b7g_env3b_counterfactual_headroom_decomposition_table.csv`
- `checkpoint_1j40b7g_env3b_counterfactual_headroom_decomposition.md`

## 3. Commands Run

```powershell
& "D:\conda\python.exe" "_block1j40b7g_env3b_counterfactual_headroom_decomposition.py"
```

## 4. Why 7g Was Run

- 7f-b showed compositional non-global estimates can exist without beating always_observe.
- 7g decomposes whether failure is due to weak oracle headroom, strong always_observe, poor sign prediction, fallback, or coverage effects.

## 5. Headroom Decomposition

| Distribution | Cost | Test | always_try | always_observe | oracle_selective | oracle-always_obs | always_obs-always_try |
|--------------|------|------|------------|----------------|------------------|-------------------|-----------------------|
| counterfactual_cov0.25 | 0.01 | A | 0.2500 | 0.4400 | 0.4433 | +0.0033 | +0.1900 |
| counterfactual_cov0.25 | 0.01 | B | 0.2500 | 0.4400 | 0.4433 | +0.0033 | +0.1900 |
| counterfactual_cov0.25 | 0.01 | C | 0.2500 | 0.4400 | 0.4433 | +0.0033 | +0.1900 |
| counterfactual_cov0.25 | 0.03 | A | 0.2500 | 0.4200 | 0.4300 | +0.0100 | +0.1700 |
| counterfactual_cov0.25 | 0.03 | B | 0.2500 | 0.4200 | 0.4300 | +0.0100 | +0.1700 |
| counterfactual_cov0.25 | 0.03 | C | 0.2500 | 0.4200 | 0.4300 | +0.0100 | +0.1700 |
| counterfactual_cov0.25 | 0.05 | A | 0.2500 | 0.4000 | 0.4167 | +0.0167 | +0.1500 |
| counterfactual_cov0.25 | 0.05 | B | 0.2500 | 0.4000 | 0.4167 | +0.0167 | +0.1500 |
| counterfactual_cov0.25 | 0.05 | C | 0.2500 | 0.4000 | 0.4167 | +0.0167 | +0.1500 |
| counterfactual_cov0.25 | 0.07 | A | 0.2500 | 0.3800 | 0.4033 | +0.0233 | +0.1300 |
| counterfactual_cov0.25 | 0.07 | B | 0.2500 | 0.3800 | 0.4033 | +0.0233 | +0.1300 |
| counterfactual_cov0.25 | 0.07 | C | 0.2500 | 0.3800 | 0.4033 | +0.0233 | +0.1300 |
| counterfactual_cov0.50 | 0.01 | A | 0.2500 | 0.4400 | 0.4433 | +0.0033 | +0.1900 |
| counterfactual_cov0.50 | 0.01 | B | 0.2500 | 0.4400 | 0.4433 | +0.0033 | +0.1900 |
| counterfactual_cov0.50 | 0.01 | C | 0.2500 | 0.4400 | 0.4433 | +0.0033 | +0.1900 |
| counterfactual_cov0.50 | 0.03 | A | 0.2500 | 0.4200 | 0.4300 | +0.0100 | +0.1700 |
| counterfactual_cov0.50 | 0.03 | B | 0.2500 | 0.4200 | 0.4300 | +0.0100 | +0.1700 |
| counterfactual_cov0.50 | 0.03 | C | 0.2500 | 0.4200 | 0.4300 | +0.0100 | +0.1700 |
| counterfactual_cov0.50 | 0.05 | A | 0.2500 | 0.4000 | 0.4167 | +0.0167 | +0.1500 |
| counterfactual_cov0.50 | 0.05 | B | 0.2500 | 0.4000 | 0.4167 | +0.0167 | +0.1500 |
| counterfactual_cov0.50 | 0.05 | C | 0.2500 | 0.4000 | 0.4167 | +0.0167 | +0.1500 |
| counterfactual_cov0.50 | 0.07 | A | 0.2500 | 0.3800 | 0.4033 | +0.0233 | +0.1300 |
| counterfactual_cov0.50 | 0.07 | B | 0.2500 | 0.3800 | 0.4033 | +0.0233 | +0.1300 |
| counterfactual_cov0.50 | 0.07 | C | 0.2500 | 0.3800 | 0.4033 | +0.0233 | +0.1300 |
| counterfactual_cov1.00 | 0.01 | A | 0.2500 | 0.4400 | 0.4433 | +0.0033 | +0.1900 |
| counterfactual_cov1.00 | 0.01 | B | 0.2500 | 0.4400 | 0.4433 | +0.0033 | +0.1900 |
| counterfactual_cov1.00 | 0.01 | C | 0.2500 | 0.4400 | 0.4433 | +0.0033 | +0.1900 |
| counterfactual_cov1.00 | 0.03 | A | 0.2500 | 0.4200 | 0.4300 | +0.0100 | +0.1700 |
| counterfactual_cov1.00 | 0.03 | B | 0.2500 | 0.4200 | 0.4300 | +0.0100 | +0.1700 |
| counterfactual_cov1.00 | 0.03 | C | 0.2500 | 0.4200 | 0.4300 | +0.0100 | +0.1700 |
| counterfactual_cov1.00 | 0.05 | A | 0.2500 | 0.4000 | 0.4167 | +0.0167 | +0.1500 |
| counterfactual_cov1.00 | 0.05 | B | 0.2500 | 0.4000 | 0.4167 | +0.0167 | +0.1500 |
| counterfactual_cov1.00 | 0.05 | C | 0.2500 | 0.4000 | 0.4167 | +0.0167 | +0.1500 |
| counterfactual_cov1.00 | 0.07 | A | 0.2500 | 0.3800 | 0.4033 | +0.0233 | +0.1300 |
| counterfactual_cov1.00 | 0.07 | B | 0.2500 | 0.3800 | 0.4033 | +0.0233 | +0.1300 |
| counterfactual_cov1.00 | 0.07 | C | 0.2500 | 0.3800 | 0.4033 | +0.0233 | +0.1300 |
| original | 0.01 | A | 0.2500 | 0.4400 | 0.4433 | +0.0033 | +0.1900 |
| original | 0.01 | B | 0.2500 | 0.4400 | 0.4433 | +0.0033 | +0.1900 |
| original | 0.01 | C | 0.2500 | 0.4400 | 0.4433 | +0.0033 | +0.1900 |
| original | 0.03 | A | 0.2500 | 0.4200 | 0.4300 | +0.0100 | +0.1700 |
| original | 0.03 | B | 0.2500 | 0.4200 | 0.4300 | +0.0100 | +0.1700 |
| original | 0.03 | C | 0.2500 | 0.4200 | 0.4300 | +0.0100 | +0.1700 |
| original | 0.05 | A | 0.2500 | 0.4000 | 0.4167 | +0.0167 | +0.1500 |
| original | 0.05 | B | 0.2500 | 0.4000 | 0.4167 | +0.0167 | +0.1500 |
| original | 0.05 | C | 0.2500 | 0.4000 | 0.4167 | +0.0167 | +0.1500 |
| original | 0.07 | A | 0.2500 | 0.3800 | 0.4033 | +0.0233 | +0.1300 |
| original | 0.07 | B | 0.2500 | 0.3800 | 0.4033 | +0.0233 | +0.1300 |
| original | 0.07 | C | 0.2500 | 0.3800 | 0.4033 | +0.0233 | +0.1300 |

## 6. Positive / Non-Positive Observe Value Distribution

| Distribution | Cost | Test | positive_rate | mean_true_E | mean_positive | mean_nonpositive | min_true_E | max_true_E |
|--------------|------|------|---------------|-------------|---------------|------------------|------------|------------|
| counterfactual_cov0.25 | 0.01 | A | 0.6667 | 0.1400 | 0.2900 | -0.0100 | -0.0100 | 0.2900 |
| counterfactual_cov0.25 | 0.01 | B | 0.6667 | 0.1400 | 0.2900 | -0.0100 | -0.0100 | 0.2900 |
| counterfactual_cov0.25 | 0.01 | C | 0.6667 | 0.1400 | 0.2900 | -0.0100 | -0.0100 | 0.2900 |
| counterfactual_cov0.25 | 0.03 | A | 0.6667 | 0.1200 | 0.2700 | -0.0300 | -0.0300 | 0.2700 |
| counterfactual_cov0.25 | 0.03 | B | 0.6667 | 0.1200 | 0.2700 | -0.0300 | -0.0300 | 0.2700 |
| counterfactual_cov0.25 | 0.03 | C | 0.6667 | 0.1200 | 0.2700 | -0.0300 | -0.0300 | 0.2700 |
| counterfactual_cov0.25 | 0.05 | A | 0.6667 | 0.1000 | 0.2500 | -0.0500 | -0.0500 | 0.2500 |
| counterfactual_cov0.25 | 0.05 | B | 0.6667 | 0.1000 | 0.2500 | -0.0500 | -0.0500 | 0.2500 |
| counterfactual_cov0.25 | 0.05 | C | 0.6667 | 0.1000 | 0.2500 | -0.0500 | -0.0500 | 0.2500 |
| counterfactual_cov0.25 | 0.07 | A | 0.6667 | 0.0800 | 0.2300 | -0.0700 | -0.0700 | 0.2300 |
| counterfactual_cov0.25 | 0.07 | B | 0.6667 | 0.0800 | 0.2300 | -0.0700 | -0.0700 | 0.2300 |
| counterfactual_cov0.25 | 0.07 | C | 0.6667 | 0.0800 | 0.2300 | -0.0700 | -0.0700 | 0.2300 |
| counterfactual_cov0.50 | 0.01 | A | 0.6667 | 0.1400 | 0.2900 | -0.0100 | -0.0100 | 0.2900 |
| counterfactual_cov0.50 | 0.01 | B | 0.6667 | 0.1400 | 0.2900 | -0.0100 | -0.0100 | 0.2900 |
| counterfactual_cov0.50 | 0.01 | C | 0.6667 | 0.1400 | 0.2900 | -0.0100 | -0.0100 | 0.2900 |
| counterfactual_cov0.50 | 0.03 | A | 0.6667 | 0.1200 | 0.2700 | -0.0300 | -0.0300 | 0.2700 |
| counterfactual_cov0.50 | 0.03 | B | 0.6667 | 0.1200 | 0.2700 | -0.0300 | -0.0300 | 0.2700 |
| counterfactual_cov0.50 | 0.03 | C | 0.6667 | 0.1200 | 0.2700 | -0.0300 | -0.0300 | 0.2700 |
| counterfactual_cov0.50 | 0.05 | A | 0.6667 | 0.1000 | 0.2500 | -0.0500 | -0.0500 | 0.2500 |
| counterfactual_cov0.50 | 0.05 | B | 0.6667 | 0.1000 | 0.2500 | -0.0500 | -0.0500 | 0.2500 |
| counterfactual_cov0.50 | 0.05 | C | 0.6667 | 0.1000 | 0.2500 | -0.0500 | -0.0500 | 0.2500 |
| counterfactual_cov0.50 | 0.07 | A | 0.6667 | 0.0800 | 0.2300 | -0.0700 | -0.0700 | 0.2300 |
| counterfactual_cov0.50 | 0.07 | B | 0.6667 | 0.0800 | 0.2300 | -0.0700 | -0.0700 | 0.2300 |
| counterfactual_cov0.50 | 0.07 | C | 0.6667 | 0.0800 | 0.2300 | -0.0700 | -0.0700 | 0.2300 |
| counterfactual_cov1.00 | 0.01 | A | 0.6667 | 0.1400 | 0.2900 | -0.0100 | -0.0100 | 0.2900 |
| counterfactual_cov1.00 | 0.01 | B | 0.6667 | 0.1400 | 0.2900 | -0.0100 | -0.0100 | 0.2900 |
| counterfactual_cov1.00 | 0.01 | C | 0.6667 | 0.1400 | 0.2900 | -0.0100 | -0.0100 | 0.2900 |
| counterfactual_cov1.00 | 0.03 | A | 0.6667 | 0.1200 | 0.2700 | -0.0300 | -0.0300 | 0.2700 |
| counterfactual_cov1.00 | 0.03 | B | 0.6667 | 0.1200 | 0.2700 | -0.0300 | -0.0300 | 0.2700 |
| counterfactual_cov1.00 | 0.03 | C | 0.6667 | 0.1200 | 0.2700 | -0.0300 | -0.0300 | 0.2700 |
| counterfactual_cov1.00 | 0.05 | A | 0.6667 | 0.1000 | 0.2500 | -0.0500 | -0.0500 | 0.2500 |
| counterfactual_cov1.00 | 0.05 | B | 0.6667 | 0.1000 | 0.2500 | -0.0500 | -0.0500 | 0.2500 |
| counterfactual_cov1.00 | 0.05 | C | 0.6667 | 0.1000 | 0.2500 | -0.0500 | -0.0500 | 0.2500 |
| counterfactual_cov1.00 | 0.07 | A | 0.6667 | 0.0800 | 0.2300 | -0.0700 | -0.0700 | 0.2300 |
| counterfactual_cov1.00 | 0.07 | B | 0.6667 | 0.0800 | 0.2300 | -0.0700 | -0.0700 | 0.2300 |
| counterfactual_cov1.00 | 0.07 | C | 0.6667 | 0.0800 | 0.2300 | -0.0700 | -0.0700 | 0.2300 |
| original | 0.01 | A | 0.6667 | 0.1400 | 0.2900 | -0.0100 | -0.0100 | 0.2900 |
| original | 0.01 | B | 0.6667 | 0.1400 | 0.2900 | -0.0100 | -0.0100 | 0.2900 |
| original | 0.01 | C | 0.6667 | 0.1400 | 0.2900 | -0.0100 | -0.0100 | 0.2900 |
| original | 0.03 | A | 0.6667 | 0.1200 | 0.2700 | -0.0300 | -0.0300 | 0.2700 |
| original | 0.03 | B | 0.6667 | 0.1200 | 0.2700 | -0.0300 | -0.0300 | 0.2700 |
| original | 0.03 | C | 0.6667 | 0.1200 | 0.2700 | -0.0300 | -0.0300 | 0.2700 |
| original | 0.05 | A | 0.6667 | 0.1000 | 0.2500 | -0.0500 | -0.0500 | 0.2500 |
| original | 0.05 | B | 0.6667 | 0.1000 | 0.2500 | -0.0500 | -0.0500 | 0.2500 |
| original | 0.05 | C | 0.6667 | 0.1000 | 0.2500 | -0.0500 | -0.0500 | 0.2500 |
| original | 0.07 | A | 0.6667 | 0.0800 | 0.2300 | -0.0700 | -0.0700 | 0.2300 |
| original | 0.07 | B | 0.6667 | 0.0800 | 0.2300 | -0.0700 | -0.0700 | 0.2300 |
| original | 0.07 | C | 0.6667 | 0.0800 | 0.2300 | -0.0700 | -0.0700 | 0.2300 |

## 7. Always-Observe Strength Source

### counterfactual_cov0.25|0.03|A

| Signature | n_objects | always_obs | oracle_sel | no_observe | contrib_always_obs | contrib_oracle_gap |
|-----------|-----------|------------|------------|------------|--------------------|--------------------|
| amb(brownish,heavy_weight)+discolored,has_bark_texture | 15 | 0.4200 | 0.4500 | 0.4500 | 0.0350 | +0.0025 |
| amb(brownish,heavy_weight)_only | 30 | 0.4200 | 0.4200 | 0.1500 | 0.0700 | +0.0000 |
| amb(brownish,heavy_weight)+has_crystal_flecks | 15 | 0.4200 | 0.4500 | 0.4500 | 0.0350 | +0.0025 |
| amb(brownish,heavy_weight)+spotted | 30 | 0.4200 | 0.4200 | 0.1500 | 0.0700 | +0.0000 |
| amb(greenish,light_weight)+discolored | 30 | 0.4200 | 0.4200 | 0.1500 | 0.0700 | +0.0000 |
| amb(greenish,light_weight)+has_grip_area | 15 | 0.4200 | 0.4500 | 0.4500 | 0.0350 | +0.0025 |
| amb(greenish,light_weight)+has_stem_remnant,spotted | 15 | 0.4200 | 0.4500 | 0.4500 | 0.0350 | +0.0025 |
| amb(greenish,light_weight)_only | 30 | 0.4200 | 0.4200 | 0.1500 | 0.0700 | +0.0000 |

### counterfactual_cov0.25|0.03|B

| Signature | n_objects | always_obs | oracle_sel | no_observe | contrib_always_obs | contrib_oracle_gap |
|-----------|-----------|------------|------------|------------|--------------------|--------------------|
| amb(greenish,light_weight)+has_stem_remnant,spotted | 15 | 0.4200 | 0.4500 | 0.4500 | 0.0700 | +0.0050 |
| amb(greenish,light_weight)_only | 30 | 0.4200 | 0.4200 | 0.1500 | 0.1400 | +0.0000 |
| amb(brownish,heavy_weight)+discolored,has_bark_texture | 15 | 0.4200 | 0.4500 | 0.4500 | 0.0700 | +0.0050 |
| amb(brownish,heavy_weight)_only | 30 | 0.4200 | 0.4200 | 0.1500 | 0.1400 | +0.0000 |

## 8. Level 1/2/3 Error Decomposition

| Distribution | Cost | Test | Level | sign_correct | false_observe | false_direct | mean_loss_false_observe | mean_loss_false_direct | fallback_rate | non_global_rate |
|--------------|------|------|-------|--------------|---------------|--------------|-------------------------|------------------------|---------------|-----------------|
| counterfactual_cov0.25 | 0.01 | A | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.01 | A | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.01 | A | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.01 | B | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.01 | B | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.01 | B | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.01 | C | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.01 | C | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.01 | C | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.03 | A | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.03 | A | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.03 | A | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.03 | B | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.03 | B | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.03 | B | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.03 | C | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.03 | C | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.03 | C | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.05 | A | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.05 | A | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.05 | A | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.05 | B | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.05 | B | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.05 | B | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.05 | C | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.05 | C | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.05 | C | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.07 | A | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.07 | A | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.07 | A | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.07 | B | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.07 | B | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.07 | B | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.07 | C | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.07 | C | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.25 | 0.07 | C | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.01 | A | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.01 | A | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.01 | A | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.01 | B | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.01 | B | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.01 | B | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.01 | C | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.01 | C | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.01 | C | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.03 | A | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.03 | A | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.03 | A | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.03 | B | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.03 | B | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.03 | B | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.03 | C | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.03 | C | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.03 | C | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.05 | A | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.05 | A | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.05 | A | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.05 | B | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.05 | B | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.05 | B | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.05 | C | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.05 | C | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.05 | C | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.07 | A | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.07 | A | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.07 | A | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.07 | B | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.07 | B | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.07 | B | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.07 | C | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.07 | C | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov0.50 | 0.07 | C | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.01 | A | level1_cue_marginal | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2900 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.01 | A | level2_knn_jaccard_k3 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2900 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.01 | A | level3_candidate_set_tau0.5 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2900 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.01 | B | level1_cue_marginal | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2900 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.01 | B | level2_knn_jaccard_k3 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2900 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.01 | B | level3_candidate_set_tau0.5 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2900 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.01 | C | level1_cue_marginal | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2900 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.01 | C | level2_knn_jaccard_k3 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2900 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.01 | C | level3_candidate_set_tau0.5 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2900 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.03 | A | level1_cue_marginal | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2700 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.03 | A | level2_knn_jaccard_k3 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2700 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.03 | A | level3_candidate_set_tau0.5 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2700 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.03 | B | level1_cue_marginal | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2700 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.03 | B | level2_knn_jaccard_k3 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2700 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.03 | B | level3_candidate_set_tau0.5 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2700 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.03 | C | level1_cue_marginal | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2700 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.03 | C | level2_knn_jaccard_k3 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2700 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.03 | C | level3_candidate_set_tau0.5 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2700 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.05 | A | level1_cue_marginal | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2500 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.05 | A | level2_knn_jaccard_k3 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2500 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.05 | A | level3_candidate_set_tau0.5 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2500 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.05 | B | level1_cue_marginal | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2500 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.05 | B | level2_knn_jaccard_k3 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2500 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.05 | B | level3_candidate_set_tau0.5 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2500 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.05 | C | level1_cue_marginal | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2500 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.05 | C | level2_knn_jaccard_k3 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2500 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.05 | C | level3_candidate_set_tau0.5 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2500 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.07 | A | level1_cue_marginal | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2300 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.07 | A | level2_knn_jaccard_k3 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2300 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.07 | A | level3_candidate_set_tau0.5 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2300 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.07 | B | level1_cue_marginal | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2300 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.07 | B | level2_knn_jaccard_k3 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2300 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.07 | B | level3_candidate_set_tau0.5 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2300 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.07 | C | level1_cue_marginal | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2300 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.07 | C | level2_knn_jaccard_k3 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2300 | 0.0000 | 1.0000 |
| counterfactual_cov1.00 | 0.07 | C | level3_candidate_set_tau0.5 | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.2300 | 0.0000 | 1.0000 |
| original | 0.01 | A | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.01 | A | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.01 | A | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.01 | B | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.01 | B | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.01 | B | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.01 | C | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.01 | C | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.01 | C | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0100 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.03 | A | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.03 | A | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.03 | A | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.03 | B | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.03 | B | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.03 | B | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.03 | C | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.03 | C | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.03 | C | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0300 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.05 | A | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.05 | A | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.05 | A | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.05 | B | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.05 | B | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.05 | B | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.05 | C | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.05 | C | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.05 | C | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.07 | A | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.07 | A | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.07 | A | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.07 | B | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.07 | B | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.07 | B | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.07 | C | level1_cue_marginal | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.07 | C | level2_knn_jaccard_k3 | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |
| original | 0.07 | C | level3_candidate_set_tau0.5 | 0.6667 | 0.3333 | 0.0000 | 0.0700 | 0.0000 | 0.0000 | 1.0000 |

## 9. Pre-Cue Predictability Audit

| Distribution | Cost | Test | Level | pearson | spearman | sign_agreement | estE_pos | estE_nonpos | separation |
|--------------|------|------|-------|---------|----------|----------------|----------|-------------|------------|
| counterfactual_cov0.25 | 0.01 | A | level1_cue_marginal | -0.7365 | -0.7178 | 0.5000 | 0.1667 | 0.1773 | -0.0107 |
| counterfactual_cov0.25 | 0.01 | A | level2_knn_jaccard_k3 | -0.3456 | -0.3293 | 0.5000 | 0.1597 | 0.1639 | -0.0042 |
| counterfactual_cov0.25 | 0.01 | A | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1733 | 0.1733 | 0.0000 |
| counterfactual_cov0.25 | 0.01 | B | level1_cue_marginal | -0.5523 | -0.4472 | 0.5000 | 0.1700 | 0.1780 | -0.0080 |
| counterfactual_cov0.25 | 0.01 | B | level2_knn_jaccard_k3 | -0.4184 | -0.4472 | 0.5000 | 0.1550 | 0.1594 | -0.0044 |
| counterfactual_cov0.25 | 0.01 | B | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1700 | 0.1700 | 0.0000 |
| counterfactual_cov0.25 | 0.01 | C | level1_cue_marginal | 0.3625 | 0.4085 | 0.5000 | 0.1737 | 0.1676 | 0.0062 |
| counterfactual_cov0.25 | 0.01 | C | level2_knn_jaccard_k3 | 0.4331 | 0.4205 | 0.5000 | 0.1764 | 0.1678 | 0.0086 |
| counterfactual_cov0.25 | 0.01 | C | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1742 | 0.1742 | 0.0000 |
| counterfactual_cov0.25 | 0.03 | A | level1_cue_marginal | -0.7365 | -0.7730 | 0.5000 | 0.1467 | 0.1573 | -0.0107 |
| counterfactual_cov0.25 | 0.03 | A | level2_knn_jaccard_k3 | -0.3456 | -0.3313 | 0.5000 | 0.1397 | 0.1439 | -0.0042 |
| counterfactual_cov0.25 | 0.03 | A | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1533 | 0.1533 | 0.0000 |
| counterfactual_cov0.25 | 0.03 | B | level1_cue_marginal | -0.5523 | -0.4472 | 0.5000 | 0.1500 | 0.1580 | -0.0080 |
| counterfactual_cov0.25 | 0.03 | B | level2_knn_jaccard_k3 | -0.4184 | -0.4472 | 0.5000 | 0.1350 | 0.1394 | -0.0044 |
| counterfactual_cov0.25 | 0.03 | B | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1500 | 0.1500 | 0.0000 |
| counterfactual_cov0.25 | 0.03 | C | level1_cue_marginal | 0.3625 | 0.3839 | 0.5000 | 0.1537 | 0.1476 | 0.0062 |
| counterfactual_cov0.25 | 0.03 | C | level2_knn_jaccard_k3 | 0.4331 | 0.4183 | 0.5000 | 0.1564 | 0.1478 | 0.0086 |
| counterfactual_cov0.25 | 0.03 | C | level3_candidate_set_tau0.5 | -0.0000 | 0.0000 | 0.5000 | 0.1542 | 0.1542 | 0.0000 |
| counterfactual_cov0.25 | 0.05 | A | level1_cue_marginal | -0.7365 | -0.7178 | 0.5000 | 0.1267 | 0.1373 | -0.0107 |
| counterfactual_cov0.25 | 0.05 | A | level2_knn_jaccard_k3 | -0.3456 | -0.3273 | 0.5000 | 0.1197 | 0.1239 | -0.0042 |
| counterfactual_cov0.25 | 0.05 | A | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1333 | 0.1333 | 0.0000 |
| counterfactual_cov0.25 | 0.05 | B | level1_cue_marginal | -0.5523 | -0.4472 | 0.5000 | 0.1300 | 0.1380 | -0.0080 |
| counterfactual_cov0.25 | 0.05 | B | level2_knn_jaccard_k3 | -0.4184 | -0.4472 | 0.5000 | 0.1150 | 0.1194 | -0.0044 |
| counterfactual_cov0.25 | 0.05 | B | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1300 | 0.1300 | 0.0000 |
| counterfactual_cov0.25 | 0.05 | C | level1_cue_marginal | 0.3625 | 0.4317 | 0.5000 | 0.1337 | 0.1276 | 0.0062 |
| counterfactual_cov0.25 | 0.05 | C | level2_knn_jaccard_k3 | 0.4331 | 0.4204 | 0.5000 | 0.1364 | 0.1278 | 0.0086 |
| counterfactual_cov0.25 | 0.05 | C | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1342 | 0.1342 | 0.0000 |
| counterfactual_cov0.25 | 0.07 | A | level1_cue_marginal | -0.7365 | -0.7730 | 0.5000 | 0.1067 | 0.1173 | -0.0107 |
| counterfactual_cov0.25 | 0.07 | A | level2_knn_jaccard_k3 | -0.3456 | -0.3293 | 0.5000 | 0.0997 | 0.1039 | -0.0042 |
| counterfactual_cov0.25 | 0.07 | A | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1133 | 0.1133 | 0.0000 |
| counterfactual_cov0.25 | 0.07 | B | level1_cue_marginal | -0.5523 | -0.4472 | 0.5000 | 0.1100 | 0.1180 | -0.0080 |
| counterfactual_cov0.25 | 0.07 | B | level2_knn_jaccard_k3 | -0.4184 | -0.4472 | 0.5000 | 0.0950 | 0.0994 | -0.0044 |
| counterfactual_cov0.25 | 0.07 | B | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1100 | 0.1100 | 0.0000 |
| counterfactual_cov0.25 | 0.07 | C | level1_cue_marginal | 0.3625 | 0.4298 | 0.5000 | 0.1137 | 0.1076 | 0.0062 |
| counterfactual_cov0.25 | 0.07 | C | level2_knn_jaccard_k3 | 0.4331 | 0.4181 | 0.5000 | 0.1164 | 0.1078 | 0.0086 |
| counterfactual_cov0.25 | 0.07 | C | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1142 | 0.1142 | 0.0000 |
| counterfactual_cov0.50 | 0.01 | A | level1_cue_marginal | 0.9244 | 0.8835 | 0.5000 | 0.1807 | 0.1653 | 0.0153 |
| counterfactual_cov0.50 | 0.01 | A | level2_knn_jaccard_k3 | 0.3262 | 0.4444 | 0.5000 | 0.1915 | 0.1855 | 0.0060 |
| counterfactual_cov0.50 | 0.01 | A | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1700 | 0.1700 | 0.0000 |
| counterfactual_cov0.50 | 0.01 | B | level1_cue_marginal | 1.0000 | 0.8944 | 0.5000 | 0.1767 | 0.1673 | 0.0093 |
| counterfactual_cov0.50 | 0.01 | B | level2_knn_jaccard_k3 | 1.0000 | 0.9428 | 0.5000 | 0.2000 | 0.1944 | 0.0056 |
| counterfactual_cov0.50 | 0.01 | B | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1767 | 0.1767 | 0.0000 |
| counterfactual_cov0.50 | 0.01 | C | level1_cue_marginal | -0.8528 | -0.8686 | 0.5000 | 0.1752 | 0.1962 | -0.0211 |
| counterfactual_cov0.50 | 0.01 | C | level2_knn_jaccard_k3 | -0.6942 | -0.8233 | 0.5000 | 0.1674 | 0.2014 | -0.0341 |
| counterfactual_cov0.50 | 0.01 | C | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1750 | 0.1750 | 0.0000 |
| counterfactual_cov0.50 | 0.03 | A | level1_cue_marginal | 0.9244 | 0.8781 | 0.5000 | 0.1607 | 0.1453 | 0.0153 |
| counterfactual_cov0.50 | 0.03 | A | level2_knn_jaccard_k3 | 0.3262 | 0.4391 | 0.5000 | 0.1715 | 0.1655 | 0.0060 |
| counterfactual_cov0.50 | 0.03 | A | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1500 | 0.1500 | 0.0000 |
| counterfactual_cov0.50 | 0.03 | B | level1_cue_marginal | 1.0000 | 0.8944 | 0.5000 | 0.1567 | 0.1473 | 0.0093 |
| counterfactual_cov0.50 | 0.03 | B | level2_knn_jaccard_k3 | 1.0000 | 0.8944 | 0.5000 | 0.1800 | 0.1744 | 0.0056 |
| counterfactual_cov0.50 | 0.03 | B | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1567 | 0.1567 | 0.0000 |
| counterfactual_cov0.50 | 0.03 | C | level1_cue_marginal | -0.8528 | -0.8629 | 0.5000 | 0.1552 | 0.1762 | -0.0211 |
| counterfactual_cov0.50 | 0.03 | C | level2_knn_jaccard_k3 | -0.6942 | -0.8233 | 0.5000 | 0.1474 | 0.1814 | -0.0341 |
| counterfactual_cov0.50 | 0.03 | C | level3_candidate_set_tau0.5 | -0.0000 | 0.0000 | 0.5000 | 0.1550 | 0.1550 | 0.0000 |
| counterfactual_cov0.50 | 0.05 | A | level1_cue_marginal | 0.9244 | 0.8729 | 0.5000 | 0.1407 | 0.1253 | 0.0153 |
| counterfactual_cov0.50 | 0.05 | A | level2_knn_jaccard_k3 | 0.3262 | 0.4391 | 0.5000 | 0.1515 | 0.1455 | 0.0060 |
| counterfactual_cov0.50 | 0.05 | A | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1300 | 0.1300 | 0.0000 |
| counterfactual_cov0.50 | 0.05 | B | level1_cue_marginal | 1.0000 | 0.8944 | 0.5000 | 0.1367 | 0.1273 | 0.0093 |
| counterfactual_cov0.50 | 0.05 | B | level2_knn_jaccard_k3 | 1.0000 | 0.9428 | 0.5000 | 0.1600 | 0.1544 | 0.0056 |
| counterfactual_cov0.50 | 0.05 | B | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1367 | 0.1367 | 0.0000 |
| counterfactual_cov0.50 | 0.05 | C | level1_cue_marginal | -0.8528 | -0.8678 | 0.5000 | 0.1352 | 0.1562 | -0.0211 |
| counterfactual_cov0.50 | 0.05 | C | level2_knn_jaccard_k3 | -0.6942 | -0.8233 | 0.5000 | 0.1274 | 0.1614 | -0.0341 |
| counterfactual_cov0.50 | 0.05 | C | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1350 | 0.1350 | 0.0000 |
| counterfactual_cov0.50 | 0.07 | A | level1_cue_marginal | 0.9244 | 0.8729 | 0.5000 | 0.1207 | 0.1053 | 0.0153 |
| counterfactual_cov0.50 | 0.07 | A | level2_knn_jaccard_k3 | 0.3262 | 0.4391 | 0.5000 | 0.1315 | 0.1255 | 0.0060 |
| counterfactual_cov0.50 | 0.07 | A | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1100 | 0.1100 | 0.0000 |
| counterfactual_cov0.50 | 0.07 | B | level1_cue_marginal | 1.0000 | 0.8944 | 0.5000 | 0.1167 | 0.1073 | 0.0093 |
| counterfactual_cov0.50 | 0.07 | B | level2_knn_jaccard_k3 | 1.0000 | 0.8944 | 0.5000 | 0.1400 | 0.1344 | 0.0056 |
| counterfactual_cov0.50 | 0.07 | B | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1167 | 0.1167 | 0.0000 |
| counterfactual_cov0.50 | 0.07 | C | level1_cue_marginal | -0.8528 | -0.8591 | 0.5000 | 0.1152 | 0.1362 | -0.0211 |
| counterfactual_cov0.50 | 0.07 | C | level2_knn_jaccard_k3 | -0.6942 | -0.8232 | 0.5000 | 0.1074 | 0.1414 | -0.0341 |
| counterfactual_cov0.50 | 0.07 | C | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1150 | 0.1150 | 0.0000 |
| counterfactual_cov1.00 | 0.01 | A | level1_cue_marginal | 0.0000 | -0.3842 | 0.5000 | -0.0100 | -0.0100 | -0.0000 |
| counterfactual_cov1.00 | 0.01 | A | level2_knn_jaccard_k3 | 0.0000 | -0.5095 | 0.5000 | -0.0100 | -0.0100 | -0.0000 |
| counterfactual_cov1.00 | 0.01 | A | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | -0.0100 | -0.0100 | 0.0000 |
| counterfactual_cov1.00 | 0.01 | B | level1_cue_marginal | 0.0000 | -0.8944 | 0.5000 | -0.0100 | -0.0100 | -0.0000 |
| counterfactual_cov1.00 | 0.01 | B | level2_knn_jaccard_k3 | 0.0000 | -0.9428 | 0.5000 | -0.0100 | -0.0100 | -0.0000 |
| counterfactual_cov1.00 | 0.01 | B | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | -0.0100 | -0.0100 | 0.0000 |
| counterfactual_cov1.00 | 0.01 | C | level1_cue_marginal | 0.0000 | 0.6626 | 0.5000 | -0.0100 | -0.0100 | 0.0000 |
| counterfactual_cov1.00 | 0.01 | C | level2_knn_jaccard_k3 | 0.0000 | 0.7178 | 0.5000 | -0.0100 | -0.0100 | 0.0000 |
| counterfactual_cov1.00 | 0.01 | C | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | -0.0100 | -0.0100 | 0.0000 |
| counterfactual_cov1.00 | 0.03 | A | level1_cue_marginal | 0.0000 | -0.3313 | 0.5000 | -0.0300 | -0.0300 | -0.0000 |
| counterfactual_cov1.00 | 0.03 | A | level2_knn_jaccard_k3 | 0.0000 | -0.2236 | 0.5000 | -0.0300 | -0.0300 | -0.0000 |
| counterfactual_cov1.00 | 0.03 | A | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | -0.0300 | -0.0300 | 0.0000 |
| counterfactual_cov1.00 | 0.03 | B | level1_cue_marginal | 0.0000 | -0.4472 | 0.5000 | -0.0300 | -0.0300 | -0.0000 |
| counterfactual_cov1.00 | 0.03 | B | level2_knn_jaccard_k3 | 0.0000 | -0.8944 | 0.5000 | -0.0300 | -0.0300 | -0.0000 |
| counterfactual_cov1.00 | 0.03 | B | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | -0.0300 | -0.0300 | 0.0000 |
| counterfactual_cov1.00 | 0.03 | C | level1_cue_marginal | 0.0000 | 0.2761 | 0.5000 | -0.0300 | -0.0300 | 0.0000 |
| counterfactual_cov1.00 | 0.03 | C | level2_knn_jaccard_k3 | 0.0000 | 0.4969 | 0.5000 | -0.0300 | -0.0300 | 0.0000 |
| counterfactual_cov1.00 | 0.03 | C | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | -0.0300 | -0.0300 | 0.0000 |
| counterfactual_cov1.00 | 0.05 | A | level1_cue_marginal | 0.0000 | -0.3487 | 0.5000 | -0.0500 | -0.0500 | -0.0000 |
| counterfactual_cov1.00 | 0.05 | A | level2_knn_jaccard_k3 | 0.0000 | -0.2250 | 0.5000 | -0.0500 | -0.0500 | 0.0000 |
| counterfactual_cov1.00 | 0.05 | A | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | -0.0500 | -0.0500 | 0.0000 |
| counterfactual_cov1.00 | 0.05 | B | level1_cue_marginal | 0.0000 | -0.5774 | 0.5000 | -0.0500 | -0.0500 | -0.0000 |
| counterfactual_cov1.00 | 0.05 | B | level2_knn_jaccard_k3 | 0.0000 | -1.0000 | 0.5000 | -0.0500 | -0.0500 | -0.0000 |
| counterfactual_cov1.00 | 0.05 | B | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | -0.0500 | -0.0500 | 0.0000 |
| counterfactual_cov1.00 | 0.05 | C | level1_cue_marginal | 0.0000 | 0.9428 | 0.5000 | -0.0500 | -0.0500 | 0.0000 |
| counterfactual_cov1.00 | 0.05 | C | level2_knn_jaccard_k3 | 0.0000 | 0.7826 | 0.5000 | -0.0500 | -0.0500 | 0.0000 |
| counterfactual_cov1.00 | 0.05 | C | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | -0.0500 | -0.0500 | 0.0000 |
| counterfactual_cov1.00 | 0.07 | A | level1_cue_marginal | 0.0000 | -0.3586 | 0.5000 | -0.0700 | -0.0700 | 0.0000 |
| counterfactual_cov1.00 | 0.07 | A | level2_knn_jaccard_k3 | 0.0000 | 0.0000 | 0.5000 | -0.0700 | -0.0700 | 0.0000 |
| counterfactual_cov1.00 | 0.07 | A | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | -0.0700 | -0.0700 | 0.0000 |
| counterfactual_cov1.00 | 0.07 | B | level1_cue_marginal | 0.0000 | -0.5774 | 0.5000 | -0.0700 | -0.0700 | -0.0000 |
| counterfactual_cov1.00 | 0.07 | B | level2_knn_jaccard_k3 | 0.0000 | 0.0000 | 0.5000 | -0.0700 | -0.0700 | 0.0000 |
| counterfactual_cov1.00 | 0.07 | B | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | -0.0700 | -0.0700 | 0.0000 |
| counterfactual_cov1.00 | 0.07 | C | level1_cue_marginal | 0.0000 | 0.3780 | 0.5000 | -0.0700 | -0.0700 | 0.0000 |
| counterfactual_cov1.00 | 0.07 | C | level2_knn_jaccard_k3 | 0.0000 | 0.9562 | 0.5000 | -0.0700 | -0.0700 | 0.0000 |
| counterfactual_cov1.00 | 0.07 | C | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | -0.0700 | -0.0700 | 0.0000 |
| original | 0.01 | A | level1_cue_marginal | -0.6882 | -0.6626 | 0.5000 | 0.1700 | 0.2000 | -0.0300 |
| original | 0.01 | A | level2_knn_jaccard_k3 | -0.6772 | -0.4417 | 0.5000 | 0.1466 | 0.1585 | -0.0120 |
| original | 0.01 | A | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1900 | 0.1900 | 0.0000 |
| original | 0.01 | B | level1_cue_marginal | -1.0000 | -0.8944 | 0.5000 | 0.1900 | 0.2100 | -0.0200 |
| original | 0.01 | B | level2_knn_jaccard_k3 | -1.0000 | -0.8944 | 0.5000 | 0.1400 | 0.1521 | -0.0121 |
| original | 0.01 | B | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1900 | 0.1900 | 0.0000 |
| original | 0.01 | C | level1_cue_marginal | 0.9918 | 0.9428 | 0.5000 | 0.1900 | 0.1533 | 0.0367 |
| original | 0.01 | C | level2_knn_jaccard_k3 | 0.7506 | 0.8830 | 0.5000 | 0.2033 | 0.1430 | 0.0603 |
| original | 0.01 | C | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1900 | 0.1900 | 0.0000 |
| original | 0.03 | A | level1_cue_marginal | -0.6882 | -0.7926 | 0.5000 | 0.1500 | 0.1800 | -0.0300 |
| original | 0.03 | A | level2_knn_jaccard_k3 | -0.6772 | -0.4417 | 0.5000 | 0.1266 | 0.1385 | -0.0120 |
| original | 0.03 | A | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1700 | 0.1700 | 0.0000 |
| original | 0.03 | B | level1_cue_marginal | -1.0000 | -0.9428 | 0.5000 | 0.1700 | 0.1900 | -0.0200 |
| original | 0.03 | B | level2_knn_jaccard_k3 | -1.0000 | -0.8944 | 0.5000 | 0.1200 | 0.1321 | -0.0121 |
| original | 0.03 | B | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1700 | 0.1700 | 0.0000 |
| original | 0.03 | C | level1_cue_marginal | 0.9918 | 0.9223 | 0.5000 | 0.1700 | 0.1333 | 0.0367 |
| original | 0.03 | C | level2_knn_jaccard_k3 | 0.7506 | 0.8826 | 0.5000 | 0.1833 | 0.1230 | 0.0603 |
| original | 0.03 | C | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1700 | 0.1700 | 0.0000 |
| original | 0.05 | A | level1_cue_marginal | -0.6882 | -0.6626 | 0.5000 | 0.1300 | 0.1600 | -0.0300 |
| original | 0.05 | A | level2_knn_jaccard_k3 | -0.6772 | -0.4417 | 0.5000 | 0.1066 | 0.1185 | -0.0120 |
| original | 0.05 | A | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1500 | 0.1500 | 0.0000 |
| original | 0.05 | B | level1_cue_marginal | -1.0000 | -0.8944 | 0.5000 | 0.1500 | 0.1700 | -0.0200 |
| original | 0.05 | B | level2_knn_jaccard_k3 | -1.0000 | -0.8944 | 0.5000 | 0.1000 | 0.1121 | -0.0121 |
| original | 0.05 | B | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1500 | 0.1500 | 0.0000 |
| original | 0.05 | C | level1_cue_marginal | 0.9918 | 0.9428 | 0.5000 | 0.1500 | 0.1133 | 0.0367 |
| original | 0.05 | C | level2_knn_jaccard_k3 | 0.7506 | 0.8845 | 0.5000 | 0.1633 | 0.1030 | 0.0603 |
| original | 0.05 | C | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1500 | 0.1500 | 0.0000 |
| original | 0.07 | A | level1_cue_marginal | -0.6882 | -0.6626 | 0.5000 | 0.1100 | 0.1400 | -0.0300 |
| original | 0.07 | A | level2_knn_jaccard_k3 | -0.6772 | -0.4417 | 0.5000 | 0.0866 | 0.0985 | -0.0120 |
| original | 0.07 | A | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1300 | 0.1300 | 0.0000 |
| original | 0.07 | B | level1_cue_marginal | -1.0000 | -0.8944 | 0.5000 | 0.1300 | 0.1500 | -0.0200 |
| original | 0.07 | B | level2_knn_jaccard_k3 | -1.0000 | -0.8944 | 0.5000 | 0.0800 | 0.0921 | -0.0121 |
| original | 0.07 | B | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1300 | 0.1300 | 0.0000 |
| original | 0.07 | C | level1_cue_marginal | 0.9918 | 0.9201 | 0.5000 | 0.1300 | 0.0933 | 0.0367 |
| original | 0.07 | C | level2_knn_jaccard_k3 | 0.7506 | 0.8826 | 0.5000 | 0.1433 | 0.0830 | 0.0603 |
| original | 0.07 | C | level3_candidate_set_tau0.5 | 0.0000 | 0.0000 | 0.5000 | 0.1300 | 0.1300 | 0.0000 |

## 10. Coverage Effect Audit

| Distribution | Cost | Test | positive_rate | oracle-always_obs | L1 sign | L2 sign | L3 sign | L1 policy | L2 policy | L3 policy |
|--------------|------|------|---------------|-------------------|---------|---------|---------|-----------|-----------|-----------|
| original | 0.01 | A | 0.6667 | +0.0033 | 0.6667 | 0.6667 | 0.6667 | 0.4400 | 0.4400 | 0.4400 |
| original | 0.01 | B | 0.6667 | +0.0033 | 0.6667 | 0.6667 | 0.6667 | 0.4400 | 0.4400 | 0.4400 |
| original | 0.01 | C | 0.6667 | +0.0033 | 0.6667 | 0.6667 | 0.6667 | 0.4400 | 0.4400 | 0.4400 |
| original | 0.03 | A | 0.6667 | +0.0100 | 0.6667 | 0.6667 | 0.6667 | 0.4200 | 0.4200 | 0.4200 |
| original | 0.03 | B | 0.6667 | +0.0100 | 0.6667 | 0.6667 | 0.6667 | 0.4200 | 0.4200 | 0.4200 |
| original | 0.03 | C | 0.6667 | +0.0100 | 0.6667 | 0.6667 | 0.6667 | 0.4200 | 0.4200 | 0.4200 |
| original | 0.05 | A | 0.6667 | +0.0167 | 0.6667 | 0.6667 | 0.6667 | 0.4000 | 0.4000 | 0.4000 |
| original | 0.05 | B | 0.6667 | +0.0167 | 0.6667 | 0.6667 | 0.6667 | 0.4000 | 0.4000 | 0.4000 |
| original | 0.05 | C | 0.6667 | +0.0167 | 0.6667 | 0.6667 | 0.6667 | 0.4000 | 0.4000 | 0.4000 |
| original | 0.07 | A | 0.6667 | +0.0233 | 0.6667 | 0.6667 | 0.6667 | 0.3800 | 0.3800 | 0.3800 |
| original | 0.07 | B | 0.6667 | +0.0233 | 0.6667 | 0.6667 | 0.6667 | 0.3800 | 0.3800 | 0.3800 |
| original | 0.07 | C | 0.6667 | +0.0233 | 0.6667 | 0.6667 | 0.6667 | 0.3800 | 0.3800 | 0.3800 |
| counterfactual_cov0.25 | 0.01 | A | 0.6667 | +0.0033 | 0.6667 | 0.6667 | 0.6667 | 0.4400 | 0.4400 | 0.4400 |
| counterfactual_cov0.25 | 0.01 | B | 0.6667 | +0.0033 | 0.6667 | 0.6667 | 0.6667 | 0.4400 | 0.4400 | 0.4400 |
| counterfactual_cov0.25 | 0.01 | C | 0.6667 | +0.0033 | 0.6667 | 0.6667 | 0.6667 | 0.4400 | 0.4400 | 0.4400 |
| counterfactual_cov0.25 | 0.03 | A | 0.6667 | +0.0100 | 0.6667 | 0.6667 | 0.6667 | 0.4200 | 0.4200 | 0.4200 |
| counterfactual_cov0.25 | 0.03 | B | 0.6667 | +0.0100 | 0.6667 | 0.6667 | 0.6667 | 0.4200 | 0.4200 | 0.4200 |
| counterfactual_cov0.25 | 0.03 | C | 0.6667 | +0.0100 | 0.6667 | 0.6667 | 0.6667 | 0.4200 | 0.4200 | 0.4200 |
| counterfactual_cov0.25 | 0.05 | A | 0.6667 | +0.0167 | 0.6667 | 0.6667 | 0.6667 | 0.4000 | 0.4000 | 0.4000 |
| counterfactual_cov0.25 | 0.05 | B | 0.6667 | +0.0167 | 0.6667 | 0.6667 | 0.6667 | 0.4000 | 0.4000 | 0.4000 |
| counterfactual_cov0.25 | 0.05 | C | 0.6667 | +0.0167 | 0.6667 | 0.6667 | 0.6667 | 0.4000 | 0.4000 | 0.4000 |
| counterfactual_cov0.25 | 0.07 | A | 0.6667 | +0.0233 | 0.6667 | 0.6667 | 0.6667 | 0.3800 | 0.3800 | 0.3800 |
| counterfactual_cov0.25 | 0.07 | B | 0.6667 | +0.0233 | 0.6667 | 0.6667 | 0.6667 | 0.3800 | 0.3800 | 0.3800 |
| counterfactual_cov0.25 | 0.07 | C | 0.6667 | +0.0233 | 0.6667 | 0.6667 | 0.6667 | 0.3800 | 0.3800 | 0.3800 |
| counterfactual_cov0.50 | 0.01 | A | 0.6667 | +0.0033 | 0.6667 | 0.6667 | 0.6667 | 0.4400 | 0.4400 | 0.4400 |
| counterfactual_cov0.50 | 0.01 | B | 0.6667 | +0.0033 | 0.6667 | 0.6667 | 0.6667 | 0.4400 | 0.4400 | 0.4400 |
| counterfactual_cov0.50 | 0.01 | C | 0.6667 | +0.0033 | 0.6667 | 0.6667 | 0.6667 | 0.4400 | 0.4400 | 0.4400 |
| counterfactual_cov0.50 | 0.03 | A | 0.6667 | +0.0100 | 0.6667 | 0.6667 | 0.6667 | 0.4200 | 0.4200 | 0.4200 |
| counterfactual_cov0.50 | 0.03 | B | 0.6667 | +0.0100 | 0.6667 | 0.6667 | 0.6667 | 0.4200 | 0.4200 | 0.4200 |
| counterfactual_cov0.50 | 0.03 | C | 0.6667 | +0.0100 | 0.6667 | 0.6667 | 0.6667 | 0.4200 | 0.4200 | 0.4200 |
| counterfactual_cov0.50 | 0.05 | A | 0.6667 | +0.0167 | 0.6667 | 0.6667 | 0.6667 | 0.4000 | 0.4000 | 0.4000 |
| counterfactual_cov0.50 | 0.05 | B | 0.6667 | +0.0167 | 0.6667 | 0.6667 | 0.6667 | 0.4000 | 0.4000 | 0.4000 |
| counterfactual_cov0.50 | 0.05 | C | 0.6667 | +0.0167 | 0.6667 | 0.6667 | 0.6667 | 0.4000 | 0.4000 | 0.4000 |
| counterfactual_cov0.50 | 0.07 | A | 0.6667 | +0.0233 | 0.6667 | 0.6667 | 0.6667 | 0.3800 | 0.3800 | 0.3800 |
| counterfactual_cov0.50 | 0.07 | B | 0.6667 | +0.0233 | 0.6667 | 0.6667 | 0.6667 | 0.3800 | 0.3800 | 0.3800 |
| counterfactual_cov0.50 | 0.07 | C | 0.6667 | +0.0233 | 0.6667 | 0.6667 | 0.6667 | 0.3800 | 0.3800 | 0.3800 |
| counterfactual_cov1.00 | 0.01 | A | 0.6667 | +0.0033 | 0.3333 | 0.3333 | 0.3333 | 0.2500 | 0.2500 | 0.2500 |
| counterfactual_cov1.00 | 0.01 | B | 0.6667 | +0.0033 | 0.3333 | 0.3333 | 0.3333 | 0.2500 | 0.2500 | 0.2500 |
| counterfactual_cov1.00 | 0.01 | C | 0.6667 | +0.0033 | 0.3333 | 0.3333 | 0.3333 | 0.2500 | 0.2500 | 0.2500 |
| counterfactual_cov1.00 | 0.03 | A | 0.6667 | +0.0100 | 0.3333 | 0.3333 | 0.3333 | 0.2500 | 0.2500 | 0.2500 |
| counterfactual_cov1.00 | 0.03 | B | 0.6667 | +0.0100 | 0.3333 | 0.3333 | 0.3333 | 0.2500 | 0.2500 | 0.2500 |
| counterfactual_cov1.00 | 0.03 | C | 0.6667 | +0.0100 | 0.3333 | 0.3333 | 0.3333 | 0.2500 | 0.2500 | 0.2500 |
| counterfactual_cov1.00 | 0.05 | A | 0.6667 | +0.0167 | 0.3333 | 0.3333 | 0.3333 | 0.2500 | 0.2500 | 0.2500 |
| counterfactual_cov1.00 | 0.05 | B | 0.6667 | +0.0167 | 0.3333 | 0.3333 | 0.3333 | 0.2500 | 0.2500 | 0.2500 |
| counterfactual_cov1.00 | 0.05 | C | 0.6667 | +0.0167 | 0.3333 | 0.3333 | 0.3333 | 0.2500 | 0.2500 | 0.2500 |
| counterfactual_cov1.00 | 0.07 | A | 0.6667 | +0.0233 | 0.3333 | 0.3333 | 0.3333 | 0.2500 | 0.2500 | 0.2500 |
| counterfactual_cov1.00 | 0.07 | B | 0.6667 | +0.0233 | 0.3333 | 0.3333 | 0.3333 | 0.2500 | 0.2500 | 0.2500 |
| counterfactual_cov1.00 | 0.07 | C | 0.6667 | +0.0233 | 0.3333 | 0.3333 | 0.3333 | 0.2500 | 0.2500 | 0.2500 |

## 11. Validity / Leakage Audit

- no learner training
- no environment change
- no 1J40b-8 implementation
- forbidden test-time keys absent
- audit-only diagnostic use of true labels

## 12. Diagnostic Flags

- **weak_oracle_headroom** = True
- **always_observe_too_strong** = True
- **nonpositive_penalty_too_weak** = True
- **pre_cue_predictability_low** = True
- **cf_cov1_destroyed_signal** = True
- **level_estimate_sign_mismatch** = True
- **fallback_problem** = False
- **distribution_review_needed** = True

## 13. Recommended Resume Point

- Resume at env3b / counterfactual design review before any 1J40b-8 work.
