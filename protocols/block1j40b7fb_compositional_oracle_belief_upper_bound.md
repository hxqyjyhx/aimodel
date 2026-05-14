# Block 1J40b-7f-b: Compositional Oracle-Belief Upper-Bound Control

## 1. Checkpoint Summary

- **Status**: NO_HEADROOM
- **Elapsed**: 23.70s
- **cf_cov1.00**: RUN

## 2. Files Changed

- `_block1j40b7fb_compositional_oracle_belief_upper_bound.py`
- `runs/block1j40b7fb_compositional_oracle_belief_upper_bound.json`
- `protocols/block1j40b7fb_compositional_oracle_belief_upper_bound.md`
- `protocols/block1j40b7fb_compositional_oracle_belief_upper_bound_table.csv`
- `checkpoint_1j40b7fb_compositional_oracle_belief_upper_bound.md`

## 3. Commands Run

```powershell
& "D:\conda\python.exe" "_block1j40b7fb_compositional_oracle_belief_upper_bound.py"
```

## 4. Why 7f Exact-Signature Oracle Was Inadequate

- Held-out Test A/B signatures are unseen by construction.
- Exact-signature oracle-belief therefore key-misses on Test A/B and falls back to global mean.
- Because the fallback is global, 7f cannot test whether compositional legal cue structure contains selective-observe headroom.
- 7f-b is required before deciding whether 1J40b-8 is justified.

## 5. Level 1 / Level 2 / Level 3 Definitions

- **Level 1**: cue-marginal mean over known legal pre_surface cues.
- **Level 2**: top-3 training signatures by Jaccard overlap.
- **Level 3**: candidate signatures satisfying subset relation or Jaccard >= 0.5, support-weighted.

## 6. Distribution Conditions

- ['counterfactual_cov0.25', 'counterfactual_cov0.50', 'counterfactual_cov1.00', 'original']

## 7. Observe Costs

- [0.01, 0.03, 0.05, 0.07]

## 8. Test A Results

| Distribution | Cost | always_obs | exact | level1 | level2 | level3 | oracle_sel | learned | best_level | best_minus_always_obs |
|--------------|------|------------|-------|--------|--------|--------|------------|---------|------------|-----------------------|
| counterfactual_cov0.25 | 0.01 | 0.4400 | 0.4400 | 0.4400 | 0.4400 | 0.4400 | 0.4433 | 0.4400 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.25 | 0.03 | 0.4200 | 0.4200 | 0.4200 | 0.4200 | 0.4200 | 0.4300 | 0.4200 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.25 | 0.05 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4167 | 0.4000 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.25 | 0.07 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.4033 | 0.3800 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.50 | 0.01 | 0.4400 | 0.4400 | 0.4400 | 0.4400 | 0.4400 | 0.4433 | 0.4400 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.50 | 0.03 | 0.4200 | 0.4200 | 0.4200 | 0.4200 | 0.4200 | 0.4300 | 0.4200 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.50 | 0.05 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4167 | 0.4000 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.50 | 0.07 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.4033 | 0.3800 | level1_cue_marginal | +0.0000 |
| counterfactual_cov1.00 | 0.01 | 0.4400 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.4433 | 0.0492 | level1_cue_marginal | -0.1900 |
| counterfactual_cov1.00 | 0.03 | 0.4200 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.4300 | 0.0475 | level1_cue_marginal | -0.1700 |
| counterfactual_cov1.00 | 0.05 | 0.4000 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.4167 | 0.0500 | level1_cue_marginal | -0.1500 |
| counterfactual_cov1.00 | 0.07 | 0.3800 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.4033 | 0.0500 | level1_cue_marginal | -0.1300 |
| original | 0.01 | 0.4400 | 0.4400 | 0.4400 | 0.4400 | 0.4400 | 0.4433 | 0.4400 | level1_cue_marginal | +0.0000 |
| original | 0.03 | 0.4200 | 0.4200 | 0.4200 | 0.4200 | 0.4200 | 0.4300 | 0.4200 | level1_cue_marginal | +0.0000 |
| original | 0.05 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4167 | 0.4000 | level1_cue_marginal | +0.0000 |
| original | 0.07 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.4033 | 0.3800 | level1_cue_marginal | +0.0000 |

## 9. Test B Results

| Distribution | Cost | always_obs | exact | level1 | level2 | level3 | oracle_sel | learned | best_level | best_minus_always_obs |
|--------------|------|------------|-------|--------|--------|--------|------------|---------|------------|-----------------------|
| counterfactual_cov0.25 | 0.01 | 0.4400 | 0.4400 | 0.4400 | 0.4400 | 0.4400 | 0.4433 | 0.4400 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.25 | 0.03 | 0.4200 | 0.4200 | 0.4200 | 0.4200 | 0.4200 | 0.4300 | 0.4200 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.25 | 0.05 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4167 | 0.4000 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.25 | 0.07 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.4033 | 0.3800 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.50 | 0.01 | 0.4400 | 0.4400 | 0.4400 | 0.4400 | 0.4400 | 0.4433 | 0.4400 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.50 | 0.03 | 0.4200 | 0.4200 | 0.4200 | 0.4200 | 0.4200 | 0.4300 | 0.4200 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.50 | 0.05 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4167 | 0.4000 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.50 | 0.07 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.4033 | 0.3800 | level1_cue_marginal | +0.0000 |
| counterfactual_cov1.00 | 0.01 | 0.4400 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.4433 | 0.2483 | level1_cue_marginal | -0.1900 |
| counterfactual_cov1.00 | 0.03 | 0.4200 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.4300 | 0.2450 | level1_cue_marginal | -0.1700 |
| counterfactual_cov1.00 | 0.05 | 0.4000 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.4167 | 0.2500 | level1_cue_marginal | -0.1500 |
| counterfactual_cov1.00 | 0.07 | 0.3800 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.4033 | 0.2500 | level1_cue_marginal | -0.1300 |
| original | 0.01 | 0.4400 | 0.4400 | 0.4400 | 0.4400 | 0.4400 | 0.4433 | 0.4400 | level1_cue_marginal | +0.0000 |
| original | 0.03 | 0.4200 | 0.4200 | 0.4200 | 0.4200 | 0.4200 | 0.4300 | 0.4200 | level1_cue_marginal | +0.0000 |
| original | 0.05 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4167 | 0.4000 | level1_cue_marginal | +0.0000 |
| original | 0.07 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.4033 | 0.3800 | level1_cue_marginal | +0.0000 |

## 10. Test C Results

| Distribution | Cost | always_obs | exact | level1 | level2 | level3 | oracle_sel | learned | best_level | best_minus_always_obs |
|--------------|------|------------|-------|--------|--------|--------|------------|---------|------------|-----------------------|
| counterfactual_cov0.25 | 0.01 | 0.4400 | 0.4400 | 0.4400 | 0.4400 | 0.4400 | 0.4433 | 0.4400 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.25 | 0.03 | 0.4200 | 0.4200 | 0.4200 | 0.4200 | 0.4200 | 0.4300 | 0.4200 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.25 | 0.05 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4167 | 0.4000 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.25 | 0.07 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.4033 | 0.3800 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.50 | 0.01 | 0.4400 | 0.4400 | 0.4400 | 0.4400 | 0.4400 | 0.4433 | 0.4400 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.50 | 0.03 | 0.4200 | 0.4200 | 0.4200 | 0.4200 | 0.4200 | 0.4300 | 0.4200 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.50 | 0.05 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4167 | 0.4000 | level1_cue_marginal | +0.0000 |
| counterfactual_cov0.50 | 0.07 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.3800 | 0.4033 | 0.3800 | level1_cue_marginal | +0.0000 |
| counterfactual_cov1.00 | 0.01 | 0.4400 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.4433 | 0.0867 | level1_cue_marginal | -0.1900 |
| counterfactual_cov1.00 | 0.03 | 0.4200 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.4300 | -0.0405 | level1_cue_marginal | -0.1700 |
| counterfactual_cov1.00 | 0.05 | 0.4000 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.4167 | -0.0408 | level1_cue_marginal | -0.1500 |
| counterfactual_cov1.00 | 0.07 | 0.3800 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.4033 | -0.0500 | level1_cue_marginal | -0.1300 |
| original | 0.01 | 0.4400 | 0.4433 | 0.4400 | 0.4400 | 0.4400 | 0.4433 | 0.4433 | level1_cue_marginal | +0.0000 |
| original | 0.03 | 0.4200 | 0.4300 | 0.4200 | 0.4200 | 0.4200 | 0.4300 | 0.4300 | level1_cue_marginal | +0.0000 |
| original | 0.05 | 0.4000 | 0.4167 | 0.4000 | 0.4000 | 0.4000 | 0.4167 | 0.4167 | level1_cue_marginal | +0.0000 |
| original | 0.07 | 0.3800 | 0.4033 | 0.3800 | 0.3800 | 0.3800 | 0.4033 | 0.4033 | level1_cue_marginal | +0.0000 |

## 11. Fallback Analysis

| Distribution | Cost | Test | Level | n_signatures | n_non_global | n_fallbacks | fallback_rate |
|--------------|------|------|-------|--------------|--------------|-------------|---------------|
| counterfactual_cov0.25 | 0.01 | A | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov0.25 | 0.01 | A | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.01 | A | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.01 | A | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.01 | B | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov0.25 | 0.01 | B | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.01 | B | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.01 | B | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.01 | C | exact_signature_oracle_belief | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.01 | C | level1_cue_marginal | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.01 | C | level2_knn_jaccard_k3 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.01 | C | level3_candidate_set_tau0.5 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.03 | A | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov0.25 | 0.03 | A | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.03 | A | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.03 | A | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.03 | B | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov0.25 | 0.03 | B | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.03 | B | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.03 | B | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.03 | C | exact_signature_oracle_belief | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.03 | C | level1_cue_marginal | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.03 | C | level2_knn_jaccard_k3 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.03 | C | level3_candidate_set_tau0.5 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.05 | A | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov0.25 | 0.05 | A | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.05 | A | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.05 | A | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.05 | B | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov0.25 | 0.05 | B | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.05 | B | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.05 | B | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.05 | C | exact_signature_oracle_belief | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.05 | C | level1_cue_marginal | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.05 | C | level2_knn_jaccard_k3 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.05 | C | level3_candidate_set_tau0.5 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.07 | A | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov0.25 | 0.07 | A | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.07 | A | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.07 | A | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.07 | B | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov0.25 | 0.07 | B | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.07 | B | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.07 | B | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.07 | C | exact_signature_oracle_belief | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.07 | C | level1_cue_marginal | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.07 | C | level2_knn_jaccard_k3 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.25 | 0.07 | C | level3_candidate_set_tau0.5 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.01 | A | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov0.50 | 0.01 | A | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.01 | A | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.01 | A | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.01 | B | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov0.50 | 0.01 | B | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.01 | B | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.01 | B | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.01 | C | exact_signature_oracle_belief | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.01 | C | level1_cue_marginal | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.01 | C | level2_knn_jaccard_k3 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.01 | C | level3_candidate_set_tau0.5 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.03 | A | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov0.50 | 0.03 | A | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.03 | A | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.03 | A | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.03 | B | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov0.50 | 0.03 | B | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.03 | B | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.03 | B | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.03 | C | exact_signature_oracle_belief | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.03 | C | level1_cue_marginal | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.03 | C | level2_knn_jaccard_k3 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.03 | C | level3_candidate_set_tau0.5 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.05 | A | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov0.50 | 0.05 | A | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.05 | A | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.05 | A | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.05 | B | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov0.50 | 0.05 | B | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.05 | B | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.05 | B | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.05 | C | exact_signature_oracle_belief | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.05 | C | level1_cue_marginal | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.05 | C | level2_knn_jaccard_k3 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.05 | C | level3_candidate_set_tau0.5 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.07 | A | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov0.50 | 0.07 | A | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.07 | A | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.07 | A | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.07 | B | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov0.50 | 0.07 | B | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.07 | B | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.07 | B | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.07 | C | exact_signature_oracle_belief | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.07 | C | level1_cue_marginal | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.07 | C | level2_knn_jaccard_k3 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov0.50 | 0.07 | C | level3_candidate_set_tau0.5 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.01 | A | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov1.00 | 0.01 | A | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.01 | A | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.01 | A | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.01 | B | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov1.00 | 0.01 | B | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.01 | B | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.01 | B | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.01 | C | exact_signature_oracle_belief | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.01 | C | level1_cue_marginal | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.01 | C | level2_knn_jaccard_k3 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.01 | C | level3_candidate_set_tau0.5 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.03 | A | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov1.00 | 0.03 | A | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.03 | A | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.03 | A | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.03 | B | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov1.00 | 0.03 | B | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.03 | B | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.03 | B | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.03 | C | exact_signature_oracle_belief | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.03 | C | level1_cue_marginal | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.03 | C | level2_knn_jaccard_k3 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.03 | C | level3_candidate_set_tau0.5 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.05 | A | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov1.00 | 0.05 | A | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.05 | A | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.05 | A | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.05 | B | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov1.00 | 0.05 | B | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.05 | B | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.05 | B | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.05 | C | exact_signature_oracle_belief | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.05 | C | level1_cue_marginal | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.05 | C | level2_knn_jaccard_k3 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.05 | C | level3_candidate_set_tau0.5 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.07 | A | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov1.00 | 0.07 | A | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.07 | A | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.07 | A | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.07 | B | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| counterfactual_cov1.00 | 0.07 | B | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.07 | B | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.07 | B | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.07 | C | exact_signature_oracle_belief | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.07 | C | level1_cue_marginal | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.07 | C | level2_knn_jaccard_k3 | 8 | 8 | 0 | 0.0000 |
| counterfactual_cov1.00 | 0.07 | C | level3_candidate_set_tau0.5 | 8 | 8 | 0 | 0.0000 |
| original | 0.01 | A | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| original | 0.01 | A | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| original | 0.01 | A | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| original | 0.01 | A | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| original | 0.01 | B | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| original | 0.01 | B | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| original | 0.01 | B | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| original | 0.01 | B | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| original | 0.01 | C | exact_signature_oracle_belief | 8 | 8 | 0 | 0.0000 |
| original | 0.01 | C | level1_cue_marginal | 8 | 8 | 0 | 0.0000 |
| original | 0.01 | C | level2_knn_jaccard_k3 | 8 | 8 | 0 | 0.0000 |
| original | 0.01 | C | level3_candidate_set_tau0.5 | 8 | 8 | 0 | 0.0000 |
| original | 0.03 | A | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| original | 0.03 | A | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| original | 0.03 | A | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| original | 0.03 | A | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| original | 0.03 | B | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| original | 0.03 | B | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| original | 0.03 | B | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| original | 0.03 | B | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| original | 0.03 | C | exact_signature_oracle_belief | 8 | 8 | 0 | 0.0000 |
| original | 0.03 | C | level1_cue_marginal | 8 | 8 | 0 | 0.0000 |
| original | 0.03 | C | level2_knn_jaccard_k3 | 8 | 8 | 0 | 0.0000 |
| original | 0.03 | C | level3_candidate_set_tau0.5 | 8 | 8 | 0 | 0.0000 |
| original | 0.05 | A | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| original | 0.05 | A | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| original | 0.05 | A | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| original | 0.05 | A | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| original | 0.05 | B | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| original | 0.05 | B | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| original | 0.05 | B | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| original | 0.05 | B | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| original | 0.05 | C | exact_signature_oracle_belief | 8 | 8 | 0 | 0.0000 |
| original | 0.05 | C | level1_cue_marginal | 8 | 8 | 0 | 0.0000 |
| original | 0.05 | C | level2_knn_jaccard_k3 | 8 | 8 | 0 | 0.0000 |
| original | 0.05 | C | level3_candidate_set_tau0.5 | 8 | 8 | 0 | 0.0000 |
| original | 0.07 | A | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| original | 0.07 | A | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| original | 0.07 | A | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| original | 0.07 | A | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| original | 0.07 | B | exact_signature_oracle_belief | 2 | 0 | 2 | 1.0000 |
| original | 0.07 | B | level1_cue_marginal | 2 | 2 | 0 | 0.0000 |
| original | 0.07 | B | level2_knn_jaccard_k3 | 2 | 2 | 0 | 0.0000 |
| original | 0.07 | B | level3_candidate_set_tau0.5 | 2 | 2 | 0 | 0.0000 |
| original | 0.07 | C | exact_signature_oracle_belief | 8 | 8 | 0 | 0.0000 |
| original | 0.07 | C | level1_cue_marginal | 8 | 8 | 0 | 0.0000 |
| original | 0.07 | C | level2_knn_jaccard_k3 | 8 | 8 | 0 | 0.0000 |
| original | 0.07 | C | level3_candidate_set_tau0.5 | 8 | 8 | 0 | 0.0000 |

## 12. Per-Signature Examples

- original cost=0.01 Test A amb(brownish,heavy_weight)+discolored,has_bark_texture: true_E=-0.0100, L1=0.2100 (fallback=False), L2=0.1521 (fallback=False), L3=0.1900 (fallback=False)
- original cost=0.01 Test A amb(brownish,heavy_weight)_only: true_E=0.2900, L1=0.1900 (fallback=False), L2=0.1400 (fallback=False), L3=0.1900 (fallback=False)
- original cost=0.01 Test A amb(brownish,heavy_weight)+has_crystal_flecks: true_E=-0.0100, L1=0.1900 (fallback=False), L2=0.1650 (fallback=False), L3=0.1900 (fallback=False)
- original cost=0.01 Test A amb(brownish,heavy_weight)+spotted: true_E=0.2900, L1=0.1500 (fallback=False), L2=0.1531 (fallback=False), L3=0.1900 (fallback=False)
- original cost=0.01 Test A amb(greenish,light_weight)+discolored: true_E=0.2900, L1=0.1500 (fallback=False), L2=0.1531 (fallback=False), L3=0.1900 (fallback=False)
- original cost=0.01 Test A amb(greenish,light_weight)+has_grip_area: true_E=-0.0100, L1=0.1900 (fallback=False), L2=0.1650 (fallback=False), L3=0.1900 (fallback=False)
- original cost=0.01 Test A amb(greenish,light_weight)+has_stem_remnant,spotted: true_E=-0.0100, L1=0.2100 (fallback=False), L2=0.1521 (fallback=False), L3=0.1900 (fallback=False)
- original cost=0.01 Test A amb(greenish,light_weight)_only: true_E=0.2900, L1=0.1900 (fallback=False), L2=0.1400 (fallback=False), L3=0.1900 (fallback=False)
- original cost=0.01 Test B amb(greenish,light_weight)+has_stem_remnant,spotted: true_E=-0.0100, L1=0.2100 (fallback=False), L2=0.1521 (fallback=False), L3=0.1900 (fallback=False)
- original cost=0.01 Test B amb(greenish,light_weight)_only: true_E=0.2900, L1=0.1900 (fallback=False), L2=0.1400 (fallback=False), L3=0.1900 (fallback=False)
- original cost=0.01 Test B amb(brownish,heavy_weight)+discolored,has_bark_texture: true_E=-0.0100, L1=0.2100 (fallback=False), L2=0.1521 (fallback=False), L3=0.1900 (fallback=False)
- original cost=0.01 Test B amb(brownish,heavy_weight)_only: true_E=0.2900, L1=0.1900 (fallback=False), L2=0.1400 (fallback=False), L3=0.1900 (fallback=False)

## 13. Headroom Summary

- any_level_beats_always_observe_on_A = False
- any_level_beats_always_observe_on_B = False
- any_level_beats_always_observe_on_A_or_B = False
- final_headroom_status = NO_HEADROOM

## 14. Whether cf_cov1.00 Was Run

- status = RUN
- reason = n/a

## 15. Validity / Leakage Audit

- forbidden test-time keys absent
- k and tau fixed in advance
- no hidden-label keying at test time
- no learner training added for oracle-belief levels

## 16. Final Status: NO_HEADROOM

No compositional level materially beat always_observe in the executed conditions.

## 17. Exact Next Suggested Resume Point

- Resume with design review of env3b / counterfactual construction before 1J40b-8.
