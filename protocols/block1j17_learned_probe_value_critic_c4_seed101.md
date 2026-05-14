# Block 1J17 — Learned Probe-Value Critic Diagnostic

## 1. Objective

Test whether a small learned critic (logistic regression / MLP) trained on
deployable IOM features can predict which (object, action) probes are valuable.
Teacher labels use oracle ground truth ONLY for training, never for deployment.

## 2. Setup

| Parameter | Value |
|-----------|-------|
| Condition | C4_instance_subtype_cued_v1 |
| Seed | 101 |
| Budget | 1.5 |
| Cost Weight | 0.5 |
| Probe Cost | 0.05 |
| Observe Cost | 0.005 |
| Reach Cost/unit | 0.01 |
| IOM k | 10 |
| Training pairs | 342 (57 objects x 6 actions) |
| Feature dim | 60 |
| Models | Logistic Regression (L2), Small MLP (hidden=16, ReLU) |

## 3. Training Label Distribution

| Label | Count | Pct |
|-------|-------|-----|
| oracle_topk_probe | 8 | 2.3% |
| effective | 263 | 76.9% |
| zero | 79 | 23.1% |
| harmful | 0 | 0.0% |

## 4. Leave-Object-Out Cross-Validation

| Metric | Logistic Regression | Small MLP |
|--------|---------------------|-----------|
| Overall AUC | 0.6841 | 0.6119 |
| Overall Precision@8 | 0.0000 | 0.1250 |
| Mean fold AUC | 0.5500 | 0.5250 |
| Valid folds | 8 | 8 |

## 5. Deployment Results

| Policy | Macro BAcc | Pos Rec | Neg Rec | Visited | Probed | Delta vs C15b |
|--------|-----------|---------|---------|---------|--------|---------------|
| C0b (observe only) | 0.5871 | 0.3100 | 0.8642 | 60 | 0 | — |
| C15b (VOI reserve) | 0.6142 | 0.3600 | 0.8683 | 57 | 8 | 0 |
| C18b (query-rel) | 0.5975 | 0.3267 | 0.8683 | 57 | 6 | -0.0167 |
| C19a (best from 1J16) | 0.6067 | — | — | — | 5 | -0.0075 |
| **LR Critic** | **0.5942** | 0.3200 | 0.8683 | 57 | 5 | **-0.0200** |
| **MLP Critic** | **0.6225** | 0.3700 | 0.8750 | 57 | 5 | **+0.0083** |
| Oracle k=8 | 0.6488 | — | — | — | 8 | +0.0346 |
| Oracle (full) | 1.0000 | — | — | — | — | — |

## 6. Probe Pair Overlap

| Comparison | Count |
|------------|-------|
| C15b pairs | [('test_wooden_pickaxe_006', 'burn_as_fuel'), ('test_wooden_pickaxe_003', 'burn_as_fuel'), ('test_stone_block_001', 'mine_by_hand'), ('test_apple_007', 'eat'), ('test_wood_log_002', 'craft_plank'), ('test_stone_block_009', 'mine_by_hand'), ('test_apple_010', 'eat'), ('test_stone_block_000', 'mine_by_hand')] |
| Oracle k=8 pairs | [('test_stone_block_003', 'mine_by_hand'), ('test_stone_block_008', 'burn_as_fuel'), ('test_apple_014', 'eat'), ('test_wooden_pickaxe_005', 'use_as_tool'), ('test_wooden_pickaxe_003', 'use_as_tool'), ('test_apple_012', 'eat'), ('test_stone_block_014', 'mine_by_hand'), ('test_stone_block_011', 'mine_by_hand')] |
| LR critic overlap with C15b | 0 |
| LR critic overlap with oracle k=8 | 1 |
| MLP critic overlap with C15b | 0 |
| MLP critic overlap with oracle k=8 | 5 |

## 7. Probe Effects (LR Critic)

| Metric | Count |
|--------|-------|
| Effective | 3 |
| Zero gain | 2 |
| Harmful | 0 |

## 8. Probe Effects (MLP Critic)

| Metric | Count |
|--------|-------|
| Effective | 5 |
| Zero gain | 0 |
| Harmful | 0 |

## 9. Gap Capture

| Policy | Gap Capture % |
|--------|--------------|
| C15b | 6.6% |
| C18b | 2.5% |
| C19a (best from 1J16) | 4.8% |
| LR Critic | 1.7% |
| MLP Critic | 8.6% |
| Oracle k=8 | 14.9% |

## 10. Per-Query Breakdown

| Query | C15b BAcc | Best Critic BAcc | Delta |
|-------|----------|------------------|-------|
| need_food | 0.6250 | 0.6250 | +0.0000 |
| need_fuel | 0.6333 | 0.6333 | +0.0000 |
| need_planks | 0.5729 | 0.5312 | -0.0417 |
| need_stone | 0.5521 | 0.5938 | +0.0417 |
| need_tool | 0.6875 | 0.7292 | +0.0417 |

## 11. Interpretation

Best learned critic (small_mlp) reaches 0.6225 (++0.0083 vs C15b), improving over C15b but not reaching 10% threshold (0.6284). Learned probe-value route is PARTIALLY supported.

### Tiered Rules

- **If learned critic > C15b**: learned probe-value route is supported.
  → **Result**: SUPPORTED (delta=+0.0083)
- **If learned critic >= 0.6284**: candidate for small multiseed later.
  → **Result**: NOT CANDIDATE (macro_bal=0.6225)
- **If critic predicts oracle probes offline but deployment does not improve**:
  execution/budget ordering remains limiting.
  → **Result**: Offline AUC=0.6119, overlap_oracle_k8=5
- **If critic cannot predict oracle probes offline**:
  available deployable features are insufficient.
  → **Result**: AUC=0.6119

### Diagnostic Caveats

1. IOM has no cross-test-object propagation. A probe on test object A only affects
   predictions for A. Therefore the probe ranking problem reduces to per-object ranking.
2. Training labels use oracle ground truth (probe outcomes). Deployment uses only
   deployable IOM features (visible features, IOM internals, query context).
3. Models are low-capacity (logistic regression, small MLP) trained on 342 pairs.
   Overfitting risk with 60 features is addressed by L2 regularization and LOO CV.
4. Single seed (101) only. Multi-seed generalization unknown.

## 12. Summary

```
[block_done]
block_id=1J17
c15b_macro_bal=0.6142
best_critic_variant=small_mlp
best_critic_auc=0.6119
best_critic_precision_at_8=0.1250
best_critic_overlap_with_oracle_k8=5
best_critic_macro_bal=0.6225
best_critic_delta_vs_c15b=+0.0083
learned_probe_value_supported=true
candidate_ready_for_small_multiseed=false
ready_for_multiseed=false
```
