# Block 1J17_patch — Cross-Fitted Probe-Value Critic Validation

## 1. Objective

Validate the 1J17 learned critic using leave-object-out cross-fitting:
each candidate (oid, action) is scored by a critic trained WITHOUT that
object's oracle labels. Eliminates same-object supervision leakage concern.

## 2. Method

1. Run C15b observe phase → candidate universe (57 objects)
2. Extract deployable features + oracle labels (342 pairs, 60 features)
3. For each test object, train SmallMLP on all OTHER objects' data
4. Score this object's pairs → OOF scores
5. Deploy: rank by OOF scores, probe top k under C15b-aligned budget

## 3. Results

| Policy | Macro BAcc | Probed | Delta vs C15b | Overlap Oracle k=8 | Eff / Zero / Harm |
|--------|-----------|--------|---------------|--------------------|--------------------|
| C0b (observe only) | 0.5871 | 0 | — | — | — |
| C15b (VOI reserve) | 0.6142 | 8 | 0 | — | — |
| C18b (query-rel) | 0.5975 | — | -0.0167 | — | — |
| Train-all MLP (1J17 ref) | 0.6225 | 5 | +0.0083 | 5 | 5/0/0 |
| **Cross-fit MLP** | **0.5942** | **5** | **-0.0200** | **0** | **4/1/0** |
| Oracle k=8 (non-deployable) | 0.6488 | 8 | +0.0346 | 8 | — |

## 4. OOF Scoring Quality

| Metric | Value |
|--------|-------|
| OOF AUC | 0.6538 |
| OOF Precision@8 | 0.1250 |

## 5. Probe Pairs

| Policy | Pairs |
|--------|-------|
| C15b | [('test_wooden_pickaxe_006', 'burn_as_fuel'), ('test_wooden_pickaxe_003', 'burn_as_fuel'), ('test_stone_block_001', 'mine_by_hand'), ('test_apple_007', 'eat'), ('test_wood_log_002', 'craft_plank'), ('test_stone_block_009', 'mine_by_hand'), ('test_apple_010', 'eat'), ('test_stone_block_000', 'mine_by_hand')] |
| Oracle k=8 | [('test_stone_block_003', 'mine_by_hand'), ('test_stone_block_008', 'burn_as_fuel'), ('test_apple_014', 'eat'), ('test_wooden_pickaxe_005', 'use_as_tool'), ('test_wooden_pickaxe_003', 'use_as_tool'), ('test_apple_012', 'eat'), ('test_stone_block_014', 'mine_by_hand'), ('test_stone_block_011', 'mine_by_hand')] |
| Train-all MLP | [('test_stone_block_003', 'mine_by_hand'), ('test_stone_block_011', 'mine_by_hand'), ('test_wooden_pickaxe_005', 'use_as_tool'), ('test_stone_block_008', 'burn_as_fuel'), ('test_apple_014', 'eat')] |
| Cross-fit MLP | [('test_stone_block_011', 'use_as_tool'), ('test_stone_block_003', 'eat'), ('test_wood_log_003', 'mine_by_hand'), ('test_apple_006', 'eat'), ('test_wood_log_007', 'eat')] |

## 6. Gap Capture

| Policy | Gap Capture % |
|--------|--------------|
| C15b | 6.6% |
| Train-all MLP | 8.6% |
| Cross-fit MLP | 1.7% |
| Oracle k=8 | 14.9% |

## 7. Per-Query Breakdown

| Query | C15b BAcc | Cross-fit MLP BAcc | Delta |
|-------|----------|-------------------|-------|
| need_food | 0.6250 | 0.6250 | +0.0000 |
| need_fuel | 0.6333 | 0.6167 | -0.0167 |
| need_planks | 0.5729 | 0.5312 | -0.0417 |
| need_stone | 0.5521 | 0.5104 | -0.0417 |
| need_tool | 0.6875 | 0.6875 | +0.0000 |

## 8. Interpretation

Cross-fit MLP (0.5942, -0.0200 vs C15b) does NOT improve over C15b, while train-all MLP (0.6225) does. 1J17 improvement was likely same-candidate supervision overfit. Deployable features with OOF critic scores are insufficient.

### Decision Rules

- **If crossfit_mlp > C15b**: learned critic generalization is supported.
  → **Result**: NOT SUPPORTED (delta=-0.0200)
- **If train_all > C15b but crossfit <= C15b**: 1J17 improvement was same-candidate overfit.
  → **Result**: OVERFIT CONFIRMED
- **If crossfit selects oracle-like probes but macro does not improve**:
  execution/budget ordering remains limiting.
  → Overlap oracle k=8: 0/5

## 9. Summary

```
[block_done]
block_id=1J17_patch
c15b_macro_bal=0.6142
train_all_mlp_macro_bal=0.6225
crossfit_mlp_macro_bal=0.5942
crossfit_delta_vs_c15b=-0.0200
crossfit_overlap_with_oracle_k8=0
crossfit_effective_probe_count=4
crossfit_harmful_probe_count=0
deployable_generalization_supported=false
learned_probe_value_route_status=not_supported
candidate_ready_for_small_multiseed=false
ready_for_multiseed=false
```
