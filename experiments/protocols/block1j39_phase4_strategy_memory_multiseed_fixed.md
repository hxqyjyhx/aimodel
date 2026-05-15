# Block 1J39_fixed: Phase 4 Strategy Memory Multiseed Robustness (FRESH RUN)

- **Seeds**: [101, 103, 107, 109, 113]
- **Condition**: C4_instance_subtype_cued_v1
- **Episodes per seed**: 5
- **Total elapsed**: 26.3s
- **old_1j39_invalidated**: true
- **old_outputs_copied**: false
- **rerun_completed**: true
- **fixed_script_path**: C:\Users\hxqyjyhx\experiments\exp004_5o_mini_mc_v0\_block1j39_phase4_strategy_memory_multiseed.py
- **fixed_script_hash**: 68f64f3fc29b
- **rerun_timestamp**: 2026-05-12T21:45:16
- **fixed_ocm_matches_1j38**: true
- **fixed_experience_matches_1j38**: true
- **feature_false_treated_as_positive**: False
- **sanity_check_passed**: True

## Bug Description (Fixed)

The original 1J39 multiseed script used a simplified OCM that added all feature dict keys to `positive_feature_set` regardless of True/False values. This is now fixed by using the exact `ObjectCandidate`/`ObjectCandidateMemory` implementation from `_block1j38_strategy_memory_shadow.py`.
- Sanity check: `positive_feature_set` contains `feat_true`=True, does NOT contain `feat_false`=True
- Feature False treated as positive: False

## Per-Seed Results

### Phase 4 Strategy Statistics

| Seed | Total | Try | Observe | Skip | Switch | Proxy | Mixed | Avg Try Val | Avg Risk | Obs Assoc | Re-deriv |
|------|-------|-----|---------|------|--------|-------|-------|------------|----------|-----------|----------|
| 101 | 36 | 24 | 4 | 4 | 4 | 8 | 9 | 0.5083 | 0.4917 | 0.8333 | True |
| 103 | 36 | 24 | 4 | 4 | 4 | 8 | 9 | 0.5273 | 0.4727 | 0.8333 | True |
| 107 | 36 | 24 | 4 | 4 | 4 | 8 | 9 | 0.5120 | 0.4880 | 0.8333 | True |
| 109 | 36 | 24 | 4 | 4 | 4 | 8 | 9 | 0.4732 | 0.5268 | 0.8333 | True |
| 113 | 36 | 24 | 4 | 4 | 4 | 8 | 9 | 0.4759 | 0.5241 | 0.8333 | True |

### Safety Flags

| Seed | Hidden Leak | Oracle Leak | Handwritten | Proxy Unlabeled | Causal Claim | Policy Chg |
|------|-------------|-------------|-------------|-----------------|--------------|------------|
| 101 | False | False | False | False | False | False |
| 103 | False | False | False | False | False | False |
| 107 | False | False | False | False | False | False |
| 109 | False | False | False | False | False | False |
| 113 | False | False | False | False | False | False |

## Aggregate Statistics

| Metric | Mean | Stdev |
|--------|------|-------|
| average_observation_value | 1.0000 | 0.0000 |
| average_risk_score | 0.5007 | 0.0212 |
| average_try_value | 0.4993 | 0.0212 |
| mixed_candidate_strategy_count | 9.0000 | 0.0000 |
| observation_association_score | 0.8333 | 0.0000 |
| observe_strategy_records | 4.0000 | 0.0000 |
| proxy_estimate_strategy_records | 8.0000 | 0.0000 |
| skip_strategy_records | 4.0000 | 0.0000 |
| switch_strategy_records | 4.0000 | 0.0000 |
| total_strategy_records | 36.0000 | 0.0000 |
| try_strategy_records | 24.0000 | 0.0000 |

## Action Value Analysis Across Seeds

| Action | Category | Mean Value | Stdev |
|--------|----------|------------|-------|
| burn_as_fuel | all | 0.5000 | 0.0000 |
| burn_as_fuel | provisional | 0.4833 | 0.0083 |
| burn_as_fuel | stable_consistent | 0.6000 | 0.3742 |
| burn_as_fuel | stable_mixed | 0.5583 | 0.1740 |
| craft_plank | all | 0.2000 | 0.0000 |
| craft_plank | provisional | 0.1914 | 0.0168 |
| craft_plank | stable_consistent | 0.4500 | 0.4583 |
| craft_plank | stable_mixed | 0.1650 | 0.1463 |
| eat | all | 0.2000 | 0.0000 |
| eat | provisional | 0.2122 | 0.0315 |
| eat | stable_consistent | 0.2000 | 0.4000 |
| eat | stable_mixed | 0.1733 | 0.0923 |
| mine_by_hand | all | 0.8000 | 0.0000 |
| mine_by_hand | provisional | 0.8017 | 0.0287 |
| mine_by_hand | stable_consistent | 0.8000 | 0.2449 |
| mine_by_hand | stable_mixed | 0.8517 | 0.0847 |
| mine_with_pickaxe | all | 1.0000 | 0.0000 |
| mine_with_pickaxe | provisional | 1.0000 | 0.0000 |
| mine_with_pickaxe | stable_consistent | 1.0000 | 0.0000 |
| mine_with_pickaxe | stable_mixed | 1.0000 | 0.0000 |
| use_as_tool | all | 0.2000 | 0.0000 |
| use_as_tool | provisional | 0.1905 | 0.0169 |
| use_as_tool | stable_consistent | 0.1500 | 0.2000 |
| use_as_tool | stable_mixed | 0.2567 | 0.0429 |

## Diagnostics

### mine_with_pickaxe dominance

| Category | Mean Value | Dominates? |
|----------|------------|------------|
| all | 1.0000 | True |
| provisional | 1.0000 | True |
| stable_consistent | 1.0000 | True |
| stable_mixed | 1.0000 | True |

### Actions with consistently high value across all categories

- `mine_by_hand`, `mine_with_pickaxe`

### Actions with high variance across seeds

- `burn_as_fuel`: max stdev = 0.3742
- `craft_plank`: max stdev = 0.4583
- `eat`: max stdev = 0.4000
- `mine_by_hand`: max stdev = 0.2449
- `use_as_tool`: max stdev = 0.2000

## Flagged Seeds

| Flag | Seeds |
|------|-------|
| hidden_oracle_leakage | none |
| policy_decisions_changed | none |
| handwritten_strategy_rules | none |
| unlabeled_proxy_estimates | none |
| observation_causal_claims | none |
| not_rederivable | none |
| crashed | none |

## Decision Rule

| Criterion | Value |
|-----------|-------|
| hidden_feature_leakage_detected=false for all seeds | True |
| oracle_leakage_detected=false for all seeds | True |
| policy_decisions_changed=false for all seeds | True |
| handwritten_strategy_rule_detected=false for all seeds | True |
| proxy_estimates_labeled=true for all seeds | True |
| observation_causal_claim_made=false for all seeds | True |
| strategy_memory_rederivable=true for all seeds | True |
| No seed crashes | True |
| **phase4_shadow_robust_enough_for_next_phase** | **True** |

## Important Caveats

- Strategy Memory is **shadow-only**. It does not influence policy decisions.
- Skip/switch values are **proxy estimates** -- no real skip/switch events exist in the current environment.
- Observation value is reported as **association**, not causal.
- ~57% mixed outcome candidate rate from 1J37 is explicitly tracked, not hidden.
- This robustness check verifies the Phase 4 shadow pipeline is stable across seeds.
- **This does NOT claim Strategy Memory is ready to replace FCRM counters or detect_type_family().**


```
[block_done]
block_id=1J39_fixed
old_1j39_invalidated=true
old_outputs_copied=false
rerun_completed=true
fixed_ocm_matches_1j38=true
fixed_experience_matches_1j38=true
feature_false_treated_as_positive=false
seeds=[101, 103, 107, 109, 113]
phase4_shadow_robust_enough_for_next_phase=true
hidden_feature_leakage_detected_any=false
oracle_leakage_detected_any=false
policy_decisions_changed_any=false
handwritten_strategy_rule_detected_any=false
proxy_estimates_unlabeled_any=false
observation_causal_claim_made_any=false
strategy_memory_not_rederivable_any=false
implementation_status=pass
failure_reason=none
```
