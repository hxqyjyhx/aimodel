# Phase 4 Strategy Memory Shadow Summary

- **Block**: 1J38
- **Seed**: 109
- **Elapsed**: 1.7s

## Phase 4 Strategy Statistics

| Metric | Value |
|--------|-------|
| Total strategy records | 36 |
| Stable / Provisional / Contradicted | 19 / 0 / 17 |
| Try / Observe / Skip / Switch | 24 / 4 / 4 / 4 |
| Proxy estimate records | 8 |
| Skip/switch observed event count | 0 |
| Average try value | 0.4732 |
| Average observation value | 1.0000 |
| Average risk score (try) | 0.5268 |
| Mixed candidate strategy count | 9 |
| Records with source IDs | 36 / 36 |
| Proxy estimates labeled | True |
| Causal claim made | False |
| Handwritten rule detected | False |
| Strategy memory re-derivable | True |
| Hidden/oracle leakage | False / False |
| Policy decisions changed | False |

## Observation Associations (NOT Causal)

| Metric | Value |
|--------|-------|
| observation_revealed_new_features_rate | 1.0 |
| observation_candidate_assignment_rate | 1.0 |
| observation_before_success_rate | 0.5 |
| observation_before_failure_rate | 0.5 |
| observation_association_score | 0.8333 |
| total_observe_events | 60 |
| total_objects_observed | 60 |
| total_observed_with_try | 60 |

## Try Strategies (derived_from_experience)

| Strategy | Category | Action | Success | Support | Value | Risk | Status |
|----------|----------|--------|---------|---------|-------|------|--------|
| strat_stable_consistent_try_burn_as_fuel | stable_consistent | burn_as_fuel | 4 | 8 | 0.500 | 0.500 | stable |
| strat_stable_consistent_try_craft_plank | stable_consistent | craft_plank | 2 | 8 | 0.250 | 0.750 | stable |
| strat_stable_consistent_try_eat | stable_consistent | eat | 0 | 8 | 0.000 | 1.000 | stable |
| strat_stable_consistent_try_mine_by_hand | stable_consistent | mine_by_hand | 4 | 8 | 0.500 | 0.500 | stable |
| strat_stable_consistent_try_mine_with_pickaxe | stable_consistent | mine_with_pickaxe | 8 | 8 | 1.000 | 0.000 | stable |
| strat_stable_consistent_try_use_as_tool | stable_consistent | use_as_tool | 2 | 8 | 0.250 | 0.750 | stable |
| strat_stable_mixed_try_burn_as_fuel | stable_mixed | burn_as_fuel | 4 | 6 | 0.667 | 0.333 | contradicted |
| strat_stable_mixed_try_craft_plank | stable_mixed | craft_plank | 0 | 6 | 0.000 | 1.000 | stable |
| strat_stable_mixed_try_eat | stable_mixed | eat | 1 | 6 | 0.167 | 0.833 | contradicted |
| strat_stable_mixed_try_mine_by_hand | stable_mixed | mine_by_hand | 5 | 6 | 0.833 | 0.167 | contradicted |
| strat_stable_mixed_try_mine_with_pickaxe | stable_mixed | mine_with_pickaxe | 6 | 6 | 1.000 | 0.000 | stable |
| strat_stable_mixed_try_use_as_tool | stable_mixed | use_as_tool | 2 | 6 | 0.333 | 0.667 | contradicted |
| strat_provisional_try_burn_as_fuel | provisional | burn_as_fuel | 22 | 46 | 0.478 | 0.522 | contradicted |
| strat_provisional_try_craft_plank | provisional | craft_plank | 10 | 46 | 0.217 | 0.783 | contradicted |
| strat_provisional_try_eat | provisional | eat | 11 | 46 | 0.239 | 0.761 | contradicted |
| strat_provisional_try_mine_by_hand | provisional | mine_by_hand | 39 | 46 | 0.848 | 0.152 | contradicted |
| strat_provisional_try_mine_with_pickaxe | provisional | mine_with_pickaxe | 46 | 46 | 1.000 | 0.000 | stable |
| strat_provisional_try_use_as_tool | provisional | use_as_tool | 8 | 46 | 0.174 | 0.826 | contradicted |
| strat_all_try_burn_as_fuel | all | burn_as_fuel | 30 | 60 | 0.500 | 0.500 | contradicted |
| strat_all_try_craft_plank | all | craft_plank | 12 | 60 | 0.200 | 0.800 | contradicted |
| strat_all_try_eat | all | eat | 12 | 60 | 0.200 | 0.800 | contradicted |
| strat_all_try_mine_by_hand | all | mine_by_hand | 48 | 60 | 0.800 | 0.200 | contradicted |
| strat_all_try_mine_with_pickaxe | all | mine_with_pickaxe | 60 | 60 | 1.000 | 0.000 | stable |
| strat_all_try_use_as_tool | all | use_as_tool | 12 | 60 | 0.200 | 0.800 | contradicted |

## Observe / Skip / Switch Strategies

| Strategy | Category | Type | Evidence Mode | Is Proxy | Value | Obs Value | Status |
|----------|----------|------|---------------|----------|-------|-----------|--------|
| strat_stable_consistent_observe | stable_consistent | observe | observed_action | False | 1.0000 | 1.0000 | stable |
| strat_stable_mixed_observe | stable_mixed | observe | observed_action | False | 1.0000 | 1.0000 | stable |
| strat_provisional_observe | provisional | observe | observed_action | False | 1.0000 | 1.0000 | stable |
| strat_all_observe | all | observe | observed_action | False | 1.0000 | 1.0000 | stable |
| strat_stable_consistent_skip | stable_consistent | skip | proxy_estimate | True | 0.0000 | 0.0000 | stable |
| strat_stable_mixed_skip | stable_mixed | skip | proxy_estimate | True | 0.0000 | 0.1667 | contradicted |
| strat_provisional_skip | provisional | skip | proxy_estimate | True | 0.0000 | 0.1268 | contradicted |
| strat_all_skip | all | skip | proxy_estimate | True | 0.0000 | 0.1139 | contradicted |
| strat_stable_consistent_switch | stable_consistent | switch | proxy_estimate | True | 0.0000 | 0.2065 | stable |
| strat_stable_mixed_switch | stable_mixed | switch | proxy_estimate | True | 0.0000 | 0.2222 | stable |
| strat_provisional_switch | provisional | switch | proxy_estimate | True | 0.0000 | 0.2387 | stable |
| strat_all_switch | all | switch | proxy_estimate | True | 0.0000 | 0.2325 | stable |

## Acceptance Checks

| Check | Result |
|-------|--------|
| no_hidden_leakage | PASS |
| no_oracle_leakage | PASS |
| strategy_records_cite_sources | PASS |
| no_handwritten_rules | PASS |
| mixed_candidates_reported | PASS |
| policy_unchanged | PASS |
| strategy_memory_rederivable | PASS |
| proxy_estimates_labeled | PASS |
| skip_switch_not_overclaimed | PASS |
| observation_causal_claim_made_false | PASS |
| reproducible | PASS |

**All checks passed: True**


```
[phase_done]
phase=4
doc=protocols/phase4_strategy_memory_v0_1.md
strategy_memory_shadow=true
shadow_json=present
shadow_summary=present
total_strategy_records=36
stable_strategy_records=19
mixed_candidate_strategy_count=9
skip_switch_observed_event_count=0
proxy_estimates_labeled=true
observation_causal_claim_made=false
handwritten_strategy_rule_detected=false
hidden_feature_leakage_detected=false
oracle_leakage_detected=false
policy_decisions_changed=false
implementation_status=pass
failure_reason=none
```
