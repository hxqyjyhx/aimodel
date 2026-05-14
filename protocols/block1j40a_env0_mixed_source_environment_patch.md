# Block 1J40a-env0: Mixed-Source Environment Patch

- **Seed**: 109
- **Episodes**: 5
- **Elapsed**: 2.8s
- **New Condition**: C5_mixed_source_identifiability_v0
- **Config C5 idempotent**: True

## Smoke Test Results

- **total_objects**: 60
- **total_events**: 525
- **repeated_observe_token_count**: 30
- **visible_state_feature_count**: 5
- **state_features**: ['clean', 'damaged', 'fresh', 'open', 'wet']
- **mixed_source_types_present**: ['identity_split', 'irreducible_noise', 'observation_gap', 'state_condition']
- **mixed_rate_by_source_type**: {"identity_split": {"count": 15, "mean_mixed_rate": 0.7556, "min": 0.6667, "max": 1.0}, "observation_gap": {"count": 15, "mean_mixed_rate": 0.6667, "min": 0.3333, "max": 1.0}, "state_condition": {"count": 15, "mean_mixed_rate": 0.6815, "min": 0.2222, "max": 0.8889}, "irreducible_noise": {"count": 15, "mean_mixed_rate": 0.7167, "min": 0.5, "max": 1.0}}
- **audit_only_source_labels**: True
- **state_in_feature_delta_count**: 0 (must be 0)
- **noise_same_features_state_for_repeats**: True

## Audit Results

| # | Condition | Available | Key Detail |
|---|-----------|-----------|-------------|
| 1 | token_persistence | **YES** | unique=60, multi-ep=0, repeated_obs=30 |
| 2 | repeated_observation | **YES** | count=30, mean=1.5, feat_change=15, state_change=15 |
| 3 | state_observation | **YES** | fields=5, events=267, features=['clean', 'damaged', 'fresh', 'open', 'wet'] |
| 4 | multi_action_covariation | **YES** | mixed_objects=60, inconsistent_repeats=23 |
| 5 | heldout_split | **YES** | episodes=5, temporal=True |
| 6 | irreducible_noise_control | **YES** | inconsistent_pairs=26, repeated_tries=75 |
| 7 | surface_matched_mixed_rates | **YES** | matched=True, max_gap=0.0889 |

## Mixed Source Types

| Source Type | Object Count | Mean Mixed Rate | Rate Range |
|-------------|-------------|-----------------|------------|
| observation_gap | 15 | 0.6667 | [0.3333, 1.0] |
| state_condition | 15 | 0.6815 | [0.2222, 0.8889] |
| identity_split | 15 | 0.7556 | [0.6667, 1.0] |
| irreducible_noise | 15 | 0.7167 | [0.5, 1.0] |

## Implementation Details

### Observation Gap
- Hidden diagnostic features are OMITTED from first `observed_features_delta` (not set to False)
- 15 tokens show feature change on re-observation
- Hidden feature details stored only in audit labels (`gap_hidden_features`)

### State Condition
- Same object token persists across time (within episode)
- `visible_state` changes between observations (`observed_state_delta` changes)
- Outcome depends on `visible_state` at try time
- State features go into `observed_state_delta`, NOT `observed_features_delta`
- state_in_feature_delta_count: 0
- 15 tokens show state change across re-observes

### Irreducible Noise
- Same (object, action) retried under SAME visible features and state
- Outcomes vary stochastically across repeated tries
- noise_same_features_state_for_repeats: True
- 26 inconsistent (oid, action) pairs

### Identity Split
- Surface-similar objects (same family) with different affordance profiles
- Split detectable via multi-action covariation, not single-action success/failure
- 60 objects show different outcomes across different actions

## Acceptance Checks

| Check | Result |
|-------|--------|
| repeated_observation_available | **PASS** |
| state_observation_available | **PASS** |
| irreducible_noise_control_available | **PASS** |
| all_four_mixed_source_types_present | **PASS** |
| audit_only_source_labels_clean | **PASS** |
| old_c4_preserved | **PASS** |
| surface_matched_mixed_rates_available | **PASS** |
| state_in_feature_delta_clean | **PASS** |
| noise_same_features_state_for_repeats | **PASS** |
| config_c5_idempotent | **PASS** |
| policy_decisions_NOT_changed | **PASS** |
| hidden_feature_leakage_NOT_detected | **PASS** |
| oracle_leakage_NOT_detected | **PASS** |
| **overall_acceptance** | **PASS** |

## Backward Compatibility

- **C4_instance_subtype_cued_v1 preserved**: True
- **C4 test**: C4 works: 20 objects generated
- **Config C5 idempotent**: True (checked before appending)


```
[block_done]
block_id=1J40a-env0
new_condition=C5_mixed_source_identifiability_v0
old_c4_preserved=true
token_persistence_available=true
repeated_observation_available=true
state_observation_available=true
multi_action_covariation_available=true
irreducible_noise_control_available=true
surface_matched_mixed_rates_available=true
mixed_source_types_present=identity_split,irreducible_noise,observation_gap,state_condition
audit_only_source_labels=true
hidden_feature_leakage_detected=false
oracle_leakage_detected=false
policy_decisions_changed=false
implementation_status=pass
failure_reason=none
```
