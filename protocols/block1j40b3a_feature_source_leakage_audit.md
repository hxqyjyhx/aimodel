# Block 1J40b-3a: Feature Source Leakage Audit

- **Total elapsed**: 0.9s
- **Implementation Status**: PARTIAL

## Part 1: RoleSignal Source Audit

| Feature | Mapped From | In Observed Delta | First Obs Step | First Use Step | Timing OK |
|---------|------------|-------------------|---------------|---------------|----------|
| uniform_texture | observe_neutral | True | 7 | 8 | True |
| variable_texture | observe_helps | True | 1 | 2 | True |
| worn_texture | observe_wasteful | True | 23 | 24 | True |

- rolesignal_audit_passed: True
- all_in_observed_delta: True
- all_timing_ok: True
- any_derived_from_audit: False

## Part 2: Hidden Feature Timing Audit

- premature_use_count: 0
- no_observe_unrevealed_hidden_count: 0
- one_observe_unrevealed_hidden_count: 0
- repeated_observe_unrevealed_hidden_count: 0
- hidden_timing_audit_passed: True

## Part 3: Ablation Models

### Full Model (all features)

- MSE: 0.026648
- overall_observe_rate: 0.3725
- observe_helps_rate: 0.8636
- observe_neutral_rate: 0.0
- observe_wasteful_rate: 0.0

### No RoleSignal

- MSE: 0.045735
- overall_observe_rate: 0.0196
- observe_helps_rate: 0.0
- observe_neutral_rate: 0.0625
- observe_wasteful_rate: 0.0
- shows_separation: False

### No Hidden Diagnostic Features

- MSE: 0.019427
- overall_observe_rate: 0.2353
- observe_helps_rate: 0.5455
- observe_neutral_rate: 0.0
- observe_wasteful_rate: 0.0

### Visible-Only (No RoleSignal + No Hidden)

- MSE: 0.032878
- overall_observe_rate: 0.0
- observe_helps_rate: 0.0
- observe_neutral_rate: 0.0
- observe_wasteful_rate: 0.0
- shows_separation: False

## Decision Rule

- rolesignal_audit_passed: True
- hidden_timing_audit_passed: True
- no-RoleSignal shows separation: False
- visible-only shows separation: False
- result_valid_after_audit: False

## Acceptance Checks

| Check | Result |
|-------|--------|
| rolesignal_audit_passed | **PASS** |
| hidden_timing_audit_passed | **PASS** |
| no_observe_unrevealed_hidden = 0 | **PASS** |
| one_observe_unrevealed_hidden = 0 | **PASS** |
| rolesignal_not_derived_from_audit | **PASS** |
| hidden_feature_premature_use = false | **PASS** |
| no_rolesignal_helps > neutral | **FAIL** |
| visible_only_helps > neutral | **FAIL** |
| result_valid_after_audit | **FAIL** |
| policy_decisions_changed = false | **PASS** |
| no_seed_crashes | **PASS** |
| **overall_acceptance** | **FAIL** |


```
[block_done]
block_id=1J40b-3a
rolesignal_audit_passed=true
hidden_timing_audit_passed=true
no_observe_unrevealed_hidden_count=0
one_observe_unrevealed_hidden_count=0
rolesignal_derived_from_audit=false
hidden_feature_premature_use=false
no_rolesignal_helps_rate=0.0
no_rolesignal_neutral_rate=0.0625
no_rolesignal_wasteful_rate=0.0
visible_only_helps_rate=0.0
visible_only_neutral_rate=0.0
visible_only_wasteful_rate=0.0
result_valid_after_audit=false
implementation_status=partial
failure_reason=no_rolesignal_helps > neutral; visible_only_helps > neutral; result_valid_after_audit
```
