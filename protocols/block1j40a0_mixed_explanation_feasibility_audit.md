# Block 1J40a-0: Mixed Candidate Explanation Feasibility Audit

- **Seed**: 109
- **Episodes**: 5
- **Elapsed**: 2.7s

## Audit Results

| # | Condition | Available | Key Detail |
|---|-----------|-----------|-------------|
| 1 | token_persistence | **YES** | unique=60, multi-ep=0, repeated_obs=0 |
| 2 | repeated_observation | **NO** | count=0, mean=1.0 |
| 3 | state_observation | **NO** | fields=0, events_with_state=0 |
| 4 | multi_action_covariation | **YES** | mixed_objects=60, action_pairs=462 |
| 5 | heldout_split | **YES** | episodes=5, temporal_possible=True |
| 6 | irreducible_noise_control | **NO** | inconsistent_pairs=0 (not required, future) |
| 7 | surface_matched_mixed_rates | **NO** | mixed_objects=60, sources=1 |

## Detailed Results

### 1. Token Persistence

- **token_persistence_available**: True
- Unique tokens: 60
- Tokens in multiple episodes: 0
- Tokens observed more than once: 0
- Tokens with observe and try: 60
- Note: Tokens persist within a single episode (observe + 6 tries) but do NOT reappear across episodes. Token persistence available at within-episode level only.

### 2. Repeated Observation

- **repeated_observation_available**: False
- Repeated observe tokens: 0
- Mean repeated observes: 1.0
- Max repeated observes: 1
- Note: Each object is observed exactly once. No repeated observations exist. This is a structural limitation of the current environment design (round-robin episode assignment, one observe per object).

### 3. State Observation

- **state_observation_available**: False
- Visible state features: 0
- State distinct from identity: False
- Note: observed_state_delta is empty for all events. No agent-visible state features exist in the current environment. All tracked features are identity-like (object type indicators), not mutable state.

### 4. Multi-Action Covariation

- **multi_action_covariation_available**: True
- Mixed objects: 60
- Mixed action pairs: 462
- Objects mixed in 2+ actions: 60
- Mixed rate: 1.0

### 5. Heldout Split

- **heldout_split_available**: True
- Episodes: 5
- Temporally heldout possible: True
- Suggested split: train on episodes 0-3, test on episode 4 (time-ordered heldout)

### 6. Irreducible Noise Control

- **irreducible_noise_control_available**: False
- Inconsistent (oid, action) pairs: 0
- Repeated try count: 0
- Note: Each (object, action) pair is tried exactly once. Outcomes are deterministic from hidden_affordance_profile. No genuine stochastic component exists. This is a structural limitation ¡ª irreducible noise control requires either: (a) repeated tries of same (object, action), (b) an explicit noise source in the environment, or (c) objects with probabilistic hidden affordances.

### 7. Surface-Matched Mixed Rates

- **surface_matched_mixed_rates_available**: False
- Objects with mixed outcomes: 60
- Mean mixed rate: 0.7
- Mixed source types detectable: 1
- Note: Mixed outcomes arise from object-level action profile variation (e.g., stone is minable but not edible). Candidate grouping confounds object-level variation with group-level mixedness. Without stochastic noise or distinct mixed-generation mechanisms, different mixed sources cannot be separated by raw rate matching alone. Environment redesign would be needed for genuine rate-matching across mixed sources.

## Decision Rule

| Condition | Value | Required |
|-----------|-------|----------|
| token_persistence_available | True | YES |
| repeated_observation_available | False | YES |
| state_observation_available | False | YES |
| multi_action_covariation_available | True | YES |
| heldout_split_available | True | YES |
| irreducible_noise_control_available | False | NO (future) |
| surface_matched_mixed_rates_available | False | NO (info) |
| **mixed_explanation_identifiable_now** | **False** | |
| **can_proceed_to_1J40a1** | **False** | |

**BLOCKED**: Do not proceed to explanation scoring. Reasons:

- repeated_observation_available=false: each object observed exactly once
- state_observation_available=false: no state features in events


```
[block_done]
block_id=1J40a-0
token_persistence_available=true
repeated_observation_available=false
state_observation_available=false
multi_action_covariation_available=true
heldout_split_available=true
irreducible_noise_control_available=false
surface_matched_mixed_rates_available=false
mixed_explanation_identifiable_now=false
can_proceed_to_1J40a1=false
policy_decisions_changed=false
hidden_feature_leakage_detected=false
oracle_leakage_detected=false
implementation_status=fail
failure_reason=repeated_observation_available=false: each object observed exactly once; state_observation_available=false: no state features in events
```
