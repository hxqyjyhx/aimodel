# Phase 1: Event Memory Action-Result Schema v0.1

## 1. Core Invariant

Event Memory is an **ordered, append-only action-result sequence**. It records what the agent did and what the environment returned. It does not store internal inference, hidden labels, or oracle fields.

```
Event Memory = [e0, e1, e2, ..., eN]   (append-only, immutable)
```

Each event binds exactly one action to exactly one environment-returned result. Observation IS an action. Internal inference is NOT an event.

## 2. Required Event Fields

| # | Field | Type | Description |
|---|-------|------|-------------|
| 1 | `step_id` | int | Unique sequential identifier, assigned at append time |
| 2 | `episode_id` | int | Which episode this event belongs to |
| 3 | `order_index` | int | Monotonic index within episode (0, 1, 2, ...) |
| 4 | `timestamp` | float | Wall-clock or step-clock time (for ordering) |
| 5 | `action_type` | str | One of: `scan`, `observe`, `try`, `skip`, `switch` |
| 6 | `action_target` | str | Object ID the action is directed at (or `null` for scan/switch) |
| 7 | `action_params` | dict | Action-specific parameters (e.g., `{"affordance": "eat"}` for try) |
| 8 | `result_type` | str | One of: `features_observed`, `action_outcome`, `no_result`, `position_change` |
| 9 | `observed_features_delta` | dict[str, bool] | Features that became visible as a result of this action (empty if action produced no new observations) |
| 10 | `observed_state_delta` | dict | State changes returned by environment (e.g., `{"position": "left_side"}`, `{"durability_change": -1}`) |
| 11 | `action_cost` | float | Cost incurred for this action |
| 12 | `success_or_failure` | bool or null | `true` if action succeeded, `false` if it failed, `null` if not applicable (scan, observe, skip, switch) |
| 13 | `evidence_source` | str | Always `"environment_return"` — records that this event came from the environment, not from internal reasoning |

### 2.1 `action_type` Enumeration

| action_type | Meaning | Has target? | Has success/failure? |
|-------------|---------|-------------|---------------------|
| `scan` | Survey visible area, discover object positions | No | No |
| `observe` | Inspect one object to reveal its visible features | Yes | No |
| `try` | Attempt an affordance on a target object | Yes | Yes |
| `skip` | Decide not to probe a specific object | Yes | No |
| `switch` | Change attention from one object to another | Yes (new target) | No |

### 2.2 `result_type` Enumeration

| result_type | When | Example |
|-------------|------|---------|
| `features_observed` | Action revealed new visible features on the target | `observe` returned `{red, round, stem}` |
| `action_outcome` | A `try` action produced success or failure | `try eat` → `success=false, reason=bad_taste` |
| `no_result` | Action produced no new information | `skip` on an already-known object |
| `position_change` | Object positions changed (e.g., after scan or switch) | `scan` returned updated object list |

## 3. Audit-Only Fields (Must NOT Enter Agent Event Memory)

These fields may appear in experiment logs for evaluation purposes but **must never be written into agent Event Memory records**.

| Field | Why excluded |
|-------|-------------|
| `true_family` | Hidden ground truth about object type family |
| `hidden_subtype` | Subtype variant not observable by agent |
| `prior_violation` | Pre-computed violation label from oracle |
| `oracle_outcome` | Ground-truth success/failure from hidden affordance profile |
| `raw_internal_score` | Policy internal scoring (risk penalty, soft prior, final score) |
| `unobserved_features` | Features present on object but not yet observed by agent |
| `full_object_state` | Complete environment object state including hidden fields |
| `deceptive_flag` | Whether the object was injected with deceptive features |
| `deviation_features` | Which deviation features were injected (agent may observe them, but the injection list itself is audit-only) |
| `teacher_label` | Human-provided label — belongs in a separate teacher-feedback channel, not in raw Event Memory |

### 3.1 Audit Log vs Agent Memory Separation

```
Agent Event Memory          |  Experiment Audit Log
(what the agent knows)      |  (what the experimenter measures)
----------------------------+-------------------------------
step_id                     |  step_id (shared key)
episode_id                  |  episode_id
order_index                 |  order_index
timestamp                   |  timestamp
action_type                 |  action_type
action_target               |  action_target
action_params               |  action_params
result_type                 |  result_type
observed_features_delta     |  observed_features_delta
observed_state_delta        |  observed_state_delta
action_cost                 |  action_cost
success_or_failure          |  success_or_failure
evidence_source             |  evidence_source
                            |  true_family          ← audit-only
                            |  hidden_subtype       ← audit-only
                            |  prior_violation      ← audit-only
                            |  oracle_outcome       ← audit-only
                            |  raw_internal_score   ← audit-only
                            |  unobserved_features  ← audit-only
                            |  full_object_state    ← audit-only
                            |  deceptive_flag       ← audit-only
```

## 4. Action-Result Examples

### 4.1 `scan` — Survey the scene

```json
{
  "step_id": 0,
  "episode_id": 0,
  "order_index": 0,
  "timestamp": 0.0,
  "action_type": "scan",
  "action_target": null,
  "action_params": {},
  "result_type": "position_change",
  "observed_features_delta": {},
  "observed_state_delta": {
    "objects_visible": ["test_apple_004", "test_wood_log_001", "test_stone_block_007"],
    "positions": {
      "test_apple_004": "on_left_side",
      "test_wood_log_001": "near_table",
      "test_stone_block_007": "on_left_side"
    }
  },
  "action_cost": 0.0,
  "success_or_failure": null,
  "evidence_source": "environment_return"
}
```

### 4.2 `observe` — Inspect one object

```json
{
  "step_id": 1,
  "episode_id": 0,
  "order_index": 1,
  "timestamp": 0.5,
  "action_type": "observe",
  "action_target": "test_apple_004",
  "action_params": {},
  "result_type": "features_observed",
  "observed_features_delta": {
    "round_small": true,
    "greenish": true,
    "has_stem_remnant": true,
    "smooth_texture": true,
    "light_weight": true,
    "on_left_side": true
  },
  "observed_state_delta": {},
  "action_cost": 0.1,
  "success_or_failure": null,
  "evidence_source": "environment_return"
}
```

### 4.3 `try` — Attempt an affordance (success)

```json
{
  "step_id": 2,
  "episode_id": 0,
  "order_index": 2,
  "timestamp": 1.0,
  "action_type": "try",
  "action_target": "test_wood_log_001",
  "action_params": {"affordance": "burn_as_fuel"},
  "result_type": "action_outcome",
  "observed_features_delta": {
    "has_bark_texture": true,
    "brownish": true,
    "has_wood_grain": true
  },
  "observed_state_delta": {},
  "action_cost": 0.3,
  "success_or_failure": true,
  "evidence_source": "environment_return"
}
```

### 4.4 `try` — Attempt an affordance (failure on deceptive object)

```json
{
  "step_id": 3,
  "episode_id": 0,
  "order_index": 3,
  "timestamp": 1.5,
  "action_type": "try",
  "action_target": "test_apple_004",
  "action_params": {"affordance": "eat"},
  "result_type": "action_outcome",
  "observed_features_delta": {
    "has_peel_texture": true,
    "damp_texture": true
  },
  "observed_state_delta": {},
  "action_cost": 0.3,
  "success_or_failure": false,
  "evidence_source": "environment_return"
}
```

Note: The agent observes `damp_texture` as a visible feature (it is present on the object surface). The agent records the failure outcome. The agent does NOT know that `damp_texture` was an injected deviation feature — it is just another observed feature. The audit log separately tracks that this object was deceptive.

### 4.5 `skip` — Decide not to probe an object

```json
{
  "step_id": 4,
  "episode_id": 0,
  "order_index": 4,
  "timestamp": 2.0,
  "action_type": "skip",
  "action_target": "test_stone_block_007",
  "action_params": {"reason": "below_exploration_threshold"},
  "result_type": "no_result",
  "observed_features_delta": {},
  "observed_state_delta": {},
  "action_cost": 0.0,
  "success_or_failure": null,
  "evidence_source": "environment_return"
}
```

### 4.6 `switch` — Change attention to another object

```json
{
  "step_id": 5,
  "episode_id": 0,
  "order_index": 5,
  "timestamp": 2.5,
  "action_type": "switch",
  "action_target": "test_stone_block_007",
  "action_params": {"from_target": "test_apple_004"},
  "result_type": "position_change",
  "observed_features_delta": {},
  "observed_state_delta": {"attention": "test_stone_block_007"},
  "action_cost": 0.0,
  "success_or_failure": null,
  "evidence_source": "environment_return"
}
```

## 5. What Event Memory Does NOT Store

These are explicitly excluded and must never appear in agent Event Memory records:

| Excluded | Why | Where it belongs (if anywhere) |
|----------|-----|-------------------------------|
| Internal inferences ("this is an apple") | Inference is not evidence | Strategy Memory or working state |
| Category assignments | Categories are built in Object Memory from events | Object Memory |
| Predicted outcomes | Predictions are not results | Strategy Memory or working state |
| Risk penalty scores | Internal policy computation | Audit log only |
| Soft prior values | Internal policy computation | Audit log only |
| Hidden affordance profiles | Agent cannot observe these | Environment / audit log |
| Teacher labels | Separate feedback channel | Teacher-feedback events (distinct from Event Memory) |
| Unobserved feature names | Agent does not yet know these exist | Audit log only |
| Object type families | Categories emerge from evidence, not from constants | Object Memory (learned) |

## 6. Compatibility with Current Experiments (1J33/1J34)

### 6.1 Mapping Current Probe Logs to Event Memory

Current probe_detail records can be converted into Event Memory records without changing policy behavior:

| Current probe_detail field | Maps to Event Memory field | Notes |
|---------------------------|---------------------------|-------|
| `oid` | `action_target` | Direct mapping |
| `action` | `action_params.affordance` | The action string becomes the affordance parameter |
| `outcome` | `success_or_failure` | 0.0 → false, 1.0 → true |
| `is_violation` | — | Audit-only, not stored in agent Event Memory |
| `is_nonviolation` | — | Audit-only |
| `soft_prior` | — | Audit-only (internal policy score) |
| `risk_penalty` | — | Audit-only (internal policy score) |
| `final_score` | — | Audit-only (internal policy score) |
| `family` | — | Not stored; agent must learn families from observed features |
| `injected_deviation_present` | — | Audit-only; agent sees deviation features as normal visible features |
| `is_deceptive` | — | Audit-only |
| `repeated` | — | Redundant; ordering is captured by `order_index` |

### 6.2 What Changes and What Doesn't

**Changes (data format only):**
- Probe actions gain explicit `action_type`: `try` for affordance probes
- Pre-probe feature checks become `observe` events
- `skip` decisions get explicit event records
- Episode boundaries become explicit via `episode_id`

**Does NOT change:**
- Policy decision logic (same FCRM, same fallback, same scoring)
- Environment dynamics (same objects, same hidden affordances, same features)
- Feature universe (same 26 features)
- Action space (same 6 try-affordances)
- Any 1J33/1J34 result

### 6.3 Conversion Pseudocode

```python
def probe_detail_to_event(detail, episode_id, order_index, timestamp):
    """Convert a 1J33/1J34 probe_detail dict to a Phase 1 Event Memory record.
    Only environment-returned data enters the event. Policy internals are excluded."""
    return {
        "step_id": next_step_id(),
        "episode_id": episode_id,
        "order_index": order_index,
        "timestamp": timestamp,
        "action_type": "try",
        "action_target": detail["oid"],
        "action_params": {"affordance": detail["action"]},
        "result_type": "action_outcome",
        "observed_features_delta": {},
        "observed_state_delta": {},
        "action_cost": 0.3,
        "success_or_failure": (detail["outcome"] == 1.0),
        "evidence_source": "environment_return",
    }
    # Fields NOT copied: is_violation, is_nonviolation, soft_prior,
    #   risk_penalty, final_score, family, injected_deviation_present,
    #   is_deceptive, repeated
```

## 7. Acceptance Checks

| # | Check | Requirement |
|---|-------|-------------|
| 1 | No hidden feature leakage | `observed_features_delta` contains only features actually returned by the environment for that action. No `unobserved_features`, no pre-populated feature lists. |
| 2 | No oracle leakage | `success_or_failure` comes from environment outcome, not from `hidden_affordance_profile` or `prior_violation`. The agent learns success/failure by trying. |
| 3 | One action per event | Each event record has exactly one `action_type` and one `action_target` (or null for scan). No compound actions. |
| 4 | One result per event | Each event has exactly one `result_type` and one `success_or_failure` value. No batched outcomes. |
| 5 | Observation is an action | `observe` and `scan` are `action_type` values, not passive background processes. They consume steps and have costs. |
| 6 | Internal inference not stored | No event contains `inferred_category`, `predicted_outcome`, `belief_state`, or `judgment`. These live in Object/Experience/Strategy Memory or working state. |
| 7 | Append-only immutability | Once written, an event record cannot be modified. Any attempt to overwrite an existing `step_id` is an error. |
| 8 | Evidence source tagged | Every event has `evidence_source = "environment_return"`. No event is created from internal reasoning alone. |
| 9 | Audit fields separated | The experiment audit log may join agent Event Memory with audit-only fields via `step_id`, but those fields never enter the agent's Event Memory store. |
| 10 | Teacher labels excluded | Teacher labels (if present) go to a separate teacher-feedback channel, not into raw Event Memory. |

## 8. Schema Summary

```
EventRecord:
    step_id:          int                        # unique, sequential, immutable
    episode_id:       int                        # episode boundary
    order_index:      int                        # monotonic within episode
    timestamp:        float                      # ordering clock
    action_type:      "scan"|"observe"|"try"|"skip"|"switch"
    action_target:    str|null                   # object ID
    action_params:    dict                       # e.g. {"affordance": "eat"}
    result_type:      "features_observed"|"action_outcome"|"no_result"|"position_change"
    observed_features_delta:  dict[str, bool]    # newly visible features from this action
    observed_state_delta:     dict               # state changes from this action
    action_cost:      float                      # cost incurred
    success_or_failure: bool|null                # outcome of try; null for non-try actions
    evidence_source:  "environment_return"       # always this value
```


```
[phase_done]
phase=1
doc=protocols/phase1_event_memory_action_result_schema_v0_1.md
example_json=present
event_memory_hidden_leakage_allowed=false
observation_is_action=true
implementation_status=pass
failure_reason=none
```
