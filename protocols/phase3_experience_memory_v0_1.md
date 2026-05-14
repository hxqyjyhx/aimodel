# Phase 3: Experience Memory v0.1

## 1. Purpose

Experience Memory (L3) learns **what happens when the agent performs an action on an object/candidate under observed features/state**. It binds object/candidate + observed features/state + action + outcome into experience records. These records will eventually replace FCRM counters (Phase 4), but do not yet influence policy decisions.

Experience Memory sits above Object Candidate Memory (L2) in the architecture. When an object is assigned to a Phase 2 candidate, Experience Memory attributes action outcomes to that candidate. When no candidate matches, the object_id serves as the anchor directly.

It is currently a **shadow/sidecar module**: it reads from Event Memory and Object Candidate Memory, builds experience records, but does NOT influence policy decisions. It will eventually feed Strategy Memory (Phase 4).

## 2. Inputs

### 2.1 Allowed

| Source | Field | Usage |
|--------|-------|-------|
| Event Memory (L1) | `action_type` | Only `try` events produce outcome evidence; `observe` events record with outcome=null |
| Event Memory (L1) | `action_target` (object_id) | Anchors experience to a specific object |
| Event Memory (L1) | `action_params` | Which affordance was attempted |
| Event Memory (L1) | `observed_features_delta` | Features visible at decision time — recorded as feature snapshot |
| Event Memory (L1) | `observed_state_delta` | State context at decision time |
| Event Memory (L1) | `success_or_failure` | Environment-returned outcome (true/false/null) |
| Event Memory (L1) | `step_id`, `episode_id` | Provenance: recorded in `source_event_ids` |
| Object Candidate Memory (L2) | `candidate_id` (provisional or stable) | Anchors experience to a candidate when object is assigned to one |
| Object Candidate Memory (L2) | `positive_feature_set` | Context for which features define the candidate |
| Teacher Feedback Channel | `teacher_label` (outcome confirm) | Optional supervised evidence; treated as informed source, not ground truth |

### 2.2 Forbidden

`true_family`, `hidden_subtype`, `prior_violation`, `oracle_outcome`, `deceptive_flag`, `full_object_state`, `unobserved_features`, `raw_internal_score`, and any audit-only field. Feature-only outcome rules (feature → outcome without object/candidate anchor) are explicitly forbidden.

## 3. Experience Record Fields

| # | Field | Type | Description |
|---|-------|------|-------------|
| 1 | `experience_id` | str | Unique identifier (e.g., `"exp_cand_0001_eat"`) |
| 2 | `object_anchor_id` | str | Object instance this experience was initially bound to |
| 3 | `candidate_anchor_id` | str or null | Phase 2 candidate this object belongs to (null if unassigned) |
| 4 | `source_event_ids` | list[int] | All Event Memory step_ids that contributed to this record |
| 5 | `observed_feature_snapshots` | list[dict] | Feature sets observed at each decision time |
| 6 | `observed_state_snapshots` | list[dict] | State context at each decision time |
| 7 | `action_type` | str | Always `"try"` for outcome-bearing experiences |
| 8 | `action_params` | dict | Which affordance was attempted (e.g., `{"affordance": "eat"}`) |
| 9 | `outcome_history` | list[dict] | Chronological list of `{outcome, step_id, episode_id}` |
| 10 | `success_count` | int | Number of times this binding produced success |
| 11 | `failure_count` | int | Number of times this binding produced failure |
| 12 | `support_count` | int | Total distinct observations of this binding |
| 13 | `contradiction_count` | int | Number of minority-outcome observations (min(success, failure)) |
| 14 | `confidence_score` | float | Proportion of majority outcome (confidence that the binding is consistent) |
| 15 | `episode_spread` | int | Number of distinct episodes evidence is drawn from |
| 16 | `last_updated_step` | int | Most recent Event Memory step that updated this record |
| 17 | `status` | str | One of: `"provisional"`, `"stable"`, `"contradicted"`, `"rejected"` |

## 4. Required Rule

Experience Memory must **never** store:

```
feature → outcome
```

It must store **only**:

```
object/candidate + features/state + action → outcome
```

The object/candidate anchor is mandatory. A feature can inform outcome prediction only through its association with an object/candidate that has action-outcome history.

Violation check: scan all experience records for any that lack both `object_anchor_id` and `candidate_anchor_id`. If any exist, `feature_only_rule_detected = true`.

## 5. Experience Record Lifecycle

### 5.1 Creation

When a `try` event occurs:
1. Look up the object's current candidate assignment from OCM (via `_object_candidate_map`).
2. Create an experience key: `(candidate_id or object_id, action_type, action_affordance)`.
3. If no record exists for this key, create one with `status = "provisional"`.
4. If a record exists, update it with the new outcome.

### 5.2 Update

When an existing experience record receives new evidence:
- `support_count` increments.
- `success_count` or `failure_count` increments based on outcome.
- If both success and failure have been observed, `contradiction_count = min(success_count, failure_count)`.
- `confidence_score = max(success_count, failure_count) / support_count`.
- `episode_spread` recomputed from `outcome_history`.
- New `source_event_ids` and `feature_snapshots` appended.

### 5.3 Status Transitions

| Status | Condition |
|--------|-----------|
| `provisional` | Default for new records with support_count < 2 |
| `stable` | `support_count >= 2`, `contradiction_count == 0`, `episode_spread >= 1` |
| `contradicted` | `contradiction_count > 0` (both success and failure observed for same binding) |
| `rejected` | `support_count >= 3` AND `contradiction_count >= 2` AND `confidence_score < 0.60` |

A contradicted record is NOT rejected — it indicates the binding is context-dependent (features/state matter). A rejected record indicates the binding is too noisy to be useful.

### 5.4 Re-anchoring

When an object is re-assigned to a different Phase 2 candidate (because OCM updated), existing experience records anchored to the old candidate are NOT retroactively modified. New experience records use the new candidate assignment. The old record's `object_anchor_id` preserves the original object anchor for provenance.

## 6. Shadow Evaluation Metrics

When run in shadow mode, report:

| Metric | Description |
|--------|-------------|
| `total_experience_records` | Count of all ExperienceRecord instances |
| `stable_experience_records` | Count with status "stable" |
| `provisional_experience_records` | Count with status "provisional" |
| `contradicted_experience_records` | Count with status "contradicted" |
| `object_action_pairs_count` | Unique (object_id, action) bindings observed |
| `candidate_action_pairs_count` | Unique (candidate_id, action) bindings observed |
| `action_outcome_support_count` | Sum of support_count across all records |
| `contradiction_count` | Sum of contradiction_count across all records |
| `feature_only_rule_detected` | Must be false |
| `hidden_feature_leakage_detected` | Must be false |
| `oracle_leakage_detected` | Must be false |
| `policy_decisions_changed` | Must be false |

## 7. Candidate Action-Outcome Diagnostic

Because Phase 2 candidates are noisy (built from feature evidence only), report whether action-outcome evidence sharpens candidate boundaries:

| Diagnostic | Description |
|------------|-------------|
| `candidates_with_consistent_action_outcomes` | Stable candidates where all actions have consistent (non-contradicted) outcomes |
| `candidates_with_mixed_action_outcomes` | Stable candidates where at least one action has mixed outcomes |
| `candidates_split_needed_by_action_outcome` | Candidates where action-outcome pattern suggests >=2 distinct underlying types (different actions succeed/fail on different subsets of the candidate's objects) |
| `candidates_reinforced_by_action_outcome` | Candidates where action-outcome evidence confirms feature-based grouping (same action, same outcome across all objects in candidate) |
| `candidates_with_no_action_evidence` | Candidates with no try events recorded |

### 7.1 Split Detection Logic

A candidate "needs splitting by action outcome" when:
- It has >= 2 objects
- At least two objects have different action-outcome profiles AND different feature sets
- The action-outcome difference is systematic (not one-off noise)

Specifically: for each action, group the candidate's objects by outcome. If the same action produces different outcomes on different subsets and those subsets have non-overlapping feature patterns, flag as `split_needed`.

## 8. Forbidden Operations

The Experience Memory must NEVER:

1. Read `true_family`, `hidden_subtype`, or any audit-only field
2. Store feature → outcome rules without object/candidate anchor
3. Write into Event Memory (L1) or Object Candidate Memory (L2) — data flows upward only
4. Directly influence policy action selection (in shadow mode)
5. Treat Phase 2 stable candidates as ground-truth categories
6. Hide contradictory outcomes — mixed outcomes must be recorded, not suppressed

## 9. Acceptance Checks

| # | Check | Requirement |
|---|-------|-------------|
| 1 | No hidden/oracle/audit-only fields enter Experience Memory | Code audit: only Event Memory fields and OCM fields are read |
| 2 | Every experience cites source_event_ids | Every record's `source_event_ids` is non-empty |
| 3 | Every experience has object/candidate anchor | No record has both `object_anchor_id` and `candidate_anchor_id` as null |
| 4 | No feature-only outcome rule exists | `feature_only_rule_detected == false` |
| 5 | Policy decisions remain unchanged | Shadow mode: Experience Memory output is not consumed by policy |
| 6 | Contradictory outcomes are reported | `contradicted_experience_records` accurately reflects mixed-outcome bindings |
| 7 | Phase 2 candidates treated as provisional anchors | Experience records note candidate status at time of binding; stable Phase 2 candidates are still provisional for Phase 3 |
| 8 | Output is reproducible | Same seed produces same experience records |


```
[phase_done]
phase=3
doc=protocols/phase3_experience_memory_v0_1.md
shadow_json=TBD
shadow_summary=TBD
total_experience_records=TBD
stable_experience_records=TBD
feature_only_rule_detected=TBD
hidden_feature_leakage_detected=TBD
oracle_leakage_detected=TBD
policy_decisions_changed=false
implementation_status=TBD
failure_reason=TBD
```
