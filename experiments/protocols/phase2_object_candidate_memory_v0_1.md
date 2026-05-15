# Phase 2: Object Candidate Memory v0.1

## 1. Purpose

Object Candidate Memory (L2) tracks **what an observed object seems to be** based on accumulated feature evidence from Event Memory (L1). It groups objects with similar observed feature patterns into provisional or stable candidates. Categories emerge from evidence, not from hard-coded constants.

It is currently a **shadow/sidecar module**: it reads from Event Memory and builds candidates, but does NOT influence policy decisions. It will eventually replace `detect_type_family()` (Phase 4).

## 2. Inputs

### 2.1 Allowed

| Source | Field | Usage |
|--------|-------|-------|
| Event Memory (L1) | `observed_features_delta` | Primary evidence for feature co-occurrence |
| Event Memory (L1) | `action_target` (object_id) | Anchors features to a specific object instance |
| Event Memory (L1) | `step_id` | Recorded in `source_step_ids` for provenance |
| Event Memory (L1) | `action_type` | `observe` events carry the richest feature delta |
| Event Memory (L1) | `success_or_failure` | Used later (Phase 3) for action-suitability evidence |
| Teacher Feedback Channel | `teacher_label` (category hint) | Optional supervised evidence; treated as informed source, not ground truth |

### 2.2 Forbidden

`true_family`, `hidden_subtype`, `prior_violation`, `oracle_outcome`, `deceptive_flag`, `full_object_state`, `unobserved_features`, `deviation_features`, `raw_internal_score`, and any audit-only field. Teacher labels must arrive through the Teacher Feedback Channel, not raw Event Memory.

## 3. Candidate Record Fields

| # | Field | Type | Description |
|---|-------|------|-------------|
| 1 | `candidate_id` | str | Unique identifier (e.g., `"cand_0003"`) |
| 2 | `object_ids` | list[str] | Object instances assigned to this candidate |
| 3 | `observed_feature_evidence` | dict[str, dict] | Per-feature: `{count, source_step_ids, first_seen_step, last_seen_step}` |
| 4 | `positive_feature_set` | set[str] | Features with support >= threshold and no contradiction |
| 5 | `negative_feature_set` | set[str] | Features consistently absent when candidate features are present |
| 6 | `contradicted_features` | dict[str, list] | Features present on some instances but absent on others, with per-instance detail |
| 7 | `confidence_score` | float | Posterior-like score from feature diagnosticity |
| 8 | `stability_score` | float | Inverse variance of feature pattern across instances |
| 9 | `support_count` | int | Number of distinct object instances assigned |
| 10 | `episode_spread` | int | Number of distinct episodes evidence is drawn from |
| 11 | `creation_step_id` | int | Event Memory step that triggered candidate creation |
| 12 | `last_updated_step_id` | int | Most recent Event Memory step that updated this candidate |
| 13 | `source_step_ids` | list[int] | All Event Memory step_ids that contributed evidence |
| 14 | `promotion_status` | str | One of: `"provisional"`, `"stable"`, `"rejected"` |
| 15 | `promotion_step_id` | int or null | Step at which status changed; null if still provisional |

## 4. Candidate Formation Rules

### 4.1 Feature Accumulation

When an Event Memory record with `action_type = "observe"` or `"try"` contains `observed_features_delta`, the features are accumulated per object_id:

```
object_feature_map[oid] += observed_features_delta
```

`observed_features_delta` contains only features the environment returned for that action. Features not yet observed are simply absent from the map — they are NOT recorded as "missing" or "false."

### 4.2 Provisional Candidate Formation

A **provisional candidate** is created when an object has >= 2 observed features and does not match any existing candidate (overlap < 2 features with all existing candidates).

Rules:
- The first object with a novel feature pattern creates a provisional candidate in the **candidate buffer**.
- A provisional candidate has `promotion_status = "provisional"`.
- Single-instance candidates stay in the buffer; they never become stable.

### 4.3 Candidate Matching

When processing a new object, compute the overlap score with each existing candidate:

```
overlap(candidate, object_features) = |candidate.positive_feature_set ∩ object_features|
```

Assign the object to the candidate with the highest overlap score, provided the score >= 2. If no candidate matches, create a new provisional candidate.

### 4.4 Evidence Updates

When an object is assigned to a candidate:
- `support_count` increments (if this object_id is new to the candidate).
- Each matched feature's count increments in `observed_feature_evidence`.
- The event's `step_id` is appended to `source_step_ids` and to each matched feature's `source_step_ids`.
- Features present on the object but NOT in the candidate's `positive_feature_set` are either added (if consistent with existing members) or recorded in `contradicted_features` (if inconsistent).

### 4.5 Contradiction Handling

A feature is contradicted when:
- It is present in the candidate's `positive_feature_set` but **absent** from a newly assigned object.
- It is in the candidate's `negative_feature_set` but **present** on a newly assigned object.

Contradictions reduce `confidence_score` but do NOT immediately reject the candidate. A candidate with > 30% contradiction rate on identity features is flagged for review but remains provisional.

### 4.6 Teacher Feedback

If teacher labels arrive through the Teacher Feedback Channel:
- A label like `"test_apple_004 is apple-like"` adds supervised evidence to the candidate's feature set.
- Teacher-labeled features are tracked separately from environment-observed features: `teacher_sourced_features` vs `env_sourced_features`.
- A teacher label alone (without environment feature evidence) does NOT create a candidate. The object must still be observed through Event Memory.

## 5. Promotion Criteria

A candidate moves from `provisional` to `stable` only when ALL of the following are met:

| # | Criterion | Threshold | Rationale |
|---|-----------|-----------|-----------|
| 1 | `support_count >= 2` | At least 2 distinct object instances | No single-instance categories |
| 2 | `episode_spread >= 2` | Evidence from at least 2 different episodes | Not an artifact of one episode's object set |
| 3 | `stability_score >= 0.70` | Feature variance across instances is low | Pattern is consistent, not coincidental |
| 4 | Contradiction rate < 0.30 | Fewer than 30% of identity features contradicted | No mixed/confused candidates |
| 5 | `|positive_feature_set| >= 3` | At least 3 consistent features | Enough signal to distinguish from other candidates |
| 6 | Max overlap with any other candidate < 0.60 | Jaccard similarity with nearest candidate below threshold | Candidate is separable in feature space |

`stability_score` is computed as:

```
stability_score = 1.0 - (contradicted_identity_features / max(positive_identity_features, 1))
```

Single-event promotion count must be **zero**. A candidate created and promoted within a single episode is an error.

## 6. Forbidden Operations

The Object Candidate Memory must NEVER:

1. Read `true_family`, `hidden_subtype`, or any audit-only field
2. Initialize candidates from `TYPE_FAMILIES` or any hard-coded category constant
3. Promote a candidate with `support_count < 2`
4. Use unobserved features as negative evidence ("this object doesn't have X" when X was never checked)
5. Write into Event Memory (L1) — data flows upward only
6. Directly influence policy action selection (in shadow mode)

## 7. Shadow Evaluation Metrics

When run in shadow mode, report:

| Metric | Description |
|--------|-------------|
| `provisional_candidates` | Count of candidates with status "provisional" |
| `stable_candidates` | Count of candidates with status "stable" |
| `average_support_count` | Mean support_count across all candidates |
| `max_support_count` | Highest support_count across candidates |
| `conflict_candidate_count` | Candidates with contradiction rate > 0 |
| `single_event_promotion_count` | Must be 0 |
| `hidden_feature_leakage_detected` | Must be false |
| `oracle_leakage_detected` | Must be false |
| `candidate_buffer_size` | Number of single-instance objects in buffer |
| `feature_overlap_with_type_families` | For evaluation only: Jaccard similarity between learned candidates and TYPE_FAMILIES |

Note: `feature_overlap_with_type_families` is computed by the audit layer for evaluation. It does NOT mean TYPE_FAMILIES was used in candidate formation. It measures whether the learned candidates naturally align with the observable feature clusters.

## 8. Acceptance Checks

| # | Check | Requirement |
|---|-------|-------------|
| 1 | No hidden/oracle/audit-only fields enter Object Candidate Memory | Code audit: only Event Memory fields and Teacher Feedback Channel fields are read |
| 2 | No single-event stable category | `single_event_promotion_count == 0` |
| 3 | Candidate records cite `source_step_ids` | Every candidate's features link back to specific Event Memory steps |
| 4 | Object candidates are provisional unless all promotion criteria met | `support_count >= 2`, `episode_spread >= 2`, `stability_score >= 0.70`, contradiction rate < 0.30, `|positive_features| >= 3`, max overlap < 0.60 |
| 5 | Policy decisions remain unchanged | Shadow mode: Object Candidate Memory output is not consumed by policy |
| 6 | Output is reproducible | Same seed produces same candidates |
| 7 | Teacher labels separated from raw Event Memory | Teacher-sourced features tagged distinctly from env-sourced features |


```
[phase_done]
phase=2
doc=protocols/phase2_object_candidate_memory_v0_1.md
shadow_json=present
shadow_summary=present
provisional_candidates=TBD
stable_candidates=TBD
single_event_promotion_count=TBD
hidden_feature_leakage_detected=TBD
oracle_leakage_detected=TBD
policy_decisions_changed=false
implementation_status=TBD
failure_reason=TBD
```
