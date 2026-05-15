# Memory Architecture Master Spec v0.1

## 1. Architecture Overview

```
                    ┌──────────────────────────────┐
                    │      EXPERIMENT AUDIT LOG     │  (parallel, read-only for evaluation)
                    │  true_family, hidden_subtype, │
                    │  prior_violation, oracle,     │
                    │  deceptive_flag, raw_scores   │
                    └──────────────────────────────┘
                         ↑ join on step_id (eval only)
                         │ never feeds downward into agent memory

┌──────────┐   ┌──────────────┐   ┌─────────────────┐   ┌──────────────────┐
│  EVENT   │──→│   OBJECT     │──→│   EXPERIENCE    │──→│    STRATEGY      │
│  MEMORY  │   │  CANDIDATE   │   │    MEMORY       │   │    MEMORY        │
│  (L1)    │   │  MEMORY (L2) │   │    (L3)         │   │    (L4)          │
└──────────┘   └──────────────┘   └─────────────────┘   └──────────────────┘
      ↑                ↑                   ↑                      │
      │                │                   │                      │
      │                ├───────────────────┼──────────────────────┘
      │                │                   │   (reads upward)
      │                │                   │
      │         ┌──────┴───────────────────┴──────┐
      │         │     TEACHER FEEDBACK CHANNEL     │  (separate input stream)
      │         │  teacher_label, correction,      │
      │         │  demonstration                  │
      │         └─────────────────────────────────┘
      │
environment return (features, outcome, state delta)
```

**Data flow direction:** environment → Event Memory → Object Memory → Experience Memory → Strategy Memory → next action.

**Decision flow:** Strategy Memory reads from all lower layers; lower layers never read downward.

**Audit Log:** runs in parallel. Joined to agent memory by `step_id` for evaluation only. Audit fields never enter agent memory.

**Teacher Feedback Channel:** separate input stream. Feeds Object Memory and Experience Memory as informed evidence. Does NOT enter raw Event Memory.

---

## 2. Layer Definitions

### 2.1 Event Memory (L1)

| Property | Value |
|----------|-------|
| **Purpose** | Ordered, append-only record of every action the agent took and every result the environment returned |
| **Input** | Environment return from each agent action |
| **Output** | Sequence of `EventRecord` objects, queried by step_id, episode_id, action_type, action_target |
| **Update trigger** | After every action (scan, observe, try, skip, switch) |
| **Decision-time usage** | Read-only queries by upper layers; Event Memory never directly drives decisions |

**Allowed fields (13):**
```
step_id, episode_id, order_index, timestamp,
action_type, action_target, action_params,
result_type, observed_features_delta, observed_state_delta,
action_cost, success_or_failure, evidence_source
```

**Forbidden fields:**
```
true_family, hidden_subtype, prior_violation, oracle_outcome,
raw_internal_score, unobserved_features, full_object_state,
deceptive_flag, deviation_features, teacher_label
```

**Invariants:**
- Append-only and immutable. Once written, a `step_id` cannot be overwritten.
- Each event binds exactly one action to exactly one environment-returned result.
- `scan`, `observe`, `try`, `skip`, `switch` are all actions.
- Internal inference (classification, prediction, belief state) is never stored as an event.
- `evidence_source` is always `"environment_return"`.

---

### 2.2 Object Candidate Memory (L2)

| Property | Value |
|----------|-------|
| **Purpose** | Learn what objects are. Categories emerge from repeated event evidence; no categories are hard-coded. |
| **Input** | Event Memory records containing `observed_features_delta` and `success_or_failure`; Teacher Feedback Channel labels (as informed evidence, not ground truth) |
| **Output** | Object candidates with: `category_id`, positive/negative feature sets with provenance `(support_count, contradiction_count, source_step_ids)`, feature diagnosticity weights |
| **Update trigger** | After each event that reveals new features or after a teacher-label event |
| **Decision-time usage** | Provides category membership posteriors and feature diagnosticity to Experience Memory and Strategy Memory |

**Allowed fields (per ObjectCandidate):**
```
category_id, features_positive, features_negative,
feature_diagnosticity, support_count, contradiction_count,
source_step_ids, creation_step_id, last_update_step_id
```

**Forbidden fields:**
```
true_family (may be used only for evaluation, not for category creation)
hidden_subtype, prior_violation, oracle_outcome
any hard-coded TYPE_FAMILIES constant as initializer
```

**Invariants:**
- No hard-coded categories. `TYPE_FAMILIES` is an evaluation constant, not a category initializer.
- Every feature carries provenance: `(support_count, contradiction_count, source_step_ids)`.
- Single-instance anomalies stay in a candidate buffer; promotion requires k > 1 instances plus additional criteria (see §2.2.1).
- Feature diagnosticity is learned from co-occurrence counts, not set manually.
- A teacher label is evidence from an informed source, not built-in ground truth. It may seed a category but does not bypass the evidence requirement.

#### 2.2.1 Candidate Buffer and Promotion

Objects that don't fit existing categories are held in a candidate buffer. Promotion to a full category requires ALL of:

| Criterion | Description |
|-----------|-------------|
| Poor fit | Low posterior under any existing category |
| Recurrence | k > 1 instances observed |
| Feature stability | Low within-cluster feature variance |
| Outcome difference | Action outcomes differ from neighboring categories |
| Feature-space gap | Separable from nearest existing category |
| Predictive value | Adding the category improves outcome prediction |

#### 2.2.2 Feature Role Separation

Features occupy three slots with distinct update dynamics:

| Role | Answers | Update speed | Updated from |
|------|---------|-------------|--------------|
| Identity | "What is this?" | Slow, high inertia | Repeated evidence across exemplars |
| State/quality | "What condition is it in?" | Fast, instance-scoped | Current observation, outcome feedback |
| Action suitability | "What can I do with it?" | Moderate | Experience Memory outcomes |

---

### 2.3 Experience Memory (L3)

| Property | Value |
|----------|-------|
| **Purpose** | Learn what happens when the agent acts on objects. Every experience binds object anchor + action + observed features + outcome. |
| **Input** | Event Memory records (especially `try` outcomes); Object Memory category assignments; Teacher Feedback Channel (evaluated judgments) |
| **Output** | `ExperienceRecord` tuples: `(object_anchor, action, feature_context, outcome)` |
| **Update trigger** | After each `try` event with a success/failure outcome; after a teacher-confirmed judgment evaluation |
| **Decision-time usage** | Queried by Strategy Memory to estimate action values given object category and feature context |

**Allowed fields (per ExperienceRecord):**
```
object_anchor (object_id or category_id), action,
feature_context (dict of observed features at decision time),
outcome (success, failure, or null for observe-only),
source_step_ids
```

**Forbidden fields:**
```
oracle_outcome (outcome comes from environment return, not hidden profile)
raw_internal_score, prior_violation
teacher_label as outcome (teacher label is evidence; the evaluated judgment outcome is stored)
```

**Invariants:**
- Every experience binds object + action + features + outcome. No feature→outcome shortcut without an object anchor.
- `observe(feature)` records have `outcome = null`.
- Judgment is not experience by default. A teacher-confirmed judgment becomes experience only after the agent evaluates its own prior judgment against the teacher label. The *evaluation outcome* (correct/incorrect) is the experience — not the judgment itself.
- Observation IS an action. `observe(feature)` produces an Event Memory record and may update Experience Memory with `outcome = null`.

---

### 2.4 Strategy Memory (L4)

| Property | Value |
|----------|-------|
| **Purpose** | Learn when to observe, try, skip, or switch. Distill accumulated evidence from L1–L3 into action-value estimates. |
| **Input** | Object Memory (category posteriors), Experience Memory (action-outcome history), Event Memory (recent trajectory context) |
| **Output** | Value estimate `V(action | belief_state)` for each candidate action; biases action selection |
| **Update trigger** | After each episode or after significant Experience Memory accumulation |
| **Decision-time usage** | Drives action selection: which objects to probe, which affordances to try, when to skip, when to switch |

**Allowed fields (conceptual):**
```
action_value_estimates (per action, per object category, per feature context),
trajectory_success_counts, trajectory_preconditions,
exploration_bonus (derived from visit counts, not hard-coded)
```

**Forbidden fields:**
```
hand-written rules, hard-coded action preferences,
direct Event Memory access (must read through Object + Experience layers),
strategy persistence that cannot be rebuilt from lower layers
```

**Invariants:**
- No hand-written rules. All strategies are learned summaries.
- Strategies are probabilistic and context-conditioned.
- Strategies must be re-derivable from lower layers when evidence changes.
- Strategy output is a bias on action selection, not a hard override.

---

### 2.5 Audit Log (Parallel)

| Property | Value |
|----------|-------|
| **Purpose** | Record experimenter-measured fields for evaluation. Never feeds into agent memory. |
| **Input** | Environment hidden state, policy internals |
| **Output** | Per-step audit records joined to agent Event Memory by `step_id` |
| **Update trigger** | After every agent action (mirrors Event Memory) |
| **Decision-time usage** | NONE — audit log is read-only for post-hoc evaluation |

**Allowed fields:**
```
step_id (join key), true_family, hidden_subtype, prior_violation,
oracle_outcome, raw_internal_score, unobserved_features,
full_object_state, deceptive_flag, deviation_features
```

**Invariants:**
- Audit fields are NEVER read by agent memory layers.
- The join key `step_id` is the only connection between agent memory and audit log.
- Audit log can be used for acceptance checks (hidden feature leakage, oracle leakage) but not for agent decisions.

---

### 2.6 Teacher Feedback Channel (Separate Input Stream)

| Property | Value |
|----------|-------|
| **Purpose** | Accept human/teacher labels, corrections, and demonstrations as informed evidence. Distinct from raw environment returns. |
| **Input** | Teacher-provided labels: object category hints, action outcome confirmations, corrections |
| **Output** | Teacher-feedback events that feed Object Memory (category seeding) and Experience Memory (judgment evaluation) |
| **Update trigger** | When teacher provides a label (may be sparse, episode-end, or on-demand) |
| **Decision-time usage** | Indirect — teacher feedback updates Object and Experience Memory, which Strategy Memory reads |

**Allowed fields:**
```
feedback_id, step_id_ref (which agent event this feedback references),
label_type (category_hint, outcome_confirm, correction, demonstration),
label_value, confidence (teacher's stated confidence, if any),
timestamp
```

**Forbidden fields:**
```
(None of the Event Memory forbidden fields apply here — this is a different channel.
 But teacher feedback must not bypass the evidence requirement:
 a label alone does not create a category without supporting event evidence.)
```

**Invariants:**
- Teacher labels enter through this channel, NOT through raw Event Memory.
- A teacher label is evidence from an informed source, not built-in ground truth.
- Teacher labels may seed or accelerate category formation in Object Memory but do not bypass the evidence requirement.
- Teacher-confirmed judgment evaluation: the agent compares its prior judgment to the teacher label and records the evaluation outcome (correct/incorrect) in Experience Memory — not the raw teacher label.
- The teacher feedback channel is optional. The system must function without it (purely from environment returns).

---

## 3. Data Flow Diagram

```
EPISODE START
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Agent selects action                                │
│  (Strategy Memory reads L2 + L3 → action choice)     │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  Environment executes action                         │
│  Returns: features_delta, state_delta, outcome, cost │
└───────────────────────┬─────────────────────────────┘
                        │
                        ├──→ Event Memory (L1): append event record
                        │
                        ├──→ Audit Log: append audit record (parallel, walled off)
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  Object Memory (L2): update feature evidence,        │
│  category posteriors, candidate buffer               │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  Experience Memory (L3): if action was try, record   │
│  (object_anchor, action, feature_context, outcome)   │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  Strategy Memory (L4): update action-value estimates │
│  from new experience evidence                        │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
              NEXT ACTION SELECTION
                        │
                        ▼
              (loop until episode end)

TEACHER FEEDBACK CHANNEL (asynchronous, separate stream):
    teacher_label ──→ Object Memory (L2): category evidence
                   ──→ Experience Memory (L3): judgment evaluation
```

---

## 4. Core Invariants

These must hold for any implementation derived from this spec.

| # | Invariant | Scope |
|---|-----------|-------|
| 1 | Event Memory is an ordered, append-only action-result sequence | L1 |
| 2 | `scan`, `observe`, `try`, `skip`, `switch` are all actions | L1 |
| 3 | Each event has exactly one action and exactly one environment-returned result | L1 |
| 4 | Internal inference is not Event Memory | L1 |
| 5 | Object classes are learned or taught, not hard-coded | L2 |
| 6 | Every object feature carries provenance: `(support_count, contradiction_count, source_step_ids)` | L2 |
| 7 | Single-instance anomalies stay in candidate buffer; promotion requires k > 1 plus additional criteria | L2 |
| 8 | Experience Memory binds object/candidate + observed features + action + outcome | L3 |
| 9 | No feature→outcome shortcut without an object anchor | L3 |
| 10 | Judgment is inference, not experience, unless validated by outcome or teacher-feedback evaluation | L3 |
| 11 | Strategy Memory is learned from evidence, not hand-written rules | L4 |
| 12 | Strategies are re-derivable from lower layers when evidence changes | L4 |
| 13 | Audit-only fields must never enter agent memory (any layer) | All |
| 14 | Teacher labels enter through the Teacher Feedback Channel, not through raw Event Memory | L1, TF |
| 15 | A teacher label is evidence from an informed source, not built-in ground truth | L2, TF |
| 16 | The system must function without teacher feedback (environment returns alone are sufficient for core learning) | All |

---

## 5. Phase Mapping

| Phase | Block(s) | What | Status |
|-------|----------|------|--------|
| **0** | 1J33, 1J34 | Fallback help-vs-harm evidence. Family+action fallback tested; direct_harmful=0; safe for next phase. | **Done** |
| **1** | — | Event Memory action-result schema. Defines 13 agent fields, 10 audit-only fields, 5 action types. | **Done** |
| **2** | 1J35 | Object Candidate Memory. Implement `ObjectCandidate`, feature diagnosticity, candidate buffer, promotion criteria. Categories emerge from event evidence only. | Pending |
| **3** | 1J36 | Experience Memory. Implement `ExperienceRecord` with object+action+features+outcome binding. Judgment evaluation from teacher feedback. | Pending |
| **4** | 1J37 | Strategy Memory. Learned action-value estimates for observe/try/skip/switch. No hand-written rules. Compare against Phase 0 baseline. | Pending |
| **5** | — | Environment/doc update. Incorporate experimental findings into design doc and environment. Resolve open questions. | Pending |

**Phase dependencies:**
```
Phase 0 ── independent (complete)
Phase 1 ── independent (complete)
Phase 2 ── depends on Phase 1 (reads Event Memory schema)
Phase 3 ── depends on Phase 1 + 2
Phase 4 ── depends on Phase 1 + 2 + 3
Phase 5 ── depends on Phase 0–4 results
```

---

## 6. Current Fallback Placement

### 6.1 What the 1J33/1J34 fallback is

The `DevPlusFamilyActionFallbackPolicy` uses dual FCRM instances:
- `fcrm_dev`: 4-key memory `(goal, action, family, dev_feat)` — signed risk weights via log-odds ratio
- `fcrm_family`: 3-key memory `(goal, action, family)` — activated when dev_feat signal is zero

When an object has zero deviation features and `fcrm_dev` produces zero penalty, the fallback checks `fcrm_family` for a family-level signal. This rescues raw-zero cases where the dev-level signal is silent but the family-level signal indicates risk.

### 6.2 Transitional status

The fallback is a **transitional mechanism**. It bridges the gap between the current policy-only architecture and the full memory architecture:

| Now (Phase 0) | Later (Phase 3+) |
|---------------|------------------|
| Family-level FCRM with hand-crafted `TYPE_FAMILIES` | Object-category-level experience from observed features |
| Hard-coded `detect_type_family()` using feature set overlap | Learned category membership from event evidence |
| Separate dev and family FCRM instances | Unified Experience Memory with object-anchored outcomes |
| Fallback triggered by `dev_penalty == 0` | Experience Memory lookup: "what happens when I try this action on objects with these features?" |

### 6.3 Migration path

1. Phase 2 Object Memory learns categories from event evidence (replacing `detect_type_family`).
2. Phase 3 Experience Memory records object+action+features+outcome tuples (replacing FCRM counters).
3. Phase 4 Strategy Memory learns when to skip/try based on experience (replacing the fallback activation rule).
4. After Phase 4, the fallback is either subsumed by learned strategy or becomes unnecessary.

### 6.4 Constraint

The fallback **must not remain a hard-coded special-case rule** in the final architecture. It is acceptable as a Phase 0–3 bridge but must be replaced by learned object-action matching in Phase 4.

---

## 7. Teacher-Label Consistency

### 7.1 Resolution of design doc vs Phase 1 doc

| Source | Position | Resolution |
|--------|----------|------------|
| `memory_architecture_v0_1_design.md` §2.2 | Teacher labels stored in Event Memory `teacher_label` field | **Revised.** Teacher labels now enter through the separate Teacher Feedback Channel. |
| `phase1_event_memory_action_result_schema_v0_1.md` §3 | `teacher_label` is audit-only; belongs in separate teacher-feedback channel | **Confirmed.** This is the canonical position. |

### 7.2 Teacher Feedback Channel boundary

```
TEACHER FEEDBACK CHANNEL (separate from Event Memory)
    │
    ├──→ Object Memory (L2): "this object is an apple" → evidence for apple-like category
    │
    ├──→ Experience Memory (L3): agent evaluates own judgment against teacher label
    │       records (object, judge_is_apple, features, outcome=correct/incorrect)
    │
    └── NOT → Event Memory (L1): teacher labels do not enter the raw event stream

AUDIT LOG: may record teacher_label for evaluation, but this is walled off from agent memory.
```

### 7.3 Why not in Event Memory

- Event Memory is strictly "what the agent did and what the environment returned."
- A teacher label is neither an agent action nor an environment return.
- Mixing teacher labels into Event Memory creates ambiguity: is this evidence from the world or from an external informant?
- Keeping them separate allows the agent to distinguish "I observed this" from "I was told this."

### 7.4 Field inventory consistency

| Field | Event Memory (L1) | Object Memory (L2) | Experience Memory (L3) | Strategy Memory (L4) | Audit Log | Teacher Feedback |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| `step_id` | R | — | — | — | R (join) | R (ref) |
| `action_type` | R | — | — | — | R | — |
| `action_target` | R | — | R (as object_anchor) | — | R | — |
| `observed_features_delta` | R | R (evidence) | R (feature_context) | — | R | — |
| `success_or_failure` | R | — | R (outcome) | R (value update) | R | — |
| `true_family` | ✗ | ✗ | ✗ | ✗ | R | — |
| `oracle_outcome` | ✗ | ✗ | ✗ | ✗ | R | — |
| `teacher_label` | ✗ | R (as evidence) | R (via evaluation) | — | R | R |
| `raw_internal_score` | ✗ | ✗ | ✗ | — | R | — |
| `deceptive_flag` | ✗ | ✗ | ✗ | ✗ | R | — |

R = allowed/required, ✗ = forbidden, — = not applicable to this layer

---

## 8. Acceptance Checks

| # | Check | Requirement | Verified by |
|---|-------|-------------|-------------|
| 1 | No hidden feature leakage | `observed_features_delta` contains only features returned by environment for that action | L1 schema, implementation audit |
| 2 | No oracle leakage | `success_or_failure` from environment outcome, not from `hidden_affordance_profile` | L1 schema, implementation audit |
| 3 | Clear memory/audit separation | Audit fields joined by `step_id` only; never read by agent memory layers | Cross-layer field inventory (§7.4) |
| 4 | Teacher feedback boundary defined | Teacher labels in separate channel; not in raw Event Memory | §7.2, §7.3 |
| 5 | Each layer has defined input/output | Every layer (§2.1–§2.6) specifies input, output, update trigger, decision-time usage | §2 |
| 6 | No direct use of true labels by agent memory | `true_family`, `hidden_subtype`, `oracle_outcome` are ✗ in all agent layers | §7.4 |
| 7 | No large environment rewrite required | Phases 2–4 use existing C4 environment; Phase 5 may add features but is backward-compatible | §5 |
| 8 | Implementation phases ordered and minimal | Phase dependency graph is acyclic; each phase builds on prior without rework | §5 |
| 9 | Fallback is transitional | Migration path defined; constraint that fallback must not remain hard-coded | §6 |
| 10 | Design invariants traceable | All 16 invariants (§4) map to specific layers and are checkable | §4 |


```
[master_spec_done]
doc=protocols/memory_architecture_master_spec_v0_1.md
layers=Event_Memory,Object_Candidate_Memory,Experience_Memory,Strategy_Memory,Audit_Log,Teacher_Feedback_Channel
phase_mapping_complete=true
audit_memory_separation=true
teacher_feedback_boundary_defined=true
implementation_status=pass
failure_reason=none
```
