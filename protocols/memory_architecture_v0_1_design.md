# Memory Architecture v0.1 — Design Document

## 1. Four-Layer Structure

```
Layer 4: Strategy Memory        ← when to observe, try, skip, switch
Layer 3: Experience Memory      ← object + action + observed features + outcome
Layer 2: Object Memory          ← object categories, features, confidence
Layer 1: Event Memory           ← raw evidence, immutable
```

Data flow: Event Memory → Object Memory + Experience Memory → Strategy Memory.
Decision flow: only lower layers write upward; higher layers never write downward.

---

## 2. Event Memory

Event Memory is the sole ground-truth store. It records raw evidence and does not directly participate in decisions.

### 2.1 What an event record contains

| Field | Description |
|-------|-------------|
| `event_id` | unique sequential identifier |
| `observed_features` | features visible at event time (free + observed-on-probe) |
| `unobserved_fields` | features known to exist but not yet observed |
| `inference_made` | any classification or judgment the agent made at that moment |
| `action_taken` | observe-X, try-Y, skip, switch, judgment-Z |
| `outcome` | reward, failure, teacher feedback, or null if observe-only |
| `teacher_label` | optional human-provided label (recorded as evidence, not as truth) |

### 2.2 Rules

1. Event Memory is append-only and immutable.
2. Event Memory does not directly drive decisions — decisions read from Object, Experience, and Strategy Memory.
3. An event with no action and no outcome is still a valid record (e.g., passive observation).
4. Teacher labels are stored in the `teacher_label` field and treated as evidence from an informed source, not as built-in ground truth.

---

## 3. Object Memory

Object Memory learns what objects are. Categories emerge from repeated event evidence, not from hard-coded definitions.

### 3.1 Rules

1. **No hard-coded categories.** All object categories form from event evidence or from teacher-labeled events.
2. **Teacher labels are evidence, not truth.** A teacher label "apple" is recorded in an event. Object Memory may form an apple-like category from those labels, but the label does not bypass the evidence requirement.
3. **Every feature must link back to Event Memory.** Positive and negative features both carry provenance: `(support_count, contradiction_count, source_event_ids)`.
4. **Confidence uses feature diagnosticity, not raw feature count.** A feature shared across many categories contributes less discriminative weight than a feature unique to one category.

### 3.2 Feature diagnosticity (conceptual)

```
diagnosticity(f, C) = MI(f; C)    -- mutual information between feature f and category C
conf(C | observed_features) ∝ Σ diagnosticity(f, C) over matched features
                              − Σ diagnosticity(f, not-C) over contradicted features
```

Weights are learned from event co-occurrence counts, not set manually.

---

## 4. Feature Role Separation

For each object category, features occupy three distinct slots with different update dynamics.

| Role | Answers | Update speed | Updates from |
|------|---------|-------------|--------------|
| Identity | "What is this?" | Slow, high inertia | Repeated evidence across exemplars |
| State/quality | "What condition is it in?" | Fast, instance-scoped | Current observation, outcome feedback |
| Action suitability | "What can I do with it?" | Moderate | Experience Memory outcomes |

### 4.1 Example: Apple

| Role | Positive | Negative |
|------|----------|----------|
| Identity-positive | red, round, peel texture, stem | — |
| Identity-negative | — | wood grain, stone grain, metal handle |
| Quality-positive | full shape, normal surface, firm | — |
| Quality-negative | — | shriveled, yellowing, rotten smell |
| Action-eat-negative | — | rotten smell (for eat action only) |

### 4.2 Rule

- A state/quality feature may make an object a "bad instance" of its category without removing category membership.
- An identity-contradicting feature (e.g., wood grain on an apple candidate) triggers re-classification, not "bad apple."
- Action-suitability features are scoped to specific actions. "Rotten smell" is negative for eat but not for burn.

---

## 5. Experience Memory

Experience Memory learns what happens when the agent acts on objects.

### 5.1 Rules

1. **Every experience must bind object + action + observed features + outcome.** The tuple is `(object_id_or_category, action, feature_context, outcome)`.
2. **No feature → outcome without object.** "Wet → failure" is invalid. "Wood + wet + burn → failure" is valid. The object anchor prevents overgeneralization.
3. **Observation is an action.** `observe(feature)` is an action with cost and information outcome.
4. **Judgment is not experience by default.** "Is this an apple?" is an inference.
5. **Teacher-confirmed judgments are first recorded as teacher-label events in Event Memory.** A teacher-label event becomes eligible to update Experience Memory only when the agent explicitly evaluates whether its own prior judgment was correct against that teacher label. The evaluation outcome — not the judgment itself — is the experience. Judgment is still not experience by default.

### 5.2 Example records

```
Record: (apple_category,  observe_color,  {red},           outcome=null)
Record: (object_47,      try_eat,        {red, shriveled}, outcome=bad_taste)
Record: (object_47,      judge_is_apple, {red, shriveled}, outcome=correct, teacher=yes)
Record: (object_48,      try_burn,       {wood_grain},     outcome=good_fire)
```

---

## 6. Strategy Memory

Strategy Memory learns when to observe, try, skip, or switch. It distills accumulated evidence from the lower three layers.

### 6.1 Rules

1. **No hand-written rules.** Strategies are learned summaries, not hard-coded policies.
2. **Input sources:** Event Memory (what happened), Object Memory (what things are), Experience Memory (what actions do).
3. **Output:** A value estimate for each action (observe-X, try-Y, skip, switch) given current belief state.
4. **Strategies are probabilistic and context-conditioned.** They must be re-derivable from lower layers when evidence changes.
5. **Update mechanism (conceptual):** Strategy Memory tracks successful trajectories and their pre-conditions. It biases action selection toward actions that historically led to success given similar object/state/experience context.

---

## 7. Candidate New Object Rule

The system must not create new object categories from a single anomalous instance.

### 7.1 Candidate buffer

A candidate-object buffer holds anomalous clusters before promotion.

### 7.2 Promotion criteria (all must be met)

| Criterion | Description |
|-----------|-------------|
| Poor fit | Existing categories explain the cluster poorly (low posterior under any existing category) |
| Recurrence | At least k instances observed (k > 1, exact value TBD) |
| Feature stability | Low within-cluster feature variance across instances |
| Outcome difference | Action outcomes differ meaningfully from neighboring categories |
| Feature-space gap | Cluster is separable from the nearest existing category in feature space |
| Predictive value | Adding the new category improves outcome prediction for future objects |

### 7.3 Sibling-feature handling

A feature diagnostic of a sibling category does not automatically become a negative feature of the current category. Wood grain raises P(wood-like) and only indirectly lowers P(apple-like) through probability normalization. The system must distinguish between "this is not an apple" and "this is wood-like."

---

## 8. Open Questions

| # | Question | Status |
|---|----------|--------|
| 1 | Exact category formation threshold (k, posterior gap, cluster distance) | unresolved |
| 2 | How to weight feature diagnosticity — MI, likelihood ratio, or Bayesian update | unresolved |
| 3 | How to update Strategy Memory — trajectory distillation, meta-learning, or Q-estimate over belief states | unresolved |
| 4 | How teacher labels interact with self-formed categories when they conflict | unresolved |
| 5 | Whether object hierarchy (object → food → fruit → apple) is learned bottom-up or top-down | unresolved |
| 6 | How to set observation cost vs. try cost in a domain-independent way | unresolved |
| 7 | Whether shared features (e.g., roundness across apples and stones) need explicit cross-category accounting | unresolved |

---

## 9. Design Invariants

These must hold for any implementation derived from this design.

1. **Event Memory is evidence, not policy.** It records; it does not decide.
2. **Object classes are learned or taught, not hard-coded.** No category exists before event evidence supports it.
3. **All object features need event evidence.** Every feature carries source event IDs and support/contradiction counts.
4. **Judgment is inference, not experience, unless validated by outcome or teacher feedback.** A classification is a hypothesis until confirmed.
5. **Experience must bind object and action.** No feature→outcome shortcut without an object anchor.
6. **Strategy is learned from accumulated evidence.** No hand-written rules in Strategy Memory.

---

```
[design_doc_patch_done]
doc=protocols/memory_architecture_v0_1_design.md
implementation_status=pass
failure_reason=none
```
