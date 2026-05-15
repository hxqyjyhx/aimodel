# Implementation Plan: memory_architecture_v0_1

## Current Issue

Signed memory keys `(goal, action, family, dev_feat)` require injected deviation features. When objects have zero deviation features (cross-family disguises), raw_penalty = 0. Family_only avoids 8/12 such cases via `(goal, action, family)` keys, but this signal is not integrated.

Reference: [memory_architecture_v0_1_design.md](memory_architecture_v0_1_design.md)

---

## Phase 0: 1J33 Fallback Test

**Goal:** Confirm that adding family+action fallback when dev_feat signal is zero rescues raw-zero cases without regression.

**Allowed changes:**
- Add `DevPlusFamilyActionFallbackPolicy` with dual FCRM (family_dev + family_only)
- Run 5-variant comparison on seed109
- Compare: A_no_memory, B_signed_sum, Family_only, B_fallback, Offline

**Forbidden changes:**
- No environment generation changes
- No new memory classes beyond existing FCRM
- No oracle/hidden feature access
- No hard exclusion

**Expected output:** `runs/block1j33_family_action_fallback_fix.json`, `protocols/block1j33_family_action_fallback_fix.md`

**Acceptance checks:**
- `raw_zero_cases_rescued >= 8`
- `hidden_feature_leakage_detected = false`
- `no_oracle_leakage_confirmed = true`
- No regression on objects WITH deviation features

**Risks:**
- Family_only signal may be too coarse (family-level, not object-level)
- 4/12 cases still unresolved (cross-family conflict too ambiguous even for family_only)
- Fallback may cause harmful avoidance on non-violation cases

---

## Phase 1: Clean Action-Result Event Memory Schema

**Goal:** Implement Layer 1 (Event Memory) as append-only evidence store per design doc §2.

**Allowed changes:**
- Define `EventRecord` dataclass with fields: `event_id, timestamp, episode, step, object_id, observed_features, unobserved_fields, inference_made, action_taken, outcome, teacher_label`
- Implement append-only `EventMemory` store with query methods
- Record every probe action + outcome as an event
- Record every observation as an event
- Store teacher labels as evidence field only (not as ground truth)

**Forbidden changes:**
- No decision logic in Event Memory
- No mutable records (append-only)
- No direct use of Event Memory for action selection
- No hard-coded categories

**Expected output:** `_block1j34_event_memory_schema.py` — runs Phase A training + probe episodes, records all events, exports event log JSON

**Acceptance checks:**
- Every probe action produces exactly one event record
- Passive observations (no action) also produce valid event records
- Teacher labels stored in `teacher_label` field, not used for decisions
- Event count matches expected: `N_episodes * probes_per_episode + observation_events`
- Immutability enforced (append attempted on existing event_id raises error)

**Risks:**
- Event volume may be large — need efficient query by object_id, episode, feature
- Distinguishing "observed features" vs "unobserved fields" requires environment introspection

---

## Phase 2: Minimal Object Candidate Memory

**Goal:** Implement Layer 2 (Object Memory) basics per design doc §3–4. Categories emerge from event evidence only.

**Allowed changes:**
- Implement `ObjectCandidate` with: `category_id, features (positive + negative), support_count, contradiction_count, source_event_ids`
- Implement `CandidateBuffer` for anomalous clusters before promotion (§7)
- Feature diagnosticity using simple mutual information from co-occurrence counts (§3.2)
- Feature role separation: identity vs state/quality vs action-suitability (§4)
- Link every feature back to source event IDs

**Forbidden changes:**
- No hard-coded categories (wood-like, apple-like, etc. must emerge from evidence)
- No category creation from single instance (k > 1, per §7.2)
- No bypassing event evidence requirement

**Expected output:** `_block1j35_object_candidate_memory.py` — runs episodes, forms object categories from event evidence, exports category state

**Acceptance checks:**
- Categories emerge from repeated event co-occurrence, not from TYPE_FAMILIES constant
- TYPE_FAMILIES used only as evaluation ground truth, not as category initializer
- Every feature carries `(support_count, contradiction_count, source_event_ids)`
- Single-instance anomalies stay in candidate buffer, not promoted
- Feature diagnosticity weights learned from data, not hard-coded

**Risks:**
- With only 5 episodes, may not have enough evidence for reliable category formation
- Teacher labels provide category hints but don't count as truth — cold-start may be slow
- Feature role separation requires careful update dynamics (identity = slow, quality = fast)

---

## Phase 3: Object-State-Action-Outcome Experience Memory

**Goal:** Implement Layer 3 (Experience Memory) per design doc §5. Every experience binds object + action + features + outcome.

**Allowed changes:**
- Implement `ExperienceRecord` with tuple: `(object_id_or_category, action, feature_context, outcome)`
- Query by object, by action, by outcome, by feature_context
- Observation as action: `observe(feature)` records with outcome=null
- Judgment evaluation: when teacher label available, compare prior judgment → outcome = correct/incorrect
- Experience update only from evaluated outcomes, not from raw judgments

**Forbidden changes:**
- No feature→outcome without object anchor (§5.1 rule 2)
- No raw judgment as experience (§5.1 rule 4)
- No teacher label bypassing judgment evaluation step (§5.1 rule 5)

**Expected output:** `_block1j36_experience_memory.py` — builds on Phase 1–2, adds experience records, exports experience DB

**Acceptance checks:**
- Every experience has all 4 fields populated (object, action, feature_context, outcome)
- `observe(feature)` records have outcome=null
- Judgment evaluation records have outcome=correct/incorrect, not the judgment itself
- Query "what happens when I eat apple-like objects" returns only eat+apple-like experiences

**Risks:**
- Object anchor granularity: category-level vs instance-level anchoring affects generalization
- Sparse outcomes — many observe actions with null outcome
- Judgment evaluation requires storing prior judgment to compare against teacher label

---

## Phase 4: Minimal Strategy Memory Statistics

**Goal:** Implement Layer 4 (Strategy Memory) per design doc §6. Learned summaries from lower 3 layers, no hand-written rules.

**Allowed changes:**
- Track successful trajectories and their pre-conditions from Experience Memory
- Compute action value estimates: `V(action | belief_state)` from historical outcomes
- Bias action selection toward actions with historically better outcomes given similar context
- Strategies must be re-derivable when lower-layer evidence changes
- Probabilistic, context-conditioned output

**Forbidden changes:**
- No hand-written rules or hard-coded policies
- No direct Event Memory access (read through Object + Experience layers)
- No strategy persistence that can't be rebuilt from lower layers

**Expected output:** `_block1j37_strategy_memory.py` — builds on Phase 1–3, adds strategy layer, compares against Phase 0 baseline

**Acceptance checks:**
- Strategy output changes when lower-layer evidence changes (re-derivability)
- No hard-coded action preferences
- Value estimates are probabilistic (not binary good/bad)
- Strategy improves over episodes as evidence accumulates

**Risks:**
- Cold start: no strategy before sufficient experience
- Exploration-exploitation tension in early episodes
- Belief state representation may be too coarse for effective conditioning
- Phase 4 depends on Phases 1–3 being solid — cascading failure risk

---

## Phase 5: Update Environment/Docs After Evidence

**Goal:** Incorporate experimental findings back into design doc and environment. Resolve open questions from §8.

**Allowed changes:**
- Update `memory_architecture_v0_1_design.md` with resolved open questions
- Update environment if needed to better test memory architecture (e.g., richer feature spaces, more diverse deceptive patterns)
- Update constants, feature universes, type families based on evidence
- Document which design invariants held and which needed revision

**Forbidden changes:**
- No design invariant removal without explicit justification from experimental evidence
- No environment changes that invalidate prior block comparisons

**Expected output:** Updated `protocols/memory_architecture_v0_1_design.md` with resolved questions, `protocols/memory_architecture_v0_1_evidence_log.md`

**Acceptance checks:**
- Each resolved open question (§8) has a citation to at least one block result
- Design invariants (§9) are either confirmed or revised with evidence
- Environment changes (if any) are backward-compatible or clearly versioned

**Risks:**
- Some open questions may remain unresolved after Phase 4
- Design doc updates may reveal gaps requiring additional phases

---

## Phase Dependency Graph

```
Phase 0 (1J33 fallback) ── independent, run immediately
    │
Phase 1 (Event Memory) ── independent foundation
    │
Phase 2 (Object Memory) ── depends on Phase 1 (reads event records)
    │
Phase 3 (Experience Memory) ── depends on Phase 1 + 2
    │
Phase 4 (Strategy Memory) ── depends on Phase 1 + 2 + 3
    │
Phase 5 (Docs/Env update) ── depends on Phase 0–4 results
```

Phase 0 can proceed in parallel with Phase 1 planning.

---

```
[plan_done]
doc=protocols/memory_architecture_implementation_plan_v0_1.md
phases=6
implementation_status=pass
failure_reason=none
```
