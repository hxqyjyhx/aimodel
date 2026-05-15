# Block 1J9: Route F Blurred Peripheral Observation — Preregistration

**Date:** 2026-05-10
**Block:** 1J9 — Route F interface preregistration (no implementation, no simulation)
**Condition:** C4_instance_subtype_cued_v1
**Seed:** 101 (reference only; audits will use seed=101)
**Budget:** 1.5
**Cost Weight:** 0.5

---

## 1. Purpose

Route F is an **observation-interface hypothesis**, not a policy optimization. It tests whether providing coarse local-scene context before committing to focal observation/probe improves object selection and probe targeting.

The current C15/C15B observation interface is point-like: the agent sees only the focal object's visible features after reaching it. The agent has no information about nearby unvisited objects, forcing a nearest-first visit strategy that Block 1J8 showed is suboptimal (oracle object selection achieves 0.6442 vs C15b 0.6142).

Route F adds blurred peripheral observation layers that give the agent coarse, lossy information about nearby objects before deciding where to visit and probe. The hypothesis: richer local context enables better object prioritization while preserving enough uncertainty for focused probing to remain valuable.

### Accepted Findings From 1J8

| Finding | Value |
|---------|-------|
| C15b macro_bal | 0.6142 |
| C15B cross-action diagnostic VOI macro_bal | 0.6142 |
| Route B marginal gain over Route A | ~0 |
| Oracle object selection macro_bal | 0.6442 |
| Oracle object selection delta vs C15b | +0.0300 |
| Oracle object selection exceeds 10% gap threshold | Yes (13.82%) |
| Full oracle macro_bal | 1.0000 |
| Environment ceiling bottleneck | false |
| Object selection bottleneck | true |
| Budget binding bottleneck | true |
| Probe informativeness bottleneck | true |

---

## 2. Design Principle

Route F peripheral observation must be a **deterministic, lossy function of public visible state**:

```
peripheral_signal = f(public_visible_state, geometry, distance, blur_level)
```

**Forbidden:** peripheral_signal must not reference hidden_subtype, true affordance labels, probe outcomes for any object, hidden category, or any aggregated hidden-state variable.

The blur function is:
- **Distance-decayed**: information density decreases monotonically with Manhattan distance from the focal object
- **Lossy**: cannot reconstruct exact hidden or true labels from peripheral signals
- **Fixed before evaluation**: blur parameters are preregistered, not tuned based on C15/C15B performance

---

## 3. Observation Layers

Route F defines four observation layers. Layer 0 is the existing focal observation. Layers 1-3 are new peripheral layers, each providing progressively coarser information at greater distance.

### Layer 0: Focal Object (existing, unchanged)

When the agent reaches and observes an object:
- Full public visible features (same as current C15/C15B interface)
- No hidden subtype
- No true affordance label
- Cost: OBSERVE_COST (0.005) — unchanged

### Layer 1: Near Peripheral Objects (distance ≤ d1)

For unvisited objects within Manhattan distance d1 of the focal object:
- Nearby object count (total unvisited objects within radius)
- Coarse public visible cue histogram: for each visible feature name, count of nearby objects exhibiting that feature, binned into coarse categories (e.g., 0, 1, 2-3, 4+)
- Relative distance (Manhattan) to each nearby object, binned (e.g., adjacent, near, edge)
- Relative bearing (N/S/E/W quadrant or octant) to each nearby object
- Blurred surface-category-like cue only if derived from public visible features (see Section 6)

**No per-object exact feature vectors** — only aggregate histograms and binned relational geometry.

**Cost:** included in OBSERVE_COST (no additional budget charge; peripheral information is a side effect of focal observation).

### Layer 2: Mid-Range Peripheral Objects (d1 < distance ≤ d2)

For unvisited objects between d1 and d2 of the focal object:
- Lower-resolution object count / density (e.g., "sparse", "moderate", "dense")
- Coarser visible cue histogram with fewer bins (e.g., presence/absence per feature rather than counts)
- Stronger blur: feature values are aggregated across wider spatial bins
- No individual object-level features
- Relative bearing at coarser resolution (quadrant only)

### Layer 3: Background Local Distribution (distance > d2, or full-grid)

For the full grid beyond d2:
- Aggregate public visible cue distribution across all remaining unvisited objects
- Local density / sparsity estimate
- No individual hidden states or object identities
- Provides a prior for "what kinds of objects are still out there"

### Candidate Distance Thresholds (preregistered, not tuned)

| Variant | d1 (near) | d2 (mid) | Description |
|---------|-----------|----------|-------------|
| F_low | 1 | 2 | Minimal peripheral information — only immediate neighbors |
| F_mid | 2 | 4 | Moderate coarse local context — small neighborhood |
| F_high | 3 | 6 | Stronger local context — larger perceptual field, higher leakage risk |

These are preregistered variants representing different points on the information/leakage tradeoff curve. They are NOT performance-tuned — they are fixed before any policy evaluation.

---

## 4. Allowed Signals

Route F peripheral observation may include ONLY the following categories of information, all derived exclusively from public visible state and geometry:

| Signal Category | Description | Source |
|----------------|-------------|--------|
| Distance-decayed visible cue histograms | Per-feature counts of nearby objects exhibiting each visible feature, binned and distance-weighted | Public visible features × geometry |
| Coarse nearby object counts | Count of unvisited objects within each distance band, binned into categories | Geometry only |
| Public surface-category-like cues | Blurred category hints derived from visible features (e.g., k-NN or histogram-based), only if the derivation function uses only public visible features | Public visible features |
| Relative distance | Manhattan distance to nearby objects, binned | Geometry only |
| Relative bearing | Direction (quadrant/octant) to nearby objects | Geometry only |
| Local object density / sparsity | Count of objects per unit area in each distance band | Geometry only |
| Uncertainty-preserving coarse bins | All continuous or count signals binned coarsely so that exact values cannot be reconstructed | Post-processing |

---

## 5. Forbidden Signals

Route F peripheral observation must NOT expose any of the following, in any form (including noised, binned, aggregated, or derived):

- Hidden subtype labels (for any object)
- True affordance labels (ground-truth affordance values for any action-object pair)
- Probe outcomes for non-focal objects
- Hidden category (if category is not observable from public visible features)
- Any noised subtype label (adding noise does not make hidden state public)
- Any noised affordance label
- Aggregated hidden-state statistics (e.g., "3/5 nearby objects are minority subtype", "most nearby objects can be crafted")
- Hand-coded minority flags (e.g., "this object is atypical for its category")
- Action-success priors computed from hidden state (e.g., true P(craft_plank | category))
- Any function of the above, even if binned or aggregated

The distinction between Sections 4 and 5 is structural: peripheral signals must be computable from only (visible_features, grid_positions, agent_position) with no access to SimulatorTruth.

---

## 6. Blur / Degradation Rules

### Core Rules

1. **Distance-decayed**: information content decreases monotonically with Manhattan distance from the focal object. Farther objects contribute less information to the peripheral signal.

2. **Monotonic**: if d_a < d_b, then object A contributes at least as much information to the peripheral signal as object B (all else equal).

3. **Lossy**: the peripheral signal cannot be inverted to reconstruct exact public visible features of individual non-focal objects. Coarse binning ensures this.

4. **Fixed before evaluation**: all blur parameters (distance thresholds, bin boundaries, decay rates) are preregistered and must not be tuned based on policy performance.

5. **Layer-consistent**: the same blur function applies to all policies using the Route F interface. No policy-specific tuning.

### Blur Function Specification

For a visible feature f at an unvisited object o at Manhattan distance d from the focal object:

```
contribution(f, o, d) = presence(f, o) * decay(d, layer)
```

Where:
- `presence(f, o)` = 1 if object o has visible feature f, else 0 (public information only)
- `decay(d, layer)` = decay factor based on distance band and layer

Layer contribution is aggregated across all objects in the distance band:

```
layer_signal(f, band) = aggregate({contribution(f, o, d) for o in band})
```

Where `aggregate` is a binning function (e.g., sum → {0, 1, 2-3, 4+}) that prevents reconstruction of individual object features.

### Preregistered Blur Variants

| Parameter | F_low | F_mid | F_high |
|-----------|-------|-------|--------|
| d1 (near, full histogram) | 1 | 2 | 3 |
| d2 (mid, coarse histogram) | 2 | 4 | 6 |
| Near bin resolution | {0, 1, 2+} | {0, 1, 2-3, 4+} | {0, 1, 2-3, 4-6, 7+} |
| Mid bin resolution | presence only | presence only | {0, 1, 2+} |
| Background detail | density only | density + feature prevalence | density + feature prevalence |
| Bearing resolution | quadrant | quadrant | octant |
| Distance binning | {adjacent, near} | {adjacent, near, edge} | {adjacent, near, mid, edge} |

These variants will be evaluated in the leakage audit (Block 1J10) before any policy comparison.

---

## 7. Leakage Audit Plan

Before any C15F policy evaluation under Route F, the peripheral signal must pass a leakage audit (Block 1J10). The audit tests whether peripheral information inadvertently exposes hidden state. **Block 1J10 is the audit block itself** — it implements the peripheral_signal function, runs leakage and marginal-value audits, but makes no C15F policy claim. Route F proceeds to C15F policy evaluation / policy implementation in **Block 1J11** only if Block 1J10 leakage and marginal-value audits pass.

### Required Audit Tests

Every leakage probe test (A, B, C, and their sub-variants) must report all of the following controls:

- **base_rate_accuracy**: accuracy of always predicting the most common class
- **cross_validated_accuracy**: leave-one-out or k-fold cross-validated classifier accuracy
- **shuffled_label_control**: classifier accuracy when target labels are randomly permuted (breaks any real signal; measures overfitting / classifier capacity)
- **real_minus_shuffle_gap**: cross_validated_accuracy − shuffled_label_control — the portion of accuracy attributable to real signal in the peripheral features
- **fail condition**: a leakage test fails only if BOTH:
  1. cross_validated_accuracy > base_rate + 0.10, AND
  2. real_minus_shuffle_gap is nontrivial (≥ 0.05)

A test where accuracy exceeds base_rate + 0.10 but real_minus_shuffle_gap < 0.05 indicates classifier overfitting rather than genuine leakage and should be treated as a PASS with a note.

---

**A1. per_category_hidden_subtype_probe_from_peripheral_signal**

For each public/visible category group (if category grouping can be formed from public features in a principled way), train a classifier to predict hidden_subtype within that group from the peripheral signal alone. This avoids confounding subtype leakage with category-level signal — global subtype labels are category-specific and may appear predictable if the peripheral signal encodes category-like information (which is allowed at the coarse/bin level).

Threshold: within-group subtype prediction accuracy must not exceed within-group base_rate + 0.10, AND real_minus_shuffle_gap must be nontrivial for a fail.

**A2. global_majority_vs_minority_probe_from_peripheral_signal**

Train a classifier to predict whether an object is majority-subtype vs minority-subtype (binary) from the peripheral signal alone, across all objects regardless of category. This tests whether peripheral information reveals subtype typicality without requiring category-specific subtype labeling.

Threshold: majority/minority prediction accuracy must not exceed base_rate + 0.10, AND real_minus_shuffle_gap must be nontrivial for a fail.

If A2 fails but A1 passes for all categories, the peripheral signal may be encoding coarse category structure (allowed) rather than true subtype information (forbidden). Investigate before deciding.

**B. true_affordance_probe_from_peripheral_signal**

Train a classifier to predict any true affordance label from the peripheral signal alone. If any affordance is predictable beyond base rate, the peripheral signal leaks affordance information.

Threshold: per-affordance prediction accuracy must not exceed base_rate + 0.10, AND real_minus_shuffle_gap must be nontrivial for a fail.

**C. hidden_category_probe_from_peripheral_signal**

Train a classifier to predict hidden_category from the peripheral signal alone. If category prediction accuracy substantially exceeds chance (0.25 for 4 categories), the peripheral signal leaks category structure beyond what is allowed through coarse public-feature-derived cues.

Threshold: category prediction accuracy must not exceed 0.35 (chance + 0.10), AND real_minus_shuffle_gap must be nontrivial for a fail.

**D. public_category_like_baseline_recheck**

Recompute the public-category-like baseline (from 1J8 Diagnostic 3B) using peripheral-enhanced information. If this baseline jumps substantially from the original 0.5000, the peripheral signal has converted the non-deployable category shortcut into a deployable one.

Threshold: public_category_like macro_bal must not exceed 0.60 (original 0.5000 + 0.10 margin). If it exceeds 0.70, Route F is invalid.

**E. C0b_observe_only_under_RouteF**

Run C0b (observe-only, no probes) with the Route F peripheral interface. If C0b's macro_bal jumps substantially from the original 0.5871, peripheral information alone is resolving too much uncertainty — probes become redundant.

Threshold: C0b_RouteF macro_bal must not exceed 0.70. If it approaches oracle, Route F over-exposes information.

**F. oracle_gap_under_RouteF**

Compute the oracle-C0b gap under Route F. If this gap collapses below 0.10, there is not enough remaining uncertainty for selective probing to matter — Route F is invalid as a probing-evaluation interface.

Threshold: oracle - C0b_RouteF >= 0.10.

### Audit Decision Matrix

| Test | Pass Condition | Controls Required | Fail Implication |
|------|---------------|-------------------|------------------|
| A1: per-category subtype probe | acc ≤ base_rate + 0.10, AND gap < 0.05 | base_rate, CV, shuffle, real−shuffle gap | Peripheral leaks subtype — redesign blur |
| A2: majority/minority probe | acc ≤ base_rate + 0.10, AND gap < 0.05 | base_rate, CV, shuffle, real−shuffle gap | Peripheral encodes subtype typicality — investigate A1 |
| B: affordance probe | acc ≤ base_rate + 0.10, AND gap < 0.05 | base_rate, CV, shuffle, real−shuffle gap | Peripheral leaks affordance — redesign blur |
| C: category probe | acc ≤ 0.35, AND gap < 0.05 | base_rate, CV, shuffle, real−shuffle gap | Peripheral leaks category — redesign blur |
| D: public cat recheck | mb ≤ 0.60 | N/A (baseline comparison) | Shortcut became deployable — redesign or reject |
| E: C0b saturation | mb ≤ 0.70 | N/A (policy evaluation) | Observation too informative — increase blur |
| F: oracle gap | gap ≥ 0.10 | N/A (policy evaluation) | Probing has no room to matter — reject Route F |

All tests must pass for Route F to proceed to C15F policy evaluation in Block 1J11. If any test fails, the blur parameters or interface design must be revised and re-audited before any policy comparison.

### Leakage Test Reporting Template

Each leakage probe test in Block 1J10 must report:

```
Test: [name]
base_rate_accuracy:        [value]
cross_validated_accuracy:  [value]
shuffled_label_control:    [value]
real_minus_shuffle_gap:    [value]
exceeds_base_rate_plus_10: [true/false]
gap_is_nontrivial:         [true/false]
verdict:                   [PASS / FAIL / PASS-WITH-NOTE]
notes:                     [if overfitting suspected: "accuracy > base_rate+0.10 but gap < 0.05 — classifier overfitting, not genuine leakage"]
```

---

## 8. Marginal-Value Audit

After the leakage audit passes (Block 1J10, Part 1), a marginal-value audit (Block 1J10, Part 2) confirms that probing still has value under the Route F interface. This audit compares basic baselines to establish the new observation-conditioned performance landscape. Only if both parts pass will Block 1J11 proceed to C15F policy evaluation.

### Required Baselines

| Baseline | Description |
|----------|-------------|
| all_false | Predict False for all queries. Provides the 0.5000 floor. |
| C_minus_prior_only | Global base-rate predictions, no observation. Provides the prior-only floor. |
| C0b_observe_only_RouteF | Observe with peripheral information, never probe. Main no-probe baseline under Route F. |
| random_probe_budgeted_RouteF | Random probe selection under Route F interface. Tests whether probes still require intelligence. |
| C_oracle_full_information | Full ground-truth labels. Provides the 1.0000 ceiling (interface-independent). |

### Success Conditions

| Condition | Threshold | Rationale |
|-----------|-----------|-----------|
| C0b_RouteF improves over original C0b | C0b_RouteF > 0.5871 | Peripheral info should help, but not too much |
| C0b_RouteF does not saturate | C0b_RouteF ≤ 0.70 | Must leave room for probes to add value |
| Oracle-C0b gap remains meaningful | oracle - C0b_RouteF ≥ 0.10 | Enough residual uncertainty for probing to matter |
| random_probe does not trivially reach oracle | random_probe_mb ≤ oracle_mb - 0.15 | Probes require intelligence, not just budget |
| Public category shortcut remains controlled | public_cat_mb ≤ 0.60 | Same as leakage audit D |
| Peripheral info improves but probing still valuable | C0b_RouteF < oracle - 0.10 | The core hypothesis requires both observation AND probing |

If all success conditions are met, Route F provides a valid new observation interface where:
- Observation gives useful local context (C0b improves)
- Observation alone is not sufficient (C0b does not saturate)
- Probing still has meaningful value (oracle gap remains)
- Probe selection requires intelligence (random doesn't trivially win)
- Category shortcut is not exploitable (public cat stays low)

---

## 9. Policy Evaluation Plan (After Audits Pass)

Only if Block 1J10 passes both the leakage audit (Section 7) and marginal-value audit (Section 8), Block 1J11 will evaluate the policy:

**C15F_two_pass_with_blurred_peripheral_observation**

This policy uses the C15 two-pass architecture (nearest-first observe with reserve transition, VOI-gated probe pass) but with Route F peripheral information available during object selection and probe decisions.

### Comparison Baselines

All baselines use the Route F interface unless explicitly labeled as original-interface:

| Baseline | Interface | Description |
|----------|-----------|-------------|
| C0b_RouteF | Route F | Observe-only with peripheral info |
| C15b_original_interface | Original | C15b from 1J7 (reference point for improvement) |
| C15B_original_interface | Original | C15B from 1J7 (reference point, null result vs C15b) |
| random_probe_RouteF | Route F | Random probe selection with peripheral info |
| C_oracle_full_information | N/A | Full oracle (interface-independent ceiling) |

**Labeling rule:** baselines using the Route F interface must be clearly labeled with `_RouteF` suffix. Original-interface baselines must be clearly labeled with `_original_interface` suffix. Never mix interfaces without explicit labeling.

### Evaluation Metrics

Same as 1J7/1J8: macro_bal, pos_recall, neg_recall, visit_count, probe_count, normalized_cost, gap_capture, effective_probe_rate, diff_action_probe_rate, per-query breakdown.

### Required Comparisons

1. C15F_RouteF vs C0b_RouteF — does probing add value above peripheral observation alone?
2. C15F_RouteF vs C15b_original_interface — does the peripheral interface improve over the original?
3. C15F_RouteF vs C15B_original_interface — does Route F unlock value that Route B could not?
4. C15F_RouteF vs random_probe_RouteF — is the VOI gate still selective under the richer interface?
5. C15F_RouteF vs oracle — gap capture analysis

---

## 10. Decision Rules

### Route F Proceeds to C15F Policy Evaluation (Block 1J11) If:

- [ ] All six leakage audit tests pass (A-F, Section 7)
- [ ] Oracle-C0b gap under Route F remains ≥ 0.10
- [ ] C0b_RouteF does not saturate (≤ 0.70)
- [ ] Public category shortcut remains controlled (≤ 0.60)
- [ ] Route F has a clear, preregistered hypothesis tied to object selection and probe targeting
- [ ] Blur parameters are fixed before evaluation (not post-hoc tuned)

### Route F Is Rejected or Redesigned If:

- [ ] Any leakage audit test fails — peripheral signal exposes hidden state
- [ ] C0b_RouteF approaches oracle (gap < 0.10) — observation alone is sufficient
- [ ] Public category-like shortcut becomes too strong (≥ 0.70) — diagnostic risk became deployable
- [ ] Blur parameters are selected post-hoc based on C15F performance
- [ ] Peripheral signal quality depends on hidden state access

### If Route F Is Rejected:

Fallback options (not yet designed):
- Simpler observation enhancement (e.g., visit-count-based heuristics without peripheral features)
- Budget/cost redesign (Routes D/E)
- Interleaved breadth-depth without peripheral information
- Accept C15b as the practical ceiling under current constraints

---

## 11. Relation to MCTS / Search-Tree Framing

Route F is **not MCTS** (Monte Carlo Tree Search). It does not perform rollouts, backpropagation, or tree-policy learning.

Route F provides richer low-cost local observations that may later support search-tree-like policies:

| Concept | Route F Analogue |
|---------|-----------------|
| Breadth expansion | Peripheral local context (Layers 1-3) — cheap, lossy information about many objects |
| Depth expansion | Focal observe/probe (Layer 0) — expensive, precise information about one object |
| UCB/selection | Not implemented in Route F; would be a future policy on top of Route F interface |
| Rollout | Not applicable — Route F provides observations, not simulations |
| Backpropagation | Not applicable — IOM update is handled by existing instance memory, not tree search |

**Future interleaved policies** (breadth-depth, MCTS-inspired) may be considered:
- After Route F passes leakage and marginal-value audits (Route F provides the richest substrate), OR
- If Route F is rejected — an interleaved-only diagnostic (without peripheral information) may still be tested, using focal observations alone to guide breadth-vs-depth decisions

Route F is a likely useful substrate for later interleaved breadth-depth policies, but not a strict prerequisite. If Route F fails audits, an interleaved-only diagnostic may still be considered — testing whether interleaving alone (without peripheral information, using only focal observations to guide breadth-vs-depth decisions) can improve over the strict two-pass architecture.

---

## 12. Final Recommendation

| Decision | Value |
|----------|-------|
| implement_route_f_now | **false** |
| route_f_preregistered | **true** |
| next_block | **1J10** — Route F leakage + marginal-value audit |
| ready_for_multiseed | **false** |

**Route F is preregistered but NOT implemented.** Block 1J10 will implement the peripheral signal function and run the leakage audit (Section 7) and marginal-value audit (Section 8) on seed=101. Block 1J10 is the audit block only — it makes no C15F policy claim. Only if both audits pass will Block 1J11 implement and evaluate C15F.

The preregistration ensures that:
- Blur parameters are fixed before seeing any policy performance
- Leakage tests are defined before the peripheral function exists
- Success/failure criteria are objective and pre-specified
- There is no post-hoc parameter tuning or threshold adjustment

---

*Generated by Block 1J9 — Route F blurred peripheral observation preregistration. No implementation. No simulation.*
