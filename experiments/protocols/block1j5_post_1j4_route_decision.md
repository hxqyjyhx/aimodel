# Block 1J5: Post-1J4 Route Decision Checkpoint

**Date:** 2026-05-10
**Block:** 1J5 — Route decision checkpoint
**Condition:** C4_instance_subtype_cued_v1
**Based on:** Block 1J4 failure decomposition

---

## 1. Accepted Findings from 1J4

| # | Finding | Detail |
|---|---------|--------|
| 1 | C13 loses to C0b overall | C13 macro_bal=0.5733, C0b macro_bal=0.5871, C0b Pareto-dominates C13 |
| 2 | C13 is better on visited objects | C13 visited-subset macro_bal=0.7163 vs C0b same-subset=0.5570 |
| 3 | posterior_prediction_bias is NOT the problem | C13 outperforms C0b on every query when restricted to visited objects |
| 4 | Main diagnosis | visit_breadth_loss + poor_differentiating_action_selection + poor_minority_subtype_targeting + persistent_negative_only_tradeoff + environment_category_shortcut_risk |
| 5 | Visit breadth | 22/60 objects visited. 82.5% of C13 false negatives from unvisited positives |
| 6 | Action relevance | 40.9% of probes target differentiating actions. 59.1% are non-differentiating |
| 7 | Minority targeting | 2/12 minority objects visited. Enrichment ratio=1.0. No preferential targeting |
| 8 | C13 beats random | +0.0308 macro_bal at same cost. Real but weak action-selection signal |
| 9 | category_majority is diagnostic only | 0.9750 is a leakage-risk upper bound, not a deployable policy |

---

## 2. Route Comparison

### Route A: Breadth-Preserving Policy Redesign

**Candidate:** C15_two_pass_observe_then_probe

**What failure mechanism it addresses:** visit_breadth_loss (primary). C13 visits too few objects because probe costs consume budget that could be spent on observation.

**Why it is justified by 1J4:**
- C13 achieves 0.7163 macro_bal on visited objects vs C0b's 0.5570 on the same subset. The probing works — it just can't compensate for 38 unvisited objects.
- 82.5% of false negatives are unvisited objects. Any fix must address breadth.
- C0b visits all 60 objects at cost 1.13 (ncost 0.7533), leaving 0.37 budget unused. This residual budget could fund selective probing after broad observation.

**Design sketch:**
1. **Pass 1 (observe):** Visit objects nearest-first, observe only (no probes), until all objects are observed or probe-reserve budget threshold is reached.
2. **Pass 2 (probe):** Rank observed objects by post-observation uncertainty or predicted VOI. Visit (re-reach) and probe the highest-value objects using remaining budget.
3. Probe gate: same C13 VOI logic, but operating on a shortlist of already-observed high-uncertainty objects.

**What would count as success:**
- C15 macro_bal > C0b macro_bal (0.5871) at acceptable cost
- C15 positive recall > C0b positive recall (0.3100)
- C15 maintains or improves negative recall vs C0b (0.8642)
- C15 probe coverage selective (not all objects probed)

**What would count as invalid / metric gaming:**
- C15 probes nearly all objects (defeats the purpose of selectivity)
- C15 requires hidden_category or ground-truth information
- C15 achieves success only by exploiting negative-only tradeoff
- Budget constraint is effectively bypassed (e.g., "observe all then probe as many as we want")

**Recommendation:** **IMPLEMENT NOW as primary route.** This directly addresses the dominant failure mechanism with a principled structural change: separate the observation decision from the probe decision. The two-pass design is a natural generalization of C13 — it preserves VOI-gated probing but removes the forced visit-probe coupling.

---

### Route B: Better Differentiating-Action Selection

**Candidate:** subtype-diagnostic action VOI / cross-action VOI

**What failure mechanism it addresses:** poor_differentiating_action_selection. Only 40.9% of C13 probes target the differentiating action for the object's category.

**Why it is justified by 1J4:**
- 59.1% of probes are on non-differentiating actions. Of those, 61.5% are zero-gain.
- When C13 does probe the differentiating action, effectiveness is 66.7% (vs 38.5% for non-differentiating).
- C13 beats random primarily through better action selection (40.9% diff-action rate vs 22.7%).
- Improving action selection could raise the effective probe rate and improve macro_bal even at the same visit count.

**Design sketch:**
1. Per category, identify which actions vary between subtypes (requires structural knowledge or learned from training data).
2. Modify VOI computation to consider cross-action inference: how probing action A changes beliefs about the cluster of query-relevant affordances, not just about action A itself.
3. Weight differentiating actions higher in the VOI ranking, or add a penalty term for non-differentiating actions.

**What would count as success:**
- Differentiating-action probe rate substantially above 40.9% (e.g., >60%)
- Effective probe rate improves
- macro_bal improves at same or lower cost

**What would count as invalid / metric gaming:**
- Hard-coding the differentiating action per category from ground truth (oracle leakage)
- Using hidden_category to look up the differentiating action at probe time
- Achieving high diff-action rate only by probing fewer objects (trivial if you probe 1 object and happen to pick the right action)

**Recommendation:** **IMPLEMENT as secondary route.** Action selection is the second-largest failure after breadth. Even with Route A's two-pass design, probes on non-differentiating actions waste budget. The two improvements compound: Route A fixes which objects get probed; Route B fixes what gets probed on each object.

---

### Route C: Better Minority-Subtype Object Targeting

**Candidate:** visible-cue outlier targeting / cluster uncertainty / likely-minority detector

**What failure mechanism it addresses:** poor_minority_subtype_targeting. C13 visits only 2/12 minority objects and has no enrichment over random.

**Why it is justified by 1J4:**
- Minority enrichment ratio = 1.0. C13 has zero preference for minority objects.
- wood_log and wooden_pickaxe minorities (3 each) are never visited at all.
- Minority objects are where probing should add the most value — majority objects are already well-predicted by category-level information.
- Probe value by subtype: majority probes are 55% effective. The 2 minority probes were both zero-gain (insufficient sample).

**Design sketch:**
1. After observing an object, compute a "minority-likelihood score" based on how much the visible features deviate from the majority-subtype prototype for that category.
2. Use this score to rank objects for selective probing in a two-pass design.
3. Alternatively: compute the expected VOI of probing the differentiating action conditioned on the visible features — objects where the posterior is near 0.5 on the differentiating feature are candidates.

**What would count as success:**
- Minority visit or probe rate substantially above the population base rate (minority_enrichment > 1.5)
- More than 2/12 minority objects probed
- Effective probe rate on minority objects improves

**What would count as invalid / metric gaming:**
- Using hidden_subtype directly to find minority objects (oracle leakage)
- Targeting minority objects by any mechanism that requires knowing subtype before probing
- Achieving enrichment only by chance on a single seed

**Recommendation:** **DEFER.** Minority targeting is a real issue, but it is largely a consequence of the visit-breadth problem. If Route A enables observing all 60 objects, the post-observation uncertainty ranking will naturally surface minority objects (they will have higher prediction uncertainty after observation). Route C may emerge as a natural consequence of Route A + B without needing a separate mechanism. Re-evaluate after Route A results.

---

### Route D: Environment Redesign v2

**Candidate:** C5_balanced_subtypes_or_cross_category_queries

**What failure mechanism it addresses:** environment_category_shortcut_risk. category_majority = 0.9750 shows C4 has a strong latent category shortcut.

**Why it is justified by 1J4:**
- category_majority achieves 0.9750 using only hidden_category (not available to agent).
- The 0.025 gap to oracle is small — probing can add at most 0.025 macro_bal over category-level knowledge.
- C4's 80/20 subtype ratio means 80% of objects are correctly predicted by category majority alone.
- A harder environment (balanced subtypes, cross-category queries) would make probing more necessary and provide a clearer signal for policy comparison.

**Design sketch:**
1. Create C5 with 50/50 subtype ratios per category (or remove majority entirely).
2. Add cross-category query patterns where knowing one object's subtype informs another object's query answer.
3. Reduce or remove visible cue differential between subtypes to make observation-only harder.

**What would count as success:**
- category_majority macro_bal drops significantly (e.g., below 0.70)
- C0b macro_bal remains in the 0.50-0.65 range (observation provides signal but not saturation)
- Oracle gap remains large (>0.20)
- C13 or its successor has room to demonstrate probe value

**What would count as invalid / metric gaming:**
- Designing C5 specifically to make C13 win (anti-tuning safeguard from 1J0)
- Changing the environment without pre-registering the design before running policies
- Iterating environment parameters until C13 looks good

**Recommendation:** **DEFER.** Environment redesign is a valid long-term concern — C4's category shortcut is real — but it should not be the first response to C13's failure. The failure mechanisms identified in 1J4 (breadth, action selection, minority targeting) are policy-design problems, not environment-design problems. Fix the policy first. If Route A + B still cannot beat C0b after addressing breadth and action selection, then environment redesign becomes more justified. When/if Route D is pursued, the new environment design must be pre-registered before any policy runs on it.

---

### Route E: Cost/Budget Redesign

**Candidate:** separate observe/probe budgets, lower probe cost, larger total budget

**What failure mechanism it addresses:** visit_breadth_loss (indirectly, by making the visit-probe tradeoff less severe).

**Why it is justified by 1J4:**
- Current budget 1.5 forces C13 to visit only 22/60 objects (probe_cost=0.05 dominates reach+observe≈0.016).
- C0b visits all 60 objects at cost 1.13, leaving 0.37 unused — but C0b never probes.
- A larger budget or cheaper probes would let C13 visit more objects, reducing the breadth gap.

**Design sketch:**
1. Increase total budget (e.g., 2.5 or 3.0) to allow more visits.
2. Or: separate the budget into observation_budget and probe_budget, requiring the policy to manage both.
3. Or: reduce probe_cost from 0.05 to 0.02-0.03 to reduce the per-object cost of probing.

**What would count as success:**
- C13 visits more objects (e.g., 35-45 out of 60)
- C13 macro_bal improves and potentially beats C0b
- The visit-probe tradeoff becomes less extreme

**What would count as invalid / metric gaming:**
- Tuning budget until C13 wins without addressing the underlying design issues
- Making probe_cost so low that probing is essentially free (removes the selectivity challenge)
- Changing costs post-hoc to make a failing policy look good

**Recommendation:** **DEFER.** Cost redesign is a calibration concern, not a policy-design concern. The current budget and costs were set during the initial Mini-MC v0 design and validated by Block 1J2's marginal-value audit. Changing them now, before attempting any policy redesign, would be premature optimization. If Route A + B still show promise but are budget-constrained, a principled budget recalibration could follow. Any cost changes must be pre-registered before running policies.

---

### Route F: Multi-Layer Blurred Peripheral Observation Interface

**Candidate:** C16_multi_layer_blurred_observation

**What failure mechanism it addresses:** visit_breadth_loss (indirectly), poor_minority_subtype_targeting, lack of local scene/context distribution.

**Why it is justified by 1J4:**
- Current observation is point-like: observing one object returns only that object's visible features. The agent gets no information about nearby objects, the local category mix, or whether the immediately reachable area is majority-dominated or contains minority candidates.
- C13 fails partly because narrow focal probing loses scene coverage — it visits objects one at a time with no awareness of what surrounds them.
- Multi-layer observation would let the agent estimate the local positive/negative sample balance and identify better candidate objects for focused probing without having to visit every object first (which C0b does blindly).
- This is an interface-level design change, not a policy-design change: it enriches AgentObsView without touching policy internals.

**Design sketch:**
1. When observing a focal object, additionally expose coarse/blurred information about nearby objects within a configurable radius.
2. The blur controls information amount: distant objects return fuzzier signals; very distant objects return no signal.
3. Blurred information could include:
   - Distance-decayed local category/cue distribution (e.g., "there are 3±2 wood_log-like objects within radius 2")
   - Binarized or noised visible cues for nearby (but not focal) objects
   - Aggregate statistics: "this region is 60±15% majority-subtype objects"
4. Hard constraints on what blurred observation MUST NOT expose:
   - No true affordance labels (probe outcomes)
   - No hidden subtype labels
   - No deterministic identification of minority objects at distance

**What would count as success:**
- C0b improves moderately but does not saturate (macro_bal remains well below oracle)
- C_oracle - C0b gap remains >= 0.10 (observation alone does not solve the task)
- No true label or hidden subtype leakage through the blurred channel
- Policy can use local context to improve object selection (e.g., minority enrichment > 1.0, more informed visit ordering)
- Visit breadth improves without requiring full observation of all objects

**What would count as invalid / metric gaming:**
- Blurred observation reveals hidden subtype directly or with near-certainty
- Local context makes the category_majority shortcut even stronger (C0b jumps to ~0.90+)
- Observation alone reaches near-oracle accuracy and removes probe value entirely
- Blur radius is tuned per-seed to favor specific policies

**Recommendation:** **DEFER for now.** Keep as interface-level Route F to revisit after C15 (Route A) and C15+B (Route A+B) results. If 1J6/1J7 show that point-like observation remains a bottleneck even after fixing visit-probe coupling and action selection — i.e., the two-pass design still cannot close the breadth gap efficiently — then blurred peripheral observation becomes the next structural fix. This is an observation-interface change, not an environment redesign: it enriches what the agent sees without changing ground truth.

---

## 3. Search-Tree / MCTS-Inspired Framing

Mini-MC can be framed as cost-constrained search over an object-action decision tree. This is a design analogy, not a claim that any current or planned policy implements MCTS:

| Search Concept | Mini-MC Analogue |
|----------------|------------------|
| **Breadth expansion** | Observe more objects / expand local scene coverage |
| **Depth expansion** | Probe selected object-action branches to reduce subtype uncertainty |
| **Value estimate** | Expected query-level gain (macro_bal improvement) from visiting+probing |
| **Rollout / leaf evaluation** | IOM posterior prediction given current observations and probe outcomes |
| **Stopping criterion** | Budget-aware termination — stop when no remaining object has positive expected net VOI |
| **Exploration-exploitation** | Visit new objects (explore) vs probe already-visited objects deeper (exploit) |
| **Search budget** | Total cost budget (reach + observe + probe) |

The two-pass design (Route A / C15) maps naturally to this framing:
- **Pass 1 (breadth-first):** Expand observation breadth cheaply — visit many objects, observe only, no probes. Analogous to building a wide search frontier.
- **Pass 2 (depth-selective):** On the observed frontier, select high-value nodes for depth expansion via probing. Analogous to selective deep rollouts on promising branches.

Current C13 is depth-biased: it spends budget on probe depth for a small set of objects, leaving most of the search space unexplored. C0b is breadth-only: it sees all objects but never probes depth. C15 aims for breadth-then-depth balance within a single budget envelope.

This framing may inform future route design (e.g., interleaved breadth-depth, UCB-style object selection, explicit value-of-information bandits) but no MCTS implementation is planned for the immediate next blocks.

---

## 4. Route Comparison Matrix

| Dimension | Route A (Breadth) | Route B (Action) | Route C (Minority) | Route D (Env) | Route E (Cost) | Route F (Blur) |
|-----------|-------------------|------------------|--------------------|---------------|----------------|-----------------|
| Addresses primary failure? | **Yes** (breadth) | Partial (action) | Partial (targeting) | No (env risk) | Indirect | Indirect (breadth) |
| Justified by 1J4 data? | **Strong** | Strong | Moderate | Weak | Weak | Moderate |
| Risk of metric gaming? | Low | Moderate | Moderate | **High** | **High** | Moderate |
| Requires env change? | No | No | No | **Yes** | **Yes** | No (interface) |
| Complements Route A? | — | **Yes** | Yes | N/A | N/A | **Yes** |
| Implementation complexity | Moderate | Low-Moderate | Moderate | High | Low | Moderate-High |
| Timing | **Now** | **Now (secondary)** | Defer | Defer | Defer | Defer (post-A+B) |

---

## 5. Final Recommendation

### Primary Next Route: **Route A — Breadth-Preserving Two-Pass Design (C15)**

C13's dominant failure is visiting too few objects (82.5% of FNs from unvisited positives). The two-pass design directly addresses this: observe broadly first (like C0b), then probe selectively on high-uncertainty objects. This preserves C13's demonstrated strength on visited objects (0.7163 vs C0b's 0.5570) while removing the forced visit-probe coupling that causes the breadth loss.

### Secondary Next Route: **Route B — Differentiating-Action Selection**

Even with better object coverage from Route A, 59.1% of probes landing on non-differentiating actions wastes budget. Improving action selection to prefer subtype-diagnostic actions compounds with the breadth fix: more objects get probed AND probes are more likely to be informative.

### Deferred Routes

- **Route C (minority targeting):** Likely addressed as a side effect of Route A + B. Re-evaluate after A+B results.
- **Route D (environment redesign):** Valid concern but premature. Fix policy design first. Pre-register before any env changes.
- **Route E (cost redesign):** Premature. Current costs/budget were validated by 1J2. Revisit if A+B show promise but are budget-constrained.
- **Route F (blurred peripheral observation):** Deferred for now. Keep as interface-level Route F to revisit after C15 and C15+B results. If 1J6/1J7 show that point-like observation remains a bottleneck even after fixing visit-probe coupling and action selection, blurred peripheral observation becomes the next structural fix.

### Search-Tree / MCTS-Inspired Framing

Mini-MC is framed as cost-constrained search over an object-action decision tree (see Section 3 for full mapping). This is a design analogy, not a claim that any current or planned policy implements MCTS. The two-pass design (Route A) maps naturally to breadth-first frontier expansion followed by selective depth expansion.

### Explicit Constraints

- **Do NOT run multi-seed yet.** Single-seed (seed=101) is sufficient for initial Route A+B evaluation.
- **Do NOT treat original C13 as promising.** C13 is Pareto-dominated by C0b and should not proceed to multi-seed in its current form.
- **Do NOT change the environment.** C4_instance_subtype_cued_v1 remains the test environment. Route D is deferred.
- **Implement Route A first, then layer Route B.** Evaluate each increment separately to isolate effects.

---

## 6. Implementation Preview (Block 1J6)

Block 1J6 will implement C15_two_pass_observe_then_probe:

1. **Phase 1 (observe pass):** Visit objects nearest-first, observe only (no probes), until either:
   - All objects are observed, or
   - Remaining budget drops below a probe-reserve threshold (e.g., enough for K probes)
2. **Phase 2 (probe pass):** Rank already-observed objects by post-observation VOI (using C13's VOI gate). Visit and probe the highest-VOI objects until budget exhausted.
3. **Probe gate:** Same C13 VOI logic (net_voi > 0), applied to already-observed objects.

If Route A alone still doesn't beat C0b, Block 1J7 will layer Route B (differentiating-action VOI) on top.

---

*Generated by Block 1J5 — route decision checkpoint, no simulations, no implementation.*
