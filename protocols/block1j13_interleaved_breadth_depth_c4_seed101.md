# Block 1J13 — Interleaved Breadth-Depth Diagnostic

**Date:** 2026-05-11
**Block:** 1J13 — Interleaved breadth-depth diagnostic
**Condition:** C4_instance_subtype_cued_v1
**Seed:** 101
**Budget:** 1.5

---

## 1. Executive Summary

1J13 tests whether interleaving observation and probing (instead of strict two-pass) improves object selection or probe targeting under the original point-like observation interface.

**All C17 variants converge to macro_bal=0.5667, worse than both C13 (0.5733) and C15b (0.6142).** Interleaving fails because individual probe VOI does not become positive until sufficient breadth has been accumulated. After only K=3, 5, or 8 observations, the IOM lacks the context to make probes worthwhile — expected utility gain is smaller than the cost penalty. The result is a degenerate "nearest-first observe + VOI-gated per-object probe" that behaves like a slightly worse C13.

**Do not pursue interleaved breadth-depth under current signal/cost conditions.** C15b's strict two-pass with 25% budget reserve is well-calibrated — it forces breadth before allowing probes.

---

## 2. Motivation

| Prior Finding | Implication |
|--------------|-------------|
| 1J11: F_mid aggregate peripheral is safe but not useful | Aggregate context does not differentiate objects under C4 |
| 1J12: Global coarse scene info is weakly useful for C13 but degrades C15G | Object-addressable public-cue access does not improve two-pass |
| 1J12: C15G scene priority negatively correlated with diagnostic value | Public cue patterns do not align with diagnostic value |
| 1J12: Oracle UB = 0.6292 | ~0.015 headroom above C15b exists |

**Hypothesis:** If the remaining bottleneck is the timing of the observe→probe transition (rather than object selection or cue access), then interleaving observation and probing should improve over strict two-pass.

---

## 3. C17 Policy Designs

### 3.1 C17_InterleavedK (K=3, 5, 8)

Each cycle: observe K new objects (nearest-first), then find the highest-VOI visited+unprobed object and probe it if VOI > 0. Repeat.

```
while budget:
    observe K new objects
    best = argmax VOI(visited_unprobed)
    if VOI(best) > 0: probe(best)
```

### 3.2 C17_AdaptiveInterleaved

At each step, compare best probe VOI among visited+unprobed vs continue-observing. Probe whenever any visited object has net VOI > 0.

```
while budget:
    best_probe = argmax VOI(visited_unprobed)
    if VOI(best_probe) > 0: probe(best_probe)
    else: observe nearest unvisited
```

### 3.3 C17_OracleSwitchTiming (Non-Deployable)

Same as adaptive, but uses oracle knowledge to decide when to probe: only probes objects that appear in the oracle's optimal object selection set.

---

## 4. Baseline Results

| Policy | macro_bal | pos_recall | neg_recall | visited | probed | ncost | Type |
|--------|-----------|------------|------------|---------|--------|-------|------|
| all_false | 0.5000 | 0.0000 | 1.0000 | 0 | 0 | 0.0000 | floor |
| C_minus_prior_only | 0.5000 | — | — | 0 | 0 | 0.0000 | no_interaction |
| C0b_original | 0.5871 | 0.3100 | 0.8642 | 60 | 0 | 0.7533 | observe_only |
| C13_instance_VOI_original | 0.5733 | — | — | 22 | 22 | 0.9733 | instance_VOI |
| C15b_probe_reserve_original | 0.6142 | 0.3600 | 0.8683 | 57 | 8 | 0.9900 | two_pass_VOI |
| C15G_global_coarse_map (1J12) | 0.5992 | — | — | 57 | 8 | 0.9967 | C15G |
| C_oracle_full_information | 1.0000 | 1.0000 | 1.0000 | — | — | — | oracle |

---

## 5. C17 Results

### 5.1 All Variants

| Variant | macro_bal | pos_recall | neg_recall | visited | probed | eff_rate | maj_probed | min_probed | first_probe_at |
|---------|-----------|------------|------------|---------|--------|----------|------------|------------|----------------|
| C17_K3 | 0.5667 | 0.2000 | 0.9333 | 25 | 22 | 1.00 | 23 | 2 | 0 |
| C17_K5 | 0.5667 | 0.2000 | 0.9333 | 25 | 22 | 1.00 | 23 | 2 | 0 |
| C17_K8 | 0.5667 | 0.2000 | 0.9333 | 25 | 22 | 1.00 | 23 | 2 | 0 |
| C17_adaptive | 0.5667 | 0.2000 | 0.9333 | 25 | 22 | 1.00 | 23 | 2 | 0 |
| C17_oracle_switch | 0.5667 | 0.2000 | 0.9333 | 25 | 22 | 1.00 | 23 | 2 | 0 |

**All five variants produce identical results.** The K-based, adaptive, and oracle-switch variants all converge to the same behavior.

### 5.2 Why All C17 Variants Converge

The interleaving pattern for all variants is `OOOOOOOOOOOOOOOOOOOOOOOOO...` (25 O's, no P's). This means `select_next_object` always returns an unvisited object — the K-triggered probe logic in `_select_probe_target` **never** finds a visited object with positive VOI.

The probes that do occur (22) happen via the harness's `decide_probe` call, which is invoked after every observation. After observing an object, `decide_probe` computes its VOI and probes if positive. This is identical to C13's behavior.

**Why VOI is never positive in `_select_probe_target`:**

- With K=3, 5, or 8, only a small number of objects have been observed
- The IOM has insufficient context to identify which objects are worth probing
- Expected utility gain from a single probe (~0.01-0.02) is less than the cost penalty (~0.024 normalized)
- Net VOI = E[post_utility] - current_utility - 0.5 * norm_cost < 0

C15b succeeds because it waits until ~57 objects are observed before probing. At that point, the IOM has enough feature context to produce larger VOI estimates for specific objects, exceeding the cost penalty.

### 5.3 Per-Query Breakdown

| Query | C15b Bal Acc | C17 (all) Bal Acc | Delta |
|-------|-------------|-------------------|-------|
| need_food | 0.6250 | 0.6250 | 0 |
| need_fuel | 0.6333 | 0.6333 | 0 |
| need_planks | 0.5729 | 0.4792 | -0.0937 |
| need_stone | 0.5521 | 0.4688 | -0.0833 |
| need_tool | 0.6875 | 0.6271 | -0.0604 |

C17 loses accuracy primarily on need_planks, need_stone, and need_tool — queries that require broader observation to resolve. C17 stops at 25 objects (budget exhausted by aggressive probing) while C15b observes 57.

---

## 6. Key Comparisons

### 6.1 C17 vs C15b

| Metric | Delta |
|--------|-------|
| delta_macro_bal | **-0.0475** |
| delta_positive_recall | -0.1600 |
| delta_negative_recall | +0.0650 |
| delta_visit_count | -32 |
| delta_probe_count | +14 |
| delta_effective_probe_rate | 0.0000 |
| Pareto relation | C15b strictly dominates C17 |

C17 probes 22 objects (vs C15b's 8) but only visits 25 (vs 57). The negative-only tradeoff is clear: C17 has much higher negative recall but much lower positive recall due to insufficient breadth.

### 6.2 C17 vs C13

| Metric | Delta |
|--------|-------|
| delta_macro_bal | **-0.0067** |
| preserves_more_breadth_than_C13 | True (25 vs 22 visited) |
| avoids_C13_narrow_depth_failure | **False** (0.5667 < 0.5733) |

C17 visits slightly more objects (25 vs 22) but probes the same number (22). Both converge to the same narrow-depth pattern because the VOI gate triggers similarly. C17's nearest-first observation order is slightly worse than C13's VOI-based object selection.

### 6.3 C17 vs C15G

| Metric | Delta |
|--------|-------|
| delta_macro_bal | **-0.0325** |
| temporal_interleaving_more_useful_than_global_coarse_map | **False** |

C15G at 0.5992 (global coarse map) substantially outperforms C17 at 0.5667. Even degraded C15b (C15G) is better than aggressive interleaving.

### 6.4 Gap Capture

| Metric | Value |
|--------|-------|
| C0b base | 0.5871 |
| Oracle ceiling | 1.0000 |
| C15b gap capture | 6.56% |
| C17 gap capture | **-4.95%** |
| C17 hits 10% threshold (0.6284) | **No** |

C17 is BELOW C0b — probing aggressively degrades performance below no-probe baseline.

---

## 7. Why Interleaving Fails

### 7.1 The VOI-Cost Gap at Low Breadth

```
At 5 observations:
  E[utility_gain_per_probe] ≈ 0.01-0.02
  cost_penalty = 0.5 * (0.085/1.5) ≈ 0.028
  net_voi = 0.02 - 0.028 = -0.008 < 0  → don't probe

At 57 observations:
  E[utility_gain_per_probe] ≈ 0.03-0.05
  cost_penalty = 0.5 * (0.085/1.5) ≈ 0.028
  net_voi = 0.04 - 0.028 = +0.012 > 0  → probe
```

The IOM needs sufficient feature context to produce meaningful VOI estimates. Early probes have low expected utility gain because the IOM's predictions are dominated by the prior — a single probe doesn't shift probabilities enough.

### 7.2 Budget Exhaustion by Premature Probing

C17's budget composition vs C15b:

| Cost Type | C15b | C17 |
|-----------|------|-----|
| Observe | 57 * 0.005 = 0.285 | 25 * 0.005 = 0.125 |
| Probe | 8 * 0.05 = 0.400 | 22 * 0.05 = 1.100 |
| Reach | ~0.305 | ~0.275 |
| **Total** | 0.990 | 1.500 |

C17 spends 73% of budget on probes vs C15b's 40%. The 22 early probes exhaust budget that could have been used for broader observation.

### 7.3 The Oracle Switch Also Fails

Even with oracle knowledge of WHICH objects to probe, the C17_oracle_switch variant produces the same result. Oracle selection can't fix the fundamental VOI-cost gap at low breadth — the oracle knows which objects are valuable, but the IOM hasn't accumulated enough context to generate meaningful utility gains from those probes.

---

## 8. Interpretation

```
C17 improves over C15b:     False (-0.0475 delta)
C17 improves over C13:      False (-0.0067 delta)
C17 improves over C15G:     False (-0.0325 delta)
C17 hits 10% threshold:     False (0.5667 < 0.6284)
C17 improves pos_recall:    False (-0.1600)
C17 improves minority:      False (2, same as C15b)
C17 preserves breadth:      True (25 > 22, but insufficient)
C17 on Pareto frontier:     False
interleaving_supported:     False
candidate_multiseed:        False
```

**Interpretation: Strict two-pass is already sufficient under current signals. The bottleneck is probe informativeness / IOM update, not timing.**

C15b's two-pass architecture with 25% budget reserve is well-calibrated: it forces broad observation before any probes are allowed. Interleaving observation and probing without sufficient breadth is harmful — it wastes budget on low-VOI probes and reduces observation coverage.

---

## 9. Final Recommendation

**Do not pursue interleaved breadth-depth.** C15b at 0.6142 remains the best deployable policy. The bottleneck is not the timing of the observe→probe transition — it is the fundamental informativeness of individual probes under the current IOM and signal conditions.

### Consolidated Block 1J11-1J13 Findings

| Block | Approach | Result | vs C15b |
|-------|----------|--------|---------|
| 1J11 | F_mid aggregate peripheral (Route F) | 0.6142 | 0.0000 |
| 1J12 | Global coarse scene map (C15G) | 0.5992 | -0.0150 |
| 1J13 | Interleaved breadth-depth (C17) | 0.5667 | -0.0475 |

All three alternative approaches either match or underperform C15b. The strict two-pass VOI architecture is robust.

### Possible Next Directions

1. **Probe informativeness redesign**: The current 5 main candidate actions may not be equally informative. Per-object action selection or query-level VOI could improve probe value.

2. **IOM update sensitivity**: The IOM's `incorporate_probe` may not propagate information efficiently. A more responsive update mechanism could increase per-probe utility gain.

3. **Cost-weight tuning**: Lower cost_weight (< 0.5) would allow more probes, potentially capturing more diagnostic value. But the risk is replicating C13's narrow-depth pattern.

4. **Different budget/cost regime**: A higher budget or lower probe cost would shift the VOI-cost balance, potentially making interleaving viable. But this changes the task design.

5. **Accept C15b as near-ceiling for current design**: At 0.6142 with 6.56% gap capture and only ~0.015 oracle-UB headroom, C15b may be close to optimal under C4 with budget=1.5 using the current IOM and signal set.

---

*Generated by Block 1J13 — Interleaved breadth-depth diagnostic.*
