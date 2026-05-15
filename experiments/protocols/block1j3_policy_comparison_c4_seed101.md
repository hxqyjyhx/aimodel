# Block 1J3: Single-Seed Policy Comparison — C4_instance_subtype_cued_v1

**Date:** 2026-05-10
**Block:** 1J3 — Single-seed Pareto policy comparison
**Condition:** C4_instance_subtype_cued_v1
**Seed:** 101
**Budget:** 1.5
**C13 Cost Weight:** 0.5

---

## 1. Configuration

| Parameter | Value |
|-----------|-------|
| Condition | C4_instance_subtype_cued_v1 |
| Seed | 101 |
| Budget | 1.5 |
| C13 cost_weight | 0.5 |
| Primary metric | macro_query_balanced_accuracy |
| Reporting mode | Pareto: macro_bal vs normalized_cost |
| IOM | k=10, nb_likelihood |
| Training coverage | 0.5714 (640 outcome rows) |
| Pre-accuracy | 0.4625 |

## 2. All Policy Results

| Policy | macro_bal | mean_pos_r | mean_neg_r | visits | probes | total_cost | ncost |
|--------|-----------|------------|------------|--------|--------|------------|-------|
| all_false | 0.5000 | 0.0000 | 1.0000 | 0 | 0 | 0.0000 | 0.0000 |
| C_minus_prior_only | 0.5000 | 0.0000 | 1.0000 | 0 | 0 | 0.0000 | 0.0000 |
| C0b_observe_only | **0.5871** | 0.3100 | 0.8642 | 60 | 0 | 1.1300 | 0.7533 |
| random_probe_budgeted | 0.5425 | 0.1500 | 0.9350 | 22 | 22 | 1.4500 | 0.9667 |
| **C13_instance_VOI** | **0.5733** | 0.2000 | 0.9467 | 22 | 22 | 1.4500 | 0.9667 |
| category_majority | 0.9750 | 1.0000 | 0.9500 | 0 | 0 | 0.0000 | 0.0000 |
| C_oracle_full_info | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0.0000 | 0.0000 |

## 3. Per-Query Detail

### all_false

| Query | pos_recall | neg_recall | bal |
|-------|-----------|-----------|-----|
| need_food | 0.0000 | 1.0000 | 0.5000 |
| need_fuel | 0.0000 | 1.0000 | 0.5000 |
| need_planks | 0.0000 | 1.0000 | 0.5000 |
| need_stone | 0.0000 | 1.0000 | 0.5000 |
| need_tool | 0.0000 | 1.0000 | 0.5000 |

### C_minus_prior_only

Identical to all_false. IOM global base rates predict below 0.5 for all query-determining features.

| Query | pos_recall | neg_recall | bal |
|-------|-----------|-----------|-----|
| need_food | 0.0000 | 1.0000 | 0.5000 |
| need_fuel | 0.0000 | 1.0000 | 0.5000 |
| need_planks | 0.0000 | 1.0000 | 0.5000 |
| need_stone | 0.0000 | 1.0000 | 0.5000 |
| need_tool | 0.0000 | 1.0000 | 0.5000 |

### C0b_observe_only

Visits all 60 objects. Observes visible features. No probes.

| Query | pos_recall | neg_recall | bal |
|-------|-----------|-----------|-----|
| need_food | 0.1667 | 1.0000 | 0.5833 |
| need_fuel | 0.6333 | 0.6333 | 0.6333 |
| need_planks | 0.2500 | 0.8125 | 0.5312 |
| need_stone | 0.0833 | 0.9375 | 0.5104 |
| need_tool | 0.4167 | 0.9375 | 0.6771 |

### random_probe_budgeted

Visits 22 nearest affordable objects, probes random action on each.

| Query | pos_recall | neg_recall | bal |
|-------|-----------|-----------|-----|
| need_food | 0.3333 | 1.0000 | 0.6667 |
| need_fuel | 0.1667 | 0.8000 | 0.4833 |
| need_planks | 0.0833 | 0.9167 | 0.5000 |
| need_stone | 0.0833 | 0.9792 | 0.5312 |
| need_tool | 0.0833 | 0.9792 | 0.5312 |

### C13_instance_VOI_original (CW=0.5)

Visits 22 objects (same count as random), probes via VOI gate.

| Query | pos_recall | neg_recall | bal | vs C0b |
|-------|-----------|-----------|-----|--------|
| need_food | 0.4167 | 1.0000 | 0.7083 | +0.1250 |
| need_fuel | 0.1667 | 0.9000 | 0.5333 | -0.1000 |
| need_planks | 0.1667 | 0.8542 | 0.5104 | -0.0208 |
| need_stone | 0.1667 | 1.0000 | 0.5833 | +0.0729 |
| need_tool | 0.0833 | 0.9792 | 0.5312 | -0.1459 |

### category_majority

| Query | pos_recall | neg_recall | bal |
|-------|-----------|-----------|-----|
| need_food | 1.0000 | 0.9375 | 0.9688 |
| need_fuel | 1.0000 | 1.0000 | 1.0000 |
| need_planks | 1.0000 | 0.9375 | 0.9688 |
| need_stone | 1.0000 | 0.9375 | 0.9688 |
| need_tool | 1.0000 | 0.9375 | 0.9688 |

### C_oracle_full_information

All queries at 1.0000 (perfect affordance knowledge).

## 4. Key Comparisons

### C13 vs C0b

| Metric | C13 | C0b | Delta |
|--------|-----|-----|-------|
| macro_bal | 0.5733 | 0.5871 | **-0.0138** |
| normalized_cost | 0.9667 | 0.7533 | +0.2133 |
| mean_pos_recall | 0.2000 | 0.3100 | -0.1100 |
| mean_neg_recall | 0.9467 | 0.8642 | +0.0825 |
| visits | 22 | 60 | -38 |

**Pareto: C0b_dominates_C13.** C0b has higher macro_bal (+0.0138) AND lower cost (-0.2133 ncost). C13 is strictly worse on both dimensions.

### C13 vs random_probe

| Metric | C13 | random | Delta |
|--------|-----|--------|-------|
| macro_bal | 0.5733 | 0.5425 | **+0.0308** |
| normalized_cost | 0.9667 | 0.9667 | 0.0000 |

C13 beats random probing at identical cost, demonstrating that VOI-based action selection provides some signal. However, the advantage is small (+0.0308) and insufficient to beat observation-only.

### C13 vs C_minus

| Metric | C13 | C_minus | Delta |
|--------|-----|---------|-------|
| macro_bal | 0.5733 | 0.5000 | +0.0733 |

C13 adds +0.0733 over the zero-information prior.

### Gap Capture

| Metric | Value |
|--------|-------|
| Oracle - C0b gap | 0.4129 |
| C13 gain over C0b | -0.0138 |
| C13 gap capture | **-3.3%** |
| Practical threshold (10%) | 0.0413 |
| C13 macro_bal needed | 0.6284 |
| C13 passes practical | **No** |

C13 does not capture any of the C0b-to-oracle gap. Instead, it moves away from oracle (loses ground).

### Probe Efficiency

| Metric | Count | Rate |
|--------|-------|------|
| Effective probes | 11 | 0.500 |
| Zero-gain probes | 11 | 0.500 |
| Harmful probes | 0 | 0.000 |
| Total probes | 22 | 1.000 |

50% of VOI-selected probes produce zero accuracy gain on the probed object. This occurs because each category only differs on 1/6 actions; probing any of the 5 non-differentiating actions yields no information. The VOI gate eliminates harmful probes but can't avoid zero-gain probes when object identity uncertainty is high.

### Positive Discovery

| Metric | C13 | C0b | Delta |
|--------|-----|-----|-------|
| mean_pos_recall | 0.2000 | 0.3100 | **-0.1100** |

**C13 harms positive recall.** The reduced visit count (22 vs 60) means C13 misses many positive objects entirely. The probe information on visited objects can't compensate for the objects never seen.

### Negative-Recall Tradeoff

**Recall source: negative_only_tradeoff.** C13 gains +0.0825 in negative recall but loses -0.1100 in positive recall. This is the same asymmetric failure mode identified in Block 1I0: C13 becomes more conservative (predicting False more often), which improves negative recall at the cost of positive recall.

## 5. Pareto Frontier (Normal Policies Only)

Non-oracle **deployable** policies plotted by (normalized_cost, macro_bal). category_majority is excluded (see Section 5a).

| Policy | macro_bal | ncost | On Frontier? |
|--------|-----------|-------|--------------|
| all_false | 0.5000 | 0.0000 | **Yes** (cost floor) |
| C_minus_prior_only | 0.5000 | 0.0000 | No (identical to all_false) |
| C0b_observe_only | 0.5871 | 0.7533 | **Yes** (best budgeted) |
| C13_instance_VOI | 0.5733 | 0.9667 | **No (dominated by C0b)** |
| random_probe | 0.5425 | 0.9667 | No (dominated by C13 at same cost) |

**Oracle reference:** macro_bal=1.0000, ncost=0.0000

The normal-policy Pareto frontier has two points: **all_false** at (0.5000, 0.0000) and **C0b_observe_only** at (0.5871, 0.7533). C_minus is identical to all_false (same cost, same accuracy) so it does not add a frontier point. C13_instance_VOI is Pareto-dominated by C0b (lower macro_bal, higher cost). random_probe is dominated by C13 at the same cost.

**C0b is the best budgeted deployable policy on this seed.** It achieves the highest macro_bal among non-oracle policies that use only information available to the agent.

## 5a. Category Shortcut / Leakage-Risk Diagnostic

**category_majority = 0.9750** with zero cost. This is NOT on the normal-policy Pareto frontier because:

1. **It uses test-time hidden_category information.** The policy reads `test_objects[oid]["hidden_category"]` to look up the training-category majority affordance profile. This information is NOT available to the agent at test time — it is hidden behind observation and probing.

2. **It is a leakage-risk diagnostic, not a deployable policy.** The agent cannot implement category_majority without first identifying which category each object belongs to, which is itself the core inference challenge.

3. **C_minus = 0.5000 confirms the agent does NOT have free access to category identity.** If the agent had cheap category-level information, C_minus would score higher. It doesn't — the IOM prior predicts global base rates below 0.5 for all query features, producing the same result as all_false.

### What category_majority Tells Us

- The task has a strong **latent** category shortcut: knowing only the object's category, without any subtype information, achieves 0.975 macro_bal.
- The 0.025 gap to oracle comes from the 20% minority-subtype objects (3 per category) that violate the training-category majority assumption.
- This means the probing challenge is concentrated on identifying those 12 minority-subtype objects and probing their differentiating actions.

### Environment Design Risk

- **category_shortcut_still_strong = true.** C4's category-level signal is very strong (0.975). This is a structural property of using training-category majority as a predictor.
- **Mitigation**: C_minus = 0.5000 shows the agent cannot exploit this shortcut without observation or probing. The shortcut is latent, not active.
- For future environments, consider designs where category-majority does not trivially solve the task (e.g., balanced subtypes, cross-category affordance patterns).

---

## 6. C13 Single-Seed Assessment

| Criterion | Result | Detail |
|-----------|--------|--------|
| A. C13 > C0b macro_bal | **FAIL** | 0.5733 < 0.5871 |
| B. C13 > random_probe macro_bal | **PASS** | 0.5733 > 0.5425 |
| C. Improves positive recall | **FAIL** | -0.1100 (worse) |
| D. >= 10% gap capture | **FAIL** | -3.3% (negative) |
| E. Not Pareto-dominated | **FAIL** | C0b dominates C13 |
| F. No metric exploit | **FAIL** | negative_only_tradeoff |

### Verdict: **c13_single_seed_promising = FALSE**

### Final Interpretation

1. **C13 has weak selectivity signal** over random probing (+0.0308 macro_bal at identical cost), confirming VOI-based action selection provides some signal over random choice. But the advantage is too small to justify the probe cost.

2. **C13 is Pareto-dominated by C0b** (lower macro_bal 0.5733 vs 0.5871, higher ncost 0.9667 vs 0.7533). C13 should NOT proceed to multi-seed.

3. **C13 does not improve positive recall.** It worsens positive recall (-0.11) while improving negative recall (+0.0825). This is the same negative-only asymmetric failure mode from Block 1I0.

4. **Recommendation**: Do NOT proceed to multi-seed with original C13 on C4. C13 needs redesign before further evaluation.

5. **category_majority = 0.975 is a leakage-risk diagnostic**, not a deployable policy point. It uses test-time hidden_category information not available to the agent.

6. **C4 environment design note**: C4 has a strong category shortcut (0.975), but the agent's IOM prior (C_minus = 0.5000) does not exploit it. The shortcut is latent, not active.

---

## 7. Interpretation

### Why C13 Fails on This Environment

1. **Visit-probe tradeoff is too severe.** With budget 1.5, probing costs 0.05 per object (on top of reach+observe). C13 visits only 22/60 objects while C0b visits all 60. The probe information on 22 objects can't outweigh the lost observations on 38 objects.

2. **50% zero-gain probe rate.** Only 1/6 actions per category differentiates subtypes. Even VOI-based selection can't avoid probing non-differentiating actions when the object's subtype is highly uncertain. Half the probes yield no accuracy gain.

3. **Negative-only tradeoff persists.** C13's predictions become more conservative (higher negative recall, lower positive recall). This is the same failure mode from Block 1I0 — the VOI gate favors predicting False because it's "safer" given the low positive base rates (0.20 per query).

4. **Observation is surprisingly strong.** C0b at 0.5871 is solidly above prior (0.5000). The visible cues, though partial (max differential 0.37), provide enough signal when combined across all 60 objects to achieve reasonable performance. C13 sacrifices this breadth for depth that doesn't pay off.

5. **category_majority is a diagnostic upper bound, not a deployable policy.** At 0.9750 with zero cost, category-level knowledge nearly solves the task. However, this uses test-time hidden_category information not available to the agent. It tells us the probing challenge is concentrated on the 12 minority-subtype objects (3 per category), but C13 fails to target them efficiently. See Section 5a.

### The Core Failure Mechanism

C13's VOI gate evaluates whether probing a specific action on a specific object is worth the probe cost. In this environment:
- The prior (IOM) predicts global base rates below 0.5 for all query features
- After observation, predictions shift toward the majority subtype per category
- For majority-subtype objects (80%), no probe is needed — observation already pushes predictions in the right direction
- For minority-subtype objects (20%), probing the differentiating action would help
- But C13 can't identify WHICH objects are minority-subtype without probing them first
- So C13 either probes many objects (wasting budget on majority cases) or probes too few objects (missing minority cases)

### What Would Need to Change

This single-seed result suggests that for C13 to succeed on this environment, it would need:
- Better object selection to target likely-minority-subtype objects
- Better action selection to target the differentiating action per category
- A different cost structure or higher budget
- Some form of category-level reasoning to narrow the probe candidate set

These are design questions for future blocks, not changes to make now.

---

*Generated by Block 1J3 — single-seed Pareto policy comparison, no multi-seed, no tuning.*
