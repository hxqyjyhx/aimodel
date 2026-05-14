# Block 1J2: Marginal-Value Audit — C4_instance_subtype_cued_v1

**Date:** 2026-05-10
**Block:** 1J2 — Marginal-value audit (no C13, no policy comparison)
**Condition:** C4_instance_subtype_cued_v1
**Budget:** 1.5

---

## 1. Audit Configuration

| Parameter | Value |
|-----------|-------|
| Seed | 101 |
| Test objects | 60 (15 per category, deterministic subtype assignment) |
| Budget | 1.5 |
| Grid | 7×9, agent start at (3, 4) |
| IOM | k=10, nb_likelihood |
| Training objects | 160 standard (40 per category) |
| Sparse outcome coverage | 0.5714 (640 rows) |
| Pre-accuracy | 0.4625 |

## 2. Raw Results

| Baseline | macro_bal | raw_comp | visits | probes | total_cost | ncost |
|----------|-----------|----------|--------|--------|------------|-------|
| all_false | 0.5000 | 0.7400 | 0 | 0 | 0.0000 | 0.0000 |
| C_minus_prior_only | 0.5000 | 0.7400 | 0 | 0 | 0.0000 | 0.0000 |
| C0b_observe_only | 0.5871 | 0.7533 | 60 | 0 | 1.1300 | 0.7533 |
| C_oracle_full_information | 1.0000 | 1.0000 | 0 | 0 | 0.0000 | 0.0000 |
| random_probe_budgeted | 0.5425 | 0.7400 | 22 | 22 | 1.4500 | 0.9667 |
| category_majority | 0.9750 | 0.9600 | 0 | 0 | 0.0000 | 0.0000 |

## 3. Per-Query Details

### all_false

| Query | pos_recall | neg_recall | bal | accuracy |
|-------|-----------|-----------|-----|----------|
| need_food | 0.0000 | 1.0000 | 0.5000 | 0.8000 |
| need_fuel | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| need_planks | 0.0000 | 1.0000 | 0.5000 | 0.8000 |
| need_stone | 0.0000 | 1.0000 | 0.5000 | 0.8000 |
| need_tool | 0.0000 | 1.0000 | 0.5000 | 0.8000 |

### C_minus_prior_only

| Query | pos_recall | neg_recall | bal | accuracy |
|-------|-----------|-----------|-----|----------|
| need_food | 0.0000 | 1.0000 | 0.5000 | 0.8000 |
| need_fuel | 0.0000 | 1.0000 | 0.5000 | 0.5000 |
| need_planks | 0.0000 | 1.0000 | 0.5000 | 0.8000 |
| need_stone | 0.0000 | 1.0000 | 0.5000 | 0.8000 |
| need_tool | 0.0000 | 1.0000 | 0.5000 | 0.8000 |

C_minus = all_false because the IOM's global base rates for all query-determining features are below 0.5. Training data has only 25% of objects with craft_plank=success (wood_log only), 25% with eat=success (apple only), etc. The prior predicts False for all query-relevant affordances. This means **the IOM prior provides zero lift over all_false** — all information must come from observation or probing.

### C0b_observe_only

| Query | pos_recall | neg_recall | bal | accuracy |
|-------|-----------|-----------|-----|----------|
| need_food | 0.1667 | 1.0000 | 0.5833 | 0.8333 |
| need_fuel | 0.6333 | 0.6333 | 0.6333 | 0.6333 |
| need_planks | 0.2500 | 0.8125 | 0.5312 | 0.7000 |
| need_stone | 0.0833 | 0.9375 | 0.5104 | 0.7667 |
| need_tool | 0.4167 | 0.9375 | 0.6771 | 0.8333 |

Observation provides visible cues that partially indicate subtype. Positive recall improves from 0.0 (C_minus) to 0.083-0.417 across queries. The single-category queries all show low positive recall (0.08-0.42), meaning the 12 positive objects per query are hard to find via observation alone.

### C_oracle_full_information

| Query | pos_recall | neg_recall | bal | accuracy |
|-------|-----------|-----------|-----|----------|
| need_food | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| need_fuel | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| need_planks | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| need_stone | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| need_tool | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Ground-truth affordances are perfectly query-predictive. The task has a well-defined ceiling at 1.0.

### random_probe_budgeted

| Query | pos_recall | neg_recall | bal | accuracy |
|-------|-----------|-----------|-----|----------|
| need_food | 0.3333 | 1.0000 | 0.6667 | 0.8667 |
| need_fuel | 0.1667 | 0.8000 | 0.4833 | 0.4833 |
| need_planks | 0.0833 | 0.9167 | 0.5000 | 0.7500 |
| need_stone | 0.0833 | 0.9792 | 0.5312 | 0.8000 |
| need_tool | 0.0833 | 0.9792 | 0.5312 | 0.8000 |

Random probing (22 objects visited and probed) underperforms C0b on macro_bal (0.5425 vs 0.5871) despite higher cost. The budget spent on random probes (22 × 0.05 = 1.10) reduces the number of objects that can be visited, and randomly selected probe actions often don't target the differentiating affordance for the visited object.

### category_majority

| Query | pos_recall | neg_recall | bal | accuracy |
|-------|-----------|-----------|-----|----------|
| need_food | 1.0000 | 0.9375 | 0.9688 | 0.9500 |
| need_fuel | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| need_planks | 1.0000 | 0.9375 | 0.9688 | 0.9500 |
| need_stone | 1.0000 | 0.9375 | 0.9688 | 0.9500 |
| need_tool | 1.0000 | 0.9375 | 0.9688 | 0.9500 |

Category-majority achieves 0.975 macro_bal, close to oracle (1.0). The 0.025 gap comes from the 3/15 minority-subtype objects per category that violate the training-category majority assumption (e.g., 3 rotten wood_log are predicted as need_planks=True but are actually False).

## 4. Deltas

| Delta | Value | Threshold | Status |
|-------|-------|-----------|--------|
| C0b - C_minus | +0.0871 | >= +0.0500 | PASS |
| C_oracle - C0b | +0.4129 | >= +0.1000 | PASS |
| C_oracle - C_minus | +0.5000 | — | — |
| random_probe - C0b | -0.0446 | — | random < C0b |
| category_majority - C0b | +0.3879 | — | — |
| category_majority - C_minus | +0.4750 | — | — |
| oracle pos_recall - C0b pos_recall | +0.6900 | >= +0.1000 | PASS |
| oracle neg_recall - C0b neg_recall | +0.1358 | >= +0.1000 | PASS |

## 5. Acceptance Criteria

### Mandatory (all 6 must pass)

| # | Criterion | Value | Result |
|---|-----------|-------|--------|
| 1 | all_false macro_bal ≈ 0.5000 | 0.5000 | PASS |
| 2 | C_oracle macro_bal >= 0.7000 | 1.0000 | PASS |
| 3 | C_oracle - C0b >= +0.1000 | +0.4129 | PASS |
| 4 | C0b - C_minus >= +0.0500 | +0.0871 | PASS |
| 5 | Oracle pos_recall - C0b pos_recall >= +0.1000 | +0.6900 | PASS |
| 6 | Oracle neg_recall - C0b neg_recall >= +0.1000 | +0.1358 | PASS |

**All 6 mandatory criteria PASS.** The environment is accepted for policy comparison (Block 1J3).

### Recommended

| # | Criterion | Value | Result |
|---|-----------|-------|--------|
| 7 | C0b macro_bal 0.55-0.65 | 0.5871 | PASS (ideal range) |
| 8 | Oracle improves both pos and neg recall | pos +0.69, neg +0.14 | PASS |
| 9 | random_probe reported | yes | PASS |

C0b at 0.5871 is solidly in the ideal 0.55-0.65 range — observation provides meaningful but incomplete information, leaving substantial room for probing to add value.

### Special Flags

| Flag | Value | Meaning |
|------|-------|---------|
| category_majority_dominates_environment | false | cat_maj (0.975) is close to oracle (1.0) but gap is 0.025, just above 0.02 threshold |
| probe_marginal_value_insufficient | false | Oracle-C0b gap is +0.4129, well above 0.10 |
| observation_value_insufficient | false | C0b-C_minus gap is +0.0871, above 0.05 |
| positive_discovery_still_hard | true | C_minus pos_recall = 0.0, C0b pos_recall = 0.31 (low) |

**positive_discovery_still_hard = true**: C_minus has zero positive recall (prior predicts False for everything), and C0b achieves only 0.08-0.63 positive recall across queries. Finding the 12 positive objects out of 60 is intrinsically hard with observation alone. This is where selective probing should add value.

## 6. Interpretation

### Key Findings

1. **The IOM prior is useless without observation.** C_minus = all_false (macro_bal = 0.5000). The global base rates from training data (standard affordance profiles, 25% positive rate per differentiating action) all fall below the 0.5 threshold, so the prior predicts False for every query. All information must come from observation or probing.

2. **Observation provides meaningful but incomplete signal.** C0b raises macro_bal to 0.5871 (+0.0871 over prior). Visible cues partially indicate subtype, allowing the IOM to find some positive objects. But positive recall remains low (0.08-0.42 for single-category queries).

3. **Probe marginal value is very large.** Oracle - C0b = +0.4129. This is 4.7× the mandatory threshold of +0.1000. The environment has very strong probe value — knowing the exact affordance profile transforms performance from 0.59 to 1.00.

4. **Random probing is worse than no probing.** random_probe_budgeted (0.5425) underperforms C0b (0.5871) despite spending more budget. Without selective action choice, probing wastes budget on non-diagnostic actions. This validates the need for VOI-based probe selection (C13).

5. **Category knowledge nearly solves the task.** category_majority = 0.9750. The 20% minority-subtype objects are the only challenge. This is structurally fine — it means probing's value is concentrated in identifying those 12 minority objects (3 per category).

### Why the Environment Passes

- **Large oracle gap** (+0.4129): Full information is dramatically better than observation alone.
- **Moderate observation value** (+0.0871): Observation helps but doesn't saturate.
- **Oracle improves both sides**: Both positive and negative recall benefit from full information.
- **C0b in ideal range** (0.5871): The task is neither too easy nor too hard for observation alone.
- **Random probing fails**: Unselective probing wastes budget, creating room for C13 to demonstrate VOI-based selection advantage.

### Risks Carried Forward from 1J1

1. **Single-category queries at 0.20 boundary**: The low positive rate (12/60) makes positive recall inherently hard. C0b's low pos_recall (0.08-0.42) reflects this difficulty.

2. **category_majority at 0.975**: This is high but does not dominate the environment (gap to oracle is 0.025). The 20% minority cases provide a clear but narrow window where probing must add value.

3. **C_minus = all_false**: The IOM prior adds zero lift. This is a consequence of the training distribution (standard objects) differing from the test distribution (subtype objects). C13 and other probing policies must overcome this zero-prior baseline.

---

*Generated by Block 1J2 — marginal-value audit, no C13, no policy comparison.*
