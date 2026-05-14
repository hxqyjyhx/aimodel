# Block 1J0: Environment Redesign Pre-registration / Design Criteria

**Date:** 2026-05-10
**Status:** Pre-registration — no simulations run.

---

## 1. Purpose

The purpose of this pre-registration is to define design criteria for a Mini-MC environment where selective probing has measurable marginal value under balanced metrics (macro_query_balanced_accuracy).

The redesign must be evaluated by marginal-value baselines (all_false, C_minus_prior_only, C0b_observe_only, C_oracle_full_information, random_probe_budgeted) before any C13 policy comparison. Environment acceptance is gated on these baselines, not on C13 performance.

---

## 2. Current Failure Summary

| # | Finding | Source |
|---|---------|--------|
| 1 | Raw composite accuracy is majority-class dominated. | Block 1H0 |
| 2 | macro_query_balanced_accuracy is ready as primary task metric. | Block 1H1 |
| 3 | Scalar utility (bal_ub25) is not ready for policy comparison. | Block 1H2 |
| 4 | C13 has weak asymmetric signal: macro_bal = 0.5578 vs C0b = 0.5444, delta = +0.0133. | Block 1H1 |
| 5 | C13 positive recall sum is lower than C0b (0.733 vs 1.033). C13 trades positive recall for higher negative recall. | Block 1H1 |
| 6 | Threshold fix is not promising. Best global threshold (0.2) gives macro_bal = 0.5767 vs C13 default (0.5) = 0.5578. Gain is small and does not address root cause. | Block 1I0 |
| 7 | Posterior separation is weak. Mean AUC = 0.6136, 4/5 queries have AUC < 0.65. True and false posterior distributions overlap heavily. | Block 1I0 |
| 8 | Aggregation bias is supported. Similarity-weighted (C13 default) = 0.5578, unweighted = 0.5211, single-nearest = 0.5144. Aggregation affects the recall trade-off but does not solve the underlying posterior-quality problem. | Block 1I0 |
| 9 | Oracle ceiling reaches macro_bal = 1.0000, positive recall = 1.0000, negative recall = 1.0000. The task is recoverable in principle. | Block 1I0 |
| 10 | Current policy_comparison_ready = false. No interactive policy beats all_false under scalar utility. Pareto reporting is the fallback. | Block 1H3 |

**Diagnosis label:** `mixed` (posterior_nonseparation + aggregation_bias)

**Core problem:** C13's positive-recall failure is not caused by negative evidence skew or threshold miscalibration. Posterior scores weakly separate true and false cases, and aggregation favors high negative recall over positive recall. Oracle shows the signal is recoverable, but current realizable C13 posterior cannot recover it reliably.

---

## 3. Redesign Principles

The new environment must satisfy the following principles. Each principle includes a justification independent of C13 performance.

### Principle A: Category prior must be insufficient.

Category membership alone should not perfectly determine affordance success. If knowing an object's category is sufficient to predict all affordances, probing adds no value beyond observation.

*Justification:* This is a task-design constraint. If category determines affordance, the task reduces to category classification and probe selectivity is irrelevant.

### Principle B: Structure must be recoverable.

The task should not become random noise. Oracle (full access to hidden ground truth) should still achieve high macro_bal. If oracle cannot recover the signal, the environment is structurally impossible and no policy can succeed.

*Justification:* This is a feasibility constraint. An impossible environment provides no information about policy quality.

### Principle C: Both observation and probing must have value.

Observation (visible features) should reveal useful but incomplete information about affordances. Probing should add additional value above observation. Neither observation alone nor probing alone should be sufficient.

*Justification:* This is the core experimental question. If observation alone suffices (C0b ~= oracle), there is no probe-value problem to study. If probing alone suffices (observation adds nothing), the task is a pure exploration problem rather than an observation+probe integration problem.

### Principle D: Support both positive and negative evidence.

The query suite must include:
- **Positive-discovery queries:** "Does this object provide X?" where true cases exist.
- **Negative-discovery queries:** "Is this object safe/unsuitable for X?" where false cases are informative.
- **Composite queries:** Queries requiring both positive and negative evidence.

Do not redesign only to favor positive recall. A probe that returns "failure" should be as informative as one that returns "success."

*Justification:* Real-world affordance learning involves both discovering what objects can do and discovering what they cannot do. A benchmark that only rewards positive discovery is incomplete.

### Principle E: No policy-favoring benchmark tuning.

Environment design decisions must be justified by task semantics, developmental plausibility, or principled difficulty calibration. Design choices must not be selected because they help C13 win. The marginal-value audit uses baselines (all_false, C_minus, C0b, oracle, random_probe), not C13.

*Justification:* This is an anti-overfitting constraint. If the environment is tuned so that C13 wins, the result does not generalize.

---

## 4. Candidate Redesign Directions

These are candidates for Block 1J1 implementation. None are implemented yet.

### Candidate A: Instance-level affordance variation with visible but imperfect cues

Objects within the same category vary in their affordance profiles. Visible features partially indicate affordance but are not perfectly predictive. Probing confirms or disconfirms the affordance.

*Example:* Some apples are ripe (edible), some are unripe (inedible). Visible color/texture partially indicates ripeness. Probing (bite/eat) confirms edibility.

*Expected effect:* Increases posterior uncertainty after observation, creating room for probe value. Preserves recoverable structure if visible cues correlate with hidden state.

### Candidate B: Hidden subtypes within categories

Each category contains hidden subtypes with different affordance profiles. Visible features are shared within category but differ between subtypes in ways that are not directly observable.

*Example:* Wood_log subtype A crafts planks well but burns poorly. Wood_log subtype B burns well but crafts poorly. Both look like wood_logs. Probing reveals the subtype.

*Expected effect:* Category prior becomes insufficient. Observation alone cannot distinguish subtypes. Probing is necessary for high accuracy.

### Candidate C: Query suite expansion

Add query templates beyond the current 5-action affordance queries:

| Query type | Example | Evidence needed |
|------------|---------|-----------------|
| Positive discovery | "Can this object be used as food?" | Positive probe |
| Negative discovery | "Is this object NOT suitable as fuel?" | Negative probe |
| Positive conjunction | "Can this object be both food AND a tool?" | Two positive probes |
| Negative conjunction | "Is this object NEITHER food NOR fuel?" | Two negative probes |
| Mixed conjunction | "Can this object be food but NOT fuel?" | Positive + negative |

*Expected effect:* Creates queries where negative evidence has explicit value, reducing the asymmetry between positive and negative probe outcomes. Composite queries require integrating multiple probe results.

### Candidate D: Observation noise or partial observability

Observation gives useful features but not the full set of affordance-relevant features. Some features are only revealed through probing.

*Example:* A rock's visible features (color, texture, size) are observed. Its hardness and fracture pattern are only revealed by striking it (probe).

*Expected effect:* Creates a natural distinction between "what you can see" and "what you must test." Increases the marginal value of probing.

### Candidate E: Balanced query construction

Ensure each query has both positive and negative examples in the test set. The current environment has 25/75 pos/neg splits for 4 of 5 queries, which creates majority-class artifacts. The redesigned environment should balance query construction, evaluated primarily with macro_query_balanced_accuracy.

*Expected effect:* Fixes the majority-class domination problem at the query-design level rather than only at the metric level.

---

## 5. Pre-registered Marginal-Value Audit

Before running any policy comparison (C13, C2, C8, etc.), the redesigned environment must pass a marginal-value audit. This audit establishes that probing and observation each have measurable value above simpler baselines.

### Required Baselines

| Baseline | Description | Purpose |
|----------|-------------|---------|
| `all_false` | Predicts False for all query-object pairs. | Chance baseline for macro_bal. |
| `C_minus_prior_only` | Uses only category-level priors from IOM, no observation, no probing. | Measures value of prior information alone. |
| `C0b_observe_only` | Visits objects, observes visible features, never probes. | Measures value of observation without probing. |
| `C_oracle_full_information` | Has access to hidden ground truth. | Upper bound on what is recoverable. |
| `random_probe_budgeted` | Visits objects randomly, probes randomly within budget. | Measures value of random probing vs selective probing. |
| `category_majority` (if applicable) | Predicts based on category-level majority class. | Measures value of category information. |

### Required Metrics (per baseline)

| Metric | Definition |
|--------|-----------|
| `raw_comp` | Raw composite task accuracy (secondary) |
| `macro_query_balanced_accuracy` | Mean[0.5*(pos_recall + neg_recall)] over all queries — **primary** |
| `positive_recall_per_query` | Per-query positive recall |
| `negative_recall_per_query` | Per-query negative recall |
| `normalized_cost` | total_cost / max_possible_cost |
| `probe_count` | Number of probe actions executed |
| `visit_count` | Number of objects visited |

---

## 6. Acceptance Criteria for New Environment

The environment is acceptable for policy comparison only if all mandatory criteria are met.

### Mandatory Criteria

| # | Criterion | Threshold | Rationale |
|---|-----------|-----------|-----------|
| 1 | `all_false` macro_bal | ≈ 0.5000 | Chance baseline must be calibrated. |
| 2 | `C_oracle` macro_bal | >= 0.7000 | Task must be solvable with full information. |
| 3 | `C_oracle - C0b` macro_bal | >= +0.1000 | Probing must have meaningful marginal value above observation. |
| 4 | `C0b - C_minus` macro_bal | >= +0.0500 | Observation must have meaningful value above prior. |
| 5 | `C_oracle_pos_recall - C0b_pos_recall` | >= +0.1000 | Probing must improve positive recall, not just negative. |
| 6 | `C_oracle_neg_recall - C0b_neg_recall` | >= +0.1000 | Probing must improve negative recall, not just positive. |

### Recommended Criteria

| # | Criterion | Threshold | Rationale |
|---|-----------|-----------|-----------|
| 7 | `C0b` macro_bal | 0.55–0.65 ideally | If C0b >= 0.70, observation alone nearly solves the task, leaving little room for probe value. |
| 8 | Oracle recall balance | Oracle should improve both positive and negative recall, not only one side. | Prevents redesign that creates an asymmetric oracle (e.g., oracle only helps positive recall). |
| 9 | `random_probe_budgeted` reported | Always reported as a baseline. | C13 later must beat random probing at comparable cost to support a selectivity claim. |

### Failure Handling

If the environment fails the mandatory criteria:
- Do NOT proceed to policy comparison.
- Record the failure in the design log.
- Adjust one design parameter and re-audit.
- Limit to at most 3 audit attempts before reporting that the redesign approach is blocked.

---

## 7. Pareto Policy Comparison Criteria

After the environment passes the marginal-value audit, policy comparison uses Pareto reporting (as established in Block 1H3).

### Per-Policy Required Fields

| Field | Status |
|-------|--------|
| `macro_query_balanced_accuracy` | Primary |
| `normalized_cost` | Primary |
| `probe_count` | Primary |
| `positive_recall_per_query` | Primary |
| `negative_recall_per_query` | Primary |
| `raw_comp` | Secondary |
| `scalar_utility` | Secondary only |

### Dominance Rule

Policy A **dominates** Policy B if and only if:

```
macro_bal_A >= macro_bal_B
normalized_cost_A <= normalized_cost_B
at least one inequality is strict
```

If neither dominates, report the trade-off explicitly rather than forcing a scalar ranking.

### Meaningful Improvement Over C0b

A policy claims meaningful improvement over C0b only if all three criteria are met:

**Statistical criterion:** macro_bal gain over C0b exceeds 2 standard errors across seeds.

**Practical criterion:** macro_bal gain is at least 10% of the C0b-to-oracle gap:

```
macro_bal_policy - macro_bal_c0b >= 0.10 * (macro_bal_oracle - macro_bal_c0b)
```

**Cost criterion:** The policy outperforms random_probe_budgeted or a proportional oracle-mixing baseline at comparable cost. If the policy achieves higher macro_bal but at higher cost, the trade-off must be reported explicitly.

---

## 8. Anti-Tuning Safeguards

The following safeguards are pre-registered to prevent benchmark overfitting:

| # | Safeguard | Enforcement |
|---|-----------|-------------|
| 1 | Environment criteria are written before policy runs. | This document serves as the pre-registration. |
| 2 | Environment redesign is evaluated using marginal-value baselines (all_false, C_minus, C0b, oracle, random_probe), not C13. | Block 1J2 audit uses only baselines. |
| 3 | Do not iterate environment indefinitely until C13 wins. | Maximum 3 audit attempts. |
| 4 | Keep a design log for every environment change. | Each attempt recorded in protocols/. |
| 5 | Report negative results if the redesigned environment passes the audit but C13 fails. | Block 1J3/1J4 must report regardless of outcome. |
| 6 | Do not change cost or beta to make C13 win. | Cost scale and beta are fixed from Block 1H3. |
| 7 | Do not tune probe actions to match C13's information preferences. | Probe actions are determined by task semantics. |
| 8 | Pre-register all query templates before seeing policy results. | Query suite is part of environment design, fixed before Block 1J3. |

---

## 9. Next Blocks

| Block | Description | Prerequisite |
|-------|-------------|-------------|
| **1J1** | Implement one minimal redesigned environment candidate. Select from Section 4 candidates. Produce environment spec, config diff, and rationale. | Block 1J0 (this document) |
| **1J2** | Run marginal-value audit on the redesigned environment. Compute all baselines and check against Section 6 acceptance criteria. | Block 1J1 |
| **1J3** | If and only if 1J2 passes: run C0b / C13 / random_probe / oracle policy comparison under Pareto reporting. Single seed only. | Block 1J2 (pass) |
| **1J4** | Multi-seed sweep only after 1J2 and 1J3 are promising. 5 seeds x N policies. | Block 1J3 (promising) |

Block 1J3 and 1J4 are **gated** on Block 1J2 passing the acceptance criteria. If 1J2 fails after 3 audit attempts, the redesign approach is blocked and the finding must be reported.

---

## 10. Final Decision

| Flag | Value | Rationale |
|------|-------|-----------|
| `balanced_metric_ready` | **true** | macro_query_balanced_accuracy neutralizes majority-class bias (Block 1H1). |
| `pareto_reporting_ready` | **true** | Pareto dominance framework is defined and validated (Block 1H3). |
| `scalar_utility_ready` | **false** | No single beta works; scalar utility overshoots achievable gain by ~4x (Block 1H2). |
| `current_policy_comparison_ready` | **false** | C13 has weak asymmetric signal; policy gains too small for meaningful comparison (Block 1H3, 1I0). |
| `environment_redesign_preregistered` | **true** | This document. |
| `ready_to_implement_environment_candidate` | **true** | Diagnosis is clear, principles are defined, acceptance criteria are pre-registered. Proceed to Block 1J1. |

### Decision

The current environment (C3_P060_O040, budget=1.5, seed=101) does not support meaningful policy comparison. C13's macro_bal gain over C0b (+0.0133) is too small relative to the oracle gap (0.442). The diagnosis is mixed: posterior non-separation + aggregation bias, with no simple threshold fix.

The environment must be redesigned so that:
1. Observation alone is insufficient (C0b leaves room for probe value).
2. Oracle information recovers the signal (task is not impossible).
3. Both positive and negative evidence have value.

Proceed to Block 1J1: implement one minimal redesigned environment candidate, then audit with marginal-value baselines in Block 1J2.

---

*Generated by Block 1J0 — pre-registration only, no simulations run.*
