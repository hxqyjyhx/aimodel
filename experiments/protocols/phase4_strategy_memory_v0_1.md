# Phase 4: Strategy Memory v0.1

## 1. Purpose

Strategy Memory (L4) learns **evidence-based statistics for action selection** — when to observe, try, skip, or switch. It synthesizes evidence from Event Memory (L1), Object Candidate Memory (L2), and Experience Memory (L3) into probabilistic action-value estimates. These estimates will eventually bias action selection (Phase 4+), but do not yet influence policy decisions.

Strategy Memory is the final upward-reading layer in the memory architecture. It reads from all lower layers; no layer reads downward from it.

It is currently a **shadow/sidecar module**: it computes what the strategy WOULD be, without activating it in the decision loop. It will eventually replace hand-crafted FCRM counters and the `detect_type_family()` fallback.

## 2. Inputs

### 2.1 Allowed

| Source | Field | Usage |
|--------|-------|-------|
| Event Memory (L1) | `action_type`, `action_target`, `observed_features_delta`, `success_or_failure` | Compute observation helpfulness, track action sequences |
| Object Candidate Memory (L2) | `candidate_id`, `promotion_status`, `stability_score`, `positive_feature_set` | Group strategies by candidate status |
| Experience Memory (L3) | `candidate_anchor_id`, `action_affordance`, `success_count`, `failure_count`, `contradiction_count`, `status` | Compute try-action value estimates and risk scores |
| Experience Memory (L3) | Candidate diagnostics: consistent/mixed/reinforced/split-needed | Categorize candidates for strategy grouping |
| Teacher Feedback Channel | `teacher_label` (outcome confirm) | Optional; treated as informed evidence |

### 2.2 Forbidden

`true_family`, `hidden_subtype`, `prior_violation`, `oracle_outcome`, `deceptive_flag`, `full_object_state`, `unobserved_features`, `raw_internal_score`, and any audit-only field. Hand-written strategy rules are explicitly forbidden.

## 3. Strategy Record Fields

| # | Field | Type | Description |
|---|-------|------|-------------|
| 1 | `strategy_id` | str | Unique identifier (e.g., `"strat_stable_consistent_try_eat"`) |
| 2 | `strategy_context_key` | str | Composite key: `{candidate_status}_{action_type}_{target_action?}` |
| 3 | `candidate_status_category` | str | One of: `"stable_consistent"`, `"stable_mixed"`, `"provisional"`, `"all"` |
| 4 | `action_type` | str | One of: `"observe"`, `"try"`, `"skip"`, `"switch"` |
| 5 | `target_action` | str or null | Affordance for try strategies; null otherwise |
| 6 | `evidence_mode` | str | One of: `"observed_action"`, `"derived_from_experience"`, `"proxy_estimate"` |
| 7 | `is_proxy_estimate` | bool | True if this strategy is estimated from related evidence, not directly observed actions |
| 8 | `causal_claim_allowed` | bool | True only if a controlled comparison supports causal interpretation; usually false for shadow |
| 9 | `source_event_ids` | list[int] | Event Memory step_ids that contributed |
| 10 | `source_experience_ids` | list[str] | Experience record IDs that contributed |
| 11 | `source_candidate_ids` | list[str] | Candidate IDs whose data backs this strategy |
| 12 | `support_count` | int | Number of data points backing this strategy |
| 13 | `success_count` | int | For try: successes. For observe: useful observations. |
| 14 | `failure_count` | int | For try: failures. For observe: observations that didn't help. |
| 15 | `contradiction_count` | int | Number of minority outcomes |
| 16 | `estimated_action_value` | float | Estimated value of taking this action in this context (0–1) |
| 17 | `estimated_observation_value` | float | For observe/skip/switch: estimated value of gathering more evidence |
| 18 | `risk_score` | float | Estimated risk of negative outcome (0–1) |
| 19 | `confidence_score` | float | Confidence in the value estimate |
| 20 | `episode_spread` | int | Number of distinct episodes evidence is drawn from |
| 21 | `status` | str | One of: `"provisional"`, `"stable"`, `"contradicted"` |

### 3.1 Evidence Mode Semantics

| evidence_mode | When used | Example |
|---------------|-----------|---------|
| `observed_action` | Real events of this action_type exist in Event Memory | `try` events exist; `observe` events exist |
| `derived_from_experience` | Value computed from Experience Memory records, which bind actions to outcomes | Try success rate for (candidate, eat) from Experience Memory |
| `proxy_estimate` | No real events of this action_type exist; estimate is inferred from related evidence | No real `skip` or `switch` events in simulation; estimate derived from contradiction/confidence patterns |

### 3.2 Proxy Estimate Labeling

If there are no real skip/switch action-result events in Event Memory:
- skip/switch strategy values are labeled `evidence_mode = "proxy_estimate"` and `is_proxy_estimate = true`.
- Their `estimated_action_value` field is renamed to `estimated_skip_proxy` / `estimated_switch_proxy` in reporting.
- `skip_switch_observed_event_count` is reported in the evaluation.
- They are NOT called "learned strategy values" — they are "proxy estimates."
- Do NOT overclaim them as evidence-backed strategies.

## 4. Required Learned Statistics

The Strategy Memory must compute the following from lower-layer evidence:

| Statistic | Derived from | Meaning |
|-----------|-------------|---------|
| `observe_value_by_candidate_status` | Event Memory observe events | How often observing reveals useful new features, grouped by candidate status |
| `try_value_by_candidate_action` | Experience Memory records | Success rate for each (candidate_status, action) pair |
| `skip_value_when_candidate_mixed_or_contradicted` | Experience Memory contradictions | Value of skipping a candidate that has mixed action outcomes |
| `switch_value_when_current_candidate_low_confidence` | OCM stability + Experience confidence | Value of switching attention away from uncertain candidates |
| `action_risk_by_candidate_action` | Experience Memory failure rates | Risk of failure for each (candidate_status, action) |
| `contradiction_penalty_by_candidate_action` | Experience Memory contradiction counts | Penalty applied when a (candidate, action) has mixed outcomes |
| `observation_helpfulness_rate` | Event Memory observe→try sequences | Rate at which observing precedes a successful try vs observing precedes nothing useful |

## 5. Observation Value Rule (Association-Based Unless Controlled)

Observation value must be **estimated from evidence as association**, not declared by hand and not claimed as causal:

1. For each observe event in Event Memory, check whether `observed_features_delta` was non-empty.
2. Compute `observation_revealed_new_features_rate`: proportion of observe events that revealed >= 1 feature.
3. Compute `observation_candidate_assignment_rate`: proportion of observed objects that were subsequently assigned to a candidate in OCM.
4. Compute `observation_before_success_rate`: proportion of objects where observing preceded a successful try action.
5. Compute `observation_before_failure_rate`: proportion of objects where observing preceded a failed try action.
6. Compute `observation_association_score`: a composite score combining the above rates.

**Causal claim restriction:** Do NOT claim "observation caused success" or "observation caused correct assignment." These are associations unless a controlled comparison (observed vs unobserved, same object type) is implemented. For Phase 4 shadow, `causal_claim_allowed` is usually `false`.

Association score is reported as:
- `observation_revealed_new_features_rate`
- `observation_candidate_assignment_rate`
- `observation_before_success_rate`
- `observation_before_failure_rate`
- `observation_association_score` (composite)

Do NOT use hand-written rules like "always observe mixed candidates." The association must emerge from the event record counts. Any hard-coded strategy rule (e.g., `if candidate.mixed: return "observe"`) sets `handwritten_strategy_rule_detected = true`.

## 6. Mixed-Outcome Handling

Because 1J37 found ~57% of candidates have mixed action outcomes:

- Strategy Memory must **explicitly track** mixed/contradicted candidates as a distinct strategy category.
- It must NOT silently treat mixed candidates as stable-consistent.
- It must compute **separate strategy records** for `stable_consistent`, `stable_mixed`, and `provisional` candidate status categories.
- It must report `mixed_candidate_strategy_count`.
- It must report observe/try risk statistics for mixed candidates.
- It can report whether mixed candidates show different observe/try/skip/switch patterns as **associations**, not as causal prescriptions.
- Do NOT encode "always observe mixed candidates" or "always skip mixed candidates" as decision rules.
- Any such hard-coded rule sets `handwritten_strategy_rule_detected = true`.
- Mixed outcome evidence is a **feature** — it drives better strategy differentiation. Do not hide it.

## 7. Strategy Categorization

Candidates are categorized for strategy grouping:

| Category | Definition | Expected strategy bias |
|----------|-----------|----------------------|
| `stable_consistent` | OCM stable + all Experience records non-contradicted | Confident try, low skip |
| `stable_mixed` | OCM stable + at least one Experience record contradicted | Observe more, cautious try |
| `provisional` | OCM provisional (in buffer) | Observe first, try cautiously |
| `all` | Aggregate across all candidates | Baseline for comparison |

For try strategies, further subdivide by `target_action` (eat, craft_plank, etc.).
For observe/skip/switch strategies, aggregate by candidate category only.

## 8. Shadow Evaluation Metrics

When run in shadow mode, report:

| Metric | Description |
|--------|-------------|
| `total_strategy_records` | Count of all StrategyRecord instances |
| `stable_strategy_records` | Count with status "stable" |
| `provisional_strategy_records` | Count with status "provisional" |
| `contradicted_strategy_records` | Count with status "contradicted" |
| `observe_strategy_records` | Count with action_type "observe" |
| `try_strategy_records` | Count with action_type "try" |
| `skip_strategy_records` | Count with action_type "skip" (all proxy_estimate) |
| `switch_strategy_records` | Count with action_type "switch" (all proxy_estimate) |
| `proxy_estimate_strategy_records` | Count with is_proxy_estimate=true |
| `skip_switch_observed_event_count` | Number of real skip/switch events observed (0 in current env) |
| `average_observation_value` | Mean estimated_observation_value across observe strategies |
| `average_try_value` | Mean estimated_action_value across try strategies |
| `average_risk_score` | Mean risk_score across try strategies |
| `mixed_candidate_strategy_count` | Number of strategy records for stable_mixed category |
| `strategy_records_with_source_ids` | Count of records citing source events or experiences |
| `observation_revealed_new_features_rate` | Association: rate of observe events revealing >= 1 feature |
| `observation_candidate_assignment_rate` | Association: rate of observed objects assigned to a candidate |
| `observation_before_success_rate` | Association: rate of observe events followed by successful try |
| `observation_before_failure_rate` | Association: rate of observe events followed by failed try |
| `observation_association_score` | Composite association score |
| `proxy_estimates_labeled` | Must be true |
| `observation_causal_claim_made` | Must be false |
| `hidden_feature_leakage_detected` | Must be false |
| `oracle_leakage_detected` | Must be false |
| `handwritten_strategy_rule_detected` | Must be false |
| `strategy_memory_rederivable` | Must be true |
| `policy_decisions_changed` | Must be false |

## 9. Forbidden Operations

The Strategy Memory must NEVER:

1. Read `true_family`, `hidden_subtype`, or any audit-only field
2. Directly influence policy action selection (in shadow mode)
3. Encode hand-written strategy rules like "always observe mixed candidates"
4. Hide or suppress contradictory evidence
5. Write into Event Memory (L1), Object Memory (L2), or Experience Memory (L3)
6. Use MCTS, planning, or lookahead
7. Replace FCRM counters or `detect_type_family()`

## 10. Re-derivability

Strategy Memory must be **re-derivable**: all strategy records can be recomputed from Event Memory + Object Candidate Memory + Experience Memory alone. No strategy record persists state that cannot be rebuilt from lower layers.

Acceptance check: run the strategy computation twice on the same lower-layer state and verify identical output.

## 11. Acceptance Checks

| # | Check | Requirement |
|---|-------|-------------|
| 1 | No hidden/oracle/audit-only fields enter Strategy Memory | Code audit |
| 2 | Every strategy record cites source_event_ids or source_experience_ids | All records have non-empty source lists |
| 3 | No hand-written strategy rule | `handwritten_strategy_rule_detected == false` |
| 4 | Mixed/contradicted candidates reported | `mixed_candidate_strategy_count > 0` (expected given 1J37 rate) |
| 5 | Policy decisions unchanged | Shadow mode |
| 6 | Strategy Memory is re-derivable | Run twice, compare outputs |
| 7 | No MCTS/planning added | Code audit |
| 8 | Proxy estimates are clearly labeled | `proxy_estimates_labeled == true` |
| 9 | Skip/switch not overclaimed if no real events | `skip_switch_observed_event_count` reported; all skip/switch records have `is_proxy_estimate = true` |
| 10 | Observation value reported as association | `observation_causal_claim_made == false` |
| 11 | Output is reproducible | Same seed produces same strategy records |


```
[phase_done]
phase=4
doc=protocols/phase4_strategy_memory_v0_1.md
strategy_memory_shadow=true
shadow_json=TBD
shadow_summary=TBD
total_strategy_records=TBD
stable_strategy_records=TBD
mixed_candidate_strategy_count=TBD
skip_switch_observed_event_count=TBD
proxy_estimates_labeled=TBD
observation_causal_claim_made=TBD
handwritten_strategy_rule_detected=TBD
hidden_feature_leakage_detected=TBD
oracle_leakage_detected=TBD
policy_decisions_changed=false
implementation_status=TBD
failure_reason=TBD
```

