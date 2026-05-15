# Block 1J37: Phase 2+3 Shadow Multiseed Robustness

- **Seeds**: [101, 103, 107, 109, 113]
- **Condition**: C4_instance_subtype_cued_v1
- **Episodes per seed**: 5
- **Total elapsed**: 17.2s

## Per-Seed Results

### Phase 2 OCM

| Seed | Total Cand | Stable | Provisional | Conflicted | Single-Event Prom | Hidden Leak | Oracle Leak |
|------|-----------|--------|-------------|------------|-------------------|-------------|-------------|
| 101 | 25 | 5 | 20 | 17 | 0 | False | False |
| 103 | 31 | 6 | 25 | 17 | 0 | False | False |
| 107 | 26 | 3 | 23 | 17 | 0 | False | False |
| 109 | 29 | 7 | 22 | 21 | 0 | False | False |
| 113 | 28 | 6 | 22 | 16 | 0 | False | False |

### Phase 3 Experience Memory

| Seed | Total Exp | Stable | Prov | Contra | Rejected | Contradictions | FeatureOnly | HiddenLeak | OracleLeak | PolicyChg |
|------|-----------|--------|------|--------|----------|---------------|-------------|------------|------------|------------|
| 101 | 150 | 67 | 48 | 29 | 6 | 41 | False | False | False | False |
| 103 | 186 | 74 | 78 | 34 | 0 | 34 | False | False | False | False |
| 107 | 156 | 61 | 54 | 35 | 6 | 47 | False | False | False | False |
| 109 | 174 | 88 | 48 | 35 | 3 | 41 | False | False | False | False |
| 113 | 168 | 71 | 60 | 35 | 2 | 39 | False | False | False | False |

### Candidate Action-Outcome Diagnostics

| Seed | Consistent | Mixed | Split-Needed | Reinforced | No Evidence |
|------|------------|-------|--------------|------------|-------------|
| 101 | 10 | 15 | 0 | 10 | 0 |
| 103 | 15 | 16 | 0 | 15 | 0 |
| 107 | 10 | 16 | 0 | 10 | 0 |
| 109 | 13 | 16 | 0 | 13 | 0 |
| 113 | 12 | 16 | 0 | 12 | 0 |

## Aggregate Statistics

| Metric | Mean | Stdev |
|--------|------|-------|
| Total candidates | 27.8 | 2.1 |
| Stable candidates | 5.4 | 1.4 |
| Total experience records | 166.8 | 12.8 |
| Contradictions | 40.4 | 4.2 |
| Mixed outcome candidate rate | 0.570 | 0.040 |
| Reinforced candidate rate | 0.430 | 0.040 |

## Flagged Seeds

| Flag | Seeds |
|------|-------|
| feature_only_rule_detected | none |
| hidden_oracle_leakage | none |
| policy_decisions_changed | none |
| single_event_promotion | none |

## Decision Rule

| Criterion | Value |
|-----------|-------|
| feature_only_rule_detected=false for all seeds | True |
| hidden_feature_leakage_detected=false for all seeds | True |
| oracle_leakage_detected=false for all seeds | True |
| policy_decisions_changed=false for all seeds | True |
| single_event_promotion_count=0 for all seeds | True |
| No seed crashes | True |
| **phase3_shadow_robust_enough_for_phase4** | **True** |

## Important Caveats

- Experience Memory is **shadow-only**. It does not influence policy decisions.
- Phase 2 candidates are **provisional anchors**, not true categories.
- This robustness check verifies the shadow pipeline is stable across seeds.
- Phase 4 may use these outputs as diagnostic inputs for strategy statistics.
- **This does NOT claim Experience Memory is ready to replace FCRM.**


```
[block_done]
block_id=1J37
seeds=[101, 103, 107, 109, 113]
phase3_shadow_robust_enough_for_phase4=true
feature_only_rule_detected_any=false
hidden_feature_leakage_detected_any=false
oracle_leakage_detected_any=false
policy_decisions_changed_any=false
single_event_promotion_any=false
implementation_status=pass
failure_reason=none
```
