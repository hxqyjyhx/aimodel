# Block 1J40b-7f: Oracle-Belief Upper-Bound Control

- **Implementation Status**: COMPLETED
- **Elapsed**: 20.2s
- **Seeds**: [101, 103, 107, 109, 113]
- **Observe Costs**: [0.01, 0.03, 0.05, 0.07]
- **Coverage Fractions**: [0.25, 0.5]

## 1. Files

- `_block1j40b7f_oracle_belief_upper_bound_control.py` -- Written
- `runs/block1j40b7f_oracle_belief_upper_bound_control.json` -- Generated
- `protocols/block1j40b7f_oracle_belief_upper_bound_control.md` -- Generated
- `protocols/block1j40b7f_oracle_belief_upper_bound_control_table.csv` -- Generated

## 2. Oracle-Belief Definition

- **Primary rule**: observe if E_observe_net > 0 (threshold on empirical mean VOI per signature)
- **E_observe_net**: mean(true_oracle_VOI | pre_surface_signature) computed from TRAINING ONLY
- **P_positive**: empirical P(positive_net | pre_surface_signature) from TRAINING ONLY
- **Fallback for held-out signatures**: global mean E_observe_net from training
- **Audit-only**: never fed to learner, never written into model-visible records

## 3. Policies Compared

1. always_try (no observe)
2. always_observe
3. oracle_selective (test-side oracle, pre_best from signature groups)
4. current_learned_policy (7e D0: raw pre_vec observe_value_model)
5. signature_lookup_policy (training-side majority optimal per sig)
6. oracle_belief_policy (E_observe_net > 0 per sig, global fallback)

## 4. Headroom Criteria

- **PASS / structural VOI justified**: oracle_belief > always_observe by meaningful margin on Test A/B, selective observe rate, discriminates positive from non-positive
- **PARTIAL**: slight improvement but marginal or only one test improves
- **FAIL / insufficient headroom**: oracle_belief <= always_observe, also collapses

## 5. Results at observe_cost = 0.01

| Test | Condition | always_try | always_obs | oracle_sel | learned | sig_lookup | oracle_belief | ob-ao | ob-at | osel-ob | learned-ob | ob_rate | pos_obs | nonpos_obs |
|------|-----------|------------|------------|------------|---------|------------|---------------|-------|-------|---------|------------|--------|---------|------------|
| A | original | 0.2500 | 0.4400 | 0.4433 | 0.4400 | 0.4400 | 0.4400 | +0.0000 | +0.1900 | +0.0033 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| B | original | 0.2500 | 0.4400 | 0.4433 | 0.4400 | 0.4400 | 0.4400 | +0.0000 | +0.1900 | +0.0033 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| C | original | 0.2500 | 0.4400 | 0.4433 | 0.4433 | 0.4433 | 0.4433 | +0.0033 | +0.1933 | +0.0000 | +0.0000 | 0.6667 | 1.0000 | 0.0000 |
| A | counterfactual_cov0.25 | 0.2500 | 0.4400 | 0.4433 | 0.4400 | 0.4400 | 0.4400 | +0.0000 | +0.1900 | +0.0033 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| B | counterfactual_cov0.25 | 0.2500 | 0.4400 | 0.4433 | 0.4400 | 0.4400 | 0.4400 | +0.0000 | +0.1900 | +0.0033 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| C | counterfactual_cov0.25 | 0.2500 | 0.4400 | 0.4433 | 0.4400 | 0.4400 | 0.4400 | +0.0000 | +0.1900 | +0.0033 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| A | counterfactual_cov0.50 | 0.2500 | 0.4400 | 0.4433 | 0.4400 | 0.4400 | 0.4400 | +0.0000 | +0.1900 | +0.0033 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| B | counterfactual_cov0.50 | 0.2500 | 0.4400 | 0.4433 | 0.4400 | 0.4400 | 0.4400 | +0.0000 | +0.1900 | +0.0033 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| C | counterfactual_cov0.50 | 0.0500 | 0.4400 | 0.4433 | 0.4400 | 0.4400 | 0.4400 | +0.0000 | +0.3900 | +0.0033 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |

## 5. Results at observe_cost = 0.03

| Test | Condition | always_try | always_obs | oracle_sel | learned | sig_lookup | oracle_belief | ob-ao | ob-at | osel-ob | learned-ob | ob_rate | pos_obs | nonpos_obs |
|------|-----------|------------|------------|------------|---------|------------|---------------|-------|-------|---------|------------|--------|---------|------------|
| A | original | 0.2500 | 0.4200 | 0.4300 | 0.4200 | 0.4200 | 0.4200 | +0.0000 | +0.1700 | +0.0100 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| B | original | 0.2500 | 0.4200 | 0.4300 | 0.4200 | 0.4200 | 0.4200 | +0.0000 | +0.1700 | +0.0100 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| C | original | 0.2500 | 0.4200 | 0.4300 | 0.4300 | 0.4300 | 0.4300 | +0.0100 | +0.1800 | +0.0000 | +0.0000 | 0.6667 | 1.0000 | 0.0000 |
| A | counterfactual_cov0.25 | 0.2500 | 0.4200 | 0.4300 | 0.4200 | 0.4200 | 0.4200 | +0.0000 | +0.1700 | +0.0100 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| B | counterfactual_cov0.25 | 0.2500 | 0.4200 | 0.4300 | 0.4200 | 0.4200 | 0.4200 | +0.0000 | +0.1700 | +0.0100 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| C | counterfactual_cov0.25 | 0.2500 | 0.4200 | 0.4300 | 0.4200 | 0.4200 | 0.4200 | +0.0000 | +0.1700 | +0.0100 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| A | counterfactual_cov0.50 | 0.2500 | 0.4200 | 0.4300 | 0.4200 | 0.4200 | 0.4200 | +0.0000 | +0.1700 | +0.0100 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| B | counterfactual_cov0.50 | 0.2500 | 0.4200 | 0.4300 | 0.4200 | 0.4200 | 0.4200 | +0.0000 | +0.1700 | +0.0100 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| C | counterfactual_cov0.50 | 0.0500 | 0.4200 | 0.4300 | 0.4200 | 0.4200 | 0.4200 | +0.0000 | +0.3700 | +0.0100 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |

## 5. Results at observe_cost = 0.05

| Test | Condition | always_try | always_obs | oracle_sel | learned | sig_lookup | oracle_belief | ob-ao | ob-at | osel-ob | learned-ob | ob_rate | pos_obs | nonpos_obs |
|------|-----------|------------|------------|------------|---------|------------|---------------|-------|-------|---------|------------|--------|---------|------------|
| A | original | 0.2500 | 0.4000 | 0.4167 | 0.4000 | 0.4000 | 0.4000 | +0.0000 | +0.1500 | +0.0167 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| B | original | 0.2500 | 0.4000 | 0.4167 | 0.4000 | 0.4000 | 0.4000 | +0.0000 | +0.1500 | +0.0167 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| C | original | 0.2500 | 0.4000 | 0.4167 | 0.4167 | 0.4167 | 0.4167 | +0.0167 | +0.1667 | +0.0000 | +0.0000 | 0.6667 | 1.0000 | 0.0000 |
| A | counterfactual_cov0.25 | 0.2500 | 0.4000 | 0.4167 | 0.4000 | 0.4000 | 0.4000 | +0.0000 | +0.1500 | +0.0167 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| B | counterfactual_cov0.25 | 0.2500 | 0.4000 | 0.4167 | 0.4000 | 0.4000 | 0.4000 | +0.0000 | +0.1500 | +0.0167 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| C | counterfactual_cov0.25 | 0.2500 | 0.4000 | 0.4167 | 0.4000 | 0.4000 | 0.4000 | +0.0000 | +0.1500 | +0.0167 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| A | counterfactual_cov0.50 | 0.2500 | 0.4000 | 0.4167 | 0.4000 | 0.4000 | 0.4000 | +0.0000 | +0.1500 | +0.0167 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| B | counterfactual_cov0.50 | 0.2500 | 0.4000 | 0.4167 | 0.4000 | 0.4000 | 0.4000 | +0.0000 | +0.1500 | +0.0167 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| C | counterfactual_cov0.50 | 0.0500 | 0.4000 | 0.4167 | 0.4000 | 0.4000 | 0.4000 | +0.0000 | +0.3500 | +0.0167 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |

## 5. Results at observe_cost = 0.07

| Test | Condition | always_try | always_obs | oracle_sel | learned | sig_lookup | oracle_belief | ob-ao | ob-at | osel-ob | learned-ob | ob_rate | pos_obs | nonpos_obs |
|------|-----------|------------|------------|------------|---------|------------|---------------|-------|-------|---------|------------|--------|---------|------------|
| A | original | 0.2500 | 0.3800 | 0.4033 | 0.3800 | 0.3800 | 0.3800 | +0.0000 | +0.1300 | +0.0233 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| B | original | 0.2500 | 0.3800 | 0.4033 | 0.3800 | 0.3800 | 0.3800 | +0.0000 | +0.1300 | +0.0233 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| C | original | 0.2500 | 0.3800 | 0.4033 | 0.4033 | 0.4033 | 0.4033 | +0.0233 | +0.1533 | +0.0000 | +0.0000 | 0.6667 | 1.0000 | 0.0000 |
| A | counterfactual_cov0.25 | 0.2500 | 0.3800 | 0.4033 | 0.3800 | 0.3800 | 0.3800 | +0.0000 | +0.1300 | +0.0233 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| B | counterfactual_cov0.25 | 0.2500 | 0.3800 | 0.4033 | 0.3800 | 0.3800 | 0.3800 | +0.0000 | +0.1300 | +0.0233 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| C | counterfactual_cov0.25 | 0.2500 | 0.3800 | 0.4033 | 0.3800 | 0.3800 | 0.3800 | +0.0000 | +0.1300 | +0.0233 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| A | counterfactual_cov0.50 | 0.2500 | 0.3800 | 0.4033 | 0.3800 | 0.3800 | 0.3800 | +0.0000 | +0.1300 | +0.0233 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| B | counterfactual_cov0.50 | 0.2500 | 0.3800 | 0.4033 | 0.3800 | 0.3800 | 0.3800 | +0.0000 | +0.1300 | +0.0233 | +0.0000 | 1.0000 | 1.0000 | 1.0000 |
| C | counterfactual_cov0.50 | 0.0500 | 0.3800 | 0.4033 | 0.3723 | 0.3800 | 0.3800 | +0.0000 | +0.3300 | +0.0233 | -0.0077 | 1.0000 | 1.0000 | 1.0000 |

## 6. Headroom Verdicts

| Cost | Condition | Verdict | ob_vs_always_obs(A) | ob_rate(A) | pos_obs | nonpos_obs | Selective? | Discriminates? |
|------|-----------|---------|---------------------|------------|---------|------------|------------|----------------|
| 0.01 | counterfactual_cov0.25 | FAIL: insufficient selective-observe headroom | +0.0000 | 1.0000 | 1.0000 | 1.0000 | False | False |
| 0.01 | counterfactual_cov0.50 | FAIL: insufficient selective-observe headroom | +0.0000 | 1.0000 | 1.0000 | 1.0000 | False | False |
| 0.01 | original | FAIL: insufficient selective-observe headroom | +0.0000 | 1.0000 | 1.0000 | 1.0000 | False | False |
| 0.03 | counterfactual_cov0.25 | FAIL: insufficient selective-observe headroom | +0.0000 | 1.0000 | 1.0000 | 1.0000 | False | False |
| 0.03 | counterfactual_cov0.50 | FAIL: insufficient selective-observe headroom | +0.0000 | 1.0000 | 1.0000 | 1.0000 | False | False |
| 0.03 | original | FAIL: insufficient selective-observe headroom | +0.0000 | 1.0000 | 1.0000 | 1.0000 | False | False |
| 0.05 | counterfactual_cov0.25 | FAIL: insufficient selective-observe headroom | +0.0000 | 1.0000 | 1.0000 | 1.0000 | False | False |
| 0.05 | counterfactual_cov0.50 | FAIL: insufficient selective-observe headroom | +0.0000 | 1.0000 | 1.0000 | 1.0000 | False | False |
| 0.05 | original | FAIL: insufficient selective-observe headroom | +0.0000 | 1.0000 | 1.0000 | 1.0000 | False | False |
| 0.07 | counterfactual_cov0.25 | FAIL: insufficient selective-observe headroom | +0.0000 | 1.0000 | 1.0000 | 1.0000 | False | False |
| 0.07 | counterfactual_cov0.50 | FAIL: insufficient selective-observe headroom | +0.0000 | 1.0000 | 1.0000 | 1.0000 | False | False |
| 0.07 | original | FAIL: insufficient selective-observe headroom | +0.0000 | 1.0000 | 1.0000 | 1.0000 | False | False |

## 7. Interpretation

**Counterfactual distribution still lacks sufficient selective-observe headroom;** 
**do not implement 1J40b-8 yet.**

The oracle-belief upper bound does NOT materially beat always_observe on held-out 
tests. Even with oracle knowledge of empirical observe usefulness by signature, 
the counterfactual distribution does not provide enough advantage for selective 
observation to outperform always-observe.

## 8. Commands Run

```
& "D:\conda\python.exe" "_block1j40b7f_oracle_belief_upper_bound_control.py"
```

```
[block_done]
block_id=1J40b-7f
implementation_status=COMPLETED
```
