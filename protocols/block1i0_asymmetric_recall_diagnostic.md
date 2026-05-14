# Block 1I0: Asymmetric Recall / Calibration Diagnostic

**Date:** 2026-05-10
**Condition:** C3_P060_O040, budget=1.5, CW=0.5, seed=101
**Primary metric:** macro_query_balanced_accuracy

---

## 1. C0b vs C13 Confusion Table

| GT | c0b_pred | c13_pred | Count |
|----|----------|----------|-------|
| all | c0b=False | c13=False | 253 |
| all | c0b=True | c13=False | 29 |
| all | c0b=True | c13=True | 14 |
| all | c0b=False | c13=True | 4 |
| gt=False | c0b=False | c13=False | 189 |
| gt=False | c0b=True | c13=False | 16 |
| gt=False | c0b=True | c13=True | 5 |
| gt=True | c0b=False | c13=False | 64 |
| gt=True | c0b=True | c13=False | 13 |
| gt=True | c0b=True | c13=True | 9 |
| gt=True | c0b=False | c13=True | 4 |


## 2. C0b -> C13 Prediction Transition Table

| GT | c0b -> c13 | Count | Pct |
|----|-----------|-------|-----|
| True | False -> False | 64 | 71.1% |
| True | False -> True | 4 | 4.4% |
| True | True -> False | 13 | 14.4% |
| True | True -> True | 9 | 10.0% |
| False | False -> False | 189 | 90.0% |
| False | True -> False | 16 | 7.6% |
| False | True -> True | 5 | 2.4% |


## 3. Diagnostic A: Threshold Miscalibration

### Reliability Deciles

**need_planks** (n=60):

| bin_low | bin_high | n | mean_post | emp_pos_rate |
|---------|----------|---|-----------|---------------|
| 0.0000 | 0.0999 | 6 | 0.0833 | 0.0000 |
| 0.0999 | 0.0999 | 6 | 0.0999 | 0.0000 |
| 0.0999 | 0.0999 | 6 | 0.0999 | 0.0000 |
| 0.0999 | 0.0999 | 6 | 0.0999 | 0.8333 |
| 0.0999 | 0.0999 | 6 | 0.0999 | 0.5000 |
| 0.0999 | 0.0999 | 6 | 0.0999 | 0.0000 |
| 0.0999 | 0.1968 | 6 | 0.1186 | 0.0000 |
| 0.1994 | 0.2946 | 6 | 0.2317 | 0.3333 |
| 0.2962 | 0.3961 | 6 | 0.3154 | 0.3333 |
| 0.3980 | 1.0000 | 6 | 0.6346 | 0.5000 |

**need_stone** (n=60):

| bin_low | bin_high | n | mean_post | emp_pos_rate |
|---------|----------|---|-----------|---------------|
| 0.0000 | 0.1960 | 6 | 0.0994 | 0.0000 |
| 0.1979 | 0.2972 | 6 | 0.2491 | 0.3333 |
| 0.2979 | 0.4013 | 6 | 0.3185 | 0.3333 |
| 0.4885 | 0.4885 | 6 | 0.4885 | 0.0000 |
| 0.4885 | 0.4885 | 6 | 0.4885 | 0.3333 |
| 0.4885 | 0.4885 | 6 | 0.4885 | 1.0000 |
| 0.4885 | 0.4885 | 6 | 0.4885 | 0.0000 |
| 0.4885 | 0.4885 | 6 | 0.4885 | 0.0000 |
| 0.4885 | 0.4885 | 6 | 0.4885 | 0.0000 |
| 0.4885 | 1.0000 | 6 | 0.7628 | 0.5000 |

**need_food** (n=60):

| bin_low | bin_high | n | mean_post | emp_pos_rate |
|---------|----------|---|-----------|---------------|
| 0.0000 | 0.1017 | 6 | 0.0503 | 0.0000 |
| 0.1022 | 0.1977 | 6 | 0.1785 | 0.1667 |
| 0.1988 | 0.2026 | 6 | 0.2020 | 0.8333 |
| 0.2026 | 0.2026 | 6 | 0.2026 | 0.8333 |
| 0.2026 | 0.2026 | 6 | 0.2026 | 0.0000 |
| 0.2026 | 0.2026 | 6 | 0.2026 | 0.0000 |
| 0.2026 | 0.2026 | 6 | 0.2026 | 0.0000 |
| 0.2026 | 0.2026 | 6 | 0.2026 | 0.0000 |
| 0.2026 | 0.2984 | 6 | 0.2498 | 0.0000 |
| 0.3041 | 1.0000 | 6 | 0.5169 | 0.6667 |

**need_tool** (n=60):

| bin_low | bin_high | n | mean_post | emp_pos_rate |
|---------|----------|---|-----------|---------------|
| 0.0000 | 0.0000 | 6 | 0.0000 | 0.0000 |
| 0.0000 | 0.0999 | 6 | 0.0488 | 0.0000 |
| 0.1028 | 0.1028 | 6 | 0.1028 | 0.0000 |
| 0.1028 | 0.1028 | 6 | 0.1028 | 0.0000 |
| 0.1028 | 0.1028 | 6 | 0.1028 | 0.0000 |
| 0.1028 | 0.1028 | 6 | 0.1028 | 0.0000 |
| 0.1028 | 0.1028 | 6 | 0.1028 | 0.6667 |
| 0.1028 | 0.1028 | 6 | 0.1028 | 1.0000 |
| 0.1028 | 0.1060 | 6 | 0.1040 | 0.3333 |
| 0.1892 | 1.0000 | 6 | 0.3956 | 0.5000 |

**need_fuel** (n=60):

| bin_low | bin_high | n | mean_post | emp_pos_rate |
|---------|----------|---|-----------|---------------|
| 0.0000 | 0.3014 | 6 | 0.1834 | 0.1667 |
| 0.3014 | 0.3014 | 6 | 0.3014 | 0.0000 |
| 0.3014 | 0.3014 | 6 | 0.3014 | 0.0000 |
| 0.3014 | 0.3014 | 6 | 0.3014 | 0.3333 |
| 0.3014 | 0.3014 | 6 | 0.3014 | 1.0000 |
| 0.3014 | 0.3014 | 6 | 0.3014 | 1.0000 |
| 0.3014 | 0.3014 | 6 | 0.3014 | 1.0000 |
| 0.3021 | 0.4015 | 6 | 0.3510 | 0.5000 |
| 0.4022 | 0.5987 | 6 | 0.4854 | 0.5000 |
| 0.6019 | 1.0000 | 6 | 0.7865 | 0.5000 |

### Threshold Sweep Summary

| Query | Best Thr | Best macro_bal | Best pos_rec | Best neg_rec | Default(0.5) macro_bal |
|-------|----------|----------------|--------------|--------------|------------------------|
| need_planks | 0.3 | 0.6111 | 0.3333 | 0.8889 | 0.6000 |
| need_stone | 0.7 | 0.6000 | 0.2000 | 1.0000 | 0.5889 |
| need_food | 0.3 | 0.6111 | 0.2667 | 0.9556 | 0.5667 |
| need_tool | 0.1 | 0.6333 | 1.0000 | 0.2667 | 0.5333 |
| need_fuel | 0.8 | 0.5500 | 0.1000 | 1.0000 | 0.5000 |

**Global best threshold:** 0.2 (macro_bal=0.5767)
**C13 default (thr=0.5):** macro_bal=0.5578
**Threshold fix promising:** False

## 4. Diagnostic B: Negative Evidence Skew

| Evidence Type | Count | Pct |
|--------------|-------|-----|
| positive_evidence | 18 | 81.8% |
| negative_evidence | 4 | 18.2% |
| ambiguous | 0 | 0.0% |

**Negative:positive ratio:** 0.22 (skewed toward positive)

### Per-Action Probe Rates vs Base Rates

| Action | n_probes | probe_succ | env_base | bias |
|--------|----------|------------|----------|------|
| mine_by_hand | 4 | 0.2500 | 0.7500 | -0.5000 |
| craft_plank | 3 | 0.6667 | 0.2500 | +0.4167 |
| eat | 3 | 0.3333 | 0.2500 | +0.0833 |
| use_as_tool | 8 | 0.1250 | 0.2500 | -0.1250 |
| burn_as_fuel | 4 | 0.5000 | 0.5000 | +0.0000 |


## 5. Diagnostic C: Posterior Separation

| Query | n_true | n_false | mean_true | mean_false | gap | AUC |
|-------|--------|---------|-----------|------------|-----|-----|
| need_planks | 15 | 45 | 0.3060 | 0.1491 | +0.1569 | 0.6193 |
| need_stone | 15 | 45 | 0.5470 | 0.3991 | +0.1478 | 0.6104 |
| need_food | 15 | 45 | 0.3082 | 0.1920 | +0.1163 | 0.6667 |
| need_tool | 15 | 45 | 0.1948 | 0.0904 | +0.1044 | 0.6430 |
| need_fuel | 30 | 30 | 0.3876 | 0.3354 | +0.0522 | 0.5289 |

**Posterior separable:** True

## 6. Diagnostic D: Aggregation Ablation

| Variant | macro_bal | mean_pos_rec | mean_neg_rec |
|---------|-----------|-------------|-------------|
| 1. similarity-weighted (C13 default) | 0.5578 | 0.1467 | 0.9689 |
| 2. unweighted average | 0.5211 | 0.1133 | 0.9289 |
| 3. single nearest instance | 0.5144 | 0.2800 | 0.7489 |

**Max variant diff:** 0.0433
**Aggregation bias supported:** True

## 7. Diagnostic E: Oracle Ceiling

| Metric | Oracle | C13 | C0b |
|--------|--------|-----|-----|
| macro_bal | 1.0000 | 0.5578 | 0.5444 |
| mean_pos_recall | 1.0000 | 0.1467 | 0.2067 |
| mean_neg_recall | 1.0000 | 0.9689 | 0.8822 |

### Per-Query Oracle vs C13

| Query | oracle_pos | c13_pos | oracle_neg | c13_neg |
|-------|-----------|---------|-----------|---------|
| need_planks | 1.0000 | 0.2000 | 1.0000 | 1.0000 |
| need_stone | 1.0000 | 0.2000 | 1.0000 | 0.9778 |
| need_food | 1.0000 | 0.1333 | 1.0000 | 1.0000 |
| need_tool | 1.0000 | 0.0667 | 1.0000 | 1.0000 |
| need_fuel | 1.0000 | 0.1333 | 1.0000 | 0.8667 |

**Oracle positive recoverable:** True

## 8. Diagnosis

| Condition | Detected |
|-----------|----------|
| threshold_miscalibration | no |
| posterior_nonseparation | YES |
| negative_evidence_skew | no |
| aggregation_bias | YES |
| environment_structure_limitation | no |

**Diagnosis label:** `mixed`

### Explicit Boolean Flags

| Flag | Value |
|------|-------|
| posterior_nonseparation_supported | True |
| aggregation_bias_supported | True |
| negative_evidence_skew_supported | False |
| threshold_fix_promising | False |
| strict_environment_structure_limitation_supported | False |
| environment_redesign_still_needed | True |

### Detailed Stats

| Stat | Value |
|------|-------|
| mean_auc | 0.6136 |
| min_auc | 0.5289 |
| n_low_auc (AUC < 0.65) | 4/5 |
| median_overlap_count | 5/5 |
| neg_skew_ratio | 0.22 |
| oracle_positive_recoverable | True |
| c13_mean_pos_recall | 0.1467 |
| c0b_mean_pos_recall | 0.2067 |
| c13_macro_bal | 0.5578 |
| c0b_macro_bal | 0.5444 |

### Interpretation

C13's positive-recall failure is not caused by negative evidence skew.
Probes mostly produce positive evidence.
The main problem is that posterior scores weakly separate true and false cases,
and the current aggregation method favors high negative recall over positive recall.
Aggregation changes the recall trade-off, but tested aggregation variants do not solve the problem.
Oracle shows that positive affordance signal is recoverable in principle.
Therefore strict_environment_structure_limitation_supported = false —
the environment is not structurally impossible.
However, environment_redesign_still_needed = true because current realizable
C13 posterior remains weak, policy gains are too small, and
policy_comparison_ready remains false.
The core issue is a mixed posterior-quality / aggregation / realizable-information problem.

---
*Generated by Block 1I0 — diagnostic only, no policy changes implemented.*
