# Block 1J34 ¡ª 5-Seed Fallback Help-vs-Harm Audit

## Per-Seed Audit Summary

| Seed | A PV | B PV | FB PV | Dir Helpful | Dir Harmful | Dir Net | Traj Helpful | Traj Harmful | Ep PV Delta | Ep Net |
|------|------|------|-------|-------------|-------------|---------|--------------|--------------|-------------|--------|
| 101 | 17 | 8 | 9 | 4 | 0 | +4 | 6 | 8 | +1 | -2 |
| 103 | 12 | 9 | 9 | 1 | 0 | +1 | 7 | 7 | +0 | +0 |
| 107 | 15 | 16 | 14 | 2 | 0 | +2 | 17 | 13 | -2 | +4 |
| 109 | 13 | 8 | 7 | 0 | 0 | +0 | 15 | 13 | -1 | +2 |
| 113 | 11 | 11 | 10 | 2 | 0 | +2 | 8 | 6 | -1 | +2 |

## Aggregate Audit Totals

- **Direct helpful:** 9
- **Direct harmful:** 0
- **Trajectory helpful:** 53
- **Trajectory harmful:** 47
- **Episode net PV delta:** -3
- **Episode net helpful:** +6

## Harmful Cases Analysis

| Seed | Direct Harmful | Fallback-Activated Direct Harmful |
|------|----------------|-----------------------------------|
| 101 | 0 | 0 |
| 103 | 0 | 0 |
| 107 | 0 | 0 |
| 109 | 0 | 0 |
| 113 | 0 | 0 |

- **Harmful concentration in seed103:** False
- **Seed103 direct harmful cases:** 0
- **Seed103 fallback-activated direct harmful:** 0

## Per-Seed Mean Metrics

- **Mean A PV:** 13.6
- **Mean B PV:** 10.4
- **Mean FB PV:** 9.8
- **Mean A macro_bal:** 0.5615
- **Mean B macro_bal:** 0.5649
- **Mean FB macro_bal:** 0.5582

## Safety Decision

- **episode_net_pv_delta:** -3
- **harmful_avoidance_risk_detected:** False
- **any_direct_harmful_systematic:** False
- **any_harmful_override_strong_prior:** False
- **hidden_feature_leakage_detected:** False
- **no_oracle_leakage_confirmed:** True
- **fallback_safe_enough_for_next_phase:** True


```
[block_done]
block_id=1J34
direct_helpful=9
direct_harmful=0
trajectory_helpful=53
trajectory_harmful=47
episode_net_pv_delta=-3
seed103_direct_harmful=0
harmful_avoidance_risk_detected=false
fallback_safe_enough_for_next_phase=true
hidden_feature_leakage_detected=false
no_oracle_leakage_confirmed=true
implementation_status=pass
failure_reason=none
```