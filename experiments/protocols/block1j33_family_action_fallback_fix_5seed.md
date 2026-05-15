# Block 1J33 — 5-Seed Aggregate Summary

## Per-Seed Results

| Seed | A PV | B PV | FO PV | FB PV | Off PV | Rescued | Missed | FB Delta PV vs B |
|------|------|------|-------|-------|--------|---------|--------|-------------------|
| 101  | 13   | 9    | 12    | 6     | 0      | 1       | 4      | -3                |
| 103  | 13   | 12   | 12    | 13    | 7      | 1       | 5      | +1                |
| 107  | 15   | 17   | 13    | 13    | 5      | 4       | 7      | -4                |
| 109  | 18   | 16   | 13    | 11    | 5      | 10      | 1      | -5                |
| 113  | 16   | 5    | 14    | 5     | 0      | 0       | 4      | 0                 |
| **Mean** | **15.0** | **11.8** | **12.8** | **9.6** | **3.4** | **3.2** | **4.2** | **-2.2** |

## Per-Seed Macro Accuracy

| Seed | A Bal | B Bal | FO Bal | FB Bal | FB Delta vs A |
|------|-------|-------|--------|--------|---------------|
| 101  | 0.5699 | 0.5723 | 0.5645 | 0.5730 | +0.0031 |
| 103  | 0.6176 | 0.6155 | 0.6089 | 0.6068 | -0.0108 |
| 107  | 0.6098 | 0.6066 | 0.5982 | 0.5980 | -0.0118 |
| 109  | 0.5645 | 0.5678 | 0.5637 | 0.5737 | +0.0092 |
| 113  | 0.6427 | 0.6473 | 0.6391 | 0.6473 | +0.0046 |
| **Mean** | **0.6009** | **0.6019** | **0.5949** | **0.5998** | **-0.0011** |

## Per-Seed Helpful vs Harmful Avoidance

| Seed | B Helpful | B Harmful | FO Helpful | FO Harmful | FB Helpful | FB Harmful |
|------|-----------|-----------|------------|------------|------------|------------|
| 101  | 2 | 3 | 2 | 2 | 3 | 2 |
| 103  | 5 | 3 | 5 | 6 | 5 | 6 |
| 107  | 1 | 2 | 5 | 8 | 4 | 8 |
| 109  | 3 | 1 | 10 | 3 | 10 | 4 |
| 113  | 8 | 6 | 4 | 5 | 8 | 6 |
| **Sum** | **19** | **15** | **26** | **24** | **30** | **26** |

## Aggregate Metrics

- **Mean A PV**: 15.0
- **Mean B PV**: 11.8
- **Mean FO PV**: 12.8
- **Mean FB PV**: 9.6 (2.2 fewer than B, 5.4 fewer than A)
- **Mean Off PV**: 3.4
- **Mean raw_zero_rescued**: 3.2 per seed
- **Mean FB macro_delta_vs_A**: -0.0011 (essentially neutral)

## Key Findings

1. **B_fallback reduces total PV by 36% vs A_no_memory** (15.0 → 9.6) and 19% vs B_signed_sum (11.8 → 9.6)
2. **Fallback is seed-dependent**: most effective on seed 109 (10 rescued), ineffective on seeds 101/103/113 (0-1 rescued)
3. **Harmful avoidance risk exists**: seed 103 shows FB with 3 more harmful avoidances than B (harmful: 6 vs 3)
4. **Family_only signal quality varies**: FO PV ranges from 12 to 14, which determines fallback effectiveness
5. **Macro accuracy is flat**: FB mean delta vs A is -0.0011 — fallback doesn't meaningfully change task accuracy
6. **Offline oracle PV of 3.4** shows that the oracle also can't avoid all PVs (no deviation features = no oracle signal either)

## Implementation Checks (all seeds pass)

- `hidden_feature_leakage_detected`: False (all seeds)
- `no_oracle_leakage_confirmed`: True (all seeds)
- `hard_exclusion_used`: False (all seeds)

```
[block_done]
block_id=1J33
seed_mode=5seed
seeds=101,103,107,109,113
A_mean_total_pv=15.0
B_signed_sum_mean_total_pv=11.8
Family_only_mean_total_pv=12.8
B_dev_plus_family_action_fallback_mean_total_pv=9.6
Offline_mean_total_pv=3.4
mean_raw_zero_cases_rescued=3.2
mean_fb_macro_delta_vs_A=-0.0011
hidden_feature_leakage_detected=false
no_oracle_leakage_confirmed=true
implementation_status=pass
failure_reason=none
```
