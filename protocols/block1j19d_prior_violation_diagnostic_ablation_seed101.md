# Block 1J19d -- Prior-Violation Memory Diagnostic Ablation

## 1. Objective

Diagnose why B_full does not beat C_full on prior violations and why B macro drops below tolerance.

## 2. Setup

| Parameter | Value |
|-----------|-------|
| Seed | 101 |
| Episodes | 5 |
| Prior Violation Threshold | 0.6 |
| Risk Alpha | 5.0 |
| Max Risk | 2.0 |
| Injected Deviation Features | ['brittle_surface', 'damp_texture', 'hollow_sound', 'treated_surface'] |
| Deceptive Objects | 13 |

## 3. B_full vs A vs C_full

| Condition | Mean Bal | Total PV | Repeated PV |
|-----------|----------|----------|-------------|
| A | 0.5889 | 19 | - |
| B_full | 0.5662 | 12 | 1 |
| C_full | 0.5705 | 19 | - |
| B delta vs A | -0.0227 | -7 | |
| B beats C on PV | true | | |

## 4. Risk Weight Distribution (B_full)

| Group | Count | Total Abs | Max | Avg | % Mass |
|-------|-------|-----------|-----|-----|--------|
| A_injected_deviation | 14 | 7.0621 | 0.9673 | 0.5044 | 25.5% |
| B_type_family | 56 | 18.8550 | 0.9669 | 0.3367 | 68.1% |
| C_positional_or_context | 6 | 1.3868 | 0.4883 | 0.2311 | 5.0% |
| D_other | 2 | 0.3731 | 0.1929 | 0.1866 | 1.4% |
| Risk memory diffusion | true |

## 5. Macro-Drop Attribution

| Metric | Value |
|--------|-------|
| Primary query | need_food |
| Avoided effective probes | 26 |
| Avoided true-positive probes | 26 |
| Conservative underprobing | true |

## 6. B_dev_only Ablation

| Metric | B_full | B_dev_only | C_dev_only |
|--------|--------|------------|------------|
| Mean Bal | 0.5662 | 0.5843 | 0.5845 |
| Total PV | 12 | 20 | 21 |
| B beats C | true | true | - |
| Macro delta vs A | -0.0227 | -0.0046 | - |

## 7. Risk Penalty Scale Grid

| Scale | Variant | PV | Delta PV | Bal | Delta Bal |
|-------|---------|-----|----------|-----|-----------|
| 0.25 | B_full | 12 | -7 | 0.5662 | -0.0227 |
| 0.25 | B_dev_only | 20 | +1 | 0.5843 | -0.0046 |
| 0.5 | B_full | 12 | -7 | 0.5662 | -0.0227 |
| 0.5 | B_dev_only | 20 | +1 | 0.5843 | -0.0046 |
| 1.0 | B_full | 12 | -7 | 0.5662 | -0.0227 |
| 1.0 | B_dev_only | 20 | +1 | 0.5843 | -0.0046 |

## 8. Interpretation

| Metric | Value |
|--------|-------|
| Diagnostic result | ready_for_small_multiseed |
| Risk memory diffusion | true |
| Penalty too strong | false |
| Insufficient sample | false |
| Best risk penalty scale | 0.25 |
| Best variant | B_full |
| Macro tradeoff acceptable (best) | false |
| Ready for small multiseed | true |
| Next route | run_small_multiseed_1j19d |

## 9. Summary

```
[block_done]
block_id=1J19d
B_full_total_pv=12
C_full_total_pv=19
B_dev_only_total_pv=20
C_dev_only_total_pv=21
B_full_macro_delta_vs_A=-0.0227
B_dev_only_macro_delta_vs_A=-0.0046
best_risk_penalty_scale=0.25
best_variant=B_full
risk_memory_diffusion_detected=true
penalty_too_strong_detected=false
insufficient_sample_problem=false
macro_tradeoff_acceptable=false
recommended_next_route=run_small_multiseed_1j19d
ready_for_small_multiseed=true
feature_risk_memory_supported=false
elapsed=50.3s
```