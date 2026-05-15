# Block 1J19_patch_smoke ¡ª Multi-Episode Feature-Risk Memory Validation

## 1. Objective

Test whether feature-keyed risk memory updated across episodes improves probe selection over the soft-prior-only baseline, and whether any improvement is due to meaningful feature-risk associations (not random perturbation).

## 2. Setup

| Parameter | Value |
|-----------|-------|
| Condition | C4_instance_subtype_cued_v1 |
| Smoke Seed | 101 |
| Budget | 1.5 |
| Episodes | 5 |
| Risk Alpha | 5.0 |
| Max Risk | 2.0 |
| Feature Universe | 24 features |
| Generalization | within_object_set_adaptation |

## 3. Patch Validation

| Check | Value |
|-------|-------|
| IOM isolation | True |
| Absence counts valid | True |
| Hard exclusion used | False |
| Permuted control valid | True |

## 4. Baselines

| Policy | Macro BAcc |
|--------|------------|
| C0b (observe only) | 0.5871 |
| C15b (VOI reserve) | 0.5975 |

## 5. Per-Episode Results

| Episode | A (soft prior) | B (risk memory) | C (permuted) | B Probed | B Eff | B Zero | B Harm |
|---------|----------------|-----------------|--------------|----------|-------|--------|--------|
| 1 | 0.5887 | 0.5887 | 0.5837 | 7 | 7 | 0 | 0 |

| 2 | 0.5921 | 0.5921 | 0.5921 | 5 | 5 | 0 | 0 |

| 3 | 0.5929 | 0.5908 | 0.5742 | 5 | 5 | 0 | 0 |

| 4 | 0.6046 | 0.6025 | 0.6025 | 5 | 5 | 0 | 0 |

| 5 | 0.5958 | 0.5792 | 0.5958 | 4 | 4 | 0 | 0 |

## 6. Aggregated Metrics

| Metric | A (soft prior) | B (risk memory) | C (permuted) |
|--------|----------------|-----------------|--------------|
| Mean macro BAcc | 0.5948 +/- 0.0060 | 0.5907 +/- 0.0083 | 0.5897 +/- 0.0110 |
| Final macro BAcc | 0.5958 | 0.5792 | 0.5958 |
| Learning slope | +0.0027/ep | -0.0009/ep | +0.0035/ep |

## 7. Deltas

| Comparison | Delta |
|------------|-------|
| B - A (real memory vs no memory) | -0.0042 |
| C - A (permuted vs no memory) | -0.0052 |
| B - C (real vs permuted) | +0.0010 |

## 8. Failure Signal Diagnostics

| Metric | Value |
|--------|-------|
| Total effective (B) | 26 |
| Total zero (B) | 0 |
| Total harmful (B) | 0 |
| Failure signal count | 0 |
| Memory keys from failures | 0 |
| Memory keys from successes | 96 |
| Memory update nontrivial | False |
| Total memory updates | 26 |

## 9. Hard Exclusion Check

| Metric | Value |
|--------|-------|
| Min final score overall | 0.8333 |
| Any candidates at floor | False |
| Hard exclusion used | False |

## 10. Interpretation

| Criterion | Result |
|-----------|--------|
| Feature-risk memory supported | False |
| Beats permuted control | True |
| Interpretation | insufficient_failure_signal |
| Next route | increase_episodes_or_reduce_probe_budget |
| Ready for full 1J19 | False |

## 11. Summary

```
[block_done]
block_id=1J19_patch_smoke
iom_isolated=true
absence_counts_valid=true
generalization_mode=within_object_set_adaptation
failure_signal_count=0
memory_update_nontrivial=false
hard_exclusion_used=false
permuted_control_valid=true
ready_for_full_1j19=false
A_mean_macro_bal=0.5948
B_mean_macro_bal=0.5907
C_mean_macro_bal=0.5897
B_delta_vs_A=-0.0042
B_learning_slope=-0.0009
elapsed=28.6s
```