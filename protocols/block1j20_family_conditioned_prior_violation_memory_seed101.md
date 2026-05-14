# Block 1J20 -- Family-Conditioned Prior-Violation Memory Diagnostic

## 1. Objective

Test whether family-conditioned compound keys (goal, action, family, dev_feature) can learn deviation-specific risk without type-family diffusion.

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

## 3. Main Comparison Table

| Variant | PV | DecPV | NorPV | Bal | dBAL | Eff | AvdEff | NZw |
|---------|-----|-------|-------|-----|------|-----|--------|-----|
| A_no_memory | 22 | 7 | 15 | 0.5937 | +0.0000 | 35 | 0 | 0 |
| B_full_old | 15 | 4 | 11 | 0.5712 | -0.0165 | 21 | 21 | 89 |
| B_dev_only_old | 22 | 4 | 18 | 0.5885 | +0.0056 | 35 | 7 | 2 |
| B_family_dev | 21 | 5 | 16 | 0.5899 | -0.0038 | 35 | 6 | 2 |
| C_family_dev_permuted | 24 | 2 | 22 | 0.5890 | +0.0057 | 34 | 8 | 2 |
| C_family_only | 16 | 6 | 10 | 0.5745 | -0.0095 | 25 | 19 | 7 |

## 4. Part 1: Risk Mass by Key Type (B_family_dev)

| Key Type | Mass |
|----------|------|
| family_only_key_mass | 0.0000 |
| deviation_only_key_mass | 0.0000 |
| family_conditioned_deviation_key_mass | 1.7225 |
| other_key_mass | 0.0000 |
| family_diffusion_leak_detected | false |

## 5. Part 2: Compound Key Support Table (B_family_dev)

| Metric | Value |
|--------|-------|
| num_keys_total | 46 |
| num_keys_with_nonzero_weight | 2 |
| num_keys_above_support_threshold | 2 |
| insufficient_family_deviation_support | true |

### Top 10 Keys by Weight

| Key | Risk | VWith | NVWith | VWithout | NVWithout | Support | Shrinkage |
|-----|------|-------|--------|----------|-----------|---------|-----------|
| need_food/eat/apple-like/brittle_surface | 0.8613 | 4 | 1 | 4 | 6 | 11 | 0.6875 |
| need_food/eat/apple-like/damp_texture | 0.8613 | 4 | 1 | 4 | 6 | 11 | 0.6875 |

## 6. Part 3: Selection Differences (B_family_dev vs A)

| Metric | Value |
|--------|-------|
| avoided_prior_violation_count | 4 |
| avoided_nonviolation_count | 2 |
| avoided_effective_probe_count | 6 |
| avoided_true_positive_probe_count | 6 |
| conservative_underprobing_detected | false |

## 7. Part 5: Implementation Sanity Checks

| Metric | Value |
|--------|-------|
| family_diffusion_leak_detected | false |
| insufficient_family_deviation_support | true |
| still_family_diffusion_dominated | true |
| conservative_underprobing_detected | false |
| hard_exclusion_used | false |
| deceptive_family_preservation_valid | true |
| implementation_status | partial |
| failure_reason | insufficient_family_deviation_support |

## 8. Summary

```
[block_done]
block_id=1J20
A_total_pv=22
B_full_old_total_pv=15
B_dev_only_old_total_pv=22
B_family_dev_total_pv=21
C_family_dev_permuted_total_pv=24
C_family_only_total_pv=16
B_family_dev_macro_delta_vs_A=-0.0038
B_family_dev_deceptive_pv_delta_vs_A=-2
family_diffusion_leak_detected=false
insufficient_family_deviation_support=true
still_family_diffusion_dominated=true
conservative_underprobing_detected=false
deceptive_family_preservation_valid=true
hard_exclusion_used=false
implementation_status=partial
failure_reason=insufficient_family_deviation_support
elapsed=62.7s
```