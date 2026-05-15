# Block 1J19b -- Failure-Signal Elicitation for Negative-Exception Feature-Risk Memory

## 1. Objective

Test whether a patched negative-exception feature-risk memory (V2) improves probe selection when failure signals are present, and verify that the memory is inert (B ~= A) when no failures occur.

## 2. Setup

| Parameter | Value |
|-----------|-------|
| Condition | C4_instance_subtype_cued_v1 |
| Seed | 101 |
| Budget | 1.5 |
| Episodes | 5 |
| Risk Alpha | 5.0 |
| Max Risk | 2.0 |
| Min Failure Support | 2 |
| Exploration Floor | 0.05 |
| Feature Universe | 24 |

## 3. Patched Memory Semantics (V2)

- Risk weight active only if actual failure observations >= min_failure_support
- Only positive risk weights (penalty); negative weights clipped to 0
- Successes calibrate the denominator (P(feature|success)) but don't create bonuses
- When failure_signal_count == 0, all risk weights = 0, so B ~= A

## 4. Baselines

| Policy | Macro BAcc |
|--------|------------|
| C0b (observe only) | 0.5871 |
| C15b (VOI reserve) | 0.5975 |

## 5. Mode: Original Safe

Explore fraction: 0.0  |  Objects: 60

### 5.1 Per-Episode Results

| Ep | A (soft prior) | B (risk memory) | C (permuted) | B Eff | B Zero | B Harm | B RiskAdj |
|----|----------------|-----------------|--------------|-------|--------|--------|-----------|
| 1 | 0.5887 | 0.5887 | 0.5887 | 7 | 0 | 0 | 0.0000 |
| 2 | 0.5921 | 0.5921 | 0.5921 | 7 | 0 | 0 | 0.0000 |
| 3 | 0.5929 | 0.5929 | 0.5929 | 7 | 0 | 0 | 0.0000 |
| 4 | 0.6046 | 0.6046 | 0.6046 | 7 | 0 | 0 | 0.0000 |
| 5 | 0.5958 | 0.5958 | 0.5958 | 7 | 0 | 0 | 0.0000 |

### 5.2 Aggregated Metrics

| Metric | A (soft prior) | B (risk memory) | C (permuted) |
|--------|----------------|-----------------|--------------|
| Mean macro BAcc | 0.5948 +/- 0.0060 | 0.5948 +/- 0.0060 | 0.5948 +/- 0.0060 |
| Final macro BAcc | 0.5958 | 0.5958 | 0.5958 |
| Learning slope | +0.0027/ep | +0.0027/ep | +0.0027/ep |

### 5.3 Deltas

| Comparison | Delta |
|------------|-------|
| B - A | +0.0000 |
| C - A | +0.0000 |
| B - C | +0.0000 |

### 5.4 Probe Effect Diagnostics

| Metric | Value |
|--------|-------|
| Total probes (B) | 35 |
| Effective | 35 |
| Zero | 0 |
| Harmful | 0 |
| Failure signal count | 0 |

### 5.5 Memory Diagnostics

| Metric | Value |
|--------|-------|
| Memory update nontrivial | false |
| Total updates | 35 |
| Failure updates | 0 |
| Success updates | 35 |
| Risk weights nonzero | 0 |
| Max risk weight | 0.0000 |
| Mean risk adjustment | 0.0000 |
| Keys with data | 144 |
| Keys from failures | 0 |
| Keys from successes | 96 |
| Success-only bias removed | true |
| B - A (when no failures) | +0.000000 |

### 5.6 Validity Checks

| Check | Value |
|-------|-------|
| IOM isolated | true |
| Absence counts valid | true |
| Hard exclusion used | false |
| Min final score | 0.8333 |

### 5.7 Top Learned Risk Features

(none -- no risk weights above threshold)

## 5. Mode: Epsilon-Boundary Explore

Explore fraction: 0.3  |  Objects: 60

### 5.1 Per-Episode Results

| Ep | A (soft prior) | B (risk memory) | C (permuted) | B Eff | B Zero | B Harm | B RiskAdj |
|----|----------------|-----------------|--------------|-------|--------|--------|-----------|
| 1 | 0.6071 | 0.6008 | 0.5925 | 7 | 0 | 0 | 0.0000 |
| 2 | 0.5992 | 0.5846 | 0.6221 | 6 | 0 | 0 | 0.0000 |
| 3 | 0.6096 | 0.6117 | 0.6096 | 6 | 0 | 0 | 0.0000 |
| 4 | 0.6000 | 0.5979 | 0.5979 | 6 | 0 | 0 | 0.0000 |
| 5 | 0.5962 | 0.5942 | 0.6025 | 6 | 0 | 0 | 0.0000 |

### 5.2 Aggregated Metrics

| Metric | A (soft prior) | B (risk memory) | C (permuted) |
|--------|----------------|-----------------|--------------|
| Mean macro BAcc | 0.6024 +/- 0.0056 | 0.5978 +/- 0.0099 | 0.6049 +/- 0.0115 |
| Final macro BAcc | 0.5962 | 0.5942 | 0.6025 |
| Learning slope | -0.0021/ep | +0.0000/ep | -0.0004/ep |

### 5.3 Deltas

| Comparison | Delta |
|------------|-------|
| B - A | -0.0046 |
| C - A | +0.0025 |
| B - C | -0.0071 |

### 5.4 Probe Effect Diagnostics

| Metric | Value |
|--------|-------|
| Total probes (B) | 31 |
| Effective | 31 |
| Zero | 0 |
| Harmful | 0 |
| Failure signal count | 0 |

### 5.5 Memory Diagnostics

| Metric | Value |
|--------|-------|
| Memory update nontrivial | false |
| Total updates | 31 |
| Failure updates | 0 |
| Success updates | 31 |
| Risk weights nonzero | 0 |
| Max risk weight | 0.0000 |
| Mean risk adjustment | 0.0000 |
| Keys with data | 144 |
| Keys from failures | 0 |
| Keys from successes | 96 |
| Success-only bias removed | true |
| B - A (when no failures) | -0.004583 |

### 5.6 Validity Checks

| Check | Value |
|-------|-------|
| IOM isolated | true |
| Absence counts valid | true |
| Hard exclusion used | false |
| Min final score | 0.8333 |

### 5.7 Top Learned Risk Features

(none -- no risk weights above threshold)

## 5. Mode: Deceptive High-Prior Diagnostic

Explore fraction: 0.0  |  Objects: 60

### 5.1 Per-Episode Results

| Ep | A (soft prior) | B (risk memory) | C (permuted) | B Eff | B Zero | B Harm | B RiskAdj |
|----|----------------|-----------------|--------------|-------|--------|--------|-----------|
| 1 | 0.5613 | 0.5613 | 0.5613 | 6 | 1 | 0 | 0.0000 |
| 2 | 0.5842 | 0.5842 | 0.5842 | 7 | 0 | 0 | 0.0000 |
| 3 | 0.5758 | 0.5758 | 0.5758 | 7 | 0 | 0 | 0.0000 |
| 4 | 0.5862 | 0.5862 | 0.5714 | 6 | 1 | 0 | 0.0000 |
| 5 | 0.5880 | 0.5797 | 0.5769 | 7 | 0 | 0 | 0.0000 |

### 5.2 Aggregated Metrics

| Metric | A (soft prior) | B (risk memory) | C (permuted) |
|--------|----------------|-----------------|--------------|
| Mean macro BAcc | 0.5791 +/- 0.0110 | 0.5774 +/- 0.0099 | 0.5739 +/- 0.0084 |
| Final macro BAcc | 0.5880 | 0.5797 | 0.5769 |
| Learning slope | +0.0056/ep | +0.0039/ep | +0.0019/ep |

### 5.3 Deltas

| Comparison | Delta |
|------------|-------|
| B - A | -0.0017 |
| C - A | -0.0052 |
| B - C | +0.0035 |

### 5.4 Probe Effect Diagnostics

| Metric | Value |
|--------|-------|
| Total probes (B) | 35 |
| Effective | 33 |
| Zero | 2 |
| Harmful | 0 |
| Failure signal count | 2 |

### 5.5 Memory Diagnostics

| Metric | Value |
|--------|-------|
| Memory update nontrivial | true |
| Total updates | 35 |
| Failure updates | 2 |
| Success updates | 33 |
| Risk weights nonzero | 8 |
| Max risk weight | 0.6109 |
| Mean risk adjustment | 0.0000 |
| Keys with data | 144 |
| Keys from failures | 24 |
| Keys from successes | 120 |
| Success-only bias removed | false |

### 5.6 Validity Checks

| Check | Value |
|-------|-------|
| IOM isolated | true |
| Absence counts valid | true |
| Hard exclusion used | false |
| Min final score | 0.8333 |

### 5.7 Top Learned Risk Features

| Feature | Goal | Action | Risk Weight | FailWith | SuccWith | FailWithout | SuccWithout | ActFail |
|---------|------|--------|-------------|----------|----------|-------------|-------------|---------|
| grayish | need_planks | craft_plank | 0.6109 | 3 | 3 | 1 | 7 | 2 |
| near_table | need_planks | craft_plank | 0.4191 | 3 | 4 | 1 | 6 | 2 |
| brownish | need_planks | craft_plank | 0.4191 | 3 | 4 | 1 | 6 | 2 |
| on_left_side | need_planks | craft_plank | 0.4191 | 3 | 4 | 1 | 6 | 2 |
| light_weight | need_planks | craft_plank | 0.1488 | 1 | 2 | 3 | 8 | 2 |
| movable | need_planks | craft_plank | 0.1488 | 3 | 6 | 1 | 4 | 2 |
| has_grip_area | need_planks | craft_plank | 0.1488 | 1 | 2 | 3 | 8 | 2 |
| heavy_weight | need_planks | craft_plank | 0.1488 | 3 | 6 | 1 | 4 | 2 |
## 6. Cross-Mode Interpretation

| Criterion | Value |
|-----------|-------|
| Mode 1 (original) failure signal count | 0 |
| Mode 2 (boundary) failure signal count | 0 |
| Mode 3 (deceptive) failure signal count | 2 |
| Mode 3 has observable deviations | true |
| Mode 3 deceptive objects | 13 |
| Patched: B ~= A when no failures | true |
| Feature-risk memory supported | false |
| Beats permuted control | false |
| Hard exclusion used | false |
| Success-only bias removed | false |
| Ready for full 1J19 | false |
| Next route | inspect_feature_space_and_risk_update |

## 7. Summary

```
[block_done]
block_id=1J19b
mode_original_failure_signal_count=0
mode_boundary_failure_signal_count=0
mode_deceptive_failure_signal_count=2
mode3_has_observable_deviations=true
patched_no_failure_B_matches_A=true
feature_risk_memory_supported=false
beats_permuted_control=false
hard_exclusion_used=false
success_only_bias_removed=false
ready_for_full_1j19=false
next_recommended_route=inspect_feature_space_and_risk_update
elapsed=38.7s
```