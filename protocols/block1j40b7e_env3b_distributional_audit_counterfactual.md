# Block 1J40b-7e: env3b Distributional Audit + Minimal Counterfactual Coverage

- **Implementation Status**: COMPLETED
- **Elapsed**: 47.7s
- **Seeds**: [101, 103, 107, 109, 113]
- **Coverage Fractions**: [0.25, 0.5]

## 1. Files

- `_block1j40b7e_env3b_distributional_audit_counterfactual.py` -- Written
- `runs/block1j40b7e_env3b_distributional_audit_counterfactual.json` -- Generated
- `protocols/block1j40b7e_env3b_distributional_audit_counterfactual.md` -- Generated
- `protocols/block1j40b7e_env3b_distributional_audit_counterfactual_table.csv` -- Generated

## 2. Distributional Audit (TRUE/ORACLE VOI Primary)

**Method**: Pre-oracle computed from legal pre_signature groups only (NOT hidden category or diagnostic feature identity). Q-based targets reported as sanity check only.

### Original Distribution (cost=0.03, TRUE/ORACLE)

| Signature | n | +true | -true | frac+ | entropy | Deterministic? |
|-----------|----|-------|-------|-------|---------|----------------|
| ('brownish', 'heavy_weight', 'long_shape', 'rough_ | 30 | 30 | 0 | 1.0000 | 0.0000 | True |
| ('brownish', 'heavy_weight', 'long_shape', 'rough_ | 30 | 30 | 0 | 1.0000 | 0.0000 | True |
| ('discolored', 'greenish', 'light_weight', 'round_ | 30 | 30 | 0 | 1.0000 | 0.0000 | True |
| ('greenish', 'light_weight', 'round_small', 'smoot | 30 | 30 | 0 | 1.0000 | 0.0000 | True |
| ('brownish', 'discolored', 'has_bark_texture', 'he | 15 | 0 | 15 | 0.0000 | 0.0000 | True |
| ('brownish', 'has_crystal_flecks', 'heavy_weight', | 15 | 0 | 15 | 0.0000 | 0.0000 | True |
| ('greenish', 'has_grip_area', 'light_weight', 'rou | 15 | 0 | 15 | 0.0000 | 0.0000 | True |
| ('greenish', 'has_stem_remnant', 'light_weight', ' | 15 | 0 | 15 | 0.0000 | 0.0000 | True |

**Result**: 8/8 signatures are TRUE-deterministic at cost=0.03. Every training signature perfectly predicts the optimal observe decision. Signature lookup is the optimal training strategy.

## 3. Counterfactual Coverage Interventions

**IMPORTANT**: These are EXPLICIT INTERVENTIONS on the training distribution, NOT a natural env3b redesign.

### Intervention Types

| Signature Type | Mechanism | Effect | Label |
|---------------|-----------|--------|-------|
| Shared (positive_net) | Flat affordance (all try actions same outcome) | Observation decision-irrelevant (pre_best == post_best, VOI = -cost) | `decision_irrelevant_flat` |
| Unique (non_positive_net) | Swap to other category's affordance in same ambient group | Lookalike/deceptive/exception: pre-surface diagnostic feature misleading, post-observe corrective | `lookalike_deceptive_exception` |

### Post-Coverage Signature Determinism

| Cost | Coverage | Deterministic | Total | Fraction | Signatures Broken |
|------|----------|--------------|-------|----------|-------------------|
| 0.03 | 0.25 | 4 | 8 | 0.5000 | 4 |
| 0.03 | 0.5 | 4 | 8 | 0.5000 | 4 |
| 0.05 | 0.25 | 4 | 8 | 0.5000 | 4 |
| 0.05 | 0.5 | 4 | 8 | 0.5000 | 4 |

### Corrective Post-Evidence Audit (Lookalike Interventions)

Verifies that unique-signature -> positive counterfactual interventions have post_observe_features containing diagnostic evidence consistent with the counterfactual affordance profile.

| Coverage | n Lookalike | n Corrective | n Missing | Pass Rate | Design Valid |
|----------|-------------|--------------|-----------|-----------|--------------|
| 0.25 | 12 | 12 | 0 | 1.0000 | True |
| 0.5 | 28 | 28 | 0 | 1.0000 | True |


## 4. Question A: Does counterfactual coverage break signature determinism?

**Original**: 8/8 signatures deterministic. All shared sigs: frac+=1.0. All unique sigs: frac+=0.0. Entropy=0 for all signatures.

**After counterfactual coverage**:
- Cost=0.03, coverage=0.25: 4/8 signatures have mixed VOI (entropy > 0)
  - ('brownish', 'heavy_weight', 'long_shape', 'rough_: frac+=0.9667, entropy=0.2108
  - ('brownish', 'heavy_weight', 'long_shape', 'rough_: frac+=0.9333, entropy=0.3534
  - ('discolored', 'greenish', 'light_weight', 'round_: frac+=0.8000, entropy=0.7219
  - ('greenish', 'light_weight', 'round_small', 'smoot: frac+=0.8333, entropy=0.6500
- Cost=0.03, coverage=0.5: 4/8 signatures have mixed VOI (entropy > 0)
  - ('brownish', 'heavy_weight', 'long_shape', 'rough_: frac+=0.7000, entropy=0.8813
  - ('brownish', 'heavy_weight', 'long_shape', 'rough_: frac+=0.7333, entropy=0.8366
  - ('discolored', 'greenish', 'light_weight', 'round_: frac+=0.7000, entropy=0.8813
  - ('greenish', 'light_weight', 'round_small', 'smoot: frac+=0.7000, entropy=0.8813
- Cost=0.05, coverage=0.25: 4/8 signatures have mixed VOI (entropy > 0)
  - ('brownish', 'heavy_weight', 'long_shape', 'rough_: frac+=0.9667, entropy=0.2108
  - ('brownish', 'heavy_weight', 'long_shape', 'rough_: frac+=0.9333, entropy=0.3534
  - ('discolored', 'greenish', 'light_weight', 'round_: frac+=0.8000, entropy=0.7219
  - ('greenish', 'light_weight', 'round_small', 'smoot: frac+=0.8333, entropy=0.6500
- Cost=0.05, coverage=0.5: 4/8 signatures have mixed VOI (entropy > 0)
  - ('brownish', 'heavy_weight', 'long_shape', 'rough_: frac+=0.7000, entropy=0.8813
  - ('brownish', 'heavy_weight', 'long_shape', 'rough_: frac+=0.7333, entropy=0.8366
  - ('discolored', 'greenish', 'light_weight', 'round_: frac+=0.7000, entropy=0.8813
  - ('greenish', 'light_weight', 'round_small', 'smoot: frac+=0.7000, entropy=0.8813

## 5. Question B: Does the learned policy improve on held-out Test A/B?

### Primary (cost=0.03)

| Test | Condition | Net | ObsRate | PosObs | NonposObs | vsAlwaysObs | vsOracle | Collapsed? |
|------|-----------|-----|---------|--------|-----------|-------------|----------|------------|
| A | original | 0.4200 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0100 | always-observe |
| A | cf_cov0.25 | 0.4200 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0100 | always-observe |
| A | cf_cov0.5 | 0.3700 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0600 | always-observe |
| B | original | 0.4200 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0100 | always-observe |
| B | cf_cov0.25 | 0.4200 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0100 | always-observe |
| B | cf_cov0.5 | 0.4200 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0100 | always-observe |
| C | original | 0.4300 | 0.6667 | 1.0000 | 0.0000 | +0.0100 | +0.0000 | no |
| C | cf_cov0.25 | 0.4200 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0100 | always-observe |
| C | cf_cov0.5 | 0.4200 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0100 | always-observe |

### Robustness (cost=0.05)

| Test | Condition | Net | ObsRate | PosObs | NonposObs | vsAlwaysObs | vsOracle | Collapsed? |
|------|-----------|-----|---------|--------|-----------|-------------|----------|------------|
| A | original | 0.4000 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0167 | always-observe |
| A | cf_cov0.25 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0167 | always-observe |
| A | cf_cov0.5 | 0.3500 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0667 | always-observe |
| B | original | 0.4000 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0167 | always-observe |
| B | cf_cov0.25 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0167 | always-observe |
| B | cf_cov0.5 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0167 | always-observe |
| C | original | 0.4167 | 0.6667 | 1.0000 | 0.0000 | +0.0167 | +0.0000 | no |
| C | cf_cov0.25 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0167 | always-observe |
| C | cf_cov0.5 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0167 | always-observe |

## 6. Question C: Does signature_lookup_policy remain strong?

A signature_lookup_policy means the observe_value_model predictions are perfectly predictable from signature identity alone.

### Primary (cost=0.03)

- original Test A: obs_rate=1.0000 (COLLAPSED)
- original Test B: obs_rate=1.0000 (COLLAPSED)
- original Test C: obs_rate=0.6667 (selective)
- cf_cov0.25 Test A: obs_rate=1.0000 (COLLAPSED)
- cf_cov0.25 Test B: obs_rate=1.0000 (COLLAPSED)
- cf_cov0.25 Test C: obs_rate=1.0000 (COLLAPSED)
- cf_cov0.5 Test A: obs_rate=1.0000 (COLLAPSED)
- cf_cov0.5 Test B: obs_rate=1.0000 (COLLAPSED)
- cf_cov0.5 Test C: obs_rate=1.0000 (COLLAPSED)

### Robustness (cost=0.05)

- original Test A: obs_rate=1.0000 (COLLAPSED)
- original Test B: obs_rate=1.0000 (COLLAPSED)
- original Test C: obs_rate=0.6667 (selective)
- cf_cov0.25 Test A: obs_rate=1.0000 (COLLAPSED)
- cf_cov0.25 Test B: obs_rate=1.0000 (COLLAPSED)
- cf_cov0.25 Test C: obs_rate=1.0000 (COLLAPSED)
- cf_cov0.5 Test A: obs_rate=1.0000 (COLLAPSED)
- cf_cov0.5 Test B: obs_rate=1.0000 (COLLAPSED)
- cf_cov0.5 Test C: obs_rate=1.0000 (COLLAPSED)

## 7. Overall Assessment

**Critical constraint (8)**: Do NOT claim success if same-signature mixed coverage exists but learned_policy still collapses to always_observe.

**Corrective evidence constraint**: Lookalike interventions require corrective post-observe evidence consistent with the counterfactual affordance profile. If absent, the counterfactual design is INVALID.

| Cost | Coverage | Sigs Broken | TestA Collapsed | TestA > AlwaysObs | TestC Passes | Corrective Evid | Design Valid | Verdict |
|------|----------|-------------|-----------------|-------------------|--------------|-----------------|--------------|---------|
| 0.03 | 0.25 | 4/8 | True | False | False | 1.0000 | True | FAIL: Mixed coverage exists but policy still collapses to always-observe on Test A |
| 0.03 | 0.5 | 4/8 | True | False | False | 1.0000 | True | FAIL: Mixed coverage exists but policy still collapses to always-observe on Test A |
| 0.05 | 0.25 | 4/8 | True | False | False | 1.0000 | True | FAIL: Mixed coverage exists but policy still collapses to always-observe on Test A |
| 0.05 | 0.5 | 4/8 | True | False | False | 1.0000 | True | FAIL: Mixed coverage exists but policy still collapses to always-observe on Test A |

## 8. Commands Run

```
& "D:\conda\python.exe" "_block1j40b7e_env3b_distributional_audit_counterfactual.py"
```

```
[block_done]
block_id=1J40b-7e
implementation_status=COMPLETED
```
