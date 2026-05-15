# Block 1J40b-7c: env3b Held-Out Signature Stress Audit

- **Implementation Status**: FAIL
- **Elapsed**: 4.4s

## 1. Files

- `_block1j40b7c_env3b_heldout_signature_stress_audit.py` ！ Created
- `runs/block1j40b7c_env3b_heldout_signature_stress_audit.json` ！ Generated
- `protocols/block1j40b7c_env3b_heldout_signature_stress_audit.md` ！ Generated
- `protocols/block1j40b7c_env3b_heldout_signature_stress_audit_table.csv` ！ Generated

## 2. Commands Run

```
& "D:\conda\python.exe" "_block1j40b7c_env3b_heldout_signature_stress_audit.py"
```

## 3. Stress Split Definitions

### Test A: Held-out Signature Split

4 variants, each holding out 1 shared (positive_net) + 1 unique (non_positive_net) signature.

**A1**: Hold out shared sig amb(brownish,heavy_weight)_only (30 obj, positive_net) + unique sig amb(brownish,heavy_weight)+discolored,has_bark_texture (15 obj, non_positive_net)
  - Train: 135 objects, Test: 45 objects (30 pos + 15 nonpos)

**A2**: Hold out shared sig amb(brownish,heavy_weight)+spotted (30 obj, positive_net) + unique sig amb(brownish,heavy_weight)+has_crystal_flecks (15 obj, non_positive_net)
  - Train: 135 objects, Test: 45 objects (30 pos + 15 nonpos)

**A3**: Hold out shared sig amb(greenish,light_weight)+discolored (30 obj, positive_net) + unique sig amb(greenish,light_weight)+has_grip_area (15 obj, non_positive_net)
  - Train: 135 objects, Test: 45 objects (30 pos + 15 nonpos)

**A4**: Hold out shared sig amb(greenish,light_weight)_only (30 obj, positive_net) + unique sig amb(greenish,light_weight)+has_stem_remnant,spotted (15 obj, non_positive_net)
  - Train: 135 objects, Test: 45 objects (30 pos + 15 nonpos)

### Test B: Held-out Cue-Combination Split

2 variants, holding out non-positive uses of specific cues.

**B1**: Hold out apple N (spotted in nonpos context) + ambient_B only (pos). Training has spotted only in positive contexts (wood_log Z, stone_block Z).
  - Train: 135 objects, Test: 45 objects (30 pos + 15 nonpos)

**B2**: Hold out wood_log N (discolored in nonpos context) + ambient_A only (pos). Training has discolored only in positive contexts (apple Z, pickaxe Z).
  - Train: 135 objects, Test: 45 objects (30 pos + 15 nonpos)

### Test C: Held-out Seed Split

Standard LOSO across 5 seeds (already tested in 7b).

## 4. Forbidden-Field Audit

- Forbidden fields in model keys: 0
- Hidden-before-reveal violations: 0

## 5. Results at observe_cost=0.03

### Test A: held-out signatures

**Status: FAIL**

| Metric | Value |
|--------|-------|
| n_test | 180 |
| policy_net | 0.4200 |
| always_observe_net | 0.4200 |
| always_try | 0.2500 |
| oracle_selective | 0.4300 |
| pos_obs_rate | 1.0000 |
| nonpos_obs_rate | 1.0000 |
| harmful_obs_rate | 1.0000 |

| Check | Result |
|-------|--------|
| learned_policy_net > always_observe_net | **FAIL** |
| learned_policy_net > always_try | **PASS** |
| observe_rate between 0 and 1 | **FAIL** |
| pos_obs_rate > nonpos_obs_rate | **FAIL** |

**Per-variant results:**

| Variant | Policy Net | AlwaysObs | OracleSel | ObsRate | PosObs | NonposObs |
|---------|------------|-----------|-----------|---------|--------|-----------|
| A1 | 0.4200 | 0.4200 | 0.4300 | 1.0000 | 1.0000 | 1.0000 |
| A2 | 0.4200 | 0.4200 | 0.4300 | 1.0000 | 1.0000 | 1.0000 |
| A3 | 0.4200 | 0.4200 | 0.4300 | 1.0000 | 1.0000 | 1.0000 |
| A4 | 0.4200 | 0.4200 | 0.4300 | 1.0000 | 1.0000 | 1.0000 |
### Test B: held-out cue combos

**Status: FAIL**

| Metric | Value |
|--------|-------|
| n_test | 90 |
| policy_net | 0.4200 |
| always_observe_net | 0.4200 |
| always_try | 0.2500 |
| oracle_selective | 0.4300 |
| pos_obs_rate | 1.0000 |
| nonpos_obs_rate | 1.0000 |
| harmful_obs_rate | 1.0000 |

| Check | Result |
|-------|--------|
| learned_policy_net > always_observe_net | **FAIL** |
| learned_policy_net > always_try | **PASS** |
| observe_rate between 0 and 1 | **FAIL** |
| pos_obs_rate > nonpos_obs_rate | **FAIL** |

**Per-variant results:**

| Variant | Policy Net | AlwaysObs | OracleSel | ObsRate | PosObs | NonposObs |
|---------|------------|-----------|-----------|---------|--------|-----------|
| B1 | 0.4200 | 0.4200 | 0.4300 | 1.0000 | 1.0000 | 1.0000 |
| B2 | 0.4200 | 0.4200 | 0.4300 | 1.0000 | 1.0000 | 1.0000 |
### Test C: held-out seeds

**Status: PASS**

| Metric | Value |
|--------|-------|
| n_test | 180 |
| policy_net | 0.4300 |
| always_observe_net | 0.4200 |
| always_try | 0.2500 |
| oracle_selective | 0.4300 |
| pos_obs_rate | 1.0000 |
| nonpos_obs_rate | 0.0000 |
| harmful_obs_rate | 0.0000 |

| Check | Result |
|-------|--------|
| learned_policy_net > always_observe_net | **PASS** |
| learned_policy_net > always_try | **PASS** |
| observe_rate between 0 and 1 | **PASS** |
| pos_obs_rate > nonpos_obs_rate | **PASS** |

**Per-variant results:**

| Variant | Policy Net | AlwaysObs | OracleSel | ObsRate | PosObs | NonposObs |
|---------|------------|-----------|-----------|---------|--------|-----------|
| C_seed101 | 0.4300 | 0.4200 | 0.4300 | 0.6667 | 1.0000 | 0.0000 |
| C_seed103 | 0.4300 | 0.4200 | 0.4300 | 0.6667 | 1.0000 | 0.0000 |
| C_seed107 | 0.4300 | 0.4200 | 0.4300 | 0.6667 | 1.0000 | 0.0000 |
| C_seed109 | 0.4300 | 0.4200 | 0.4300 | 0.6667 | 1.0000 | 0.0000 |
| C_seed113 | 0.4300 | 0.4200 | 0.4300 | 0.6667 | 1.0000 | 0.0000 |
## 5. Results at observe_cost=0.05

### Test A: held-out signatures

**Status: FAIL**

| Metric | Value |
|--------|-------|
| n_test | 180 |
| policy_net | 0.4000 |
| always_observe_net | 0.4000 |
| always_try | 0.2500 |
| oracle_selective | 0.4167 |
| pos_obs_rate | 1.0000 |
| nonpos_obs_rate | 1.0000 |
| harmful_obs_rate | 1.0000 |

| Check | Result |
|-------|--------|
| learned_policy_net > always_observe_net | **FAIL** |
| learned_policy_net > always_try | **PASS** |
| observe_rate between 0 and 1 | **FAIL** |
| pos_obs_rate > nonpos_obs_rate | **FAIL** |

**Per-variant results:**

| Variant | Policy Net | AlwaysObs | OracleSel | ObsRate | PosObs | NonposObs |
|---------|------------|-----------|-----------|---------|--------|-----------|
| A1 | 0.4000 | 0.4000 | 0.4167 | 1.0000 | 1.0000 | 1.0000 |
| A2 | 0.4000 | 0.4000 | 0.4167 | 1.0000 | 1.0000 | 1.0000 |
| A3 | 0.4000 | 0.4000 | 0.4167 | 1.0000 | 1.0000 | 1.0000 |
| A4 | 0.4000 | 0.4000 | 0.4167 | 1.0000 | 1.0000 | 1.0000 |
### Test B: held-out cue combos

**Status: FAIL**

| Metric | Value |
|--------|-------|
| n_test | 90 |
| policy_net | 0.4000 |
| always_observe_net | 0.4000 |
| always_try | 0.2500 |
| oracle_selective | 0.4167 |
| pos_obs_rate | 1.0000 |
| nonpos_obs_rate | 1.0000 |
| harmful_obs_rate | 1.0000 |

| Check | Result |
|-------|--------|
| learned_policy_net > always_observe_net | **FAIL** |
| learned_policy_net > always_try | **PASS** |
| observe_rate between 0 and 1 | **FAIL** |
| pos_obs_rate > nonpos_obs_rate | **FAIL** |

**Per-variant results:**

| Variant | Policy Net | AlwaysObs | OracleSel | ObsRate | PosObs | NonposObs |
|---------|------------|-----------|-----------|---------|--------|-----------|
| B1 | 0.4000 | 0.4000 | 0.4167 | 1.0000 | 1.0000 | 1.0000 |
| B2 | 0.4000 | 0.4000 | 0.4167 | 1.0000 | 1.0000 | 1.0000 |
### Test C: held-out seeds

**Status: PASS**

| Metric | Value |
|--------|-------|
| n_test | 180 |
| policy_net | 0.4167 |
| always_observe_net | 0.4000 |
| always_try | 0.2500 |
| oracle_selective | 0.4167 |
| pos_obs_rate | 1.0000 |
| nonpos_obs_rate | 0.0000 |
| harmful_obs_rate | 0.0000 |

| Check | Result |
|-------|--------|
| learned_policy_net > always_observe_net | **PASS** |
| learned_policy_net > always_try | **PASS** |
| observe_rate between 0 and 1 | **PASS** |
| pos_obs_rate > nonpos_obs_rate | **PASS** |

**Per-variant results:**

| Variant | Policy Net | AlwaysObs | OracleSel | ObsRate | PosObs | NonposObs |
|---------|------------|-----------|-----------|---------|--------|-----------|
| C_seed101 | 0.4167 | 0.4000 | 0.4167 | 0.6667 | 1.0000 | 0.0000 |
| C_seed103 | 0.4167 | 0.4000 | 0.4167 | 0.6667 | 1.0000 | 0.0000 |
| C_seed107 | 0.4167 | 0.4000 | 0.4167 | 0.6667 | 1.0000 | 0.0000 |
| C_seed109 | 0.4167 | 0.4000 | 0.4167 | 0.6667 | 1.0000 | 0.0000 |
| C_seed113 | 0.4167 | 0.4000 | 0.4167 | 0.6667 | 1.0000 | 0.0000 |

## 6. Overall

**FAIL**

| Test | Primary (0.03) | Robustness (0.05) |
|------|---------------|-------------------|
| Test A (held-out signatures) | FAIL | FAIL |
| Test B (held-out cue combos) | FAIL | FAIL |
| Test C (held-out seeds) | PASS | PASS |

## 7. Key Findings

- Test A (held-out signatures): FAIL
- Test B (held-out cue combos): FAIL
- Test C (held-out seeds): PASS

The policy fails to generalize to held-out signatures despite passing in-distribution (Test C).
This suggests signature memorization rather than VOI reasoning.

## 8. Output Files

- `runs/block1j40b7c_env3b_heldout_signature_stress_audit.json`
- `protocols/block1j40b7c_env3b_heldout_signature_stress_audit.md`
- `protocols/block1j40b7c_env3b_heldout_signature_stress_audit_table.csv`

## 9. Next Step

Stress test fails. The policy does not generalize to held-out signatures.
The 7b result may be driven by signature memorization rather than VOI reasoning.
Consider: richer training data, more diverse signatures, or different model architecture.

```
[block_done]
block_id=1J40b-7c
implementation_status=FAIL
```
