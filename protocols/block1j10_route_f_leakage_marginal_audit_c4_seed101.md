# Block 1J10 — Route F Leakage + Marginal-Value Audit

**Date:** 2026-05-11
**Block:** 1J10 — Route F peripheral observation interface audit
**Condition:** C4_instance_subtype_cued_v1
**Seed:** 101
**Budget:** 1.5

---

## 1. Executive Summary

1J10 implements Route F peripheral observation interface variants (F_low, F_mid, F_high), runs the leakage audit (A1, A2, B, C, D) and marginal-value audit with clean deployable/diagnostic/oracle separation.

**All three variants pass both leakage and marginal-value audits.** No hidden information leakage detected through the peripheral signal. C0b_RouteF_deployable predictions are identical to original C0b (0.5871) — as expected for aggregate-only peripheral signals. The peripheral information upper bound (diagnostic, non-deployable) shows no classifier-extractable hidden info.

**F_mid recommended** for 1J11 C15F policy evaluation.

---

## 2. Design Validation

Before running, the script was validated against the preregistered design constraints:

| Check | Status |
|-------|--------|
| `_peripheral_prior_for_oid` removed | PASS |
| `local_feature_prevalence` blending removed | PASS |
| `C0b_RouteFPolicy` class removed | PASS |
| `RandomProbe_RouteFPolicy` class removed | PASS |
| `C0b_RouteF_deployable` uses only IOM + global prior | PASS |
| `peripheral_information_upper_bound` labeled diagnostic_non_deployable | PASS |
| No ad-hoc pseudo-observation conversion | PASS |
| Deployable vs diagnostic separation clean | PASS |

---

## 3. Blur Variants

| Variant | d1 | d2 | Description |
|---------|----|----|-------------|
| F_low | 1 | 2 | Minimal peripheral — only immediate neighbors |
| F_mid | 2 | 4 | Moderate coarse local context — small neighborhood |
| F_high | 3 | 6 | Stronger local context — larger perceptual field |

---

## 4. Leakage Audit

### 4.1 Test Design

Five leakage probe types applied to each variant:

| Test | What it probes | Method |
|------|---------------|--------|
| A1 | Per-category hidden subtype | 1-NN LOO on peripheral features → subtype within each category |
| A2 | Global majority vs minority | 1-NN LOO on peripheral features → majority/minority label |
| B | True affordance values | 1-NN LOO on peripheral features → per-affordance binary |
| C | Hidden category | 1-NN LOO on peripheral features → category |
| D | Public category-like recheck | k-NN peripheral features → predicted category → majority subtype profile |

All tests use base_rate, CV, shuffle control, and real-minus-shuffle gap. Fail condition: `CV > base_rate + 0.10 AND gap >= 0.05`.

### 4.2 Results

| Variant | A1 Fails | A2 Verdict | B Fails | C Verdict | D macro_bal | Overall |
|---------|----------|-----------|---------|-----------|------------|---------|
| F_low | none | PASS | none | PASS | 0.5000 | **PASS** |
| F_mid | none | PASS | none | PASS | 0.5000 | **PASS** |
| F_high | none | PASS | none | PASS | 0.5000 | **PASS** |

All variants: 0 hard fails. Public category-like macro_bal = 0.5000 for all (identical to 1J8 baseline).

### 4.3 Interpretation

The peripheral signal — even at F_high (d1=3, d2=6) — does not leak hidden subtype, affordance, or category information beyond what is detectable by base-rate classification. The C4_instance_subtype_cued_v1 condition's visible features don't encode between-category discriminability, and the aggregate-only peripheral histograms don't recover it.

---

## 5. Marginal-Value Audit

### 5.1 Baseline Architecture

Three cleanly separated result types:

| Result | Label | What it measures |
|--------|-------|-----------------|
| `C0b_RouteF_deployable` | **deployable** | C0b policy with Route F peripheral interface. Predictions identical to original C0b (IOM for visited, global prior for unvisited). |
| `peripheral_information_upper_bound` | **diagnostic_non_deployable** | Best-signal selection + 1-NN LOO classifier. Estimates maximum extractable information. NOT a deployable policy. |
| `random_probe_RouteF_deployable` | **deployable** | Random-probe policy. Standard predictions (IOM for visited, global prior for unvisited). |

### 5.2 Deployable Results

| Variant | C0b_RouteF_deployable | vs Original C0b | Identical? | Random Probe | vs C0b_RouteF |
|---------|----------------------|-----------------|------------|-------------|---------------|
| F_low | 0.5871 | -0.0000 | yes | 0.5750 | -0.0121 |
| F_mid | 0.5871 | -0.0000 | yes | 0.5750 | -0.0121 |
| F_high | 0.5871 | -0.0000 | yes | 0.5750 | -0.0121 |

C0b_RouteF_deployable is identical to original C0b for all variants. This is the expected result: the peripheral signal is aggregate-only (histograms, counts, densities) and cannot produce per-object predictions for unvisited objects. **Route F's value is in policy targeting / object selection, not direct observe-only classification.**

### 5.3 Diagnostic Upper Bound

| Variant | A2 CV | Near Oracle? | Hard Fails | Contains Info? |
|---------|-------|-------------|------------|---------------|
| F_low | 0.6333 | no | 0 | no |
| F_mid | 0.7333 | no | 0 | no |
| F_high | 0.7667 | no | 0 | no |

The peripheral information upper bound uses best-signal selection + 1-NN LOO. Even with optimal access, no variant extracts hidden information beyond base rates (all hard_fail_count=0). A2 CV increases with blur radius (0.63 → 0.77) but stays below the 0.85 "near oracle" threshold.

### 5.4 Oracle Gap

| Variant | Oracle Gap | Saturation? |
|---------|-----------|-------------|
| F_low | 0.4129 | no |
| F_mid | 0.4129 | no |
| F_high | 0.4129 | no |

Oracle gap is 0.4129 (identical across variants — C0b predictions unchanged). Well above the 0.10 minimum. Task remains challenging.

---

## 6. Variant Assessment

| Variant | Leakage | Marginal Value | C0b mb | Oracle Gap | Pub Cat | UB Near Oracle |
|---------|---------|---------------|--------|------------|---------|---------------|
| F_low | PASS | **PASS** | 0.5871 | 0.4129 | 0.5000 | no |
| F_mid | PASS | **PASS** | 0.5871 | 0.4129 | 0.5000 | no |
| F_high | PASS | **PASS** | 0.5871 | 0.4129 | 0.5000 | no |

Marginal value criteria:
1. Leakage must pass ✓ (all)
2. C0b must not saturate (≤ 0.70) ✓ (0.5871)
3. Oracle gap ≥ 0.10 ✓ (0.4129)
4. Public category shortcut ≤ 0.60 ✓ (0.5000)
5. Peripheral UB not near oracle ✓ (all)

C0b_RouteF_deployable is NOT required to improve over original C0b — for aggregate-only signals, identity is expected and acceptable.

---

## 7. Recommendation

**Recommended variant: F_mid**

All three variants pass both audits. F_mid is preferred:
- Larger perceptual field than F_low (d1=2, d2=4 vs d1=1, d2=2) — more pre-probe context
- Lower leakage risk ceiling than F_high — F_high A2 CV (0.7667) is closer to the near-oracle threshold
- F_mid provides the best balance of information richness vs leakage safety

**Route F audit: PASSED**
**Proceed to 1J11 C15F policy evaluation: YES**

### Implementation Notes for 1J11

1. C15F policy should use peripheral signals for **object selection and probe targeting only** — not for direct prediction
2. C0b_RouteF observation is identical to C0b observation; the value is in giving the policy richer pre-visit information
3. 1J11 design should preregister: how C15F uses peripheral signals in select_next_object, how it uses them in decide_probe, and all leakage constraints
4. Multi-seed deferred until single-seed improvement confirmed on 1J11

---

## 8. Reference Baselines

| Baseline | macro_bal | Label |
|----------|-----------|-------|
| all_false | 0.5000 | floor |
| C_minus_prior_only | 0.5000 | global base rate |
| C0b_original | 0.5871 | no-probe baseline |
| C15b_original | 0.6142 | best C15 variant |
| oracle | 1.0000 | ceiling |
| hidden_category_majority | 0.9750 | diagnostic (from 1J2/1J3) |

---

*Generated by Block 1J10 — Route F leakage and marginal-value audit.*
