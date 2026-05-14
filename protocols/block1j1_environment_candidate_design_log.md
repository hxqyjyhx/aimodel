# Block 1J1: Environment Candidate Design Log — C4_instance_subtype_cued_v1

**Date:** 2026-05-10
**Block:** 1J1 — Environment candidate implementation (no policy runs)
**Condition:** C4_instance_subtype_cued_v1

---

## 1. Condition Name

`C4_instance_subtype_cued_v1`

## 2. Files Changed / Created

| File | Action | Purpose |
|------|--------|---------|
| `config.py` | Modified (line 33-38) | Added C4 condition to CUE_CONDITIONS with `subtype: True` flag |
| `subtype_objects.py` | Created | Subtype definitions, visible cue probabilities, deterministic + random generators, query ground truth computer |
| `_block1j1_static_check.py` | Created | Static generation sanity check script |
| `protocols/block1j1_environment_candidate_design_log.md` | Created | This design log |
| `runs/calibration_block1j1_static_environment_candidate_summary.json` | Created | Static label distribution summary |

## 3. Subtype Definitions

Each of the 4 categories has 2 hidden subtypes. One subtype is the "majority" (80% of objects), the other is the "minority" (20%).

### wood_log

| Property | craftable (80%) | rotten_or_brittle (20%) |
|----------|-----------------|-------------------------|
| mine_by_hand | success | success |
| mine_with_pickaxe | success | success |
| craft_plank | **success** | **fail** |
| eat | fail | fail |
| use_as_tool | fail | fail |
| burn_as_fuel | success | success |

Category-majority guess (craft_plank=True): 80% correct for wood_log objects.

### stone_block

| Property | brittle (20%) | hard (80%) |
|----------|---------------|------------|
| mine_by_hand | **success** | **fail** |
| mine_with_pickaxe | success | success |
| craft_plank | fail | fail |
| eat | fail | fail |
| use_as_tool | fail | fail |
| burn_as_fuel | fail | fail |

Category-majority guess (mine_by_hand=False): 80% correct for stone_block objects.

### apple

| Property | ripe (80%) | unripe_or_spoiled (20%) |
|----------|------------|-------------------------|
| mine_by_hand | success | success |
| mine_with_pickaxe | success | success |
| craft_plank | fail | fail |
| eat | **success** | **fail** |
| use_as_tool | fail | fail |
| burn_as_fuel | fail | fail |

Category-majority guess (eat=True): 80% correct for apple objects.

### wooden_pickaxe

| Property | intact (80%) | damaged (20%) |
|----------|-------------|---------------|
| mine_by_hand | success | success |
| mine_with_pickaxe | success | success |
| craft_plank | fail | fail |
| eat | fail | fail |
| use_as_tool | **success** | **fail** |
| burn_as_fuel | success | success |

Category-majority guess (use_as_tool=True): 80% correct for wooden_pickaxe objects.

## 4. Visible Cue Definitions

Visible features are sampled with subtype-dependent probabilities. The diagnostic visible features per category:

### wood_log cues

| Feature | craftable | rotten_or_brittle | abs diff |
|---------|-----------|-------------------|----------|
| has_wood_grain | 0.75 | 0.38 | 0.37 |
| has_bark_texture | 0.72 | 0.42 | 0.30 |
| rough_texture | 0.62 | 0.48 | 0.14 |
| solid | 0.85 | 0.70 | 0.15 |

### stone_block cues

| Feature | brittle | hard | abs diff |
|---------|---------|------|----------|
| has_crystal_flecks | 0.70 | 0.42 | 0.28 |
| has_granular_surface | 0.65 | 0.42 | 0.23 |
| solid | 0.78 | 0.90 | 0.12 |

### apple cues

| Feature | ripe | unripe_or_spoiled | abs diff |
|---------|------|-------------------|----------|
| has_stem_remnant | 0.72 | 0.40 | 0.32 |
| has_peel_texture | 0.70 | 0.42 | 0.28 |
| greenish | 0.55 | 0.40 | 0.15 |
| smooth_texture | 0.58 | 0.42 | 0.16 |

### wooden_pickaxe cues

| Feature | intact | damaged | abs diff |
|---------|--------|---------|----------|
| has_grip_area | 0.75 | 0.38 | 0.37 |
| has_shaft_shape | 0.72 | 0.42 | 0.30 |
| elongated_with_handle | 0.65 | 0.48 | 0.17 |
| solid | 0.80 | 0.65 | 0.15 |

**Key property:** No single visible feature perfectly determines subtype. The maximum differential is 0.37 (has_wood_grain, has_grip_area). Multiple features combined provide partial subtype information. Probing confirms the affordance ground truth.

## 5. Query Design

The existing 5-query suite is preserved:

| Query | Condition | Positive subtypes | Expected pos_rate |
|-------|-----------|-------------------|-------------------|
| need_planks | craft_plank_success=1.0 | wood_log::craftable only | 0.20 |
| need_stone | mine_by_hand_success=0.0, mine_with_pickaxe_success=1.0 | stone_block::hard only | 0.20 |
| need_food | eat_success=1.0 | apple::ripe only | 0.20 |
| need_tool | use_as_tool_success=1.0 | wooden_pickaxe::intact only | 0.20 |
| need_fuel | burn_as_fuel_success=1.0 | wood_log (both) + wooden_pickaxe (both) | 0.50 |

All 5 queries have both positive and negative examples. All positive rates are within [0.20, 0.80]. need_fuel is at 0.50 (ideal range). The other 4 queries are at 0.20 (boundary, but pass the acceptance criterion of >= 0.20).

## 6. Rationale for Each Environment Choice

### Why 80/20 subtype ratios?

With 4 categories and equal counts (15 each = 60 total), a query satisfied by a single subtype in a single category can achieve at most 25% positive rate. The 80/20 split gives 20% positive rate for single-subtype queries, which is at the acceptance boundary of 0.20. A 50/50 split would give only 12.5% — below the 0.20 floor.

The 80/20 split also means category-majority guessing achieves 80% per-category accuracy, creating a realistic "category is useful but not sufficient" regime. The 20% minority cases are where probing is necessary.

### Why one differentiating affordance per category?

Each category differs from its subtype on exactly one core action. This is the minimal change from the original fixed-profile design. It keeps the environment simple while creating the key property: category alone is insufficient, probing confirms subtype.

### Why partial visible cues (max diff ~0.35)?

Visible feature probabilities differ between subtypes by 0.12-0.37. This means:
- Observation provides real but imperfect information about subtype
- Neither observation alone nor probing alone suffices
- The best strategy combines observation (partial cues) with probing (confirmation)

If cues were too strong (diff > 0.5), observation alone might solve the task. If cues were too weak (diff < 0.1), observation would add no value above prior.

### Why deterministic subtype assignment for static check?

With n=15 per category, random sampling produces high variance in subtype counts (e.g., seed=101 gave 7 hard stone_blocks instead of the expected 12). Deterministic assignment ensures exact target ratios, removing seed-dependent noise from the label distribution. The shuffle prevents object ID from revealing subtype.

## 7. Why This Is Not Policy-Favoring C13 Tuning

1. **The design is driven by task semantics, not C13 performance.** Subtypes model real-world variation (ripe vs unripe apples, intact vs damaged tools). The visible cues (grain, texture, color) are natural partial indicators.

2. **The acceptance criteria are based on baselines, not C13.** The 1J2 audit evaluates all_false, C_minus, C0b, oracle, and random_probe. C13 is not run.

3. **The design does not target C13's weaknesses.** C13's failure mode in Block 1I0 was posterior non-separation (weak AUC) and aggregation bias favoring negative recall. This design does not specifically fix those issues — it creates a different information structure where probing has clearer marginal value.

4. **The anti-tuning safeguards from Block 1J0 are followed.** The design was pre-registered before implementation. The environment is evaluated by marginal-value baselines.

## 8. Expected Effect on Baselines

These are qualitative predictions, not results. Actual values will be measured in Block 1J2.

| Baseline | Expected behavior | Rationale |
|----------|-------------------|-----------|
| all_false | macro_bal ≈ 0.5000 | Always predicts False; 5 queries all have both classes, so pos_recall=0, neg_recall=1 for all → macro_bal=0.5 |
| C_minus_prior_only | macro_bal moderately above 0.50 | IOM base rates reflect the 80/20 split; predicts majority outcome per category, achieves ~80% per-category accuracy but no subtype discrimination |
| C0b_observe_only | macro_bal between C_minus and oracle | Visible features provide partial subtype signal; observation should improve over prior but not reach oracle. Expected macro_bal in 0.55-0.70 range. |
| C_oracle | macro_bal >= 0.70 (target 1.0 ideally) | Oracle knows hidden subtype → perfect affordance knowledge. Should achieve near-perfect accuracy. |
| random_probe_budgeted | macro_bal between C_minus and C0b | Random probes occasionally hit the differentiating action, revealing subtype for probed objects. |
| category_majority | macro_bal ≈ 0.975 | Predicts majority affordance per category; 80% correct per category, 100% correct for need_fuel. |

### Key expected gaps:

- **C_oracle - C0b >= +0.1000:** Depends on visible cue strength. With max feature differentials of 0.35, C0b should have meaningful error, leaving room for oracle improvement.
- **C0b - C_minus >= +0.0500:** Visible features carry partial subtype signal, so observation should improve over prior. The 0.35 differentials should support this.
- **C_oracle positive AND negative recall improvement:** Oracle knows exact subtype for every object, enabling both positive and negative recall improvements.

## 9. Static Sanity Results (seed=101, deterministic assignment)

| Metric | Value | Pass? |
|--------|-------|-------|
| Total objects | 60 (15 per category) | — |
| Subtype distribution | 12/3 or 3/12 per category (80/20) | — |
| need_planks pos_rate | 0.2000 | Pass (>= 0.20) |
| need_stone pos_rate | 0.2000 | Pass (>= 0.20) |
| need_food pos_rate | 0.2000 | Pass (>= 0.20) |
| need_tool pos_rate | 0.2000 | Pass (>= 0.20) |
| need_fuel pos_rate | 0.5000 | Pass (ideal 0.35-0.65) |
| All queries have both classes | True | Pass |
| all_false macro_bal | 0.5000 | Pass |
| category_majority macro_bal | 0.9750 | — |
| static_sanity_passed | True | — |
| ready_for_1J2 | True | — |

## 10. Known Limitations / Risks

1. **Single-category queries at 0.20 boundary.** need_planks, need_stone, need_food, need_tool are all at exactly 0.20 positive rate. This is the minimum acceptable. If the 1J2 audit reveals issues, adjusting subtype ratios or adding cross-category query satisfaction may be needed.

2. **category_majority baseline is very high (0.975).** This means category knowledge alone nearly solves the task for 80% of objects. The remaining 20% (minority subtype objects) are where probing must add value. This is structurally fine — it means the marginal value of probing is concentrated in a minority of cases — but it may make the oracle - C0b gap harder to demonstrate across all queries.

3. **Visible cue differentials may be too weak or too strong.** The 0.30-0.37 max differentials were chosen to balance observability vs probe value. The 1J2 audit will reveal whether they are calibrated correctly.

4. **Only one differentiating affordance per category.** This is minimal but may not create enough probe value diversity. Future iterations could add more differentiating actions.

---

*Generated by Block 1J1 — environment candidate implemented, no policy runs.*
