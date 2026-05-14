# Block 1J30 — Agent Decision Pipeline and Raw-Penalty Failure Flowcharts

Based on seed 109 decision-path audit and 1J31 raw-penalty-zero source audit.

---

## 1. Agent Decision Pipeline (Per-Episode Loop)

```mermaid
flowchart TD
    START(["Episode Start"]) --> RESET["policy.reset(view)"]
    RESET --> OBSERVE["PHASE_OBSERVE"]

    OBSERVE --> NEXT["select_next_object(view)"]
    NEXT --> |unvisited exists| VISIT["Visit cheapest unvisited object"]
    VISIT --> OBSERVE_FEATURES["view.get_observed_features(oid)"]
    OBSERVE_FEATURES --> CHECK_TRANSITION{"should_transition<br/>to probe phase?"}
    CHECK_TRANSITION --> |no| NEXT
    CHECK_TRANSITION --> |yes| PROBE["PHASE_PROBE"]

    PROBE --> SELECT_TARGET["_select_probe_target(view)"]
    SELECT_TARGET --> POOL_BUILD["Build candidate pool:<br/>visited + not probed +<br/>features available +<br/>can afford cost"]
    POOL_BUILD --> SCORE_CANDIDATES["_score_candidate(features, norm_cost)<br/>for each candidate"]
    SCORE_CANDIDATES --> RANK["Sort by score, assign ranks"]
    RANK --> EXPLORE_CHECK{"explore_fraction<br/>roll?"}
    EXPLORE_CHECK --> |yes| PICK_RANDOM["Pick from [30%, 80%] rank tier"]
    EXPLORE_CHECK --> |no| PICK_TOP["Pick top-ranked candidate"]

    PICK_RANDOM --> DECIDE_PROBE
    PICK_TOP --> DECIDE_PROBE["decide_probe(view, object_id)"]
    DECIDE_PROBE --> SCORE_ACTIONS["For each action:<br/>_compute_adjusted_score(features, action, norm_cost)"]
    SCORE_ACTIONS --> PICK_BEST_ACTION["Select action with max final_score"]
    PICK_BEST_ACTION --> EXECUTE["Execute probe, observe outcome"]
    EXECUTE --> UPDATE_IM["Update InstanceOutcomeMemory<br/>with probe result"]
    UPDATE_IM --> UPDATE_RISK["on_probe_result: check_prior_violation<br/>→ FamilyConditionedRiskMemory.update()"]
    UPDATE_RISK --> CHECK_DONE{"probed_oids >=<br/>target_probe_count?"}
    CHECK_DONE --> |no| SELECT_TARGET
    CHECK_DONE --> |yes| GET_ANSWER["get_answer(view):<br/>predict_all_affordances per object"]
    GET_ANSWER --> END(["Episode End"])
```

---

## 2. Raw Penalty Formation (Signed Sum)

Shows how `get_total_risk_penalty(features, action)` computes the final signed penalty used in score adjustment.

```mermaid
flowchart TD
    START(["get_total_risk_penalty(features, action)"]) --> MODE{"aggregation_mode?"}
    MODE --> |signed_sum| DETECT_FAMILY["family = detect_type_family(features)"]
    DETECT_FAMILY --> GOAL["goal = ACTION_TO_QUERY[action]"]

    GOAL --> LOOP_START["For each dev_feat in<br/>INJECTED_DEVIATION_FEATURES"]

    LOOP_START --> FEAT_PRESENT{"features.get(dev_feat, False)?"}
    FEAT_PRESENT --> |no| SKIP_FEAT["Skip (not present on object)"]
    SKIP_FEAT --> LOOP_NEXT

    FEAT_PRESENT --> |yes| KEY_BUILD["key = (goal, action, family, dev_feat)"]
    KEY_BUILD --> GET_SIGNED["fcrm.get_raw_risk_weight_signed(key)"]
    GET_SIGNED --> CACHE_CHECK{"key in _signed_cache?"}
    CACHE_CHECK --> |yes| RETURN_CACHED["Return cached value"]
    CACHE_CHECK --> |no| GET_COUNTS["c = counts[key]:<br/>violation_with, nonviolation_with,<br/>violation_without, nonviolation_without"]

    GET_COUNTS --> CHECK_SUPPORT{"actual_violations<br/>= (v_with+v_without)-2.0<br/>>= min_violation_support?"}
    CHECK_SUPPORT --> |no| CACHE_ZERO["_signed_cache[key] = 0.0<br/>→ return 0.0"]
    CACHE_ZERO --> ADD_ZERO["Add 0.0 to sum"]
    ADD_ZERO --> LOOP_NEXT

    CHECK_SUPPORT --> |yes| COMPUTE["p_feat_given_v = v_with / total_v<br/>p_feat_given_nv = nv_with / total_nv<br/>ratio = p_feat_given_v / p_feat_given_nv<br/>raw_lr = log(ratio)"]

    COMPUTE --> SHRINK["support = total_v + total_nv - 4<br/>reliability = support / (support + alpha)<br/>usable = reliability * raw_lr"]
    SHRINK --> CACHE_RESULT["_signed_cache[key] = usable"]
    CACHE_RESULT --> ADD_WEIGHT["Add usable to sum"]
    ADD_WEIGHT --> LOOP_NEXT

    LOOP_NEXT --> LOOP_END{"More dev_feats?"}
    LOOP_END --> |yes| LOOP_START
    LOOP_END --> |no| CLAMP["total = sum(weights)<br/>clamp to [-max_risk, max_risk]"]

    CLAMP --> SCALE["scaled_penalty = total * risk_penalty_scale"]
    SCALE --> SCORE["raw_score = base_prior * exp(-scaled_penalty) - COST_WEIGHT * norm_cost"]
    SCORE --> FLOOR["final_score = max(EXPLORATION_FLOOR, raw_score)"]
    FLOOR --> END(["Return final_score"])

    MODE --> |sum_positive| POSITIVE["Same loop but uses<br/>get_risk_weight() (clipped to [0,max_risk])<br/>sum only positive weights"]

    MODE --> |signed_max_abs| MAX_ABS["Same loop, signed weights<br/>select max by abs(value)<br/>clamp to [-max_risk, max_risk]"]
```

---

## 3. Seed109 Failure Path — Why raw_signed_penalty = 0 for 12/16 Missed PVs

Root cause from 1J31: `feature_not_in_memory_universe` — the 12 objects lack ALL injected deviation features.

```mermaid
flowchart TD
    START(["Missed offline-avoidable PV<br/>(A has it, Offline doesn't, Online has it)"]) --> QUESTION{"Does the object have<br/>ANY injected deviation feature<br/>(damp_texture, brittle_surface,<br/>treated_surface, hollow_sound)?"}

    QUESTION --> |"YES (3 cases:<br/>apple_006, apple_007, apple_011<br/>have damp_texture)"| HAS_FEATS["raw_signed_penalty > 0<br/>(0.634838 for damp_texture)"]
    HAS_FEATS --> RANK_UNCHANGED["rank_unchanged (4 cases):<br/>penalty is nonzero but not<br/>large enough to change ranking<br/>at scale 1.0 or 3.0"]

    QUESTION --> |"NO (12 cases:<br/>apple_001, apple_004, apple_005,<br/>stone_block_002/003/004/006/008/012,<br/>wood_log_001,<br/>wooden_pickaxe_002/005)"| NO_FEATS["Zero injected deviation features<br/>attached to object"]

    NO_FEATS --> LOOP["For each dev_feat in<br/>INJECTED_DEVIATION_FEATURES:"]
    LOOP --> CHECK{"features.get(dev_feat, False)?"}
    CHECK --> |all False| ZERO_TOTAL["total = 0.0<br/>(nothing to sum)"]
    ZERO_TOTAL --> ZERO_PENALTY["raw_signed_penalty = 0.0"]
    ZERO_PENALTY --> ZERO_SCALED["scaled_penalty = 0.0 * scale = 0.0"]
    ZERO_SCALED --> NO_EFFECT["raw_score = base_prior * exp(0) - cost<br/>= base_prior - cost<br/>(same as A_no_memory!)"]
    NO_EFFECT --> PV_NOT_AVOIDED["Prior violation NOT avoided<br/>→ Online still has this PV"]

    RANK_UNCHANGED --> PV_NOT_AVOIDED

    NO_EFFECT --> DIAGNOSIS["Diagnosis: feature_universe_issue_detected=True<br/>These objects are cross-family deceptive —<br/>they look like family A but ARE family B.<br/>Their injected deviation features are<br/>structural features of the wrong family,<br/>not deviation markers."]
    DIAGNOSIS --> ROOT["ROOT CAUSE: Memory keys are<br/>(goal, action, family, dev_feat) but<br/>dev_feat only fires if present on object.<br/>Cross-family deceptive objects have<br/>ZERO deviation features.<br/>→ Memory has nothing to penalize."]
```

---

## 4. Observation → Try-Action → Memory Update Loop

Shows how a single probe decision flows through observation, scoring, execution, and memory update.

```mermaid
flowchart TD
    START(["Agent at object oid"]) --> OBSERVE["features = view.get_observed_features(oid)"]
    OBSERVE --> OBSERVED_FEATS["Observed features include:<br/>- Type-family features (brownish, wood_grain...)<br/>- Deviation features IF present<br/>(damp_texture, brittle_surface, etc.)"]

    OBSERVED_FEATS --> SCORE_LOOP["For each action in MAIN_CANDIDATE_ACTIONS"]

    SCORE_LOOP --> COMPUTE_PRIOR["base_prior = get_goal_soft_prior(features, action)<br/>Uses detect_type_family → DEFAULT_GOAL_PRIOR[family, goal, action]"]
    COMPUTE_PRIOR --> COMPUTE_PENALTY["risk_penalty = get_total_risk_penalty(features, action)<br/>Only sums weights for deviation features<br/>PRESENT on the object"]

    COMPUTE_PENALTY --> CHECK_PENALTY{"risk_penalty == 0?"}
    CHECK_PENALTY --> |"yes (12/16 cases)"| NO_ADJUST["score = base_prior - COST_WEIGHT * norm_cost<br/>No risk adjustment at all"]
    CHECK_PENALTY --> |"no (3/38 candidates)"| ADJUST["scaled = risk_penalty * risk_penalty_scale<br/>raw_score = base_prior * exp(-scaled) - cost<br/>final_score = max(floor, raw_score)"]

    NO_ADJUST --> SELECT_ACTION
    ADJUST --> SELECT_ACTION["Select action with max final_score"]
    SELECT_ACTION --> EXECUTE["Execute probe: try the action"]

    EXECUTE --> OUTCOME["Observe real outcome (success/fail)<br/>from simulator ground truth"]

    OUTCOME --> CHECK_PV["check_prior_violation(features, action, outcome):<br/>is_eligible = soft_prior >= 0.60 AND<br/>is_violation = eligible AND outcome == fail"]

    CHECK_PV --> MEM_UPDATE["FamilyConditionedRiskMemory.update()"]
    MEM_UPDATE --> WHICH_MODE{"key_mode?"}
    WHICH_MODE --> |family_dev| UPDATE_DEV["For each dev_feat in INJECTED_DEVIATION_FEATURES:<br/>key = (goal, action, family, dev_feat)<br/>If dev_feat present on object: increment _with<br/>If dev_feat absent: increment _without"]
    WHICH_MODE --> |family_only| UPDATE_FAM["key_detected = (goal, action, family)<br/>Increment _with for detected family<br/>Increment _without for all other families"]

    UPDATE_DEV --> CLEAR_CACHE["Clear _risk_cache and _signed_cache<br/>(forces recomputation on next query)"]
    UPDATE_FAM --> CLEAR_CACHE

    CLEAR_CACHE --> LOG["Append to update_log:<br/>{episode_id, step_id, object_id,<br/>family, dev_feats_present,<br/>outcome, prior_violation, eligible}"]

    LOG --> NEXT_ACTION{"More probes<br/>in budget?"}
    NEXT_ACTION --> |yes| NEXT_OBJ["Move to next selected object"]
    NEXT_ACTION --> |no| END(["Probe phase complete"])
```

---

## 5. Leakage Boundary — What Separates Online from Offline

The audit confirms no hidden feature leakage and no oracle leakage. This diagram shows the boundaries.

```mermaid
flowchart TD
    subgraph OFFLINE["OFFLINE (Oracle) — NOT allowed online"]
        direction LR
        GT["Ground truth outcomes<br/>for ALL (object, action) pairs"]
        FULL_OBS["All features observed<br/>(including hidden)"]
        FULL_TRAJ["Full episode trajectory<br/>known in advance"]
        ORACLE_MEM["Memory trained on<br/>all decisions at once"]
    end

    subgraph BOUNDARY["LEAKAGE BOUNDARY — Causally Valid Wall"]
        direction LR
        NO_ORACLE["no_oracle_leakage_confirmed=True<br/>Online variants never access<br/>offline/oracle evidence"]
        NO_HIDDEN["hidden_feature_leakage_detected=False<br/>Features used by policy are subset<br/>of observed features"]
        CAUSAL["Causally valid: memory at step N<br/>only reflects outcomes from steps 0..N-1"]
    end

    subgraph ONLINE["ONLINE — What agent actually has"]
        direction LR
        OBS_FEATS["Observed features only<br/>via get_observed_features(oid)"]
        PRIOR_OUTCOMES["Only outcomes of<br/>already-executed probes"]
        STEP_MEMORY["Memory built incrementally<br/>from observed outcomes"]
        IOM["InstanceOutcomeMemory<br/>updated per probe result"]
    end

    OFFLINE --> |"Audit checks for leaks"| BOUNDARY
    BOUNDARY --> |"Confirmed clean"| ONLINE

    LEAK_CHECK1["Leakage Check 1:<br/>Are online variants using any<br/>offline precomputed scores?"]
    LEAK_CHECK2["Leakage Check 2:<br/>Do online features include<br/>any ground-truth-only features?"]
    LEAK_CHECK3["Leakage Check 3:<br/>Is memory updated with<br/>future-episode outcomes?"]

    LEAK_CHECK1 --> |"No — confirmed"| BOUNDARY
    LEAK_CHECK2 --> |"No — confirmed"| BOUNDARY
    LEAK_CHECK3 --> |"No — stepwise only"| BOUNDARY
```

---

## 6. Next-Test Decision Tree (Based on 1J30 + 1J31 Findings)

1J30 found `raw_penalty_zero` as dominant failure. 1J31 found `feature_not_in_memory_universe` as root cause: cross-family deceptive objects have NO injected deviation features.

```mermaid
flowchart TD
    FINDINGS["1J30/1J31 Findings:<br/>12/16 missed PVs: raw_penalty=0<br/>Root cause: feature_not_in_memory_universe<br/>Cross-family deceptive objects have<br/>ZERO deviation features attached"]

    FINDINGS --> QUESTION{"Why do deceptive objects<br/>lack deviation features?"}

    QUESTION --> BRANCH_A["Hypothesis A:<br/>Feature coverage gap<br/>Cross-family deceptives are constructed<br/>with structural features of the WRONG family<br/>(e.g., an apple that looks like wood)<br/>but NO deviation marker features"]
    QUESTION --> BRANCH_B["Hypothesis B:<br/>Key design gap<br/>Memory keys = (goal, action, family, dev_feat)<br/>requires BOTH family context AND<br/>present deviation feature.<br/>If dev_feat absent → key never fires"]

    BRANCH_A --> TEST_A["Test 1J32: Expand deviation features<br/>to cover cross-family structural markers<br/>OR add 'wrong_family' synthetic feature"]
    BRANCH_B --> TEST_B["Test 1J33: Change key to<br/>(goal, action, dev_feat) — drop family<br/>OR use (goal, action, wrong_family_flag)<br/>to directly penalize cross-family cases"]

    BRANCH_A --> TEST_C["Test 1J34: Audit which features<br/>ARE attached to cross-family deceptives<br/>→ add those to INJECTED_DEVIATION_FEATURES"]

    BRANCH_B --> TEST_D["Test 1J35: Per-family penalty<br/>B_family_only already has signal<br/>(family_only PV=24 vs A=29)<br/>Can family-only penalty + deviation<br/>penalty be combined?"]

    QUESTION --> BRANCH_C["Hypothesis C:<br/>Observation bottleneck<br/>The deviation features exist on<br/>ground-truth objects but are not<br/>observable by the agent"]
    BRANCH_C --> TEST_E["Test 1J36: Check ground truth<br/>feature set of deceptive objects<br/>vs observed feature set"]

    FINDINGS --> NOTE["Note: 3 cases (apple_006/007/011)<br/>DO have damp_texture and nonzero penalty<br/>but still fail due to rank_unchanged.<br/>Scaling from 1.0→3.0 doesn't help<br/>because penalty at scale 3 still doesn't<br/>change the relative ranking."]

    NOTE --> TEST_F["Test 1J37: Try larger penalty scale<br/>(5.0, 10.0) OR multiplicative penalty<br/>instead of exponential discount<br/>to amplify signal for these 3 cases"]
```

---

## Summary of Key Numbers

| Metric | Value |
|--------|-------|
| Seed 109 | A PV=29, FamilyOnly=24, Offline=6 |
| Missed offline-avoidable PVs | 16 |
| Dominant failure (1J30) | raw_penalty_zero (12/16) |
| Root cause (1J31) | feature_not_in_memory_universe (12/12) |
| Candidates with nonzero raw signed | 3/38 |
| Rank unchanged cases | 4 (the 3 nonzero cases + stone_block_013) |
| Scale 1 vs Scale 3 probe set identical | True |
| Observation bottleneck | False |
| Memory support bottleneck | False |
| No oracle/hidden leakage | Confirmed |
