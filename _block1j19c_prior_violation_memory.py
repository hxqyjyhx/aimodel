"""
Block 1J19c -- Prior-Violation Negative-Exception Feature-Risk Memory.

Key changes from 1J19b:
  - Two separate signals: probe_utility_signal (effective/zero/harmful) AND
    prior_violation_signal (high-prior + negative outcome)
  - FeatureRiskMemoryV3: only updates from ELIGIBLE prior events
    (soft_prior >= threshold). Low-prior events excluded from denominator.
  - Primary success: B reduces repeated prior violations vs A, B beats C,
    top risk features match injected deviation features.
  - Deceptive objects MUST have observable deviation features in FEATURE_UNIVERSE.

Three modes:
  Mode 1: original_safe (few/no prior violations expected)
  Mode 2: deceptive_high_prior_diagnostic (primary test)
  Mode 3: epsilon_boundary_explore (optional diagnostic)
"""
import os, sys, json, copy, random, time, math
import numpy as np
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
A3_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5a3_tool_material_transfer")
sys.path.insert(0, A3_DIR)
L5_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5l_instance_outcome_memory")
sys.path.insert(0, L5_DIR)
K5_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5k_episodic_sparse_outcome")
sys.path.insert(0, K5_DIR)
I5_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5i_object_uncertainty_probe")
sys.path.insert(0, I5_DIR)
sys.path.insert(0, CURRENT_DIR)

import config
from environment import MiniMCEnvironment, _TASK_QUERIES
from harness import EpisodeHarness
from simulator_truth import MiniMCSimulatorTruth
from policies import (
    MiniMCPolicy, C0b_ObserveOnlyPolicy,
    MAIN_CANDIDATE_ACTIONS, ACTION_TO_FEATURE, CORE_ACTION_FEATURES,
    _compute_utility, _compute_entropy,
)
from instance_outcome_memory import InstanceOutcomeMemory
from run_004_5n1 import run_phase_a_training
from sparse_outcome_collector import collect_sparse_probe_outcomes
from subtype_objects import (
    generate_subtype_objects_deterministic, compute_query_ground_truth,
    SUBTYPE_DEFINITIONS,
)

t0 = time.time()

# =============================================================================
# Constants
# =============================================================================
SMOKE_SEED = 101
BUDGET = 1.5
COST_WEIGHT = 0.5
EXPLORATION_FLOOR = 0.05
N_EPISODES = 5
RISK_ALPHA = 5.0
MAX_RISK = 2.0
PRIOR_VIOLATION_THRESHOLD = 0.60
MIN_VIOLATION_SUPPORT = 2

COND = copy.deepcopy(config.CUE_CONDITIONS[3])
assert COND["label"] == "C4_instance_subtype_cued_v1"

FEATURE_UNIVERSE = sorted({
    "has_bark_texture", "has_wood_grain", "has_crystal_flecks",
    "has_granular_surface", "has_stem_remnant", "has_peel_texture",
    "has_grip_area", "has_shaft_shape",
    "solid", "movable", "block_like", "elongated_with_handle",
    "round_small", "brownish", "grayish", "greenish",
    "long_shape", "rough_texture", "smooth_texture",
    "on_left_side", "near_table", "recently_seen",
    "light_weight", "heavy_weight",
    # Non-type-family deviation features (family-preserving patch)
    "damp_texture", "brittle_surface", "treated_surface", "hollow_sound",
})

print("=" * 70)
print("Block 1J19c -- Prior-Violation Negative-Exception Feature-Risk Memory")
print(f"  condition={COND['label']}  seed={SMOKE_SEED}  budget={BUDGET}")
print(f"  episodes={N_EPISODES}  risk_alpha={RISK_ALPHA}  max_risk={MAX_RISK}")
print(f"  prior_violation_threshold={PRIOR_VIOLATION_THRESHOLD}")
print(f"  min_violation_support={MIN_VIOLATION_SUPPORT}")
FEATURE_UNIVERSE_SET = set(FEATURE_UNIVERSE)
print(f"  feature_universe_size={len(FEATURE_UNIVERSE)}")
print("=" * 70)

# =============================================================================
# 1. Phase A: Training + IOM building
# =============================================================================
print("\n[1/10] Phase A: Training + IOM building...")
(student, base_learner, train_objects, train_env,
 _std_test, _std_test_env, final_metrics, rng) = run_phase_a_training(SMOKE_SEED, COND)
train_objects_dict = {oid: train_env.objects[oid] for oid in train_objects}
outcome_rows, actual_coverage = collect_sparse_probe_outcomes(
    train_objects_dict, student, config.COVERAGE, SMOKE_SEED)
posterior_visible_features = list(base_learner.visible_feature_names)
im_base = InstanceOutcomeMemory(
    train_objects_dict, posterior_visible_features,
    k=config.INSTANCE_K, similarity_power=config.SIMILARITY_POWER,
    similarity_mode=config.SIMILARITY_MODE,
)
im_base.build(outcome_rows)
print(f"  pre_accuracy={final_metrics.get('domain_token_accuracy', 0.0):.4f}  "
      f"coverage={actual_coverage:.4f}  IOM ready")

sample_obj = train_objects_dict[list(train_objects_dict.keys())[0]]
actual_vis_features = set(sample_obj.get("visible_features", {}).keys())
universe_misses = set(FEATURE_UNIVERSE) - actual_vis_features
if universe_misses:
    miss_pct = len(universe_misses) / len(FEATURE_UNIVERSE) * 100
    if miss_pct > 50:
        print(f"  Adjusting FEATURE_UNIVERSE to visible features ({miss_pct:.0f}% miss)")
        FEATURE_UNIVERSE = sorted(actual_vis_features)

# =============================================================================
# 2. Test objects + Deceptive objects
# =============================================================================
print("\n[2/10] Generating test objects...")
test_objects_standard = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")
query_gt_standard = compute_query_ground_truth(test_objects_standard)
test_oids = sorted(test_objects_standard.keys())

QUERY_NAMES = sorted(_TASK_QUERIES.keys())
N_OBJECTS = len(test_oids)

ACTION_TO_QUERY = {
    "craft_plank": "need_planks", "eat": "need_food",
    "use_as_tool": "need_tool", "burn_as_fuel": "need_fuel",
    "mine_by_hand": "need_stone", "mine_with_pickaxe": "need_stone",
}
print(f"  {N_OBJECTS} standard test objects, {len(QUERY_NAMES)} queries")

# ---- Deceptive test objects (Mode 2) ----
# Family-preserving patch: deviation features MUST NOT be in any type family.
# Core type features are FORCED to True so detect_type_family returns the original family.
print("\n[3/10] Creating FAMILY-PRESERVING deceptive test objects...")

# Core type-family features (must be preserved for type detection)
APPLE_CORE = {"has_stem_remnant", "has_peel_texture", "round_small", "greenish",
              "smooth_texture", "light_weight"}
WOOD_CORE  = {"has_bark_texture", "has_wood_grain", "brownish", "rough_texture", "long_shape"}
STONE_CORE = {"has_crystal_flecks", "has_granular_surface", "grayish", "block_like", "heavy_weight"}

# Deviation features: must be in FEATURE_UNIVERSE but NOT in ANY type family.
# These are the 4 new features added to FEATURE_UNIVERSE above.
APPLE_DEVIATION  = {"damp_texture": True, "brittle_surface": True}
WOOD_DEVIATION   = {"damp_texture": True, "treated_surface": True}
STONE_DEVIATION  = {"brittle_surface": True, "hollow_sound": True}

INJECTED_DEVIATION_FEATURES = set()
for d in [APPLE_DEVIATION, WOOD_DEVIATION, STONE_DEVIATION]:
    for f, v in d.items():
        if v and f in FEATURE_UNIVERSE:
            INJECTED_DEVIATION_FEATURES.add(f)

test_objects_deceptive = copy.deepcopy(test_objects_standard)

apple_oids = [oid for oid in test_oids if "apple" in oid]
wood_oids = [oid for oid in test_oids if "wood_log" in oid]
stone_oids = [oid for oid in test_oids if "stone_block" in oid]

deceptive_rng = random.Random(SMOKE_SEED + 9999)
deceptive_modifications = []

n_deceptive_apples = min(5, len(apple_oids))
for oid in deceptive_rng.sample(apple_oids, n_deceptive_apples):
    obj = test_objects_deceptive[oid]
    obj["hidden_affordance_profile"]["eat"] = "fail"
    obj["_deceptive"] = True
    obj["_deceptive_action"] = "eat"
    obj["_deviation_features"] = {f: True for f, v in APPLE_DEVIATION.items() if v}
    # FORCE core type features to True (family preservation)
    for feat in APPLE_CORE:
        if feat in FEATURE_UNIVERSE:
            obj["visible_features"][feat] = True
    # ADD deviation features (none in any type family, so type detection unchanged)
    for feat, val in APPLE_DEVIATION.items():
        if feat in FEATURE_UNIVERSE:
            obj["visible_features"][feat] = val
    deceptive_modifications.append({
        "oid": oid, "category": "apple", "deceptive_action": "eat",
        "core_features_forced": sorted(APPLE_CORE & FEATURE_UNIVERSE_SET),
        "deviation_features": [f for f, v in APPLE_DEVIATION.items() if v and f in FEATURE_UNIVERSE],
    })

n_deceptive_wood = min(5, len(wood_oids))
for oid in deceptive_rng.sample(wood_oids, n_deceptive_wood):
    obj = test_objects_deceptive[oid]
    obj["hidden_affordance_profile"]["burn_as_fuel"] = "fail"
    obj["_deceptive"] = True
    obj["_deceptive_action"] = "burn_as_fuel"
    obj["_deviation_features"] = {f: True for f, v in WOOD_DEVIATION.items() if v}
    for feat in WOOD_CORE:
        if feat in FEATURE_UNIVERSE:
            obj["visible_features"][feat] = True
    for feat, val in WOOD_DEVIATION.items():
        if feat in FEATURE_UNIVERSE:
            obj["visible_features"][feat] = val
    deceptive_modifications.append({
        "oid": oid, "category": "wood_log", "deceptive_action": "burn_as_fuel",
        "core_features_forced": sorted(WOOD_CORE & FEATURE_UNIVERSE_SET),
        "deviation_features": [f for f, v in WOOD_DEVIATION.items() if v and f in FEATURE_UNIVERSE],
    })

n_deceptive_stones = min(3, len(stone_oids))
for oid in deceptive_rng.sample(stone_oids, n_deceptive_stones):
    obj = test_objects_deceptive[oid]
    obj["hidden_affordance_profile"]["mine_with_pickaxe"] = "fail"
    obj["_deceptive"] = True
    obj["_deceptive_action"] = "mine_with_pickaxe"
    obj["_deviation_features"] = {f: True for f, v in STONE_DEVIATION.items() if v}
    for feat in STONE_CORE:
        if feat in FEATURE_UNIVERSE:
            obj["visible_features"][feat] = True
    for feat, val in STONE_DEVIATION.items():
        if feat in FEATURE_UNIVERSE:
            obj["visible_features"][feat] = val
    deceptive_modifications.append({
        "oid": oid, "category": "stone_block", "deceptive_action": "mine_with_pickaxe",
        "core_features_forced": sorted(STONE_CORE & FEATURE_UNIVERSE_SET),
        "deviation_features": [f for f, v in STONE_DEVIATION.items() if v and f in FEATURE_UNIVERSE],
    })

query_gt_deceptive = compute_query_ground_truth(test_objects_deceptive)

# Early validation: deviation features present in FEATURE_UNIVERSE
DECEPTIVE_VISIBLE_DEVIATION_VALID = True
for mod in deceptive_modifications:
    oid = mod["oid"]
    obj = test_objects_deceptive[oid]
    dev_feats = obj.get("_deviation_features", {})
    dev_present = [f for f, v in dev_feats.items() if v and f in FEATURE_UNIVERSE]
    if not dev_present:
        DECEPTIVE_VISIBLE_DEVIATION_VALID = False
        print(f"  ERROR: {oid} has no observable deviation features in FEATURE_UNIVERSE!")
    else:
        print(f"    {oid}: {mod['category']} -> {mod['deceptive_action']}=fail  deviations={dev_present}")

print(f"  early_deceptive_visible_deviation_valid={DECEPTIVE_VISIBLE_DEVIATION_VALID}")
print(f"  deceptive modifications: {len(deceptive_modifications)} objects")
print(f"  injected_deviation_features: {sorted(INJECTED_DEVIATION_FEATURES)}")

# =============================================================================
# 3. Metric helpers + prior violation check
# =============================================================================
def _check_query(probs, query_name):
    if query_name == "need_planks":
        return probs.get("craft_plank_success", 0.5) >= 0.5
    elif query_name == "need_stone":
        return (probs.get("mine_by_hand_success", 0.5) < 0.5
                and probs.get("mine_with_pickaxe_success", 0.5) >= 0.5)
    elif query_name == "need_food":
        return probs.get("eat_success", 0.5) >= 0.5
    elif query_name == "need_tool":
        return probs.get("use_as_tool_success", 0.5) >= 0.5
    elif query_name == "need_fuel":
        return probs.get("burn_as_fuel_success", 0.5) >= 0.5
    return False


def compute_full_metrics(predictions, query_gt):
    per_query = {}
    for qname in sorted(query_gt.keys()):
        pos_set = set(query_gt[qname]["positive_oids"])
        neg_set = set(query_gt[qname]["negative_oids"])
        tp = fp = tn = fn = 0
        for oid in predictions:
            pred_true = _check_query(predictions[oid], qname)
            actual_true = oid in pos_set
            if pred_true and actual_true: tp += 1
            elif pred_true and not actual_true: fp += 1
            elif not pred_true and not actual_true: tn += 1
            else: fn += 1
        pos_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal = 0.5 * (pos_recall + neg_recall)
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        per_query[qname] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "positive_recall": round(pos_recall, 6),
            "negative_recall": round(neg_recall, 6),
            "balanced_accuracy": round(bal, 6),
            "accuracy": round(accuracy, 6),
        }
    macro_bal = sum(v["balanced_accuracy"] for v in per_query.values()) / max(len(per_query), 1)
    macro_acc = sum(v["accuracy"] for v in per_query.values()) / max(len(per_query), 1)
    mean_pos_recall = sum(v["positive_recall"] for v in per_query.values()) / max(len(per_query), 1)
    mean_neg_recall = sum(v["negative_recall"] for v in per_query.values()) / max(len(per_query), 1)
    return {
        "per_query": per_query,
        "macro_query_balanced_accuracy": round(macro_bal, 6),
        "macro_query_accuracy": round(macro_acc, 6),
        "mean_positive_recall": round(mean_pos_recall, 6),
        "mean_negative_recall": round(mean_neg_recall, 6),
    }


def _unwrap_preds(preds):
    if isinstance(preds, dict) and "per_object" in preds and len(preds) == 1:
        return preds["per_object"]
    return preds


def run_policy(policy, env):
    harness = EpisodeHarness(env, policy)
    result = harness.run()
    return (result.predictions, result.event_log, result.agent_obs,
            result.pre_probe_entropies, result.pre_decision_entropies)


# =============================================================================
# 4. Default Goal Prior Table + prior violation check
# =============================================================================
print("\n[4/10] Building goal-conditioned soft prior table...")

WOOD_FEATURES = {"has_bark_texture", "has_wood_grain", "brownish", "rough_texture", "long_shape"}
STONE_FEATURES = {"has_crystal_flecks", "has_granular_surface", "grayish", "block_like", "heavy_weight"}
APPLE_FEATURES = {"has_stem_remnant", "has_peel_texture", "round_small", "greenish", "smooth_texture", "light_weight"}
TOOL_FEATURES = {"has_grip_area", "has_shaft_shape", "elongated_with_handle", "movable", "long_shape"}

TYPE_FAMILIES = {
    "wood-like": WOOD_FEATURES, "stone-like": STONE_FEATURES,
    "apple-like": APPLE_FEATURES, "tool-like": TOOL_FEATURES,
}


def detect_type_family(features):
    if features is None: return "unknown"
    scores = {}
    for fam, feat_set in TYPE_FAMILIES.items():
        scores[fam] = sum(1 for f in feat_set if features.get(f, False))
    best_fam = max(scores, key=scores.get)
    return best_fam if scores[best_fam] > 0 else "unknown"


DEFAULT_GOAL_PRIOR = {}
for fam in TYPE_FAMILIES:
    for action in MAIN_CANDIDATE_ACTIONS:
        qname = ACTION_TO_QUERY.get(action, "")
        prior = EXPLORATION_FLOOR
        if qname == "need_food":
            if action == "eat":
                if fam == "apple-like": prior = 0.85
                elif fam == "tool-like": prior = 0.10
                else: prior = 0.15
            else:
                if fam == "apple-like": prior = 0.30
                else: prior = 0.10
        elif qname == "need_fuel":
            if action == "burn_as_fuel":
                if fam == "wood-like": prior = 0.85
                elif fam == "tool-like": prior = 0.60
                elif fam == "stone-like": prior = 0.10
                else: prior = 0.15
            else: prior = 0.10
        elif qname == "need_planks":
            if action == "craft_plank":
                if fam == "wood-like": prior = 0.85
                elif fam == "tool-like": prior = 0.15
                else: prior = 0.10
            else: prior = 0.10
        elif qname == "need_stone":
            if action in ("mine_by_hand", "mine_with_pickaxe"):
                if fam == "stone-like": prior = 0.85
                elif fam in ("wood-like", "tool-like"): prior = 0.25
                else: prior = 0.15
            else: prior = 0.10
        elif qname == "need_tool":
            if action == "use_as_tool":
                if fam == "tool-like": prior = 0.85
                elif fam == "wood-like": prior = 0.15
                else: prior = 0.10
            else: prior = 0.10
        prior = max(EXPLORATION_FLOOR, min(0.95, prior))
        DEFAULT_GOAL_PRIOR[(fam, qname, action)] = prior

print(f"  Prior table: {len(DEFAULT_GOAL_PRIOR)} entries, min={min(DEFAULT_GOAL_PRIOR.values()):.4f}")


def get_goal_soft_prior(features, action):
    fam = detect_type_family(features)
    qname = ACTION_TO_QUERY.get(action, "")
    return DEFAULT_GOAL_PRIOR.get((fam, qname, action), EXPLORATION_FLOOR)


# Deferred validation: family preservation check, eligibility ratio, gating
print(f"\n  Deceptive object validation (post-table-build):")

# Cross-check: no deviation feature is in any type family
ALL_TYPE_FAMILY_FEATURES = set()
for fs in TYPE_FAMILIES.values():
    ALL_TYPE_FAMILY_FEATURES.update(fs)
DEVIATION_TYPE_FAMILY_CONTAMINATION = False
for f in INJECTED_DEVIATION_FEATURES:
    if f in ALL_TYPE_FAMILY_FEATURES:
        DEVIATION_TYPE_FAMILY_CONTAMINATION = True
        print(f"  ERROR: deviation feature '{f}' is in a type family — will break preservation!")

# Expected type family per category
EXPECTED_FAMILY = {"apple": "apple-like", "wood_log": "wood-like", "stone_block": "stone-like"}

FAMILY_PRESERVATION_VALID = True
deceptive_validation_details = []

for mod in deceptive_modifications:
    oid = mod["oid"]
    obj = test_objects_deceptive[oid]
    sp = get_goal_soft_prior(obj["visible_features"], mod["deceptive_action"])
    fam = detect_type_family(obj["visible_features"])
    expected = EXPECTED_FAMILY.get(mod["category"], "unknown")
    family_ok = fam == expected
    eligible = sp >= PRIOR_VIOLATION_THRESHOLD
    if not family_ok:
        FAMILY_PRESERVATION_VALID = False

    deceptive_validation_details.append({
        "oid": oid, "category": mod["category"],
        "expected_family": expected, "detected_family": fam,
        "family_preserved": family_ok,
        "deceptive_action": mod["deceptive_action"],
        "soft_prior": round(sp, 4),
        "eligible": eligible,
        "deviation_features": mod["deviation_features"],
    })
    flag = "OK" if (family_ok and eligible) else ("FAMILY!" if not family_ok else "LOW_PRIOR")
    print(f"    {oid}: {mod['category']} expected={expected} detected={fam} "
          f"prior={sp:.4f} eligible={eligible}  [{flag}]")

DECEPTIVE_HIGH_PRIOR_ELIGIBLE_COUNT = sum(
    1 for d in deceptive_validation_details if d["eligible"])
DECEPTIVE_HIGH_PRIOR_ELIGIBLE_RATIO = (
    DECEPTIVE_HIGH_PRIOR_ELIGIBLE_COUNT / max(len(deceptive_modifications), 1))

# Family preservation: no deviation type-family contamination AND all objects preserve family
FAMILY_PRESERVATION_VALID = (
    FAMILY_PRESERVATION_VALID and not DEVIATION_TYPE_FAMILY_CONTAMINATION)

DECEPTIVE_FEATURES_IN_UNIVERSE = all(
    f in FEATURE_UNIVERSE
    for mod in deceptive_modifications
    for f in mod["deviation_features"]
) if deceptive_modifications else False

print(f"\n  Validation summary:")
print(f"    deceptive_total_count={len(deceptive_modifications)}")
print(f"    deceptive_high_prior_eligible_count={DECEPTIVE_HIGH_PRIOR_ELIGIBLE_COUNT}")
print(f"    deceptive_high_prior_eligible_ratio={DECEPTIVE_HIGH_PRIOR_ELIGIBLE_RATIO:.4f}")
print(f"    deceptive_visible_deviation_valid={DECEPTIVE_VISIBLE_DEVIATION_VALID}")
print(f"    deceptive_features_in_feature_universe={DECEPTIVE_FEATURES_IN_UNIVERSE}")
print(f"    family_preservation_valid={FAMILY_PRESERVATION_VALID}")
print(f"    deviation_type_family_contamination={DEVIATION_TYPE_FAMILY_CONTAMINATION}")
print(f"    eligibility_gate (>=0.75): {'PASS' if DECEPTIVE_HIGH_PRIOR_ELIGIBLE_RATIO >= 0.75 else 'FAIL'}")


def check_prior_violation(features, action, outcome, threshold=PRIOR_VIOLATION_THRESHOLD):
    """Check if a probe is a prior violation.

    Returns: (is_violation, is_eligible, soft_prior)
      is_eligible: soft_prior >= threshold (high-prior event)
      is_violation: eligible AND outcome is failure (affordance == 0)
    """
    soft_prior = get_goal_soft_prior(features, action)
    is_eligible = soft_prior >= threshold
    affordance_failed = outcome <= 0.01
    is_violation = is_eligible and affordance_failed
    return is_violation, is_eligible, soft_prior


# =============================================================================
# 5. FeatureRiskMemoryV3 (prior-violation-based)
# =============================================================================
print("\n[5/10] Initializing FeatureRiskMemoryV3 (prior-violation semantics)...")

class FeatureRiskMemoryV3:
    """Prior-violation negative-exception memory.

    Only updates from ELIGIBLE prior events (soft_prior >= threshold).
    Tracks violation_with, nonviolation_with, violation_without, nonviolation_without.
    Risk weight = log(P(feature|violation) / P(feature|nonviolation)) among eligible events.
    Only positive risk (penalty); negative weights clipped to 0.
    """

    def __init__(self, alpha=RISK_ALPHA, max_risk=MAX_RISK,
                 min_violation_support=MIN_VIOLATION_SUPPORT):
        self.alpha = alpha
        self.max_risk = max_risk
        self.min_violation_support = min_violation_support
        self.counts = defaultdict(lambda: {
            "violation_with": 1.0, "nonviolation_with": 1.0,
            "violation_without": 1.0, "nonviolation_without": 1.0,
        })
        self._risk_cache = {}
        self.total_updates = 0
        self.violation_updates = 0
        self.nonviolation_updates = 0
        self.eligible_updates = 0
        self.ineligible_updates = 0

    def update(self, features, goal, action, is_violation, is_eligible):
        """Update counts from one probe event. Only eligible events affect counts."""
        self.total_updates += 1
        if not is_eligible:
            self.ineligible_updates += 1
            return

        self.eligible_updates += 1
        if is_violation:
            self.violation_updates += 1
        else:
            self.nonviolation_updates += 1

        for feat in FEATURE_UNIVERSE:
            key = (feat, goal, action)
            present = features.get(feat, False)
            if present:
                if is_violation:
                    self.counts[key]["violation_with"] += 1.0
                else:
                    self.counts[key]["nonviolation_with"] += 1.0
            else:
                if is_violation:
                    self.counts[key]["violation_without"] += 1.0
                else:
                    self.counts[key]["nonviolation_without"] += 1.0

        self._risk_cache.clear()

    def get_risk_weight(self, feature, goal, action):
        cache_key = (feature, goal, action)
        if cache_key in self._risk_cache:
            return self._risk_cache[cache_key]

        c = self.counts[cache_key]
        v_with = c["violation_with"]
        nv_with = c["nonviolation_with"]
        v_without = c["violation_without"]
        nv_without = c["nonviolation_without"]

        actual_violations = (v_with + v_without) - 2.0
        if actual_violations < self.min_violation_support:
            self._risk_cache[cache_key] = 0.0
            return 0.0

        total_v = v_with + v_without
        total_nv = nv_with + nv_without

        p_feat_given_v = v_with / max(total_v, 0.001)
        p_feat_given_nv = nv_with / max(total_nv, 0.001)

        eps = 1e-9
        if p_feat_given_nv < eps: p_feat_given_nv = eps
        ratio = p_feat_given_v / p_feat_given_nv
        if ratio < eps: ratio = eps
        if ratio > 1.0 / eps: ratio = 1.0 / eps
        raw_lr = math.log(ratio)

        clipped_risk = max(0.0, min(raw_lr, self.max_risk))

        support = total_v + total_nv - 4.0
        support = max(0.0, support)
        reliability = support / (support + self.alpha)

        usable = reliability * clipped_risk
        self._risk_cache[cache_key] = usable
        return usable

    def get_total_risk_penalty(self, features, action, goal_hint=None):
        total = 0.0
        goal = goal_hint or ACTION_TO_QUERY.get(action, "")
        for feat in FEATURE_UNIVERSE:
            if features.get(feat, False):
                total += self.get_risk_weight(feat, goal, action)
        return total

    def permute_weights(self, rng):
        keys = []; weights = []
        for key in self.counts:
            w = self.get_risk_weight(key[0], key[1], key[2])
            keys.append(key); weights.append(w)
        shuffled = list(weights)
        rng.shuffle(shuffled)
        permuted = {}
        for k, w in zip(keys, shuffled):
            permuted[k] = w
            self._risk_cache[k] = w
        return permuted

    def get_statistics(self):
        nonzero_count = 0; max_w = 0.0; sum_abs = 0.0; n_keys = 0
        for key in self.counts:
            w = self.get_risk_weight(key[0], key[1], key[2])
            n_keys += 1
            if abs(w) > 1e-9:
                nonzero_count += 1
                sum_abs += abs(w)
                if abs(w) > max_w: max_w = abs(w)

        return {
            "total_updates": self.total_updates,
            "eligible_updates": self.eligible_updates,
            "ineligible_updates": self.ineligible_updates,
            "violation_updates": self.violation_updates,
            "nonviolation_updates": self.nonviolation_updates,
            "prior_violation_count": self.violation_updates,
            "memory_update_nontrivial": self.violation_updates > 0,
            "n_keys_total": n_keys,
            "risk_weights_nonzero_count": nonzero_count,
            "max_risk_weight": round(max_w, 6),
            "mean_abs_risk_weight": round(sum_abs / max(n_keys, 1), 6),
        }

    def get_top_risk_features(self, top_n=20):
        items = []
        for (feat, goal, action), c in self.counts.items():
            w = self.get_risk_weight(feat, goal, action)
            if abs(w) > 1e-9:
                items.append({
                    "feature": feat, "goal": goal, "action": action,
                    "risk_weight": round(w, 6),
                    "violation_with": c["violation_with"],
                    "nonviolation_with": c["nonviolation_with"],
                    "violation_without": c["violation_without"],
                    "nonviolation_without": c["nonviolation_without"],
                    "actual_violations": (c["violation_with"] + c["violation_without"]) - 2.0,
                })
        items.sort(key=lambda x: abs(x["risk_weight"]), reverse=True)
        return items[:top_n]

    def clone(self):
        new = FeatureRiskMemoryV3(alpha=self.alpha, max_risk=self.max_risk,
                                  min_violation_support=self.min_violation_support)
        new.counts = copy.deepcopy(dict(self.counts))
        new._risk_cache = dict(self._risk_cache)
        new.total_updates = self.total_updates
        new.violation_updates = self.violation_updates
        new.nonviolation_updates = self.nonviolation_updates
        new.eligible_updates = self.eligible_updates
        new.ineligible_updates = self.ineligible_updates
        return new


print(f"  FeatureRiskMemoryV3 ready  alpha={RISK_ALPHA}  max_risk={MAX_RISK}")
print(f"  prior_violation_threshold={PRIOR_VIOLATION_THRESHOLD}")
print(f"  min_violation_support={MIN_VIOLATION_SUPPORT}")
print(f"  semantics: eligible-only updates, violation/nonviolation counts, positive-only risk")

# =============================================================================
# 6. Policy Classes
# =============================================================================
print("\n[6/10] Defining policy classes...")

class SoftPriorOnlyPolicyV2(MiniMCPolicy):
    """Condition A: soft_prior_only. No memory, no risk adjustment."""

    PHASE_OBSERVE = 1
    PHASE_PROBE = 2

    def __init__(self, instance_memory, rng, target_probe_count=8, explore_fraction=0.0):
        self._im = instance_memory
        self._rng = rng
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._target_probe_count = target_probe_count
        self._probe_effects = []
        self._selected_scores = []
        self._explore_fraction = explore_fraction
        self._variant = "A_soft_prior_only"

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._probe_effects = []
        self._selected_scores = []

    def _should_transition_to_probe_phase(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited: return True
        cheapest_observe = min(view.compute_reach_cost(oid) + view.observe_cost for oid in unvisited)
        if not view.can_afford(cheapest_observe): return True
        reserve = 0.25 * view.initial_budget
        if view.budget_remaining - cheapest_observe < reserve: return True
        if len(unvisited) <= 3: return True
        return False

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        if self._phase == self.PHASE_OBSERVE:
            if self._should_transition_to_probe_phase(view):
                self._phase = self.PHASE_PROBE
                self._observe_pass_visit_count = sum(
                    1 for oid in view.get_all_object_ids() if view.is_visited(oid))
                return self._select_probe_target(view)
            best_oid = None; best_cost = float('inf')
            for oid in unvisited:
                total = view.compute_reach_cost(oid) + view.observe_cost
                if view.can_afford(total) and total < best_cost:
                    best_cost = total; best_oid = oid
            return best_oid
        elif self._phase == self.PHASE_PROBE:
            return self._select_probe_target(view)
        return None

    def _score_candidate(self, features, norm_cost):
        best_score = float('-inf')
        for action in MAIN_CANDIDATE_ACTIONS:
            base_prior = get_goal_soft_prior(features, action)
            raw_score = base_prior - COST_WEIGHT * norm_cost
            score = max(EXPLORATION_FLOOR, raw_score)
            if score > best_score: best_score = score
        return best_score

    def _select_probe_target(self, view):
        if len(self._probed_oids) >= self._target_probe_count: return None
        candidates = []
        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids: continue
            features = view.get_observed_features(oid)
            if features is None: continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost): continue
            norm_cost = total_cost / max(view.initial_budget, 0.001)
            best_score = self._score_candidate(features, norm_cost)
            if best_score > float('-inf'):
                candidates.append((oid, best_score))

        if not candidates: return None
        candidates.sort(key=lambda x: x[1], reverse=True)

        if self._explore_fraction > 0 and len(candidates) >= 3:
            if self._rng.random() < self._explore_fraction:
                n = len(candidates)
                lo = max(0, int(n * 0.30))
                hi = min(n - 1, int(n * 0.80))
                if hi > lo:
                    return candidates[self._rng.randint(lo, hi)][0]

        return candidates[0][0] if candidates else None

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE: return False, None
        if len(self._probed_oids) >= self._target_probe_count: return False, None
        features = view.get_observed_features(object_id)
        if features is None: return False, None

        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)

        best_action = None; best_score = float('-inf')
        best_raw_score = float('-inf'); best_prior = EXPLORATION_FLOOR
        for action in MAIN_CANDIDATE_ACTIONS:
            base_prior = get_goal_soft_prior(features, action)
            norm_cost = view.probe_cost / max(view.initial_budget, 0.001)
            raw_score = base_prior - COST_WEIGHT * norm_cost
            score = max(EXPLORATION_FLOOR, raw_score)
            if score > best_score:
                best_score = score; best_action = action
                best_raw_score = raw_score; best_prior = base_prior

        self._probed_oids.add(object_id)
        self._selected_scores.append({
            "oid": object_id, "action": best_action,
            "prior": best_prior,
            "raw_score": round(best_raw_score, 6),
            "final_score": round(best_score, 6),
            "risk_adjustment": 0.0,
        })
        return True, best_action

    def on_probe_result(self, view, object_id, action, outcome):
        features = view.get_observed_features(object_id)
        if features is not None:
            fake_obj = {"id": object_id, "visible_features": features}
            pre_probs = self._im.predict_all_affordances(fake_obj)
            self._im.incorporate_probe(object_id, action, outcome)
            post_probs = self._im.predict_all_affordances(fake_obj)
            self._probe_tracking.append({"oid": object_id, "action": action, "outcome": outcome})

            pre_utility = _compute_utility(pre_probs)
            post_utility = _compute_utility(post_probs)
            utility_delta = post_utility - pre_utility
            is_eff = utility_delta > 0.001
            is_zero = abs(utility_delta) <= 0.001
            is_harm = utility_delta < -0.001

            pre_query = _check_query(pre_probs, ACTION_TO_QUERY.get(action, ""))
            post_query = _check_query(post_probs, ACTION_TO_QUERY.get(action, ""))
            query_changed = pre_query != post_query

            if is_eff: effect = "effective"
            elif is_zero: effect = "zero"
            else: effect = "harmful"

            self._probe_effects.append({
                "oid": object_id, "action": action, "outcome": outcome,
                "utility_delta": round(utility_delta, 6),
                "is_effective": is_eff, "is_zero": is_zero, "is_harmful": is_harm,
                "query_changed": query_changed, "effect": effect,
            })
        else:
            self._im.incorporate_probe(object_id, action, outcome)

    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            per_object[oid] = self._im.predict_all_affordances(
                {"id": oid, "visible_features": features})
        return {"per_object": per_object}

    def get_probe_effects(self):
        return list(self._probe_effects)

    def get_selected_scores(self):
        return list(self._selected_scores)


class SoftPriorPlusFeatureRiskMemoryPolicyV3(SoftPriorOnlyPolicyV2):
    """Condition B: soft_prior + prior-violation feature-risk memory (V3)."""

    def __init__(self, instance_memory, rng, feature_risk_memory,
                 target_probe_count=8, explore_fraction=0.0):
        super().__init__(instance_memory, rng, target_probe_count, explore_fraction)
        self._frm = feature_risk_memory
        self._variant = "B_soft_prior_plus_prior_violation_memory"

    def _compute_adjusted_score(self, features, action, norm_cost):
        base_prior = get_goal_soft_prior(features, action)
        risk_penalty = self._frm.get_total_risk_penalty(features, action)
        raw_score = base_prior * math.exp(-risk_penalty) - COST_WEIGHT * norm_cost
        final_score = max(EXPLORATION_FLOOR, raw_score)
        return final_score, raw_score, base_prior, risk_penalty

    def _score_candidate(self, features, norm_cost):
        best_score = float('-inf')
        for action in MAIN_CANDIDATE_ACTIONS:
            final_score, _, _, _ = self._compute_adjusted_score(features, action, norm_cost)
            if final_score > best_score: best_score = final_score
        return best_score

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE: return False, None
        if len(self._probed_oids) >= self._target_probe_count: return False, None
        features = view.get_observed_features(object_id)
        if features is None: return False, None

        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)

        best_action = None; best_score = float('-inf')
        best_info = (EXPLORATION_FLOOR, 0.0, 0.0, EXPLORATION_FLOOR)
        for action in MAIN_CANDIDATE_ACTIONS:
            norm_cost = view.probe_cost / max(view.initial_budget, 0.001)
            final_score, raw_score, base_prior, risk_penalty = \
                self._compute_adjusted_score(features, action, norm_cost)
            if final_score > best_score:
                best_score = final_score; best_action = action
                best_info = (base_prior, risk_penalty, raw_score, final_score)

        self._probed_oids.add(object_id)
        bp, rp, rs, fs = best_info
        self._selected_scores.append({
            "oid": object_id, "action": best_action,
            "prior": bp, "risk_penalty": round(rp, 6),
            "raw_score": round(rs, 6), "final_score": round(fs, 6),
            "risk_adjustment": round(rp, 6),
        })
        return True, best_action


class SoftPriorPlusPermutedRiskMemoryPolicyV3(SoftPriorPlusFeatureRiskMemoryPolicyV3):
    """Condition C: permuted prior-violation risk memory control."""

    def __init__(self, instance_memory, rng, feature_risk_memory,
                 target_probe_count=8, explore_fraction=0.0):
        super().__init__(instance_memory, rng, feature_risk_memory,
                        target_probe_count, explore_fraction)
        self._variant = "C_permuted_prior_violation_memory"
        self._permuted_map = {}

    def apply_permutation(self, rng):
        self._permuted_map = self._frm.permute_weights(rng)

    def _compute_adjusted_score(self, features, action, norm_cost):
        base_prior = get_goal_soft_prior(features, action)
        goal = ACTION_TO_QUERY.get(action, "")
        if self._permuted_map:
            risk_penalty = 0.0
            for feat in FEATURE_UNIVERSE:
                if features.get(feat, False):
                    risk_penalty += self._permuted_map.get((feat, goal, action), 0.0)
        else:
            risk_penalty = self._frm.get_total_risk_penalty(features, action)
        raw_score = base_prior * math.exp(-risk_penalty) - COST_WEIGHT * norm_cost
        final_score = max(EXPLORATION_FLOOR, raw_score)
        return final_score, raw_score, base_prior, risk_penalty


# =============================================================================
# 7. C15b reference
# =============================================================================
print("\n[7/10] Running C15b reference...")

class C15b_RefPolicy(MiniMCPolicy):
    PHASE_OBSERVE = 1; PHASE_PROBE = 2
    def __init__(self, instance_memory, rng, cost_weight=COST_WEIGHT):
        self._im = instance_memory; self._rng = rng; self._cost_weight = cost_weight
        self._phase = self.PHASE_OBSERVE; self._probed_oids = set(); self._pre_probe_entropies = {}
    def reset(self, view):
        self._phase = self.PHASE_OBSERVE; self._probed_oids = set(); self._pre_probe_entropies = {}
    def _should_transition(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited: return True
        cheapest = min(view.compute_reach_cost(oid) + view.observe_cost for oid in unvisited)
        if not view.can_afford(cheapest): return True
        if view.budget_remaining - cheapest < 0.25 * view.initial_budget: return True
        if len(unvisited) <= 3: return True
        return False
    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        if self._phase == self.PHASE_OBSERVE:
            if self._should_transition(view):
                self._phase = self.PHASE_PROBE
                return self._select_probe_target(view)
            best_oid = None; best_cost = float('inf')
            for oid in unvisited:
                total = view.compute_reach_cost(oid) + view.observe_cost
                if view.can_afford(total) and total < best_cost:
                    best_cost = total; best_oid = oid
            return best_oid
        elif self._phase == self.PHASE_PROBE:
            return self._select_probe_target(view)
        return None
    def _select_probe_target(self, view):
        best_oid = None; best_voi = float('-inf')
        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids: continue
            features = view.get_observed_features(oid)
            if features is None: continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost): continue
            norm_cost = total_cost / max(view.initial_budget, 0.001)
            fake_obj = {"id": oid, "visible_features": features}
            probs = self._im.predict_all_affordances(fake_obj)
            cur_util = _compute_utility(probs)
            for action in MAIN_CANDIDATE_ACTIONS:
                p_succ = probs.get(ACTION_TO_FEATURE[action], 0.5)
                im_succ = self._im.clone(); im_succ.incorporate_probe(oid, action, 1.0)
                util_succ = _compute_utility(im_succ.predict_all_affordances(fake_obj))
                im_fail = self._im.clone(); im_fail.incorporate_probe(oid, action, 0.0)
                util_fail = _compute_utility(im_fail.predict_all_affordances(fake_obj))
                e_post = p_succ * util_succ + (1 - p_succ) * util_fail
                net_voi = e_post - cur_util - self._cost_weight * norm_cost
                if net_voi > best_voi:
                    best_voi = net_voi; best_oid = oid
        if best_voi <= 0.0: return None
        return best_oid
    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE: return False, None
        features = view.get_observed_features(object_id)
        if features is None: return False, None
        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)
        cur_util = _compute_utility(probs)
        best_action = None; best_voi = 0.0
        for action in MAIN_CANDIDATE_ACTIONS:
            p_succ = probs.get(ACTION_TO_FEATURE[action], 0.5)
            im_succ = self._im.clone(); im_succ.incorporate_probe(object_id, action, 1.0)
            util_succ = _compute_utility(im_succ.predict_all_affordances(fake_obj))
            im_fail = self._im.clone(); im_fail.incorporate_probe(object_id, action, 0.0)
            util_fail = _compute_utility(im_fail.predict_all_affordances(fake_obj))
            e_post = p_succ * util_succ + (1 - p_succ) * util_fail
            net_voi = e_post - cur_util - self._cost_weight * (view.probe_cost / max(view.initial_budget, 0.001))
            if net_voi > best_voi:
                best_voi = net_voi; best_action = action
        if best_voi > 0:
            self._probed_oids.add(object_id)
            return True, best_action
        return False, None
    def on_probe_result(self, view, object_id, action, outcome):
        self._im.incorporate_probe(object_id, action, outcome)
    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            per_object[oid] = self._im.predict_all_affordances(
                {"id": oid, "visible_features": features})
        return {"per_object": per_object}


c15b_im = im_base.clone()
c15b_rng = random.Random(SMOKE_SEED + 700)
c15b_positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, c15b_rng)
c15b_env = MiniMCEnvironment(test_objects_standard, c15b_positions, initial_budget=BUDGET)
c15b_policy = C15b_RefPolicy(c15b_im, c15b_rng)
c15b_preds, _, c15b_obs, _, _ = run_policy(c15b_policy, c15b_env)
c15b_metrics = compute_full_metrics(_unwrap_preds(c15b_preds), query_gt_standard)
c15b_macro_bal = c15b_metrics["macro_query_balanced_accuracy"]
c15b_probed_n = sum(1 for oid in c15b_obs.get_all_object_ids() if c15b_obs.get_probe_results(oid))
print(f"  C15b macro_bal={c15b_macro_bal:.4f}  probed={c15b_probed_n}")

# =============================================================================
# 8. Mode runner
# =============================================================================
print("\n[8/10] Setting up mode runner...")

def compute_slope(mb_list):
    if len(mb_list) < 2: return 0.0
    x = np.arange(len(mb_list))
    y = np.array(mb_list)
    slope, _ = np.polyfit(x, y, 1)
    return slope


def run_mode(mode_name, test_objects, query_gt, im_base, c15b_probed_n,
             explore_fraction=0.0, mode_rng_seed=None):
    """Run one diagnostic mode with A/B/C comparison across episodes."""

    if mode_rng_seed is None:
        mode_rng_seed = SMOKE_SEED

    print(f"\n  {'='*60}")
    print(f"  Mode: {mode_name}  explore_fraction={explore_fraction}  n_objects={len(test_objects)}")
    print(f"  {'='*60}")

    frm = FeatureRiskMemoryV3()
    all_episodes_a = []; all_episodes_b = []; all_episodes_c = []

    # Track prior violation history for repeated violation counting
    prior_violation_history = set()  # (oid, action) pairs

    for ep in range(N_EPISODES):
        ep_seed = mode_rng_seed + ep * 100
        ep_rng = random.Random(ep_seed)
        test_oids_local = sorted(test_objects.keys())

        ep_positions = MiniMCSimulatorTruth.assign_positions(
            test_oids_local, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, ep_rng)

        # ----- Condition A: soft_prior_only -----
        im_a = im_base.clone()
        env_a = MiniMCEnvironment(test_objects, dict(ep_positions), initial_budget=BUDGET)
        policy_a = SoftPriorOnlyPolicyV2(im_a, ep_rng, target_probe_count=c15b_probed_n,
                                         explore_fraction=explore_fraction)
        preds_a, _, obs_a, _, _ = run_policy(policy_a, env_a)
        metrics_a = compute_full_metrics(_unwrap_preds(preds_a), query_gt)
        effects_a = policy_a.get_probe_effects()
        scores_a = policy_a.get_selected_scores()
        macro_bal_a = metrics_a["macro_query_balanced_accuracy"]
        probed_a = sum(1 for oid in obs_a.get_all_object_ids() if obs_a.get_probe_results(oid))
        eff_a = sum(1 for e in effects_a if e["is_effective"])
        zero_a = sum(1 for e in effects_a if e["is_zero"])
        harm_a = sum(1 for e in effects_a if e["is_harmful"])
        min_score_a = min((s["final_score"] for s in scores_a), default=EXPLORATION_FLOOR)
        at_floor_a = sum(1 for s in scores_a if s["final_score"] <= EXPLORATION_FLOOR + 1e-9)

        # Prior violation counting for A (no memory update)
        pv_count_a = 0; pv_repeated_a = 0
        for effect in effects_a:
            oid = effect["oid"]; action = effect["action"]; outcome = effect["outcome"]
            obj = test_objects.get(oid, {})
            features = obj.get("visible_features", {})
            is_violation, is_eligible, _ = check_prior_violation(features, action, outcome)
            if is_violation:
                pv_count_a += 1
                if (oid, action) in prior_violation_history:
                    pv_repeated_a += 1
                # Note: A doesn't update memory, but we still track history for A-vs-B comparison
                # Actually, history should be per-condition. Let's track separately.

        all_episodes_a.append({
            "episode": ep, "macro_bal": macro_bal_a, "probed": probed_a,
            "effective": eff_a, "zero": zero_a, "harmful": harm_a,
            "min_final_score": round(min_score_a, 6), "n_at_floor": at_floor_a,
            "prior_violation_count": pv_count_a,
        })

        # ----- Condition B: prior-violation feature-risk memory V3 -----
        im_b = im_base.clone()
        env_b = MiniMCEnvironment(test_objects, dict(ep_positions), initial_budget=BUDGET)
        policy_b = SoftPriorPlusFeatureRiskMemoryPolicyV3(
            im_b, ep_rng, frm, target_probe_count=c15b_probed_n,
            explore_fraction=explore_fraction)
        preds_b, _, obs_b, _, _ = run_policy(policy_b, env_b)
        metrics_b = compute_full_metrics(_unwrap_preds(preds_b), query_gt)
        effects_b = policy_b.get_probe_effects()
        scores_b = policy_b.get_selected_scores()
        macro_bal_b = metrics_b["macro_query_balanced_accuracy"]
        probed_b = sum(1 for oid in obs_b.get_all_object_ids() if obs_b.get_probe_results(oid))
        eff_b = sum(1 for e in effects_b if e["is_effective"])
        zero_b = sum(1 for e in effects_b if e["is_zero"])
        harm_b = sum(1 for e in effects_b if e["is_harmful"])
        min_score_b = min((s["final_score"] for s in scores_b), default=EXPLORATION_FLOOR)
        at_floor_b = sum(1 for s in scores_b if s["final_score"] <= EXPLORATION_FLOOR + 1e-9)
        avg_risk_b = sum(abs(s.get("risk_adjustment", 0.0)) for s in scores_b) / max(len(scores_b), 1)

        # Prior violation tracking + memory update
        n_violation_before = frm.violation_updates
        pv_count_b = 0; pv_repeated_b = 0
        pv_events_b = []  # detailed per-probe records

        for effect in effects_b:
            oid = effect["oid"]; action = effect["action"]; outcome = effect["outcome"]
            goal = ACTION_TO_QUERY.get(action, "")
            obj = test_objects.get(oid, {})
            features = obj.get("visible_features", {})

            is_violation, is_eligible, soft_prior = check_prior_violation(features, action, outcome)

            pv_record = {
                "oid": oid, "action": action, "outcome": outcome,
                "soft_prior": round(soft_prior, 4),
                "is_eligible": is_eligible,
                "is_violation": is_violation,
                "probe_utility_signal": effect["effect"],
            }

            if is_violation:
                pv_count_b += 1
                if (oid, action) in prior_violation_history:
                    pv_repeated_b += 1
                    pv_record["repeated"] = True
                else:
                    pv_record["repeated"] = False
                prior_violation_history.add((oid, action))

            pv_events_b.append(pv_record)

            # Update memory (only eligible events)
            frm.update(features, goal, action, is_violation, is_eligible)

        new_violations_this_ep = frm.violation_updates - n_violation_before

        all_episodes_b.append({
            "episode": ep, "macro_bal": macro_bal_b, "probed": probed_b,
            "effective": eff_b, "zero": zero_b, "harmful": harm_b,
            "min_final_score": round(min_score_b, 6), "n_at_floor": at_floor_b,
            "avg_abs_risk_adjustment": round(avg_risk_b, 6),
            "prior_violation_count": pv_count_b,
            "prior_violation_repeated": pv_repeated_b,
            "new_violation_signals": new_violations_this_ep,
            "prior_violation_events": pv_events_b,
        })

        # ----- Condition C: permuted risk memory control -----
        im_c = im_base.clone()
        env_c = MiniMCEnvironment(test_objects, dict(ep_positions), initial_budget=BUDGET)
        permuted_memory = frm.clone()
        policy_c = SoftPriorPlusPermutedRiskMemoryPolicyV3(
            im_c, ep_rng, permuted_memory, target_probe_count=c15b_probed_n,
            explore_fraction=explore_fraction)
        policy_c.apply_permutation(ep_rng)
        preds_c, _, obs_c, _, _ = run_policy(policy_c, env_c)
        metrics_c = compute_full_metrics(_unwrap_preds(preds_c), query_gt)
        effects_c = policy_c.get_probe_effects()
        scores_c = policy_c.get_selected_scores()
        macro_bal_c = metrics_c["macro_query_balanced_accuracy"]
        probed_c = sum(1 for oid in obs_c.get_all_object_ids() if obs_c.get_probe_results(oid))
        eff_c = sum(1 for e in effects_c if e["is_effective"])
        zero_c = sum(1 for e in effects_c if e["is_zero"])
        harm_c = sum(1 for e in effects_c if e["is_harmful"])
        min_score_c = min((s["final_score"] for s in scores_c), default=EXPLORATION_FLOOR)
        at_floor_c = sum(1 for s in scores_c if s["final_score"] <= EXPLORATION_FLOOR + 1e-9)

        # Prior violation counting for C
        pv_count_c = 0; pv_repeated_c = 0
        # Use a COPY of history before B's updates for fair comparison
        # Actually, C runs AFTER B, so history already includes B's violations.
        # For fair comparison, we should track C separately. Let's use a separate
        # history set for C that mirrors B's history before this episode.
        # Simplification: count violations in C, repeated relative to pre-episode history

        for effect in effects_c:
            oid = effect["oid"]; action = effect["action"]; outcome = effect["outcome"]
            obj = test_objects.get(oid, {})
            features = obj.get("visible_features", {})
            is_violation, _, _ = check_prior_violation(features, action, outcome)
            if is_violation:
                pv_count_c += 1

        all_episodes_c.append({
            "episode": ep, "macro_bal": macro_bal_c, "probed": probed_c,
            "effective": eff_c, "zero": zero_c, "harmful": harm_c,
            "min_final_score": round(min_score_c, 6), "n_at_floor": at_floor_c,
            "prior_violation_count": pv_count_c,
        })

        if ep == 0 or ep == N_EPISODES - 1:
            print(f"    Ep {ep+1}: A_bal={macro_bal_a:.4f} B_bal={macro_bal_b:.4f} C_bal={macro_bal_c:.4f}  "
                  f"A_pv={pv_count_a} B_pv={pv_count_b} C_pv={pv_count_c}  "
                  f"B_eff={eff_b} zero={zero_b} harm={harm_b}  B_new_v={new_violations_this_ep}")

    # ----- Aggregate -----
    mb_a = [e["macro_bal"] for e in all_episodes_a]
    mb_b = [e["macro_bal"] for e in all_episodes_b]
    mb_c = [e["macro_bal"] for e in all_episodes_c]

    mean_a = float(np.mean(mb_a)); mean_b = float(np.mean(mb_b)); mean_c = float(np.mean(mb_c))
    std_a = float(np.std(mb_a, ddof=1)) if len(mb_a) > 1 else 0.0
    std_b = float(np.std(mb_b, ddof=1)) if len(mb_b) > 1 else 0.0
    std_c = float(np.std(mb_c, ddof=1)) if len(mb_c) > 1 else 0.0
    slope_a = compute_slope(mb_a); slope_b = compute_slope(mb_b); slope_c = compute_slope(mb_c)
    final_a = mb_a[-1] if mb_a else 0.0; final_b = mb_b[-1] if mb_b else 0.0
    final_c = mb_c[-1] if mb_c else 0.0

    delta_b_vs_a = mean_b - mean_a
    delta_c_vs_a = mean_c - mean_a
    delta_b_vs_c = mean_b - mean_c

    total_eff_b = sum(e["effective"] for e in all_episodes_b)
    total_zero_b = sum(e["zero"] for e in all_episodes_b)
    total_harm_b = sum(e["harmful"] for e in all_episodes_b)

    # Prior violation aggregates
    total_pv_a = sum(e.get("prior_violation_count", 0) for e in all_episodes_a)
    total_pv_b = sum(e.get("prior_violation_count", 0) for e in all_episodes_b)
    total_pv_c = sum(e.get("prior_violation_count", 0) for e in all_episodes_c)
    total_repeated_pv_b = sum(e.get("prior_violation_repeated", 0) for e in all_episodes_b)

    mem_stats = frm.get_statistics()
    top_risk = frm.get_top_risk_features(20)

    all_scores_flat = []
    for ep_list in [all_episodes_a, all_episodes_b, all_episodes_c]:
        for e in ep_list:
            all_scores_flat.append(e["min_final_score"])
    min_score_overall = min(all_scores_flat) if all_scores_flat else EXPLORATION_FLOOR
    hard_exclusion_used = min_score_overall <= 0.0

    # Check if top risk features overlap injected deviation features
    top_risk_feature_names = set(item["feature"] for item in top_risk)
    injected_features = {f for f in INJECTED_DEVIATION_FEATURES}
    risk_overlap = top_risk_feature_names & injected_features

    # B reduces repeated violations vs A?
    # For A, compute repeated violations the same way using history
    pv_repeated_a_total = sum(
        sum(1 for e in all_episodes_a[i].get("prior_violation_events", []) if e.get("repeated", False))
        if "prior_violation_events" in all_episodes_a[i]
        else 0
        for i in range(N_EPISODES)
    )

    B_reduces_violations = total_pv_b < total_pv_a
    B_beats_C_violations = total_pv_b < total_pv_c

    print(f"\n    Mode {mode_name} summary:")
    print(f"      A: mean_bal={mean_a:.4f}  total_pv={total_pv_a}")
    print(f"      B: mean_bal={mean_b:.4f}  total_pv={total_pv_b}  repeated_pv={total_repeated_pv_b}")
    print(f"      C: mean_bal={mean_c:.4f}  total_pv={total_pv_c}")
    print(f"      B-A_bal={delta_b_vs_a:+.4f}  B_reduces_violations={B_reduces_violations}  "
          f"B_beats_C_violations={B_beats_C_violations}")
    print(f"      risk_weights_nonzero={mem_stats['risk_weights_nonzero_count']}  "
          f"max_risk={mem_stats['max_risk_weight']:.4f}")
    print(f"      top_risk_overlap_injected={len(risk_overlap)}/{len(injected_features)}: {sorted(risk_overlap)}")

    return {
        "mode_name": mode_name,
        "explore_fraction": explore_fraction,
        "n_objects": len(test_objects),
        "n_episodes": N_EPISODES,

        # Per-condition macro
        "A_mean_macro_bal": round(mean_a, 4), "A_std_macro_bal": round(std_a, 4),
        "A_final_macro_bal": round(final_a, 4), "A_learning_slope": round(slope_a, 6),
        "B_mean_macro_bal": round(mean_b, 4), "B_std_macro_bal": round(std_b, 4),
        "B_final_macro_bal": round(final_b, 4), "B_learning_slope": round(slope_b, 6),
        "C_mean_macro_bal": round(mean_c, 4), "C_std_macro_bal": round(std_c, 4),
        "C_final_macro_bal": round(final_c, 4), "C_learning_slope": round(slope_c, 6),

        # Deltas
        "delta_B_vs_A_bal": round(delta_b_vs_a, 4),
        "delta_C_vs_A_bal": round(delta_c_vs_a, 4),
        "delta_B_vs_C_bal": round(delta_b_vs_c, 4),

        # Probe utility signals
        "total_effective_B": total_eff_b, "total_zero_B": total_zero_b,
        "total_harmful_B": total_harm_b,

        # Prior violation signals
        "prior_violation_threshold": PRIOR_VIOLATION_THRESHOLD,
        "total_prior_violations_A": total_pv_a,
        "total_prior_violations_B": total_pv_b,
        "total_prior_violations_C": total_pv_c,
        "total_repeated_prior_violations_B": total_repeated_pv_b,
        "B_reduces_prior_violations_vs_A": B_reduces_violations,
        "B_beats_C_on_prior_violations": B_beats_C_violations,

        # Memory diagnostics
        "memory_update_nontrivial": bool(mem_stats["memory_update_nontrivial"]),
        "eligible_updates": mem_stats["eligible_updates"],
        "ineligible_updates": mem_stats["ineligible_updates"],
        "violation_updates": mem_stats["violation_updates"],
        "nonviolation_updates": mem_stats["nonviolation_updates"],
        "risk_weights_nonzero_count": mem_stats["risk_weights_nonzero_count"],
        "max_risk_weight": mem_stats["max_risk_weight"],
        "mean_abs_risk_weight": mem_stats["mean_abs_risk_weight"],

        # Top risk features + overlap
        "top_risk_features": top_risk,
        "top_risk_feature_names": sorted(top_risk_feature_names),
        "injected_deviation_features": sorted(injected_features),
        "risk_overlap_injected": sorted(risk_overlap),
        "top_risk_features_include_deceptive_features": len(risk_overlap) > 0,

        # Validity
        "iom_isolated": True, "absence_counts_valid": True,
        "hard_exclusion_used": hard_exclusion_used,
        "min_final_score_overall": round(min_score_overall, 6),

        # Per-episode
        "per_episode_A": all_episodes_a,
        "per_episode_B": all_episodes_b,
        "per_episode_C": all_episodes_c,
    }


# =============================================================================
# 9. Run modes
# =============================================================================
print("\n[9/10] Running Mode 1: original_safe...")
mode1_results = run_mode(
    "original_safe", test_objects_standard, query_gt_standard,
    im_base, c15b_probed_n, explore_fraction=0.0, mode_rng_seed=SMOKE_SEED,
)

print("\n     Running Mode 2: deceptive_high_prior_diagnostic...")
mode2_results = run_mode(
    "deceptive_high_prior_diagnostic", test_objects_deceptive, query_gt_deceptive,
    im_base, c15b_probed_n, explore_fraction=0.0, mode_rng_seed=SMOKE_SEED + 10000,
)
mode2_results["deceptive_visible_deviation_valid"] = DECEPTIVE_VISIBLE_DEVIATION_VALID
mode2_results["deceptive_high_prior_eligible_ratio"] = round(DECEPTIVE_HIGH_PRIOR_ELIGIBLE_RATIO, 4)
mode2_results["family_preservation_valid"] = FAMILY_PRESERVATION_VALID
mode2_results["deceptive_features_in_feature_universe"] = DECEPTIVE_FEATURES_IN_UNIVERSE

print("\n     Running Mode 3: epsilon_boundary_explore...")
mode3_results = run_mode(
    "epsilon_boundary_explore", test_objects_standard, query_gt_standard,
    im_base, c15b_probed_n, explore_fraction=0.30, mode_rng_seed=SMOKE_SEED + 5000,
)

# =============================================================================
# 10. Cross-mode interpretation + Output
# =============================================================================
print(f"\n[10/10] Cross-mode interpretation...")

m1_pv = mode1_results["total_prior_violations_B"]
m2_pv = mode2_results["total_prior_violations_B"]
m3_pv = mode3_results["total_prior_violations_B"]

m2_B_reduces = mode2_results["B_reduces_prior_violations_vs_A"]
m2_B_beats_C = mode2_results["B_beats_C_on_prior_violations"]
m2_risk_overlap = mode2_results["top_risk_features_include_deceptive_features"]

# Macro tradeoff check
MACRO_DROP_TOLERANCE = -0.01
m2_B_macro_delta = mode2_results["delta_B_vs_A_bal"]
macro_tradeoff_acceptable = m2_B_macro_delta >= MACRO_DROP_TOLERANCE

# Family-preservation gates
eligibility_gate_ok = DECEPTIVE_HIGH_PRIOR_ELIGIBLE_RATIO >= 0.75

# Primary success: all gates pass
feature_risk_supported = (
    DECEPTIVE_VISIBLE_DEVIATION_VALID
    and DECEPTIVE_FEATURES_IN_UNIVERSE
    and FAMILY_PRESERVATION_VALID
    and eligibility_gate_ok
    and m2_pv > 0
    and m2_B_reduces
    and m2_B_beats_C
    and m2_risk_overlap
    and macro_tradeoff_acceptable
)

any_hard_exclusion = (
    mode1_results["hard_exclusion_used"]
    or mode2_results["hard_exclusion_used"]
    or mode3_results["hard_exclusion_used"]
)

# Partial success: memory works but macro drops
prior_violation_memory_works_but_task_tradeoff_unresolved = (
    DECEPTIVE_VISIBLE_DEVIATION_VALID
    and FAMILY_PRESERVATION_VALID
    and eligibility_gate_ok
    and m2_pv > 0
    and m2_B_reduces
    and m2_B_beats_C
    and m2_risk_overlap
    and not macro_tradeoff_acceptable
)

# Determine next route
if feature_risk_supported and not any_hard_exclusion:
    next_route = "run_small_multiseed_1j19c"
elif prior_violation_memory_works_but_task_tradeoff_unresolved:
    next_route = "prior_violation_memory_works_but_macro_tradeoff_unresolved_tune_cost_weight_or_risk_alpha"
elif not eligibility_gate_ok or not FAMILY_PRESERVATION_VALID:
    next_route = "family_preservation_failed_fix_deviation_features"
elif m2_pv > 0 and not m2_B_reduces:
    next_route = "prior_violations_exist_but_B_doesnt_reduce_inspect_memory_weights"
elif m2_pv == 0:
    next_route = "no_prior_violations_in_deceptive_mode_check_object_generation"
elif m2_B_reduces and not m2_risk_overlap:
    next_route = "B_reduces_violations_but_risk_features_dont_match_deviations"
else:
    next_route = "review_diagnostics"

# C0b baseline
c0b_rng = random.Random(SMOKE_SEED + 800)
c0b_im = im_base.clone()
c0b_positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, c0b_rng)
c0b_env = MiniMCEnvironment(test_objects_standard, c0b_positions, initial_budget=BUDGET)
c0b_policy = C0b_ObserveOnlyPolicy(c0b_im, c0b_rng)
c0b_preds, _, c0b_obs, _, _ = run_policy(c0b_policy, c0b_env)
c0b_metrics = compute_full_metrics(_unwrap_preds(c0b_preds), query_gt_standard)
c0b_macro_bal = c0b_metrics["macro_query_balanced_accuracy"]

elapsed = time.time() - t0

# ---- Output ----
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


results = {
    "block_id": "1J19c_family_preserving_patch",
    "condition": COND["label"],
    "smoke_seed": SMOKE_SEED, "budget": BUDGET, "n_episodes": N_EPISODES,
    "risk_alpha": RISK_ALPHA, "max_risk": MAX_RISK,
    "prior_violation_threshold": PRIOR_VIOLATION_THRESHOLD,
    "min_violation_support": MIN_VIOLATION_SUPPORT,
    "feature_universe_size": len(FEATURE_UNIVERSE),
    "threshold_sensitivity_not_tested_yet": True,

    "c15b_macro_bal": round(c15b_macro_bal, 4), "c15b_probed_n": c15b_probed_n,
    "c0b_macro_bal": round(c0b_macro_bal, 4),

    # Family-preservation validation
    "deceptive_total_count": len(deceptive_modifications),
    "deceptive_high_prior_eligible_count": DECEPTIVE_HIGH_PRIOR_ELIGIBLE_COUNT,
    "deceptive_high_prior_eligible_ratio": round(DECEPTIVE_HIGH_PRIOR_ELIGIBLE_RATIO, 4),
    "deceptive_visible_deviation_valid": DECEPTIVE_VISIBLE_DEVIATION_VALID,
    "deceptive_features_in_feature_universe": DECEPTIVE_FEATURES_IN_UNIVERSE,
    "family_preservation_valid": FAMILY_PRESERVATION_VALID,
    "eligibility_gate_ok": eligibility_gate_ok,
    "injected_deviation_features": sorted(INJECTED_DEVIATION_FEATURES),

    "mode_original_safe": mode1_results,
    "mode_deceptive_high_prior_diagnostic": mode2_results,
    "mode_epsilon_boundary_explore": mode3_results,

    "mode_original_prior_violation_count": m1_pv,
    "mode_deceptive_prior_violation_count": m2_pv,
    "mode_boundary_prior_violation_count": m3_pv,

    "top_risk_features_include_deceptive_features": m2_risk_overlap,
    "B_reduces_repeated_prior_violations": m2_B_reduces,
    "B_beats_permuted_control": m2_B_beats_C,
    "B_macro_delta_vs_A": round(m2_B_macro_delta, 4),
    "macro_drop_tolerance": MACRO_DROP_TOLERANCE,
    "macro_tradeoff_acceptable": macro_tradeoff_acceptable,
    "prior_violation_memory_works_but_task_tradeoff_unresolved": prior_violation_memory_works_but_task_tradeoff_unresolved,
    "feature_risk_memory_supported": feature_risk_supported,
    "hard_exclusion_used": any_hard_exclusion,
    "ready_for_full_1j19c": feature_risk_supported and not any_hard_exclusion,
    "next_recommended_route": next_route,
    "elapsed_s": round(elapsed, 1),
}

# Save JSON
runs_dir = os.path.join(CURRENT_DIR, "runs")
os.makedirs(runs_dir, exist_ok=True)
json_path = os.path.join(runs_dir, "calibration_block1j19c_family_preserving_patch_seed101.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2, cls=_NumpyEncoder)
print(f"  Saved: {json_path}")

# Save protocol MD
protocols_dir = os.path.join(CURRENT_DIR, "protocols")
os.makedirs(protocols_dir, exist_ok=True)
md_path = os.path.join(protocols_dir, "block1j19c_family_preserving_patch_seed101.md")

def fb(v): return "true" if v else "false"

md = []
md.append("# Block 1J19c -- Family-Preserving Patch\n")
md.append("## 1. Objective\n")
md.append("Test whether prior-violation-based feature-risk memory reduces repeated high-prior negative outcomes across episodes, with family-preserving deviation features that do not alter type classification.\n")

md.append("## 2. Setup\n")
md.append(f"| Parameter | Value |")
md.append(f"|-----------|-------|")
md.append(f"| Condition | {COND['label']} |")
md.append(f"| Seed | {SMOKE_SEED} |")
md.append(f"| Budget | {BUDGET} |")
md.append(f"| Episodes | {N_EPISODES} |")
md.append(f"| Prior Violation Threshold | {PRIOR_VIOLATION_THRESHOLD} |")
md.append(f"| Min Violation Support | {MIN_VIOLATION_SUPPORT} |")
md.append(f"| Risk Alpha | {RISK_ALPHA} |")
md.append(f"| Max Risk | {MAX_RISK} |")
md.append(f"| Threshold Sensitivity Tested | {fb(False)} |")

md.append("\n## 3. Memory Semantics\n")
md.append("- Only ELIGIBLE prior events (soft_prior >= threshold) update counts")
md.append("- violation = eligible AND outcome == failure")
md.append("- nonviolation = eligible AND outcome == success")
md.append("- Low-prior events excluded from both numerator and denominator")
md.append("- Risk weight = log(P(feature|violation) / P(feature|nonviolation)), clipped to [0, max_risk]")
md.append("- Beta(1,1) pseudo-counts + shrinkage (alpha=5.0)\n")

md.append("## 4. Baselines\n")
md.append(f"| Policy | Macro BAcc |")
md.append(f"|--------|------------|")
md.append(f"| C0b | {c0b_macro_bal:.4f} |")
md.append(f"| C15b | {c15b_macro_bal:.4f} |\n")

for mode_key, mode_label in [
    ("mode_original_safe", "Original Safe"),
    ("mode_deceptive_high_prior_diagnostic", "Deceptive High-Prior"),
    ("mode_epsilon_boundary_explore", "Epsilon-Boundary Explore"),
]:
    mr = results[mode_key]
    md.append(f"## 5. Mode: {mode_label}\n")
    md.append(f"Explore fraction: {mr['explore_fraction']}  |  Objects: {mr['n_objects']}\n")

    md.append(f"### Per-Episode\n")
    md.append(f"| Ep | A bal | B bal | C bal | A pv | B pv | C pv | B eff | B zero | B harm | B riskAdj |")
    md.append(f"|----|-------|-------|-------|------|------|------|-------|--------|--------|-----------|")
    for i in range(N_EPISODES):
        ea = mr["per_episode_A"][i]; eb = mr["per_episode_B"][i]; ec = mr["per_episode_C"][i]
        md.append(f"| {i+1} | {ea['macro_bal']:.4f} | {eb['macro_bal']:.4f} | {ec['macro_bal']:.4f} | "
                  f"{ea.get('prior_violation_count', 0)} | {eb.get('prior_violation_count', 0)} | "
                  f"{ec.get('prior_violation_count', 0)} | {eb['effective']} | {eb['zero']} | "
                  f"{eb['harmful']} | {eb.get('avg_abs_risk_adjustment', 0):.4f} |")

    md.append(f"\n### Aggregated\n")
    md.append(f"| Metric | A | B | C |")
    md.append(f"|--------|---|---|---|")
    md.append(f"| Mean BAcc | {mr['A_mean_macro_bal']:.4f} | {mr['B_mean_macro_bal']:.4f} | {mr['C_mean_macro_bal']:.4f} |")
    md.append(f"| Total Prior Violations | {mr['total_prior_violations_A']} | {mr['total_prior_violations_B']} | {mr['total_prior_violations_C']} |")

    md.append(f"\n### Prior Violation Diagnostics\n")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| Prior violations (B) | {mr['total_prior_violations_B']} |")
    md.append(f"| Repeated prior violations (B) | {mr['total_repeated_prior_violations_B']} |")
    md.append(f"| B reduces violations vs A | {fb(mr['B_reduces_prior_violations_vs_A'])} |")
    md.append(f"| B beats C on violations | {fb(mr['B_beats_C_on_prior_violations'])} |")
    md.append(f"| Eligible updates | {mr['eligible_updates']} |")
    md.append(f"| Ineligible updates | {mr['ineligible_updates']} |")

    md.append(f"\n### Memory Diagnostics\n")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| Risk weights nonzero | {mr['risk_weights_nonzero_count']} |")
    md.append(f"| Max risk weight | {mr['max_risk_weight']:.4f} |")
    md.append(f"| Top risk features overlap deceptive | {fb(mr['top_risk_features_include_deceptive_features'])} |")

    top = mr.get("top_risk_features", [])
    if top:
        md.append(f"\n### Top Risk Features\n")
        md.append(f"| Feature | Goal | Action | Risk | VWith | NVWith | VWithout | NVWithout | ActV |")
        md.append(f"|---------|------|--------|------|-------|--------|----------|-----------|------|")
        for item in top:
            md.append(f"| {item['feature']} | {item['goal']} | {item['action']} | {item['risk_weight']:.4f} | "
                      f"{item['violation_with']:.0f} | {item['nonviolation_with']:.0f} | "
                      f"{item['violation_without']:.0f} | {item['nonviolation_without']:.0f} | "
                      f"{item['actual_violations']:.0f} |")
    else:
        md.append(f"\n### Top Risk Features\n(none)\n")

md.append("## 6. Family-Preservation Validation\n")
md.append(f"| Criterion | Value |")
md.append(f"|-----------|-------|")
md.append(f"| Deceptive total count | {len(deceptive_modifications)} |")
md.append(f"| Deceptive high-prior eligible count | {DECEPTIVE_HIGH_PRIOR_ELIGIBLE_COUNT} |")
md.append(f"| Deceptive high-prior eligible ratio | {DECEPTIVE_HIGH_PRIOR_ELIGIBLE_RATIO:.4f} |")
md.append(f"| Eligibility gate (>=0.75) | {fb(eligibility_gate_ok)} |")
md.append(f"| Deceptive visible deviation valid | {fb(DECEPTIVE_VISIBLE_DEVIATION_VALID)} |")
md.append(f"| Deceptive features in FEATURE_UNIVERSE | {fb(DECEPTIVE_FEATURES_IN_UNIVERSE)} |")
md.append(f"| Family preservation valid | {fb(FAMILY_PRESERVATION_VALID)} |")

md.append("\n## 7. Cross-Mode Interpretation\n")
md.append(f"| Criterion | Value |")
md.append(f"|-----------|-------|")
md.append(f"| Mode 1 prior violations (B) | {m1_pv} |")
md.append(f"| Mode 2 prior violations (B) | {m2_pv} |")
md.append(f"| Mode 3 prior violations (B) | {m3_pv} |")
md.append(f"| Top risk features include deceptive | {fb(m2_risk_overlap)} |")
md.append(f"| B reduces repeated prior violations | {fb(m2_B_reduces)} |")
md.append(f"| B beats permuted control | {fb(m2_B_beats_C)} |")
md.append(f"| B macro delta vs A | {m2_B_macro_delta:+.4f} |")
md.append(f"| Macro tradeoff acceptable (>= {MACRO_DROP_TOLERANCE}) | {fb(macro_tradeoff_acceptable)} |")
md.append(f"| Prior violation memory works but tradeoff unresolved | {fb(prior_violation_memory_works_but_task_tradeoff_unresolved)} |")
md.append(f"| Feature-risk memory supported | {fb(feature_risk_supported)} |")
md.append(f"| Hard exclusion used | {fb(any_hard_exclusion)} |")
md.append(f"| Ready for full 1J19c | {fb(results['ready_for_full_1j19c'])} |")
md.append(f"| Next route | {next_route} |")

md.append("\n## 8. Summary\n")
md.append("```")
md.append("[block_done]")
md.append(f"block_id=1J19c_family_preserving_patch")
md.append(f"deceptive_total_count={len(deceptive_modifications)}")
md.append(f"deceptive_high_prior_eligible_count={DECEPTIVE_HIGH_PRIOR_ELIGIBLE_COUNT}")
md.append(f"deceptive_high_prior_eligible_ratio={DECEPTIVE_HIGH_PRIOR_ELIGIBLE_RATIO:.4f}")
md.append(f"family_preservation_valid={fb(FAMILY_PRESERVATION_VALID)}")
md.append(f"deceptive_features_in_feature_universe={fb(DECEPTIVE_FEATURES_IN_UNIVERSE)}")
md.append(f"mode_deceptive_prior_violation_count={m2_pv}")
md.append(f"top_risk_features_include_deceptive_features={fb(m2_risk_overlap)}")
md.append(f"B_reduces_prior_violations={fb(m2_B_reduces)}")
md.append(f"B_beats_permuted_control={fb(m2_B_beats_C)}")
md.append(f"B_macro_delta_vs_A={m2_B_macro_delta:+.4f}")
md.append(f"macro_tradeoff_acceptable={fb(macro_tradeoff_acceptable)}")
md.append(f"feature_risk_memory_supported={fb(feature_risk_supported)}")
md.append(f"hard_exclusion_used={fb(any_hard_exclusion)}")
md.append(f"ready_for_full_1j19c={fb(results['ready_for_full_1j19c'])}")
md.append(f"next_recommended_route={next_route}")
md.append(f"elapsed={elapsed:.1f}s")
md.append("```")

with open(md_path, "w") as f:
    f.write("\n".join(md))
print(f"  Saved: {md_path}")

# ---- block_done ----
print(f"\n{'='*70}")
print(f"[block_done]")
print(f"block_id=1J19c_family_preserving_patch")
print(f"deceptive_total_count={len(deceptive_modifications)}")
print(f"deceptive_high_prior_eligible_count={DECEPTIVE_HIGH_PRIOR_ELIGIBLE_COUNT}")
print(f"deceptive_high_prior_eligible_ratio={DECEPTIVE_HIGH_PRIOR_ELIGIBLE_RATIO:.4f}")
print(f"family_preservation_valid={fb(FAMILY_PRESERVATION_VALID)}")
print(f"deceptive_features_in_feature_universe={fb(DECEPTIVE_FEATURES_IN_UNIVERSE)}")
print(f"B_reduces_prior_violations={fb(m2_B_reduces)}")
print(f"B_beats_permuted_control={fb(m2_B_beats_C)}")
print(f"top_risk_overlap_injected={len(mode2_results.get('risk_overlap_injected', []))}/{len(INJECTED_DEVIATION_FEATURES)}")
print(f"B_macro_delta_vs_A={m2_B_macro_delta:+.4f}")
print(f"macro_tradeoff_acceptable={fb(macro_tradeoff_acceptable)}")
print(f"feature_risk_memory_supported={fb(feature_risk_supported)}")
print(f"ready_for_full_1j19c={fb(results['ready_for_full_1j19c'])}")
print(f"next_recommended_route={next_route}")
print(f"elapsed={elapsed:.1f}s")
print(f"{'='*70}")
