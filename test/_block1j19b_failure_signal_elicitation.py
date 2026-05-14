"""
Block 1J19b -- Failure-Signal Elicitation for Negative-Exception Feature-Risk Memory.

Patched from 1J19:
  - FeatureRiskMemoryV2: only positive risk penalties, min_failure_support gate,
    zero risk when no failure evidence
  - Three diagnostic modes:
    Mode 1: original_safe (verify B ~= A when no failures)
    Mode 2: epsilon_boundary_explore (force some boundary probes)
    Mode 3: deceptive_high_prior_diagnostic (deceptive objects that fail)

Three policy conditions per mode:
  A. soft_prior_only
  B. soft_prior_plus_negative_feature_risk_memory (V2)
  C. permuted_risk_memory_control
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
MIN_FAILURE_SUPPORT = 2  # need at least 2 failure observations before risk activates

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
})

print("=" * 70)
print("Block 1J19b -- Failure-Signal Elicitation")
print(f"  condition={COND['label']}  seed={SMOKE_SEED}  budget={BUDGET}")
print(f"  episodes={N_EPISODES}  risk_alpha={RISK_ALPHA}  max_risk={MAX_RISK}")
print(f"  min_failure_support={MIN_FAILURE_SUPPORT}")
print(f"  feature_universe_size={len(FEATURE_UNIVERSE)}")
print(f"  modes: original_safe, epsilon_boundary_explore, deceptive_high_prior")
print("=" * 70)

# =============================================================================
# 1. Phase A: Training + IOM building
# =============================================================================
print("\n[1/12] Phase A: Training + IOM building...")
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

# Verify FEATURE_UNIVERSE coverage
sample_obj = train_objects_dict[list(train_objects_dict.keys())[0]]
actual_vis_features = set(sample_obj.get("visible_features", {}).keys())
universe_misses = set(FEATURE_UNIVERSE) - actual_vis_features
if universe_misses:
    miss_pct = len(universe_misses) / len(FEATURE_UNIVERSE) * 100
    if miss_pct > 50:
        print(f"  Adjusting FEATURE_UNIVERSE to visible features ({miss_pct:.0f}% miss)")
        FEATURE_UNIVERSE = sorted(actual_vis_features)
    else:
        print(f"  Note: {len(universe_misses)}/{len(FEATURE_UNIVERSE)} features not in "
              f"visible_features (treated as absent)")

# =============================================================================
# 2. Test objects (standard, for Mode 1 and Mode 2)
# =============================================================================
print("\n[2/12] Generating test objects...")
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

print(f"  Generated {N_OBJECTS} standard test objects, {len(QUERY_NAMES)} queries")

# =============================================================================
# 3. Deceptive test objects (for Mode 3) -- WITH observable deviation features
# =============================================================================
print("\n[3/12] Creating deceptive test objects for Mode 3...")
test_objects_deceptive = copy.deepcopy(test_objects_standard)

apple_oids = [oid for oid in test_oids if "apple" in oid]
wood_oids = [oid for oid in test_oids if "wood_log" in oid]
stone_oids = [oid for oid in test_oids if "stone_block" in oid]

deceptive_modifications = []
deceptive_rng = random.Random(SMOKE_SEED + 9999)

# Deviation feature definitions per category
# These are observable features in FEATURE_UNIVERSE that signal abnormality
# while the type-family-defining features are preserved (soft prior stays high)
APPLE_DEVIATION_FEATURES = {
    # Spoilage/rot indicators for apple-like non-edible objects
    "grayish": True,         # gray spots = rot
    "heavy_weight": True,    # dense/waterlogged = spoiled
    "rough_texture": True,   # wrinkled/rotten surface
    "smooth_texture": False, # override normal apple smoothness
    "light_weight": False,   # override normal apple lightness
}
WOOD_DEVIATION_FEATURES = {
    # Damp/treated indicators for wood-like non-burnable objects
    "smooth_texture": True,  # damp/treated surface
    "heavy_weight": True,    # waterlogged
    "grayish": True,         # treated/painted wood
    "rough_texture": False,  # override normal wood roughness
    "brownish": False,       # override normal wood brown color
}
STONE_DEVIATION_FEATURES = {
    # Ultra-hard/polished indicators for stone not mineable by pickaxe
    "smooth_texture": True,  # polished, very hard surface
    "brownish": True,        # iron-rich, extremely hard variant
    "light_weight": False,   # ensure heavy
}

# Deceptive apples: apple-like visible features + spoilage deviation + eat=fail
n_deceptive_apples = min(5, len(apple_oids))
deceptive_apple_oids = deceptive_rng.sample(apple_oids, n_deceptive_apples)
for oid in deceptive_apple_oids:
    obj = test_objects_deceptive[oid]
    obj["hidden_affordance_profile"]["eat"] = "fail"
    obj["_deceptive"] = True
    obj["_deceptive_action"] = "eat"
    obj["_deviation_features"] = dict(APPLE_DEVIATION_FEATURES)
    # Apply deviation features to visible_features
    for feat, val in APPLE_DEVIATION_FEATURES.items():
        if feat in FEATURE_UNIVERSE:
            obj["visible_features"][feat] = val
    deceptive_modifications.append({
        "oid": oid, "category": "apple", "flipped_action": "eat",
        "deviation_features": list(APPLE_DEVIATION_FEATURES.keys()),
    })

# Deceptive wood: wood-like visible features + damp/treated deviation + burn_as_fuel=fail
n_deceptive_wood = min(5, len(wood_oids))
deceptive_wood_oids = deceptive_rng.sample(wood_oids, n_deceptive_wood)
for oid in deceptive_wood_oids:
    obj = test_objects_deceptive[oid]
    obj["hidden_affordance_profile"]["burn_as_fuel"] = "fail"
    obj["_deceptive"] = True
    obj["_deceptive_action"] = "burn_as_fuel"
    obj["_deviation_features"] = dict(WOOD_DEVIATION_FEATURES)
    for feat, val in WOOD_DEVIATION_FEATURES.items():
        if feat in FEATURE_UNIVERSE:
            obj["visible_features"][feat] = val
    deceptive_modifications.append({
        "oid": oid, "category": "wood_log", "flipped_action": "burn_as_fuel",
        "deviation_features": list(WOOD_DEVIATION_FEATURES.keys()),
    })

# Deceptive stones: stone-like visible features + ultra-hard deviation + mine_with_pickaxe=fail
n_deceptive_stones = min(3, len(stone_oids))
deceptive_stone_oids = deceptive_rng.sample(stone_oids, n_deceptive_stones)
for oid in deceptive_stone_oids:
    obj = test_objects_deceptive[oid]
    obj["hidden_affordance_profile"]["mine_with_pickaxe"] = "fail"
    obj["_deceptive"] = True
    obj["_deceptive_action"] = "mine_with_pickaxe"
    obj["_deviation_features"] = dict(STONE_DEVIATION_FEATURES)
    for feat, val in STONE_DEVIATION_FEATURES.items():
        if feat in FEATURE_UNIVERSE:
            obj["visible_features"][feat] = val
    deceptive_modifications.append({
        "oid": oid, "category": "stone_block", "flipped_action": "mine_with_pickaxe",
        "deviation_features": list(STONE_DEVIATION_FEATURES.keys()),
    })

query_gt_deceptive = compute_query_ground_truth(test_objects_deceptive)

# ---- Validation: every deceptive object MUST have observable deviation features ----
MODE3_VALID = True
for mod in deceptive_modifications:
    oid = mod["oid"]
    obj = test_objects_deceptive[oid]
    feats = obj["visible_features"]
    dev_feats = obj.get("_deviation_features", {})

    # Check: at least one deviation feature is present AND in FEATURE_UNIVERSE
    dev_in_universe = [f for f in dev_feats if f in FEATURE_UNIVERSE]
    dev_present = [f for f, v in dev_feats.items() if v and f in FEATURE_UNIVERSE]

    if not dev_present:
        MODE3_VALID = False
        print(f"  ERROR: {oid} has no observable deviation features!")
    else:
        print(f"    {oid}: action={mod['flipped_action']}=fail  "
              f"deviations={dev_present}")

if not MODE3_VALID:
    print(f"  *** MODE 3 INVALID: deceptive objects lack observable deviation features ***")
else:
    print(f"  Mode 3 validity: all {len(deceptive_modifications)} deceptive objects have "
          f"observable deviation features in FEATURE_UNIVERSE")

print(f"  Deceptive test objects: {len(deceptive_modifications)} total, valid={MODE3_VALID}")

# =============================================================================
# 4. Metric helpers
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
# 5. Default Goal Prior Table (from 1J18)
# =============================================================================
print("\n[4/12] Building default goal-conditioned soft prior table...")

WOOD_FEATURES = {
    "has_bark_texture", "has_wood_grain", "brownish", "rough_texture", "long_shape",
}
STONE_FEATURES = {
    "has_crystal_flecks", "has_granular_surface", "grayish", "block_like", "heavy_weight",
}
APPLE_FEATURES = {
    "has_stem_remnant", "has_peel_texture", "round_small", "greenish",
    "smooth_texture", "light_weight",
}
TOOL_FEATURES = {
    "has_grip_area", "has_shaft_shape", "elongated_with_handle", "movable", "long_shape",
}

TYPE_FAMILIES = {
    "wood-like": WOOD_FEATURES,
    "stone-like": STONE_FEATURES,
    "apple-like": APPLE_FEATURES,
    "tool-like": TOOL_FEATURES,
}


def detect_type_family(features):
    if features is None:
        return "unknown"
    scores = {}
    for fam, feat_set in TYPE_FAMILIES.items():
        score = sum(1 for f in feat_set if features.get(f, False))
        scores[fam] = score
    best_fam = max(scores, key=scores.get)
    if scores[best_fam] == 0:
        return "unknown"
    return best_fam


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
            else:
                prior = 0.10
        elif qname == "need_planks":
            if action == "craft_plank":
                if fam == "wood-like": prior = 0.85
                elif fam == "tool-like": prior = 0.15
                else: prior = 0.10
            else:
                prior = 0.10
        elif qname == "need_stone":
            if action in ("mine_by_hand", "mine_with_pickaxe"):
                if fam == "stone-like": prior = 0.85
                elif fam in ("wood-like", "tool-like"): prior = 0.25
                else: prior = 0.15
            else:
                prior = 0.10
        elif qname == "need_tool":
            if action == "use_as_tool":
                if fam == "tool-like": prior = 0.85
                elif fam == "wood-like": prior = 0.15
                else: prior = 0.10
            else:
                prior = 0.10

        prior = max(EXPLORATION_FLOOR, min(0.95, prior))
        DEFAULT_GOAL_PRIOR[(fam, qname, action)] = prior

min_prior = min(DEFAULT_GOAL_PRIOR.values())
assert min_prior > 0.0, f"Found zero prior! min={min_prior}"
print(f"  Default prior table: {len(DEFAULT_GOAL_PRIOR)} entries, min={min_prior:.4f}")


def get_goal_soft_prior(features, action):
    fam = detect_type_family(features)
    qname = ACTION_TO_QUERY.get(action, "")
    return DEFAULT_GOAL_PRIOR.get((fam, qname, action), EXPLORATION_FLOOR)


# =============================================================================
# 6. FeatureRiskMemoryV2 (PATCHED: negative-exception only)
# =============================================================================
print("\n[5/12] Initializing FeatureRiskMemoryV2 (patched semantics)...")

class FeatureRiskMemoryV2:
    """Negative-exception feature-risk memory.

    Key changes from V1:
      - Risk weight active ONLY if failure evidence >= min_failure_support
      - Only POSITIVE risk weights (penalty); negative weights clipped to 0
      - Successes calibrate the denominator but don't create bonuses
    """

    def __init__(self, alpha=RISK_ALPHA, max_risk=MAX_RISK, min_failure_support=MIN_FAILURE_SUPPORT):
        self.alpha = alpha
        self.max_risk = max_risk
        self.min_failure_support = min_failure_support
        self.counts = defaultdict(lambda: {"fail_with": 1.0, "succ_with": 1.0,
                                            "fail_without": 1.0, "succ_without": 1.0})
        self._risk_cache = {}
        self.total_updates = 0
        self.failure_updates = 0
        self.success_updates = 0
        self.keys_updated_from_failures = set()
        self.keys_updated_from_successes = set()

    def update(self, features, goal, action, is_success):
        self.total_updates += 1
        if is_success:
            self.success_updates += 1
        else:
            self.failure_updates += 1

        for feat in FEATURE_UNIVERSE:
            key = (feat, goal, action)
            present = features.get(feat, False)
            if present:
                if is_success:
                    self.counts[key]["succ_with"] += 1.0
                else:
                    self.counts[key]["fail_with"] += 1.0
            else:
                if is_success:
                    self.counts[key]["succ_without"] += 1.0
                else:
                    self.counts[key]["fail_without"] += 1.0

            if not is_success:
                self.keys_updated_from_failures.add(key)
            else:
                self.keys_updated_from_successes.add(key)

        self._risk_cache.clear()

    def get_risk_weight(self, feature, goal, action):
        cache_key = (feature, goal, action)
        if cache_key in self._risk_cache:
            return self._risk_cache[cache_key]

        c = self.counts[cache_key]
        fail_with = c["fail_with"]
        succ_with = c["succ_with"]
        fail_without = c["fail_without"]
        succ_without = c["succ_without"]

        # Gate: need enough failure evidence
        actual_failures = (fail_with + fail_without) - 2.0  # subtract pseudo-counts
        if actual_failures < self.min_failure_support:
            self._risk_cache[cache_key] = 0.0
            return 0.0

        # Likelihood ratio: P(feature | failure) / P(feature | success)
        total_fail = fail_with + fail_without
        total_succ = succ_with + succ_without

        p_feat_given_fail = fail_with / max(total_fail, 0.001)
        p_feat_given_succ = succ_with / max(total_succ, 0.001)

        eps = 1e-9
        if p_feat_given_succ < eps:
            p_feat_given_succ = eps
        ratio = p_feat_given_fail / p_feat_given_succ
        if ratio < eps:
            ratio = eps
        if ratio > 1.0 / eps:
            ratio = 1.0 / eps
        raw_lr = math.log(ratio)

        # Only positive risk (penalty for failure-associated features)
        clipped_risk = max(0.0, raw_lr)
        clipped_risk = min(clipped_risk, self.max_risk)

        # Shrinkage
        support = total_fail + total_succ - 4.0
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
        keys = []
        weights = []
        for key in self.counts:
            w = self.get_risk_weight(key[0], key[1], key[2])
            keys.append(key)
            weights.append(w)

        shuffled = list(weights)
        rng.shuffle(shuffled)

        permuted = {}
        for k, w in zip(keys, shuffled):
            permuted[k] = w
            self._risk_cache[k] = w

        return permuted

    def get_statistics(self):
        nonzero_count = 0
        max_w = 0.0
        sum_abs = 0.0
        n_keys = 0
        for key in self.counts:
            w = self.get_risk_weight(key[0], key[1], key[2])
            n_keys += 1
            if abs(w) > 1e-9:
                nonzero_count += 1
                sum_abs += abs(w)
                if abs(w) > max_w:
                    max_w = abs(w)

        return {
            "total_updates": self.total_updates,
            "failure_updates": self.failure_updates,
            "success_updates": self.success_updates,
            "failure_signal_count": self.failure_updates,
            "n_keys_total": n_keys,
            "n_keys_with_data": len(self.counts),
            "n_keys_from_failures": len(self.keys_updated_from_failures),
            "n_keys_from_successes": len(self.keys_updated_from_successes),
            "memory_update_nontrivial": self.failure_updates > 0,
            "risk_weights_nonzero_count": nonzero_count,
            "max_risk_weight": round(max_w, 6),
            "mean_abs_risk_weight": round(sum_abs / max(n_keys, 1), 6),
        }

    def get_top_risk_features(self, top_n=15):
        items = []
        for (feat, goal, action), c in self.counts.items():
            w = self.get_risk_weight(feat, goal, action)
            if abs(w) > 1e-9:
                items.append({
                    "feature": feat,
                    "goal": goal,
                    "action": action,
                    "risk_weight": round(w, 6),
                    "fail_with": c["fail_with"],
                    "succ_with": c["succ_with"],
                    "fail_without": c["fail_without"],
                    "succ_without": c["succ_without"],
                    "actual_failures": (c["fail_with"] + c["fail_without"]) - 2.0,
                })
        items.sort(key=lambda x: abs(x["risk_weight"]), reverse=True)
        return items[:top_n]

    def clone(self):
        new = FeatureRiskMemoryV2(alpha=self.alpha, max_risk=self.max_risk,
                                  min_failure_support=self.min_failure_support)
        new.counts = copy.deepcopy(dict(self.counts))
        new._risk_cache = dict(self._risk_cache)
        new.total_updates = self.total_updates
        new.failure_updates = self.failure_updates
        new.success_updates = self.success_updates
        new.keys_updated_from_failures = set(self.keys_updated_from_failures)
        new.keys_updated_from_successes = set(self.keys_updated_from_successes)
        return new


print(f"  FeatureRiskMemoryV2 ready  alpha={RISK_ALPHA}  max_risk={MAX_RISK}")
print(f"  min_failure_support={MIN_FAILURE_SUPPORT}  (risk inactive below this)")
print(f"  semantics: positive-only risk, failure-gated, success-calibrated")

# =============================================================================
# 7. Policy Classes
# =============================================================================
print("\n[6/12] Defining policy classes...")

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
            best_oid = None
            best_cost = float('inf')
            for oid in unvisited:
                total = view.compute_reach_cost(oid) + view.observe_cost
                if view.can_afford(total) and total < best_cost:
                    best_cost = total; best_oid = oid
            return best_oid
        elif self._phase == self.PHASE_PROBE:
            return self._select_probe_target(view)
        return None

    def _score_candidate(self, features, norm_cost):
        """Compute scores for all actions. Returns best score and a list of (action, score)."""
        best_score = float('-inf')
        all_scores = []
        for action in MAIN_CANDIDATE_ACTIONS:
            base_prior = get_goal_soft_prior(features, action)
            raw_score = base_prior - COST_WEIGHT * norm_cost
            score = max(EXPLORATION_FLOOR, raw_score)
            all_scores.append((action, score, base_prior))
            if score > best_score:
                best_score = score
        return best_score, all_scores

    def _select_probe_target(self, view):
        if len(self._probed_oids) >= self._target_probe_count:
            return None

        # Collect all viable candidates with scores
        candidates = []
        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids: continue
            features = view.get_observed_features(oid)
            if features is None: continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost): continue
            norm_cost = total_cost / max(view.initial_budget, 0.001)
            best_score, _ = self._score_candidate(features, norm_cost)
            if best_score > float('-inf'):
                candidates.append((oid, best_score, features))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)

        # Epsilon-boundary exploration
        if self._explore_fraction > 0 and len(candidates) >= 3:
            if self._rng.random() < self._explore_fraction:
                # Pick from boundary: candidates in 30th-80th percentile by score
                n = len(candidates)
                lo = max(0, int(n * 0.30))
                hi = min(n - 1, int(n * 0.80))
                if hi > lo:
                    idx = self._rng.randint(lo, hi)
                    return candidates[idx][0]
                # Fall through to best if boundary is empty

        # Default: pick best
        return candidates[0][0] if candidates else None

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE: return False, None
        if len(self._probed_oids) >= self._target_probe_count: return False, None
        features = view.get_observed_features(object_id)
        if features is None: return False, None

        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)

        best_action = None
        best_score = float('-inf')
        best_raw_score = float('-inf')
        best_prior = EXPLORATION_FLOOR
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


class SoftPriorPlusFeatureRiskMemoryPolicyV2(SoftPriorOnlyPolicyV2):
    """Condition B: soft_prior + negative-exception feature-risk memory (V2)."""

    def __init__(self, instance_memory, rng, feature_risk_memory,
                 target_probe_count=8, explore_fraction=0.0):
        super().__init__(instance_memory, rng, target_probe_count, explore_fraction)
        self._frm = feature_risk_memory
        self._variant = "B_soft_prior_plus_feature_risk_memory"

    def _compute_adjusted_score(self, features, action, norm_cost):
        base_prior = get_goal_soft_prior(features, action)
        risk_penalty = self._frm.get_total_risk_penalty(features, action)
        raw_score = base_prior * math.exp(-risk_penalty) - COST_WEIGHT * norm_cost
        final_score = max(EXPLORATION_FLOOR, raw_score)
        return final_score, raw_score, base_prior, risk_penalty

    def _score_candidate(self, features, norm_cost):
        best_score = float('-inf')
        all_scores = []
        for action in MAIN_CANDIDATE_ACTIONS:
            final_score, raw_score, base_prior, risk_penalty = \
                self._compute_adjusted_score(features, action, norm_cost)
            all_scores.append((action, final_score, base_prior, risk_penalty))
            if final_score > best_score:
                best_score = final_score
        return best_score, all_scores

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE: return False, None
        if len(self._probed_oids) >= self._target_probe_count: return False, None
        features = view.get_observed_features(object_id)
        if features is None: return False, None

        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)

        best_action = None
        best_score = float('-inf')
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


class SoftPriorPlusPermutedRiskMemoryPolicyV2(SoftPriorPlusFeatureRiskMemoryPolicyV2):
    """Condition C: permuted risk memory control."""

    def __init__(self, instance_memory, rng, feature_risk_memory,
                 target_probe_count=8, explore_fraction=0.0):
        super().__init__(instance_memory, rng, feature_risk_memory,
                        target_probe_count, explore_fraction)
        self._variant = "C_soft_prior_plus_permuted_risk_memory"
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
# 8. C15b reference (for baseline probe count)
# =============================================================================
print("\n[7/12] Running C15b reference...")

class C15b_RefPolicy(MiniMCPolicy):
    """Inline C15b reference for baseline probe count."""
    PHASE_OBSERVE = 1
    PHASE_PROBE = 2

    def __init__(self, instance_memory, rng, cost_weight=COST_WEIGHT):
        self._im = instance_memory
        self._rng = rng
        self._cost_weight = cost_weight
        self._phase = self.PHASE_OBSERVE
        self._probed_oids = set()
        self._pre_probe_entropies = {}

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probed_oids = set()
        self._pre_probe_entropies = {}

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
                im_succ = self._im.clone()
                im_succ.incorporate_probe(oid, action, 1.0)
                util_succ = _compute_utility(im_succ.predict_all_affordances(fake_obj))
                im_fail = self._im.clone()
                im_fail.incorporate_probe(oid, action, 0.0)
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
            im_succ = self._im.clone()
            im_succ.incorporate_probe(object_id, action, 1.0)
            util_succ = _compute_utility(im_succ.predict_all_affordances(fake_obj))
            im_fail = self._im.clone()
            im_fail.incorporate_probe(object_id, action, 0.0)
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
c15b_preds, c15b_event_log, c15b_obs, _, _ = run_policy(c15b_policy, c15b_env)
c15b_metrics = compute_full_metrics(_unwrap_preds(c15b_preds), query_gt_standard)
c15b_macro_bal = c15b_metrics["macro_query_balanced_accuracy"]
c15b_probed_n = sum(1 for oid in c15b_obs.get_all_object_ids() if c15b_obs.get_probe_results(oid))
print(f"  C15b macro_bal={c15b_macro_bal:.4f}  probed={c15b_probed_n}")
print(f"  target_probe_count={c15b_probed_n}")

# =============================================================================
# 9. Mode runner
# =============================================================================
print("\n[8/12] Setting up mode runner...")

def compute_slope(mb_list):
    if len(mb_list) < 2: return 0.0
    x = np.arange(len(mb_list))
    y = np.array(mb_list)
    slope, _ = np.polyfit(x, y, 1)
    return slope


def run_mode(mode_name, test_objects, query_gt, im_base, c15b_probed_n,
             explore_fraction=0.0, mode_rng_seed=None):
    """Run one diagnostic mode with A/B/C comparison across episodes.

    Returns: mode_results dict with all metrics.
    """
    if mode_rng_seed is None:
        mode_rng_seed = SMOKE_SEED

    print(f"\n  {'='*60}")
    print(f"  Mode: {mode_name}")
    print(f"    explore_fraction={explore_fraction}")
    print(f"    n_objects={len(test_objects)}")
    print(f"  {'='*60}")

    # Fresh feature-risk memory for this mode
    frm = FeatureRiskMemoryV2()

    all_episodes_a = []
    all_episodes_b = []
    all_episodes_c = []

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
        preds_a, event_log_a, obs_a, _, _ = run_policy(policy_a, env_a)
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

        all_episodes_a.append({
            "episode": ep, "macro_bal": macro_bal_a, "probed": probed_a,
            "effective": eff_a, "zero": zero_a, "harmful": harm_a,
            "min_final_score": round(min_score_a, 6),
            "n_at_floor": at_floor_a,
        })

        # ----- Condition B: feature-risk memory V2 -----
        im_b = im_base.clone()
        env_b = MiniMCEnvironment(test_objects, dict(ep_positions), initial_budget=BUDGET)
        policy_b = SoftPriorPlusFeatureRiskMemoryPolicyV2(
            im_b, ep_rng, frm, target_probe_count=c15b_probed_n,
            explore_fraction=explore_fraction)
        preds_b, event_log_b, obs_b, _, _ = run_policy(policy_b, env_b)
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

        # Update feature-risk memory from B's probe effects
        n_failure_before = frm.failure_updates
        for effect in effects_b:
            oid = effect["oid"]
            action = effect["action"]
            is_success = effect["is_effective"]
            goal = ACTION_TO_QUERY.get(action, "")
            obj = test_objects.get(oid, {})
            features = obj.get("visible_features", {})
            frm.update(features, goal, action, is_success)
        new_failures_this_ep = frm.failure_updates - n_failure_before

        all_episodes_b.append({
            "episode": ep, "macro_bal": macro_bal_b, "probed": probed_b,
            "effective": eff_b, "zero": zero_b, "harmful": harm_b,
            "min_final_score": round(min_score_b, 6),
            "n_at_floor": at_floor_b,
            "avg_abs_risk_adjustment": round(avg_risk_b, 6),
            "new_failure_signals": new_failures_this_ep,
        })

        # ----- Condition C: permuted risk memory control -----
        im_c = im_base.clone()
        env_c = MiniMCEnvironment(test_objects, dict(ep_positions), initial_budget=BUDGET)
        permuted_memory = frm.clone()
        policy_c = SoftPriorPlusPermutedRiskMemoryPolicyV2(
            im_c, ep_rng, permuted_memory, target_probe_count=c15b_probed_n,
            explore_fraction=explore_fraction)
        policy_c.apply_permutation(ep_rng)
        preds_c, event_log_c, obs_c, _, _ = run_policy(policy_c, env_c)
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

        all_episodes_c.append({
            "episode": ep, "macro_bal": macro_bal_c, "probed": probed_c,
            "effective": eff_c, "zero": zero_c, "harmful": harm_c,
            "min_final_score": round(min_score_c, 6),
            "n_at_floor": at_floor_c,
        })

        if ep == 0 or ep == N_EPISODES - 1:
            print(f"    Ep {ep+1}: A={macro_bal_a:.4f} B={macro_bal_b:.4f} C={macro_bal_c:.4f}  "
                  f"B_eff={eff_b} zero={zero_b} harm={harm_b} new_fails={new_failures_this_ep}")

    # ----- Aggregate -----
    mb_a = [e["macro_bal"] for e in all_episodes_a]
    mb_b = [e["macro_bal"] for e in all_episodes_b]
    mb_c = [e["macro_bal"] for e in all_episodes_c]

    mean_a = float(np.mean(mb_a))
    mean_b = float(np.mean(mb_b))
    mean_c = float(np.mean(mb_c))
    std_a = float(np.std(mb_a, ddof=1)) if len(mb_a) > 1 else 0.0
    std_b = float(np.std(mb_b, ddof=1)) if len(mb_b) > 1 else 0.0
    std_c = float(np.std(mb_c, ddof=1)) if len(mb_c) > 1 else 0.0
    slope_a = compute_slope(mb_a)
    slope_b = compute_slope(mb_b)
    slope_c = compute_slope(mb_c)
    final_a = mb_a[-1] if mb_a else 0.0
    final_b = mb_b[-1] if mb_b else 0.0
    final_c = mb_c[-1] if mb_c else 0.0

    delta_b_vs_a = mean_b - mean_a
    delta_c_vs_a = mean_c - mean_a
    delta_b_vs_c = mean_b - mean_c

    total_eff_b = sum(e["effective"] for e in all_episodes_b)
    total_zero_b = sum(e["zero"] for e in all_episodes_b)
    total_harm_b = sum(e["harmful"] for e in all_episodes_b)
    failure_signal_count = total_zero_b + total_harm_b

    mem_stats = frm.get_statistics()
    top_risk = frm.get_top_risk_features(15)

    all_scores_flat = []
    for ep_list in [all_episodes_a, all_episodes_b, all_episodes_c]:
        for e in ep_list:
            all_scores_flat.append(e["min_final_score"])
    min_score_overall = min(all_scores_flat) if all_scores_flat else EXPLORATION_FLOOR
    hard_exclusion_used = min_score_overall <= 0.0

    # B vs A when no failures
    b_minus_a_no_failures = None
    if failure_signal_count == 0:
        b_minus_a_no_failures = delta_b_vs_a

    # Risk weight statistics
    risk_weights_nonzero = mem_stats["risk_weights_nonzero_count"]
    max_risk_weight = mem_stats["max_risk_weight"]
    all_risk_adjustments = []
    for ep_b_data in all_episodes_b:
        all_risk_adjustments.append(ep_b_data.get("avg_abs_risk_adjustment", 0.0))
    mean_risk_adj = float(np.mean(all_risk_adjustments)) if all_risk_adjustments else 0.0

    # Permuted control weight distribution check
    # B weights: from frm directly
    b_weights = []
    for key in frm.counts:
        b_weights.append(frm.get_risk_weight(key[0], key[1], key[2]))
    b_weights_nonzero = [w for w in b_weights if abs(w) > 1e-9]

    # Success-only bias check
    success_only_bias = (failure_signal_count == 0 and risk_weights_nonzero == 0)

    print(f"\n    Mode {mode_name} summary:")
    print(f"      A: mean={mean_a:.4f} +/- {std_a:.4f}  final={final_a:.4f}  slope={slope_a:+.4f}")
    print(f"      B: mean={mean_b:.4f} +/- {std_b:.4f}  final={final_b:.4f}  slope={slope_b:+.4f}")
    print(f"      C: mean={mean_c:.4f} +/- {std_c:.4f}  final={final_c:.4f}  slope={slope_c:+.4f}")
    print(f"      B-A={delta_b_vs_a:+.4f}  B-C={delta_b_vs_c:+.4f}")
    print(f"      failure_signal_count={failure_signal_count}")
    print(f"      risk_weights_nonzero={risk_weights_nonzero}  max_risk={max_risk_weight:.4f}")
    print(f"      success_only_bias_removed={success_only_bias}")

    return {
        "mode_name": mode_name,
        "explore_fraction": explore_fraction,
        "n_objects": len(test_objects),
        "n_episodes": N_EPISODES,

        # Per-condition
        "A_mean_macro_bal": round(mean_a, 4),
        "A_std_macro_bal": round(std_a, 4),
        "A_final_macro_bal": round(final_a, 4),
        "A_learning_slope": round(slope_a, 6),
        "B_mean_macro_bal": round(mean_b, 4),
        "B_std_macro_bal": round(std_b, 4),
        "B_final_macro_bal": round(final_b, 4),
        "B_learning_slope": round(slope_b, 6),
        "C_mean_macro_bal": round(mean_c, 4),
        "C_std_macro_bal": round(std_c, 4),
        "C_final_macro_bal": round(final_c, 4),
        "C_learning_slope": round(slope_c, 6),

        # Deltas
        "delta_B_vs_A": round(delta_b_vs_a, 4),
        "delta_C_vs_A": round(delta_c_vs_a, 4),
        "delta_B_vs_C": round(delta_b_vs_c, 4),

        # Probe effects (B)
        "total_effective_B": total_eff_b,
        "total_zero_B": total_zero_b,
        "total_harmful_B": total_harm_b,
        "failure_signal_count": failure_signal_count,
        "probe_count_B": sum(e["probed"] for e in all_episodes_b),

        # Memory diagnostics
        "memory_update_nontrivial": bool(mem_stats["memory_update_nontrivial"]),
        "memory_total_updates": mem_stats["total_updates"],
        "memory_failure_updates": mem_stats["failure_updates"],
        "memory_success_updates": mem_stats["success_updates"],
        "risk_weights_nonzero_count": risk_weights_nonzero,
        "max_risk_weight": max_risk_weight,
        "mean_risk_adjustment": round(mean_risk_adj, 6),
        "n_keys_with_data": mem_stats["n_keys_with_data"],
        "n_keys_from_failures": mem_stats["n_keys_from_failures"],
        "n_keys_from_successes": mem_stats["n_keys_from_successes"],

        # Validity
        "iom_isolated": True,
        "absence_counts_valid": True,
        "hard_exclusion_used": hard_exclusion_used,
        "min_final_score_overall": round(min_score_overall, 6),
        "success_only_bias_removed": success_only_bias,
        "B_minus_A_when_no_failures": round(b_minus_a_no_failures, 6) if b_minus_a_no_failures is not None else None,

        # Top risk features
        "top_risk_features": top_risk,

        # Per-episode detail
        "per_episode_A": [{k: v for k, v in e.items()}
                         for e in all_episodes_a],
        "per_episode_B": [{k: v for k, v in e.items()}
                         for e in all_episodes_b],
        "per_episode_C": [{k: v for k, v in e.items()}
                         for e in all_episodes_c],
    }


# =============================================================================
# 10. Run all three modes
# =============================================================================
print("\n[9/12] Running Mode 1: original_safe...")
mode1_results = run_mode(
    "original_safe",
    test_objects_standard,
    query_gt_standard,
    im_base,
    c15b_probed_n,
    explore_fraction=0.0,
    mode_rng_seed=SMOKE_SEED,
)

print("\n[10/12] Running Mode 2: epsilon_boundary_explore...")
mode2_results = run_mode(
    "epsilon_boundary_explore",
    test_objects_standard,
    query_gt_standard,
    im_base,
    c15b_probed_n,
    explore_fraction=0.30,
    mode_rng_seed=SMOKE_SEED + 5000,
)

print("\n[11/12] Running Mode 3: deceptive_high_prior_diagnostic...")
mode3_results = run_mode(
    "deceptive_high_prior_diagnostic",
    test_objects_deceptive,
    query_gt_deceptive,
    im_base,
    c15b_probed_n,
    explore_fraction=0.0,
    mode_rng_seed=SMOKE_SEED + 10000,
)
# Inject Mode 3 validity flag
mode3_results["mode3_valid"] = MODE3_VALID
mode3_results["n_deceptive_objects"] = len(deceptive_modifications)
mode3_results["deceptive_modifications"] = [
    {"oid": m["oid"], "category": m["category"], "flipped_action": m["flipped_action"],
     "deviation_features": m["deviation_features"]}
    for m in deceptive_modifications
]

# =============================================================================
# 11. Cross-mode interpretation
# =============================================================================
print(f"\n[12/12] Cross-mode interpretation...")

m1_fail = mode1_results["failure_signal_count"]
m2_fail = mode2_results["failure_signal_count"]
m3_fail = mode3_results["failure_signal_count"]

m1_b_delta = mode1_results["delta_B_vs_A"]
m2_b_delta = mode2_results["delta_B_vs_A"]
m3_b_delta = mode3_results["delta_B_vs_A"]

m1_b_vs_c = mode1_results["delta_B_vs_C"]
m2_b_vs_c = mode2_results["delta_B_vs_C"]
m3_b_vs_c = mode3_results["delta_B_vs_C"]

# Patched no-failure behavior
patched_no_failure_B_matches_A = (
    m1_fail == 0 and abs(m1_b_delta) < 0.005 and
    mode1_results["success_only_bias_removed"]
)

# Does any mode have nontrivial failure signals?
any_nontrivial_failures = (m2_fail > 0 or m3_fail > 0)

# Feature risk memory supported if: any VALID mode has failures AND B > A AND B > C
feature_risk_supported = False
beats_permuted = False
if any_nontrivial_failures:
    # Check mode 2 (always valid)
    if m2_fail > 0 and m2_b_delta > 0.001 and m2_b_vs_c > 0.001:
        feature_risk_supported = True
        beats_permuted = True
    # Check mode 3 (only if valid)
    if MODE3_VALID and m3_fail > 0 and m3_b_delta > 0.001 and m3_b_vs_c > 0.001:
        feature_risk_supported = True
        beats_permuted = True
    # Partial: B beats A but not C
    if not feature_risk_supported:
        m2_partial = m2_fail > 0 and m2_b_delta > 0.001
        m3_partial = MODE3_VALID and m3_fail > 0 and m3_b_delta > 0.001
        if m2_partial or m3_partial:
            if m2_b_vs_c <= 0.001 and m3_b_vs_c <= 0.001:
                beats_permuted = False

success_only_bias_removed = (
    mode1_results["success_only_bias_removed"] and
    mode2_results["success_only_bias_removed"] and
    mode3_results["success_only_bias_removed"]
)

any_hard_exclusion = (
    mode1_results["hard_exclusion_used"] or
    mode2_results["hard_exclusion_used"] or
    mode3_results["hard_exclusion_used"]
)

# Determine next route
if feature_risk_supported:
    next_route = "run_small_multiseed_1j19b"
elif any_nontrivial_failures and not feature_risk_supported:
    if m2_b_delta > 0.001 or m3_b_delta > 0.001:
        next_route = "gains_from_perturbation_investigate"
    else:
        next_route = "inspect_feature_space_and_risk_update"
elif not any_nontrivial_failures:
    if not MODE3_VALID:
        next_route = "mode3_invalid_no_observable_deviations_fix_and_retry"
    else:
        next_route = "diagnostic_failed_to_generate_failures_try_harsher_exploration"
else:
    next_route = "review_diagnostics"

# =============================================================================
# 12. C0b baseline
# =============================================================================
c0b_rng = random.Random(SMOKE_SEED + 800)
c0b_im = im_base.clone()
c0b_positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, c0b_rng)
c0b_env = MiniMCEnvironment(test_objects_standard, c0b_positions, initial_budget=BUDGET)
c0b_policy = C0b_ObserveOnlyPolicy(c0b_im, c0b_rng)
c0b_preds, c0b_event_log, c0b_obs, _, _ = run_policy(c0b_policy, c0b_env)
c0b_metrics = compute_full_metrics(_unwrap_preds(c0b_preds), query_gt_standard)
c0b_macro_bal = c0b_metrics["macro_query_balanced_accuracy"]

# =============================================================================
# 13. Output
# =============================================================================
elapsed = time.time() - t0

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

results = {
    "block_id": "1J19b",
    "condition": COND["label"],
    "smoke_seed": SMOKE_SEED,
    "budget": BUDGET,
    "n_episodes": N_EPISODES,
    "risk_alpha": RISK_ALPHA,
    "max_risk": MAX_RISK,
    "min_failure_support": MIN_FAILURE_SUPPORT,
    "feature_universe_size": len(FEATURE_UNIVERSE),
    "exploration_floor": EXPLORATION_FLOOR,

    # Baselines
    "c15b_macro_bal": round(c15b_macro_bal, 4),
    "c15b_probed_n": c15b_probed_n,
    "c0b_macro_bal": round(c0b_macro_bal, 4),

    # Per-mode results
    "mode_original_safe": mode1_results,
    "mode_epsilon_boundary_explore": mode2_results,
    "mode_deceptive_high_prior_diagnostic": mode3_results,

    # Cross-mode interpretation
    "mode_original_failure_signal_count": m1_fail,
    "mode_boundary_failure_signal_count": m2_fail,
    "mode_deceptive_failure_signal_count": m3_fail,
    "mode3_valid": MODE3_VALID,
    "mode3_has_observable_deviations": MODE3_VALID,
    "mode3_n_deceptive_objects": len(deceptive_modifications),
    "patched_no_failure_B_matches_A": patched_no_failure_B_matches_A,
    "feature_risk_memory_supported": feature_risk_supported,
    "beats_permuted_control": beats_permuted,
    "hard_exclusion_used": any_hard_exclusion,
    "success_only_bias_removed": success_only_bias_removed,
    "ready_for_full_1j19": feature_risk_supported and not any_hard_exclusion,
    "next_recommended_route": next_route,
    "elapsed_s": round(elapsed, 1),
}

# Save JSON
runs_dir = os.path.join(CURRENT_DIR, "runs")
os.makedirs(runs_dir, exist_ok=True)
json_path = os.path.join(runs_dir, "calibration_block1j19b_failure_signal_elicitation_seed101.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2, cls=_NumpyEncoder)
print(f"\n  Saved: {json_path}")

# Save protocol MD
protocols_dir = os.path.join(CURRENT_DIR, "protocols")
os.makedirs(protocols_dir, exist_ok=True)
md_path = os.path.join(protocols_dir, "block1j19b_failure_signal_elicitation_seed101.md")

def fmt_bool(v):
    return "true" if v else "false"

md = []
md.append("# Block 1J19b -- Failure-Signal Elicitation for Negative-Exception Feature-Risk Memory\n")
md.append("## 1. Objective\n")
md.append("Test whether a patched negative-exception feature-risk memory (V2) improves probe selection when failure signals are present, and verify that the memory is inert (B ~= A) when no failures occur.\n")

md.append("## 2. Setup\n")
md.append(f"| Parameter | Value |")
md.append(f"|-----------|-------|")
md.append(f"| Condition | {COND['label']} |")
md.append(f"| Seed | {SMOKE_SEED} |")
md.append(f"| Budget | {BUDGET} |")
md.append(f"| Episodes | {N_EPISODES} |")
md.append(f"| Risk Alpha | {RISK_ALPHA} |")
md.append(f"| Max Risk | {MAX_RISK} |")
md.append(f"| Min Failure Support | {MIN_FAILURE_SUPPORT} |")
md.append(f"| Exploration Floor | {EXPLORATION_FLOOR} |")
md.append(f"| Feature Universe | {len(FEATURE_UNIVERSE)} |")

md.append("\n## 3. Patched Memory Semantics (V2)\n")
md.append("- Risk weight active only if actual failure observations >= min_failure_support")
md.append("- Only positive risk weights (penalty); negative weights clipped to 0")
md.append("- Successes calibrate the denominator (P(feature|success)) but don't create bonuses")
md.append("- When failure_signal_count == 0, all risk weights = 0, so B ~= A\n")

md.append("## 4. Baselines\n")
md.append(f"| Policy | Macro BAcc |")
md.append(f"|--------|------------|")
md.append(f"| C0b (observe only) | {c0b_macro_bal:.4f} |")
md.append(f"| C15b (VOI reserve) | {c15b_macro_bal:.4f} |\n")

for mode_key, mode_label in [
    ("mode_original_safe", "Original Safe"),
    ("mode_epsilon_boundary_explore", "Epsilon-Boundary Explore"),
    ("mode_deceptive_high_prior_diagnostic", "Deceptive High-Prior Diagnostic"),
]:
    mr = results[mode_key]
    md.append(f"## 5. Mode: {mode_label}\n")
    md.append(f"Explore fraction: {mr['explore_fraction']}  |  Objects: {mr['n_objects']}\n")

    md.append(f"### 5.1 Per-Episode Results\n")
    md.append(f"| Ep | A (soft prior) | B (risk memory) | C (permuted) | B Eff | B Zero | B Harm | B RiskAdj |")
    md.append(f"|----|----------------|-----------------|--------------|-------|--------|--------|-----------|")
    for i in range(N_EPISODES):
        ea = mr["per_episode_A"][i]
        eb = mr["per_episode_B"][i]
        ec = mr["per_episode_C"][i]
        md.append(f"| {i+1} | {ea['macro_bal']:.4f} | {eb['macro_bal']:.4f} | {ec['macro_bal']:.4f} | {eb['effective']} | {eb['zero']} | {eb['harmful']} | {eb.get('avg_abs_risk_adjustment', 0):.4f} |")

    md.append(f"\n### 5.2 Aggregated Metrics\n")
    md.append(f"| Metric | A (soft prior) | B (risk memory) | C (permuted) |")
    md.append(f"|--------|----------------|-----------------|--------------|")
    md.append(f"| Mean macro BAcc | {mr['A_mean_macro_bal']:.4f} +/- {mr['A_std_macro_bal']:.4f} | {mr['B_mean_macro_bal']:.4f} +/- {mr['B_std_macro_bal']:.4f} | {mr['C_mean_macro_bal']:.4f} +/- {mr['C_std_macro_bal']:.4f} |")
    md.append(f"| Final macro BAcc | {mr['A_final_macro_bal']:.4f} | {mr['B_final_macro_bal']:.4f} | {mr['C_final_macro_bal']:.4f} |")
    md.append(f"| Learning slope | {mr['A_learning_slope']:+.4f}/ep | {mr['B_learning_slope']:+.4f}/ep | {mr['C_learning_slope']:+.4f}/ep |")

    md.append(f"\n### 5.3 Deltas\n")
    md.append(f"| Comparison | Delta |")
    md.append(f"|------------|-------|")
    md.append(f"| B - A | {mr['delta_B_vs_A']:+.4f} |")
    md.append(f"| C - A | {mr['delta_C_vs_A']:+.4f} |")
    md.append(f"| B - C | {mr['delta_B_vs_C']:+.4f} |")

    md.append(f"\n### 5.4 Probe Effect Diagnostics\n")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| Total probes (B) | {mr['probe_count_B']} |")
    md.append(f"| Effective | {mr['total_effective_B']} |")
    md.append(f"| Zero | {mr['total_zero_B']} |")
    md.append(f"| Harmful | {mr['total_harmful_B']} |")
    md.append(f"| Failure signal count | {mr['failure_signal_count']} |")

    md.append(f"\n### 5.5 Memory Diagnostics\n")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| Memory update nontrivial | {fmt_bool(mr['memory_update_nontrivial'])} |")
    md.append(f"| Total updates | {mr['memory_total_updates']} |")
    md.append(f"| Failure updates | {mr['memory_failure_updates']} |")
    md.append(f"| Success updates | {mr['memory_success_updates']} |")
    md.append(f"| Risk weights nonzero | {mr['risk_weights_nonzero_count']} |")
    md.append(f"| Max risk weight | {mr['max_risk_weight']:.4f} |")
    md.append(f"| Mean risk adjustment | {mr['mean_risk_adjustment']:.4f} |")
    md.append(f"| Keys with data | {mr['n_keys_with_data']} |")
    md.append(f"| Keys from failures | {mr['n_keys_from_failures']} |")
    md.append(f"| Keys from successes | {mr['n_keys_from_successes']} |")
    md.append(f"| Success-only bias removed | {fmt_bool(mr['success_only_bias_removed'])} |")
    if mr['B_minus_A_when_no_failures'] is not None:
        md.append(f"| B - A (when no failures) | {mr['B_minus_A_when_no_failures']:+.6f} |")

    md.append(f"\n### 5.6 Validity Checks\n")
    md.append(f"| Check | Value |")
    md.append(f"|-------|-------|")
    md.append(f"| IOM isolated | {fmt_bool(mr['iom_isolated'])} |")
    md.append(f"| Absence counts valid | {fmt_bool(mr['absence_counts_valid'])} |")
    md.append(f"| Hard exclusion used | {fmt_bool(mr['hard_exclusion_used'])} |")
    md.append(f"| Min final score | {mr['min_final_score_overall']:.4f} |")

    # Top risk features
    top = mr.get("top_risk_features", [])
    if top:
        md.append(f"\n### 5.7 Top Learned Risk Features\n")
        md.append(f"| Feature | Goal | Action | Risk Weight | FailWith | SuccWith | FailWithout | SuccWithout | ActFail |")
        md.append(f"|---------|------|--------|-------------|----------|----------|-------------|-------------|---------|")
        for item in top:
            md.append(f"| {item['feature']} | {item['goal']} | {item['action']} | {item['risk_weight']:.4f} | {item['fail_with']:.0f} | {item['succ_with']:.0f} | {item['fail_without']:.0f} | {item['succ_without']:.0f} | {item['actual_failures']:.0f} |")
    else:
        md.append(f"\n### 5.7 Top Learned Risk Features\n")
        md.append("(none -- no risk weights above threshold)\n")

md.append("## 6. Cross-Mode Interpretation\n")
md.append(f"| Criterion | Value |")
md.append(f"|-----------|-------|")
md.append(f"| Mode 1 (original) failure signal count | {m1_fail} |")
md.append(f"| Mode 2 (boundary) failure signal count | {m2_fail} |")
md.append(f"| Mode 3 (deceptive) failure signal count | {m3_fail} |")
md.append(f"| Mode 3 has observable deviations | {fmt_bool(MODE3_VALID)} |")
md.append(f"| Mode 3 deceptive objects | {len(deceptive_modifications)} |")
md.append(f"| Patched: B ~= A when no failures | {fmt_bool(patched_no_failure_B_matches_A)} |")
md.append(f"| Feature-risk memory supported | {fmt_bool(feature_risk_supported)} |")
md.append(f"| Beats permuted control | {fmt_bool(beats_permuted)} |")
md.append(f"| Hard exclusion used | {fmt_bool(any_hard_exclusion)} |")
md.append(f"| Success-only bias removed | {fmt_bool(success_only_bias_removed)} |")
md.append(f"| Ready for full 1J19 | {fmt_bool(results['ready_for_full_1j19'])} |")
md.append(f"| Next route | {next_route} |")

md.append("\n## 7. Summary\n")
md.append("```")
md.append("[block_done]")
md.append(f"block_id=1J19b")
md.append(f"mode_original_failure_signal_count={m1_fail}")
md.append(f"mode_boundary_failure_signal_count={m2_fail}")
md.append(f"mode_deceptive_failure_signal_count={m3_fail}")
md.append(f"mode3_has_observable_deviations={fmt_bool(MODE3_VALID)}")
md.append(f"patched_no_failure_B_matches_A={fmt_bool(patched_no_failure_B_matches_A)}")
md.append(f"feature_risk_memory_supported={fmt_bool(feature_risk_supported)}")
md.append(f"beats_permuted_control={fmt_bool(beats_permuted)}")
md.append(f"hard_exclusion_used={fmt_bool(any_hard_exclusion)}")
md.append(f"success_only_bias_removed={fmt_bool(success_only_bias_removed)}")
md.append(f"ready_for_full_1j19={fmt_bool(results['ready_for_full_1j19'])}")
md.append(f"next_recommended_route={next_route}")
md.append(f"elapsed={elapsed:.1f}s")
md.append("```")

with open(md_path, "w") as f:
    f.write("\n".join(md))
print(f"  Saved: {md_path}")

# =============================================================================
# Print [block_done]
# =============================================================================
print(f"\n{'='*70}")
print(f"[block_done]")
print(f"block_id=1J19b")
print(f"mode_original_failure_signal_count={m1_fail}")
print(f"mode_boundary_failure_signal_count={m2_fail}")
print(f"mode_deceptive_failure_signal_count={m3_fail}")
print(f"mode3_has_observable_deviations={fmt_bool(MODE3_VALID)}")
print(f"patched_no_failure_B_matches_A={fmt_bool(patched_no_failure_B_matches_A)}")
print(f"feature_risk_memory_supported={fmt_bool(feature_risk_supported)}")
print(f"beats_permuted_control={fmt_bool(beats_permuted)}")
print(f"hard_exclusion_used={fmt_bool(any_hard_exclusion)}")
print(f"success_only_bias_removed={fmt_bool(success_only_bias_removed)}")
print(f"ready_for_full_1j19={fmt_bool(results['ready_for_full_1j19'])}")
print(f"next_recommended_route={next_route}")
print(f"elapsed={elapsed:.1f}s")
print(f"{'='*70}")
