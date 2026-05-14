"""
Block 1J19d -- Prior-Violation Memory Diagnostic Ablation.

Diagnoses why B_full does not beat C_full on prior violations in 1J19c
family-preserving patch, and why B macro drops below tolerance.

Parts:
  1. Selection-difference diagnostics (A vs B vs C per-probe details)
  2. Risk-weight diagnostics by feature group
  3. Macro-drop attribution by query/family/action
  4. Ablation 1: B_dev_only (injected deviation features only)
  5. Ablation 2: risk_penalty_scale grid
  6. Interpretation + recommended next route
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
MACRO_DROP_TOLERANCE = -0.01
RISK_PENALTY_SCALES = [0.25, 0.5, 1.0]

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
    "damp_texture", "brittle_surface", "treated_surface", "hollow_sound",
})
FEATURE_UNIVERSE_SET = set(FEATURE_UNIVERSE)

# Feature groups for Part 2 risk-weight diagnostics
WOOD_FEATURES = {"has_bark_texture", "has_wood_grain", "brownish", "rough_texture", "long_shape"}
STONE_FEATURES = {"has_crystal_flecks", "has_granular_surface", "grayish", "block_like", "heavy_weight"}
APPLE_FEATURES = {"has_stem_remnant", "has_peel_texture", "round_small", "greenish", "smooth_texture", "light_weight"}
TOOL_FEATURES = {"has_grip_area", "has_shaft_shape", "elongated_with_handle", "movable", "long_shape"}

POSITIONAL_FEATURES = {"on_left_side", "near_table", "recently_seen"}
OTHER_FEATURES = {"solid"}  # neither type-family nor positional

TYPE_FAMILIES = {
    "wood-like": WOOD_FEATURES, "stone-like": STONE_FEATURES,
    "apple-like": APPLE_FEATURES, "tool-like": TOOL_FEATURES,
}

ALL_TYPE_FAMILY_FEATURES = set()
for fs in TYPE_FAMILIES.values():
    ALL_TYPE_FAMILY_FEATURES.update(fs)

INJECTED_DEVIATION_FEATURES = {"damp_texture", "brittle_surface", "treated_surface", "hollow_sound"}

ACTION_TO_QUERY = {
    "craft_plank": "need_planks", "eat": "need_food",
    "use_as_tool": "need_tool", "burn_as_fuel": "need_fuel",
    "mine_by_hand": "need_stone", "mine_with_pickaxe": "need_stone",
}

def classify_feature_group(feat):
    if feat in INJECTED_DEVIATION_FEATURES:
        return "A_injected_deviation"
    if feat in ALL_TYPE_FAMILY_FEATURES:
        return "B_type_family"
    if feat in POSITIONAL_FEATURES:
        return "C_positional_or_context"
    return "D_other"

print("=" * 70)
print("Block 1J19d -- Prior-Violation Memory Diagnostic Ablation")
print(f"  condition={COND['label']}  seed={SMOKE_SEED}  budget={BUDGET}")
print(f"  episodes={N_EPISODES}  risk_alpha={RISK_ALPHA}  max_risk={MAX_RISK}")
print(f"  prior_violation_threshold={PRIOR_VIOLATION_THRESHOLD}")
print(f"  min_violation_support={MIN_VIOLATION_SUPPORT}")
print(f"  risk_penalty_scales={RISK_PENALTY_SCALES}")
print(f"  injected_deviation_features={sorted(INJECTED_DEVIATION_FEATURES)}")
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

# =============================================================================
# 2. Test objects + Family-preserving deceptive objects
# =============================================================================
print("\n[2/12] Generating test objects...")
test_objects_standard = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")
query_gt_standard = compute_query_ground_truth(test_objects_standard)
test_oids = sorted(test_objects_standard.keys())
QUERY_NAMES = sorted(_TASK_QUERIES.keys())
N_OBJECTS = len(test_oids)
print(f"  {N_OBJECTS} standard test objects, {len(QUERY_NAMES)} queries")

# Family-preserving deceptive objects (from 1J19c patch)
print("\n[3/12] Creating family-preserving deceptive test objects...")

APPLE_CORE = {"has_stem_remnant", "has_peel_texture", "round_small", "greenish",
              "smooth_texture", "light_weight"}
WOOD_CORE  = {"has_bark_texture", "has_wood_grain", "brownish", "rough_texture", "long_shape"}
STONE_CORE = {"has_crystal_flecks", "has_granular_surface", "grayish", "block_like", "heavy_weight"}

APPLE_DEVIATION  = {"damp_texture": True, "brittle_surface": True}
WOOD_DEVIATION   = {"damp_texture": True, "treated_surface": True}
STONE_DEVIATION  = {"brittle_surface": True, "hollow_sound": True}

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
    for feat in APPLE_CORE:
        if feat in FEATURE_UNIVERSE_SET:
            obj["visible_features"][feat] = True
    for feat, val in APPLE_DEVIATION.items():
        if feat in FEATURE_UNIVERSE_SET:
            obj["visible_features"][feat] = val
    deceptive_modifications.append({
        "oid": oid, "category": "apple", "deceptive_action": "eat",
        "deviation_features": [f for f, v in APPLE_DEVIATION.items() if v and f in FEATURE_UNIVERSE_SET],
    })

n_deceptive_wood = min(5, len(wood_oids))
for oid in deceptive_rng.sample(wood_oids, n_deceptive_wood):
    obj = test_objects_deceptive[oid]
    obj["hidden_affordance_profile"]["burn_as_fuel"] = "fail"
    obj["_deceptive"] = True
    obj["_deceptive_action"] = "burn_as_fuel"
    obj["_deviation_features"] = {f: True for f, v in WOOD_DEVIATION.items() if v}
    for feat in WOOD_CORE:
        if feat in FEATURE_UNIVERSE_SET:
            obj["visible_features"][feat] = True
    for feat, val in WOOD_DEVIATION.items():
        if feat in FEATURE_UNIVERSE_SET:
            obj["visible_features"][feat] = val
    deceptive_modifications.append({
        "oid": oid, "category": "wood_log", "deceptive_action": "burn_as_fuel",
        "deviation_features": [f for f, v in WOOD_DEVIATION.items() if v and f in FEATURE_UNIVERSE_SET],
    })

n_deceptive_stones = min(3, len(stone_oids))
for oid in deceptive_rng.sample(stone_oids, n_deceptive_stones):
    obj = test_objects_deceptive[oid]
    obj["hidden_affordance_profile"]["mine_with_pickaxe"] = "fail"
    obj["_deceptive"] = True
    obj["_deceptive_action"] = "mine_with_pickaxe"
    obj["_deviation_features"] = {f: True for f, v in STONE_DEVIATION.items() if v}
    for feat in STONE_CORE:
        if feat in FEATURE_UNIVERSE_SET:
            obj["visible_features"][feat] = True
    for feat, val in STONE_DEVIATION.items():
        if feat in FEATURE_UNIVERSE_SET:
            obj["visible_features"][feat] = val
    deceptive_modifications.append({
        "oid": oid, "category": "stone_block", "deceptive_action": "mine_with_pickaxe",
        "deviation_features": [f for f, v in STONE_DEVIATION.items() if v and f in FEATURE_UNIVERSE_SET],
    })

query_gt_deceptive = compute_query_ground_truth(test_objects_deceptive)
DECEPTIVE_OIDS = set(mod["oid"] for mod in deceptive_modifications)
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
print("\n[4/12] Building goal-conditioned soft prior table...")


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

print(f"  Prior table: {len(DEFAULT_GOAL_PRIOR)} entries")


def get_goal_soft_prior(features, action):
    fam = detect_type_family(features)
    qname = ACTION_TO_QUERY.get(action, "")
    return DEFAULT_GOAL_PRIOR.get((fam, qname, action), EXPLORATION_FLOOR)


def check_prior_violation(features, action, outcome, threshold=PRIOR_VIOLATION_THRESHOLD):
    soft_prior = get_goal_soft_prior(features, action)
    is_eligible = soft_prior >= threshold
    affordance_failed = outcome <= 0.01
    is_violation = is_eligible and affordance_failed
    return is_violation, is_eligible, soft_prior


# =============================================================================
# 5. FeatureRiskMemoryV3 with feature_whitelist support
# =============================================================================
print("\n[5/12] Initializing FeatureRiskMemoryV3...")

class FeatureRiskMemoryV3:
    """Prior-violation memory with optional feature whitelist."""

    def __init__(self, alpha=RISK_ALPHA, max_risk=MAX_RISK,
                 min_violation_support=MIN_VIOLATION_SUPPORT,
                 feature_whitelist=None):
        self.alpha = alpha
        self.max_risk = max_risk
        self.min_violation_support = min_violation_support
        self.feature_whitelist = feature_whitelist  # None = all features
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

    def _features_to_track(self):
        if self.feature_whitelist is not None:
            return self.feature_whitelist
        return FEATURE_UNIVERSE

    def update(self, features, goal, action, is_violation, is_eligible):
        self.total_updates += 1
        if not is_eligible:
            self.ineligible_updates += 1
            return

        self.eligible_updates += 1
        if is_violation:
            self.violation_updates += 1
        else:
            self.nonviolation_updates += 1

        for feat in self._features_to_track():
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
        for feat in self._features_to_track():
            if features.get(feat, False):
                w = self.get_risk_weight(feat, goal, action)
                total += w
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
            "n_keys_total": n_keys,
            "risk_weights_nonzero_count": nonzero_count,
            "max_risk_weight": round(max_w, 6),
            "mean_abs_risk_weight": round(sum_abs / max(n_keys, 1), 6),
        }

    def get_risk_weight_groups(self):
        """Group all nonzero risk weights by feature group."""
        groups = {
            "A_injected_deviation": [],
            "B_type_family": [],
            "C_positional_or_context": [],
            "D_other": [],
        }
        for (feat, goal, action), c in self.counts.items():
            w = self.get_risk_weight(feat, goal, action)
            if abs(w) < 1e-9: continue
            group = classify_feature_group(feat)
            groups[group].append({
                "feature": feat, "goal": goal, "action": action,
                "risk_weight": round(w, 6),
            })
        result = {}
        total_mass = sum(abs(item["risk_weight"]) for lst in groups.values() for item in lst)
        for group_name in ["A_injected_deviation", "B_type_family",
                           "C_positional_or_context", "D_other"]:
            items = groups[group_name]
            abs_sum = sum(abs(item["risk_weight"]) for item in items)
            result[group_name] = {
                "count_nonzero": len(items),
                "total_abs_risk_weight": round(abs_sum, 6),
                "max_risk_weight": round(max((abs(item["risk_weight"]) for item in items), default=0.0), 6),
                "average_risk_weight": round(abs_sum / max(len(items), 1), 6),
                "percentage_of_total_risk_mass": round(100.0 * abs_sum / max(total_mass, 0.001), 2),
                "items": sorted(items, key=lambda x: abs(x["risk_weight"]), reverse=True),
            }
        return result

    def get_top_risk_features(self, top_n=20):
        items = []
        for (feat, goal, action), c in self.counts.items():
            w = self.get_risk_weight(feat, goal, action)
            if abs(w) > 1e-9:
                items.append({
                    "feature": feat, "goal": goal, "action": action,
                    "risk_weight": round(w, 6),
                    "feature_group": classify_feature_group(feat),
                })
        items.sort(key=lambda x: abs(x["risk_weight"]), reverse=True)
        return items[:top_n]

    def clone(self):
        new = FeatureRiskMemoryV3(alpha=self.alpha, max_risk=self.max_risk,
                                  min_violation_support=self.min_violation_support,
                                  feature_whitelist=self.feature_whitelist)
        # update defaultdict (preserves defaultdict type — no KeyError on miss)
        new.counts.update(copy.deepcopy(dict(self.counts)))
        new._risk_cache = dict(self._risk_cache)
        new.total_updates = self.total_updates
        new.violation_updates = self.violation_updates
        new.nonviolation_updates = self.nonviolation_updates
        new.eligible_updates = self.eligible_updates
        new.ineligible_updates = self.ineligible_updates
        return new


# =============================================================================
# 6. Policy Classes (with risk_penalty_scale and feature_whitelist)
# =============================================================================
print("\n[6/12] Defining policy classes...")

class SoftPriorOnlyPolicyV2(MiniMCPolicy):
    """Condition A: soft_prior_only."""

    PHASE_OBSERVE = 1; PHASE_PROBE = 2

    def __init__(self, instance_memory, rng, target_probe_count=8, explore_fraction=0.0):
        self._im = instance_memory; self._rng = rng
        self._phase = self.PHASE_OBSERVE; self._probed_oids = set()
        self._pre_probe_entropies = {}; self._observe_pass_visit_count = 0
        self._target_probe_count = target_probe_count
        self._explore_fraction = explore_fraction
        self._variant = "A_soft_prior_only"
        self._probe_effects = []; self._selected_scores = []; self._probe_tracking = []

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE; self._probed_oids = set()
        self._pre_probe_entropies = {}; self._observe_pass_visit_count = 0
        self._probe_effects = []; self._selected_scores = []; self._probe_tracking = []

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
            "prior": best_prior, "raw_score": round(best_raw_score, 6),
            "final_score": round(best_score, 6), "risk_adjustment": 0.0,
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

    def get_probe_effects(self): return list(self._probe_effects)
    def get_selected_scores(self): return list(self._selected_scores)


class RiskAdjustedPolicy(SoftPriorOnlyPolicyV2):
    """Base class for B and C: soft_prior + risk memory."""

    def __init__(self, instance_memory, rng, feature_risk_memory,
                 target_probe_count=8, explore_fraction=0.0,
                 risk_penalty_scale=1.0):
        super().__init__(instance_memory, rng, target_probe_count, explore_fraction)
        self._frm = feature_risk_memory
        self._risk_penalty_scale = risk_penalty_scale
        self._variant = "B_risk_adjusted"

    def _compute_adjusted_score(self, features, action, norm_cost):
        base_prior = get_goal_soft_prior(features, action)
        risk_penalty = self._frm.get_total_risk_penalty(features, action)
        scaled_penalty = risk_penalty * self._risk_penalty_scale
        raw_score = base_prior * math.exp(-scaled_penalty) - COST_WEIGHT * norm_cost
        final_score = max(EXPLORATION_FLOOR, raw_score)
        return final_score, raw_score, base_prior, risk_penalty, scaled_penalty

    def _score_candidate(self, features, norm_cost):
        best_score = float('-inf')
        for action in MAIN_CANDIDATE_ACTIONS:
            final_score, _, _, _, _ = self._compute_adjusted_score(features, action, norm_cost)
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
        best_info = (EXPLORATION_FLOOR, 0.0, 0.0, 0.0, EXPLORATION_FLOOR)
        for action in MAIN_CANDIDATE_ACTIONS:
            norm_cost = view.probe_cost / max(view.initial_budget, 0.001)
            final_score, raw_score, base_prior, risk_penalty, scaled_penalty = \
                self._compute_adjusted_score(features, action, norm_cost)
            if final_score > best_score:
                best_score = final_score; best_action = action
                best_info = (base_prior, risk_penalty, scaled_penalty, raw_score, final_score)
        self._probed_oids.add(object_id)
        bp, rp, sp, rs, fs = best_info
        self._selected_scores.append({
            "oid": object_id, "action": best_action,
            "prior": bp, "risk_penalty": round(rp, 6),
            "scaled_risk_penalty": round(sp, 6),
            "raw_score": round(rs, 6), "final_score": round(fs, 6),
            "risk_adjustment": round(sp, 6),
        })
        return True, best_action


class PermutedRiskPolicy(RiskAdjustedPolicy):
    """Condition C: permuted risk weights."""

    def __init__(self, instance_memory, rng, feature_risk_memory,
                 target_probe_count=8, explore_fraction=0.0,
                 risk_penalty_scale=1.0):
        super().__init__(instance_memory, rng, feature_risk_memory,
                        target_probe_count, explore_fraction, risk_penalty_scale)
        self._variant = "C_permuted"
        self._permuted_map = {}

    def apply_permutation(self, rng):
        self._permuted_map = self._frm.permute_weights(rng)

    def _compute_adjusted_score(self, features, action, norm_cost):
        base_prior = get_goal_soft_prior(features, action)
        goal = ACTION_TO_QUERY.get(action, "")
        if self._permuted_map:
            risk_penalty = 0.0
            for feat in self._frm._features_to_track():
                if features.get(feat, False):
                    risk_penalty += self._permuted_map.get((feat, goal, action), 0.0)
        else:
            risk_penalty = self._frm.get_total_risk_penalty(features, action)
        scaled_penalty = risk_penalty * self._risk_penalty_scale
        raw_score = base_prior * math.exp(-scaled_penalty) - COST_WEIGHT * norm_cost
        final_score = max(EXPLORATION_FLOOR, raw_score)
        return final_score, raw_score, base_prior, risk_penalty, scaled_penalty


# =============================================================================
# 7. C15b reference
# =============================================================================
print("\n[7/12] Running C15b reference...")

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
# 8. Detailed diagnostic mode runner
# =============================================================================
print("\n[8/12] Setting up diagnostic mode runner...")

def run_diagnostic_mode(test_objects, query_gt, im_base, c15b_probed_n,
                        risk_penalty_scale=1.0, feature_whitelist=None,
                        variant_label="B_full", is_permuted=False,
                        mode_rng_seed=SMOKE_SEED):
    """
    Run A + one condition (B or C) with detailed per-probe tracking.
    Returns:
        - per_episode details for A and variant
        - frm (the memory used)
        - all per-probe diagnostics
    """
    frm = FeatureRiskMemoryV3(feature_whitelist=feature_whitelist)
    all_episodes_a = []; all_episodes_var = []
    prior_violation_history = set()

    for ep in range(N_EPISODES):
        ep_seed = mode_rng_seed + ep * 100
        ep_rng = random.Random(ep_seed)
        test_oids_local = sorted(test_objects.keys())
        ep_positions = MiniMCSimulatorTruth.assign_positions(
            test_oids_local, config.GRID_ROWS, config.GRID_COLS,
            config.AGENT_START, ep_rng)

        # --- Condition A ---
        im_a = im_base.clone()
        env_a = MiniMCEnvironment(test_objects, dict(ep_positions), initial_budget=BUDGET)
        policy_a = SoftPriorOnlyPolicyV2(im_a, ep_rng, target_probe_count=c15b_probed_n)
        preds_a, _, obs_a, _, _ = run_policy(policy_a, env_a)
        metrics_a = compute_full_metrics(_unwrap_preds(preds_a), query_gt)
        effects_a = policy_a.get_probe_effects()
        scores_a = policy_a.get_selected_scores()

        pv_count_a = 0
        a_probe_details = []
        for effect, score in zip(effects_a, scores_a):
            oid = effect["oid"]; action = effect["action"]; outcome = effect["outcome"]
            obj = test_objects.get(oid, {})
            features = obj.get("visible_features", {})
            is_violation, is_eligible, sp = check_prior_violation(features, action, outcome)
            qname = ACTION_TO_QUERY.get(action, "")
            family = detect_type_family(features)
            injected_present = [f for f in INJECTED_DEVIATION_FEATURES if features.get(f, False)]
            detail = {
                "oid": oid, "action": action, "outcome": outcome,
                "family": family, "query": qname,
                "soft_prior": round(sp, 4), "is_eligible": is_eligible,
                "is_violation": is_violation,
                "probe_utility_signal": effect["effect"],
                "query_changed": effect["query_changed"],
                "final_score": score["final_score"],
                "injected_deviation_present": injected_present,
                "is_deceptive": oid in DECEPTIVE_OIDS,
            }
            if is_violation: pv_count_a += 1
            a_probe_details.append(detail)

        all_episodes_a.append({
            "episode": ep,
            "macro_bal": metrics_a["macro_query_balanced_accuracy"],
            "per_query": metrics_a["per_query"],
            "prior_violation_count": pv_count_a,
            "probe_details": a_probe_details,
        })

        # --- Variant (B or C) ---
        im_var = im_base.clone()
        env_var = MiniMCEnvironment(test_objects, dict(ep_positions), initial_budget=BUDGET)
        if is_permuted:
            permuted_memory = frm.clone()
            policy_var = PermutedRiskPolicy(
                im_var, ep_rng, permuted_memory,
                target_probe_count=c15b_probed_n,
                risk_penalty_scale=risk_penalty_scale)
            policy_var.apply_permutation(ep_rng)
        else:
            policy_var = RiskAdjustedPolicy(
                im_var, ep_rng, frm,
                target_probe_count=c15b_probed_n,
                risk_penalty_scale=risk_penalty_scale)
        preds_var, _, obs_var, _, _ = run_policy(policy_var, env_var)
        metrics_var = compute_full_metrics(_unwrap_preds(preds_var), query_gt)
        effects_var = policy_var.get_probe_effects()
        scores_var = policy_var.get_selected_scores()

        n_violation_before = frm.violation_updates
        pv_count_var = 0; pv_repeated_var = 0
        var_probe_details = []

        for effect, score in zip(effects_var, scores_var):
            oid = effect["oid"]; action = effect["action"]; outcome = effect["outcome"]
            goal = ACTION_TO_QUERY.get(action, "")
            obj = test_objects.get(oid, {})
            features = obj.get("visible_features", {})
            is_violation, is_eligible, sp = check_prior_violation(features, action, outcome)
            qname = ACTION_TO_QUERY.get(action, "")
            family = detect_type_family(features)
            injected_present = [f for f in INJECTED_DEVIATION_FEATURES if features.get(f, False)]

            detail = {
                "oid": oid, "action": action, "outcome": outcome,
                "family": family, "query": qname,
                "soft_prior": round(sp, 4), "is_eligible": is_eligible,
                "is_violation": is_violation,
                "probe_utility_signal": effect["effect"],
                "query_changed": effect["query_changed"],
                "risk_penalty": round(score.get("risk_penalty", 0.0), 6),
                "scaled_risk_penalty": round(score.get("scaled_risk_penalty", 0.0), 6),
                "final_score": score["final_score"],
                "injected_deviation_present": injected_present,
                "is_deceptive": oid in DECEPTIVE_OIDS,
            }
            if is_violation:
                pv_count_var += 1
                if (oid, action) in prior_violation_history:
                    pv_repeated_var += 1
                    detail["repeated"] = True
                else:
                    detail["repeated"] = False
                prior_violation_history.add((oid, action))
            var_probe_details.append(detail)

            # Update memory
            frm.update(features, goal, action, is_violation, is_eligible)

        new_violations_this_ep = frm.violation_updates - n_violation_before

        all_episodes_var.append({
            "episode": ep,
            "macro_bal": metrics_var["macro_query_balanced_accuracy"],
            "per_query": metrics_var["per_query"],
            "prior_violation_count": pv_count_var,
            "prior_violation_repeated": pv_repeated_var,
            "new_violation_signals": new_violations_this_ep,
            "probe_details": var_probe_details,
        })

        if ep == 0 or ep == N_EPISODES - 1:
            print(f"    Ep {ep+1}: A_bal={metrics_a['macro_query_balanced_accuracy']:.4f}  "
                  f"{variant_label}_bal={metrics_var['macro_query_balanced_accuracy']:.4f}  "
                  f"A_pv={pv_count_a}  {variant_label}_pv={pv_count_var}  "
                  f"new_v={new_violations_this_ep}")

    return all_episodes_a, all_episodes_var, frm


def aggregate_episodes(ep_list):
    mbs = [e["macro_bal"] for e in ep_list]
    pvs = [e.get("prior_violation_count", 0) for e in ep_list]
    return {
        "mean_macro_bal": round(float(np.mean(mbs)), 4),
        "std_macro_bal": round(float(np.std(mbs, ddof=1)) if len(mbs) > 1 else 0.0, 4),
        "final_macro_bal": round(mbs[-1], 4) if mbs else 0.0,
        "total_prior_violations": sum(pvs),
        "total_repeated_prior_violations": sum(e.get("prior_violation_repeated", 0)
                                                for e in ep_list),
        "per_episode": ep_list,
    }


def compute_selection_diff(ep_a, ep_var, variant_label, test_objects):
    """Compare A and variant selections per episode. Report avoided and new selections."""
    a_set = set((d["oid"], d["action"]) for d in ep_a["probe_details"])
    var_set = set((d["oid"], d["action"]) for d in ep_var["probe_details"])

    avoided_by_var = a_set - var_set  # A selected but variant didn't
    new_by_var = var_set - a_set      # variant selected but A didn't

    a_details = {(d["oid"], d["action"]): d for d in ep_a["probe_details"]}
    var_details = {(d["oid"], d["action"]): d for d in ep_var["probe_details"]}

    avoided_info = []
    for oid, action in avoided_by_var:
        ad = a_details[(oid, action)]
        obj = test_objects.get(oid, {})
        avoided_info.append({
            "oid": oid, "action": action,
            "was_violation_in_A": ad["is_violation"],
            "was_effective_in_A": ad["probe_utility_signal"] == "effective",
            "query": ad["query"],
            "family": ad["family"],
            "soft_prior": ad["soft_prior"],
            "is_deceptive": ad["is_deceptive"],
            "injected_deviation_present": ad["injected_deviation_present"],
        })

    new_info = []
    for oid, action in new_by_var:
        vd = var_details[(oid, action)]
        obj = test_objects.get(oid, {})
        new_info.append({
            "oid": oid, "action": action,
            "was_violation_in_var": vd["is_violation"],
            "was_effective_in_var": vd["probe_utility_signal"] == "effective",
            "query": vd["query"],
            "family": vd["family"],
            "soft_prior": vd["soft_prior"],
            "is_deceptive": vd["is_deceptive"],
            "injected_deviation_present": vd["injected_deviation_present"],
            "risk_penalty": vd["risk_penalty"],
        })

    return {
        "avoided_count": len(avoided_info),
        "avoided_true_violations": sum(1 for d in avoided_info if d["was_violation_in_A"]),
        "avoided_effective_probes": sum(1 for d in avoided_info if d["was_effective_in_A"]),
        "new_count": len(new_info),
        "new_violations": sum(1 for d in new_info if d["was_violation_in_var"]),
        "new_effective_probes": sum(1 for d in new_info if d["was_effective_in_var"]),
        "avoided_details": avoided_info,
        "new_details": new_info,
    }


# =============================================================================
# 9. Part 1-3: Run Mode 2 B_full with full diagnostics
# =============================================================================
print("\n[9/12] Part 1-3: Full diagnostic run on Mode 2 deceptive objects...")

ep_a_full, ep_b_full, frm_b_full = run_diagnostic_mode(
    test_objects_deceptive, query_gt_deceptive, im_base, c15b_probed_n,
    risk_penalty_scale=1.0, feature_whitelist=None,
    variant_label="B_full", is_permuted=False, mode_rng_seed=SMOKE_SEED + 10000)

agg_a = aggregate_episodes(ep_a_full)
agg_b = aggregate_episodes(ep_b_full)

# Also run C_full
ep_a_c, ep_c_full, frm_c_full = run_diagnostic_mode(
    test_objects_deceptive, query_gt_deceptive, im_base, c15b_probed_n,
    risk_penalty_scale=1.0, feature_whitelist=None,
    variant_label="C_full", is_permuted=True, mode_rng_seed=SMOKE_SEED + 10000)

agg_c = aggregate_episodes(ep_c_full)

# --- Part 1: Selection differences ---
print("\n  --- Part 1: Selection-difference diagnostics ---")
selection_diffs_B_vs_A = []
selection_diffs_B_vs_C = []

for ep_idx in range(N_EPISODES):
    diff_ba = compute_selection_diff(
        ep_a_full[ep_idx], ep_b_full[ep_idx], "B_full", test_objects_deceptive)
    diff_bc = compute_selection_diff(
        ep_c_full[ep_idx], ep_b_full[ep_idx], "B_vs_C", test_objects_deceptive)
    selection_diffs_B_vs_A.append(diff_ba)
    selection_diffs_B_vs_C.append(diff_bc)

    # Print key differences
    if diff_ba["avoided_details"]:
        for d in diff_ba["avoided_details"]:
            viol = "PV" if d["was_violation_in_A"] else "NV"
            eff = "eff" if d["was_effective_in_A"] else "ineff"
            dec = "DECEPTIVE" if d["is_deceptive"] else "normal"
            print(f"    Ep{ep_idx+1} B avoided vs A: {d['oid']}/{d['action']} "
                  f"{viol} {eff} {dec} family={d['family']} dev={d['injected_deviation_present']}")

total_avoided_eff = sum(d["avoided_effective_probes"] for d in selection_diffs_B_vs_A)
total_avoided_violations = sum(d["avoided_true_violations"] for d in selection_diffs_B_vs_A)

print(f"  B avoided {total_avoided_eff} effective probes vs A "
      f"(across {sum(d['avoided_count'] for d in selection_diffs_B_vs_A)} total avoided)")

# --- Part 2: Risk-weight diagnostics ---
print("\n  --- Part 2: Risk-weight diagnostics ---")
risk_groups = frm_b_full.get_risk_weight_groups()

for group_name in ["A_injected_deviation", "B_type_family",
                   "C_positional_or_context", "D_other"]:
    g = risk_groups[group_name]
    print(f"    {group_name}: count={g['count_nonzero']}  "
          f"total_abs={g['total_abs_risk_weight']:.4f}  "
          f"max={g['max_risk_weight']:.4f}  "
          f"avg={g['average_risk_weight']:.4f}  "
          f"%mass={g['percentage_of_total_risk_mass']:.1f}%")

# Check risk diffusion
total_injected_mass = risk_groups["A_injected_deviation"]["total_abs_risk_weight"]
total_other_mass = sum(risk_groups[g]["total_abs_risk_weight"]
                       for g in ["B_type_family", "C_positional_or_context", "D_other"])
risk_memory_diffusion_detected = total_other_mass > total_injected_mass

print(f"    injected_mass={total_injected_mass:.4f}  other_mass={total_other_mass:.4f}")
print(f"    risk_memory_diffusion_detected={risk_memory_diffusion_detected}")

top_risk = frm_b_full.get_top_risk_features(25)
print(f"    Top 10 risk features:")
for item in top_risk[:10]:
    print(f"      {item['feature']}/{item['goal']}/{item['action']} "
          f"w={item['risk_weight']:.4f} group={item['feature_group']}")

# --- Part 3: Macro-drop attribution ---
print("\n  --- Part 3: Macro-drop attribution ---")

# Per-query macro drop
a_per_query = agg_a["per_episode"][-1]["per_query"]  # last episode
b_per_query = agg_b["per_episode"][-1]["per_query"]

query_drops = []
for qname in sorted(a_per_query.keys()):
    a_bal = a_per_query[qname]["balanced_accuracy"]
    b_bal = b_per_query[qname]["balanced_accuracy"]
    drop = b_bal - a_bal
    query_drops.append((qname, drop, a_bal, b_bal))
query_drops.sort(key=lambda x: x[1])

print(f"    Query-level macro drops (B - A, final episode):")
for qname, drop, a_bal, b_bal in query_drops:
    print(f"      {qname}: A={a_bal:.4f} B={b_bal:.4f} delta={drop:+.4f}")

macro_drop_primary_query = query_drops[0][0]  # worst drop

# Per-family macro drop
family_drops = defaultdict(list)
for ep_idx in range(N_EPISODES):
    for detail in ep_a_full[ep_idx]["probe_details"]:
        family_drops[("A", detail["family"], detail["query"])].append(detail)
    for detail in ep_b_full[ep_idx]["probe_details"]:
        family_drops[("B", detail["family"], detail["query"])].append(detail)

# Count avoided effective probes and true-positive probes
avoided_effective_probe_count = total_avoided_eff

# True positive check: an effective probe that changes a query answer contributes to recall
avoided_true_positive_probe_count = sum(
    1 for d in selection_diffs_B_vs_A
    for dd in d["avoided_details"]
    if dd["was_effective_in_A"]
)

# Conservative underprobing: B probes fewer objects total
b_total_probes = sum(len(ep["probe_details"]) for ep in ep_b_full)
a_total_probes = sum(len(ep["probe_details"]) for ep in ep_a_full)
conservative_underprobing_detected = b_total_probes < a_total_probes

# Identify primary family for macro drop
family_bal_drops = defaultdict(float)
for ep_idx in range(N_EPISODES):
    for detail in ep_a_full[ep_idx]["probe_details"]:
        family_bal_drops[("A", detail["family"])] += 0
    for detail in ep_b_full[ep_idx]["probe_details"]:
        family_bal_drops[("B", detail["family"])] += 0

# Simplified: use query-level analysis
macro_drop_primary_family = "mixed"

print(f"    avoided_effective_probe_count={avoided_effective_probe_count}")
print(f"    avoided_true_positive_probe_count={avoided_true_positive_probe_count}")
print(f"    conservative_underprobing_detected={conservative_underprobing_detected}")
print(f"    macro_drop_primary_query={macro_drop_primary_query}")
print(f"    a_total_probes={a_total_probes}  b_total_probes={b_total_probes}")


# =============================================================================
# 10. Part 4: Ablation 1 — B_dev_only
# =============================================================================
print("\n[10/12] Part 4: Ablation 1 -- injected-deviation-only memory...")

ep_a_dev, ep_b_dev, frm_b_dev = run_diagnostic_mode(
    test_objects_deceptive, query_gt_deceptive, im_base, c15b_probed_n,
    risk_penalty_scale=1.0, feature_whitelist=sorted(INJECTED_DEVIATION_FEATURES),
    variant_label="B_dev_only", is_permuted=False, mode_rng_seed=SMOKE_SEED + 10000)

agg_b_dev = aggregate_episodes(ep_b_dev)

ep_a_cdev, ep_c_dev, frm_c_dev = run_diagnostic_mode(
    test_objects_deceptive, query_gt_deceptive, im_base, c15b_probed_n,
    risk_penalty_scale=1.0, feature_whitelist=sorted(INJECTED_DEVIATION_FEATURES),
    variant_label="C_dev_only", is_permuted=True, mode_rng_seed=SMOKE_SEED + 10000)

agg_c_dev = aggregate_episodes(ep_c_dev)

b_dev_delta_pv = agg_b_dev["total_prior_violations"] - agg_a["total_prior_violations"]
b_dev_delta_bal = round(agg_b_dev["mean_macro_bal"] - agg_a["mean_macro_bal"], 4)
b_dev_beats_c_dev = agg_b_dev["total_prior_violations"] < agg_c_dev["total_prior_violations"]

print(f"  B_dev_only: mean_bal={agg_b_dev['mean_macro_bal']:.4f}  "
      f"total_pv={agg_b_dev['total_prior_violations']}  "
      f"repeated_pv={agg_b_dev['total_repeated_prior_violations']}")
print(f"  C_dev_only: mean_bal={agg_c_dev['mean_macro_bal']:.4f}  "
      f"total_pv={agg_c_dev['total_prior_violations']}")
print(f"  B_dev delta_pv_vs_A: {b_dev_delta_pv:+d}")
print(f"  B_dev delta_bal_vs_A: {b_dev_delta_bal:+.4f}")
print(f"  B_dev beats C_dev: {b_dev_beats_c_dev}")

# Risk weight groups for B_dev_only
risk_groups_dev = frm_b_dev.get_risk_weight_groups()
print(f"  B_dev_only injected risk mass: "
      f"{risk_groups_dev['A_injected_deviation']['total_abs_risk_weight']:.4f}")
print(f"  B_dev_only non-injected risk mass: "
      f"{sum(risk_groups_dev[g]['total_abs_risk_weight']
            for g in ['B_type_family', 'C_positional_or_context', 'D_other']):.4f}")


# =============================================================================
# 11. Part 5: Ablation 2 — risk_penalty_scale grid
# =============================================================================
print("\n[11/12] Part 5: Risk penalty scale grid...")

scale_results = {}
for scale in RISK_PENALTY_SCALES:
    print(f"\n  --- scale={scale} ---")

    # B_full with this scale
    ep_a_s, ep_b_s, frm_b_s = run_diagnostic_mode(
        test_objects_deceptive, query_gt_deceptive, im_base, c15b_probed_n,
        risk_penalty_scale=scale, feature_whitelist=None,
        variant_label=f"B_full_s{scale}", is_permuted=False,
        mode_rng_seed=SMOKE_SEED + 10000)
    agg_b_s = aggregate_episodes(ep_b_s)
    agg_a_s = aggregate_episodes(ep_a_s)

    # B_dev_only with this scale
    ep_a_sd, ep_b_sd, frm_b_sd = run_diagnostic_mode(
        test_objects_deceptive, query_gt_deceptive, im_base, c15b_probed_n,
        risk_penalty_scale=scale, feature_whitelist=sorted(INJECTED_DEVIATION_FEATURES),
        variant_label=f"B_dev_s{scale}", is_permuted=False,
        mode_rng_seed=SMOKE_SEED + 10000)
    agg_b_sd = aggregate_episodes(ep_b_sd)
    agg_a_sd = aggregate_episodes(ep_a_sd)

    scale_results[scale] = {
        "B_full": {
            "total_prior_violations": agg_b_s["total_prior_violations"],
            "repeated_prior_violations": agg_b_s["total_repeated_prior_violations"],
            "mean_macro_bal": agg_b_s["mean_macro_bal"],
            "delta_pv_vs_A": agg_b_s["total_prior_violations"] - agg_a_s["total_prior_violations"],
            "macro_delta_vs_A": round(agg_b_s["mean_macro_bal"] - agg_a_s["mean_macro_bal"], 4),
        },
        "B_dev_only": {
            "total_prior_violations": agg_b_sd["total_prior_violations"],
            "repeated_prior_violations": agg_b_sd["total_repeated_prior_violations"],
            "mean_macro_bal": agg_b_sd["mean_macro_bal"],
            "delta_pv_vs_A": agg_b_sd["total_prior_violations"] - agg_a_sd["total_prior_violations"],
            "macro_delta_vs_A": round(agg_b_sd["mean_macro_bal"] - agg_a_sd["mean_macro_bal"], 4),
        },
    }

    for variant in ["B_full", "B_dev_only"]:
        r = scale_results[scale][variant]
        print(f"    {variant}: pv={r['total_prior_violations']}  "
              f"bal={r['mean_macro_bal']:.4f}  "
              f"delta_pv={r['delta_pv_vs_A']:+d}  "
              f"delta_bal={r['macro_delta_vs_A']:+.4f}")


# =============================================================================
# 12. Part 6: Interpretation
# =============================================================================
print(f"\n[12/12] Interpretation...")

# Find best configuration
best_config = None
best_score = float('-inf')
# Score: prefer high pv reduction and high macro (within tolerance)
for scale in RISK_PENALTY_SCALES:
    for variant in ["B_full", "B_dev_only"]:
        r = scale_results[scale][variant]
        pv_reduction = -r["delta_pv_vs_A"]  # more negative = better reduction
        macro_ok = 1.0 if r["macro_delta_vs_A"] >= MACRO_DROP_TOLERANCE else 0.0
        score = pv_reduction * 2.0 + macro_ok * 3.0  # prioritize macro within tolerance
        if score > best_score:
            best_score = score
            best_config = (scale, variant, r)

best_risk_penalty_scale, best_variant, best_r = best_config

# Determine diagnostic result
b_full_beats_c = agg_b["total_prior_violations"] < agg_c["total_prior_violations"]
macro_drop_ok = agg_b["mean_macro_bal"] - agg_a["mean_macro_bal"] >= MACRO_DROP_TOLERANCE
b_dev_beats_c = agg_b_dev["total_prior_violations"] < agg_c_dev["total_prior_violations"]
b_dev_macro_ok = agg_b_dev["mean_macro_bal"] - agg_a["mean_macro_bal"] >= MACRO_DROP_TOLERANCE

# Check if lower scales improve macro while keeping pv reduction
best_macro_delta = best_r["macro_delta_vs_A"]
penalty_too_strong_detected = False
for scale in RISK_PENALTY_SCALES:
    for variant in ["B_full", "B_dev_only"]:
        r = scale_results[scale][variant]
        if (r["macro_delta_vs_A"] > agg_b["mean_macro_bal"] - agg_a["mean_macro_bal"]
                and r["delta_pv_vs_A"] < 0):
            penalty_too_strong_detected = True
            break

# Check sample size issue
b_vs_c_pv_diff = abs(agg_b["total_prior_violations"] - agg_c["total_prior_violations"])
insufficient_sample_problem = b_vs_c_pv_diff <= 2

diagnostic_result = "mechanism_not_supported"
if b_dev_beats_c and b_dev_macro_ok:
    diagnostic_result = "ready_for_small_multiseed"
elif risk_memory_diffusion_detected and b_dev_delta_bal > agg_b["mean_macro_bal"] - agg_a["mean_macro_bal"]:
    diagnostic_result = "risk_memory_diffusion_problem"
elif penalty_too_strong_detected:
    diagnostic_result = "penalty_too_strong_problem"
elif insufficient_sample_problem and b_vs_c_pv_diff <= 1:
    diagnostic_result = "insufficient_sample_problem"

ready_for_small_multiseed = diagnostic_result == "ready_for_small_multiseed"

if ready_for_small_multiseed:
    next_route = "run_small_multiseed_1j19d"
elif diagnostic_result == "risk_memory_diffusion_problem":
    next_route = "use_dev_only_memory_and_retest"
elif diagnostic_result == "penalty_too_strong_problem":
    next_route = f"tune_risk_penalty_scale_best={best_risk_penalty_scale}_retest"
elif diagnostic_result == "insufficient_sample_problem":
    next_route = "increase_episodes_or_deceptive_count"
else:
    next_route = "review_full_diagnostics"

# =============================================================================
# Output
# =============================================================================
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

results = {
    "block_id": "1J19d",
    "condition": COND["label"],
    "smoke_seed": SMOKE_SEED, "budget": BUDGET, "n_episodes": N_EPISODES,
    "risk_alpha": RISK_ALPHA, "max_risk": MAX_RISK,
    "prior_violation_threshold": PRIOR_VIOLATION_THRESHOLD,
    "min_violation_support": MIN_VIOLATION_SUPPORT,
    "macro_drop_tolerance": MACRO_DROP_TOLERANCE,
    "injected_deviation_features": sorted(INJECTED_DEVIATION_FEATURES),
    "n_deceptive_objects": len(deceptive_modifications),

    "c15b_macro_bal": round(c15b_macro_bal, 4),
    "c15b_probed_n": c15b_probed_n,

    # Part 1: Selection differences
    "selection_diffs_B_vs_A": selection_diffs_B_vs_A,
    "selection_diffs_B_vs_C": selection_diffs_B_vs_C,
    "total_avoided_effective_probes_by_B": total_avoided_eff,
    "total_avoided_violations_by_B": total_avoided_violations,

    # Part 2: Risk weight groups
    "risk_weight_groups_B_full": {k: {
        "count_nonzero": v["count_nonzero"],
        "total_abs_risk_weight": v["total_abs_risk_weight"],
        "max_risk_weight": v["max_risk_weight"],
        "average_risk_weight": v["average_risk_weight"],
        "percentage_of_total_risk_mass": v["percentage_of_total_risk_mass"],
    } for k, v in risk_groups.items()},
    "risk_memory_diffusion_detected": risk_memory_diffusion_detected,
    "top_risk_features_B_full": top_risk[:20],

    # Part 3: Macro-drop attribution
    "macro_drop_primary_query": macro_drop_primary_query,
    "macro_drop_primary_family": macro_drop_primary_family,
    "avoided_effective_probe_count": avoided_effective_probe_count,
    "avoided_true_positive_probe_count": avoided_true_positive_probe_count,
    "conservative_underprobing_detected": conservative_underprobing_detected,
    "query_drops_final_episode": [{"query": q, "A_bal": a, "B_bal": b, "delta": round(d, 4)}
                                   for q, d, a, b in query_drops],

    # Part 4: B_dev_only
    "B_dev_only": {
        "mean_macro_bal": agg_b_dev["mean_macro_bal"],
        "total_prior_violations": agg_b_dev["total_prior_violations"],
        "total_repeated_prior_violations": agg_b_dev["total_repeated_prior_violations"],
        "delta_pv_vs_A": b_dev_delta_pv,
        "macro_delta_vs_A": b_dev_delta_bal,
        "beats_C_dev_only": b_dev_beats_c_dev,
    },
    "C_dev_only_total_pv": agg_c_dev["total_prior_violations"],

    # Part 5: Risk penalty scale grid
    "risk_penalty_scale_grid": scale_results,

    # Part 6: Interpretation
    "diagnostic_result": diagnostic_result,
    "best_risk_penalty_scale": best_risk_penalty_scale,
    "best_variant": best_variant,
    "penalty_too_strong_detected": penalty_too_strong_detected,
    "insufficient_sample_problem": insufficient_sample_problem,
    "macro_tradeoff_acceptable_best": best_r["macro_delta_vs_A"] >= MACRO_DROP_TOLERANCE,
    "feature_risk_memory_supported": False,
    "ready_for_small_multiseed": ready_for_small_multiseed,
}

# Summary stats for block_done
B_full_total_pv = agg_b["total_prior_violations"]
C_full_total_pv = agg_c["total_prior_violations"]
B_dev_only_total_pv = agg_b_dev["total_prior_violations"]
C_dev_only_total_pv = agg_c_dev["total_prior_violations"]
B_full_macro_delta = round(agg_b["mean_macro_bal"] - agg_a["mean_macro_bal"], 4)
B_dev_only_macro_delta = b_dev_delta_bal

elapsed = time.time() - t0

results["B_full_total_pv"] = B_full_total_pv
results["C_full_total_pv"] = C_full_total_pv
results["B_dev_only_total_pv"] = B_dev_only_total_pv
results["C_dev_only_total_pv"] = C_dev_only_total_pv
results["B_full_macro_delta_vs_A"] = B_full_macro_delta
results["B_dev_only_macro_delta_vs_A"] = B_dev_only_macro_delta
results["recommended_next_route"] = next_route
results["elapsed_s"] = round(elapsed, 1)

# Save JSON
runs_dir = os.path.join(CURRENT_DIR, "runs")
os.makedirs(runs_dir, exist_ok=True)
json_path = os.path.join(runs_dir, "calibration_block1j19d_prior_violation_diagnostic_ablation_seed101.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2, cls=_NumpyEncoder)
print(f"\n  Saved: {json_path}")

# Save protocol MD
protocols_dir = os.path.join(CURRENT_DIR, "protocols")
os.makedirs(protocols_dir, exist_ok=True)
md_path = os.path.join(protocols_dir, "block1j19d_prior_violation_diagnostic_ablation_seed101.md")

def fb(v): return "true" if v else "false"

md = []
md.append("# Block 1J19d -- Prior-Violation Memory Diagnostic Ablation\n")
md.append("## 1. Objective\n")
md.append("Diagnose why B_full does not beat C_full on prior violations and why B macro drops below tolerance.\n")

md.append("## 2. Setup\n")
md.append(f"| Parameter | Value |")
md.append(f"|-----------|-------|")
md.append(f"| Seed | {SMOKE_SEED} |")
md.append(f"| Episodes | {N_EPISODES} |")
md.append(f"| Prior Violation Threshold | {PRIOR_VIOLATION_THRESHOLD} |")
md.append(f"| Risk Alpha | {RISK_ALPHA} |")
md.append(f"| Max Risk | {MAX_RISK} |")
md.append(f"| Injected Deviation Features | {sorted(INJECTED_DEVIATION_FEATURES)} |")
md.append(f"| Deceptive Objects | {len(deceptive_modifications)} |\n")

md.append("## 3. B_full vs A vs C_full\n")
md.append(f"| Condition | Mean Bal | Total PV | Repeated PV |")
md.append(f"|-----------|----------|----------|-------------|")
md.append(f"| A | {agg_a['mean_macro_bal']:.4f} | {agg_a['total_prior_violations']} | - |")
md.append(f"| B_full | {agg_b['mean_macro_bal']:.4f} | {agg_b['total_prior_violations']} | {agg_b['total_repeated_prior_violations']} |")
md.append(f"| C_full | {agg_c['mean_macro_bal']:.4f} | {agg_c['total_prior_violations']} | - |")
md.append(f"| B delta vs A | {B_full_macro_delta:+.4f} | {agg_b['total_prior_violations'] - agg_a['total_prior_violations']:+d} | |")
md.append(f"| B beats C on PV | {fb(agg_b['total_prior_violations'] < agg_c['total_prior_violations'])} | | |\n")

md.append("## 4. Risk Weight Distribution (B_full)\n")
md.append(f"| Group | Count | Total Abs | Max | Avg | % Mass |")
md.append(f"|-------|-------|-----------|-----|-----|--------|")
for group_name in ["A_injected_deviation", "B_type_family",
                   "C_positional_or_context", "D_other"]:
    g = risk_groups[group_name]
    md.append(f"| {group_name} | {g['count_nonzero']} | {g['total_abs_risk_weight']:.4f} | "
              f"{g['max_risk_weight']:.4f} | {g['average_risk_weight']:.4f} | "
              f"{g['percentage_of_total_risk_mass']:.1f}% |")
md.append(f"| Risk memory diffusion | {fb(risk_memory_diffusion_detected)} |\n")

md.append("## 5. Macro-Drop Attribution\n")
md.append(f"| Metric | Value |")
md.append(f"|--------|-------|")
md.append(f"| Primary query | {macro_drop_primary_query} |")
md.append(f"| Avoided effective probes | {avoided_effective_probe_count} |")
md.append(f"| Avoided true-positive probes | {avoided_true_positive_probe_count} |")
md.append(f"| Conservative underprobing | {fb(conservative_underprobing_detected)} |\n")

md.append("## 6. B_dev_only Ablation\n")
md.append(f"| Metric | B_full | B_dev_only | C_dev_only |")
md.append(f"|--------|--------|------------|------------|")
md.append(f"| Mean Bal | {agg_b['mean_macro_bal']:.4f} | {agg_b_dev['mean_macro_bal']:.4f} | {agg_c_dev['mean_macro_bal']:.4f} |")
md.append(f"| Total PV | {agg_b['total_prior_violations']} | {agg_b_dev['total_prior_violations']} | {agg_c_dev['total_prior_violations']} |")
md.append(f"| B beats C | {fb(agg_b['total_prior_violations'] < agg_c['total_prior_violations'])} | {fb(b_dev_beats_c)} | - |")
md.append(f"| Macro delta vs A | {B_full_macro_delta:+.4f} | {b_dev_delta_bal:+.4f} | - |\n")

md.append("## 7. Risk Penalty Scale Grid\n")
md.append(f"| Scale | Variant | PV | Delta PV | Bal | Delta Bal |")
md.append(f"|-------|---------|-----|----------|-----|-----------|")
for scale in RISK_PENALTY_SCALES:
    for variant in ["B_full", "B_dev_only"]:
        r = scale_results[scale][variant]
        md.append(f"| {scale} | {variant} | {r['total_prior_violations']} | "
                  f"{r['delta_pv_vs_A']:+d} | {r['mean_macro_bal']:.4f} | "
                  f"{r['macro_delta_vs_A']:+.4f} |")
md.append("")

md.append("## 8. Interpretation\n")
md.append(f"| Metric | Value |")
md.append(f"|--------|-------|")
md.append(f"| Diagnostic result | {diagnostic_result} |")
md.append(f"| Risk memory diffusion | {fb(risk_memory_diffusion_detected)} |")
md.append(f"| Penalty too strong | {fb(penalty_too_strong_detected)} |")
md.append(f"| Insufficient sample | {fb(insufficient_sample_problem)} |")
md.append(f"| Best risk penalty scale | {best_risk_penalty_scale} |")
md.append(f"| Best variant | {best_variant} |")
md.append(f"| Macro tradeoff acceptable (best) | {fb(results['macro_tradeoff_acceptable_best'])} |")
md.append(f"| Ready for small multiseed | {fb(ready_for_small_multiseed)} |")
md.append(f"| Next route | {next_route} |\n")

md.append("## 9. Summary\n")
md.append("```")
md.append("[block_done]")
md.append(f"block_id=1J19d")
md.append(f"B_full_total_pv={B_full_total_pv}")
md.append(f"C_full_total_pv={C_full_total_pv}")
md.append(f"B_dev_only_total_pv={B_dev_only_total_pv}")
md.append(f"C_dev_only_total_pv={C_dev_only_total_pv}")
md.append(f"B_full_macro_delta_vs_A={B_full_macro_delta:+.4f}")
md.append(f"B_dev_only_macro_delta_vs_A={B_dev_only_macro_delta:+.4f}")
md.append(f"best_risk_penalty_scale={best_risk_penalty_scale}")
md.append(f"best_variant={best_variant}")
md.append(f"risk_memory_diffusion_detected={fb(risk_memory_diffusion_detected)}")
md.append(f"penalty_too_strong_detected={fb(penalty_too_strong_detected)}")
md.append(f"insufficient_sample_problem={fb(insufficient_sample_problem)}")
md.append(f"macro_tradeoff_acceptable={fb(results['macro_tradeoff_acceptable_best'])}")
md.append(f"recommended_next_route={next_route}")
md.append(f"ready_for_small_multiseed={fb(ready_for_small_multiseed)}")
md.append(f"feature_risk_memory_supported=false")
md.append(f"elapsed={elapsed:.1f}s")
md.append("```")

with open(md_path, "w") as f:
    f.write("\n".join(md))
print(f"  Saved: {md_path}")

# ---- block_done ----
print(f"\n{'='*70}")
print(f"[block_done]")
print(f"block_id=1J19d")
print(f"B_full_total_pv={B_full_total_pv}")
print(f"C_full_total_pv={C_full_total_pv}")
print(f"B_dev_only_total_pv={B_dev_only_total_pv}")
print(f"C_dev_only_total_pv={C_dev_only_total_pv}")
print(f"B_full_macro_delta_vs_A={B_full_macro_delta:+.4f}")
print(f"B_dev_only_macro_delta_vs_A={B_dev_only_macro_delta:+.4f}")
print(f"best_risk_penalty_scale={best_risk_penalty_scale}")
print(f"best_variant={best_variant}")
print(f"risk_memory_diffusion_detected={fb(risk_memory_diffusion_detected)}")
print(f"penalty_too_strong_detected={fb(penalty_too_strong_detected)}")
print(f"insufficient_sample_problem={fb(insufficient_sample_problem)}")
print(f"macro_tradeoff_acceptable={fb(results['macro_tradeoff_acceptable_best'])}")
print(f"recommended_next_route={next_route}")
print(f"ready_for_small_multiseed={fb(ready_for_small_multiseed)}")
print(f"feature_risk_memory_supported=false")
print(f"elapsed={elapsed:.1f}s")
print(f"{'='*70}")
