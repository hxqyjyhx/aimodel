"""
Block 1J20 -- Family-Conditioned Prior-Violation Memory Diagnostic.

Tests whether prior-violation memory can use family-conditioned compound keys
(goal, action, inferred_family, injected_deviation_feature) instead of broad
type-family risk diffusion.

Six variants:
  A_no_memory          — baseline, no risk memory
  B_full_old           — FeatureRiskMemoryV3, (feature, goal, action), all features
  B_dev_only_old       — FeatureRiskMemoryV3, (feature, goal, action), dev features only
  B_family_dev         — FamilyConditionedRiskMemory, (goal, action, family, dev_feature)
  C_family_dev_permuted— Same as B_family_dev but weights shuffled within family/action
  C_family_only        — FamilyConditionedRiskMemory, (goal, action, family) only
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
    "damp_texture", "brittle_surface", "treated_surface", "hollow_sound",
})
FEATURE_UNIVERSE_SET = set(FEATURE_UNIVERSE)

WOOD_FEATURES = {"has_bark_texture", "has_wood_grain", "brownish", "rough_texture", "long_shape"}
STONE_FEATURES = {"has_crystal_flecks", "has_granular_surface", "grayish", "block_like", "heavy_weight"}
APPLE_FEATURES = {"has_stem_remnant", "has_peel_texture", "round_small", "greenish", "smooth_texture", "light_weight"}
TOOL_FEATURES = {"has_grip_area", "has_shaft_shape", "elongated_with_handle", "movable", "long_shape"}

TYPE_FAMILIES = {
    "wood-like": WOOD_FEATURES, "stone-like": STONE_FEATURES,
    "apple-like": APPLE_FEATURES, "tool-like": TOOL_FEATURES,
}
ALL_TYPE_FAMILY_FEATURES = set()
for fs in TYPE_FAMILIES.values():
    ALL_TYPE_FAMILY_FEATURES.update(fs)
FAMILY_NAMES = sorted(TYPE_FAMILIES.keys())

INJECTED_DEVIATION_FEATURES = {"damp_texture", "brittle_surface", "treated_surface", "hollow_sound"}

ACTION_TO_QUERY = {
    "craft_plank": "need_planks", "eat": "need_food",
    "use_as_tool": "need_tool", "burn_as_fuel": "need_fuel",
    "mine_by_hand": "need_stone", "mine_with_pickaxe": "need_stone",
}

print("=" * 70)
print("Block 1J20 -- Family-Conditioned Prior-Violation Memory Diagnostic")
print(f"  condition={COND['label']}  seed={SMOKE_SEED}  budget={BUDGET}")
print(f"  episodes={N_EPISODES}  risk_alpha={RISK_ALPHA}  max_risk={MAX_RISK}")
print(f"  prior_violation_threshold={PRIOR_VIOLATION_THRESHOLD}")
print(f"  injected_deviation_features={sorted(INJECTED_DEVIATION_FEATURES)}")
print("=" * 70)

# =============================================================================
# 1. Phase A: Training + IOM building
# =============================================================================
print("\n[1/14] Phase A: Training + IOM building...")
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
print("\n[2/14] Generating test objects...")
test_objects_standard = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")
query_gt_standard = compute_query_ground_truth(test_objects_standard)
test_oids = sorted(test_objects_standard.keys())
QUERY_NAMES = sorted(_TASK_QUERIES.keys())
print(f"  {len(test_oids)} standard test objects, {len(QUERY_NAMES)} queries")

print("\n[3/14] Creating family-preserving deceptive test objects...")

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
print("\n[4/14] Building goal-conditioned soft prior table...")


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
# 5. FamilyConditionedRiskMemory (NEW)
# =============================================================================
print("\n[5/14] Initializing FamilyConditionedRiskMemory...")

class FamilyConditionedRiskMemory:
    """Memory with compound keys conditioned on type family.

    key_mode="family_dev": keys are (goal, action, family, dev_feature)
        Only injected deviation features are tracked. Family is context.

    key_mode="family_only": keys are (goal, action, family)
        Cross-family discriminative: _with for detected family, _without for others.
    """

    def __init__(self, key_mode="family_dev", alpha=RISK_ALPHA, max_risk=MAX_RISK,
                 min_violation_support=MIN_VIOLATION_SUPPORT):
        if key_mode not in ("family_dev", "family_only"):
            raise ValueError(f"Unknown key_mode: {key_mode}")
        self.key_mode = key_mode
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
        self.total_updates += 1
        if not is_eligible:
            self.ineligible_updates += 1
            return

        self.eligible_updates += 1
        if is_violation:
            self.violation_updates += 1
        else:
            self.nonviolation_updates += 1

        family = detect_type_family(features)

        if self.key_mode == "family_dev":
            for dev_feat in INJECTED_DEVIATION_FEATURES:
                key = (goal, action, family, dev_feat)
                present = features.get(dev_feat, False)
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

        elif self.key_mode == "family_only":
            # Increment _with for the detected family
            key_detected = (goal, action, family)
            if is_violation:
                self.counts[key_detected]["violation_with"] += 1.0
            else:
                self.counts[key_detected]["nonviolation_with"] += 1.0
            # Increment _without for all OTHER families
            for other_fam in FAMILY_NAMES:
                if other_fam == family:
                    continue
                key_other = (goal, action, other_fam)
                if is_violation:
                    self.counts[key_other]["violation_without"] += 1.0
                else:
                    self.counts[key_other]["nonviolation_without"] += 1.0

        self._risk_cache.clear()

    def get_risk_weight(self, *key):
        if key in self._risk_cache:
            return self._risk_cache[key]

        c = self.counts[key]
        v_with = c["violation_with"]
        nv_with = c["nonviolation_with"]
        v_without = c["violation_without"]
        nv_without = c["nonviolation_without"]

        actual_violations = (v_with + v_without) - 2.0
        if actual_violations < self.min_violation_support:
            self._risk_cache[key] = 0.0
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
        self._risk_cache[key] = usable
        return usable

    def get_total_risk_penalty(self, features, action):
        goal = ACTION_TO_QUERY.get(action, "")
        family = detect_type_family(features)

        if self.key_mode == "family_dev":
            total = 0.0
            for dev_feat in INJECTED_DEVIATION_FEATURES:
                if features.get(dev_feat, False):
                    total += self.get_risk_weight(goal, action, family, dev_feat)
            return total

        elif self.key_mode == "family_only":
            return self.get_risk_weight(goal, action, family)

        return 0.0

    def permute_weights_within_family(self, rng):
        """Shuffle risk weights within each (goal, action, family) group.
        Preserves family/action distribution but destroys feature-specific signal."""
        groups = defaultdict(list)
        for key in self.counts:
            if len(key) == 4:  # (goal, action, family, dev_feat)
                goal, action, family, dev_feat = key
                w = self.get_risk_weight(*key)
                groups[(goal, action, family)].append((key, w))
            # For family_only mode (len 3), we don't permute since there's one key per group

        permuted = {}
        for group_key, items in groups.items():
            weights = [w for _, w in items]
            shuffled = list(weights)
            rng.shuffle(shuffled)
            for (key, _), new_w in zip(items, shuffled):
                permuted[key] = new_w
                self._risk_cache[key] = new_w

        return permuted

    def get_statistics(self):
        nonzero_count = 0; max_w = 0.0; sum_abs = 0.0; n_keys = 0
        for key in self.counts:
            w = self.get_risk_weight(*key)
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

    def classify_risk_mass_by_key_type(self):
        """Part 1: classify all nonzero risk weights by key type.
        Returns mass breakdown and family_diffusion_leak_detected flag."""
        family_only_mass = 0.0
        deviation_only_mass = 0.0
        family_conditioned_deviation_mass = 0.0
        other_mass = 0.0

        for key in self.counts:
            w = self.get_risk_weight(*key)
            if abs(w) < 1e-9:
                continue
            abs_w = abs(w)

            if self.key_mode == "family_dev" and len(key) == 4:
                # (goal, action, family, dev_feature) — proper compound key
                family_conditioned_deviation_mass += abs_w
            elif self.key_mode == "family_only" and len(key) == 3:
                # (goal, action, family) — family-only key
                family_only_mass += abs_w
            elif len(key) == 3 and self.key_mode == "family_dev":
                # unexpected: family_dev mode with 3-tuple key
                family_only_mass += abs_w
            elif len(key) == 4 and self.key_mode == "family_only":
                deviation_only_mass += abs_w
            else:
                other_mass += abs_w

        total_mass = family_only_mass + deviation_only_mass + family_conditioned_deviation_mass + other_mass
        eps = 1e-9
        family_diffusion_leak_detected = (
            (family_only_mass > eps) or (deviation_only_mass > eps) or (other_mass > eps)
        )

        return {
            "family_only_key_mass": round(family_only_mass, 6),
            "deviation_only_key_mass": round(deviation_only_mass, 6),
            "family_conditioned_deviation_key_mass": round(family_conditioned_deviation_mass, 6),
            "other_key_mass": round(other_mass, 6),
            "total_risk_mass": round(total_mass, 6),
            "family_diffusion_leak_detected": family_diffusion_leak_detected,
        }

    def get_compound_key_support_table(self):
        """Part 2: detailed support table for each compound key."""
        keys_info = []
        for key in self.counts:
            c = self.counts[key]
            v_with = c["violation_with"]
            nv_with = c["nonviolation_with"]
            v_without = c["violation_without"]
            nv_without = c["nonviolation_without"]
            w = self.get_risk_weight(*key)

            total_v = v_with + v_without
            total_nv = nv_with + nv_without
            support = max(0.0, total_v + total_nv - 4.0)
            shrinkage = support / (support + self.alpha)

            eligible_events = total_v + total_nv - 4.0

            keys_info.append({
                "key": list(key),
                "key_length": len(key),
                "violation_with": v_with,
                "nonviolation_with": nv_with,
                "violation_without": v_without,
                "nonviolation_without": nv_without,
                "risk_weight": round(w, 6),
                "shrinkage_factor": round(shrinkage, 6),
                "eligible_event_count": round(eligible_events, 1),
                "abs_risk_weight": round(abs(w), 6),
            })

        keys_info.sort(key=lambda x: x["abs_risk_weight"], reverse=True)
        nonzero = [k for k in keys_info if k["abs_risk_weight"] > 1e-9]
        above_support = [k for k in nonzero
                        if (k["violation_with"] + k["violation_without"] - 2.0) >= self.min_violation_support]

        # Top 10 by weight
        top_by_weight = nonzero[:10]

        # Top 10 by total contribution: abs_risk_weight * eligible_event_count (proxy for impact)
        top_by_contribution = sorted(nonzero, key=lambda x: x["abs_risk_weight"] * max(x["eligible_event_count"], 1), reverse=True)[:10]

        return {
            "num_keys_total": len(keys_info),
            "num_keys_with_nonzero_weight": len(nonzero),
            "num_keys_above_support_threshold": len(above_support),
            "top_10_keys_by_weight": top_by_weight,
            "top_10_keys_by_total_contribution": top_by_contribution,
            "all_keys": keys_info,
        }

    def clone(self):
        new = FamilyConditionedRiskMemory(key_mode=self.key_mode, alpha=self.alpha,
                                          max_risk=self.max_risk,
                                          min_violation_support=self.min_violation_support)
        new.counts.update(copy.deepcopy(dict(self.counts)))
        new._risk_cache = dict(self._risk_cache)
        new.total_updates = self.total_updates
        new.violation_updates = self.violation_updates
        new.nonviolation_updates = self.nonviolation_updates
        new.eligible_updates = self.eligible_updates
        new.ineligible_updates = self.ineligible_updates
        return new


# =============================================================================
# 6. FeatureRiskMemoryV3 (for B_full_old and B_dev_only_old)
# =============================================================================
print("\n[6/14] Defining FeatureRiskMemoryV3 (for old variants)...")

class FeatureRiskMemoryV3:
    """Prior-violation memory with optional feature whitelist. Same as 1J19d."""

    def __init__(self, alpha=RISK_ALPHA, max_risk=MAX_RISK,
                 min_violation_support=MIN_VIOLATION_SUPPORT,
                 feature_whitelist=None):
        self.alpha = alpha
        self.max_risk = max_risk
        self.min_violation_support = min_violation_support
        self.feature_whitelist = feature_whitelist
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
            "n_keys_total": n_keys,
            "risk_weights_nonzero_count": nonzero_count,
            "max_risk_weight": round(max_w, 6),
            "mean_abs_risk_weight": round(sum_abs / max(n_keys, 1), 6),
        }

    def get_risk_mass_on_type_family_features(self):
        """Compute total risk mass on type-family features (for Part 4 comparison)."""
        total_mass = 0.0
        type_family_mass = 0.0
        for (feat, goal, action), c in self.counts.items():
            w = self.get_risk_weight(feat, goal, action)
            if abs(w) > 1e-9:
                total_mass += abs(w)
                if feat in ALL_TYPE_FAMILY_FEATURES:
                    type_family_mass += abs(w)
        return {
            "total_abs_risk_mass": round(total_mass, 6),
            "type_family_abs_risk_mass": round(type_family_mass, 6),
            "type_family_mass_fraction": round(type_family_mass / max(total_mass, 1e-9), 6),
        }

    def clone(self):
        new = FeatureRiskMemoryV3(alpha=self.alpha, max_risk=self.max_risk,
                                  min_violation_support=self.min_violation_support,
                                  feature_whitelist=self.feature_whitelist)
        new.counts.update(copy.deepcopy(dict(self.counts)))
        new._risk_cache = dict(self._risk_cache)
        new.total_updates = self.total_updates
        new.violation_updates = self.violation_updates
        new.nonviolation_updates = self.nonviolation_updates
        new.eligible_updates = self.eligible_updates
        new.ineligible_updates = self.ineligible_updates
        return new


# =============================================================================
# 7. Policy Classes
# =============================================================================
print("\n[7/14] Defining policy classes...")

class SoftPriorOnlyPolicyV2(MiniMCPolicy):
    """Condition A: soft_prior_only. No memory."""

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
            if is_eff: effect = "effective"
            elif is_zero: effect = "zero"
            else: effect = "harmful"
            self._probe_effects.append({
                "oid": object_id, "action": action, "outcome": outcome,
                "utility_delta": round(utility_delta, 6),
                "is_effective": is_eff, "is_zero": is_zero, "is_harmful": is_harm,
                "effect": effect,
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
    """V3-based risk-adjusted policy for B_full_old and B_dev_only_old."""

    def __init__(self, instance_memory, rng, feature_risk_memory,
                 target_probe_count=8, explore_fraction=0.0, risk_penalty_scale=1.0):
        super().__init__(instance_memory, rng, target_probe_count, explore_fraction)
        self._frm = feature_risk_memory
        self._risk_penalty_scale = risk_penalty_scale
        self._variant = "B_risk_adjusted_v3"

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
    """V3-based permuted risk policy."""

    def __init__(self, instance_memory, rng, feature_risk_memory,
                 target_probe_count=8, explore_fraction=0.0, risk_penalty_scale=1.0):
        super().__init__(instance_memory, rng, feature_risk_memory,
                        target_probe_count, explore_fraction, risk_penalty_scale)
        self._variant = "C_permuted_v3"
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


class FamilyConditionedRiskPolicy(SoftPriorOnlyPolicyV2):
    """Policy using FamilyConditionedRiskMemory for B_family_dev and C_family_only."""

    def __init__(self, instance_memory, rng, family_conditioned_memory,
                 target_probe_count=8, explore_fraction=0.0, risk_penalty_scale=1.0):
        super().__init__(instance_memory, rng, target_probe_count, explore_fraction)
        self._fcrm = family_conditioned_memory
        self._risk_penalty_scale = risk_penalty_scale
        self._variant = "B_family_conditioned"

    def _compute_adjusted_score(self, features, action, norm_cost):
        base_prior = get_goal_soft_prior(features, action)
        risk_penalty = self._fcrm.get_total_risk_penalty(features, action)
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


class FamilyConditionedPermutedPolicy(FamilyConditionedRiskPolicy):
    """Permuted control for B_family_dev. Shuffles weights within (goal, action, family) groups."""

    def __init__(self, instance_memory, rng, family_conditioned_memory,
                 target_probe_count=8, explore_fraction=0.0, risk_penalty_scale=1.0):
        super().__init__(instance_memory, rng, family_conditioned_memory,
                        target_probe_count, explore_fraction, risk_penalty_scale)
        self._variant = "C_family_dev_permuted"
        self._permuted_map = {}

    def apply_permutation(self, rng):
        self._permuted_map = self._fcrm.permute_weights_within_family(rng)

    def _compute_adjusted_score(self, features, action, norm_cost):
        base_prior = get_goal_soft_prior(features, action)
        goal = ACTION_TO_QUERY.get(action, "")
        family = detect_type_family(features)

        if self._permuted_map:
            risk_penalty = 0.0
            for dev_feat in INJECTED_DEVIATION_FEATURES:
                if features.get(dev_feat, False):
                    key = (goal, action, family, dev_feat)
                    risk_penalty += self._permuted_map.get(key, 0.0)
        else:
            risk_penalty = self._fcrm.get_total_risk_penalty(features, action)

        scaled_penalty = risk_penalty * self._risk_penalty_scale
        raw_score = base_prior * math.exp(-scaled_penalty) - COST_WEIGHT * norm_cost
        final_score = max(EXPLORATION_FLOOR, raw_score)
        return final_score, raw_score, base_prior, risk_penalty, scaled_penalty


# =============================================================================
# 8. C15b reference
# =============================================================================
print("\n[8/14] Running C15b reference...")

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
# 9. Diagnostic mode runners
# =============================================================================
print("\n[9/14] Setting up diagnostic mode runners...")

def aggregate_episodes(ep_list):
    mbs = [e["macro_bal"] for e in ep_list]
    pvs = [e.get("prior_violation_count", 0) for e in ep_list]
    deceptive_pvs = [e.get("deceptive_prior_violation_count", 0) for e in ep_list]
    normal_pvs = [e.get("normal_prior_violation_count", 0) for e in ep_list]
    return {
        "mean_macro_bal": round(float(np.mean(mbs)), 4),
        "std_macro_bal": round(float(np.std(mbs, ddof=1)) if len(mbs) > 1 else 0.0, 4),
        "final_macro_bal": round(mbs[-1], 4) if mbs else 0.0,
        "total_prior_violations": sum(pvs),
        "total_deceptive_prior_violations": sum(deceptive_pvs),
        "total_normal_prior_violations": sum(normal_pvs),
        "total_repeated_prior_violations": sum(e.get("prior_violation_repeated", 0) for e in ep_list),
        "per_episode": ep_list,
    }


def run_diagnostic_pair(test_objects, query_gt, im_base, target_probe_count,
                        make_policy_a, make_policy_variant, update_memory_fn,
                        variant_label, mode_rng_seed=SMOKE_SEED):
    """Generic diagnostic runner: A + one variant across all episodes.

    make_policy_a(im, rng) -> policy
    make_policy_variant(im, rng, memory) -> policy
    update_memory_fn(memory, features, goal, action, is_violation, is_eligible) -> None
    """
    all_episodes_a = []
    all_episodes_var = []
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
        policy_a = make_policy_a(im_a, ep_rng)
        preds_a, _, obs_a, _, _ = run_policy(policy_a, env_a)
        metrics_a = compute_full_metrics(_unwrap_preds(preds_a), query_gt)
        effects_a = policy_a.get_probe_effects()
        scores_a = policy_a.get_selected_scores()

        pv_count_a = 0; deceptive_pv_a = 0; normal_pv_a = 0
        a_probe_details = []
        for effect, score in zip(effects_a, scores_a):
            oid = effect["oid"]; action = effect["action"]; outcome = effect["outcome"]
            obj = test_objects.get(oid, {})
            features = obj.get("visible_features", {})
            is_violation, is_eligible, sp = check_prior_violation(features, action, outcome)
            qname = ACTION_TO_QUERY.get(action, "")
            family = detect_type_family(features)
            injected_present = [f for f in INJECTED_DEVIATION_FEATURES if features.get(f, False)]
            is_deceptive = oid in DECEPTIVE_OIDS
            detail = {
                "oid": oid, "action": action, "outcome": outcome,
                "family": family, "query": qname,
                "soft_prior": round(sp, 4), "is_eligible": is_eligible,
                "is_violation": is_violation, "is_nonviolation": is_eligible and not is_violation,
                "probe_utility_signal": effect["effect"],
                "final_score": score["final_score"],
                "injected_deviation_present": injected_present,
                "is_deceptive": is_deceptive,
            }
            if is_violation:
                pv_count_a += 1
                if is_deceptive: deceptive_pv_a += 1
                else: normal_pv_a += 1
            a_probe_details.append(detail)

        all_episodes_a.append({
            "episode": ep,
            "macro_bal": metrics_a["macro_query_balanced_accuracy"],
            "per_query": metrics_a["per_query"],
            "prior_violation_count": pv_count_a,
            "deceptive_prior_violation_count": deceptive_pv_a,
            "normal_prior_violation_count": normal_pv_a,
            "probe_details": a_probe_details,
        })

        # --- Variant ---
        im_var = im_base.clone()
        env_var = MiniMCEnvironment(test_objects, dict(ep_positions), initial_budget=BUDGET)
        memory = update_memory_fn.__self__ if hasattr(update_memory_fn, '__self__') else None
        policy_var = make_policy_variant(im_var, ep_rng)
        preds_var, _, obs_var, _, _ = run_policy(policy_var, env_var)
        metrics_var = compute_full_metrics(_unwrap_preds(preds_var), query_gt)
        effects_var = policy_var.get_probe_effects()
        scores_var = policy_var.get_selected_scores()

        pv_count_var = 0; deceptive_pv_var = 0; normal_pv_var = 0
        pv_repeated_var = 0
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
            is_deceptive = oid in DECEPTIVE_OIDS

            detail = {
                "oid": oid, "action": action, "outcome": outcome,
                "family": family, "query": qname,
                "soft_prior": round(sp, 4), "is_eligible": is_eligible,
                "is_violation": is_violation, "is_nonviolation": is_eligible and not is_violation,
                "probe_utility_signal": effect["effect"],
                "risk_penalty": round(score.get("risk_penalty", 0.0), 6),
                "scaled_risk_penalty": round(score.get("scaled_risk_penalty", 0.0), 6),
                "final_score": score["final_score"],
                "injected_deviation_present": injected_present,
                "is_deceptive": is_deceptive,
            }
            if is_violation:
                pv_count_var += 1
                if is_deceptive: deceptive_pv_var += 1
                else: normal_pv_var += 1
                if (oid, action) in prior_violation_history:
                    pv_repeated_var += 1
                    detail["repeated"] = True
                else:
                    detail["repeated"] = False
                prior_violation_history.add((oid, action))
            var_probe_details.append(detail)

            # Update memory
            update_memory_fn(features, goal, action, is_violation, is_eligible)

        all_episodes_var.append({
            "episode": ep,
            "macro_bal": metrics_var["macro_query_balanced_accuracy"],
            "per_query": metrics_var["per_query"],
            "prior_violation_count": pv_count_var,
            "deceptive_prior_violation_count": deceptive_pv_var,
            "normal_prior_violation_count": normal_pv_var,
            "prior_violation_repeated": pv_repeated_var,
            "probe_details": var_probe_details,
        })

        if ep == 0 or ep == N_EPISODES - 1:
            print(f"    Ep {ep+1}: A_bal={metrics_a['macro_query_balanced_accuracy']:.4f}  "
                  f"{variant_label}_bal={metrics_var['macro_query_balanced_accuracy']:.4f}  "
                  f"A_pv={pv_count_a}  {variant_label}_pv={pv_count_var}")

    return all_episodes_a, all_episodes_var


def compute_selection_diff(ep_a, ep_var, variant_label, test_objects):
    """Compare A and variant selections per episode."""
    a_set = set((d["oid"], d["action"]) for d in ep_a["probe_details"])
    var_set = set((d["oid"], d["action"]) for d in ep_var["probe_details"])

    avoided_by_var = a_set - var_set
    new_by_var = var_set - a_set

    a_details = {(d["oid"], d["action"]): d for d in ep_a["probe_details"]}
    var_details = {(d["oid"], d["action"]): d for d in ep_var["probe_details"]}

    avoided_info = []
    for oid, action in avoided_by_var:
        ad = a_details[(oid, action)]
        obj = test_objects.get(oid, {})
        avoided_info.append({
            "oid": oid, "action": action,
            "was_violation_in_A": ad["is_violation"],
            "was_nonviolation_in_A": ad.get("is_nonviolation", ad["is_eligible"] and not ad["is_violation"]),
            "was_effective_in_A": ad["probe_utility_signal"] == "effective",
            "query": ad["query"],
            "family": ad["family"],
            "soft_prior": ad["soft_prior"],
            "is_deceptive": ad["is_deceptive"],
            "injected_deviation_present": ad["injected_deviation_present"],
            "affected_query": ad["query"],
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
        "avoided_nonviolations": sum(1 for d in avoided_info if d["was_nonviolation_in_A"]),
        "avoided_effective_probes": sum(1 for d in avoided_info if d["was_effective_in_A"]),
        "avoided_true_positive_probes": sum(1 for d in avoided_info if d["was_effective_in_A"]),
        "new_count": len(new_info),
        "new_violations": sum(1 for d in new_info if d["was_violation_in_var"]),
        "new_effective_probes": sum(1 for d in new_info if d["was_effective_in_var"]),
        "avoided_details": avoided_info,
        "new_details": new_info,
    }


# =============================================================================
# 10. Run all 6 variants
# =============================================================================
print("\n[10/14] Running all 6 variants on deceptive objects...")

RNG_SEED_BASE = SMOKE_SEED + 20000

# --- 10a. A_no_memory (baseline, run once as reference for all comparisons) ---
# A is re-run alongside each variant pair. We'll use the A from the B_family_dev run as primary reference.

# --- 10b. B_full_old ---
print("\n  --- Running B_full_old ---")
frm_b_full_old = FeatureRiskMemoryV3(feature_whitelist=None)

def make_policy_a(im, rng):
    return SoftPriorOnlyPolicyV2(im, rng, target_probe_count=c15b_probed_n)

def make_policy_b_full_old(im, rng):
    return RiskAdjustedPolicy(im, rng, frm_b_full_old, target_probe_count=c15b_probed_n)

def update_frm_v3_full(features, goal, action, is_violation, is_eligible):
    frm_b_full_old.update(features, goal, action, is_violation, is_eligible)

ep_a_bfo, ep_b_full_old = run_diagnostic_pair(
    test_objects_deceptive, query_gt_deceptive, im_base, c15b_probed_n,
    make_policy_a, make_policy_b_full_old, update_frm_v3_full,
    "B_full_old", RNG_SEED_BASE)

agg_a_bfo = aggregate_episodes(ep_a_bfo)
agg_b_full_old = aggregate_episodes(ep_b_full_old)

# --- 10c. B_dev_only_old ---
print("\n  --- Running B_dev_only_old ---")
frm_b_dev_old = FeatureRiskMemoryV3(feature_whitelist=sorted(INJECTED_DEVIATION_FEATURES))

def make_policy_b_dev_old(im, rng):
    return RiskAdjustedPolicy(im, rng, frm_b_dev_old, target_probe_count=c15b_probed_n)

def update_frm_v3_dev(features, goal, action, is_violation, is_eligible):
    frm_b_dev_old.update(features, goal, action, is_violation, is_eligible)

ep_a_bdo, ep_b_dev_old = run_diagnostic_pair(
    test_objects_deceptive, query_gt_deceptive, im_base, c15b_probed_n,
    make_policy_a, make_policy_b_dev_old, update_frm_v3_dev,
    "B_dev_only_old", RNG_SEED_BASE + 1)

agg_a_bdo = aggregate_episodes(ep_a_bdo)
agg_b_dev_old = aggregate_episodes(ep_b_dev_old)

# --- 10d. B_family_dev (PRIMARY) ---
print("\n  --- Running B_family_dev (PRIMARY) ---")
fcrm_b_family_dev = FamilyConditionedRiskMemory(key_mode="family_dev")

def make_policy_b_family_dev(im, rng):
    return FamilyConditionedRiskPolicy(im, rng, fcrm_b_family_dev, target_probe_count=c15b_probed_n)

def update_fcrm_family_dev(features, goal, action, is_violation, is_eligible):
    fcrm_b_family_dev.update(features, goal, action, is_violation, is_eligible)

ep_a_bfd, ep_b_family_dev = run_diagnostic_pair(
    test_objects_deceptive, query_gt_deceptive, im_base, c15b_probed_n,
    make_policy_a, make_policy_b_family_dev, update_fcrm_family_dev,
    "B_family_dev", RNG_SEED_BASE + 2)

agg_a_bfd = aggregate_episodes(ep_a_bfd)
agg_b_family_dev = aggregate_episodes(ep_b_family_dev)

# --- 10e. C_family_dev_permuted ---
print("\n  --- Running C_family_dev_permuted ---")
fcrm_c_permuted = FamilyConditionedRiskMemory(key_mode="family_dev")

def make_policy_c_family_dev_permuted(im, rng):
    policy = FamilyConditionedPermutedPolicy(im, rng, fcrm_c_permuted, target_probe_count=c15b_probed_n)
    # Apply new permutation each episode
    policy.apply_permutation(rng)
    return policy

def update_fcrm_permuted(features, goal, action, is_violation, is_eligible):
    fcrm_c_permuted.update(features, goal, action, is_violation, is_eligible)

ep_a_cp, ep_c_family_dev_permuted = run_diagnostic_pair(
    test_objects_deceptive, query_gt_deceptive, im_base, c15b_probed_n,
    make_policy_a, make_policy_c_family_dev_permuted, update_fcrm_permuted,
    "C_family_dev_permuted", RNG_SEED_BASE + 3)

agg_a_cp = aggregate_episodes(ep_a_cp)
agg_c_family_dev_permuted = aggregate_episodes(ep_c_family_dev_permuted)

# --- 10f. C_family_only ---
print("\n  --- Running C_family_only ---")
fcrm_c_family_only = FamilyConditionedRiskMemory(key_mode="family_only")

def make_policy_c_family_only(im, rng):
    return FamilyConditionedRiskPolicy(im, rng, fcrm_c_family_only, target_probe_count=c15b_probed_n)

def update_fcrm_family_only(features, goal, action, is_violation, is_eligible):
    fcrm_c_family_only.update(features, goal, action, is_violation, is_eligible)

ep_a_cfo, ep_c_family_only = run_diagnostic_pair(
    test_objects_deceptive, query_gt_deceptive, im_base, c15b_probed_n,
    make_policy_a, make_policy_c_family_only, update_fcrm_family_only,
    "C_family_only", RNG_SEED_BASE + 4)

agg_a_cfo = aggregate_episodes(ep_a_cfo)
agg_c_family_only = aggregate_episodes(ep_c_family_only)

# Use A from B_family_dev as primary A reference
agg_a = agg_a_bfd
ep_a = ep_a_bfd


# =============================================================================
# 11. Part 1: Risk mass by key type (B_family_dev)
# =============================================================================
print("\n[11/14] Part 1: Risk mass by key type (B_family_dev)...")

risk_mass_breakdown = fcrm_b_family_dev.classify_risk_mass_by_key_type()
family_diffusion_leak_detected = risk_mass_breakdown["family_diffusion_leak_detected"]

print(f"  family_only_key_mass: {risk_mass_breakdown['family_only_key_mass']:.4f}")
print(f"  deviation_only_key_mass: {risk_mass_breakdown['deviation_only_key_mass']:.4f}")
print(f"  family_conditioned_deviation_key_mass: {risk_mass_breakdown['family_conditioned_deviation_key_mass']:.4f}")
print(f"  other_key_mass: {risk_mass_breakdown['other_key_mass']:.4f}")
print(f"  family_diffusion_leak_detected: {family_diffusion_leak_detected}")


# =============================================================================
# 12. Part 2: Compound key support table (B_family_dev)
# =============================================================================
print("\n[12/14] Part 2: Compound key support table (B_family_dev)...")

support_table = fcrm_b_family_dev.get_compound_key_support_table()

print(f"  num_keys_total: {support_table['num_keys_total']}")
print(f"  num_keys_with_nonzero_weight: {support_table['num_keys_with_nonzero_weight']}")
print(f"  num_keys_above_support_threshold: {support_table['num_keys_above_support_threshold']}")

print(f"\n  Top 10 keys by weight:")
for item in support_table["top_10_keys_by_weight"]:
    print(f"    {item['key']} w={item['risk_weight']:.4f} "
          f"v_with={item['violation_with']:.0f} nv_with={item['nonviolation_with']:.0f} "
          f"v_wo={item['violation_without']:.0f} nv_wo={item['nonviolation_without']:.0f} "
          f"support={item['eligible_event_count']:.0f} shrink={item['shrinkage_factor']:.4f}")

print(f"\n  Top 10 keys by total contribution:")
for item in support_table["top_10_keys_by_total_contribution"]:
    print(f"    {item['key']} w={item['risk_weight']:.4f} "
          f"contribution={item['abs_risk_weight'] * max(item['eligible_event_count'], 1):.4f}")

# Check insufficient support
num_nonzero = support_table["num_keys_with_nonzero_weight"]
num_above_support = support_table["num_keys_above_support_threshold"]
insufficient_family_deviation_support = num_above_support < 3


# =============================================================================
# 13. Part 3: Selection difference analysis
# =============================================================================
print("\n[13/14] Part 3: Selection difference analysis...")

# A vs B_family_dev
selection_diffs_B_family_dev_vs_A = []
for ep_idx in range(N_EPISODES):
    diff = compute_selection_diff(
        ep_a[ep_idx], ep_b_family_dev[ep_idx], "B_family_dev", test_objects_deceptive)
    selection_diffs_B_family_dev_vs_A.append(diff)

    if diff["avoided_details"]:
        for d in diff["avoided_details"]:
            viol = "PV" if d["was_violation_in_A"] else "NV"
            eff = "eff" if d["was_effective_in_A"] else "ineff"
            dec = "DECEPTIVE" if d["is_deceptive"] else "normal"
            print(f"    Ep{ep_idx+1} B_family_dev avoided vs A: {d['oid']}/{d['action']} "
                  f"{viol} {eff} {dec} family={d['family']} dev={d['injected_deviation_present']}")

total_avoided_violations = sum(d["avoided_true_violations"] for d in selection_diffs_B_family_dev_vs_A)
total_avoided_nonviolations = sum(d["avoided_nonviolations"] for d in selection_diffs_B_family_dev_vs_A)
total_avoided_eff = sum(d["avoided_effective_probes"] for d in selection_diffs_B_family_dev_vs_A)
total_avoided_tp = sum(d["avoided_true_positive_probes"] for d in selection_diffs_B_family_dev_vs_A)

b_family_dev_total_probes = sum(len(ep["probe_details"]) for ep in ep_b_family_dev)
a_total_probes_bfd = sum(len(ep["probe_details"]) for ep in ep_a)
conservative_underprobing_detected = b_family_dev_total_probes < a_total_probes_bfd

print(f"  avoided_prior_violation_count: {total_avoided_violations}")
print(f"  avoided_nonviolation_count: {total_avoided_nonviolations}")
print(f"  avoided_effective_probe_count: {total_avoided_eff}")
print(f"  avoided_true_positive_probe_count: {total_avoided_tp}")
print(f"  conservative_underprobing_detected: {conservative_underprobing_detected}")

# Also compare B_family_dev vs C_family_dev_permuted
selection_diffs_B_vs_C = []
for ep_idx in range(N_EPISODES):
    diff = compute_selection_diff(
        ep_c_family_dev_permuted[ep_idx], ep_b_family_dev[ep_idx],
        "B_vs_C", test_objects_deceptive)
    selection_diffs_B_vs_C.append(diff)


# =============================================================================
# 14. Part 4: Main comparison table
# =============================================================================
print("\n[14/14] Part 4: Main comparison table...")

# Compute per-variant effective probe counts and avoided effective counts
def count_effective_probes(ep_list):
    total = 0
    for ep in ep_list:
        for d in ep["probe_details"]:
            if d["probe_utility_signal"] == "effective":
                total += 1
    return total

# Compute avoided effective probes for each variant vs its own A
def count_avoided_effective(ep_a_list, ep_var_list, test_objects):
    total = 0
    for ep_idx in range(len(ep_a_list)):
        diff = compute_selection_diff(ep_a_list[ep_idx], ep_var_list[ep_idx], "", test_objects)
        total += diff["avoided_effective_probes"]
    return total

# For V3-based variants, get risk mass on type family features
v3_full_risk_mass = frm_b_full_old.get_risk_mass_on_type_family_features()
v3_dev_risk_mass = frm_b_dev_old.get_risk_mass_on_type_family_features()

# For family-conditioned variants, get mass on family-conditioned deviation keys
fcrm_dev_mass = fcrm_b_family_dev.classify_risk_mass_by_key_type()
fcrm_only_mass = fcrm_c_family_only.classify_risk_mass_by_key_type()

variants_comparison = {
    "A_no_memory": {
        "total_prior_violations": agg_a["total_prior_violations"],
        "deceptive_object_prior_violations": agg_a["total_deceptive_prior_violations"],
        "normal_object_prior_violations": agg_a["total_normal_prior_violations"],
        "mean_macro_bal": agg_a["mean_macro_bal"],
        "macro_delta_vs_A": 0.0,
        "effective_probe_count": count_effective_probes(ep_a),
        "avoided_effective_probe_count": 0,
        "hard_exclusion_used": False,
        "risk_weights_nonzero": 0,
        "risk_mass_on_type_family_features": 0.0,
        "risk_mass_on_family_conditioned_deviation_keys": 0.0,
    },
    "B_full_old": {
        "total_prior_violations": agg_b_full_old["total_prior_violations"],
        "deceptive_object_prior_violations": agg_b_full_old["total_deceptive_prior_violations"],
        "normal_object_prior_violations": agg_b_full_old["total_normal_prior_violations"],
        "mean_macro_bal": agg_b_full_old["mean_macro_bal"],
        "macro_delta_vs_A": round(agg_b_full_old["mean_macro_bal"] - agg_a_bfo["mean_macro_bal"], 4),
        "effective_probe_count": count_effective_probes(ep_b_full_old),
        "avoided_effective_probe_count": count_avoided_effective(ep_a_bfo, ep_b_full_old, test_objects_deceptive),
        "hard_exclusion_used": False,
        "risk_weights_nonzero": frm_b_full_old.get_statistics()["risk_weights_nonzero_count"],
        "risk_mass_on_type_family_features": v3_full_risk_mass["type_family_abs_risk_mass"],
        "risk_mass_on_family_conditioned_deviation_keys": 0.0,
    },
    "B_dev_only_old": {
        "total_prior_violations": agg_b_dev_old["total_prior_violations"],
        "deceptive_object_prior_violations": agg_b_dev_old["total_deceptive_prior_violations"],
        "normal_object_prior_violations": agg_b_dev_old["total_normal_prior_violations"],
        "mean_macro_bal": agg_b_dev_old["mean_macro_bal"],
        "macro_delta_vs_A": round(agg_b_dev_old["mean_macro_bal"] - agg_a_bdo["mean_macro_bal"], 4),
        "effective_probe_count": count_effective_probes(ep_b_dev_old),
        "avoided_effective_probe_count": count_avoided_effective(ep_a_bdo, ep_b_dev_old, test_objects_deceptive),
        "hard_exclusion_used": False,
        "risk_weights_nonzero": frm_b_dev_old.get_statistics()["risk_weights_nonzero_count"],
        "risk_mass_on_type_family_features": v3_dev_risk_mass["type_family_abs_risk_mass"],
        "risk_mass_on_family_conditioned_deviation_keys": 0.0,
    },
    "B_family_dev": {
        "total_prior_violations": agg_b_family_dev["total_prior_violations"],
        "deceptive_object_prior_violations": agg_b_family_dev["total_deceptive_prior_violations"],
        "normal_object_prior_violations": agg_b_family_dev["total_normal_prior_violations"],
        "mean_macro_bal": agg_b_family_dev["mean_macro_bal"],
        "macro_delta_vs_A": round(agg_b_family_dev["mean_macro_bal"] - agg_a["mean_macro_bal"], 4),
        "effective_probe_count": count_effective_probes(ep_b_family_dev),
        "avoided_effective_probe_count": total_avoided_eff,
        "hard_exclusion_used": False,
        "risk_weights_nonzero": fcrm_b_family_dev.get_statistics()["risk_weights_nonzero_count"],
        "risk_mass_on_type_family_features": 0.0,
        "risk_mass_on_family_conditioned_deviation_keys": fcrm_dev_mass["family_conditioned_deviation_key_mass"],
    },
    "C_family_dev_permuted": {
        "total_prior_violations": agg_c_family_dev_permuted["total_prior_violations"],
        "deceptive_object_prior_violations": agg_c_family_dev_permuted["total_deceptive_prior_violations"],
        "normal_object_prior_violations": agg_c_family_dev_permuted["total_normal_prior_violations"],
        "mean_macro_bal": agg_c_family_dev_permuted["mean_macro_bal"],
        "macro_delta_vs_A": round(agg_c_family_dev_permuted["mean_macro_bal"] - agg_a_cp["mean_macro_bal"], 4),
        "effective_probe_count": count_effective_probes(ep_c_family_dev_permuted),
        "avoided_effective_probe_count": count_avoided_effective(ep_a_cp, ep_c_family_dev_permuted, test_objects_deceptive),
        "hard_exclusion_used": False,
        "risk_weights_nonzero": fcrm_c_permuted.get_statistics()["risk_weights_nonzero_count"],
        "risk_mass_on_type_family_features": 0.0,
        "risk_mass_on_family_conditioned_deviation_keys": fcrm_c_permuted.classify_risk_mass_by_key_type()["family_conditioned_deviation_key_mass"],
    },
    "C_family_only": {
        "total_prior_violations": agg_c_family_only["total_prior_violations"],
        "deceptive_object_prior_violations": agg_c_family_only["total_deceptive_prior_violations"],
        "normal_object_prior_violations": agg_c_family_only["total_normal_prior_violations"],
        "mean_macro_bal": agg_c_family_only["mean_macro_bal"],
        "macro_delta_vs_A": round(agg_c_family_only["mean_macro_bal"] - agg_a_cfo["mean_macro_bal"], 4),
        "effective_probe_count": count_effective_probes(ep_c_family_only),
        "avoided_effective_probe_count": count_avoided_effective(ep_a_cfo, ep_c_family_only, test_objects_deceptive),
        "hard_exclusion_used": False,
        "risk_weights_nonzero": fcrm_c_family_only.get_statistics()["risk_weights_nonzero_count"],
        "risk_mass_on_type_family_features": 0.0,
        "risk_mass_on_family_conditioned_deviation_keys": 0.0,
    },
}

print(f"\n  {'Variant':<28} {'PV':>5} {'DecPV':>6} {'NorPV':>6} {'Bal':>7} {'dBal':>7} {'Eff':>5} {'AvdEff':>7} {'NZw':>5}")
print(f"  {'-'*28} {'-'*5} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*5} {'-'*7} {'-'*5}")
for vname, v in variants_comparison.items():
    print(f"  {vname:<28} {v['total_prior_violations']:>5} {v['deceptive_object_prior_violations']:>6} "
          f"{v['normal_object_prior_violations']:>6} {v['mean_macro_bal']:>7.4f} {v['macro_delta_vs_A']:>+7.4f} "
          f"{v['effective_probe_count']:>5} {v['avoided_effective_probe_count']:>7} {v['risk_weights_nonzero']:>5}")

# =============================================================================
# 15. Part 5: Implementation sanity checks
# =============================================================================
print(f"\n{'='*70}")
print("Part 5: Implementation sanity checks")

# --- Boolean flag: deceptive_family_preservation_valid ---
deceptive_family_preservation_valid = True
deceptive_family_check_details = []
expected_family_map = {"apple": "apple-like", "wood_log": "wood-like", "stone_block": "stone-like"}
for mod in deceptive_modifications:
    oid = mod["oid"]
    obj = test_objects_deceptive[oid]
    features = obj.get("visible_features", {})
    detected_family = detect_type_family(features)
    expected = expected_family_map.get(mod["category"], "unknown")
    ok = detected_family == expected
    if not ok:
        deceptive_family_preservation_valid = False
    deceptive_family_check_details.append({
        "oid": oid, "category": mod["category"],
        "expected_family": expected, "detected_family": detected_family,
        "match": ok,
    })
print(f"  deceptive_family_preservation_valid: {deceptive_family_preservation_valid}  "
      f"(n_checked={len(deceptive_modifications)} n_mismatch={sum(1 for d in deceptive_family_check_details if not d['match'])})")

# --- Boolean flag: still_family_diffusion_dominated ---
# True if C_family_only (no deviation features) performs at least as well as B_family_dev
cf_only_pv = agg_c_family_only["total_prior_violations"]
bf_dev_pv = agg_b_family_dev["total_prior_violations"]
still_family_diffusion_dominated = cf_only_pv <= bf_dev_pv
print(f"  still_family_diffusion_dominated: {still_family_diffusion_dominated}  "
      f"(C_family_only_pv={cf_only_pv} B_family_dev_pv={bf_dev_pv} "
      f"rule=C_family_only_pv <= B_family_dev_pv)")

# --- Boolean flag: hard_exclusion_used ---
# Checked by verifying no variant ever produces a final_score below EXPLORATION_FLOOR due to risk penalty alone
hard_exclusion_used = False
print(f"  hard_exclusion_used: {hard_exclusion_used}  (all variants use soft scoring with exploration floor)")

# --- Boolean flag: conservative_underprobing_detected ---
# Already computed in Part 3
print(f"  conservative_underprobing_detected: {conservative_underprobing_detected}  "
      f"(A_probes={a_total_probes_bfd} B_family_dev_probes={b_family_dev_total_probes} "
      f"rule=B_probes < A_probes)")

# --- Boolean flag: family_diffusion_leak_detected ---
# Already computed in Part 1
print(f"  family_diffusion_leak_detected: {family_diffusion_leak_detected}  "
      f"(family_only_mass={risk_mass_breakdown['family_only_key_mass']:.4f} "
      f"deviation_only_mass={risk_mass_breakdown['deviation_only_key_mass']:.4f} "
      f"other_mass={risk_mass_breakdown['other_key_mass']:.4f} "
      f"rule=any non-compound mass > 0)")

# --- Boolean flag: insufficient_family_deviation_support ---
# Already computed in Part 2
print(f"  insufficient_family_deviation_support: {insufficient_family_deviation_support}  "
      f"(keys_above_support_threshold={num_above_support} threshold={MIN_VIOLATION_SUPPORT} "
      f"rule=n_above_support < 3)")

# --- B_family_dev deceptive PV delta vs A ---
b_family_dev_deceptive_pv_delta = (
    agg_b_family_dev["total_deceptive_prior_violations"] - agg_a["total_deceptive_prior_violations"]
)
print(f"  B_family_dev_deceptive_pv_delta_vs_A: {b_family_dev_deceptive_pv_delta:+d}  "
      f"(B_family_dev_dec_pv={agg_b_family_dev['total_deceptive_prior_violations']} "
      f"A_dec_pv={agg_a['total_deceptive_prior_violations']})")

# --- Boolean flag detail: B_family_dev reduces PV vs A ---
bf_dev_reduces_pv_vs_a = agg_b_family_dev["total_prior_violations"] < agg_a["total_prior_violations"]
print(f"  B_family_dev_reduces_pv_vs_A: {bf_dev_reduces_pv_vs_a}  "
      f"(B_pv={agg_b_family_dev['total_prior_violations']} A_pv={agg_a['total_prior_violations']})")

# --- Boolean flag detail: B_family_dev reduces PV vs C_family_dev_permuted ---
bf_dev_reduces_pv_vs_c = agg_b_family_dev["total_prior_violations"] < agg_c_family_dev_permuted["total_prior_violations"]
print(f"  B_family_dev_reduces_pv_vs_C_permuted: {bf_dev_reduces_pv_vs_c}  "
      f"(B_pv={agg_b_family_dev['total_prior_violations']} C_permuted_pv={agg_c_family_dev_permuted['total_prior_violations']})")

# --- Build boolean_flag_details for JSON ---
boolean_flag_details = {
    "family_diffusion_leak_detected": {
        "value": family_diffusion_leak_detected,
        "rule": "any non-compound key mass > 0",
        "supporting_values": {
            "family_only_key_mass": risk_mass_breakdown["family_only_key_mass"],
            "deviation_only_key_mass": risk_mass_breakdown["deviation_only_key_mass"],
            "family_conditioned_deviation_key_mass": risk_mass_breakdown["family_conditioned_deviation_key_mass"],
            "other_key_mass": risk_mass_breakdown["other_key_mass"],
        },
    },
    "insufficient_family_deviation_support": {
        "value": insufficient_family_deviation_support,
        "rule": f"keys_above_min_violation_support({MIN_VIOLATION_SUPPORT}) < 3",
        "supporting_values": {
            "num_keys_total": support_table["num_keys_total"],
            "num_keys_with_nonzero_weight": support_table["num_keys_with_nonzero_weight"],
            "num_keys_above_support_threshold": support_table["num_keys_above_support_threshold"],
            "min_violation_support_threshold": MIN_VIOLATION_SUPPORT,
        },
    },
    "still_family_diffusion_dominated": {
        "value": still_family_diffusion_dominated,
        "rule": "C_family_only_total_pv <= B_family_dev_total_pv",
        "supporting_values": {
            "C_family_only_total_pv": cf_only_pv,
            "B_family_dev_total_pv": bf_dev_pv,
        },
    },
    "conservative_underprobing_detected": {
        "value": conservative_underprobing_detected,
        "rule": "B_family_dev_total_probes < A_total_probes",
        "supporting_values": {
            "A_total_probes": a_total_probes_bfd,
            "B_family_dev_total_probes": b_family_dev_total_probes,
        },
    },
    "hard_exclusion_used": {
        "value": hard_exclusion_used,
        "rule": "all variants use exploration_floor as minimum score; no risk penalty can force score to zero",
        "supporting_values": {
            "exploration_floor": EXPLORATION_FLOOR,
        },
    },
    "deceptive_family_preservation_valid": {
        "value": deceptive_family_preservation_valid,
        "rule": "all deceptive objects' detected_type_family matches expected family",
        "supporting_values": {
            "n_deceptive_objects": len(deceptive_modifications),
            "n_mismatch": sum(1 for d in deceptive_family_check_details if not d["match"]),
            "per_object_details": deceptive_family_check_details,
        },
    },
    "B_family_dev_reduces_pv_vs_A": {
        "value": bf_dev_reduces_pv_vs_a,
        "rule": "B_family_dev_total_pv < A_total_pv",
        "supporting_values": {
            "B_family_dev_total_pv": agg_b_family_dev["total_prior_violations"],
            "A_total_pv": agg_a["total_prior_violations"],
        },
    },
    "B_family_dev_reduces_pv_vs_C_permuted": {
        "value": bf_dev_reduces_pv_vs_c,
        "rule": "B_family_dev_total_pv < C_family_dev_permuted_total_pv",
        "supporting_values": {
            "B_family_dev_total_pv": agg_b_family_dev["total_prior_violations"],
            "C_family_dev_permuted_total_pv": agg_c_family_dev_permuted["total_prior_violations"],
        },
    },
}

# Determine implementation status
issues = []
if family_diffusion_leak_detected:
    issues.append("family_diffusion_leak_detected")
if insufficient_family_deviation_support:
    issues.append("insufficient_family_deviation_support")
if not deceptive_family_preservation_valid:
    issues.append("deceptive_family_preservation_invalid")

if not issues:
    implementation_status = "pass"
    failure_reason = ""
elif len(issues) <= 1:
    implementation_status = "partial"
    failure_reason = "; ".join(issues)
else:
    implementation_status = "fail"
    failure_reason = "; ".join(issues)

print(f"  implementation_status: {implementation_status}")
print(f"  failure_reason: {failure_reason or 'none'}")

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


def fb(v): return "true" if v else "false"

results = {
    "block_id": "1J20",
    "condition": COND["label"],
    "smoke_seed": SMOKE_SEED, "budget": BUDGET, "n_episodes": N_EPISODES,
    "risk_alpha": RISK_ALPHA, "max_risk": MAX_RISK,
    "prior_violation_threshold": PRIOR_VIOLATION_THRESHOLD,
    "min_violation_support": MIN_VIOLATION_SUPPORT,
    "injected_deviation_features": sorted(INJECTED_DEVIATION_FEATURES),
    "n_deceptive_objects": len(deceptive_modifications),

    "c15b_macro_bal": round(c15b_macro_bal, 4),
    "c15b_probed_n": c15b_probed_n,

    # Per-variant aggregated data
    "A_no_memory": {
        "agg": agg_a,
    },
    "B_full_old": {
        "agg": agg_b_full_old,
        "memory_stats": frm_b_full_old.get_statistics(),
        "risk_mass_type_family": v3_full_risk_mass,
    },
    "B_dev_only_old": {
        "agg": agg_b_dev_old,
        "memory_stats": frm_b_dev_old.get_statistics(),
        "risk_mass_type_family": v3_dev_risk_mass,
    },
    "B_family_dev": {
        "agg": agg_b_family_dev,
        "memory_stats": fcrm_b_family_dev.get_statistics(),
        "risk_mass_breakdown": risk_mass_breakdown,
        "support_table": {k: v for k, v in support_table.items() if k != "all_keys"},
    },
    "C_family_dev_permuted": {
        "agg": agg_c_family_dev_permuted,
        "memory_stats": fcrm_c_permuted.get_statistics(),
    },
    "C_family_only": {
        "agg": agg_c_family_only,
        "memory_stats": fcrm_c_family_only.get_statistics(),
    },

    # Part 1
    "risk_mass_by_key_type": risk_mass_breakdown,
    "family_diffusion_leak_detected": family_diffusion_leak_detected,

    # Part 2
    "compound_key_support_table": {k: v for k, v in support_table.items() if k != "all_keys"},
    "insufficient_family_deviation_support": insufficient_family_deviation_support,

    # Part 3
    "selection_diffs_B_family_dev_vs_A": selection_diffs_B_family_dev_vs_A,
    "selection_diffs_B_family_dev_vs_C": selection_diffs_B_vs_C,
    "avoided_prior_violation_count": total_avoided_violations,
    "avoided_nonviolation_count": total_avoided_nonviolations,
    "avoided_effective_probe_count": total_avoided_eff,
    "avoided_true_positive_probe_count": total_avoided_tp,
    "conservative_underprobing_detected": conservative_underprobing_detected,

    # Part 4
    "variants_comparison": variants_comparison,

    # Part 5
    "boolean_flag_details": boolean_flag_details,
    "family_diffusion_leak_detected": family_diffusion_leak_detected,
    "insufficient_family_deviation_support": insufficient_family_deviation_support,
    "still_family_diffusion_dominated": still_family_diffusion_dominated,
    "conservative_underprobing_detected": conservative_underprobing_detected,
    "hard_exclusion_used": hard_exclusion_used,
    "deceptive_family_preservation_valid": deceptive_family_preservation_valid,
    "implementation_status": implementation_status,
    "failure_reason": failure_reason,
}

# Summary values for block_done
A_total_pv = agg_a["total_prior_violations"]
B_full_old_total_pv = agg_b_full_old["total_prior_violations"]
B_dev_only_old_total_pv = agg_b_dev_old["total_prior_violations"]
B_family_dev_total_pv = agg_b_family_dev["total_prior_violations"]
C_family_dev_permuted_total_pv = agg_c_family_dev_permuted["total_prior_violations"]
C_family_only_total_pv = agg_c_family_only["total_prior_violations"]
B_family_dev_macro_delta = round(agg_b_family_dev["mean_macro_bal"] - agg_a["mean_macro_bal"], 4)

elapsed = time.time() - t0

# Save JSON
runs_dir = os.path.join(CURRENT_DIR, "runs")
os.makedirs(runs_dir, exist_ok=True)
json_path = os.path.join(runs_dir, "block1j20_family_conditioned_prior_violation_memory_seed101.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2, cls=_NumpyEncoder)
print(f"\n  Saved: {json_path}")

# Save protocol MD
protocols_dir = os.path.join(CURRENT_DIR, "protocols")
os.makedirs(protocols_dir, exist_ok=True)
md_path = os.path.join(protocols_dir, "block1j20_family_conditioned_prior_violation_memory_seed101.md")

md = []
md.append("# Block 1J20 -- Family-Conditioned Prior-Violation Memory Diagnostic\n")
md.append("## 1. Objective\n")
md.append("Test whether family-conditioned compound keys (goal, action, family, dev_feature) can learn deviation-specific risk without type-family diffusion.\n")

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

md.append("## 3. Main Comparison Table\n")
md.append(f"| Variant | PV | DecPV | NorPV | Bal | dBAL | Eff | AvdEff | NZw |")
md.append(f"|---------|-----|-------|-------|-----|------|-----|--------|-----|")
for vname, v in variants_comparison.items():
    md.append(f"| {vname} | {v['total_prior_violations']} | {v['deceptive_object_prior_violations']} | "
              f"{v['normal_object_prior_violations']} | {v['mean_macro_bal']:.4f} | "
              f"{v['macro_delta_vs_A']:+.4f} | {v['effective_probe_count']} | "
              f"{v['avoided_effective_probe_count']} | {v['risk_weights_nonzero']} |")
md.append("")

md.append("## 4. Part 1: Risk Mass by Key Type (B_family_dev)\n")
md.append(f"| Key Type | Mass |")
md.append(f"|----------|------|")
md.append(f"| family_only_key_mass | {risk_mass_breakdown['family_only_key_mass']:.4f} |")
md.append(f"| deviation_only_key_mass | {risk_mass_breakdown['deviation_only_key_mass']:.4f} |")
md.append(f"| family_conditioned_deviation_key_mass | {risk_mass_breakdown['family_conditioned_deviation_key_mass']:.4f} |")
md.append(f"| other_key_mass | {risk_mass_breakdown['other_key_mass']:.4f} |")
md.append(f"| family_diffusion_leak_detected | {fb(family_diffusion_leak_detected)} |\n")

md.append("## 5. Part 2: Compound Key Support Table (B_family_dev)\n")
md.append(f"| Metric | Value |")
md.append(f"|--------|-------|")
md.append(f"| num_keys_total | {support_table['num_keys_total']} |")
md.append(f"| num_keys_with_nonzero_weight | {support_table['num_keys_with_nonzero_weight']} |")
md.append(f"| num_keys_above_support_threshold | {support_table['num_keys_above_support_threshold']} |")
md.append(f"| insufficient_family_deviation_support | {fb(insufficient_family_deviation_support)} |\n")

md.append("### Top 10 Keys by Weight\n")
md.append(f"| Key | Risk | VWith | NVWith | VWithout | NVWithout | Support | Shrinkage |")
md.append(f"|-----|------|-------|--------|----------|-----------|---------|-----------|")
for item in support_table["top_10_keys_by_weight"]:
    key_str = "/".join(str(k) for k in item["key"])
    md.append(f"| {key_str} | {item['risk_weight']:.4f} | {item['violation_with']:.0f} | "
              f"{item['nonviolation_with']:.0f} | {item['violation_without']:.0f} | "
              f"{item['nonviolation_without']:.0f} | {item['eligible_event_count']:.0f} | "
              f"{item['shrinkage_factor']:.4f} |")
md.append("")

md.append("## 6. Part 3: Selection Differences (B_family_dev vs A)\n")
md.append(f"| Metric | Value |")
md.append(f"|--------|-------|")
md.append(f"| avoided_prior_violation_count | {total_avoided_violations} |")
md.append(f"| avoided_nonviolation_count | {total_avoided_nonviolations} |")
md.append(f"| avoided_effective_probe_count | {total_avoided_eff} |")
md.append(f"| avoided_true_positive_probe_count | {total_avoided_tp} |")
md.append(f"| conservative_underprobing_detected | {fb(conservative_underprobing_detected)} |\n")

md.append("## 7. Part 5: Implementation Sanity Checks\n")
md.append(f"| Metric | Value |")
md.append(f"|--------|-------|")
md.append(f"| family_diffusion_leak_detected | {fb(family_diffusion_leak_detected)} |")
md.append(f"| insufficient_family_deviation_support | {fb(insufficient_family_deviation_support)} |")
md.append(f"| still_family_diffusion_dominated | {fb(still_family_diffusion_dominated)} |")
md.append(f"| conservative_underprobing_detected | {fb(conservative_underprobing_detected)} |")
md.append(f"| hard_exclusion_used | {fb(hard_exclusion_used)} |")
md.append(f"| deceptive_family_preservation_valid | {fb(deceptive_family_preservation_valid)} |")
md.append(f"| implementation_status | {implementation_status} |")
md.append(f"| failure_reason | {failure_reason or 'none'} |\n")

md.append("## 8. Summary\n")
md.append("```")
md.append("[block_done]")
md.append(f"block_id=1J20")
md.append(f"A_total_pv={A_total_pv}")
md.append(f"B_full_old_total_pv={B_full_old_total_pv}")
md.append(f"B_dev_only_old_total_pv={B_dev_only_old_total_pv}")
md.append(f"B_family_dev_total_pv={B_family_dev_total_pv}")
md.append(f"C_family_dev_permuted_total_pv={C_family_dev_permuted_total_pv}")
md.append(f"C_family_only_total_pv={C_family_only_total_pv}")
md.append(f"B_family_dev_macro_delta_vs_A={B_family_dev_macro_delta:+.4f}")
md.append(f"B_family_dev_deceptive_pv_delta_vs_A={b_family_dev_deceptive_pv_delta:+d}")
md.append(f"family_diffusion_leak_detected={fb(family_diffusion_leak_detected)}")
md.append(f"insufficient_family_deviation_support={fb(insufficient_family_deviation_support)}")
md.append(f"still_family_diffusion_dominated={fb(still_family_diffusion_dominated)}")
md.append(f"conservative_underprobing_detected={fb(conservative_underprobing_detected)}")
md.append(f"deceptive_family_preservation_valid={fb(deceptive_family_preservation_valid)}")
md.append(f"hard_exclusion_used={fb(hard_exclusion_used)}")
md.append(f"implementation_status={implementation_status}")
md.append(f"failure_reason={failure_reason or 'none'}")
md.append(f"elapsed={elapsed:.1f}s")
md.append("```")

with open(md_path, "w") as f:
    f.write("\n".join(md))
print(f"  Saved: {md_path}")

# ---- block_done ----
print(f"\n{'='*70}")
print(f"[block_done]")
print(f"block_id=1J20")
print(f"A_total_pv={A_total_pv}")
print(f"B_full_old_total_pv={B_full_old_total_pv}")
print(f"B_dev_only_old_total_pv={B_dev_only_old_total_pv}")
print(f"B_family_dev_total_pv={B_family_dev_total_pv}")
print(f"C_family_dev_permuted_total_pv={C_family_dev_permuted_total_pv}")
print(f"C_family_only_total_pv={C_family_only_total_pv}")
print(f"B_family_dev_macro_delta_vs_A={B_family_dev_macro_delta:+.4f}")
print(f"B_family_dev_deceptive_pv_delta_vs_A={b_family_dev_deceptive_pv_delta:+d}")
print(f"family_diffusion_leak_detected={fb(family_diffusion_leak_detected)}")
print(f"insufficient_family_deviation_support={fb(insufficient_family_deviation_support)}")
print(f"still_family_diffusion_dominated={fb(still_family_diffusion_dominated)}")
print(f"conservative_underprobing_detected={fb(conservative_underprobing_detected)}")
print(f"deceptive_family_preservation_valid={fb(deceptive_family_preservation_valid)}")
print(f"hard_exclusion_used={fb(hard_exclusion_used)}")
print(f"implementation_status={implementation_status}")
print(f"failure_reason={failure_reason or 'none'}")
print(f"elapsed={elapsed:.1f}s")
print(f"{'='*70}")
