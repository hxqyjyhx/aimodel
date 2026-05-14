"""
Block 1J21 -- Evidence Bottleneck and Offline Upper-Bound Diagnostic.

Diagnoses whether the weak 1J20 B_family_dev result is caused by:
1. insufficient online evidence collection, or
2. family-conditioned deviation keys being uninformative even with enough evidence.

Five variants:
  A_no_memory                    — baseline, no risk memory
  B_family_dev_online            — same as 1J20 B_family_dev (online learning only)
  B_family_dev_offline_upper     — oracle: pre-built from ALL eligible prior events (diagnostic)
  C_family_dev_offline_permuted  — permuted control for offline upper-bound
  C_family_only_offline          — family-only offline oracle (no deviation features)
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
print("Block 1J21 -- Evidence Bottleneck and Offline Upper-Bound Diagnostic")
print(f"  condition={COND['label']}  seed={SMOKE_SEED}  budget={BUDGET}")
print(f"  episodes={N_EPISODES}  risk_alpha={RISK_ALPHA}  max_risk={MAX_RISK}")
print(f"  prior_violation_threshold={PRIOR_VIOLATION_THRESHOLD}")
print(f"  injected_deviation_features={sorted(INJECTED_DEVIATION_FEATURES)}")
print("=" * 70)

# =============================================================================
# 1. Phase A: Training + IOM building
# =============================================================================
print("\n[1/15] Phase A: Training + IOM building...")
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
print("\n[2/15] Generating test objects...")
test_objects_standard = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")
query_gt_standard = compute_query_ground_truth(test_objects_standard)
test_oids = sorted(test_objects_standard.keys())
QUERY_NAMES = sorted(_TASK_QUERIES.keys())
print(f"  {len(test_oids)} standard test objects, {len(QUERY_NAMES)} queries")

print("\n[3/15] Creating family-preserving deceptive test objects...")

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
print("\n[4/15] Building goal-conditioned soft prior table...")


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
# 5. FamilyConditionedRiskMemory (reused from 1J20)
# =============================================================================
print("\n[5/15] Defining FamilyConditionedRiskMemory...")

class FamilyConditionedRiskMemory:
    """Memory with compound keys conditioned on type family.

    key_mode="family_dev": keys are (goal, action, family, dev_feature)
    key_mode="family_only": keys are (goal, action, family)
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
            key_detected = (goal, action, family)
            if is_violation:
                self.counts[key_detected]["violation_with"] += 1.0
            else:
                self.counts[key_detected]["nonviolation_with"] += 1.0
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
        groups = defaultdict(list)
        for key in self.counts:
            if len(key) == 4:
                goal, action, family, dev_feat = key
                w = self.get_risk_weight(*key)
                groups[(goal, action, family)].append((key, w))
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

    def get_num_keys_above_support_threshold(self):
        count = 0
        for key in self.counts:
            c = self.counts[key]
            actual_violations = (c["violation_with"] + c["violation_without"]) - 2.0
            if actual_violations >= self.min_violation_support:
                count += 1
        return count

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
# 6. Policy Classes
# =============================================================================
print("\n[6/15] Defining policy classes...")

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


class FamilyConditionedRiskPolicy(SoftPriorOnlyPolicyV2):
    """Policy using FamilyConditionedRiskMemory. Used for all memory variants."""

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
    """Permuted control. Shuffles weights within (goal, action, family) groups."""

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
# 7. C15b reference
# =============================================================================
print("\n[7/15] Running C15b reference...")

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
# 8. Offline Oracle Memory Builder
# =============================================================================
print("\n[8/15] Building offline oracle memory...")

def get_ground_truth_outcome(obj, action):
    """Get ground truth outcome for an (object, action) pair from hidden_affordance_profile."""
    profile = obj.get("hidden_affordance_profile", {})
    result = profile.get(action, "success")
    return 1.0 if result == "success" else 0.0


def build_offline_oracle_memory(test_objects, key_mode="family_dev"):
    """Build FamilyConditionedRiskMemory from ALL eligible prior events using ground truth.

    For each object-action-goal candidate in the environment:
    - compute soft_prior
    - if soft_prior >= threshold: eligible prior event
    - use ground truth outcome to determine violation/nonviolation
    - update memory counters
    """
    memory = FamilyConditionedRiskMemory(key_mode=key_mode)

    for oid, obj in test_objects.items():
        features = obj.get("visible_features", {})
        for action in MAIN_CANDIDATE_ACTIONS:
            soft_prior = get_goal_soft_prior(features, action)
            is_eligible = soft_prior >= PRIOR_VIOLATION_THRESHOLD
            if not is_eligible:
                continue
            outcome = get_ground_truth_outcome(obj, action)
            is_violation = outcome <= 0.01  # failure = violation
            goal = ACTION_TO_QUERY.get(action, "")
            memory.update(features, goal, action, is_violation, is_eligible)

    return memory


# =============================================================================
# 9. Evidence Funnel Data Collection (Part 1)
# =============================================================================
print("\n[9/15] Collecting evidence funnel data...")

def collect_evidence_funnel_data(test_objects):
    """For each (goal, action, family, dev_feature) key, collect all counts."""
    keys_list = []
    for goal in sorted(set(ACTION_TO_QUERY.values())):
        for action in MAIN_CANDIDATE_ACTIONS:
            if ACTION_TO_QUERY.get(action, "") != goal:
                continue
            for family in FAMILY_NAMES:
                for dev_feat in sorted(INJECTED_DEVIATION_FEATURES):
                    keys_list.append((goal, action, family, dev_feat))

    funnel_data = []

    for goal, action, family, dev_feat in keys_list:
        available_objects = []
        available_deceptive = []
        available_normal = []
        high_prior_eligible = []
        # For offline: we need ground truth outcomes for eligible events
        offline_eligible_violations = 0
        offline_eligible_nonviolations = 0

        for oid, obj in test_objects.items():
            features = obj.get("visible_features", {})
            obj_family = detect_type_family(features)
            has_dev_feat = features.get(dev_feat, False)

            if obj_family == family and has_dev_feat:
                available_objects.append(oid)
                if oid in DECEPTIVE_OIDS:
                    available_deceptive.append(oid)
                else:
                    available_normal.append(oid)

                soft_prior = get_goal_soft_prior(features, action)
                if soft_prior >= PRIOR_VIOLATION_THRESHOLD:
                    high_prior_eligible.append(oid)
                    outcome = get_ground_truth_outcome(obj, action)
                    if outcome <= 0.01:
                        offline_eligible_violations += 1
                    else:
                        offline_eligible_nonviolations += 1

        funnel_data.append({
            "key": [goal, action, family, dev_feat],
            "available_objects_with_key": len(available_objects),
            "available_deceptive_objects_with_key": len(available_deceptive),
            "available_normal_objects_with_key": len(available_normal),
            "high_prior_eligible_objects_with_key": len(high_prior_eligible),
            "online_selected_events_with_key": 0,        # filled during online run
            "online_prior_violations_with_key": 0,       # filled during online run
            "online_prior_nonviolations_with_key": 0,    # filled during online run
            "offline_eligible_events_with_key": len(high_prior_eligible),
            "offline_prior_violations_with_key": offline_eligible_violations,
            "offline_prior_nonviolations_with_key": offline_eligible_nonviolations,
            "online_above_support_threshold": False,     # filled during online run
            "offline_above_support_threshold": (offline_eligible_violations >= MIN_VIOLATION_SUPPORT),
        })

    return funnel_data


# Pre-collect funnel data (online fields to be filled after online run)
funnel_data_template = collect_evidence_funnel_data(test_objects_deceptive)

# Build a lookup by key for fast access
funnel_by_key = {}
for entry in funnel_data_template:
    funnel_by_key[tuple(entry["key"])] = entry


# =============================================================================
# 10. Diagnostic runner
# =============================================================================
print("\n[10/15] Setting up diagnostic runners...")

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


def run_episode_and_extract(test_objects, query_gt, policy, env, ep,
                            prior_violation_history=None):
    """Run episode via EpisodeHarness and extract probe-level diagnostics."""
    if prior_violation_history is None:
        prior_violation_history = set()

    preds, event_log, obs, pre_probe_entropies, pre_decision_entropies = run_policy(policy, env)
    metrics = compute_full_metrics(_unwrap_preds(preds), query_gt)
    effects = policy.get_probe_effects()
    scores = policy.get_selected_scores()

    pv_count = 0; deceptive_pv = 0; normal_pv = 0; pv_repeated = 0
    probe_details = []

    for effect, score in zip(effects, scores):
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
            "risk_penalty": round(score.get("risk_penalty", 0.0), 6),
            "scaled_risk_penalty": round(score.get("scaled_risk_penalty", 0.0), 6),
            "final_score": score["final_score"],
            "injected_deviation_present": injected_present,
            "is_deceptive": is_deceptive,
        }
        if is_violation:
            pv_count += 1
            if is_deceptive: deceptive_pv += 1
            else: normal_pv += 1
            if (oid, action) in prior_violation_history:
                pv_repeated += 1
                detail["repeated"] = True
            else:
                detail["repeated"] = False
            prior_violation_history.add((oid, action))
        probe_details.append(detail)

    return {
        "episode": ep,
        "macro_bal": metrics["macro_query_balanced_accuracy"],
        "per_query": metrics["per_query"],
        "prior_violation_count": pv_count,
        "deceptive_prior_violation_count": deceptive_pv,
        "normal_prior_violation_count": normal_pv,
        "prior_violation_repeated": pv_repeated,
        "probe_details": probe_details,
    }, prior_violation_history


def update_memory_online_from_episode(ep_data, test_objects, memory):
    """Update online memory and funnel data from episode probe results."""
    for detail in ep_data["probe_details"]:
        oid = detail["oid"]
        action = detail["action"]
        outcome = detail["outcome"]
        obj = test_objects.get(oid, {})
        features = obj.get("visible_features", {})
        is_violation, is_eligible, sp = check_prior_violation(features, action, outcome)
        goal = ACTION_TO_QUERY.get(action, "")
        family = detect_type_family(features)
        memory.update(features, goal, action, is_violation, is_eligible)

        if is_eligible:
            for dev_feat in INJECTED_DEVIATION_FEATURES:
                funnel_key = (goal, action, family, dev_feat)
                if funnel_key in funnel_by_key:
                    funnel_by_key[funnel_key]["online_selected_events_with_key"] += 1
                    if is_violation:
                        funnel_by_key[funnel_key]["online_prior_violations_with_key"] += 1
                    else:
                        funnel_by_key[funnel_key]["online_prior_nonviolations_with_key"] += 1


# =============================================================================
# 11. Run all 5 variants
# =============================================================================
print("\n[11/15] Running all 5 variants on deceptive objects...")

RNG_SEED_BASE = SMOKE_SEED + 20000

# --- Shared state ---
# A is re-run alongside each variant for paired comparison.
# We'll use the A from B_family_dev_online run as primary reference.

# --- 11a. A_no_memory (baseline) + B_family_dev_online ---
print("\n  --- Running A_no_memory + B_family_dev_online ---")
fcrm_online = FamilyConditionedRiskMemory(key_mode="family_dev")

all_episodes_a_primary = []
all_episodes_online = []
pv_history_online = set()

for ep in range(N_EPISODES):
    ep_seed = RNG_SEED_BASE + 2 + ep * 100
    ep_rng = random.Random(ep_seed)
    test_oids_local = sorted(test_objects_deceptive.keys())
    ep_positions = MiniMCSimulatorTruth.assign_positions(
        test_oids_local, config.GRID_ROWS, config.GRID_COLS,
        config.AGENT_START, ep_rng)

    # Condition A
    im_a = im_base.clone()
    env_a = MiniMCEnvironment(test_objects_deceptive, dict(ep_positions), initial_budget=BUDGET)
    policy_a = SoftPriorOnlyPolicyV2(im_a, ep_rng, target_probe_count=c15b_probed_n)
    ep_a, _ = run_episode_and_extract(
        test_objects_deceptive, query_gt_deceptive, policy_a, env_a, ep)
    all_episodes_a_primary.append(ep_a)

    # Condition B_family_dev_online
    im_online = im_base.clone()
    env_online = MiniMCEnvironment(test_objects_deceptive, dict(ep_positions), initial_budget=BUDGET)
    policy_online = FamilyConditionedRiskPolicy(im_online, ep_rng, fcrm_online, target_probe_count=c15b_probed_n)
    ep_online, pv_history_online = run_episode_and_extract(
        test_objects_deceptive, query_gt_deceptive, policy_online, env_online, ep,
        prior_violation_history=pv_history_online)
    # Update online memory + funnel from this episode's probe results
    update_memory_online_from_episode(ep_online, test_objects_deceptive, fcrm_online)
    all_episodes_online.append(ep_online)

    if ep == 0 or ep == N_EPISODES - 1:
        print(f"    Ep {ep+1}: A_bal={ep_a['macro_bal']:.4f}  "
              f"B_online_bal={ep_online['macro_bal']:.4f}  "
              f"A_pv={ep_a['prior_violation_count']}  B_online_pv={ep_online['prior_violation_count']}")

agg_a_primary = aggregate_episodes(all_episodes_a_primary)
agg_online = aggregate_episodes(all_episodes_online)

# Update online funnel data support threshold flags
for entry in funnel_data_template:
    key = tuple(entry["key"])
    if key in funnel_by_key:
        fd = funnel_by_key[key]
        fd["online_above_support_threshold"] = (fd["online_prior_violations_with_key"] >= MIN_VIOLATION_SUPPORT)

# Use A from B_family_dev_online as primary A reference
agg_a = agg_a_primary
ep_a = all_episodes_a_primary

# --- 11b. B_family_dev_offline_upper ---
print("\n  --- Running B_family_dev_offline_upper (oracle) ---")
fcrm_offline = build_offline_oracle_memory(test_objects_deceptive, key_mode="family_dev")
offline_stats = fcrm_offline.get_statistics()
print(f"    Offline oracle memory: {offline_stats['eligible_updates']} eligible events, "
      f"{offline_stats['violation_updates']} violations, "
      f"{offline_stats['nonviolation_updates']} nonviolations, "
      f"{offline_stats['risk_weights_nonzero_count']} nonzero weights")

all_episodes_offline = []
pv_history_offline = set()

for ep in range(N_EPISODES):
    ep_seed = RNG_SEED_BASE + 10 + ep * 100
    ep_rng = random.Random(ep_seed)
    test_oids_local = sorted(test_objects_deceptive.keys())
    ep_positions = MiniMCSimulatorTruth.assign_positions(
        test_oids_local, config.GRID_ROWS, config.GRID_COLS,
        config.AGENT_START, ep_rng)

    fcrm_offline_frozen = fcrm_offline.clone()
    im_off = im_base.clone()
    env_off = MiniMCEnvironment(test_objects_deceptive, dict(ep_positions), initial_budget=BUDGET)
    policy_offline = FamilyConditionedRiskPolicy(im_off, ep_rng, fcrm_offline_frozen, target_probe_count=c15b_probed_n)
    ep_off, pv_history_offline = run_episode_and_extract(
        test_objects_deceptive, query_gt_deceptive, policy_offline, env_off, ep,
        prior_violation_history=pv_history_offline)
    all_episodes_offline.append(ep_off)

    if ep == 0 or ep == N_EPISODES - 1:
        print(f"    Ep {ep+1}: offline_bal={ep_off['macro_bal']:.4f}  "
              f"offline_pv={ep_off['prior_violation_count']}")

agg_offline = aggregate_episodes(all_episodes_offline)

# --- 11c. C_family_dev_offline_permuted ---
print("\n  --- Running C_family_dev_offline_permuted ---")
fcrm_offline_base = build_offline_oracle_memory(test_objects_deceptive, key_mode="family_dev")

all_episodes_permuted = []
pv_history_permuted = set()

for ep in range(N_EPISODES):
    ep_seed = RNG_SEED_BASE + 20 + ep * 100
    ep_rng = random.Random(ep_seed)
    test_oids_local = sorted(test_objects_deceptive.keys())
    ep_positions = MiniMCSimulatorTruth.assign_positions(
        test_oids_local, config.GRID_ROWS, config.GRID_COLS,
        config.AGENT_START, ep_rng)

    fcrm_permuted_ep = fcrm_offline_base.clone()
    permuted_map = fcrm_permuted_ep.permute_weights_within_family(ep_rng)

    im_perm = im_base.clone()
    env_perm = MiniMCEnvironment(test_objects_deceptive, dict(ep_positions), initial_budget=BUDGET)
    policy_permuted = FamilyConditionedPermutedPolicy(im_perm, ep_rng, fcrm_permuted_ep, target_probe_count=c15b_probed_n)
    policy_permuted._permuted_map = permuted_map
    ep_perm, pv_history_permuted = run_episode_and_extract(
        test_objects_deceptive, query_gt_deceptive, policy_permuted, env_perm, ep,
        prior_violation_history=pv_history_permuted)
    all_episodes_permuted.append(ep_perm)

    if ep == 0 or ep == N_EPISODES - 1:
        print(f"    Ep {ep+1}: permuted_bal={ep_perm['macro_bal']:.4f}  "
              f"permuted_pv={ep_perm['prior_violation_count']}")

agg_permuted = aggregate_episodes(all_episodes_permuted)

# --- 11d. C_family_only_offline ---
print("\n  --- Running C_family_only_offline ---")
fcrm_family_only = build_offline_oracle_memory(test_objects_deceptive, key_mode="family_only")
fo_stats = fcrm_family_only.get_statistics()
print(f"    Family-only offline memory: {fo_stats['eligible_updates']} eligible events, "
      f"{fo_stats['violation_updates']} violations, "
      f"{fo_stats['risk_weights_nonzero_count']} nonzero weights")

all_episodes_family_only = []
pv_history_fo = set()

for ep in range(N_EPISODES):
    ep_seed = RNG_SEED_BASE + 30 + ep * 100
    ep_rng = random.Random(ep_seed)
    test_oids_local = sorted(test_objects_deceptive.keys())
    ep_positions = MiniMCSimulatorTruth.assign_positions(
        test_oids_local, config.GRID_ROWS, config.GRID_COLS,
        config.AGENT_START, ep_rng)

    fcrm_fo_frozen = fcrm_family_only.clone()
    im_fo = im_base.clone()
    env_fo = MiniMCEnvironment(test_objects_deceptive, dict(ep_positions), initial_budget=BUDGET)
    policy_fo = FamilyConditionedRiskPolicy(im_fo, ep_rng, fcrm_fo_frozen, target_probe_count=c15b_probed_n)
    ep_fo, pv_history_fo = run_episode_and_extract(
        test_objects_deceptive, query_gt_deceptive, policy_fo, env_fo, ep,
        prior_violation_history=pv_history_fo)
    all_episodes_family_only.append(ep_fo)

    if ep == 0 or ep == N_EPISODES - 1:
        print(f"    Ep {ep+1}: family_only_bal={ep_fo['macro_bal']:.4f}  "
              f"family_only_pv={ep_fo['prior_violation_count']}")

agg_family_only = aggregate_episodes(all_episodes_family_only)


# =============================================================================
# 12. Part 1: Evidence Funnel by Family/Action/Deviation Key
# =============================================================================
print("\n[12/15] Part 1: Evidence funnel...")

# Finalize funnel data with all fields
funnel_final = []
for entry in funnel_data_template:
    key = tuple(entry["key"])
    fd = funnel_by_key.get(key, entry)
    funnel_final.append({
        "key": list(key),
        "available_objects_with_key": fd["available_objects_with_key"],
        "available_deceptive_objects_with_key": fd["available_deceptive_objects_with_key"],
        "available_normal_objects_with_key": fd["available_normal_objects_with_key"],
        "high_prior_eligible_objects_with_key": fd["high_prior_eligible_objects_with_key"],
        "online_selected_events_with_key": fd["online_selected_events_with_key"],
        "online_prior_violations_with_key": fd["online_prior_violations_with_key"],
        "online_prior_nonviolations_with_key": fd["online_prior_nonviolations_with_key"],
        "offline_eligible_events_with_key": fd["offline_eligible_events_with_key"],
        "offline_prior_violations_with_key": fd["offline_prior_violations_with_key"],
        "offline_prior_nonviolations_with_key": fd["offline_prior_nonviolations_with_key"],
        "online_above_support_threshold": fd["online_above_support_threshold"],
        "offline_above_support_threshold": fd["offline_above_support_threshold"],
    })

num_keys_total = len(funnel_final)
num_keys_online_above = sum(1 for e in funnel_final if e["online_above_support_threshold"])
num_keys_offline_above = sum(1 for e in funnel_final if e["offline_above_support_threshold"])
num_keys_online_zero = sum(1 for e in funnel_final if e["online_selected_events_with_key"] == 0)
num_keys_offline_zero = sum(1 for e in funnel_final if e["offline_eligible_events_with_key"] == 0)

print(f"  Keys total={num_keys_total}  online_above_support={num_keys_online_above}  "
      f"offline_above_support={num_keys_offline_above}")
print(f"  online_zero_evidence={num_keys_online_zero}  offline_zero_evidence={num_keys_offline_zero}")

part1 = {
    "num_keys_total": num_keys_total,
    "num_keys_online_above_support_threshold": num_keys_online_above,
    "num_keys_offline_above_support_threshold": num_keys_offline_above,
    "num_keys_online_zero_evidence": num_keys_online_zero,
    "num_keys_offline_zero_evidence": num_keys_offline_zero,
    "per_key_details": funnel_final,
}

# =============================================================================
# 13. Part 2: Bottleneck Reason Summary
# =============================================================================
print("\n[13/15] Part 2: Bottleneck reason summary...")

bottleneck_by_family_action = []
for goal in sorted(set(ACTION_TO_QUERY.values())):
    for action in MAIN_CANDIDATE_ACTIONS:
        if ACTION_TO_QUERY.get(action, "") != goal:
            continue
        for family in FAMILY_NAMES:
            # Aggregate across all 4 deviation features for this family/action
            keys_for_group = [e for e in funnel_final
                             if e["key"][0] == goal and e["key"][1] == action
                             and e["key"][2] == family]

            missing_no_objects = 0
            missing_low_prior = 0
            missing_not_selected = 0
            missing_no_failures = 0
            missing_min_threshold = 0

            for k in keys_for_group:
                if k["available_objects_with_key"] == 0:
                    missing_no_objects += 1
                elif k["high_prior_eligible_objects_with_key"] == 0:
                    missing_low_prior += 1
                elif k["online_selected_events_with_key"] == 0:
                    missing_not_selected += 1
                elif k["online_prior_violations_with_key"] == 0:
                    missing_no_failures += 1
                elif not k["online_above_support_threshold"]:
                    missing_min_threshold += 1

            total_eligible = sum(k["offline_eligible_events_with_key"] for k in keys_for_group)
            total_violations = sum(k["offline_prior_violations_with_key"] for k in keys_for_group)

            bottleneck_by_family_action.append({
                "goal": goal,
                "action": action,
                "family": family,
                "num_dev_keys": len(keys_for_group),
                "missing_support_due_to_no_objects": missing_no_objects,
                "missing_support_due_to_low_soft_prior": missing_low_prior,
                "missing_support_due_to_not_selected_online": missing_not_selected,
                "missing_support_due_to_no_failures": missing_no_failures,
                "missing_support_due_to_min_threshold": missing_min_threshold,
                "total_offline_eligible_events": total_eligible,
                "total_offline_violations": total_violations,
            })

part2 = {
    "bottleneck_by_family_action": bottleneck_by_family_action,
}

# =============================================================================
# 14. Part 3: Main Comparison Table
# =============================================================================
print("\n[14/15] Part 3: Main comparison table...")

def compute_effective_probe_count(ep_list):
    total = 0
    for ep in ep_list:
        for d in ep.get("probe_details", []):
            if d.get("probe_utility_signal") == "effective":
                total += 1
    return total

def compute_avoided_effective(ep_a_list, ep_var_list):
    """Count effective probes in A that were avoided by variant."""
    total_avoided = 0
    for ep_a, ep_var in zip(ep_a_list, ep_var_list):
        a_probes = {(d["oid"], d["action"]): d for d in ep_a["probe_details"]}
        var_probes = {(d["oid"], d["action"]): d for d in ep_var["probe_details"]}
        avoided = set(a_probes.keys()) - set(var_probes.keys())
        for oid, action in avoided:
            if a_probes[(oid, action)].get("probe_utility_signal") == "effective":
                total_avoided += 1
    return total_avoided

comparison_variants = [
    ("A_no_memory", agg_a, ep_a, None, False),
    ("B_family_dev_online", agg_online, all_episodes_online, fcrm_online, False),
    ("B_family_dev_offline_upper", agg_offline, all_episodes_offline, fcrm_offline, True),
    ("C_family_dev_offline_permuted", agg_permuted, all_episodes_permuted, fcrm_offline_base, True),
    ("C_family_only_offline", agg_family_only, all_episodes_family_only, fcrm_family_only, True),
]

comparison_rows = []
for label, agg, ep_list, memory, is_oracle in comparison_variants:
    macro_delta = round(agg["mean_macro_bal"] - agg_a["mean_macro_bal"], 4)
    effective_count = compute_effective_probe_count(ep_list) if label != "A_no_memory" else compute_effective_probe_count(ep_a)
    avoided_eff = compute_avoided_effective(ep_a, ep_list) if label != "A_no_memory" else 0

    nonzero_weights = 0
    n_keys_above = 0
    if memory is not None:
        nonzero_weights = memory.get_statistics()["risk_weights_nonzero_count"]
        n_keys_above = memory.get_num_keys_above_support_threshold()

    comparison_rows.append({
        "variant": label,
        "total_prior_violations": agg["total_prior_violations"],
        "deceptive_object_prior_violations": agg["total_deceptive_prior_violations"],
        "normal_object_prior_violations": agg["total_normal_prior_violations"],
        "mean_macro_bal": agg["mean_macro_bal"],
        "macro_delta_vs_A": macro_delta,
        "effective_probe_count": effective_count,
        "avoided_effective_probe_count": avoided_eff,
        "hard_exclusion_used": False,
        "risk_weights_nonzero": nonzero_weights,
        "num_keys_above_support_threshold": n_keys_above,
        "diagnostic_oracle_evidence": is_oracle,
    })

part3 = {"comparison_table": comparison_rows}

# =============================================================================
# 15. Part 4: Offline-vs-Online Risk Comparison
# =============================================================================
print("\n[15/15] Part 4: Offline-vs-online risk comparison...")

online_stats = fcrm_online.get_statistics()
offline_stats_full = fcrm_offline.get_statistics()

# Get top-10 keys by risk weight for each
def get_sorted_risk_keys(memory):
    result = []
    for key in memory.counts:
        w = memory.get_risk_weight(*key)
        if abs(w) > 1e-9:
            result.append((key, w))
    result.sort(key=lambda x: abs(x[1]), reverse=True)
    return result

online_risk_keys = get_sorted_risk_keys(fcrm_online)
offline_risk_keys = get_sorted_risk_keys(fcrm_offline)

online_top10 = set(k for k, _ in online_risk_keys[:10])
offline_top10 = set(k for k, _ in offline_risk_keys[:10])
overlap_top10 = online_top10 & offline_top10

online_key_set = set(k for k, w in online_risk_keys)
offline_key_set = set(k for k, w in offline_risk_keys)
offline_only = offline_key_set - online_key_set
online_only = online_key_set - offline_key_set

online_total_mass = sum(abs(w) for _, w in online_risk_keys)
offline_total_mass = sum(abs(w) for _, w in offline_risk_keys)

part4 = {
    "online_risk_weights_nonzero": online_stats["risk_weights_nonzero_count"],
    "offline_risk_weights_nonzero": offline_stats_full["risk_weights_nonzero_count"],
    "online_total_abs_risk_mass": round(online_total_mass, 6),
    "offline_total_abs_risk_mass": round(offline_total_mass, 6),
    "overlap_top_10_online_offline_keys": len(overlap_top10),
    "overlap_top_10_keys": [list(k) for k in overlap_top10],
    "offline_only_supported_keys": len(offline_only),
    "online_only_supported_keys": len(online_only),
    "top_10_online": [{"key": list(k), "risk_weight": round(w, 6)} for k, w in online_risk_keys[:10]],
    "top_10_offline": [{"key": list(k), "risk_weight": round(w, 6)} for k, w in offline_risk_keys[:10]],
}

# =============================================================================
# 16. Part 5: Boolean Flag Details
# =============================================================================
print("\n[16/15] Part 5: Computing boolean flags...")

# Compute all flags from explicit metric rules
online_evidence_bottleneck = num_keys_online_above < 3
offline_support_sufficient = num_keys_offline_above >= 3
offline_family_dev_beats_A = agg_offline["total_prior_violations"] < agg_a["total_prior_violations"]
offline_family_dev_beats_permuted = agg_offline["total_prior_violations"] < agg_permuted["total_prior_violations"]
offline_family_dev_beats_family_only = agg_offline["total_prior_violations"] < agg_family_only["total_prior_violations"]
family_only_still_dominates = agg_family_only["total_prior_violations"] <= agg_offline["total_prior_violations"]
hard_exclusion_used = False

# Check deceptive family preservation: all deceptive objects should preserve family
deceptive_family_preservation_checks = []
for oid in DECEPTIVE_OIDS:
    obj = test_objects_deceptive.get(oid, {})
    features = obj.get("visible_features", {})
    family = detect_type_family(features)
    expected_family = None
    if "apple" in oid: expected_family = "apple-like"
    elif "wood_log" in oid: expected_family = "wood-like"
    elif "stone_block" in oid: expected_family = "stone-like"
    else: expected_family = "tool-like"
    deceptive_family_preservation_checks.append({
        "oid": oid, "family": family, "expected_family": expected_family,
        "preserved": family == expected_family,
    })

deceptive_family_preservation_valid = all(c["preserved"] for c in deceptive_family_preservation_checks)

# Determine implementation status
if offline_support_sufficient and offline_family_dev_beats_A and offline_family_dev_beats_permuted:
    implementation_status = "pass"
elif num_keys_offline_above > 0 and not offline_family_dev_beats_A:
    implementation_status = "partial"
    failure_reason = "offline_support_exists_but_no_benefit"
elif num_keys_offline_above == 0:
    implementation_status = "partial"
    failure_reason = "zero_offline_keys_above_support"
elif online_evidence_bottleneck and not offline_support_sufficient:
    implementation_status = "partial"
    failure_reason = "both_online_bottleneck_and_offline_insufficient"
else:
    implementation_status = "partial"
    failure_reason = "mixed_results"

boolean_flag_details = {
    "online_evidence_bottleneck_detected": {
        "value": online_evidence_bottleneck,
        "rule": "num_keys_online_above_support_threshold < 3",
        "supporting_values": {
            "num_keys_online_above_support_threshold": num_keys_online_above,
            "num_keys_total": num_keys_total,
            "num_keys_online_zero_evidence": num_keys_online_zero,
        },
    },
    "offline_support_sufficient": {
        "value": offline_support_sufficient,
        "rule": "num_keys_offline_above_support_threshold >= 3",
        "supporting_values": {
            "num_keys_offline_above_support_threshold": num_keys_offline_above,
            "num_keys_total": num_keys_total,
        },
    },
    "offline_family_dev_beats_A": {
        "value": offline_family_dev_beats_A,
        "rule": "B_family_dev_offline_upper total_prior_violations < A_no_memory total_prior_violations",
        "supporting_values": {
            "A_total_pv": agg_a["total_prior_violations"],
            "offline_total_pv": agg_offline["total_prior_violations"],
        },
    },
    "offline_family_dev_beats_permuted": {
        "value": offline_family_dev_beats_permuted,
        "rule": "B_family_dev_offline_upper total_prior_violations < C_family_dev_offline_permuted total_prior_violations",
        "supporting_values": {
            "offline_total_pv": agg_offline["total_prior_violations"],
            "permuted_total_pv": agg_permuted["total_prior_violations"],
        },
    },
    "offline_family_dev_beats_family_only": {
        "value": offline_family_dev_beats_family_only,
        "rule": "B_family_dev_offline_upper total_prior_violations < C_family_only_offline total_prior_violations",
        "supporting_values": {
            "offline_total_pv": agg_offline["total_prior_violations"],
            "family_only_total_pv": agg_family_only["total_prior_violations"],
        },
    },
    "family_only_still_dominates_offline": {
        "value": family_only_still_dominates,
        "rule": "C_family_only_offline total_prior_violations <= B_family_dev_offline_upper total_prior_violations",
        "supporting_values": {
            "family_only_total_pv": agg_family_only["total_prior_violations"],
            "offline_total_pv": agg_offline["total_prior_violations"],
        },
    },
    "hard_exclusion_used": {
        "value": hard_exclusion_used,
        "rule": "hard_exclusion is always false for all variants",
        "supporting_values": {},
    },
    "deceptive_family_preservation_valid": {
        "value": deceptive_family_preservation_valid,
        "rule": "all deceptive objects preserve expected family identity",
        "supporting_values": {
            "total_deceptive_objects": len(DECEPTIVE_OIDS),
            "preserved_family_count": sum(1 for c in deceptive_family_preservation_checks if c["preserved"]),
            "violated_family_count": sum(1 for c in deceptive_family_preservation_checks if not c["preserved"]),
        },
    },
    "implementation_status": {
        "value": implementation_status,
        "rule": "pass if offline support sufficient AND beats A AND beats permuted; partial otherwise",
        "supporting_values": {
            "num_keys_offline_above_support_threshold": num_keys_offline_above,
            "offline_beats_A": offline_family_dev_beats_A,
            "offline_beats_permuted": offline_family_dev_beats_permuted,
        },
    },
    "failure_reason": {
        "value": failure_reason,
        "rule": "derived from combination of support, bottleneck, and comparison metrics",
        "supporting_values": {},
    },
}

# =============================================================================
# 17. Output: JSON
# =============================================================================
elapsed = time.time() - t0
print(f"\n  elapsed={elapsed:.1f}s  writing outputs...")

output = {
    "block_id": "1J21",
    "condition": COND["label"],
    "smoke_seed": SMOKE_SEED,
    "budget": BUDGET,
    "n_episodes": N_EPISODES,
    "risk_alpha": RISK_ALPHA,
    "max_risk": MAX_RISK,
    "prior_violation_threshold": PRIOR_VIOLATION_THRESHOLD,
    "min_violation_support": MIN_VIOLATION_SUPPORT,
    "injected_deviation_features": sorted(INJECTED_DEVIATION_FEATURES),
    "n_deceptive_objects": len(DECEPTIVE_OIDS),
    "c15b_macro_bal": c15b_macro_bal,
    "c15b_probed_n": c15b_probed_n,
    "A_no_memory": {"agg": agg_a, "episodes": ep_a},
    "B_family_dev_online": {
        "agg": agg_online,
        "episodes": all_episodes_online,
        "memory_stats": fcrm_online.get_statistics(),
    },
    "B_family_dev_offline_upper": {
        "agg": agg_offline,
        "episodes": all_episodes_offline,
        "memory_stats": fcrm_offline.get_statistics(),
        "diagnostic_oracle_evidence": True,
    },
    "C_family_dev_offline_permuted": {
        "agg": agg_permuted,
        "episodes": all_episodes_permuted,
    },
    "C_family_only_offline": {
        "agg": agg_family_only,
        "episodes": all_episodes_family_only,
        "memory_stats": fcrm_family_only.get_statistics(),
        "diagnostic_oracle_evidence": True,
    },
    "part1_evidence_funnel": part1,
    "part2_bottleneck_summary": part2,
    "part3_comparison_table": part3,
    "part4_offline_vs_online": part4,
    "part5_boolean_flags": boolean_flag_details,
    "deceptive_family_preservation_checks": deceptive_family_preservation_checks,
    "elapsed_seconds": round(elapsed, 1),
}

out_json_path = os.path.join(CURRENT_DIR, "runs",
    "block1j21_evidence_bottleneck_offline_upper_bound_seed101.json")
os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
with open(out_json_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"  JSON -> {out_json_path}")

# =============================================================================
# 18. Output: Markdown Protocol
# =============================================================================
md_lines = []
md_lines.append("# Block 1J21 -- Evidence Bottleneck and Offline Upper-Bound Diagnostic")
md_lines.append("")
md_lines.append("## 1. Objective")
md_lines.append("")
md_lines.append("Diagnose whether the weak 1J20 B_family_dev result is caused by "
               "insufficient online evidence collection or by family-conditioned "
               "deviation keys being uninformative even with enough evidence.")
md_lines.append("")
md_lines.append("## 2. Setup")
md_lines.append("")
md_lines.append("| Parameter | Value |")
md_lines.append("|-----------|-------|")
md_lines.append(f"| Condition | {COND['label']} |")
md_lines.append(f"| Seed | {SMOKE_SEED} |")
md_lines.append(f"| Budget | {BUDGET} |")
md_lines.append(f"| Episodes | {N_EPISODES} |")
md_lines.append(f"| Prior Violation Threshold | {PRIOR_VIOLATION_THRESHOLD} |")
md_lines.append(f"| Min Violation Support | {MIN_VIOLATION_SUPPORT} |")
md_lines.append(f"| Risk Alpha | {RISK_ALPHA} |")
md_lines.append(f"| Max Risk | {MAX_RISK} |")
md_lines.append("")

md_lines.append("## 3. Baselines")
md_lines.append("")
md_lines.append("| Policy | Macro BAcc |")
md_lines.append("|--------|------------|")
md_lines.append(f"| C15b | {c15b_macro_bal:.4f} |")
md_lines.append("")

# Part 1
md_lines.append("## 4. Part 1: Evidence Funnel Aggregates")
md_lines.append("")
md_lines.append("| Metric | Value |")
md_lines.append("|--------|-------|")
md_lines.append(f"| num_keys_total | {num_keys_total} |")
md_lines.append(f"| num_keys_online_above_support_threshold | {num_keys_online_above} |")
md_lines.append(f"| num_keys_offline_above_support_threshold | {num_keys_offline_above} |")
md_lines.append(f"| num_keys_online_zero_evidence | {num_keys_online_zero} |")
md_lines.append(f"| num_keys_offline_zero_evidence | {num_keys_offline_zero} |")
md_lines.append("")

# Top keys by offline support
md_lines.append("### Top Keys by Offline Violations")
md_lines.append("")
md_lines.append("| Goal | Action | Family | Dev Feature | Off Eligible | Off PV | Off NPV | Online Selected | Online PV | Online Above | Offline Above |")
md_lines.append("|------|--------|--------|-------------|-------------|--------|---------|-----------------|-----------|--------------|---------------|")
for entry in sorted(funnel_final, key=lambda x: x["offline_prior_violations_with_key"], reverse=True)[:20]:
    k = entry["key"]
    md_lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | "
                   f"{entry['offline_eligible_events_with_key']} | "
                   f"{entry['offline_prior_violations_with_key']} | "
                   f"{entry['offline_prior_nonviolations_with_key']} | "
                   f"{entry['online_selected_events_with_key']} | "
                   f"{entry['online_prior_violations_with_key']} | "
                   f"{entry['online_above_support_threshold']} | "
                   f"{entry['offline_above_support_threshold']} |")
md_lines.append("")

# Part 2
md_lines.append("## 5. Part 2: Bottleneck Reason Summary")
md_lines.append("")
md_lines.append("| Goal | Action | Family | Dev Keys | No Obj | Low Prior | Not Selected | No Fail | Min Thresh | Off Eligible | Off PV |")
md_lines.append("|------|--------|--------|----------|--------|-----------|--------------|---------|------------|-------------|--------|")
for row in bottleneck_by_family_action:
    md_lines.append(f"| {row['goal']} | {row['action']} | {row['family']} | "
                   f"{row['num_dev_keys']} | {row['missing_support_due_to_no_objects']} | "
                   f"{row['missing_support_due_to_low_soft_prior']} | "
                   f"{row['missing_support_due_to_not_selected_online']} | "
                   f"{row['missing_support_due_to_no_failures']} | "
                   f"{row['missing_support_due_to_min_threshold']} | "
                   f"{row['total_offline_eligible_events']} | "
                   f"{row['total_offline_violations']} |")
md_lines.append("")

# Part 3
md_lines.append("## 6. Part 3: Main Comparison Table")
md_lines.append("")
md_lines.append("| Variant | Total PV | Deceptive PV | Normal PV | Mean BAcc | Delta vs A | Eff Probes | Avoided Eff | Nonzero W | Keys Above | Oracle |")
md_lines.append("|---------|----------|-------------|-----------|-----------|------------|------------|-------------|-----------|------------|--------|")
for row in comparison_rows:
    md_lines.append(f"| {row['variant']} | {row['total_prior_violations']} | "
                   f"{row['deceptive_object_prior_violations']} | "
                   f"{row['normal_object_prior_violations']} | "
                   f"{row['mean_macro_bal']} | {row['macro_delta_vs_A']} | "
                   f"{row['effective_probe_count']} | {row['avoided_effective_probe_count']} | "
                   f"{row['risk_weights_nonzero']} | {row['num_keys_above_support_threshold']} | "
                   f"{row['diagnostic_oracle_evidence']} |")
md_lines.append("")

# Part 4
md_lines.append("## 7. Part 4: Offline-vs-Online Risk Comparison")
md_lines.append("")
md_lines.append("| Metric | Online | Offline |")
md_lines.append("|--------|--------|---------|")
md_lines.append(f"| risk_weights_nonzero | {part4['online_risk_weights_nonzero']} | {part4['offline_risk_weights_nonzero']} |")
md_lines.append(f"| total_abs_risk_mass | {part4['online_total_abs_risk_mass']} | {part4['offline_total_abs_risk_mass']} |")
md_lines.append(f"| overlap_top_10_keys | {part4['overlap_top_10_online_offline_keys']} | - |")
md_lines.append(f"| offline_only_supported_keys | - | {part4['offline_only_supported_keys']} |")
md_lines.append(f"| online_only_supported_keys | {part4['online_only_supported_keys']} | - |")
md_lines.append("")

md_lines.append("### Top 10 Online Risk Keys")
md_lines.append("")
md_lines.append("| Key | Risk Weight |")
md_lines.append("|-----|-------------|")
for entry in part4["top_10_online"]:
    md_lines.append(f"| {entry['key']} | {entry['risk_weight']} |")
md_lines.append("")

md_lines.append("### Top 10 Offline Risk Keys")
md_lines.append("")
md_lines.append("| Key | Risk Weight |")
md_lines.append("|-----|-------------|")
for entry in part4["top_10_offline"]:
    md_lines.append(f"| {entry['key']} | {entry['risk_weight']} |")
md_lines.append("")

# Part 5
md_lines.append("## 8. Part 5: Boolean Flag Details")
md_lines.append("")
for flag_name, flag_info in boolean_flag_details.items():
    md_lines.append(f"### {flag_name}: {flag_info['value']}")
    md_lines.append(f"  Rule: {flag_info['rule']}")
    if flag_info["supporting_values"]:
        for sv_key, sv_val in flag_info["supporting_values"].items():
            md_lines.append(f"  - {sv_key}: {sv_val}")
    md_lines.append("")

# Summary
md_lines.append("## 9. Summary")
md_lines.append("")
md_lines.append("```")
md_lines.append("[block_done]")
md_lines.append(f"block_id=1J21")
md_lines.append(f"A_total_pv={agg_a['total_prior_violations']}")
md_lines.append(f"B_family_dev_online_total_pv={agg_online['total_prior_violations']}")
md_lines.append(f"B_family_dev_offline_upper_total_pv={agg_offline['total_prior_violations']}")
md_lines.append(f"C_family_dev_offline_permuted_total_pv={agg_permuted['total_prior_violations']}")
md_lines.append(f"C_family_only_offline_total_pv={agg_family_only['total_prior_violations']}")
md_lines.append(f"B_family_dev_online_macro_delta_vs_A={round(agg_online['mean_macro_bal'] - agg_a['mean_macro_bal'], 4)}")
md_lines.append(f"B_family_dev_offline_macro_delta_vs_A={round(agg_offline['mean_macro_bal'] - agg_a['mean_macro_bal'], 4)}")
md_lines.append(f"num_keys_online_above_support_threshold={num_keys_online_above}")
md_lines.append(f"num_keys_offline_above_support_threshold={num_keys_offline_above}")
md_lines.append(f"online_evidence_bottleneck_detected={online_evidence_bottleneck}")
md_lines.append(f"offline_support_sufficient={offline_support_sufficient}")
md_lines.append(f"offline_family_dev_beats_A={offline_family_dev_beats_A}")
md_lines.append(f"offline_family_dev_beats_permuted={offline_family_dev_beats_permuted}")
md_lines.append(f"offline_family_dev_beats_family_only={offline_family_dev_beats_family_only}")
md_lines.append(f"family_only_still_dominates_offline={family_only_still_dominates}")
md_lines.append(f"deceptive_family_preservation_valid={deceptive_family_preservation_valid}")
md_lines.append(f"hard_exclusion_used={hard_exclusion_used}")
md_lines.append(f"implementation_status={implementation_status}")
md_lines.append(f"failure_reason={failure_reason}")
md_lines.append(f"elapsed={elapsed:.1f}s")
md_lines.append("```")

out_md_path = os.path.join(CURRENT_DIR, "protocols",
    "block1j21_evidence_bottleneck_offline_upper_bound_seed101.md")
os.makedirs(os.path.dirname(out_md_path), exist_ok=True)
with open(out_md_path, "w") as f:
    f.write("\n".join(md_lines))
print(f"  MD  -> {out_md_path}")

print("\n" + "=" * 70)
print("[block_done]")
print(f"block_id=1J21")
print(f"A_total_pv={agg_a['total_prior_violations']}")
print(f"B_family_dev_online_total_pv={agg_online['total_prior_violations']}")
print(f"B_family_dev_offline_upper_total_pv={agg_offline['total_prior_violations']}")
print(f"C_family_dev_offline_permuted_total_pv={agg_permuted['total_prior_violations']}")
print(f"C_family_only_offline_total_pv={agg_family_only['total_prior_violations']}")
print(f"B_family_dev_online_macro_delta_vs_A={round(agg_online['mean_macro_bal'] - agg_a['mean_macro_bal'], 4)}")
print(f"B_family_dev_offline_macro_delta_vs_A={round(agg_offline['mean_macro_bal'] - agg_a['mean_macro_bal'], 4)}")
print(f"num_keys_online_above_support_threshold={num_keys_online_above}")
print(f"num_keys_offline_above_support_threshold={num_keys_offline_above}")
print(f"online_evidence_bottleneck_detected={online_evidence_bottleneck}")
print(f"offline_support_sufficient={offline_support_sufficient}")
print(f"offline_family_dev_beats_A={offline_family_dev_beats_A}")
print(f"offline_family_dev_beats_permuted={offline_family_dev_beats_permuted}")
print(f"offline_family_dev_beats_family_only={offline_family_dev_beats_family_only}")
print(f"family_only_still_dominates_offline={family_only_still_dominates}")
print(f"deceptive_family_preservation_valid={deceptive_family_preservation_valid}")
print(f"hard_exclusion_used={hard_exclusion_used}")
print(f"implementation_status={implementation_status}")
print(f"failure_reason={failure_reason}")
print(f"elapsed={elapsed:.1f}s")
print("=" * 70)
