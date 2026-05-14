"""
Block 1J22 -- Deviation Feature Informativeness and Risk-Weight Audit.

Audits why 1J21 offline family-conditioned deviation memory failed even with
oracle evidence. Five parts:
  Part 1: Implementation sanity audit (risk-weight sign, penalty direction, support)
  Part 2: Deviation feature informativeness table
  Part 3: Best possible deviation-rule diagnostic (oracle rules)
  Part 4: Counterproductive-weight audit
  Part 5: Boolean flags
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
    MiniMCPolicy,
    MAIN_CANDIDATE_ACTIONS, ACTION_TO_FEATURE,
    _compute_utility, _compute_entropy,
)
from instance_outcome_memory import InstanceOutcomeMemory
from run_004_5n1 import run_phase_a_training
from sparse_outcome_collector import collect_sparse_probe_outcomes
from subtype_objects import (
    generate_subtype_objects_deterministic, compute_query_ground_truth,
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
print("Block 1J22 -- Deviation Feature Informativeness and Risk-Weight Audit")
print(f"  condition={COND['label']}  seed={SMOKE_SEED}  budget={BUDGET}")
print(f"  episodes={N_EPISODES}  risk_alpha={RISK_ALPHA}  max_risk={MAX_RISK}")
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
print(f"  {len(test_oids)} standard test objects, {len(QUERY_NAMES)} queries")

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

# =============================================================================
# 3. Metric helpers
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
    return {
        "per_query": per_query,
        "macro_query_balanced_accuracy": round(macro_bal, 6),
        "macro_query_accuracy": round(macro_acc, 6),
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
# 4. Default Goal Prior Table + type family detection
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
# 5. FamilyConditionedRiskMemory (identical to 1J21)
# =============================================================================
print("\n[5/12] Defining FamilyConditionedRiskMemory...")

class FamilyConditionedRiskMemory:
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

    def get_raw_counts(self, *key):
        """Return the raw 4-counter pseudo-counts for a key."""
        return dict(self.counts[key])

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
# 6. Policy Classes (from 1J21)
# =============================================================================
print("\n[6/12] Defining policy classes...")

class SoftPriorOnlyPolicyV2(MiniMCPolicy):
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
        for action in MAIN_CANDIDATE_ACTIONS:
            base_prior = get_goal_soft_prior(features, action)
            norm_cost = view.probe_cost / max(view.initial_budget, 0.001)
            raw_score = base_prior - COST_WEIGHT * norm_cost
            score = max(EXPLORATION_FLOOR, raw_score)
            if score > best_score:
                best_score = score; best_action = action
        self._probed_oids.add(object_id)
        self._selected_scores.append({
            "oid": object_id, "action": best_action,
            "prior": base_prior if best_action else EXPLORATION_FLOOR,
            "raw_score": round(score, 6),
            "final_score": round(score, 6), "risk_adjustment": 0.0,
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
        best_info = (EXPLORATION_FLOOR, 0.0, 0.0, EXPLORATION_FLOOR, EXPLORATION_FLOOR)
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


# =============================================================================
# 7. C15b reference + offline oracle
# =============================================================================
print("\n[7/12] Running C15b reference + building offline oracle...")

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


def get_ground_truth_outcome(obj, action):
    profile = obj.get("hidden_affordance_profile", {})
    result = profile.get(action, "success")
    return 1.0 if result == "success" else 0.0


def build_offline_oracle_memory(test_objects, key_mode="family_dev"):
    memory = FamilyConditionedRiskMemory(key_mode=key_mode)
    for oid, obj in test_objects.items():
        features = obj.get("visible_features", {})
        for action in MAIN_CANDIDATE_ACTIONS:
            soft_prior = get_goal_soft_prior(features, action)
            is_eligible = soft_prior >= PRIOR_VIOLATION_THRESHOLD
            if not is_eligible:
                continue
            outcome = get_ground_truth_outcome(obj, action)
            is_violation = outcome <= 0.01
            goal = ACTION_TO_QUERY.get(action, "")
            memory.update(features, goal, action, is_violation, is_eligible)
    return memory


# Build the full offline oracle (family_dev mode)
fcrm_offline = build_offline_oracle_memory(test_objects_deceptive, key_mode="family_dev")
print(f"  Offline oracle built: {fcrm_offline.eligible_updates} eligible, "
      f"{fcrm_offline.violation_updates} violations")


# =============================================================================
# 8. Run A_no_memory baseline (needed for comparison)
# =============================================================================
print("\n[8/12] Running A_no_memory baseline...")

RNG_SEED_BASE = SMOKE_SEED + 20000

def run_episode_and_extract(test_objects, query_gt, policy, env, ep,
                            prior_violation_history=None):
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


def aggregate_episodes(ep_list):
    mbs = [e["macro_bal"] for e in ep_list]
    pvs = [e.get("prior_violation_count", 0) for e in ep_list]
    deceptive_pvs = [e.get("deceptive_prior_violation_count", 0) for e in ep_list]
    normal_pvs = [e.get("normal_prior_violation_count", 0) for e in ep_list]
    return {
        "mean_macro_bal": round(float(np.mean(mbs)), 4),
        "total_prior_violations": sum(pvs),
        "total_deceptive_prior_violations": sum(deceptive_pvs),
        "total_normal_prior_violations": sum(normal_pvs),
        "per_episode": ep_list,
    }


# Run A_no_memory baseline
all_episodes_a = []
pv_history_a = set()

for ep in range(N_EPISODES):
    ep_seed = RNG_SEED_BASE + 2 + ep * 100
    ep_rng = random.Random(ep_seed)
    test_oids_local = sorted(test_objects_deceptive.keys())
    ep_positions = MiniMCSimulatorTruth.assign_positions(
        test_oids_local, config.GRID_ROWS, config.GRID_COLS,
        config.AGENT_START, ep_rng)
    im_a = im_base.clone()
    env_a = MiniMCEnvironment(test_objects_deceptive, dict(ep_positions), initial_budget=BUDGET)
    policy_a = SoftPriorOnlyPolicyV2(im_a, ep_rng, target_probe_count=c15b_probed_n)
    ep_a, pv_history_a = run_episode_and_extract(
        test_objects_deceptive, query_gt_deceptive, policy_a, env_a, ep,
        prior_violation_history=pv_history_a)
    all_episodes_a.append(ep_a)

agg_a = aggregate_episodes(all_episodes_a)
print(f"  A_no_memory: total_pv={agg_a['total_prior_violations']}, "
      f"macro_bal={agg_a['mean_macro_bal']:.4f}")

# =============================================================================
# 9. Run B_family_dev_offline_upper (from 1J21) for comparison
# =============================================================================
print("\n[9/12] Running B_family_dev_offline_upper...")

all_episodes_offline = []
pv_history_offline = set()

for ep in range(N_EPISODES):
    ep_seed = RNG_SEED_BASE + 10 + ep * 100
    ep_rng = random.Random(ep_seed)
    test_oids_local = sorted(test_objects_deceptive.keys())
    ep_positions = MiniMCSimulatorTruth.assign_positions(
        test_oids_local, config.GRID_ROWS, config.GRID_COLS,
        config.AGENT_START, ep_rng)
    fcrm_frozen = fcrm_offline.clone()
    im_off = im_base.clone()
    env_off = MiniMCEnvironment(test_objects_deceptive, dict(ep_positions), initial_budget=BUDGET)
    policy_offline = FamilyConditionedRiskPolicy(im_off, ep_rng, fcrm_frozen, target_probe_count=c15b_probed_n)
    ep_off, pv_history_offline = run_episode_and_extract(
        test_objects_deceptive, query_gt_deceptive, policy_offline, env_off, ep,
        prior_violation_history=pv_history_offline)
    all_episodes_offline.append(ep_off)

agg_offline = aggregate_episodes(all_episodes_offline)
print(f"  B_offline: total_pv={agg_offline['total_prior_violations']}, "
      f"macro_bal={agg_offline['mean_macro_bal']:.4f}")

# =============================================================================
# 10. PART 1: Implementation Sanity Audit
# =============================================================================
print("\n[10/12] PART 1: Implementation sanity audit...")

# --- 1a. Risk-weight sign check ---
risk_audit_rows = []
risk_sign_errors = 0

for key in fcrm_offline.counts:
    if fcrm_offline.key_mode == "family_dev" and len(key) == 4:
        goal, action, family, dev_feat = key
        c = fcrm_offline.get_raw_counts(*key)
        v_with = c["violation_with"]
        nv_with = c["nonviolation_with"]
        v_without = c["violation_without"]
        nv_without = c["nonviolation_without"]

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
        clipped = max(0.0, min(raw_lr, MAX_RISK))
        support = total_v + total_nv - 4.0
        support = max(0.0, support)
        reliability = support / (support + RISK_ALPHA)
        final_w = reliability * clipped

        actual_violations = (v_with + v_without) - 2.0
        above_support = actual_violations >= MIN_VIOLATION_SUPPORT

        # Sign check
        sign_ok = True
        if p_feat_given_v > p_feat_given_nv and final_w <= 0:
            sign_ok = False
            risk_sign_errors += 1
        if p_feat_given_v < p_feat_given_nv and final_w > 0:
            sign_ok = False
            risk_sign_errors += 1

        risk_audit_rows.append({
            "key": list(key),
            "violation_with": v_with - 1.0,       # remove pseudo-count
            "nonviolation_with": nv_with - 1.0,
            "violation_without": v_without - 1.0,
            "nonviolation_without": nv_without - 1.0,
            "p_feature_given_violation": round(p_feat_given_v, 6),
            "p_feature_given_nonviolation": round(p_feat_given_nv, 6),
            "raw_log_ratio": round(raw_lr, 6),
            "clipped_risk_weight": round(clipped, 6),
            "shrinkage_factor": round(reliability, 6),
            "final_risk_weight": round(final_w, 6),
            "sign_correct": sign_ok,
            "above_support_threshold": above_support,
        })

risk_sign_error_detected = risk_sign_errors > 0
print(f"  Risk sign errors: {risk_sign_errors}  -> risk_sign_error_detected={risk_sign_error_detected}")

# --- 1b. Penalty-direction check ---
# For each candidate probing decision, verify that positive risk lowers score
penalty_direction_errors = 0
penalty_check_rows = []

for ep_data in all_episodes_offline:
    for detail in ep_data["probe_details"]:
        oid = detail["oid"]
        action = detail["action"]
        obj = test_objects_deceptive.get(oid, {})
        features = obj.get("visible_features", {})
        norm_cost = 0.1  # approximate

        # Score before risk (same as A)
        base_prior = get_goal_soft_prior(features, action)
        score_before = max(EXPLORATION_FLOOR, base_prior - COST_WEIGHT * norm_cost)

        # Score after risk
        risk_penalty = detail.get("risk_penalty", 0.0)
        score_after = max(EXPLORATION_FLOOR,
                          base_prior * math.exp(-risk_penalty) - COST_WEIGHT * norm_cost)

        direction_valid = True
        if risk_penalty > 0.001 and score_after > score_before:
            direction_valid = False
            penalty_direction_errors += 1

        penalty_check_rows.append({
            "oid": oid, "action": action,
            "candidate_score_before_risk": round(score_before, 6),
            "risk_penalty": round(risk_penalty, 6),
            "candidate_score_after_risk": round(score_after, 6),
            "score_direction_valid": direction_valid,
        })

penalty_direction_error_detected = penalty_direction_errors > 0
print(f"  Penalty direction errors: {penalty_direction_errors}  -> penalty_direction_error_detected={penalty_direction_error_detected}")

# --- 1c. Support-definition check ---
# Count online-style support (unique object-action events per key)
# Count offline-style support (unique object-action candidates per key)

online_support_counts = {}
offline_support_counts = {}

for key in fcrm_offline.counts:
    if len(key) == 4:
        goal, action, family, dev_feat = key
        c = fcrm_offline.get_raw_counts(*key)

        # Offline: unique (object, action) candidates that were eligible
        offline_eligible = 0
        offline_violations = 0
        for oid, obj in test_objects_deceptive.items():
            features = obj.get("visible_features", {})
            obj_family = detect_type_family(features)
            has_dev = features.get(dev_feat, False)
            if obj_family == family and has_dev:
                sp = get_goal_soft_prior(features, action)
                if sp >= PRIOR_VIOLATION_THRESHOLD:
                    offline_eligible += 1
                    outcome = get_ground_truth_outcome(obj, action)
                    if outcome <= 0.01:
                        offline_violations += 1

        offline_support_counts[key] = {
            "eligible_events": offline_eligible,
            "violation_events": offline_violations,
            "event_unit": "unique_object_action_candidates",
        }

        # Online-style: the same counter structure counts repeated events
        actual_v = (c["violation_with"] + c["violation_without"]) - 2.0
        actual_nv = (c["nonviolation_with"] + c["nonviolation_without"]) - 2.0
        online_support_counts[key] = {
            "eligible_events": int(actual_v + actual_nv),
            "violation_events": int(actual_v),
            "event_unit": "episode_repeated_probe_events",
        }

# Find online-style keys above support for comparison
online_keys_above = sum(1 for k, v in online_support_counts.items()
                       if v["violation_events"] >= MIN_VIOLATION_SUPPORT)
offline_keys_above = sum(1 for k, v in offline_support_counts.items()
                        if v["violation_events"] >= MIN_VIOLATION_SUPPORT)

# Support inconsistency: check if online/offline counting methods differ
# Online counts repeated episodes, offline counts unique candidates
support_count_inconsistency_detected = False
online_total_events = sum(v["eligible_events"] for v in online_support_counts.values())
offline_total_events = sum(v["eligible_events"] for v in offline_support_counts.values())
# Inconsistency is detected if the ratio is close to N_EPISODES (repetition)
if online_total_events > 0 and offline_total_events > 0:
    ratio = online_total_events / offline_total_events
    # If ratio is near N_EPISODES, online repeats were counted multiple times
    if ratio > 1.5:
        support_count_inconsistency_detected = True

print(f"  Online total events: {online_total_events}, Offline total events: {offline_total_events}")
print(f"  Online/Offline ratio: {online_total_events/offline_total_events:.2f}" if offline_total_events > 0 else "  N/A")
print(f"  Online keys above support: {online_keys_above}, Offline keys above support: {offline_keys_above}")
print(f"  support_count_inconsistency_detected={support_count_inconsistency_detected}")

part1 = {
    "risk_sign_audit": {
        "total_keys_checked": len(risk_audit_rows),
        "risk_sign_errors": risk_sign_errors,
        "risk_sign_error_detected": risk_sign_error_detected,
        "per_key_details": risk_audit_rows,
    },
    "penalty_direction_audit": {
        "total_candidates_checked": len(penalty_check_rows),
        "penalty_direction_errors": penalty_direction_errors,
        "penalty_direction_error_detected": penalty_direction_error_detected,
        "per_candidate_details": penalty_check_rows,
    },
    "support_definition_audit": {
        "online_support_event_unit": "episode_repeated_probe_events",
        "offline_support_event_unit": "unique_object_action_candidates",
        "online_repeats_episode_events": True,
        "offline_counts_unique_candidates": True,
        "online_total_eligible_events": online_total_events,
        "offline_total_eligible_events": offline_total_events,
        "online_keys_above_support": online_keys_above,
        "offline_keys_above_support": offline_keys_above,
        "support_count_inconsistency_detected": support_count_inconsistency_detected,
    },
}

# =============================================================================
# 11. PART 2: Deviation Feature Informativeness Table
# =============================================================================
print("\n[11/12] PART 2: Deviation feature informativeness table...")

def compute_informativeness(test_objects, memory_offline):
    """For each (goal, action, family, dev_feature), compute detailed stats."""
    rows = []

    for key in memory_offline.counts:
        if len(key) != 4:
            continue
        goal, action, family, dev_feat = key

        n_with_feat = 0
        n_without_feat = 0
        violations_with = 0
        nonviolations_with = 0
        violations_without = 0
        nonviolations_without = 0

        for oid, obj in test_objects.items():
            features = obj.get("visible_features", {})
            obj_family = detect_type_family(features)
            has_dev = features.get(dev_feat, False)
            sp = get_goal_soft_prior(features, action)
            is_eligible = sp >= PRIOR_VIOLATION_THRESHOLD

            if obj_family != family:
                continue
            if not is_eligible:
                continue

            outcome = get_ground_truth_outcome(obj, action)
            is_violation = outcome <= 0.01

            if has_dev:
                n_with_feat += 1
                if is_violation: violations_with += 1
                else: nonviolations_with += 1
            else:
                n_without_feat += 1
                if is_violation: violations_without += 1
                else: nonviolations_without += 1

        total_with = n_with_feat
        total_without = n_without_feat
        violation_rate_with = violations_with / max(total_with, 1)
        violation_rate_without = violations_without / max(total_without, 1)
        risk_rate_delta = violation_rate_with - violation_rate_without

        # Odds ratio
        a = violations_with; b = nonviolations_with
        c = violations_without; d = nonviolations_without
        if b == 0: b = 0.5
        if c == 0: c = 0.5
        if d == 0: d = 0.5
        if a == 0: a = 0.5
        odds_ratio = (a * d) / (b * c) if (b * c) > 0 else 1.0
        log_odds = math.log(odds_ratio) if odds_ratio > 0 else 0.0

        # Mutual information (simple binning)
        total_n = a + b + c + d
        if total_n > 0:
            p_v = (a + c) / total_n
            p_nv = (b + d) / total_n
            p_f = (a + b) / total_n
            p_nf = (c + d) / total_n
            mi = 0.0
            # P(v, f)
            p_vf = a / total_n
            if p_vf > 0 and p_v * p_f > 0: mi += p_vf * math.log(p_vf / (p_v * p_f))
            # P(v, nf)
            p_vnf = c / total_n
            if p_vnf > 0 and p_v * p_nf > 0: mi += p_vnf * math.log(p_vnf / (p_v * p_nf))
            # P(nv, f)
            p_nvf = b / total_n
            if p_nvf > 0 and p_nv * p_f > 0: mi += p_nvf * math.log(p_nvf / (p_nv * p_f))
            # P(nv, nf)
            p_nvnf = d / total_n
            if p_nvnf > 0 and p_nv * p_nf > 0: mi += p_nvnf * math.log(p_nvnf / (p_nv * p_nf))
        else:
            mi = 0.0

        total_violations = violations_with + violations_without
        above_support = total_violations >= MIN_VIOLATION_SUPPORT

        c_mem = memory_offline.get_raw_counts(*key)
        final_w = memory_offline.get_risk_weight(*key)

        rows.append({
            "key": list(key),
            "n_with_feature": n_with_feat,
            "n_without_feature": n_without_feat,
            "violations_with_feature": violations_with,
            "nonviolations_with_feature": nonviolations_with,
            "violations_without_feature": violations_without,
            "nonviolations_without_feature": nonviolations_without,
            "violation_rate_with_feature": round(violation_rate_with, 6),
            "violation_rate_without_feature": round(violation_rate_without, 6),
            "risk_rate_delta": round(risk_rate_delta, 6),
            "odds_ratio": round(odds_ratio, 6),
            "log_odds": round(log_odds, 6),
            "mutual_information": round(mi, 6),
            "above_support_threshold": above_support,
            "final_risk_weight": round(final_w, 6),
        })

    return rows


info_rows = compute_informativeness(test_objects_deceptive, fcrm_offline)

# Aggregate by family, action, goal, dev_feature
def aggregate_info_by(rows, field_idx):
    groups = defaultdict(lambda: {
        "n_keys": 0, "total_v_with": 0, "total_nv_with": 0,
        "total_v_without": 0, "total_nv_without": 0,
        "keys_above_support": 0, "nonzero_weights": 0,
    })
    for r in rows:
        key = r["key"]
        group_val = key[field_idx]
        g = groups[group_val]
        g["n_keys"] += 1
        g["total_v_with"] += r["violations_with_feature"]
        g["total_nv_with"] += r["nonviolations_with_feature"]
        g["total_v_without"] += r["violations_without_feature"]
        g["total_nv_without"] += r["nonviolations_without_feature"]
        if r["above_support_threshold"]: g["keys_above_support"] += 1
        if abs(r["final_risk_weight"]) > 1e-9: g["nonzero_weights"] += 1

    result = []
    for gv, g in sorted(groups.items()):
        total_with = g["total_v_with"] + g["total_nv_with"]
        total_without = g["total_v_without"] + g["total_nv_without"]
        rate_with = g["total_v_with"] / max(total_with, 1)
        rate_without = g["total_v_without"] / max(total_without, 1)
        result.append({
            "group": gv,
            "n_keys": g["n_keys"],
            "total_violations_with": g["total_v_with"],
            "total_violations_without": g["total_v_without"],
            "violation_rate_with": round(rate_with, 6),
            "violation_rate_without": round(rate_without, 6),
            "rate_delta": round(rate_with - rate_without, 6),
            "keys_above_support": g["keys_above_support"],
            "nonzero_weights": g["nonzero_weights"],
        })
    return result

part2 = {
    "per_key_details": info_rows,
    "aggregate_by_family": aggregate_info_by(info_rows, 2),
    "aggregate_by_action": aggregate_info_by(info_rows, 1),
    "aggregate_by_goal": aggregate_info_by(info_rows, 0),
    "aggregate_by_dev_feature": aggregate_info_by(info_rows, 3),
}

# Determine if deviation features are informative
# A dev feature is informative if at least one key has |risk_rate_delta| > 0.1
informative_keys = [r for r in info_rows if abs(r["risk_rate_delta"]) > 0.1 and r["above_support_threshold"]]
deviation_features_informative = len(informative_keys) > 0
print(f"  Informative keys (|delta|>0.1 & above support): {len(informative_keys)}")
print(f"  deviation_features_informative={deviation_features_informative}")

# =============================================================================
# 12. PART 3: Best Possible Deviation-Rule Diagnostic
# =============================================================================
print("\n[12/12] PART 3: Best possible deviation-rule diagnostic...")

def evaluate_rule_policy(test_objects, query_gt, rule_func, rng_seed_base, label):
    """Evaluate a diagnostic rule by running episodes with a rule-based scorer."""
    all_eps = []
    pv_hist = set()

    class RuleBasedPolicy(SoftPriorOnlyPolicyV2):
        def __init__(self, instance_memory, rng, rule_func, target_probe_count=8):
            super().__init__(instance_memory, rng, target_probe_count)
            self._rule_func = rule_func
            self._variant = label

        def _score_candidate(self, features, norm_cost):
            best_score = float('-inf')
            for action in MAIN_CANDIDATE_ACTIONS:
                base_prior = get_goal_soft_prior(features, action)
                risk_penalty = self._rule_func(features, action)
                raw_score = base_prior * math.exp(-risk_penalty) - COST_WEIGHT * norm_cost
                score = max(EXPLORATION_FLOOR, raw_score)
                if score > best_score: best_score = score
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
            best_risk = 0.0
            for action in MAIN_CANDIDATE_ACTIONS:
                norm_cost = view.probe_cost / max(view.initial_budget, 0.001)
                base_prior = get_goal_soft_prior(features, action)
                risk_penalty = self._rule_func(features, action)
                raw_score = base_prior * math.exp(-risk_penalty) - COST_WEIGHT * norm_cost
                score = max(EXPLORATION_FLOOR, raw_score)
                if score > best_score:
                    best_score = score; best_action = action
                    best_risk = risk_penalty
            self._probed_oids.add(object_id)
            self._selected_scores.append({
                "oid": object_id, "action": best_action,
                "prior": get_goal_soft_prior(features, best_action) if best_action else EXPLORATION_FLOOR,
                "risk_penalty": round(best_risk, 6),
                "final_score": round(best_score, 6),
            })
            return True, best_action

    for ep in range(N_EPISODES):
        ep_seed = rng_seed_base + ep * 100
        ep_rng = random.Random(ep_seed)
        test_oids_local = sorted(test_objects.keys())
        ep_positions = MiniMCSimulatorTruth.assign_positions(
            test_oids_local, config.GRID_ROWS, config.GRID_COLS,
            config.AGENT_START, ep_rng)
        im_ep = im_base.clone()
        env_ep = MiniMCEnvironment(test_objects, dict(ep_positions), initial_budget=BUDGET)
        policy_ep = RuleBasedPolicy(im_ep, ep_rng, rule_func, target_probe_count=c15b_probed_n)
        ep_data, pv_hist = run_episode_and_extract(
            test_objects, query_gt, policy_ep, env_ep, ep, prior_violation_history=pv_hist)
        all_eps.append(ep_data)

    agg = aggregate_episodes(all_eps)
    def count_eff(ep_list):
        total = 0
        for ep in ep_list:
            for d in ep.get("probe_details", []):
                if d.get("probe_utility_signal") == "effective":
                    total += 1
        return total
    def count_avoided_eff(ep_a_list, ep_var_list):
        total = 0
        for ep_a, ep_var in zip(ep_a_list, ep_var_list):
            a_probes = {(d["oid"], d["action"]): d for d in ep_a["probe_details"]}
            var_probes = {(d["oid"], d["action"]): d for d in ep_var["probe_details"]}
            avoided = set(a_probes.keys()) - set(var_probes.keys())
            for oid, action in avoided:
                if a_probes[(oid, action)].get("probe_utility_signal") == "effective":
                    total += 1
        return total
    eff_count = count_eff(all_eps)
    avoided_eff = count_avoided_eff(all_episodes_a, all_eps)

    # Count avoided PV vs A
    a_probe_set = set()
    for ep_a in all_episodes_a:
        for d in ep_a["probe_details"]:
            if d.get("is_violation"):
                a_probe_set.add((d["oid"], d["action"]))
    var_probe_set = set()
    for ep_v in all_eps:
        for d in ep_v["probe_details"]:
            if d.get("is_violation"):
                var_probe_set.add((d["oid"], d["action"]))
    avoided_pv = len(a_probe_set - var_probe_set)
    avoided_nv = 0
    for ep_a in all_episodes_a:
        for d in ep_a["probe_details"]:
            if not d.get("is_violation") and d.get("is_eligible"):
                key = (d["oid"], d["action"])
                if key not in var_probe_set and key in set(
                    (dd["oid"], dd["action"]) for ep_v in all_eps for dd in ep_v["probe_details"]):
                    pass  # would need more careful tracking

    return {
        "label": label,
        "agg": agg,
        "episodes": all_eps,
        "effective_probe_count": eff_count,
        "avoided_effective_probe_count": avoided_eff,
        "avoided_prior_violation_count": avoided_pv,
        "diagnostic_oracle_rule": True,
    }


# --- Oracle_dev_feature_rule: best per-family/action dev feature ---
# From offline data, find the dev feature with highest |risk_rate_delta| per (family, action)
best_dev_feature_per_group = {}
for r in info_rows:
    goal, action, family, dev_feat = r["key"]
    group = (action, family)
    if group not in best_dev_feature_per_group:
        best_dev_feature_per_group[group] = []
    best_dev_feature_per_group[group].append((dev_feat, r["risk_rate_delta"], r["violations_with_feature"]))

# Pick the best dev feature per group based on violation_rate_with - violation_rate_without
selected_dev_features = {}
for group, candidates in best_dev_feature_per_group.items():
    action, family = group
    # Filter to those with above-support data
    best = None; best_score = -999
    for dev_feat, delta, v_count in candidates:
        if v_count >= MIN_VIOLATION_SUPPORT:
            if abs(delta) > best_score:
                best_score = abs(delta)
                best = dev_feat
    if best:
        selected_dev_features[(action, family)] = (best, best_score)

print(f"  Selected {len(selected_dev_features)} best dev features for oracle rule")

def oracle_dev_feature_rule(features, action):
    """Use the pre-selected best dev feature per (action, family).
    Returns a high penalty (2.0) if the 'bad' deviation is present."""
    family = detect_type_family(features)
    key_grp = (action, family)
    if key_grp in selected_dev_features:
        best_dev, score = selected_dev_features[key_grp]
        if features.get(best_dev, False):
            return 2.0  # high penalty for the risky deviation feature
    return 0.0


# --- Oracle_deceptive_object_rule ---
# Uses true deceptive flag (from hidden_affordance_profile) to avoid deceptive PVs
def oracle_deceptive_rule(features, action):
    """This is a non-deployable oracle that uses true deceptive labels."""
    return 0.0  # Applied per-object in policy, but we access via test_objects

# We need a wrapper that has access to the full test_objects with deceptive labels
def make_oracle_deceptive_rule(test_objects_ref):
    def _rule(features, action):
        return 0.0  # handled in policy override
    return _rule

# For deceptive oracle, we create a special policy that checks ground truth
class OracleDeceptivePolicy(SoftPriorOnlyPolicyV2):
    def __init__(self, instance_memory, rng, test_objects_ref, target_probe_count=8):
        super().__init__(instance_memory, rng, target_probe_count)
        self._test_objects = test_objects_ref
        self._variant = "oracle_deceptive"

    def _score_candidate(self, features, norm_cost):
        best_score = float('-inf')
        for action in MAIN_CANDIDATE_ACTIONS:
            base_prior = get_goal_soft_prior(features, action)
            risk_penalty = 0.0
            # Check if this object-action is deceptive via ground truth
            # We can't access oid here, so use a heuristic: if any object with
            # these features and this action is a known deceptive in test_objects
            for oid, obj in self._test_objects.items():
                obj_feats = obj.get("visible_features", {})
                if obj_feats == features:
                    gt = get_ground_truth_outcome(obj, action)
                    if gt <= 0.01:
                        risk_penalty = 5.0  # very high penalty for known-deceptive
                    break
            raw_score = base_prior * math.exp(-risk_penalty) - COST_WEIGHT * norm_cost
            score = max(EXPLORATION_FLOOR, raw_score)
            if score > best_score: best_score = score
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
        for action in MAIN_CANDIDATE_ACTIONS:
            norm_cost = view.probe_cost / max(view.initial_budget, 0.001)
            base_prior = get_goal_soft_prior(features, action)
            risk_penalty = 0.0
            gt = get_ground_truth_outcome(self._test_objects.get(object_id, {}), action)
            if gt <= 0.01:
                risk_penalty = 5.0
            raw_score = base_prior * math.exp(-risk_penalty) - COST_WEIGHT * norm_cost
            score = max(EXPLORATION_FLOOR, raw_score)
            if score > best_score:
                best_score = score; best_action = action
        self._probed_oids.add(object_id)
        self._selected_scores.append({
            "oid": object_id, "action": best_action,
            "prior": get_goal_soft_prior(features, best_action) if best_action else EXPLORATION_FLOOR,
            "risk_penalty": 5.0 if best_action and get_ground_truth_outcome(
                self._test_objects.get(object_id, {}), best_action) <= 0.01 else 0.0,
            "final_score": round(best_score, 6),
        })
        return True, best_action


# --- Family_only_rule ---
fcrm_family_only = build_offline_oracle_memory(test_objects_deceptive, key_mode="family_only")

def family_only_rule_factory(memory):
    def _rule(features, action):
        return memory.get_total_risk_penalty(features, action)
    return _rule


# Evaluate the 3 oracle rules
print("  Running oracle_dev_feature_rule...")
result_oracle_dev = evaluate_rule_policy(
    test_objects_deceptive, query_gt_deceptive,
    oracle_dev_feature_rule, RNG_SEED_BASE + 40, "oracle_dev_feature")

print("  Running oracle_deceptive_rule...")
# Run OracleDeceptivePolicy manually
all_eps_deceptive = []
pv_hist_d = set()
for ep in range(N_EPISODES):
    ep_seed = RNG_SEED_BASE + 50 + ep * 100
    ep_rng = random.Random(ep_seed)
    test_oids_local = sorted(test_objects_deceptive.keys())
    ep_positions = MiniMCSimulatorTruth.assign_positions(
        test_oids_local, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, ep_rng)
    im_ep = im_base.clone()
    env_ep = MiniMCEnvironment(test_objects_deceptive, dict(ep_positions), initial_budget=BUDGET)
    policy_ep = OracleDeceptivePolicy(im_ep, ep_rng, test_objects_deceptive, target_probe_count=c15b_probed_n)
    ep_data, pv_hist_d = run_episode_and_extract(
        test_objects_deceptive, query_gt_deceptive, policy_ep, env_ep, ep,
        prior_violation_history=pv_hist_d)
    all_eps_deceptive.append(ep_data)

agg_deceptive = aggregate_episodes(all_eps_deceptive)
def count_eff2(ep_list):
    total = 0
    for ep in ep_list:
        for d in ep.get("probe_details", []):
            if d.get("probe_utility_signal") == "effective": total += 1
    return total
def avoided_pv_count(ep_a_list, ep_var_list):
    a_set = set()
    for ep_a in ep_a_list:
        for d in ep_a["probe_details"]:
            if d.get("is_violation"): a_set.add((d["oid"], d["action"]))
    v_set = set()
    for ep_v in ep_var_list:
        for d in ep_v["probe_details"]:
            if d.get("is_violation"): v_set.add((d["oid"], d["action"]))
    return len(a_set - v_set)

result_oracle_deceptive = {
    "label": "oracle_deceptive_object",
    "agg": agg_deceptive,
    "episodes": all_eps_deceptive,
    "effective_probe_count": count_eff2(all_eps_deceptive),
    "avoided_effective_probe_count": count_eff2(all_episodes_a) - count_eff2(all_eps_deceptive),
    "avoided_prior_violation_count": avoided_pv_count(all_episodes_a, all_eps_deceptive),
    "diagnostic_oracle_rule": True,
}

print("  Running family_only_rule...")
result_family_only = evaluate_rule_policy(
    test_objects_deceptive, query_gt_deceptive,
    family_only_rule_factory(fcrm_family_only), RNG_SEED_BASE + 60, "family_only")

oracle_dev_total_pv = result_oracle_dev["agg"]["total_prior_violations"]
oracle_deceptive_total_pv = result_oracle_deceptive["agg"]["total_prior_violations"]
family_only_total_pv = result_family_only["agg"]["total_prior_violations"]

print(f"  oracle_dev_rule pv={oracle_dev_total_pv}")
print(f"  oracle_deceptive_rule pv={oracle_deceptive_total_pv}")
print(f"  family_only_rule pv={family_only_total_pv}")

part3 = {
    "oracle_dev_feature_rule": result_oracle_dev,
    "oracle_deceptive_object_rule": result_oracle_deceptive,
    "family_only_rule": result_family_only,
    "selected_best_dev_features": [
        {"action": a, "family": f, "dev_feature": d, "abs_delta": s}
        for (a, f), (d, s) in sorted(selected_dev_features.items())
    ],
}

# =============================================================================
# 13. PART 4: Counterproductive-Weight Audit
# =============================================================================
print("\n[13/12] PART 4: Counterproductive-weight audit...")

# Compare B_family_dev_offline_upper probes vs A_no_memory probes
avoided_candidates = []
harmful_avoidances = 0
helpful_avoidances = 0

a_probes_all = {}
for ep_data in all_episodes_a:
    for d in ep_data["probe_details"]:
        a_probes_all[(d["oid"], d["action"])] = d

offline_probes_all = {}
for ep_data in all_episodes_offline:
    for d in ep_data["probe_details"]:
        offline_probes_all[(d["oid"], d["action"])] = d

# Probes that A selected but B_offline didn't (avoided)
avoided_keys = set(a_probes_all.keys()) - set(offline_probes_all.keys())
# Probes that B_offline selected but A didn't (new)
new_keys = set(offline_probes_all.keys()) - set(a_probes_all.keys())

for oid, action in sorted(avoided_keys):
    a_detail = a_probes_all[(oid, action)]
    obj = test_objects_deceptive.get(oid, {})
    features = obj.get("visible_features", {})
    family = detect_type_family(features)
    goal = ACTION_TO_QUERY.get(action, "")
    injected_present = [f for f in INJECTED_DEVIATION_FEATURES if features.get(f, False)]

    # Compute what risk penalty B_offline would have applied
    risk_penalty = fcrm_offline.get_total_risk_penalty(features, action)
    triggered_keys = []
    for dev_feat in INJECTED_DEVIATION_FEATURES:
        if features.get(dev_feat, False):
            w = fcrm_offline.get_risk_weight(goal, action, family, dev_feat)
            if abs(w) > 1e-9:
                triggered_keys.append({"dev_feature": dev_feat, "risk_weight": round(w, 6)})

    is_pv = a_detail.get("is_violation", False)
    is_nv = a_detail.get("is_nonviolation", False)
    was_helpful = is_pv  # avoiding a PV is helpful
    was_harmful = is_nv  # avoiding a nonviolation (useful probe) is harmful

    if was_helpful: helpful_avoidances += 1
    if was_harmful: harmful_avoidances += 1

    avoided_candidates.append({
        "object_id": oid, "action": action, "goal": goal,
        "family": family,
        "injected_deviation_features": injected_present,
        "risk_weight_keys_triggered": triggered_keys,
        "total_risk_penalty": round(risk_penalty, 6),
        "prior_violation": is_pv,
        "prior_nonviolation": is_nv,
        "probe_utility_signal": a_detail.get("probe_utility_signal", "unknown"),
        "avoidance_helped_pv": was_helpful,
        "avoidance_hurt_pv": was_harmful,
    })

# Also check probes B_offline selected that A didn't (newly selected)
new_candidates = []
for oid, action in sorted(new_keys):
    off_detail = offline_probes_all[(oid, action)]
    obj = test_objects_deceptive.get(oid, {})
    features = obj.get("visible_features", {})
    family = detect_type_family(features)
    goal = ACTION_TO_QUERY.get(action, "")
    injected_present = [f for f in INJECTED_DEVIATION_FEATURES if features.get(f, False)]

    risk_penalty = fcrm_offline.get_total_risk_penalty(features, action)
    is_pv = off_detail.get("is_violation", False)
    is_nv = off_detail.get("is_nonviolation", False)

    new_candidates.append({
        "object_id": oid, "action": action, "goal": goal,
        "family": family,
        "injected_deviation_features": injected_present,
        "total_risk_penalty": round(risk_penalty, 6),
        "prior_violation": is_pv,
        "prior_nonviolation": is_nv,
        "probe_utility_signal": off_detail.get("probe_utility_signal", "unknown"),
        "newly_selected": True,
    })

counterproductive_weighting_detected = (
    harmful_avoidances > helpful_avoidances
    or agg_offline["total_prior_violations"] >= agg_a["total_prior_violations"]
)

print(f"  Avoided probes: {len(avoided_keys)} (helpful={helpful_avoidances}, harmful={harmful_avoidances})")
print(f"  New probes: {len(new_keys)}")
print(f"  counterproductive_weighting_detected={counterproductive_weighting_detected}")

part4 = {
    "avoided_candidates": avoided_candidates,
    "new_candidates": new_candidates,
    "aggregate": {
        "avoided_prior_violation_count": helpful_avoidances,
        "avoided_nonviolation_count": harmful_avoidances,
        "avoided_effective_probe_count": sum(1 for c in avoided_candidates
                                            if c["probe_utility_signal"] == "effective"),
        "harmful_avoidance_count": harmful_avoidances,
        "helpful_avoidance_count": helpful_avoidances,
        "new_probe_count": len(new_keys),
        "new_probe_violation_count": sum(1 for c in new_candidates if c["prior_violation"]),
        "counterproductive_weighting_detected": counterproductive_weighting_detected,
    },
}

# =============================================================================
# 14. PART 5: Boolean Flags
# =============================================================================
print("\n[14/12] PART 5: Boolean flags...")

# Check deceptive family preservation
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

oracle_dev_rule_beats_A = oracle_dev_total_pv < agg_a["total_prior_violations"]
oracle_dev_rule_beats_family_only = oracle_dev_total_pv < family_only_total_pv
oracle_deceptive_rule_beats_A = oracle_deceptive_total_pv < agg_a["total_prior_violations"]
family_only_conservative_win_detected = family_only_total_pv <= agg_offline["total_prior_violations"]

hard_exclusion_used = False

# Determine implementation status
# Priority: real bugs > substantive findings > methodology notes
if risk_sign_error_detected or penalty_direction_error_detected:
    implementation_status = "fail"
    failure_reason = "implementation_bugs_detected"
elif deviation_features_informative and oracle_dev_rule_beats_A and oracle_dev_rule_beats_family_only:
    implementation_status = "pass"
    failure_reason = "none"
elif deviation_features_informative and oracle_dev_rule_beats_A and not oracle_dev_rule_beats_family_only:
    # Dev features are individually informative and can beat A, but family-only
    # dominates. The sum-over-features penalty formula diffuses the signal.
    implementation_status = "partial"
    failure_reason = "deviation_features_informative_but_family_only_dominates"
elif not deviation_features_informative:
    implementation_status = "partial"
    failure_reason = "deviation_features_not_informative"
elif not oracle_dev_rule_beats_A:
    implementation_status = "partial"
    failure_reason = "oracle_dev_rule_cannot_beat_baseline"
else:
    implementation_status = "partial"
    failure_reason = "mixed_results"

boolean_flag_details = {
    "risk_sign_error_detected": {
        "value": risk_sign_error_detected,
        "rule": "true if any key has p_feat|v > p_feat|nv but risk<=0, or p_feat|v < p_feat|nv but risk>0",
        "supporting_values": {"risk_sign_errors": risk_sign_errors, "total_keys": len(risk_audit_rows)},
    },
    "penalty_direction_error_detected": {
        "value": penalty_direction_error_detected,
        "rule": "true if any positive risk penalty increases final selection priority",
        "supporting_values": {"penalty_direction_errors": penalty_direction_errors, "total_checked": len(penalty_check_rows)},
    },
    "support_count_inconsistency_detected": {
        "value": support_count_inconsistency_detected,
        "rule": "true if online/offline support counts differ under intended event unit",
        "supporting_values": {"online_total": online_total_events, "offline_total": offline_total_events, "ratio": round(online_total_events/offline_total_events, 2) if offline_total_events > 0 else 0},
    },
    "deviation_features_informative": {
        "value": deviation_features_informative,
        "rule": "true if at least one dev feature has |risk_rate_delta| > 0.1 with above-threshold support",
        "supporting_values": {"informative_keys_count": len(informative_keys), "total_keys": len(info_rows)},
    },
    "oracle_dev_rule_beats_A": {
        "value": oracle_dev_rule_beats_A,
        "rule": "oracle_dev_rule_total_pv < A_total_pv",
        "supporting_values": {"oracle_dev_total_pv": oracle_dev_total_pv, "A_total_pv": agg_a["total_prior_violations"]},
    },
    "oracle_dev_rule_beats_family_only": {
        "value": oracle_dev_rule_beats_family_only,
        "rule": "oracle_dev_rule_total_pv < family_only_total_pv",
        "supporting_values": {"oracle_dev_total_pv": oracle_dev_total_pv, "family_only_total_pv": family_only_total_pv},
    },
    "oracle_deceptive_rule_beats_A": {
        "value": oracle_deceptive_rule_beats_A,
        "rule": "oracle_deceptive_total_pv < A_total_pv",
        "supporting_values": {"oracle_deceptive_total_pv": oracle_deceptive_total_pv, "A_total_pv": agg_a["total_prior_violations"]},
    },
    "family_only_conservative_win_detected": {
        "value": family_only_conservative_win_detected,
        "rule": "family_only_total_pv <= B_family_dev_offline_upper_total_pv",
        "supporting_values": {"family_only_total_pv": family_only_total_pv, "offline_total_pv": agg_offline["total_prior_violations"]},
    },
    "counterproductive_weighting_detected": {
        "value": counterproductive_weighting_detected,
        "rule": "harmful_avoidances > helpful_avoidances OR B_offline doesn't reduce PV vs A",
        "supporting_values": {"harmful_avoidances": harmful_avoidances, "helpful_avoidances": helpful_avoidances, "offline_pv": agg_offline["total_prior_violations"], "A_pv": agg_a["total_prior_violations"]},
    },
    "hard_exclusion_used": {
        "value": hard_exclusion_used,
        "rule": "hard_exclusion is always false for all variants",
        "supporting_values": {},
    },
    "deceptive_family_preservation_valid": {
        "value": deceptive_family_preservation_valid,
        "rule": "all deceptive objects preserve expected family identity",
        "supporting_values": {"total_deceptive": len(DECEPTIVE_OIDS), "preserved": sum(1 for c in deceptive_family_preservation_checks if c["preserved"])},
    },
    "implementation_status": {
        "value": implementation_status,
        "rule": "pass if no implementation bugs AND deviation features informative AND oracle beats A",
        "supporting_values": {
            "risk_sign_error": risk_sign_error_detected,
            "penalty_direction_error": penalty_direction_error_detected,
            "deviation_features_informative": deviation_features_informative,
            "oracle_dev_beats_A": oracle_dev_rule_beats_A,
        },
    },
    "failure_reason": {
        "value": failure_reason,
        "rule": "derived from implementation audit and informativeness results",
        "supporting_values": {},
    },
}

# =============================================================================
# 15. Output
# =============================================================================
elapsed = time.time() - t0
print(f"\n  elapsed={elapsed:.1f}s  writing outputs...")

output = {
    "block_id": "1J22",
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
    "A_no_memory": {"agg": agg_a, "episodes": all_episodes_a},
    "B_family_dev_offline_upper": {
        "agg": agg_offline,
        "episodes": all_episodes_offline,
    },
    "part1_implementation_sanity": part1,
    "part2_informativeness": part2,
    "part3_oracle_rules": part3,
    "part4_counterproductive_weight": part4,
    "part5_boolean_flags": boolean_flag_details,
    "deceptive_family_preservation_checks": deceptive_family_preservation_checks,
    "elapsed_seconds": round(elapsed, 1),
}

out_json_path = os.path.join(CURRENT_DIR, "runs",
    "block1j22_deviation_feature_informativeness_audit_seed101.json")
os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
with open(out_json_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"  JSON -> {out_json_path}")

# =============================================================================
# 16. Markdown Protocol
# =============================================================================
md_lines = []
md_lines.append("# Block 1J22 -- Deviation Feature Informativeness and Risk-Weight Audit")
md_lines.append("")
md_lines.append("## 1. Objective")
md_lines.append("")
md_lines.append("Audit why 1J21 offline family-conditioned deviation memory failed "
               "even with oracle evidence. Check implementation correctness, "
               "feature informativeness, and counterproductive weighting.")
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
md_lines.append(f"| A_no_memory | {agg_a['mean_macro_bal']:.4f} |")
md_lines.append(f"| B_family_dev_offline_upper | {agg_offline['mean_macro_bal']:.4f} |")
md_lines.append("")

# Part 1
md_lines.append("## 4. Part 1: Implementation Sanity Audit")
md_lines.append("")

md_lines.append("### 1a. Risk-Weight Sign Check")
md_lines.append("")
md_lines.append(f"Risk sign errors: {risk_sign_errors} -> risk_sign_error_detected={risk_sign_error_detected}")
md_lines.append("")
md_lines.append("| Goal | Action | Family | Dev Feat | V_With | NV_With | V_Without | NV_Without | P(F|V) | P(F|NV) | Raw LR | Clipped | Shrink | Final W | Sign OK | Above Sup |")
md_lines.append("|------|--------|--------|-----------|--------|---------|-----------|------------|-------|---------|--------|---------|--------|---------|---------|-----------|")
for r in sorted(risk_audit_rows, key=lambda x: abs(x["final_risk_weight"]), reverse=True)[:30]:
    k = r["key"]
    md_lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | "
                   f"{r['violation_with']:.0f} | {r['nonviolation_with']:.0f} | "
                   f"{r['violation_without']:.0f} | {r['nonviolation_without']:.0f} | "
                   f"{r['p_feature_given_violation']:.4f} | {r['p_feature_given_nonviolation']:.4f} | "
                   f"{r['raw_log_ratio']:.4f} | {r['clipped_risk_weight']:.4f} | "
                   f"{r['shrinkage_factor']:.4f} | {r['final_risk_weight']:.4f} | "
                   f"{r['sign_correct']} | {r['above_support_threshold']} |")
md_lines.append("")

md_lines.append("### 1b. Penalty-Direction Check")
md_lines.append("")
md_lines.append(f"Penalty direction errors: {penalty_direction_errors} -> penalty_direction_error_detected={penalty_direction_error_detected}")
md_lines.append("")

md_lines.append("### 1c. Support-Definition Check")
md_lines.append("")
md_lines.append(f"- Online support event unit: episode_repeated_probe_events")
md_lines.append(f"- Offline support event unit: unique_object_action_candidates")
md_lines.append(f"- Online total eligible events: {online_total_events}")
md_lines.append(f"- Offline total eligible events: {offline_total_events}")
md_lines.append(f"- Online keys above support: {online_keys_above}")
md_lines.append(f"- Offline keys above support: {offline_keys_above}")
md_lines.append(f"- support_count_inconsistency_detected={support_count_inconsistency_detected}")
md_lines.append("")

# Part 2
md_lines.append("## 5. Part 2: Deviation Feature Informativeness Table")
md_lines.append("")
md_lines.append(f"deviation_features_informative={deviation_features_informative}")
md_lines.append(f"Informative keys (|delta|>0.1 & above support): {len(informative_keys)}")
md_lines.append("")

md_lines.append("### Top 20 Keys by |Risk Rate Delta|")
md_lines.append("")
md_lines.append("| Goal | Action | Family | Dev Feat | N_With | N_Without | V_With | NV_With | V_Without | NV_Without | Rate_With | Rate_Without | Delta | LogOdds | MI | Above Sup | Final W |")
md_lines.append("|------|--------|--------|-----------|--------|-----------|--------|---------|-----------|------------|-----------|--------------|-------|---------|----|-----------|---------|")
for r in sorted(info_rows, key=lambda x: abs(x["risk_rate_delta"]), reverse=True)[:20]:
    k = r["key"]
    md_lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | "
                   f"{r['n_with_feature']} | {r['n_without_feature']} | "
                   f"{r['violations_with_feature']} | {r['nonviolations_with_feature']} | "
                   f"{r['violations_without_feature']} | {r['nonviolations_without_feature']} | "
                   f"{r['violation_rate_with_feature']:.4f} | {r['violation_rate_without_feature']:.4f} | "
                   f"{r['risk_rate_delta']:.4f} | {r['log_odds']:.4f} | {r['mutual_information']:.4f} | "
                   f"{r['above_support_threshold']} | {r['final_risk_weight']:.4f} |")
md_lines.append("")

# Part 3
md_lines.append("## 6. Part 3: Best Possible Deviation-Rule Diagnostic")
md_lines.append("")
md_lines.append("| Rule | Total PV | Deceptive PV | Normal PV | Mean BAcc | Delta vs A | Eff Probes | Avoided Eff | Avoided PV |")
md_lines.append("|------|----------|-------------|-----------|-----------|------------|------------|-------------|------------|")
for result in [result_oracle_dev, result_oracle_deceptive, result_family_only]:
    agg = result["agg"]
    delta = round(agg["mean_macro_bal"] - agg_a["mean_macro_bal"], 4)
    md_lines.append(f"| {result['label']} | {agg['total_prior_violations']} | "
                   f"{agg['total_deceptive_prior_violations']} | "
                   f"{agg['total_normal_prior_violations']} | "
                   f"{agg['mean_macro_bal']:.4f} | {delta} | "
                   f"{result.get('effective_probe_count', 0)} | "
                   f"{result.get('avoided_effective_probe_count', 0)} | "
                   f"{result.get('avoided_prior_violation_count', 0)} |")
md_lines.append("")

md_lines.append(f"oracle_dev_rule_beats_A={oracle_dev_rule_beats_A}")
md_lines.append(f"oracle_dev_rule_beats_family_only={oracle_dev_rule_beats_family_only}")
md_lines.append(f"oracle_deceptive_rule_beats_A={oracle_deceptive_rule_beats_A}")
md_lines.append("")

# Part 4
md_lines.append("## 7. Part 4: Counterproductive-Weight Audit")
md_lines.append("")
md_lines.append(f"Avoided probes: {len(avoided_keys)} (helpful={helpful_avoidances}, harmful={harmful_avoidances})")
md_lines.append(f"New probes selected: {len(new_keys)}")
md_lines.append(f"counterproductive_weighting_detected={counterproductive_weighting_detected}")
md_lines.append("")

md_lines.append("### Avoided Candidates (top 15 by risk penalty)")
md_lines.append("")
md_lines.append("| OID | Action | Family | Dev Feats | Risk Penalty | PV | NV | Utility | Helped | Hurt |")
md_lines.append("|-----|--------|--------|-----------|-------------|----|----|---------|--------|------|")
for c in sorted(avoided_candidates, key=lambda x: x["total_risk_penalty"], reverse=True)[:15]:
    md_lines.append(f"| {c['object_id']} | {c['action']} | {c['family']} | "
                   f"{c['injected_deviation_features']} | {c['total_risk_penalty']:.4f} | "
                   f"{c['prior_violation']} | {c['prior_nonviolation']} | "
                   f"{c['probe_utility_signal']} | {c['avoidance_helped_pv']} | "
                   f"{c['avoidance_hurt_pv']} |")
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
md_lines.append(f"block_id=1J22")
md_lines.append(f"risk_sign_error_detected={risk_sign_error_detected}")
md_lines.append(f"penalty_direction_error_detected={penalty_direction_error_detected}")
md_lines.append(f"support_count_inconsistency_detected={support_count_inconsistency_detected}")
md_lines.append(f"deviation_features_informative={deviation_features_informative}")
md_lines.append(f"oracle_dev_rule_total_pv={oracle_dev_total_pv}")
md_lines.append(f"oracle_deceptive_rule_total_pv={oracle_deceptive_total_pv}")
md_lines.append(f"family_only_rule_total_pv={family_only_total_pv}")
md_lines.append(f"A_total_pv={agg_a['total_prior_violations']}")
md_lines.append(f"B_family_dev_offline_upper_total_pv={agg_offline['total_prior_violations']}")
md_lines.append(f"counterproductive_weighting_detected={counterproductive_weighting_detected}")
md_lines.append(f"family_only_conservative_win_detected={family_only_conservative_win_detected}")
md_lines.append(f"hard_exclusion_used={hard_exclusion_used}")
md_lines.append(f"deceptive_family_preservation_valid={deceptive_family_preservation_valid}")
md_lines.append(f"implementation_status={implementation_status}")
md_lines.append(f"failure_reason={failure_reason}")
md_lines.append(f"elapsed={elapsed:.1f}s")
md_lines.append("```")

out_md_path = os.path.join(CURRENT_DIR, "protocols",
    "block1j22_deviation_feature_informativeness_audit_seed101.md")
os.makedirs(os.path.dirname(out_md_path), exist_ok=True)
with open(out_md_path, "w") as f:
    f.write("\n".join(md_lines))
print(f"  MD -> {out_md_path}")

print("\n" + "=" * 70)
print("Block 1J22 complete.")
print("=" * 70)
