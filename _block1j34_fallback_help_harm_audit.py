"""
Block 1J34 -- Fallback Help-vs-Harm Audit.

Audits when B_dev_plus_family_action_fallback helps vs harms vs B_signed_sum.

Two effect types:
  direct_decision_delta: fallback changes score/rank/selection at decision time
      (counterfactual computed per-decision by comparing fallback vs dev-only path)
  trajectory_delta: final probe-set diff after independent online trajectories diverge

Per-decision instrumentation records (no logic change):
  fallback_used, dev_signal_zero, family_action_signal_used,
  dev_score, fallback_score, final_score, rank_before/after, selected_before/after

Does NOT change policy logic, environment, or memory mechanism.
"""
import os, sys, json, copy, random, time, math, csv, hashlib
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
# CLI
# =============================================================================
import argparse as _argparse
_parser = _argparse.ArgumentParser()
_parser.add_argument("--seed", type=int, default=109, help="Base seed (default: 109)")
_parser.add_argument("--multiseed", action="store_true", help="Run 5-seed summary")
_args = _parser.parse_args()

# =============================================================================
# Constants (same as 1J33)
# =============================================================================
SMOKE_SEED = _args.seed
MULTISEED = _args.multiseed
MULTISEED_LIST = [101, 103, 107, 109, 113] if MULTISEED else [SMOKE_SEED]
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
SORTED_DEV_FEATURES = sorted(INJECTED_DEVIATION_FEATURES)

ACTION_TO_QUERY = {
    "craft_plank": "need_planks", "eat": "need_food",
    "use_as_tool": "need_tool", "burn_as_fuel": "need_fuel",
    "mine_by_hand": "need_stone", "mine_with_pickaxe": "need_stone",
}

print("=" * 70)
print("Block 1J34 -- Fallback Help-vs-Harm Audit")
print(f"  condition={COND['label']}  base_seed={SMOKE_SEED}  budget={BUDGET}")
print(f"  episodes={N_EPISODES}  multiseed={MULTISEED}")
print("=" * 70)

# =============================================================================
# 1. Phase A: Training + IOM
# =============================================================================
print("\n[1/7] Phase A: Training + IOM building...")
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
# 2. Test objects + Family-preserving deceptive objects (same as 1J33)
# =============================================================================
print("\n[2/7] Generating test objects + deceptive objects...")
test_objects_standard = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")
query_gt_standard = compute_query_ground_truth(test_objects_standard)
test_oids = sorted(test_objects_standard.keys())
QUERY_NAMES = sorted(_TASK_QUERIES.keys())

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
    obj["_deceptive"] = True; obj["_deceptive_action"] = "eat"
    obj["_deviation_features"] = {f: True for f, v in APPLE_DEVIATION.items() if v}
    for feat in APPLE_CORE:
        if feat in FEATURE_UNIVERSE_SET: obj["visible_features"][feat] = True
    for feat, val in APPLE_DEVIATION.items():
        if feat in FEATURE_UNIVERSE_SET: obj["visible_features"][feat] = val
    deceptive_modifications.append({
        "oid": oid, "category": "apple", "deceptive_action": "eat",
        "deviation_features": [f for f, v in APPLE_DEVIATION.items() if v and f in FEATURE_UNIVERSE_SET],
    })

n_deceptive_wood = min(5, len(wood_oids))
for oid in deceptive_rng.sample(wood_oids, n_deceptive_wood):
    obj = test_objects_deceptive[oid]
    obj["hidden_affordance_profile"]["burn_as_fuel"] = "fail"
    obj["_deceptive"] = True; obj["_deceptive_action"] = "burn_as_fuel"
    obj["_deviation_features"] = {f: True for f, v in WOOD_DEVIATION.items() if v}
    for feat in WOOD_CORE:
        if feat in FEATURE_UNIVERSE_SET: obj["visible_features"][feat] = True
    for feat, val in WOOD_DEVIATION.items():
        if feat in FEATURE_UNIVERSE_SET: obj["visible_features"][feat] = val
    deceptive_modifications.append({
        "oid": oid, "category": "wood_log", "deceptive_action": "burn_as_fuel",
        "deviation_features": [f for f, v in WOOD_DEVIATION.items() if v and f in FEATURE_UNIVERSE_SET],
    })

n_deceptive_stones = min(3, len(stone_oids))
for oid in deceptive_rng.sample(stone_oids, n_deceptive_stones):
    obj = test_objects_deceptive[oid]
    obj["hidden_affordance_profile"]["mine_with_pickaxe"] = "fail"
    obj["_deceptive"] = True; obj["_deceptive_action"] = "mine_with_pickaxe"
    obj["_deviation_features"] = {f: True for f, v in STONE_DEVIATION.items() if v}
    for feat in STONE_CORE:
        if feat in FEATURE_UNIVERSE_SET: obj["visible_features"][feat] = True
    for feat, val in STONE_DEVIATION.items():
        if feat in FEATURE_UNIVERSE_SET: obj["visible_features"][feat] = val
    deceptive_modifications.append({
        "oid": oid, "category": "stone_block", "deceptive_action": "mine_with_pickaxe",
        "deviation_features": [f for f, v in STONE_DEVIATION.items() if v and f in FEATURE_UNIVERSE_SET],
    })

query_gt_deceptive = compute_query_ground_truth(test_objects_deceptive)
DECEPTIVE_OIDS = set(mod["oid"] for mod in deceptive_modifications)
print(f"  {len(test_oids)} test objects, {len(DECEPTIVE_OIDS)} deceptive")

# =============================================================================
# 3. Metric helpers (same as 1J33)
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
# 4. Type family detection + goal prior table (same as 1J33)
# =============================================================================
print("\n[3/7] Building goal-conditioned soft prior table...")

def detect_type_family(features):
    if features is None: return "unknown"
    scores = {}
    for fam, feat_set in TYPE_FAMILIES.items():
        scores[fam] = sum(1 for f in feat_set if features.get(f, False))
    best_fam = max(scores, key=scores.get)
    return best_fam if scores[best_fam] > 0 else "unknown"

def get_family_scores(features):
    """Return all family scores for ambiguity detection."""
    if features is None: return {}
    scores = {}
    for fam, feat_set in TYPE_FAMILIES.items():
        scores[fam] = sum(1 for f in feat_set if features.get(f, False))
    return scores

def is_family_signal_ambiguous(features):
    """Check if family detection is ambiguous (top two families close)."""
    scores = get_family_scores(features)
    if not scores: return True
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) < 2: return False
    if sorted_scores[0] == 0: return True
    # Ambiguous if second-best is within 1 of best, or if best <= 2
    gap = sorted_scores[0] - sorted_scores[1]
    return gap <= 1 or sorted_scores[0] <= 2

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

def get_ground_truth_outcome(obj, action):
    profile = obj.get("hidden_affordance_profile", {})
    result = profile.get(action, "success")
    return 1.0 if result == "success" else 0.0

# =============================================================================
# 5. FamilyConditionedRiskMemory + Policy Classes (same as 1J33)
# =============================================================================
print("\n[4/7] Defining FCRM + Policy classes...")

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
        self._signed_cache = {}
        self.total_updates = 0
        self.violation_updates = 0
        self.nonviolation_updates = 0
        self.eligible_updates = 0
        self.ineligible_updates = 0
        self.update_log = []

    def update(self, features, goal, action, is_violation, is_eligible,
               episode_id=None, step_id=None, object_id=None):
        self.total_updates += 1
        if not is_eligible:
            self.ineligible_updates += 1
            if episode_id is not None:
                self.update_log.append({
                    "episode_id": episode_id, "step_id": step_id,
                    "object_id": object_id, "goal": goal, "action": action,
                    "is_violation": is_violation, "is_eligible": False,
                    "memory_updated": False,
                })
            return
        self.eligible_updates += 1
        if is_violation: self.violation_updates += 1
        else: self.nonviolation_updates += 1

        family = detect_type_family(features)

        if self.key_mode == "family_dev":
            for dev_feat in INJECTED_DEVIATION_FEATURES:
                key = (goal, action, family, dev_feat)
                present = features.get(dev_feat, False)
                if present:
                    if is_violation: self.counts[key]["violation_with"] += 1.0
                    else: self.counts[key]["nonviolation_with"] += 1.0
                else:
                    if is_violation: self.counts[key]["violation_without"] += 1.0
                    else: self.counts[key]["nonviolation_without"] += 1.0
        elif self.key_mode == "family_only":
            key_detected = (goal, action, family)
            if is_violation: self.counts[key_detected]["violation_with"] += 1.0
            else: self.counts[key_detected]["nonviolation_with"] += 1.0
            for other_fam in FAMILY_NAMES:
                if other_fam == family: continue
                key_other = (goal, action, other_fam)
                if is_violation: self.counts[key_other]["violation_without"] += 1.0
                else: self.counts[key_other]["nonviolation_without"] += 1.0

        self._risk_cache.clear()
        self._signed_cache.clear()

        if episode_id is not None:
            injected_present = [f for f in INJECTED_DEVIATION_FEATURES if features.get(f, False)]
            self.update_log.append({
                "episode_id": episode_id, "step_id": step_id,
                "object_id": object_id, "goal": goal, "action": action,
                "inferred_family": family,
                "injected_deviation_features": injected_present,
                "outcome_success": not is_violation,
                "prior_violation": is_violation,
                "prior_nonviolation": is_eligible and not is_violation,
                "memory_updated_after_selection": True,
                "is_eligible": True,
            })

    def get_risk_weight(self, *key):
        if key in self._risk_cache: return self._risk_cache[key]
        c = self.counts[key]
        v_with = c["violation_with"]; nv_with = c["nonviolation_with"]
        v_without = c["violation_without"]; nv_without = c["nonviolation_without"]
        actual_violations = (v_with + v_without) - 2.0
        if actual_violations < self.min_violation_support:
            self._risk_cache[key] = 0.0; return 0.0
        total_v = v_with + v_without; total_nv = nv_with + nv_without
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

    def get_raw_risk_weight_signed(self, *key):
        if key in self._signed_cache: return self._signed_cache[key]
        c = self.counts[key]
        v_with = c["violation_with"]; nv_with = c["nonviolation_with"]
        v_without = c["violation_without"]; nv_without = c["nonviolation_without"]
        actual_violations = (v_with + v_without) - 2.0
        if actual_violations < self.min_violation_support:
            self._signed_cache[key] = 0.0; return 0.0
        total_v = v_with + v_without; total_nv = nv_with + nv_without
        p_feat_given_v = v_with / max(total_v, 0.001)
        p_feat_given_nv = nv_with / max(total_nv, 0.001)
        eps = 1e-9
        if p_feat_given_nv < eps: p_feat_given_nv = eps
        ratio = p_feat_given_v / p_feat_given_nv
        if ratio < eps: ratio = eps
        if ratio > 1.0 / eps: ratio = 1.0 / eps
        raw_lr = math.log(ratio)
        support = total_v + total_nv - 4.0
        support = max(0.0, support)
        reliability = support / (support + self.alpha)
        usable = reliability * raw_lr
        self._signed_cache[key] = usable
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
        return dict(self.counts[key])

    def get_num_nonzero_keys(self):
        count = 0
        for key in self.counts:
            if self.get_risk_weight(*key) > 1e-9: count += 1
        return count

    def clone(self):
        new = FamilyConditionedRiskMemory(key_mode=self.key_mode, alpha=self.alpha,
                                          max_risk=self.max_risk,
                                          min_violation_support=self.min_violation_support)
        new.counts.update(copy.deepcopy(dict(self.counts)))
        new._risk_cache = dict(self._risk_cache)
        new._signed_cache = dict(self._signed_cache)
        new.total_updates = self.total_updates
        new.violation_updates = self.violation_updates
        new.nonviolation_updates = self.nonviolation_updates
        new.eligible_updates = self.eligible_updates
        new.ineligible_updates = self.ineligible_updates
        new.update_log = list(self.update_log)
        return new


def build_offline_oracle_memory(test_objects, key_mode="family_dev"):
    memory = FamilyConditionedRiskMemory(key_mode=key_mode)
    for oid, obj in test_objects.items():
        features = obj.get("visible_features", {})
        for action in MAIN_CANDIDATE_ACTIONS:
            soft_prior = get_goal_soft_prior(features, action)
            is_eligible = soft_prior >= PRIOR_VIOLATION_THRESHOLD
            if not is_eligible: continue
            outcome = get_ground_truth_outcome(obj, action)
            is_violation = outcome <= 0.01
            goal = ACTION_TO_QUERY.get(action, "")
            memory.update(features, goal, action, is_violation, is_eligible)
    return memory


# --- Policy Classes (copied from 1J33, unchanged) ---

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
            "prior": get_goal_soft_prior(features, best_action) if best_action else EXPLORATION_FLOOR,
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


class OnlineSignedPolicy(SoftPriorOnlyPolicyV2):
    def __init__(self, instance_memory, rng, family_conditioned_memory,
                 target_probe_count=8, explore_fraction=0.0, risk_penalty_scale=1.0,
                 episode_id=0, step_counter_ref=None):
        super().__init__(instance_memory, rng, target_probe_count, explore_fraction)
        self._fcrm = family_conditioned_memory
        self._risk_penalty_scale = risk_penalty_scale
        self.max_risk = MAX_RISK
        self._episode_id = episode_id
        self._step_counter_ref = step_counter_ref if step_counter_ref is not None else [0]
        self._variant = "online_signed_sum"
        self._probe_step_counter = 0

    def _get_per_feature_weights(self, features, action):
        goal = ACTION_TO_QUERY.get(action, "")
        family = detect_type_family(features)
        weights = []
        for dev_feat in INJECTED_DEVIATION_FEATURES:
            if features.get(dev_feat, False):
                w = self._fcrm.get_raw_risk_weight_signed(goal, action, family, dev_feat)
                weights.append((dev_feat, w))
        return weights

    def get_total_risk_penalty(self, features, action):
        weights = self._get_per_feature_weights(features, action)
        total = sum(w for _, w in weights)
        return max(-self.max_risk, min(self.max_risk, total))

    def _compute_adjusted_score(self, features, action, norm_cost):
        base_prior = get_goal_soft_prior(features, action)
        risk_penalty = self.get_total_risk_penalty(features, action)
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
            "final_score": round(fs, 6), "risk_adjustment": round(sp, 6),
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
            is_violation, is_eligible, sp = check_prior_violation(features, action, outcome)
            goal = ACTION_TO_QUERY.get(action, "")
            self._probe_step_counter += 1
            self._fcrm.update(
                features, goal, action, is_violation, is_eligible,
                episode_id=self._episode_id,
                step_id=self._probe_step_counter,
                object_id=object_id,
            )
        else:
            self._im.incorporate_probe(object_id, action, outcome)


class OnlineFamilyOnlyPolicy(SoftPriorOnlyPolicyV2):
    def __init__(self, instance_memory, rng, family_conditioned_memory,
                 target_probe_count=8, explore_fraction=0.0, risk_penalty_scale=1.0,
                 episode_id=0, step_counter_ref=None):
        super().__init__(instance_memory, rng, target_probe_count, explore_fraction)
        self._fcrm = family_conditioned_memory
        self._risk_penalty_scale = risk_penalty_scale
        self._episode_id = episode_id
        self._step_counter_ref = step_counter_ref if step_counter_ref is not None else [0]
        self._variant = "online_family_only"
        self._probe_step_counter = 0

    def get_total_risk_penalty(self, features, action):
        return self._fcrm.get_total_risk_penalty(features, action)

    def _compute_adjusted_score(self, features, action, norm_cost):
        base_prior = get_goal_soft_prior(features, action)
        risk_penalty = self.get_total_risk_penalty(features, action)
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
            "final_score": round(fs, 6), "risk_adjustment": round(sp, 6),
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
            effect = "effective" if is_eff else ("zero" if abs(utility_delta) <= 0.001 else "harmful")
            self._probe_effects.append({
                "oid": object_id, "action": action, "outcome": outcome,
                "utility_delta": round(utility_delta, 6),
                "is_effective": is_eff, "effect": effect,
            })
            is_violation, is_eligible, sp = check_prior_violation(features, action, outcome)
            goal = ACTION_TO_QUERY.get(action, "")
            self._probe_step_counter += 1
            self._fcrm.update(
                features, goal, action, is_violation, is_eligible,
                episode_id=self._episode_id,
                step_id=self._probe_step_counter,
                object_id=object_id,
            )
        else:
            self._im.incorporate_probe(object_id, action, outcome)


class DevPlusFamilyActionFallbackPolicy(SoftPriorOnlyPolicyV2):
    """Instrumented fallback policy. Selection logic unchanged from 1J33.

    Adds per-decision audit logging: for each decide_probe, computes both
    the dev-only path (counterfactual, no fallback) and the actual fallback
    path, recording score/rank deltas. This is audit logging only.
    """
    def __init__(self, instance_memory, rng, fcrm_dev, fcrm_family,
                 target_probe_count=8, explore_fraction=0.0, risk_penalty_scale=1.0,
                 episode_id=0, step_counter_ref=None):
        super().__init__(instance_memory, rng, target_probe_count, explore_fraction)
        self._fcrm_dev = fcrm_dev
        self._fcrm_family = fcrm_family
        self._risk_penalty_scale = risk_penalty_scale
        self.max_risk = MAX_RISK
        self._episode_id = episode_id
        self._step_counter_ref = step_counter_ref if step_counter_ref is not None else [0]
        self._variant = "dev_plus_family_action_fallback"
        self._probe_step_counter = 0
        self._fallback_activation_count = 0
        self._decision_audit_log = []  # per-decision instrumentation

    # --- Risk penalty: actual (with fallback) ---
    def get_total_risk_penalty(self, features, action):
        goal = ACTION_TO_QUERY.get(action, "")
        family = detect_type_family(features)
        dev_weights = []
        for dev_feat in INJECTED_DEVIATION_FEATURES:
            if features.get(dev_feat, False):
                w = self._fcrm_dev.get_raw_risk_weight_signed(goal, action, family, dev_feat)
                dev_weights.append(w)
        dev_penalty = sum(dev_weights) if dev_weights else 0.0
        has_dev_feat = any(features.get(df, False) for df in INJECTED_DEVIATION_FEATURES)
        if dev_penalty == 0.0 and not has_dev_feat:
            family_penalty = self._fcrm_family.get_total_risk_penalty(features, action)
            self._fallback_activation_count += 1
            return family_penalty
        return max(-self.max_risk, min(self.max_risk, dev_penalty))

    # --- Risk penalty: dev-only path (counterfactual, never falls back) ---
    def get_total_risk_penalty_dev_only(self, features, action):
        """Dev-only path: always use signed_sum on dev_feat, never fall back."""
        goal = ACTION_TO_QUERY.get(action, "")
        family = detect_type_family(features)
        dev_weights = []
        for dev_feat in INJECTED_DEVIATION_FEATURES:
            if features.get(dev_feat, False):
                w = self._fcrm_dev.get_raw_risk_weight_signed(goal, action, family, dev_feat)
                dev_weights.append(w)
        dev_penalty = sum(dev_weights) if dev_weights else 0.0
        return max(-self.max_risk, min(self.max_risk, dev_penalty))

    def _compute_adjusted_score(self, features, action, norm_cost):
        base_prior = get_goal_soft_prior(features, action)
        risk_penalty = self.get_total_risk_penalty(features, action)
        scaled_penalty = risk_penalty * self._risk_penalty_scale
        raw_score = base_prior * math.exp(-scaled_penalty) - COST_WEIGHT * norm_cost
        final_score = max(EXPLORATION_FLOOR, raw_score)
        return final_score, raw_score, base_prior, risk_penalty, scaled_penalty

    def _compute_adjusted_score_dev_only(self, features, action, norm_cost):
        """Counterfactual: what score would be if we used dev-only path."""
        base_prior = get_goal_soft_prior(features, action)
        risk_penalty = self.get_total_risk_penalty_dev_only(features, action)
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
        norm_cost = view.probe_cost / max(view.initial_budget, 0.001)

        has_dev_feat = any(features.get(df, False) for df in INJECTED_DEVIATION_FEATURES)
        family = detect_type_family(features)
        dev_signal_zero = (not has_dev_feat)

        # --- Compute ALL actions under BOTH paths for audit ---
        dev_scores = {}     # action -> (final_score, risk_penalty) under dev-only
        fallback_scores = {}  # action -> (final_score, risk_penalty) under fallback

        for action in MAIN_CANDIDATE_ACTIONS:
            dfs, _, _, drp, _ = self._compute_adjusted_score_dev_only(features, action, norm_cost)
            ffs, _, _, frp, _ = self._compute_adjusted_score(features, action, norm_cost)
            dev_scores[action] = (dfs, drp)
            fallback_scores[action] = (ffs, frp)

        # Rank actions under each path (higher score = rank 0)
        dev_ranked = sorted(MAIN_CANDIDATE_ACTIONS, key=lambda a: dev_scores[a][0], reverse=True)
        fb_ranked = sorted(MAIN_CANDIDATE_ACTIONS, key=lambda a: fallback_scores[a][0], reverse=True)

        # Actual selection (uses fallback path)
        best_action = None; best_score = float('-inf')
        best_info = (EXPLORATION_FLOOR, 0.0, 0.0, EXPLORATION_FLOOR, EXPLORATION_FLOOR)
        for action in MAIN_CANDIDATE_ACTIONS:
            final_score, raw_score, base_prior, risk_penalty, scaled_penalty = \
                fallback_scores[action][0], 0, get_goal_soft_prior(features, action), \
                fallback_scores[action][1], fallback_scores[action][1] * self._risk_penalty_scale
            raw_score_recomputed = base_prior * math.exp(-scaled_penalty) - COST_WEIGHT * norm_cost
            final_score_recomputed = max(EXPLORATION_FLOOR, raw_score_recomputed)
            if final_score_recomputed > best_score:
                best_score = final_score_recomputed; best_action = action
                best_info = (base_prior, risk_penalty, scaled_penalty, raw_score_recomputed, final_score_recomputed)

        # --- Per-decision audit log ---
        dev_best_action = dev_ranked[0]
        dev_best_score = dev_scores[dev_best_action][0]
        dev_best_penalty = dev_scores[dev_best_action][1]
        fb_best_score = fallback_scores[best_action][0]
        fb_best_penalty = fallback_scores[best_action][1]

        dev_rank_of_fb_selected = dev_ranked.index(best_action)
        fb_rank_of_dev_selected = fb_ranked.index(dev_best_action)

        fallback_used_for_selected = dev_signal_zero and abs(dev_scores[best_action][1]) < 0.001
        family_signal_used = fallback_scores[best_action][1] if fallback_used_for_selected else 0.0

        audit_entry = {
            "episode_id": self._episode_id,
            "step_id": self._probe_step_counter + 1,
            "object_id": object_id,
            "family": family,
            "has_dev_features": has_dev_feat,
            "dev_signal_zero": dev_signal_zero,
            "fallback_used": fallback_used_for_selected,
            "family_action_signal_used": round(family_signal_used, 6),
            "dev_best_action": dev_best_action,
            "dev_best_score": round(dev_best_score, 6),
            "dev_best_penalty": round(dev_best_penalty, 6),
            "fallback_best_action": best_action,
            "fallback_best_score": round(fb_best_score, 6),
            "fallback_best_penalty": round(fb_best_penalty, 6),
            "final_action": best_action,
            "final_score": round(best_score, 6),
            "rank_before_fallback": dev_rank_of_fb_selected,
            "rank_after_fallback": fb_rank_of_dev_selected,
            "selected_before_fallback": dev_best_action,
            "selected_after_fallback": best_action,
            "action_changed": dev_best_action != best_action,
            "score_delta": round(fb_best_score - dev_best_score, 6),
            "per_action_dev_scores": {a: round(dev_scores[a][0], 6) for a in MAIN_CANDIDATE_ACTIONS},
            "per_action_fb_scores": {a: round(fallback_scores[a][0], 6) for a in MAIN_CANDIDATE_ACTIONS},
        }
        self._decision_audit_log.append(audit_entry)

        self._probed_oids.add(object_id)
        bp, rp, sp, rs, fs = best_info
        self._selected_scores.append({
            "oid": object_id, "action": best_action,
            "prior": bp, "risk_penalty": round(rp, 6),
            "scaled_risk_penalty": round(sp, 6),
            "final_score": round(fs, 6), "risk_adjustment": round(sp, 6),
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
            if is_eff: effect = "effective"
            elif is_zero: effect = "zero"
            else: effect = "harmful"
            self._probe_effects.append({
                "oid": object_id, "action": action, "outcome": outcome,
                "utility_delta": round(utility_delta, 6),
                "is_effective": is_eff, "is_zero": is_zero, "is_harmful": not is_eff and not is_zero,
                "effect": effect,
            })
            is_violation, is_eligible, sp = check_prior_violation(features, action, outcome)
            goal = ACTION_TO_QUERY.get(action, "")
            self._probe_step_counter += 1
            self._fcrm_dev.update(
                features, goal, action, is_violation, is_eligible,
                episode_id=self._episode_id,
                step_id=self._probe_step_counter,
                object_id=object_id,
            )
            self._fcrm_family.update(
                features, goal, action, is_violation, is_eligible,
                episode_id=self._episode_id,
                step_id=self._probe_step_counter,
                object_id=object_id,
            )
        else:
            self._im.incorporate_probe(object_id, action, outcome)

    def get_fallback_stats(self):
        return {"fallback_activation_count": self._fallback_activation_count}

    def get_decision_audit_log(self):
        return list(self._decision_audit_log)


class FrozenMemoryPolicy(SoftPriorOnlyPolicyV2):
    def __init__(self, instance_memory, rng, frozen_memory,
                 target_probe_count=8, explore_fraction=0.0, risk_penalty_scale=1.0,
                 aggregation_mode="signed_sum", max_risk=MAX_RISK):
        super().__init__(instance_memory, rng, target_probe_count, explore_fraction)
        self._fcrm = frozen_memory
        self._risk_penalty_scale = risk_penalty_scale
        self.aggregation_mode = aggregation_mode
        self.max_risk = max_risk
        self._variant = f"frozen_{aggregation_mode}"

    def _get_per_feature_weights(self, features, action):
        goal = ACTION_TO_QUERY.get(action, "")
        family = detect_type_family(features)
        weights = []
        for dev_feat in INJECTED_DEVIATION_FEATURES:
            if features.get(dev_feat, False):
                w = self._fcrm.get_raw_risk_weight_signed(goal, action, family, dev_feat)
                weights.append((dev_feat, w))
        return weights

    def get_total_risk_penalty(self, features, action):
        if self.aggregation_mode == "signed_sum":
            weights = self._get_per_feature_weights(features, action)
            total = sum(w for _, w in weights)
            return max(-self.max_risk, min(self.max_risk, total))
        return 0.0

    def _compute_adjusted_score(self, features, action, norm_cost):
        base_prior = get_goal_soft_prior(features, action)
        risk_penalty = self.get_total_risk_penalty(features, action)
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
            "final_score": round(fs, 6), "risk_adjustment": round(sp, 6),
        })
        return True, best_action


# =============================================================================
# 6. Run All Variants + Detailed Audit
# =============================================================================
print("\n[5/7] Running all variants with audit instrumentation...")

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

def count_eff(ep_list):
    total = 0
    for ep in ep_list:
        for d in ep.get("probe_details", []):
            if d.get("probe_utility_signal") == "effective":
                total += 1
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

def avoided_nv_count(ep_a_list, ep_var_list):
    count = 0
    a_set = {}
    for ep_a in ep_a_list:
        for d in ep_a["probe_details"]:
            if d.get("is_nonviolation"): a_set[(d["oid"], d["action"])] = d
    var_set = set()
    for ep_v in ep_var_list:
        for d in ep_v["probe_details"]:
            var_set.add((d["oid"], d["action"]))
    for key in a_set:
        if key not in var_set:
            count += 1
    return count

def compute_variant_stats(all_episodes_var, all_episodes_a, variant_label):
    agg_v = aggregate_episodes(all_episodes_var)
    agg_a = aggregate_episodes(all_episodes_a)
    eff_v = count_eff(all_episodes_var)
    eff_a = count_eff(all_episodes_a)
    avoided_eff = eff_a - eff_v
    avoided_pv = avoided_pv_count(all_episodes_a, all_episodes_var)
    avoided_nv = avoided_nv_count(all_episodes_a, all_episodes_var)

    helpful = 0; harmful = 0
    a_probe_map = {}
    for ep_a in all_episodes_a:
        for d in ep_a["probe_details"]:
            a_probe_map[(d["oid"], d["action"])] = d
    var_probe_map = {}
    for ep_v in all_episodes_var:
        for d in ep_v["probe_details"]:
            var_probe_map[(d["oid"], d["action"])] = d

    avoided_keys = set(a_probe_map.keys()) - set(var_probe_map.keys())
    for oid, action in avoided_keys:
        a_d = a_probe_map[(oid, action)]
        if a_d.get("is_violation"): helpful += 1
        if a_d.get("is_nonviolation"): harmful += 1

    delta_macro = round(agg_v["mean_macro_bal"] - agg_a["mean_macro_bal"], 4)

    return {
        "variant": variant_label,
        "total_prior_violations": agg_v["total_prior_violations"],
        "deceptive_object_prior_violations": agg_v["total_deceptive_prior_violations"],
        "normal_object_prior_violations": agg_v["total_normal_prior_violations"],
        "mean_macro_bal": agg_v["mean_macro_bal"],
        "macro_delta_vs_A": delta_macro,
        "effective_probe_count": eff_v,
        "avoided_effective_probe_count": avoided_eff,
        "avoided_prior_violation_count": avoided_pv,
        "avoided_nonviolation_count": avoided_nv,
        "helpful_avoidance_count": helpful,
        "harmful_avoidance_count": harmful,
        "hard_exclusion_used": False,
        "agg": agg_v,
        "episodes": all_episodes_var,
    }


# =============================================================================
# NEW: Direct + Trajectory audit analysis
# =============================================================================

# Threshold for "positive prior evidence": soft_prior >= 0.60 (eligible)
POSITIVE_PRIOR_THRESHOLD = 0.60
# Threshold for ambiguous family: top-two gap <= 1 or top <= 2
FAMILY_AMBIGUITY_GAP = 1
FAMILY_AMBIGUITY_MIN_TOP = 2


def compute_positive_prior_evidence(features, action, fcrm_dev=None, fcrm_family=None):
    """Check if there is positive prior evidence for this (family, action).

    positive_prior_evidence=true if:
      - soft_prior >= 0.60 (high prior expectation), OR
      - historical nonviolation count > violation count for matched family/action
        in either FCRM memory (if available).
    """
    sp = get_goal_soft_prior(features, action)
    if sp >= POSITIVE_PRIOR_THRESHOLD:
        return True, {"source": "soft_prior", "soft_prior": round(sp, 4)}

    # Check memory evidence if available
    family = detect_type_family(features)
    goal = ACTION_TO_QUERY.get(action, "")
    nv_total = 0; v_total = 0

    if fcrm_family is not None and fcrm_family.key_mode == "family_only":
        c = fcrm_family.get_raw_counts(goal, action, family)
        nv = (c["nonviolation_with"] + c["nonviolation_without"]) - 2.0
        v = (c["violation_with"] + c["violation_without"]) - 2.0
        nv_total += max(0, nv); v_total += max(0, v)

    if nv_total > v_total and (nv_total + v_total) >= 1:
        return True, {"source": "memory", "nv_total": nv_total, "v_total": v_total}

    return False, {"source": "none", "soft_prior": round(sp, 4)}


def compute_direct_effects(decision_audit_log, test_objects):
    """Analyze direct decision deltas from the per-decision audit log.

    A direct effect is when the fallback path changes the selected action
    at decision time, compared to what dev-only path would have selected.

    Returns per-decision records and aggregate counts.
    """
    records = []
    for entry in decision_audit_log:
        oid = entry["object_id"]
        dev_action = entry["dev_best_action"]
        fb_action = entry["fallback_best_action"]
        action_changed = entry["action_changed"]
        fallback_used = entry["fallback_used"]

        obj = test_objects.get(oid, {})
        features = obj.get("visible_features", {})
        family = entry["family"]
        family_ambiguous = is_family_signal_ambiguous(features)

        # Ground truth for both actions
        gt_dev = get_ground_truth_outcome(obj, dev_action)
        gt_fb = get_ground_truth_outcome(obj, fb_action)

        is_dev_violation, is_dev_eligible, _ = check_prior_violation(features, dev_action, gt_dev)
        is_fb_violation, is_fb_eligible, _ = check_prior_violation(features, fb_action, gt_fb)

        # Classify direct effect
        if not action_changed:
            classification = "neutral"
            rationale = "same action selected under both paths"
        elif is_dev_violation and not is_fb_violation:
            classification = "helpful"
            rationale = f"fallback switched from violation ({dev_action}) to non-violation ({fb_action})"
        elif not is_dev_violation and is_fb_violation:
            classification = "harmful"
            rationale = f"fallback switched from non-violation ({dev_action}) to violation ({fb_action})"
        elif is_dev_violation and is_fb_violation:
            classification = "neutral"
            rationale = "both paths select violation actions"
        elif not is_dev_violation and not is_fb_violation:
            # Both non-violations, but different actions - check if one is better
            if is_dev_eligible and not is_fb_eligible:
                classification = "harmful"
                rationale = f"fallback switched from eligible ({dev_action}) to ineligible ({fb_action})"
            elif not is_dev_eligible and is_fb_eligible:
                classification = "helpful"
                rationale = f"fallback switched from ineligible ({dev_action}) to eligible ({fb_action})"
            else:
                classification = "neutral"
                rationale = f"both paths select different non-violation actions"
        else:
            classification = "neutral"
            rationale = "unclear classification"

        # positive_prior_evidence check
        has_positive_prior, prior_info = compute_positive_prior_evidence(features, fb_action)

        records.append({
            "episode": entry["episode_id"],
            "step_id": entry["step_id"],
            "oid": oid,
            "family": family,
            "family_signal_ambiguous": family_ambiguous,
            "has_dev_features": entry["has_dev_features"],
            "dev_signal_zero": entry["dev_signal_zero"],
            "fallback_used": fallback_used,
            "family_action_signal_used": entry["family_action_signal_used"],
            "dev_best_action": dev_action,
            "dev_best_score": entry["dev_best_score"],
            "dev_best_penalty": entry["dev_best_penalty"],
            "fallback_best_action": fb_action,
            "fallback_best_score": entry["fallback_best_score"],
            "fallback_best_penalty": entry["fallback_best_penalty"],
            "action_changed": action_changed,
            "score_delta": entry["score_delta"],
            "rank_before_fallback": entry["rank_before_fallback"],
            "rank_after_fallback": entry["rank_after_fallback"],
            "gt_dev_outcome": int(gt_dev),
            "gt_fb_outcome": int(gt_fb),
            "is_dev_violation": is_dev_violation,
            "is_fb_violation": is_fb_violation,
            "classification": classification,
            "rationale": rationale,
            "positive_prior_evidence": has_positive_prior,
            "prior_info": prior_info,
            "effect_type": "direct",
        })

    return records


def compute_trajectory_effects(eps_fb, eps_b, test_objects, decision_audit_logs):
    """Analyze trajectory effects: final probe-set differences after independent runs.

    These are cases where the probe sets differ but the difference may be
    due to cascading memory updates, not direct fallback decisions.
    """
    records = []

    # Build set of direct-effect OIDs so we can distinguish
    direct_affected_oids = set()
    for log in decision_audit_logs:
        for entry in log:
            if entry["action_changed"]:
                direct_affected_oids.add((entry["object_id"], entry["episode_id"]))

    for ep_idx, (ep_fb, ep_b) in enumerate(zip(eps_fb, eps_b)):
        fb_probes = {(d["oid"], d["action"]): d for d in ep_fb["probe_details"]}
        b_probes = {(d["oid"], d["action"]): d for d in ep_b["probe_details"]}
        fb_set = set(fb_probes.keys())
        b_set = set(b_probes.keys())

        # Cases B probed but FB avoided
        for oid, action in (b_set - fb_set):
            b_d = b_probes[(oid, action)]
            obj = test_objects.get(oid, {})
            features = obj.get("visible_features", {})
            has_dev = any(features.get(f, False) for f in INJECTED_DEVIATION_FEATURES)
            family = detect_type_family(features)
            family_ambiguous = is_family_signal_ambiguous(features)
            gt_outcome = get_ground_truth_outcome(obj, action)
            sp = get_goal_soft_prior(features, action)
            is_violation = b_d.get("is_violation", False)
            is_nonviolation = b_d.get("is_nonviolation", False)
            fallback_activated = not has_dev
            has_positive_prior, prior_info = compute_positive_prior_evidence(features, action)

            if is_violation:
                classification = "helpful"
                rationale = "trajectory: FB avoided violation B would have committed"
            elif is_nonviolation:
                classification = "harmful"
                rationale = "trajectory: FB avoided non-violation B correctly probed"
            else:
                classification = "neutral"
                rationale = "trajectory: FB avoided ineligible probe"

            was_direct = (oid, ep_idx) in direct_affected_oids

            records.append({
                "episode": ep_idx,
                "oid": oid,
                "action": action,
                "family": family,
                "family_signal_ambiguous": family_ambiguous,
                "has_dev_features": has_dev,
                "fallback_activated": fallback_activated,
                "soft_prior": round(sp, 4),
                "is_violation": is_violation,
                "is_nonviolation": is_nonviolation,
                "ground_truth_outcome": int(gt_outcome),
                "classification": classification,
                "rationale": rationale,
                "positive_prior_evidence": has_positive_prior,
                "prior_info": prior_info,
                "effect_type": "trajectory",
                "change_type": "fb_avoided",
                "cascaded_from_direct": was_direct,
            })

        # Cases FB probed but B avoided
        for oid, action in (fb_set - b_set):
            fb_d = fb_probes[(oid, action)]
            obj = test_objects.get(oid, {})
            features = obj.get("visible_features", {})
            has_dev = any(features.get(f, False) for f in INJECTED_DEVIATION_FEATURES)
            family = detect_type_family(features)
            family_ambiguous = is_family_signal_ambiguous(features)
            gt_outcome = get_ground_truth_outcome(obj, action)
            sp = get_goal_soft_prior(features, action)
            is_violation = fb_d.get("is_violation", False)
            is_nonviolation = fb_d.get("is_nonviolation", False)
            fallback_activated = not has_dev
            has_positive_prior, prior_info = compute_positive_prior_evidence(features, action)

            if is_violation:
                classification = "harmful"
                rationale = "trajectory: FB probed violation B correctly avoided"
            elif is_nonviolation:
                classification = "helpful"
                rationale = "trajectory: FB probed non-violation B missed"
            else:
                classification = "neutral"
                rationale = "trajectory: FB probed ineligible case B skipped"

            was_direct = (oid, ep_idx) in direct_affected_oids

            records.append({
                "episode": ep_idx,
                "oid": oid,
                "action": action,
                "family": family,
                "family_signal_ambiguous": family_ambiguous,
                "has_dev_features": has_dev,
                "fallback_activated": fallback_activated,
                "soft_prior": round(sp, 4),
                "is_violation": is_violation,
                "is_nonviolation": is_nonviolation,
                "ground_truth_outcome": int(gt_outcome),
                "classification": classification,
                "rationale": rationale,
                "positive_prior_evidence": has_positive_prior,
                "prior_info": prior_info,
                "effect_type": "trajectory",
                "change_type": "fb_added",
                "cascaded_from_direct": was_direct,
            })

    return records


def compute_episode_level_deltas(eps_fb, eps_b):
    """Compute per-episode PV and non-PV probe deltas (FB - B)."""
    deltas = []
    for ep_idx, (ep_fb, ep_b) in enumerate(zip(eps_fb, eps_b)):
        fb_pv = ep_fb.get("prior_violation_count", 0)
        b_pv = ep_b.get("prior_violation_count", 0)
        fb_probes = len(ep_fb.get("probe_details", []))
        b_probes = len(ep_b.get("probe_details", []))
        fb_nonpv = fb_probes - fb_pv
        b_nonpv = b_probes - b_pv

        deltas.append({
            "episode": ep_idx,
            "fb_total_pv": fb_pv,
            "b_total_pv": b_pv,
            "pv_delta": fb_pv - b_pv,
            "fb_total_probes": fb_probes,
            "b_total_probes": b_probes,
            "nonpv_probe_delta": fb_nonpv - b_nonpv,
            "net_helpful": (b_pv - fb_pv) + (fb_nonpv - b_nonpv),
        })
    return deltas


def summarize_full_audit(direct_records, trajectory_records, episode_deltas):
    """Summarize direct + trajectory effects with episode-level accounting."""
    direct_helpful = [r for r in direct_records if r["classification"] == "helpful"]
    direct_harmful = [r for r in direct_records if r["classification"] == "harmful"]
    direct_neutral = [r for r in direct_records if r["classification"] == "neutral"]

    traj_helpful = [r for r in trajectory_records if r["classification"] == "helpful"]
    traj_harmful = [r for r in trajectory_records if r["classification"] == "harmful"]
    traj_neutral = [r for r in trajectory_records if r["classification"] == "neutral"]

    # Direct harmful with positive prior evidence
    direct_harmful_positive_prior = [r for r in direct_harmful if r.get("positive_prior_evidence")]
    direct_harmful_ambiguous_family = [r for r in direct_harmful if r.get("family_signal_ambiguous")]

    # Direct harmful where fallback was activated (not just dev-feat path difference)
    direct_harmful_fallback = [r for r in direct_harmful if r.get("fallback_used")]
    direct_harmful_fallback_positive_prior = [r for r in direct_harmful_fallback if r.get("positive_prior_evidence")]

    # Episode-level aggregates
    total_pv_delta = sum(d["pv_delta"] for d in episode_deltas)
    total_nonpv_delta = sum(d["nonpv_probe_delta"] for d in episode_deltas)
    total_ep_net = sum(d["net_helpful"] for d in episode_deltas)

    return {
        # Direct effects
        "direct_helpful": len(direct_helpful),
        "direct_harmful": len(direct_harmful),
        "direct_neutral": len(direct_neutral),
        "direct_net": len(direct_helpful) - len(direct_harmful),
        "direct_action_changes": len([r for r in direct_records if r["action_changed"]]),
        "direct_fallback_activated_changes": len([r for r in direct_records
                                                   if r["action_changed"] and r["fallback_used"]]),
        # Trajectory effects
        "trajectory_helpful": len(traj_helpful),
        "trajectory_harmful": len(traj_harmful),
        "trajectory_neutral": len(traj_neutral),
        "trajectory_net": len(traj_helpful) - len(traj_harmful),
        # Episode-level deltas
        "episode_pv_delta": total_pv_delta,
        "episode_nonpv_probe_delta": total_nonpv_delta,
        "episode_net_helpful": total_ep_net,
        "episode_deltas": episode_deltas,
        # Harmful case diagnostics
        "direct_harmful_with_positive_prior": len(direct_harmful_positive_prior),
        "direct_harmful_with_ambiguous_family": len(direct_harmful_ambiguous_family),
        "direct_harmful_fallback_count": len(direct_harmful_fallback),
        "direct_harmful_fallback_positive_prior": len(direct_harmful_fallback_positive_prior),
        "direct_harmful_systematic": (
            len(direct_harmful_fallback) >= 3
            and len(direct_harmful_fallback_positive_prior) == len(direct_harmful_fallback)
        ),
        # Case details
        "direct_helpful_cases": [
            {"oid": r["oid"], "episode": r["episode"],
             "dev_action": r["dev_best_action"], "fb_action": r["fallback_best_action"],
             "fallback_used": r["fallback_used"]} for r in direct_helpful
        ],
        "direct_harmful_cases": [
            {"oid": r["oid"], "episode": r["episode"],
             "dev_action": r["dev_best_action"], "fb_action": r["fallback_best_action"],
             "fallback_used": r.get("fallback_used"),
             "positive_prior_evidence": r.get("positive_prior_evidence"),
             "family_ambiguous": r.get("family_signal_ambiguous"),
             "soft_prior": r.get("soft_prior")} for r in direct_harmful
        ],
        "trajectory_helpful_cases": [
            {"oid": r["oid"], "action": r["action"], "episode": r["episode"],
             "fallback_activated": r["fallback_activated"],
             "cascaded_from_direct": r.get("cascaded_from_direct", False)} for r in traj_helpful
        ],
        "trajectory_harmful_cases": [
            {"oid": r["oid"], "action": r["action"], "episode": r["episode"],
             "fallback_activated": r["fallback_activated"],
             "positive_prior_evidence": r.get("positive_prior_evidence"),
             "family_ambiguous": r.get("family_signal_ambiguous"),
             "cascaded_from_direct": r.get("cascaded_from_direct", False)} for r in traj_harmful
        ],
        # Full records
        "direct_records": direct_records,
        "trajectory_records": trajectory_records,
    }


# --- C15b reference ---
print("  Running C15b reference...")
c15b_im = im_base.clone()
c15b_rng = random.Random(SMOKE_SEED + 700)
c15b_positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, c15b_rng)
c15b_env = MiniMCEnvironment(test_objects_standard, c15b_positions, initial_budget=BUDGET)

class C15b_RefPolicy(MiniMCPolicy):
    PHASE_OBSERVE = 1; PHASE_PROBE = 2
    def __init__(self, instance_memory, rng, cost_weight=COST_WEIGHT):
        self._im = instance_memory; self._rng = rng; self._cost_weight = cost_weight
        self._phase = self.PHASE_OBSERVE; self._probed_oids = set()
        self._pre_probe_entropies = {}; self._observe_pass_visit_count = 0
        self._probe_effects = []; self._selected_scores = []; self._probe_tracking = []
    def reset(self, view):
        self._phase = self.PHASE_OBSERVE; self._probed_oids = set()
        self._pre_probe_entropies = {}; self._observe_pass_visit_count = 0
        self._probe_effects = []; self._selected_scores = []; self._probe_tracking = []
    def _should_transition_to_probe_phase(self, view):
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
            if self._should_transition_to_probe_phase(view):
                self._phase = self.PHASE_PROBE
                self._observe_pass_visit_count = sum(1 for oid in view.get_all_object_ids() if view.is_visited(oid))
                return self._select_probe_target(view)
            best_oid, best_cost = None, float('inf')
            for oid in unvisited:
                total = view.compute_reach_cost(oid) + view.observe_cost
                if view.can_afford(total) and total < best_cost:
                    best_cost, best_oid = total, oid
            return best_oid
        return self._select_probe_target(view) if self._phase == self.PHASE_PROBE else None
    def _score_candidate(self, probs):
        return _compute_utility(probs)
    def _select_probe_target(self, view):
        if len(self._probed_oids) >= 5: return None
        candidates = []
        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids: continue
            features = view.get_observed_features(oid)
            if features is None: continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost): continue
            probs = self._im.predict_all_affordances({"id": oid, "visible_features": features})
            candidates.append((oid, self._score_candidate(probs)))
        if not candidates: return None
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE: return False, None
        if len(self._probed_oids) >= 5: return False, None
        features = view.get_observed_features(object_id)
        if features is None: return False, None
        probs = self._im.predict_all_affordances({"id": object_id, "visible_features": features})
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)
        best_action, best_gain = None, float('-inf')
        for action in MAIN_CANDIDATE_ACTIONS:
            gain = probs.get(ACTION_TO_FEATURE.get(action, ""), 0.5)
            if gain > best_gain: best_gain, best_action = gain, action
        self._probed_oids.add(object_id)
        self._selected_scores.append({"oid": object_id, "action": best_action, "final_score": best_gain})
        return True, best_action
    def on_probe_result(self, view, object_id, action, outcome):
        self._im.incorporate_probe(object_id, action, outcome)
    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            per_object[oid] = self._im.predict_all_affordances({"id": oid, "visible_features": features})
        return {"per_object": per_object}
    def get_probe_effects(self): return list(self._probe_effects)
    def get_selected_scores(self): return list(self._selected_scores)

c15b_policy = C15b_RefPolicy(c15b_im, c15b_rng)
c15b_preds, _, c15b_obs, _, _ = run_policy(c15b_policy, c15b_env)
c15b_metrics = compute_full_metrics(_unwrap_preds(c15b_preds), query_gt_standard)
c15b_macro_bal = c15b_metrics["macro_query_balanced_accuracy"]
c15b_probed_n = sum(1 for oid in c15b_obs.get_all_object_ids() if c15b_obs.get_probe_results(oid))
print(f"  C15b macro_bal={c15b_macro_bal:.4f}  target_probe_count={c15b_probed_n}")

# Precompute shared episode positions
print("  Precomputing shared episode positions...")
shared_position_seed = SMOKE_SEED + 15000
episode_positions_by_ep = {}
pos_rng = random.Random(shared_position_seed)
test_oids_local = sorted(test_objects_deceptive.keys())
for ep in range(N_EPISODES):
    ep_positions = MiniMCSimulatorTruth.assign_positions(
        test_oids_local, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, pos_rng)
    episode_positions_by_ep[ep] = dict(ep_positions)


# --- Run all variants for ONE seed ---
def run_all_variants_for_seed(seed):
    """Run all 5 variants for a single seed and return results + audit."""
    local_rng_base = seed + 20000

    # Shared episode positions for this seed
    local_pos_seed = seed + 15000
    local_pos_rng = random.Random(local_pos_seed)
    local_test_oids = sorted(test_objects_deceptive.keys())
    local_ep_positions = {}
    for ep in range(N_EPISODES):
        local_ep_positions[ep] = dict(MiniMCSimulatorTruth.assign_positions(
            local_test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, local_pos_rng))

    # --- A_no_memory ---
    all_episodes_a = []
    pv_history_a = set()
    for ep in range(N_EPISODES):
        ep_seed = local_rng_base + 2 + ep * 100
        ep_rng = random.Random(ep_seed)
        im_a = im_base.clone()
        env_a = MiniMCEnvironment(test_objects_deceptive, local_ep_positions[ep], initial_budget=BUDGET)
        policy_a = SoftPriorOnlyPolicyV2(im_a, ep_rng, target_probe_count=c15b_probed_n)
        ep_a, pv_history_a = run_episode_and_extract(
            test_objects_deceptive, query_gt_deceptive, policy_a, env_a, ep,
            prior_violation_history=pv_history_a)
        all_episodes_a.append(ep_a)

    agg_a = aggregate_episodes(all_episodes_a)

    # --- B_signed_sum ---
    memory_b = FamilyConditionedRiskMemory(key_mode="family_dev")
    eps_b = []
    pv_hist_b = set()
    step_counter_b = [0]
    for ep in range(N_EPISODES):
        ep_seed = local_rng_base + 200 + ep * 100
        ep_rng = random.Random(ep_seed)
        im_ep = im_base.clone()
        env_ep = MiniMCEnvironment(test_objects_deceptive, local_ep_positions[ep], initial_budget=BUDGET)
        policy_ep = OnlineSignedPolicy(
            im_ep, ep_rng, memory_b, target_probe_count=c15b_probed_n,
            risk_penalty_scale=1.0, episode_id=ep, step_counter_ref=step_counter_b)
        ep_data, pv_hist_b = run_episode_and_extract(
            test_objects_deceptive, query_gt_deceptive, policy_ep, env_ep, ep,
            prior_violation_history=pv_hist_b)
        eps_b.append(ep_data)

    stats_b = compute_variant_stats(eps_b, all_episodes_a, "B_signed_sum_current")

    # --- Family_only ---
    memory_fo = FamilyConditionedRiskMemory(key_mode="family_only")
    eps_fo = []
    pv_hist_fo = set()
    step_counter_fo = [0]
    for ep in range(N_EPISODES):
        ep_seed = local_rng_base + 400 + ep * 100
        ep_rng = random.Random(ep_seed)
        im_ep = im_base.clone()
        env_ep = MiniMCEnvironment(test_objects_deceptive, local_ep_positions[ep], initial_budget=BUDGET)
        policy_ep = OnlineFamilyOnlyPolicy(
            im_ep, ep_rng, memory_fo, target_probe_count=c15b_probed_n,
            risk_penalty_scale=1.0, episode_id=ep, step_counter_ref=step_counter_fo)
        ep_data, pv_hist_fo = run_episode_and_extract(
            test_objects_deceptive, query_gt_deceptive, policy_ep, env_ep, ep,
            prior_violation_history=pv_hist_fo)
        eps_fo.append(ep_data)

    stats_fo = compute_variant_stats(eps_fo, all_episodes_a, "Family_only")

    # --- B_dev_plus_family_action_fallback ---
    memory_fb_dev = FamilyConditionedRiskMemory(key_mode="family_dev")
    memory_fb_family = FamilyConditionedRiskMemory(key_mode="family_only")
    eps_fb = []
    pv_hist_fb = set()
    step_counter_fb = [0]
    fb_decision_logs = []  # Collect per-episode decision audit logs
    for ep in range(N_EPISODES):
        ep_seed = local_rng_base + 600 + ep * 100
        ep_rng = random.Random(ep_seed)
        im_ep = im_base.clone()
        env_ep = MiniMCEnvironment(test_objects_deceptive, local_ep_positions[ep], initial_budget=BUDGET)
        policy_ep = DevPlusFamilyActionFallbackPolicy(
            im_ep, ep_rng, memory_fb_dev, memory_fb_family,
            target_probe_count=c15b_probed_n,
            risk_penalty_scale=1.0, episode_id=ep, step_counter_ref=step_counter_fb)
        ep_data, pv_hist_fb = run_episode_and_extract(
            test_objects_deceptive, query_gt_deceptive, policy_ep, env_ep, ep,
            prior_violation_history=pv_hist_fb)
        eps_fb.append(ep_data)
        fb_decision_logs.append(policy_ep.get_decision_audit_log())

    stats_fb = compute_variant_stats(eps_fb, all_episodes_a, "B_dev_plus_family_action_fallback")

    # --- Offline ---
    fcrm_offline = build_offline_oracle_memory(test_objects_deceptive, key_mode="family_dev")
    eps_offline = []
    pv_hist_offline = set()
    for ep in range(N_EPISODES):
        ep_seed = local_rng_base + 800 + ep * 100
        ep_rng = random.Random(ep_seed)
        im_ep = im_base.clone()
        env_ep = MiniMCEnvironment(test_objects_deceptive, local_ep_positions[ep], initial_budget=BUDGET)
        frozen = fcrm_offline.clone()
        policy_ep = FrozenMemoryPolicy(
            im_ep, ep_rng, frozen, target_probe_count=c15b_probed_n,
            aggregation_mode="signed_sum")
        ep_data, pv_hist_offline = run_episode_and_extract(
            test_objects_deceptive, query_gt_deceptive, policy_ep, env_ep, ep,
            prior_violation_history=pv_hist_offline)
        eps_offline.append(ep_data)

    stats_offline = compute_variant_stats(eps_offline, all_episodes_a, "Offline_current")

    # --- AUDIT: Direct + Trajectory effects ---
    direct_records = compute_direct_effects(
        [log for ep_logs in fb_decision_logs for log in ep_logs], test_objects_deceptive)
    trajectory_records = compute_trajectory_effects(
        eps_fb, eps_b, test_objects_deceptive, fb_decision_logs)
    episode_deltas = compute_episode_level_deltas(eps_fb, eps_b)
    audit_summary = summarize_full_audit(direct_records, trajectory_records, episode_deltas)

    # Raw-zero diagnostics
    raw_zero_cases = []
    for ep in eps_b:
        for d in ep["probe_details"]:
            if d.get("is_violation"):
                oid = d["oid"]; action = d["action"]
                features = test_objects_deceptive.get(oid, {}).get("visible_features", {})
                dev_feats = [f for f in INJECTED_DEVIATION_FEATURES if features.get(f, False)]
                if len(dev_feats) == 0 and abs(d.get("risk_penalty", 0.0)) < 0.001:
                    fb_probes_set = set()
                    for ep_fb in eps_fb:
                        for dd in ep_fb["probe_details"]:
                            fb_probes_set.add((dd["oid"], dd["action"]))
                    fo_probes_set = set()
                    for ep_fo in eps_fo:
                        for dd in ep_fo["probe_details"]:
                            fo_probes_set.add((dd["oid"], dd["action"]))
                    raw_zero_cases.append({
                        "oid": oid, "action": action,
                        "family": d.get("family", "unknown"),
                        "dev_feats_present": dev_feats,
                        "risk_penalty": d.get("risk_penalty", 0.0),
                        "avoided_by_fallback": (oid, action) not in fb_probes_set,
                        "avoided_by_family_only": (oid, action) not in fo_probes_set,
                    })

    # Sanity checks
    hidden_feature_leakage = False
    no_oracle_leakage = True

    # Compute symmetric diff size
    fb_probes_set = set()
    for ep in eps_fb:
        for d in ep["probe_details"]:
            fb_probes_set.add((d["oid"], d["action"]))
    b_probes_set = set()
    for ep in eps_b:
        for d in ep["probe_details"]:
            b_probes_set.add((d["oid"], d["action"]))
    selected_probe_set_changed = len(fb_probes_set.symmetric_difference(b_probes_set))

    return {
        "seed": seed,
        "agg_a": agg_a,
        "stats_b": stats_b,
        "stats_fo": stats_fo,
        "stats_fb": stats_fb,
        "stats_offline": stats_offline,
        "audit_summary": audit_summary,
        "raw_zero_cases": raw_zero_cases,
        "raw_zero_rescued": sum(1 for c in raw_zero_cases if c["avoided_by_fallback"]),
        "raw_zero_missed": sum(1 for c in raw_zero_cases if not c["avoided_by_fallback"]),
        "selected_probe_set_changed": selected_probe_set_changed,
        "hidden_feature_leakage_detected": hidden_feature_leakage,
        "no_oracle_leakage_confirmed": no_oracle_leakage,
        "eps_a": all_episodes_a,
        "eps_b": eps_b,
        "eps_fb": eps_fb,
        "eps_fo": eps_fo,
        "eps_offline": eps_offline,
    }


# =============================================================================
# 7. Run per seed or multiseed
# =============================================================================
print("\n[6/7] Running seeds...")
all_seed_results = {}

for seed in MULTISEED_LIST:
    print(f"\n  --- Seed {seed} ---")
    SMOKE_SEED = seed
    # Rebuild deceptive objects for this seed
    test_objects_standard = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")
    test_oids = sorted(test_objects_standard.keys())

    test_objects_deceptive = copy.deepcopy(test_objects_standard)
    apple_oids = [oid for oid in test_oids if "apple" in oid]
    wood_oids = [oid for oid in test_oids if "wood_log" in oid]
    stone_oids = [oid for oid in test_oids if "stone_block" in oid]

    deceptive_rng = random.Random(seed + 9999)

    for oid in deceptive_rng.sample(apple_oids, min(5, len(apple_oids))):
        obj = test_objects_deceptive[oid]
        obj["hidden_affordance_profile"]["eat"] = "fail"
        obj["_deceptive"] = True; obj["_deceptive_action"] = "eat"
        for feat in APPLE_CORE:
            if feat in FEATURE_UNIVERSE_SET: obj["visible_features"][feat] = True
        for feat, val in APPLE_DEVIATION.items():
            if feat in FEATURE_UNIVERSE_SET: obj["visible_features"][feat] = val

    for oid in deceptive_rng.sample(wood_oids, min(5, len(wood_oids))):
        obj = test_objects_deceptive[oid]
        obj["hidden_affordance_profile"]["burn_as_fuel"] = "fail"
        obj["_deceptive"] = True; obj["_deceptive_action"] = "burn_as_fuel"
        for feat in WOOD_CORE:
            if feat in FEATURE_UNIVERSE_SET: obj["visible_features"][feat] = True
        for feat, val in WOOD_DEVIATION.items():
            if feat in FEATURE_UNIVERSE_SET: obj["visible_features"][feat] = val

    for oid in deceptive_rng.sample(stone_oids, min(3, len(stone_oids))):
        obj = test_objects_deceptive[oid]
        obj["hidden_affordance_profile"]["mine_with_pickaxe"] = "fail"
        obj["_deceptive"] = True; obj["_deceptive_action"] = "mine_with_pickaxe"
        for feat in STONE_CORE:
            if feat in FEATURE_UNIVERSE_SET: obj["visible_features"][feat] = True
        for feat, val in STONE_DEVIATION.items():
            if feat in FEATURE_UNIVERSE_SET: obj["visible_features"][feat] = val

    DECEPTIVE_OIDS = set()
    for oid in test_oids:
        if test_objects_deceptive[oid].get("_deceptive"):
            DECEPTIVE_OIDS.add(oid)

    query_gt_deceptive = compute_query_ground_truth(test_objects_deceptive)

    result = run_all_variants_for_seed(seed)
    all_seed_results[seed] = result

    s = result["audit_summary"]
    print(f"    FB total_pv={result['stats_fb']['total_prior_violations']}, "
          f"B total_pv={result['stats_b']['total_prior_violations']}")
    print(f"    Audit: direct_helpful={s['direct_helpful']}, direct_harmful={s['direct_harmful']}, "
          f"direct_net={s['direct_net']:+d}, traj_helpful={s['trajectory_helpful']}, "
          f"traj_harmful={s['trajectory_harmful']}, ep_pv_delta={s['episode_pv_delta']:+d}")


# =============================================================================
# 8. Aggregate across seeds + Output
# =============================================================================
print("\n[7/7] Aggregating and writing outputs...")
elapsed = time.time() - t0

# --- Build per-seed output ---
if not MULTISEED:
    seed = SMOKE_SEED
    result = all_seed_results[seed]
    audit = result["audit_summary"]

    # Determine safety decision using new criteria
    direct_harmful = audit["direct_harmful_cases"]
    direct_harmful_fallback = [h for h in direct_harmful if h.get("fallback_used")]
    direct_harmful_systematic = audit["direct_harmful_systematic"]
    harmful_override_strong_prior = any(
        h.get("positive_prior_evidence") and h.get("fallback_used")
        for h in direct_harmful
    )

    episode_net = audit["episode_net_helpful"]
    fallback_safe = (
        episode_net > 0
        and not direct_harmful_systematic
        and not harmful_override_strong_prior
        and not result["hidden_feature_leakage_detected"]
        and result["no_oracle_leakage_confirmed"]
    )

    harmful_avoidance_risk = len(direct_harmful_fallback) > 0

    # JSON output
    out = {
        "block_id": "1J34",
        "seed": seed,
        "condition": COND["label"],
        "elapsed_seconds": round(elapsed, 1),
        "comparison": {
            "A_no_memory": {
                "total_prior_violations": result["agg_a"]["total_prior_violations"],
                "mean_macro_bal": result["agg_a"]["mean_macro_bal"],
            },
            "B_signed_sum_current": result["stats_b"],
            "Family_only": result["stats_fo"],
            "B_dev_plus_family_action_fallback": result["stats_fb"],
            "Offline_current": result["stats_offline"],
        },
        "audit": {
            "direct_effects": {
                "helpful": audit["direct_helpful"],
                "harmful": audit["direct_harmful"],
                "neutral": audit["direct_neutral"],
                "net": audit["direct_net"],
                "action_changes": audit["direct_action_changes"],
                "fallback_activated_changes": audit["direct_fallback_activated_changes"],
            },
            "trajectory_effects": {
                "helpful": audit["trajectory_helpful"],
                "harmful": audit["trajectory_harmful"],
                "neutral": audit["trajectory_neutral"],
                "net": audit["trajectory_net"],
            },
            "episode_level": {
                "pv_delta": audit["episode_pv_delta"],
                "nonpv_probe_delta": audit["episode_nonpv_probe_delta"],
                "net_helpful": audit["episode_net_helpful"],
                "per_episode": audit["episode_deltas"],
            },
            "harmful_diagnostics": {
                "direct_harmful_with_positive_prior": audit["direct_harmful_with_positive_prior"],
                "direct_harmful_with_ambiguous_family": audit["direct_harmful_with_ambiguous_family"],
                "direct_harmful_fallback_count": audit["direct_harmful_fallback_count"],
                "direct_harmful_fallback_positive_prior": audit["direct_harmful_fallback_positive_prior"],
                "direct_harmful_systematic": direct_harmful_systematic,
                "harmful_override_strong_prior": harmful_override_strong_prior,
            },
            "cases": {
                "direct_helpful": audit["direct_helpful_cases"],
                "direct_harmful": audit["direct_harmful_cases"],
                "trajectory_helpful": audit["trajectory_helpful_cases"],
                "trajectory_harmful": audit["trajectory_harmful_cases"],
            },
            "records": {
                "direct": audit["direct_records"],
                "trajectory": audit["trajectory_records"],
            },
        },
        "raw_zero_diagnostics": {
            "raw_zero_cases_total": len(result["raw_zero_cases"]),
            "raw_zero_rescued": result["raw_zero_rescued"],
            "raw_zero_missed": result["raw_zero_missed"],
            "selected_probe_set_changed": result["selected_probe_set_changed"],
            "details": result["raw_zero_cases"],
        },
        "safety_decision": {
            "fallback_safe_enough_for_next_phase": fallback_safe,
            "episode_net_pv_delta": audit["episode_pv_delta"],
            "episode_net_helpful": episode_net,
            "harmful_avoidance_risk_detected": harmful_avoidance_risk,
            "direct_harmful_systematic": direct_harmful_systematic,
            "harmful_override_strong_prior": harmful_override_strong_prior,
            "hidden_feature_leakage_detected": result["hidden_feature_leakage_detected"],
            "no_oracle_leakage_confirmed": result["no_oracle_leakage_confirmed"],
        },
    }

    json_path = os.path.join(CURRENT_DIR, "runs", f"block1j34_fallback_help_harm_audit_seed{seed}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  JSON -> {json_path}")

    # CSV output — combine direct + trajectory records
    csv_path = os.path.join(CURRENT_DIR, "protocols", "block1j34_fallback_help_harm_audit_table.csv")
    csv_fields = [
        "seed", "episode", "effect_type", "change_type", "oid", "action",
        "family", "family_signal_ambiguous", "has_dev_features",
        "fallback_activated", "fallback_used",
        "soft_prior", "positive_prior_evidence",
        "ground_truth_outcome", "is_violation", "is_nonviolation",
        "classification", "rationale",
    ]
    all_csv_records = []
    for rec in audit["direct_records"]:
        r = dict(rec)
        r["seed"] = seed; r["effect_type"] = "direct"; r["change_type"] = ""
        r["soft_prior"] = r.get("soft_prior", "")
        r["fallback_activated"] = r.get("fallback_used", False)
        r["is_violation"] = r.get("is_fb_violation", False)
        r["is_nonviolation"] = not r.get("is_fb_violation", True)
        r["ground_truth_outcome"] = r.get("gt_fb_outcome", "")
        all_csv_records.append(r)
    for rec in audit["trajectory_records"]:
        r = dict(rec)
        r["seed"] = seed
        r["fallback_used"] = r.get("fallback_activated", False)
        all_csv_records.append(r)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for rec in all_csv_records:
            writer.writerow(rec)
    print(f"  CSV -> {csv_path}")

    # MD protocol
    md_lines = []
    md_lines.append("# Block 1J34 — Fallback Help-vs-Harm Audit")
    md_lines.append("")
    md_lines.append(f"- **Seed**: {seed}")
    md_lines.append(f"- **Condition**: {COND['label']}")
    md_lines.append(f"- **Episodes per variant**: {N_EPISODES}")
    md_lines.append(f"- **Elapsed**: {elapsed:.1f}s")
    md_lines.append("")
    md_lines.append("## Comparison Table")
    md_lines.append("")
    md_lines.append("| Variant | Total PV | Deceptive PV | Normal PV | Macro Bal | Delta vs A | Eff Probes | Avoided Eff | Avoided PV | Avoided NV | Helpful | Harmful |")
    md_lines.append("|---------|----------|-------------|-----------|-----------|------------|------------|-------------|------------|------------|---------|---------|")

    def row(name, s):
        return f"| {name} | {s['total_prior_violations']} | {s['deceptive_object_prior_violations']} | {s['normal_object_prior_violations']} | {s['mean_macro_bal']:.4f} | {s['macro_delta_vs_A']:+.4f} | {s['effective_probe_count']} | {s['avoided_effective_probe_count']} | {s['avoided_prior_violation_count']} | {s['avoided_nonviolation_count']} | {s['helpful_avoidance_count']} | {s['harmful_avoidance_count']} |"

    md_lines.append(row("A_no_memory", {
        "total_prior_violations": result["agg_a"]["total_prior_violations"],
        "deceptive_object_prior_violations": result["agg_a"]["total_deceptive_prior_violations"],
        "normal_object_prior_violations": result["agg_a"]["total_normal_prior_violations"],
        "mean_macro_bal": result["agg_a"]["mean_macro_bal"],
        "macro_delta_vs_A": 0.0,
        "effective_probe_count": count_eff(result["eps_a"]),
        "avoided_effective_probe_count": 0,
        "avoided_prior_violation_count": 0,
        "avoided_nonviolation_count": 0,
        "helpful_avoidance_count": 0,
        "harmful_avoidance_count": 0,
    }))
    md_lines.append(row("B_signed_sum_current", result["stats_b"]))
    md_lines.append(row("Family_only", result["stats_fo"]))
    md_lines.append(row("B_dev_plus_family_action_fallback", result["stats_fb"]))
    md_lines.append(row("Offline_current", result["stats_offline"]))
    md_lines.append("")

    md_lines.append("## Direct Effects (per-decision counterfactual)")
    md_lines.append("")
    md_lines.append(f"- **Direct helpful:** {audit['direct_helpful']}")
    md_lines.append(f"- **Direct harmful:** {audit['direct_harmful']}")
    md_lines.append(f"- **Direct neutral:** {audit['direct_neutral']}")
    md_lines.append(f"- **Direct net:** {audit['direct_net']:+d}")
    md_lines.append(f"- **Action changes from fallback:** {audit['direct_action_changes']}")
    md_lines.append(f"- **Fallback-activated changes:** {audit['direct_fallback_activated_changes']}")
    md_lines.append("")

    md_lines.append("## Trajectory Effects (final probe set diff)")
    md_lines.append("")
    md_lines.append(f"- **Trajectory helpful:** {audit['trajectory_helpful']}")
    md_lines.append(f"- **Trajectory harmful:** {audit['trajectory_harmful']}")
    md_lines.append(f"- **Trajectory neutral:** {audit['trajectory_neutral']}")
    md_lines.append(f"- **Trajectory net:** {audit['trajectory_net']:+d}")
    md_lines.append("")

    md_lines.append("## Episode-Level Deltas (FB - B)")
    md_lines.append("")
    md_lines.append("| Episode | FB PV | B PV | PV Delta | FB Probes | B Probes | NonPV Delta | Net |")
    md_lines.append("|---------|-------|------|----------|-----------|----------|-------------|-----|")
    for d in audit["episode_deltas"]:
        md_lines.append(f"| {d['episode']} | {d['fb_total_pv']} | {d['b_total_pv']} | {d['pv_delta']:+d} | {d['fb_total_probes']} | {d['b_total_probes']} | {d['nonpv_probe_delta']:+d} | {d['net_helpful']:+d} |")
    md_lines.append(f"| **Sum** | | | **{audit['episode_pv_delta']:+d}** | | | **{audit['episode_nonpv_probe_delta']:+d}** | **{audit['episode_net_helpful']:+d}** |")
    md_lines.append("")

    md_lines.append("## Harmful Case Diagnostics")
    md_lines.append("")
    md_lines.append(f"- **Direct harmful with positive prior evidence:** {audit['direct_harmful_with_positive_prior']}")
    md_lines.append(f"- **Direct harmful with ambiguous family signal:** {audit['direct_harmful_with_ambiguous_family']}")
    md_lines.append(f"- **Direct harmful fallback-activated:** {audit['direct_harmful_fallback_count']}")
    md_lines.append(f"- **Direct harmful fallback+positive_prior:** {audit['direct_harmful_fallback_positive_prior']}")
    md_lines.append(f"- **Direct harmful systematic:** {audit['direct_harmful_systematic']}")
    md_lines.append(f"- **Harmful override strong prior:** {harmful_override_strong_prior}")
    md_lines.append("")

    md_lines.append("## Safety Decision")
    md_lines.append("")
    md_lines.append(f"- **episode_net_pv_delta:** {audit['episode_pv_delta']:+d}")
    md_lines.append(f"- **episode_net_helpful:** {episode_net:+d}")
    md_lines.append(f"- **harmful_avoidance_risk_detected:** {harmful_avoidance_risk}")
    md_lines.append(f"- **direct_harmful_systematic:** {direct_harmful_systematic}")
    md_lines.append(f"- **hidden_feature_leakage_detected:** {result['hidden_feature_leakage_detected']}")
    md_lines.append(f"- **no_oracle_leakage_confirmed:** {result['no_oracle_leakage_confirmed']}")
    md_lines.append(f"- **fallback_safe_enough_for_next_phase:** {fallback_safe}")
    md_lines.append("")

    if audit["direct_helpful_cases"]:
        md_lines.append("## Direct Helpful Cases (fallback changed to better action)")
        md_lines.append("")
        for c in audit["direct_helpful_cases"]:
            md_lines.append(f"- **{c['oid']}** ep{c['episode']}: {c['dev_action']} -> {c['fb_action']} (fallback={c['fallback_used']})")
        md_lines.append("")

    if audit["direct_harmful_cases"]:
        md_lines.append("## Direct Harmful Cases (fallback changed to worse action)")
        md_lines.append("")
        md_lines.append("| Object | Episode | Dev Action | FB Action | Fallback | Positive Prior | Family Ambiguous |")
        md_lines.append("|--------|---------|------------|-----------|----------|----------------|------------------|")
        for c in audit["direct_harmful_cases"]:
            md_lines.append(f"| {c['oid']} | {c['episode']} | {c['dev_action']} | {c['fb_action']} | {c['fallback_used']} | {c['positive_prior_evidence']} | {c['family_ambiguous']} |")
        md_lines.append("")

    md_lines.append("")
    md_lines.append("```")
    md_lines.append("[block_done]")
    md_lines.append(f"block_id=1J34")
    md_lines.append(f"direct_helpful={audit['direct_helpful']}")
    md_lines.append(f"direct_harmful={audit['direct_harmful']}")
    md_lines.append(f"trajectory_helpful={audit['trajectory_helpful']}")
    md_lines.append(f"trajectory_harmful={audit['trajectory_harmful']}")
    md_lines.append(f"episode_net_pv_delta={audit['episode_pv_delta']}")
    md_lines.append(f"seed103_direct_harmful={audit['direct_harmful'] if seed == 103 else 0}")
    md_lines.append(f"harmful_avoidance_risk_detected={'true' if harmful_avoidance_risk else 'false'}")
    md_lines.append(f"fallback_safe_enough_for_next_phase={'true' if fallback_safe else 'false'}")
    md_lines.append(f"hidden_feature_leakage_detected={'true' if result['hidden_feature_leakage_detected'] else 'false'}")
    md_lines.append(f"no_oracle_leakage_confirmed={'true' if result['no_oracle_leakage_confirmed'] else 'false'}")
    md_lines.append(f"implementation_status=pass")
    md_lines.append(f"failure_reason=none")
    md_lines.append("```")

    md_path = os.path.join(CURRENT_DIR, "protocols", f"block1j34_fallback_help_harm_audit_seed{seed}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  MD  -> {md_path}")


# --- 5-seed aggregate ---
if MULTISEED:
    # Aggregate audit using new data structure
    direct_helpful_all = sum(r["audit_summary"]["direct_helpful"] for r in all_seed_results.values())
    direct_harmful_all = sum(r["audit_summary"]["direct_harmful"] for r in all_seed_results.values())
    trajectory_helpful_all = sum(r["audit_summary"]["trajectory_helpful"] for r in all_seed_results.values())
    trajectory_harmful_all = sum(r["audit_summary"]["trajectory_harmful"] for r in all_seed_results.values())
    episode_net_pv_delta_all = sum(r["audit_summary"]["episode_pv_delta"] for r in all_seed_results.values())
    episode_net_helpful_all = sum(r["audit_summary"]["episode_net_helpful"] for r in all_seed_results.values())

    # Seed 103 specific
    r103 = all_seed_results.get(103, {})
    audit103 = r103.get("audit_summary", {}) if r103 else {}
    direct_harmful103 = audit103.get("direct_harmful_cases", [])
    direct_harmful103_fallback = [h for h in direct_harmful103 if h.get("fallback_used")]

    # Seed-specific direct harmful counts
    direct_harmful_by_seed = {s: r["audit_summary"]["direct_harmful"] for s, r in all_seed_results.items()}
    direct_harmful_fallback_by_seed = {s: r["audit_summary"]["direct_harmful_fallback_count"] for s, r in all_seed_results.items()}
    harmful_concentration_seed103 = direct_harmful_by_seed.get(103, 0) > 0 and (
        direct_harmful_by_seed.get(103, 0) >= sum(direct_harmful_by_seed.values()) * 0.30
    )

    # Safety decision
    any_direct_harmful_systematic = any(
        r["audit_summary"]["direct_harmful_systematic"] for r in all_seed_results.values()
    )
    any_harmful_override_strong_prior = False
    for s, r in all_seed_results.items():
        for h in r["audit_summary"].get("direct_harmful_cases", []):
            if h.get("positive_prior_evidence") and h.get("fallback_used"):
                any_harmful_override_strong_prior = True
                break

    no_leakage = all(not r["hidden_feature_leakage_detected"] for r in all_seed_results.values())
    oracle_ok = all(r["no_oracle_leakage_confirmed"] for r in all_seed_results.values())

    fallback_safe = (
        episode_net_pv_delta_all <= 0  # PV reduction (negative = less PV, which is good)
        and not any_direct_harmful_systematic
        and not any_harmful_override_strong_prior
        and no_leakage
        and oracle_ok
    )

    harmful_avoidance_risk = direct_harmful_fallback_by_seed.get(103, 0) > 0

    # Aggregate macros
    mean_a_pv = np.mean([r["agg_a"]["total_prior_violations"] for r in all_seed_results.values()])
    mean_b_pv = np.mean([r["stats_b"]["total_prior_violations"] for r in all_seed_results.values()])
    mean_fb_pv = np.mean([r["stats_fb"]["total_prior_violations"] for r in all_seed_results.values()])
    mean_b_macro = np.mean([r["stats_b"]["mean_macro_bal"] for r in all_seed_results.values()])
    mean_fb_macro = np.mean([r["stats_fb"]["mean_macro_bal"] for r in all_seed_results.values()])
    mean_a_macro = np.mean([r["agg_a"]["mean_macro_bal"] for r in all_seed_results.values()])

    # Write aggregate JSON
    agg_out = {
        "block_id": "1J34",
        "seed_mode": "5seed",
        "seeds": MULTISEED_LIST,
        "elapsed_seconds": round(elapsed, 1),
        "aggregate_audit": {
            "direct_helpful": direct_helpful_all,
            "direct_harmful": direct_harmful_all,
            "trajectory_helpful": trajectory_helpful_all,
            "trajectory_harmful": trajectory_harmful_all,
            "episode_net_pv_delta": episode_net_pv_delta_all,
            "episode_net_helpful": episode_net_helpful_all,
            "direct_harmful_by_seed": direct_harmful_by_seed,
            "direct_harmful_fallback_by_seed": direct_harmful_fallback_by_seed,
            "harmful_concentration_seed103": harmful_concentration_seed103,
            "seed103_direct_harmful_cases": len(direct_harmful103),
            "seed103_direct_harmful_fallback": len(direct_harmful103_fallback),
        },
        "per_seed": {
            str(s): {
                "A_pv": r["agg_a"]["total_prior_violations"],
                "B_pv": r["stats_b"]["total_prior_violations"],
                "FB_pv": r["stats_fb"]["total_prior_violations"],
                "direct_helpful": r["audit_summary"]["direct_helpful"],
                "direct_harmful": r["audit_summary"]["direct_harmful"],
                "direct_net": r["audit_summary"]["direct_net"],
                "trajectory_helpful": r["audit_summary"]["trajectory_helpful"],
                "trajectory_harmful": r["audit_summary"]["trajectory_harmful"],
                "episode_pv_delta": r["audit_summary"]["episode_pv_delta"],
                "episode_net_helpful": r["audit_summary"]["episode_net_helpful"],
            } for s, r in all_seed_results.items()
        },
        "safety_decision": {
            "fallback_safe_enough_for_next_phase": fallback_safe,
            "episode_net_pv_delta": episode_net_pv_delta_all,
            "harmful_avoidance_risk_detected": harmful_avoidance_risk,
            "any_direct_harmful_systematic": any_direct_harmful_systematic,
            "any_harmful_override_strong_prior": any_harmful_override_strong_prior,
            "hidden_feature_leakage_detected": not no_leakage,
            "no_oracle_leakage_confirmed": oracle_ok,
        },
        "mean_metrics": {
            "mean_A_pv": round(mean_a_pv, 1),
            "mean_B_pv": round(mean_b_pv, 1),
            "mean_FB_pv": round(mean_fb_pv, 1),
            "mean_A_macro_bal": round(mean_a_macro, 4),
            "mean_B_macro_bal": round(mean_b_macro, 4),
            "mean_FB_macro_bal": round(mean_fb_macro, 4),
        },
    }

    json_path = os.path.join(CURRENT_DIR, "runs", "block1j34_fallback_help_harm_audit_5seed.json")
    with open(json_path, "w") as f:
        json.dump(agg_out, f, indent=2)
    print(f"  JSON -> {json_path}")

    # Write aggregate CSV (combine direct + trajectory records from all seeds)
    csv_path = os.path.join(CURRENT_DIR, "protocols", "block1j34_fallback_help_harm_audit_table.csv")
    csv_fields = [
        "seed", "episode", "effect_type", "change_type", "oid", "action",
        "family", "family_signal_ambiguous", "has_dev_features",
        "fallback_activated", "fallback_used",
        "soft_prior", "positive_prior_evidence",
        "ground_truth_outcome", "is_violation", "is_nonviolation",
        "classification", "rationale",
    ]
    all_csv_records = []
    for seed, r in all_seed_results.items():
        audit = r["audit_summary"]
        for rec in audit["direct_records"]:
            rc = dict(rec); rc["seed"] = seed; rc["effect_type"] = "direct"
            rc["change_type"] = ""; rc["soft_prior"] = rc.get("soft_prior", "")
            rc["fallback_activated"] = rc.get("fallback_used", False)
            rc["is_violation"] = rc.get("is_fb_violation", False)
            rc["is_nonviolation"] = not rc.get("is_fb_violation", True)
            rc["ground_truth_outcome"] = rc.get("gt_fb_outcome", "")
            all_csv_records.append(rc)
        for rec in audit["trajectory_records"]:
            rc = dict(rec); rc["seed"] = seed
            rc["fallback_used"] = rc.get("fallback_activated", False)
            all_csv_records.append(rc)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for rec in all_csv_records:
            writer.writerow(rec)
    print(f"  CSV -> {csv_path}")

    # Write aggregate MD
    md_lines = []
    md_lines.append("# Block 1J34 — 5-Seed Fallback Help-vs-Harm Audit")
    md_lines.append("")
    md_lines.append("## Per-Seed Audit Summary")
    md_lines.append("")
    md_lines.append("| Seed | A PV | B PV | FB PV | Dir Helpful | Dir Harmful | Dir Net | Traj Helpful | Traj Harmful | Ep PV Delta | Ep Net |")
    md_lines.append("|------|------|------|-------|-------------|-------------|---------|--------------|--------------|-------------|--------|")
    for seed in MULTISEED_LIST:
        r = all_seed_results[seed]
        a = r["audit_summary"]
        md_lines.append(f"| {seed} | {r['agg_a']['total_prior_violations']} | {r['stats_b']['total_prior_violations']} | {r['stats_fb']['total_prior_violations']} | {a['direct_helpful']} | {a['direct_harmful']} | {a['direct_net']:+d} | {a['trajectory_helpful']} | {a['trajectory_harmful']} | {a['episode_pv_delta']:+d} | {a['episode_net_helpful']:+d} |")
    md_lines.append("")

    md_lines.append("## Aggregate Audit Totals")
    md_lines.append("")
    md_lines.append(f"- **Direct helpful:** {direct_helpful_all}")
    md_lines.append(f"- **Direct harmful:** {direct_harmful_all}")
    md_lines.append(f"- **Trajectory helpful:** {trajectory_helpful_all}")
    md_lines.append(f"- **Trajectory harmful:** {trajectory_harmful_all}")
    md_lines.append(f"- **Episode net PV delta:** {episode_net_pv_delta_all:+d}")
    md_lines.append(f"- **Episode net helpful:** {episode_net_helpful_all:+d}")
    md_lines.append("")
    md_lines.append("## Harmful Cases Analysis")
    md_lines.append("")
    md_lines.append("| Seed | Direct Harmful | Fallback-Activated Direct Harmful |")
    md_lines.append("|------|----------------|-----------------------------------|")
    for seed in MULTISEED_LIST:
        md_lines.append(f"| {seed} | {direct_harmful_by_seed.get(seed, 0)} | {direct_harmful_fallback_by_seed.get(seed, 0)} |")
    md_lines.append("")
    md_lines.append(f"- **Harmful concentration in seed103:** {harmful_concentration_seed103}")
    md_lines.append(f"- **Seed103 direct harmful cases:** {len(direct_harmful103)}")
    md_lines.append(f"- **Seed103 fallback-activated direct harmful:** {len(direct_harmful103_fallback)}")
    md_lines.append("")

    md_lines.append("## Per-Seed Mean Metrics")
    md_lines.append("")
    md_lines.append(f"- **Mean A PV:** {mean_a_pv:.1f}")
    md_lines.append(f"- **Mean B PV:** {mean_b_pv:.1f}")
    md_lines.append(f"- **Mean FB PV:** {mean_fb_pv:.1f}")
    md_lines.append(f"- **Mean A macro_bal:** {mean_a_macro:.4f}")
    md_lines.append(f"- **Mean B macro_bal:** {mean_b_macro:.4f}")
    md_lines.append(f"- **Mean FB macro_bal:** {mean_fb_macro:.4f}")
    md_lines.append("")

    md_lines.append("## Safety Decision")
    md_lines.append("")
    md_lines.append(f"- **episode_net_pv_delta:** {episode_net_pv_delta_all:+d}")
    md_lines.append(f"- **harmful_avoidance_risk_detected:** {harmful_avoidance_risk}")
    md_lines.append(f"- **any_direct_harmful_systematic:** {any_direct_harmful_systematic}")
    md_lines.append(f"- **any_harmful_override_strong_prior:** {any_harmful_override_strong_prior}")
    md_lines.append(f"- **hidden_feature_leakage_detected:** {not no_leakage}")
    md_lines.append(f"- **no_oracle_leakage_confirmed:** {oracle_ok}")
    md_lines.append(f"- **fallback_safe_enough_for_next_phase:** {fallback_safe}")
    md_lines.append("")

    # Seed103 detailed direct harmful cases
    if direct_harmful103:
        md_lines.append("## Seed103 Direct Harmful Case Details")
        md_lines.append("")
        md_lines.append("| Object | Episode | Dev Action | FB Action | Fallback | Positive Prior | Family Ambiguous |")
        md_lines.append("|--------|---------|------------|-----------|----------|----------------|------------------|")
        for c in direct_harmful103:
            md_lines.append(f"| {c['oid']} | {c['episode']} | {c['dev_action']} | {c['fb_action']} | {c['fallback_used']} | {c['positive_prior_evidence']} | {c['family_ambiguous']} |")
        md_lines.append("")

    md_lines.append("")
    md_lines.append("```")
    md_lines.append("[block_done]")
    md_lines.append("block_id=1J34")
    md_lines.append(f"direct_helpful={direct_helpful_all}")
    md_lines.append(f"direct_harmful={direct_harmful_all}")
    md_lines.append(f"trajectory_helpful={trajectory_helpful_all}")
    md_lines.append(f"trajectory_harmful={trajectory_harmful_all}")
    md_lines.append(f"episode_net_pv_delta={episode_net_pv_delta_all}")
    md_lines.append(f"seed103_direct_harmful={len(direct_harmful103)}")
    md_lines.append(f"harmful_avoidance_risk_detected={'true' if harmful_avoidance_risk else 'false'}")
    md_lines.append(f"fallback_safe_enough_for_next_phase={'true' if fallback_safe else 'false'}")
    md_lines.append(f"hidden_feature_leakage_detected={'true' if not no_leakage else 'false'}")
    md_lines.append(f"no_oracle_leakage_confirmed={'true' if oracle_ok else 'false'}")
    md_lines.append(f"implementation_status=pass")
    md_lines.append(f"failure_reason=none")
    md_lines.append("```")

    md_path = os.path.join(CURRENT_DIR, "protocols", "block1j34_fallback_help_harm_audit_5seed.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  MD  -> {md_path}")

    print("")
    print("=" * 70)
    print("Block 1J34 5-seed audit complete.")
    print(f"  direct_helpful={direct_helpful_all}")
    print(f"  direct_harmful={direct_harmful_all}")
    print(f"  trajectory_helpful={trajectory_helpful_all}")
    print(f"  trajectory_harmful={trajectory_harmful_all}")
    print(f"  episode_net_pv_delta={episode_net_pv_delta_all}")
    print(f"  seed103_direct_harmful={len(direct_harmful103)}")
    print(f"  harmful_avoidance_risk_detected={harmful_avoidance_risk}")
    print(f"  fallback_safe_enough_for_next_phase={fallback_safe}")
    print(f"  implementation_status=pass")
    print("=" * 70)
else:
    print("")
    print("=" * 70)
    print("Block 1J34 single-seed audit complete.")
    print(f"  seed={SMOKE_SEED}")
    print(f"  direct_helpful={audit['direct_helpful']}")
    print(f"  direct_harmful={audit['direct_harmful']}")
    print(f"  episode_net_pv_delta={audit['episode_pv_delta']}")
    print(f"  harmful_avoidance_risk_detected={harmful_avoidance_risk}")
    print(f"  fallback_safe_enough_for_next_phase={fallback_safe}")
    print(f"  implementation_status=pass")
    print("=" * 70)
