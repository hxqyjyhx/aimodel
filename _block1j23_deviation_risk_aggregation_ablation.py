"""
Block 1J23 -- Deviation Risk Aggregation Ablation.

Tests whether the failure of 1J21/1J22 family-conditioned deviation memory
is caused by the current risk aggregation rule (sum all positive risk weights).

Variants:
  A_no_memory               -- Baseline
  B_sum_positive_current    -- Current sum-positive (reproduce 1J22)
  B_max_positive            -- Max positive weight only
  B_top2_positive_capped    -- Sum top 2 positive, capped at max_risk
  B_mean_positive           -- Average positive weights
  B_signed_sum_capped       -- Signed sum, clipped to [-max_risk, max_risk]
  B_signed_max_abs          -- Max absolute signed weight
  C_max_positive_permuted   -- Permuted control for max_positive
  C_signed_sum_permuted     -- Permuted control for signed_sum
  Family_only_control       -- Family-only rule from 1J22
  Oracle_dev_feature_rule   -- Oracle single-dev-feature from 1J22

Six diagnostic parts:
  Part 1: Aggregation formula comparison table
  Part 2: Candidate-level aggregation audit
  Part 3: Helpful vs harmful avoidance
  Part 4: Positive vs negative signal audit
  Part 5: Matching permuted controls
  Part 6: Boolean flag details
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
print("Block 1J23 -- Deviation Risk Aggregation Ablation")
print(f"  condition={COND['label']}  seed={SMOKE_SEED}  budget={BUDGET}")
print(f"  episodes={N_EPISODES}  risk_alpha={RISK_ALPHA}  max_risk={MAX_RISK}")
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

# =============================================================================
# 2. Test objects + Family-preserving deceptive objects
# =============================================================================
print("\n[2/10] Generating test objects + deceptive objects...")
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
print(f"  {len(test_oids)} test objects, {len(DECEPTIVE_OIDS)} deceptive")

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
print("\n[3/10] Building goal-conditioned soft prior table...")


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
# 5. FamilyConditionedRiskMemory (identical to 1J22)
# =============================================================================
print("\n[4/10] Defining FamilyConditionedRiskMemory...")

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

    def get_raw_risk_weight_signed(self, *key):
        """Return the signed raw log-ratio (unclipped, with shrinkage).
        This is the weight before positive-clipping, used by signed variants."""
        c = self.counts[key]
        v_with = c["violation_with"]
        nv_with = c["nonviolation_with"]
        v_without = c["violation_without"]
        nv_without = c["nonviolation_without"]
        actual_violations = (v_with + v_without) - 2.0
        if actual_violations < self.min_violation_support:
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
        support = total_v + total_nv - 4.0
        support = max(0.0, support)
        reliability = support / (support + self.alpha)
        return reliability * raw_lr

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
            if self.get_risk_weight(*key) > 1e-9:
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
# 6. Aggregation functions + Permuted memory builder
# =============================================================================
print("\n[5/10] Defining aggregation functions...")

def aggregate_risk_weights(weights, mode, max_risk=MAX_RISK):
    """Apply aggregation function to list of (dev_feat_name, weight) pairs.

    weights: list of (dev_feat_name, risk_weight) for present features.
    mode: one of sum_positive, max_positive, top2_positive_capped,
          mean_positive, signed_sum_capped, signed_max_abs.
    max_risk: cap value for capping modes.
    Returns (final_penalty, detail_dict).
    """
    detail = {"mode": mode, "weights": [(n, round(w, 6)) for n, w in weights]}

    if mode == "sum_positive":
        positive = [w for _, w in weights if w > 0]
        penalty = sum(positive)
        detail["positive_weights"] = [round(w, 6) for w in positive]
        detail["penalty"] = round(penalty, 6)
        return penalty, detail

    elif mode == "max_positive":
        positive = [w for _, w in weights if w > 0]
        penalty = max(positive) if positive else 0.0
        detail["positive_weights"] = [round(w, 6) for w in sorted(positive, reverse=True)]
        detail["penalty"] = round(penalty, 6)
        return penalty, detail

    elif mode == "top2_positive_capped":
        positive = sorted([w for _, w in weights if w > 0], reverse=True)
        total = sum(positive[:2])
        penalty = min(total, max_risk)
        detail["positive_weights"] = [round(w, 6) for w in positive]
        detail["top2_sum"] = round(total, 6)
        detail["cap"] = max_risk
        detail["penalty"] = round(penalty, 6)
        return penalty, detail

    elif mode == "mean_positive":
        positive = [w for _, w in weights if w > 0]
        penalty = sum(positive) / len(positive) if positive else 0.0
        detail["positive_weights"] = [round(w, 6) for w in positive]
        detail["n_positive"] = len(positive)
        detail["penalty"] = round(penalty, 6)
        return penalty, detail

    elif mode == "signed_sum_capped":
        total = sum(w for _, w in weights)
        penalty = max(-max_risk, min(max_risk, total))
        detail["signed_sum_raw"] = round(total, 6)
        detail["cap"] = max_risk
        detail["positive_weights"] = [round(w, 6) for _, w in weights if w > 0]
        detail["negative_weights"] = [round(w, 6) for _, w in weights if w < 0]
        detail["penalty"] = round(penalty, 6)
        return penalty, detail

    elif mode == "signed_max_abs":
        if not weights:
            detail["penalty"] = 0.0
            return 0.0, detail
        best = max(weights, key=lambda x: abs(x[1]))
        penalty = max(-max_risk, min(max_risk, best[1]))
        detail["selected_feature"] = best[0]
        detail["selected_raw_weight"] = round(best[1], 6)
        detail["cap"] = max_risk
        detail["penalty"] = round(penalty, 6)
        return penalty, detail

    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")


def build_permuted_memory(base_memory):
    """Create a permuted copy of the memory.

    Within each (goal, action, family) group, shuffle the risk weights across
    deviation features. This preserves the marginal distribution of risk weights
    per family/action group but destroys feature-specific signal.
    """
    permuted = base_memory.clone()

    # Group keys by (goal, action, family)
    groups = defaultdict(list)
    for key in base_memory.counts:
        if len(key) == 4:
            goal, action, family, dev_feat = key
            groups[(goal, action, family)].append(key)

    # Within each group, collect all the count dicts and shuffle them
    for grp_key, keys in groups.items():
        count_dicts = [dict(base_memory.counts[k]) for k in keys]
        rng_shuffle = random.Random(SMOKE_SEED + hash(grp_key) % 100000)
        rng_shuffle.shuffle(count_dicts)
        for k, cd in zip(keys, count_dicts):
            permuted.counts[k] = defaultdict(float, cd)

    permuted._risk_cache.clear()
    return permuted


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


# =============================================================================
# 7. Policy Classes
# =============================================================================
print("\n[6/10] Defining policy classes...")

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
            "raw_score": round(best_score, 6),
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


class DeviationRiskPolicy(SoftPriorOnlyPolicyV2):
    """Policy with parameterized risk aggregation.

    aggregation_mode: one of sum_positive, max_positive, top2_positive_capped,
                      mean_positive, signed_sum_capped, signed_max_abs.
    family_conditioned_memory: FamilyConditionedRiskMemory instance.
    max_risk: cap value.
    use_signed_weights: if True, use get_raw_risk_weight_signed() instead of
                        get_risk_weight() for the per-feature weights.
    """
    def __init__(self, instance_memory, rng, family_conditioned_memory,
                 target_probe_count=8, explore_fraction=0.0, risk_penalty_scale=1.0,
                 aggregation_mode="sum_positive", max_risk=MAX_RISK,
                 use_signed_weights=False):
        super().__init__(instance_memory, rng, target_probe_count, explore_fraction)
        self._fcrm = family_conditioned_memory
        self._risk_penalty_scale = risk_penalty_scale
        self.aggregation_mode = aggregation_mode
        self.max_risk = max_risk
        self.use_signed_weights = use_signed_weights
        self._variant = f"B_{aggregation_mode}"

    def _get_per_feature_weights(self, features, action):
        """Get list of (dev_feat, weight) for present deviation features."""
        goal = ACTION_TO_QUERY.get(action, "")
        family = detect_type_family(features)
        weights = []
        for dev_feat in INJECTED_DEVIATION_FEATURES:
            if features.get(dev_feat, False):
                if self.use_signed_weights:
                    w = self._fcrm.get_raw_risk_weight_signed(goal, action, family, dev_feat)
                else:
                    w = self._fcrm.get_risk_weight(goal, action, family, dev_feat)
                weights.append((dev_feat, w))
        return weights

    def get_total_risk_penalty(self, features, action):
        weights = self._get_per_feature_weights(features, action)
        penalty, _ = aggregate_risk_weights(weights, self.aggregation_mode, self.max_risk)
        return penalty

    def _compute_adjusted_score(self, features, action, norm_cost):
        base_prior = get_goal_soft_prior(features, action)
        risk_penalty = self.get_total_risk_penalty(features, action)
        scaled_penalty = risk_penalty * self._risk_penalty_scale
        # For signed variants, a negative penalty should increase score,
        # so we use exp(-scaled_penalty) consistently. Negative penalty -> exp(+|penalty|) > 1
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


class FamilyOnlyPolicy(SoftPriorOnlyPolicyV2):
    """Family-only risk memory policy (from 1J22)."""
    def __init__(self, instance_memory, rng, family_conditioned_memory,
                 target_probe_count=8, explore_fraction=0.0, risk_penalty_scale=1.0):
        super().__init__(instance_memory, rng, target_probe_count, explore_fraction)
        self._fcrm = family_conditioned_memory
        self._risk_penalty_scale = risk_penalty_scale
        self._variant = "family_only_control"

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


class OracleDevFeaturePolicy(SoftPriorOnlyPolicyV2):
    """Oracle single-best-dev-feature rule from 1J22."""
    def __init__(self, instance_memory, rng, selected_dev_features,
                 target_probe_count=8, explore_fraction=0.0):
        super().__init__(instance_memory, rng, target_probe_count, explore_fraction)
        self._selected_dev_features = selected_dev_features
        self._variant = "oracle_dev_feature"

    def _compute_adjusted_score(self, features, action, norm_cost):
        base_prior = get_goal_soft_prior(features, action)
        family = detect_type_family(features)
        key_grp = (action, family)
        risk_penalty = 0.0
        if key_grp in self._selected_dev_features:
            best_dev, score = self._selected_dev_features[key_grp]
            if features.get(best_dev, False):
                risk_penalty = 2.0
        raw_score = base_prior * math.exp(-risk_penalty) - COST_WEIGHT * norm_cost
        final_score = max(EXPLORATION_FLOOR, raw_score)
        return final_score, raw_score, base_prior, risk_penalty, risk_penalty

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
# 8. C15b reference + offline oracle memory
# =============================================================================
print("\n[7/10] Running C15b reference + building offline oracle...")

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
print(f"  C15b macro_bal={c15b_macro_bal:.4f}  target_probe_count={c15b_probed_n}")

# Build offline oracle memory (family_dev mode)
fcrm_offline = build_offline_oracle_memory(test_objects_deceptive, key_mode="family_dev")
fcrm_family_only = build_offline_oracle_memory(test_objects_deceptive, key_mode="family_only")
fcrm_permuted = build_permuted_memory(fcrm_offline)

# Build informativeness data for oracle dev feature rule
info_rows = []
for key in fcrm_offline.counts:
    if len(key) != 4:
        continue
    goal, action, family, dev_feat = key
    n_with = 0; n_without = 0
    violations_with = 0; nonviolations_with = 0
    violations_without = 0; nonviolations_without = 0
    for oid, obj in test_objects_deceptive.items():
        features = obj.get("visible_features", {})
        obj_family = detect_type_family(features)
        has_dev = features.get(dev_feat, False)
        sp = get_goal_soft_prior(features, action)
        is_eligible = sp >= PRIOR_VIOLATION_THRESHOLD
        if obj_family != family: continue
        if not is_eligible: continue
        outcome = get_ground_truth_outcome(obj, action)
        is_violation = outcome <= 0.01
        if has_dev:
            n_with += 1
            if is_violation: violations_with += 1
            else: nonviolations_with += 1
        else:
            n_without += 1
            if is_violation: violations_without += 1
            else: nonviolations_without += 1
    violation_rate_with = violations_with / max(n_with, 1)
    violation_rate_without = violations_without / max(n_without, 1)
    risk_rate_delta = violation_rate_with - violation_rate_without
    info_rows.append({
        "goal": goal, "action": action, "family": family, "dev_feat": dev_feat,
        "risk_rate_delta": risk_rate_delta,
        "violations_with_feature": violations_with,
    })

# Select best dev feature per (action, family) for oracle rule
best_dev_feature_per_group = {}
for r in info_rows:
    group = (r["action"], r["family"])
    if group not in best_dev_feature_per_group:
        best_dev_feature_per_group[group] = []
    best_dev_feature_per_group[group].append((r["dev_feat"], r["risk_rate_delta"], r["violations_with_feature"]))

selected_dev_features = {}
for group, candidates in best_dev_feature_per_group.items():
    action, family = group
    best = None; best_score = -999
    for dev_feat, delta, v_count in candidates:
        if v_count >= MIN_VIOLATION_SUPPORT:
            if abs(delta) > best_score:
                best_score = abs(delta)
                best = dev_feat
    if best:
        selected_dev_features[(action, family)] = (best, best_score)

# Count nonzero weights in offline memory
offline_nonzero = fcrm_offline.get_num_nonzero_keys()
print(f"  Offline oracle: {fcrm_offline.eligible_updates} eligible, "
      f"{fcrm_offline.violation_updates} violations, {offline_nonzero} nonzero keys")
print(f"  Selected {len(selected_dev_features)} best dev features for oracle rule")

# =============================================================================
# 9. Run all variants
# =============================================================================
print("\n[8/10] Running all variants...")

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
    """Count nonviolation probes that A made but variant didn't."""
    count = 0
    a_set = {}
    for ep_a in ep_a_list:
        for d in ep_a["probe_details"]:
            if d.get("is_nonviolation"):
                a_set[(d["oid"], d["action"])] = d
    var_set = set()
    for ep_v in ep_var_list:
        for d in ep_v["probe_details"]:
            var_set.add((d["oid"], d["action"]))
    for key in a_set:
        if key not in var_set:
            count += 1
    return count


def compute_variant_stats(all_episodes_var, all_episodes_a, variant_label,
                          aggregation_formula, diagnostic_oracle_rule=False,
                          diagnostic_oracle_evidence=False):
    """Compute standard stats for a variant."""
    agg_v = aggregate_episodes(all_episodes_var)
    eff_v = count_eff(all_episodes_var)
    eff_a = count_eff(all_episodes_a)
    avoided_eff = eff_a - eff_v
    avoided_pv = avoided_pv_count(all_episodes_a, all_episodes_var)
    avoided_nv = avoided_nv_count(all_episodes_a, all_episodes_var)

    # Helpful vs harmful avoidance
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
        "aggregation_formula": aggregation_formula,
        "diagnostic_oracle_rule": diagnostic_oracle_rule,
        "diagnostic_oracle_evidence": diagnostic_oracle_evidence,
        "agg": agg_v,
        "episodes": all_episodes_var,
    }


# --- Run A_no_memory baseline ---
print("  Running A_no_memory...")
all_episodes_a = []
pv_history_a = set()

for ep in range(N_EPISODES):
    ep_seed = RNG_SEED_BASE + 2 + ep * 100
    ep_rng = random.Random(ep_seed)
    test_oids_local = sorted(test_objects_deceptive.keys())
    ep_positions = MiniMCSimulatorTruth.assign_positions(
        test_oids_local, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, ep_rng)
    im_a = im_base.clone()
    env_a = MiniMCEnvironment(test_objects_deceptive, dict(ep_positions), initial_budget=BUDGET)
    policy_a = SoftPriorOnlyPolicyV2(im_a, ep_rng, target_probe_count=c15b_probed_n)
    ep_a, pv_history_a = run_episode_and_extract(
        test_objects_deceptive, query_gt_deceptive, policy_a, env_a, ep,
        prior_violation_history=pv_history_a)
    all_episodes_a.append(ep_a)

agg_a = aggregate_episodes(all_episodes_a)
print(f"    A_no_memory: total_pv={agg_a['total_prior_violations']}, "
      f"macro_bal={agg_a['mean_macro_bal']:.4f}")


def run_variant_episodes(memory, policy_class, policy_kwargs, label):
    """Run 5 episodes for a variant."""
    all_eps = []
    pv_hist = set()
    for ep in range(N_EPISODES):
        ep_seed = RNG_SEED_BASE + policy_kwargs.get("seed_offset", 0) + ep * 100
        ep_rng = random.Random(ep_seed)
        test_oids_local = sorted(test_objects_deceptive.keys())
        ep_positions = MiniMCSimulatorTruth.assign_positions(
            test_oids_local, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, ep_rng)
        im_ep = im_base.clone()
        env_ep = MiniMCEnvironment(test_objects_deceptive, dict(ep_positions), initial_budget=BUDGET)
        if policy_class == DeviationRiskPolicy:
            policy_ep = DeviationRiskPolicy(
                im_ep, ep_rng, memory.clone(),
                target_probe_count=c15b_probed_n,
                aggregation_mode=policy_kwargs["aggregation_mode"],
                max_risk=policy_kwargs.get("max_risk", MAX_RISK),
                use_signed_weights=policy_kwargs.get("use_signed_weights", False),
            )
        elif policy_class == FamilyOnlyPolicy:
            policy_ep = FamilyOnlyPolicy(im_ep, ep_rng, memory.clone(), target_probe_count=c15b_probed_n)
        elif policy_class == OracleDevFeaturePolicy:
            policy_ep = OracleDevFeaturePolicy(
                im_ep, ep_rng, selected_dev_features, target_probe_count=c15b_probed_n)
        else:
            raise ValueError(f"Unknown policy class: {policy_class}")
        ep_data, pv_hist = run_episode_and_extract(
            test_objects_deceptive, query_gt_deceptive, policy_ep, env_ep, ep,
            prior_violation_history=pv_hist)
        all_eps.append(ep_data)
    return all_eps


# Define all variants to run
VARIANTS = [
    {
        "name": "B_sum_positive_current",
        "policy_class": DeviationRiskPolicy,
        "memory": fcrm_offline,
        "kwargs": {"aggregation_mode": "sum_positive", "seed_offset": 10},
        "formula": "sum(max(0, w_i) for each present dev_feat i)",
        "diagnostic_oracle_rule": False,
        "diagnostic_oracle_evidence": False,
    },
    {
        "name": "B_max_positive",
        "policy_class": DeviationRiskPolicy,
        "memory": fcrm_offline,
        "kwargs": {"aggregation_mode": "max_positive", "seed_offset": 20},
        "formula": "max(max(0, w_i) for each present dev_feat i)",
        "diagnostic_oracle_rule": False,
        "diagnostic_oracle_evidence": False,
    },
    {
        "name": "B_top2_positive_capped",
        "policy_class": DeviationRiskPolicy,
        "memory": fcrm_offline,
        "kwargs": {"aggregation_mode": "top2_positive_capped", "seed_offset": 30,
                   "max_risk": MAX_RISK},
        "formula": "min(max_risk, sum(top_2_positive_weights))",
        "diagnostic_oracle_rule": False,
        "diagnostic_oracle_evidence": False,
    },
    {
        "name": "B_mean_positive",
        "policy_class": DeviationRiskPolicy,
        "memory": fcrm_offline,
        "kwargs": {"aggregation_mode": "mean_positive", "seed_offset": 40},
        "formula": "mean(max(0, w_i) for each present dev_feat i)",
        "diagnostic_oracle_rule": False,
        "diagnostic_oracle_evidence": False,
    },
    {
        "name": "B_signed_sum_capped",
        "policy_class": DeviationRiskPolicy,
        "memory": fcrm_offline,
        "kwargs": {"aggregation_mode": "signed_sum_capped", "seed_offset": 50,
                   "max_risk": MAX_RISK, "use_signed_weights": True},
        "formula": "clip(sum(signed_w_i), -max_risk, +max_risk)",
        "diagnostic_oracle_rule": False,
        "diagnostic_oracle_evidence": False,
    },
    {
        "name": "B_signed_max_abs",
        "policy_class": DeviationRiskPolicy,
        "memory": fcrm_offline,
        "kwargs": {"aggregation_mode": "signed_max_abs", "seed_offset": 60,
                   "max_risk": MAX_RISK, "use_signed_weights": True},
        "formula": "clip(argmax(|w_i|).signed_weight, -max_risk, +max_risk)",
        "diagnostic_oracle_rule": False,
        "diagnostic_oracle_evidence": False,
    },
    {
        "name": "C_max_positive_permuted",
        "policy_class": DeviationRiskPolicy,
        "memory": fcrm_permuted,
        "kwargs": {"aggregation_mode": "max_positive", "seed_offset": 70},
        "formula": "max(max(0, w_i) for each present dev_feat i) [PERMUTED within (goal,action,family)]",
        "diagnostic_oracle_rule": False,
        "diagnostic_oracle_evidence": False,
    },
    {
        "name": "C_signed_sum_permuted",
        "policy_class": DeviationRiskPolicy,
        "memory": fcrm_permuted,
        "kwargs": {"aggregation_mode": "signed_sum_capped", "seed_offset": 80,
                   "max_risk": MAX_RISK, "use_signed_weights": True},
        "formula": "clip(sum(signed_w_i), -max_risk, +max_risk) [PERMUTED within (goal,action,family)]",
        "diagnostic_oracle_rule": False,
        "diagnostic_oracle_evidence": False,
    },
    {
        "name": "Family_only_control",
        "policy_class": FamilyOnlyPolicy,
        "memory": fcrm_family_only,
        "kwargs": {"seed_offset": 90},
        "formula": "sum(risk_weight(goal, action, family)) across families",
        "diagnostic_oracle_rule": False,
        "diagnostic_oracle_evidence": False,
    },
    {
        "name": "Oracle_dev_feature_rule",
        "policy_class": OracleDevFeaturePolicy,
        "memory": None,
        "kwargs": {"seed_offset": 100},
        "formula": "2.0 if best_dev_feature_per(action,family) present else 0.0",
        "diagnostic_oracle_rule": True,
        "diagnostic_oracle_evidence": True,
    },
]

# Run all variants
all_results = {}
for vdef in VARIANTS:
    name = vdef["name"]
    print(f"  Running {name}...")
    if vdef["policy_class"] == OracleDevFeaturePolicy:
        all_eps = run_variant_episodes(None, OracleDevFeaturePolicy, vdef["kwargs"], name)
    elif vdef["policy_class"] == FamilyOnlyPolicy:
        all_eps = run_variant_episodes(vdef["memory"], FamilyOnlyPolicy, vdef["kwargs"], name)
    else:
        all_eps = run_variant_episodes(vdef["memory"], DeviationRiskPolicy, vdef["kwargs"], name)

    stats = compute_variant_stats(
        all_eps, all_episodes_a, name,
        vdef["formula"],
        diagnostic_oracle_rule=vdef["diagnostic_oracle_rule"],
        diagnostic_oracle_evidence=vdef["diagnostic_oracle_evidence"],
    )
    all_results[name] = stats
    print(f"    {name}: total_pv={stats['total_prior_violations']}, "
          f"macro_bal={stats['mean_macro_bal']:.4f}, delta_vs_A={stats['macro_delta_vs_A']:.4f}")

# =============================================================================
# 10. Diagnostics
# =============================================================================
print("\n[9/10] Computing diagnostics...")

# --- Part 1: Aggregation formula comparison table ---
part1_rows = []
for vdef in VARIANTS:
    name = vdef["name"]
    stats = all_results[name]
    part1_rows.append({
        "variant": name,
        "total_prior_violations": stats["total_prior_violations"],
        "deceptive_object_prior_violations": stats["deceptive_object_prior_violations"],
        "normal_object_prior_violations": stats["normal_object_prior_violations"],
        "mean_macro_bal": stats["mean_macro_bal"],
        "macro_delta_vs_A": stats["macro_delta_vs_A"],
        "effective_probe_count": stats["effective_probe_count"],
        "avoided_effective_probe_count": stats["avoided_effective_probe_count"],
        "avoided_prior_violation_count": stats["avoided_prior_violation_count"],
        "avoided_nonviolation_count": stats["avoided_nonviolation_count"],
        "helpful_avoidance_count": stats["helpful_avoidance_count"],
        "harmful_avoidance_count": stats["harmful_avoidance_count"],
        "hard_exclusion_used": stats["hard_exclusion_used"],
        "aggregation_formula": stats["aggregation_formula"],
        "diagnostic_oracle_rule": stats["diagnostic_oracle_rule"],
        "diagnostic_oracle_evidence": stats["diagnostic_oracle_evidence"],
    })

# --- Part 2: Candidate-level aggregation audit ---
# Build a comprehensive per-candidate audit using all unique probed (oid, action) pairs
candidate_audit_rows = []

# Collect all unique candidates across all variants
all_probe_keys = set()
for ep_data in all_episodes_a:
    for d in ep_data["probe_details"]:
        all_probe_keys.add((d["oid"], d["action"]))
for name, stats in all_results.items():
    for ep_data in stats["episodes"]:
        for d in ep_data["probe_details"]:
            all_probe_keys.add((d["oid"], d["action"]))

for oid, action in sorted(all_probe_keys):
    obj = test_objects_deceptive.get(oid, {})
    features = obj.get("visible_features", {})
    family = detect_type_family(features)
    goal = ACTION_TO_QUERY.get(action, "")
    injected_present = [f for f in INJECTED_DEVIATION_FEATURES if features.get(f, False)]

    # Compute all aggregation values for this candidate
    weights_positive = []
    weights_signed = []
    triggered_keys = []
    for dev_feat in INJECTED_DEVIATION_FEATURES:
        if features.get(dev_feat, False):
            w_pos = fcrm_offline.get_risk_weight(goal, action, family, dev_feat)
            w_signed = fcrm_offline.get_raw_risk_weight_signed(goal, action, family, dev_feat)
            weights_positive.append((dev_feat, w_pos))
            weights_signed.append((dev_feat, w_signed))
            if abs(w_pos) > 1e-9 or abs(w_signed) > 1e-9:
                triggered_keys.append({
                    "dev_feature": dev_feat,
                    "risk_weight_positive": round(w_pos, 6),
                    "risk_weight_signed": round(w_signed, 6),
                })

    sum_pos_penalty = sum(w for _, w in weights_positive if w > 0)
    max_pos_penalty = max([w for _, w in weights_positive if w > 0]) if any(w > 0 for _, w in weights_positive) else 0.0
    positive_sorted = sorted([w for _, w in weights_positive if w > 0], reverse=True)
    top2_pos_penalty = min(sum(positive_sorted[:2]), MAX_RISK)
    mean_pos_penalty = sum_pos_penalty / len([w for _, w in weights_positive if w > 0]) if any(w > 0 for _, w in weights_positive) else 0.0
    signed_sum_penalty_raw = sum(w for _, w in weights_signed)
    signed_sum_penalty = max(-MAX_RISK, min(MAX_RISK, signed_sum_penalty_raw))
    signed_max_abs_penalty = 0.0
    if weights_signed:
        best_signed = max(weights_signed, key=lambda x: abs(x[1]))
        signed_max_abs_penalty = max(-MAX_RISK, min(MAX_RISK, best_signed[1]))

    # Check which variants selected this (oid, action)
    selected_by_variant = {}
    selected_by_variant["A_no_memory"] = any(
        d["oid"] == oid and d["action"] == action
        for ep_data in all_episodes_a for d in ep_data["probe_details"])
    for name in all_results:
        selected_by_variant[name] = any(
            d["oid"] == oid and d["action"] == action
            for ep_data in all_results[name]["episodes"]
            for d in ep_data["probe_details"])

    # Find prior violation status from ground truth
    outcome = get_ground_truth_outcome(obj, action)
    sp = get_goal_soft_prior(features, action)
    is_eligible = sp >= PRIOR_VIOLATION_THRESHOLD
    is_pv = is_eligible and outcome <= 0.01
    is_nv = is_eligible and outcome > 0.01

    # Get probe utility signal from A
    utility_signal = "unknown"
    for ep_data in all_episodes_a:
        for d in ep_data["probe_details"]:
            if d["oid"] == oid and d["action"] == action:
                utility_signal = d.get("probe_utility_signal", "unknown")

    candidate_audit_rows.append({
        "object_id": oid,
        "goal": goal,
        "action": action,
        "inferred_family": family,
        "injected_deviation_features": injected_present,
        "triggered_risk_keys": triggered_keys,
        "triggered_positive_weights": [round(w, 6) for _, w in weights_positive if w > 0],
        "triggered_negative_weights": [round(w, 6) for _, w in weights_signed if w < 0],
        "sum_positive_penalty": round(sum_pos_penalty, 6),
        "max_positive_penalty": round(max_pos_penalty, 6),
        "top2_positive_penalty": round(top2_pos_penalty, 6),
        "mean_positive_penalty": round(mean_pos_penalty, 6),
        "signed_sum_penalty": round(signed_sum_penalty, 6),
        "signed_max_abs_penalty": round(signed_max_abs_penalty, 6),
        "selected_by_variant": selected_by_variant,
        "prior_violation": is_pv,
        "prior_nonviolation": is_nv,
        "probe_utility_signal": utility_signal,
    })

part2 = {
    "total_candidates_audited": len(candidate_audit_rows),
    "candidates": candidate_audit_rows,
}

# --- Part 3: Helpful vs harmful avoidance ---
part3_rows = []
for vdef in VARIANTS:
    name = vdef["name"]
    stats = all_results[name]
    net = stats["helpful_avoidance_count"] - stats["harmful_avoidance_count"]

    # Count avoided true positive probes
    a_probe_map = {}
    for ep_a in all_episodes_a:
        for d in ep_a["probe_details"]:
            a_probe_map[(d["oid"], d["action"])] = d
    var_probe_map = {}
    for ep_v in stats["episodes"]:
        for d in ep_v["probe_details"]:
            var_probe_map[(d["oid"], d["action"])] = d

    avoided_keys = set(a_probe_map.keys()) - set(var_probe_map.keys())
    avoided_tp = sum(1 for k in avoided_keys
                     if a_probe_map[k].get("probe_utility_signal") == "effective"
                     and a_probe_map[k].get("is_violation"))

    part3_rows.append({
        "variant": name,
        "avoided_prior_violation_count": stats["avoided_prior_violation_count"],
        "avoided_nonviolation_count": stats["avoided_nonviolation_count"],
        "avoided_effective_probe_count": stats["avoided_effective_probe_count"],
        "avoided_true_positive_probe_count": avoided_tp,
        "helpful_avoidance_count": stats["helpful_avoidance_count"],
        "harmful_avoidance_count": stats["harmful_avoidance_count"],
        "net_helpful_minus_harmful": net,
    })

part3 = {"per_variant": part3_rows}

# --- Part 4: Positive vs negative signal audit ---
signed_keys_audit = []
for key in fcrm_offline.counts:
    if len(key) != 4:
        continue
    w_signed = fcrm_offline.get_raw_risk_weight_signed(*key)
    w_positive = fcrm_offline.get_risk_weight(*key)
    signed_keys_audit.append({
        "key": list(key),
        "signed_weight": round(w_signed, 6),
        "positive_weight": round(w_positive, 6),
        "is_positive": w_signed > 1e-9,
        "is_negative": w_signed < -1e-9,
        "is_zero": abs(w_signed) < 1e-9,
    })

num_positive_keys = sum(1 for k in signed_keys_audit if k["is_positive"])
num_negative_keys = sum(1 for k in signed_keys_audit if k["is_negative"])
total_abs_positive = sum(k["signed_weight"] for k in signed_keys_audit if k["is_positive"])
total_abs_negative = sum(abs(k["signed_weight"]) for k in signed_keys_audit if k["is_negative"])

top_positive = sorted([k for k in signed_keys_audit if k["is_positive"]],
                       key=lambda x: x["signed_weight"], reverse=True)[:10]
top_negative = sorted([k for k in signed_keys_audit if k["is_negative"]],
                       key=lambda x: x["signed_weight"])[:10]

# Count candidates with protective (negative) signal
candidates_with_protective = 0
candidates_where_negative_changed = 0
for cand in candidate_audit_rows:
    if cand["triggered_negative_weights"]:
        candidates_with_protective += 1
        # Check if signed_sum_penalty differs from sum_positive_penalty
        if abs(cand["signed_sum_penalty"] - cand["sum_positive_penalty"]) > 1e-6:
            candidates_where_negative_changed += 1

part4 = {
    "num_positive_weight_keys": num_positive_keys,
    "num_negative_weight_keys": num_negative_keys,
    "num_zero_weight_keys": sum(1 for k in signed_keys_audit if k["is_zero"]),
    "total_abs_positive_weight": round(total_abs_positive, 6),
    "total_abs_negative_weight": round(total_abs_negative, 6),
    "top_10_positive_keys": top_positive,
    "top_10_negative_keys": top_negative,
    "number_of_candidates_with_protective_signal": candidates_with_protective,
    "number_of_candidates_where_negative_signal_changed_selection": candidates_where_negative_changed,
    "all_signed_keys": signed_keys_audit,
}

# --- Part 5: Matching permuted controls ---
part5_rows = []
permuted_pairs = [
    ("B_max_positive", "C_max_positive_permuted"),
    ("B_signed_sum_capped", "C_signed_sum_permuted"),
]
for real_name, perm_name in permuted_pairs:
    real = all_results[real_name]
    perm = all_results[perm_name]
    real_beats = real["total_prior_violations"] < perm["total_prior_violations"]
    part5_rows.append({
        "real_variant_name": real_name,
        "permuted_variant_name": perm_name,
        "real_total_pv": real["total_prior_violations"],
        "permuted_total_pv": perm["total_prior_violations"],
        "real_macro_bal": real["mean_macro_bal"],
        "permuted_macro_bal": perm["mean_macro_bal"],
        "real_beats_permuted": real_beats,
    })

part5 = {"permuted_controls": part5_rows}

# --- Part 6: Boolean flags ---
print("\n[10/10] Computing boolean flags...")

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

# Compute flags
a_total_pv = agg_a["total_prior_violations"]
sum_pos_pv = all_results["B_sum_positive_current"]["total_prior_violations"]
max_pos_pv = all_results["B_max_positive"]["total_prior_violations"]
top2_pv = all_results["B_top2_positive_capped"]["total_prior_violations"]
mean_pos_pv = all_results["B_mean_positive"]["total_prior_violations"]
signed_sum_pv = all_results["B_signed_sum_capped"]["total_prior_violations"]
signed_max_abs_pv = all_results["B_signed_max_abs"]["total_prior_violations"]
oracle_pv = all_results["Oracle_dev_feature_rule"]["total_prior_violations"]
family_only_pv = all_results["Family_only_control"]["total_prior_violations"]
max_perm_pv = all_results["C_max_positive_permuted"]["total_prior_violations"]
signed_perm_pv = all_results["C_signed_sum_permuted"]["total_prior_violations"]

current_sum_reproduced_1j22 = sum_pos_pv == a_total_pv or sum_pos_pv >= a_total_pv
max_positive_beats_current_sum = max_pos_pv < sum_pos_pv
max_positive_beats_A = max_pos_pv < a_total_pv
top2_positive_beats_current_sum = top2_pv < sum_pos_pv
signed_sum_beats_current_sum = signed_sum_pv < sum_pos_pv
signed_sum_beats_A = signed_sum_pv < a_total_pv

signed_variant_uses_protective_signal = (
    signed_sum_pv != sum_pos_pv or signed_max_abs_pv != sum_pos_pv
)

dev_aggregators = ["B_sum_positive_current", "B_max_positive", "B_top2_positive_capped",
                   "B_mean_positive", "B_signed_sum_capped", "B_signed_max_abs"]
any_dev_aggregation_beats_A = any(
    all_results[name]["total_prior_violations"] < a_total_pv
    for name in dev_aggregators)

any_dev_aggregation_beats_matching_permuted = (
    (all_results["B_max_positive"]["total_prior_violations"] < max_perm_pv) or
    (all_results["B_signed_sum_capped"]["total_prior_violations"] < signed_perm_pv)
)

any_dev_aggregation_beats_oracle_dev_feature = any(
    all_results[name]["total_prior_violations"] < oracle_pv
    for name in dev_aggregators)

family_only_still_dominates_all_dev_aggregators = all(
    family_only_pv <= all_results[name]["total_prior_violations"]
    for name in dev_aggregators)

counterproductive_weighting_reduced = any(
    all_results[name]["harmful_avoidance_count"] <= all_results[name]["helpful_avoidance_count"]
    for name in dev_aggregators)

hard_exclusion_used = False

# Determine best non-oracle dev aggregator
best_dev_name = None
best_dev_pv = 999
for name in dev_aggregators:
    pv = all_results[name]["total_prior_violations"]
    if pv < best_dev_pv:
        best_dev_pv = pv
        best_dev_name = name

best_dev_macro_delta = all_results[best_dev_name]["macro_delta_vs_A"] if best_dev_name else 0.0

# Determine implementation status and failure_reason
if any_dev_aggregation_beats_A:
    implementation_status = "pass"
    failure_reason = "none"
elif any_dev_aggregation_beats_matching_permuted:
    implementation_status = "partial"
    failure_reason = "dev_aggregation_beats_permuted_but_not_baseline"
else:
    implementation_status = "partial"
    failure_reason = "no_aggregation_beats_A"

boolean_flag_details = {
    "current_sum_reproduced_1j22": {
        "value": current_sum_reproduced_1j22,
        "rule": "B_sum_positive_current PV >= A PV (same as 1J22 finding)",
        "supporting_values": {"sum_positive_pv": sum_pos_pv, "A_pv": a_total_pv},
    },
    "max_positive_beats_current_sum": {
        "value": max_positive_beats_current_sum,
        "rule": "B_max_positive_total_pv < B_sum_positive_current_total_pv",
        "supporting_values": {"max_positive_pv": max_pos_pv, "sum_positive_pv": sum_pos_pv},
    },
    "max_positive_beats_A": {
        "value": max_positive_beats_A,
        "rule": "B_max_positive_total_pv < A_total_pv",
        "supporting_values": {"max_positive_pv": max_pos_pv, "A_pv": a_total_pv},
    },
    "top2_positive_beats_current_sum": {
        "value": top2_positive_beats_current_sum,
        "rule": "B_top2_positive_capped_total_pv < B_sum_positive_current_total_pv",
        "supporting_values": {"top2_pv": top2_pv, "sum_positive_pv": sum_pos_pv},
    },
    "signed_sum_beats_current_sum": {
        "value": signed_sum_beats_current_sum,
        "rule": "B_signed_sum_capped_total_pv < B_sum_positive_current_total_pv",
        "supporting_values": {"signed_sum_pv": signed_sum_pv, "sum_positive_pv": sum_pos_pv},
    },
    "signed_sum_beats_A": {
        "value": signed_sum_beats_A,
        "rule": "B_signed_sum_capped_total_pv < A_total_pv",
        "supporting_values": {"signed_sum_pv": signed_sum_pv, "A_pv": a_total_pv},
    },
    "signed_variant_uses_protective_signal": {
        "value": signed_variant_uses_protective_signal,
        "rule": "signed_sum or signed_max_abs produces different result than sum_positive_only",
        "supporting_values": {
            "signed_sum_pv": signed_sum_pv,
            "sum_positive_pv": sum_pos_pv,
            "signed_max_abs_pv": signed_max_abs_pv,
        },
    },
    "any_dev_aggregation_beats_A": {
        "value": any_dev_aggregation_beats_A,
        "rule": "at least one non-oracle dev aggregation variant has PV < A_total_pv",
        "supporting_values": {name: all_results[name]["total_prior_violations"] for name in dev_aggregators},
    },
    "any_dev_aggregation_beats_matching_permuted": {
        "value": any_dev_aggregation_beats_matching_permuted,
        "rule": "at least one real variant beats its permuted control",
        "supporting_values": {
            "max_positive_vs_permuted": f"{max_pos_pv} vs {max_perm_pv}",
            "signed_sum_vs_permuted": f"{signed_sum_pv} vs {signed_perm_pv}",
        },
    },
    "any_dev_aggregation_beats_oracle_dev_feature": {
        "value": any_dev_aggregation_beats_oracle_dev_feature,
        "rule": "at least one dev aggregation variant has PV < oracle_dev_feature PV",
        "supporting_values": {"oracle_dev_feature_pv": oracle_pv},
    },
    "family_only_still_dominates_all_dev_aggregators": {
        "value": family_only_still_dominates_all_dev_aggregators,
        "rule": "family_only PV <= every dev aggregation variant PV",
        "supporting_values": {
            "family_only_pv": family_only_pv,
            "dev_aggregator_pvs": {name: all_results[name]["total_prior_violations"] for name in dev_aggregators},
        },
    },
    "counterproductive_weighting_reduced": {
        "value": counterproductive_weighting_reduced,
        "rule": "at least one variant has helpful >= harmful avoidances",
        "supporting_values": {
            name: f"helpful={all_results[name]['helpful_avoidance_count']}, harmful={all_results[name]['harmful_avoidance_count']}"
            for name in dev_aggregators
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
            "total_deceptive": len(DECEPTIVE_OIDS),
            "preserved": sum(1 for c in deceptive_family_preservation_checks if c["preserved"]),
        },
    },
    "implementation_status": {
        "value": implementation_status,
        "rule": "pass if any dev aggregation beats A; partial otherwise",
        "supporting_values": {"any_dev_aggregation_beats_A": any_dev_aggregation_beats_A},
    },
    "failure_reason": {
        "value": failure_reason,
        "rule": "derived from aggregation comparison results",
        "supporting_values": {},
    },
}

# =============================================================================
# 11. Output
# =============================================================================
elapsed = time.time() - t0
print(f"\n  elapsed={elapsed:.1f}s  writing outputs...")

output = {
    "block_id": "1J23",
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
    "c15b_target_probe_count": c15b_probed_n,
    "A_no_memory": {
        "agg": agg_a,
        "episodes": all_episodes_a,
    },
    "variants": {name: {
        "agg": stats["agg"],
        "episodes": stats["episodes"],
        "stats": {k: v for k, v in stats.items() if k not in ("agg", "episodes")},
    } for name, stats in all_results.items()},
    "part1_aggregation_comparison": part1_rows,
    "part2_candidate_audit": part2,
    "part3_helpful_harmful": part3,
    "part4_signed_signal_audit": part4,
    "part5_permuted_controls": part5,
    "part6_boolean_flags": boolean_flag_details,
    "deceptive_family_preservation_checks": deceptive_family_preservation_checks,
    "offline_memory_nonzero_keys": offline_nonzero,
    "best_non_oracle_dev_aggregation_name": best_dev_name,
    "best_non_oracle_dev_aggregation_total_pv": best_dev_pv,
    "best_non_oracle_dev_aggregation_macro_delta_vs_A": best_dev_macro_delta,
    "elapsed_seconds": round(elapsed, 1),
}

out_json_path = os.path.join(CURRENT_DIR, "runs",
    "block1j23_deviation_risk_aggregation_ablation_seed101.json")
os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
with open(out_json_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"  JSON -> {out_json_path}")

# =============================================================================
# 12. Markdown Protocol
# =============================================================================
md_lines = []
md_lines.append("# Block 1J23 -- Deviation Risk Aggregation Ablation")
md_lines.append("")
md_lines.append("## 1. Objective")
md_lines.append("")
md_lines.append("Test whether the failure of 1J21/1J22 family-conditioned deviation memory "
               "is caused by the sum-all-positive-weights risk aggregation rule.")
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
md_lines.append("")

# Part 1
md_lines.append("## 4. Part 1: Aggregation Formula Comparison")
md_lines.append("")
md_lines.append("| Variant | Total PV | Dec PV | Norm PV | Macro BAcc | Delta vs A | Eff Probes | Avoided Eff | Avoided PV | Avoided NV | Helpful | Harmful | Hard Excl | Aggregation Formula | Oracle |")
md_lines.append("|---------|----------|--------|---------|------------|------------|------------|-------------|------------|------------|---------|---------|-----------|---------------------|--------|")
for row in part1_rows:
    md_lines.append(f"| {row['variant']} | {row['total_prior_violations']} | "
                   f"{row['deceptive_object_prior_violations']} | "
                   f"{row['normal_object_prior_violations']} | "
                   f"{row['mean_macro_bal']:.4f} | {row['macro_delta_vs_A']} | "
                   f"{row['effective_probe_count']} | {row['avoided_effective_probe_count']} | "
                   f"{row['avoided_prior_violation_count']} | {row['avoided_nonviolation_count']} | "
                   f"{row['helpful_avoidance_count']} | {row['harmful_avoidance_count']} | "
                   f"{row['hard_exclusion_used']} | "
                   f"{row['aggregation_formula'][:60]} | "
                   f"{row['diagnostic_oracle_rule']} |")
md_lines.append("")

# Part 2
md_lines.append("## 5. Part 2: Candidate-Level Aggregation Audit")
md_lines.append("")
md_lines.append(f"Total candidates audited: {len(candidate_audit_rows)}")
md_lines.append("")
md_lines.append("### Top 30 Candidates by Sum Positive Penalty")
md_lines.append("")
md_lines.append("| OID | Action | Family | Dev Feats | SumPos | MaxPos | Top2 | MeanPos | SignedSum | SignedMaxAbs | PV | NV | Utility | Sel by A | Sel by SumPos | Sel by MaxPos | Sel by SignedSum |")
md_lines.append("|-----|--------|--------|-----------|--------|--------|------|---------|-----------|-------------|----|----|---------|----------|--------------|--------------|-----------------|")
for c in sorted(candidate_audit_rows, key=lambda x: x["sum_positive_penalty"], reverse=True)[:30]:
    sel_a = "Y" if c["selected_by_variant"].get("A_no_memory", False) else ""
    sel_sp = "Y" if c["selected_by_variant"].get("B_sum_positive_current", False) else ""
    sel_mp = "Y" if c["selected_by_variant"].get("B_max_positive", False) else ""
    sel_ss = "Y" if c["selected_by_variant"].get("B_signed_sum_capped", False) else ""
    md_lines.append(f"| {c['object_id']} | {c['action']} | {c['inferred_family']} | "
                   f"{c['injected_deviation_features']} | "
                   f"{c['sum_positive_penalty']:.4f} | {c['max_positive_penalty']:.4f} | "
                   f"{c['top2_positive_penalty']:.4f} | {c['mean_positive_penalty']:.4f} | "
                   f"{c['signed_sum_penalty']:.4f} | {c['signed_max_abs_penalty']:.4f} | "
                   f"{c['prior_violation']} | {c['prior_nonviolation']} | "
                   f"{c['probe_utility_signal']} | {sel_a} | {sel_sp} | {sel_mp} | {sel_ss} |")
md_lines.append("")

# Part 3
md_lines.append("## 6. Part 3: Helpful vs Harmful Avoidance")
md_lines.append("")
md_lines.append("| Variant | Avoided PV | Avoided NV | Avoided Eff | Avoided TP | Helpful | Harmful | Net |")
md_lines.append("|---------|------------|------------|-------------|------------|---------|---------|-----|")
for row in part3_rows:
    md_lines.append(f"| {row['variant']} | {row['avoided_prior_violation_count']} | "
                   f"{row['avoided_nonviolation_count']} | {row['avoided_effective_probe_count']} | "
                   f"{row['avoided_true_positive_probe_count']} | "
                   f"{row['helpful_avoidance_count']} | {row['harmful_avoidance_count']} | "
                   f"{row['net_helpful_minus_harmful']} |")
md_lines.append("")

# Part 4
md_lines.append("## 7. Part 4: Positive vs Negative Signal Audit")
md_lines.append("")
md_lines.append(f"- num_positive_weight_keys: {num_positive_keys}")
md_lines.append(f"- num_negative_weight_keys: {num_negative_keys}")
md_lines.append(f"- total_abs_positive_weight: {total_abs_positive:.4f}")
md_lines.append(f"- total_abs_negative_weight: {total_abs_negative:.4f}")
md_lines.append(f"- candidates_with_protective_signal: {candidates_with_protective}")
md_lines.append(f"- candidates_where_negative_changed_selection: {candidates_where_negative_changed}")
md_lines.append("")
md_lines.append("### Top 10 Positive Keys")
md_lines.append("")
md_lines.append("| Goal | Action | Family | Dev Feat | Signed Weight |")
md_lines.append("|------|--------|--------|-----------|---------------|")
for k in top_positive:
    key = k["key"]
    md_lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {k['signed_weight']:.4f} |")
md_lines.append("")
md_lines.append("### Top 10 Negative Keys")
md_lines.append("")
md_lines.append("| Goal | Action | Family | Dev Feat | Signed Weight |")
md_lines.append("|------|--------|--------|-----------|---------------|")
for k in top_negative:
    key = k["key"]
    md_lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {k['signed_weight']:.4f} |")
md_lines.append("")

# Part 5
md_lines.append("## 8. Part 5: Matching Permuted Controls")
md_lines.append("")
md_lines.append("| Real Variant | Permuted Variant | Real PV | Permuted PV | Real BAcc | Permuted BAcc | Real Beats Permuted |")
md_lines.append("|-------------|-----------------|---------|-------------|-----------|---------------|---------------------|")
for row in part5_rows:
    md_lines.append(f"| {row['real_variant_name']} | {row['permuted_variant_name']} | "
                   f"{row['real_total_pv']} | {row['permuted_total_pv']} | "
                   f"{row['real_macro_bal']:.4f} | {row['permuted_macro_bal']:.4f} | "
                   f"{row['real_beats_permuted']} |")
md_lines.append("")

# Part 6
md_lines.append("## 9. Part 6: Boolean Flag Details")
md_lines.append("")
for flag_name, flag_info in boolean_flag_details.items():
    md_lines.append(f"### {flag_name}: {flag_info['value']}")
    md_lines.append(f"  Rule: {flag_info['rule']}")
    if flag_info["supporting_values"]:
        for sv_key, sv_val in flag_info["supporting_values"].items():
            md_lines.append(f"  - {sv_key}: {sv_val}")
    md_lines.append("")

# Summary
md_lines.append("## 10. Summary")
md_lines.append("")
md_lines.append("```")
md_lines.append("[block_done]")
md_lines.append(f"block_id=1J23")
md_lines.append(f"A_total_pv={a_total_pv}")
md_lines.append(f"B_sum_positive_current_total_pv={sum_pos_pv}")
md_lines.append(f"B_max_positive_total_pv={max_pos_pv}")
md_lines.append(f"B_top2_positive_capped_total_pv={top2_pv}")
md_lines.append(f"B_mean_positive_total_pv={mean_pos_pv}")
md_lines.append(f"B_signed_sum_capped_total_pv={signed_sum_pv}")
md_lines.append(f"B_signed_max_abs_total_pv={signed_max_abs_pv}")
md_lines.append(f"Oracle_dev_feature_rule_total_pv={oracle_pv}")
md_lines.append(f"Family_only_control_total_pv={family_only_pv}")
md_lines.append(f"best_non_oracle_dev_aggregation_name={best_dev_name}")
md_lines.append(f"best_non_oracle_dev_aggregation_total_pv={best_dev_pv}")
md_lines.append(f"best_non_oracle_dev_aggregation_macro_delta_vs_A={best_dev_macro_delta}")
md_lines.append(f"current_sum_reproduced_1j22={current_sum_reproduced_1j22}")
md_lines.append(f"max_positive_beats_current_sum={max_positive_beats_current_sum}")
md_lines.append(f"signed_sum_beats_current_sum={signed_sum_beats_current_sum}")
md_lines.append(f"any_dev_aggregation_beats_A={any_dev_aggregation_beats_A}")
md_lines.append(f"any_dev_aggregation_beats_matching_permuted={any_dev_aggregation_beats_matching_permuted}")
md_lines.append(f"family_only_still_dominates_all_dev_aggregators={family_only_still_dominates_all_dev_aggregators}")
md_lines.append(f"counterproductive_weighting_reduced={counterproductive_weighting_reduced}")
md_lines.append(f"hard_exclusion_used={hard_exclusion_used}")
md_lines.append(f"deceptive_family_preservation_valid={deceptive_family_preservation_valid}")
md_lines.append(f"implementation_status={implementation_status}")
md_lines.append(f"failure_reason={failure_reason}")
md_lines.append(f"elapsed={elapsed:.1f}s")
md_lines.append("```")

out_md_path = os.path.join(CURRENT_DIR, "protocols",
    "block1j23_deviation_risk_aggregation_ablation_seed101.md")
os.makedirs(os.path.dirname(out_md_path), exist_ok=True)
with open(out_md_path, "w") as f:
    f.write("\n".join(md_lines))
print(f"  MD -> {out_md_path}")

print("\n" + "=" * 70)
print("Block 1J23 complete.")
print(f"  A PV={a_total_pv}  best_dev_aggregator='{best_dev_name}' PV={best_dev_pv} delta={best_dev_macro_delta}")
print("=" * 70)
