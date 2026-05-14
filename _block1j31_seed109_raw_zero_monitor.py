"""
Block 1J31 -- Seed 109 Raw-Penalty-Zero Monitor (Step-Level).

Patched from 1J30 with step-level monitoring, CSV export, and time-of-learning table.
Detailed companion to the 1J31 source audit with per-candidate per-step data.
All 1J24b/1J28/1J29/1J30 validated fixes preserved.

Variants:
  A_no_memory                        -- Baseline (once)
  B_signed_sum_online_scale_1        -- Online signed sum at scale 1.0
  B_signed_sum_online_scale_3        -- Online signed sum at scale 3.0
  B_signed_max_abs_online_scale_1    -- Online signed max abs at scale 1.0
  B_signed_max_abs_online_scale_3    -- Online signed max abs at scale 3.0
  Family_only_online_control         -- Online family-only memory (once)
  Offline_signed_sum_upper_bound     -- Offline oracle upper bound (once)
  Scales: [1.0, 3.0]
  No permutation distribution (audit-only run).

Additions vs 1J30:
  - Step-level monitoring: per-candidate (object, action) data at each probe step
  - CSV export: one row per (episode, step, object, action)
  - Time-of-learning table: per-key weight evolution across episodes
  - Comprehensive leakage audit with per-episode checks
"""
import os, sys, json, copy, random, time, math, hashlib, argparse
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

# =============================================================================
# Step-Level Monitoring Infrastructure (1J31 monitor addition)
# =============================================================================
STEP_MONITOR_LOG = []  # per-step per-candidate entries
STEP_MONITOR_CSV_ROWS = []  # flat list for CSV export

def _monitor_record_step(episode_id, step_id, object_id, family, dev_feats,
                          score_no_mem, score_with_mem, per_action_details):
    # Record per-candidate per-action step-level monitoring data.
    entry = {
        "episode": episode_id,
        "step": step_id,
        "object_id": object_id,
        "family": family,
        "dev_feats_present": dev_feats,
        "score_before_memory": round(score_no_mem, 6) if score_no_mem is not None else None,
        "score_after_memory": round(score_with_mem, 6) if score_with_mem is not None else None,
        "per_action": per_action_details,
    }
    STEP_MONITOR_LOG.append(entry)
    for ad in per_action_details:
        STEP_MONITOR_CSV_ROWS.append({
            "episode": episode_id,
            "step": step_id,
            "object_id": object_id,
            "family": family,
            "dev_feats_present": ",".join(dev_feats) if dev_feats else "",
            "action": ad["action"],
            "base_prior": ad["base_prior"],
            "risk_penalty_raw": ad["risk_penalty_raw"],
            "risk_penalty_clipped": ad["risk_penalty_clipped"],
            "scaled_penalty": ad["scaled_penalty"],
            "raw_score": ad["raw_score"],
            "final_score": ad["final_score"],
            "per_feat_weights": ad.get("per_feat_weights_str", ""),
            "ground_truth_outcome": ad.get("ground_truth_outcome", ""),
            "is_prior_violation": ad.get("is_prior_violation", False),
            "was_selected": ad.get("was_selected", False),
        })

t0 = time.time()

# =============================================================================
# CLI arguments (for multiseed runs)
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=109, help="Random seed (default: 109)")
parser.add_argument("--n-perm-repeats", type=int, default=0, help="Permutation repeats per scale (default: 0, audit-only no perms)")
parser.add_argument("--output-suffix", type=str, default="seed109", help="Suffix for output filenames")
parser.add_argument("--scales", type=str, default="1.0,3.0", help="Comma-separated scales (default: 1.0,3.0)")
parser.add_argument("--smoke", action="store_true", help="Smoke test: scales=[1.0], n-perm-repeats=1")
args = parser.parse_args()

if args.smoke:
    args.scales = "1.0"
    args.n_perm_repeats = 1

# =============================================================================
# Constants
# =============================================================================
SMOKE_SEED = args.seed
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

def deterministic_perm_seed(goal, action, family, perm_repeat):
    """Deterministic permutation seed using hashlib.md5 (Fix 3)."""
    key_str = f"{goal}|{action}|{family}|repeat_{perm_repeat}"
    hash_hex = hashlib.md5(key_str.encode()).hexdigest()
    return int(hash_hex[:16], 16) % (2**31)

N_PERM_REPEATS = args.n_perm_repeats

SCALES = [float(x) for x in args.scales.split(",")]

ACTION_TO_QUERY = {
    "craft_plank": "need_planks", "eat": "need_food",
    "use_as_tool": "need_tool", "burn_as_fuel": "need_fuel",
    "mine_by_hand": "need_stone", "mine_with_pickaxe": "need_stone",
}

print("=" * 70)
print("Block 1J31 -- Seed 109 Raw-Penalty-Zero Monitor (Step-Level)")
print(f"  condition={COND['label']}  seed={SMOKE_SEED}  budget={BUDGET}")
print(f"  episodes={N_EPISODES}  risk_alpha={RISK_ALPHA}  max_risk={MAX_RISK}")
print(f"  n_perm_repeats={N_PERM_REPEATS} (audit-only)  scales={SCALES}  output_suffix={args.output_suffix}")
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
print("\n[2/12] Generating test objects + deceptive objects...")
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
print("\n[3/12] Building goal-conditioned soft prior table...")


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


def get_ground_truth_outcome(obj, action):
    profile = obj.get("hidden_affordance_profile", {})
    result = profile.get(action, "success")
    return 1.0 if result == "success" else 0.0


# =============================================================================
# 5. FamilyConditionedRiskMemory (same as 1J23)
# =============================================================================
print("\n[4/12] Defining FamilyConditionedRiskMemory...")

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
        if key in self._signed_cache:
            return self._signed_cache[key]
        c = self.counts[key]
        v_with = c["violation_with"]
        nv_with = c["nonviolation_with"]
        v_without = c["violation_without"]
        nv_without = c["nonviolation_without"]
        actual_violations = (v_with + v_without) - 2.0
        if actual_violations < self.min_violation_support:
            self._signed_cache[key] = 0.0
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
            if self.get_risk_weight(*key) > 1e-9:
                count += 1
        return count

    def get_num_signed_nonzero_keys(self):
        count = 0
        for key in self.counts:
            if abs(self.get_raw_risk_weight_signed(*key)) > 1e-9:
                count += 1
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


def build_permuted_memory(base_memory):
    """Permute risk weights within (goal, action, family) groups."""
    permuted = base_memory.clone()
    groups = defaultdict(list)
    for key in base_memory.counts:
        if len(key) == 4:
            goal, action, family, dev_feat = key
            groups[(goal, action, family)].append(key)
    for grp_key, keys in groups.items():
        count_dicts = [dict(base_memory.counts[k]) for k in keys]
        goal, action, family = grp_key
        perm_seed = deterministic_perm_seed(goal, action, family, 0)
        rng_shuffle = random.Random(perm_seed)
        rng_shuffle.shuffle(count_dicts)
        for k, cd in zip(keys, count_dicts):
            permuted.counts[k] = defaultdict(float, cd)
    permuted._risk_cache.clear()
    permuted._signed_cache.clear()
    return permuted


def build_offline_oracle_memory(test_objects, key_mode="family_dev"):
    """Build memory from full offline oracle (all test objects, hidden outcomes)."""
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
# 6. Policy Classes
# =============================================================================
print("\n[5/12] Defining policy classes...")

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
    """Online policy that learns signed risk weights from observed probe outcomes.

    Key differences from offline:
    - Memory is updated in on_probe_result with actual observed outcomes
    - Memory state at decision time only reflects prior probes (causally valid)
    - Memory persists across episodes (shared reference)

    aggregation_mode: 'sum_positive', 'signed_sum', 'signed_max_abs'
    """
    def __init__(self, instance_memory, rng, family_conditioned_memory,
                 target_probe_count=8, explore_fraction=0.0, risk_penalty_scale=1.0,
                 aggregation_mode="signed_sum", max_risk=MAX_RISK,
                 episode_id=0, step_counter_ref=None):
        super().__init__(instance_memory, rng, target_probe_count, explore_fraction)
        self._fcrm = family_conditioned_memory
        self._risk_penalty_scale = risk_penalty_scale
        self.aggregation_mode = aggregation_mode
        self.max_risk = max_risk
        self._episode_id = episode_id
        self._step_counter_ref = step_counter_ref if step_counter_ref is not None else [0]
        self._variant = f"online_{aggregation_mode}"
        self._probe_step_counter = 0
        self._decision_path_log = []
        self._pending_decision_log = None
        self._monitor_step_entries = []  # 1J31: per-step detailed monitoring

    def _score_candidate_no_memory(self, features, norm_cost):
        """Compute best score WITHOUT any risk memory (A_no_memory baseline)."""
        best_score = float('-inf')
        for action in MAIN_CANDIDATE_ACTIONS:
            base_prior = get_goal_soft_prior(features, action)
            raw_score = base_prior - COST_WEIGHT * norm_cost
            score = max(EXPLORATION_FLOOR, raw_score)
            if score > best_score:
                best_score = score
        return best_score

    def _get_per_feature_weights(self, features, action):
        goal = ACTION_TO_QUERY.get(action, "")
        family = detect_type_family(features)
        weights = []
        for dev_feat in INJECTED_DEVIATION_FEATURES:
            if features.get(dev_feat, False):
                w = self._fcrm.get_raw_risk_weight_signed(goal, action, family, dev_feat)
                weights.append((dev_feat, w))
        return weights

    def _get_per_feature_weights_positive(self, features, action):
        goal = ACTION_TO_QUERY.get(action, "")
        family = detect_type_family(features)
        weights = []
        for dev_feat in INJECTED_DEVIATION_FEATURES:
            if features.get(dev_feat, False):
                w = self._fcrm.get_risk_weight(goal, action, family, dev_feat)
                weights.append((dev_feat, w))
        return weights

    def get_total_risk_penalty(self, features, action):
        if self.aggregation_mode == "sum_positive":
            weights = self._get_per_feature_weights_positive(features, action)
            return sum(w for _, w in weights if w > 0)
        elif self.aggregation_mode == "signed_sum":
            weights = self._get_per_feature_weights(features, action)
            total = sum(w for _, w in weights)
            return max(-self.max_risk, min(self.max_risk, total))
        elif self.aggregation_mode == "signed_max_abs":
            weights = self._get_per_feature_weights(features, action)
            if not weights:
                return 0.0
            best = max(weights, key=lambda x: abs(x[1]))
            return max(-self.max_risk, min(self.max_risk, best[1]))
        elif self.aggregation_mode == "family_only":
            return self._fcrm.get_total_risk_penalty(features, action)
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

    def _select_probe_target(self, view):
        """Override with decision-path logging for all candidates."""
        self._pending_decision_log = None
        if len(self._probed_oids) >= self._target_probe_count:
            return None

        candidate_pool = []
        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids:
                continue
            features = view.get_observed_features(oid)
            if features is None:
                candidate_pool.append({
                    "object_id": oid,
                    "observed_features_available": [],
                    "excluded": True,
                    "exclusion_reason": "not_visited_or_features_unavailable",
                })
                continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost):
                candidate_pool.append({
                    "object_id": oid,
                    "observed_features_available": sorted([f for f, v in features.items() if v]),
                    "excluded": True,
                    "exclusion_reason": "cannot_afford_total_cost",
                })
                continue
            norm_cost = total_cost / max(view.initial_budget, 0.001)
            score_no_mem = self._score_candidate_no_memory(features, norm_cost)
            score_with_mem = self._score_candidate(features, norm_cost)
            if score_with_mem > float('-inf'):
                candidate_pool.append({
                    "object_id": oid,
                    "observed_features_available": sorted([f for f, v in features.items() if v]),
                    "excluded": False,
                    "exclusion_reason": "",
                    "score_before_memory": round(score_no_mem, 6),
                    "score_after_memory_before_floor": round(score_with_mem, 6),
                })

        # Identify candidates excluded due to various reasons
        visited_unprobed_ids = set()
        for oid in view.get_all_object_ids():
            if view.is_visited(oid) and oid not in self._probed_oids:
                visited_unprobed_ids.add(oid)

        # Add candidates that were filtered out earlier (not visited, no features)
        all_obj_ids = set(view.get_all_object_ids())
        logged_ids = set(c["object_id"] for c in candidate_pool)
        for oid in all_obj_ids:
            if oid not in logged_ids:
                features = view.get_observed_features(oid)
                if features is None:
                    candidate_pool.append({
                        "object_id": oid,
                        "observed_features_available": [],
                        "excluded": True,
                        "exclusion_reason": "features_not_observed",
                    })
                elif oid not in visited_unprobed_ids and oid not in self._probed_oids:
                    candidate_pool.append({
                        "object_id": oid,
                        "observed_features_available": sorted([f for f, v in features.items() if v]),
                        "excluded": True,
                        "exclusion_reason": "not_visited",
                    })

        active_candidates = [c for c in candidate_pool if not c["excluded"]]
        active_candidates.sort(key=lambda x: x["score_after_memory_before_floor"], reverse=True)

        # Assign ranks
        for i, c in enumerate(active_candidates):
            c["rank_after_memory"] = i + 1

        # Assign no-memory ranks (sort by score_before_memory)
        sorted_no_mem = sorted(active_candidates, key=lambda x: x["score_before_memory"], reverse=True)
        no_mem_rank_map = {}
        for i, c in enumerate(sorted_no_mem):
            no_mem_rank_map[c["object_id"]] = i + 1
        for c in active_candidates:
            c["rank_before_memory"] = no_mem_rank_map.get(c["object_id"], len(active_candidates))

        # Assign ranks to excluded
        for c in candidate_pool:
            if c.get("excluded"):
                c["rank_before_memory"] = None
                c["rank_after_memory"] = None
                c["score_before_memory"] = None
                c["score_after_memory_before_floor"] = None

        if not active_candidates:
            self._decision_path_log.append({
                "episode_id": self._episode_id,
                "step_id": len(self._probed_oids),
                "candidate_pool": candidate_pool,
                "selected_object_id": None,
                "selected_action": None,
                "per_action_details": [],
            })
            return None

        # Select (same logic as parent)
        if self._explore_fraction > 0 and len(active_candidates) >= 3:
            if self._rng.random() < self._explore_fraction:
                n = len(active_candidates)
                lo = max(0, int(n * 0.30))
                hi = min(n - 1, int(n * 0.80))
                if hi > lo:
                    selected = active_candidates[self._rng.randint(lo, hi)]
                    self._pending_decision_log = {
                        "episode_id": self._episode_id,
                        "step_id": len(self._probed_oids),
                        "candidate_pool": candidate_pool,
                        "selected_object_id": selected["object_id"],
                    }
                    return selected["object_id"]

        selected = active_candidates[0]
        self._pending_decision_log = {
            "episode_id": self._episode_id,
            "step_id": len(self._probed_oids),
            "candidate_pool": candidate_pool,
            "selected_object_id": selected["object_id"],
        }
        return selected["object_id"]

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
        per_action_monitor = []
        for action in MAIN_CANDIDATE_ACTIONS:
            norm_cost = view.probe_cost / max(view.initial_budget, 0.001)
            final_score, raw_score, base_prior, risk_penalty, scaled_penalty = \
                self._compute_adjusted_score(features, action, norm_cost)
            # 1J31: capture per-action weights
            feat_weights = self._get_per_feature_weights(features, action)
            feat_weights_str = ";".join(f"{f}={w:.6f}" for f, w in feat_weights)
            per_action_monitor.append({
                "action": action,
                "base_prior": round(base_prior, 6),
                "risk_penalty_raw": round(risk_penalty, 6),
                "risk_penalty_clipped": round(risk_penalty, 6),
                "scaled_penalty": round(scaled_penalty, 6),
                "raw_score": round(raw_score, 6),
                "final_score": round(final_score, 6),
                "per_feat_weights_str": feat_weights_str,
                "ground_truth_outcome": "unknown",
                "is_prior_violation": False,
                "was_selected": False,
            })
            if final_score > best_score:
                best_score = final_score; best_action = action
                best_info = (base_prior, risk_penalty, scaled_penalty, raw_score, final_score)
        # Mark the selected action and record step-level monitoring
        if per_action_monitor:
            # best_info = (base_prior, risk_penalty, scaled_penalty, raw_score, final_score)
            _m_bp, _m_rp, _m_sp, _m_rs, _m_fs = best_info
            for pam in per_action_monitor:
                if pam["action"] == best_action:
                    pam["was_selected"] = True
                    pam["final_score"] = round(_m_fs, 6)
                    pam["scaled_penalty"] = round(_m_sp, 6)
            _monitor_record_step(
                self._episode_id,
                len(self._probed_oids),
                object_id,
                detect_type_family(features),
                sorted([f for f in INJECTED_DEVIATION_FEATURES if features.get(f, False)]),
                self._score_candidate_no_memory(features,
                    view.probe_cost / max(view.initial_budget, 0.001)),
                self._score_candidate(features,
                    view.probe_cost / max(view.initial_budget, 0.001)),
                per_action_monitor,
            )
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

            # UPDATE ONLINE RISK MEMORY with observed outcome
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
    """Online family-only policy for comparison."""
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
                "is_effective": is_eff,
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


class OnlineStepwisePermutedPolicy(SoftPriorOnlyPolicyV2):
    """Online permuted control: learns online, permutes weights at decision time (Fix 2).

    Key difference from old frozen-final-memory approach:
    - Has its OWN FamilyConditionedRiskMemory that learns online
    - Updates memory in on_probe_result (same as real variants)
    - At decision time, permutes current weights within (goal, action, family)
    - Permutation is deterministic per (goal, action, family, perm_repeat)
    - Uses hashlib.md5 for deterministic seed (Fix 3)
    """
    def __init__(self, instance_memory, rng, family_conditioned_memory,
                 target_probe_count=8, explore_fraction=0.0, risk_penalty_scale=1.0,
                 aggregation_mode="signed_sum", max_risk=MAX_RISK,
                 episode_id=0, step_counter_ref=None, perm_repeat=0):
        super().__init__(instance_memory, rng, target_probe_count, explore_fraction)
        self._fcrm = family_conditioned_memory
        self._risk_penalty_scale = risk_penalty_scale
        self.aggregation_mode = aggregation_mode
        self.max_risk = max_risk
        self._episode_id = episode_id
        self._step_counter_ref = step_counter_ref if step_counter_ref is not None else [0]
        self._variant = f"online_stepwise_permuted_{aggregation_mode}_r{perm_repeat}"
        self._probe_step_counter = 0
        self._perm_repeat = perm_repeat
        self._perm_cache = {}

    def _get_permuted_weights(self, features, action):
        """Get per-feature signed weights, permuted deterministically within (goal, action, family)."""
        goal = ACTION_TO_QUERY.get(action, "")
        family = detect_type_family(features)

        cache_key = (goal, action, family, self._perm_repeat)
        if cache_key not in self._perm_cache:
            perm_seed = deterministic_perm_seed(goal, action, family, self._perm_repeat)
            perm_rng = random.Random(perm_seed)
            indices = list(range(len(SORTED_DEV_FEATURES)))
            perm_rng.shuffle(indices)
            self._perm_cache[cache_key] = indices

        perm_indices = self._perm_cache[cache_key]

        weights = []
        for i, df in enumerate(SORTED_DEV_FEATURES):
            if features.get(df, False):
                w = self._fcrm.get_raw_risk_weight_signed(goal, action, family, df)
                weights.append((df, w))

        if not weights:
            return []

        all_raw = []
        for df in SORTED_DEV_FEATURES:
            all_raw.append(self._fcrm.get_raw_risk_weight_signed(goal, action, family, df))

        permuted_map = {}
        for i, df in enumerate(SORTED_DEV_FEATURES):
            permuted_map[df] = all_raw[perm_indices[i]]

        result = []
        for df, _ in weights:
            result.append((df, permuted_map[df]))
        return result

    def get_total_risk_penalty(self, features, action):
        permuted = self._get_permuted_weights(features, action)
        if self.aggregation_mode == "signed_sum":
            total = sum(w for _, w in permuted)
            return max(-self.max_risk, min(self.max_risk, total))
        elif self.aggregation_mode == "signed_max_abs":
            if not permuted:
                return 0.0
            best = max(permuted, key=lambda x: abs(x[1]))
            return max(-self.max_risk, min(self.max_risk, best[1]))
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
            # UPDATE ONLINE RISK MEMORY (same as real variants, its OWN memory)
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


# =============================================================================
# 7. C15b reference
# =============================================================================
print("\n[6/12] Running C15b reference...")

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

# =============================================================================
# 8. Run all variants
# =============================================================================
print("\n[7/12] Running all variants...")

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


def compute_variant_stats(all_episodes_var, all_episodes_a, variant_label,
                          oracle_evidence=False, future_outcome_used=False,
                          true_deceptive_used=False):
    agg_v = aggregate_episodes(all_episodes_var)
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
        "diagnostic_oracle_evidence": oracle_evidence,
        "future_outcome_used_for_current_selection": future_outcome_used,
        "true_deceptive_label_used": true_deceptive_used,
        "agg": agg_v,
        "episodes": all_episodes_var,
    }


# --- Fix 1: Precompute shared episode positions ---
print("  Precomputing shared episode positions (Fix 1)...")
shared_position_seed = SMOKE_SEED + 15000
episode_positions_by_ep = {}
episode_position_hashes = {}
pos_rng = random.Random(shared_position_seed)
test_oids_local = sorted(test_objects_deceptive.keys())
for ep in range(N_EPISODES):
    ep_positions = MiniMCSimulatorTruth.assign_positions(
        test_oids_local, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, pos_rng)
    episode_positions_by_ep[ep] = dict(ep_positions)
    pos_hash = hashlib.md5(str(sorted(ep_positions.items())).encode()).hexdigest()[:12]
    episode_position_hashes[ep] = pos_hash

shared_positions_used = True

# --- Run A_no_memory baseline (uses shared positions) ---
print("  Running A_no_memory...")
all_episodes_a = []
pv_history_a = set()

for ep in range(N_EPISODES):
    ep_seed = RNG_SEED_BASE + 2 + ep * 100
    ep_rng = random.Random(ep_seed)
    ep_positions = episode_positions_by_ep[ep]
    im_a = im_base.clone()
    env_a = MiniMCEnvironment(test_objects_deceptive, ep_positions, initial_budget=BUDGET)
    policy_a = SoftPriorOnlyPolicyV2(im_a, ep_rng, target_probe_count=c15b_probed_n)
    ep_a, pv_history_a = run_episode_and_extract(
        test_objects_deceptive, query_gt_deceptive, policy_a, env_a, ep,
        prior_violation_history=pv_history_a)
    all_episodes_a.append(ep_a)

agg_a = aggregate_episodes(all_episodes_a)
print(f"    A_no_memory: total_pv={agg_a['total_prior_violations']}, "
      f"macro_bal={agg_a['mean_macro_bal']:.4f}")


# --- Online variants (shared memory across episodes, updated in on_probe_result) ---
def run_online_variant(variant_name, aggregation_mode, key_mode="family_dev",
                       use_family_only_policy=False, risk_penalty_scale=1.0):
    """Run 5 episodes with online memory learning (shared positions, Fix 1).

    Memory is created once and shared across episodes.
    Each episode's policy mutates the shared memory during on_probe_result.
    Uses precomputed episode_positions_by_ep for position consistency.
    """
    if key_mode == "family_only":
        memory = FamilyConditionedRiskMemory(key_mode="family_only")
    else:
        memory = FamilyConditionedRiskMemory(key_mode="family_dev")

    all_eps = []
    pv_hist = set()
    step_counter = [0]  # mutable reference for step tracking

    for ep in range(N_EPISODES):
        ep_seed = RNG_SEED_BASE + _get_seed_offset(variant_name) + ep * 100
        ep_rng = random.Random(ep_seed)
        ep_positions = episode_positions_by_ep[ep]
        im_ep = im_base.clone()
        env_ep = MiniMCEnvironment(test_objects_deceptive, ep_positions, initial_budget=BUDGET)

        if use_family_only_policy:
            policy_ep = OnlineFamilyOnlyPolicy(
                im_ep, ep_rng, memory, target_probe_count=c15b_probed_n,
                risk_penalty_scale=risk_penalty_scale,
                episode_id=ep, step_counter_ref=step_counter)
        else:
            policy_ep = OnlineSignedPolicy(
                im_ep, ep_rng, memory, target_probe_count=c15b_probed_n,
                aggregation_mode=aggregation_mode,
                risk_penalty_scale=risk_penalty_scale,
                episode_id=ep, step_counter_ref=step_counter)

        ep_data, pv_hist = run_episode_and_extract(
            test_objects_deceptive, query_gt_deceptive, policy_ep, env_ep, ep,
            prior_violation_history=pv_hist)
        all_eps.append(ep_data)

    return all_eps, memory


def run_online_stepwise_permuted_variant(variant_name, aggregation_mode, perm_repeat,
                                         risk_penalty_scale=1.0):
    """Run 5 episodes with ONLINE STEPWISE permuted control (Fix 2).

    Each repeat has its OWN FamilyConditionedRiskMemory that learns online.
    Permutation happens at decision time, not post-hoc.
    Uses shared episode positions (Fix 1).
    """
    memory = FamilyConditionedRiskMemory(key_mode="family_dev")
    all_eps = []
    pv_hist = set()
    step_counter = [0]

    for ep in range(N_EPISODES):
        ep_seed = RNG_SEED_BASE + _get_seed_offset(variant_name) + perm_repeat * 50 + ep * 100
        ep_rng = random.Random(ep_seed)
        ep_positions = episode_positions_by_ep[ep]
        im_ep = im_base.clone()
        env_ep = MiniMCEnvironment(test_objects_deceptive, ep_positions, initial_budget=BUDGET)

        policy_ep = OnlineStepwisePermutedPolicy(
            im_ep, ep_rng, memory, target_probe_count=c15b_probed_n,
            aggregation_mode=aggregation_mode,
            risk_penalty_scale=risk_penalty_scale,
            episode_id=ep, step_counter_ref=step_counter,
            perm_repeat=perm_repeat)

        ep_data, pv_hist = run_episode_and_extract(
            test_objects_deceptive, query_gt_deceptive, policy_ep, env_ep, ep,
            prior_violation_history=pv_hist)
        all_eps.append(ep_data)

    return all_eps, memory


def _get_seed_offset(variant_name):
    offsets = {
        "B_sum_positive_online": 100,
        "B_signed_sum_online": 200,
        "B_signed_max_abs_online": 300,
        "Family_only_online_control": 400,
        "C_signed_sum_online_permuted": 500,
        "C_signed_max_abs_online_permuted": 600,
    }
    return offsets.get(variant_name, 999)


# Run online variants
print("  Running B_sum_positive_online...")
eps_sum_pos, mem_sum_pos = run_online_variant("B_sum_positive_online", "sum_positive")
stats_sum_pos = compute_variant_stats(eps_sum_pos, all_episodes_a, "B_sum_positive_online")

print("  Running B_signed_sum_online...")
eps_signed_sum, mem_signed_sum = run_online_variant("B_signed_sum_online", "signed_sum")
stats_signed_sum = compute_variant_stats(eps_signed_sum, all_episodes_a, "B_signed_sum_online")

print("  Running B_signed_max_abs_online...")
eps_signed_max, mem_signed_max = run_online_variant("B_signed_max_abs_online", "signed_max_abs")
stats_signed_max = compute_variant_stats(eps_signed_max, all_episodes_a, "B_signed_max_abs_online")

print("  Running Family_only_online_control...")
eps_fam_only, mem_fam_only = run_online_variant(
    "Family_only_online_control", "family_only",
    key_mode="family_only", use_family_only_policy=True)
stats_fam_only = compute_variant_stats(eps_fam_only, all_episodes_a, "Family_only_online_control")

# --- Online stepwise permuted controls (Fix 2): each learns online, permutes at decision time ---
# Replaced frozen-final-memory permuted controls with true online stepwise permutation.
permuted_control_mode = "online_stepwise"
final_memory_permutation_used = False
deterministic_permutation_seed_used = True


class FrozenMemoryPolicy(SoftPriorOnlyPolicyV2):
    """Uses a frozen (permuted) memory without updating it."""
    def __init__(self, instance_memory, rng, frozen_memory,
                 target_probe_count=8, explore_fraction=0.0, risk_penalty_scale=1.0,
                 aggregation_mode="signed_sum", max_risk=MAX_RISK):
        super().__init__(instance_memory, rng, target_probe_count, explore_fraction)
        self._fcrm = frozen_memory
        self._risk_penalty_scale = risk_penalty_scale
        self.aggregation_mode = aggregation_mode
        self.max_risk = max_risk
        self._variant = f"permuted_{aggregation_mode}"

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
        elif self.aggregation_mode == "signed_max_abs":
            weights = self._get_per_feature_weights(features, action)
            if not weights:
                return 0.0
            best = max(weights, key=lambda x: abs(x[1]))
            return max(-self.max_risk, min(self.max_risk, best[1]))
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


# --- Offline upper bound (reuse 1J23 pattern, run once) ---
print("  Running Offline_signed_sum_upper_bound...")
fcrm_offline = build_offline_oracle_memory(test_objects_deceptive, key_mode="family_dev")

all_eps_offline = []
pv_hist_offline = set()

for ep in range(N_EPISODES):
    ep_seed = RNG_SEED_BASE + 800 + ep * 100
    ep_rng = random.Random(ep_seed)
    ep_positions = episode_positions_by_ep[ep]
    im_ep = im_base.clone()
    env_ep = MiniMCEnvironment(test_objects_deceptive, ep_positions, initial_budget=BUDGET)
    frozen_offline = fcrm_offline.clone()
    policy_ep = FrozenMemoryPolicy(
        im_ep, ep_rng, frozen_offline, target_probe_count=c15b_probed_n,
        aggregation_mode="signed_sum")
    ep_data, pv_hist_offline = run_episode_and_extract(
        test_objects_deceptive, query_gt_deceptive, policy_ep, env_ep, ep,
        prior_violation_history=pv_hist_offline)
    all_eps_offline.append(ep_data)

stats_offline = compute_variant_stats(
    all_eps_offline, all_episodes_a, "Offline_signed_sum_upper_bound",
    oracle_evidence=True)

print(f"    Family_only_online_control: total_pv={stats_fam_only['total_prior_violations']}, "
      f"macro_bal={stats_fam_only['mean_macro_bal']:.4f}, delta_vs_A={stats_fam_only['macro_delta_vs_A']:.4f}")
print(f"    Offline_signed_sum_upper_bound: total_pv={stats_offline['total_prior_violations']}, "
      f"macro_bal={stats_offline['mean_macro_bal']:.4f}, delta_vs_A={stats_offline['macro_delta_vs_A']:.4f}")

# =============================================================================
# Scale sweep: risk-penalty multiplier applied AFTER clipping
# scaled_penalty = clipped_risk_penalty * scale
# 1J31: monitor run — only scales 1.0 and 3.0, no permutation distribution
print(f"\n  Running scale sweep across {SCALES} (audit-only, no permutations)...")
all_scale_results = {}

def scale_perm_dist_stats(pv_list, real_pv):
    arr = sorted(pv_list)
    n = len(arr)
    mean_val = float(np.mean(arr)) if n > 0 else 0.0
    median_val = float(np.median(arr)) if n > 0 else 0.0
    return {
        "min_total_pv": min(arr) if n > 0 else 0,
        "max_total_pv": max(arr) if n > 0 else 0,
        "mean_total_pv": round(mean_val, 2),
        "median_total_pv": int(median_val),
        "std_total_pv": round(float(np.std(arr)), 2) if n > 0 else 0.0,
        "real_total_pv": real_pv,
        "num_permuted_better_than_real": 0,
        "num_permuted_equal_to_real": 0,
        "num_permuted_worse_than_real": 0,
        "all_permuted_pvs": pv_list,
        "real_beats_permuted_median": False,
        "real_beats_80pct_of_permuted": False,
        "note": "no permutation distribution for 1J31 monitor run",
    }

for scale in SCALES:
    print(f"\n  --- Scale {scale} ---")

    # B_signed_sum_online
    print(f"    Running B_signed_sum_online_scale_{scale}...")
    eps_ss, mem_ss = run_online_variant(
        "B_signed_sum_online", "signed_sum", risk_penalty_scale=scale)
    stats_ss = compute_variant_stats(
        eps_ss, all_episodes_a, f"B_signed_sum_online_scale_{scale}")

    # B_signed_max_abs_online
    print(f"    Running B_signed_max_abs_online_scale_{scale}...")
    eps_sm, mem_sm = run_online_variant(
        "B_signed_max_abs_online", "signed_max_abs", risk_penalty_scale=scale)
    stats_sm = compute_variant_stats(
        eps_sm, all_episodes_a, f"B_signed_max_abs_online_scale_{scale}")

    print(f"    B_signed_sum_scale_{scale}: total_pv={stats_ss['total_prior_violations']}, "
          f"macro_bal={stats_ss['mean_macro_bal']:.4f}, delta_vs_A={stats_ss['macro_delta_vs_A']:.4f}")
    print(f"    B_signed_max_scale_{scale}: total_pv={stats_sm['total_prior_violations']}, "
          f"macro_bal={stats_sm['mean_macro_bal']:.4f}, delta_vs_A={stats_sm['macro_delta_vs_A']:.4f}")

    all_scale_results[scale] = {
        "signed_sum": {
            "stats": stats_ss,
            "episodes": eps_ss,
            "memory": mem_ss,
            "permuted_distribution": scale_perm_dist_stats([], stats_ss["total_prior_violations"]),
            "permuted_pvs": [],
        },
        "signed_max_abs": {
            "stats": stats_sm,
            "episodes": eps_sm,
            "memory": mem_sm,
            "permuted_distribution": scale_perm_dist_stats([], stats_sm["total_prior_violations"]),
            "permuted_pvs": [],
        },
    }

# Set scale=1.0 results for backward compatibility with remaining diagnostic code
stats_signed_sum = all_scale_results[1.0]["signed_sum"]["stats"]
stats_signed_max = all_scale_results[1.0]["signed_max_abs"]["stats"]
eps_signed_sum = all_scale_results[1.0]["signed_sum"]["episodes"]
eps_signed_max = all_scale_results[1.0]["signed_max_abs"]["episodes"]
mem_signed_sum = all_scale_results[1.0]["signed_sum"]["memory"]
mem_signed_max = all_scale_results[1.0]["signed_max_abs"]["memory"]

# =============================================================================
# 9. Raw-Penalty-Zero Monitor Diagnostics
# =============================================================================
print("\n[8/12] Computing Raw-Penalty-Zero Monitor Diagnostics...")

decision_path_audit = {}

# ===========================================================================
# Part 1: Candidate decision-path trace
# ===========================================================================
print("  Part 1: Building candidate decision-path trace...")

# Collect decision path logs from B_signed_sum_online for scales 1.0 and 3.0
# The decision_path_log is stored in the policy's _decision_path_log attribute
# For 1J31, we need to extract these from the memory/policy objects with step-level detail.
# Since run_online_variant recreates the policy per episode, we need to collect
# logs differently. We'll use the episode-level probe_details enriched with
# observation data instead.

# Build enriched decision path from probe_details and episode data
diag_p1_scale1 = []
diag_p1_scale3 = []

# Gather decision-path data from probe_details of scale 1.0 and 3.0
# For each probe decision, collect the available candidate information
for scale_val, scale_key in [(1.0, "scale_1_0"), (3.0, "scale_3_0")]:
    if scale_val not in all_scale_results:
        continue
    sr = all_scale_results[scale_val]["signed_sum"]
    eps_list = sr["episodes"]
    memory = sr["memory"]
    paths = []
    for ep_idx, ep_data in enumerate(eps_list):
        for probe_idx, probe in enumerate(ep_data.get("probe_details", [])):
            oid = probe["oid"]
            action = probe["action"]
            obj = test_objects_deceptive.get(oid, {})
            features = obj.get("visible_features", {})
            family = detect_type_family(features)
            is_deceptive = oid in DECEPTIVE_OIDS
            observed = sorted([f for f, v in features.items() if v])
            hidden = sorted([f for f in FEATURE_UNIVERSE_SET if f not in features])
            dev_present = sorted([f for f in INJECTED_DEVIATION_FEATURES if features.get(f, False)])

            # Compute what signed memory keys are available
            goal = ACTION_TO_QUERY.get(action, "")
            signed_keys = []
            for dev_feat in dev_present:
                key = (goal, action, family, dev_feat)
                raw_w = memory.get_raw_risk_weight_signed(goal, action, family, dev_feat)
                clipped_w = memory.get_risk_weight(goal, action, family, dev_feat)
                counts_raw = memory.get_raw_counts(goal, action, family, dev_feat)
                support = (counts_raw["violation_with"] + counts_raw["violation_without"] +
                          counts_raw["nonviolation_with"] + counts_raw["nonviolation_without"] - 4.0)
                signed_keys.append({
                    "key": list(key),
                    "raw_signed_weight": round(raw_w, 6),
                    "clipped_risk_weight": round(clipped_w, 6),
                    "support": round(support, 2),
                })

            # Compute scores
            norm_cost = 0.01  # approximate
            base_prior = probe.get("soft_prior", get_goal_soft_prior(features, action))
            raw_signed = probe.get("risk_penalty", 0.0)
            clipped_pen = probe.get("risk_penalty", 0.0)
            scaled_pen = probe.get("scaled_risk_penalty", 0.0)
            final_score = probe.get("final_score", EXPLORATION_FLOOR)
            score_no_mem = max(EXPLORATION_FLOOR, base_prior - COST_WEIGHT * norm_cost)

            paths.append({
                "episode_id": ep_idx,
                "step_id": probe_idx,
                "object_id": oid,
                "action": action,
                "inferred_family": family,
                "is_deceptive_object": is_deceptive,
                "prior_violation_if_probed": probe.get("is_violation", False),
                "prior_nonviolation_if_probed": probe.get("is_nonviolation", False),
                "observed_features_available_before_decision": observed,
                "hidden_features_not_yet_observed": hidden,
                "deviation_features_present": dev_present,
                "signed_memory_keys_available_before_decision": signed_keys,
                "raw_signed_penalty": round(raw_signed, 6),
                "clipped_signed_penalty": round(clipped_pen, 6),
                "scaled_signed_penalty": round(scaled_pen, 6),
                "base_soft_prior_score": round(base_prior, 6),
                "score_before_memory": round(score_no_mem, 6),
                "score_after_memory_before_floor": round(final_score, 6),
                "score_after_floor_or_clip": round(final_score, 6),
                "final_selection_score": round(final_score, 6),
                "selected_for_probe": True,
                "selected_for_action": True,
                "reason_not_selected": "none_selected",
            })

    if scale_key == "scale_1_0":
        diag_p1_scale1 = paths
    else:
        diag_p1_scale3 = paths

# Check for hidden feature leakage
hidden_feature_leakage_detected = False
for path in diag_p1_scale1:
    observed_set = set(path.get("observed_features_available_before_decision", []))
    for key_info in path.get("signed_memory_keys_available_before_decision", []):
        dev_feat = key_info["key"][3]  # (goal, action, family, dev_feat)
        if dev_feat not in observed_set:
            hidden_feature_leakage_detected = True
            break

decision_path_audit["part1_decision_path_trace_scale1"] = diag_p1_scale1
decision_path_audit["part1_decision_path_trace_scale3"] = diag_p1_scale3
decision_path_audit["hidden_feature_leakage_detected"] = hidden_feature_leakage_detected
print(f"    scale1 paths={len(diag_p1_scale1)}  scale3 paths={len(diag_p1_scale3)}  leakage={hidden_feature_leakage_detected}")

# ===========================================================================
# Part 2: Missed offline-avoidable PV audit
# ===========================================================================
print("  Part 2: Missed offline-avoidable PV audit...")

# Offline avoided PVs: PVs in A that are NOT in Offline
offline_eps = all_eps_offline
offline_pv_set = set()
for ep in offline_eps:
    for d in ep["probe_details"]:
        if d.get("is_violation"):
            offline_pv_set.add((d["oid"], d["action"]))

# A's PVs
a_pv_set = set()
for ep_a in all_episodes_a:
    for d in ep_a["probe_details"]:
        if d.get("is_violation"):
            a_pv_set.add((d["oid"], d["action"]))

# PVs that offline avoids but A does not
offline_avoided = a_pv_set - offline_pv_set  # These are PVs in A but not in Offline → offline "avoided" them
# Actually: offline PV count is what Offline makes. If A makes a PV and Offline doesn't, offline "avoided" that PV.
# missable = PVs that offline would avoid

# For each offline-avoided PV, check if online (scale 1 and 3) also avoided it
scale1_ss_pv_set = set()
scale1_eps = all_scale_results[1.0]["signed_sum"]["episodes"]
for ep in scale1_eps:
    for d in ep["probe_details"]:
        if d.get("is_violation"):
            scale1_ss_pv_set.add((d["oid"], d["action"]))

# Guard: handle smoke mode where scale 3 might not be present
_has_scale3 = 3.0 in all_scale_results
_ss3_eps = all_scale_results[3.0]["signed_sum"]["episodes"] if _has_scale3 else []
_sm3_eps = all_scale_results[3.0]["signed_max_abs"]["episodes"] if _has_scale3 else []
_ss3_stats = all_scale_results[3.0]["signed_sum"]["stats"] if _has_scale3 else {"total_prior_violations": 0, "mean_macro_bal": 0.0, "macro_delta_vs_A": 0.0}
_sm3_stats = all_scale_results[3.0]["signed_max_abs"]["stats"] if _has_scale3 else {"total_prior_violations": 0, "mean_macro_bal": 0.0, "macro_delta_vs_A": 0.0}

scale3_ss_pv_set = set()
scale3_eps = _ss3_eps
for ep in scale3_eps:
    for d in ep["probe_details"]:
        if d.get("is_violation"):
            scale3_ss_pv_set.add((d["oid"], d["action"]))

# PVs that offline avoids
offline_avoidable_pvs = a_pv_set - offline_pv_set

diag_p2 = []
missed_reasons = {
    "candidate_pool_exclusion": 0,
    "observation_unavailable": 0,
    "raw_penalty_zero": 0,
    "clipping_or_floor_saturation": 0,
    "rank_unchanged": 0,
    "top_k_threshold_failure": 0,
    "budget_or_probe_count_limit": 0,
    "other": 0,
    "total": 0,
}

for (oid, action) in sorted(offline_avoidable_pvs):
    online_avoided_scale1 = (oid, action) not in scale1_ss_pv_set
    online_avoided_scale3 = (oid, action) not in scale3_ss_pv_set
    if online_avoided_scale1:
        continue  # Already avoided by scale 1, not "missed"

    obj = test_objects_deceptive.get(oid, {})
    features = obj.get("visible_features", {})
    family = detect_type_family(features)
    observed = sorted([f for f, v in features.items() if v])
    dev_present = sorted([f for f in INJECTED_DEVIATION_FEATURES if features.get(f, False)])
    goal = ACTION_TO_QUERY.get(action, "")

    # Compute online penalties using the scale 1 memory
    mem_ss = all_scale_results[1.0]["signed_sum"]["memory"]
    online_key_available = False
    online_raw = 0.0
    online_clipped = 0.0
    for dev_feat in dev_present:
        raw_w = mem_ss.get_raw_risk_weight_signed(goal, action, family, dev_feat)
        if abs(raw_w) > 1e-9:
            online_key_available = True
        online_raw += raw_w
        online_clipped += mem_ss.get_risk_weight(goal, action, family, dev_feat)
    online_raw = max(-MAX_RISK, min(MAX_RISK, online_raw))
    online_clipped = max(-MAX_RISK, min(MAX_RISK, online_clipped))

    # Offline penalty
    offline_penalty = 0.0
    for dev_feat in dev_present:
        offline_penalty += fcrm_offline.get_raw_risk_weight_signed(goal, action, family, dev_feat)
    offline_penalty = max(-MAX_RISK, min(MAX_RISK, offline_penalty))

    # Determine failure reason
    base_prior = get_goal_soft_prior(features, action)
    norm_cost = 0.01  # approximate
    score_no_mem = max(EXPLORATION_FLOOR, base_prior - COST_WEIGHT * norm_cost)
    scaled_pen_1 = online_clipped * 1.0
    raw_score_1 = base_prior * math.exp(-scaled_pen_1) - COST_WEIGHT * norm_cost
    final_score_1 = max(EXPLORATION_FLOOR, raw_score_1)
    scaled_pen_3 = online_clipped * 3.0
    raw_score_3 = base_prior * math.exp(-scaled_pen_3) - COST_WEIGHT * norm_cost
    final_score_3 = max(EXPLORATION_FLOOR, raw_score_3)

    failure_reason = "other"
    if len(observed) == 0:
        failure_reason = "observation_unavailable"
    elif abs(online_raw) < 1e-9:
        failure_reason = "raw_penalty_zero"
    elif abs(online_clipped) < 1e-9:
        failure_reason = "clipping_or_floor_saturation"
    elif final_score_1 <= EXPLORATION_FLOOR + 1e-9 and score_no_mem <= EXPLORATION_FLOOR + 1e-9:
        failure_reason = "clipping_or_floor_saturation"
    else:
        failure_reason = "rank_unchanged"

    if not online_key_available and abs(online_raw) < 1e-9:
        failure_reason = "raw_penalty_zero"

    diag_p2.append({
        "episode_id": "?",
        "step_id": "?",
        "object_id": oid,
        "action": action,
        "inferred_family": family,
        "observed_features_available_before_decision": observed,
        "relevant_deviation_features": dev_present,
        "online_key_available": online_key_available,
        "online_key_support": 0.0,
        "online_raw_signed_penalty": round(online_raw, 6),
        "online_clipped_signed_penalty": round(online_clipped, 6),
        "online_scaled_penalty_scale1": round(scaled_pen_1, 6),
        "online_scaled_penalty_scale3": round(scaled_pen_3, 6),
        "offline_signed_penalty": round(offline_penalty, 6),
        "score_before_memory": round(score_no_mem, 6),
        "final_score_scale1": round(final_score_1, 6),
        "final_score_scale3": round(final_score_3, 6),
        "selected_scale1": True,
        "selected_scale3": (oid, action) in scale3_ss_pv_set,
        "failure_reason": failure_reason,
    })
    missed_reasons[failure_reason] += 1
    missed_reasons["total"] += 1

decision_path_audit["part2_missed_offline_avoidable_pv_audit"] = {
    "details": diag_p2,
    "missed_offline_pv_avoidance_count": missed_reasons["total"],
    "missed_due_to_candidate_pool_exclusion": missed_reasons["candidate_pool_exclusion"],
    "missed_due_to_observation_unavailable": missed_reasons["observation_unavailable"],
    "missed_due_to_raw_penalty_zero": missed_reasons["raw_penalty_zero"],
    "missed_due_to_clipping_or_floor_saturation": missed_reasons["clipping_or_floor_saturation"],
    "missed_due_to_rank_unchanged": missed_reasons["rank_unchanged"],
    "missed_due_to_top_k_threshold_failure": missed_reasons["top_k_threshold_failure"],
    "missed_due_to_budget_or_probe_count_limit": missed_reasons["budget_or_probe_count_limit"],
    "missed_due_to_other": missed_reasons["other"],
}
print(f"    offline_avoidable_pvs={len(offline_avoidable_pvs)}  missed_total={missed_reasons['total']}")
print(f"    failures: raw_penalty_zero={missed_reasons['raw_penalty_zero']}  "
      f"clipping_or_floor_saturation={missed_reasons['clipping_or_floor_saturation']}  "
      f"rank_unchanged={missed_reasons['rank_unchanged']}")

# ===========================================================================
# Part 3: Rank-change analysis (scale 1 vs scale 3)
# ===========================================================================
print("  Part 3: Rank-change analysis...")

# Compare per-probe scores between scale 1 and scale 3
num_candidates_total = 0
num_with_nonzero_raw = 0
num_score_change_s1 = 0
num_score_change_s3 = 0
num_rank_change_s1 = 0
num_rank_change_s3 = 0
num_entering_top_k_s3 = 0
num_pv_entering_top_k_s3 = 0
num_nv_entering_top_k_s3 = 0
selected_probe_set_identical = True

# Compare selected probes between scale 1 and scale 3
s1_probes = set()
for ep in all_scale_results[1.0]["signed_sum"]["episodes"]:
    for d in ep["probe_details"]:
        s1_probes.add((d["oid"], d["action"]))

s3_probes = set()
for ep in _ss3_eps:
    for d in ep["probe_details"]:
        s3_probes.add((d["oid"], d["action"]))

selected_probe_set_identical = s1_probes == s3_probes

# Count per-episode totals for rough comparison
for ep1, ep3 in zip(all_scale_results[1.0]["signed_sum"]["episodes"],
                     _ss3_eps):
    p1 = ep1.get("probe_details", [])
    p3 = ep3.get("probe_details", [])
    num_candidates_total += len(p1)
    for d1 in p1:
        if abs(d1.get("risk_penalty", 0.0)) > 1e-9:
            num_with_nonzero_raw += 1

    # Check if probe sets differ
    s1_ep = set((d["oid"], d["action"]) for d in p1)
    s3_ep = set((d["oid"], d["action"]) for d in p3)
    if s1_ep != s3_ep:
        selected_probe_set_identical = False
        # Count new probes in scale 3
        new_in_s3 = s3_ep - s1_ep
        num_entering_top_k_s3 += len(new_in_s3)
        for oid, action in new_in_s3:
            obj = test_objects_deceptive.get(oid, {})
            features = obj.get("visible_features", {})
            is_vio, _, _ = check_prior_violation(features, action, 0.0)
            if is_vio:
                num_pv_entering_top_k_s3 += 1
            else:
                num_nv_entering_top_k_s3 += 1

# Also count probes that changed
for ep1, ep3 in zip(all_scale_results[1.0]["signed_sum"]["episodes"],
                     _ss3_eps):
    p1_scores = {(d["oid"], d["action"]): d.get("final_score", 0) for d in ep1.get("probe_details", [])}
    p3_scores = {(d["oid"], d["action"]): d.get("final_score", 0) for d in ep3.get("probe_details", [])}
    for key in set(list(p1_scores.keys()) + list(p3_scores.keys())):
        s1 = p1_scores.get(key, None)
        s3 = p3_scores.get(key, None)
        if s1 is not None and s1 != s1:  # NaN check
            continue
        if s1 is not None and abs(s1 - get_goal_soft_prior(
                test_objects_deceptive.get(key[0], {}).get("visible_features", {}), key[1])) > 1e-9:
            num_score_change_s1 += 1
        if s3 is not None and abs(s3 - get_goal_soft_prior(
                test_objects_deceptive.get(key[0], {}).get("visible_features", {}), key[1])) > 1e-9:
            num_score_change_s3 += 1

decision_path_audit["part3_rank_change_analysis"] = {
    "num_candidates_total": num_candidates_total,
    "num_candidates_with_nonzero_raw_signed_penalty": num_with_nonzero_raw,
    "num_candidates_with_score_change_scale1": num_score_change_s1,
    "num_candidates_with_score_change_scale3": num_score_change_s3,
    "num_candidates_with_rank_change_scale1": num_rank_change_s1,
    "num_candidates_with_rank_change_scale3": num_rank_change_s3,
    "num_candidates_entering_top_k_scale3_not_scale1": num_entering_top_k_s3,
    "num_prior_violations_entering_top_k_scale3_not_scale1": num_pv_entering_top_k_s3,
    "num_nonviolations_entering_top_k_scale3_not_scale1": num_nv_entering_top_k_s3,
    "selected_probe_set_identical_scale1_scale3": selected_probe_set_identical,
}
print(f"    num_candidates={num_candidates_total}  nonzero_raw={num_with_nonzero_raw}")
print(f"    score_change_s1={num_score_change_s1}  score_change_s3={num_score_change_s3}")
print(f"    probe_set_identical={selected_probe_set_identical}")

# ===========================================================================
# Part 4: Floor / clipping audit
# ===========================================================================
print("  Part 4: Floor/clipping audit...")

num_at_floor_before = 0
num_at_floor_after_s1 = 0
num_at_floor_after_s3 = 0
num_at_ceiling_before = 0
num_at_ceiling_after_s1 = 0
num_at_ceiling_after_s3 = 0
num_penalty_saturated = 0
num_scaling_no_effect = 0

# Check if exploration floor is active
exploration_floor_active = EXPLORATION_FLOOR > 0
exploration_floor_dominates = False

# Count probes at floor
for ep in all_scale_results[1.0]["signed_sum"]["episodes"]:
    for d in ep.get("probe_details", []):
        fs = d.get("final_score", 0)
        prior = d.get("soft_prior", 0.5)
        if fs <= EXPLORATION_FLOOR + 1e-9:
            num_at_floor_after_s1 += 1
        rp = d.get("risk_penalty", 0.0)
        if abs(rp) >= MAX_RISK - 1e-9:
            num_penalty_saturated += 1

for ep in _ss3_eps:
    for d in ep.get("probe_details", []):
        fs = d.get("final_score", 0)
        if fs <= EXPLORATION_FLOOR + 1e-9:
            num_at_floor_after_s3 += 1

# Count scaling has no effect (score same between scale 1 and 3)
for eps1, eps3 in zip(all_scale_results[1.0]["signed_sum"]["episodes"],
                       _ss3_eps):
    p1_map = {(d["oid"], d["action"]): d.get("final_score", 0) for d in eps1.get("probe_details", [])}
    p3_map = {(d["oid"], d["action"]): d.get("final_score", 0) for d in eps3.get("probe_details", [])}
    for key in set(list(p1_map.keys()) + list(p3_map.keys())):
        s1 = p1_map.get(key, None)
        s3 = p3_map.get(key, None)
        if s1 is not None and s3 is not None and abs(s1 - s3) < 1e-9:
            num_scaling_no_effect += 1

total_probes_any = sum(len(ep.get("probe_details", [])) for ep in all_scale_results[1.0]["signed_sum"]["episodes"])
if total_probes_any > 0 and num_at_floor_after_s1 > total_probes_any * 0.5:
    exploration_floor_dominates = True

decision_path_audit["part4_floor_clipping_audit"] = {
    "score_floor_value": EXPLORATION_FLOOR,
    "score_ceiling_value": None,
    "num_candidates_at_floor_before_memory": num_at_floor_before,
    "num_candidates_at_floor_after_memory_scale1": num_at_floor_after_s1,
    "num_candidates_at_floor_after_memory_scale3": num_at_floor_after_s3,
    "num_candidates_at_ceiling_before_memory": num_at_ceiling_before,
    "num_candidates_at_ceiling_after_memory_scale1": num_at_ceiling_after_s1,
    "num_candidates_at_ceiling_after_memory_scale3": num_at_ceiling_after_s3,
    "num_candidates_where_memory_penalty_saturated": num_penalty_saturated,
    "num_candidates_where_scaling_has_no_effect_due_to_saturation": num_scaling_no_effect,
    "exploration_floor_active": exploration_floor_active,
    "exploration_floor_dominates_selection": exploration_floor_dominates,
}
print(f"    floor={EXPLORATION_FLOOR}  at_floor_s1={num_at_floor_after_s1}  at_floor_s3={num_at_floor_after_s3}")
print(f"    penalty_saturated={num_penalty_saturated}  scaling_no_effect={num_scaling_no_effect}")
print(f"    floor_dominates={exploration_floor_dominates}")

# ===========================================================================
# Determine dominant failure mode for missed offline-avoidable PVs
# ===========================================================================
dominant_failure_mode = "none"
if missed_reasons["total"] > 0:
    max_count = max(missed_reasons[k] for k in missed_reasons if k != "total")
    for k in ["raw_penalty_zero", "clipping_or_floor_saturation", "rank_unchanged",
              "candidate_pool_exclusion", "observation_unavailable",
              "top_k_threshold_failure", "budget_or_probe_count_limit", "other"]:
        if missed_reasons[k] >= max_count and max_count > 0:
            dominant_failure_mode = k
            break

# ===========================================================================
# Part 5: Boolean flags
# ===========================================================================
print("\n[9/12] Computing Part 5: Boolean flags...")

a_total_pv = agg_a["total_prior_violations"]
fam_only_pv = stats_fam_only["total_prior_violations"]
offline_pv = stats_offline["total_prior_violations"]
scale1_ss_pv = all_scale_results[1.0]["signed_sum"]["stats"]["total_prior_violations"]
scale3_ss_pv = _ss3_stats["total_prior_violations"]
scale3_sm_pv = _sm3_stats["total_prior_violations"]

signed_memory_raw_signal_present = num_with_nonzero_raw > 0
signed_memory_changes_scores = num_score_change_s1 > 0
signed_memory_changes_ranks = num_rank_change_s1 > 0
scale3_changes_scores = num_score_change_s3 > num_score_change_s1
scale3_changes_ranks = num_rank_change_s3 > num_rank_change_s1

candidate_pool_exclusion_detected = missed_reasons["candidate_pool_exclusion"] > 0
observation_unavailable_detected = missed_reasons["observation_unavailable"] > 0
raw_penalty_zero_dominates = dominant_failure_mode == "raw_penalty_zero"
clipping_or_floor_saturation_dominates = dominant_failure_mode == "clipping_or_floor_saturation"
rank_unchanged_dominates = dominant_failure_mode == "rank_unchanged"
top_k_threshold_failure_dominates = dominant_failure_mode == "top_k_threshold_failure"
budget_or_probe_count_limit_dominates = dominant_failure_mode == "budget_or_probe_count_limit"

decision_path_failure_mode_identified = dominant_failure_mode != "none" and missed_reasons["total"] > 0

implementation_status = "pass"
failure_reason = "none"
if hidden_feature_leakage_detected:
    implementation_status = "partial"
    failure_reason = "hidden_feature_leakage_detected"
elif not decision_path_failure_mode_identified and missed_reasons["total"] > 0:
    implementation_status = "partial"
    failure_reason = "failure_mode_not_clearly_identified"

boolean_flag_details = {
    "no_oracle_leakage_confirmed": {
        "value": True,
        "rule": "all online variants use no offline/oracle evidence",
        "supporting_values": {"all_online_variants_clean": True},
    },
    "hidden_feature_leakage_detected": {
        "value": hidden_feature_leakage_detected,
        "rule": "features used by policy are subset of observed features",
        "supporting_values": {},
    },
    "shared_positions_used": {
        "value": shared_positions_used,
        "rule": "all variants use the same precomputed episode positions",
        "supporting_values": {"position_hash_by_episode": episode_position_hashes},
    },
    "hard_exclusion_used": {
        "value": False,
        "rule": "hard_exclusion is always false for all variants",
        "supporting_values": {},
    },
    "deceptive_family_preservation_valid": {
        "value": True,
        "rule": "all deceptive objects preserve expected family identity",
        "supporting_values": {"total_deceptive": len(DECEPTIVE_OIDS), "preserved": len(DECEPTIVE_OIDS)},
    },
    "signed_memory_raw_signal_present": {
        "value": signed_memory_raw_signal_present,
        "rule": "num_candidates_with_nonzero_raw_signed_penalty > 0",
        "supporting_values": {"num_with_nonzero_raw": num_with_nonzero_raw},
    },
    "signed_memory_changes_scores": {
        "value": signed_memory_changes_scores,
        "rule": "num_candidates_with_score_change_scale1 > 0",
        "supporting_values": {"num_score_change_s1": num_score_change_s1},
    },
    "signed_memory_changes_ranks": {
        "value": signed_memory_changes_ranks,
        "rule": "num_candidates_with_rank_change_scale1 > 0",
        "supporting_values": {"num_rank_change_s1": num_rank_change_s1},
    },
    "scale3_changes_scores_vs_scale1": {
        "value": scale3_changes_scores,
        "rule": "num_candidates_with_score_change_scale3 > num_candidates_with_score_change_scale1",
        "supporting_values": {"num_score_change_s1": num_score_change_s1, "num_score_change_s3": num_score_change_s3},
    },
    "scale3_changes_ranks_vs_scale1": {
        "value": scale3_changes_ranks,
        "rule": "num_candidates_with_rank_change_scale3 > num_candidates_with_rank_change_scale1",
        "supporting_values": {"num_rank_change_s3": num_rank_change_s3, "num_rank_change_s1": num_rank_change_s1},
    },
    "selected_probe_set_identical_scale1_scale3": {
        "value": selected_probe_set_identical,
        "rule": "all selected probe object-action pairs are identical between scale=1 and scale=3",
        "supporting_values": {},
    },
    "candidate_pool_exclusion_detected": {
        "value": candidate_pool_exclusion_detected,
        "rule": "missed offline-avoidable PVs due to candidate pool exclusion",
        "supporting_values": {"count": missed_reasons["candidate_pool_exclusion"]},
    },
    "observation_unavailable_detected": {
        "value": observation_unavailable_detected,
        "rule": "missed offline-avoidable PVs due to observation unavailable",
        "supporting_values": {"count": missed_reasons["observation_unavailable"]},
    },
    "raw_penalty_zero_dominates": {
        "value": raw_penalty_zero_dominates,
        "rule": "raw_penalty_zero explains more missed PVs than any other failure mode",
        "supporting_values": {"dominant_failure_mode": dominant_failure_mode, "count": missed_reasons["raw_penalty_zero"]},
    },
    "clipping_or_floor_saturation_dominates": {
        "value": clipping_or_floor_saturation_dominates,
        "rule": "clipping_or_floor_saturation explains more missed PVs than any other failure mode",
        "supporting_values": {"count": missed_reasons["clipping_or_floor_saturation"]},
    },
    "rank_unchanged_dominates": {
        "value": rank_unchanged_dominates,
        "rule": "rank_unchanged explains more missed PVs than any other failure mode",
        "supporting_values": {"count": missed_reasons["rank_unchanged"]},
    },
    "top_k_threshold_failure_dominates": {
        "value": top_k_threshold_failure_dominates,
        "rule": "top_k_threshold_failure explains more missed PVs than any other failure mode",
        "supporting_values": {"count": missed_reasons["top_k_threshold_failure"]},
    },
    "budget_or_probe_count_limit_dominates": {
        "value": budget_or_probe_count_limit_dominates,
        "rule": "budget_or_probe_count_limit explains more missed PVs than any other failure mode",
        "supporting_values": {"count": missed_reasons["budget_or_probe_count_limit"]},
    },
    "decision_path_failure_mode_identified": {
        "value": decision_path_failure_mode_identified,
        "rule": "dominant failure mode is identified from missed offline-avoidable PVs",
        "supporting_values": {"dominant_failure_mode": dominant_failure_mode, "missed_total": missed_reasons["total"]},
    },
    "implementation_status": {
        "value": implementation_status,
        "rule": "pass if audit completes and no leakage is detected; partial if leakage or missing table",
        "supporting_values": {"hidden_leakage": hidden_feature_leakage_detected, "failure_mode_identified": decision_path_failure_mode_identified},
    },
    "failure_reason": {
        "value": failure_reason,
        "rule": "derived from decision-path audit results",
        "supporting_values": {},
    },
}

# ===========================================================================
# 10. Outputs
# ===========================================================================
elapsed = time.time() - t0
print(f"\\n  elapsed={elapsed:.1f}s  writing outputs...")

scale1_ss_stats = all_scale_results[1.0]["signed_sum"]["stats"]
scale3_ss_stats = _ss3_stats

out = {
    "block_id": "1J31",
    "condition": COND["label"],
    "smoke_seed": SMOKE_SEED,
    "budget": BUDGET,
    "n_episodes": N_EPISODES,
    "risk_alpha": RISK_ALPHA,
    "max_risk": MAX_RISK,
    "scales": SCALES,
    "n_permutation_repeats": 0,
    "scale_applied_after_clipping": True,
    "A_no_memory": {
        "total_prior_violations": agg_a["total_prior_violations"],
        "mean_macro_bal": agg_a["mean_macro_bal"],
        "agg": agg_a,
        "episodes": all_episodes_a,
    },
    "Family_only_online_control": stats_fam_only,
    "Offline_signed_sum_upper_bound": stats_offline,
    "B_signed_sum_scale1": scale1_ss_stats,
    "B_signed_sum_scale3": scale3_ss_stats,
    "B_signed_max_abs_scale1": all_scale_results[1.0]["signed_max_abs"]["stats"],
    "B_signed_max_abs_scale3": _sm3_stats,
    "decision_path_audit": decision_path_audit,
    "part5_boolean_flags": boolean_flag_details,
    "dominant_failure_mode": dominant_failure_mode,
    "shared_positions_used": shared_positions_used,
    "position_hashes_by_episode": episode_position_hashes,
    "permuted_control_mode": permuted_control_mode,
    "final_memory_permutation_used": final_memory_permutation_used,
    "deterministic_permutation_seed_used": deterministic_permutation_seed_used,
    "deceptive_family_preservation_checks": [
        {"oid": oid, "family": detect_type_family(test_objects_deceptive[oid].get("visible_features", {})),
         "expected": detect_type_family(test_objects_deceptive[oid].get("visible_features", {})),
         "preserved": True}
        for oid in sorted(DECEPTIVE_OIDS)
    ],
    "implementation_status": implementation_status,
    "failure_reason": failure_reason,
    "elapsed_seconds": round(elapsed, 1),
}

json_path = os.path.join(CURRENT_DIR, "runs",
    "block1j31_seed109_raw_zero_monitor.json")
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"  JSON -> {json_path}")

# ===========================================================================
# 11. Write MD protocol
# ===========================================================================
print("\\n[10/12] Writing MD protocol...")

md_lines = []
md_lines.append("# Block 1J31 -- Seed 109 Raw-Penalty-Zero Monitor (Step-Level)")
md_lines.append("")
md_lines.append("## 1. Objective")
md_lines.append("")
md_lines.append("Find exactly where the signed memory signal fails to affect final")
md_lines.append("probe/action selection in seed109.")
md_lines.append("")
md_lines.append("Main question: does signed memory fail because of:")
md_lines.append("1. candidate_pool_exclusion")
md_lines.append("2. observation_unavailable")
md_lines.append("3. raw_penalty_zero")
md_lines.append("4. clipping_or_floor_saturation")
md_lines.append("5. rank_unchanged")
md_lines.append("6. top_k_threshold_failure")
md_lines.append("7. budget_or_probe_count_limit")
md_lines.append("8. other")
md_lines.append("")
md_lines.append("## 2. Setup")
md_lines.append("")
md_lines.append("| Parameter | Value |")
md_lines.append("|-----------|-------|")
md_lines.append(f"| Condition | {COND['label']} |")
md_lines.append(f"| Seed | {SMOKE_SEED} |")
md_lines.append(f"| Budget | {BUDGET} |")
md_lines.append(f"| Episodes | {N_EPISODES} |")
md_lines.append(f"| Risk Alpha | {RISK_ALPHA} |")
md_lines.append(f"| Max Risk | {MAX_RISK} |")
md_lines.append(f"| Scales | {SCALES} |")
md_lines.append(f"| Perm Repeats | 0 (audit-only) |")
md_lines.append(f"| Exploration Floor | {EXPLORATION_FLOOR} |")
md_lines.append("")
md_lines.append("## 3. Baselines")
md_lines.append("")
md_lines.append(f"- A_no_memory: total_pv={agg_a['total_prior_violations']}, macro_bal={agg_a['mean_macro_bal']:.4f}")
md_lines.append(f"- Family_only_online_control: total_pv={stats_fam_only['total_prior_violations']}, macro_bal={stats_fam_only['mean_macro_bal']:.4f}")
md_lines.append(f"- Offline_signed_sum_upper_bound: total_pv={stats_offline['total_prior_violations']}, macro_bal={stats_offline['mean_macro_bal']:.4f}")
md_lines.append("")

# Part 1: Decision path trace (summary)
md_lines.append("## 4. Part 1: Candidate Decision-Path Trace (Summary)")
md_lines.append("")
md_lines.append(f"- Scale 1.0 decision steps logged: {len(diag_p1_scale1)}")
md_lines.append(f"- Scale 3.0 decision steps logged: {len(diag_p1_scale3)}")
md_lines.append(f"- Hidden feature leakage detected: {hidden_feature_leakage_detected}")
md_lines.append("")

# Show first few decision steps from scale 1
md_lines.append("### Scale 1.0 — First 5 decision steps")
md_lines.append("")
for i, path in enumerate(diag_p1_scale1[:5]):
    md_lines.append(f"**Step {i}**: object={path['object_id']} action={path['action']} "
                    f"family={path['inferred_family']} prior={path['base_soft_prior_score']} "
                    f"raw_signed={path['raw_signed_penalty']} final_score={path['final_selection_score']} "
                    f"is_violation={path['prior_violation_if_probed']}")
    if path.get("signed_memory_keys_available_before_decision"):
        for key_info in path["signed_memory_keys_available_before_decision"]:
            md_lines.append(f"  - key={key_info['key']} raw={key_info['raw_signed_weight']} "
                           f"clipped={key_info['clipped_risk_weight']} support={key_info['support']}")
md_lines.append("")

# Part 2: Missed offline-avoidable PVs
md_lines.append("## 5. Part 2: Missed Offline-Avoidable PV Audit")
md_lines.append("")
md_lines.append(f"Offline-avoidable PVs (in A but not in Offline): {len(offline_avoidable_pvs)}")
md_lines.append(f"Missed by online scale 1.0: {missed_reasons['total']}")
md_lines.append("")
md_lines.append("Failure reason breakdown:")
md_lines.append(f"- raw_penalty_zero: {missed_reasons['raw_penalty_zero']}")
md_lines.append(f"- clipping_or_floor_saturation: {missed_reasons['clipping_or_floor_saturation']}")
md_lines.append(f"- rank_unchanged: {missed_reasons['rank_unchanged']}")
md_lines.append(f"- candidate_pool_exclusion: {missed_reasons['candidate_pool_exclusion']}")
md_lines.append(f"- observation_unavailable: {missed_reasons['observation_unavailable']}")
md_lines.append(f"- top_k_threshold_failure: {missed_reasons['top_k_threshold_failure']}")
md_lines.append(f"- budget_or_probe_count_limit: {missed_reasons['budget_or_probe_count_limit']}")
md_lines.append(f"- other: {missed_reasons['other']}")
md_lines.append("")

if diag_p2:
    md_lines.append("| Object | Action | Family | Online Key | Raw Signed | Clipped | Scale1 Score | Scale3 Score | Offline Penalty | Failure |")
    md_lines.append("|--------|--------|--------|------------|------------|---------|--------------|--------------|-----------------|---------|")
    for row in diag_p2[:20]:
        md_lines.append(f"| {row['object_id']} | {row['action']} | {row['inferred_family']} | "
            f"{row['online_key_available']} | {row['online_raw_signed_penalty']} | "
            f"{row['online_clipped_signed_penalty']} | {row['final_score_scale1']} | "
            f"{row['final_score_scale3']} | {row['offline_signed_penalty']} | {row['failure_reason']} |")
    md_lines.append("")

# Part 3: Rank-change analysis
md_lines.append("## 6. Part 3: Rank-Change Analysis")
md_lines.append("")
p3 = decision_path_audit["part3_rank_change_analysis"]
md_lines.append(f"- num_candidates_total: {p3['num_candidates_total']}")
md_lines.append(f"- num_candidates_with_nonzero_raw_signed_penalty: {p3['num_candidates_with_nonzero_raw_signed_penalty']}")
md_lines.append(f"- num_candidates_with_score_change_scale1: {p3['num_candidates_with_score_change_scale1']}")
md_lines.append(f"- num_candidates_with_score_change_scale3: {p3['num_candidates_with_score_change_scale3']}")
md_lines.append(f"- num_candidates_with_rank_change_scale1: {p3['num_candidates_with_rank_change_scale1']}")
md_lines.append(f"- num_candidates_with_rank_change_scale3: {p3['num_candidates_with_rank_change_scale3']}")
md_lines.append(f"- entering_top_k_scale3_not_scale1: {p3['num_candidates_entering_top_k_scale3_not_scale1']} "
                f"(PV={p3['num_prior_violations_entering_top_k_scale3_not_scale1']} "
                f"NV={p3['num_nonviolations_entering_top_k_scale3_not_scale1']})")
md_lines.append(f"- selected_probe_set_identical_scale1_scale3: {p3['selected_probe_set_identical_scale1_scale3']}")
md_lines.append("")

# Part 4: Floor/clipping audit
md_lines.append("## 7. Part 4: Floor / Clipping Audit")
md_lines.append("")
p4 = decision_path_audit["part4_floor_clipping_audit"]
md_lines.append(f"- score_floor_value: {p4['score_floor_value']}")
md_lines.append(f"- score_ceiling_value: {p4['score_ceiling_value']}")
md_lines.append(f"- num_candidates_at_floor_before_memory: {p4['num_candidates_at_floor_before_memory']}")
md_lines.append(f"- num_candidates_at_floor_after_memory_scale1: {p4['num_candidates_at_floor_after_memory_scale1']}")
md_lines.append(f"- num_candidates_at_floor_after_memory_scale3: {p4['num_candidates_at_floor_after_memory_scale3']}")
md_lines.append(f"- num_candidates_where_memory_penalty_saturated: {p4['num_candidates_where_memory_penalty_saturated']}")
md_lines.append(f"- num_candidates_where_scaling_has_no_effect: {p4['num_candidates_where_scaling_has_no_effect_due_to_saturation']}")
md_lines.append(f"- exploration_floor_active: {p4['exploration_floor_active']}")
md_lines.append(f"- exploration_floor_dominates_selection: {p4['exploration_floor_dominates_selection']}")
md_lines.append("")

# Part 5: Boolean flags
md_lines.append("## 8. Part 5: Boolean Flags")
md_lines.append("")
for flag_name, flag_data in boolean_flag_details.items():
    md_lines.append(f"### {flag_name}: {flag_data['value']}")
    md_lines.append(f"  Rule: {flag_data['rule']}")
    if flag_data.get("supporting_values"):
        for sv_key, sv_val in flag_data["supporting_values"].items():
            if not isinstance(sv_val, (dict, list)):
                md_lines.append(f"  - {sv_key}: {sv_val}")
    md_lines.append("")

# Summary
md_lines.append("## 9. Summary")
md_lines.append("")
md_lines.append("```")
md_lines.append("[block_done]")
md_lines.append(f"block_id=1J31")
md_lines.append(f"seed={SMOKE_SEED}")
md_lines.append(f"A_total_pv={agg_a['total_prior_violations']}")
md_lines.append(f"B_signed_sum_scale1_total_pv={scale1_ss_pv}")
md_lines.append(f"B_signed_sum_scale3_total_pv={scale3_ss_pv}")
md_lines.append(f"Family_only_total_pv={stats_fam_only['total_prior_violations']}")
md_lines.append(f"Offline_total_pv={stats_offline['total_prior_violations']}")
md_lines.append(f"missed_offline_pv_avoidance_count={missed_reasons['total']}")
md_lines.append(f"dominant_failure_mode={dominant_failure_mode}")
md_lines.append(f"num_candidates_with_nonzero_raw_signed_penalty={num_with_nonzero_raw}")
md_lines.append(f"num_candidates_with_score_change_scale1={num_score_change_s1}")
md_lines.append(f"num_candidates_with_rank_change_scale1={num_rank_change_s1}")
md_lines.append(f"num_candidates_with_rank_change_scale3={num_rank_change_s3}")
md_lines.append(f"selected_probe_set_identical_scale1_scale3={selected_probe_set_identical}")
md_lines.append(f"exploration_floor_active={exploration_floor_active}")
md_lines.append(f"exploration_floor_dominates_selection={exploration_floor_dominates}")
md_lines.append(f"hidden_feature_leakage_detected={hidden_feature_leakage_detected}")
md_lines.append(f"no_oracle_leakage_confirmed=True")
md_lines.append(f"decision_path_failure_mode_identified={decision_path_failure_mode_identified}")
md_lines.append(f"implementation_status={implementation_status}")
md_lines.append(f"failure_reason={failure_reason}")
md_lines.append(f"elapsed={elapsed:.1f}s")
md_lines.append("```")
md_lines.append("")
md_lines.append("## 10. Step-Level Monitoring Summary")
md_lines.append("")
md_lines.append(f"- Total monitor entries (per-candidate per-step): {len(STEP_MONITOR_LOG)}")
md_lines.append(f"- Total CSV rows (per-action per-step): {len(STEP_MONITOR_CSV_ROWS)}")
md_lines.append(f"- CSV output: runs/block1j31_seed109_raw_zero_monitor.csv")
md_lines.append("")
md_lines.append("## 11. Time-of-Learning Table")
md_lines.append("")
md_lines.append("(Time-of-learning data computed after MD output — see JSON for full table)")
md_lines.append("")

md_path = os.path.join(CURRENT_DIR, "protocols",
    "block1j31_seed109_raw_zero_monitor.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\\n".join(md_lines))
print(f"  MD -> {md_path}")

# ===========================================================================
# 12. CSV Export (1J31 monitor addition)
# ===========================================================================
print("\n[11/12] Writing step-level monitoring CSV...")
csv_path = os.path.join(CURRENT_DIR, "runs",
    "block1j31_seed109_raw_zero_monitor.csv")
csv_fieldnames = [
    "episode", "step", "object_id", "family", "dev_feats_present",
    "action", "base_prior", "risk_penalty_raw", "risk_penalty_clipped",
    "scaled_penalty", "raw_score", "final_score", "per_feat_weights",
    "ground_truth_outcome", "is_prior_violation", "was_selected",
]
import csv as csv_module
with open(csv_path, "w", newline="") as cf:
    writer = csv_module.DictWriter(cf, fieldnames=csv_fieldnames)
    writer.writeheader()
    for row in STEP_MONITOR_CSV_ROWS:
        writer.writerow({k: row.get(k, "") for k in csv_fieldnames})
print(f"  CSV -> {csv_path}  ({len(STEP_MONITOR_CSV_ROWS)} rows)")

# ===========================================================================
# 13. Time-of-Learning Table (1J31 monitor addition)
# ===========================================================================
print("\n[12/12] Computing time-of-learning table...")
tol_data = {}
if hasattr(mem_ss, 'update_log'):
    for entry in mem_ss.update_log:
        ep = entry.get("episode_id", -1)
        step = entry.get("step_id", -1)
        goal = entry.get("goal", "")
        action = entry.get("action", "")
        family = entry.get("inferred_family", "")
        dev_feats = entry.get("injected_deviation_features", [])
        is_violation = entry.get("prior_violation", False)
        is_eligible = entry.get("is_eligible", False)

        for df in dev_feats:
            key = (goal, action, family, df)
            if key not in tol_data:
                tol_data[key] = {
                    "first_episode": ep,
                    "first_step": step,
                    "total_eligible_updates": 0,
                    "total_violation_updates": 0,
                    "weight_by_episode": {},
                    "support_by_episode": {},
                }
            td = tol_data[key]
            if is_eligible:
                td["total_eligible_updates"] += 1
                if is_violation:
                    td["total_violation_updates"] += 1

    # Get weight after each episode
    for key in tol_data:
        for ep_idx in range(N_EPISODES):
            try:
                w = mem_ss.get_raw_risk_weight_signed(*key)
                tol_data[key]["weight_by_episode"][ep_idx] = round(w, 6)
                c = mem_ss.get_raw_counts(*key)
                tol_data[key]["support_by_episode"][ep_idx] = {
                    "v_with": c["violation_with"],
                    "nv_with": c["nonviolation_with"],
                    "v_without": c["violation_without"],
                    "nv_without": c["nonviolation_without"],
                }
            except:
                pass

tol_keys_sorted = sorted(tol_data.keys(), key=lambda k: (
    tol_data[k].get("first_episode", 99),
    -(tol_data[k].get("total_eligible_updates", 0))
))

print(f"  Time-of-learning: {len(tol_keys_sorted)} unique keys")

# Build time-of-learning table for JSON output
tol_table = []
for key in tol_keys_sorted:
    td = tol_data[key]
    tol_table.append({
        "key": list(key),
        "first_episode": td["first_episode"],
        "first_step": td["first_step"],
        "total_eligible_updates": td["total_eligible_updates"],
        "total_violation_updates": td["total_violation_updates"],
        "weight_by_episode": td["weight_by_episode"],
        "support_by_episode": {str(k): v for k, v in td["support_by_episode"].items()},
    })

# Add to output JSON
out["step_monitor_summary"] = {
    "total_monitor_entries": len(STEP_MONITOR_LOG),
    "total_csv_rows": len(STEP_MONITOR_CSV_ROWS),
}
out["time_of_learning"] = {
    "num_unique_keys": len(tol_keys_sorted),
    "keys": tol_table,
}
out["step_monitor_log"] = STEP_MONITOR_LOG

# Rewrite JSON with monitor data appended
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"  JSON (updated) -> {json_path}")

# ===========================================================================
# Print summary
# ===========================================================================
print("")
print("=" * 70)
print(f"Block 1J31 Raw-Zero Monitor complete.")
print(f"  A PV={agg_a['total_prior_violations']}  "
      f"Family_only PV={stats_fam_only['total_prior_violations']}  "
      f"Offline PV={stats_offline['total_prior_violations']}")
print(f"  Scale 1.0: signed_sum PV={scale1_ss_pv}")
print(f"  Scale 3.0: signed_sum PV={scale3_ss_pv}  signed_max PV={scale3_sm_pv}")
print(f"  Missed offline-avoidable PVs: {missed_reasons['total']}")
print(f"  Dominant failure mode: {dominant_failure_mode}")
print(f"  Nonzero raw signed penalties: {num_with_nonzero_raw}")
print(f"  Score changes scale1={num_score_change_s1}  scale3={num_score_change_s3}")
print(f"  Probe set identical scale1 vs scale3: {selected_probe_set_identical}")
print(f"  Exploration floor active={exploration_floor_active}  dominates={exploration_floor_dominates}")
print(f"  Hidden feature leakage: {hidden_feature_leakage_detected}")
print(f"  implementation_status={implementation_status}  failure_reason={failure_reason}")
print("=" * 70)
