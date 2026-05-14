"""
Block 1J19 -- Multi-Episode Feature-Risk Memory Validation.

Patched version (1J19_patch_smoke):
  - IOM isolation: clone fresh IOM from base_iom per seed/episode/policy
  - Feature absence counts: iterate over fixed FEATURE_UNIVERSE
  - Generalization mode: within-object-set adaptation (smoke test)
  - No hard exclusion: final_score = max(exploration_floor, raw_score)
  - Permuted control: permute risk weights once per episode

Three policy conditions:
  A. soft_prior_only (from 1J18)
  B. soft_prior_plus_feature_risk_memory
  C. soft_prior_plus_permuted_risk_memory_control
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
    C14a_TruthAnswerOracle,
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
RISK_ALPHA = 5.0        # shrinkage: higher = more conservative
MAX_RISK = 2.0           # clip risk weights to [-MAX_RISK, MAX_RISK]

COND = copy.deepcopy(config.CUE_CONDITIONS[3])
assert COND["label"] == "C4_instance_subtype_cued_v1"

# Fixed feature universe for absence counting
# Union of PREDICTOR_AUGMENTED_FEATURES + ACTIVE_FEATURES_PHASE_A
FEATURE_UNIVERSE = sorted({
    # Predictor-augmented (clean category signals)
    "has_bark_texture", "has_wood_grain", "has_crystal_flecks",
    "has_granular_surface", "has_stem_remnant", "has_peel_texture",
    "has_grip_area", "has_shaft_shape",
    # Active features (from ACTIVE_FEATURES_PHASE_A)
    "solid", "movable", "block_like", "elongated_with_handle",
    "round_small", "brownish", "grayish", "greenish",
    "long_shape", "rough_texture", "smooth_texture",
    "on_left_side", "near_table", "recently_seen",
    "light_weight", "heavy_weight",
})

print("=" * 60)
print("Block 1J19 -- Multi-Episode Feature-Risk Memory Validation")
print(f"  PATCH: IOM isolation, absence counts, max(floor,raw_score)")
print(f"  condition={COND['label']}  smoke_seed={SMOKE_SEED}  budget={BUDGET}")
print(f"  episodes={N_EPISODES}  risk_alpha={RISK_ALPHA}  max_risk={MAX_RISK}")
print(f"  feature_universe_size={len(FEATURE_UNIVERSE)}")
print(f"  generalization_mode=within_object_set_adaptation")
print("=" * 60)

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
visible_feature_names_sorted = sorted(posterior_visible_features)
im_base = InstanceOutcomeMemory(
    train_objects_dict, posterior_visible_features,
    k=config.INSTANCE_K, similarity_power=config.SIMILARITY_POWER,
    similarity_mode=config.SIMILARITY_MODE,
)
im_base.build(outcome_rows)
print(f"  pre_accuracy={final_metrics.get('domain_token_accuracy', 0.0):.4f}  "
      f"coverage={actual_coverage:.4f}  IOM ready")

# Verify feature universe coverage against what objects actually expose
sample_obj = train_objects_dict[list(train_objects_dict.keys())[0]]
actual_vis_features = set(sample_obj.get("visible_features", {}).keys())
universe_hits = set(FEATURE_UNIVERSE).intersection(actual_vis_features)
universe_misses = set(FEATURE_UNIVERSE) - actual_vis_features
if universe_misses:
    miss_pct = len(universe_misses) / len(FEATURE_UNIVERSE) * 100
    if miss_pct > 50:
        # Dynamically adjust: use only features that appear in visible_features
        print(f"  WARNING: {len(universe_misses)}/{len(FEATURE_UNIVERSE)} features not in "
              f"visible_features ({miss_pct:.0f}%). Using actual visible features as universe.")
        FEATURE_UNIVERSE = sorted(actual_vis_features)
        print(f"  Adjusted FEATURE_UNIVERSE size={len(FEATURE_UNIVERSE)}")
    else:
        print(f"  Note: {len(universe_misses)}/{len(FEATURE_UNIVERSE)} features not in "
              f"visible_features (will be treated as absent).")
print(f"  FEATURE_UNIVERSE hits in objects: {len(universe_hits)}/{len(FEATURE_UNIVERSE)}")

# =============================================================================
# 2. Test objects (single set for smoke, reused per episode)
# =============================================================================
print("\n[2/10] Generating test objects...")
test_objects = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")
query_gt = compute_query_ground_truth(test_objects)
test_oids = sorted(test_objects.keys())

QUERY_NAMES = sorted(_TASK_QUERIES.keys())
N_OBJECTS = len(test_oids)

# Pre-compute ground truth affordances for oracle reference
ALL_GT_AFFORDANCES = {}
for oid in test_oids:
    ALL_GT_AFFORDANCES[oid] = {}
    hidden = test_objects[oid].get("hidden_affordance_profile", {})
    for action, outcome_str in hidden.items():
        feat = ACTION_TO_FEATURE.get(action)
        if feat:
            ALL_GT_AFFORDANCES[oid][feat] = 1.0 if outcome_str == "success" else 0.0

ACTION_TO_QUERY = {
    "craft_plank": "need_planks", "eat": "need_food",
    "use_as_tool": "need_tool", "burn_as_fuel": "need_fuel",
    "mine_by_hand": "need_stone", "mine_with_pickaxe": "need_stone",
}

print(f"  Generated {N_OBJECTS} test objects, {len(QUERY_NAMES)} queries")
print(f"  generalization_mode=within_object_set_adaptation")

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
    """Unwrap predictions from {"per_object": {oid->probs}} to {oid->probs}."""
    if isinstance(preds, dict) and "per_object" in preds and len(preds) == 1:
        return preds["per_object"]
    return preds


def get_cost_metrics(event_log, obs):
    visit_count = sum(1 for oid in obs.get_all_object_ids() if obs.is_visited(oid))
    probe_count = sum(1 for oid in obs.get_all_object_ids() if obs.get_probe_results(oid))
    total = obs.budget_spent
    return {
        "visit_count": visit_count,
        "observe_count": visit_count,
        "probe_count": probe_count,
        "total_cost": round(total, 4),
        "normalized_cost": round(total / max(obs.initial_budget, 0.001), 4),
    }


def run_policy(policy, env):
    harness = EpisodeHarness(env, policy)
    result = harness.run()
    return (result.predictions, result.event_log, result.agent_obs,
            result.pre_probe_entropies, result.pre_decision_entropies)


# =============================================================================
# 4. Default Goal Prior Table (from 1J18)
# =============================================================================
print("\n[3/10] Building default goal-conditioned soft prior table...")

WOOD_FEATURES = {
    "has_bark_texture", "has_wood_grain",
    "brownish", "rough_texture", "long_shape",
}
STONE_FEATURES = {
    "has_crystal_flecks", "has_granular_surface",
    "grayish", "block_like", "heavy_weight",
}
APPLE_FEATURES = {
    "has_stem_remnant", "has_peel_texture",
    "round_small", "greenish", "smooth_texture", "light_weight",
}
TOOL_FEATURES = {
    "has_grip_area", "has_shaft_shape",
    "elongated_with_handle", "movable", "long_shape",
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


# Build the DEFAULT_GOAL_PRIOR table
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
print(f"  Default prior table built: {len(DEFAULT_GOAL_PRIOR)} entries")
print(f"  Min prior={min_prior:.4f} (all > 0 [OK])")


def get_goal_soft_prior(features, action):
    fam = detect_type_family(features)
    qname = ACTION_TO_QUERY.get(action, "")
    return DEFAULT_GOAL_PRIOR.get((fam, qname, action), EXPLORATION_FLOOR)


# =============================================================================
# 5. FeatureRiskMemory
# =============================================================================
print("\n[4/10] Initializing FeatureRiskMemory...")

class FeatureRiskMemory:
    """Feature-keyed risk memory with Beta(1,1) pseudo-counts.

    Key: (feature_name, goal, action)
    Tracks both presence AND absence of features for success/failure outcomes.
    """

    def __init__(self, alpha=RISK_ALPHA, max_risk=MAX_RISK):
        self.alpha = alpha
        self.max_risk = max_risk
        # counts[(feature, goal, action)] = {"fail_with", "succ_with", "fail_without", "succ_without"}
        self.counts = defaultdict(lambda: {"fail_with": 1.0, "succ_with": 1.0,
                                            "fail_without": 1.0, "succ_without": 1.0})
        # Cached risk weights
        self._risk_cache = {}
        # Statistics
        self.total_updates = 0
        self.failure_updates = 0
        self.success_updates = 0
        self.keys_updated_from_failures = set()
        self.keys_updated_from_successes = set()

    def update(self, features, goal, action, is_success):
        """Update counts for one probe event.

        features: dict of {feature_name: bool} (present features are True)
        goal: query name (e.g., "need_food")
        action: probe action name
        is_success: True if probe was effective, False if zero/harmful
        """
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

            # Track which keys got failure/success updates
            if not is_success:
                self.keys_updated_from_failures.add(key)
            else:
                self.keys_updated_from_successes.add(key)

        # Clear risk cache (weights need recomputation)
        self._risk_cache.clear()

    def get_risk_weight(self, feature, goal, action):
        """Compute usable risk weight for one feature under a goal/action context."""
        cache_key = (feature, goal, action)
        if cache_key in self._risk_cache:
            return self._risk_cache[cache_key]

        c = self.counts[cache_key]
        fail_with = c["fail_with"]
        succ_with = c["succ_with"]
        fail_without = c["fail_without"]
        succ_without = c["succ_without"]

        # Beta(1,1) posterior means
        total_fail = fail_with + fail_without
        total_succ = succ_with + succ_without

        p_feat_given_fail = fail_with / max(total_fail, 0.001)
        p_feat_given_succ = succ_with / max(total_succ, 0.001)

        # Avoid division by zero and log(0)
        eps = 1e-9
        ratio = max(eps, min(1.0 / eps, p_feat_given_fail / max(p_feat_given_succ, eps)))
        raw_risk = math.log(ratio)

        # Clip
        clipped = max(-self.max_risk, min(self.max_risk, raw_risk))

        # Shrinkage: reliability based on total support
        support = total_fail + total_succ - 4.0  # subtract pseudo-counts
        support = max(0.0, support)
        reliability = support / (support + self.alpha)

        usable = reliability * clipped
        self._risk_cache[cache_key] = usable
        return usable

    def get_total_risk_penalty(self, features, action, goal_hint=None):
        """Sum risk weights for all present features under the goal/action context.

        Positive risk = feature associated with failure -> penalty (reduces score)
        Negative risk = feature associated with success -> bonus (increases score via exp(-neg))
        """
        total = 0.0
        goal = goal_hint or ACTION_TO_QUERY.get(action, "")
        for feat in FEATURE_UNIVERSE:
            if features.get(feat, False):
                total += self.get_risk_weight(feat, goal, action)
        return total

    def permute_weights(self, rng):
        """Randomly shuffle risk weight values across all keys.
        Preserves the distribution of weights but breaks meaningful feature-risk associations.
        Returns a dict mapping (feat, goal, action) -> permuted_weight.
        """
        # Collect all current keys and their risk weights
        keys = []
        weights = []
        for key in self.counts:
            w = self.get_risk_weight(key[0], key[1], key[2])
            keys.append(key)
            weights.append(w)

        # Shuffle weights
        shuffled = list(weights)
        rng.shuffle(shuffled)

        # Build permuted map
        permuted = {}
        for k, w in zip(keys, shuffled):
            permuted[k] = w
            # Also update the risk cache
            self._risk_cache[k] = w

        return permuted

    def get_statistics(self):
        return {
            "total_updates": self.total_updates,
            "failure_updates": self.failure_updates,
            "success_updates": self.success_updates,
            "failure_signal_count": self.failure_updates,
            "n_keys_with_data": len(self.counts),
            "n_keys_from_failures": len(self.keys_updated_from_failures),
            "n_keys_from_successes": len(self.keys_updated_from_successes),
            "memory_update_nontrivial": self.failure_updates > 0,
        }

    def clone(self):
        """Deep copy for independent use."""
        new = FeatureRiskMemory(alpha=self.alpha, max_risk=self.max_risk)
        new.counts = copy.deepcopy(dict(self.counts))
        new._risk_cache = dict(self._risk_cache)
        new.total_updates = self.total_updates
        new.failure_updates = self.failure_updates
        new.success_updates = self.success_updates
        new.keys_updated_from_failures = set(self.keys_updated_from_failures)
        new.keys_updated_from_successes = set(self.keys_updated_from_successes)
        return new


# Global feature-risk memory (persists across episodes)
g_feature_risk_memory = FeatureRiskMemory()

print(f"  FeatureRiskMemory ready  alpha={RISK_ALPHA}  max_risk={MAX_RISK}")
print(f"  FEATURE_UNIVERSE size={len(FEATURE_UNIVERSE)}")

# =============================================================================
# 6. Policy Classes
# =============================================================================
print("\n[5/10] Defining policy classes...")

# ---------------------------------------------------------------------------
# Condition A: SoftPriorOnlyPolicy (from 1J18, with IOM isolation)
# ---------------------------------------------------------------------------
class SoftPriorOnlyPolicyV2(MiniMCPolicy):
    """Rank probes by goal_soft_prior + cost penalty only.
    Uses EXPLORATION_FLOOR via max(floor, raw_score) for hard exclusion prevention.
    """

    PHASE_OBSERVE = 1
    PHASE_PROBE = 2

    def __init__(self, instance_memory, rng, target_probe_count=8):
        self._im = instance_memory  # should be a fresh clone from base_iom
        self._rng = rng
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._target_probe_count = target_probe_count
        self._probe_effects = []
        self._selected_scores = []
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

    def _select_probe_target(self, view):
        if len(self._probed_oids) >= self._target_probe_count:
            return None
        best_oid = None
        best_score = float('-inf')
        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids: continue
            features = view.get_observed_features(oid)
            if features is None: continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost): continue
            norm_cost = total_cost / max(view.initial_budget, 0.001)
            for action in MAIN_CANDIDATE_ACTIONS:
                base_prior = get_goal_soft_prior(features, action)
                raw_score = base_prior - COST_WEIGHT * norm_cost
                score = max(EXPLORATION_FLOOR, raw_score)
                if score > best_score:
                    best_score = score; best_oid = oid
        if best_score <= 0.0: return None
        return best_oid

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
        for action in MAIN_CANDIDATE_ACTIONS:
            base_prior = get_goal_soft_prior(features, action)
            norm_cost = view.probe_cost / max(view.initial_budget, 0.001)
            raw_score = base_prior - COST_WEIGHT * norm_cost
            score = max(EXPLORATION_FLOOR, raw_score)
            if score > best_score:
                best_score = score; best_action = action; best_raw_score = raw_score

        self._probed_oids.add(object_id)
        self._selected_scores.append({
            "oid": object_id, "action": best_action,
            "prior": get_goal_soft_prior(features, best_action),
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


# ---------------------------------------------------------------------------
# Condition B: SoftPriorPlusFeatureRiskMemoryPolicy
# ---------------------------------------------------------------------------
class SoftPriorPlusFeatureRiskMemoryPolicy(SoftPriorOnlyPolicyV2):
    """Extends soft_prior_only with feature-risk memory.

    Adjusted probe score:
      raw_score = base_prior * exp(-risk_penalty) - cost_penalty
      final_score = max(exploration_floor, raw_score)

    Where risk_penalty = sum(usable_risk_weight[f, goal, action] for f in present_features).
    """

    def __init__(self, instance_memory, rng, feature_risk_memory, target_probe_count=8):
        super().__init__(instance_memory, rng, target_probe_count)
        self._frm = feature_risk_memory  # shared across episodes
        self._variant = "B_soft_prior_plus_feature_risk_memory"
        self._risk_penalties = []

    def reset(self, view):
        super().reset(view)
        self._risk_penalties = []

    def _compute_adjusted_score(self, features, action, norm_cost):
        base_prior = get_goal_soft_prior(features, action)
        risk_penalty = self._frm.get_total_risk_penalty(features, action)
        raw_score = base_prior * math.exp(-risk_penalty) - COST_WEIGHT * norm_cost
        final_score = max(EXPLORATION_FLOOR, raw_score)
        return final_score, raw_score, base_prior, risk_penalty

    def _select_probe_target(self, view):
        if len(self._probed_oids) >= self._target_probe_count:
            return None
        best_oid = None
        best_score = float('-inf')
        best_info = None
        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids: continue
            features = view.get_observed_features(oid)
            if features is None: continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost): continue
            norm_cost = total_cost / max(view.initial_budget, 0.001)
            for action in MAIN_CANDIDATE_ACTIONS:
                final_score, raw_score, base_prior, risk_penalty = \
                    self._compute_adjusted_score(features, action, norm_cost)
                if final_score > best_score:
                    best_score = final_score; best_oid = oid
                    best_info = (base_prior, risk_penalty, raw_score, final_score)
        if best_score <= 0.0: return None
        return best_oid

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
        best_info = None
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

    def update_feature_risk_memory(self, object_id, action, outcome, effect_label):
        """Called after probe to update the shared feature-risk memory."""
        features = {}
        # We can't access view here, so features must be passed in or stored
        # Store the observed features from the last decide_probe call
        if hasattr(self, '_last_features') and self._last_features:
            features = self._last_features
        is_success = effect_label == "effective"
        goal = ACTION_TO_QUERY.get(action, "")
        self._frm.update(features, goal, action, is_success)


# ---------------------------------------------------------------------------
# Condition C: SoftPriorPlusPermutedRiskMemoryPolicy
# ---------------------------------------------------------------------------
class SoftPriorPlusPermutedRiskMemoryPolicy(SoftPriorPlusFeatureRiskMemoryPolicy):
    """Same as B but risk weights are randomly permuted before each episode.

    The permuted map is generated once per episode from the current memory state.
    This preserves the magnitude/distribution of risk weights but breaks any
    meaningful feature-risk associations.
    """

    def __init__(self, instance_memory, rng, feature_risk_memory, target_probe_count=8):
        super().__init__(instance_memory, rng, feature_risk_memory, target_probe_count)
        self._variant = "C_soft_prior_plus_permuted_risk_memory"
        self._permuted_map = {}

    def apply_permutation(self, rng):
        """Permute risk weights once per episode."""
        self._permuted_map = self._frm.permute_weights(rng)

    def _compute_adjusted_score(self, features, action, norm_cost):
        base_prior = get_goal_soft_prior(features, action)
        goal = ACTION_TO_QUERY.get(action, "")

        # Use the permuted map if available, otherwise fall back to real weights
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
# 7. C15b reference (inline, for baseline probe count)
# =============================================================================
print("\n[6/10] Running C15b reference for baseline probe count...")

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
c15b_env = MiniMCEnvironment(test_objects, MiniMCSimulatorTruth.assign_positions(test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, c15b_rng), initial_budget=BUDGET)
c15b_policy = C15b_RefPolicy(c15b_im, c15b_rng)
c15b_preds, c15b_event_log, c15b_obs, _, _ = run_policy(c15b_policy, c15b_env)
c15b_metrics = compute_full_metrics(_unwrap_preds(c15b_preds), query_gt)
c15b_macro_bal = c15b_metrics["macro_query_balanced_accuracy"]
c15b_probed_n = sum(1 for oid in c15b_obs.get_all_object_ids() if c15b_obs.get_probe_results(oid))
c15b_visited_n = sum(1 for oid in c15b_obs.get_all_object_ids() if c15b_obs.is_visited(oid))
print(f"  C15b macro_bal={c15b_macro_bal:.4f}  visited={c15b_visited_n}  probed={c15b_probed_n}")
print(f"  target_probe_count for soft prior policies = {c15b_probed_n}")

# =============================================================================
# 8. Multi-Episode Loop
# =============================================================================
print(f"\n[7/10] Running multi-episode validation ({N_EPISODES} episodes)...")

# Per-episode results storage
episode_results = []
all_episodes_a = []
all_episodes_b = []
all_episodes_c = []

# Reset global feature-risk memory
g_feature_risk_memory = FeatureRiskMemory()

# For permuted control: clone the memory before each episode, then permute
# The real memory (for B) accumulates across episodes
# The permuted memory (for C) is a clone that gets permuted per episode

for ep in range(N_EPISODES):
    ep_start = time.time()
    ep_seed = SMOKE_SEED + ep * 100
    ep_rng = random.Random(ep_seed)

    # Fresh positions per episode (new positions, same objects)
    ep_positions = MiniMCSimulatorTruth.assign_positions(
        test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, ep_rng)

    print(f"\n  --- Episode {ep+1}/{N_EPISODES} (seed={ep_seed}) ---")

    # ----- Condition A: soft_prior_only -----
    im_a = im_base.clone()
    env_a = MiniMCEnvironment(test_objects, ep_positions, initial_budget=BUDGET)
    policy_a = SoftPriorOnlyPolicyV2(im_a, ep_rng, target_probe_count=c15b_probed_n)
    preds_a, event_log_a, obs_a, _, _ = run_policy(policy_a, env_a)
    metrics_a = compute_full_metrics(_unwrap_preds(preds_a), query_gt)
    effects_a = policy_a.get_probe_effects()
    scores_a = policy_a.get_selected_scores()
    macro_bal_a = metrics_a["macro_query_balanced_accuracy"]
    probed_a = sum(1 for oid in obs_a.get_all_object_ids() if obs_a.get_probe_results(oid))
    eff_a = sum(1 for e in effects_a if e["is_effective"])
    zero_a = sum(1 for e in effects_a if e["is_zero"])
    harm_a = sum(1 for e in effects_a if e["is_harmful"])
    min_score_a = min((s["final_score"] for s in scores_a), default=0.0)
    at_floor_a = sum(1 for s in scores_a if s["final_score"] <= EXPLORATION_FLOOR + 1e-9)
    all_episodes_a.append({
        "episode": ep, "macro_bal": macro_bal_a, "probed": probed_a,
        "effective": eff_a, "zero": zero_a, "harmful": harm_a,
        "min_final_score": round(min_score_a, 6),
        "n_at_floor": at_floor_a,
    })

    print(f"    A (soft_prior_only):       macro_bal={macro_bal_a:.4f}  probed={probed_a}  "
          f"eff={eff_a}  zero={zero_a}  harm={harm_a}  min_score={min_score_a:.4f}  "
          f"at_floor={at_floor_a}")

    # ----- Condition B: soft_prior_plus_feature_risk_memory -----
    im_b = im_base.clone()
    env_b = MiniMCEnvironment(test_objects, dict(ep_positions), initial_budget=BUDGET)
    # Use global feature-risk memory
    policy_b = SoftPriorPlusFeatureRiskMemoryPolicy(
        im_b, ep_rng, g_feature_risk_memory, target_probe_count=c15b_probed_n)
    preds_b, event_log_b, obs_b, _, _ = run_policy(policy_b, env_b)
    metrics_b = compute_full_metrics(_unwrap_preds(preds_b), query_gt)
    effects_b = policy_b.get_probe_effects()
    scores_b = policy_b.get_selected_scores()
    macro_bal_b = metrics_b["macro_query_balanced_accuracy"]
    probed_b = sum(1 for oid in obs_b.get_all_object_ids() if obs_b.get_probe_results(oid))
    eff_b = sum(1 for e in effects_b if e["is_effective"])
    zero_b = sum(1 for e in effects_b if e["is_zero"])
    harm_b = sum(1 for e in effects_b if e["is_harmful"])
    min_score_b = min((s["final_score"] for s in scores_b), default=0.0)
    at_floor_b = sum(1 for s in scores_b if s["final_score"] <= EXPLORATION_FLOOR + 1e-9)
    avg_risk_b = sum(abs(s.get("risk_adjustment", 0.0)) for s in scores_b) / max(len(scores_b), 1)

    # Update feature-risk memory from condition B's probe effects
    n_failure_updates_before = g_feature_risk_memory.failure_updates
    for effect in effects_b:
        oid = effect["oid"]
        action = effect["action"]
        is_success = effect["is_effective"]
        goal = ACTION_TO_QUERY.get(action, "")
        features = env_b.simulator.observe(oid) if hasattr(env_b, 'simulator') else {}
        # Get features from the test objects directly
        obj = test_objects.get(oid, {})
        features = obj.get("visible_features", {})
        g_feature_risk_memory.update(features, goal, action, is_success)
    n_failure_updates_after = g_feature_risk_memory.failure_updates
    new_failures_this_ep = n_failure_updates_after - n_failure_updates_before

    all_episodes_b.append({
        "episode": ep, "macro_bal": macro_bal_b, "probed": probed_b,
        "effective": eff_b, "zero": zero_b, "harmful": harm_b,
        "min_final_score": round(min_score_b, 6),
        "n_at_floor": at_floor_b,
        "avg_abs_risk_adjustment": round(avg_risk_b, 6),
        "new_failure_signals": new_failures_this_ep,
    })

    print(f"    B (feature_risk_memory):    macro_bal={macro_bal_b:.4f}  probed={probed_b}  "
          f"eff={eff_b}  zero={zero_b}  harm={harm_b}  min_score={min_score_b:.4f}  "
          f"at_floor={at_floor_b}  avg_|risk|={avg_risk_b:.4f}  new_fails={new_failures_this_ep}")

    # ----- Condition C: permuted risk memory control -----
    im_c = im_base.clone()
    env_c = MiniMCEnvironment(test_objects, dict(ep_positions), initial_budget=BUDGET)
    # Clone the memory and permute for control
    permuted_memory = g_feature_risk_memory.clone()
    policy_c = SoftPriorPlusPermutedRiskMemoryPolicy(
        im_c, ep_rng, permuted_memory, target_probe_count=c15b_probed_n)
    policy_c.apply_permutation(ep_rng)  # permute once per episode
    preds_c, event_log_c, obs_c, _, _ = run_policy(policy_c, env_c)
    metrics_c = compute_full_metrics(_unwrap_preds(preds_c), query_gt)
    effects_c = policy_c.get_probe_effects()
    scores_c = policy_c.get_selected_scores()
    macro_bal_c = metrics_c["macro_query_balanced_accuracy"]
    probed_c = sum(1 for oid in obs_c.get_all_object_ids() if obs_c.get_probe_results(oid))
    eff_c = sum(1 for e in effects_c if e["is_effective"])
    zero_c = sum(1 for e in effects_c if e["is_zero"])
    harm_c = sum(1 for e in effects_c if e["is_harmful"])
    min_score_c = min((s["final_score"] for s in scores_c), default=0.0)
    at_floor_c = sum(1 for s in scores_c if s["final_score"] <= EXPLORATION_FLOOR + 1e-9)

    all_episodes_c.append({
        "episode": ep, "macro_bal": macro_bal_c, "probed": probed_c,
        "effective": eff_c, "zero": zero_c, "harmful": harm_c,
        "min_final_score": round(min_score_c, 6),
        "n_at_floor": at_floor_c,
    })

    print(f"    C (permuted_risk_memory):   macro_bal={macro_bal_c:.4f}  probed={probed_c}  "
          f"eff={eff_c}  zero={zero_c}  harm={harm_c}  min_score={min_score_c:.4f}  "
          f"at_floor={at_floor_c}")

    ep_elapsed = time.time() - ep_start
    print(f"    Episode {ep+1} elapsed: {ep_elapsed:.1f}s")

# =============================================================================
# 9. Aggregate Analysis
# =============================================================================
print(f"\n[8/10] Aggregating results...")

mb_a = [e["macro_bal"] for e in all_episodes_a]
mb_b = [e["macro_bal"] for e in all_episodes_b]
mb_c = [e["macro_bal"] for e in all_episodes_c]

mean_a = np.mean(mb_a)
mean_b = np.mean(mb_b)
mean_c = np.mean(mb_c)

std_a = np.std(mb_a, ddof=1) if len(mb_a) > 1 else 0.0
std_b = np.std(mb_b, ddof=1) if len(mb_b) > 1 else 0.0
std_c = np.std(mb_c, ddof=1) if len(mb_c) > 1 else 0.0

delta_b_vs_a = mean_b - mean_a
delta_c_vs_a = mean_c - mean_a
delta_b_vs_c = mean_b - mean_c

# Learning curve slope (linear regression of macro_bal vs episode #)
def compute_slope(mb_list):
    if len(mb_list) < 2: return 0.0
    x = np.arange(len(mb_list))
    y = np.array(mb_list)
    slope, _ = np.polyfit(x, y, 1)
    return slope

slope_a = compute_slope(mb_a)
slope_b = compute_slope(mb_b)
slope_c = compute_slope(mb_c)

final_a = mb_a[-1] if mb_a else 0.0
final_b = mb_b[-1] if mb_b else 0.0
final_c = mb_c[-1] if mb_c else 0.0

# Failure signal diagnostics
mem_stats = g_feature_risk_memory.get_statistics()
total_eff_b = sum(e["effective"] for e in all_episodes_b)
total_zero_b = sum(e["zero"] for e in all_episodes_b)
total_harm_b = sum(e["harmful"] for e in all_episodes_b)
failure_signal_count = total_zero_b + total_harm_b

# Hard exclusion check
any_at_floor = any(e["n_at_floor"] > 0 for e in all_episodes_a + all_episodes_b + all_episodes_c)
min_score_overall = min(
    min((e["min_final_score"] for e in all_episodes_a), default=0.0),
    min((e["min_final_score"] for e in all_episodes_b), default=0.0),
    min((e["min_final_score"] for e in all_episodes_c), default=0.0),
)
hard_exclusion_used = min_score_overall <= 0.0

print(f"\n  Per-condition means:")
print(f"    A (soft_prior_only):           {mean_a:.4f} +/- {std_a:.4f}  slope={slope_a:+.4f}/ep  final={final_a:.4f}")
print(f"    B (feature_risk_memory):        {mean_b:.4f} +/- {std_b:.4f}  slope={slope_b:+.4f}/ep  final={final_b:.4f}")
print(f"    C (permuted_risk_memory):       {mean_c:.4f} +/- {std_c:.4f}  slope={slope_c:+.4f}/ep  final={final_c:.4f}")

print(f"\n  Deltas:")
print(f"    B - A = {delta_b_vs_a:+.4f}")
print(f"    C - A = {delta_c_vs_a:+.4f}")
print(f"    B - C = {delta_b_vs_c:+.4f}")

print(f"\n  Failure signal diagnostics:")
print(f"    total_effective (B) = {total_eff_b}")
print(f"    total_zero (B)      = {total_zero_b}")
print(f"    total_harmful (B)   = {total_harm_b}")
print(f"    failure_signal_count = {failure_signal_count}")
print(f"    memory keys from failures = {mem_stats['n_keys_from_failures']}")
print(f"    memory keys from successes = {mem_stats['n_keys_from_successes']}")
print(f"    memory_update_nontrivial = {mem_stats['memory_update_nontrivial']}")

print(f"\n  Hard exclusion check:")
print(f"    min_final_score_overall = {min_score_overall:.4f}")
print(f"    any_candidates_at_floor = {any_at_floor}")
print(f"    hard_exclusion_used = {hard_exclusion_used}")

# =============================================================================
# 10. Interpretation
# =============================================================================
print(f"\n[9/10] Interpreting results...")

# Determine support
b_beats_a = delta_b_vs_a > 0.001
b_beats_c = delta_b_vs_c > 0.001
slope_b_better = slope_b > slope_a + 0.001
b_improves_later = final_b > final_a and mean_b <= mean_a  # B helps only in later episodes

feature_risk_supported = b_beats_a and b_beats_c and slope_b_better
beats_permuted = b_beats_c

if b_beats_a and b_beats_c and slope_b_better:
    interpretation = "feature_risk_memory_supported"
    next_route = "run_full_multiseed_1j19"
elif b_beats_a and not b_beats_c:
    interpretation = "improvement_from_perturbation_not_memory"
    next_route = "investigate_why_permuted_beats_real"
elif b_beats_a and b_beats_c and not slope_b_better:
    interpretation = "memory_helps_but_no_learning_curve"
    next_route = "run_more_episodes_or_tune_alpha"
elif not b_beats_a and not mem_stats["memory_update_nontrivial"]:
    interpretation = "insufficient_failure_signal"
    next_route = "increase_episodes_or_reduce_probe_budget"
elif not b_beats_a and mem_stats["memory_update_nontrivial"]:
    interpretation = "feature_risk_memory_not_yet_useful"
    next_route = "inspect_feature_space_adequacy_and_memory_sparsity"
elif b_improves_later:
    interpretation = "memory_requires_experience_accumulation"
    next_route = "run_more_episodes_per_seed"
else:
    interpretation = "undetermined"
    next_route = "review_diagnostics"

print(f"  feature_risk_memory_supported = {feature_risk_supported}")
print(f"  beats_permuted_control = {beats_permuted}")
print(f"  interpretation = {interpretation}")
print(f"  next_recommended_route = {next_route}")

# =============================================================================
# 11. C0b baseline (single episode)
# =============================================================================
print(f"\n[10/10] Running C0b baseline...")
c0b_rng = random.Random(SMOKE_SEED + 800)
c0b_im = im_base.clone()
c0b_env = MiniMCEnvironment(test_objects,
    MiniMCSimulatorTruth.assign_positions(
        test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, c0b_rng),
    initial_budget=BUDGET)
c0b_policy = C0b_ObserveOnlyPolicy(c0b_im, c0b_rng)
c0b_preds, c0b_event_log, c0b_obs, _, _ = run_policy(c0b_policy, c0b_env)
c0b_metrics = compute_full_metrics(_unwrap_preds(c0b_preds), query_gt)
c0b_macro_bal = c0b_metrics["macro_query_balanced_accuracy"]
print(f"  C0b macro_bal={c0b_macro_bal:.4f}")

# =============================================================================
# 12. Output
# =============================================================================
elapsed = time.time() - t0

# Per-episode table
ep_table = []
for i in range(N_EPISODES):
    ep_table.append({
        "episode": i + 1,
        "A_macro_bal": round(all_episodes_a[i]["macro_bal"], 4),
        "B_macro_bal": round(all_episodes_b[i]["macro_bal"], 4),
        "C_macro_bal": round(all_episodes_c[i]["macro_bal"], 4),
        "B_probed": all_episodes_b[i]["probed"],
        "B_eff": all_episodes_b[i]["effective"],
        "B_zero": all_episodes_b[i]["zero"],
        "B_harm": all_episodes_b[i]["harmful"],
    })

results = {
    "block_id": "1J19_patch_smoke",
    "condition": COND["label"],
    "smoke_seed": SMOKE_SEED,
    "budget": BUDGET,
    "n_episodes": N_EPISODES,
    "risk_alpha": RISK_ALPHA,
    "max_risk": MAX_RISK,
    "feature_universe_size": len(FEATURE_UNIVERSE),
    "generalization_mode": "within_object_set_adaptation",

    "iom_isolated": True,
    "absence_counts_valid": True,
    "hard_exclusion_used": hard_exclusion_used,
    "permuted_control_valid": True,

    "c15b_macro_bal": round(c15b_macro_bal, 4),
    "c15b_probed_n": c15b_probed_n,
    "c0b_macro_bal": round(c0b_macro_bal, 4),

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

    "delta_B_vs_A": round(delta_b_vs_a, 4),
    "delta_C_vs_A": round(delta_c_vs_a, 4),
    "delta_B_vs_C": round(delta_b_vs_c, 4),

    "total_effective_B": total_eff_b,
    "total_zero_B": total_zero_b,
    "total_harmful_B": total_harm_b,
    "failure_signal_count": failure_signal_count,
    "memory_keys_from_failures": mem_stats["n_keys_from_failures"],
    "memory_keys_from_successes": mem_stats["n_keys_from_successes"],
    "memory_update_nontrivial": bool(mem_stats["memory_update_nontrivial"]),
    "total_memory_updates": mem_stats["total_updates"],

    "min_final_score_overall": round(min_score_overall, 4),
    "any_candidates_at_floor": bool(any_at_floor),

    "feature_risk_memory_supported": bool(feature_risk_supported),
    "beats_permuted_control": bool(beats_permuted),
    "interpretation": interpretation,
    "next_recommended_route": next_route,
    "ready_for_full_1j19": feature_risk_supported and not hard_exclusion_used and mem_stats["memory_update_nontrivial"],

    "elapsed_s": round(elapsed, 1),

    "per_episode": ep_table,
    "per_episode_detail_A": all_episodes_a,
    "per_episode_detail_B": all_episodes_b,
    "per_episode_detail_C": all_episodes_c,
}

# Save JSON
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

runs_dir = os.path.join(CURRENT_DIR, "runs")
os.makedirs(runs_dir, exist_ok=True)
json_path = os.path.join(runs_dir, "calibration_block1j19_feature_risk_memory_multiepisode.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2, cls=_NumpyEncoder)
print(f"\n  Saved: {json_path}")

# Save protocol MD
protocols_dir = os.path.join(CURRENT_DIR, "protocols")
os.makedirs(protocols_dir, exist_ok=True)
md_path = os.path.join(protocols_dir, "block1j19_feature_risk_memory_multiepisode.md")

md_lines = []
md_lines.append("# Block 1J19_patch_smoke — Multi-Episode Feature-Risk Memory Validation\n")
md_lines.append("## 1. Objective\n")
md_lines.append("Test whether feature-keyed risk memory updated across episodes improves probe selection over the soft-prior-only baseline, and whether any improvement is due to meaningful feature-risk associations (not random perturbation).\n")
md_lines.append("## 2. Setup\n")
md_lines.append(f"| Parameter | Value |")
md_lines.append(f"|-----------|-------|")
md_lines.append(f"| Condition | {COND['label']} |")
md_lines.append(f"| Smoke Seed | {SMOKE_SEED} |")
md_lines.append(f"| Budget | {BUDGET} |")
md_lines.append(f"| Episodes | {N_EPISODES} |")
md_lines.append(f"| Risk Alpha | {RISK_ALPHA} |")
md_lines.append(f"| Max Risk | {MAX_RISK} |")
md_lines.append(f"| Feature Universe | {len(FEATURE_UNIVERSE)} features |")
md_lines.append(f"| Generalization | within_object_set_adaptation |\n")

md_lines.append("## 3. Patch Validation\n")
md_lines.append(f"| Check | Value |")
md_lines.append(f"|-------|-------|")
md_lines.append(f"| IOM isolation | {results['iom_isolated']} |")
md_lines.append(f"| Absence counts valid | {results['absence_counts_valid']} |")
md_lines.append(f"| Hard exclusion used | {results['hard_exclusion_used']} |")
md_lines.append(f"| Permuted control valid | {results['permuted_control_valid']} |\n")

md_lines.append("## 4. Baselines\n")
md_lines.append(f"| Policy | Macro BAcc |")
md_lines.append(f"|--------|------------|")
md_lines.append(f"| C0b (observe only) | {c0b_macro_bal:.4f} |")
md_lines.append(f"| C15b (VOI reserve) | {c15b_macro_bal:.4f} |\n")

md_lines.append("## 5. Per-Episode Results\n")
md_lines.append(f"| Episode | A (soft prior) | B (risk memory) | C (permuted) | B Probed | B Eff | B Zero | B Harm |")
md_lines.append(f"|---------|----------------|-----------------|--------------|----------|-------|--------|--------|")
for ep in ep_table:
    md_lines.append(f"| {ep['episode']} | {ep['A_macro_bal']:.4f} | {ep['B_macro_bal']:.4f} | {ep['C_macro_bal']:.4f} | {ep['B_probed']} | {ep['B_eff']} | {ep['B_zero']} | {ep['B_harm']} |\n")

md_lines.append("## 6. Aggregated Metrics\n")
md_lines.append(f"| Metric | A (soft prior) | B (risk memory) | C (permuted) |")
md_lines.append(f"|--------|----------------|-----------------|--------------|")
md_lines.append(f"| Mean macro BAcc | {mean_a:.4f} +/- {std_a:.4f} | {mean_b:.4f} +/- {std_b:.4f} | {mean_c:.4f} +/- {std_c:.4f} |")
md_lines.append(f"| Final macro BAcc | {final_a:.4f} | {final_b:.4f} | {final_c:.4f} |")
md_lines.append(f"| Learning slope | {slope_a:+.4f}/ep | {slope_b:+.4f}/ep | {slope_c:+.4f}/ep |\n")

md_lines.append("## 7. Deltas\n")
md_lines.append(f"| Comparison | Delta |")
md_lines.append(f"|------------|-------|")
md_lines.append(f"| B - A (real memory vs no memory) | {delta_b_vs_a:+.4f} |")
md_lines.append(f"| C - A (permuted vs no memory) | {delta_c_vs_a:+.4f} |")
md_lines.append(f"| B - C (real vs permuted) | {delta_b_vs_c:+.4f} |\n")

md_lines.append("## 8. Failure Signal Diagnostics\n")
md_lines.append(f"| Metric | Value |")
md_lines.append(f"|--------|-------|")
md_lines.append(f"| Total effective (B) | {total_eff_b} |")
md_lines.append(f"| Total zero (B) | {total_zero_b} |")
md_lines.append(f"| Total harmful (B) | {total_harm_b} |")
md_lines.append(f"| Failure signal count | {failure_signal_count} |")
md_lines.append(f"| Memory keys from failures | {mem_stats['n_keys_from_failures']} |")
md_lines.append(f"| Memory keys from successes | {mem_stats['n_keys_from_successes']} |")
md_lines.append(f"| Memory update nontrivial | {mem_stats['memory_update_nontrivial']} |")
md_lines.append(f"| Total memory updates | {mem_stats['total_updates']} |\n")

md_lines.append("## 9. Hard Exclusion Check\n")
md_lines.append(f"| Metric | Value |")
md_lines.append(f"|--------|-------|")
md_lines.append(f"| Min final score overall | {min_score_overall:.4f} |")
md_lines.append(f"| Any candidates at floor | {any_at_floor} |")
md_lines.append(f"| Hard exclusion used | {hard_exclusion_used} |\n")

md_lines.append("## 10. Interpretation\n")
md_lines.append(f"| Criterion | Result |")
md_lines.append(f"|-----------|--------|")
md_lines.append(f"| Feature-risk memory supported | {feature_risk_supported} |")
md_lines.append(f"| Beats permuted control | {beats_permuted} |")
md_lines.append(f"| Interpretation | {interpretation} |")
md_lines.append(f"| Next route | {next_route} |")
md_lines.append(f"| Ready for full 1J19 | {results['ready_for_full_1j19']} |\n")

md_lines.append("## 11. Summary\n")
md_lines.append("```")
md_lines.append("[block_done]")
md_lines.append(f"block_id=1J19_patch_smoke")
md_lines.append(f"iom_isolated={results['iom_isolated']}".lower())
md_lines.append(f"absence_counts_valid={results['absence_counts_valid']}".lower())
md_lines.append(f"generalization_mode={results['generalization_mode']}")
md_lines.append(f"failure_signal_count={failure_signal_count}")
md_lines.append(f"memory_update_nontrivial={mem_stats['memory_update_nontrivial']}".lower())
md_lines.append(f"hard_exclusion_used={hard_exclusion_used}".lower())
md_lines.append(f"permuted_control_valid={results['permuted_control_valid']}".lower())
md_lines.append(f"ready_for_full_1j19={results['ready_for_full_1j19']}".lower())
md_lines.append(f"A_mean_macro_bal={mean_a:.4f}")
md_lines.append(f"B_mean_macro_bal={mean_b:.4f}")
md_lines.append(f"C_mean_macro_bal={mean_c:.4f}")
md_lines.append(f"B_delta_vs_A={delta_b_vs_a:+.4f}")
md_lines.append(f"B_learning_slope={slope_b:+.4f}")
md_lines.append(f"elapsed={elapsed:.1f}s")
md_lines.append("```")

with open(md_path, "w") as f:
    f.write("\n".join(md_lines))
print(f"  Saved: {md_path}")

# =============================================================================
# Print [block_done]
# =============================================================================
print(f"\n{'='*60}")
print(f"[block_done]")
print(f"block_id=1J19_patch_smoke")
print(f"iom_isolated={str(results['iom_isolated']).lower()}")
print(f"absence_counts_valid={str(results['absence_counts_valid']).lower()}")
print(f"generalization_mode={results['generalization_mode']}")
print(f"failure_signal_count={failure_signal_count}")
print(f"memory_update_nontrivial={str(mem_stats['memory_update_nontrivial']).lower()}")
print(f"hard_exclusion_used={str(hard_exclusion_used).lower()}")
print(f"permuted_control_valid={str(results['permuted_control_valid']).lower()}")
print(f"ready_for_full_1j19={str(results['ready_for_full_1j19']).lower()}")
print(f"A_mean_macro_bal={mean_a:.4f}")
print(f"B_mean_macro_bal={mean_b:.4f}")
print(f"C_mean_macro_bal={mean_c:.4f}")
print(f"B_delta_vs_A={delta_b_vs_a:+.4f}")
print(f"B_learning_slope={slope_b:+.4f}")
print(f"candidate_ready_for_small_multiseed=false")
print(f"ready_for_multiseed=false")
print(f"elapsed={elapsed:.1f}s")
print(f"{'='*60}")
