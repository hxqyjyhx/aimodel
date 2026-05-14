"""
Block 1J16 -- Calibration-Aware / Confident-Error Probe Ranking Diagnostic.

Three C19 variants testing whether support/disagreement-based risk signals
can identify diagnostic probes better than uncertainty-based ranking:

  C19a_neighbor_disagreement_risk:
    Score = outcome_variance among kNN neighbors for the action.
    Higher = more disagreement = more probe value.

  C19b_low_support_risk:
    Score = 1.0 - composite_support_score.
    Lower support = higher risk = more probe value.
    Composite: neighbor_count, total_sim_weight, outcome_agreement.

  C19c_confident_error_risk — PRIMARY:
    Score = confidence * disagreement * (1 + query_relevance).
    Targets confident-but-wrong cases, not p≈0.5 uncertainty.

All variants keep C15b's observe phase (nearest-first, 25% reserve).
Only the probe ranking is replaced. Fixed probe budget aligned with C15b.

Single seed (101) only. No multi-seed.
"""
import os, sys, json, copy, random, time, math
from collections import Counter

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
    C14a_TruthAnswerOracle, C13_InstanceVOIPolicy,
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

SEED = 101
BUDGET = 1.5
COST_WEIGHT = 0.5

COND = copy.deepcopy(config.CUE_CONDITIONS[3])
assert COND["label"] == "C4_instance_subtype_cued_v1"

print("=" * 60)
print("Block 1J16 -- Confident-Error Probe Ranking Diagnostic")
print(f"  condition={COND['label']}  seed={SEED}  budget={BUDGET}")
print(f"  no_cross_test_object_propagation=true")
print(f"  primary_variant=C19c_confident_error_risk")
print("=" * 60)

# =============================================================================
# 1. Phase A: Training + IOM building
# =============================================================================
print("\n[1/9] Phase A: Training + IOM building...")
(student, base_learner, train_objects, train_env,
 _std_test, _std_test_env, final_metrics, rng) = run_phase_a_training(SEED, COND)
train_objects_dict = {oid: train_env.objects[oid] for oid in train_objects}
outcome_rows, actual_coverage = collect_sparse_probe_outcomes(
    train_objects_dict, student, config.COVERAGE, SEED)
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
# 2. Test objects
# =============================================================================
print("\n[2/9] Generating test objects...")
test_objects = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")
query_gt = compute_query_ground_truth(test_objects)
test_oids = sorted(test_objects.keys())
positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, rng)
sim_gt = MiniMCSimulatorTruth(test_objects, positions, config.AGENT_START)

QUERY_NAMES = sorted(_TASK_QUERIES.keys())
N_OBJECTS = len(test_oids)

ALL_GT_AFFORDANCES = {}
for oid in test_oids:
    ALL_GT_AFFORDANCES[oid] = sim_gt.get_ground_truth_affordances(oid)

# Action-to-query mapping
ACTION_TO_QUERY = {
    "craft_plank": "need_planks", "eat": "need_food",
    "use_as_tool": "need_tool", "burn_as_fuel": "need_fuel",
    "mine_by_hand": "need_stone", "mine_with_pickaxe": "need_stone",
}

print(f"  Generated {N_OBJECTS} test objects, {len(QUERY_NAMES)} queries")


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
        "per_query_positive_recall": {q: per_query[q]["positive_recall"] for q in per_query},
        "per_query_negative_recall": {q: per_query[q]["negative_recall"] for q in per_query},
    }


def get_cost_metrics(event_log, obs):
    visit_count = sum(1 for oid in obs.get_all_object_ids() if obs.is_visited(oid))
    probe_count = sum(1 for oid in obs.get_all_object_ids() if obs.get_probe_results(oid))
    total = obs.budget_spent
    observe_count = visit_count
    return {
        "visit_count": visit_count,
        "observe_count": observe_count,
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
# 4. IOM Support/Disagreement Scoring Helpers
# =============================================================================

def _compute_iom_confidence(per_action_stats, action, k=10):
    """Replicate IOM's internal confidence formula from dynamic indicators."""
    stats = per_action_stats.get(action, {})
    n_neighbors = stats.get("neighbor_count", 0)
    total_weight = stats.get("total_sim_weight", 0.0)
    outcome_variance = stats.get("outcome_variance", 0.25)
    if stats.get("source") in ("direct_override",):
        return 1.0
    if stats.get("source") in ("unsupported", "global_base_rate"):
        return 0.1
    if n_neighbors == 0:
        return 0.1
    confidence = (
        0.4 * min(total_weight / max(n_neighbors, 1), 1.0) +
        0.3 * min(n_neighbors / k, 1.0) +
        0.3 * (1.0 - min(outcome_variance / 0.25, 1.0))
    )
    return max(0.0, min(1.0, confidence))


def compute_risk_scores(im, oid, features):
    """Compute neighbor_disagreement, support_score, confidence for each action.

    Returns dict action -> {
        'probability', 'confidence', 'outcome_variance',
        'total_sim_weight', 'neighbor_count', 'query_relevance',
        'neighbor_disagreement_risk', 'low_support_risk', 'confident_error_risk'
    }
    """
    fake_obj = {"id": oid, "visible_features": features}
    probs = im.predict_all_affordances(fake_obj)
    indicators = im.get_object_dynamic_indicators(fake_obj)
    per_action = indicators.get("per_action", {})

    scores = {}
    for action in MAIN_CANDIDATE_ACTIONS:
        feature = ACTION_TO_FEATURE[action]
        p = probs.get(feature, 0.5)
        p = max(0.001, min(0.999, p))
        stats = per_action.get(action, {})
        outcome_variance = stats.get("outcome_variance", 0.25)
        total_sim_weight = stats.get("total_sim_weight", 0.0)
        neighbor_count = stats.get("neighbor_count", 0)
        source = stats.get("source", "unknown")

        confidence = _compute_iom_confidence(per_action, action, im.k)
        query_relevance = 1.0 - max(p, 1.0 - p)  # min(p, 1-p), near boundary

        # C19a: neighbor disagreement risk = outcome_variance
        neighbor_disagreement_risk = outcome_variance

        # C19b: low support risk
        # Support is high when: many neighbors, high total weight, high agreement
        support_score = (
            0.35 * min(neighbor_count / im.k, 1.0) +
            0.35 * min(total_sim_weight / max(neighbor_count, 1), 1.0) +
            0.30 * (1.0 - min(outcome_variance / 0.25, 1.0))
        )
        low_support_risk = 1.0 - support_score

        # C19c: confident error risk
        # High when: confident prediction + high neighbor disagreement
        # Multiply by (1 + query_relevance) as tiebreaker toward query boundaries
        confident_error_risk = confidence * outcome_variance * (0.5 + 0.5 * query_relevance)

        scores[action] = {
            "probability": float(p),
            "confidence": confidence,
            "outcome_variance": outcome_variance,
            "total_sim_weight": total_sim_weight,
            "neighbor_count": neighbor_count,
            "query_relevance": query_relevance,
            "source": source,
            "neighbor_disagreement_risk": neighbor_disagreement_risk,
            "low_support_risk": low_support_risk,
            "confident_error_risk": confident_error_risk,
        }
    return scores, probs


# =============================================================================
# 5. C15b Inline Policy (baseline)
# =============================================================================
print("\n[3/9] Defining C15b, C19a, C19b, C19c policies...")

class C15b_ProbeReservePolicy(MiniMCPolicy):
    """C15b: two-pass VOI with 25% budget reserve."""
    PHASE_OBSERVE = 1
    PHASE_PROBE = 2

    def __init__(self, instance_memory, rng, cost_weight=0.5):
        self._im = instance_memory
        self._rng = rng
        self._cost_weight = cost_weight
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._visit_order = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._visit_order = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0

    def _normalized_probe_cost(self, view):
        return view.probe_cost / max(view.initial_budget, 0.001)

    def _should_transition_to_probe_phase(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited:
            return True
        cheapest_observe = min(
            view.compute_reach_cost(oid) + view.observe_cost
            for oid in unvisited)
        if not view.can_afford(cheapest_observe):
            return True
        reserve = 0.25 * view.initial_budget
        if view.budget_remaining - cheapest_observe < reserve:
            return True
        if len(unvisited) <= 3:
            return True
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
                    best_cost = total
                    best_oid = oid
            if best_oid is not None:
                self._visit_order.append(("observe", best_oid))
            return best_oid
        elif self._phase == self.PHASE_PROBE:
            return self._select_probe_target(view)
        return None

    def _select_probe_target(self, view):
        best_oid = None
        best_score = float('-inf')
        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids:
                continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            affordable = view.can_afford(total_cost)
            features = view.get_observed_features(oid)
            if features is None:
                continue
            fake_obj = {"id": oid, "visible_features": features}
            probs = self._im.predict_all_affordances(fake_obj)
            current_utility = _compute_utility(probs)
            norm_cost = total_cost / max(view.initial_budget, 0.001)
            best_action_voi = 0.0
            for action in MAIN_CANDIDATE_ACTIONS:
                p_success = probs.get(ACTION_TO_FEATURE[action], 0.5)
                im_succ = self._im.clone()
                im_succ.incorporate_probe(oid, action, 1.0)
                utility_succ = _compute_utility(im_succ.predict_all_affordances(fake_obj))
                im_fail = self._im.clone()
                im_fail.incorporate_probe(oid, action, 0.0)
                utility_fail = _compute_utility(im_fail.predict_all_affordances(fake_obj))
                e_post = p_success * utility_succ + (1 - p_success) * utility_fail
                net_voi = e_post - current_utility - self._cost_weight * norm_cost
                if net_voi > best_action_voi:
                    best_action_voi = net_voi
            if affordable and best_action_voi > best_score:
                best_score = best_action_voi
                best_oid = oid
        if best_oid is not None:
            self._visit_order.append(("probe", best_oid))
        return best_oid

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE:
            return False, None
        features = view.get_observed_features(object_id)
        if features is None:
            return False, None
        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        current_utility = _compute_utility(probs)
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)
        norm_probe_cost = self._normalized_probe_cost(view)
        best_action = None
        best_net_voi = 0.0
        for action in MAIN_CANDIDATE_ACTIONS:
            p_success = probs.get(ACTION_TO_FEATURE[action], 0.5)
            im_succ = self._im.clone()
            im_succ.incorporate_probe(object_id, action, 1.0)
            utility_succ = _compute_utility(im_succ.predict_all_affordances(fake_obj))
            im_fail = self._im.clone()
            im_fail.incorporate_probe(object_id, action, 0.0)
            utility_fail = _compute_utility(im_fail.predict_all_affordances(fake_obj))
            e_post = p_success * utility_succ + (1 - p_success) * utility_fail
            net_voi = e_post - current_utility - self._cost_weight * norm_probe_cost
            if net_voi > best_net_voi:
                best_net_voi = net_voi
                best_action = action
        should_probe = best_net_voi > 0
        if should_probe:
            self._probed_oids.add(object_id)
        return should_probe, best_action

    def on_probe_result(self, view, object_id, action, outcome):
        features = view.get_observed_features(object_id)
        if features is not None:
            fake_obj = {"id": object_id, "visible_features": features}
            pre_probs = self._im.predict_all_affordances(fake_obj)
            self._im.incorporate_probe(object_id, action, outcome)
            post_probs = self._im.predict_all_affordances(fake_obj)
            self._probe_tracking.append({
                "oid": object_id, "action": action, "outcome": outcome,
                "pre_probs": dict(pre_probs), "post_probs": dict(post_probs),
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

    def get_pre_probe_entropies(self):
        return dict(self._pre_probe_entropies)


# =============================================================================
# 6. C19a — Neighbor Disagreement Risk
# =============================================================================
class C19a_NeighborDisagreementRiskPolicy(MiniMCPolicy):
    """Score probes by outcome_variance among kNN neighbors.

    Higher variance = neighbors disagree more about this outcome = more probe value.
    Same observe phase as C15b. Fixed probe budget (k = C15b probe count).
    """
    PHASE_OBSERVE = 1
    PHASE_PROBE = 2

    def __init__(self, instance_memory, rng, target_probe_count=8):
        self._im = instance_memory
        self._rng = rng
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._target_probe_count = target_probe_count
        self._probe_effects = []

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._probe_effects = []

    def _should_transition_to_probe_phase(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited:
            return True
        cheapest_observe = min(
            view.compute_reach_cost(oid) + view.observe_cost
            for oid in unvisited)
        if not view.can_afford(cheapest_observe):
            return True
        reserve = 0.25 * view.initial_budget
        if view.budget_remaining - cheapest_observe < reserve:
            return True
        if len(unvisited) <= 3:
            return True
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
                    best_cost = total
                    best_oid = oid
            return best_oid
        elif self._phase == self.PHASE_PROBE:
            return self._select_probe_target(view)
        return None

    def _select_probe_target(self, view):
        if len(self._probed_oids) >= self._target_probe_count:
            return None
        best_oid = None
        best_action = None
        best_score = float('-inf')
        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids:
                continue
            features = view.get_observed_features(oid)
            if features is None:
                continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost):
                continue
            scores, probs = compute_risk_scores(self._im, oid, features)
            for action in MAIN_CANDIDATE_ACTIONS:
                score = scores[action]["neighbor_disagreement_risk"]
                if score > best_score:
                    best_score = score
                    best_oid = oid
                    best_action = action
        return best_oid

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE:
            return False, None
        if len(self._probed_oids) >= self._target_probe_count:
            return False, None
        features = view.get_observed_features(object_id)
        if features is None:
            return False, None
        scores, probs = compute_risk_scores(self._im, object_id, features)
        best_action = None
        best_score = 0.0
        for action in MAIN_CANDIDATE_ACTIONS:
            score = scores[action]["neighbor_disagreement_risk"]
            if score > best_score:
                best_score = score
                best_action = action
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)
        self._probed_oids.add(object_id)
        return True, best_action

    def on_probe_result(self, view, object_id, action, outcome):
        features = view.get_observed_features(object_id)
        if features is not None:
            fake_obj = {"id": object_id, "visible_features": features}
            pre_scores, _ = compute_risk_scores(self._im, object_id, features)
            self._im.incorporate_probe(object_id, action, outcome)
            post_scores, _ = compute_risk_scores(self._im, object_id, features)
            self._probe_tracking.append({
                "oid": object_id, "action": action, "outcome": outcome,
            })
            pre_risk = pre_scores[action]["neighbor_disagreement_risk"]
            post_risk = post_scores[action]["neighbor_disagreement_risk"]
            self._probe_effects.append({
                "oid": object_id, "action": action, "outcome": outcome,
                "pre_risk_score": round(pre_risk, 6),
                "post_risk_score": round(post_risk, 6),
                "delta_risk": round(post_risk - pre_risk, 6),
                "is_zero_gain": abs(post_risk - pre_risk) < 0.0001,
                "is_harmful": post_risk > pre_risk + 0.0001,  # risk increased = unhelpful
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

    def get_pre_probe_entropies(self):
        return dict(self._pre_probe_entropies)

    def get_probe_effects(self):
        return list(self._probe_effects)


# =============================================================================
# 7. C19b — Low Support Risk
# =============================================================================
class C19b_LowSupportRiskPolicy(MiniMCPolicy):
    """Score probes by inverse support strength.

    Low support = few neighbors, low similarity weights, low agreement
    = high risk = more probe value.
    Same observe phase as C15b. Fixed probe budget (k = C15b probe count).
    """
    PHASE_OBSERVE = 1
    PHASE_PROBE = 2

    def __init__(self, instance_memory, rng, target_probe_count=8):
        self._im = instance_memory
        self._rng = rng
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._target_probe_count = target_probe_count
        self._probe_effects = []

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._probe_effects = []

    def _should_transition_to_probe_phase(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited:
            return True
        cheapest_observe = min(
            view.compute_reach_cost(oid) + view.observe_cost
            for oid in unvisited)
        if not view.can_afford(cheapest_observe):
            return True
        reserve = 0.25 * view.initial_budget
        if view.budget_remaining - cheapest_observe < reserve:
            return True
        if len(unvisited) <= 3:
            return True
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
                    best_cost = total
                    best_oid = oid
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
            if not view.is_visited(oid) or oid in self._probed_oids:
                continue
            features = view.get_observed_features(oid)
            if features is None:
                continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost):
                continue
            scores, probs = compute_risk_scores(self._im, oid, features)
            for action in MAIN_CANDIDATE_ACTIONS:
                score = scores[action]["low_support_risk"]
                if score > best_score:
                    best_score = score
                    best_oid = oid
        return best_oid

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE:
            return False, None
        if len(self._probed_oids) >= self._target_probe_count:
            return False, None
        features = view.get_observed_features(object_id)
        if features is None:
            return False, None
        scores, probs = compute_risk_scores(self._im, object_id, features)
        best_action = None
        best_score = 0.0
        for action in MAIN_CANDIDATE_ACTIONS:
            score = scores[action]["low_support_risk"]
            if score > best_score:
                best_score = score
                best_action = action
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)
        self._probed_oids.add(object_id)
        return True, best_action

    def on_probe_result(self, view, object_id, action, outcome):
        features = view.get_observed_features(object_id)
        if features is not None:
            pre_scores, _ = compute_risk_scores(self._im, object_id, features)
            self._im.incorporate_probe(object_id, action, outcome)
            post_scores, _ = compute_risk_scores(self._im, object_id, features)
            self._probe_tracking.append({
                "oid": object_id, "action": action, "outcome": outcome,
            })
            pre_risk = pre_scores[action]["low_support_risk"]
            post_risk = post_scores[action]["low_support_risk"]
            self._probe_effects.append({
                "oid": object_id, "action": action, "outcome": outcome,
                "pre_risk_score": round(pre_risk, 6),
                "post_risk_score": round(post_risk, 6),
                "delta_risk": round(post_risk - pre_risk, 6),
                "is_zero_gain": abs(post_risk - pre_risk) < 0.0001,
                "is_harmful": post_risk > pre_risk + 0.0001,
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

    def get_pre_probe_entropies(self):
        return dict(self._pre_probe_entropies)

    def get_probe_effects(self):
        return list(self._probe_effects)


# =============================================================================
# 8. C19c — Confident Error Risk (PRIMARY)
# =============================================================================
class C19c_ConfidentErrorRiskPolicy(MiniMCPolicy):
    """Score probes by confident-error risk = confidence * disagreement.

    Targets predictions where IOM is confident BUT neighbors disagree —
    the hallmark of confident errors. Multiply by (1 + query_relevance) as
    tiebreaker toward query decision boundaries.

    PRIMARY variant. Same observe phase as C15b. Fixed probe budget.
    """
    PHASE_OBSERVE = 1
    PHASE_PROBE = 2

    def __init__(self, instance_memory, rng, target_probe_count=8):
        self._im = instance_memory
        self._rng = rng
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._target_probe_count = target_probe_count
        self._probe_effects = []
        self._probe_scores = []

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._probe_effects = []
        self._probe_scores = []

    def _should_transition_to_probe_phase(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited:
            return True
        cheapest_observe = min(
            view.compute_reach_cost(oid) + view.observe_cost
            for oid in unvisited)
        if not view.can_afford(cheapest_observe):
            return True
        reserve = 0.25 * view.initial_budget
        if view.budget_remaining - cheapest_observe < reserve:
            return True
        if len(unvisited) <= 3:
            return True
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
                    best_cost = total
                    best_oid = oid
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
            if not view.is_visited(oid) or oid in self._probed_oids:
                continue
            features = view.get_observed_features(oid)
            if features is None:
                continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost):
                continue
            scores, probs = compute_risk_scores(self._im, oid, features)
            for action in MAIN_CANDIDATE_ACTIONS:
                score = scores[action]["confident_error_risk"]
                if score > best_score:
                    best_score = score
                    best_oid = oid
        return best_oid

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE:
            return False, None
        if len(self._probed_oids) >= self._target_probe_count:
            return False, None
        features = view.get_observed_features(object_id)
        if features is None:
            return False, None
        scores, probs = compute_risk_scores(self._im, object_id, features)
        best_action = None
        best_score = 0.0
        for action in MAIN_CANDIDATE_ACTIONS:
            score = scores[action]["confident_error_risk"]
            if score > best_score:
                best_score = score
                best_action = action
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)
        self._probed_oids.add(object_id)
        self._probe_scores.append({
            "oid": object_id, "action": best_action,
            "score": best_score,
            "scores_detail": {a: scores[a]["confident_error_risk"]
                              for a in MAIN_CANDIDATE_ACTIONS},
        })
        return True, best_action

    def on_probe_result(self, view, object_id, action, outcome):
        features = view.get_observed_features(object_id)
        if features is not None:
            pre_scores, _ = compute_risk_scores(self._im, object_id, features)
            self._im.incorporate_probe(object_id, action, outcome)
            post_scores, _ = compute_risk_scores(self._im, object_id, features)
            self._probe_tracking.append({
                "oid": object_id, "action": action, "outcome": outcome,
            })
            pre_risk = pre_scores[action]["confident_error_risk"]
            post_risk = post_scores[action]["confident_error_risk"]
            pre_conf = pre_scores[action]["confidence"]
            pre_disag = pre_scores[action]["outcome_variance"]
            self._probe_effects.append({
                "oid": object_id, "action": action, "outcome": outcome,
                "pre_confidence": round(pre_conf, 6),
                "pre_disagreement": round(pre_disag, 6),
                "pre_risk_score": round(pre_risk, 6),
                "post_risk_score": round(post_risk, 6),
                "delta_risk": round(post_risk - pre_risk, 6),
                "is_zero_gain": abs(post_risk - pre_risk) < 0.0001,
                "is_harmful": post_risk > pre_risk + 0.0001,
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

    def get_pre_probe_entropies(self):
        return dict(self._pre_probe_entropies)

    def get_probe_effects(self):
        return list(self._probe_effects)

    def get_probe_scores(self):
        return list(self._probe_scores)


# =============================================================================
# 9. Oracle k=8 Reference (Non-Deployable)
# =============================================================================
print("\n[4/9] Computing oracle k=8 reference (non-deployable)...")

def compute_oracle_k8_ranking(im, test_oids, gt_affordances, k=8):
    """Rank (oid, action) by |IOM_prediction - ground_truth| using ALL visible features."""
    errors = []
    for oid in test_oids:
        obj = test_objects[oid]
        fake_obj = {"id": oid, "visible_features": obj.get("visible_features", {})}
        probs = im.predict_all_affordances(fake_obj)
        for action in MAIN_CANDIDATE_ACTIONS:
            feature = ACTION_TO_FEATURE[action]
            pred = probs.get(feature, 0.5)
            gt = gt_affordances[oid].get(feature, 0.0)
            error = abs(pred - gt)
            errors.append((error, oid, action, feature, pred, gt))
    errors.sort(key=lambda x: x[0], reverse=True)
    top_k = errors[:k]
    return [(e[1], e[2], e[3], e[0], e[5]) for e in top_k], errors

oracle_k8_pairs, all_errors_ranked = compute_oracle_k8_ranking(
    im_base, test_oids, ALL_GT_AFFORDANCES, k=8)
oracle_k8_pair_set = set((oid, action) for oid, action, _, _, _ in oracle_k8_pairs)
print(f"  Oracle k=8 pairs: {[(oid, action) for oid, action, _, _, _ in oracle_k8_pairs]}")

# Compute oracle k=8 macro_bal by injecting true outcomes
def inject_and_predict(im, injections, test_oids):
    """Inject true outcomes into a fresh IOM clone and predict."""
    im_clone = im.clone()
    for oid, action, outcome in injections:
        feature = ACTION_TO_FEATURE[action]
        im_clone._direct_outcomes[(oid, feature)] = outcome
    predictions = {}
    for oid in test_oids:
        obj = test_objects[oid]
        fake_obj = {"id": oid, "visible_features": obj.get("visible_features", {})}
        predictions[oid] = im_clone.predict_all_affordances(fake_obj)
    return predictions

oracle_k8_injections = [(oid, action, ALL_GT_AFFORDANCES[oid][feature])
                        for oid, action, feature, error, gt in oracle_k8_pairs]
oracle_k8_preds = inject_and_predict(im_base, oracle_k8_injections, test_oids)
oracle_k8_metrics = compute_full_metrics(oracle_k8_preds, query_gt)
oracle_k8_mb = oracle_k8_metrics["macro_query_balanced_accuracy"]
print(f"  Oracle k=8 macro_bal={oracle_k8_mb:.4f}  (reference only)")


# =============================================================================
# 10. Run Baselines + C19a + C19b + C19c
# =============================================================================
print("\n[5/9] Running baselines...")
results = {}

# --- all_false ---
all_false_preds = {oid: {f: 0.0 for f in CORE_ACTION_FEATURES} for oid in test_oids}
results["all_false"] = {
    "predictions": all_false_preds,
    "metrics": compute_full_metrics(all_false_preds, query_gt),
    "cost": {"visit_count": 0, "observe_count": 0, "probe_count": 0,
             "total_cost": 0.0, "normalized_cost": 0.0},
    "policy_type": "floor",
}
print(f"  all_false macro_bal={results['all_false']['metrics']['macro_query_balanced_accuracy']:.4f}")

# --- C_minus_prior_only ---
c_minus_preds = {oid: {f: 0.5 for f in CORE_ACTION_FEATURES} for oid in test_oids}
results["C_minus_prior_only"] = {
    "predictions": c_minus_preds,
    "metrics": compute_full_metrics(c_minus_preds, query_gt),
    "cost": {"visit_count": 0, "observe_count": 0, "probe_count": 0,
             "total_cost": 0.0, "normalized_cost": 0.0},
    "policy_type": "no_interaction",
}
print(f"  C_minus_prior_only macro_bal={results['C_minus_prior_only']['metrics']['macro_query_balanced_accuracy']:.4f}")

# --- C0b_original ---
c0b_im = im_base.clone()
c0b_policy = C0b_ObserveOnlyPolicy(c0b_im, random.Random(SEED + 500))
c0b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c0b_preds, c0b_log, c0b_obs, _, _ = run_policy(c0b_policy, c0b_env)
c0b_visited_oids = [oid for oid in test_oids if c0b_obs.is_visited(oid)]
results["C0b_original"] = {
    "predictions": c0b_preds["per_object"],
    "metrics": compute_full_metrics(c0b_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c0b_log, c0b_obs),
    "policy_type": "observe_only",
    "visited_oids": c0b_visited_oids,
}
c0b_mb = results["C0b_original"]["metrics"]["macro_query_balanced_accuracy"]
print(f"  C0b_original macro_bal={c0b_mb:.4f}  visited={len(c0b_visited_oids)}")

# --- C13_instance_VOI_original ---
c13_im = im_base.clone()
c13_policy = C13_InstanceVOIPolicy(c13_im, random.Random(SEED + 600), cost_weight=COST_WEIGHT)
c13_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c13_preds, c13_log, c13_obs, _, _ = run_policy(c13_policy, c13_env)
c13_visited_oids = [oid for oid in test_oids if c13_obs.is_visited(oid)]
results["C13_instance_VOI_original"] = {
    "predictions": c13_preds["per_object"],
    "metrics": compute_full_metrics(c13_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c13_log, c13_obs),
    "policy_type": "instance_VOI",
    "visited_oids": c13_visited_oids,
}
c13_mb = results["C13_instance_VOI_original"]["metrics"]["macro_query_balanced_accuracy"]
c13_visited_n = len(c13_visited_oids)
c13_probed_n = results["C13_instance_VOI_original"]["cost"]["probe_count"]
print(f"  C13_original macro_bal={c13_mb:.4f}  visited={c13_visited_n}  probed={c13_probed_n}")

# --- C15b_probe_reserve_original ---
c15b_im = im_base.clone()
c15b_policy = C15b_ProbeReservePolicy(c15b_im, random.Random(SEED + 700), cost_weight=COST_WEIGHT)
c15b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c15b_preds, c15b_log, c15b_obs, _, _ = run_policy(c15b_policy, c15b_env)
c15b_visited_oids = [oid for oid in test_oids if c15b_obs.is_visited(oid)]
c15b_probe_pairs = [(e["oid"], e["action"]) for e in c15b_policy._probe_tracking]
c15b_probe_pair_set = set(c15b_probe_pairs)
c15b_selected_actions = [e["action"] for e in c15b_policy._probe_tracking]
results["C15b_probe_reserve_original"] = {
    "predictions": c15b_preds["per_object"],
    "metrics": compute_full_metrics(c15b_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c15b_log, c15b_obs),
    "policy_type": "two_pass_VOI",
    "visited_oids": c15b_visited_oids,
    "probe_pairs": c15b_probe_pairs,
}
c15b_mb = results["C15b_probe_reserve_original"]["metrics"]["macro_query_balanced_accuracy"]
c15b_pos_recall = results["C15b_probe_reserve_original"]["metrics"]["mean_positive_recall"]
c15b_neg_recall = results["C15b_probe_reserve_original"]["metrics"]["mean_negative_recall"]
c15b_visited_n = len(c15b_visited_oids)
c15b_probed_n = results["C15b_probe_reserve_original"]["cost"]["probe_count"]
print(f"  C15b_original macro_bal={c15b_mb:.4f}  visited={c15b_visited_n}  probed={c15b_probed_n}")
print(f"  C15b probe pairs: {c15b_probe_pairs}")
print(f"  C15b probe actions: {dict(Counter(c15b_selected_actions))}")

# --- C18b from 1J15 (re-run inline) ---
# Use 1J15's C18b as a reference: query-relevant ranking, fixed budget.
# We re-run it inline.
print("\n  Running C18b_query_relevant (from 1J15) as reference...")
t_c18b = time.time()
# We need the C18b policy from 1J15. Define it inline.
class C18b_QueryRelevantFixedBudgetPolicy(MiniMCPolicy):
    """C18b: Query-relevant ranking, fixed budget. Re-run inline from 1J15 spec."""
    PHASE_OBSERVE = 1
    PHASE_PROBE = 2

    def __init__(self, instance_memory, rng, target_probe_count=8):
        self._im = instance_memory
        self._rng = rng
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._target_probe_count = target_probe_count
        self._probe_effects = []

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._probe_effects = []

    def _should_transition_to_probe_phase(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited:
            return True
        cheapest_observe = min(
            view.compute_reach_cost(oid) + view.observe_cost for oid in unvisited)
        if not view.can_afford(cheapest_observe):
            return True
        reserve = 0.25 * view.initial_budget
        if view.budget_remaining - cheapest_observe < reserve:
            return True
        if len(unvisited) <= 3:
            return True
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
                    best_cost = total
                    best_oid = oid
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
            if not view.is_visited(oid) or oid in self._probed_oids:
                continue
            features = view.get_observed_features(oid)
            if features is None:
                continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost):
                continue
            fake_obj = {"id": oid, "visible_features": features}
            probs = self._im.predict_all_affordances(fake_obj)
            for action in MAIN_CANDIDATE_ACTIONS:
                p = probs.get(ACTION_TO_FEATURE[action], 0.5)
                p = max(0.001, min(0.999, p))
                score = 1.0 - max(p, 1.0 - p)  # min(p,1-p) = query relevance
                if score > best_score:
                    best_score = score
                    best_oid = oid
        return best_oid

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE:
            return False, None
        if len(self._probed_oids) >= self._target_probe_count:
            return False, None
        features = view.get_observed_features(object_id)
        if features is None:
            return False, None
        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        best_action = None
        best_score = 0.0
        for action in MAIN_CANDIDATE_ACTIONS:
            p = probs.get(ACTION_TO_FEATURE[action], 0.5)
            p = max(0.001, min(0.999, p))
            score = 1.0 - max(p, 1.0 - p)
            if score > best_score:
                best_score = score
                best_action = action
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)
        self._probed_oids.add(object_id)
        return True, best_action

    def on_probe_result(self, view, object_id, action, outcome):
        features = view.get_observed_features(object_id)
        if features is not None:
            fake_obj = {"id": object_id, "visible_features": features}
            pre_probs = self._im.predict_all_affordances(fake_obj)
            self._im.incorporate_probe(object_id, action, outcome)
            post_probs = self._im.predict_all_affordances(fake_obj)
            pre_score = 1.0 - max(pre_probs.get(ACTION_TO_FEATURE[action], 0.5),
                                  1.0 - pre_probs.get(ACTION_TO_FEATURE[action], 0.5))
            post_score = 1.0 - max(post_probs.get(ACTION_TO_FEATURE[action], 0.5),
                                   1.0 - post_probs.get(ACTION_TO_FEATURE[action], 0.5))
            self._probe_tracking.append({"oid": object_id, "action": action, "outcome": outcome})
            self._probe_effects.append({
                "oid": object_id, "action": action, "outcome": outcome,
                "pre_score": round(pre_score, 6),
                "post_score": round(post_score, 6),
                "delta": round(post_score - pre_score, 6),
                "is_zero_gain": abs(post_score - pre_score) < 0.0001,
                "is_harmful": post_score < pre_score - 0.0001,
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

    def get_pre_probe_entropies(self):
        return dict(self._pre_probe_entropies)

    def get_probe_effects(self):
        return list(self._probe_effects)

c18b_im = im_base.clone()
c18b_policy = C18b_QueryRelevantFixedBudgetPolicy(c18b_im, random.Random(SEED + 850),
                                                   target_probe_count=c15b_probed_n)
c18b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c18b_preds, c18b_log, c18b_obs, _, _ = run_policy(c18b_policy, c18b_env)
c18b_visited_oids = [oid for oid in test_oids if c18b_obs.is_visited(oid)]
c18b_probe_effects = c18b_policy.get_probe_effects()
c18b_probe_pairs = [(e["oid"], e["action"]) for e in c18b_probe_effects]
c18b_selected_actions = [e["action"] for e in c18b_probe_effects]
results["C18b_query_relevant_fixed_budget"] = {
    "predictions": c18b_preds["per_object"],
    "metrics": compute_full_metrics(c18b_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c18b_log, c18b_obs),
    "policy_type": "query_relevant_fixed_budget_from_1J15",
    "probe_pairs": c18b_probe_pairs,
}
c18b_mb = results["C18b_query_relevant_fixed_budget"]["metrics"]["macro_query_balanced_accuracy"]
c18b_pos_recall = results["C18b_query_relevant_fixed_budget"]["metrics"]["mean_positive_recall"]
c18b_neg_recall = results["C18b_query_relevant_fixed_budget"]["metrics"]["mean_negative_recall"]
c18b_probed_n = results["C18b_query_relevant_fixed_budget"]["cost"]["probe_count"]
c18b_visited_n = len(c18b_visited_oids)
c18b_delta = c18b_mb - c15b_mb
c18b_runtime = time.time() - t_c18b
print(f"  C18b macro_bal={c18b_mb:.4f}  probed={c18b_probed_n}  delta_vs_C15b={c18b_delta:+.4f}")

# --- C19a_neighbor_disagreement_risk ---
print("\n  Running C19a_neighbor_disagreement_risk...")
t_c19a = time.time()
c19a_im = im_base.clone()
c19a_policy = C19a_NeighborDisagreementRiskPolicy(c19a_im, random.Random(SEED + 900),
                                                    target_probe_count=c15b_probed_n)
c19a_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c19a_preds, c19a_log, c19a_obs, _, _ = run_policy(c19a_policy, c19a_env)
c19a_visited_oids = [oid for oid in test_oids if c19a_obs.is_visited(oid)]
c19a_probe_effects = c19a_policy.get_probe_effects()
c19a_probe_pairs = [(e["oid"], e["action"]) for e in c19a_probe_effects]
c19a_selected_actions = [e["action"] for e in c19a_probe_effects]
c19a_zero_gain = sum(1 for e in c19a_probe_effects if e["is_zero_gain"])
c19a_harmful = sum(1 for e in c19a_probe_effects if e["is_harmful"])
c19a_effective = len(c19a_probe_effects) - c19a_zero_gain - c19a_harmful
c19a_mean_disagreement = (sum(e["pre_risk_score"] for e in c19a_probe_effects) /
                           max(len(c19a_probe_effects), 1))
results["C19a_neighbor_disagreement_risk"] = {
    "predictions": c19a_preds["per_object"],
    "metrics": compute_full_metrics(c19a_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c19a_log, c19a_obs),
    "policy_type": "neighbor_disagreement_risk",
    "visited_oids": c19a_visited_oids,
    "probe_pairs": c19a_probe_pairs,
    "probe_effects": c19a_probe_effects,
}
c19a_mb = results["C19a_neighbor_disagreement_risk"]["metrics"]["macro_query_balanced_accuracy"]
c19a_pos_recall = results["C19a_neighbor_disagreement_risk"]["metrics"]["mean_positive_recall"]
c19a_neg_recall = results["C19a_neighbor_disagreement_risk"]["metrics"]["mean_negative_recall"]
c19a_visited_n = len(c19a_visited_oids)
c19a_probed_n = results["C19a_neighbor_disagreement_risk"]["cost"]["probe_count"]
c19a_delta = c19a_mb - c15b_mb
c19a_runtime = time.time() - t_c19a
c19a_overlap_c15b = len(set(c19a_probe_pairs) & c15b_probe_pair_set)
c19a_overlap_oracle = len(set(c19a_probe_pairs) & oracle_k8_pair_set)
print(f"  C19a macro_bal={c19a_mb:.4f}  visited={c19a_visited_n}  probed={c19a_probed_n}  "
      f"delta_vs_C15b={c19a_delta:+.4f}  overlap_C15b={c19a_overlap_c15b}  "
      f"overlap_oracle_k8={c19a_overlap_oracle}  runtime={c19a_runtime:.1f}s")

# --- C19b_low_support_risk ---
print("\n  Running C19b_low_support_risk...")
t_c19b = time.time()
c19b_im = im_base.clone()
c19b_policy = C19b_LowSupportRiskPolicy(c19b_im, random.Random(SEED + 1000),
                                          target_probe_count=c15b_probed_n)
c19b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c19b_preds, c19b_log, c19b_obs, _, _ = run_policy(c19b_policy, c19b_env)
c19b_visited_oids = [oid for oid in test_oids if c19b_obs.is_visited(oid)]
c19b_probe_effects = c19b_policy.get_probe_effects()
c19b_probe_pairs = [(e["oid"], e["action"]) for e in c19b_probe_effects]
c19b_selected_actions = [e["action"] for e in c19b_probe_effects]
c19b_zero_gain = sum(1 for e in c19b_probe_effects if e["is_zero_gain"])
c19b_harmful = sum(1 for e in c19b_probe_effects if e["is_harmful"])
c19b_effective = len(c19b_probe_effects) - c19b_zero_gain - c19b_harmful
c19b_mean_support = (sum(1.0 - e["pre_risk_score"] for e in c19b_probe_effects) /
                      max(len(c19b_probe_effects), 1))
results["C19b_low_support_risk"] = {
    "predictions": c19b_preds["per_object"],
    "metrics": compute_full_metrics(c19b_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c19b_log, c19b_obs),
    "policy_type": "low_support_risk",
    "visited_oids": c19b_visited_oids,
    "probe_pairs": c19b_probe_pairs,
    "probe_effects": c19b_probe_effects,
}
c19b_mb = results["C19b_low_support_risk"]["metrics"]["macro_query_balanced_accuracy"]
c19b_pos_recall = results["C19b_low_support_risk"]["metrics"]["mean_positive_recall"]
c19b_neg_recall = results["C19b_low_support_risk"]["metrics"]["mean_negative_recall"]
c19b_visited_n = len(c19b_visited_oids)
c19b_probed_n = results["C19b_low_support_risk"]["cost"]["probe_count"]
c19b_delta = c19b_mb - c15b_mb
c19b_runtime = time.time() - t_c19b
c19b_overlap_c15b = len(set(c19b_probe_pairs) & c15b_probe_pair_set)
c19b_overlap_oracle = len(set(c19b_probe_pairs) & oracle_k8_pair_set)
print(f"  C19b macro_bal={c19b_mb:.4f}  visited={c19b_visited_n}  probed={c19b_probed_n}  "
      f"delta_vs_C15b={c19b_delta:+.4f}  overlap_C15b={c19b_overlap_c15b}  "
      f"overlap_oracle_k8={c19b_overlap_oracle}  runtime={c19b_runtime:.1f}s")

# --- C19c_confident_error_risk (PRIMARY) ---
print("\n  Running C19c_confident_error_risk (PRIMARY)...")
t_c19c = time.time()
c19c_im = im_base.clone()
c19c_policy = C19c_ConfidentErrorRiskPolicy(c19c_im, random.Random(SEED + 1100),
                                              target_probe_count=c15b_probed_n)
c19c_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c19c_preds, c19c_log, c19c_obs, _, _ = run_policy(c19c_policy, c19c_env)
c19c_visited_oids = [oid for oid in test_oids if c19c_obs.is_visited(oid)]
c19c_probe_effects = c19c_policy.get_probe_effects()
c19c_probe_scores = c19c_policy.get_probe_scores()
c19c_probe_pairs = [(e["oid"], e["action"]) for e in c19c_probe_effects]
c19c_selected_actions = [e["action"] for e in c19c_probe_effects]
c19c_zero_gain = sum(1 for e in c19c_probe_effects if e["is_zero_gain"])
c19c_harmful = sum(1 for e in c19c_probe_effects if e["is_harmful"])
c19c_effective = len(c19c_probe_effects) - c19c_zero_gain - c19c_harmful
c19c_mean_confidence = (sum(e.get("pre_confidence", 0) for e in c19c_probe_effects) /
                        max(len(c19c_probe_effects), 1))
c19c_mean_disagreement = (sum(e.get("pre_disagreement", 0) for e in c19c_probe_effects) /
                           max(len(c19c_probe_effects), 1))
c19c_mean_risk = (sum(e["pre_risk_score"] for e in c19c_probe_effects) /
                   max(len(c19c_probe_effects), 1))
results["C19c_confident_error_risk"] = {
    "predictions": c19c_preds["per_object"],
    "metrics": compute_full_metrics(c19c_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c19c_log, c19c_obs),
    "policy_type": "confident_error_risk_primary",
    "visited_oids": c19c_visited_oids,
    "probe_pairs": c19c_probe_pairs,
    "probe_effects": c19c_probe_effects,
    "probe_scores": c19c_probe_scores,
}
c19c_mb = results["C19c_confident_error_risk"]["metrics"]["macro_query_balanced_accuracy"]
c19c_pos_recall = results["C19c_confident_error_risk"]["metrics"]["mean_positive_recall"]
c19c_neg_recall = results["C19c_confident_error_risk"]["metrics"]["mean_negative_recall"]
c19c_visited_n = len(c19c_visited_oids)
c19c_probed_n = results["C19c_confident_error_risk"]["cost"]["probe_count"]
c19c_delta = c19c_mb - c15b_mb
c19c_runtime = time.time() - t_c19c
c19c_overlap_c15b = len(set(c19c_probe_pairs) & c15b_probe_pair_set)
c19c_overlap_oracle = len(set(c19c_probe_pairs) & oracle_k8_pair_set)
print(f"  C19c macro_bal={c19c_mb:.4f}  visited={c19c_visited_n}  probed={c19c_probed_n}  "
      f"delta_vs_C15b={c19c_delta:+.4f}  overlap_C15b={c19c_overlap_c15b}  "
      f"overlap_oracle_k8={c19c_overlap_oracle}  runtime={c19c_runtime:.1f}s")
print(f"  C19c mean_confidence={c19c_mean_confidence:.4f}  "
      f"mean_disagreement={c19c_mean_disagreement:.4f}  mean_risk={c19c_mean_risk:.4f}")
print(f"  C19c effective={c19c_effective}  zero_gain={c19c_zero_gain}  harmful={c19c_harmful}")

# --- Oracle ---
oracle_preds = C14a_TruthAnswerOracle(sim_gt).get_answer()
results["C_oracle_full_information"] = {
    "predictions": oracle_preds["per_object"],
    "metrics": compute_full_metrics(oracle_preds["per_object"], query_gt),
    "policy_type": "oracle",
    "cost": {"visit_count": 0, "observe_count": 0, "probe_count": 0,
             "total_cost": 0.0, "normalized_cost": 0.0},
}
oracle_mb = results["C_oracle_full_information"]["metrics"]["macro_query_balanced_accuracy"]
print(f"  C_oracle macro_bal={oracle_mb:.4f}")


# =============================================================================
# 11. Comparisons
# =============================================================================
print("\n[6/9] Computing comparisons...")
target_10pct = 0.6284
oracle_c0b_gap = oracle_mb - c0b_mb

# Find best C19 variant
c19_variants = {
    "C19a": (c19a_mb, c19a_delta, c19a_probed_n, c19a_overlap_oracle, c19a_effective, c19a_harmful),
    "C19b": (c19b_mb, c19b_delta, c19b_probed_n, c19b_overlap_oracle, c19b_effective, c19b_harmful),
    "C19c": (c19c_mb, c19c_delta, c19c_probed_n, c19c_overlap_oracle, c19c_effective, c19c_harmful),
}
best_c19_name = max(c19_variants, key=lambda k: c19_variants[k][0])
best_c19_mb, best_c19_delta, best_c19_probed, best_c19_overlap_oracle, best_c19_eff, best_c19_harm = c19_variants[best_c19_name]

comparisons = {}

# Primary: best C19 vs C15b
comparisons["best_C19_vs_C15b"] = {
    "best_variant": best_c19_name,
    "delta_macro_bal": round(best_c19_delta, 6),
    "best_macro_bal": best_c19_mb,
}

# All C19 vs C15b
for name, (mb, delta, probed, ov_oracle, eff, harm) in c19_variants.items():
    comparisons[f"{name}_vs_C15b"] = {
        "delta_macro_bal": round(delta, 6),
        "macro_bal": mb,
        "overlap_with_C15b": len(set(eval(f"c{name.lower()[1:]}_probe_pairs")) & c15b_probe_pair_set),
        "overlap_with_oracle_k8": ov_oracle,
        "effective_probe_count": eff,
        "harmful_probe_count": harm,
    }

# Gap capture
comparisons["gap_capture"] = {
    "C0b_base": c0b_mb,
    "oracle_ceiling": oracle_mb,
    "oracle_c0b_gap": oracle_c0b_gap,
    "target_10pct_threshold": target_10pct,
    "C15b_gap_capture_pct": round((c15b_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
    "C18b_gap_capture_pct": round((c18b_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
    "C19a_gap_capture_pct": round((c19a_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
    "C19b_gap_capture_pct": round((c19b_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
    "C19c_gap_capture_pct": round((c19c_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
    "oracle_k8_gap_capture_pct": round((oracle_k8_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
}

# Per-query breakdown
comparisons["per_query_breakdown"] = {}
for qname in QUERY_NAMES:
    c15b_q = results["C15b_probe_reserve_original"]["metrics"]["per_query"][qname]
    c19c_q = results["C19c_confident_error_risk"]["metrics"]["per_query"][qname]
    comparisons["per_query_breakdown"][qname] = {
        "C15b_bal_acc": c15b_q["balanced_accuracy"],
        "C19c_bal_acc": c19c_q["balanced_accuracy"],
        "C19c_delta_vs_C15b": round(c19c_q["balanced_accuracy"] - c15b_q["balanced_accuracy"], 6),
    }

# Probe selection overlap summary
comparisons["probe_pair_overlap"] = {
    "C15b_probe_pairs": str(c15b_probe_pairs),
    "oracle_k8_pairs": str([(oid, action) for oid, action, _, _, _ in oracle_k8_pairs]),
    "C19a_probe_pairs": str(c19a_probe_pairs),
    "C19a_overlap_C15b": c19a_overlap_c15b,
    "C19a_overlap_oracle_k8": c19a_overlap_oracle,
    "C19b_probe_pairs": str(c19b_probe_pairs),
    "C19b_overlap_C15b": c19b_overlap_c15b,
    "C19b_overlap_oracle_k8": c19b_overlap_oracle,
    "C19c_probe_pairs": str(c19c_probe_pairs),
    "C19c_overlap_C15b": c19c_overlap_c15b,
    "C19c_overlap_oracle_k8": c19c_overlap_oracle,
}

# Oracle reference
comparisons["oracle_references"] = {
    "1J14_oracle_k8_macro_bal": 0.6488,
    "1J14_C15b_UB_macro_bal": 0.8546,
    "1J14_Full_UB_macro_bal": 1.0000,
    "1J16_oracle_k8_macro_bal": oracle_k8_mb,
    "1J16_oracle_k8_pairs": str([(oid, action) for oid, action, _, _, _ in oracle_k8_pairs]),
    "note": "Oracle k=8 uses |IOM_pred - gt| ranking with ALL visible features (non-deployable).",
}

# IOM architecture notes
comparisons["iom_architecture_notes"] = {
    "no_cross_test_object_propagation": True,
    "global_recompute_equivalent_to_local": True,
}


# =============================================================================
# 12. Interpretation
# =============================================================================
print("\n[7/9] Interpreting results...")

c19c_hits_10pct = c19c_mb >= target_10pct
c19c_improves_c15b = c19c_mb > c15b_mb + 0.001

if c19c_hits_10pct:
    c19c_interpretation = (
        f"C19c (confident-error risk) reaches {c19c_mb:.4f}, exceeding the 10% "
        f"gap threshold ({target_10pct}). Confident-error proxy is USEFUL for "
        f"probe selection. Targets confident-but-possibly-wrong predictions."
    )
    confident_error_proxy_supported = True
    candidate_ready_for_small_multiseed = True
    next_recommended_route = "learned_probe_value_critic_from_logs"
elif c19c_improves_c15b:
    c19c_interpretation = (
        f"C19c reaches {c19c_mb:.4f} (+{c19c_delta:+.4f} vs C15b), improving over "
        f"C15b but not reaching the 10% threshold ({target_10pct}). Confident-error "
        f"proxy provides partial signal."
    )
    confident_error_proxy_supported = True
    candidate_ready_for_small_multiseed = False
    next_recommended_route = "enhanced_confident_error_or_learned_critic"
elif abs(c19c_mb - c15b_mb) < 0.001:
    c19c_interpretation = (
        f"C19c reaches {c19c_mb:.4f}, essentially identical to C15b ({c15b_mb:.4f}). "
        f"The confident-error proxy does NOT differentiate from C15b's VOI ranking. "
        f"Under current IOM, confidence * disagreement is not more informative than "
        f"simple affordance-utility maximization for probe selection."
    )
    confident_error_proxy_supported = False
    candidate_ready_for_small_multiseed = False
    next_recommended_route = "learned_probe_value_critic_from_effective_harmful_logs"
else:
    c19c_interpretation = (
        f"C19c reaches {c19c_mb:.4f} ({c19c_delta:+.4f} vs C15b), below C15b. "
        f"Confident-error proxy DEGRADES performance. The hand-coded risk proxy "
        f"(confidence * disagreement) selects probes that are individually risky "
        f"but collectively unhelpful. Current hand-coded proxies cannot approximate "
        f"oracle error ranking."
    )
    confident_error_proxy_supported = False
    candidate_ready_for_small_multiseed = False
    next_recommended_route = "learned_probe_value_critic_from_effective_harmful_logs"

# Determine if any C19 variant improved or matched C15b
if c19c_overlap_oracle > c19a_overlap_oracle and c19c_overlap_oracle > c19b_overlap_oracle:
    ranking_improvement_note = (
        f"C19c selected more oracle-like probes ({c19c_overlap_oracle}/{c19c_probed_n} overlap "
        f"with oracle k=8) than C19a ({c19a_overlap_oracle}) or C19b ({c19b_overlap_oracle}), "
        f"but macro_bal did {'improve' if c19c_improves_c15b else 'not improve'}."
    )
else:
    best_oracle_overlap = max(c19a_overlap_oracle, c19b_overlap_oracle, c19c_overlap_oracle)
    best_overlap_var = [n for n, (_, _, _, ov, _, _) in c19_variants.items() if ov == best_oracle_overlap][0]
    ranking_improvement_note = (
        f"Best oracle k=8 overlap: {best_overlap_var} at {best_oracle_overlap} pairs. "
        f"None of the C19 variants recover oracle-like probe pairs at high rate."
    )

print(f"  C19c macro_bal={c19c_mb:.4f}  delta_vs_C15b={c19c_delta:+.4f}")
print(f"  C19c hits 10% target: {c19c_hits_10pct}")
print(f"  C19c improves C15b: {c19c_improves_c15b}")
print(f"  confident_error_proxy_supported: {confident_error_proxy_supported}")
print(f"  candidate_ready_for_small_multiseed: {candidate_ready_for_small_multiseed}")
print(f"  next_recommended_route: {next_recommended_route}")

# Per-query
per_query_interp = {}
for qname in QUERY_NAMES:
    c15b_bal = results["C15b_probe_reserve_original"]["metrics"]["per_query"][qname]["balanced_accuracy"]
    c19c_bal = results["C19c_confident_error_risk"]["metrics"]["per_query"][qname]["balanced_accuracy"]
    delta = c19c_bal - c15b_bal
    if delta > 0.01:
        per_query_interp[qname] = f"improved (+{delta:.4f})"
    elif delta < -0.01:
        per_query_interp[qname] = f"degraded ({delta:.4f})"
    else:
        per_query_interp[qname] = f"unchanged ({delta:.4f})"
    print(f"  {qname}: {per_query_interp[qname]}")


# =============================================================================
# 13. Save outputs
# =============================================================================
print("\n[8/9] Saving outputs...")

output = {
    "block_id": "1J16",
    "condition": "C4_instance_subtype_cued_v1",
    "seed": SEED,
    "budget": BUDGET,
    "cost_weight": COST_WEIGHT,
    "desc": "Confident-Error Probe Ranking Diagnostic. Tests support/disagreement-based risk signals (C19a/b/c) for probe selection.",
    "iom_architecture": {
        "no_cross_test_object_propagation": True,
        "global_recompute_equivalent_to_local": True,
    },
    "diagnostic_caveats": {
        "hand_coded_proxies_not_full_solution": True,
        "no_cross_test_object_propagation_limits_value": True,
        "confident_error_proxy_uses_only_iom_internals": True,
    },
    "design_validation": {
        "no_environment_change": True,
        "no_budget_cost_change": True,
        "no_multi_seed": True,
        "no_hidden_labels_in_policy": True,
        "no_oracle_probe_outcomes": True,
        "no_IOM_update_change": True,
        "no_Route_F_F2_global_coarse_map": True,
    },
    "baselines": {},
    "C19a_details": {},
    "C19b_details": {},
    "C19c_details": {},
    "oracle_k8_reference": {},
    "comparisons": comparisons,
    "interpretation": {
        "primary_variant": "C19c_confident_error_risk",
        "best_c19_variant": best_c19_name,
        "best_c19_macro_bal": best_c19_mb,
        "c19c_macro_bal": c19c_mb,
        "c19c_delta_vs_c15b": round(c19c_delta, 6),
        "c19c_gap_capture_vs_c0b": round((c19c_mb - c0b_mb) / oracle_c0b_gap, 6) if oracle_c0b_gap > 0 else 0.0,
        "c19c_hits_10pct_target": c19c_hits_10pct,
        "confident_error_proxy_supported": confident_error_proxy_supported,
        "candidate_ready_for_small_multiseed": candidate_ready_for_small_multiseed,
        "ready_for_multiseed": False,
        "next_recommended_route": next_recommended_route,
        "c19c_interpretation": c19c_interpretation,
        "ranking_improvement_note": ranking_improvement_note,
        "per_query_interpretation": per_query_interp,
    },
}

# Serialize baselines
for name, r in results.items():
    output["baselines"][name] = {
        "macro_query_balanced_accuracy": r["metrics"]["macro_query_balanced_accuracy"],
        "macro_query_accuracy": r["metrics"]["macro_query_accuracy"],
        "mean_positive_recall": r["metrics"]["mean_positive_recall"],
        "mean_negative_recall": r["metrics"]["mean_negative_recall"],
        "per_query": r["metrics"]["per_query"],
        "visit_count": r["cost"]["visit_count"],
        "observe_count": r["cost"]["observe_count"],
        "probe_count": r["cost"]["probe_count"],
        "total_cost": r["cost"]["total_cost"],
        "normalized_cost": r["cost"]["normalized_cost"],
        "policy_type": r["policy_type"],
    }

# C19a details
output["C19a_details"] = {
    "variant": "neighbor_disagreement_risk",
    "scoring": "outcome_variance_among_kNN_neighbors",
    "macro_query_balanced_accuracy": c19a_mb,
    "mean_positive_recall": c19a_pos_recall,
    "mean_negative_recall": c19a_neg_recall,
    "per_query": results["C19a_neighbor_disagreement_risk"]["metrics"]["per_query"],
    "visit_count": c19a_visited_n,
    "probe_count": c19a_probed_n,
    "total_cost": results["C19a_neighbor_disagreement_risk"]["cost"]["total_cost"],
    "normalized_cost": results["C19a_neighbor_disagreement_risk"]["cost"]["normalized_cost"],
    "probe_pairs": str(c19a_probe_pairs),
    "selected_probe_actions": dict(Counter(c19a_selected_actions)),
    "effective_probe_count": c19a_effective,
    "zero_gain_probe_count": c19a_zero_gain,
    "harmful_probe_count": c19a_harmful,
    "overlap_with_C15b": c19a_overlap_c15b,
    "overlap_with_oracle_k8": c19a_overlap_oracle,
    "mean_selected_disagreement": round(c19a_mean_disagreement, 6),
    "delta_vs_C15b": round(c19a_delta, 6),
    "runtime_seconds": round(c19a_runtime, 1),
}

# C19b details
output["C19b_details"] = {
    "variant": "low_support_risk",
    "scoring": "inverse_composite_support_score",
    "macro_query_balanced_accuracy": c19b_mb,
    "mean_positive_recall": c19b_pos_recall,
    "mean_negative_recall": c19b_neg_recall,
    "per_query": results["C19b_low_support_risk"]["metrics"]["per_query"],
    "visit_count": c19b_visited_n,
    "probe_count": c19b_probed_n,
    "total_cost": results["C19b_low_support_risk"]["cost"]["total_cost"],
    "normalized_cost": results["C19b_low_support_risk"]["cost"]["normalized_cost"],
    "probe_pairs": str(c19b_probe_pairs),
    "selected_probe_actions": dict(Counter(c19b_selected_actions)),
    "effective_probe_count": c19b_effective,
    "zero_gain_probe_count": c19b_zero_gain,
    "harmful_probe_count": c19b_harmful,
    "overlap_with_C15b": c19b_overlap_c15b,
    "overlap_with_oracle_k8": c19b_overlap_oracle,
    "mean_selected_support_score": round(c19b_mean_support, 6),
    "delta_vs_C15b": round(c19b_delta, 6),
    "runtime_seconds": round(c19b_runtime, 1),
}

# C19c details
output["C19c_details"] = {
    "variant": "confident_error_risk_primary",
    "scoring": "confidence_times_disagreement_times_query_tiebreaker",
    "macro_query_balanced_accuracy": c19c_mb,
    "mean_positive_recall": c19c_pos_recall,
    "mean_negative_recall": c19c_neg_recall,
    "per_query": results["C19c_confident_error_risk"]["metrics"]["per_query"],
    "visit_count": c19c_visited_n,
    "probe_count": c19c_probed_n,
    "total_cost": results["C19c_confident_error_risk"]["cost"]["total_cost"],
    "normalized_cost": results["C19c_confident_error_risk"]["cost"]["normalized_cost"],
    "probe_pairs": str(c19c_probe_pairs),
    "probe_scores": c19c_probe_scores,
    "selected_probe_actions": dict(Counter(c19c_selected_actions)),
    "effective_probe_count": c19c_effective,
    "zero_gain_probe_count": c19c_zero_gain,
    "harmful_probe_count": c19c_harmful,
    "overlap_with_C15b": c19c_overlap_c15b,
    "overlap_with_oracle_k8": c19c_overlap_oracle,
    "mean_selected_confidence": round(c19c_mean_confidence, 6),
    "mean_selected_disagreement": round(c19c_mean_disagreement, 6),
    "mean_selected_risk_score": round(c19c_mean_risk, 6),
    "delta_vs_C15b": round(c19c_delta, 6),
    "runtime_seconds": round(c19c_runtime, 1),
}

# Oracle k=8 reference
output["oracle_k8_reference"] = {
    "label": "diagnostic_non_deployable",
    "k": 8,
    "pairs": str([(oid, action) for oid, action, _, _, _ in oracle_k8_pairs]),
    "macro_bal": oracle_k8_mb,
    "delta_vs_C15b": round(oracle_k8_mb - c15b_mb, 6),
    "uses_all_visible_features": True,
    "ranking_method": "|IOM_prediction - ground_truth|",
    "note": "Non-deployable reference. Uses oracle ground truth for ranking only.",
}

# Save JSON
json_path = os.path.join(CURRENT_DIR, "runs",
                         "calibration_block1j16_confident_error_probe_ranking_c4_seed101.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"  Saved: {json_path}")

elapsed = time.time() - t0
print(f"\n{'=' * 60}")
print(f"[block_done]")
print(f"block_id=1J16")
print(f"c15b_macro_bal={c15b_mb:.6f}")
print(f"c15b_probe_count={c15b_probed_n}")
print(f"c15b_probe_pairs={c15b_probe_pairs}")
print(f"c18b_macro_bal={c18b_mb:.6f}")
print(f"c18b_delta_vs_c15b={c18b_delta:+.6f}")
print(f"c19a_macro_bal={c19a_mb:.6f}")
print(f"c19a_delta_vs_c15b={c19a_delta:+.6f}")
print(f"c19a_overlap_oracle_k8={c19a_overlap_oracle}")
print(f"c19b_macro_bal={c19b_mb:.6f}")
print(f"c19b_delta_vs_c15b={c19b_delta:+.6f}")
print(f"c19b_overlap_oracle_k8={c19b_overlap_oracle}")
print(f"c19c_macro_bal={c19c_mb:.6f}")
print(f"c19c_delta_vs_c15b={c19c_delta:+.6f}")
print(f"c19c_overlap_oracle_k8={c19c_overlap_oracle}")
print(f"c19c_gap_capture_vs_c0b={(c19c_mb - c0b_mb) / oracle_c0b_gap:.6f}")
print(f"c19c_hits_10pct_target={'true' if c19c_hits_10pct else 'false'}")
print(f"best_c19_variant={best_c19_name}")
print(f"best_c19_macro_bal={best_c19_mb:.6f}")
print(f"best_c19_delta_vs_c15b={best_c19_delta:+.6f}")
print(f"best_c19_gap_capture_vs_c0b={(best_c19_mb - c0b_mb) / oracle_c0b_gap:.6f}")
print(f"best_c19_overlap_with_oracle_k8={c19_variants[best_c19_name][3]}")
print(f"best_c19_effective_probe_count={best_c19_eff}")
print(f"best_c19_harmful_probe_count={best_c19_harm}")
print(f"oracle_k8_macro_bal={oracle_k8_mb:.6f}")
print(f"oracle_k8_pairs={[(oid, action) for oid, action, _, _, _ in oracle_k8_pairs]}")
print(f"confident_error_proxy_supported={'true' if confident_error_proxy_supported else 'false'}")
print(f"candidate_ready_for_small_multiseed={'true' if candidate_ready_for_small_multiseed else 'false'}")
print(f"next_recommended_route={next_recommended_route}")
print(f"ready_for_multiseed=false")
print(f"no_cross_test_object_propagation=true")
print(f"c19a_runtime_seconds={c19a_runtime:.1f}")
print(f"c19b_runtime_seconds={c19b_runtime:.1f}")
print(f"c19c_runtime_seconds={c19c_runtime:.1f}")
print(f"elapsed_seconds={elapsed:.1f}")


# =============================================================================
# 14. Protocol MD
# =============================================================================
print("\n[9/9] Writing protocol MD...")

per_query_rows = []
for qname in QUERY_NAMES:
    c15b_q = results["C15b_probe_reserve_original"]["metrics"]["per_query"][qname]
    c18b_q = results["C18b_query_relevant_fixed_budget"]["metrics"]["per_query"][qname]
    c19c_q = results["C19c_confident_error_risk"]["metrics"]["per_query"][qname]
    per_query_rows.append(
        f"| {qname} | {c15b_q['balanced_accuracy']:.4f} | {c18b_q['balanced_accuracy']:.4f} | "
        f"{c19c_q['balanced_accuracy']:.4f} | {c19c_q['balanced_accuracy'] - c15b_q['balanced_accuracy']:+.4f} |"
    )

c19_summary_rows = []
for name in ["C19a_neighbor_disagreement_risk", "C19b_low_support_risk", "C19c_confident_error_risk"]:
    r = results[name]
    mb = r["metrics"]["macro_query_balanced_accuracy"]
    delta = mb - c15b_mb
    probed = r["cost"]["probe_count"]
    pairs = r.get("probe_pairs", [])
    ov_c15b = len(set(pairs) & c15b_probe_pair_set)
    ov_oracle = len(set(pairs) & oracle_k8_pair_set)
    c19_summary_rows.append(
        f"| {name.split('_')[0].upper()} | {mb:.4f} | {delta:+.4f} | {probed} | "
        f"{ov_c15b} | {ov_oracle} |"
    )

protocol_md = f"""# Block 1J16 — Confident-Error Probe Ranking Diagnostic

**Date:** 2026-05-11
**Block:** 1J16 — Confident-error / calibration-aware probe ranking diagnostic
**Condition:** C4_instance_subtype_cued_v1
**Seed:** 101
**Budget:** 1.5

---

## 1. Executive Summary

1J16 tests whether support/disagreement-based risk signals can identify diagnostic probes better than uncertainty-based ranking. Three C19 variants replace C15b's affordance-utility VOI with calibration-aware risk scoring while keeping the same observe phase:

- **C19a**: Neighbor disagreement risk — score = outcome_variance among kNN neighbors.
- **C19b**: Low support risk — score = 1.0 - composite support (neighbor count, similarity weight, agreement).
- **C19c (PRIMARY)**: Confident error risk — score = confidence * disagreement * (1 + query_relevance tiebreaker).

**C19c reaches macro_bal={c19c_mb:.4f} (delta_vs_C15b={c19c_delta:+.4f}).** {c19c_interpretation}

**Oracle k=8 reference**: {oracle_k8_mb:.4f} (non-deployable, ranks by |IOM_pred - gt| on all 60 objects).

---

## 2. Motivation

| Prior Finding | Implication |
|--------------|-------------|
| 1J14: C15b_UB reaches 0.8546 with oracle probe selection | Probe information has substantial usable value |
| 1J14: Oracle k=8 reaches 0.6488 vs C15b 0.6142 | C15b's VOI selects wrong probes — +0.0346 from better selection |
| 1J15: Query-relevant ranking (C18b) = 0.5975 | Uncertainty-based ranking does not differentiate from affordance utility |
| 1J15: C18a = C15b = 0.6142 | min(p,1-p) is monotonically equivalent to C15b's utility |
| IOM: predict_outcome returns confidence + support info | IOM has internal calibration data that VOI ignores |

**Key question**: Can the IOM's internal calibration signals (confidence, neighbor disagreement, support strength) identify probes that oracle error ranking would select, better than uncertainty-based proxies?

**Hypothesis**: The oracle k=8 ranking selects pairs where IOM is confident but WRONG (|pred - gt| is large). These are cases where neighbor disagreement should be high (neighbors split on outcomes) while IOM confidence may still be moderate-to-high (kNN produces a weighted average that happens to be far from truth). A risk proxy of confidence * disagreement should correlate with oracle error better than raw uncertainty.

---

## 3. C19 Variant Designs

### 3.1 Common Architecture

All C19 variants share C15b's observe phase:
- Nearest-first object visitation
- 25% budget reserve
- Same transition logic to probe phase

Only the probe ranking is replaced. Fixed probe budget = C15b's probe count (k≈{c15b_probed_n}).

### 3.2 Risk Scoring Functions

The scoring uses `InstanceOutcomeMemory.get_object_dynamic_indicators()` which returns per-action:
- `outcome_variance`: variance of neighbor outcomes (disagreement)
- `total_sim_weight`: sum of similarity weights
- `neighbor_count`: number of kNN matches found

**IOM confidence** (replicated from `predict_outcome` internals):
```
confidence = 0.4 * min(total_weight / n_neighbors, 1)
           + 0.3 * min(n_neighbors / k, 1)
           + 0.3 * (1 - min(outcome_variance / 0.25, 1))
```

### 3.3 C19a — Neighbor Disagreement Risk

Score = `outcome_variance`

Higher variance = neighbors disagree more about this outcome = more probe value.

### 3.4 C19b — Low Support Risk

```
support = 0.35 * min(neighbor_count/k, 1)
        + 0.35 * min(total_sim_weight/neighbor_count, 1)
        + 0.30 * (1 - min(outcome_variance/0.25, 1))
risk = 1.0 - support
```

Low support = few/weak neighbors or high disagreement = high risk.

### 3.5 C19c — Confident Error Risk (PRIMARY)

```
risk = confidence * outcome_variance * (0.5 + 0.5 * min(p, 1-p))
```

- **confidence**: high when IOM is sure (many similar neighbors agree)
- **outcome_variance**: high when neighbors disagree
- **min(p, 1-p)**: query-relevance tiebreaker (near decision boundary)

The product `confidence * disagreement` targets the confident-error pattern: IOM is confident but the evidence base is split. This should correlate with oracle |pred - gt|.

---

## 4. Forbidden-Information Compliance

| Check | Status |
|-------|--------|
| No environment change | ✓ |
| No budget/cost change | ✓ |
| Single seed only | ✓ |
| No hidden labels in policy | ✓ |
| No oracle probe outcomes | ✓ |
| No IOM update change | ✓ |
| No Route F/F2/global coarse map | ✓ |
| No hidden subtype/category in ranking | ✓ |
| Uses only IOM internals + visible features | ✓ |

---

## 5. Baseline Results

| Policy | macro_bal | pos_recall | neg_recall | visited | probed | ncost | Type |
|--------|-----------|------------|------------|---------|--------|-------|------|
| all_false | {results['all_false']['metrics']['macro_query_balanced_accuracy']:.4f} | {results['all_false']['metrics']['mean_positive_recall']:.4f} | {results['all_false']['metrics']['mean_negative_recall']:.4f} | 0 | 0 | 0.0000 | floor |
| C_minus_prior_only | {results['C_minus_prior_only']['metrics']['macro_query_balanced_accuracy']:.4f} | — | — | 0 | 0 | 0.0000 | no_interaction |
| C0b_original | {c0b_mb:.4f} | {results['C0b_original']['metrics']['mean_positive_recall']:.4f} | {results['C0b_original']['metrics']['mean_negative_recall']:.4f} | {results['C0b_original']['cost']['visit_count']} | 0 | {results['C0b_original']['cost']['normalized_cost']:.4f} | observe_only |
| C13_instance_VOI_original | {c13_mb:.4f} | — | — | {c13_visited_n} | {c13_probed_n} | {results['C13_instance_VOI_original']['cost']['normalized_cost']:.4f} | instance_VOI |
| C15b_probe_reserve_original | {c15b_mb:.4f} | {c15b_pos_recall:.4f} | {c15b_neg_recall:.4f} | {c15b_visited_n} | {c15b_probed_n} | {results['C15b_probe_reserve_original']['cost']['normalized_cost']:.4f} | two_pass_VOI |
| C18b_query_relevant (1J15) | {c18b_mb:.4f} | {c18b_pos_recall:.4f} | {c18b_neg_recall:.4f} | {c18b_visited_n} | {c18b_probed_n} | {results['C18b_query_relevant_fixed_budget']['cost']['normalized_cost']:.4f} | query_relevant_fixed |
| C_oracle_full_information | {oracle_mb:.4f} | 1.0000 | 1.0000 | — | — | — | oracle |

### Oracle References (non-deployable)

| Variant | macro_bal | delta_vs_C15b |
|---------|-----------|---------------|
| 1J14 C15b_UB | 0.8546 | +0.2404 |
| 1J14 IOM k=8 | 0.6488 | +0.0346 |
| 1J16 Oracle k=8 | {oracle_k8_mb:.4f} | {oracle_k8_mb - c15b_mb:+.4f} |
| Full_UB | 1.0000 | +0.3858 |

Oracle-C0b gap: **{oracle_c0b_gap:.4f}**. C15b captures {(c15b_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}%.

---

## 6. C19 Results

### 6.1 Aggregate Metrics

| Metric | C15b | C18b | C19a | C19b | C19c (PRIMARY) |
|--------|------|------|------|------|----------------|
| macro_bal | {c15b_mb:.4f} | {c18b_mb:.4f} | {c19a_mb:.4f} | {c19b_mb:.4f} | **{c19c_mb:.4f}** |
| mean_positive_recall | {c15b_pos_recall:.4f} | {c18b_pos_recall:.4f} | {c19a_pos_recall:.4f} | {c19b_pos_recall:.4f} | {c19c_pos_recall:.4f} |
| mean_negative_recall | {c15b_neg_recall:.4f} | {c18b_neg_recall:.4f} | {c19a_neg_recall:.4f} | {c19b_neg_recall:.4f} | {c19c_neg_recall:.4f} |
| visit_count | {c15b_visited_n} | {c18b_visited_n} | {c19a_visited_n} | {c19b_visited_n} | {c19c_visited_n} |
| probe_count | {c15b_probed_n} | {c18b_probed_n} | {c19a_probed_n} | {c19b_probed_n} | {c19c_probed_n} |
| delta_vs_C15b | — | {c18b_delta:+.4f} | {c19a_delta:+.4f} | {c19b_delta:+.4f} | **{c19c_delta:+.4f}** |
| gap_capture | {(c15b_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}% | {(c18b_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}% | {(c19a_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}% | {(c19b_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}% | **{(c19c_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}%** |

### 6.2 Per-Query Breakdown

| Query | C15b Bal | C18b Bal | C19c Bal | C19c Δ vs C15b |
|-------|----------|----------|----------|----------------|
{chr(10).join(per_query_rows)}

### 6.3 Probe Selection

| Metric | C15b | C18b | C19a | C19b | C19c |
|--------|------|------|------|------|------|
| probe_count | {c15b_probed_n} | {c18b_probed_n} | {c19a_probed_n} | {c19b_probed_n} | {c19c_probed_n} |
| actions | {dict(Counter(c15b_selected_actions))} | {dict(Counter(c18b_selected_actions))} | {dict(Counter(c19a_selected_actions))} | {dict(Counter(c19b_selected_actions))} | {dict(Counter(c19c_selected_actions))} |
| effective | — | — | {c19a_effective} | {c19b_effective} | {c19c_effective} |
| zero_gain | — | — | {c19a_zero_gain} | {c19b_zero_gain} | {c19c_zero_gain} |
| harmful | — | — | {c19a_harmful} | {c19b_harmful} | {c19c_harmful} |
| mean_confidence | — | — | — | — | {c19c_mean_confidence:.4f} |
| mean_disagreement | — | — | {c19a_mean_disagreement:.4f} | — | {c19c_mean_disagreement:.4f} |

### 6.4 Pair Overlap Analysis

| Variant | Pairs | Overlap C15b | Overlap Oracle k=8 |
|---------|-------|-------------|-------------------|
| C15b | {c15b_probe_pairs} | {c15b_probed_n} | {len(c15b_probe_pair_set & oracle_k8_pair_set)} |
| C18b | {c18b_probe_pairs} | {len(set(c18b_probe_pairs) & c15b_probe_pair_set)} | {len(set(c18b_probe_pairs) & oracle_k8_pair_set)} |
| C19a | {c19a_probe_pairs} | {c19a_overlap_c15b} | {c19a_overlap_oracle} |
| C19b | {c19b_probe_pairs} | {c19b_overlap_c15b} | {c19b_overlap_oracle} |
| C19c | {c19c_probe_pairs} | {c19c_overlap_c15b} | {c19c_overlap_oracle} |
| Oracle k=8 | {[(oid, action) for oid, action, _, _, _ in oracle_k8_pairs]} | {len(oracle_k8_pair_set & c15b_probe_pair_set)} | {len(oracle_k8_pairs)} |

---

## 7. Key Comparisons

### 7.1 C19c vs C15b (PRIMARY)

| Metric | Delta |
|--------|-------|
| delta_macro_bal | **{c19c_delta:+.4f}** |
| delta_positive_recall | {c19c_pos_recall - c15b_pos_recall:+.4f} |
| delta_negative_recall | {c19c_neg_recall - c15b_neg_recall:+.4f} |
| delta_visit_count | {c19c_visited_n - c15b_visited_n} |
| delta_probe_count | {c19c_probed_n - c15b_probed_n} |
| C19c hits 10% threshold ({target_10pct}) | **{'Yes' if c19c_hits_10pct else 'No'}** |
| overlap_with_oracle_k8 | {c19c_overlap_oracle}/{len(oracle_k8_pairs)} |

### 7.2 C19 Variant Comparison

| Variant | macro_bal | delta_vs_C15b | probed | overlap_C15b | overlap_oracle_k8 |
|---------|-----------|---------------|--------|-------------|-------------------|
{chr(10).join(c19_summary_rows)}

### 7.3 Gap Capture

| Policy | gap_capture |
|--------|-------------|
| C15b | {(c15b_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}% |
| C18b (1J15) | {(c18b_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}% |
| C19a | {(c19a_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}% |
| C19b | {(c19b_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}% |
| C19c | {(c19c_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}% |
| Oracle k=8 | {(oracle_k8_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}% |
| 1J14 C15b_UB | 64.78% |

---

## 8. Interpretation

```
C19c improves over C15b:     {'True' if c19c_improves_c15b else 'False'} ({c19c_delta:+.4f})
C19c hits 10% threshold:     {'True' if c19c_hits_10pct else 'False'} ({c19c_mb:.4f} vs {target_10pct})
confident_error_proxy_supported: {confident_error_proxy_supported}
candidate_ready_for_small_multiseed: {candidate_ready_for_small_multiseed}
ready_for_multiseed:          False
no_cross_test_object_propagation: true
```

**C19c interpretation: {c19c_interpretation}**

**{ranking_improvement_note}**

### Per-Query Interpretation

{chr(10).join(f"- **{q}**: {interp}" for q, interp in per_query_interp.items())}

---

## 9. Why Hand-Coded Risk Proxies May Not Differentiate

### 9.1 The Oracle Error Signal

Oracle k=8 ranks by `|IOM_pred - ground_truth|`. This is a SUPERVISED error signal — it knows the correct answer. Hand-coded proxies attempt to approximate this from internal IOM state alone.

### 9.2 Limitations of Current Proxies

1. **Disagreement != error**: Neighbors can disagree because the test case is genuinely ambiguous (true p≈0.5), not because the IOM is wrong. Disagreement captures ambiguity, not error.

2. **Confidence * disagreement captures a specific pattern**: High confidence + high disagreement suggests the IOM is interpolating between conflicting evidence. But this pattern may not dominate the oracle error distribution.

3. **Low support != wrong**: A prediction with few weak neighbors may be correct (the training data is complete enough). Low support correlates with extrapolation, not necessarily error.

4. **No ground-truth signal**: Oracle k=8 uses `|pred - gt|` — the hand-coded proxy has no access to gt. Without error supervision, the proxy can only guess which predictions are wrong.

### 9.3 Diagnostic Caveats

**Under current IOM architecture:**
- Test-object probes do NOT propagate to other test objects.
- The IOM's internal calibration signals (confidence, variance, support) are the only available risk indicators besides prediction probabilities.
- None of these signals have access to ground truth.

**C19 is NOT claimed as a full solution.** C19 tests whether simple hand-coded proxies can approximate oracle error ranking from IOM internals alone.

---

## 10. Final Recommendation

**{"Confident-error proxy is supported. Candidate for small multiseed (3 seeds)." if candidate_ready_for_small_multiseed else "Confident-error proxy does not differentiate from C15b's VOI. Hand-coded risk proxies cannot approximate oracle error from IOM internals alone." if not confident_error_proxy_supported else "Confident-error proxy shows marginal improvement but insufficient for multiseed."}**

**Next recommended route: {next_recommended_route}.**

The effective/harmful probe logs from 1J14, 1J15, and 1J16 provide labeled data: which (object, action) probes were effective (improved query scores) and which were harmful (degraded scores). A learned probe-value critic trained on these logs could potentially identify patterns that hand-coded proxies miss.

### Consolidated Block 1J11-1J16 Findings

| Block | Approach | Result | vs C15b | Key Finding |
|-------|----------|--------|---------|-------------|
| 1J11 | F_mid aggregate peripheral | 0.6142 | +0.0000 | Safe but not useful |
| 1J12 | Global coarse scene map (C15G) | 0.5992 | -0.0150 | Scene priority misaligns with diagnostic value |
| 1J13 | Interleaved breadth-depth (C17) | 0.5667 | -0.0475 | VOI doesn't activate at low breadth |
| 1J14 | Oracle probe injection (C15b_UB) | 0.8546 | +0.2404 | Substantial probe value exists; VOI fails to find it |
| 1J15 | Query-relevant ranking (C18b) | {c18b_mb:.4f} | {c18b_delta:+.4f} | Uncertainty proxy = affordance utility; fixed budget hurts |
| **1J16** | **Confident-error risk (C19c)** | **{c19c_mb:.4f}** | **{c19c_delta:+.4f}** | **{c19c_interpretation[:120]}...** |

---

*Generated by Block 1J16 — Confident-Error / Calibration-Aware Probe Ranking Diagnostic.*
"""

protocol_path = os.path.join(CURRENT_DIR, "protocols",
                             "block1j16_confident_error_probe_ranking_c4_seed101.md")
with open(protocol_path, "w", encoding="utf-8") as f:
    f.write(protocol_md)
print(f"  Saved: {protocol_path}")

print(f"\nDone! elapsed={elapsed:.1f}s")
