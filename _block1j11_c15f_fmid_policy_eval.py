"""
Block 1J11 — C15F Policy Evaluation using F_mid Peripheral Observation Interface.

Evaluates C15F variants (context-guided observe and/or probe) atop the C15b
two-pass architecture, using the audited F_mid peripheral signal.

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
from event_log import EventLog
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
F_MID_CFG = {
    "d1": 2, "d2": 4,
    "bin_layer1_near": [1, 3],
    "bin_layer2_mid": [0],
    "bin_layer3_background": [0],
    "count_bins": [1, 3, 6],
    "description": "F_mid — moderate coarse local context",
}

COND = copy.deepcopy(config.CUE_CONDITIONS[3])
assert COND["label"] == "C4_instance_subtype_cued_v1"

print("=" * 60)
print("Block 1J11 — C15F Policy Evaluation (F_mid)")
print(f"  condition={COND['label']}  seed={SEED}  budget={BUDGET}")
print("=" * 60)

# =============================================================================
# 1. Training + IOM building
# =============================================================================
print("\n[1/8] Phase A: Training + IOM building...")
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
print("\n[2/8] Generating test objects...")
test_objects = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")
query_gt = compute_query_ground_truth(test_objects)
test_oids = sorted(test_objects.keys())
positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, rng)
sim_gt = MiniMCSimulatorTruth(test_objects, positions, config.AGENT_START)

# Object metadata
object_meta = {}
for oid in test_oids:
    obj = test_objects[oid]
    cat = obj["hidden_category"]
    subtype = obj["hidden_subtype"]
    gt_aff = sim_gt.get_ground_truth_affordances(oid)
    subtype_def = SUBTYPE_DEFINITIONS[cat]
    ratios = subtype_def["subtype_ratio"]
    is_majority = ratios.get(subtype, 0.0) >= 0.5
    object_meta[oid] = {
        "category": cat, "subtype": subtype, "is_majority": is_majority,
        "gt_affordances": gt_aff,
    }

# Public visible features for all objects
ALL_VISIBLE_FEATURES = {}
for oid in test_oids:
    ALL_VISIBLE_FEATURES[oid] = dict(test_objects[oid].get("visible_features", {}))

# Global base rates
GLOBAL_FEATURE_PREVALENCE = {}
for f in posterior_visible_features:
    prevalence = sum(1 for oid in test_oids if ALL_VISIBLE_FEATURES[oid].get(f, False)) / len(test_oids)
    GLOBAL_FEATURE_PREVALENCE[f] = prevalence

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


def get_cost_metrics(event_log, agent_obs):
    events = event_log.to_list()
    total_reach = sum(e.get("cost", 0.0) for e in events if e.get("event") == "reach")
    total_observe = sum(e.get("cost", 0.0) for e in events if e.get("event") == "observe")
    total_probe = sum(e.get("cost", 0.0) for e in events if e.get("event") == "probe")
    total_cost = event_log.total_cost()
    n_visited = sum(1 for oid in agent_obs.get_all_object_ids() if agent_obs.is_visited(oid))
    n_probed = sum(1 for oid in agent_obs.get_all_object_ids() if agent_obs.get_probe_results(oid))
    init_budget = agent_obs.initial_budget
    return {
        "visit_count": n_visited,
        "observe_count": n_visited,
        "probe_count": n_probed,
        "total_cost": total_cost,
        "normalized_cost": total_cost / max(init_budget, 0.001),
        "total_reach_cost": total_reach,
        "total_observe_cost": total_observe,
        "total_probe_cost": total_probe,
        "budget_remaining": agent_obs.budget_remaining,
        "budget_spent": agent_obs.budget_spent,
        "initial_budget": init_budget,
    }


def run_policy(policy, env):
    harness = EpisodeHarness(env, policy)
    result = harness.run()
    return (result.predictions, result.event_log, result.agent_obs,
            result.pre_probe_entropies, result.pre_decision_entropies)


# Pareto check
def is_pareto_dominated(a_mb, a_cost, others):
    """Return True if (a_mb, a_cost) is Pareto-dominated by any entry in others.
    Each entry in others is (mb, cost). Higher mb is better, lower cost is better."""
    for o_mb, o_cost in others:
        if o_mb >= a_mb and o_cost <= a_cost and (o_mb > a_mb or o_cost < a_cost):
            return True
    return False


def pareto_frontier(policies_dict):
    """policies_dict: {name: (mb, cost)}. Returns set of non-dominated names."""
    items = list(policies_dict.items())
    nondominated = set()
    for name, (mb, cost) in items:
        others = [(v[0], v[1]) for k, v in items if k != name]
        if not is_pareto_dominated(mb, cost, others):
            nondominated.add(name)
    return nondominated


# =============================================================================
# 4. Peripheral signal functions (F_mid only)
# =============================================================================
def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def bin_count(count, bins):
    if count <= 0:
        return "0"
    for threshold in bins:
        if count <= threshold:
            return str(threshold) if threshold <= 1 else f"<={threshold}"
    return f">{bins[-1]}"


def compute_peripheral_signal(focal_position, focal_oid, blur_cfg):
    d1 = blur_cfg["d1"]
    d2 = blur_cfg["d2"]
    layers = {"layer1_near": [], "layer2_mid": [], "layer3_background": []}
    for oid in test_oids:
        if oid == focal_oid:
            continue
        pos = positions[oid]
        dist = manhattan(focal_position, pos)
        if dist == 0:
            continue
        if dist <= d1:
            layers["layer1_near"].append((oid, dist))
        elif dist <= d2:
            layers["layer2_mid"].append((oid, dist))
        else:
            layers["layer3_background"].append((oid, dist))
    result = {}
    for layer_name, objects_in_layer in layers.items():
        n = len(objects_in_layer)
        if n == 0:
            result[layer_name] = {"object_count": 0, "count_binned": "0",
                                  "feature_histogram": {}, "density": 0.0}
            continue
        feature_counts = {}
        for oid, dist in objects_in_layer:
            vis = ALL_VISIBLE_FEATURES.get(oid, {})
            for f in posterior_visible_features:
                if vis.get(f, False):
                    feature_counts[f] = feature_counts.get(f, 0) + 1
        bin_cfg = blur_cfg.get(f"bin_{layer_name}", [1])
        result[layer_name] = {
            "object_count": n,
            "count_binned": bin_count(n, blur_cfg.get("count_bins", [1, 3, 6])),
            "feature_histogram": {
                f: bin_count(feature_counts.get(f, 0), bin_cfg)
                for f in posterior_visible_features
            },
            "density": round(n / max(config.GRID_ROWS * config.GRID_COLS, 1), 4),
        }
    return result


def compute_context_diversity(periph):
    """Compute a public-visible-cue diversity score from a peripheral signal.

    Measures how varied the public visible features are in the local neighborhood.
    Uses ONLY public visible features — never hidden state.

    Returns a float in [0, 1]. Higher = more diverse local context.
    """
    # Combine layer 1 (near) and layer 2 (mid) histograms
    combined_counts = {}
    for layer_name in ["layer1_near", "layer2_mid"]:
        ld = periph.get(layer_name, {})
        hist = ld.get("feature_histogram", {})
        for f, val in hist.items():
            if val == "0":
                continue
            elif val == "1":
                combined_counts[f] = combined_counts.get(f, 0) + 1
            elif val.startswith("<="):
                try:
                    combined_counts[f] = combined_counts.get(f, 0) + int(val[2:])
                except ValueError:
                    combined_counts[f] = combined_counts.get(f, 0) + 1
            elif val.startswith(">"):
                combined_counts[f] = combined_counts.get(f, 0) + 3

    total_count = sum(combined_counts.values())
    if total_count == 0:
        return 0.0

    # Compute entropy of the feature distribution
    entropy = 0.0
    for f, count in combined_counts.items():
        p = count / total_count
        if p > 0:
            entropy -= p * math.log(p)

    # Normalize by max entropy (log of number of features present)
    n_features = len(posterior_visible_features)
    max_entropy = math.log(max(n_features, 1))
    if max_entropy == 0:
        return 0.0

    # Also factor in density
    n_near = periph.get("layer1_near", {}).get("object_count", 0)
    n_mid = periph.get("layer2_mid", {}).get("object_count", 0)
    density_factor = min((n_near + 0.5 * n_mid) / 10.0, 1.0)

    diversity = 0.7 * (entropy / max_entropy) + 0.3 * density_factor
    return round(diversity, 4)


def estimate_regional_diversity(oid, peripheral_signals, blur_cfg):
    """Estimate regional public-cue diversity around an unvisited object.

    Uses peripheral signals from already-visited positions. Finds the best
    (closest layer) peripheral signal that includes this object's position.
    Returns the context diversity score of that peripheral signal.

    Uses ONLY public visible features. Does not predict hidden state.
    """
    oid_pos = positions[oid]
    d1 = blur_cfg["d1"]
    d2 = blur_cfg["d2"]

    best_layer_rank = None
    best_periph = None

    for visitor_oid, periph in peripheral_signals.items():
        visitor_pos = positions[visitor_oid]
        dist = manhattan(visitor_pos, oid_pos)
        for layer_name, layer_rank in [("layer1_near", 0), ("layer2_mid", 1), ("layer3_background", 2)]:
            in_layer = False
            if layer_name == "layer1_near" and dist <= d1:
                in_layer = True
            elif layer_name == "layer2_mid" and d1 < dist <= d2:
                in_layer = True
            elif layer_name == "layer3_background" and dist > d2:
                in_layer = True
            if in_layer and (best_layer_rank is None or layer_rank < best_layer_rank):
                best_layer_rank = layer_rank
                best_periph = periph

    if best_periph is not None:
        return compute_context_diversity(best_periph)
    return 0.0


# =============================================================================
# 5. C15F Policy Variants
# =============================================================================

class C15F_ContextGuidedProbePolicy(MiniMCPolicy):
    """C15F V2: F_mid context guides probe prioritization among visited objects.

    Pass 1: Same broad observe as C15b (nearest-first).
    Pass 2: Rank already focally observed objects using C15b VOI + F_mid
            local-context diversity bonus. Probe only focally observed objects.
    """

    PHASE_OBSERVE = 1
    PHASE_PROBE = 2

    def __init__(self, instance_memory, rng, cost_weight=0.5, blur_cfg=None,
                 context_bonus_weight=0.15):
        self._im = instance_memory
        self._rng = rng
        self._cost_weight = cost_weight
        self._blur_cfg = blur_cfg or F_MID_CFG
        self._context_bonus_weight = context_bonus_weight
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._visit_order = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._observe_pass_cost = 0.0
        self._remaining_budget_after_observe = 0.0
        self._all_voi_scores = []
        self._peripheral_signals = {}
        self._context_scores = {}
        self._context_guided_decisions = 0
        self._selected_context_scores = []
        self._unselected_context_scores = []

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._visit_order = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._observe_pass_cost = 0.0
        self._remaining_budget_after_observe = 0.0
        self._all_voi_scores = []
        self._peripheral_signals = {}
        self._context_scores = {}
        self._context_guided_decisions = 0
        self._selected_context_scores = []
        self._unselected_context_scores = []

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
                self._observe_pass_cost = view.budget_spent
                self._remaining_budget_after_observe = view.budget_remaining
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

            # F_mid context bonus: boost VOI for objects in high-diversity regions
            ctx_score = self._context_scores.get(oid, 0.0)
            adjusted_voi = best_action_voi * (1.0 + self._context_bonus_weight * ctx_score)

            self._all_voi_scores.append({
                "oid": oid, "voi": best_action_voi, "adjusted_voi": adjusted_voi,
                "context_score": ctx_score, "affordable": affordable,
            })
            if affordable and adjusted_voi > best_score:
                best_score = adjusted_voi
                best_oid = oid

        if best_oid is not None:
            self._visit_order.append(("probe", best_oid))
            self._selected_context_scores.append(
                self._context_scores.get(best_oid, 0.0))
            self._context_guided_decisions += 1

        # Track unselected context scores for comparison
        for oid in view.get_all_object_ids():
            if view.is_visited(oid) and oid not in self._probed_oids and oid != best_oid:
                if oid in self._context_scores:
                    self._unselected_context_scores.append(self._context_scores[oid])

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

    def on_visit(self, view, object_id):
        """Called after observation — compute peripheral signal and context score."""
        pos = view.get_object_position(object_id)
        periph = compute_peripheral_signal(pos, object_id, self._blur_cfg)
        self._peripheral_signals[object_id] = periph
        ctx = compute_context_diversity(periph)
        self._context_scores[object_id] = ctx

    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            per_object[oid] = self._im.predict_all_affordances(
                {"id": oid, "visible_features": features})
        return {"per_object": per_object}

    def get_pre_probe_entropies(self):
        return dict(self._pre_probe_entropies)

    def get_probe_efficiency(self, view):
        """Classify probes as effective, zero_gain, or harmful."""
        effective = 0
        zero_gain = 0
        harmful = 0
        for pt in self._probe_tracking:
            oid = pt["oid"]
            pre = pt["pre_probs"]
            post = pt["post_probs"]
            pre_u = _compute_utility(pre)
            post_u = _compute_utility(post)
            delta = post_u - pre_u
            if delta > 0.001:
                effective += 1
            elif delta < -0.001:
                harmful += 1
            else:
                zero_gain += 1
        total = effective + zero_gain + harmful
        return {
            "effective_probe_count": effective,
            "zero_gain_probe_count": zero_gain,
            "harmful_probe_count": harmful,
            "total_probe_count": total,
            "effective_probe_rate": effective / max(total, 1),
        }

    def get_selected_object_stats(self, view):
        """Majority vs minority breakdown of selected probe objects."""
        majority = 0
        minority = 0
        per_query_pos = {q: 0 for q in query_gt}
        per_query_neg = {q: 0 for q in query_gt}
        per_query_visited_pos = {q: 0 for q in query_gt}
        per_query_visited_neg = {q: 0 for q in query_gt}

        for oid in self._probed_oids:
            meta = object_meta.get(oid, {})
            if meta.get("is_majority", True):
                majority += 1
            else:
                minority += 1
            for qname, qdata in query_gt.items():
                if oid in qdata["positive_oids"]:
                    per_query_pos[qname] += 1
                else:
                    per_query_neg[qname] += 1

        for oid in view.get_all_object_ids():
            if view.is_visited(oid):
                for qname, qdata in query_gt.items():
                    if oid in qdata["positive_oids"]:
                        per_query_visited_pos[qname] += 1
                    else:
                        per_query_visited_neg[qname] += 1

        return {
            "selected_probe_objects_majority": majority,
            "selected_probe_objects_minority": minority,
            "selected_probe_objects_per_query_positive": per_query_pos,
            "selected_probe_objects_per_query_negative": per_query_neg,
            "visited_positive_coverage_per_query": per_query_visited_pos,
            "visited_negative_coverage_per_query": per_query_visited_neg,
        }


class C15F_ContextGuidedObservePolicy(C15F_ContextGuidedProbePolicy):
    """C15F V1: F_mid context guides observation target selection.

    Pass 1: Broad observe preferring objects in high public-cue-diversity regions.
    Pass 2: Same probe selection as C15b (VOI-based, no context bonus).
    """

    def __init__(self, instance_memory, rng, cost_weight=0.5, blur_cfg=None,
                 diversity_weight=0.5):
        super().__init__(instance_memory, rng, cost_weight, blur_cfg,
                         context_bonus_weight=0.0)  # no context bonus in probe phase
        self._diversity_weight = diversity_weight

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        if self._phase == self.PHASE_OBSERVE:
            if self._should_transition_to_probe_phase(view):
                self._phase = self.PHASE_PROBE
                self._observe_pass_visit_count = sum(
                    1 for oid in view.get_all_object_ids() if view.is_visited(oid))
                self._observe_pass_cost = view.budget_spent
                self._remaining_budget_after_observe = view.budget_remaining
                return self._select_probe_target(view)

            # Context-guided observe: prefer nearest, but boost diversity
            best_oid = None
            best_score = float('-inf')
            for oid in unvisited:
                reach_cost = view.compute_reach_cost(oid)
                total = reach_cost + view.observe_cost
                if not view.can_afford(total):
                    continue
                # Estimate regional diversity from already-visited positions
                reg_div = estimate_regional_diversity(
                    oid, self._peripheral_signals, self._blur_cfg)
                # Score: diversity bonus normalized by cost
                diversity_bonus = self._diversity_weight * reg_div
                score = diversity_bonus / (1.0 + reach_cost)
                if score > best_score:
                    best_score = score
                    best_oid = oid
            if best_oid is not None:
                self._visit_order.append(("observe", best_oid))
                self._context_guided_decisions += 1
            return best_oid
        elif self._phase == self.PHASE_PROBE:
            return self._select_probe_target(view)
        return None

    def _select_probe_target(self, view):
        # V1: standard C15b VOI-based probe selection (no context bonus)
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
            self._all_voi_scores.append({
                "oid": oid, "voi": best_action_voi, "affordable": affordable,
            })
            if affordable and best_action_voi > best_score:
                best_score = best_action_voi
                best_oid = oid
        if best_oid is not None:
            self._visit_order.append(("probe", best_oid))
        return best_oid


# =============================================================================
# 6. Custom harness wrapper (calls policy.on_visit after observe)
# =============================================================================

class C15FHarness:
    """Harness that calls policy.on_visit() after each observation.

    Standard EpisodeHarness doesn't call on_visit. We need peripheral signal
    computation after each successful observation.
    """

    def __init__(self, environment, policy):
        self._env = environment
        self._policy = policy

    def run(self):
        obs = self._env.reset()
        self._policy.reset(obs)
        obs.event_log.append({"event": "episode_loop_start"})

        while not obs.is_terminal():
            target = self._policy.select_next_object(obs)
            if target is None:
                obs.event_log.append({
                    "event": "policy_stop",
                    "reason": "policy returned None",
                    "budget_remaining": obs.budget_remaining,
                })
                break

            reach_cost = obs.compute_reach_cost(target)
            total_pre_probe = reach_cost + obs.observe_cost
            if not obs.can_afford(total_pre_probe):
                obs.event_log.append({
                    "event": "policy_stop",
                    "reason": f"cannot afford reach+observe for {target}",
                    "budget_remaining": obs.budget_remaining,
                })
                break

            self._env.reach(obs, target)
            self._env.observe(obs, target)

            # Call on_visit for peripheral-aware policies
            if hasattr(self._policy, 'on_visit'):
                self._policy.on_visit(obs, target)

            should_probe, action = self._policy.decide_probe(obs, target)
            if should_probe:
                assert action != "tap_sound"
                if obs.can_afford(obs.probe_cost):
                    outcome_float, outcome_str = self._env.probe(obs, target, action)
                    self._policy.on_probe_result(obs, target, action, outcome_float)

        predictions = self._policy.get_answer(obs)
        obs.event_log.append({
            "event": "episode_end",
            "budget_remaining": obs.budget_remaining,
            "budget_spent": obs.budget_spent,
            "objects_visited": sum(1 for oid in obs.get_all_object_ids()
                                   if obs.is_visited(oid)),
        })

        pre_probe = {}
        pre_decision = {}
        if hasattr(self._policy, 'get_pre_probe_entropies'):
            pre_probe = self._policy.get_pre_probe_entropies()

        from harness import HarnessResult
        return HarnessResult(predictions, obs.event_log, obs,
                             pre_probe, pre_decision)


# =============================================================================
# 7. Run all baselines + C15F variants
# =============================================================================
print("\n[3/8] Running baselines...")

results = {}

# --- all_false ---
ALL_FALSE_PREDS = {oid: {f: 0.0 for f in CORE_ACTION_FEATURES} for oid in test_oids}
results["all_false"] = {
    "predictions": ALL_FALSE_PREDS,
    "metrics": compute_full_metrics(ALL_FALSE_PREDS, query_gt),
    "policy_type": "floor",
    "cost": {"visit_count": 0, "observe_count": 0, "probe_count": 0,
             "total_cost": 0.0, "normalized_cost": 0.0},
}
print(f"  all_false macro_bal={results['all_false']['metrics']['macro_query_balanced_accuracy']:.4f}")

# --- C_minus_prior_only ---
PRIOR_ONLY_PREDS = {oid: {f: GLOBAL_FEATURE_PREVALENCE.get(f, 0.5) for f in CORE_ACTION_FEATURES}
                    for oid in test_oids}
results["C_minus_prior_only"] = {
    "predictions": PRIOR_ONLY_PREDS,
    "metrics": compute_full_metrics(PRIOR_ONLY_PREDS, query_gt),
    "policy_type": "no_interaction",
    "cost": {"visit_count": 0, "observe_count": 0, "probe_count": 0,
             "total_cost": 0.0, "normalized_cost": 0.0},
}
print(f"  C_minus_prior_only macro_bal={results['C_minus_prior_only']['metrics']['macro_query_balanced_accuracy']:.4f}")

# --- C0b_original ---
c0b_im = im_base.clone()
c0b_policy = C0b_ObserveOnlyPolicy(c0b_im, random.Random(SEED + 100))
c0b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c0b_preds, c0b_log, c0b_obs, _, _ = run_policy(c0b_policy, c0b_env)
results["C0b_original"] = {
    "predictions": c0b_preds["per_object"],
    "metrics": compute_full_metrics(c0b_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c0b_log, c0b_obs),
    "policy_type": "observe_only_no_peripheral",
}
c0b_mb = results["C0b_original"]["metrics"]["macro_query_balanced_accuracy"]
print(f"  C0b_original macro_bal={c0b_mb:.4f}  visited={results['C0b_original']['cost']['visit_count']}")

# --- C0b_RouteF_Fmid_deployable (predictions identical to C0b) ---
results["C0b_RouteF_Fmid_deployable"] = {
    "predictions": c0b_preds["per_object"],  # identical to original C0b
    "metrics": compute_full_metrics(c0b_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c0b_log, c0b_obs),
    "policy_type": "observe_only_with_peripheral_aggregate_only",
    "note": "Predictions identical to original C0b — peripheral signal is aggregate-only.",
}
print(f"  C0b_RouteF_Fmid_deployable macro_bal={results['C0b_RouteF_Fmid_deployable']['metrics']['macro_query_balanced_accuracy']:.4f}  (identical to C0b)")

# --- C13_instance_VOI_original ---
c13_im = im_base.clone()
c13_policy = C13_InstanceVOIPolicy(c13_im, random.Random(SEED + 500), cost_weight=COST_WEIGHT)
c13_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c13_preds, c13_log, c13_obs, _, _ = run_policy(c13_policy, c13_env)
results["C13_instance_VOI_original"] = {
    "predictions": c13_preds["per_object"],
    "metrics": compute_full_metrics(c13_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c13_log, c13_obs),
    "policy_type": "instance_VOI_no_peripheral",
}
print(f"  C13_instance_VOI_original macro_bal={results['C13_instance_VOI_original']['metrics']['macro_query_balanced_accuracy']:.4f}  "
      f"visited={results['C13_instance_VOI_original']['cost']['visit_count']}  "
      f"probed={results['C13_instance_VOI_original']['cost']['probe_count']}")

# --- random_probe_original ---
class RandomProbePolicy(MiniMCPolicy):
    def __init__(self, im, rng):
        super().__init__()
        self._im = im
        self._rng = rng
        self._probed = []

    def reset(self, view):
        self._probed = []

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited:
            return None
        best_oid = None
        best_cost = float('inf')
        for oid in unvisited:
            total = view.compute_reach_cost(oid) + view.observe_cost
            if view.can_afford(total) and total < best_cost:
                best_cost = total
                best_oid = oid
        return best_oid

    def decide_probe(self, view, object_id):
        if view.can_afford(view.probe_cost) and self._rng.random() < 0.3:
            candidates = [a for a in MAIN_CANDIDATE_ACTIONS if a != "tap_sound"]
            action = self._rng.choice(candidates) if candidates else None
            if action:
                self._probed.append(object_id)
                return True, action
        return False, None

    def on_probe_result(self, view, object_id, action, outcome):
        pass

    def get_answer(self, view):
        per_object = {}
        for oid in test_oids:
            feats = view.get_observed_features(oid)
            if feats is not None:
                fake_obj = {"id": oid, "visible_features": dict(feats)}
                probs = self._im.predict_all_affordances(fake_obj)
                per_object[oid] = {f: probs.get(f, 0.5) for f in CORE_ACTION_FEATURES}
            else:
                per_object[oid] = {f: GLOBAL_FEATURE_PREVALENCE.get(f, 0.5)
                                   for f in CORE_ACTION_FEATURES}
        return {"per_object": per_object}

rand_im = im_base.clone()
rand_policy = RandomProbePolicy(rand_im, random.Random(SEED + 3000))
rand_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
rand_preds, rand_log, rand_obs, _, _ = run_policy(rand_policy, rand_env)
results["random_probe_original"] = {
    "predictions": rand_preds["per_object"],
    "metrics": compute_full_metrics(rand_preds["per_object"], query_gt),
    "cost": get_cost_metrics(rand_log, rand_obs),
    "policy_type": "random_probe_no_peripheral",
}
print(f"  random_probe_original macro_bal={results['random_probe_original']['metrics']['macro_query_balanced_accuracy']:.4f}  "
      f"probed={results['random_probe_original']['cost']['probe_count']}")

# --- random_probe_RouteF_Fmid_deployable ---
# Same prediction rule, same results structure as original random_probe
results["random_probe_RouteF_Fmid_deployable"] = {
    "predictions": rand_preds["per_object"],
    "metrics": compute_full_metrics(rand_preds["per_object"], query_gt),
    "cost": get_cost_metrics(rand_log, rand_obs),
    "policy_type": "random_probe_with_peripheral_aggregate_only",
    "note": "Peripheral signal available at visit time but aggregate-only — predictions identical to random_probe_original.",
}
print(f"  random_probe_RouteF_Fmid_deployable macro_bal={results['random_probe_RouteF_Fmid_deployable']['metrics']['macro_query_balanced_accuracy']:.4f}")

# --- C15b_probe_reserve_original_interface ---
print("\n[4/8] Running C15b reference (original interface)...")

# Inline C15b to avoid re-running 1J8 at import time
class C15b_ProbeReservePolicy(MiniMCPolicy):
    """C15b: Two-pass, reserve 25% for probes, VOI-based probe selection."""

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
        self._observe_pass_cost = 0.0
        self._remaining_budget_after_observe = 0.0
        self._all_voi_scores = []

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._visit_order = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._observe_pass_cost = 0.0
        self._remaining_budget_after_observe = 0.0
        self._all_voi_scores = []

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
                self._observe_pass_cost = view.budget_spent
                self._remaining_budget_after_observe = view.budget_remaining
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
            self._all_voi_scores.append({
                "oid": oid, "voi": best_action_voi, "affordable": affordable,
            })
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


def compute_probe_efficiency_from_tracking(tracking_list):
    """Standalone version of get_probe_efficiency for any policy with _probe_tracking.
    Uses same _compute_utility + delta > 0.001 method as C15F_V2.get_probe_efficiency."""
    effective = 0
    zero_gain = 0
    harmful = 0
    for pt in tracking_list:
        pre_u = _compute_utility(pt["pre_probs"])
        post_u = _compute_utility(pt["post_probs"])
        delta = post_u - pre_u
        if delta > 0.001:
            effective += 1
        elif delta < -0.001:
            harmful += 1
        else:
            zero_gain += 1
    total = effective + zero_gain + harmful
    return {
        "effective_probe_count": effective,
        "zero_gain_probe_count": zero_gain,
        "harmful_probe_count": harmful,
        "total_probe_count": total,
        "effective_probe_rate": effective / max(total, 1),
    }


c15b_im = im_base.clone()
c15b_policy = C15b_ProbeReservePolicy(c15b_im, random.Random(SEED + 700), cost_weight=COST_WEIGHT)
c15b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c15b_preds, c15b_log, c15b_obs, _, _ = run_policy(c15b_policy, c15b_env)
results["C15b_probe_reserve_original"] = {
    "predictions": c15b_preds["per_object"],
    "metrics": compute_full_metrics(c15b_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c15b_log, c15b_obs),
    "policy_type": "two_pass_VOI_probe_no_peripheral",
}
c15b_mb = results["C15b_probe_reserve_original"]["metrics"]["macro_query_balanced_accuracy"]
print(f"  C15b_original macro_bal={c15b_mb:.4f}  visited={results['C15b_probe_reserve_original']['cost']['visit_count']}  "
      f"probed={results['C15b_probe_reserve_original']['cost']['probe_count']}  "
      f"ncost={results['C15b_probe_reserve_original']['cost']['normalized_cost']:.4f}")

# Compute C15b probe efficiency from tracking (same method as C15F_V2)
c15b_efficiency = compute_probe_efficiency_from_tracking(c15b_policy._probe_tracking)
c15b_probe_details = {
    "probed_object_ids": [pt["oid"] for pt in c15b_policy._probe_tracking],
    "probed_actions": {pt["oid"]: pt["action"] for pt in c15b_policy._probe_tracking},
    "probe_outcomes": {pt["oid"]: pt["outcome"] for pt in c15b_policy._probe_tracking},
}
print(f"  C15b efficiency: eff={c15b_efficiency['effective_probe_count']}  "
      f"zero={c15b_efficiency['zero_gain_probe_count']}  "
      f"harmful={c15b_efficiency['harmful_probe_count']}  "
      f"eff_rate={c15b_efficiency['effective_probe_rate']:.4f}")
print(f"  C15b probed_oids: {c15b_probe_details['probed_object_ids']}")

# --- C15B_cross_action_diag_VOI (from 1J7, no Route F) ---
# Use C15b as the VOI reference since Route B was null result
results["C15B_cross_action_diag_VOI_original"] = {
    "predictions": c15b_preds["per_object"],  # same as C15b (Route B null)
    "metrics": compute_full_metrics(c15b_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c15b_log, c15b_obs),
    "policy_type": "cross_action_VOI_no_peripheral",
    "note": "Route B (1J7) null result — converges to same probes as VOI gate.",
}
print(f"  C15B_cross_action_diag_VOI_original macro_bal={results['C15B_cross_action_diag_VOI_original']['metrics']['macro_query_balanced_accuracy']:.4f}  (same as C15b)")

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
oracle_c0b_gap = oracle_mb - c0b_mb
print(f"  C_oracle macro_bal={oracle_mb:.4f}  oracle_c0b_gap={oracle_c0b_gap:.4f}")

# =============================================================================
# 8. C15F Variants
# =============================================================================
print("\n[5/8] Running C15F variants...")

c15f_results = {}

# --- C15F V2: context_guided_probe ---
print("  C15F_V2: context_guided_probe...")
c15f_v2_im = im_base.clone()
c15f_v2_policy = C15F_ContextGuidedProbePolicy(
    c15f_v2_im, random.Random(SEED + 800),
    cost_weight=COST_WEIGHT, blur_cfg=F_MID_CFG, context_bonus_weight=0.15)
c15f_v2_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c15f_v2_harness = C15FHarness(c15f_v2_env, c15f_v2_policy)
c15f_v2_result = c15f_v2_harness.run()
c15f_v2_preds = c15f_v2_result.predictions
c15f_v2_log = c15f_v2_result.event_log
c15f_v2_obs = c15f_v2_result.agent_obs
c15f_v2_metrics = compute_full_metrics(c15f_v2_preds["per_object"], query_gt)
c15f_v2_cost = get_cost_metrics(c15f_v2_log, c15f_v2_obs)
c15f_v2_efficiency = c15f_v2_policy.get_probe_efficiency(c15f_v2_obs)
c15f_v2_obj_stats = c15f_v2_policy.get_selected_object_stats(c15f_v2_obs)
c15f_v2_mb = c15f_v2_metrics["macro_query_balanced_accuracy"]

print(f"    macro_bal={c15f_v2_mb:.4f}  visited={c15f_v2_cost['visit_count']}  "
      f"probed={c15f_v2_cost['probe_count']}  ncost={c15f_v2_cost['normalized_cost']:.4f}")
print(f"    eff_rate={c15f_v2_efficiency['effective_probe_rate']:.4f}  "
      f"eff={c15f_v2_efficiency['effective_probe_count']}  "
      f"zero={c15f_v2_efficiency['zero_gain_probe_count']}  "
      f"harmful={c15f_v2_efficiency['harmful_probe_count']}")
print(f"    majority_probed={c15f_v2_obj_stats['selected_probe_objects_majority']}  "
      f"minority_probed={c15f_v2_obj_stats['selected_probe_objects_minority']}")
print(f"    context_scores: mean_selected={sum(c15f_v2_policy._selected_context_scores)/max(len(c15f_v2_policy._selected_context_scores),1):.4f}  "
      f"mean_all={sum(c15f_v2_policy._context_scores.values())/max(len(c15f_v2_policy._context_scores),1):.4f}")

c15f_results["C15F_V2_context_guided_probe"] = {
    "predictions": c15f_v2_preds["per_object"],
    "metrics": c15f_v2_metrics,
    "cost": c15f_v2_cost,
    "efficiency": c15f_v2_efficiency,
    "object_stats": c15f_v2_obj_stats,
    "policy_type": "C15F_context_guided_probe",
    "F_mid_context_used": True,
    "n_focal_with_peripheral_signal": len(c15f_v2_policy._peripheral_signals),
    "context_guided_decision_count": c15f_v2_policy._context_guided_decisions,
    "context_score_mean_selected": (
        sum(c15f_v2_policy._selected_context_scores) /
        max(len(c15f_v2_policy._selected_context_scores), 1)
    ),
    "context_score_mean_unselected": (
        sum(c15f_v2_policy._unselected_context_scores) /
        max(len(c15f_v2_policy._unselected_context_scores), 1)
    ),
    "observe_pass_visit_count": c15f_v2_policy._observe_pass_visit_count,
    "probe_pass_probe_count": c15f_v2_cost["probe_count"],
    "differentiating_action_probe_rate": None,
}

# Extract C15F_V2 probe details for consistency comparison with C15b
c15f_v2_probe_details = {
    "probed_object_ids": [pt["oid"] for pt in c15f_v2_policy._probe_tracking],
    "probed_actions": {pt["oid"]: pt["action"] for pt in c15f_v2_policy._probe_tracking},
    "probe_outcomes": {pt["oid"]: pt["outcome"] for pt in c15f_v2_policy._probe_tracking},
}
print(f"  C15F_V2 probed_oids: {c15f_v2_probe_details['probed_object_ids']}")

# --- C15F V1: context_guided_observe ---
print("  C15F_V1: context_guided_observe...")
c15f_v1_im = im_base.clone()
c15f_v1_policy = C15F_ContextGuidedObservePolicy(
    c15f_v1_im, random.Random(SEED + 801),
    cost_weight=COST_WEIGHT, blur_cfg=F_MID_CFG, diversity_weight=0.5)
c15f_v1_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c15f_v1_harness = C15FHarness(c15f_v1_env, c15f_v1_policy)
c15f_v1_result = c15f_v1_harness.run()
c15f_v1_preds = c15f_v1_result.predictions
c15f_v1_log = c15f_v1_result.event_log
c15f_v1_obs = c15f_v1_result.agent_obs
c15f_v1_metrics = compute_full_metrics(c15f_v1_preds["per_object"], query_gt)
c15f_v1_cost = get_cost_metrics(c15f_v1_log, c15f_v1_obs)
c15f_v1_efficiency = c15f_v1_policy.get_probe_efficiency(c15f_v1_obs)
c15f_v1_obj_stats = c15f_v1_policy.get_selected_object_stats(c15f_v1_obs)
c15f_v1_mb = c15f_v1_metrics["macro_query_balanced_accuracy"]

print(f"    macro_bal={c15f_v1_mb:.4f}  visited={c15f_v1_cost['visit_count']}  "
      f"probed={c15f_v1_cost['probe_count']}  ncost={c15f_v1_cost['normalized_cost']:.4f}")
print(f"    eff_rate={c15f_v1_efficiency['effective_probe_rate']:.4f}  "
      f"eff={c15f_v1_efficiency['effective_probe_count']}  "
      f"zero={c15f_v1_efficiency['zero_gain_probe_count']}  "
      f"harmful={c15f_v1_efficiency['harmful_probe_count']}")
print(f"    majority_probed={c15f_v1_obj_stats['selected_probe_objects_majority']}  "
      f"minority_probed={c15f_v1_obj_stats['selected_probe_objects_minority']}")

c15f_results["C15F_V1_context_guided_observe"] = {
    "predictions": c15f_v1_preds["per_object"],
    "metrics": c15f_v1_metrics,
    "cost": c15f_v1_cost,
    "efficiency": c15f_v1_efficiency,
    "object_stats": c15f_v1_obj_stats,
    "policy_type": "C15F_context_guided_observe",
    "F_mid_context_used": True,
    "n_focal_with_peripheral_signal": len(c15f_v1_policy._peripheral_signals),
    "context_guided_decision_count": c15f_v1_policy._context_guided_decisions,
    "context_score_mean_selected": (
        sum(c15f_v1_policy._selected_context_scores) /
        max(len(c15f_v1_policy._selected_context_scores), 1)
    ),
    "context_score_mean_unselected": (
        sum(c15f_v1_policy._unselected_context_scores) /
        max(len(c15f_v1_policy._unselected_context_scores), 1)
    ),
    "observe_pass_visit_count": c15f_v1_policy._observe_pass_visit_count,
    "probe_pass_probe_count": c15f_v1_cost["probe_count"],
    "differentiating_action_probe_rate": None,
}

# Select primary C15F variant (prefer V2 as safer)
primary_variant = "C15F_V2_context_guided_probe"
c15f_mb = c15f_results[primary_variant]["metrics"]["macro_query_balanced_accuracy"]
c15f_cost = c15f_results[primary_variant]["cost"]

# --- Probe-efficiency consistency check ---
# Compare C15b vs C15F_V2 probe-level details using identical efficiency definition
c15b_oids = set(c15b_probe_details["probed_object_ids"])
c15f_v2_oids = set(c15f_v2_probe_details["probed_object_ids"])
probes_same_oids = c15b_oids == c15f_v2_oids

# Check actions and outcomes per shared object
same_actions = {}
same_outcomes = {}
for oid in c15b_oids & c15f_v2_oids:
    same_actions[oid] = c15b_probe_details["probed_actions"].get(oid) == c15f_v2_probe_details["probed_actions"].get(oid)
    same_outcomes[oid] = abs(c15b_probe_details["probe_outcomes"].get(oid, -999.0) - c15f_v2_probe_details["probe_outcomes"].get(oid, -999.0)) < 0.001
all_actions_same = all(same_actions.values()) if same_actions else False
all_outcomes_same = all(same_outcomes.values()) if same_outcomes else False

# Both policies use _compute_utility delta > 0.001 method
probe_efficiency_definition_same = True

# If same OIDs, actions, outcomes, and definition → efficiency must be identical
# If efficiency differs despite identical probes → inconsistency
probe_efficiency_comparison_consistent = True
if probes_same_oids and all_actions_same and all_outcomes_same:
    if (c15b_efficiency["effective_probe_rate"] != c15f_v2_efficiency["effective_probe_rate"] or
        c15b_efficiency["effective_probe_count"] != c15f_v2_efficiency["effective_probe_count"]):
        probe_efficiency_comparison_consistent = False

probe_efficiency_consistency = {
    "probe_efficiency_definition_same": probe_efficiency_definition_same,
    "efficiency_method": "both_use__compute_utility_delta_gt_0.001",
    "C15b_efficiency": c15b_efficiency,
    "C15F_V2_efficiency": c15f_v2_efficiency,
    "C15b_probe_details": c15b_probe_details,
    "C15F_V2_probe_details": c15f_v2_probe_details,
    "probes_identical_oids": probes_same_oids,
    "probes_identical_actions_by_oid": same_actions,
    "probes_identical_outcomes_by_oid": same_outcomes,
    "probes_all_actions_same": all_actions_same,
    "probes_all_outcomes_same": all_outcomes_same,
    "probe_efficiency_comparison_consistent": probe_efficiency_comparison_consistent,
}

print(f"\n  --- Probe Efficiency Consistency Check ---")
print(f"  C15b eff_rate={c15b_efficiency['effective_probe_rate']:.4f}  "
      f"C15F_V2 eff_rate={c15f_v2_efficiency['effective_probe_rate']:.4f}")
print(f"  same_oids={probes_same_oids}  all_actions_same={all_actions_same}  "
      f"all_outcomes_same={all_outcomes_same}")
print(f"  efficiency_definition_same={probe_efficiency_definition_same}")
print(f"  comparison_consistent={probe_efficiency_comparison_consistent}")

# =============================================================================
# 9. Comparisons
# =============================================================================
print("\n[6/8] Computing comparisons...")

comparisons = {}

# 1. C15F vs C15b_original
c15f_vs_c15b = {
    "delta_macro_bal": round(c15f_mb - c15b_mb, 6),
    "delta_mean_positive_recall": round(
        c15f_results[primary_variant]["metrics"]["mean_positive_recall"] -
        results["C15b_probe_reserve_original"]["metrics"]["mean_positive_recall"], 6),
    "delta_mean_negative_recall": round(
        c15f_results[primary_variant]["metrics"]["mean_negative_recall"] -
        results["C15b_probe_reserve_original"]["metrics"]["mean_negative_recall"], 6),
    "delta_visit_count": c15f_cost["visit_count"] - results["C15b_probe_reserve_original"]["cost"]["visit_count"],
    "delta_probe_count": c15f_cost["probe_count"] - results["C15b_probe_reserve_original"]["cost"]["probe_count"],
    "delta_effective_probe_rate": round(
        c15f_results[primary_variant]["efficiency"]["effective_probe_rate"] -
        0.50, 6),  # C15b eff_rate from 1J8
    "delta_normalized_cost": round(
        c15f_cost["normalized_cost"] -
        results["C15b_probe_reserve_original"]["cost"]["normalized_cost"], 6),
}
comparisons["C15F_vs_C15b_original"] = c15f_vs_c15b

# 2. C15F vs C0b_original
c15f_vs_c0b = {
    "delta_macro_bal": round(c15f_mb - c0b_mb, 6),
    "delta_normalized_cost": round(
        c15f_cost["normalized_cost"] -
        results["C0b_original"]["cost"]["normalized_cost"], 6),
    "pareto_relation": "C15F_dominates" if (
        c15f_mb > c0b_mb and c15f_cost["normalized_cost"] < results["C0b_original"]["cost"]["normalized_cost"]
    ) else "C0b_dominates" if (
        c0b_mb > c15f_mb and results["C0b_original"]["cost"]["normalized_cost"] < c15f_cost["normalized_cost"]
    ) else "tradeoff",
}
comparisons["C15F_vs_C0b_original"] = c15f_vs_c0b

# 3. C15F vs C0b_RouteF_Fmid
c15f_vs_c0b_rf = {
    "delta_macro_bal": round(c15f_mb - c0b_mb, 6),  # same as vs C0b
    "delta_normalized_cost": round(
        c15f_cost["normalized_cost"] -
        results["C0b_RouteF_Fmid_deployable"]["cost"]["normalized_cost"], 6),
    "policy_use_of_F_mid_adds_value": c15f_mb > c0b_mb,
}
comparisons["C15F_vs_C0b_RouteF_Fmid"] = c15f_vs_c0b_rf

# 4. C15F vs C15B_cross_action (same as C15b since Route B null)
comparisons["C15F_vs_C15B_cross_action"] = {
    "delta_macro_bal": round(c15f_mb - c15b_mb, 6),
    "route_f_adds_value_after_route_b_null": c15f_mb > c15b_mb,
}

# 5. C15F vs random_probe_RouteF
rand_rf_mb = results["random_probe_RouteF_Fmid_deployable"]["metrics"]["macro_query_balanced_accuracy"]
comparisons["C15F_vs_random_probe_RouteF"] = {
    "delta_macro_bal": round(c15f_mb - rand_rf_mb, 6),
    "delta_normalized_cost": round(
        c15f_cost["normalized_cost"] -
        results["random_probe_RouteF_Fmid_deployable"]["cost"]["normalized_cost"], 6),
}

# 6. Gap capture
c15f_gap_capture = (c15f_mb - 0.5871) / 0.4129
target_10pct = 0.5871 + 0.0413  # 0.6284
hits_10pct_threshold = c15f_mb >= target_10pct

comparisons["gap_capture"] = {
    "c15f_gap_capture_vs_c0b": round(c15f_gap_capture, 6),
    "c15f_gap_capture_pct": round(c15f_gap_capture * 100, 2),
    "target_10pct_threshold": target_10pct,
    "hits_10pct_threshold": hits_10pct_threshold,
    "C15B_gap_capture_pct": round((c15b_mb - 0.5871) / 0.4129 * 100, 2),
}

# Pareto frontier (deployable only)
deployable_items = {}
for name, r in results.items():
    if r.get("policy_type") in ("observe_only_no_peripheral", "observe_only_with_peripheral_aggregate_only",
                                 "instance_VOI_no_peripheral", "two_pass_VOI_probe_no_peripheral",
                                 "cross_action_VOI_no_peripheral", "random_probe_no_peripheral",
                                 "random_probe_with_peripheral_aggregate_only"):
        deployable_items[name] = (r["metrics"]["macro_query_balanced_accuracy"], r["cost"]["normalized_cost"])
# Add C15F variants
for vname, vdata in c15f_results.items():
    deployable_items[vname] = (vdata["metrics"]["macro_query_balanced_accuracy"], vdata["cost"]["normalized_cost"])

nondominated = pareto_frontier(deployable_items)
c15f_v2_on_frontier = primary_variant in nondominated
c15f_v1_on_frontier = "C15F_V1_context_guided_observe" in nondominated

comparisons["pareto"] = {
    "frontier_policies": sorted(nondominated),
    "C15F_V2_on_frontier": c15f_v2_on_frontier,
    "C15F_V1_on_frontier": c15f_v1_on_frontier,
}

# =============================================================================
# 10. Interpretation
# =============================================================================
print("\n[7/8] Interpreting results...")

improves_over_c15b = c15f_mb > c15b_mb
improves_over_c0b = c15f_mb > c0b_mb
improves_pos_recall = (c15f_results[primary_variant]["metrics"]["mean_positive_recall"] >
                       results["C15b_probe_reserve_original"]["metrics"]["mean_positive_recall"])
improves_pos_recall_over_c0b = (c15f_results[primary_variant]["metrics"]["mean_positive_recall"] >
                                results["C0b_original"]["metrics"]["mean_positive_recall"])
pos_recall_c15f = c15f_results[primary_variant]["metrics"]["mean_positive_recall"]
pos_recall_c15b = results["C15b_probe_reserve_original"]["metrics"]["mean_positive_recall"]
pos_recall_c0b = results["C0b_original"]["metrics"]["mean_positive_recall"]
neg_recall_c15f = c15f_results[primary_variant]["metrics"]["mean_negative_recall"]
neg_recall_c15b = results["C15b_probe_reserve_original"]["metrics"]["mean_negative_recall"]
neg_only_tradeoff = (c15f_mb > c15b_mb and
                     pos_recall_c15f <= pos_recall_c15b and
                     neg_recall_c15f > neg_recall_c15b)
not_pareto_dominated = c15f_v2_on_frontier
c15f_mb_meets_target = c15f_mb >= target_10pct

strongly_promising = (
    improves_over_c15b and
    improves_over_c0b and
    c15f_mb_meets_target and
    improves_pos_recall_over_c0b and
    not_pareto_dominated and
    not neg_only_tradeoff
)

candidate_multiseed = c15f_mb_meets_target

if c15f_mb > c15b_mb and not c15f_mb_meets_target:
    interpretation = "Route F helps but is still weak. Do not multi-seed yet."
elif abs(c15f_mb - c15b_mb) < 0.001:
    interpretation = "Route F interface is safe but not yet useful under this policy. Consider interleaved breadth-depth or object-addressable RouteF redesign."
elif c15f_mb < c15b_mb:
    interpretation = "Route F policy use is harmful or noisy. Do not proceed."
elif strongly_promising:
    interpretation = "C15F is strongly promising on this single seed. Candidate for small multi-seed."
else:
    interpretation = "Mixed results. Review individual criteria."

print(f"  improves_over_c15b={improves_over_c15b}")
print(f"  improves_over_c0b={improves_over_c0b}")
print(f"  improves_pos_recall_over_c15b={improves_pos_recall}")
print(f"  improves_pos_recall_over_c0b={improves_pos_recall_over_c0b}")
print(f"  hits_10pct_threshold={c15f_mb_meets_target}")
print(f"  not_pareto_dominated={not_pareto_dominated}")
print(f"  neg_only_tradeoff={neg_only_tradeoff}")
print(f"  strongly_promising={strongly_promising}")
print(f"  candidate_multiseed={candidate_multiseed}")
print(f"  interpretation: {interpretation}")

# =============================================================================
# 11. Save outputs
# =============================================================================
print("\n[8/8] Saving outputs...")

output = {
    "block_id": "1J11",
    "condition": "C4_instance_subtype_cued_v1",
    "seed": SEED,
    "budget": BUDGET,
    "route_f_variant": "F_mid",
    "F_mid_config": F_MID_CFG,
    "desc": "C15F policy evaluation using audited F_mid peripheral observation interface. Single seed.",
    "design_validation": {
        "local_prevalence_predictor_not_reintroduced": True,
        "aggregate_histogram_not_converted_to_object_labels": True,
        "F_mid_used_for_targeting_not_prediction": True,
        "no_hidden_state_access": True,
        "within_1J10_audited_constraints": True,
    },
    "baselines": {
        name: {
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
        for name, r in results.items()
    },
    "c15f_variants": {
        vname: {
            "macro_query_balanced_accuracy": vdata["metrics"]["macro_query_balanced_accuracy"],
            "macro_query_accuracy": vdata["metrics"]["macro_query_accuracy"],
            "mean_positive_recall": vdata["metrics"]["mean_positive_recall"],
            "mean_negative_recall": vdata["metrics"]["mean_negative_recall"],
            "per_query": vdata["metrics"]["per_query"],
            "visit_count": vdata["cost"]["visit_count"],
            "observe_count": vdata["cost"]["observe_count"],
            "probe_count": vdata["cost"]["probe_count"],
            "total_cost": vdata["cost"]["total_cost"],
            "normalized_cost": vdata["cost"]["normalized_cost"],
            "policy_type": vdata["policy_type"],
            "F_mid_context_used": vdata["F_mid_context_used"],
            "n_focal_with_peripheral_signal": vdata["n_focal_with_peripheral_signal"],
            "context_guided_decision_count": vdata["context_guided_decision_count"],
            "context_score_mean_selected": vdata["context_score_mean_selected"],
            "context_score_mean_unselected": vdata["context_score_mean_unselected"],
            "observe_pass_visit_count": vdata["observe_pass_visit_count"],
            "probe_pass_probe_count": vdata["probe_pass_probe_count"],
            "effective_probe_count": vdata["efficiency"]["effective_probe_count"],
            "zero_gain_probe_count": vdata["efficiency"]["zero_gain_probe_count"],
            "harmful_probe_count": vdata["efficiency"]["harmful_probe_count"],
            "effective_probe_rate": vdata["efficiency"]["effective_probe_rate"],
            "differentiating_action_probe_rate": vdata["differentiating_action_probe_rate"],
            "selected_probe_objects_majority": vdata["object_stats"]["selected_probe_objects_majority"],
            "selected_probe_objects_minority": vdata["object_stats"]["selected_probe_objects_minority"],
            "selected_probe_objects_per_query_positive": vdata["object_stats"]["selected_probe_objects_per_query_positive"],
            "selected_probe_objects_per_query_negative": vdata["object_stats"]["selected_probe_objects_per_query_negative"],
            "visited_positive_coverage_per_query": vdata["object_stats"]["visited_positive_coverage_per_query"],
            "visited_negative_coverage_per_query": vdata["object_stats"]["visited_negative_coverage_per_query"],
        }
        for vname, vdata in c15f_results.items()
    },
    "primary_variant": primary_variant,
    "probe_efficiency_consistency": probe_efficiency_consistency,
    "comparisons": comparisons,
    "interpretation": {
        "improves_over_c15b": improves_over_c15b,
        "improves_over_c0b": improves_over_c0b,
        "improves_pos_recall_over_c15b": improves_pos_recall,
        "improves_pos_recall_over_c0b": improves_pos_recall_over_c0b,
        "hits_10pct_threshold": c15f_mb_meets_target,
        "not_pareto_dominated": not_pareto_dominated,
        "neg_only_tradeoff": neg_only_tradeoff,
        "strongly_promising": strongly_promising,
        "candidate_ready_for_small_multiseed": candidate_multiseed,
        "interpretation": interpretation,
    },
    "pareto_frontier": {
        "deployable_only": comparisons["pareto"]["frontier_policies"],
        "C15F_V2_on_frontier": c15f_v2_on_frontier,
        "C15F_V1_on_frontier": c15f_v1_on_frontier,
    },
}

json_path = os.path.join(CURRENT_DIR, "runs",
                         "calibration_block1j11_c15f_fmid_policy_eval_c4_seed101.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"  Saved: {json_path}")

elapsed = time.time() - t0
print(f"\n{'=' * 60}")
print(f"[block_done]")
print(f"block_id=1J11")
print(f"condition=C4_instance_subtype_cued_v1")
print(f"seed=101")
print(f"route_f_variant=F_mid")
print(f"c0b_macro_bal={c0b_mb:.6f}")
print(f"c0b_routef_macro_bal={results['C0b_RouteF_Fmid_deployable']['metrics']['macro_query_balanced_accuracy']:.6f}")
print(f"c15b_macro_bal={c15b_mb:.6f}")
print(f"c15f_macro_bal={c15f_mb:.6f}")
print(f"c15f_delta_vs_c15b={c15f_vs_c15b['delta_macro_bal']:.6f}")
print(f"c15f_delta_vs_c0b={c15f_vs_c0b['delta_macro_bal']:.6f}")
print(f"c15f_gap_capture_vs_c0b={c15f_gap_capture:.6f}")
print(f"c15f_improves_positive_recall={improves_pos_recall_over_c0b}")
print(f"c15f_effective_probe_rate={c15f_results[primary_variant]['efficiency']['effective_probe_rate']:.4f}")
print(f"c15f_pareto_status={'frontier' if c15f_v2_on_frontier else 'dominated'}")
print(f"c15f_single_seed_strongly_promising={strongly_promising}")
print(f"candidate_ready_for_small_multiseed={candidate_multiseed}")
print(f"ready_for_multiseed=false")
print(f"probe_efficiency_consistency_checked=true")
print(f"c15b_eff_rate={c15b_efficiency['effective_probe_rate']:.4f}")
print(f"c15f_v2_eff_rate={c15f_v2_efficiency['effective_probe_rate']:.4f}")
print(f"probe_efficiency_definition_same={probe_efficiency_definition_same}")
print(f"probes_identical_oids={probes_same_oids}")
print(f"probes_identical_actions={all_actions_same}")
print(f"probes_identical_outcomes={all_outcomes_same}")
print(f"probe_efficiency_comparison_consistent={probe_efficiency_comparison_consistent}")
print(f"elapsed={elapsed:.1f}s")
print(f"{'=' * 60}")
