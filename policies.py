"""Mini-MC v0 policies.

All policies receive ONLY AgentObsView. No simulator access.
C14b (budgeted oracle) receives a reference to MiniMCSimulatorTruth.
C14a (truth answer oracle) bypasses the episode harness entirely.
"""

import random
import math
from abc import ABC, abstractmethod

from agent_obs import AgentObsView
import config

# Reuse constants from InstanceOutcomeMemory
MAIN_CANDIDATE_ACTIONS = [
    "mine_by_hand", "mine_with_pickaxe", "craft_plank",
    "eat", "use_as_tool", "burn_as_fuel",
]
ACTION_TO_FEATURE = {
    "mine_by_hand": "mine_by_hand_success",
    "mine_with_pickaxe": "mine_with_pickaxe_success",
    "craft_plank": "craft_plank_success",
    "eat": "eat_success",
    "use_as_tool": "use_as_tool_success",
    "burn_as_fuel": "burn_as_fuel_success",
}
CORE_ACTION_FEATURES = list(ACTION_TO_FEATURE.values())


def _compute_utility(probs):
    """Utility = mean over 6 core features of max(p, 1-p)."""
    return sum(max(probs.get(f, 0.5), 1.0 - probs.get(f, 0.5))
               for f in CORE_ACTION_FEATURES) / len(CORE_ACTION_FEATURES)


def _compute_entropy(probs):
    """Mean binary entropy over features."""
    total = 0.0
    for f in CORE_ACTION_FEATURES:
        p = max(1e-9, min(1.0 - 1e-9, probs.get(f, 0.5)))
        total += -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    return total / len(CORE_ACTION_FEATURES)


# ----- Base class -----
class MiniMCPolicy(ABC):
    @abstractmethod
    def reset(self, view):
        pass

    @abstractmethod
    def select_next_object(self, view):
        pass

    @abstractmethod
    def decide_probe(self, view, object_id):
        pass

    @abstractmethod
    def on_probe_result(self, view, object_id, action, outcome):
        pass

    @abstractmethod
    def get_answer(self, view):
        pass

    def get_pre_probe_entropies(self):
        """Return {oid: pre-probe entropy} for entropy gap analysis."""
        return {}

    def get_pre_decision_entropies(self):
        """Return {oid: pre-decision entropy} for objects considered but skipped."""
        return {}


# =========================================================================
# C0a — No interaction baseline
# =========================================================================
class C0a_NoInteractionPolicy(MiniMCPolicy):
    """Never observes, never probes, never moves. Uses global base rates only."""

    def __init__(self, instance_memory, rng):
        self._im = instance_memory
        self._rng = rng

    def reset(self, view):
        pass

    def select_next_object(self, view):
        return None

    def decide_probe(self, view, object_id):
        return False, None

    def on_probe_result(self, view, object_id, action, outcome):
        pass

    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            fake_obj = {"id": oid, "visible_features": {}}
            per_object[oid] = self._im.predict_all_affordances(fake_obj)
        return {"per_object": per_object}


# =========================================================================
# C0b — Observe only, no probe
# =========================================================================
class C0b_ObserveOnlyPolicy(MiniMCPolicy):
    """Visits objects (nearest first), observes, never probes."""

    def __init__(self, instance_memory, rng):
        self._im = instance_memory
        self._rng = rng
        self._pre_decision_entropies = {}

    def reset(self, view):
        self._pre_decision_entropies = {}

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited:
            return None
        best_oid = None
        best_cost = float('inf')
        for oid in unvisited:
            reach_cost = view.compute_reach_cost(oid)
            total = reach_cost + view.observe_cost
            if view.can_afford(total) and total < best_cost:
                best_cost = total
                best_oid = oid
        # Record pre-decision entropies for objects we're skipping
        # (C0b visits until budget exhausted, so all unvisited are "considered-skipped")
        for oid in unvisited:
            if oid != best_oid:
                fake_obj = {"id": oid, "visible_features": {}}
                probs = self._im.predict_all_affordances(fake_obj)
                self._pre_decision_entropies[oid] = _compute_entropy(probs)
        return best_oid

    def decide_probe(self, view, object_id):
        return False, None

    def on_probe_result(self, view, object_id, action, outcome):
        pass

    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            fake_obj = {"id": oid, "visible_features": features}
            per_object[oid] = self._im.predict_all_affordances(fake_obj)
        return {"per_object": per_object}

    def get_pre_decision_entropies(self):
        return dict(self._pre_decision_entropies)


# =========================================================================
# C2 — Cluster reference (cost gate)
# =========================================================================
class C2_ClusterRefPolicy(MiniMCPolicy):
    """Visits by cluster-EIG score, probes via CategoryPosteriorLearner gate.
    select_next_object requires net score > 0 (EIG - cost > 0)."""

    def __init__(self, cluster_posterior, rng):
        self._cp = cluster_posterior
        self._rng = rng
        self._pre_probe_entropies = {}
        self._pre_decision_entropies = {}

    def _global_eig(self, object_id, view):
        try:
            fake_obj = {"id": object_id, "visible_features": {}}
            posterior = self._cp.compute_posterior(fake_obj)
            return self._cp.compute_max_eig(posterior.get("label_probs", {}))
        except Exception:
            return 0.0

    def reset(self, view):
        self._pre_probe_entropies = {}
        self._pre_decision_entropies = {}

    def _object_score(self, oid, view):
        total_cost = (view.compute_reach_cost(oid) + view.observe_cost
                      + view.probe_cost)
        return self._global_eig(oid, view) - total_cost

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        best_oid = None
        best_score = float('-inf')
        for oid in unvisited:
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost):
                continue
            score = self._object_score(oid, view)
            if score > best_score:
                best_score = score
                best_oid = oid
        # Record pre-decision entropies for skipped objects
        if best_score <= 0:
            # All remaining unvisited are "considered and skipped"
            for oid in unvisited:
                if view.can_afford(view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost):
                    fake_obj = {"id": oid, "visible_features": {}}
                    try:
                        posterior = self._cp.compute_posterior(fake_obj)
                        probs = self._cp.predict_affordances(posterior.get("label_probs", {}))
                    except Exception:
                        probs = {f: 0.5 for f in CORE_ACTION_FEATURES}
                    self._pre_decision_entropies[oid] = _compute_entropy(probs)
            return None
        return best_oid

    def decide_probe(self, view, object_id):
        features = view.get_observed_features(object_id) or {}
        fake_obj = {"id": object_id, "visible_features": features}
        try:
            posterior = self._cp.compute_posterior(fake_obj)
            probs = self._cp.predict_affordances(posterior.get("label_probs", {}))
            self._pre_probe_entropies[object_id] = _compute_entropy(probs)

            selection = self._cp.select_best_probe(
                posterior.get("label_probs", {}), MAIN_CANDIDATE_ACTIONS
            )
            if selection is None or selection.get("best_action") is None:
                return False, None
            action = selection["best_action"]
            if action == "tap_sound":
                return False, None
            net_benefit = selection.get("net_benefit", 0.0)
            return net_benefit > 0, action
        except Exception:
            return False, None

    def on_probe_result(self, view, object_id, action, outcome):
        try:
            self._cp.incorporate_probe(object_id, action, outcome)
        except Exception:
            pass

    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            fake_obj = {"id": oid, "visible_features": features}
            try:
                posterior = self._cp.compute_posterior(fake_obj)
                probs = self._cp.predict_affordances(posterior.get("label_probs", {}))
                per_object[oid] = {f: probs.get(f, 0.5) for f in CORE_ACTION_FEATURES}
            except Exception:
                per_object[oid] = {f: 0.5 for f in CORE_ACTION_FEATURES}
        return {"per_object": per_object}

    def get_pre_probe_entropies(self):
        return dict(self._pre_probe_entropies)

    def get_pre_decision_entropies(self):
        return dict(self._pre_decision_entropies)


# =========================================================================
# C8 — Forced EIG (instance-based, cost gate)
# =========================================================================
class C8_ForcedEIGPolicy(MiniMCPolicy):
    """Visits by instance-EIG score, always probes best-EIG action.
    select_next_object requires net score > 0 (EIG - cost > 0)."""

    def __init__(self, instance_memory, rng):
        self._im = instance_memory
        self._rng = rng
        self._pre_probe_entropies = {}
        self._pre_decision_entropies = {}

    def _global_eig(self, oid, view):
        try:
            total_eig = 0.0
            for action in MAIN_CANDIDATE_ACTIONS:
                fake_obj = {"id": oid, "visible_features": {}}
                eig, _, _, _ = self._im.compute_action_outcome_eig(fake_obj, action)
                total_eig += eig
            return total_eig / len(MAIN_CANDIDATE_ACTIONS)
        except Exception:
            return 0.0

    def reset(self, view):
        self._pre_probe_entropies = {}
        self._pre_decision_entropies = {}

    def _object_score(self, oid, view):
        total_cost = (view.compute_reach_cost(oid) + view.observe_cost
                      + view.probe_cost)
        return self._global_eig(oid, view) - total_cost

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        best_oid = None
        best_score = float('-inf')
        for oid in unvisited:
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost):
                continue
            score = self._object_score(oid, view)
            if score > best_score:
                best_score = score
                best_oid = oid
        if best_score <= 0:
            for oid in unvisited:
                if view.can_afford(view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost):
                    fake_obj = {"id": oid, "visible_features": {}}
                    probs = self._im.predict_all_affordances(fake_obj)
                    self._pre_decision_entropies[oid] = _compute_entropy(probs)
            return None
        return best_oid

    def decide_probe(self, view, object_id):
        features = view.get_observed_features(object_id) or {}
        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)

        selection = self._im.select_best_probe(fake_obj, MAIN_CANDIDATE_ACTIONS)
        action = selection["best_action"]
        assert action != "tap_sound", "C8 selected tap_sound"
        return True, action

    def on_probe_result(self, view, object_id, action, outcome):
        self._im.incorporate_probe(object_id, action, outcome)

    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            fake_obj = {"id": oid, "visible_features": features}
            per_object[oid] = self._im.predict_all_affordances(fake_obj)
        return {"per_object": per_object}

    def get_pre_probe_entropies(self):
        return dict(self._pre_probe_entropies)

    def get_pre_decision_entropies(self):
        return dict(self._pre_decision_entropies)


# =========================================================================
# C13 — Instance VOI with normalized cost_weight
# =========================================================================
class C13_InstanceVOIPolicy(MiniMCPolicy):
    """VOI gate with normalized cost.

    net_voi = E[post_utility] - current_utility
              - cost_weight * normalized_step_cost

    where normalized_step_cost = (reach + observe + probe) / initial_budget
    """

    def __init__(self, instance_memory, rng, cost_weight=1.0):
        self._im = instance_memory
        self._rng = rng
        self._cost_weight = cost_weight
        self._pre_probe_entropies = {}
        self._pre_decision_entropies = {}

    def _normalized_step_cost(self, oid, view):
        raw = (view.compute_reach_cost(oid) + view.observe_cost
               + view.probe_cost)
        return raw / max(view.initial_budget, 0.001)

    def _normalized_probe_cost(self, view):
        return view.probe_cost / max(view.initial_budget, 0.001)

    def _global_voi(self, oid, view):
        """Prior/global VOI — expected utility gain from global outcome distribution.
        Uses normalized cost."""
        try:
            fake_obj = {"id": oid, "visible_features": {}}
            probs = self._im.predict_all_affordances(fake_obj)
            current_utility = _compute_utility(probs)
            norm_cost = self._normalized_step_cost(oid, view)

            best_net_voi = 0.0
            for action in MAIN_CANDIDATE_ACTIONS:
                p_success = probs.get(ACTION_TO_FEATURE[action], 0.5)

                im_succ = self._im.clone()
                im_succ.incorporate_probe(oid, action, 1.0)
                probs_succ = im_succ.predict_all_affordances(fake_obj)
                utility_succ = _compute_utility(probs_succ)

                im_fail = self._im.clone()
                im_fail.incorporate_probe(oid, action, 0.0)
                probs_fail = im_fail.predict_all_affordances(fake_obj)
                utility_fail = _compute_utility(probs_fail)

                e_post = p_success * utility_succ + (1 - p_success) * utility_fail
                net_voi = e_post - current_utility - self._cost_weight * norm_cost
                if net_voi > best_net_voi:
                    best_net_voi = net_voi
            return best_net_voi
        except Exception:
            return 0.0

    def reset(self, view):
        self._pre_probe_entropies = {}
        self._pre_decision_entropies = {}

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        best_oid = None
        best_score = float('-inf')
        for oid in unvisited:
            total_cost = (view.compute_reach_cost(oid) + view.observe_cost
                          + view.probe_cost)
            if not view.can_afford(total_cost):
                continue
            score = self._global_voi(oid, view)
            if score > best_score:
                best_score = score
                best_oid = oid

        # Record pre-decision entropies for considered-but-skipped objects
        if best_score <= 0:
            for oid in unvisited:
                if view.can_afford(view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost):
                    fake_obj = {"id": oid, "visible_features": {}}
                    probs = self._im.predict_all_affordances(fake_obj)
                    self._pre_decision_entropies[oid] = _compute_entropy(probs)
            return None
        return best_oid

    def decide_probe(self, view, object_id):
        features = view.get_observed_features(object_id)
        if features is None:
            return False, None
        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        current_utility = _compute_utility(probs)

        # Record pre-probe entropy
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)

        norm_probe_cost = self._normalized_probe_cost(view)

        best_action = None
        best_net_voi = 0.0
        for action in MAIN_CANDIDATE_ACTIONS:
            assert action != "tap_sound"
            p_success = probs.get(ACTION_TO_FEATURE[action], 0.5)

            im_succ = self._im.clone()
            im_succ.incorporate_probe(object_id, action, 1.0)
            probs_succ = im_succ.predict_all_affordances(fake_obj)
            utility_succ = _compute_utility(probs_succ)

            im_fail = self._im.clone()
            im_fail.incorporate_probe(object_id, action, 0.0)
            probs_fail = im_fail.predict_all_affordances(fake_obj)
            utility_fail = _compute_utility(probs_fail)

            e_post = p_success * utility_succ + (1 - p_success) * utility_fail
            net_voi = e_post - current_utility - self._cost_weight * norm_probe_cost
            if net_voi > best_net_voi:
                best_net_voi = net_voi
                best_action = action

        return best_net_voi > 0, best_action

    def on_probe_result(self, view, object_id, action, outcome):
        self._im.incorporate_probe(object_id, action, outcome)

    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            fake_obj = {"id": oid, "visible_features": features}
            per_object[oid] = self._im.predict_all_affordances(fake_obj)
        return {"per_object": per_object}

    def get_pre_probe_entropies(self):
        return dict(self._pre_probe_entropies)

    def get_pre_decision_entropies(self):
        return dict(self._pre_decision_entropies)


# =========================================================================
# C14b — Budgeted oracle probe
# =========================================================================
class C14b_BudgetedOraclePolicy(MiniMCPolicy):
    """Budgeted oracle: uses hidden truth to select best probe action,
    but respects the budget constraint. Runs in separate harness."""

    def __init__(self, instance_memory, rng, simulator):
        self._im = instance_memory
        self._rng = rng
        self._simulator = simulator
        self._pre_probe_entropies = {}
        self._pre_decision_entropies = {}

    def reset(self, view):
        self._pre_probe_entropies = {}
        self._pre_decision_entropies = {}

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited:
            # Record pre-decision entropies for remaining
            for oid in unvisited:
                fake_obj = {"id": oid, "visible_features": {}}
                probs = self._im.predict_all_affordances(fake_obj)
                self._pre_decision_entropies[oid] = _compute_entropy(probs)
            return None
        # Visit nearest affordable
        best_oid = None
        best_cost = float('inf')
        for oid in unvisited:
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if view.can_afford(total_cost) and total_cost < best_cost:
                best_cost = total_cost
                best_oid = oid
        return best_oid

    def decide_probe(self, view, object_id):
        features = view.get_observed_features(object_id)
        if features is None:
            return False, None
        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)

        best_action = None
        best_post_acc = -1.0
        for action in MAIN_CANDIDATE_ACTIONS:
            true_outcome = self._simulator.probe(object_id, action)[0]
            im_clone = self._im.clone()
            im_clone.incorporate_probe(object_id, action, true_outcome)
            post_probs = im_clone.predict_all_affordances(fake_obj)
            gt = self._simulator.get_ground_truth_affordances(object_id)
            acc = sum(
                1.0 if (post_probs.get(f, 0.5) >= 0.5) == (gt.get(f, 0.0) >= 0.5)
                else 0.0
                for f in CORE_ACTION_FEATURES
            ) / len(CORE_ACTION_FEATURES)
            if acc > best_post_acc:
                best_post_acc = acc
                best_action = action

        return True, best_action

    def on_probe_result(self, view, object_id, action, outcome):
        self._im.incorporate_probe(object_id, action, outcome)

    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            fake_obj = {"id": oid, "visible_features": features}
            per_object[oid] = self._im.predict_all_affordances(fake_obj)
        return {"per_object": per_object}

    def get_pre_probe_entropies(self):
        return dict(self._pre_probe_entropies)

    def get_pre_decision_entropies(self):
        return dict(self._pre_decision_entropies)


# =========================================================================
# C14a — Truth answer oracle (no budget, no harness)
# =========================================================================
class C14a_TruthAnswerOracle:
    """Truth answer oracle. Returns ground truth directly.
    Does NOT use the harness — used only to verify evaluator ceiling.
    Should score ~1.0 on composite_task_accuracy."""

    def __init__(self, simulator):
        self._simulator = simulator

    def get_answer(self):
        """Return ground truth affordances as predictions."""
        per_object = {}
        for oid in self._simulator._objects:
            gt = self._simulator.get_ground_truth_affordances(oid)
            # Convert to probabilities: 1.0 for success, 0.0 for fail
            per_object[oid] = {f: gt.get(f, 0.0) for f in CORE_ACTION_FEATURES}
        return {"per_object": per_object}

    def get_pre_probe_entropies(self):
        return {}

    def get_pre_decision_entropies(self):
        return {}


# =========================================================================
# Diagnostic: C2 forced-budget
# =========================================================================
class C2_ForcedBudgetPolicy(C2_ClusterRefPolicy):
    """Diagnostic: visits highest-EIG affordable object without net-score gate.
    Probe decision still uses EIG gate."""

    def select_next_object(self, view):
        best_oid = None
        best_score = float('-inf')
        for oid in view.get_unvisited_objects():
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost):
                continue
            score = self._global_eig(oid, view)
            if score > best_score:
                best_score = score
                best_oid = oid
        return best_oid


# =========================================================================
# Diagnostic: C8 forced-budget
# =========================================================================
class C8_ForcedBudgetPolicy(C8_ForcedEIGPolicy):
    """Diagnostic: visits highest-EIG affordable object without net-score gate.
    Always probes best-EIG action."""

    def select_next_object(self, view):
        best_oid = None
        best_score = float('-inf')
        for oid in view.get_unvisited_objects():
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost):
                continue
            score = self._global_eig(oid, view)
            if score > best_score:
                best_score = score
                best_oid = oid
        return best_oid


# =========================================================================
# Diagnostic: C13 forced-visit probe-gate
# =========================================================================
class C13_ForcedVisitProbeGatePolicy(C13_InstanceVOIPolicy):
    """Diagnostic: visit by VOI ranking (no net-VOI gate on visit),
    but still gate probe decisions with net_voi > 0."""

    def select_next_object(self, view):
        best_oid = None
        best_score = float('-inf')
        for oid in view.get_unvisited_objects():
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost):
                continue
            score = self._global_voi(oid, view)
            if score > best_score:
                best_score = score
                best_oid = oid
        # Record pre-decision entropies if nothing found
        if best_oid is None:
            unvisited = view.get_unvisited_objects()
            for oid in unvisited:
                if view.can_afford(view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost):
                    fake_obj = {"id": oid, "visible_features": {}}
                    probs = self._im.predict_all_affordances(fake_obj)
                    self._pre_decision_entropies[oid] = _compute_entropy(probs)
        return best_oid
