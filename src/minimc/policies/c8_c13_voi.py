"""C8 (forced EIG) and C13 (instance VOI) policies."""

from .base import (
    MiniMCPolicy, MAIN_CANDIDATE_ACTIONS, CORE_ACTION_FEATURES,
    ACTION_TO_FEATURE, compute_utility, compute_entropy,
)


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
                    self._pre_decision_entropies[oid] = compute_entropy(probs)
            return None
        return best_oid

    def decide_probe(self, view, object_id):
        features = view.get_observed_features(object_id) or {}
        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        self._pre_probe_entropies[object_id] = compute_entropy(probs)

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
        """Prior/global VOI — expected utility gain from global outcome distribution."""
        try:
            fake_obj = {"id": oid, "visible_features": {}}
            probs = self._im.predict_all_affordances(fake_obj)
            current_utility = compute_utility(probs)
            norm_cost = self._normalized_step_cost(oid, view)

            best_net_voi = 0.0
            for action in MAIN_CANDIDATE_ACTIONS:
                p_success = probs.get(ACTION_TO_FEATURE[action], 0.5)

                im_succ = self._im.clone()
                im_succ.incorporate_probe(oid, action, 1.0)
                probs_succ = im_succ.predict_all_affordances(fake_obj)
                utility_succ = compute_utility(probs_succ)

                im_fail = self._im.clone()
                im_fail.incorporate_probe(oid, action, 0.0)
                probs_fail = im_fail.predict_all_affordances(fake_obj)
                utility_fail = compute_utility(probs_fail)

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

        if best_score <= 0:
            for oid in unvisited:
                if view.can_afford(view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost):
                    fake_obj = {"id": oid, "visible_features": {}}
                    probs = self._im.predict_all_affordances(fake_obj)
                    self._pre_decision_entropies[oid] = compute_entropy(probs)
            return None
        return best_oid

    def decide_probe(self, view, object_id):
        features = view.get_observed_features(object_id)
        if features is None:
            return False, None
        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        current_utility = compute_utility(probs)

        self._pre_probe_entropies[object_id] = compute_entropy(probs)

        norm_probe_cost = self._normalized_probe_cost(view)

        best_action = None
        best_net_voi = 0.0
        for action in MAIN_CANDIDATE_ACTIONS:
            assert action != "tap_sound"
            p_success = probs.get(ACTION_TO_FEATURE[action], 0.5)

            im_succ = self._im.clone()
            im_succ.incorporate_probe(object_id, action, 1.0)
            probs_succ = im_succ.predict_all_affordances(fake_obj)
            utility_succ = compute_utility(probs_succ)

            im_fail = self._im.clone()
            im_fail.incorporate_probe(object_id, action, 0.0)
            probs_fail = im_fail.predict_all_affordances(fake_obj)
            utility_fail = compute_utility(probs_fail)

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
        if best_oid is None:
            unvisited = view.get_unvisited_objects()
            for oid in unvisited:
                if view.can_afford(view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost):
                    fake_obj = {"id": oid, "visible_features": {}}
                    probs = self._im.predict_all_affordances(fake_obj)
                    self._pre_decision_entropies[oid] = compute_entropy(probs)
        return best_oid
