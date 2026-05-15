"""C2 cluster-based policy using CategoryPosteriorLearner."""

from ..deps import CategoryPosteriorLearner
from .base import MiniMCPolicy, MAIN_CANDIDATE_ACTIONS, CORE_ACTION_FEATURES, compute_entropy


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
            for oid in unvisited:
                if view.can_afford(view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost):
                    fake_obj = {"id": oid, "visible_features": {}}
                    try:
                        posterior = self._cp.compute_posterior(fake_obj)
                        probs = self._cp.predict_affordances(posterior.get("label_probs", {}))
                    except Exception:
                        probs = {f: 0.5 for f in CORE_ACTION_FEATURES}
                    self._pre_decision_entropies[oid] = compute_entropy(probs)
            return None
        return best_oid

    def decide_probe(self, view, object_id):
        features = view.get_observed_features(object_id) or {}
        fake_obj = {"id": object_id, "visible_features": features}
        try:
            posterior = self._cp.compute_posterior(fake_obj)
            probs = self._cp.predict_affordances(posterior.get("label_probs", {}))
            self._pre_probe_entropies[object_id] = compute_entropy(probs)

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
