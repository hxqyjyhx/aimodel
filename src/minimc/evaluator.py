"""MiniMCEvaluator: post-episode metrics from event log and predictions."""

import math

from .environment import _TASK_QUERIES
from .policies import CORE_ACTION_FEATURES
from . import config


class MiniMCEvaluator:
    def __init__(self, environment):
        self._env = environment

    def evaluate(self, predictions, event_log, agent_obs,
                 pre_probe_entropies=None, pre_decision_entropies=None):
        """Evaluate episode. Accepts policy-tracked pre-probe entropies for
        entropy gap and pre-decision entropies for skipped objects."""
        per_object = predictions.get("per_object", {})
        n_objects = len(per_object)

        if pre_probe_entropies is None:
            pre_probe_entropies = {}
        if pre_decision_entropies is None:
            pre_decision_entropies = {}

        # ---- Per-feature ground truth comparison ----
        aff_correct = 0
        aff_total = 0
        per_feature_correct = {f: 0 for f in CORE_ACTION_FEATURES}
        per_feature_total = {f: 0 for f in CORE_ACTION_FEATURES}
        # For balanced accuracy: per-feature success/fail tallies
        per_feature_success_correct = {f: 0 for f in CORE_ACTION_FEATURES}
        per_feature_success_total = {f: 0 for f in CORE_ACTION_FEATURES}
        per_feature_fail_correct = {f: 0 for f in CORE_ACTION_FEATURES}
        per_feature_fail_total = {f: 0 for f in CORE_ACTION_FEATURES}
        per_category_correct = {}
        per_category_total = {}

        for oid, probs in per_object.items():
            gt = self._env.get_ground_truth_affordances(oid)
            cat = self._env.get_hidden_category(oid)
            per_category_correct.setdefault(cat, 0)
            per_category_total.setdefault(cat, 0)

            for f in CORE_ACTION_FEATURES:
                pred_val = 1.0 if probs.get(f, 0.5) >= 0.5 else 0.0
                true_val = gt.get(f, 0.0)
                if pred_val == true_val:
                    aff_correct += 1
                    per_feature_correct[f] += 1
                    per_category_correct[cat] += 1
                aff_total += 1
                per_feature_total[f] += 1
                per_category_total[cat] += 1

                # Success/fail breakdown
                if true_val >= 0.5:
                    per_feature_success_total[f] += 1
                    if pred_val >= 0.5:
                        per_feature_success_correct[f] += 1
                else:
                    per_feature_fail_total[f] += 1
                    if pred_val < 0.5:
                        per_feature_fail_correct[f] += 1

        # Raw affordance accuracy
        affordance_accuracy = aff_correct / max(aff_total, 1)

        # Per-feature accuracy
        per_feature_accuracy = {
            f: per_feature_correct[f] / max(per_feature_total[f], 1)
            for f in CORE_ACTION_FEATURES
        }

        # Per-feature class coverage: which classes (success/fail) are present
        per_feature_class_coverage = {}
        for f in CORE_ACTION_FEATURES:
            has_success = per_feature_success_total[f] > 0
            has_fail = per_feature_fail_total[f] > 0
            per_feature_class_coverage[f] = {
                "n_success": per_feature_success_total[f],
                "n_fail": per_feature_fail_total[f],
                "has_both": has_success and has_fail,
            }

        # Fixed balanced affordance accuracy:
        # - both classes present: average of success_recall and fail_recall
        # - only success: use success_recall
        # - only fail: use fail_recall
        # - neither: skip feature
        feature_balanced = {}
        feature_balanced_fixed = {}
        n_balanced_features = 0
        for f in CORE_ACTION_FEATURES:
            n_succ = per_feature_success_total[f]
            n_fail = per_feature_fail_total[f]
            succ_acc = (per_feature_success_correct[f] /
                        max(n_succ, 1))
            fail_acc = (per_feature_fail_correct[f] /
                        max(n_fail, 1))
            # Original (always two-class average, penalizes single-class features)
            feature_balanced[f] = (succ_acc + fail_acc) / 2.0

            # Fixed: handle single-class features correctly
            if n_succ > 0 and n_fail > 0:
                feature_balanced_fixed[f] = (succ_acc + fail_acc) / 2.0
            elif n_succ > 0:
                feature_balanced_fixed[f] = succ_acc
            elif n_fail > 0:
                feature_balanced_fixed[f] = fail_acc
            # else: neither — skip (shouldn't happen)

        balanced_affordance_accuracy = sum(feature_balanced.values()) / max(
            len(feature_balanced), 1)
        fixed_balanced_affordance_accuracy = (
            sum(feature_balanced_fixed.values()) /
            max(len(feature_balanced_fixed), 1)
        )

        # Macro action accuracy (each action = one CORE_ACTION_FEATURE)
        macro_action_accuracy = sum(per_feature_accuracy.values()) / max(
            len(per_feature_accuracy), 1)

        # Per-category accuracy
        per_category_accuracy = {
            cat: per_category_correct[cat] / max(per_category_total[cat], 1)
            for cat in per_category_correct
        }

        # ---- Composite task accuracy ----
        task_correct = 0
        task_total = 0
        per_query_correct = {}
        per_query_total = {}
        for query_name, query_fn in _TASK_QUERIES.items():
            per_query_correct[query_name] = 0
            per_query_total[query_name] = 0
            for oid, probs in per_object.items():
                gt = self._env.get_ground_truth_affordances(oid)
                pred_match = self._check_query(probs, query_name)
                true_match = query_fn(gt)
                if pred_match == true_match:
                    task_correct += 1
                    per_query_correct[query_name] += 1
                task_total += 1
                per_query_total[query_name] += 1
        composite_task_accuracy = task_correct / max(task_total, 1)
        target_query_accuracy = {
            q: per_query_correct[q] / max(per_query_total[q], 1)
            for q in per_query_correct
        }
        macro_query_accuracy = sum(target_query_accuracy.values()) / max(
            len(target_query_accuracy), 1)

        # ---- Cost metrics from event log ----
        events = event_log.to_list()
        total_reach_cost = sum(
            e.get("cost", 0.0) for e in events if e.get("event") == "reach"
        )
        total_observe_cost = sum(
            e.get("cost", 0.0) for e in events if e.get("event") == "observe"
        )
        total_probe_cost = sum(
            e.get("cost", 0.0) for e in events if e.get("event") == "probe"
        )
        total_cost = event_log.total_cost()
        initial_budget = agent_obs.initial_budget
        normalized_cost = total_cost / max(initial_budget, 0.001)

        # ---- Probe coverage ----
        objects_probed = 0
        for oid in agent_obs.get_all_object_ids():
            if agent_obs.get_probe_results(oid):
                objects_probed += 1
        probe_coverage = objects_probed / max(n_objects, 1)

        # Objects visited
        objects_visited = sum(
            1 for oid in agent_obs.get_all_object_ids()
            if agent_obs.is_visited(oid)
        )

        # ---- Utility (normalized, with beta sweep) ----
        # task_score uses composite_task_accuracy as primary score
        task_score = composite_task_accuracy
        utility_betas = {}
        for beta in config.UTILITY_BETAS:
            utility_betas[f"utility_beta{beta}"] = (
                task_score - beta * normalized_cost
            )

        # Also report raw utility for backwards compat
        total_utility = affordance_accuracy - total_cost

        # ---- Entropy gap (from policy-tracked pre-probe entropies) ----
        probed_entropies = []
        skipped_considered_entropies = []
        never_considered_entropies = []

        probed_oids = set()
        for oid in agent_obs.get_all_object_ids():
            if agent_obs.get_probe_results(oid):
                probed_oids.add(oid)

        for oid in agent_obs.get_all_object_ids():
            if oid in probed_oids:
                if oid in pre_probe_entropies:
                    probed_entropies.append(pre_probe_entropies[oid])
            elif oid in pre_decision_entropies:
                skipped_considered_entropies.append(pre_decision_entropies[oid])
            else:
                # Object was never considered by the policy decision loop
                never_considered_entropies.append(0.0)  # placeholder

        mean_probed_pre_entropy = (
            sum(probed_entropies) / max(len(probed_entropies), 1)
        )
        mean_skipped_pre_entropy = (
            sum(skipped_considered_entropies) /
            max(len(skipped_considered_entropies), 1)
            if skipped_considered_entropies else 0.0
        )
        entropy_gap = mean_probed_pre_entropy - mean_skipped_pre_entropy

        # Fallback: if policy didn't track entropies, use post-hoc
        # (marked as fallback so report can distinguish)
        entropy_gap_fallback = not bool(pre_probe_entropies)

        n_probed_entropy = len(probed_entropies)
        n_skipped_entropy = len(skipped_considered_entropies)
        n_never_considered = len(never_considered_entropies)

        return {
            "n_objects": n_objects,
            # Primary task metrics
            "composite_task_accuracy": composite_task_accuracy,
            "macro_query_accuracy": macro_query_accuracy,
            "target_query_accuracy": target_query_accuracy,
            # Affordance metrics
            "affordance_accuracy": affordance_accuracy,
            "balanced_affordance_accuracy": balanced_affordance_accuracy,
            "fixed_balanced_affordance_accuracy": fixed_balanced_affordance_accuracy,
            "macro_action_accuracy": macro_action_accuracy,
            "per_feature_accuracy": per_feature_accuracy,
            "per_feature_class_coverage": per_feature_class_coverage,
            "feature_balanced": feature_balanced,
            "feature_balanced_fixed": feature_balanced_fixed,
            "per_category_accuracy": per_category_accuracy,
            # Cost and coverage
            "probe_coverage": probe_coverage,
            "objects_visited": objects_visited,
            "objects_probed": objects_probed,
            "total_reach_cost": total_reach_cost,
            "total_observe_cost": total_observe_cost,
            "total_probe_cost": total_probe_cost,
            "total_cost": total_cost,
            "normalized_cost": normalized_cost,
            "initial_budget": initial_budget,
            # Utility
            "total_utility": total_utility,
            "utility_betas": utility_betas,
            # Entropy
            "entropy_gap": entropy_gap,
            "entropy_gap_fallback": entropy_gap_fallback,
            "mean_probed_pre_entropy": mean_probed_pre_entropy,
            "mean_skipped_pre_entropy": mean_skipped_pre_entropy,
            "n_probed_entropy": n_probed_entropy,
            "n_skipped_entropy": n_skipped_entropy,
            "n_never_considered": n_never_considered,
            # Budget
            "budget_remaining": agent_obs.budget_remaining,
            "budget_spent": agent_obs.budget_spent,
            "budget_used": agent_obs.budget_spent,
        }

    def _check_query(self, probs, query_name):
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


def _binary_entropy(p):
    p = max(1e-9, min(1.0 - 1e-9, p))
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
