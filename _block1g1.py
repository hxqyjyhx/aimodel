"""Block 1G1: Minimal Environment Redesign v0 — instance-level affordance variation audit.

Adds within-category instance-level variation to reduce prior dominance,
then runs the same marginal value audit as Block 1G0.
"""
import sys, os, json, time, copy, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("runs", exist_ok=True)

t0 = time.time()

# ---- Paths for imports (must match run_004_5n1.py) ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_5L = os.path.join(CURRENT_DIR, "..", "exp004_5l_instance_outcome_memory")
PARENT_5A3 = os.path.join(CURRENT_DIR, "..", "exp004_5a3_tool_material_transfer")
PARENT_5K = os.path.join(CURRENT_DIR, "..", "exp004_5k_episodic_sparse_outcome")
sys.path.insert(0, PARENT_5L)
sys.path.insert(0, PARENT_5A3)
sys.path.insert(0, PARENT_5K)

import config
from environment import MiniMCEnvironment, _TASK_QUERIES
from harness import EpisodeHarness
from evaluator import MiniMCEvaluator
from policies import (
    C0b_ObserveOnlyPolicy,
    CORE_ACTION_FEATURES, MAIN_CANDIDATE_ACTIONS, ACTION_TO_FEATURE,
)
from simulator_truth import MiniMCSimulatorTruth
from instance_outcome_memory import InstanceOutcomeMemory
from sparse_outcome_collector import collect_sparse_probe_outcomes

# Import objects module for monkey-patching
import objects as obj_module

seed = 101
budget = 1.5
CONDITION_LABEL = "C3v_instance_variation_v0"

# Instance-level variation: each category splits into two hidden sub-types (A/B).
# Sub-type A = original category profile (no change).
# Sub-type B = one key action flipped (50% of instances).
# The sub-type is HIDDEN — not reflected in visible features.
# Probing the flipped action reveals the instance-specific sub-type.
# With 50-50 split, the IOM prior predicts p~0.5 for the varied action,
# putting task queries at the decision boundary and creating probe value.
INSTANCE_VARIATION = {
    "wood_log": {
        "craft_plank": 0.50,   # 50% of wood_logs cannot craft planks (sub-type B)
    },
    "apple": {
        "eat": 0.50,           # 50% of apples are inedible (sub-type B)
    },
    "stone_block": {
        "mine_with_pickaxe": 0.50,  # 50% of stone blocks resist pickaxe (sub-type B)
    },
    "wooden_pickaxe": {
        "use_as_tool": 0.50,   # 50% of pickaxes are too fragile to use (sub-type B)
    },
}

# Separate RNG for instance variation flips — keeps main RNG state unchanged.
_variation_rng = random.Random(seed * 3 + 7777)

print("=" * 70)
print("Block 1G1: Instance-Level Affordance Variation Audit")
print(f"  condition={CONDITION_LABEL}")
print(f"  budget={budget}, seed={seed}")
print(f"  instance_variation={INSTANCE_VARIATION}")
print("=" * 70)

# ============ Monkey-patch _make_hidden_profile ============
_original_make_hidden_profile = obj_module._make_hidden_profile

def _patched_make_hidden_profile(category, rng):
    profile = _original_make_hidden_profile(category, rng)
    overrides = INSTANCE_VARIATION.get(category, {})
    for action, flip_prob in overrides.items():
        if _variation_rng.random() < flip_prob:
            current = profile.get(action, "success")
            profile[action] = "fail" if current == "success" else "success"
    return profile

obj_module._make_hidden_profile = _patched_make_hidden_profile

try:
    # Now run Phase A training with the patched object generation.
    # The cue condition is still C3_P060_O040 for visible features.
    from run_004_5n1 import run_phase_a_training

    cond = copy.deepcopy({"label": "C3_P060_O040", "p_target": 0.60, "p_other": 0.40, "absent": False})

    print("\n  Phase A: Training (with instance-level variation)...")
    (student, base_learner, train_objects, train_env,
     test_objects, test_env, final_metrics, rng) = run_phase_a_training(seed, cond)

finally:
    # Restore original
    obj_module._make_hidden_profile = _original_make_hidden_profile
    print("  (restored original _make_hidden_profile)")

train_objects_dict = {oid: train_env.objects[oid] for oid in train_objects}
test_oids = sorted(test_objects)
test_objects_dict = {oid: test_env.objects[oid] for oid in test_oids}

# Verify instance variation is present
print("\n  Verifying instance-level variation in test objects...")
from collections import Counter
for category in ["wood_log", "apple", "stone_block", "wooden_pickaxe"]:
    cat_oids = [oid for oid in test_oids if oid.startswith(f"test_{category}")]
    action = list(INSTANCE_VARIATION.get(category, {}).keys())
    if action:
        action = action[0]
        outcomes = Counter()
        for oid in cat_oids:
            gt = test_env.objects[oid].get("hidden_affordance_profile", {})
            outcomes[gt.get(action, "?")] += 1
        print(f"    {category}.{action}: {dict(outcomes)}")

# ============ Helpers ============
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


def compute_comp_from_iom(im, oid_list, env):
    per_object = {}
    for oid in oid_list:
        fake_obj = {"id": oid, "visible_features": {}}
        per_object[oid] = im.predict_all_affordances(fake_obj)
    preds = {"per_object": per_object}
    correct = 0
    total = 0
    for query_name, query_fn in _TASK_QUERIES.items():
        for oid, probs in per_object.items():
            gt = env.get_ground_truth_affordances(oid)
            if _check_query(probs, query_name) == query_fn(gt):
                correct += 1
            total += 1
    return correct / max(total, 1)


# ============ Build IOM ============
outcome_rows, _ = collect_sparse_probe_outcomes(
    train_objects_dict, student, config.COVERAGE, seed)

posterior_visible_features = list(base_learner.visible_feature_names)
im_base = InstanceOutcomeMemory(
    train_objects_dict, posterior_visible_features,
    k=config.INSTANCE_K, similarity_power=config.SIMILARITY_POWER,
    similarity_mode=config.SIMILARITY_MODE,
)
im_base.build(outcome_rows)

positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, rng)

_temp_sim = MiniMCSimulatorTruth(test_objects_dict, positions, config.AGENT_START)
gt_per_object = {oid: _temp_sim.get_ground_truth_affordances(oid) for oid in test_oids}


def probe_summary(records):
    return {
        "n_probe": len(records),
        "n_effective": sum(1 for r in records if r.get("is_effective", False)),
        "n_zero_gain": sum(1 for r in records if r.get("is_zero_gain", False)),
        "n_harmful": sum(1 for r in records if r.get("is_harmful", False)),
        "actual_delta_sum": sum(r.get("actual_delta_task_correct", 0) for r in records),
    }


# ============ Policy classes (same as Block 1G0) ============

class PriorOnlyPolicy:
    def __init__(self, im): self._im = im
    def reset(self, view): pass
    def select_next_object(self, view): return None
    def decide_probe(self, view, oid): return False, None
    def on_probe_result(self, view, oid, action, outcome): pass
    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            fake_obj = {"id": oid, "visible_features": {}}
            per_object[oid] = self._im.predict_all_affordances(fake_obj)
        return {"per_object": per_object}


class ObsNPolicy:
    def __init__(self, im, n_objects, rng):
        self._im = im; self._n = n_objects; self._rng = rng
        self._visited_count = 0

    def reset(self, view): self._visited_count = 0

    def select_next_object(self, view):
        if self._visited_count >= self._n:
            return None
        unvisited = view.get_unvisited_objects()
        if not unvisited: return None
        best_oid, best_cost = None, float('inf')
        for oid in unvisited:
            cost = view.compute_reach_cost(oid) + view.observe_cost
            if view.can_afford(cost) and cost < best_cost:
                best_cost = cost; best_oid = oid
        if best_oid is not None: self._visited_count += 1
        return best_oid

    def decide_probe(self, view, oid): return False, None
    def on_probe_result(self, view, oid, action, outcome): pass
    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            fake_obj = {"id": oid, "visible_features": features}
            per_object[oid] = self._im.predict_all_affordances(fake_obj)
        return {"per_object": per_object}


def run_policy(policy, env, label):
    result = EpisodeHarness(env, policy).run()
    evaluator = MiniMCEvaluator(env)
    metrics = evaluator.evaluate(
        result.predictions, result.event_log, result.agent_obs,
        result.pre_probe_entropies, result.pre_decision_entropies,
    )
    return metrics, result


# ======== 1. C_minus_prior_only ========
print("\n  Running C_minus_prior_only...")
env_prior = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_prior = im_base.clone()
prior_metrics, _ = run_policy(PriorOnlyPolicy(im_prior), env_prior, "C_minus_prior_only")
C_prior_comp = prior_metrics["composite_task_accuracy"]

# ======== 2. C_obs2 ========
print("  Running C_obs2...")
env_obs2 = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_obs2 = im_base.clone()
rng_obs2 = random.Random(seed + 100)
obs2_metrics, _ = run_policy(ObsNPolicy(im_obs2, 2, rng_obs2), env_obs2, "C_obs2")
C_obs2_comp = obs2_metrics["composite_task_accuracy"]

# ======== 3. C_obs_full (C0b) ========
print("  Running C_obs_full (C0b)...")
env_full = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_full = im_base.clone()
rng_full = random.Random(seed + 100)
full_metrics, _ = run_policy(C0b_ObserveOnlyPolicy(im_full, rng_full), env_full, "C_obs_full")
C_full_comp = full_metrics["composite_task_accuracy"]

# ======== 4. C_oracle_budgeted ========
print("  Running C_oracle_budgeted...")
env_budgeted = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_budgeted = im_base.clone()

obs = env_budgeted.reset()

class BudgetedOraclePolicy:
    def __init__(self, im): self._im = im
    def reset(self, view): pass
    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited: return None
        best_oid, best_cost = None, float('inf')
        for oid in unvisited:
            cost = view.compute_reach_cost(oid) + view.observe_cost
            if view.can_afford(cost) and cost < best_cost:
                best_cost = cost; best_oid = oid
        return best_oid
    def decide_probe(self, view, oid): return False, None
    def on_probe_result(self, view, oid, action, outcome):
        self._im.incorporate_probe(oid, action, outcome)
    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            fake_obj = {"id": oid, "visible_features": features}
            per_object[oid] = self._im.predict_all_affordances(fake_obj)
        return {"per_object": per_object}

oracle_pol = BudgetedOraclePolicy(im_budgeted)
oracle_pol.reset(obs)

while not obs.is_terminal():
    target = oracle_pol.select_next_object(obs)
    if target is None: break
    if not obs.can_afford(obs.compute_reach_cost(target) + obs.observe_cost): break
    env_budgeted.reach(obs, target)
    env_budgeted.observe(obs, target)
    for action in MAIN_CANDIDATE_ACTIONS:
        if not obs.can_afford(obs.probe_cost): break
        outcome_float, _ = env_budgeted.probe(obs, target, action)
        oracle_pol.on_probe_result(obs, target, action, outcome_float)

preds_budgeted = oracle_pol.get_answer(obs)
eval_budgeted = MiniMCEvaluator(env_budgeted)
budgeted_metrics = eval_budgeted.evaluate(preds_budgeted, obs.event_log, obs, {}, {})
C_budgeted_comp = budgeted_metrics["composite_task_accuracy"]

n_objects_fully_probed = 0
for oid in obs.get_all_object_ids():
    results = obs.get_probe_results(oid)
    if results and len(results) >= 5:
        n_objects_fully_probed += 1

# ======== 5. C_oracle_unconstrained ========
print("  Computing C_oracle_unconstrained...")
im_unconstrained = im_base.clone()
for oid in test_oids:
    gt = gt_per_object[oid]
    for action in MAIN_CANDIDATE_ACTIONS:
        feature = ACTION_TO_FEATURE[action]
        true_val = gt.get(feature, 0.0)
        im_unconstrained.incorporate_probe(oid, action, true_val)

C_oracle_comp = compute_comp_from_iom(im_unconstrained, test_oids, env_budgeted)

# ======== Report ========
BETA = 0.25

print()
print("=" * 70)
print(f"Block 1G1 — Instance Variation Audit ({CONDITION_LABEL})")
print("=" * 70)


def print_baseline(label, metrics, extra=None):
    comp = metrics["composite_task_accuracy"]
    ncost = metrics["normalized_cost"]
    ub25 = comp - BETA * ncost
    print(f"\n--- {label} ---")
    print(f"  comp={comp:.4f}  ncost={ncost:.4f}  ub25={ub25:.4f}")
    print(f"  visited={metrics['objects_visited']}"
          f"  probe_count={metrics['objects_probed']}"
          f"  total_cost={metrics['total_cost']:.4f}"
          f"  budget_remaining={metrics['budget_remaining']:.4f}")
    if extra:
        for k, v in extra.items():
            print(f"  {k}={v}")
    return comp, ncost, ub25


C_prior_comp, _, _ = print_baseline("C_minus_prior_only", prior_metrics)
C_obs2_comp, _, _ = print_baseline("C_obs2", obs2_metrics)
C_full_comp, _, _ = print_baseline("C_obs_full (C0b)", full_metrics)
C_budgeted_comp, _, _ = print_baseline("C_oracle_budgeted", budgeted_metrics,
                                       {"n_objects_fully_probed": n_objects_fully_probed})

print(f"\n--- C_oracle_unconstrained ---")
print(f"  comp={C_oracle_comp:.4f}  (no budget constraint)")

# Deltas
delta_prior_to_obs2 = C_obs2_comp - C_prior_comp
delta_obs2_to_full = C_full_comp - C_obs2_comp
delta_full_to_budgeted = C_budgeted_comp - C_full_comp
delta_full_to_oracle = C_oracle_comp - C_full_comp
delta_prior_to_oracle = C_oracle_comp - C_prior_comp

print()
print(f"--- Marginal Value Curve (v0 vs original) ---")
print(f"  delta_Cminus_to_Cobs2:      {delta_prior_to_obs2:+.4f}")
print(f"  delta_Cobs2_to_CobsFull:    {delta_obs2_to_full:+.4f}")
print(f"  delta_CobsFull_to_Coracle_budgeted: {delta_full_to_budgeted:+.4f}")
print(f"  delta_CobsFull_to_Coracle:  {delta_full_to_oracle:+.4f}")
print(f"  delta_Cminus_to_Coracle:    {delta_prior_to_oracle:+.4f}")

# Diagnostic targets
prior_dominance = C_full_comp - C_prior_comp
prior_reduced = prior_dominance >= 0.05
budgeted_increased = delta_full_to_budgeted >= 0.03
oracle_gap_ok = delta_full_to_oracle >= 0.15
ready = prior_reduced and budgeted_increased

print()
print(f"--- Diagnostic Targets ---")
print(f"  C_obs_full - C_minus: {prior_dominance:+.4f}  (target >= +0.05)  {'OK' if prior_reduced else 'FAIL'}")
print(f"  C_full to C_oracle_budgeted: {delta_full_to_budgeted:+.4f}  (target >= +0.03)  {'OK' if budgeted_increased else 'FAIL'}")
print(f"  C_full to C_oracle_unconstrained: {delta_full_to_oracle:+.4f}  (target >= +0.15)  {'OK' if oracle_gap_ok else 'FAIL'}")

# Compare with Block 1G0 original
print(f"\n--- Comparison with Block 1G0 (original C3_P060_O040) ---")
print(f"  {'metric':<40} {'1G0_original':<16} {'1G1_v0':<16}")
print(f"  {'C_minus_comp':<40} {'0.7000':<16} {C_prior_comp:<16.4f}")
print(f"  {'C_obs_full_comp':<40} {'0.7033':<16} {C_full_comp:<16.4f}")
print(f"  {'C_obs_full - C_minus':<40} {'+0.0033':<16} {prior_dominance:<+16.4f}")
print(f"  {'C_oracle_unconstrained':<40} {'1.0000':<16} {C_oracle_comp:<16.4f}")
print(f"  {'C_full to oracle gap':<40} {'+0.2967':<16} {delta_full_to_oracle:<+16.4f}")
print(f"  {'C_oracle_budgeted':<40} {'0.7167':<16} {C_budgeted_comp:<16.4f}")
print(f"  {'C_full to budgeted gap':<40} {'+0.0133':<16} {delta_full_to_budgeted:<+16.4f}")

print()
print("[block_done]")
print(f"  block_id=1G1")
print(f"  condition={CONDITION_LABEL}")
print(f"  C_minus_comp={C_prior_comp:.4f}")
print(f"  C_obs2_comp={C_obs2_comp:.4f}")
print(f"  C_obs_full_comp={C_full_comp:.4f}")
print(f"  C_oracle_budgeted_comp={C_budgeted_comp:.4f}")
print(f"  C_oracle_unconstrained_comp={C_oracle_comp:.4f}")
print(f"  prior_dominance_reduced={'true' if prior_reduced else 'false'}")
print(f"  budgeted_probe_value_increased={'true' if budgeted_increased else 'false'}")
print(f"  ready_for_policy_rerun={'true' if ready else 'false'}")
print(f"  elapsed={time.time() - t0:.0f}s")

# ======== Save ========
block_out = {
    "block_id": "1G1",
    "elapsed_s": time.time() - t0,
    "condition": CONDITION_LABEL,
    "budget": budget, "seed": seed, "beta": BETA,
    "instance_variation": INSTANCE_VARIATION,
    "C_minus_prior_only": {
        "comp": C_prior_comp,
        "visited_count": prior_metrics["objects_visited"],
        "total_cost": prior_metrics["total_cost"],
        "budget_remaining": prior_metrics["budget_remaining"],
    },
    "C_obs2": {
        "comp": C_obs2_comp,
        "visited_count": obs2_metrics["objects_visited"],
        "total_cost": obs2_metrics["total_cost"],
        "budget_remaining": obs2_metrics["budget_remaining"],
    },
    "C_obs_full": {
        "comp": C_full_comp,
        "visited_count": full_metrics["objects_visited"],
        "total_cost": full_metrics["total_cost"],
        "budget_remaining": full_metrics["budget_remaining"],
    },
    "C_oracle_budgeted": {
        "comp": C_budgeted_comp,
        "visited_count": budgeted_metrics["objects_visited"],
        "probe_count": budgeted_metrics["objects_probed"],
        "total_cost": budgeted_metrics["total_cost"],
        "budget_remaining": budgeted_metrics["budget_remaining"],
        "n_objects_fully_probed": n_objects_fully_probed,
    },
    "C_oracle_unconstrained": {
        "comp": C_oracle_comp,
        "note": "All objects, all 5 candidate actions probed with true outcomes, budget ignored",
    },
    "marginal_value_curve": {
        "delta_Cminus_to_Cobs2": delta_prior_to_obs2,
        "delta_Cobs2_to_CobsFull": delta_obs2_to_full,
        "delta_CobsFull_to_Coracle_budgeted": delta_full_to_budgeted,
        "delta_CobsFull_to_Coracle": delta_full_to_oracle,
        "delta_Cminus_to_Coracle": delta_prior_to_oracle,
    },
    "diagnostic_targets": {
        "prior_dominance": prior_dominance,
        "prior_dominance_reduced": prior_reduced,
        "target_prior": ">= +0.05",
        "budgeted_probe_value": delta_full_to_budgeted,
        "budgeted_probe_value_increased": budgeted_increased,
        "target_budgeted": ">= +0.03",
        "oracle_gap": delta_full_to_oracle,
        "oracle_gap_ok": oracle_gap_ok,
        "target_oracle": ">= +0.15",
        "ready_for_policy_rerun": ready,
    },
    "comparison_with_1G0": {
        "C_minus_comp_1G0": 0.7000, "C_minus_comp_1G1": C_prior_comp,
        "C_obs_full_comp_1G0": 0.7033, "C_obs_full_comp_1G1": C_full_comp,
        "prior_dominance_1G0": 0.0033, "prior_dominance_1G1": prior_dominance,
        "oracle_unconstrained_1G0": 1.0000, "oracle_unconstrained_1G1": C_oracle_comp,
        "oracle_budgeted_1G0": 0.7167, "oracle_budgeted_1G1": C_budgeted_comp,
    },
}
with open("runs/calibration_block1g1_instance_variation_audit.json", "w") as f:
    json.dump(block_out, f, indent=2)
print(f"  saved: runs/calibration_block1g1_instance_variation_audit.json")
