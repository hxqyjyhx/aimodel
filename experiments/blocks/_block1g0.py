"""Block 1G0: Environment Marginal Value Audit.

Measures the comp curve from prior-only → few observations →
full observation → oracle probing, to quantify where value lives.
"""
import sys, os, json, time, copy, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("runs", exist_ok=True)

t0 = time.time()

import config
from environment import MiniMCEnvironment, _TASK_QUERIES
from harness import EpisodeHarness
from evaluator import MiniMCEvaluator
from policies import (
    C0b_ObserveOnlyPolicy,
    CORE_ACTION_FEATURES, MAIN_CANDIDATE_ACTIONS, ACTION_TO_FEATURE,
    _compute_utility,
)
from simulator_truth import MiniMCSimulatorTruth

PARENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "exp004_5l_instance_outcome_memory")
sys.path.insert(0, PARENT_DIR)
from instance_outcome_memory import InstanceOutcomeMemory
from run_004_5n1 import run_phase_a_training
from sparse_outcome_collector import collect_sparse_probe_outcomes

seed = 101
cond = copy.deepcopy({"label": "C3_P060_O040", "p_target": 0.60, "p_other": 0.40, "absent": False})
budget = 1.5

print("=" * 70)
print("Block 1G0: Environment Marginal Value Audit")
print(f"  cue=C3_P060_O040, budget={budget}, seed={seed}")
print("=" * 70)


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


def compute_comp_from_iom(im, view_or_objects, env):
    """Compute composite_task_accuracy from IOM predictions given a view or object list."""
    # Get all object IDs and their observed features
    per_object = {}
    if hasattr(view_or_objects, 'get_all_object_ids'):
        # It's an AgentObsView
        for oid in view_or_objects.get_all_object_ids():
            features = view_or_objects.get_observed_features(oid) or {}
            fake_obj = {"id": oid, "visible_features": features}
            per_object[oid] = im.predict_all_affordances(fake_obj)
    else:
        # It's a list of object IDs, use empty features
        for oid in view_or_objects:
            fake_obj = {"id": oid, "visible_features": {}}
            per_object[oid] = im.predict_all_affordances(fake_obj)

    preds = {"per_object": per_object}

    correct = 0
    total = 0
    for query_name, query_fn in _TASK_QUERIES.items():
        for oid, probs in per_object.items():
            gt = env.get_ground_truth_affordances(oid)
            pred_match = _check_query(probs, query_name)
            true_match = query_fn(gt)
            if pred_match == true_match:
                correct += 1
            total += 1
    return correct / max(total, 1)


# ============ Policy: PriorOnly ============
class PriorOnlyPolicy:
    """Visit 0 objects. Answer from IOM prior only (empty features for all)."""

    def __init__(self, im):
        self._im = im

    def reset(self, view): pass

    def select_next_object(self, view):
        return None  # Never visit anything

    def decide_probe(self, view, object_id):
        return False, None

    def on_probe_result(self, view, object_id, action, outcome): pass

    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            fake_obj = {"id": oid, "visible_features": {}}
            per_object[oid] = self._im.predict_all_affordances(fake_obj)
        return {"per_object": per_object}


# ============ Policy: ObsN (visit+observe exactly N objects, nearest first) ============
class ObsNPolicy:
    """Visit+observe exactly N objects (nearest first), then stop. Never probe."""

    def __init__(self, im, n_objects, rng):
        self._im = im
        self._n = n_objects
        self._rng = rng
        self._visited_count = 0

    def reset(self, view):
        self._visited_count = 0

    def select_next_object(self, view):
        if self._visited_count >= self._n:
            return None
        unvisited = view.get_unvisited_objects()
        if not unvisited:
            return None
        # Nearest first (same as C0b)
        best_oid = None
        best_cost = float('inf')
        for oid in unvisited:
            cost = view.compute_reach_cost(oid) + view.observe_cost
            if view.can_afford(cost) and cost < best_cost:
                best_cost = cost
                best_oid = oid
        if best_oid is not None:
            self._visited_count += 1
        return best_oid

    def decide_probe(self, view, object_id):
        return False, None

    def on_probe_result(self, view, object_id, action, outcome): pass

    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            fake_obj = {"id": oid, "visible_features": features}
            per_object[oid] = self._im.predict_all_affordances(fake_obj)
        return {"per_object": per_object}


# ============ Policy: BudgetedOracleFullProbe ============
class BudgetedOracleFullProbePolicy:
    """Visit objects (nearest first), probe ALL 5 candidate actions with true
    outcomes. Stop when budget exhausted. Oracle = uses simulator for true outcomes."""

    def __init__(self, im, simulator, rng):
        self._im = im
        self._sim = simulator
        self._rng = rng
        self._pre_probe_entropies = {}
        self._pre_decision_entropies = {}

    def reset(self, view):
        self._pre_probe_entropies = {}
        self._pre_decision_entropies = {}

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited:
            return None
        # Nearest first
        best_oid = None
        best_cost = float('inf')
        for oid in unvisited:
            cost = view.compute_reach_cost(oid) + view.observe_cost
            if view.can_afford(cost) and cost < best_cost:
                best_cost = cost
                best_oid = oid
        return best_oid

    def decide_probe(self, view, object_id):
        features = view.get_observed_features(object_id)
        if features is None:
            return False, None
        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        self._pre_probe_entropies[object_id] = sum(
            max(0.0001, min(0.9999, probs.get(f, 0.5))) for f in CORE_ACTION_FEATURES
        ) / len(CORE_ACTION_FEATURES)

        # Probe the first affordable candidate action with true outcome
        # (We'll probe ALL actions by being called repeatedly)
        for action in MAIN_CANDIDATE_ACTIONS:
            if view.can_afford(view.probe_cost):
                return True, action
        return False, None

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


# ======== Phase A: Training ========
print("\n  Phase A: Training...")
(student, base_learner, train_objects, train_env,
 test_objects, test_env, final_metrics, rng) = run_phase_a_training(seed, cond)

train_objects_dict = {oid: train_env.objects[oid] for oid in train_objects}
test_oids = sorted(test_objects)
test_objects_dict = {oid: test_env.objects[oid] for oid in test_oids}

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
prior_policy = PriorOnlyPolicy(im_prior)
# We still need to run through the harness for consistency
# PriorOnly returns None from select_next_object immediately
# But we need an AgentObs to compute the answer. Let's do it via the harness.
prior_metrics, prior_result = run_policy(prior_policy, env_prior, "C_minus_prior_only")
prior_comp = prior_metrics["composite_task_accuracy"]
# But wait — the harness calls env.reset() which creates a fresh AgentObs.
# PriorOnly returns None immediately, so the episode ends with 0 visits.
# Then get_answer computes predictions with empty features for all objects.
# This is correct.


# ======== 2. C_obs2 ========
print("  Running C_obs2...")
env_obs2 = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_obs2 = im_base.clone()
rng_obs2 = random.Random(seed + 100)
obs2_policy = ObsNPolicy(im_obs2, 2, rng_obs2)
obs2_metrics, obs2_result = run_policy(obs2_policy, env_obs2, "C_obs2")
obs2_comp = obs2_metrics["composite_task_accuracy"]


# ======== 3. C_obs_full (C0b) ========
print("  Running C_obs_full (C0b)...")
env_full = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_full = im_base.clone()
rng_full = random.Random(seed + 100)
full_policy = C0b_ObserveOnlyPolicy(im_full, rng_full)
full_metrics, full_result = run_policy(full_policy, env_full, "C_obs_full")
full_comp = full_metrics["composite_task_accuracy"]


# ======== 4. C_oracle_full_probe ========
# (a) Budgeted: Visit objects (nearest first), probe ALL 5 actions with true outcome,
#     continue until budget exhausted.
print("  Running C_oracle_budgeted...")
env_budgeted = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_budgeted = im_base.clone()
rng_budgeted = random.Random(seed + 200)

# We need a custom run that probes all 5 actions per object using true outcomes.
# The harness only probes one action per decide_probe call.
# So we need to either:
#   a. Override the harness behavior, or
#   b. Write a custom run loop
# Let's write a custom run loop for the budgeted oracle.

obs_budgeted = env_budgeted.reset()

# Use the policy for object selection
budgeted_policy = BudgetedOracleFullProbePolicy(im_budgeted,
                                                env_budgeted.simulator,
                                                rng_budgeted)
budgeted_policy.reset(obs_budgeted)

while not obs_budgeted.is_terminal():
    target = budgeted_policy.select_next_object(obs_budgeted)
    if target is None:
        break

    reach_cost = obs_budgeted.compute_reach_cost(target)
    # Check if we can afford at least reach+observe
    if not obs_budgeted.can_afford(reach_cost + obs_budgeted.observe_cost):
        break

    env_budgeted.reach(obs_budgeted, target)
    env_budgeted.observe(obs_budgeted, target)

    # Probe ALL 5 candidate actions using true outcomes, budget permitting
    for action in MAIN_CANDIDATE_ACTIONS:
        if not obs_budgeted.can_afford(obs_budgeted.probe_cost):
            break
        true_outcome, _ = env_budgeted.simulator.probe(target, action)
        env_budgeted.probe(obs_budgeted, target, action)
        # probe() in the env calls agent_obs._probe which calls simulator.probe
        # But we already know the true outcome. We need to incorporate it directly
        # into the IOM. Let me instead incorporate it into the policy's IOM.
        # Actually, env.probe already returns the outcome from the simulator.
        # The issue is that the harness normally calls policy.on_probe_result.
        # Let me do that manually.
        budgeted_policy.on_probe_result(obs_budgeted, target, action, true_outcome)

# Now get answer and evaluate
preds_budgeted = budgeted_policy.get_answer(obs_budgeted)
evaluator_budgeted = MiniMCEvaluator(env_budgeted)
budgeted_metrics = evaluator_budgeted.evaluate(
    preds_budgeted, obs_budgeted.event_log, obs_budgeted,
    budgeted_policy.get_pre_probe_entropies(),
    budgeted_policy.get_pre_decision_entropies(),
)
budgeted_comp = budgeted_metrics["composite_task_accuracy"]

# Count objects fully probed
n_objects_fully_probed = 0
for oid in obs_budgeted.get_all_object_ids():
    probe_results = obs_budgeted.get_probe_results(oid)
    if probe_results and len(probe_results) >= 5:
        n_objects_fully_probed += 1

# (b) Unconstrained oracle ceiling: incorporate true outcomes for ALL actions
# on ALL objects into a fresh IOM, ignoring budget entirely.
print("  Computing C_oracle_unconstrained...")
im_unconstrained = im_base.clone()
for oid in test_oids:
    gt = gt_per_object[oid]
    for action in MAIN_CANDIDATE_ACTIONS:
        feature = ACTION_TO_FEATURE[action]
        true_val = gt.get(feature, 0.0)
        im_unconstrained.incorporate_probe(oid, action, true_val)

# Compute comp from unconstrained IOM
oracle_comp = compute_comp_from_iom(im_unconstrained, test_oids, env_budgeted)

# ======== Report ========
BETA = 0.25

print()
print("=" * 70)
print("Block 1G0 — Environment Marginal Value Audit")
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
print(f"  comp={oracle_comp:.4f}  (no budget constraint — all objects, all actions)")
print(f"  ncost=not applicable (unconstrained)")

# ======== Curve diagnostics ========
delta_prior_to_obs2 = C_obs2_comp - C_prior_comp
delta_obs2_to_full = C_full_comp - C_obs2_comp
delta_full_to_budgeted = C_budgeted_comp - C_full_comp
delta_full_to_oracle = oracle_comp - C_full_comp
delta_prior_to_oracle = oracle_comp - C_prior_comp

print()
print(f"--- Marginal Value Curve ---")
print(f"  delta_Cminus_to_Cobs2:      {delta_prior_to_obs2:+.4f}  (prior → 2 obs)")
print(f"  delta_Cobs2_to_CobsFull:    {delta_obs2_to_full:+.4f}  (2 obs → full obs)")
print(f"  delta_CobsFull_to_Coracle_budgeted: {delta_full_to_budgeted:+.4f}  (full obs → budgeted oracle probe)")
print(f"  delta_CobsFull_to_Coracle:  {delta_full_to_oracle:+.4f}  (full obs → unconstrained oracle)")
print(f"  delta_Cminus_to_Coracle:    {delta_prior_to_oracle:+.4f}  (total possible gain)")

# Interpretation
prior_dominated = C_full_comp - C_prior_comp < 0.03
obs_saturates = delta_obs2_to_full < 0.01
probe_marginal = delta_full_to_oracle
probe_has_value = probe_marginal > 0.03

print()
print(f"--- Interpretation ---")
print(f"  prior_dominated: {prior_dominated}  (C_full - C_minus = {C_full_comp - C_prior_comp:.4f} {'<' if prior_dominated else '>='} 0.03)")
print(f"  observation_saturates_after_2: {obs_saturates}  (C_full - C_obs2 = {delta_obs2_to_full:.4f})")
print(f"  probe_marginal_value: {probe_marginal:+.4f}  ({'meaningful' if probe_has_value else 'small'})")
print(f"  note: {delta_full_to_budgeted:+.4f} of probe value is realizable under budget=1.5, "
      f"{probe_marginal - delta_full_to_budgeted:+.4f} is unconstrained only")

print()
print("[block_done]")
print(f"  block_id=1G0")
print(f"  C_minus_comp={C_prior_comp:.4f}")
print(f"  C_obs2_comp={C_obs2_comp:.4f}")
print(f"  C_obs_full_comp={C_full_comp:.4f}")
print(f"  C_oracle_comp={oracle_comp:.4f}")
print(f"  delta_obs_full_to_oracle={delta_full_to_oracle:+.4f}")
print(f"  environment_has_probe_value={'true' if probe_has_value else 'false'}")
print(f"  environment_prior_dominated={'true' if prior_dominated else 'false'}")
print(f"  elapsed={time.time() - t0:.0f}s")

# ======== Save ========
block_out = {
    "block_id": "1G0",
    "elapsed_s": time.time() - t0,
    "cue": "C3_P060_O040", "budget": budget, "seed": seed,
    "beta": BETA,
    "C_minus_prior_only": {
        "comp": C_prior_comp,
        "ncost": prior_metrics["normalized_cost"],
        "ub25": C_prior_comp - BETA * prior_metrics["normalized_cost"],
        "visited_count": prior_metrics["objects_visited"],
        "observed_count": prior_metrics["objects_visited"],
        "probe_count": prior_metrics["objects_probed"],
        "total_cost": prior_metrics["total_cost"],
        "budget_remaining": prior_metrics["budget_remaining"],
    },
    "C_obs2": {
        "comp": C_obs2_comp,
        "ncost": obs2_metrics["normalized_cost"],
        "ub25": C_obs2_comp - BETA * obs2_metrics["normalized_cost"],
        "visited_count": obs2_metrics["objects_visited"],
        "observed_count": obs2_metrics["objects_visited"],
        "probe_count": obs2_metrics["objects_probed"],
        "total_cost": obs2_metrics["total_cost"],
        "budget_remaining": obs2_metrics["budget_remaining"],
    },
    "C_obs_full": {
        "comp": C_full_comp,
        "ncost": full_metrics["normalized_cost"],
        "ub25": C_full_comp - BETA * full_metrics["normalized_cost"],
        "visited_count": full_metrics["objects_visited"],
        "observed_count": full_metrics["objects_visited"],
        "probe_count": full_metrics["objects_probed"],
        "total_cost": full_metrics["total_cost"],
        "budget_remaining": full_metrics["budget_remaining"],
    },
    "C_oracle_budgeted": {
        "comp": C_budgeted_comp,
        "ncost": budgeted_metrics["normalized_cost"],
        "ub25": C_budgeted_comp - BETA * budgeted_metrics["normalized_cost"],
        "visited_count": budgeted_metrics["objects_visited"],
        "observed_count": budgeted_metrics["objects_visited"],
        "probe_count": budgeted_metrics["objects_probed"],
        "total_cost": budgeted_metrics["total_cost"],
        "budget_remaining": budgeted_metrics["budget_remaining"],
        "n_objects_fully_probed": n_objects_fully_probed,
    },
    "C_oracle_unconstrained": {
        "comp": oracle_comp,
        "note": "All 60 objects, all 5 candidate actions probed with true outcomes, budget ignored",
    },
    "marginal_value_curve": {
        "delta_Cminus_to_Cobs2": delta_prior_to_obs2,
        "delta_Cobs2_to_CobsFull": delta_obs2_to_full,
        "delta_CobsFull_to_Coracle_budgeted": delta_full_to_budgeted,
        "delta_CobsFull_to_Coracle": delta_full_to_oracle,
        "delta_Cminus_to_Coracle": delta_prior_to_oracle,
    },
    "interpretation": {
        "prior_dominated": prior_dominated,
        "observation_saturates_after_2": obs_saturates,
        "probe_marginal_value": probe_marginal,
        "environment_has_probe_value": probe_has_value,
        "environment_prior_dominated": prior_dominated,
        "note_full_to_budgeted": f"{delta_full_to_budgeted:+.4f} of probe value is realizable under budget=1.5, "
                                 f"{probe_marginal - delta_full_to_budgeted:+.4f} is unconstrained only",
    },
}
with open("runs/calibration_block1g0_environment_marginal_value_audit.json", "w") as f:
    json.dump(block_out, f, indent=2)
print(f"  saved: runs/calibration_block1g0_environment_marginal_value_audit.json")
