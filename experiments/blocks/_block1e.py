"""Block 1E: Pre-probe filter diagnostic — can we predict zero-gain probes?"""
import sys, os, json, time, copy, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("runs", exist_ok=True)

t0 = time.time()

import config
from environment import MiniMCEnvironment
from harness import EpisodeHarness
from evaluator import _TASK_QUERIES
from policies import (
    C0b_ObserveOnlyPolicy, C13_InstanceVOIPolicy,
    CORE_ACTION_FEATURES, MAIN_CANDIDATE_ACTIONS, ACTION_TO_FEATURE,
)

PARENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "exp004_5l_instance_outcome_memory")
sys.path.insert(0, PARENT_DIR)
from instance_outcome_memory import InstanceOutcomeMemory
from run_004_5n1 import run_phase_a_training
from sparse_outcome_collector import collect_sparse_probe_outcomes

seed = 101
cond = copy.deepcopy({"label": "C3_P060_O040", "p_target": 0.60, "p_other": 0.40, "absent": False})
budget = 1.5
cw = 0.5

print("=" * 70)
print("Block 1E: Pre-Probe Filter Diagnostic")
print(f"  cue=C3_P060_O040, budget={budget}, CW={cw}, seed={seed}")
print("=" * 70)

# ============ Helpers ============
def _binary_entropy(p):
    p = max(0.0001, min(0.9999, p))
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

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

def _composite_correct_count(probs, gt):
    correct = 0
    for qname, qfn in _TASK_QUERIES.items():
        if _check_query(probs, qname) == qfn(gt):
            correct += 1
    return correct

def _mean(xs): return sum(xs) / max(len(xs), 1)
def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0: return float('nan')
    if n % 2: return s[n//2]
    return (s[n//2-1] + s[n//2]) / 2

def _recompute_net_voi(im, oid, features, action, cost_weight, norm_probe_cost):
    """Recompute net_voi for a specific action given current IOM state."""
    fake_obj = {"id": oid, "visible_features": features or {}}
    probs = im.predict_all_affordances(fake_obj)
    feature = ACTION_TO_FEATURE[action]
    p_success = probs.get(feature, 0.5)

    def _utility(probs_dict):
        return sum(max(p, 1 - p) for p in probs_dict.values()) / max(len(probs_dict), 1)

    current_utility = _utility(probs)

    im_succ = im.clone()
    im_succ.incorporate_probe(oid, action, 1.0)
    utility_succ = _utility(im_succ.predict_all_affordances(fake_obj))

    im_fail = im.clone()
    im_fail.incorporate_probe(oid, action, 0.0)
    utility_fail = _utility(im_fail.predict_all_affordances(fake_obj))

    e_post = p_success * utility_succ + (1 - p_success) * utility_fail
    return e_post - current_utility - cost_weight * norm_probe_cost


# ============ Enhanced ProbeTracker ============
class PreProbeFeatureTracker:
    """Wraps C13 to capture detailed pre-probe features for every probe decision."""

    def __init__(self, inner_policy, gt_per_object, im):
        self._inner = inner_policy
        self._gt = gt_per_object
        self._im = im
        self.probe_records = []

    def reset(self, view):       self._inner.reset(view)
    def select_next_object(self, view): return self._inner.select_next_object(view)
    def get_answer(self, view):  return self._inner.get_answer(view)
    def get_pre_probe_entropies(self):
        return getattr(self._inner, '_pre_probe_entropies', {})
    def get_pre_decision_entropies(self):
        return getattr(self._inner, '_pre_decision_entropies', {})
    def on_probe_result(self, view, object_id, action, outcome):
        self._inner.on_probe_result(view, object_id, action, outcome)

        pending = getattr(self, "_pending_pre", None)
        if pending is not None and pending["object_id"] == object_id:
            features = view.get_observed_features(object_id)
            fake_obj = {"id": object_id, "visible_features": features or {}}
            after_probs = self._im.predict_all_affordances(fake_obj)
            gt = pending["gt"]
            after_comp = _composite_correct_count(after_probs, gt)
            delta_comp = after_comp - pending["before_comp"]
            is_effective = delta_comp > 0

            changed_features = []
            for f in CORE_ACTION_FEATURES:
                bp = 1.0 if pending["before_probs"].get(f, 0.5) >= 0.5 else 0.0
                ap = 1.0 if after_probs.get(f, 0.5) >= 0.5 else 0.0
                if bp != ap:
                    changed_features.append(f)

            rec = {
                "object_id": object_id,
                "action": pending["action"],
                "is_effective_probe": is_effective,
                "delta_composite_correct": delta_comp,
                "before_comp": pending["before_comp"],
                "after_comp": after_comp,
                "changed_features": changed_features,
                "changed_to_correct": sum(1 for f in changed_features
                    if (1.0 if after_probs.get(f, 0.5) >= 0.5 else 0.0) == gt.get(f, 0.0)),
                "changed_to_wrong": sum(1 for f in changed_features
                    if (1.0 if after_probs.get(f, 0.5) >= 0.5 else 0.0) != gt.get(f, 0.0)),
            }
            # Merge pre-probe features
            for k, v in pending["pre_features"].items():
                rec[k] = v
            self.probe_records.append(rec)
        self._pending_pre = None

    def decide_probe(self, view, object_id):
        features = view.get_observed_features(object_id)
        gt = self._gt.get(object_id, {})
        fake_obj = {"id": object_id, "visible_features": features or {}}
        before_probs = self._im.predict_all_affordances(fake_obj)
        before_comp = _composite_correct_count(before_probs, gt)

        # ---- Compute all pre-probe features ----
        norm_probe_cost = view.probe_cost / max(view.initial_budget, 0.001)

        # Compute per-action pre-probe features
        per_action_features = {}
        for action in MAIN_CANDIDATE_ACTIONS:
            feature = ACTION_TO_FEATURE[action]
            prob = before_probs.get(feature, 0.5)
            entropy = _binary_entropy(prob)
            dist = abs(prob - 0.5)
            # Recompute net_voi for this action
            net_voi = _recompute_net_voi(
                self._im, object_id, features, action,
                self._inner._cost_weight, norm_probe_cost)
            per_action_features[action] = {
                "feature": feature,
                "prob": round(prob, 6),
                "entropy": round(entropy, 6),
                "distance_to_threshold": round(dist, 6),
                "net_voi": round(net_voi, 6),
            }

        # Object-level entropy stats
        all_entropies = [per_action_features[a]["entropy"] for a in MAIN_CANDIDATE_ACTIONS]
        object_mean_entropy = sum(all_entropies) / len(all_entropies)
        object_max_entropy = max(all_entropies)

        # IOM neighbor stats for each action
        # predict_outcome expects ACTION name (not feature), converts internally
        neighbor_stats = {}
        for action in MAIN_CANDIDATE_ACTIONS:
            try:
                _, _, support = self._im.predict_outcome(
                    {"id": object_id, "visible_features": features or {}}, action)
                neighbor_stats[action] = {
                    "neighbor_count": support.get("neighbor_count", 0),
                    "outcome_variance": round(support.get("outcome_variance", 0.0), 6),
                    "total_similarity_weight": round(support.get("total_similarity_weight", 0.0), 6),
                    "max_similarity": round(support.get("max_similarity", 0.0), 6),
                }
            except Exception:
                neighbor_stats[action] = {
                    "neighbor_count": -1, "outcome_variance": -1.0,
                    "total_similarity_weight": -1.0, "max_similarity": -1.0,
                }

        # Global VOI score for this object
        global_voi = self._inner._global_voi(object_id, view)

        # Now call inner.decide_probe
        should_probe, action = self._inner.decide_probe(view, object_id)

        if should_probe and action is not None:
            # Selected action features
            sel_feat = per_action_features[action]
            neigh = neighbor_stats.get(action, {})

            # Object-level support stats (mean/max across all actions)
            obj_neighbor_counts = [neighbor_stats[a].get("neighbor_count", 0) for a in MAIN_CANDIDATE_ACTIONS]
            obj_outcome_variances = [neighbor_stats[a].get("outcome_variance", 0.0) for a in MAIN_CANDIDATE_ACTIONS]

            pre_features = {
                "action_name": action,
                "feature_probed": ACTION_TO_FEATURE[action],
                "pre_probe_prob_for_action": sel_feat["prob"],
                "pre_probe_entropy_for_action": sel_feat["entropy"],
                "distance_to_threshold": sel_feat["distance_to_threshold"],
                "pre_object_mean_entropy": round(object_mean_entropy, 6),
                "pre_object_max_entropy": round(object_max_entropy, 6),
                "global_voi_score": round(global_voi, 6),
                "selected_action_score": sel_feat["net_voi"],
                "current_budget_remaining": round(view.budget_remaining, 6),
                "reach_cost": round(view.compute_reach_cost(object_id), 6),
                "observe_cost": round(view.observe_cost, 6),
                "probe_cost": round(view.probe_cost, 6),
                "normalized_probe_cost": round(norm_probe_cost, 6),
                # Selected-action support
                "selected_action_neighbor_count": neigh.get("neighbor_count", -1),
                "selected_action_total_similarity_weight": neigh.get("total_similarity_weight", -1.0),
                "selected_action_max_similarity": neigh.get("max_similarity", -1.0),
                "selected_action_outcome_variance": neigh.get("outcome_variance", -1.0),
                # Object-level support
                "object_mean_neighbor_count": round(_mean(obj_neighbor_counts), 2),
                "object_mean_outcome_variance": round(_mean(obj_outcome_variances), 6),
                "object_max_outcome_variance": round(max(obj_outcome_variances), 6),
                # Store all per-action features for detailed analysis
                "per_action_features": per_action_features,
            }

            self._pending_pre = {
                "object_id": object_id,
                "action": action,
                "before_probs": before_probs,
                "before_comp": before_comp,
                "gt": gt,
                "pre_features": pre_features,
            }
        else:
            self._pending_pre = None

        return should_probe, action


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

from simulator_truth import MiniMCSimulatorTruth
positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, rng)

_temp_sim = MiniMCSimulatorTruth(test_objects_dict, positions, config.AGENT_START)
gt_per_object = {oid: _temp_sim.get_ground_truth_affordances(oid) for oid in test_oids}

# ======== Run instrumented C13 ========
print("  Running C13 (CW=0.5, instrumented)...")
env_c13 = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_c13 = im_base.clone()
inner_c13 = C13_InstanceVOIPolicy(im_c13, random.Random(seed + 550), cost_weight=cw)
tracker = PreProbeFeatureTracker(inner_c13, gt_per_object, im_c13)
result_c13 = EpisodeHarness(env_c13, tracker).run()

probe_records = tracker.probe_records
n_probe = len(probe_records)
effective = [r for r in probe_records if r["is_effective_probe"]]
zero_gain = [r for r in probe_records if not r["is_effective_probe"]]
n_eff = len(effective)
n_zero = len(zero_gain)

# C0b reference (from Block 1C saved data, or compute)
# Use the saved Block 1C values for consistency
with open("runs/calibration_block1c_diagnostic.json") as f:
    b1c = json.load(f)
c0b_comp = 0.7033
c0b_ncost = b1c["costs"]["c0b_total"] / budget
c0b_ub25 = c0b_comp - 0.25 * c0b_ncost

c13_ncost = b1c["costs"]["c13_total"] / budget
c13_comp = c0b_comp + b1c["comp_gain"]
c13_ub25 = c13_comp - 0.25 * c13_ncost

# ======== 1. Distribution comparison ========
fields_to_compare = [
    ("pre_probe_entropy_for_action", "pre-probe entropy"),
    ("distance_to_threshold", "distance to threshold"),
    ("pre_object_mean_entropy", "object mean entropy"),
    ("pre_object_max_entropy", "object max entropy"),
    ("pre_probe_prob_for_action", "prob for probed action"),
    ("global_voi_score", "global VOI score"),
    ("selected_action_score", "selected action score"),
    ("current_budget_remaining", "budget remaining"),
    ("selected_action_neighbor_count", "sel. act. neighbor count"),
    ("selected_action_outcome_variance", "sel. act. outcome var"),
    ("object_mean_outcome_variance", "obj mean outcome var"),
    ("object_max_outcome_variance", "obj max outcome var"),
]

# ======== 2. Heuristic filter tests ========
TOTAL_COMPOSITE_DECISIONS = 60 * 5  # n_objects * n_task_queries

def test_filter(name, keep_condition, records):
    """Test a heuristic filter. Returns dict of stats.

    Two estimates (both posthoc_trajectory_estimate — NOT real policy):

    1. Conservative (primary — trajectory-conditional):
       estimated_comp_conservative = current_C13_comp - skipped_effective_delta / TOTAL_COMPOSITE_DECISIONS
       Subtracts lost probe gain from C13's observed comp. Assumes unprobed objects
       perform at C13's unprobed level (worse than C0b).

    2. Optimistic (secondary — full-observe baseline):
       estimated_comp_optimistic = C0b_comp + kept_effective_delta / TOTAL_COMPOSITE_DECISIONS
       Adds kept probe gain to C0b's full-observe baseline. Assumes unprobed objects
       perform at C0b level (better than C13 unprobed).
    """
    kept = [r for r in records if keep_condition(r)]
    skipped = [r for r in records if not keep_condition(r)]
    n_kept = len(kept)
    n_skipped = len(skipped)
    kept_eff = sum(1 for r in kept if r["is_effective_probe"])
    skipped_eff = sum(1 for r in skipped if r["is_effective_probe"])
    kept_zero = sum(1 for r in kept if not r["is_effective_probe"])
    kept_effective_delta = sum(r["delta_composite_correct"] for r in kept)
    skipped_effective_delta = sum(r["delta_composite_correct"] for r in skipped)

    # Conservative: trajectory-conditional
    estimated_comp_conservative = c13_comp - skipped_effective_delta / TOTAL_COMPOSITE_DECISIONS

    # Optimistic: full-observe baseline
    estimated_comp_optimistic = c0b_comp + kept_effective_delta / TOTAL_COMPOSITE_DECISIONS

    est_probe_cost = n_kept * config.PROBE_COST
    c13_visit_obs = b1c["costs"]["c13_visit"] + b1c["costs"]["c13_observe"]
    est_total_cost = c13_visit_obs + est_probe_cost
    est_ncost = est_total_cost / budget

    est_ub25_conservative = estimated_comp_conservative - 0.25 * est_ncost
    est_ub25_optimistic = estimated_comp_optimistic - 0.25 * est_ncost

    return {
        "name": name,
        "label_as": "posthoc_trajectory_estimate",
        "kept_count": n_kept, "skipped_count": n_skipped,
        "kept_effective": kept_eff, "skipped_effective": skipped_eff,
        "kept_zero_gain": kept_zero,
        "kept_effective_delta": kept_effective_delta,
        "skipped_effective_delta": skipped_effective_delta,
        "estimated_probe_cost": round(est_probe_cost, 6),
        "estimated_total_cost": round(est_total_cost, 6),
        "estimated_normalized_cost": round(est_ncost, 6),
        # Conservative (primary — trajectory-conditional)
        "estimated_comp_conservative": round(estimated_comp_conservative, 6),
        "estimated_utility_beta0.25_conservative": round(est_ub25_conservative, 6),
        "utility_gain_over_C0b_conservative": round(est_ub25_conservative - c0b_ub25, 6),
        # Optimistic (secondary — full-observe baseline)
        "estimated_comp_optimistic": round(estimated_comp_optimistic, 6),
        "estimated_utility_beta0.25_optimistic": round(est_ub25_optimistic, 6),
        "utility_gain_over_C0b_optimistic": round(est_ub25_optimistic - c0b_ub25, 6),
    }

# Compute thresholds as percentiles of the actual distributions
all_entropy = [r["pre_probe_entropy_for_action"] for r in probe_records]
all_dist = [r["distance_to_threshold"] for r in probe_records]
all_ovo_var = [r.get("selected_action_outcome_variance", 0) for r in probe_records]
all_sel_score = [r["selected_action_score"] for r in probe_records]

entropy_median = _median(all_entropy)
dist_median = _median(all_dist)
ovo_var_median = _median(all_ovo_var)
sel_score_median = _median(all_sel_score)

# Identify high-efficiency actions (>50% effective rate)
action_rates = {}
for r in probe_records:
    act = r["action_name"]
    action_rates.setdefault(act, {"n": 0, "eff": 0})
    action_rates[act]["n"] += 1
    action_rates[act]["eff"] += 1 if r["is_effective_probe"] else 0
high_eff_actions = {act for act, d in action_rates.items()
                    if d["eff"] / max(d["n"], 1) > 0.5}

filters = [
    # A: entropy threshold
    test_filter("A: entropy >= median",
                lambda r, t=entropy_median: r["pre_probe_entropy_for_action"] >= t,
                probe_records),
    # B: distance_to_threshold threshold
    test_filter("B: distance_to_threshold <= median",
                lambda r, t=dist_median: r["distance_to_threshold"] <= t,
                probe_records),
    # C: outcome_variance threshold
    test_filter("C: outcome_variance >= median",
                lambda r, t=ovo_var_median: r.get("selected_action_outcome_variance", 0) >= t,
                probe_records),
    # D: selected_action_score threshold
    test_filter("D: selected_action_score >= median",
                lambda r, t=sel_score_median: r["selected_action_score"] >= t,
                probe_records),
    # E: combined entropy AND distance
    test_filter("E: entropy>=median AND dist<=median",
                lambda r, te=entropy_median, td=dist_median:
                    r["pre_probe_entropy_for_action"] >= te
                    and r["distance_to_threshold"] <= td,
                probe_records),
    # F: action_name exploratory filter
    test_filter("F: EXPLORATORY action in high_efficiency",
                lambda r, ha=high_eff_actions: r["action_name"] in ha,
                probe_records),
]

# "Perfect" filter (upper bound reference using oracle knowledge)
perfect_filter = test_filter("PERFECT: oracle keep effective only",
    lambda r: r["is_effective_probe"], probe_records)

# Also compute "current C13" and "C0b" as references
total_delta = sum(r["delta_composite_correct"] for r in probe_records)
current_c13_stats = {
    "name": "REF: current_C13",
    "label_as": "REF",
    "kept_count": n_probe, "skipped_count": 0,
    "kept_effective": n_eff, "skipped_effective": 0,
    "kept_zero_gain": n_zero,
    "kept_effective_delta": total_delta,
    "skipped_effective_delta": 0,
    "estimated_probe_cost": round(n_probe * config.PROBE_COST, 6),
    "estimated_total_cost": b1c["costs"]["c13_total"],
    "estimated_normalized_cost": round(c13_ncost, 6),
    # Conservative (trajectory-conditional) — same as observed for REF since skipped_delta=0
    "estimated_comp_conservative": round(c13_comp, 6),
    "estimated_utility_beta0.25_conservative": round(c13_ub25, 6),
    "utility_gain_over_C0b_conservative": round(c13_ub25 - c0b_ub25, 6),
    # Optimistic (full-observe baseline) — same as conservative for REF
    "estimated_comp_optimistic": round(c13_comp, 6),
    "estimated_utility_beta0.25_optimistic": round(c13_ub25, 6),
    "utility_gain_over_C0b_optimistic": round(c13_ub25 - c0b_ub25, 6),
}

# ======== Output ========
print()
print("=" * 70)
print("Block 1E — Pre-Probe Filter Diagnostic")
print("=" * 70)

print(f"\n--- 1. Effective vs Zero-Gain Distribution Comparison ---")
print(f"  n_effective = {n_eff}, n_zero_gain = {n_zero}")
print(f"  {'feature':<32} {'eff_mean':<10} {'zero_mean':<10} "
      f"{'eff_med':<10} {'zero_med':<10} {'eff_min':<10} {'eff_max':<10} "
      f"{'zero_min':<10} {'zero_max':<10}")
for fname, flabel in fields_to_compare:
    eff_vals = [r[fname] for r in effective]
    zero_vals = [r[fname] for r in zero_gain]
    print(f"  {flabel:<32} {_mean(eff_vals):<10.4f} {_mean(zero_vals):<10.4f} "
          f"{_median(eff_vals):<10.4f} {_median(zero_vals):<10.4f} "
          f"{min(eff_vals):<10.4f} {max(eff_vals):<10.4f} "
          f"{min(zero_vals):<10.4f} {max(zero_vals):<10.4f}")

# Per-action effective rate
print(f"\n  Per-action effective rate:")
print(f"  {'action':<22} {'n':<5} {'eff':<5} {'rate':<8}")
for act in sorted(action_rates):
    d = action_rates[act]
    print(f"  {act:<22} {d['n']:<5} {d['eff']:<5} {d['eff']/max(d['n'],1):.3f}")

print(f"\n  High-efficiency actions (>50%): {high_eff_actions if high_eff_actions else 'NONE'}")

# Timing interpretation note
print(f"\n  NOTE — budget_remaining interpretation:")
print(f"    budget_remaining = remaining budget at probe time (not yet deducted for this probe).")
print(f"    effective_mean={_mean([r['current_budget_remaining'] for r in effective]):.4f}")
print(f"    zero_gain_mean={_mean([r['current_budget_remaining'] for r in zero_gain]):.4f}")
print(f"    Lower remaining -> more budget already spent -> probe occurs LATER in episode.")
print(f"    => effective probes tend to occur LATER (more budget consumed),")
print(f"       suggesting IOM calibration improves over time, making later probes higher-value.")

print(f"\n--- 2. Heuristic Filter Tests ---")
print(f"  ALL FILTERS LABELED: posthoc_trajectory_estimate (NOT real policy)")
print(f"  Conservative (primary):  estimated_comp = current_C13_comp - skipped_effective_delta / 300")
print(f"  Optimistic (secondary): estimated_comp = C0b_comp + kept_effective_delta / 300")
print(f"  current_C13_comp = {c13_comp:.4f}, TOTAL_COMPOSITE_DECISIONS = {TOTAL_COMPOSITE_DECISIONS}")
print(f"  C0b: comp={c0b_comp:.4f}  ncost={c0b_ncost:.4f}  ub25={c0b_ub25:.4f}")
print(f"  C13: comp={c13_comp:.4f}  ncost={c13_ncost:.4f}  ub25={c13_ub25:.4f}")
print()
print(f"  {'filter':<42} {'kept':<5} {'skip':<5} {'kE_d':<6} {'sE_d':<6} "
      f"{'e_cost':<8} {'c_comp':<7} {'c_ub25':<8} {'c_vC0b':<8} "
      f"{'o_comp':<7} {'o_ub25':<8}")
print(f"  {'':42} {'':5} {'':5} {'':6} {'':6} "
      f"{'':8} {'(consv)':<7} {'(consv)':<8} {'(consv)':<8} "
      f"{'(optim)':<7} {'(optim)':<8}")
header = (f"  {'filter':<42} {'kept':<5} {'skip':<5} {'kE_d':<6} {'sE_d':<6} "
          f"{'e_cost':<8} {'c_comp':<7} {'c_ub25':<8} {'c_vC0b':<8} "
          f"{'o_comp':<7} {'o_ub25':<8}")
print(header)
print("  " + "-" * len(header))
for f in [current_c13_stats] + filters + [perfect_filter]:
    f["skipped_zero_gain"] = n_zero - f["kept_zero_gain"]
    label = " posthoc" if "posthoc" in f.get("label_as", "") else " REF"
    print(f"  {f['name']:<42} {f['kept_count']:<5} {f['skipped_count']:<5} "
          f"{f['kept_effective_delta']:<6} {f['skipped_effective_delta']:<6} "
          f"{f['estimated_total_cost']:<8.4f} "
          f"{f['estimated_comp_conservative']:<7.4f} "
          f"{f['estimated_utility_beta0.25_conservative']:<8.4f} "
          f"{f['utility_gain_over_C0b_conservative']:<+8.4f} "
          f"{f['estimated_comp_optimistic']:<7.4f} "
          f"{f['estimated_utility_beta0.25_optimistic']:<8.4f}{label}")

# Identify candidate filters (beats C0b and retains >= half of effective probes)
print()
print("--- 3. Candidate Filter Assessment (posthoc_trajectory_estimate) ---")
candidates = []
exploratory_candidates = []
for f in filters:
    f["beats_C0b"] = f["estimated_utility_beta0.25_conservative"] > c0b_ub25
    f["retains_eff_ratio"] = f["kept_effective"] / max(n_eff, 1)
    f["retains_delta_ratio"] = f["kept_effective_delta"] / max(total_delta, 1)
    is_exploratory = "EXPLORATORY" in f["name"]
    if f["beats_C0b"] and f["retains_eff_ratio"] >= 0.5:
        if is_exploratory:
            exploratory_candidates.append(f)
        else:
            candidates.append(f)
        tag = " [EXPLORATORY]" if is_exploratory else ""
        print(f"  CANDIDATE{tag}: {f['name']}  "
              f"(label: posthoc_trajectory_estimate)")
        print(f"    kept_effective_delta={f['kept_effective_delta']}/{total_delta} "
              f"({f['retains_delta_ratio']:.1%}), "
              f"skipped_effective_delta={f['skipped_effective_delta']}/{total_delta}")
        print(f"    keeps {f['kept_effective']}/{n_eff} effective probes "
              f"({f['retains_eff_ratio']:.1%}), "
              f"skips {f['skipped_count']} probes "
              f"({f['skipped_zero_gain']} zero-gain)")
        print(f"    [conservative_trajectory_conditional_estimate]")
        print(f"      comp={f['estimated_comp_conservative']:.4f}, "
              f"cost={f['estimated_total_cost']:.4f}, "
              f"ncost={f['estimated_normalized_cost']:.4f}, "
              f"ub25={f['estimated_utility_beta0.25_conservative']:.4f}")
        print(f"      utility_gain_over_C0b = {f['utility_gain_over_C0b_conservative']:+.4f}")
        print(f"    [optimistic_full_observe_baseline_estimate]")
        print(f"      comp={f['estimated_comp_optimistic']:.4f}, "
              f"ub25={f['estimated_utility_beta0.25_optimistic']:.4f}")

if not candidates and not exploratory_candidates:
    print("  NO CANDIDATE FILTERS FOUND")
    print("  No pre-probe heuristic retains >=50% effective probes while beating C0b.")
    print("  => Zero-gain probes are NOT linearly separable with current pre-probe signals.")
    print("  => May need learned filter (e.g. logistic regression on pre-probe features)")
    print("     or additional features not currently available.")
elif not candidates and exploratory_candidates:
    print("  Only exploratory (action_name) filters pass. Main heuristic filters FAIL.")
    print("  => Current pre-probe signals do NOT separate effective from zero-gain probes.")
    print("  => action_name carries signal, but is environmentally specific and not a")
    print("     generalizable filter.")

# Correlation check: which features correlate with is_effective?
print()
print("--- 4. Feature-Effectiveness Direction ---")
for fname, flabel in fields_to_compare:
    eff_vals = [r[fname] for r in effective]
    zero_vals = [r[fname] for r in zero_gain]
    eff_m = _mean(eff_vals)
    zero_m = _mean(zero_vals)
    direction = "higher" if eff_m > zero_m else "lower" if eff_m < zero_m else "same"
    print(f"  {flabel:<32} eff_mean={eff_m:.4f} zero_mean={zero_m:.4f} "
          f"diff={eff_m-zero_m:+.4f} -> effective tends {direction}")

print()
print("[block_done]")
print(f"  block_id=1E")
print(f"  elapsed={time.time()-t0:.0f}s")
print(f"  n_probe={n_probe}, n_eff={n_eff}, n_zero={n_zero}")
print(f"  n_candidate_filters={len(candidates)}")
print(f"  c0b_ub25={c0b_ub25:.4f}, c13_ub25={c13_ub25:.4f}")

# Save
per_action_features_serializable = {}
for r in probe_records:
    per_action_features_serializable[r["object_id"]] = {
        k: {k2: v2 for k2, v2 in v.items()}  # already serializable
        for k, v in r.get("per_action_features", {}).items()
    }

block_out = {
    "block_id": "1E",
    "elapsed_s": time.time() - t0,
    "cue": "C3_P060_O040", "budget": budget, "cw": cw, "seed": seed,
    "n_probe": n_probe, "n_effective": n_eff, "n_zero_gain": n_zero,
    "total_delta": total_delta,
    "TOTAL_COMPOSITE_DECISIONS": TOTAL_COMPOSITE_DECISIONS,
    "C0b": {"comp": c0b_comp, "ncost": c0b_ncost, "ub25": c0b_ub25},
    "C13": {"comp": c13_comp, "ncost": c13_ncost, "ub25": c13_ub25},
    "comp_formula_conservative": "estimated_comp_conservative = current_C13_comp - skipped_effective_delta / TOTAL_COMPOSITE_DECISIONS",
    "comp_formula_optimistic": "estimated_comp_optimistic = C0b_comp + kept_effective_delta / TOTAL_COMPOSITE_DECISIONS",
    "primary_estimate": "conservative_trajectory_conditional_estimate",
    "distribution_comparison": [
        {"feature": flabel, "field": fname,
         "eff_mean": _mean([r[fname] for r in effective]),
         "zero_mean": _mean([r[fname] for r in zero_gain]),
         "eff_median": _median([r[fname] for r in effective]),
         "zero_median": _median([r[fname] for r in zero_gain]),
         "eff_min": min([r[fname] for r in effective]),
         "eff_max": max([r[fname] for r in effective]),
         "zero_min": min([r[fname] for r in zero_gain]),
         "zero_max": max([r[fname] for r in zero_gain]),
         } for fname, flabel in fields_to_compare
    ],
    "budget_remaining_note": (
        "Lower budget_remaining = more budget spent = probe occurs LATER. "
        "Effective probes have lower remaining, meaning they tend to occur later."
    ),
    "per_action_rates": {act: {"n": d["n"], "eff": d["eff"],
                                "rate": d["eff"]/max(d["n"],1)}
                          for act, d in action_rates.items()},
    "high_eff_actions": sorted(high_eff_actions),
    "heuristic_filters": filters,
    "perfect_filter": perfect_filter,
    "current_c13": current_c13_stats,
    "n_candidate_filters_main": len(candidates),
    "n_candidate_filters_exploratory": len(exploratory_candidates),
    "candidate_filters_main": [c["name"] for c in candidates],
    "candidate_filters_exploratory": [c["name"] for c in exploratory_candidates],
    "probe_records": probe_records,
}
# Remove per_action_features from probe_records in saved output (too large)
slim_records = []
for r in probe_records:
    slim = {k: v for k, v in r.items() if k != "per_action_features"}
    slim_records.append(slim)
block_out["probe_records"] = slim_records

with open("runs/calibration_block1e_preprobe_filter.json", "w") as f:
    json.dump(block_out, f, indent=2)
print(f"  saved: runs/calibration_block1e_preprobe_filter.json")
