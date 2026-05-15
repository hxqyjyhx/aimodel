"""Mini-MC v0 — Main experiment runner."""

import os
import sys
import json
import copy
import random
import time

# Ensure the project root is in the path so `src.minimc` is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.minimc import config
from src.minimc.environment import MiniMCEnvironment
from src.minimc.harness import EpisodeHarness, HarnessResult
from src.minimc.evaluator import MiniMCEvaluator
from tests.test_guardrails import run_all_guardrail_tests

# External dependencies (stubs in src.minimc.deps — provide real implementations)
from src.minimc.deps import (
    InstanceOutcomeMemory,
    run_phase_a_training,
    run_cluster_baselines,
    collect_sparse_probe_outcomes,
)


# =========================================================================
# Helpers
# =========================================================================

def _make_policy(cfg_id, im_base, cp_base, simulator, seed, cost_weight=None):
    """Instantiate a single policy for a given config_id."""
    from src.minimc.policies import (
        C0a_NoInteractionPolicy, C0b_ObserveOnlyPolicy,
        C2_ClusterRefPolicy, C8_ForcedEIGPolicy,
        C13_InstanceVOIPolicy, C13_ForcedVisitProbeGatePolicy,
        C14b_BudgetedOraclePolicy, C14a_TruthAnswerOracle,
        C2_ForcedBudgetPolicy, C8_ForcedBudgetPolicy,
    )
    im = im_base.clone()
    cp = cp_base.clone() if cp_base else None

    if cfg_id == "C0a_no_interaction":
        return C0a_NoInteractionPolicy(im, random.Random(seed))
    elif cfg_id == "C0b_observe_only":
        return C0b_ObserveOnlyPolicy(im, random.Random(seed + 100))
    elif cfg_id == "C2_cost_gate":
        return C2_ClusterRefPolicy(cp, random.Random(seed + 500))
    elif cfg_id == "C2_forced_budget":
        return C2_ForcedBudgetPolicy(cp, random.Random(seed + 510))
    elif cfg_id == "C8_cost_gate":
        return C8_ForcedEIGPolicy(im, random.Random(seed + 200))
    elif cfg_id == "C8_forced_budget":
        return C8_ForcedBudgetPolicy(im, random.Random(seed + 210))
    elif cfg_id == "C13_cost_gate":
        cw = cost_weight if cost_weight is not None else 1.0
        return C13_InstanceVOIPolicy(im, random.Random(seed + 500 + int(cw * 100)),
                                     cost_weight=cw)
    elif cfg_id == "C13_forced_visit_probe_gate":
        cw = cost_weight if cost_weight is not None else 0.5
        return C13_ForcedVisitProbeGatePolicy(
            im, random.Random(seed + 600 + int(cw * 100)), cost_weight=cw)
    elif cfg_id == "C14b_budgeted_oracle_probe":
        return C14b_BudgetedOraclePolicy(im, random.Random(seed + 700), simulator)
    elif cfg_id == "C14a_truth_answer_oracle":
        return C14a_TruthAnswerOracle(simulator)
    else:
        # Handle C13 variants with CW in name
        if cfg_id.startswith("C13_cost_gate_CW"):
            cw_str = cfg_id.split("CW")[-1]
            cw = {"00": 0.0, "05": 0.5, "10": 1.0, "15": 1.5, "20": 2.0}[cw_str]
            return C13_InstanceVOIPolicy(
                im, random.Random(seed + 500 + int(cw * 100)), cost_weight=cw)
        raise ValueError(f"Unknown config_id: {cfg_id}")


def _run_episode(env, policy):
    """Run an episode through the harness. Handles C14a specially."""
    from src.minimc.policies import C14a_TruthAnswerOracle
    if isinstance(policy, C14a_TruthAnswerOracle):
        predictions = policy.get_answer()
        obs = env.reset()
        obs.event_log.append({"event": "C14a_truth_oracle",
                              "note": "no budget, no harness"})
        return HarnessResult(predictions, obs.event_log, obs, {}, {})
    harness = EpisodeHarness(env, policy)
    return harness.run()


# =========================================================================
# Single-seed run
# =========================================================================

def run_single_seed(seed):
    """Run full Mini-MC v0 pipeline for one seed."""
    print(f"\n{'='*60}")
    print(f"Mini-MC v0 — seed={seed}")
    print(f"{'='*60}")

    cond = copy.deepcopy(config.FIXED_CONDITION)

    # Phase A: Training
    print("  Phase A: Training...")
    (student, base_learner, train_objects, train_env,
     test_objects, test_env, final_metrics, rng) = run_phase_a_training(seed, cond)

    train_objects_dict = {oid: train_env.objects[oid] for oid in train_objects}
    test_objects_dict = {oid: test_env.objects[oid] for oid in test_objects}

    # Phase A': Sparse outcome collection
    print(f"  Phase A': Collecting sparse outcomes at coverage={config.COVERAGE}...")
    outcome_rows, actual_coverage = collect_sparse_probe_outcomes(
        train_objects_dict, student, config.COVERAGE, seed
    )
    print(f"    actual_coverage={actual_coverage:.4f}")

    # Phase B: Build InstanceOutcomeMemory
    print("  Phase B: Building InstanceOutcomeMemory...")
    posterior_visible_features = list(base_learner.visible_feature_names)
    im_base = InstanceOutcomeMemory(
        train_objects_dict, posterior_visible_features,
        k=config.INSTANCE_K, similarity_power=config.SIMILARITY_POWER,
        similarity_mode=config.SIMILARITY_MODE,
    )
    im_base.build(outcome_rows)

    # Phase C: Build cluster posterior for C2
    print("  Phase C: Building cluster posterior...")
    cluster_results = run_cluster_baselines(
        train_objects_dict, student, train_env, base_learner,
        config.COVERAGE, seed
    )
    c2_posterior = cluster_results["C2"]
    cluster_quality = cluster_results.get("cluster_quality", {})

    test_oids = sorted(test_objects_dict.keys())

    # Phase D: Assign positions
    print("  Phase D: Assigning positions...")
    from src.minimc.simulator_truth import MiniMCSimulatorTruth
    positions = MiniMCSimulatorTruth.assign_positions(
        test_oids, config.GRID_ROWS, config.GRID_COLS,
        config.AGENT_START, rng
    )

    # Phase E: Run each policy
    print("  Phase E: Running policies...")
    results = {}

    main_env = MiniMCEnvironment(
        test_objects_dict, positions,
        initial_budget=config.DEFAULT_INITIAL_BUDGET,
    )
    oracle_env = MiniMCEnvironment(
        test_objects_dict, positions,
        initial_budget=config.DEFAULT_INITIAL_BUDGET,
    )
    evaluator_main = MiniMCEvaluator(main_env)
    evaluator_oracle = MiniMCEvaluator(oracle_env)

    # C14a (truth answer) — special case, uses simulator directly
    c14a_sim = main_env.simulator
    c14a_policy = _make_policy("C14a_truth_answer_oracle", im_base, None,
                               c14a_sim, seed)
    c14a_preds = c14a_policy.get_answer()
    # Use main evaluator for C14a
    from src.minimc.agent_obs import AgentObs
    from src.minimc.event_log import EventLog
    dummy_log = EventLog()
    dummy_obs = AgentObs(c14a_sim, config.DEFAULT_INITIAL_BUDGET,
                         config.REACH_COST_PER_UNIT, config.OBSERVE_COST,
                         config.PROBE_COST, dummy_log)
    c14a_metrics = evaluator_main.evaluate(c14a_preds, dummy_log, dummy_obs)
    results["C14a_truth_answer_oracle"] = {
        "metrics": c14a_metrics,
        "predictions": c14a_preds,
        "event_log": [],
    }
    print(f"    [C14a_truth_answer_oracle] "
          f"comp_acc={c14a_metrics['composite_task_accuracy']:.4f} "
          f"bal_aff={c14a_metrics['balanced_affordance_accuracy']:.4f}")

    for config_id in config.POLICY_CONFIGS:
        is_oracle = config_id in config.ORACLE_CONFIGS
        env = oracle_env if is_oracle else main_env
        evaluator = evaluator_oracle if is_oracle else evaluator_main

        env_copy = MiniMCEnvironment(
            env.simulator._objects, positions,
            agent_start=env.simulator.get_agent_start(),
            initial_budget=config.DEFAULT_INITIAL_BUDGET,
        )

        is_c13 = config_id.startswith("C13_cost_gate_CW")
        if is_c13:
            cw_str = config_id.split("CW")[-1]
            cw = {"00": 0.0, "05": 0.5, "10": 1.0, "15": 1.5, "20": 2.0}[cw_str]
        else:
            cw = None

        policy = _make_policy(config_id, im_base, c2_posterior,
                              env_copy.simulator, seed, cost_weight=cw)

        print(f"    [{config_id}] running episode...")
        try:
            result = _run_episode(env_copy, policy)
            metrics = evaluator.evaluate(
                result.predictions, result.event_log, result.agent_obs,
                pre_probe_entropies=result.pre_probe_entropies,
                pre_decision_entropies=result.pre_decision_entropies,
            )
            metrics["config_id"] = config_id
            results[config_id] = {
                "metrics": metrics,
                "predictions": result.predictions,
                "event_log": result.event_log.to_list(),
            }
            print(f"      comp_acc={metrics['composite_task_accuracy']:.4f} "
                  f"bal_aff={metrics['balanced_affordance_accuracy']:.4f} "
                  f"visited={metrics['objects_visited']} "
                  f"probed={metrics['objects_probed']} "
                  f"cost={metrics['total_cost']:.4f} "
                  f"ent_gap={metrics['entropy_gap']:+.4f}")
        except Exception as e:
            print(f"    [{config_id}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[config_id] = {"error": str(e)}

    return {
        "seed": seed,
        "actual_coverage": actual_coverage,
        "cluster_quality": cluster_quality,
        "positions": {oid: list(pos) for oid, pos in positions.items()},
        "results": results,
        "pre_accuracy": final_metrics.get("domain_token_accuracy", 0.0),
    }


# =========================================================================
# Calibration
# =========================================================================

def run_calibration_grid(cue_filter=None, budget_filter=None, cw_filter=None):
    """Grid calibration: 3 cues x 4 budgets x (fixed configs + C13 CW sweep).

    Optional filters for block-based running:
      cue_filter: set of cue labels (e.g. {'C3_P060_O040'})
      budget_filter: set of budget values (e.g. {1.5, 2.0})
      cw_filter: set of cost weight values (e.g. {0.5, 1.0})
    """
    print("=" * 70)
    print("CALIBRATION GRID")
    print("=" * 70)

    seed = 101
    all_cue_results = {}
    t_start = time.time()

    from src.minimc.simulator_truth import MiniMCSimulatorTruth

    # Filter cues
    cues_to_run = [c for c in config.CUE_CONDITIONS
                   if cue_filter is None or c["label"] in cue_filter]

    for cue_cond in cues_to_run:
        cue_label = cue_cond["label"]
        print(f"\n{'='*60}")
        print(f"Cue: {cue_label}  (p_target={cue_cond['p_target']}, "
              f"p_other={cue_cond['p_other']}, absent={cue_cond.get('absent', False)})")
        print(f"{'='*60}")

        cond = copy.deepcopy(cue_cond)

        # Phase A: Training
        print("  Phase A: Training...")
        try:
            (student, base_learner, train_objects, train_env,
             test_objects, test_env, final_metrics, rng) = run_phase_a_training(seed, cond)
        except Exception as e:
            print(f"  ERROR in Phase A: {e}")
            import traceback
            traceback.print_exc()
            all_cue_results[cue_label] = {"error": str(e)}
            continue

        train_objects_dict = {oid: train_env.objects[oid] for oid in train_objects}
        test_oids = sorted(test_objects)
        test_objects_dict = {oid: test_env.objects[oid] for oid in test_oids}

        print(f"    train={len(train_objects_dict)}, test={len(test_objects_dict)}")
        print(f"    pre_accuracy={final_metrics.get('domain_token_accuracy', 0.0):.4f}")

        # Phase A': Sparse outcomes
        outcome_rows, actual_coverage = collect_sparse_probe_outcomes(
            train_objects_dict, student, config.COVERAGE, seed
        )

        # Phase B: IOM
        posterior_visible_features = list(base_learner.visible_feature_names)
        im_base = InstanceOutcomeMemory(
            train_objects_dict, posterior_visible_features,
            k=config.INSTANCE_K, similarity_power=config.SIMILARITY_POWER,
            similarity_mode=config.SIMILARITY_MODE,
        )
        im_base.build(outcome_rows)

        # Phase C: Cluster posterior
        cluster_results = run_cluster_baselines(
            train_objects_dict, student, train_env, base_learner,
            config.COVERAGE, seed
        )
        c2_posterior = cluster_results.get("C2")

        # Phase D: Positions
        positions = MiniMCSimulatorTruth.assign_positions(
            test_oids, config.GRID_ROWS, config.GRID_COLS,
            config.AGENT_START, rng
        )

        # ---- Run calibration episodes ----
        cue_budget_results = {}

        # Compute total configs for progress tracking
        budgets_to_run = [b for b in config.CALIBRATION_BUDGETS
                          if budget_filter is None or b in budget_filter]
        cws_to_run = [cw for cw in config.CALIBRATION_CW_SWEEP
                      if cw_filter is None or cw in cw_filter]
        n_fixed = len(config.CALIBRATION_FIXED_CONFIGS)
        n_c13_per_budget = len(cws_to_run) * 2  # cost_gate + forced_visit
        n_per_budget = n_fixed + n_c13_per_budget
        total_configs = len(budgets_to_run) * n_per_budget
        config_idx = 0
        cue_t0 = time.time()

        for budget in budgets_to_run:
            print(f"\n  --- Budget={budget} ---")
            row = {}

            # Run fixed (non-C13) configs
            for cal_cfg in config.CALIBRATION_FIXED_CONFIGS:
                config_idx += 1
                t0 = time.time()
                is_c14b = (cal_cfg == "C14b_budgeted_oracle_probe")
                is_c14a = (cal_cfg == "C14a_truth_answer_oracle")

                env = MiniMCEnvironment(
                    test_objects_dict, positions, initial_budget=budget,
                )

                try:
                    policy = _make_policy(cal_cfg, im_base, c2_posterior,
                                          env.simulator, seed)
                    result = _run_episode(env, policy)
                    evaluator = MiniMCEvaluator(env)
                    metrics = evaluator.evaluate(
                        result.predictions, result.event_log, result.agent_obs,
                        pre_probe_entropies=result.pre_probe_entropies,
                        pre_decision_entropies=result.pre_decision_entropies,
                    )
                    row[cal_cfg] = metrics
                    n_obj = metrics.get("n_objects", 1)
                    fbal = metrics.get('fixed_balanced_affordance_accuracy',
                                       metrics.get('balanced_affordance_accuracy', 0))
                    elapsed = time.time() - t0
                    total_elapsed = time.time() - cue_t0
                    avg = total_elapsed / config_idx
                    eta = avg * (total_configs - config_idx)
                    print(f"    [progress] cue={cue_label} budget={budget} "
                          f"config={config_idx}/{total_configs} "
                          f"elapsed={elapsed:.0f}s avg={avg:.0f}s eta={eta:.0f}s")
                    print(f"    [{cal_cfg:<35}] "
                          f"comp={metrics.get('composite_task_accuracy', 0):.4f} "
                          f"fbal={fbal:.4f} "
                          f"v={metrics.get('objects_visited', 0)}/{n_obj} "
                          f"p={metrics.get('probe_coverage', 0):.2%} "
                          f"ncost={metrics.get('normalized_cost', 0):.4f} "
                          f"ub25={metrics.get('utility_betas', {}).get('utility_beta0.25', 0):+.4f} "
                          f"ent={metrics.get('entropy_gap', 0):+.4f}")
                    # For C14a, also print feature class coverage
                    if cal_cfg == "C14a_truth_answer_oracle":
                        cov = metrics.get('per_feature_class_coverage', {})
                        single_class = [f for f, v in cov.items() if not v.get('has_both', True)]
                        if single_class:
                            print(f"      [!] C14a single-class features: {single_class}")
                            for f in single_class:
                                print(f"          {f}: n_succ={cov[f]['n_success']} n_fail={cov[f]['n_fail']}")
                except Exception as e:
                    print(f"    [{cal_cfg}] ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    row[cal_cfg] = {"error": str(e)}

            # Run C13 cost gate and forced-visit for each cost_weight
            for cw in cws_to_run:
                cw_label = config.COST_WEIGHT_LABELS[cw]
                for c13_base in ["C13_cost_gate", "C13_forced_visit_probe_gate"]:
                    cal_cfg = f"{c13_base}_CW{cw_label.replace('CW', '')}"
                    config_idx += 1
                    t0 = time.time()

                    env = MiniMCEnvironment(
                        test_objects_dict, positions, initial_budget=budget,
                    )

                    try:
                        policy = _make_policy(c13_base, im_base, None,
                                              env.simulator, seed, cost_weight=cw)
                        result = _run_episode(env, policy)
                        evaluator = MiniMCEvaluator(env)
                        metrics = evaluator.evaluate(
                            result.predictions, result.event_log, result.agent_obs,
                            pre_probe_entropies=result.pre_probe_entropies,
                            pre_decision_entropies=result.pre_decision_entropies,
                        )
                        row[cal_cfg] = metrics
                        n_obj = metrics.get("n_objects", 1)
                        fbal = metrics.get('fixed_balanced_affordance_accuracy',
                                           metrics.get('balanced_affordance_accuracy', 0))
                        elapsed = time.time() - t0
                        total_elapsed = time.time() - cue_t0
                        avg = total_elapsed / config_idx
                        eta = avg * (total_configs - config_idx)
                        print(f"    [progress] cue={cue_label} budget={budget} "
                              f"cw={cw} config={config_idx}/{total_configs} "
                              f"elapsed={elapsed:.0f}s avg={avg:.0f}s eta={eta:.0f}s")
                        print(f"    [{cal_cfg:<35}] "
                              f"comp={metrics.get('composite_task_accuracy', 0):.4f} "
                              f"fbal={fbal:.4f} "
                              f"v={metrics.get('objects_visited', 0)}/{n_obj} "
                              f"p={metrics.get('probe_coverage', 0):.2%} "
                              f"ncost={metrics.get('normalized_cost', 0):.4f} "
                              f"ub25={metrics.get('utility_betas', {}).get('utility_beta0.25', 0):+.4f} "
                              f"ent={metrics.get('entropy_gap', 0):+.4f}")
                    except Exception as e:
                        print(f"    [{cal_cfg}] ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                        row[cal_cfg] = {"error": str(e)}

            cue_budget_results[budget] = row

        all_cue_results[cue_label] = {
            "pre_accuracy": final_metrics.get("domain_token_accuracy", 0.0),
            "actual_coverage": actual_coverage,
            "budgets": cue_budget_results,
        }

    # ---- Evaluate calibration targets ----
    print("\n" + "=" * 70)
    print("CALIBRATION TARGET CHECK")
    print("=" * 70)

    passing = []
    priority_passing = []
    all_rows = []

    # ---- C14b ceiling diagnostics per (cue, budget) ----
    # Use filtered cues/budgets consistent with block run
    cues_for_check = [c["label"] for c in
                      ([c2 for c2 in config.CUE_CONDITIONS
                        if cue_filter is None or c2["label"] in cue_filter])]
    budgets_for_check = [b for b in config.CALIBRATION_BUDGETS
                         if budget_filter is None or b in budget_filter]
    cws_for_check = [cw for cw in config.CALIBRATION_CW_SWEEP
                     if cw_filter is None or cw in cw_filter]

    c14b_ceilings = {}  # key=(cue_label, budget) -> dict
    for cue_label in cues_for_check:
        cue_data = all_cue_results.get(cue_label, {})
        if "error" in cue_data:
            continue
        for budget in budgets_for_check:
            row = cue_data["budgets"].get(budget, {})
            c0a = row.get("C0a_no_interaction", {})
            c0a_comp = c0a.get("composite_task_accuracy", 0.70)
            c14b = row.get("C14b_budgeted_oracle_probe", {})
            if not c14b or "error" in c14b:
                continue
            c14b_pcov = c14b.get("probe_coverage", 0.0)
            c14b_comp = c14b.get("composite_task_accuracy", 0.0)
            # Rough ceiling: probed objects perfect, unvisited at C0a (zero-info) comp
            rough_budget_coverage_ceiling = (c14b_pcov * 1.0
                                              + (1.0 - c14b_pcov) * c0a_comp)
            c14b_ceiling_ratio = c14b_comp / max(rough_budget_coverage_ceiling, 0.001)
            c14b_ceilings[(cue_label, budget)] = {
                "budget": budget,
                "c14b_pcov": c14b_pcov,
                "rough_budget_coverage_ceiling": rough_budget_coverage_ceiling,
                "c14b_comp": c14b_comp,
                "c14b_ceiling_ratio": c14b_ceiling_ratio,
            }

    print("\n  C14b Ceiling Diagnostics:")
    print(f"  {'Cue':<18} {'B':<6} {'C14b_comp':<10} {'C14b_pcov':<10} "
          f"{'rough_ceil':<12} {'ceil_ratio':<10}")
    print("  " + "-" * 67)
    for (cue_label, budget), diag in sorted(c14b_ceilings.items()):
        print(f"  {cue_label:<18} {budget:<6.1f} {diag['c14b_comp']:<10.4f} "
              f"{diag['c14b_pcov']:<10.2%} "
              f"{diag['rough_budget_coverage_ceiling']:<12.4f} "
              f"{diag['c14b_ceiling_ratio']:<10.4f}")

    # ---- Per-row target check (two-pass: collect, then finalize cross-CW) ----
    # Pass 1: collect all rows
    PRIORITY_BUDGETS = {1.5, 2.0}
    PRIORITY_CW = {0.5, 1.0}

    for cue_label in cues_for_check:
        cue_data = all_cue_results.get(cue_label, {})
        if "error" in cue_data:
            continue

        for budget in budgets_for_check:
            row = cue_data["budgets"].get(budget, {})

            c0b = row.get("C0b_observe_only", {})
            c0b_comp = c0b.get("composite_task_accuracy", 1.0)
            c0b_bal = c0b.get("balanced_affordance_accuracy", 1.0)
            c0b_ub25 = c0b.get("utility_betas", {}).get("utility_beta0.25", 0.0)
            c0b_ncost = c0b.get("normalized_cost", 0.0)

            c14a = row.get("C14a_truth_answer_oracle", {})
            c14a_comp = c14a.get("composite_task_accuracy", 0.0)
            c14a_fbal = c14a.get("fixed_balanced_affordance_accuracy", 0.0)

            c14b = row.get("C14b_budgeted_oracle_probe", {})
            c14b_comp = c14b.get("composite_task_accuracy", 0.0)
            c14b_pcov = c14b.get("probe_coverage", 0.0)

            for cw in cws_for_check:
                cw_label = config.COST_WEIGHT_LABELS[cw]
                c13_key = f"C13_cost_gate_CW{cw_label.replace('CW', '')}"
                c13 = row.get(c13_key, {})
                if not c13 or "error" in c13:
                    continue

                c13_comp = c13.get("composite_task_accuracy", 0.0)
                c13_bal = c13.get("fixed_balanced_affordance_accuracy", 0.0)
                c13_cov = c13.get("probe_coverage", 1.0)
                c13_ent = c13.get("entropy_gap", -999.0)
                c13_ub25 = c13.get("utility_betas", {}).get("utility_beta0.25", -999.0)
                c13_ncost = c13.get("normalized_cost", 0.0)
                c13_gt_c0b_comp = c13_comp > c0b_comp
                c13_gt_c0b_bal = c13_bal > c0b_bal

                comp_gain = c13_comp - c0b_comp
                utility_gain = c13_ub25 - c0b_ub25
                cost_reduction = c0b_ncost - c13_ncost

                is_priority = budget in PRIORITY_BUDGETS and cw in PRIORITY_CW

                row_data = {
                    "cue_label": cue_label,
                    "cue_cond": config.CUE_CONDITIONS[
                        [c["label"] for c in config.CUE_CONDITIONS].index(cue_label)
                    ],
                    "budget": budget,
                    "cost_weight": cw,
                    "is_priority": is_priority,
                    "c0b_comp": c0b_comp, "c0b_bal": c0b_bal,
                    "c0b_ub25": c0b_ub25, "c0b_ncost": c0b_ncost,
                    "c13_comp": c13_comp, "c13_bal": c13_bal,
                    "c13_cov": c13_cov, "c13_ent": c13_ent,
                    "c13_ub25": c13_ub25, "c13_ncost": c13_ncost,
                    "c13_gt_c0b_comp": c13_gt_c0b_comp,
                    "c13_gt_c0b_bal": c13_gt_c0b_bal,
                    "c14a_comp": c14a_comp, "c14a_fbal": c14a_fbal,
                    "c14b_comp": c14b_comp, "c14b_pcov": c14b_pcov,
                    "comp_gain": comp_gain,
                    "utility_gain": utility_gain,
                    "cost_reduction": cost_reduction,
                    # C14b_rel_ok filled in pass 2
                }
                all_rows.append(row_data)

    # Pass 2: compute best_C13_comp per (cue, budget) across all CW, then finalize
    c13_best_by_cue_budget = {}
    cw_count_by_cue_budget = {}  # count distinct CWs to determine scope
    for r in all_rows:
        key = (r["cue_label"], r["budget"])
        if key not in c13_best_by_cue_budget:
            c13_best_by_cue_budget[key] = r["c13_comp"]
            cw_count_by_cue_budget[key] = {r["cost_weight"]}
        else:
            c13_best_by_cue_budget[key] = max(c13_best_by_cue_budget[key], r["c13_comp"])
            cw_count_by_cue_budget[key].add(r["cost_weight"])

    for r in all_rows:
        key = (r["cue_label"], r["budget"])
        best_c13 = c13_best_by_cue_budget.get(key, r["c13_comp"])
        n_cws = len(cw_count_by_cue_budget.get(key, {r["cost_weight"]}))
        c14b_rel_ok = r["c14b_comp"] >= best_c13 + 0.01

        # Determine C14b_relative_scope
        if n_cws <= 1:
            c14b_rel_scope = "single_cw_provisional"
        else:
            c14b_rel_scope = "budget_all_included_cw"

        # Attach ceiling diagnostics from c14b_ceilings
        ceiling_diag = c14b_ceilings.get(key, {})
        r["best_C13_comp_for_budget"] = best_c13
        r["C14b_relative_ok"] = c14b_rel_ok
        r["C14b_relative_scope"] = c14b_rel_scope
        r["rough_budget_coverage_ceiling"] = ceiling_diag.get(
            "rough_budget_coverage_ceiling", 0.0)
        r["C14b_ceiling_ratio"] = ceiling_diag.get("c14b_ceiling_ratio", 0.0)

        r["checks"] = {
            "C0b_low": (r["c0b_comp"] <= 0.65 or r["c0b_bal"] <= 0.60),
            "C14a_ok": (r["c14a_comp"] >= 0.98 and r["c14a_fbal"] >= 0.98),
            "C14b_rel_ok": c14b_rel_ok,
            "C13_pcov": 0 < r["c13_cov"] <= 0.40,
            "C13_better": r["c13_gt_c0b_comp"] or r["c13_gt_c0b_bal"],
            "C13_util_gt_c0b": r["c13_ub25"] > r["c0b_ub25"],
            "C13_ent": r["c13_ent"] > 0,
        }
        fail_reasons = [k for k, v in r["checks"].items() if not v]
        all_pass = len(fail_reasons) == 0
        r["_all_pass"] = all_pass
        r["_fail_reasons"] = fail_reasons
        r["_n_fail"] = len(fail_reasons)
        r["_verdict"] = "PASS" if all_pass else f"FAIL({len(fail_reasons)})"
        if all_pass and r["is_priority"]:
            r["_verdict"] = "PASS*"

    # ---- Print per-row table ----
    header = (f"{'Cue':<18} {'B':<5} {'CW':<5} "
              f"{'C0b_comp':<10} "
              f"{'C13_comp':<10} {'C13_cov':<10} "
              f"{'C14a_c':<8} {'C14b_c':<10} "
              f"{'bestC13':<10} {'C14b_rel':<8} {'scope':<24} "
              f"{'ub25':<9} {'EntGap':<9} "
              f"{'dComp':<7} {'dUtil':<7} {'dCost':<7} "
              f"{'rCeil':<8} {'cRatio':<7} "
              f"{'Pri':<4} {'Verdict'}")
    print("  " + header)
    print("  " + "-" * len(header))

    for r in all_rows:
        priority_flag = "*" if r["is_priority"] else " "
        print(f"  {r['cue_label']:<18} {r['budget']:<5.1f} {r['cost_weight']:<5.1f} "
              f"{r['c0b_comp']:<10.4f} "
              f"{r['c13_comp']:<10.4f} {r['c13_cov']:<10.2%} "
              f"{r['c14a_comp']:<8.4f} {r['c14b_comp']:<10.4f} "
              f"{r['best_C13_comp_for_budget']:<10.4f} "
              f"{'OK' if r['C14b_relative_ok'] else 'FAIL':<8} "
              f"{r['C14b_relative_scope']:<24} "
              f"{r['c13_ub25']:<+9.4f} {r['c13_ent']:<+9.4f} "
              f"{r['comp_gain']:<+7.4f} {r['utility_gain']:<+7.4f} "
              f"{r['cost_reduction']:<+7.4f} "
              f"{r['rough_budget_coverage_ceiling']:<8.4f} "
              f"{r['C14b_ceiling_ratio']:<7.4f} "
              f"{priority_flag:<4} {r['_verdict']}")

    passing = [r for r in all_rows if r["_all_pass"]]
    priority_passing = [r for r in passing if r["is_priority"]]

    print()

    # ---- C14b relative check summary ----
    print("  C14b relative check (C14b_comp >= best_C13_comp + 0.01):")
    for (cue_label, budget), best_c13 in sorted(c13_best_by_cue_budget.items()):
        diag = c14b_ceilings.get((cue_label, budget), {})
        c14b_c = diag.get("c14b_comp", 0.0)
        rel_ok = c14b_c >= best_c13 + 0.01
        print(f"    {cue_label} B={budget:.1f}: best_C13={best_c13:.4f}  "
              f"C14b={c14b_c:.4f}  delta={c14b_c - best_c13:+.4f}  "
              f"{'OK' if rel_ok else 'FAIL'}")
    print()

    if passing:
        print(f"  PASSING: {len(passing)} combinations "
              f"({len(priority_passing)} priority)")
        for p in priority_passing if priority_passing else passing:
            flags = ""
            if p["is_priority"]:
                flags += " [PRIORITY]"
            print(f"  {p['cue_label']} B={p['budget']:.1f} CW={p['cost_weight']:.1f} "
                  f"C0b_c={p['c0b_comp']:.4f} C13_c={p['c13_comp']:.4f} "
                  f"C13_cov={p['c13_cov']:.2%} C14a={p['c14a_comp']:.4f} "
                  f"ub25={p['c13_ub25']:+.4f} ent={p['c13_ent']:+.4f}"
                  f"{flags}")
        # Select best: prioritize priority, then lowest probe coverage, then highest C13 comp
        candidates = priority_passing if priority_passing else passing
        best = min(candidates, key=lambda p: (
            not p["is_priority"],  # priority first
            p["c13_cov"],          # lower probe coverage preferred
            -p["c13_comp"],        # higher comp accuracy preferred
            -p["c13_ub25"],        # higher utility preferred
        ))
        print(f"\n  SELECTED: Cue={best['cue_label']}, Budget={best['budget']:.1f}, "
              f"CW={best['cost_weight']:.1f}")
        return all_cue_results, best

    # No passing combination — diagnostic
    print("NO COMBINATION PASSES ALL 7 TARGETS.")
    print()

    # Identify structurally impossible criteria
    criteria_names = ["C0b_low", "C14a_ok", "C14b_rel_ok", "C13_pcov",
                      "C13_better", "C13_util_gt_c0b", "C13_ent"]
    criteria_stats = {k: {"pass": 0, "fail": 0} for k in criteria_names}
    for r in all_rows:
        for k in criteria_names:
            if r["checks"].get(k, False):
                criteria_stats[k]["pass"] += 1
            else:
                criteria_stats[k]["fail"] += 1

    print("Per-criterion pass rate:")
    for k, stats in criteria_stats.items():
        total = stats["pass"] + stats["fail"]
        pct = stats["pass"] / max(total, 1) * 100
        print(f"  {k}: {stats['pass']}/{total} ({pct:.0f}%)")

    print()
    for k, stats in criteria_stats.items():
        if stats["pass"] == 0:
            print(f"  STRUCTURAL: '{k}' never passes — may be impossible with current design.")

    return all_cue_results, None


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 70)
    print("Mini-MC v0 — Cost-Structured Instance+VOI Environment")
    print("=" * 70)

    # ---- Step 0: Calibration grid ----
    cal_results, best = run_calibration_grid()

    if best is None:
        print("\n" + "=" * 70)
        print("CALIBRATION FAILED — No setting meets all targets.")
        print("Review per-criterion diagnostics above and revise task design.")
        print("=" * 70)
        # Save calibration results for diagnosis
        results_dir = os.path.join(CURRENT_DIR, "results_004_5o")
        os.makedirs(results_dir, exist_ok=True)
        _save_calibration(cal_results, best, results_dir)
        return None, None, cal_results

    # Apply selected settings
    selected_cue = best["cue_cond"]
    selected_budget = best["budget"]
    selected_cw = best["cost_weight"]
    config.FIXED_CONDITION = selected_cue
    config.DEFAULT_INITIAL_BUDGET = selected_budget

    print("\n" + "=" * 70)
    print(f"SELECTED: Cue={selected_cue['label']}, Budget={selected_budget:.1f}, "
          f"CW={selected_cw:.1f}")
    print("=" * 70)

    # ---- Step 1: Guardrail tests ----
    print("\n" + "=" * 70)
    print("GUARDRAIL TESTS")
    print("=" * 70)

    seed_gr = 101
    cond = copy.deepcopy(selected_cue)

    (student, base_learner, train_objects, train_env,
     test_objects, test_env, final_metrics, rng) = run_phase_a_training(seed_gr, cond)

    train_objects_dict = {oid: train_env.objects[oid] for oid in train_objects}
    test_oids = sorted(test_objects)
    test_objects_dict = {oid: test_env.objects[oid] for oid in test_oids}

    outcome_rows, actual_coverage = collect_sparse_probe_outcomes(
        train_objects_dict, student, config.COVERAGE, seed_gr
    )

    posterior_visible_features = list(base_learner.visible_feature_names)
    im_base = InstanceOutcomeMemory(
        train_objects_dict, posterior_visible_features,
        k=config.INSTANCE_K, similarity_power=config.SIMILARITY_POWER,
        similarity_mode=config.SIMILARITY_MODE,
    )
    im_base.build(outcome_rows)

    cluster_results = run_cluster_baselines(
        train_objects_dict, student, train_env, base_learner,
        config.COVERAGE, seed_gr
    )
    c2_posterior = cluster_results.get("C2")

    from src.minimc.simulator_truth import MiniMCSimulatorTruth
    positions = MiniMCSimulatorTruth.assign_positions(
        test_oids, config.GRID_ROWS, config.GRID_COLS,
        config.AGENT_START, rng
    )

    gr_env = MiniMCEnvironment(
        test_objects_dict, positions,
        initial_budget=selected_budget,
    )

    guardrail_results = run_all_guardrail_tests(
        gr_env, test_objects_dict, positions, im_base,
        c2_posterior, gr_env.simulator
    )

    all_passed = True
    for name, (passed, msg) in sorted(guardrail_results.items()):
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name}: {msg}")

    if not all_passed:
        print("\n  WARNING: Some guardrail tests failed. Review before proceeding.")

    # ---- Step 2: Full sweep ----
    print("\n" + "=" * 70)
    print("FULL SWEEP")
    print("=" * 70)

    all_seed_results = {}
    for seed in config.SEEDS:
        result = run_single_seed(seed)
        all_seed_results[seed] = result

    # ---- Save results ----
    results_dir = os.path.join(CURRENT_DIR, "results_004_5o")
    os.makedirs(results_dir, exist_ok=True)
    _save_results(all_seed_results, cal_results, best, guardrail_results, results_dir)
    _save_calibration(cal_results, best, results_dir)

    print("\nDone.")
    return all_seed_results, guardrail_results, cal_results


def _save_results(all_seed_results, cal_results, best, guardrail_results, results_dir):
    """Save main results, event logs, and guardrail results."""
    # Slim results
    slim_results = {}
    for seed, data in all_seed_results.items():
        slim_configs = {}
        for cfg_id, cfg_data in data.get("results", {}).items():
            slim_configs[cfg_id] = {
                "metrics": cfg_data.get("metrics", {}),
                "event_count": len(cfg_data.get("event_log", [])),
            }
        slim_results[seed] = {
            key: val for key, val in data.items() if key != "results"
        }
        slim_results[seed]["results"] = slim_configs

    results_path = os.path.join(results_dir, "results_004_5o.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(slim_results, f, indent=2, default=str)
    print(f"Saved: {results_path}")

    # Event logs
    logs_dir = os.path.join(results_dir, "event_logs")
    os.makedirs(logs_dir, exist_ok=True)
    for seed, data in all_seed_results.items():
        for cfg_id, cfg_data in data.get("results", {}).items():
            log_path = os.path.join(logs_dir, f"seed{seed}_{cfg_id}.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(cfg_data.get("event_log", []), f, indent=2)

    # Guardrail results
    gr_path = os.path.join(results_dir, "guardrail_results.json")
    gr_serializable = {}
    for name, (passed, msg) in guardrail_results.items():
        gr_serializable[name] = {"passed": passed, "message": msg}
    with open(gr_path, "w", encoding="utf-8") as f:
        json.dump(gr_serializable, f, indent=2)
    print(f"Saved: {gr_path}")


def _save_calibration(cal_results, best, results_dir):
    """Save calibration grid results."""
    cal_serializable = {}
    for cue_label, cue_data in cal_results.items():
        if "error" in cue_data:
            cal_serializable[cue_label] = {"error": cue_data["error"]}
        else:
            cal_serializable[cue_label] = {
                "pre_accuracy": cue_data.get("pre_accuracy", 0.0),
                "actual_coverage": cue_data.get("actual_coverage", 0.0),
                "budgets": {},
            }
            for budget, row in cue_data.get("budgets", {}).items():
                cal_serializable[cue_label]["budgets"][str(budget)] = {
                    k: v for k, v in row.items()
                    if isinstance(v, dict) and "error" not in v
                }
    if best:
        cal_serializable["_selected"] = {
            "cue": best["cue_label"], "budget": best["budget"],
            "cost_weight": best["cost_weight"],
        }

    cal_path = os.path.join(results_dir, "calibration.json")
    with open(cal_path, "w", encoding="utf-8") as f:
        json.dump(cal_serializable, f, indent=2)
    print(f"Saved: {cal_path}")


if __name__ == "__main__":
    main()
