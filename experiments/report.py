"""Mini-MC v0 — Report generator."""

import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.minimc import config


def generate_report(results_path, calibration_path, guardrail_path, output_path):
    with open(results_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)
    with open(calibration_path, "r", encoding="utf-8") as f:
        cal_results = json.load(f)
    with open(guardrail_path, "r", encoding="utf-8") as f:
        guardrail_results = json.load(f)

    lines = []

    def w(text=""):
        lines.append(text)

    w("# Mini-MC v0 Report")
    w()
    w("## Experiment Configuration")
    w()
    w(f"- Grid: {config.GRID_ROWS}x{config.GRID_COLS}, Agent start: {config.AGENT_START}")
    w(f"- Costs: reach={config.REACH_COST_PER_UNIT}/unit, observe={config.OBSERVE_COST}, probe={config.PROBE_COST}")
    w(f"- Budget: {config.DEFAULT_INITIAL_BUDGET}")
    w(f"- Coverage: {config.COVERAGE}")
    w(f"- Cue condition: {config.FIXED_CONDITION['label']}")
    w(f"- Seeds: {config.SEEDS}")
    w(f"- Policy configs: {config.POLICY_CONFIGS}")
    w()

    # ---- Guardrail Results ----
    w("## Guardrail Tests")
    w()
    all_gr_pass = True
    for name, result in sorted(guardrail_results.items()):
        passed = result.get("passed", False)
        msg = result.get("message", "")
        if not passed:
            all_gr_pass = False
        status = "PASS" if passed else "FAIL"
        w(f"- [{status}] {name}: {msg}")
    w()
    if not all_gr_pass:
        w("**WARNING: Some guardrail tests failed.**")
        w()

    # ---- Calibration ----
    w("## Difficulty Calibration")
    w()
    w("| Budget | C0b Acc | C13_CW05 Acc | C13_CW05 Visit% | C13_CW05 Probe% | C14 Acc |")
    w("|--------|---------|-------------|-----------------|-----------------|---------|")
    for budget_str, row in sorted(cal_results.items(), key=lambda x: float(x[0])):
        c0b = row.get("C0b_observe_only", {})
        c13 = row.get("C13_instance_VOI_CW05", {})
        c14 = row.get("C14_oracle_appendix_only", {})
        w(f"| {float(budget_str):.1f} | {c0b.get('affordance_accuracy', 'N/A'):.4f} | "
          f"{c13.get('affordance_accuracy', 'N/A'):.4f} | "
          f"{c13.get('visit_pct', 'N/A'):.2%} | "
          f"{c13.get('probe_pct', 'N/A'):.2%} | "
          f"{c14.get('affordance_accuracy', 'N/A'):.4f} |")
    w()

    # ---- Main Results ----
    w("## Main Results")
    w()

    # Aggregate per config across seeds
    config_metrics = {}
    for seed_str, data in all_results.items():
        for cfg_id, cfg_data in data.get("results", {}).items():
            metrics = cfg_data.get("metrics", {})
            if not metrics:
                continue
            config_metrics.setdefault(cfg_id, []).append(metrics)

    # Per-config averages
    w("| Config | Aff Acc | CompTask Acc | Probe Cov | Visited | Probed | Total Cost | Utility | Entropy Gap |")
    w("|--------|---------|-------------|-----------|---------|--------|------------|---------|-------------|")
    for cfg_id in config.POLICY_CONFIGS:
        metrics_list = config_metrics.get(cfg_id, [])
        if not metrics_list:
            w(f"| {cfg_id} | — | — | — | — | — | — | — | — |")
            continue
        n = len(metrics_list)
        avg = {}
        for key in ["affordance_accuracy", "composite_task_accuracy", "probe_coverage",
                     "objects_visited", "objects_probed", "total_cost", "total_utility",
                     "entropy_gap"]:
            vals = [m.get(key, 0.0) for m in metrics_list if key in m]
            avg[key] = sum(vals) / max(len(vals), 1)

        w(f"| {cfg_id} | {avg['affordance_accuracy']:.4f} | {avg['composite_task_accuracy']:.4f} | "
          f"{avg['probe_coverage']:.2%} | {avg['objects_visited']:.1f} | "
          f"{avg['objects_probed']:.1f} | {avg['total_cost']:.4f} | {avg['total_utility']:.4f} | "
          f"{avg['entropy_gap']:+.4f} |")
    w()

    # ---- Success Criteria ----
    w("## Success Criteria Checklist")
    w()

    c13_metrics = config_metrics.get("C13_instance_VOI_CW05", [])
    c0b_metrics = config_metrics.get("C0b_observe_only", [])
    c14_metrics = config_metrics.get("C14_oracle_appendix_only", [])

    def avg_metric(metrics_list, key):
        vals = [m.get(key, 0.0) for m in metrics_list if key in m]
        return sum(vals) / max(len(vals), 1) if vals else 0.0

    c13_acc = avg_metric(c13_metrics, "composite_task_accuracy")
    c13_cov = avg_metric(c13_metrics, "probe_coverage")
    c0b_acc = avg_metric(c0b_metrics, "composite_task_accuracy")
    c14_acc = avg_metric(c14_metrics, "composite_task_accuracy")
    c13_entropy_gap = avg_metric(c13_metrics, "entropy_gap")

    criteria = [
        ("C13 >= 75% composite task accuracy", c13_acc >= 0.75, f"{c13_acc:.4f}"),
        ("C13 <= 40% probe coverage", c13_cov <= 0.40, f"{c13_cov:.2%}"),
        ("C0b <= 60% composite task accuracy", c0b_acc <= 0.60, f"{c0b_acc:.4f}"),
        ("C14 >= 80% composite task accuracy", c14_acc >= 0.80, f"{c14_acc:.4f}"),
        ("Entropy gap (probed > skipped) positive", c13_entropy_gap > 0, f"{c13_entropy_gap:+.4f}"),
    ]

    w("| Criterion | Target | Actual | Result |")
    w("|-----------|--------|--------|--------|")
    all_pass = True
    for desc, passed, actual in criteria:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        w(f"| {desc} | — | {actual} | {status} |")
    w()

    # ---- C13 vs Baselines ----
    w("## C13 vs Baselines")
    w()
    c2_metrics = config_metrics.get("C2_cluster_reference", [])
    c8_metrics = config_metrics.get("C8_forced_EIG", [])

    c2_acc = avg_metric(c2_metrics, "affordance_accuracy")
    c8_acc = avg_metric(c8_metrics, "affordance_accuracy")
    c2_cost = avg_metric(c2_metrics, "total_cost")
    c8_cost = avg_metric(c8_metrics, "total_cost")
    c13_cost = avg_metric(c13_metrics, "total_cost")
    c13_util = avg_metric(c13_metrics, "total_utility")
    c2_util = avg_metric(c2_metrics, "total_utility")
    c8_util = avg_metric(c8_metrics, "total_utility")

    w(f"C13_CW05 vs C2: acc {c13_acc:.4f} vs {c2_acc:.4f}, cost {c13_cost:.4f} vs {c2_cost:.4f}")
    w(f"C13_CW05 vs C8: acc {c13_acc:.4f} vs {c8_acc:.4f}, cost {c13_cost:.4f} vs {c8_cost:.4f}")
    w(f"C13_CW05 utility: {c13_util:.4f}, C2 utility: {c2_util:.4f}, C8 utility: {c8_util:.4f}")
    w()

    # ---- C13 Cost Weight Sweep ----
    w("## C13 Cost Weight Sweep")
    w()
    w("| Cost Weight | Aff Acc | Probe Cov | Total Cost | Utility |")
    w("|-------------|---------|-----------|------------|---------|")
    for cw in config.COST_WEIGHTS:
        cw_label = config.COST_WEIGHT_LABELS[cw]
        cfg_id = f"C13_instance_VOI_{cw_label}"
        m_list = config_metrics.get(cfg_id, [])
        if m_list:
            acc = avg_metric(m_list, "affordance_accuracy")
            cov = avg_metric(m_list, "probe_coverage")
            cost = avg_metric(m_list, "total_cost")
            util = avg_metric(m_list, "total_utility")
            w(f"| {cw:.1f} | {acc:.4f} | {cov:.2%} | {cost:.4f} | {util:.4f} |")
    w()

    # ---- Case Interpretation ----
    w("## Interpretation")
    w()
    if all_pass and c13_acc >= 0.75 and c13_cov <= 0.40 and c0b_acc <= 0.60 and c14_acc >= 0.80 and c13_entropy_gap > 0:
        w("**Case A**: Mini-MC validates selective cost-aware instance+VOI probing.")
    elif c13_acc >= 0.75 and (c13_cov > 0.40 or c13_entropy_gap <= 0):
        w("**Case B**: Instance+VOI transfers but selective gate still unresolved.")
    elif c2_acc > c13_acc:
        w("**Case C**: Need hybrid instance/category abstraction.")
    else:
        w("**Case D**: Task design may be invalid — C0b too high or C14 too low.")
    w()

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved to: {output_path}")
    return "\n".join(lines)


if __name__ == "__main__":
    results_dir = os.path.join(CURRENT_DIR, "results_004_5o")
    generate_report(
        os.path.join(results_dir, "results_004_5o.json"),
        os.path.join(results_dir, "calibration.json"),
        os.path.join(results_dir, "guardrail_results.json"),
        os.path.join(results_dir, "aggregate_report_004_5o.md"),
    )
