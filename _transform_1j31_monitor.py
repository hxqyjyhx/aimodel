"""
Transform _block1j31_seed109_raw_zero_monitor.py:
Add step-level monitoring, CSV export, and time-of-learning table.
"""
import os, sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(CURRENT_DIR, "_block1j31_seed109_raw_zero_monitor.py")

with open(TARGET, "r", encoding="utf-8") as f:
    content = f.read()

replaced_count = 0

# ===========================================================================
# 1. Add monitoring data structures after the import block
# ===========================================================================
old_import_block = '''from subtype_objects import (
    generate_subtype_objects_deterministic, compute_query_ground_truth,
)

t0 = time.time()'''

new_import_block = '''from subtype_objects import (
    generate_subtype_objects_deterministic, compute_query_ground_truth,
)

# =============================================================================
# Step-Level Monitoring Infrastructure (1J31 monitor addition)
# =============================================================================
STEP_MONITOR_LOG = []  # per-step per-candidate entries
STEP_MONITOR_CSV_ROWS = []  # flat list for CSV export

def _monitor_record_step(episode_id, step_id, object_id, family, dev_feats,
                          score_no_mem, score_with_mem, per_action_details):
    # Record per-candidate per-action step-level monitoring data.
    entry = {
        "episode": episode_id,
        "step": step_id,
        "object_id": object_id,
        "family": family,
        "dev_feats_present": dev_feats,
        "score_before_memory": round(score_no_mem, 6) if score_no_mem is not None else None,
        "score_after_memory": round(score_with_mem, 6) if score_with_mem is not None else None,
        "per_action": per_action_details,
    }
    STEP_MONITOR_LOG.append(entry)
    for ad in per_action_details:
        STEP_MONITOR_CSV_ROWS.append({
            "episode": episode_id,
            "step": step_id,
            "object_id": object_id,
            "family": family,
            "dev_feats_present": ",".join(dev_feats) if dev_feats else "",
            "action": ad["action"],
            "base_prior": ad["base_prior"],
            "risk_penalty_raw": ad["risk_penalty_raw"],
            "risk_penalty_clipped": ad["risk_penalty_clipped"],
            "scaled_penalty": ad["scaled_penalty"],
            "raw_score": ad["raw_score"],
            "final_score": ad["final_score"],
            "per_feat_weights": ad.get("per_feat_weights_str", ""),
            "ground_truth_outcome": ad.get("ground_truth_outcome", ""),
            "is_prior_violation": ad.get("is_prior_violation", False),
            "was_selected": ad.get("was_selected", False),
        })

t0 = time.time()'''

pos = content.find(old_import_block)
if pos == -1:
    print("ERROR: Could not find import block for injection")
    sys.exit(1)
content = content.replace(old_import_block, new_import_block, 1)
replaced_count += 1
print(f"  [1/6] Monitoring infrastructure injected at position {pos}")

# ===========================================================================
# 2. Add self._step_monitor to OnlineSignedPolicy.__init__
# ===========================================================================
old_init = '''        self._decision_path_log = []
        self._pending_decision_log = None'''

new_init = '''        self._decision_path_log = []
        self._pending_decision_log = None
        self._monitor_step_entries = []  # 1J31: per-step detailed monitoring'''

pos = content.find(old_init)
if pos == -1:
    print("ERROR: Could not find OnlineSignedPolicy init block")
    sys.exit(1)
content = content.replace(old_init, new_init, 1)
replaced_count += 1
print(f"  [2/6] OnlineSignedPolicy.__init__ monitor field added at position {pos}")

# ===========================================================================
# 3. Inject monitoring code in decide_probe action loop (OnlineSignedPolicy only)
# ===========================================================================
old_action_loop = '''        best_action = None; best_score = float('-inf')
        best_info = (EXPLORATION_FLOOR, 0.0, 0.0, EXPLORATION_FLOOR, EXPLORATION_FLOOR)
        for action in MAIN_CANDIDATE_ACTIONS:
            norm_cost = view.probe_cost / max(view.initial_budget, 0.001)
            final_score, raw_score, base_prior, risk_penalty, scaled_penalty = \\
                self._compute_adjusted_score(features, action, norm_cost)
            if final_score > best_score:
                best_score = final_score; best_action = action
                best_info = (base_prior, risk_penalty, scaled_penalty, raw_score, final_score)'''

new_action_loop = '''        best_action = None; best_score = float('-inf')
        best_info = (EXPLORATION_FLOOR, 0.0, 0.0, EXPLORATION_FLOOR, EXPLORATION_FLOOR)
        per_action_monitor = []
        for action in MAIN_CANDIDATE_ACTIONS:
            norm_cost = view.probe_cost / max(view.initial_budget, 0.001)
            final_score, raw_score, base_prior, risk_penalty, scaled_penalty = \\
                self._compute_adjusted_score(features, action, norm_cost)
            # 1J31: capture per-action weights
            feat_weights = self._get_per_feature_weights(features, action)
            feat_weights_str = ";".join(f"{f}={w:.6f}" for f, w in feat_weights)
            per_action_monitor.append({
                "action": action,
                "base_prior": round(base_prior, 6),
                "risk_penalty_raw": round(risk_penalty, 6),
                "risk_penalty_clipped": round(risk_penalty, 6),
                "scaled_penalty": round(scaled_penalty, 6),
                "raw_score": round(raw_score, 6),
                "final_score": round(final_score, 6),
                "per_feat_weights_str": feat_weights_str,
                "ground_truth_outcome": "unknown",
                "is_prior_violation": False,
                "was_selected": False,
            })
            if final_score > best_score:
                best_score = final_score; best_action = action
                best_info = (base_prior, risk_penalty, scaled_penalty, raw_score, final_score)'''

pos = content.find(old_action_loop)
if pos == -1:
    print("ERROR: Could not find decide_probe action loop")
    sys.exit(1)
content = content.replace(old_action_loop, new_action_loop, 1)
replaced_count += 1
print(f"  [3/6] decide_probe action monitoring injected at position {pos}")

# ===========================================================================
# 4. Inject monitoring record call + mark selected action (after best_action, before return)
# ===========================================================================
old_return = '''        self._probed_oids.add(object_id)
        bp, rp, sp, rs, fs = best_info
        self._selected_scores.append({
            "oid": object_id, "action": best_action,
            "prior": bp, "risk_penalty": round(rp, 6),
            "scaled_risk_penalty": round(sp, 6),
            "final_score": round(fs, 6), "risk_adjustment": round(sp, 6),
        })
        return True, best_action'''

new_return = '''        # Mark the selected action and record step-level monitoring
        if per_action_monitor:
            for pam in per_action_monitor:
                if pam["action"] == best_action:
                    pam["was_selected"] = True
                    pam["final_score"] = round(fs, 6)
                    pam["scaled_penalty"] = round(sp, 6)
            _monitor_record_step(
                self._episode_id,
                len(self._probed_oids),
                object_id,
                detect_type_family(features),
                sorted([f for f in INJECTED_DEVIATION_FEATURES if features.get(f, False)]),
                self._score_candidate_no_memory(features,
                    view.probe_cost / max(view.initial_budget, 0.001)),
                self._score_candidate(features,
                    view.probe_cost / max(view.initial_budget, 0.001)),
                per_action_monitor,
            )
        self._probed_oids.add(object_id)
        bp, rp, sp, rs, fs = best_info
        self._selected_scores.append({
            "oid": object_id, "action": best_action,
            "prior": bp, "risk_penalty": round(rp, 6),
            "scaled_risk_penalty": round(sp, 6),
            "final_score": round(fs, 6), "risk_adjustment": round(sp, 6),
        })
        return True, best_action'''

pos = content.find(old_return)
if pos == -1:
    print("ERROR: Could not find decide_probe return block")
    sys.exit(1)
content = content.replace(old_return, new_return, 1)
replaced_count += 1
print(f"  [4/6] decide_probe monitor recording injected at position {pos}")

# ===========================================================================
# 5. Add CSV export section before "Print summary"
# ===========================================================================
old_print_summary = '''
# ===========================================================================
# Print summary
# ===========================================================================
print("")
print("=" * 70)
print(f"Block 1J31 Raw-Zero Monitor complete.")'''

# Build the replacement — careful with triple quotes inside
csv_section = '''
# ===========================================================================
# 12. CSV Export (1J31 monitor addition)
# ===========================================================================
print("\\n[11/12] Writing step-level monitoring CSV...")
csv_path = os.path.join(CURRENT_DIR, "runs",
    "block1j31_seed109_raw_zero_monitor.csv")
csv_fieldnames = [
    "episode", "step", "object_id", "family", "dev_feats_present",
    "action", "base_prior", "risk_penalty_raw", "risk_penalty_clipped",
    "scaled_penalty", "raw_score", "final_score", "per_feat_weights",
    "ground_truth_outcome", "is_prior_violation", "was_selected",
]
import csv as csv_module
with open(csv_path, "w", newline="") as cf:
    writer = csv_module.DictWriter(cf, fieldnames=csv_fieldnames)
    writer.writeheader()
    for row in STEP_MONITOR_CSV_ROWS:
        writer.writerow({k: row.get(k, "") for k in csv_fieldnames})
print(f"  CSV -> {csv_path}  ({len(STEP_MONITOR_CSV_ROWS)} rows)")

# ===========================================================================
# 13. Time-of-Learning Table (1J31 monitor addition)
# ===========================================================================
print("\\n[12/12] Computing time-of-learning table...")
tol_data = {}
if hasattr(mem_ss, 'update_log'):
    for entry in mem_ss.update_log:
        ep = entry.get("episode_id", -1)
        step = entry.get("step_id", -1)
        goal = entry.get("goal", "")
        action = entry.get("action", "")
        family = entry.get("inferred_family", "")
        dev_feats = entry.get("injected_deviation_features", [])
        is_violation = entry.get("prior_violation", False)
        is_eligible = entry.get("is_eligible", False)

        for df in dev_feats:
            key = (goal, action, family, df)
            if key not in tol_data:
                tol_data[key] = {
                    "first_episode": ep,
                    "first_step": step,
                    "total_eligible_updates": 0,
                    "total_violation_updates": 0,
                    "weight_by_episode": {},
                    "support_by_episode": {},
                }
            td = tol_data[key]
            if is_eligible:
                td["total_eligible_updates"] += 1
                if is_violation:
                    td["total_violation_updates"] += 1

    # Get weight after each episode
    for key in tol_data:
        for ep_idx in range(N_EPISODES):
            try:
                w = mem_ss.get_raw_risk_weight_signed(*key)
                tol_data[key]["weight_by_episode"][ep_idx] = round(w, 6)
                c = mem_ss.get_raw_counts(*key)
                tol_data[key]["support_by_episode"][ep_idx] = {
                    "v_with": c["violation_with"],
                    "nv_with": c["nonviolation_with"],
                    "v_without": c["violation_without"],
                    "nv_without": c["nonviolation_without"],
                }
            except:
                pass

tol_keys_sorted = sorted(tol_data.keys(), key=lambda k: (
    tol_data[k].get("first_episode", 99),
    -(tol_data[k].get("total_eligible_updates", 0))
))

print(f"  Time-of-learning: {len(tol_keys_sorted)} unique keys")

# Build time-of-learning table for JSON output
tol_table = []
for key in tol_keys_sorted:
    td = tol_data[key]
    tol_table.append({
        "key": list(key),
        "first_episode": td["first_episode"],
        "first_step": td["first_step"],
        "total_eligible_updates": td["total_eligible_updates"],
        "total_violation_updates": td["total_violation_updates"],
        "weight_by_episode": td["weight_by_episode"],
        "support_by_episode": {str(k): v for k, v in td["support_by_episode"].items()},
    })

# Add to output JSON
out["step_monitor_summary"] = {
    "total_monitor_entries": len(STEP_MONITOR_LOG),
    "total_csv_rows": len(STEP_MONITOR_CSV_ROWS),
}
out["time_of_learning"] = {
    "num_unique_keys": len(tol_keys_sorted),
    "keys": tol_table,
}
out["step_monitor_log"] = STEP_MONITOR_LOG

# ===========================================================================
# Print summary
# ===========================================================================
print("")
print("=" * 70)
print(f"Block 1J31 Raw-Zero Monitor complete.")'''

pos = content.find(old_print_summary)
if pos == -1:
    print("ERROR: Could not find print summary section")
    sys.exit(1)
content = content.replace(old_print_summary, csv_section, 1)
replaced_count += 1
print(f"  [5/6] CSV export + time-of-learning injected at position {pos}")

# ===========================================================================
# 6. Add time-of-learning section to MD protocol
# ===========================================================================
old_md_end = '''md_lines.append(f"elapsed={elapsed:.1f}s")
md_lines.append("```")'''

new_md_end = '''md_lines.append(f"elapsed={elapsed:.1f}s")
md_lines.append("```")
md_lines.append("")
md_lines.append("## 10. Step-Level Monitoring Summary")
md_lines.append("")
md_lines.append(f"- Total monitor entries (per-candidate per-step): {len(STEP_MONITOR_LOG)}")
md_lines.append(f"- Total CSV rows (per-action per-step): {len(STEP_MONITOR_CSV_ROWS)}")
md_lines.append(f"- CSV output: runs/block1j31_seed109_raw_zero_monitor.csv")
md_lines.append("")
md_lines.append("## 11. Time-of-Learning Table")
md_lines.append("")
md_lines.append(f"- Unique compound keys with updates: {len(tol_keys_sorted)}")
md_lines.append("")
md_lines.append("| Goal | Action | Family | Dev Feat | First Ep | Eligible | Violations | Final Weight |")
md_lines.append("|------|--------|--------|-----------|----------|----------|------------|-------------|")
for t in tol_table[:30]:
    g, a, fam, df = t["key"]
    w_eps = t.get("weight_by_episode", {})
    final_w = w_eps.get(max(w_eps.keys()) if w_eps else -1, 0.0)
    md_lines.append(f"| {g} | {a} | {fam} | {df} | {t['first_episode']} | {t['total_eligible_updates']} | {t['total_violation_updates']} | {final_w:.6f} |")
md_lines.append("")
md_lines.append("### Per-Episode Weight Evolution")
md_lines.append("")
for t in tol_table[:20]:
    g, a, fam, df = t["key"]
    w_eps = t.get("weight_by_episode", {})
    w_str = " | ".join(f"ep{ep}={w_eps.get(ep, 0):.4f}" for ep in sorted(w_eps.keys()))
    md_lines.append(f"- **({g}, {a}, {fam}, {df})**: {w_str}")'''

pos = content.find(old_md_end)
if pos == -1:
    print("ERROR: Could not find MD end block")
    sys.exit(1)
content = content.replace(old_md_end, new_md_end, 1)
replaced_count += 1
print(f"  [6/6] MD time-of-learning section injected at position {pos}")

# ===========================================================================
# Write
# ===========================================================================
with open(TARGET, "w", encoding="utf-8") as f:
    f.write(content)

print(f"\nTransform complete.  {replaced_count}/6 replacements applied.")
print(f"  Total length: {len(content)} chars")
print(f"  1J31 occurrences: {content.count('1J31')}")
print(f"  1J30 occurrences: {content.count('1J30')}")
