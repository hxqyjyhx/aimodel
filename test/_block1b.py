"""Block 1B: Multi-CW test — cue=C3, budget=1.5, CW={0.5, 1.0, 1.5, 2.0}."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("runs", exist_ok=True)

t0 = time.time()
from run import run_calibration_grid
cal, best = run_calibration_grid(
    cue_filter={"C3_P060_O040"},
    budget_filter={1.5},
    cw_filter={0.5, 1.0, 1.5, 2.0},
)
elapsed = time.time() - t0

# Extract per-CW metrics
cue_data = cal.get("C3_P060_O040", {})
budget_row = cue_data.get("budgets", {}).get(1.5, {})

c0b = budget_row.get("C0b_observe_only", {})
c0b_comp = c0b.get("composite_task_accuracy", 0)
c0b_ncost = c0b.get("normalized_cost", 0)
c0b_ub25 = c0b.get("utility_betas", {}).get("utility_beta0.25", 0)
c0b_ub01 = c0b.get("utility_betas", {}).get("utility_beta0.1", 0)
c0b_ub05 = c0b.get("utility_betas", {}).get("utility_beta0.5", 0)

c14b = budget_row.get("C14b_budgeted_oracle_probe", {})
c14b_comp = c14b.get("composite_task_accuracy", 0)
c14b_pcov = c14b.get("probe_coverage", 0)
c14b_ncost = c14b.get("normalized_cost", 0)

c0a = budget_row.get("C0a_no_interaction", {})
c0a_comp = c0a.get("composite_task_accuracy", 0.70)

# Compute best_C13_comp across all CWs
c13_rows = {}
for cw in [0.5, 1.0, 1.5, 2.0]:
    cw_label = {0.0: "00", 0.5: "05", 1.0: "10", 1.5: "15", 2.0: "20"}[cw]
    c13_key = f"C13_cost_gate_CW{cw_label}"
    c13 = budget_row.get(c13_key, {})
    if c13 and "error" not in c13:
        c13_rows[cw] = c13

best_c13_comp = max(
    (r.get("composite_task_accuracy", 0) for r in c13_rows.values()),
    default=0
)

c14b_rel_ok = c14b_comp >= best_c13_comp + 0.01

# C14b ceiling
rough_ceil = c14b_pcov * 1.0 + (1.0 - c14b_pcov) * c0a_comp
ceil_ratio = c14b_comp / max(rough_ceil, 0.001)

print()
print("=" * 100)
print("Block 1B — Per-CW Summary")
print("=" * 100)
print(f"  Cue: C3_P060_O040  Budget: 1.5")
print()
print(f"  C0b baseline:     comp={c0b_comp:.4f}  ncost={c0b_ncost:.4f}  "
      f"ub01={c0b_ub01:+.4f}  ub25={c0b_ub25:+.4f}  ub05={c0b_ub05:+.4f}")
print(f"  C14b oracle:      comp={c14b_comp:.4f}  pcov={c14b_pcov:.2%}  "
      f"ncost={c14b_ncost:.4f}")
print(f"  C0a (no-interact): comp={c0a_comp:.4f}")
print()

hdr = (f"  {'CW':<5} {'C13_comp':<10} {'C13_nc':<8} "
       f"{'C13_ub25':<10} {'C13_pcov':<10} {'ent_gap':<9} "
       f"{'comp_gain':<10} {'cost_red':<10} {'util_gain':<10} "
       f"{'cov_ok':<7} {'ent_ok':<7} {'util_ok':<7}")
print(hdr)
print("  " + "-" * len(hdr))

candidates = []
for cw in [0.5, 1.0, 1.5, 2.0]:
    c13 = c13_rows.get(cw)
    if not c13:
        continue
    comp = c13.get("composite_task_accuracy", 0)
    ncost = c13.get("normalized_cost", 0)
    ub25 = c13.get("utility_betas", {}).get("utility_beta0.25", 0)
    pcov = c13.get("probe_coverage", 0)
    ent = c13.get("entropy_gap", 0)
    cgain = comp - c0b_comp
    cred = c0b_ncost - ncost
    ugain = ub25 - c0b_ub25
    cov_ok = 0 < pcov <= 0.40
    ent_ok = ent > 0
    util_ok = ub25 > c0b_ub25
    is_cand = cov_ok and ent_ok and util_ok
    if is_cand:
        candidates.append(cw)

    print(f"  {cw:<5.1f} {comp:<10.4f} {ncost:<8.4f} "
          f"{ub25:<+10.4f} {pcov:<10.2%} {ent:<+9.4f} "
          f"{cgain:<+10.4f} {cred:<+10.4f} {ugain:<+10.4f} "
          f"{'OK' if cov_ok else 'FAIL':<7} {'OK' if ent_ok else 'FAIL':<7} "
          f"{'OK' if util_ok else 'FAIL':<7}")

print()
print(f"  best_C13_comp = {best_c13_comp:.4f}")
print(f"  C14b_relative_ok = {'OK' if c14b_rel_ok else 'FAIL'} "
      f"(C14b={c14b_comp:.4f} >= {best_c13_comp + 0.01:.4f})")
print(f"  rough_budget_coverage_ceiling = {rough_ceil:.4f}")
print(f"  C14b_ceiling_ratio = {ceil_ratio:.4f}")
print()
if candidates:
    print(f"  CANDIDATES (CW passing cov + ent + util): {candidates}")
else:
    print("  NO CANDIDATES: No CW satisfies all three: cov>0 & <=40%, ent>0, util>C0b")
    print()
    print("  Utility failure analysis:")
    for cw in [0.5, 1.0, 1.5, 2.0]:
        c13 = c13_rows.get(cw)
        if not c13:
            continue
        comp = c13.get("composite_task_accuracy", 0)
        ncost = c13.get("normalized_cost", 0)
        ub25 = c13.get("utility_betas", {}).get("utility_beta0.25", 0)
        pcov = c13.get("probe_coverage", 0)
        ent = c13.get("entropy_gap", 0)
        reasons = []
        if not (0 < pcov <= 0.40):
            reasons.append(f"pcov={pcov:.2%} (not in 0..40%)")
        if ent <= 0:
            reasons.append(f"ent_gap={ent:+.4f} (<=0)")
        if ub25 <= c0b_ub25:
            reasons.append(f"ub25={ub25:+.4f} <= C0b_ub25={c0b_ub25:+.4f} "
                           f"(comp_gain={comp-c0b_comp:+.4f}, ncost_delta={ncost-c0b_ncost:+.4f})")
        print(f"    CW={cw:.1f}: {', '.join(reasons) if reasons else 'ALL OK'}")

print()
print("[block_done]")
print(f"  block_id=1B")
print(f"  n_configs=1_cue_x_1_budget_x_4_CW")
print(f"  elapsed={elapsed:.0f}s")
print(f"  n_candidates={len(candidates)}")
print(f"  best_C13_comp={best_c13_comp:.4f}")
print(f"  C14b_relative_ok={'OK' if c14b_rel_ok else 'FAIL'}")

# Save block output
block_out = {
    "block_id": "1B",
    "elapsed_s": elapsed,
    "cue": "C3_P060_O040",
    "budget": 1.5,
    "cws_tested": [0.5, 1.0, 1.5, 2.0],
    "c0b": {"comp": c0b_comp, "ncost": c0b_ncost, "ub25": c0b_ub25},
    "c14b": {"comp": c14b_comp, "pcov": c14b_pcov, "ncost": c14b_ncost,
             "rel_ok": c14b_rel_ok, "rough_ceil": rough_ceil, "ceil_ratio": ceil_ratio},
    "best_c13_comp": best_c13_comp,
    "candidates": candidates,
    "per_cw": {str(cw): {
        "comp": r.get("composite_task_accuracy", 0),
        "ncost": r.get("normalized_cost", 0),
        "ub25": r.get("utility_betas", {}).get("utility_beta0.25", 0),
        "pcov": r.get("probe_coverage", 0),
        "ent_gap": r.get("entropy_gap", 0),
    } for cw, r in c13_rows.items()},
}
with open("runs/calibration_block1b_c3_candidates.json", "w") as f:
    json.dump(block_out, f, indent=2)
print(f"  saved: runs/calibration_block1b_c3_candidates.json")
