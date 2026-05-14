"""Block 1D: Post-hoc probe filtering upper bound — oracle skip zero-gain probes."""
import json, time

t0 = time.time()

# Load Block 1C results
with open("runs/calibration_block1c_diagnostic.json") as f:
    b1c = json.load(f)

import config

budget = b1c["budget"]  # 1.5
probe_records = b1c["probe_before_after"]["per_probe"]
n_probe = len(probe_records)

# Categorize probes
effective = [r for r in probe_records if r["delta_composite_correct"] > 0]
zero_gain = [r for r in probe_records if r["delta_composite_correct"] == 0]
negative  = [r for r in probe_records if r["delta_composite_correct"] < 0]
n_effective = len(effective)
n_zero_gain = len(zero_gain)
n_negative  = len(negative)

# Current C13 (from Block 1C cost decomposition)
c13_comp = b1c["probed_comp"] * b1c["probed_n"] / 60 + b1c["unprobed_comp"] * b1c["unprobed_n"] / 60
# More precisely: use per-object counts
# Actually let me compute from the raw numbers
c13_costs = b1c["costs"]
c13_total_cost = c13_costs["c13_total"]
c13_ncost = c13_total_cost / budget
c0b_total_cost = c13_costs["c0b_total"]
c0b_ncost = c0b_total_cost / budget

# For comp, use the formula from Block 1C: probed_comp * probed_n/60 + unprobed_comp * unprobed_n/60
# But the evaluator computed c13_comp = 0.7267 using all 60 objects
# Let me use the comp_gain based on C0b comp
c0b_comp = 0.7033
c13_comp = c0b_comp + b1c["comp_gain"]

# Utility
def utility_beta(comp, ncost, beta=0.25):
    return comp - beta * ncost

c13_ub25 = utility_beta(c13_comp, c13_ncost)
c0b_ub25 = utility_beta(c0b_comp, c0b_ncost)

# ---- Oracle skip zero-gain probes ----
# Save 0.05 per zero-gain probe
cost_wasted = n_zero_gain * config.PROBE_COST
adjusted_probe_cost = c13_costs["c13_probe"] - cost_wasted
adjusted_total_cost = c13_total_cost - cost_wasted
adjusted_ncost = adjusted_total_cost / budget
adjusted_comp = c13_comp  # zero-gain probes changed nothing
adjusted_ub25 = utility_beta(adjusted_comp, adjusted_ncost)

# ---- Effective probe rate ----
effective_rate = n_effective / max(n_probe, 1)
wasted_rate = n_zero_gain / max(n_probe, 1)
cost_wasted_ratio = cost_wasted / c13_total_cost

# ---- Per-action waste breakdown ----
action_waste = {}
for r in zero_gain:
    act = r["action"]
    if act not in action_waste:
        action_waste[act] = {"n_zero": 0, "objects": []}
    action_waste[act]["n_zero"] += 1
    action_waste[act]["objects"].append(r["object_id"])
for r in effective:
    act = r["action"]
    if act not in action_waste:
        action_waste[act] = {"n_zero": 0, "n_eff": 0, "objects": []}
    action_waste[act].setdefault("n_eff", 0)
    action_waste[act]["n_eff"] = action_waste[act].get("n_eff", 0) + 1

# ===================================================================
# Output
# ===================================================================
print("=" * 70)
print("Block 1D — Post-hoc Probe Filtering Upper Bound")
print(f"  cue=C3_P060_O040, budget={budget}, CW=0.5")
print("=" * 70)

print(f"\n--- Reference: C0b ---")
print(f"  comp          = {c0b_comp:.4f}")
print(f"  total_cost    = {c0b_total_cost:.4f}")
print(f"  normalized_cost = {c0b_ncost:.4f}")
print(f"  utility_beta0.25 = {c0b_ub25:.4f}")

print(f"\n--- 1. current_C13 ---")
print(f"  comp          = {c13_comp:.4f}")
print(f"  total_cost    = {c13_total_cost:.4f}")
print(f"  normalized_cost = {c13_ncost:.4f}")
print(f"  utility_beta0.25 = {c13_ub25:.4f}")
print(f"  n_probe       = {n_probe}")
print(f"  n_effective_probe = {n_effective}")
print(f"  n_zero_gain_probe = {n_zero_gain}")
if n_negative:
    print(f"  n_negative_probe = {n_negative}")

print(f"\n--- 2. oracle_skip_zero_gain_probe_upper_bound ---")
print(f"  (post-hoc diagnostic — NOT a real strategy)")
print(f"  METHOD: skip all {n_zero_gain} probes where delta_composite_correct=0")
print(f"  comp (unchanged) = {adjusted_comp:.4f}")
print(f"  saved_probe_cost = {cost_wasted:.4f}")
print(f"  adjusted_probe_cost = {adjusted_probe_cost:.4f}")
print(f"  adjusted_total_cost = {adjusted_total_cost:.4f}")
print(f"  adjusted_normalized_cost = {adjusted_ncost:.4f}")
print(f"  adjusted_utility_beta0.25 = {adjusted_ub25:.4f}")
print(f"  utility_gain_over_C0b = {adjusted_ub25 - c0b_ub25:+.4f}")
print(f"  utility_gain_over_C13 = {adjusted_ub25 - c13_ub25:+.4f}")

print(f"\n--- 3. oracle_effective_probe_only_rate ---")
print(f"  effective_probe_rate = {n_effective}/{n_probe} = {effective_rate:.4f}")
print(f"  wasted_probe_rate    = {n_zero_gain}/{n_probe} = {wasted_rate:.4f}")
print(f"  cost_wasted_on_zero_gain_probe = {cost_wasted:.4f}")
print(f"  cost_wasted_ratio              = {cost_wasted_ratio:.4f} ({cost_wasted_ratio:.1%})")

print(f"\n--- 4. Per-action waste detail ---")
print(f"  {'action':<20} {'n_total':<8} {'n_eff':<7} {'n_zero':<7} {'eff_rate':<9} "
      f"{'zero_oids'}")
for act in sorted(action_waste):
    aw = action_waste[act]
    n_total_act = aw.get("n_eff", 0) + aw.get("n_zero", 0)
    n_eff_act = aw.get("n_eff", 0)
    n_zero_act = aw.get("n_zero", 0)
    eff_r = n_eff_act / max(n_total_act, 1)
    zero_oids = aw.get("objects", [])
    print(f"  {act:<20} {n_total_act:<8} {n_eff_act:<7} {n_zero_act:<7} "
          f"{eff_r:.3f}     {zero_oids[:3]}{'...' if len(zero_oids) > 3 else ''}")

# Interpretation
print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)
if adjusted_ub25 > c0b_ub25:
    print(f"  oracle_skip_zero > C0b: {adjusted_ub25:.4f} > {c0b_ub25:.4f}  "
          f"(delta={adjusted_ub25 - c0b_ub25:+.4f})")
    print(f"  => C13's main problem is probe filtering / marginal value gate.")
    print(f"     If zero-gain probes could be identified in advance,")
    print(f"     C13 would beat C0b on utility.")
else:
    print(f"  oracle_skip_zero <= C0b: {adjusted_ub25:.4f} <= {c0b_ub25:.4f}")
    print(f"  => Even with perfect zero-gain filtering, C13 cannot beat C0b.")
    print(f"     The effective probes alone don't deliver enough gain")

print(f"  current_C13 utility   = {c13_ub25:.4f} vs C0b = {c0b_ub25:.4f} "
      f"({'beats' if c13_ub25 > c0b_ub25 else 'loses to'} C0b)")
print(f"  oracle skip_zero util = {adjusted_ub25:.4f} vs C0b = {c0b_ub25:.4f} "
      f"({'beats' if adjusted_ub25 > c0b_ub25 else 'loses to'} C0b)")

# Block done
print()
print("[block_done]")
print(f"  block_id=1D")
print(f"  elapsed={time.time()-t0:.0f}s")
print(f"  current_C13_ub25={c13_ub25:.4f}")
print(f"  oracle_skip_zero_ub25={adjusted_ub25:.4f}")
print(f"  C0b_ub25={c0b_ub25:.4f}")
print(f"  effective_rate={effective_rate:.4f}")
print(f"  cost_wasted_ratio={cost_wasted_ratio:.4f}")
print(f"  oracle_beats_C0b={'YES' if adjusted_ub25 > c0b_ub25 else 'NO'}")

# Save
block_out = {
    "block_id": "1D",
    "elapsed_s": time.time() - t0,
    "cue": "C3_P060_O040", "budget": budget, "cw": 0.5, "seed": 101,
    "C0b": {
        "comp": c0b_comp, "total_cost": c0b_total_cost,
        "normalized_cost": c0b_ncost, "utility_beta0.25": c0b_ub25,
    },
    "current_C13": {
        "comp": c13_comp, "total_cost": c13_total_cost,
        "normalized_cost": c13_ncost, "utility_beta0.25": c13_ub25,
        "n_probe": n_probe, "n_effective_probe": n_effective,
        "n_zero_gain_probe": n_zero_gain,
    },
    "oracle_skip_zero_gain": {
        "adjusted_comp": adjusted_comp,
        "adjusted_probe_cost": adjusted_probe_cost,
        "adjusted_total_cost": adjusted_total_cost,
        "adjusted_normalized_cost": adjusted_ncost,
        "adjusted_utility_beta0.25": adjusted_ub25,
        "utility_gain_over_C0b": adjusted_ub25 - c0b_ub25,
        "cost_wasted": cost_wasted,
    },
    "effective_probe_rate": {
        "effective": effective_rate,
        "wasted": wasted_rate,
        "cost_wasted_on_zero_gain": cost_wasted,
        "cost_wasted_ratio": cost_wasted_ratio,
    },
    "per_action_waste": {
        act: {
            "n_total": aw.get("n_eff", 0) + aw.get("n_zero", 0),
            "n_eff": aw.get("n_eff", 0),
            "n_zero": aw.get("n_zero", 0),
            "eff_rate": aw.get("n_eff", 0) / max(aw.get("n_eff", 0) + aw.get("n_zero", 0), 1),
        }
        for act, aw in sorted(action_waste.items())
    },
    "oracle_beats_C0b": adjusted_ub25 > c0b_ub25,
}
with open("runs/calibration_block1d_upper_bound.json", "w") as f:
    json.dump(block_out, f, indent=2)
print(f"  saved: runs/calibration_block1d_upper_bound.json")
