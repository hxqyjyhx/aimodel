"""Block 1A: Smoke test — cue=C3, budget=1.5, CW=0.5 only."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("runs", exist_ok=True)

t0 = time.time()
from run import run_calibration_grid
cal, best = run_calibration_grid(
    cue_filter={"C3_P060_O040"},
    budget_filter={1.5},
    cw_filter={0.5},
)
elapsed = time.time() - t0

print()
print("[block_done]")
print(f"  block_id=1A")
n_configs = sum(
    1 for cue_data in cal.values()
    if isinstance(cue_data, dict) and "budgets" in cue_data
    for budget_data in cue_data["budgets"].values()
    for k, v in budget_data.items()
    if k.startswith("C13_cost_gate_CW") and not isinstance(v, dict) or "error" not in str(v)
)
print(f"  n_configs=1_cue_x_1_budget_x_1_CW")
print(f"  elapsed={elapsed:.0f}s")
if best:
    print(f"  best_candidate: cue={best['cue_label']} B={best['budget']:.1f} CW={best['cost_weight']:.1f}")
    print(f"  result=PASS")
else:
    print(f"  result=FAIL")
    print(f"  pass_count=0")

# Save block output
block_out = {
    "block_id": "1A",
    "elapsed_s": elapsed,
    "best": (
        {"cue": best["cue_label"], "budget": best["budget"], "cw": best["cost_weight"]}
        if best else None
    ),
    "pass": best is not None,
}
with open("runs/calibration_block1a_smoke.json", "w") as f:
    json.dump(block_out, f, indent=2)
print(f"  saved: runs/calibration_block1a_smoke.json")
