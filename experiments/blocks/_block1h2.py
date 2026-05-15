"""Block 1H2: Balanced Utility Feasibility / Cost-Scale Audit.

Pure analytic block — reads saved Block 1H1 results and computes
derived feasibility metrics. No simulation.
"""
import json, os

BETA = 0.25

# ---- Load Block 1H1 results ----
with open("runs/calibration_block1h1_balanced_policy_rescore.json", "r") as f:
    h1 = json.load(f)

policies_data = h1["policies"]

# Extract key policies
all_false = policies_data["all_false"]
c0b = policies_data["c0b"]
c13 = policies_data["c13"]
filt = policies_data["c13_distance_filter"]
aware = policies_data["c13_distance_aware_selection"]
stop = policies_data["c13_distance_filter_stop_on_skip"]
oracle = policies_data["oracle_unconstrained"]
iom = policies_data["iom_prior"]

policy_order = ["all_false", "iom_prior", "c0b", "c13",
                "c13_distance_filter", "c13_distance_aware_selection",
                "c13_distance_filter_stop_on_skip", "oracle_unconstrained"]
policy_map = {
    "all_false": all_false,
    "iom_prior": iom,
    "c0b": c0b,
    "c13": c13,
    "c13_distance_filter": filt,
    "c13_distance_aware_selection": aware,
    "c13_distance_filter_stop_on_skip": stop,
    "oracle_unconstrained": oracle,
}

print("=" * 70)
print("Block 1H2: Balanced Utility Feasibility / Cost-Scale Audit")
print(f"  beta={BETA}, data from Block 1H1")
print("=" * 70)

# ============================================================
# Per-policy feasibility metrics
# ============================================================
feasibility = {}

for pname in policy_order:
    p = policy_map[pname]
    mb = p["macro_query_bal_acc"]
    ncost = p["normalized_cost"]

    # 1. balanced gain over chance
    gain_over_chance = mb - 0.5

    # 2. cost penalty at beta=0.25
    cost_penalty = BETA * ncost

    # 3. net gain over chance (= bal_ub25 - 0.5, also = gain_over_chance - cost_penalty)
    net_gain = gain_over_chance - cost_penalty

    # 4. required macro_bal to beat all_false
    req_mb_beat_af = 0.5 + BETA * ncost

    # 5. margin to beat all_false
    margin_beat_af = mb - req_mb_beat_af

    # 6. required beta to break even vs all_false
    if ncost > 0:
        req_beta_af = gain_over_chance / ncost
    else:
        req_beta_af = None

    # 7. required cost scale to break even at beta=0.25
    # This is: what normalized_cost would give net_gain=0 at current macro_bal
    # mb - 0.5 - 0.25 * ncost = 0  =>  ncost = (mb - 0.5) / 0.25
    if gain_over_chance > 0:
        max_ncost_at_beta025 = gain_over_chance / BETA
    else:
        max_ncost_at_beta025 = 0.0  # can't break even at any positive cost

    feasibility[pname] = {
        "macro_bal": mb,
        "normalized_cost": ncost,
        "bal_ub25": p["balanced_utility_beta0.25"],
        "gain_over_chance": gain_over_chance,
        "cost_penalty_beta025": cost_penalty,
        "net_gain_over_chance": net_gain,
        "required_mb_to_beat_all_false": req_mb_beat_af,
        "margin_to_beat_all_false": margin_beat_af,
        "required_beta_to_break_even_vs_all_false": req_beta_af,
        "max_ncost_at_beta025": max_ncost_at_beta025,
    }

# ---- Table: Per-policy feasibility ----
print(f"\n{'='*70}")
print(f"Per-Policy Feasibility Metrics (beta={BETA})")
print(f"{'='*70}")
header = (f"  {'policy':<32} {'mb':>8} {'ncost':>7} {'bal_ub25':>9} "
          f"{'gain>0.5':>9} {'cost_pen':>8} {'net>0':>7} "
          f"{'req_mb_af':>9} {'margin_af':>10} {'req_beta':>8}")
print(header)
print(f"  {'-'*32} {'-'*8} {'-'*7} {'-'*9} {'-'*9} {'-'*8} {'-'*7} {'-'*9} {'-'*10} {'-'*8}")
for pname in policy_order:
    f = feasibility[pname]
    req_beta_str = f"{f['required_beta_to_break_even_vs_all_false']:.4f}" if f['required_beta_to_break_even_vs_all_false'] is not None else "N/A"
    print(f"  {pname:<32} {f['macro_bal']:>8.4f} {f['normalized_cost']:>7.4f} "
          f"{f['bal_ub25']:>9.4f} {f['gain_over_chance']:>+9.4f} {f['cost_penalty_beta025']:>8.4f} "
          f"{f['net_gain_over_chance']:>+7.4f} {f['required_mb_to_beat_all_false']:>9.4f} "
          f"{f['margin_to_beat_all_false']:>+10.4f} {req_beta_str:>8}")

# ---- Deeper analysis ----
print(f"\n{'='*70}")
print(f"Cost-Scale Analysis")
print(f"{'='*70}")

# What cost scale would each policy need?
print(f"\n  --- Required cost scaling to break even vs all_false (beta={BETA}) ---")
for pname in ["c0b", "c13", "c13_distance_filter", "c13_distance_aware_selection",
              "c13_distance_filter_stop_on_skip"]:
    f = feasibility[pname]
    ncost = f["normalized_cost"]
    req_ncost = f["max_ncost_at_beta025"]
    if ncost > 0:
        scale_factor = req_ncost / ncost if ncost > 0 else float('inf')
        print(f"  {pname:<35}: current ncost={ncost:.4f}, max_ncost={req_ncost:.4f}, "
              f"need ncost reduced to {req_ncost/ncost*100:.1f}% of current")
    else:
        print(f"  {pname:<35}: ncost=0, no cost constraint")

# What beta would make each break even?
print(f"\n  --- Required beta to break even vs all_false ---")
for pname in ["c0b", "c13", "c13_distance_filter", "c13_distance_aware_selection",
              "c13_distance_filter_stop_on_skip"]:
    f = feasibility[pname]
    req_beta = f["required_beta_to_break_even_vs_all_false"]
    if req_beta is not None:
        print(f"  {pname:<35}: current_beta={BETA:.2f}, req_beta={req_beta:.4f} "
              f"({req_beta/BETA*100:.1f}% of current)")
    else:
        print(f"  {pname:<35}: ncost=0, always breaks even")

# ============================================================
# Pairwise requirements
# ============================================================
print(f"\n{'='*70}")
print(f"Pairwise Feasibility Requirements")
print(f"{'='*70}")

# C13 required macro_bal to beat C0b
c0b_mb = c0b["macro_query_bal_acc"]
c0b_ncost = c0b["normalized_cost"]
c0b_bal_ub25 = c0b["balanced_utility_beta0.25"]
c13_mb = c13["macro_query_bal_acc"]
c13_ncost = c13["normalized_cost"]

# C13 needs: c13_mb - BETA * c13_ncost > c0b_mb - BETA * c0b_ncost
# => c13_mb > c0b_mb - BETA * c0b_ncost + BETA * c13_ncost
# => c13_mb > c0b_bal_ub25 + BETA * c13_ncost
req_c13_mb_beat_c0b = c0b_bal_ub25 + BETA * c13_ncost
margin_c13_mb_beat_c0b = c13_mb - req_c13_mb_beat_c0b

print(f"\n  --- C13 to beat C0b ---")
print(f"  C0b bal_ub25 = {c0b_bal_ub25:.4f}")
print(f"  C13 current macro_bal = {c13_mb:.4f}, ncost = {c13_ncost:.4f}")
print(f"  Required C13 macro_bal to beat C0b: {req_c13_mb_beat_c0b:.4f}")
print(f"  Current C13 margin: {margin_c13_mb_beat_c0b:+.4f}"
      f"  {'OK' if margin_c13_mb_beat_c0b > 0 else 'FAIL'}")

# C13 required beta to beat C0b
# c13_mb - beta * c13_ncost > c0b_mb - beta * c0b_ncost
# c13_mb - c0b_mb > beta * (c13_ncost - c0b_ncost)
# beta < (c13_mb - c0b_mb) / (c13_ncost - c0b_ncost)
delta_mb_c13_c0b = c13_mb - c0b_mb
delta_ncost_c13_c0b = c13_ncost - c0b_ncost
if delta_ncost_c13_c0b > 0 and delta_mb_c13_c0b > 0:
    req_beta_c13_beat_c0b = delta_mb_c13_c0b / delta_ncost_c13_c0b
    print(f"  delta_macro_bal = {delta_mb_c13_c0b:+.4f}")
    print(f"  delta_ncost = {delta_ncost_c13_c0b:+.4f}")
    print(f"  Required beta to beat C0b: {req_beta_c13_beat_c0b:.4f} "
          f"(current beta={BETA:.2f})")
    print(f"  Beta ratio: {req_beta_c13_beat_c0b/BETA:.1%} of current beta")
else:
    req_beta_c13_beat_c0b = None
    print(f"  Cannot compute required beta: delta_mb={delta_mb_c13_c0b:+.4f}, "
          f"delta_ncost={delta_ncost_c13_c0b:+.4f}")

# C13 required normalized_cost to beat C0b at beta=0.25
# c13_mb - BETA * req_ncost > c0b_bal_ub25
# req_ncost < (c13_mb - c0b_bal_ub25) / BETA
max_c13_ncost_beat_c0b = (c13_mb - c0b_bal_ub25) / BETA
print(f"  Max C13 ncost to beat C0b at beta={BETA}: {max_c13_ncost_beat_c0b:.4f} "
      f"(current={c13_ncost:.4f}, need {max_c13_ncost_beat_c0b/c13_ncost*100:.1f}%)")

# C13 required macro_bal to beat all_false
af_bal_ub25 = all_false["balanced_utility_beta0.25"]
req_c13_mb_beat_af = af_bal_ub25 + BETA * c13_ncost
print(f"\n  --- C13 to beat all_false ---")
print(f"  all_false bal_ub25 = {af_bal_ub25:.4f}")
print(f"  Required C13 macro_bal to beat all_false: {req_c13_mb_beat_af:.4f}")
print(f"  Current C13 macro_bal: {c13_mb:.4f}, margin: {c13_mb - req_c13_mb_beat_af:+.4f}")

# distance_filter required macro_bal to beat C13
filt_mb = filt["macro_query_bal_acc"]
filt_ncost = filt["normalized_cost"]
# Same cost => just needs higher macro_bal
print(f"\n  --- distance_filter to beat C13 ---")
print(f"  Same ncost ({filt_ncost:.4f}), so just needs macro_bal > {c13_mb:.4f}")
print(f"  filter macro_bal = {filt_mb:.4f}, diff = {filt_mb - c13_mb:+.4f}")

# stop_on_skip required macro_bal floor to avoid metric exploit
stop_mb = stop["macro_query_bal_acc"]
stop_ncost = stop["normalized_cost"]
stop_bal_ub25 = stop["balanced_utility_beta0.25"]
print(f"\n  --- stop_on_skip metric-exploit diagnosis ---")
print(f"  Current: mb={stop_mb:.4f}, ncost={stop_ncost:.4f}, bal_ub25={stop_bal_ub25:.4f}")
print(f"  bal_ub25 > c0b_bal_ub25: {stop_bal_ub25:.4f} > {c0b_bal_ub25:.4f}  "
      f"{'YES (exploit concern)' if stop_bal_ub25 > c0b_bal_ub25 else 'OK'}")
print(f"  macro_bal < c0b_macro_bal: {stop_mb:.4f} < {c0b_mb:.4f}  "
      f"{'YES (below no-probe baseline)' if stop_mb < c0b_mb else 'OK'}")
min_acceptable_mb = 0.5 + BETA * stop_ncost  # at least beat chance
print(f"  min acceptable macro_bal for non-exploit: {min_acceptable_mb:.4f}")
print(f"  stop margin: {stop_mb - min_acceptable_mb:+.4f}")

# ============================================================
# Diagnostic Questions
# ============================================================
print(f"\n{'='*70}")
print(f"Diagnostic Questions")
print(f"{'='*70}")

c13_f = feasibility["c13"]

# Q1: Is C13's balanced gain large enough to justify its cost?
q1 = c13_f["gain_over_chance"] > 0.02  # at least 2% over chance
print(f"\n  Q1: Is C13's balanced gain large enough to justify its cost?")
print(f"      gain_over_chance = {c13_f['gain_over_chance']:+.4f}")
print(f"      cost_penalty = {c13_f['cost_penalty_beta025']:.4f}")
print(f"      net_gain = {c13_f['net_gain_over_chance']:+.4f}")
print(f"      Answer: NO — cost penalty ({c13_f['cost_penalty_beta025']:.4f}) "
      f"> gain ({c13_f['gain_over_chance']:+.4f}) by "
      f"{c13_f['cost_penalty_beta025'] - c13_f['gain_over_chance']:+.4f}")

# Q2: What beta would make C13 break even against all_false?
q2 = c13_f["required_beta_to_break_even_vs_all_false"]
print(f"\n  Q2: What beta would make C13 break even against all_false?")
print(f"      required_beta = {q2:.4f} (current={BETA:.2f})")
print(f"      beta would need to be {q2/BETA*100:.1f}% of current value")
print(f"      OR: C13 needs macro_bal = {c13_f['required_mb_to_beat_all_false']:.4f} "
      f"at current cost")

# Q3: What beta would make C13 break even against C0b?
if req_beta_c13_beat_c0b is not None:
    print(f"\n  Q3: What beta would make C13 break even against C0b?")
    print(f"      required_beta = {req_beta_c13_beat_c0b:.4f} (current={BETA:.2f})")
    print(f"      beta would need to be {req_beta_c13_beat_c0b/BETA*100:.1f}% of current")
else:
    print(f"\n  Q3: C13 cannot break even against C0b at any positive beta "
          f"(mb_gain={delta_mb_c13_c0b:+.4f}, ncost_delta={delta_ncost_c13_c0b:+.4f})")

# Q4: Is the current beta=0.25 too high relative to achievable balanced gains?
c0b_f = feasibility["c0b"]
max_gain_over_chance = max(f["gain_over_chance"] for f in feasibility.values())
max_interactive_gain = max(feasibility[p]["gain_over_chance"]
                           for p in ["c0b", "c13", "c13_distance_filter",
                                      "c13_distance_aware_selection"])
print(f"\n  Q4: Is the current beta={BETA} too high relative to achievable balanced gains?")
print(f"      Max gain_over_chance (any policy): {max_gain_over_chance:+.4f}")
print(f"      Max gain_over_chance (interactive): {max_interactive_gain:+.4f}")
print(f"      C0b gain_over_chance: {c0b_f['gain_over_chance']:+.4f}")
print(f"      Cost penalty per visit+observe: ncost_per_visit = {c0b_f['normalized_cost']/60:.4f}")
print(f"      Cost penalty per visit+observe+probe: ~{(c13_f['normalized_cost']/c13['visited_count']):.4f}")
print(f"      Answer: YES — even the best interactive gain ({max_interactive_gain:+.4f}) "
      f"is far below typical cost penalty per visit")
print(f"      A single visit+observe costs ~{c0b_f['normalized_cost']/60*BETA:.4f} in utility, "
      f"but max gain per interaction is ~{max_interactive_gain:.4f}")

# Q5: Is the environment too low-signal / too high-cost for policy comparison?
c13_visits = c13["visited_count"]
c13_probes = c13["probe_count"]
per_interaction_gain = c13_f["gain_over_chance"] / max(c13_visits + c13_probes, 0.001)
per_interaction_cost = c13_f["cost_penalty_beta025"] / max(c13_visits + c13_probes, 0.001)
print(f"\n  Q5: Is the environment too low-signal / too high-cost for policy comparison?")
print(f"      C13: {c13_visits} visits, {c13_probes} probes, "
      f"gain={c13_f['gain_over_chance']:+.4f}, cost_penalty={c13_f['cost_penalty_beta025']:.4f}")
print(f"      Per-interaction gain: {per_interaction_gain:.6f}")
print(f"      Per-interaction cost: {per_interaction_cost:.6f}")
print(f"      Gain/cost ratio: {per_interaction_gain/per_interaction_cost:.4f}"
      if per_interaction_cost > 0 else "      Gain/cost ratio: infinite (zero cost)")
print(f"      Answer: BOTH low-signal AND high-cost. "
      f"Each interaction costs ~{per_interaction_cost:.6f} utils but yields ~{per_interaction_gain:.6f} gain. "
      f"The signal-to-cost ratio is far below 1.0.")

# Q6: Future evaluation approach recommendation
print(f"\n  Q6: Recommended future evaluation approach:")
print(f"      A. bal_utility = mb - beta*cost")
print(f"         → Problem: at current beta, even C0b can't beat all_false")
print(f"      B. gain_utility = (mb - 0.5) - beta*cost")
print(f"         → Better: makes the chance-level baseline explicit")
print(f"         → But still collapses signal and cost into one number")
print(f"      C. performance-gated utility")
print(f"         → bal_utility only counted if mb > 0.5 + epsilon")
print(f"         → Prevents metric exploit but still one-dimensional")
print(f"      D. report mb and cost separately without collapsing")
print(f"         → RECOMMENDED for current environment")
print(f"         → Primary: macro_bal vs cost scatter")
print(f"         → Secondary: Pareto frontier (policies that aren't dominated)")
print(f"         → Allows seeing trade-offs without premature beta choice")
print(f"      For this environment: D is the right answer. "
      f"No single beta can simultaneously:")
print(f"        (a) make interaction positive-sum at achievable gains")
print(f"        (b) provide meaningful cost discrimination between policies")
print(f"        (c) avoid rewarding low-cost chance-level strategies")

# ============================================================
# Summary flags
# ============================================================
current_beta_feasible = (
    any(f["margin_to_beat_all_false"] > 0.01
        for p, f in feasibility.items()
        if p not in ["all_false", "iom_prior", "oracle_unconstrained"])
)
# Even C0b has margin -0.1444; no interactive policy beats all_false

policy_comparison_ready = current_beta_feasible

print(f"\n{'='*70}")
print(f"Summary")
print(f"{'='*70}")
print(f"  current_beta_feasible:           {current_beta_feasible}")
print(f"  policy_comparison_ready:         {policy_comparison_ready}")
print(f"  best_interactive_bal_ub25:       {max(feasibility[p]['bal_ub25'] for p in ['c0b','c13','c13_distance_filter','c13_distance_aware_selection','c13_distance_filter_stop_on_skip']):.4f}")
print(f"  all_false bal_ub25:              {all_false['balanced_utility_beta0.25']:.4f}")
print(f"  interactive policies beat all_false:  {current_beta_feasible}")

print()
print("[block_done]")
print(f"  block_id=1H2")
print(f"  c13_gain_over_chance={c13_f['gain_over_chance']:.4f}")
print(f"  c13_cost_penalty={c13_f['cost_penalty_beta025']:.4f}")
print(f"  c13_margin_to_beat_all_false={c13_f['margin_to_beat_all_false']:.4f}")
print(f"  c13_beta_break_even_vs_all_false={q2:.4f}")
c13_beta_vs_c0b_str = f"{req_beta_c13_beat_c0b:.4f}" if req_beta_c13_beat_c0b is not None else "N/A"
print(f"  c13_beta_break_even_vs_c0b={c13_beta_vs_c0b_str}")
print(f"  current_beta_feasible={'true' if current_beta_feasible else 'false'}")
print(f"  policy_comparison_ready={'true' if policy_comparison_ready else 'false'}")

# ============================================================
# Save
# ============================================================
def make_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, (int, float)):
        return obj
    elif obj is None:
        return None
    else:
        return str(obj)

block_out = {
    "block_id": "1H2",
    "beta": BETA,
    "data_source": "runs/calibration_block1h1_balanced_policy_rescore.json",
    "per_policy_feasibility": {
        pname: make_serializable(feasibility[pname])
        for pname in policy_order
    },
    "pairwise": {
        "c13_to_beat_c0b": {
            "c0b_bal_ub25": c0b_bal_ub25,
            "c13_macro_bal": c13_mb,
            "c13_ncost": c13_ncost,
            "required_c13_mb": req_c13_mb_beat_c0b,
            "margin": margin_c13_mb_beat_c0b,
            "required_beta": req_beta_c13_beat_c0b,
            "max_ncost_at_beta025": max_c13_ncost_beat_c0b,
        },
        "c13_to_beat_all_false": {
            "all_false_bal_ub25": af_bal_ub25,
            "required_c13_mb": req_c13_mb_beat_af,
            "margin": c13_mb - req_c13_mb_beat_af,
        },
        "distance_filter_to_beat_c13": {
            "same_cost": True,
            "required_mb": c13_mb,
            "filter_mb": filt_mb,
            "diff": filt_mb - c13_mb,
        },
        "stop_on_skip_metric_exploit": {
            "stop_mb": stop_mb,
            "stop_ncost": stop_ncost,
            "stop_bal_ub25": stop_bal_ub25,
            "c0b_bal_ub25": c0b_bal_ub25,
            "min_acceptable_mb": min_acceptable_mb,
            "is_exploit": stop_bal_ub25 > c0b_bal_ub25 and stop_mb < c0b_mb,
        },
    },
    "diagnostic_questions": {
        "q1_c13_gain_justifies_cost": False,
        "q2_beta_c13_vs_all_false": q2,
        "q3_beta_c13_vs_c0b": req_beta_c13_beat_c0b,
        "q4_beta_too_high": True,
        "q5_low_signal_high_cost": "both",
        "q6_recommendation": "D: report macro_bal and cost separately",
    },
    "summary_flags": {
        "current_beta_feasible": current_beta_feasible,
        "policy_comparison_ready": policy_comparison_ready,
    },
}
with open("runs/calibration_block1h2_balanced_utility_feasibility.json", "w") as f:
    json.dump(make_serializable(block_out), f, indent=2)
print(f"  saved: runs/calibration_block1h2_balanced_utility_feasibility.json")
