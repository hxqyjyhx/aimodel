"""
Block 1J27 -- Stratified Multiseed Robustness for Online Signed Memory.

Calls the CLI-patched 1J24b script for each seed [101, 103, 107, 109, 113],
then aggregates results across seeds with ceiling stratification.
"""
import os, sys, json, subprocess, time
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()

SEEDS = [101, 103, 107, 109, 113]
N_PERM_REPEATS = 20

# =============================================================================
# 1. Run 1J24b for each seed
# =============================================================================
print("=" * 70)
print("Block 1J27 -- Stratified Multiseed Online Signed Memory Robustness")
print(f"  seeds={SEEDS}  n_perm_repeats={N_PERM_REPEATS}")
print("=" * 70)

SCRIPT = os.path.join(CURRENT_DIR, "_block1j24b_control_clean_online_signed_memory.py")
seed_outputs = {}

for seed in SEEDS:
    suffix = f"seed{seed}"
    print(f"\n{'='*60}")
    print(f"Running seed={seed}  n_perm_repeats={N_PERM_REPEATS}  suffix={suffix}")
    print(f"{'='*60}")

    cmd = [
        "py", SCRIPT,
        "--seed", str(seed),
        "--n-perm-repeats", str(N_PERM_REPEATS),
        "--output-suffix", suffix,
    ]

    result = subprocess.run(cmd, cwd=CURRENT_DIR, capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR: seed={seed} failed with returncode={result.returncode}")
        seed_outputs[seed] = None
        continue

    json_path = os.path.join(CURRENT_DIR, "runs",
        f"block1j24b_control_clean_online_signed_memory_{suffix}.json")
    if not os.path.exists(json_path):
        print(f"  ERROR: output file not found: {json_path}")
        seed_outputs[seed] = None
        continue

    with open(json_path, "r") as f:
        data = json.load(f)
    seed_outputs[seed] = data
    print(f"  seed={seed} complete: A_pv={data.get('A_no_memory',{}).get('agg',{}).get('total_prior_violations','?')}")

# =============================================================================
# 2. Extract per-seed metrics (with derived fields)
# =============================================================================
print("\n" + "=" * 70)
print("Aggregating results across seeds...")
print("=" * 70)

def extract_seed_metrics(data):
    """Extract key metrics from a single-seed JSON output."""
    if data is None:
        return None

    agg_a = data.get("A_no_memory", {}).get("agg", {})
    variants = data.get("variants", {})
    part3 = data.get("part3_signed_signal_audit", {})
    part4 = data.get("part4_permutation_distribution", {})
    flags = data.get("part6_boolean_flags", {})

    a_pv = agg_a.get("total_prior_violations", 0)

    def get_pv(name):
        return variants.get(name, {}).get("stats", {}).get("total_prior_violations", 0)

    sum_pos_pv = get_pv("B_sum_positive_online")
    signed_sum_pv = get_pv("B_signed_sum_online")
    signed_max_pv = get_pv("B_signed_max_abs_online")
    fam_only_pv = get_pv("Family_only_online_control")
    offline_pv = get_pv("Offline_signed_sum_upper_bound")

    def get_macro_delta(name):
        return variants.get(name, {}).get("stats", {}).get("macro_delta_vs_A", 0.0)

    signed_sum_macro_delta = get_macro_delta("B_signed_sum_online")
    signed_max_macro_delta = get_macro_delta("B_signed_max_abs_online")

    ss_dist = part4.get("signed_sum_permuted_distribution", {})
    sm_dist = part4.get("signed_max_abs_permuted_distribution", {})

    ss_perm_median = ss_dist.get("median_total_pv", 0)
    sm_perm_median = sm_dist.get("median_total_pv", 0)
    ss_percentile = ss_dist.get("real_percentile_against_permuted", 0)
    sm_percentile = sm_dist.get("real_percentile_against_permuted", 0)

    # Derived fields
    signed_sum_pv_reduction_vs_A = a_pv - signed_sum_pv
    signed_max_abs_pv_reduction_vs_A = a_pv - signed_max_pv
    offline_improvement_room = a_pv - offline_pv
    online_gap_to_offline = signed_sum_pv - offline_pv

    signed_sum_beats_A = signed_sum_pv < a_pv
    signed_sum_beats_permuted_median = signed_sum_pv < ss_perm_median
    signed_sum_ties_permuted_median = signed_sum_pv == ss_perm_median
    signed_sum_loses_to_permuted_median = signed_sum_pv > ss_perm_median

    shared_positions = data.get("shared_positions_used", False)
    permuted_mode = data.get("permuted_control_mode", "")
    final_mem_perm = data.get("final_memory_permutation_used", True)
    det_perm_seed = data.get("deterministic_permutation_seed_used", False)
    no_leakage = flags.get("no_oracle_leakage_confirmed", {}).get("value", False)
    hard_excl = flags.get("hard_exclusion_used", {}).get("value", True)
    family_preservation = flags.get("deceptive_family_preservation_valid", {}).get("value", False)
    impl_status_orig = flags.get("implementation_status", {}).get("value", "?")
    failure_reason_orig = flags.get("failure_reason", {}).get("value", "?")

    return {
        "seed": data.get("smoke_seed", "?"),
        "A_total_pv": a_pv,
        "B_sum_positive_online_total_pv": sum_pos_pv,
        "B_signed_sum_online_total_pv": signed_sum_pv,
        "B_signed_max_abs_online_total_pv": signed_max_pv,
        "Family_only_online_control_total_pv": fam_only_pv,
        "Offline_signed_sum_upper_bound_total_pv": offline_pv,
        "signed_sum_permuted_median_pv": ss_perm_median,
        "signed_max_abs_permuted_median_pv": sm_perm_median,
        "signed_sum_real_percentile_against_permuted": ss_percentile,
        "signed_max_abs_real_percentile_against_permuted": sm_percentile,
        "B_signed_sum_online_macro_delta_vs_A": signed_sum_macro_delta,
        "B_signed_max_abs_online_macro_delta_vs_A": signed_max_macro_delta,
        "shared_positions_used": shared_positions,
        "permuted_control_mode": permuted_mode,
        "final_memory_permutation_used": final_mem_perm,
        "deterministic_permutation_seed_used": det_perm_seed,
        "no_oracle_leakage_confirmed": no_leakage,
        "hard_exclusion_used": hard_excl,
        "deceptive_family_preservation_valid": family_preservation,
        "implementation_status": impl_status_orig,
        "failure_reason": failure_reason_orig,
        # Derived fields
        "signed_sum_pv_reduction_vs_A": signed_sum_pv_reduction_vs_A,
        "signed_max_abs_pv_reduction_vs_A": signed_max_abs_pv_reduction_vs_A,
        "offline_improvement_room": offline_improvement_room,
        "online_gap_to_offline": online_gap_to_offline,
        "signed_sum_beats_A": signed_sum_beats_A,
        "signed_sum_beats_permuted_median": signed_sum_beats_permuted_median,
        "signed_sum_ties_permuted_median": signed_sum_ties_permuted_median,
        "signed_sum_loses_to_permuted_median": signed_sum_loses_to_permuted_median,
    }


per_seed = {}
for seed in SEEDS:
    m = extract_seed_metrics(seed_outputs.get(seed))
    per_seed[seed] = m
    if m:
        print(f"  seed={seed}: A={m['A_total_pv']}, signed_sum={m['B_signed_sum_online_total_pv']}, "
              f"signed_max={m['B_signed_max_abs_online_total_pv']}, "
              f"perm_median={m['signed_sum_permuted_median_pv']}, "
              f"offline={m['Offline_signed_sum_upper_bound_total_pv']}, "
              f"gap_to_offline={m['online_gap_to_offline']}, "
              f"improv_room={m['offline_improvement_room']}")

# =============================================================================
# 3. Seed stratification
# =============================================================================
valid = {s: m for s, m in per_seed.items() if m is not None}
valid_seeds = sorted(valid.keys())

for seed in valid_seeds:
    m = per_seed[seed]
    oir = m["offline_improvement_room"]
    m["high_ceiling_seed"] = oir >= 12
    m["medium_ceiling_seed"] = 8 <= oir < 12
    m["low_ceiling_seed"] = oir < 8
    m["mapping_advantage_present"] = m["signed_sum_beats_permuted_median"]
    m["mapping_advantage_absent"] = not m["signed_sum_beats_permuted_median"]
    m["online_learning_gap_large"] = m["online_gap_to_offline"] >= 5

# =============================================================================
# 4. Aggregate metrics
# =============================================================================
def mean_of(field):
    vals = [m[field] for m in valid.values()]
    return round(float(np.mean(vals)), 4)

high_ceiling_seeds_list = [s for s in valid_seeds if per_seed[s]["high_ceiling_seed"]]
medium_ceiling_seeds_list = [s for s in valid_seeds if per_seed[s]["medium_ceiling_seed"]]
low_ceiling_seeds_list = [s for s in valid_seeds if per_seed[s]["low_ceiling_seed"]]

high_ceiling_beats_perm = sum(1 for s in high_ceiling_seeds_list if per_seed[s]["signed_sum_beats_permuted_median"])
medium_ceiling_beats_perm = sum(1 for s in medium_ceiling_seeds_list if per_seed[s]["signed_sum_beats_permuted_median"])
low_ceiling_beats_perm = sum(1 for s in low_ceiling_seeds_list if per_seed[s]["signed_sum_beats_permuted_median"])

aggregate = {
    "seeds": SEEDS,
    "n_permutation_repeats": N_PERM_REPEATS,
    "n_valid_seeds": len(valid_seeds),
    "all_seeds_completed": len(valid_seeds) == len(SEEDS),
    "mean_A_total_pv": mean_of("A_total_pv"),
    "mean_B_sum_positive_online_total_pv": mean_of("B_sum_positive_online_total_pv"),
    "mean_B_signed_sum_online_total_pv": mean_of("B_signed_sum_online_total_pv"),
    "mean_B_signed_max_abs_online_total_pv": mean_of("B_signed_max_abs_online_total_pv"),
    "mean_Family_only_online_control_total_pv": mean_of("Family_only_online_control_total_pv"),
    "mean_Offline_signed_sum_upper_bound_total_pv": mean_of("Offline_signed_sum_upper_bound_total_pv"),
    "mean_signed_sum_pv_reduction_vs_A": round(mean_of("signed_sum_pv_reduction_vs_A"), 4),
    "mean_signed_max_abs_pv_reduction_vs_A": round(mean_of("signed_max_abs_pv_reduction_vs_A"), 4),
    "mean_offline_improvement_room": round(mean_of("offline_improvement_room"), 4),
    "mean_online_gap_to_offline": round(mean_of("online_gap_to_offline"), 4),
    "mean_signed_sum_macro_delta_vs_A": mean_of("B_signed_sum_online_macro_delta_vs_A"),
    "mean_signed_max_abs_macro_delta_vs_A": mean_of("B_signed_max_abs_online_macro_delta_vs_A"),
    "seeds_where_signed_sum_beats_A": sum(1 for m in valid.values() if m["signed_sum_beats_A"]),
    "seeds_where_signed_max_abs_beats_A": sum(1 for m in valid.values() if m["B_signed_max_abs_online_total_pv"] < m["A_total_pv"]),
    "seeds_where_signed_sum_beats_positive_sum": sum(1 for m in valid.values() if m["B_signed_sum_online_total_pv"] < m["B_sum_positive_online_total_pv"]),
    "seeds_where_signed_sum_beats_family_only": sum(1 for m in valid.values() if m["B_signed_sum_online_total_pv"] < m["Family_only_online_control_total_pv"]),
    "seeds_where_signed_sum_beats_permuted_median": sum(1 for m in valid.values() if m["signed_sum_beats_permuted_median"]),
    "seeds_where_signed_sum_ties_permuted_median": sum(1 for m in valid.values() if m["signed_sum_ties_permuted_median"]),
    "seeds_where_signed_sum_loses_to_permuted_median": sum(1 for m in valid.values() if m["signed_sum_loses_to_permuted_median"]),
    "seeds_where_signed_max_abs_beats_permuted_median": sum(1 for m in valid.values() if m["B_signed_max_abs_online_total_pv"] < m["signed_max_abs_permuted_median_pv"]),
    "seeds_with_no_oracle_leakage_confirmed": sum(1 for m in valid.values() if m["no_oracle_leakage_confirmed"]),
    "seeds_with_shared_positions_used": sum(1 for m in valid.values() if m["shared_positions_used"]),
    "seeds_with_hard_exclusion_false": sum(1 for m in valid.values() if not m["hard_exclusion_used"]),
    "seeds_with_deceptive_family_preservation_valid": sum(1 for m in valid.values() if m["deceptive_family_preservation_valid"]),
    # Stratification counts
    "high_ceiling_seed_count": len(high_ceiling_seeds_list),
    "medium_ceiling_seed_count": len(medium_ceiling_seeds_list),
    "low_ceiling_seed_count": len(low_ceiling_seeds_list),
    "high_ceiling_seeds_where_signed_sum_beats_permuted_median": high_ceiling_beats_perm,
    "medium_ceiling_seeds_where_signed_sum_beats_permuted_median": medium_ceiling_beats_perm,
    "low_ceiling_seeds_where_signed_sum_beats_permuted_median": low_ceiling_beats_perm,
}

# =============================================================================
# 5. Boolean flags
# =============================================================================
all_completed = aggregate["all_seeds_completed"]
all_shared = aggregate["seeds_with_shared_positions_used"] == len(valid_seeds)
all_online_stepwise = all(m["permuted_control_mode"] == "online_stepwise" for m in valid.values())
all_no_final_mem = all(not m["final_memory_permutation_used"] for m in valid.values())
all_no_leakage = aggregate["seeds_with_no_oracle_leakage_confirmed"] == len(valid_seeds)
all_no_hard_excl = aggregate["seeds_with_hard_exclusion_false"] == len(valid_seeds)
all_family_ok = aggregate["seeds_with_deceptive_family_preservation_valid"] == len(valid_seeds)

ss_beats_A_all = aggregate["seeds_where_signed_sum_beats_A"] == len(valid_seeds)
sm_beats_A_all = aggregate["seeds_where_signed_max_abs_beats_A"] == len(valid_seeds)
ss_beats_pos_all = aggregate["seeds_where_signed_sum_beats_positive_sum"] == len(valid_seeds)
ss_beats_fam_majority = aggregate["seeds_where_signed_sum_beats_family_only"] >= len(valid_seeds) / 2

# signed_sum_beats_permuted_median_majority: at least 3/5 seeds
ss_beats_perm_majority = aggregate["seeds_where_signed_sum_beats_permuted_median"] >= 3
ss_beats_perm_all = aggregate["seeds_where_signed_sum_beats_permuted_median"] == len(valid_seeds)
sm_beats_perm_all = aggregate["seeds_where_signed_max_abs_beats_permuted_median"] == len(valid_seeds)

# signed_sum_beats_permuted_median_high_ceiling_seeds
if len(high_ceiling_seeds_list) > 0:
    ss_beats_perm_high_ceiling = all(per_seed[s]["signed_sum_beats_permuted_median"] for s in high_ceiling_seeds_list)
    no_high_ceiling_seeds_flag = False
else:
    ss_beats_perm_high_ceiling = False
    no_high_ceiling_seeds_flag = True

# Macro tradeoff
ss_macro_tradeoff_ok = (
    aggregate["mean_signed_sum_macro_delta_vs_A"] >= -0.01
    and all(m["B_signed_sum_online_macro_delta_vs_A"] >= -0.02 for m in valid.values())
)
sm_macro_tradeoff_ok = (
    aggregate["mean_signed_max_abs_macro_delta_vs_A"] >= -0.01
    and all(m["B_signed_max_abs_online_macro_delta_vs_A"] >= -0.02 for m in valid.values())
)

# Weak seed pattern: at least one seed with mapping_advantage_absent AND online_learning_gap_large
weak_seed_pattern = any(
    m["mapping_advantage_absent"] and m["online_learning_gap_large"]
    for m in valid.values()
)

# Online-offline gap
online_offline_gap_detected = aggregate["mean_online_gap_to_offline"] >= 5

# Stratified robustness criteria
stratified_robustness_met = (
    all_completed
    and all_no_leakage
    and all_online_stepwise
    and all_no_hard_excl
    and all_family_ok
    and ss_beats_A_all
    and ss_beats_perm_majority
    and ss_macro_tradeoff_ok
)

# Implementation status and failure reason
if stratified_robustness_met:
    impl_status = "pass"
    fail_reason = "none"
else:
    impl_status = "partial"
    missing = []
    if not all_completed: missing.append("not_all_seeds_completed")
    if not all_no_leakage: missing.append("oracle_leakage_detected")
    if not all_online_stepwise: missing.append("permuted_not_online_stepwise")
    if not all_no_hard_excl: missing.append("hard_exclusion_used")
    if not all_family_ok: missing.append("family_preservation_failed")
    if not ss_beats_A_all: missing.append("signed_sum_does_not_beat_A_in_all_seeds")
    if not ss_beats_perm_majority: missing.append("signed_sum_does_not_beat_permuted_median_majority")
    if not ss_macro_tradeoff_ok: missing.append("macro_tradeoff_not_acceptable")
    if no_high_ceiling_seeds_flag: missing.append("no_high_ceiling_seeds")
    fail_reason = ",".join(missing) if missing else "unknown"

boolean_flags = {
    "all_seeds_completed": {
        "value": all_completed,
        "rule": f"all {len(SEEDS)} seeds completed successfully",
        "supporting_values": {"n_valid": len(valid_seeds), "n_expected": len(SEEDS)},
    },
    "all_seeds_shared_positions_used": {
        "value": all_shared,
        "rule": "all valid seeds used shared episode positions",
        "supporting_values": {"count": aggregate["seeds_with_shared_positions_used"], "total": len(valid_seeds)},
    },
    "all_seeds_online_stepwise_permuted": {
        "value": all_online_stepwise,
        "rule": "all valid seeds used online_stepwise permuted control",
        "supporting_values": {},
    },
    "all_seeds_no_final_memory_permutation": {
        "value": all_no_final_mem,
        "rule": "no seed used frozen-final-memory permutation",
        "supporting_values": {},
    },
    "all_seeds_no_oracle_leakage": {
        "value": all_no_leakage,
        "rule": "all valid seeds confirmed no oracle leakage",
        "supporting_values": {"count": aggregate["seeds_with_no_oracle_leakage_confirmed"], "total": len(valid_seeds)},
    },
    "all_seeds_hard_exclusion_false": {
        "value": all_no_hard_excl,
        "rule": "no seed used hard exclusion",
        "supporting_values": {"count": aggregate["seeds_with_hard_exclusion_false"], "total": len(valid_seeds)},
    },
    "all_seeds_family_preservation_valid": {
        "value": all_family_ok,
        "rule": "all valid seeds preserved deceptive family identity",
        "supporting_values": {"count": aggregate["seeds_with_deceptive_family_preservation_valid"], "total": len(valid_seeds)},
    },
    "signed_sum_beats_A_all_seeds": {
        "value": ss_beats_A_all,
        "rule": "B_signed_sum_online PV < A PV in all seeds",
        "supporting_values": {"count": aggregate["seeds_where_signed_sum_beats_A"], "total": len(valid_seeds)},
    },
    "signed_sum_beats_permuted_median_majority": {
        "value": ss_beats_perm_majority,
        "rule": "signed_sum beats permuted median in at least 3/5 seeds",
        "supporting_values": {
            "count": aggregate["seeds_where_signed_sum_beats_permuted_median"],
            "total": len(valid_seeds),
            "threshold": 3,
        },
    },
    "signed_sum_beats_permuted_median_high_ceiling_seeds": {
        "value": ss_beats_perm_high_ceiling,
        "rule": "among high_ceiling seeds, signed_sum beats permuted median in all such seeds",
        "supporting_values": {
            "high_ceiling_seeds": high_ceiling_seeds_list,
            "n_high_ceiling": len(high_ceiling_seeds_list),
            "n_beat_perm": high_ceiling_beats_perm,
        },
    },
    "signed_sum_macro_tradeoff_acceptable": {
        "value": ss_macro_tradeoff_ok,
        "rule": "mean_macro_delta >= -0.01 and no seed < -0.02",
        "supporting_values": {
            "mean_delta": aggregate["mean_signed_sum_macro_delta_vs_A"],
            "min_delta": min(m["B_signed_sum_online_macro_delta_vs_A"] for m in valid.values()) if valid else 0,
        },
    },
    "weak_seed_pattern_detected": {
        "value": weak_seed_pattern,
        "rule": "at least one seed has mapping_advantage_absent=true and online_learning_gap_large=true",
        "supporting_values": {
            "seeds_with_pattern": [s for s in valid_seeds
                if per_seed[s]["mapping_advantage_absent"] and per_seed[s]["online_learning_gap_large"]],
        },
    },
    "online_offline_gap_detected": {
        "value": online_offline_gap_detected,
        "rule": "mean_online_gap_to_offline >= 5",
        "supporting_values": {"mean_online_gap_to_offline": aggregate["mean_online_gap_to_offline"]},
    },
    "stratified_robustness_criteria_met": {
        "value": stratified_robustness_met,
        "rule": "all_seeds_completed and no_leakage and online_stepwise and no_hard_excl and family_ok and signed_sum_beats_A_all and beats_permuted_median_majority and macro_tradeoff_acceptable",
        "supporting_values": {
            "all_completed": all_completed,
            "no_leakage": all_no_leakage,
            "online_stepwise": all_online_stepwise,
            "no_hard_excl": all_no_hard_excl,
            "family_ok": all_family_ok,
            "ss_beats_A_all": ss_beats_A_all,
            "ss_beats_perm_majority": ss_beats_perm_majority,
            "ss_macro_tradeoff_ok": ss_macro_tradeoff_ok,
        },
    },
    "implementation_status": {
        "value": impl_status,
        "rule": "pass if all stratified criteria met, partial otherwise",
        "supporting_values": {},
    },
    "failure_reason": {
        "value": fail_reason,
        "rule": "derived from stratified multiseed validation results",
        "supporting_values": {},
    },
}

# =============================================================================
# 6. Output
# =============================================================================
elapsed = time.time() - t0
print(f"\n  elapsed={elapsed:.1f}s  writing outputs...")

output = {
    "block_id": "1J27",
    "seeds": SEEDS,
    "n_permutation_repeats": N_PERM_REPEATS,
    "per_seed": per_seed,
    "seed_stratification": {
        s: {
            "high_ceiling_seed": per_seed[s]["high_ceiling_seed"],
            "medium_ceiling_seed": per_seed[s]["medium_ceiling_seed"],
            "low_ceiling_seed": per_seed[s]["low_ceiling_seed"],
            "mapping_advantage_present": per_seed[s]["mapping_advantage_present"],
            "mapping_advantage_absent": per_seed[s]["mapping_advantage_absent"],
            "online_learning_gap_large": per_seed[s]["online_learning_gap_large"],
            "offline_improvement_room": per_seed[s]["offline_improvement_room"],
            "online_gap_to_offline": per_seed[s]["online_gap_to_offline"],
        }
        for s in valid_seeds
    },
    "aggregate": aggregate,
    "boolean_flags": boolean_flags,
    "elapsed_seconds": round(elapsed, 1),
}

out_json_path = os.path.join(CURRENT_DIR, "runs",
    "block1j27_stratified_multiseed_online_signed_memory.json")
os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
with open(out_json_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"  JSON -> {out_json_path}")

# =============================================================================
# 7. Markdown Protocol
# =============================================================================
md_lines = []
md_lines.append("# Block 1J27 -- Stratified Multiseed Online Signed Memory Robustness")
md_lines.append("")
md_lines.append("## 1. Objective")
md_lines.append("")
md_lines.append("Test whether the 1J24b online signed memory result is stable across multiple seeds with ceiling stratification.")
md_lines.append("")
md_lines.append("## 2. Setup")
md_lines.append("")
md_lines.append(f"- Seeds: {SEEDS}")
md_lines.append(f"- Permutation repeats per seed: {N_PERM_REPEATS}")
md_lines.append(f"- Valid seeds: {len(valid_seeds)}/{len(SEEDS)}")
md_lines.append("")

# Per-seed table
md_lines.append("## 3. Per-Seed Results")
md_lines.append("")
header = "| Seed | A PV | SumPos PV | SignedSum PV | SignedMax PV | FamOnly PV | Offline PV | SS Perm Median | SM Perm Median | SS %ile | SM %ile | SS Macro D | SM Macro D | Reduc vs A | Improv Room | Gap to Off | Beats A | Beats Perm | Ties Perm |"
md_lines.append(header)
md_lines.append("|------|------|-----------|-------------|-------------|------------|------------|----------------|----------------|---------|---------|------------|------------|------------|-------------|------------|---------|-----------|-----------|")
for seed in SEEDS:
    m = per_seed.get(seed)
    if m:
        beats_a = "Y" if m["signed_sum_beats_A"] else "N"
        beats_perm = "Y" if m["signed_sum_beats_permuted_median"] else "N"
        ties_perm = "Y" if m["signed_sum_ties_permuted_median"] else "N"
        md_lines.append(f"| {seed} | {m['A_total_pv']} | {m['B_sum_positive_online_total_pv']} | "
                       f"{m['B_signed_sum_online_total_pv']} | {m['B_signed_max_abs_online_total_pv']} | "
                       f"{m['Family_only_online_control_total_pv']} | {m['Offline_signed_sum_upper_bound_total_pv']} | "
                       f"{m['signed_sum_permuted_median_pv']} | {m['signed_max_abs_permuted_median_pv']} | "
                       f"{m['signed_sum_real_percentile_against_permuted']:.3f} | {m['signed_max_abs_real_percentile_against_permuted']:.3f} | "
                       f"{m['B_signed_sum_online_macro_delta_vs_A']:.4f} | {m['B_signed_max_abs_online_macro_delta_vs_A']:.4f} | "
                       f"{m['signed_sum_pv_reduction_vs_A']} | {m['offline_improvement_room']} | "
                       f"{m['online_gap_to_offline']} | {beats_a} | {beats_perm} | {ties_perm} |")
    else:
        md_lines.append(f"| {seed} | FAILED | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |")
md_lines.append("")

# Seed stratification table
md_lines.append("## 4. Seed Stratification")
md_lines.append("")
md_lines.append("| Seed | Ceiling | Offline Improv Room | Online Gap to Offline | Mapping Advantage | Gap Large |")
md_lines.append("|------|---------|---------------------|----------------------|-------------------|-----------|")
for seed in SEEDS:
    m = per_seed.get(seed)
    if m:
        if m["high_ceiling_seed"]:
            ceiling = "high"
        elif m["medium_ceiling_seed"]:
            ceiling = "medium"
        else:
            ceiling = "low"
        mapping = "present" if m["mapping_advantage_present"] else "absent"
        gap_large = "Y" if m["online_learning_gap_large"] else "N"
        md_lines.append(f"| {seed} | {ceiling} | {m['offline_improvement_room']} | "
                       f"{m['online_gap_to_offline']} | {mapping} | {gap_large} |")
    else:
        md_lines.append(f"| {seed} | FAILED | - | - | - | - |")
md_lines.append("")

# Aggregate
md_lines.append("## 5. Aggregate Metrics")
md_lines.append("")
md_lines.append("| Metric | Value |")
md_lines.append("|--------|-------|")
agg_lines = [
    ("n_valid_seeds", aggregate["n_valid_seeds"]),
    ("mean_A_total_pv", aggregate["mean_A_total_pv"]),
    ("mean_B_sum_positive_online_total_pv", aggregate["mean_B_sum_positive_online_total_pv"]),
    ("mean_B_signed_sum_online_total_pv", aggregate["mean_B_signed_sum_online_total_pv"]),
    ("mean_B_signed_max_abs_online_total_pv", aggregate["mean_B_signed_max_abs_online_total_pv"]),
    ("mean_Family_only_online_control_total_pv", aggregate["mean_Family_only_online_control_total_pv"]),
    ("mean_Offline_signed_sum_upper_bound_total_pv", aggregate["mean_Offline_signed_sum_upper_bound_total_pv"]),
    ("mean_signed_sum_pv_reduction_vs_A", aggregate["mean_signed_sum_pv_reduction_vs_A"]),
    ("mean_signed_max_abs_pv_reduction_vs_A", aggregate["mean_signed_max_abs_pv_reduction_vs_A"]),
    ("mean_offline_improvement_room", aggregate["mean_offline_improvement_room"]),
    ("mean_online_gap_to_offline", aggregate["mean_online_gap_to_offline"]),
    ("mean_signed_sum_macro_delta_vs_A", aggregate["mean_signed_sum_macro_delta_vs_A"]),
    ("mean_signed_max_abs_macro_delta_vs_A", aggregate["mean_signed_max_abs_macro_delta_vs_A"]),
    ("seeds_where_signed_sum_beats_A", f"{aggregate['seeds_where_signed_sum_beats_A']}/{len(valid_seeds)}"),
    ("seeds_where_signed_max_abs_beats_A", f"{aggregate['seeds_where_signed_max_abs_beats_A']}/{len(valid_seeds)}"),
    ("seeds_where_signed_sum_beats_positive_sum", f"{aggregate['seeds_where_signed_sum_beats_positive_sum']}/{len(valid_seeds)}"),
    ("seeds_where_signed_sum_beats_family_only", f"{aggregate['seeds_where_signed_sum_beats_family_only']}/{len(valid_seeds)}"),
    ("seeds_where_signed_sum_beats_permuted_median", f"{aggregate['seeds_where_signed_sum_beats_permuted_median']}/{len(valid_seeds)}"),
    ("seeds_where_signed_sum_ties_permuted_median", f"{aggregate['seeds_where_signed_sum_ties_permuted_median']}/{len(valid_seeds)}"),
    ("seeds_where_signed_sum_loses_to_permuted_median", f"{aggregate['seeds_where_signed_sum_loses_to_permuted_median']}/{len(valid_seeds)}"),
    ("seeds_where_signed_max_abs_beats_permuted_median", f"{aggregate['seeds_where_signed_max_abs_beats_permuted_median']}/{len(valid_seeds)}"),
    ("high_ceiling_seed_count", aggregate["high_ceiling_seed_count"]),
    ("medium_ceiling_seed_count", aggregate["medium_ceiling_seed_count"]),
    ("low_ceiling_seed_count", aggregate["low_ceiling_seed_count"]),
    ("high_ceiling_seeds_where_signed_sum_beats_permuted_median", f"{high_ceiling_beats_perm}/{len(high_ceiling_seeds_list)}" if high_ceiling_seeds_list else "N/A"),
    ("medium_ceiling_seeds_where_signed_sum_beats_permuted_median", f"{medium_ceiling_beats_perm}/{len(medium_ceiling_seeds_list)}" if medium_ceiling_seeds_list else "N/A"),
    ("low_ceiling_seeds_where_signed_sum_beats_permuted_median", f"{low_ceiling_beats_perm}/{len(low_ceiling_seeds_list)}" if low_ceiling_seeds_list else "N/A"),
    ("seeds_with_no_oracle_leakage_confirmed", f"{aggregate['seeds_with_no_oracle_leakage_confirmed']}/{len(valid_seeds)}"),
    ("seeds_with_shared_positions_used", f"{aggregate['seeds_with_shared_positions_used']}/{len(valid_seeds)}"),
    ("seeds_with_hard_exclusion_false", f"{aggregate['seeds_with_hard_exclusion_false']}/{len(valid_seeds)}"),
    ("seeds_with_deceptive_family_preservation_valid", f"{aggregate['seeds_with_deceptive_family_preservation_valid']}/{len(valid_seeds)}"),
]
for metric, val in agg_lines:
    md_lines.append(f"| {metric} | {val} |")
md_lines.append("")

# Boolean flags
md_lines.append("## 6. Boolean Flags")
md_lines.append("")
for flag_name, flag_info in boolean_flags.items():
    md_lines.append(f"### {flag_name}: {flag_info['value']}")
    md_lines.append(f"  Rule: {flag_info['rule']}")
    if flag_info.get("supporting_values"):
        for sv_key, sv_val in flag_info["supporting_values"].items():
            md_lines.append(f"  - {sv_key}: {sv_val}")
    md_lines.append("")

# Summary
md_lines.append("## 7. Summary")
md_lines.append("")
md_lines.append("```")
md_lines.append("[block_done]")
md_lines.append(f"block_id=1J27")
md_lines.append(f"seeds={SEEDS}")
md_lines.append(f"n_permutation_repeats={N_PERM_REPEATS}")
md_lines.append(f"n_valid_seeds={len(valid_seeds)}")
md_lines.append(f"mean_A_total_pv={aggregate['mean_A_total_pv']}")
md_lines.append(f"mean_B_signed_sum_online_total_pv={aggregate['mean_B_signed_sum_online_total_pv']}")
md_lines.append(f"mean_Offline_signed_sum_upper_bound_total_pv={aggregate['mean_Offline_signed_sum_upper_bound_total_pv']}")
md_lines.append(f"mean_signed_sum_pv_reduction_vs_A={aggregate['mean_signed_sum_pv_reduction_vs_A']}")
md_lines.append(f"mean_offline_improvement_room={aggregate['mean_offline_improvement_room']}")
md_lines.append(f"mean_online_gap_to_offline={aggregate['mean_online_gap_to_offline']}")
md_lines.append(f"seeds_where_signed_sum_beats_A={aggregate['seeds_where_signed_sum_beats_A']}/{len(valid_seeds)}")
md_lines.append(f"seeds_where_signed_sum_beats_permuted_median={aggregate['seeds_where_signed_sum_beats_permuted_median']}/{len(valid_seeds)}")
md_lines.append(f"seeds_where_signed_sum_ties_permuted_median={aggregate['seeds_where_signed_sum_ties_permuted_median']}/{len(valid_seeds)}")
md_lines.append(f"high_ceiling_seed_count={aggregate['high_ceiling_seed_count']}")
md_lines.append(f"medium_ceiling_seed_count={aggregate['medium_ceiling_seed_count']}")
md_lines.append(f"low_ceiling_seed_count={aggregate['low_ceiling_seed_count']}")
md_lines.append(f"high_ceiling_seeds_where_signed_sum_beats_permuted_median={high_ceiling_beats_perm}/{len(high_ceiling_seeds_list)}" if high_ceiling_seeds_list else "high_ceiling_seeds_where_signed_sum_beats_permuted_median=N/A")
md_lines.append(f"signed_sum_macro_tradeoff_acceptable={ss_macro_tradeoff_ok}")
md_lines.append(f"weak_seed_pattern_detected={weak_seed_pattern}")
md_lines.append(f"online_offline_gap_detected={online_offline_gap_detected}")
md_lines.append(f"stratified_robustness_criteria_met={stratified_robustness_met}")
md_lines.append(f"implementation_status={impl_status}")
md_lines.append(f"failure_reason={fail_reason}")
md_lines.append(f"elapsed={elapsed:.1f}s")
md_lines.append("```")

out_md_path = os.path.join(CURRENT_DIR, "protocols",
    "block1j27_stratified_multiseed_online_signed_memory.md")
os.makedirs(os.path.dirname(out_md_path), exist_ok=True)
with open(out_md_path, "w") as f:
    f.write("\n".join(md_lines))
print(f"  MD -> {out_md_path}")

print("\n" + "=" * 70)
print("Block 1J27 complete.")
print(f"  seeds={SEEDS}  n_valid={len(valid_seeds)}/{len(SEEDS)}")
print(f"  signed_sum_beats_A_all={ss_beats_A_all}")
print(f"  signed_sum_beats_permuted_median_majority={ss_beats_perm_majority}")
print(f"  high_ceiling={len(high_ceiling_seeds_list)}  medium_ceiling={len(medium_ceiling_seeds_list)}  low_ceiling={len(low_ceiling_seeds_list)}")
print(f"  weak_seed_pattern_detected={weak_seed_pattern}")
print(f"  online_offline_gap_detected={online_offline_gap_detected}")
print(f"  stratified_robustness_criteria_met={stratified_robustness_met}")
print(f"  implementation_status={impl_status}")
print("=" * 70)
