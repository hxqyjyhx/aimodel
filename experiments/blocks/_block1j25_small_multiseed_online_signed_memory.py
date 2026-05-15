"""
Block 1J25 -- Small Multiseed Robustness for Online Signed Risk/Protection Memory.

Calls the CLI-patched 1J24b script for each seed [101, 103, 107],
then aggregates results across seeds.
"""
import os, sys, json, subprocess, time
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()

SEEDS = [101, 103, 107]
N_PERM_REPEATS = 10

# =============================================================================
# 1. Run 1J24b for each seed
# =============================================================================
print("=" * 70)
print("Block 1J25 -- Small Multiseed Online Signed Memory Robustness")
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
# 2. Extract per-seed metrics
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

    return {
        "seed": data.get("smoke_seed", "?"),
        "A_total_pv": a_pv,
        "B_sum_positive_online_total_pv": sum_pos_pv,
        "B_signed_sum_online_total_pv": signed_sum_pv,
        "B_signed_max_abs_online_total_pv": signed_max_pv,
        "Family_only_online_control_total_pv": fam_only_pv,
        "Offline_signed_sum_upper_bound_total_pv": offline_pv,
        "signed_sum_permuted_median_pv": ss_dist.get("median_total_pv", 0),
        "signed_max_abs_permuted_median_pv": sm_dist.get("median_total_pv", 0),
        "signed_sum_real_percentile_against_permuted": ss_dist.get("real_percentile_against_permuted", 0),
        "signed_max_abs_real_percentile_against_permuted": sm_dist.get("real_percentile_against_permuted", 0),
        "B_signed_sum_online_macro_delta_vs_A": signed_sum_macro_delta,
        "B_signed_max_abs_online_macro_delta_vs_A": signed_max_macro_delta,
        "shared_positions_used": data.get("shared_positions_used", False),
        "permuted_control_mode": data.get("permuted_control_mode", ""),
        "final_memory_permutation_used": data.get("final_memory_permutation_used", True),
        "deterministic_permutation_seed_used": data.get("deterministic_permutation_seed_used", False),
        "no_oracle_leakage_confirmed": flags.get("no_oracle_leakage_confirmed", {}).get("value", False),
        "hard_exclusion_used": flags.get("hard_exclusion_used", {}).get("value", True),
        "deceptive_family_preservation_valid": flags.get("deceptive_family_preservation_valid", {}).get("value", False),
        "implementation_status": flags.get("implementation_status", {}).get("value", "?"),
        "failure_reason": flags.get("failure_reason", {}).get("value", "?"),
    }

per_seed = {}
for seed in SEEDS:
    m = extract_seed_metrics(seed_outputs.get(seed))
    per_seed[seed] = m
    if m:
        print(f"  seed={seed}: A={m['A_total_pv']}, signed_sum={m['B_signed_sum_online_total_pv']}, "
              f"signed_max={m['B_signed_max_abs_online_total_pv']}, "
              f"perm_median={m['signed_sum_permuted_median_pv']}, "
              f"offline={m['Offline_signed_sum_upper_bound_total_pv']}")

# =============================================================================
# 3. Aggregate metrics
# =============================================================================
valid = {s: m for s, m in per_seed.items() if m is not None}
valid_seeds = sorted(valid.keys())

def mean_of(field):
    vals = [m[field] for m in valid.values()]
    return round(float(np.mean(vals)), 4)

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
    "mean_signed_sum_pv_reduction_vs_A": mean_of("B_signed_sum_online_total_pv") - mean_of("A_total_pv"),
    "mean_signed_max_abs_pv_reduction_vs_A": mean_of("B_signed_max_abs_online_total_pv") - mean_of("A_total_pv"),
    "mean_signed_sum_macro_delta_vs_A": mean_of("B_signed_sum_online_macro_delta_vs_A"),
    "mean_signed_max_abs_macro_delta_vs_A": mean_of("B_signed_max_abs_online_macro_delta_vs_A"),
    "seeds_where_signed_sum_beats_A": sum(1 for m in valid.values() if m["B_signed_sum_online_total_pv"] < m["A_total_pv"]),
    "seeds_where_signed_max_abs_beats_A": sum(1 for m in valid.values() if m["B_signed_max_abs_online_total_pv"] < m["A_total_pv"]),
    "seeds_where_signed_sum_beats_positive_sum": sum(1 for m in valid.values() if m["B_signed_sum_online_total_pv"] < m["B_sum_positive_online_total_pv"]),
    "seeds_where_signed_sum_beats_family_only": sum(1 for m in valid.values() if m["B_signed_sum_online_total_pv"] < m["Family_only_online_control_total_pv"]),
    "seeds_where_signed_sum_beats_permuted_median": sum(1 for m in valid.values() if m["B_signed_sum_online_total_pv"] < m["signed_sum_permuted_median_pv"]),
    "seeds_where_signed_max_abs_beats_permuted_median": sum(1 for m in valid.values() if m["B_signed_max_abs_online_total_pv"] < m["signed_max_abs_permuted_median_pv"]),
    "seeds_with_no_oracle_leakage_confirmed": sum(1 for m in valid.values() if m["no_oracle_leakage_confirmed"]),
    "seeds_with_shared_positions_used": sum(1 for m in valid.values() if m["shared_positions_used"]),
    "seeds_with_hard_exclusion_false": sum(1 for m in valid.values() if not m["hard_exclusion_used"]),
    "seeds_with_deceptive_family_preservation_valid": sum(1 for m in valid.values() if m["deceptive_family_preservation_valid"]),
}

# =============================================================================
# 4. Boolean flags
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
ss_beats_perm_all = aggregate["seeds_where_signed_sum_beats_permuted_median"] == len(valid_seeds)
sm_beats_perm_all = aggregate["seeds_where_signed_max_abs_beats_permuted_median"] == len(valid_seeds)

# Macro tradeoff: mean >= -0.01, no seed < -0.02
ss_macro_tradeoff_ok = (
    aggregate["mean_signed_sum_macro_delta_vs_A"] >= -0.01
    and all(m["B_signed_sum_online_macro_delta_vs_A"] >= -0.02 for m in valid.values())
)
sm_macro_tradeoff_ok = (
    aggregate["mean_signed_max_abs_macro_delta_vs_A"] >= -0.01
    and all(m["B_signed_max_abs_online_macro_delta_vs_A"] >= -0.02 for m in valid.values())
)

# Robustness: at least one signed variant passes all criteria
ss_robust = ss_beats_A_all and ss_beats_perm_all and ss_macro_tradeoff_ok
sm_robust = sm_beats_A_all and sm_beats_perm_all and sm_macro_tradeoff_ok
robustness_met = all_completed and all_no_leakage and all_online_stepwise and all_no_hard_excl and all_family_ok and (ss_robust or sm_robust)

if all_completed and all_no_leakage and all_online_stepwise and all_no_hard_excl and all_family_ok:
    if ss_robust or sm_robust:
        impl_status = "pass"
        fail_reason = "none"
    else:
        impl_status = "partial"
        fail_reason = "robustness_criteria_not_met"
else:
    impl_status = "partial"
    missing = []
    if not all_completed: missing.append("not_all_seeds_completed")
    if not all_no_leakage: missing.append("oracle_leakage_detected")
    if not all_online_stepwise: missing.append("permuted_not_online_stepwise")
    if not all_no_hard_excl: missing.append("hard_exclusion_used")
    if not all_family_ok: missing.append("family_preservation_failed")
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
    "signed_max_abs_beats_A_all_seeds": {
        "value": sm_beats_A_all,
        "rule": "B_signed_max_abs_online PV < A PV in all seeds",
        "supporting_values": {"count": aggregate["seeds_where_signed_max_abs_beats_A"], "total": len(valid_seeds)},
    },
    "signed_sum_beats_positive_sum_all_seeds": {
        "value": ss_beats_pos_all,
        "rule": "B_signed_sum_online PV < sum_positive_online PV in all seeds",
        "supporting_values": {"count": aggregate["seeds_where_signed_sum_beats_positive_sum"], "total": len(valid_seeds)},
    },
    "signed_sum_beats_family_only_majority": {
        "value": ss_beats_fam_majority,
        "rule": "signed_sum beats family_only in majority of seeds",
        "supporting_values": {"count": aggregate["seeds_where_signed_sum_beats_family_only"], "total": len(valid_seeds)},
    },
    "signed_sum_beats_permuted_median_all_seeds": {
        "value": ss_beats_perm_all,
        "rule": "signed_sum PV < permuted median PV in all seeds",
        "supporting_values": {"count": aggregate["seeds_where_signed_sum_beats_permuted_median"], "total": len(valid_seeds)},
    },
    "signed_max_abs_beats_permuted_median_all_seeds": {
        "value": sm_beats_perm_all,
        "rule": "signed_max_abs PV < permuted median PV in all seeds",
        "supporting_values": {"count": aggregate["seeds_where_signed_max_abs_beats_permuted_median"], "total": len(valid_seeds)},
    },
    "signed_sum_macro_tradeoff_acceptable": {
        "value": ss_macro_tradeoff_ok,
        "rule": "mean_macro_delta >= -0.01 and no seed < -0.02",
        "supporting_values": {
            "mean_delta": aggregate["mean_signed_sum_macro_delta_vs_A"],
            "min_delta": min(m["B_signed_sum_online_macro_delta_vs_A"] for m in valid.values()) if valid else 0,
        },
    },
    "signed_max_abs_macro_tradeoff_acceptable": {
        "value": sm_macro_tradeoff_ok,
        "rule": "mean_macro_delta >= -0.01 and no seed < -0.02",
        "supporting_values": {
            "mean_delta": aggregate["mean_signed_max_abs_macro_delta_vs_A"],
            "min_delta": min(m["B_signed_max_abs_online_macro_delta_vs_A"] for m in valid.values()) if valid else 0,
        },
    },
    "robustness_criteria_met": {
        "value": robustness_met,
        "rule": "all_seeds_completed and no_leakage and online_stepwise and no_hard_excl and family_ok and at least one signed variant passes all criteria",
        "supporting_values": {"signed_sum_robust": ss_robust, "signed_max_abs_robust": sm_robust},
    },
    "implementation_status": {
        "value": impl_status,
        "rule": "pass if all criteria met, partial otherwise",
        "supporting_values": {},
    },
    "failure_reason": {
        "value": fail_reason,
        "rule": "derived from multiseed validation results",
        "supporting_values": {},
    },
}

# =============================================================================
# 5. Output
# =============================================================================
elapsed = time.time() - t0
print(f"\n  elapsed={elapsed:.1f}s  writing outputs...")

output = {
    "block_id": "1J25",
    "seeds": SEEDS,
    "n_permutation_repeats": N_PERM_REPEATS,
    "per_seed": per_seed,
    "aggregate": aggregate,
    "boolean_flags": boolean_flags,
    "elapsed_seconds": round(elapsed, 1),
}

out_json_path = os.path.join(CURRENT_DIR, "runs",
    "block1j25_small_multiseed_online_signed_memory.json")
os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
with open(out_json_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"  JSON -> {out_json_path}")

# =============================================================================
# 6. Markdown Protocol
# =============================================================================
md_lines = []
md_lines.append("# Block 1J25 -- Small Multiseed Online Signed Memory Robustness")
md_lines.append("")
md_lines.append("## 1. Objective")
md_lines.append("")
md_lines.append("Test whether the 1J24b online signed memory result is stable across multiple seeds.")
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
md_lines.append("| Seed | A PV | SumPos PV | SignedSum PV | SignedMax PV | FamOnly PV | Offline PV | SS Perm Median | SM Perm Median | SS %ile | SM %ile | SS Macro D | SM Macro D |")
md_lines.append("|------|------|-----------|-------------|-------------|------------|------------|----------------|----------------|---------|---------|------------|------------|")
for seed in SEEDS:
    m = per_seed.get(seed)
    if m:
        md_lines.append(f"| {seed} | {m['A_total_pv']} | {m['B_sum_positive_online_total_pv']} | "
                       f"{m['B_signed_sum_online_total_pv']} | {m['B_signed_max_abs_online_total_pv']} | "
                       f"{m['Family_only_online_control_total_pv']} | {m['Offline_signed_sum_upper_bound_total_pv']} | "
                       f"{m['signed_sum_permuted_median_pv']} | {m['signed_max_abs_permuted_median_pv']} | "
                       f"{m['signed_sum_real_percentile_against_permuted']:.3f} | {m['signed_max_abs_real_percentile_against_permuted']:.3f} | "
                       f"{m['B_signed_sum_online_macro_delta_vs_A']:.4f} | {m['B_signed_max_abs_online_macro_delta_vs_A']:.4f} |")
    else:
        md_lines.append(f"| {seed} | FAILED | - | - | - | - | - | - | - | - | - | - | - |")
md_lines.append("")

# Aggregate
md_lines.append("## 4. Aggregate Metrics")
md_lines.append("")
md_lines.append("| Metric | Value |")
md_lines.append("|--------|-------|")
agg_lines = [
    ("mean_A_total_pv", aggregate["mean_A_total_pv"]),
    ("mean_B_sum_positive_online_total_pv", aggregate["mean_B_sum_positive_online_total_pv"]),
    ("mean_B_signed_sum_online_total_pv", aggregate["mean_B_signed_sum_online_total_pv"]),
    ("mean_B_signed_max_abs_online_total_pv", aggregate["mean_B_signed_max_abs_online_total_pv"]),
    ("mean_Family_only_online_control_total_pv", aggregate["mean_Family_only_online_control_total_pv"]),
    ("mean_Offline_signed_sum_upper_bound_total_pv", aggregate["mean_Offline_signed_sum_upper_bound_total_pv"]),
    ("mean_signed_sum_pv_reduction_vs_A", aggregate["mean_signed_sum_pv_reduction_vs_A"]),
    ("mean_signed_max_abs_pv_reduction_vs_A", aggregate["mean_signed_max_abs_pv_reduction_vs_A"]),
    ("mean_signed_sum_macro_delta_vs_A", aggregate["mean_signed_sum_macro_delta_vs_A"]),
    ("mean_signed_max_abs_macro_delta_vs_A", aggregate["mean_signed_max_abs_macro_delta_vs_A"]),
    ("seeds_where_signed_sum_beats_A", f"{aggregate['seeds_where_signed_sum_beats_A']}/{len(valid_seeds)}"),
    ("seeds_where_signed_max_abs_beats_A", f"{aggregate['seeds_where_signed_max_abs_beats_A']}/{len(valid_seeds)}"),
    ("seeds_where_signed_sum_beats_positive_sum", f"{aggregate['seeds_where_signed_sum_beats_positive_sum']}/{len(valid_seeds)}"),
    ("seeds_where_signed_sum_beats_family_only", f"{aggregate['seeds_where_signed_sum_beats_family_only']}/{len(valid_seeds)}"),
    ("seeds_where_signed_sum_beats_permuted_median", f"{aggregate['seeds_where_signed_sum_beats_permuted_median']}/{len(valid_seeds)}"),
    ("seeds_where_signed_max_abs_beats_permuted_median", f"{aggregate['seeds_where_signed_max_abs_beats_permuted_median']}/{len(valid_seeds)}"),
    ("seeds_with_no_oracle_leakage_confirmed", f"{aggregate['seeds_with_no_oracle_leakage_confirmed']}/{len(valid_seeds)}"),
    ("seeds_with_shared_positions_used", f"{aggregate['seeds_with_shared_positions_used']}/{len(valid_seeds)}"),
    ("seeds_with_hard_exclusion_false", f"{aggregate['seeds_with_hard_exclusion_false']}/{len(valid_seeds)}"),
    ("seeds_with_deceptive_family_preservation_valid", f"{aggregate['seeds_with_deceptive_family_preservation_valid']}/{len(valid_seeds)}"),
]
for metric, val in agg_lines:
    md_lines.append(f"| {metric} | {val} |")
md_lines.append("")

# Boolean flags
md_lines.append("## 5. Boolean Flags")
md_lines.append("")
for flag_name, flag_info in boolean_flags.items():
    md_lines.append(f"### {flag_name}: {flag_info['value']}")
    md_lines.append(f"  Rule: {flag_info['rule']}")
    if flag_info.get("supporting_values"):
        for sv_key, sv_val in flag_info["supporting_values"].items():
            md_lines.append(f"  - {sv_key}: {sv_val}")
    md_lines.append("")

# Summary
md_lines.append("## 6. Summary")
md_lines.append("")
md_lines.append("```")
md_lines.append("[block_done]")
md_lines.append(f"block_id=1J25")
md_lines.append(f"seeds={SEEDS}")
md_lines.append(f"n_permutation_repeats={N_PERM_REPEATS}")
md_lines.append(f"mean_A_total_pv={aggregate['mean_A_total_pv']}")
md_lines.append(f"mean_B_sum_positive_online_total_pv={aggregate['mean_B_sum_positive_online_total_pv']}")
md_lines.append(f"mean_B_signed_sum_online_total_pv={aggregate['mean_B_signed_sum_online_total_pv']}")
md_lines.append(f"mean_B_signed_max_abs_online_total_pv={aggregate['mean_B_signed_max_abs_online_total_pv']}")
md_lines.append(f"mean_Family_only_online_control_total_pv={aggregate['mean_Family_only_online_control_total_pv']}")
md_lines.append(f"mean_Offline_signed_sum_upper_bound_total_pv={aggregate['mean_Offline_signed_sum_upper_bound_total_pv']}")
md_lines.append(f"seeds_where_signed_sum_beats_A={aggregate['seeds_where_signed_sum_beats_A']}/{len(valid_seeds)}")
md_lines.append(f"seeds_where_signed_max_abs_beats_A={aggregate['seeds_where_signed_max_abs_beats_A']}/{len(valid_seeds)}")
md_lines.append(f"seeds_where_signed_sum_beats_permuted_median={aggregate['seeds_where_signed_sum_beats_permuted_median']}/{len(valid_seeds)}")
md_lines.append(f"seeds_where_signed_max_abs_beats_permuted_median={aggregate['seeds_where_signed_max_abs_beats_permuted_median']}/{len(valid_seeds)}")
md_lines.append(f"mean_signed_sum_macro_delta_vs_A={aggregate['mean_signed_sum_macro_delta_vs_A']}")
md_lines.append(f"mean_signed_max_abs_macro_delta_vs_A={aggregate['mean_signed_max_abs_macro_delta_vs_A']}")
md_lines.append(f"all_seeds_no_oracle_leakage={all_no_leakage}")
md_lines.append(f"all_seeds_shared_positions_used={all_shared}")
md_lines.append(f"all_seeds_online_stepwise_permuted={all_online_stepwise}")
md_lines.append(f"all_seeds_no_final_memory_permutation={all_no_final_mem}")
md_lines.append(f"signed_sum_macro_tradeoff_acceptable={ss_macro_tradeoff_ok}")
md_lines.append(f"signed_max_abs_macro_tradeoff_acceptable={sm_macro_tradeoff_ok}")
md_lines.append(f"robustness_criteria_met={robustness_met}")
md_lines.append(f"implementation_status={impl_status}")
md_lines.append(f"failure_reason={fail_reason}")
md_lines.append(f"elapsed={elapsed:.1f}s")
md_lines.append("```")

out_md_path = os.path.join(CURRENT_DIR, "protocols",
    "block1j25_small_multiseed_online_signed_memory.md")
os.makedirs(os.path.dirname(out_md_path), exist_ok=True)
with open(out_md_path, "w") as f:
    f.write("\n".join(md_lines))
print(f"  MD -> {out_md_path}")

print("\n" + "=" * 70)
print("Block 1J25 complete.")
print(f"  seeds={SEEDS}  n_valid={len(valid_seeds)}/{len(SEEDS)}")
print(f"  signed_sum_beats_A_all={ss_beats_A_all}  signed_max_beats_A_all={sm_beats_A_all}")
print(f"  robustness_criteria_met={robustness_met}")
print(f"  implementation_status={impl_status}")
print("=" * 70)
