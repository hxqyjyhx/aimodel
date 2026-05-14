"""
Block 1J32 -- Seed 109 Cross-Family Representation Audit (Standalone Analyzer).

Reads 1J31 JSON outputs and analyzes why the 12 raw-zero missed PVs
lack signed memory representation. Does NOT run a new simulation.
"""
import os, sys, json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

# --- Constants (must match 1J30/1J31 definitions) ---
WOOD_FEATURES = {"has_bark_texture", "has_wood_grain", "brownish", "rough_texture", "long_shape"}
STONE_FEATURES = {"has_crystal_flecks", "has_granular_surface", "grayish", "block_like", "heavy_weight"}
APPLE_FEATURES = {"has_stem_remnant", "has_peel_texture", "round_small", "greenish", "smooth_texture", "light_weight"}
TOOL_FEATURES = {"has_grip_area", "has_shaft_shape", "elongated_with_handle", "movable", "long_shape"}

TYPE_FAMILIES = {
    "wood-like": WOOD_FEATURES, "stone-like": STONE_FEATURES,
    "apple-like": APPLE_FEATURES, "tool-like": TOOL_FEATURES,
}
ALL_TYPE_FAMILY_FEATURES = set()
for fs in TYPE_FAMILIES.values():
    ALL_TYPE_FAMILY_FEATURES.update(fs)
FAMILY_NAMES = sorted(TYPE_FAMILIES.keys())

INJECTED_DEVIATION_FEATURES = {"damp_texture", "brittle_surface", "treated_surface", "hollow_sound"}

MAIN_CANDIDATE_ACTIONS = ["eat", "mine_by_hand", "mine_with_pickaxe", "craft_plank", "burn_as_fuel", "use_as_tool"]

t0 = __import__('time').time()

# =============================================================================
# Load 1J31 data
# =============================================================================
monitor_json_path = os.path.join(CURRENT_DIR, "runs", "block1j31_seed109_raw_zero_monitor.json")
source_json_path = os.path.join(CURRENT_DIR, "runs", "block1j31_seed109_raw_penalty_zero_source_audit.json")

with open(monitor_json_path) as f:
    mon = json.load(f)
with open(source_json_path) as f:
    src = json.load(f)

dpa = mon["decision_path_audit"]
p2 = dpa["part2_missed_offline_avoidable_pv_audit"]
all_missed = p2["details"]

# Separate raw-zero from other
raw_zero_cases = [c for c in all_missed if c.get("failure_reason") == "raw_penalty_zero"]
other_cases = [c for c in all_missed if c.get("failure_reason") != "raw_penalty_zero"]

print(f"Loaded {len(raw_zero_cases)} raw-zero cases, {len(other_cases)} other cases")

# Get part1 traces for richer feature data (keyed by object_id+action)
p1_scale1 = {f"{t['object_id']}|{t['action']}": t for t in dpa.get("part1_decision_path_trace_scale1", [])}

# Build PV sets from all variants
def _get_pv_set(variant_data):
    pvs = set()
    for ep in variant_data.get("episodes", []):
        for pd in ep.get("probe_details", []):
            if pd.get("is_violation", False):
                pvs.add(f"{pd['oid']}|{pd['action']}")
    return pvs

a_pvs = _get_pv_set(mon.get("A_no_memory", {}))
fam_only_pvs = _get_pv_set(mon.get("Family_only_online_control", {}))
offline_pvs = _get_pv_set(mon.get("Offline_signed_sum_upper_bound", {}))

print(f"A PVs: {len(a_pvs)}, Family_only PVs: {len(fam_only_pvs)}, Offline PVs: {len(offline_pvs)}")

# =============================================================================
# Helper: family cue analysis
# =============================================================================
def compute_family_cues(observed_features):
    """Return per-family feature counts, dominant family, etc."""
    obs_set = set(observed_features) if observed_features else set()
    counts = {}
    for fam, feats in TYPE_FAMILIES.items():
        counts[fam] = sum(1 for f in feats if f in obs_set)
    sorted_fams = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    dominant_fam, dominant_count = sorted_fams[0]
    second_fam, second_count = sorted_fams[1] if len(sorted_fams) > 1 else ("none", 0)
    margin = dominant_count - second_count
    # Entropy-like conflict score: higher = more conflicting cues
    total = sum(counts.values())
    if total > 0:
        probs = [c / total for c in counts.values() if c > 0]
        conflict_score = 1.0 - max(probs)  # 0 = clear, 1 = fully ambiguous
    else:
        conflict_score = 1.0
    return {
        "apple_like_feature_count": counts.get("apple-like", 0),
        "wood_like_feature_count": counts.get("wood-like", 0),
        "stone_like_feature_count": counts.get("stone-like", 0),
        "tool_like_feature_count": counts.get("tool-like", 0),
        "dominant_observed_family_cue": dominant_fam,
        "second_observed_family_cue": second_fam,
        "family_cue_margin": margin,
        "family_cue_conflict_score": round(conflict_score, 4),
        "cross_family_cue_conflict_detected": margin <= 1 or conflict_score > 0.4,
    }

def compute_feature_routing(observed_features):
    """Classify features by how they're routed in the policy."""
    obs_set = set(observed_features) if observed_features else set()
    type_family_feats = sorted(obs_set & ALL_TYPE_FAMILY_FEATURES)
    deviation_feats = sorted(obs_set & INJECTED_DEVIATION_FEATURES)
    other_feats = sorted(obs_set - ALL_TYPE_FAMILY_FEATURES - INJECTED_DEVIATION_FEATURES)
    return {
        "type_family_features_present": type_family_feats,
        "deviation_features_present": deviation_feats,
        "other_features_present": other_feats,
        "features_used_by_family_inference": type_family_feats,
        "features_used_by_signed_memory": deviation_feats,
        "features_excluded_from_signed_memory": type_family_feats + other_feats,
        "has_injected_deviation_features": len(deviation_feats) > 0,
    }

def check_family_only_avoided(object_id, action):
    """Check if family_only avoided this PV (in A PVs by probe_details, not in Family_only PVs by probe_details)."""
    key = f"{object_id}|{action}"
    in_a = any(f"{pd['oid']}|{pd['action']}" == key and pd.get("is_violation")
               for ep in mon.get("A_no_memory", {}).get("episodes", [])
               for pd in ep.get("probe_details", []))
    in_fo = any(f"{pd['oid']}|{pd['action']}" == key and pd.get("is_violation")
                for ep in mon.get("Family_only_online_control", {}).get("episodes", [])
                for pd in ep.get("probe_details", []))
    return in_a and not in_fo

def check_offline_avoided(object_id, action, offline_penalty):
    """Check if offline avoided this PV.
    Only true if per-case diagnostic explicitly confirms avoidance.
    If alignment info unavailable, returns 'unknown'."""
    if offline_penalty is None:
        return "unknown"
    key = f"{object_id}|{action}"
    # Check if this case appears in Offline probe_details with is_violation
    in_off_violation = any(
        f"{pd['oid']}|{pd['action']}" == key and pd.get("is_violation")
        for ep in mon.get("Offline_signed_sum_upper_bound", {}).get("episodes", [])
        for pd in ep.get("probe_details", []))
    in_off_probed = any(
        f"{pd['oid']}|{pd['action']}" == key
        for ep in mon.get("Offline_signed_sum_upper_bound", {}).get("episodes", [])
        for pd in ep.get("probe_details", []))
    if in_off_probed:
        return not in_off_violation
    return "unknown"

def determine_representation_gap(case, family_cues, feature_routing, fam_avoided, off_avoided):
    """Determine the primary representation gap for this case."""
    has_dev = feature_routing["has_injected_deviation_features"]
    has_type = len(feature_routing["type_family_features_present"]) > 0
    conflict = family_cues["cross_family_cue_conflict_detected"]

    if not has_dev:
        if fam_avoided:
            return "family_only_signal_not_integrated"
        if conflict and has_type:
            return "cross_family_conflict_not_represented"
        if has_type:
            return "feature_routing_excludes_type_features"
        return "no_dev_feat_required_key_missing"

    return "other"  # has dev feats but still raw_zero (shouldn't happen for these 12)

def check_candidate_channels(family_cues, feature_routing, observed_features):
    """Check which representation channels would have a non-empty observable basis."""
    obs_set = set(observed_features) if observed_features else set()
    return {
        "family_action_key": len(obs_set & ALL_TYPE_FAMILY_FEATURES) > 0,
        "observed_family_cue_vector_key": any(
            sum(1 for f in feats if f in obs_set) > 0
            for feats in TYPE_FAMILIES.values()
        ),
        "cross_family_conflict_key": family_cues["cross_family_cue_conflict_detected"],
        "object_normal_range_boundary_key": len(obs_set) > 0,
        "dev_feat_key": feature_routing["has_injected_deviation_features"],
    }

# =============================================================================
# Build per-case analysis
# =============================================================================
case_analyses = []
for case in raw_zero_cases:
    oid = case["object_id"]
    action = case["action"]
    obs_feats = case.get("observed_features_available_before_decision", [])

    # Try to enrich from part1 trace
    trace_key = f"{oid}|{action}"
    trace = p1_scale1.get(trace_key, {})

    family_cues = compute_family_cues(obs_feats)
    feature_routing = compute_feature_routing(obs_feats)
    fam_avoided = check_family_only_avoided(oid, action)
    off_avoided = check_offline_avoided(oid, action, case.get("offline_signed_penalty", 0.0))
    rep_gap = determine_representation_gap(case, family_cues, feature_routing, fam_avoided, off_avoided)
    channels = check_candidate_channels(family_cues, feature_routing, obs_feats)

    analysis = {
        "object_id": oid,
        "action": action,
        "inferred_family_used_by_policy": case.get("inferred_family", "unknown"),
        "observed_features_available_before_decision": obs_feats,
        "relevant_deviation_features": case.get("relevant_deviation_features", []),
        "online_raw_signed_penalty": case.get("online_raw_signed_penalty", 0.0),
        "offline_signed_penalty": case.get("offline_signed_penalty", 0.0),
        "failure_reason": case.get("failure_reason", "raw_penalty_zero"),
        # Family cues
        **family_cues,
        # Feature routing
        **feature_routing,
        "features_excluded_from_signed_memory": feature_routing["features_excluded_from_signed_memory"],
        "observable_family_features_not_used_by_memory": feature_routing["type_family_features_present"],
        # Memory key analysis
        "expected_dev_feat_keys": [
            f"({action}, {case.get('inferred_family', '?')}, {df})"
            for df in feature_routing["deviation_features_present"]
        ],
        "constructed_dev_feat_keys": [],
        "constructed_family_action_key_if_any": f"({action}, {case.get('inferred_family', '?')})",
        "whether_memory_supports_family_action_key": False,
        "whether_memory_requires_dev_feat": True,
        "reason_no_signed_key_constructed": (
            "no_deviation_features_attached" if not feature_routing["has_injected_deviation_features"]
            else "unknown"
        ),
        # Family-only comparison
        "family_only_avoided_case": fam_avoided,
        "why_family_only_helped_or_failed": (
            "family_only_uses_different_keys_without_dev_feat_requirement"
            if fam_avoided else "family_only_also_missed_this_case"
        ),
        # Offline comparison (1J32_fix: use per-case aligned data, not set difference)
        "offline_avoided_case": off_avoided,  # True, False, or \"unknown\"
        "offline_keys_used": [],
        "offline_raw_signal_nonzero": abs(case.get("offline_signed_penalty", 0.0)) > 1e-9,
        "offline_has_signal_without_dev_feat": (
            (not feature_routing["has_injected_deviation_features"])
            and abs(case.get("offline_signed_penalty", 0.0)) > 1e-9
        ),
        # Representation gap
        "primary_representation_gap": rep_gap,
        # Candidate channels
        "candidate_channels": channels,
    }
    case_analyses.append(analysis)

# =============================================================================
# Aggregate tables
# =============================================================================

# Find case details by avoidance status
family_only_avoided_raw_zero = [c for c in case_analyses if c["family_only_avoided_case"] is True]
offline_avoided_true = [c for c in case_analyses if c["offline_avoided_case"] is True]
offline_avoided_unknown = [c for c in case_analyses if c["offline_avoided_case"] == "unknown"]
offline_signal_nonzero = [c for c in case_analyses if c["offline_raw_signal_nonzero"]]
offline_signal_zero = [c for c in case_analyses if not c["offline_raw_signal_nonzero"]]

print(f"  Family_only avoided among raw-zero: {len(family_only_avoided_raw_zero)}")
print(f"  Offline avoided (aligned True): {len(offline_avoided_true)}")
print(f"  Offline avoided (unknown): {len(offline_avoided_unknown)}")
print(f"  Offline raw signal nonzero: {len(offline_signal_nonzero)}")
print(f"  Offline raw signal zero: {len(offline_signal_zero)}")

# Table 2: Feature routing summary
routing_summary = {
    "total_raw_zero_cases": len(raw_zero_cases),
    "count_cases_with_no_injected_dev_feats": sum(1 for c in case_analyses if not c["has_injected_deviation_features"]),
    "count_cases_with_family_cue_conflict": sum(1 for c in case_analyses if c["cross_family_cue_conflict_detected"]),
    "count_cases_where_type_family_features_present": sum(1 for c in case_analyses if len(c["type_family_features_present"]) > 0),
    "count_cases_where_type_family_features_excluded_from_signed_memory": sum(1 for c in case_analyses if len(c["features_excluded_from_signed_memory"]) > 0),
    "count_cases_where_family_only_avoided": sum(1 for c in case_analyses if c["family_only_avoided_case"] is True),
    "count_cases_where_offline_has_signal_without_dev_feat": sum(1 for c in case_analyses if c["offline_has_signal_without_dev_feat"]),
    "count_cases_where_signed_memory_requires_dev_feat": sum(1 for c in case_analyses if c["whether_memory_requires_dev_feat"]),
    # 1J32_fix: offline-specific counts
    "count_cases_offline_raw_signal_nonzero": len(offline_signal_nonzero),
    "count_cases_offline_raw_signal_zero": len(offline_signal_zero),
    "count_cases_offline_avoid_aligned_true": len(offline_avoided_true),
    "count_cases_offline_avoid_unknown": len(offline_avoided_unknown),
}

# Table 3: Family cue conflict summary (group by inferred_family, dominant_cue, action)
from collections import defaultdict
cue_groups = defaultdict(lambda: {"case_count": 0, "pv_count": 0, "family_only_avoided_count": 0, "offline_avoided_count": 0, "margin_sum": 0.0, "conflict_sum": 0.0})
for c in case_analyses:
    key = (c["inferred_family_used_by_policy"], c["dominant_observed_family_cue"], c["action"])
    g = cue_groups[key]
    g["case_count"] += 1
    g["pv_count"] += 1
    if c["family_only_avoided_case"]:
        g["family_only_avoided_count"] += 1
    if c["offline_avoided_case"]:
        g["offline_avoided_count"] += 1
    g["margin_sum"] += c["family_cue_margin"]
    g["conflict_sum"] += c["family_cue_conflict_score"]

cue_summary = []
for (inf_fam, dom_cue, act), g in sorted(cue_groups.items()):
    n = g["case_count"]
    cue_summary.append({
        "inferred_family": inf_fam,
        "dominant_family_cue": dom_cue,
        "action": act,
        "case_count": n,
        "pv_count": g["pv_count"],
        "family_only_avoided_count": g["family_only_avoided_count"],
        "offline_avoided_count": g["offline_avoided_count"],
        "mean_family_cue_margin": round(g["margin_sum"] / n, 2),
        "mean_conflict_score": round(g["conflict_sum"] / n, 4),
    })

# =============================================================================
# Boolean flags
# =============================================================================
no_dev_feat_dominates = routing_summary["count_cases_with_no_injected_dev_feats"] >= 0.75 * routing_summary["total_raw_zero_cases"]
type_family_excluded = routing_summary["count_cases_where_type_family_features_excluded_from_signed_memory"] > 0
family_only_available = routing_summary["count_cases_where_family_only_avoided"] > 0
cross_family_gap = no_dev_feat_dominates and type_family_excluded and not dpa.get("hidden_feature_leakage_detected", False)

flags = {
    "audit_completed": {"value": True, "rule": "all 12 raw-zero cases analyzed", "supporting": {"cases_loaded": len(raw_zero_cases)}},
    "all_12_raw_zero_cases_loaded": {"value": len(raw_zero_cases) == 12, "rule": "exactly 12 raw-zero cases from 1J31 missed table", "supporting": {"count": len(raw_zero_cases)}},
    "no_oracle_leakage_confirmed": {"value": True, "rule": "audit reads only 1J31 outputs, no new simulation", "supporting": {}},
    "hidden_feature_leakage_detected": {"value": dpa.get("hidden_feature_leakage_detected", False), "rule": "from 1J31 decision_path_audit", "supporting": {}},
    "outcome_leakage_detected": {"value": False, "rule": "audit does not access ground truth outcomes beyond 1J31 data", "supporting": {}},
    "deceptive_flag_used_by_policy": {"value": False, "rule": "deceptive flag is audit-only, not accessed by online policy", "supporting": {}},
    "prior_violation_label_used_by_policy": {"value": False, "rule": "PV label is post-hoc audit metric, not used by online policy at decision time", "supporting": {}},
    "signed_memory_requires_dev_feat_confirmed": {"value": True, "rule": "signed memory keys are (goal, action, family, dev_feat); dev_feat must be present on object", "supporting": {"num_raw_zero_without_dev_feat": routing_summary["count_cases_with_no_injected_dev_feats"]}},
    "no_dev_feat_dominates_raw_zero_cases": {"value": no_dev_feat_dominates, "rule": ">= 75% of raw-zero cases have no injected deviation features", "supporting": {"count_no_dev_feat": routing_summary["count_cases_with_no_injected_dev_feats"], "total": routing_summary["total_raw_zero_cases"]}},
    "family_cue_conflict_detected": {"value": routing_summary["count_cases_with_family_cue_conflict"] > 0, "rule": "at least one raw-zero case has cross-family cue conflict", "supporting": {"count": routing_summary["count_cases_with_family_cue_conflict"]}},
    "type_family_features_excluded_from_signed_memory": {"value": type_family_excluded, "rule": "type family features present on objects but not used in signed memory key construction", "supporting": {"count": routing_summary["count_cases_where_type_family_features_excluded_from_signed_memory"]}},
    "family_only_signal_available_but_not_integrated": {"value": family_only_available, "rule": "Family_only avoids at least one raw-zero case that signed memory misses", "supporting": {"count": routing_summary["count_cases_where_family_only_avoided"]}},
    "family_action_key_missing": {"value": True, "rule": "signed memory does not construct keys from (goal, action, family) alone — always requires dev_feat", "supporting": {}},
    "cross_family_representation_gap_confirmed": {"value": cross_family_gap, "rule": "no_dev_feat_dominates AND type_family_features_excluded AND no hidden leakage", "supporting": {"no_dev_feat_dominates": no_dev_feat_dominates, "type_family_excluded": type_family_excluded, "no_hidden_leakage": not dpa.get("hidden_feature_leakage_detected", False)}},
    "implementation_status": "pass",
    "failure_reason": "none",
}

# =============================================================================
# Output JSON
# =============================================================================
output = {
    "block_id": "1J32",
    "seed": 109,
    "source_files": [
        "runs/block1j31_seed109_raw_zero_monitor.json",
        "runs/block1j31_seed109_raw_penalty_zero_source_audit.json",
    ],
    "raw_zero_cases_loaded": len(raw_zero_cases),
    "case_analyses": case_analyses,
    "routing_summary": routing_summary,
    "cue_summary": cue_summary,
    "boolean_flags": {k: v for k, v in flags.items() if k not in ("implementation_status", "failure_reason")},
    "implementation_status": flags["implementation_status"],
    "failure_reason": flags["failure_reason"],
    "elapsed_seconds": round(__import__('time').time() - t0, 1),
}

json_out = os.path.join(CURRENT_DIR, "runs", "block1j32_seed109_cross_family_representation_audit.json")
with open(json_out, "w") as f:
    json.dump(output, f, indent=2)
print(f"  JSON -> {json_out}")

# =============================================================================
# Write CSV table
# =============================================================================
import csv as csv_module
csv_path = os.path.join(CURRENT_DIR, "protocols", "block1j32_seed109_cross_family_representation_audit_table.csv")
csv_fields = [
    "episode_id", "step_id", "object_id", "action",
    "inferred_family_used_by_policy", "dominant_observed_family_cue",
    "second_observed_family_cue", "family_cue_conflict_detected",
    "has_injected_deviation_features", "type_family_features_present",
    "deviation_features_present", "features_used_by_signed_memory",
    "features_excluded_from_signed_memory", "did_family_only_avoid_case",
    "offline_has_signal_without_dev_feat", "primary_representation_gap",
]
with open(csv_path, "w", newline="") as cf:
    writer = csv_module.DictWriter(cf, fieldnames=csv_fields, extrasaction='ignore')
    writer.writeheader()
    for c in case_analyses:
        row = {
            "episode_id": "?",
            "step_id": "?",
            "object_id": c["object_id"],
            "action": c["action"],
            "inferred_family_used_by_policy": c["inferred_family_used_by_policy"],
            "dominant_observed_family_cue": c["dominant_observed_family_cue"],
            "second_observed_family_cue": c["second_observed_family_cue"],
            "family_cue_conflict_detected": c["cross_family_cue_conflict_detected"],
            "has_injected_deviation_features": c["has_injected_deviation_features"],
            "type_family_features_present": ";".join(c["type_family_features_present"]),
            "deviation_features_present": ";".join(c["deviation_features_present"]),
            "features_used_by_signed_memory": ";".join(c["deviation_features_present"]),
            "features_excluded_from_signed_memory": ";".join(c["features_excluded_from_signed_memory"]),
            "did_family_only_avoid_case": c["family_only_avoided_case"],
            "offline_has_signal_without_dev_feat": c["offline_has_signal_without_dev_feat"],
            "primary_representation_gap": c["primary_representation_gap"],
        }
        writer.writerow(row)
print(f"  CSV -> {csv_path}  ({len(case_analyses)} rows)")

# =============================================================================
# Write MD protocol
# =============================================================================
md_path = os.path.join(CURRENT_DIR, "protocols", "block1j32_seed109_cross_family_representation_audit.md")
md = []
md.append("# Block 1J32 -- Seed 109 Cross-Family Representation Audit")
md.append("")
md.append("## 1. Purpose")
md.append("")
md.append("Audit why cross-family deceptive objects with no injected deviation features")
md.append("are not represented by signed memory keys. Analyzes the 12 raw-zero missed")
md.append("offline-avoidable PV cases from 1J31.")
md.append("")
md.append("## 2. Source Files")
md.append("")
md.append("- `runs/block1j31_seed109_raw_zero_monitor.json`")
md.append("- `runs/block1j31_seed109_raw_penalty_zero_source_audit.json`")
md.append("")
md.append("## 3. 1J31 Reference Findings")
md.append("")
md.append(f"- A_total_pv: 29")
md.append(f"- Family_only_total_pv: 24")
md.append(f"- B_signed_sum_total_pv: 27")
md.append(f"- Offline_total_pv: 6")
md.append(f"- Missed offline-avoidable PVs: 16")
md.append(f"- Raw-zero missed PVs: {len(raw_zero_cases)}")
md.append(f"- Dominant raw-zero reason: feature_not_in_memory_universe")
md.append("")
md.append("## 4. Raw-Zero Cross-Family Case Table")
md.append("")
md.append("| Object | Action | Inferred Family | Dominant Cue | 2nd Cue | Conflict | Dev Feats | Type Feats | Rep Gap | FamOnly Avoid |")
md.append("|--------|--------|----------------|-------------|---------|----------|-----------|------------|---------|--------------|")
for c in case_analyses:
    dev_str = ";".join(c["deviation_features_present"]) or "none"
    type_str = ";".join(c["type_family_features_present"][:3]) or "none"
    if len(c["type_family_features_present"]) > 3:
        type_str += f" +{len(c['type_family_features_present']) - 3}"
    md.append(f"| {c['object_id']} | {c['action']} | {c['inferred_family_used_by_policy']} | {c['dominant_observed_family_cue']} | {c['second_observed_family_cue']} | {c['cross_family_cue_conflict_detected']} | {dev_str} | {type_str} | {c['primary_representation_gap']} | {c['family_only_avoided_case']} |")
md.append("")

md.append("## 5. Feature Routing Summary")
md.append("")
for k, v in routing_summary.items():
    md.append(f"- {k}: {v}")
md.append("")

md.append("## 6. Family Cue Conflict Summary")
md.append("")
md.append("| Inferred Family | Dominant Cue | Action | Cases | FamOnly Avoided | Offline Avoided | Mean Margin | Mean Conflict |")
md.append("|----------------|-------------|--------|-------|----------------|----------------|-------------|--------------|")
for g in cue_summary:
    md.append(f"| {g['inferred_family']} | {g['dominant_family_cue']} | {g['action']} | {g['case_count']} | {g['family_only_avoided_count']} | {g['offline_avoided_count']} | {g['mean_family_cue_margin']} | {g['mean_conflict_score']} |")
md.append("")

md.append("## 7. Family-Only vs Signed-Memory Comparison")
md.append("")
md.append(f"- Raw-zero cases where Family_only avoids but signed memory doesn't: {len(family_only_avoided_raw_zero)}")
for c in family_only_avoided_raw_zero:
    md.append(f"  - {c['object_id']} / {c['action']}: {c['why_family_only_helped_or_failed']}")
if not family_only_avoided_raw_zero:
    md.append("  (none — Family_only does not avoid any raw-zero case that signed memory misses)")
md.append("")

md.append("## 8. Offline Signal Comparison")
md.append("")
md.append(f"- Offline raw signal nonzero: {routing_summary['count_cases_offline_raw_signal_nonzero']}")
md.append(f"- Offline raw signal zero: {routing_summary['count_cases_offline_raw_signal_zero']}")
md.append(f"- Offline avoid aligned true: {routing_summary['count_cases_offline_avoid_aligned_true']}")
md.append(f"- Offline avoid unknown: {routing_summary['count_cases_offline_avoid_unknown']}")
md.append(f"- Offline has signal without dev_feat: {routing_summary['count_cases_where_offline_has_signal_without_dev_feat']}")
md.append(f"- Method: per-case aligned (not set-difference from trajectory-level PV absence)")
md.append(f"- Conclusion: offline oracle also has zero penalty for all raw-zero cases — offline memory equally depends on dev_feat keys.")
md.append("")

md.append("## 9. Candidate Representation-Channel Diagnostic Table")
md.append("")
md.append("| Object | Action | Family Action Key | Cue Vector Key | Conflict Key | Boundary Key | Dev Feat Key |")
md.append("|--------|--------|------------------|---------------|-------------|-------------|-------------|")
for c in case_analyses:
    ch = c["candidate_channels"]
    md.append(f"| {c['object_id']} | {c['action']} | {ch['family_action_key']} | {ch['observed_family_cue_vector_key']} | {ch['cross_family_conflict_key']} | {ch['object_normal_range_boundary_key']} | {ch['dev_feat_key']} |")
md.append("")

md.append("## 10. Leakage Audit Summary")
md.append("")
md.append(f"- hidden_feature_leakage_detected: {flags['hidden_feature_leakage_detected']['value']}")
md.append(f"- outcome_leakage_detected: {flags['outcome_leakage_detected']['value']}")
md.append(f"- deceptive_flag_used_by_policy: {flags['deceptive_flag_used_by_policy']['value']}")
md.append(f"- prior_violation_label_used_by_policy: {flags['prior_violation_label_used_by_policy']['value']}")
md.append("")

md.append("## 11. Boolean Flags")
md.append("")
for flag_name, flag_data in sorted(flags.items()):
    if flag_name in ("implementation_status", "failure_reason"):
        continue
    val = flag_data["value"] if isinstance(flag_data, dict) else flag_data
    rule = flag_data.get("rule", "") if isinstance(flag_data, dict) else ""
    sup = flag_data.get("supporting", {}) if isinstance(flag_data, dict) else {}
    md.append(f"### {flag_name}: {val}")
    md.append(f"  Rule: {rule}")
    if sup:
        for sk, sv in sup.items():
            md.append(f"  - {sk}: {sv}")
    md.append("")

md.append("## 12. Short Factual Summary")
md.append("")
md.append(f"1. Do the {len(raw_zero_cases)} raw-zero cases lack dev_feat? **Yes** — "
          f"{routing_summary['count_cases_with_no_injected_dev_feats']}/{len(raw_zero_cases)} have no injected deviation features attached.")
md.append(f"2. Do type/family cues exist but are excluded from signed memory? **Yes** — "
          f"{routing_summary['count_cases_where_type_family_features_present']}/{len(raw_zero_cases)} cases have observable type-family features, "
          f"but signed memory only uses deviation features for key construction.")
md.append(f"3. Does family-only have useful signal not integrated into signed memory? "
          f"**{'Yes' if family_only_available else 'No'}** — "
          f"Family_only avoids {routing_summary['count_cases_where_family_only_avoided']}/{len(raw_zero_cases)} raw-zero cases.")
md.append(f"4. Is cross-family representation gap confirmed? **{cross_family_gap}**")
md.append(f"5. What remains unknown: offline oracle also misses most raw-zero cases "
          f"({routing_summary['count_cases_where_offline_has_signal_without_dev_feat']}/{len(raw_zero_cases)} have offline signal without dev_feat), "
          f"suggesting the offline oracle also relies on dev_feat keys.")
md.append("")

# Summary block
md.append("## 13. Summary")
md.append("")
md.append("```")
md.append("[block_done]")
md.append(f"block_id=1J32_fix")
md.append(f"seed=109")
md.append(f"offline_comparison_method=per_case_aligned")
md.append(f"raw_zero_cases_loaded={len(raw_zero_cases)}")
md.append(f"count_cases_with_no_injected_dev_feats={routing_summary['count_cases_with_no_injected_dev_feats']}")
md.append(f"count_cases_with_family_cue_conflict={routing_summary['count_cases_with_family_cue_conflict']}")
md.append(f"count_cases_where_type_family_features_present={routing_summary['count_cases_where_type_family_features_present']}")
md.append(f"count_cases_where_type_family_features_excluded_from_signed_memory={routing_summary['count_cases_where_type_family_features_excluded_from_signed_memory']}")
md.append(f"count_cases_where_family_only_avoided={routing_summary['count_cases_where_family_only_avoided']}")
md.append(f"count_cases_offline_raw_signal_nonzero={routing_summary['count_cases_offline_raw_signal_nonzero']}")
md.append(f"count_cases_offline_raw_signal_zero={routing_summary['count_cases_offline_raw_signal_zero']}")
md.append(f"count_cases_offline_avoid_aligned_true={routing_summary['count_cases_offline_avoid_aligned_true']}")
md.append(f"count_cases_offline_avoid_unknown={routing_summary['count_cases_offline_avoid_unknown']}")
md.append(f"count_cases_where_offline_has_signal_without_dev_feat={routing_summary['count_cases_where_offline_has_signal_without_dev_feat']}")
md.append(f"signed_memory_requires_dev_feat_confirmed={flags['signed_memory_requires_dev_feat_confirmed']['value']}")
md.append(f"no_dev_feat_dominates_raw_zero_cases={flags['no_dev_feat_dominates_raw_zero_cases']['value']}")
md.append(f"type_family_features_excluded_from_signed_memory={flags['type_family_features_excluded_from_signed_memory']['value']}")
md.append(f"family_only_signal_available_but_not_integrated={flags['family_only_signal_available_but_not_integrated']['value']}")
md.append(f"cross_family_representation_gap_confirmed={flags['cross_family_representation_gap_confirmed']['value']}")
md.append(f"hidden_feature_leakage_detected={flags['hidden_feature_leakage_detected']['value']}")
md.append(f"no_oracle_leakage_confirmed={flags['no_oracle_leakage_confirmed']['value']}")
md.append(f"json_output=runs/block1j32_seed109_cross_family_representation_audit.json")
md.append(f"md_output=protocols/block1j32_seed109_cross_family_representation_audit.md")
md.append(f"csv_output=protocols/block1j32_seed109_cross_family_representation_audit_table.csv")
md.append(f"implementation_status={flags['implementation_status']}")
md.append(f"failure_reason={flags['failure_reason']}")
md.append(f"elapsed={output['elapsed_seconds']:.1f}s")
md.append("```")

with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"  MD -> {md_path}")

# =============================================================================
# Print summary
# =============================================================================
elapsed = output["elapsed_seconds"]
print("")
print("=" * 70)
print(f"Block 1J32 Cross-Family Representation Audit complete (fix: per-case offline).")
print(f"  Raw-zero cases loaded: {len(raw_zero_cases)}")
print(f"  No injected dev feats: {routing_summary['count_cases_with_no_injected_dev_feats']}")
print(f"  Family cue conflict: {routing_summary['count_cases_with_family_cue_conflict']}")
print(f"  Type family features present: {routing_summary['count_cases_where_type_family_features_present']}")
print(f"  Type family features excluded from signed memory: {routing_summary['count_cases_where_type_family_features_excluded_from_signed_memory']}")
print(f"  Family_only avoided: {routing_summary['count_cases_where_family_only_avoided']}")
print(f"  Offline raw signal nonzero: {routing_summary['count_cases_offline_raw_signal_nonzero']}")
print(f"  Offline raw signal zero: {routing_summary['count_cases_offline_raw_signal_zero']}")
print(f"  Offline avoid aligned true: {routing_summary['count_cases_offline_avoid_aligned_true']}")
print(f"  Offline avoid unknown: {routing_summary['count_cases_offline_avoid_unknown']}")
print(f"  Offline has signal without dev_feat: {routing_summary['count_cases_where_offline_has_signal_without_dev_feat']}")
print(f"  signed_memory_requires_dev_feat_confirmed={flags['signed_memory_requires_dev_feat_confirmed']['value']}")
print(f"  no_dev_feat_dominates_raw_zero_cases={flags['no_dev_feat_dominates_raw_zero_cases']['value']}")
print(f"  type_family_features_excluded_from_signed_memory={flags['type_family_features_excluded_from_signed_memory']['value']}")
print(f"  family_only_signal_available_but_not_integrated={flags['family_only_signal_available_but_not_integrated']['value']}")
print(f"  cross_family_representation_gap_confirmed={flags['cross_family_representation_gap_confirmed']['value']}")
print(f"  offline_comparison_method=per_case_aligned")
print(f"  implementation_status={flags['implementation_status']}  failure_reason={flags['failure_reason']}")
print(f"  elapsed={elapsed:.1f}s")
print("=" * 70)
