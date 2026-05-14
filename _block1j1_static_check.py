"""
Block 1J1 — Static generation sanity check for C4_instance_subtype_cued_v1.
Generates test objects with hidden subtypes and computes label distributions.
No simulations, no policy runs, no C13.
"""
import os
import sys
import json
import random
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure objects.py is importable
A3_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5a3_tool_material_transfer")
sys.path.insert(0, A3_DIR)

from subtype_objects import (
    SUBTYPE_DEFINITIONS,
    generate_subtype_objects,
    generate_subtype_objects_deterministic,
    compute_query_ground_truth,
)
from objects import TASK_QUERIES

t0 = time.time()

# =============================================================================
# Generate test objects
# =============================================================================
SEED = 101
rng = random.Random(SEED)

# Test set: 15 per category = 60 objects (same as existing test sets)
NUM_PER_CATEGORY = 15
test_objects = generate_subtype_objects_deterministic(
    NUM_PER_CATEGORY, NUM_PER_CATEGORY, NUM_PER_CATEGORY, NUM_PER_CATEGORY,
    rng, prefix="test"
)

# =============================================================================
# Compute category and subtype counts
# =============================================================================
category_counts = {}
subtype_counts = {}
for oid, obj in test_objects.items():
    cat = obj["hidden_category"]
    st = obj.get("hidden_subtype", "unknown")
    category_counts[cat] = category_counts.get(cat, 0) + 1
    key = f"{cat}::{st}"
    subtype_counts[key] = subtype_counts.get(key, 0) + 1

# =============================================================================
# Compute query ground truth
# =============================================================================
query_gt = compute_query_ground_truth(test_objects)

per_query_stats = {}
for qname, data in query_gt.items():
    n_pos = len(data["positive_oids"])
    n_neg = len(data["negative_oids"])
    n_total = n_pos + n_neg
    pos_rate = n_pos / n_total if n_total > 0 else 0.0
    neg_rate = n_neg / n_total if n_total > 0 else 0.0
    has_both = n_pos > 0 and n_neg > 0
    per_query_stats[qname] = {
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_total": n_total,
        "positive_rate": round(pos_rate, 4),
        "negative_rate": round(neg_rate, 4),
        "has_both_classes": has_both,
        "positive_rate_in_ideal_range": 0.35 <= pos_rate <= 0.65,
    }

# =============================================================================
# Compute all_false expected macro_bal
# =============================================================================
# all_false: predicts False for every query-object pair
# pos_recall = 0.0 for all queries, neg_recall = 1.0 for all queries
# macro_bal = mean(0.5*(0.0 + 1.0)) = 0.5
all_false_macro_bal = 0.5

# =============================================================================
# Compute category_majority macro_bal
# =============================================================================
# For each query, determine which categories have majority True,
# then predict True for all objects in those categories, False otherwise.
category_majority_query_bal = {}
for qname, data in query_gt.items():
    # Per-category positive rate
    cat_pos_counts = {}
    cat_total_counts = {}
    for oid in data["positive_oids"]:
        cat = test_objects[oid]["hidden_category"]
        cat_pos_counts[cat] = cat_pos_counts.get(cat, 0) + 1
    for oid in data["negative_oids"]:
        cat = test_objects[oid]["hidden_category"]
        cat_total_counts[cat] = cat_total_counts.get(cat, 0) + 1

    # Build prediction: True for categories with majority positive
    cat_majority_true = set()
    for cat in category_counts:
        pos = cat_pos_counts.get(cat, 0)
        total = cat_total_counts.get(cat, 0) + pos  # all objects of this cat
        if total > 0 and pos / total > 0.5:
            cat_majority_true.add(cat)

    # Compute per-object correctness
    tp = 0  # predicted True, actually True
    fp = 0  # predicted True, actually False
    tn = 0  # predicted False, actually False
    fn = 0  # predicted False, actually True

    pos_set = set(data["positive_oids"])
    for oid in test_objects:
        cat = test_objects[oid]["hidden_category"]
        pred_true = cat in cat_majority_true
        actual_true = oid in pos_set
        if pred_true and actual_true:
            tp += 1
        elif pred_true and not actual_true:
            fp += 1
        elif not pred_true and not actual_true:
            tn += 1
        else:
            fn += 1

    pos_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal = 0.5 * (pos_recall + neg_recall)
    category_majority_query_bal[qname] = {
        "pos_recall": round(pos_recall, 4),
        "neg_recall": round(neg_recall, 4),
        "balanced_acc": round(bal, 4),
        "majority_true_categories": sorted(list(cat_majority_true)),
    }

cat_maj_macro_bal = sum(v["balanced_acc"] for v in category_majority_query_bal.values()) / len(category_majority_query_bal)

# =============================================================================
# Visible cue coverage summary
# =============================================================================
# For each category and subtype, show which visible features have
# meaningful probability differences vs the other subtype
visible_cue_summary = {}
for cat, cat_def in SUBTYPE_DEFINITIONS.items():
    subtypes = cat_def["subtypes"]
    cue_probs = cat_def["visible_cue_probs"]
    cat_summary = {"subtypes": subtypes, "cue_differentials": {}}
    for feat in sorted(cue_probs[subtypes[0]].keys()):
        p0 = cue_probs[subtypes[0]].get(feat, 0.5)
        p1 = cue_probs[subtypes[1]].get(feat, 0.5)
        diff = abs(p0 - p1)
        cat_summary["cue_differentials"][feat] = {
            f"{subtypes[0]}_p": p0,
            f"{subtypes[1]}_p": p1,
            "abs_diff": round(diff, 3),
        }
    visible_cue_summary[cat] = cat_summary

# =============================================================================
# Subtype affordance difference summary
# =============================================================================
affordance_diff_summary = {}
for cat, cat_def in SUBTYPE_DEFINITIONS.items():
    subtypes = cat_def["subtypes"]
    prof0 = cat_def["affordance_profiles"][subtypes[0]]
    prof1 = cat_def["affordance_profiles"][subtypes[1]]
    diffs = {}
    for action in sorted(prof0.keys()):
        if prof0[action] != prof1[action]:
            diffs[action] = {subtypes[0]: prof0[action], subtypes[1]: prof1[action]}
    affordance_diff_summary[cat] = {
        "differs_on": sorted(list(diffs.keys())),
        "details": diffs,
    }

# =============================================================================
# Sanity checks
# =============================================================================
all_queries_have_both = all(s["has_both_classes"] for s in per_query_stats.values())
all_queries_in_range = all(
    0.20 <= s["positive_rate"] <= 0.80 for s in per_query_stats.values()
)
any_query_in_ideal_range = any(
    s["positive_rate_in_ideal_range"] for s in per_query_stats.values()
)
total_objects = len(test_objects)

static_sanity_passed = all_queries_have_both and all_queries_in_range

# =============================================================================
# Print summary
# =============================================================================
print("=" * 60)
print("Block 1J1 — C4_instance_subtype_cued_v1 Static Sanity Check")
print("=" * 60)
print(f"  seed={SEED}")
print(f"  total_objects={total_objects}")
print()

print("  Category counts:")
for cat in sorted(category_counts.keys()):
    print(f"    {cat}: {category_counts[cat]}")
print()

print("  Subtype counts:")
for key in sorted(subtype_counts.keys()):
    print(f"    {key}: {subtype_counts[key]}")
print()

print("  Per-query statistics:")
for qname in sorted(per_query_stats.keys()):
    s = per_query_stats[qname]
    flags = []
    if s["has_both_classes"]:
        flags.append("both_classes")
    if s["positive_rate_in_ideal_range"]:
        flags.append("ideal_range")
    elif 0.20 <= s["positive_rate"] <= 0.80:
        flags.append("in_bounds")
    else:
        flags.append("OUT_OF_BOUNDS")
    print(f"    {qname}: pos={s['n_positive']} neg={s['n_negative']} "
          f"pos_rate={s['positive_rate']:.4f} neg_rate={s['negative_rate']:.4f} "
          f"[{', '.join(flags)}]")
print()

print(f"  all_false expected macro_bal: {all_false_macro_bal:.4f}")
print(f"  category_majority macro_bal:  {cat_maj_macro_bal:.4f}")
print(f"    per-query:")
for qname in sorted(category_majority_query_bal.keys()):
    v = category_majority_query_bal[qname]
    print(f"      {qname}: pos_rec={v['pos_recall']:.4f} neg_rec={v['neg_recall']:.4f} "
          f"bal={v['balanced_acc']:.4f}  maj_cats={v['majority_true_categories']}")
print()

print("  Affordance differences per category:")
for cat in sorted(affordance_diff_summary.keys()):
    info = affordance_diff_summary[cat]
    print(f"    {cat}: differs on {info['differs_on']}")
print()

print("  Visible cue differentials (top 3 per category):")
for cat in sorted(visible_cue_summary.keys()):
    diffs = visible_cue_summary[cat]["cue_differentials"]
    top3 = sorted(diffs.items(), key=lambda x: -x[1]["abs_diff"])[:3]
    print(f"    {cat}:")
    for feat, info in top3:
        print(f"      {feat}: diff={info['abs_diff']:.3f}  ({info[list(info.keys())[0]]:.2f} vs {info[list(info.keys())[1]]:.2f})")
print()

print("  Sanity checks:")
print(f"    all_queries_have_both_classes: {all_queries_have_both}")
print(f"    all_queries_in_0.20-0.80:      {all_queries_in_range}")
print(f"    any_query_in_0.35-0.65:        {any_query_in_ideal_range}")
print(f"    static_sanity_passed:           {static_sanity_passed}")
print(f"    ready_for_1J2:                  {static_sanity_passed}")

# =============================================================================
# Failures detail
# =============================================================================
failures = []
if not all_queries_have_both:
    for qname, s in per_query_stats.items():
        if not s["has_both_classes"]:
            failures.append(f"{qname}: only one class (pos={s['n_positive']}, neg={s['n_negative']})")
if not all_queries_in_range:
    for qname, s in per_query_stats.items():
        if not (0.20 <= s["positive_rate"] <= 0.80):
            failures.append(f"{qname}: positive_rate={s['positive_rate']:.4f} out of [0.20, 0.80]")

if failures:
    print()
    print("  FAILURES:")
    for f in failures:
        print(f"    - {f}")

# =============================================================================
# Save JSON summary
# =============================================================================
summary = {
    "block_id": "1J1",
    "condition": "C4_instance_subtype_cued_v1",
    "seed": SEED,
    "total_objects": total_objects,
    "category_counts": category_counts,
    "subtype_counts": subtype_counts,
    "subtype_definitions": {},
    "per_query": per_query_stats,
    "all_false_expected_macro_bal": all_false_macro_bal,
    "category_majority_macro_bal": round(cat_maj_macro_bal, 4),
    "category_majority_per_query": category_majority_query_bal,
    "affordance_differences": affordance_diff_summary,
    "visible_cue_summary": {
        cat: {
            "subtypes": info["subtypes"],
            "top_differentials": sorted(
                info["cue_differentials"].items(),
                key=lambda x: -x[1]["abs_diff"]
            )[:5],
        }
        for cat, info in visible_cue_summary.items()
    },
    "sanity_checks": {
        "all_queries_have_both_classes": all_queries_have_both,
        "all_queries_in_0_20_to_0_80": all_queries_in_range,
        "any_query_in_0_35_to_0_65": any_query_in_ideal_range,
        "static_sanity_passed": static_sanity_passed,
    },
    "failures": failures,
    "ready_for_1J2_marginal_audit": static_sanity_passed,
}

# Serialize subtype definitions (remove non-serializable bits)
for cat, cat_def in SUBTYPE_DEFINITIONS.items():
    summary["subtype_definitions"][cat] = {
        "subtypes": cat_def["subtypes"],
        "subtype_ratio": cat_def["subtype_ratio"],
        "affordance_profiles": cat_def["affordance_profiles"],
        "n_visible_cues": len(cat_def["visible_cue_probs"].get(cat_def["subtypes"][0], {})),
    }

output_path = os.path.join(CURRENT_DIR, "runs",
                           "calibration_block1j1_static_environment_candidate_summary.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {output_path}")

# =============================================================================
# Block done
# =============================================================================
elapsed = time.time() - t0
print()
print(f"[block_done]")
print(f"  block_id=1J1")
print(f"  condition=C4_instance_subtype_cued_v1")
print(f"  environment_candidate_implemented=true")
print(f"  static_sanity_passed={'true' if static_sanity_passed else 'false'}")
print(f"  policy_runs=false")
print(f"  ready_for_1J2={'true' if static_sanity_passed else 'false'}")
print(f"  elapsed={elapsed:.1f}s")
