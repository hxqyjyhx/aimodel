"""
Block 1J10 — Route F Leakage and Marginal-Value Audit.

Implements Route F peripheral observation interface variants (F_low, F_mid, F_high),
runs leakage probes (A1, A2, B, C, D), and marginal-value baselines.

No C15F policy claim. Audit only.
"""
import os, sys, json, copy, random, time, math
from collections import Counter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
A3_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5a3_tool_material_transfer")
sys.path.insert(0, A3_DIR)
L5_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5l_instance_outcome_memory")
sys.path.insert(0, L5_DIR)
K5_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5k_episodic_sparse_outcome")
sys.path.insert(0, K5_DIR)
I5_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5i_object_uncertainty_probe")
sys.path.insert(0, I5_DIR)
sys.path.insert(0, CURRENT_DIR)

import config
from environment import MiniMCEnvironment, _TASK_QUERIES
from harness import EpisodeHarness
from agent_obs import AgentObs
from event_log import EventLog
from simulator_truth import MiniMCSimulatorTruth
from policies import (
    MiniMCPolicy, C0a_NoInteractionPolicy, C0b_ObserveOnlyPolicy,
    C14a_TruthAnswerOracle, C13_InstanceVOIPolicy,
    MAIN_CANDIDATE_ACTIONS, ACTION_TO_FEATURE, CORE_ACTION_FEATURES,
    _compute_utility, _compute_entropy,
)
from instance_outcome_memory import InstanceOutcomeMemory
from run_004_5n1 import run_phase_a_training
from sparse_outcome_collector import collect_sparse_probe_outcomes
from subtype_objects import (
    generate_subtype_objects_deterministic, compute_query_ground_truth,
    SUBTYPE_DEFINITIONS,
)

t0 = time.time()

SEED = 101
BUDGET = 1.5
COND = copy.deepcopy(config.CUE_CONDITIONS[3])
assert COND["label"] == "C4_instance_subtype_cued_v1"

print("=" * 60)
print("Block 1J10 — Route F Leakage + Marginal-Value Audit")
print(f"  condition={COND['label']}  seed={SEED}  budget={BUDGET}")
print("=" * 60)

# =============================================================================
# 1. Training + IOM building (reused from 1J8)
# =============================================================================
print("\n[1/6] Phase A: Training + IOM building...")
(student, base_learner, train_objects, train_env,
 _std_test, _std_test_env, final_metrics, rng) = run_phase_a_training(SEED, COND)
train_objects_dict = {oid: train_env.objects[oid] for oid in train_objects}
outcome_rows, actual_coverage = collect_sparse_probe_outcomes(
    train_objects_dict, student, config.COVERAGE, SEED)
posterior_visible_features = list(base_learner.visible_feature_names)
im_base = InstanceOutcomeMemory(
    train_objects_dict, posterior_visible_features,
    k=config.INSTANCE_K, similarity_power=config.SIMILARITY_POWER,
    similarity_mode=config.SIMILARITY_MODE,
)
im_base.build(outcome_rows)
print(f"  pre_accuracy={final_metrics.get('domain_token_accuracy', 0.0):.4f}  "
      f"coverage={actual_coverage:.4f}  IOM ready")

# =============================================================================
# 2. Test objects
# =============================================================================
print("\n[2/6] Generating test objects...")
test_objects = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")
query_gt = compute_query_ground_truth(test_objects)
test_oids = sorted(test_objects.keys())
positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, rng)
sim_gt = MiniMCSimulatorTruth(test_objects, positions, config.AGENT_START)

# Object metadata
CATEGORY_DIFFERENTIATING_ACTION = {
    "wood_log": "craft_plank", "stone_block": "mine_by_hand",
    "apple": "eat", "wooden_pickaxe": "use_as_tool",
}

object_meta = {}
for oid in test_oids:
    obj = test_objects[oid]
    cat = obj["hidden_category"]
    subtype = obj["hidden_subtype"]
    gt_aff = sim_gt.get_ground_truth_affordances(oid)
    subtype_def = SUBTYPE_DEFINITIONS[cat]
    ratios = subtype_def["subtype_ratio"]
    is_majority = ratios.get(subtype, 0.0) >= 0.5
    object_meta[oid] = {
        "category": cat, "subtype": subtype, "is_majority": is_majority,
        "gt_affordances": gt_aff,
    }

# Public visible features for ALL objects (accessible without visiting)
ALL_VISIBLE_FEATURES = {}
for oid in test_oids:
    ALL_VISIBLE_FEATURES[oid] = dict(test_objects[oid].get("visible_features", {}))

# Global base rates
GLOBAL_FEATURE_PREVALENCE = {}
for f in posterior_visible_features:
    prevalence = sum(1 for oid in test_oids if ALL_VISIBLE_FEATURES[oid].get(f, False)) / len(test_oids)
    GLOBAL_FEATURE_PREVALENCE[f] = prevalence


# =============================================================================
# 3. Metric helpers (same as 1J7/1J8)
# =============================================================================
def _check_query(probs, query_name):
    if query_name == "need_planks":
        return probs.get("craft_plank_success", 0.5) >= 0.5
    elif query_name == "need_stone":
        return (probs.get("mine_by_hand_success", 0.5) < 0.5
                and probs.get("mine_with_pickaxe_success", 0.5) >= 0.5)
    elif query_name == "need_food":
        return probs.get("eat_success", 0.5) >= 0.5
    elif query_name == "need_tool":
        return probs.get("use_as_tool_success", 0.5) >= 0.5
    elif query_name == "need_fuel":
        return probs.get("burn_as_fuel_success", 0.5) >= 0.5
    return False


def compute_full_metrics(predictions, query_gt):
    per_query = {}
    for qname in sorted(query_gt.keys()):
        pos_set = set(query_gt[qname]["positive_oids"])
        neg_set = set(query_gt[qname]["negative_oids"])
        tp = fp = tn = fn = 0
        for oid in predictions:
            pred_true = _check_query(predictions[oid], qname)
            actual_true = oid in pos_set
            if pred_true and actual_true: tp += 1
            elif pred_true and not actual_true: fp += 1
            elif not pred_true and not actual_true: tn += 1
            else: fn += 1
        pos_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal = 0.5 * (pos_recall + neg_recall)
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        per_query[qname] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "positive_recall": round(pos_recall, 6),
            "negative_recall": round(neg_recall, 6),
            "balanced_accuracy": round(bal, 6),
            "accuracy": round(accuracy, 6),
        }
    macro_bal = sum(v["balanced_accuracy"] for v in per_query.values()) / max(len(per_query), 1)
    macro_acc = sum(v["accuracy"] for v in per_query.values()) / max(len(per_query), 1)
    mean_pos_recall = sum(v["positive_recall"] for v in per_query.values()) / max(len(per_query), 1)
    mean_neg_recall = sum(v["negative_recall"] for v in per_query.values()) / max(len(per_query), 1)
    return {
        "per_query": per_query,
        "macro_query_balanced_accuracy": round(macro_bal, 6),
        "macro_query_accuracy": round(macro_acc, 6),
        "mean_positive_recall": round(mean_pos_recall, 6),
        "mean_negative_recall": round(mean_neg_recall, 6),
    }


# =============================================================================
# 4. Peripheral signal function
# =============================================================================
def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def bin_count(count, bins):
    """Bin a count into a coarse category string."""
    if count <= 0:
        return "0"
    for threshold in bins:
        if count <= threshold:
            return str(threshold) if threshold <= 1 else f"<={threshold}"
    return f">{bins[-1]}"


def compute_peripheral_signal(focal_position, focal_oid, blur_cfg):
    """
    Compute layered peripheral signal from a focal position.

    Uses ONLY public visible features (ALL_VISIBLE_FEATURES) and geometry (positions).
    Never accesses hidden_subtype, true_affordances, or hidden_category.

    Returns dict with layers 1-3.
    """
    d1 = blur_cfg["d1"]
    d2 = blur_cfg["d2"]

    layers = {"layer1_near": [], "layer2_mid": [], "layer3_background": []}

    for oid in test_oids:
        if oid == focal_oid:
            continue
        pos = positions[oid]
        dist = manhattan(focal_position, pos)
        if dist == 0:
            continue  # same cell, already visited
        if dist <= d1:
            layers["layer1_near"].append((oid, dist))
        elif dist <= d2:
            layers["layer2_mid"].append((oid, dist))
        else:
            layers["layer3_background"].append((oid, dist))

    result = {}
    for layer_name, objects_in_layer in layers.items():
        n = len(objects_in_layer)
        if n == 0:
            result[layer_name] = {
                "object_count": 0,
                "count_binned": "0",
                "feature_histogram": {},
                "density": 0.0,
            }
            continue

        # Build feature histogram from public visible features
        feature_counts = {}
        for oid, dist in objects_in_layer:
            vis = ALL_VISIBLE_FEATURES.get(oid, {})
            for f in posterior_visible_features:
                if vis.get(f, False):
                    feature_counts[f] = feature_counts.get(f, 0) + 1

        bin_cfg = blur_cfg.get(f"bin_{layer_name}", [1])

        result[layer_name] = {
            "object_count": n,
            "count_binned": bin_count(n, blur_cfg.get("count_bins", [1, 3, 6])),
            "feature_histogram": {
                f: bin_count(feature_counts.get(f, 0), bin_cfg)
                for f in posterior_visible_features
            },
            "density": round(n / max(config.GRID_ROWS * config.GRID_COLS, 1), 4),
        }

    return result


def flatten_peripheral_signal(periph):
    """Flatten peripheral signal dict into a numeric feature vector for classifiers."""
    features = []
    for layer in ["layer1_near", "layer2_mid", "layer3_background"]:
        ld = periph.get(layer, {})
        features.append(float(ld.get("object_count", 0)))
        for f in posterior_visible_features:
            hist = ld.get("feature_histogram", {})
            val = hist.get(f, "0")
            # Convert binned string to numeric
            if val == "0":
                features.append(0.0)
            elif val.startswith("<="):
                try:
                    features.append(float(val[2:]) / 10.0)
                except ValueError:
                    features.append(0.5)
            elif val.startswith(">"):
                features.append(1.0)
            else:
                try:
                    features.append(float(val) / 10.0)
                except ValueError:
                    features.append(0.0)
        features.append(float(ld.get("density", 0)))
    return features


# Blur variant configurations
BLUR_VARIANTS = {
    "F_low": {
        "d1": 1, "d2": 2,
        "bin_layer1_near": [1],        # {0, 1, 2+}
        "bin_layer2_mid": [0],          # presence only
        "bin_layer3_background": [0],   # presence only
        "count_bins": [1, 3],
        "bearing_resolution": "quadrant",
        "distance_binning": ["adjacent", "near"],
        "description": "Minimal peripheral information — only immediate neighbors",
    },
    "F_mid": {
        "d1": 2, "d2": 4,
        "bin_layer1_near": [1, 3],      # {0, 1, 2-3, 4+}
        "bin_layer2_mid": [0],          # presence only
        "bin_layer3_background": [0],   # presence only
        "count_bins": [1, 3, 6],
        "bearing_resolution": "quadrant",
        "distance_binning": ["adjacent", "near", "edge"],
        "description": "Moderate coarse local context — small neighborhood",
    },
    "F_high": {
        "d1": 3, "d2": 6,
        "bin_layer1_near": [1, 3, 6],   # {0, 1, 2-3, 4-6, 7+}
        "bin_layer2_mid": [1],          # {0, 1, 2+}
        "bin_layer3_background": [0],   # presence only
        "count_bins": [2, 5, 10],
        "bearing_resolution": "octant",
        "distance_binning": ["adjacent", "near", "mid", "edge"],
        "description": "Stronger local context — larger perceptual field, higher leakage risk",
    },
}


# =============================================================================
# 5. Leakage audit
# =============================================================================
print("\n[3/6] Leakage audit...")

# Build peripheral signal dataset: for each object as focal point,
# compute the peripheral signal using positions as-is (simulating agent at that object)
# For the leakage audit, we use each object's position as the focal position
# (simulating what the agent would see if it visited that object)

def build_peripheral_dataset(blur_cfg):
    """Build dataset of peripheral signals and associated hidden labels."""
    data = []
    for focal_oid in test_oids:
        focal_pos = positions[focal_oid]
        periph = compute_peripheral_signal(focal_pos, focal_oid, blur_cfg)
        feat_vec = flatten_peripheral_signal(periph)
        meta = object_meta[focal_oid]
        data.append({
            "oid": focal_oid,
            "features": feat_vec,
            "hidden_subtype": meta["subtype"],
            "hidden_category": meta["category"],
            "is_majority": meta["is_majority"],
            "gt_affordances": meta["gt_affordances"],
        })
    return data


def leave_one_out_accuracy(X, y, clf_train_fn):
    """Leave-one-out cross-validated accuracy."""
    n = len(y)
    if n <= 1:
        return 0.0
    correct = 0
    for i in range(n):
        X_train = [X[j] for j in range(n) if j != i]
        y_train = [y[j] for j in range(n) if j != i]
        if len(set(y_train)) < 2:
            # Only one class in training data, predict majority
            pred = y_train[0]
        else:
            pred = clf_train_fn(X_train, y_train, X[i])
        if pred == y[i]:
            correct += 1
    return correct / n


def simple_nn_classifier(X_train, y_train, x_test):
    """1-NN classifier using Euclidean distance."""
    best_dist = float('inf')
    best_label = None
    for i, x_train in enumerate(X_train):
        dist = sum((a - b) ** 2 for a, b in zip(x_train, x_test))
        if dist < best_dist:
            best_dist = dist
            best_label = y_train[i]
    return best_label


def run_leakage_test(name, X, y, base_rate=None):
    """Run a leakage probe test with all required controls."""
    n = len(y)
    if n < 2:
        return {
            "test": name, "n_samples": n,
            "base_rate_accuracy": None, "cross_validated_accuracy": None,
            "shuffled_label_control": None, "real_minus_shuffle_gap": None,
            "verdict": "INCONCLUSIVE_SMALL_SAMPLE",
            "notes": "Too few samples for meaningful classification."
        }

    # Base rate: always predict most common class
    counts = Counter(y)
    most_common_count = max(counts.values())
    base_rate_acc = most_common_count / n if n > 0 else 0.0

    # Cross-validated accuracy
    cv_acc = leave_one_out_accuracy(X, y, simple_nn_classifier)

    # Shuffled-label control
    y_shuffled = list(y)
    rng.shuffle(y_shuffled)
    shuffle_acc = leave_one_out_accuracy(X, y_shuffled, simple_nn_classifier)

    # Real minus shuffle gap
    gap = cv_acc - shuffle_acc

    # Verdict
    exceeds_base = cv_acc > base_rate_acc + 0.10
    gap_nontrivial = gap >= 0.05

    if not exceeds_base:
        verdict = "PASS"
        notes = "CV accuracy within base_rate + 0.10"
    elif not gap_nontrivial:
        verdict = "PASS_WITH_OVERFITTING_NOTE"
        notes = f"CV accuracy > base_rate+0.10 ({cv_acc:.3f} > {base_rate_acc+0.10:.3f}) but real-shuffle gap ({gap:.3f}) < 0.05 — classifier overfitting, not genuine leakage"
    else:
        verdict = "FAIL"
        notes = f"CV accuracy > base_rate+0.10 AND gap >= 0.05 — genuine leakage detected"

    return {
        "test": name,
        "n_samples": n,
        "base_rate_accuracy": round(base_rate_acc, 4),
        "cross_validated_accuracy": round(cv_acc, 4),
        "shuffled_label_control": round(shuffle_acc, 4),
        "real_minus_shuffle_gap": round(gap, 4),
        "exceeds_base_rate_plus_10": exceeds_base,
        "gap_is_nontrivial": gap_nontrivial,
        "verdict": verdict,
        "notes": notes,
    }


def run_all_leakage_tests(variant_name, blur_cfg):
    """Run full leakage audit for one variant."""
    print(f"\n  --- {variant_name}: {blur_cfg['description']} ---")
    dataset = build_peripheral_dataset(blur_cfg)
    X = [d["features"] for d in dataset]

    results = {}

    # A1: Per-category hidden subtype probe
    print("    A1: per-category subtype probe...")
    a1_results = {}
    for cat in ["wood_log", "stone_block", "apple", "wooden_pickaxe"]:
        cat_data = [d for d in dataset if d["hidden_category"] == cat]
        if len(cat_data) < 2:
            a1_results[cat] = {"verdict": "INCONCLUSIVE_SMALL_SAMPLE", "n_samples": len(cat_data)}
            continue
        cat_X = [d["features"] for d in cat_data]
        cat_y = [d["hidden_subtype"] for d in cat_data]
        a1_results[cat] = run_leakage_test(f"A1_{cat}", cat_X, cat_y)
    # Aggregate A1
    a1_fails = [c for c, r in a1_results.items() if r.get("verdict") == "FAIL"]
    a1_result_agg = {
        "per_category": a1_results,
        "any_fail": len(a1_fails) > 0,
        "failed_categories": a1_fails,
    }
    results["A1_per_category_hidden_subtype_probe"] = a1_result_agg
    print(f"      fails={a1_fails if a1_fails else 'none'}")

    # A2: Global majority vs minority probe
    print("    A2: majority vs minority probe...")
    maj_min_y = ["majority" if d["is_majority"] else "minority" for d in dataset]
    results["A2_global_majority_vs_minority_probe"] = run_leakage_test(
        "A2_majority_vs_minority", X, maj_min_y)
    print(f"      verdict={results['A2_global_majority_vs_minority_probe']['verdict']}")

    # B: True affordance probe
    print("    B: affordance probe...")
    b_results = {}
    for feat in CORE_ACTION_FEATURES:
        y_aff = [1.0 if d["gt_affordances"].get(feat, 0.0) >= 0.5 else 0.0 for d in dataset]
        if len(set(y_aff)) < 2:
            b_results[feat] = {"verdict": "INCONCLUSIVE_SMALL_SAMPLE"}
            continue
        b_results[feat] = run_leakage_test(f"B_{feat}", X, y_aff)
    b_fails = [f for f, r in b_results.items() if r.get("verdict") == "FAIL"]
    results["B_true_affordance_probe"] = {
        "per_affordance": b_results,
        "any_fail": len(b_fails) > 0,
        "failed_affordances": b_fails,
    }
    print(f"      fails={b_fails if b_fails else 'none'}")

    # C: Hidden category probe
    print("    C: hidden category probe...")
    cat_y = [d["hidden_category"] for d in dataset]
    results["C_hidden_category_probe"] = run_leakage_test("C_category", X, cat_y)
    print(f"      verdict={results['C_hidden_category_probe']['verdict']}")

    # D: Public category-like baseline recheck
    print("    D: public category-like recheck...")
    # Use peripheral signal features to predict category via k-NN,
    # then use majority subtype profile for predictions
    public_cat_preds = {}
    for d in dataset:
        neighbors = []
        for d2 in dataset:
            if d2["oid"] == d["oid"]:
                continue
            # Similarity based on peripheral features
            sim = 1.0 / (1.0 + sum((a - b) ** 2 for a, b in zip(d["features"], d2["features"])) ** 0.5)
            if sim > 0:
                neighbors.append((sim, d2["hidden_category"]))
        neighbors.sort(key=lambda x: x[0], reverse=True)
        top_k = neighbors[:max(config.INSTANCE_K, 3)]
        if top_k:
            cat_votes = {}
            for sim, cat in top_k:
                cat_votes[cat] = cat_votes.get(cat, 0) + sim
            predicted_cat = max(cat_votes, key=cat_votes.get)
        else:
            predicted_cat = None

        if predicted_cat and predicted_cat in SUBTYPE_DEFINITIONS:
            subtype_def = SUBTYPE_DEFINITIONS[predicted_cat]
            ratios = subtype_def["subtype_ratio"]
            majority_subtype = max(ratios, key=ratios.get)
            majority_profile = subtype_def["affordance_profiles"][majority_subtype]
            public_cat_preds[d["oid"]] = {
                f: float(v) for f, v in majority_profile.items()
                if f in CORE_ACTION_FEATURES
            }
        else:
            public_cat_preds[d["oid"]] = {f: 0.5 for f in CORE_ACTION_FEATURES}

    public_cat_metrics = compute_full_metrics(public_cat_preds, query_gt)
    public_cat_mb = public_cat_metrics["macro_query_balanced_accuracy"]
    results["D_public_category_like_recheck"] = {
        "macro_bal": public_cat_mb,
        "per_query": public_cat_metrics["per_query"],
        "original_public_cat_mb_1J8": 0.5,
        "delta_vs_original": round(public_cat_mb - 0.5, 4),
        "shortcut_deployable": public_cat_mb > 0.60,
        "shortcut_invalid": public_cat_mb >= 0.70,
    }
    print(f"      macro_bal={public_cat_mb:.4f}  delta_vs_1J8={public_cat_mb - 0.5:+.4f}")

    # Overall leakage verdict
    all_probes = []
    for cat, r in a1_results.items():
        if isinstance(r, dict) and "verdict" in r:
            all_probes.append(r["verdict"])
    all_probes.append(results["A2_global_majority_vs_minority_probe"]["verdict"])
    for feat, r in b_results.items():
        if isinstance(r, dict) and "verdict" in r:
            all_probes.append(r["verdict"])
    all_probes.append(results["C_hidden_category_probe"]["verdict"])

    hard_fails = [v for v in all_probes if v == "FAIL"]
    leakage_pass = len(hard_fails) == 0

    results["overall_leakage_pass"] = leakage_pass
    results["hard_fail_count"] = len(hard_fails)
    results["all_probe_verdicts"] = all_probes

    print(f"    OVERALL: leakage_pass={leakage_pass}  hard_fails={len(hard_fails)}")
    return results


leakage_results = {}
for vname, vcfg in BLUR_VARIANTS.items():
    leakage_results[vname] = run_all_leakage_tests(vname, vcfg)


# =============================================================================
# 6. Marginal-value audit
# =============================================================================
print("\n[4/6] Marginal-value audit...")

# Reference baselines (used by all variants)
ALL_FALSE_PREDS = {oid: {f: 0.0 for f in CORE_ACTION_FEATURES} for oid in test_oids}
all_false_metrics = compute_full_metrics(ALL_FALSE_PREDS, query_gt)
print(f"  all_false macro_bal={all_false_metrics['macro_query_balanced_accuracy']:.4f}")

PRIOR_ONLY_PREDS = {}
for oid in test_oids:
    PRIOR_ONLY_PREDS[oid] = {f: GLOBAL_FEATURE_PREVALENCE.get(f, 0.5) for f in CORE_ACTION_FEATURES}
prior_only_metrics = compute_full_metrics(PRIOR_ONLY_PREDS, query_gt)
print(f"  C_minus_prior_only macro_bal={prior_only_metrics['macro_query_balanced_accuracy']:.4f}")

ORACLE_PREDS = {}
for oid in test_oids:
    gt = object_meta[oid]["gt_affordances"]
    ORACLE_PREDS[oid] = {f: float(gt.get(f, 0.0)) for f in CORE_ACTION_FEATURES}
oracle_metrics = compute_full_metrics(ORACLE_PREDS, query_gt)
print(f"  oracle macro_bal={oracle_metrics['macro_query_balanced_accuracy']:.4f}")


# --- Deployable baselines ---

def run_c0b_routef_deployable(blur_cfg):
    """Run standard C0b policy. Compute peripheral signals post-hoc for diagnostics.

    C0b_RouteF_deployable predictions are IDENTICAL to original C0b:
    - Visited objects: IOM predictions from focal observation
    - Unvisited objects: global base-rate prior

    The peripheral signal is aggregate-only (histograms, counts, densities) —
    it cannot produce per-object predictions for unvisited objects. Route F's
    expected value is in policy targeting / object selection, not direct
    observe-only classification.
    """
    im = copy.deepcopy(im_base)
    from policies import C0b_ObserveOnlyPolicy as C0bPol
    policy = C0bPol(im, random.Random(SEED + 2000))
    env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
    harness = EpisodeHarness(env, policy)
    result = harness.run()
    obs = result.agent_obs

    visited_oids = [oid for oid in test_oids if obs.is_visited(oid)]
    visited_features = {}
    for oid in visited_oids:
        feats = obs.get_observed_features(oid)
        if feats:
            visited_features[oid] = dict(feats)

    # Compute peripheral signals from each visited position (for diagnostics only)
    peripheral_signals = {}
    for oid in visited_oids:
        pos = positions[oid]
        periph = compute_peripheral_signal(pos, oid, blur_cfg)
        peripheral_signals[oid] = periph

    # C0b_RouteF_deployable: IOM for visited, global prior for unvisited
    # Identical prediction rule to original C0b
    per_object = {}
    for oid in test_oids:
        if oid in visited_features:
            fake_obj = {"id": oid, "visible_features": visited_features[oid]}
            probs = im.predict_all_affordances(fake_obj)
            per_object[oid] = {f: probs.get(f, 0.5) for f in CORE_ACTION_FEATURES}
        else:
            per_object[oid] = {f: GLOBAL_FEATURE_PREVALENCE.get(f, 0.5)
                               for f in CORE_ACTION_FEATURES}

    return {"per_object": per_object, "per_query": {}}, obs, peripheral_signals, visited_oids


def run_random_probe_routef_deployable(blur_cfg):
    """Run random-probe policy with standard predictions (no peripheral blending).

    Deployable baseline. Predictions use:
    - Visited objects: IOM from focal observation (with probe outcomes incorporated)
    - Unvisited objects: global base-rate prior

    No ad-hoc peripheral blending. The peripheral signal is computed post-hoc
    for diagnostics only.
    """
    im = copy.deepcopy(im_base)

    class RandProbePol(MiniMCPolicy):
        def __init__(self, im, rng):
            super().__init__()
            self._im = im
            self._rng = rng

        def reset(self, view):
            pass

        def select_next_object(self, view):
            unvisited = view.get_unvisited_objects()
            if not unvisited:
                return None
            best_oid = None
            best_cost = float('inf')
            for oid in unvisited:
                total = view.compute_reach_cost(oid) + view.observe_cost
                if view.can_afford(total) and total < best_cost:
                    best_cost = total
                    best_oid = oid
            return best_oid

        def decide_probe(self, view, object_id):
            if view.can_afford(view.probe_cost) and self._rng.random() < 0.3:
                candidates = [a for a in MAIN_CANDIDATE_ACTIONS if a != "tap_sound"]
                action = self._rng.choice(candidates) if candidates else None
                if action:
                    return True, action
            return False, None

        def on_probe_result(self, view, object_id, action, outcome):
            pass

        def get_answer(self, view):
            per_object = {}
            for oid in test_oids:
                feats = view.get_observed_features(oid)
                if feats is not None:
                    fake_obj = {"id": oid, "visible_features": dict(feats)}
                    probs = self._im.predict_all_affordances(fake_obj)
                    per_object[oid] = {f: probs.get(f, 0.5) for f in CORE_ACTION_FEATURES}
                else:
                    per_object[oid] = {f: GLOBAL_FEATURE_PREVALENCE.get(f, 0.5)
                                       for f in CORE_ACTION_FEATURES}
            return {"per_object": per_object, "per_query": {}}

    policy = RandProbePol(im, random.Random(SEED + 3000))
    env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
    harness = EpisodeHarness(env, policy)
    result = harness.run()
    obs = result.agent_obs

    # Post-hoc peripheral computation (for diagnostics only)
    visited_oids = [oid for oid in test_oids if obs.is_visited(oid)]
    peripheral_signals = {}
    for oid in visited_oids:
        pos = positions[oid]
        periph = compute_peripheral_signal(pos, oid, blur_cfg)
        peripheral_signals[oid] = periph

    # Standard predictions: IOM for visited, global prior for unvisited
    per_object = {}
    for oid in test_oids:
        feats = obs.get_observed_features(oid)
        if feats is not None:
            fake_obj = {"id": oid, "visible_features": dict(feats)}
            probs = im.predict_all_affordances(fake_obj)
            per_object[oid] = {f: probs.get(f, 0.5) for f in CORE_ACTION_FEATURES}
        else:
            per_object[oid] = {f: GLOBAL_FEATURE_PREVALENCE.get(f, 0.5)
                               for f in CORE_ACTION_FEATURES}

    return {"per_object": per_object, "per_query": {}}, obs, peripheral_signals, visited_oids


# --- Diagnostic: peripheral information upper bound ---

def _peripheral_layer_to_feature_vector(layer_dict):
    """Convert a single layer dict to a numeric feature vector for classifiers."""
    features = []
    features.append(float(layer_dict.get("object_count", 0)))
    for f in posterior_visible_features:
        hist = layer_dict.get("feature_histogram", {})
        val = hist.get(f, "0")
        if val == "0":
            features.append(0.0)
        elif val.startswith("<="):
            try:
                features.append(float(val[2:]) / 10.0)
            except ValueError:
                features.append(0.5)
        elif val.startswith(">"):
            features.append(1.0)
        else:
            try:
                features.append(float(val) / 10.0)
            except ValueError:
                features.append(0.0)
    features.append(float(layer_dict.get("density", 0.0)))
    return features


def run_peripheral_information_upper_bound(peripheral_signals, blur_cfg):
    """Estimate information content of peripheral signals using optimal access.

    DIAGNOSTIC ONLY — NON-DEPLOYABLE.

    For each unvisited object, finds the best peripheral signal (closest layer
    from any visitor), extracts its feature vector, and uses 1-NN LOO to
    predict hidden properties.

    This estimates the MAXIMUM information a classifier could extract from the
    peripheral signal — an upper bound. It is NOT a deployable policy.
    """
    d1 = blur_cfg["d1"]
    d2 = blur_cfg["d2"]

    # For each object, find the best peripheral signal that included it
    best_signals = {}
    for oid in test_oids:
        oid_pos = positions[oid]
        best_layer_rank = None
        best_layer_dict = None

        for visitor_oid, periph in peripheral_signals.items():
            visitor_pos = positions[visitor_oid]
            dist = manhattan(visitor_pos, oid_pos)

            for layer_name, layer_rank in [("layer1_near", 0), ("layer2_mid", 1), ("layer3_background", 2)]:
                in_layer = False
                if layer_name == "layer1_near" and dist <= d1:
                    in_layer = True
                elif layer_name == "layer2_mid" and d1 < dist <= d2:
                    in_layer = True
                elif layer_name == "layer3_background" and dist > d2:
                    in_layer = True

                if in_layer and (best_layer_rank is None or layer_rank < best_layer_rank):
                    best_layer_rank = layer_rank
                    best_layer_dict = periph.get(layer_name, {})

        if best_layer_dict is not None:
            best_signals[oid] = _peripheral_layer_to_feature_vector(best_layer_dict)
        else:
            best_signals[oid] = None

    oids_with_signal = [oid for oid in test_oids if best_signals[oid] is not None]
    oids_without = [oid for oid in test_oids if best_signals[oid] is None]

    results = {
        "n_with_signal": len(oids_with_signal),
        "n_without_signal": len(oids_without),
        "label": "diagnostic_non_deployable",
        "note": "Uses best-signal selection + 1-NN LOO classifier. NOT a deployable policy."
    }

    if len(oids_with_signal) < 2:
        results["verdict"] = "INCONCLUSIVE — too few objects with peripheral signal"
        return results

    X = [best_signals[oid] for oid in oids_with_signal]

    # A2-style: majority vs minority probe
    maj_min_y = ["majority" if object_meta[oid]["is_majority"] else "minority"
                 for oid in oids_with_signal]
    if len(set(maj_min_y)) >= 2:
        results["A2_majority_vs_minority"] = run_leakage_test(
            "UB_A2_majority_vs_minority", X, maj_min_y)

    # B-style: per-affordance probe
    for feat in CORE_ACTION_FEATURES:
        y_aff = [1.0 if object_meta[oid]["gt_affordances"].get(feat, 0.0) >= 0.5 else 0.0
                 for oid in oids_with_signal]
        if len(set(y_aff)) >= 2:
            results[f"B_{feat}"] = run_leakage_test(f"UB_B_{feat}", X, y_aff)

    # C-style: hidden category probe
    cat_y = [object_meta[oid]["category"] for oid in oids_with_signal]
    if len(set(cat_y)) >= 2:
        results["C_hidden_category"] = run_leakage_test("UB_C_category", X, cat_y)

    verdicts = []
    for k, v in results.items():
        if isinstance(v, dict) and "verdict" in v:
            verdicts.append(v["verdict"])
    hard_fails = [v for v in verdicts if v == "FAIL"]
    results["hard_fail_count"] = len(hard_fails)
    results["contains_classifier_extractable_info"] = len(hard_fails) > 0

    return results


# Reference baselines
ALL_FALSE_PREDS = {oid: {f: 0.0 for f in CORE_ACTION_FEATURES} for oid in test_oids}
all_false_metrics = compute_full_metrics(ALL_FALSE_PREDS, query_gt)
print(f"  all_false macro_bal={all_false_metrics['macro_query_balanced_accuracy']:.4f}")

# C_minus_prior_only: global base rate predictions
PRIOR_ONLY_PREDS = {}
for oid in test_oids:
    PRIOR_ONLY_PREDS[oid] = {f: GLOBAL_FEATURE_PREVALENCE.get(f, 0.5) for f in CORE_ACTION_FEATURES}
prior_only_metrics = compute_full_metrics(PRIOR_ONLY_PREDS, query_gt)
print(f"  C_minus_prior_only macro_bal={prior_only_metrics['macro_query_balanced_accuracy']:.4f}")

# Oracle
ORACLE_PREDS = {}
for oid in test_oids:
    gt = object_meta[oid]["gt_affordances"]
    ORACLE_PREDS[oid] = {f: float(gt.get(f, 0.0)) for f in CORE_ACTION_FEATURES}
oracle_metrics = compute_full_metrics(ORACLE_PREDS, query_gt)
print(f"  oracle macro_bal={oracle_metrics['macro_query_balanced_accuracy']:.4f}")


marginal_results = {}
for vname, vcfg in BLUR_VARIANTS.items():
    print(f"\n  --- {vname} marginal-value audit ---")

    # C0b_RouteF_deployable (predictions identical to original C0b)
    c0b_preds, c0b_obs, c0b_periph, c0b_visited = run_c0b_routef_deployable(vcfg)
    c0b_metrics = compute_full_metrics(c0b_preds["per_object"], query_gt)
    c0b_mb = c0b_metrics["macro_query_balanced_accuracy"]
    n_visited = len(c0b_visited)
    delta_vs_original = c0b_mb - 0.5871
    identical = abs(delta_vs_original) < 0.0001
    print(f"    C0b_RouteF_deployable macro_bal={c0b_mb:.4f}  visited={n_visited}  "
          f"delta_vs_original_C0b={delta_vs_original:+.4f}"
          f"{' (identical to C0b as expected)' if identical else ''}")

    # Peripheral information upper bound (DIAGNOSTIC only, non-deployable)
    ub_results = run_peripheral_information_upper_bound(c0b_periph, vcfg)
    ub_fail_count = ub_results.get("hard_fail_count", 0)
    ub_contains_info = ub_results.get("contains_classifier_extractable_info", False)
    ub_a2 = ub_results.get("A2_majority_vs_minority", {})
    ub_a2_cv = ub_a2.get("cross_validated_accuracy", 0) if isinstance(ub_a2, dict) else 0
    ub_near_oracle = ub_a2_cv > 0.85
    print(f"    peripheral_information_upper_bound: hard_fails={ub_fail_count}  "
          f"contains_info={ub_contains_info}  A2_CV={ub_a2_cv:.4f}  "
          f"near_oracle={ub_near_oracle}")

    # Random_probe_RouteF_deployable
    rand_preds, rand_obs, rand_periph, rand_visited = run_random_probe_routef_deployable(vcfg)
    rand_metrics = compute_full_metrics(rand_preds["per_object"], query_gt)
    rand_mb = rand_metrics["macro_query_balanced_accuracy"]
    print(f"    random_probe_RouteF_deployable macro_bal={rand_mb:.4f}  "
          f"delta_vs_C0b_RouteF={rand_mb - c0b_mb:+.4f}")

    # Oracle gap
    oracle_gap = oracle_metrics["macro_query_balanced_accuracy"] - c0b_mb

    # Saturation check
    saturation = c0b_mb > 0.70

    # Public category like recheck (from leakage audit)
    public_cat_mb = leakage_results[vname]["D_public_category_like_recheck"]["macro_bal"]

    marginal_results[vname] = {
        "C0b_RouteF_deployable": {
            "macro_bal": c0b_mb,
            "visit_count": n_visited,
            "delta_vs_original_C0b": round(delta_vs_original, 4),
            "identical_to_original_C0b": identical,
            "per_query": c0b_metrics["per_query"],
            "label": "deployable",
            "note": "Predictions identical to original C0b — peripheral signal is aggregate-only, cannot produce per-object predictions for unvisited objects. Route F value is in policy targeting / object selection, not direct classification."
        },
        "peripheral_information_upper_bound": {
            "hard_fail_count": ub_fail_count,
            "contains_classifier_extractable_info": ub_contains_info,
            "near_oracle": ub_near_oracle,
            "A2_majority_vs_minority_cv": ub_a2_cv,
            "A2_majority_vs_minority": ub_results.get("A2_majority_vs_minority", {}),
            "C_hidden_category": ub_results.get("C_hidden_category", {}),
            "label": "diagnostic_non_deployable",
            "note": "Uses best-signal selection + 1-NN LOO classifier. Estimates information upper bound. NOT a deployable policy. Excluded from Pareto frontier."
        },
        "random_probe_RouteF_deployable": {
            "macro_bal": rand_mb,
            "delta_vs_C0b_RouteF_deployable": round(rand_mb - c0b_mb, 4),
            "label": "deployable",
            "note": "Standard predictions (IOM for visited, global prior for unvisited). Random probing adds some probe outcomes."
        },
        "all_false_macro_bal": all_false_metrics["macro_query_balanced_accuracy"],
        "C_minus_prior_only_macro_bal": prior_only_metrics["macro_query_balanced_accuracy"],
        "oracle_macro_bal": oracle_metrics["macro_query_balanced_accuracy"],
        "oracle_minus_C0b_RouteF_deployable": round(oracle_gap, 4),
        "public_category_like_macro_bal": public_cat_mb,
        "hidden_category_diagnostic_macro_bal": 0.975,
        "saturation_status": "SATURATED" if saturation else "ok",
        "C15b_original_reference": 0.6142,
        "C0b_original_reference": 0.5871,
    }


# =============================================================================
# 7. Variant recommendation
# =============================================================================
print("\n[5/6] Variant recommendation...")

variant_assessment = {}
for vname in ["F_low", "F_mid", "F_high"]:
    leak_pass = leakage_results[vname]["overall_leakage_pass"]
    marg = marginal_results[vname]
    c0b_dep = marg["C0b_RouteF_deployable"]
    ub = marg["peripheral_information_upper_bound"]

    c0b_mb = c0b_dep["macro_bal"]
    oracle_gap = marg["oracle_minus_C0b_RouteF_deployable"]
    public_cat_mb = marg["public_category_like_macro_bal"]
    ub_near_oracle = ub["near_oracle"]
    saturation = marg["saturation_status"] == "SATURATED"

    # Marginal value pass criteria:
    # 1. Leakage must pass (no hidden info leaked through peripheral signal)
    # 2. C0b must not saturate (<= 0.70 — task remains challenging)
    # 3. Oracle gap must be meaningful (>= 0.10 — Route F doesn't trivialize task)
    # 4. Public category shortcut must not deployably exist (<= 0.60)
    # 5. Peripheral upper bound must not be near oracle (diagnostic ceiling check)
    # NOTE: C0b_RouteF_deployable is NOT required to improve over original C0b.
    # For aggregate-only signals, C0b_RouteF == C0b_original is expected.
    # Route F's value is in policy targeting / object selection, not direct classification.
    marg_pass = (
        leak_pass and
        not saturation and
        oracle_gap >= 0.10 and
        public_cat_mb <= 0.60 and
        not ub_near_oracle
    )

    variant_assessment[vname] = {
        "leakage_pass": leak_pass,
        "marginal_value_pass": marg_pass,
        "C0b_RouteF_deployable_macro_bal": c0b_mb,
        "C0b_RouteF_identical_to_original_C0b": c0b_dep["identical_to_original_C0b"],
        "oracle_gap": oracle_gap,
        "public_cat_mb": public_cat_mb,
        "saturation": marg["saturation_status"],
        "peripheral_ub_near_oracle": ub_near_oracle,
        "peripheral_ub_hard_fails": ub["hard_fail_count"],
    }

    print(f"  {vname}: leak_pass={leak_pass}  marg_pass={marg_pass}  "
          f"C0b_mb={c0b_mb:.4f}  identical_to_orig={c0b_dep['identical_to_original_C0b']}  "
          f"oracle_gap={oracle_gap:.4f}  pub_cat={public_cat_mb:.4f}  "
          f"ub_near_oracle={ub_near_oracle}")

# Choose variant
passed_variants = [v for v, a in variant_assessment.items()
                   if a["leakage_pass"] and a["marginal_value_pass"]]

if passed_variants:
    if "F_mid" in passed_variants:
        recommended = "F_mid"
    elif "F_low" in passed_variants:
        recommended = "F_low"
    else:
        recommended = passed_variants[0]
    route_f_audit_passed = True
    proceed_to_1J11 = True
else:
    recommended = "none"
    route_f_audit_passed = False
    proceed_to_1J11 = False

print(f"\n  Recommended variant: {recommended}")
print(f"  Route F audit passed: {route_f_audit_passed}")
print(f"  Proceed to 1J11 C15F policy eval: {proceed_to_1J11}")


# =============================================================================
# 8. Save outputs
# =============================================================================
print("\n[6/6] Saving outputs...")

output = {
    "block_id": "1J10",
    "condition": "C4_instance_subtype_cued_v1",
    "seed": 101,
    "budget": BUDGET,
    "desc": "Route F leakage and marginal-value audit. No C15F policy claim.",
    "design_validation": {
        "local_prevalence_predictor_removed": True,
        "deployable_vs_diagnostic_separated": True,
        "C0b_RouteF_uses_only_IOM_and_global_prior": True,
        "peripheral_information_upper_bound_is_diagnostic_non_deployable": True,
        "no_ad_hoc_blending": True,
    },
    "blur_variants": {v: {"d1": c["d1"], "d2": c["d2"], "description": c["description"]}
                      for v, c in BLUR_VARIANTS.items()},
    "reference_baselines": {
        "all_false_macro_bal": all_false_metrics["macro_query_balanced_accuracy"],
        "C_minus_prior_only_macro_bal": prior_only_metrics["macro_query_balanced_accuracy"],
        "C0b_original_macro_bal": 0.5871,
        "C15b_original_macro_bal": 0.6142,
        "oracle_original_macro_bal": 1.0,
        "hidden_category_majority_diagnostic": 0.975,
    },
    "leakage_audit": {v: leakage_results[v] for v in BLUR_VARIANTS},
    "marginal_value_audit": {v: marginal_results[v] for v in BLUR_VARIANTS},
    "variant_assessment": variant_assessment,
    "recommendation": {
        "recommended_route_f_variant": recommended,
        "route_f_audit_passed": route_f_audit_passed,
        "proceed_to_1J11_c15f_policy_eval": proceed_to_1J11,
        "no_c15f_policy_claim_in_block_1J10": True,
        "ready_for_multiseed": False,
    },
}

json_path = os.path.join(CURRENT_DIR, "runs",
                         "calibration_block1j10_route_f_leakage_marginal_audit_c4_seed101.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"  Saved: {json_path}")

elapsed = time.time() - t0
print(f"\n{'=' * 60}")
print(f"[block_done]")
print(f"block_id=1J10")
print(f"condition=C4_instance_subtype_cued_v1")
print(f"seed=101")
print(f"local_prevalence_predictor_removed=true")
print(f"deployable_vs_diagnostic_separated=true")
print(f"F_low_leakage_pass={variant_assessment['F_low']['leakage_pass']}")
print(f"F_mid_leakage_pass={variant_assessment['F_mid']['leakage_pass']}")
print(f"F_high_leakage_pass={variant_assessment['F_high']['leakage_pass']}")
print(f"F_low_marginal_value_pass={variant_assessment['F_low']['marginal_value_pass']}")
print(f"F_mid_marginal_value_pass={variant_assessment['F_mid']['marginal_value_pass']}")
print(f"F_high_marginal_value_pass={variant_assessment['F_high']['marginal_value_pass']}")
print(f"recommended_route_f_variant={recommended}")
best_v = recommended if recommended != "none" else "F_mid"
best_marg = marginal_results.get(best_v, marginal_results.get("F_mid", {}))
best_c0b = best_marg.get("C0b_RouteF_deployable", {})
best_ub = best_marg.get("peripheral_information_upper_bound", {})
if isinstance(best_c0b, dict):
    print(f"c0b_routef_deployable_macro_bal_best_variant={best_c0b.get('macro_bal', 'N/A')}")
    print(f"c0b_routef_identical_to_original={best_c0b.get('identical_to_original_C0b', 'N/A')}")
else:
    print(f"c0b_routef_deployable_macro_bal_best_variant=N/A")
    print(f"c0b_routef_identical_to_original=N/A")
print(f"public_category_like_macro_bal_best_variant={best_marg.get('public_category_like_macro_bal', 'N/A')}")
print(f"oracle_minus_c0b_best_variant={best_marg.get('oracle_minus_C0b_RouteF_deployable', 'N/A')}")
print(f"peripheral_ub_near_oracle={best_ub.get('near_oracle', 'N/A') if isinstance(best_ub, dict) else 'N/A'}")
print(f"route_f_audit_passed={route_f_audit_passed}")
print(f"proceed_to_1J11_c15f_policy_eval={proceed_to_1J11}")
print(f"ready_for_multiseed=false")
print(f"elapsed={elapsed:.1f}s")
print(f"{'=' * 60}")
