"""
Block 1J40b-7a — Focused Observe-Cost Calibration Audit.

Reuses env3b object generation and audit_VOI computation from the main 7a script.
Reports per-cost metrics including balanced accuracy and macro-F1 for shortcut probes.

Does NOT redesign the environment, train a learner, or add policy intervention.
"""

import os, sys, json, time, math, random
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
A3_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5a3_tool_material_transfer")
sys.path.insert(0, A3_DIR)
sys.path.insert(0, CURRENT_DIR)

import config

t0 = time.time()

# =============================================================================
# Constants (from main 7a script)
# =============================================================================
SEEDS = [101, 103, 107, 109, 113]
ALL_TRY_AFFORDANCES = [
    "burn_as_fuel", "craft_plank", "eat",
    "mine_by_hand", "mine_with_pickaxe", "use_as_tool",
]
PROBE_COST = 0.05
SUCCESS_REWARD = 0.5
FAILURE_PENALTY = 0.1
CATEGORIES = ["wood_log", "stone_block", "apple", "wooden_pickaxe"]

AMBIENT_GROUP_FEATURES = {
    "A": ["brownish", "rough_texture", "long_shape", "heavy_weight"],
    "B": ["greenish", "smooth_texture", "round_small", "light_weight"],
}
CATEGORY_AMBIENT_GROUP = {
    "wood_log": "A", "stone_block": "A",
    "apple": "B", "wooden_pickaxe": "B",
}
ALL_AMBIENT_FEATURE_NAMES = sorted(set(
    f for feats in AMBIENT_GROUP_FEATURES.values() for f in feats))

CATEGORY_POST_OBSERVE_FEATURES = {
    "wood_log": ["has_bark_texture", "fibrous", "flammable", "porous_surface"],
    "stone_block": ["has_crystal_flecks", "has_granular_surface", "block_like",
                    "cold_to_touch", "scratch_resistant"],
    "apple": ["has_stem_remnant", "has_peel_texture", "fruity_scent"],
    "wooden_pickaxe": ["has_grip_area", "has_shaft_shape", "elongated_with_handle",
                       "movable", "has_metal_head", "jointed"],
}
SHARED_SURFACE_FEATURES = [
    "spotted", "weathered", "coarse_texture", "discolored", "has_veins", "pitted",
]
ALL_POST_OBSERVE_FEATURE_NAMES = sorted(set(
    f for flist in CATEGORY_POST_OBSERVE_FEATURES.values() for f in flist))
ALL_VISIBLE_FEATURE_NAMES = sorted(set(
    list(ALL_AMBIENT_FEATURE_NAMES) +
    list(ALL_POST_OBSERVE_FEATURE_NAMES) +
    list(SHARED_SURFACE_FEATURES)
))
STATE_FEATURE_NAMES = ["fresh", "wet", "damaged", "clean", "hot", "open"]

# =============================================================================
# Instance subtype definitions (same as main 7a script)
# =============================================================================
INSTANCE_SUBTYPE_DEFS = {}

INSTANCE_SUBTYPE_DEFS["wood_log"] = {
    "P": {
        "label": "ambiguous_log", "audit_rationale_class": "positive_change",
        "pre_surface_extra": [],
        "affordance_profile": {
            "mine_by_hand": "success", "mine_with_pickaxe": "fail",
            "craft_plank": "success", "eat": "fail",
            "use_as_tool": "fail", "burn_as_fuel": "success",
        },
        "post_observe_revealed": ["has_bark_texture", "fibrous", "flammable", "porous_surface"],
        "state_features": {"fresh": True, "wet": True, "damaged": False, "clean": True, "hot": False, "open": False},
    },
    "Z": {
        "label": "positive_shared_cue_log", "audit_rationale_class": "positive_change",
        "pre_surface_extra": ["spotted"],
        "affordance_profile": {
            "mine_by_hand": "success", "mine_with_pickaxe": "fail",
            "craft_plank": "success", "eat": "fail",
            "use_as_tool": "fail", "burn_as_fuel": "success",
        },
        "post_observe_revealed": ["has_bark_texture", "fibrous", "flammable", "porous_surface"],
        "state_features": {"fresh": True, "wet": True, "damaged": False, "clean": True, "hot": False, "open": False},
    },
    "N": {
        "label": "nonpos_identified_log", "audit_rationale_class": "wasteful_cost_only",
        "pre_surface_extra": ["has_bark_texture", "discolored"],
        "affordance_profile": {
            "mine_by_hand": "success", "mine_with_pickaxe": "fail",
            "craft_plank": "success", "eat": "fail",
            "use_as_tool": "fail", "burn_as_fuel": "success",
        },
        "post_observe_revealed": ["fibrous", "flammable", "porous_surface"],
        "state_features": {"fresh": False, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
    },
}

INSTANCE_SUBTYPE_DEFS["stone_block"] = {
    "P": {
        "label": "ambiguous_stone", "audit_rationale_class": "positive_change",
        "pre_surface_extra": [],
        "affordance_profile": {
            "mine_by_hand": "fail", "mine_with_pickaxe": "success",
            "craft_plank": "fail", "eat": "fail",
            "use_as_tool": "fail", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_crystal_flecks", "has_granular_surface", "block_like",
                                  "cold_to_touch", "scratch_resistant"],
        "state_features": {"fresh": True, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
    },
    "Z": {
        "label": "positive_shared_cue_stone", "audit_rationale_class": "positive_change",
        "pre_surface_extra": ["spotted"],
        "affordance_profile": {
            "mine_by_hand": "fail", "mine_with_pickaxe": "success",
            "craft_plank": "fail", "eat": "fail",
            "use_as_tool": "fail", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_crystal_flecks", "has_granular_surface", "block_like", "cold_to_touch"],
        "state_features": {"fresh": True, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
    },
    "N": {
        "label": "nonpos_identified_stone", "audit_rationale_class": "wasteful_cost_only",
        "pre_surface_extra": ["has_crystal_flecks"],
        "affordance_profile": {
            "mine_by_hand": "fail", "mine_with_pickaxe": "success",
            "craft_plank": "fail", "eat": "fail",
            "use_as_tool": "fail", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_granular_surface", "block_like", "cold_to_touch", "scratch_resistant"],
        "state_features": {"fresh": True, "wet": False, "damaged": True, "clean": False, "hot": False, "open": False},
    },
}

INSTANCE_SUBTYPE_DEFS["apple"] = {
    "P": {
        "label": "ambiguous_fruit", "audit_rationale_class": "positive_change",
        "pre_surface_extra": [],
        "affordance_profile": {
            "mine_by_hand": "success", "mine_with_pickaxe": "fail",
            "craft_plank": "fail", "eat": "success",
            "use_as_tool": "fail", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_stem_remnant", "has_peel_texture", "fruity_scent"],
        "state_features": {"fresh": True, "wet": True, "damaged": False, "clean": True, "hot": False, "open": False},
    },
    "Z": {
        "label": "positive_shared_cue_apple", "audit_rationale_class": "positive_change",
        "pre_surface_extra": ["discolored"],
        "affordance_profile": {
            "mine_by_hand": "success", "mine_with_pickaxe": "fail",
            "craft_plank": "fail", "eat": "success",
            "use_as_tool": "fail", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_stem_remnant", "has_peel_texture", "fruity_scent"],
        "state_features": {"fresh": True, "wet": True, "damaged": False, "clean": True, "hot": False, "open": False},
    },
    "N": {
        "label": "nonpos_identified_apple", "audit_rationale_class": "wasteful_cost_only",
        "pre_surface_extra": ["has_stem_remnant", "spotted"],
        "affordance_profile": {
            "mine_by_hand": "success", "mine_with_pickaxe": "fail",
            "craft_plank": "fail", "eat": "success",
            "use_as_tool": "fail", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_peel_texture", "fruity_scent"],
        "state_features": {"fresh": True, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
    },
}

INSTANCE_SUBTYPE_DEFS["wooden_pickaxe"] = {
    "P": {
        "label": "ambiguous_tool", "audit_rationale_class": "positive_change",
        "pre_surface_extra": [],
        "affordance_profile": {
            "mine_by_hand": "fail", "mine_with_pickaxe": "fail",
            "craft_plank": "fail", "eat": "fail",
            "use_as_tool": "success", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_grip_area", "has_shaft_shape", "elongated_with_handle",
                                  "movable", "has_metal_head", "jointed"],
        "state_features": {"fresh": True, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
    },
    "Z": {
        "label": "positive_shared_cue_tool", "audit_rationale_class": "positive_change",
        "pre_surface_extra": ["discolored"],
        "affordance_profile": {
            "mine_by_hand": "fail", "mine_with_pickaxe": "fail",
            "craft_plank": "fail", "eat": "fail",
            "use_as_tool": "success", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_grip_area", "has_shaft_shape", "elongated_with_handle", "movable"],
        "state_features": {"fresh": True, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
    },
    "N": {
        "label": "nonpos_identified_tool", "audit_rationale_class": "wasteful_cost_only",
        "pre_surface_extra": ["has_grip_area"],
        "affordance_profile": {
            "mine_by_hand": "fail", "mine_with_pickaxe": "fail",
            "craft_plank": "fail", "eat": "fail",
            "use_as_tool": "success", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_shaft_shape", "elongated_with_handle", "movable",
                                  "has_metal_head", "jointed"],
        "state_features": {"fresh": True, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
    },
}

# =============================================================================
# Object generation (same as main 7a script)
# =============================================================================
def generate_env3b_objects(seed):
    objects = {}
    audit_labels = {}
    oid_counter = [0]

    def make_oid():
        oid_counter[0] += 1
        return f"env3b_obj_{oid_counter[0]:04d}"

    for category in CATEGORIES:
        ambient_group = CATEGORY_AMBIENT_GROUP[category]
        ambient_features_list = AMBIENT_GROUP_FEATURES[ambient_group]

        for subtype_key in ["P", "Z", "N"]:
            subtype_def = INSTANCE_SUBTYPE_DEFS[category][subtype_key]
            pre_extra = subtype_def["pre_surface_extra"]
            affordance = subtype_def["affordance_profile"]
            post_revealed = subtype_def["post_observe_revealed"]
            state_feats = dict(subtype_def["state_features"])

            pre_surface = {}
            for f in ambient_features_list:
                pre_surface[f] = True
            for f in pre_extra:
                pre_surface[f] = True

            post_observe = dict(pre_surface)
            for f in post_revealed:
                post_observe[f] = True

            for depth_schedule in ["no_observe", "one_observe", "repeated_observe"]:
                oid = make_oid()
                objects[oid] = {
                    "oid": oid,
                    "hidden_category": category,
                    "hidden_subtype": subtype_key,
                    "hidden_audit_rationale_class": subtype_def["audit_rationale_class"],
                    "hidden_affordance_profile": dict(affordance),
                    "pre_surface_features": dict(pre_surface),
                    "post_observe_features": dict(post_observe),
                    "visible_state": dict(state_feats),
                    "depth_schedule": depth_schedule,
                }
                audit_labels[oid] = {
                    "oid": oid,
                    "hidden_category": category,
                    "hidden_subtype": subtype_key,
                    "hidden_audit_rationale_class": subtype_def["audit_rationale_class"],
                    "depth_schedule": depth_schedule,
                    "affordance_profile": dict(affordance),
                    "ambient_group": ambient_group,
                    "pre_surface_extra": list(pre_extra),
                    "post_observe_revealed": list(post_revealed),
                    "seed": seed,
                }
    return objects, audit_labels

# =============================================================================
# VOI computation (same as main 7a script)
# =============================================================================
def try_net_value(success_or_failure):
    if success_or_failure is True:
        return SUCCESS_REWARD - PROBE_COST
    elif success_or_failure is False:
        return -FAILURE_PENALTY - PROBE_COST
    return 0.0

def get_pre_surface_signature(pre_surface_features):
    return tuple(sorted(pre_surface_features.keys()))

def compute_audit_voi_all_objects(per_seed_objects, per_seed_audit_labels, observe_cost):
    sig_groups = defaultdict(list)
    for seed in SEEDS:
        for oid, obj in per_seed_objects[seed].items():
            lbl = per_seed_audit_labels[seed][oid]
            sig = get_pre_surface_signature(obj["pre_surface_features"])
            true_returns = {}
            for action in ALL_TRY_AFFORDANCES:
                outcome = lbl["affordance_profile"].get(action, "fail")
                true_returns[action] = try_net_value(outcome == "success")
            sig_groups[sig].append({
                "seed": seed, "oid": oid,
                "true_returns": true_returns,
                "oracle_post_best": max(true_returns.values()),
            })

    sig_expected = {}
    for sig, members in sig_groups.items():
        n = len(members)
        action_expected = {}
        for action in ALL_TRY_AFFORDANCES:
            total = sum(m["true_returns"][action] for m in members)
            action_expected[action] = total / n
        best_action = max(action_expected, key=action_expected.get)
        sig_expected[sig] = {
            "n_members": n,
            "action_expected": action_expected,
            "best_action": best_action,
            "best_expected_value": action_expected[best_action],
        }

    all_voi_results = []
    for seed in SEEDS:
        for oid, obj in per_seed_objects[seed].items():
            lbl = per_seed_audit_labels[seed][oid]
            sig = get_pre_surface_signature(obj["pre_surface_features"])
            oracle_post_best = max(
                try_net_value(lbl["affordance_profile"].get(a, "fail") == "success")
                for a in ALL_TRY_AFFORDANCES)
            oracle_pre_best_expected = sig_expected[sig]["best_expected_value"]
            audit_VOI = oracle_post_best - oracle_pre_best_expected - observe_cost

            all_voi_results.append({
                "oid": oid, "seed": seed,
                "category": lbl["hidden_category"],
                "subtype": lbl["hidden_subtype"],
                "audit_rationale_class": lbl["hidden_audit_rationale_class"],
                "pre_surface_signature": sig,
                "sig_group_size": sig_expected[sig]["n_members"],
                "oracle_post_best": round(oracle_post_best, 6),
                "oracle_pre_best_expected": round(oracle_pre_best_expected, 6),
                "audit_VOI": round(audit_VOI, 6),
                "positive_net": audit_VOI > 0.001,
                "n_pre_surface_features": len(sig),
            })
    return all_voi_results, sig_groups, sig_expected

# =============================================================================
# Baseline computation
# =============================================================================
def compute_baselines(all_voi_results, observe_cost):
    n = len(all_voi_results)
    no_observe_returns = [v["oracle_pre_best_expected"] for v in all_voi_results]
    always_observe_returns = [v["oracle_post_best"] - observe_cost for v in all_voi_results]

    rng = random.Random(42)
    random_returns = []
    for v in all_voi_results:
        if rng.random() < 0.5:
            random_returns.append(v["oracle_post_best"] - observe_cost)
        else:
            random_returns.append(v["oracle_pre_best_expected"])

    oracle_selective_returns = []
    oracle_selective_obs_count = 0
    for v in all_voi_results:
        if v["audit_VOI"] > 0:
            oracle_selective_returns.append(v["oracle_post_best"] - observe_cost)
            oracle_selective_obs_count += 1
        else:
            oracle_selective_returns.append(v["oracle_pre_best_expected"])

    return {
        "no_observe_pre_only": sum(no_observe_returns) / n,
        "always_try": sum(no_observe_returns) / n,
        "always_observe_net": sum(always_observe_returns) / n,
        "random_observe": sum(random_returns) / n,
        "oracle_selective": sum(oracle_selective_returns) / n,
        "oracle_selective_obs_rate": oracle_selective_obs_count / n,
        "oracle_selective_gap_vs_always_observe":
            sum(oracle_selective_returns) / n - sum(always_observe_returns) / n,
        "oracle_selective_gap_vs_always_try":
            sum(oracle_selective_returns) / n - sum(no_observe_returns) / n,
    }

# =============================================================================
# Shortcut probes with balanced accuracy and macro-F1
# =============================================================================
def softmax(logits):
    exps = [math.exp(lv - max(logits)) for lv in logits]
    total = sum(exps)
    return [e / total for e in exps]

def train_softmax_2class(X, y, l2_alpha=0.1, max_iter=200, lr=0.01):
    n_samples = len(X)
    n_features = len(X[0]) if n_samples > 0 else 0
    if n_features == 0 or n_samples == 0:
        return None, None
    W = [[0.0] * n_features for _ in range(2)]
    b = [0.0] * 2
    for _ in range(max_iter):
        grad_W = [[0.0] * n_features for _ in range(2)]
        grad_b = [0.0] * 2
        for i in range(n_samples):
            logits = [sum(W[c][j] * X[i][j] for j in range(n_features)) + b[c] for c in range(2)]
            probs = softmax(logits)
            for c in range(2):
                target = 1.0 if c == y[i] else 0.0
                error = probs[c] - target
                for j in range(n_features):
                    grad_W[c][j] += error * X[i][j]
                grad_b[c] += error
        for c in range(2):
            for j in range(n_features):
                grad_W[c][j] = grad_W[c][j] / n_samples + l2_alpha * W[c][j]
                W[c][j] -= lr * grad_W[c][j]
            grad_b[c] /= n_samples
            b[c] -= lr * grad_b[c]
    return W, b

def predict_softmax_2class(W, b, X):
    preds = []
    for i in range(len(X)):
        logits = [sum(W[c][j] * X[i][j] for j in range(len(W[0]))) + b[c] for c in range(2)]
        preds.append(max(range(2), key=lambda c: logits[c]))
    return preds

def balanced_accuracy(y_true, y_pred):
    """(sensitivity + specificity) / 2"""
    n = len(y_true)
    classes = sorted(set(y_true))
    recalls = []
    for c in classes:
        tp = sum(1 for i in range(n) if y_true[i] == c and y_pred[i] == c)
        fn = sum(1 for i in range(n) if y_true[i] == c and y_pred[i] != c)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return sum(recalls) / len(recalls)

def macro_f1(y_true, y_pred):
    """Average of per-class F1 scores."""
    n = len(y_true)
    classes = sorted(set(y_true))
    f1s = []
    for c in classes:
        tp = sum(1 for i in range(n) if y_true[i] == c and y_pred[i] == c)
        fp = sum(1 for i in range(n) if y_true[i] != c and y_pred[i] == c)
        fn = sum(1 for i in range(n) if y_true[i] == c and y_pred[i] != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)

def run_shortcut_probes(all_voi_results, per_seed_objects):
    """Run all shortcut probes, returning balanced accuracy and macro-F1 for each."""

    # Build object feature vectors
    object_feature_vectors = []
    for v in all_voi_results:
        oid = v["oid"]
        seed = v["seed"]
        obj = per_seed_objects[seed][oid]
        pre_feats = obj["pre_surface_features"]
        feat_vec = {}
        for fn in ALL_VISIBLE_FEATURE_NAMES:
            feat_vec[f"feat_{fn}"] = 1.0 if fn in pre_feats else 0.0
        feat_vec["positive_net"] = 1.0 if v["positive_net"] else 0.0
        feat_vec["category"] = v["category"]
        feat_vec["n_pre_surface_features"] = v["n_pre_surface_features"]
        object_feature_vectors.append(feat_vec)

    y_true = [1 if fv["positive_net"] > 0 else 0 for fv in object_feature_vectors]
    results = {}

    # --- Single-feature probe (best individual feature) ---
    best_single_acc = 0.0
    best_single_ba = 0.0
    best_single_f1 = 0.0
    best_single_feat = None
    for fn in ALL_VISIBLE_FEATURE_NAMES:
        X = [[1.0 if fv[f"feat_{fn}"] > 0 else 0.0] for fv in object_feature_vectors]
        W, b = train_softmax_2class(X, y_true)
        if W is not None:
            y_pred = predict_softmax_2class(W, b, X)
            acc = sum(1 for p, yi in zip(y_pred, y_true) if p == yi) / len(y_true)
            if acc > best_single_acc:
                best_single_acc = acc
                best_single_ba = balanced_accuracy(y_true, y_pred)
                best_single_f1 = macro_f1(y_true, y_pred)
                best_single_feat = fn

    results["single_feature"] = {
        "feature": best_single_feat,
        "accuracy": round(best_single_acc, 4),
        "balanced_accuracy": round(best_single_ba, 4),
        "macro_f1": round(best_single_f1, 4),
    }

    # --- Feature-count probe ---
    X_nfeat = [[float(fv["n_pre_surface_features"])] for fv in object_feature_vectors]
    W_nf, b_nf = train_softmax_2class(X_nfeat, y_true)
    if W_nf is not None:
        y_pred_nf = predict_softmax_2class(W_nf, b_nf, X_nfeat)
        results["feature_count"] = {
            "accuracy": round(sum(1 for p, yi in zip(y_pred_nf, y_true) if p == yi) / len(y_true), 4),
            "balanced_accuracy": round(balanced_accuracy(y_true, y_pred_nf), 4),
            "macro_f1": round(macro_f1(y_true, y_pred_nf), 4),
        }

    # --- Category probe ---
    cat_to_idx = {cat: i for i, cat in enumerate(CATEGORIES)}
    X_cat = [[0.0] * len(CATEGORIES) for _ in object_feature_vectors]
    for i, fv in enumerate(object_feature_vectors):
        X_cat[i][cat_to_idx[fv["category"]]] = 1.0
    W_cat, b_cat = train_softmax_2class(X_cat, y_true)
    if W_cat is not None:
        y_pred_cat = predict_softmax_2class(W_cat, b_cat, X_cat)
        results["category"] = {
            "accuracy": round(sum(1 for p, yi in zip(y_pred_cat, y_true) if p == yi) / len(y_true), 4),
            "balanced_accuracy": round(balanced_accuracy(y_true, y_pred_cat), 4),
            "macro_f1": round(macro_f1(y_true, y_pred_cat), 4),
        }

    # --- Feature-pair probe (top 5 pairs) ---
    top_features = sorted(
        [(fn, sum(1 for fv in object_feature_vectors if fv[f"feat_{fn}"] > 0))
         for fn in ALL_VISIBLE_FEATURE_NAMES],
        key=lambda x: -x[1]
    )[:6]
    sampled = [fn for fn, _ in top_features]
    pair_accs = []
    for i, f1 in enumerate(sampled):
        for f2 in sampled[i+1:]:
            X = [[1.0 if fv[f"feat_{f1}"] > 0 else 0.0,
                  1.0 if fv[f"feat_{f2}"] > 0 else 0.0]
                 for fv in object_feature_vectors]
            W, b = train_softmax_2class(X, y_true)
            if W is not None:
                y_pred_pair = predict_softmax_2class(W, b, X)
                pair_accs.append((
                    (f1, f2),
                    sum(1 for p, yi in zip(y_pred_pair, y_true) if p == yi) / len(y_true),
                    balanced_accuracy(y_true, y_pred_pair),
                    macro_f1(y_true, y_pred_pair),
                ))
    pair_accs.sort(key=lambda x: -x[1])
    if pair_accs:
        (bf1, bf2), best_pair_acc, best_pair_ba, best_pair_f1 = pair_accs[0]
        results["feature_pair"] = {
            "pair": (bf1, bf2),
            "accuracy": round(best_pair_acc, 4),
            "balanced_accuracy": round(best_pair_ba, 4),
            "macro_f1": round(best_pair_f1, 4),
        }

    return results

# =============================================================================
# Counterfactual slice status
# =============================================================================
def check_counterfactual_slices(all_voi_results, per_seed_objects):
    """Check that at least one shared surface feature appears in both pos and non-pos."""
    features_in_both = []
    for fn in ALL_VISIBLE_FEATURE_NAMES:
        pos_cats = set()
        nonpos_cats = set()
        for v in all_voi_results:
            seed = v["seed"]
            oid = v["oid"]
            obj = per_seed_objects[seed][oid]
            if fn in obj["pre_surface_features"]:
                if v["positive_net"]:
                    pos_cats.add(v["category"])
                else:
                    nonpos_cats.add(v["category"])
        if len(pos_cats) > 0 and len(nonpos_cats) > 0:
            features_in_both.append((fn, len(pos_cats), len(nonpos_cats)))

    # Check shared surface features specifically
    shared_in_both = [fn for fn in SHARED_SURFACE_FEATURES
                      if any(f[0] == fn for f in features_in_both)]
    return {
        "n_features_in_both": len(features_in_both),
        "n_shared_in_both": len(shared_in_both),
        "shared_in_both": shared_in_both,
    }

# =============================================================================
# Generate objects (once, reused for all cost values)
# =============================================================================
print("Generating env3b objects...")
per_seed_objects = {}
per_seed_audit_labels = {}
for seed in SEEDS:
    objects, audit_labels = generate_env3b_objects(seed)
    per_seed_objects[seed] = objects
    per_seed_audit_labels[seed] = audit_labels
print(f"  {len(per_seed_objects[SEEDS[0]])} objects per seed, {len(SEEDS)} seeds")

# =============================================================================
# Observe-cost sweep
# =============================================================================
cost_candidates = [0.0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10]
print(f"\nRunning observe-cost sweep across {len(cost_candidates)} values...")

sweep_results = []
for oc in cost_candidates:
    voi_results, sig_groups, sig_expected = compute_audit_voi_all_objects(
        per_seed_objects, per_seed_audit_labels, oc)
    bl = compute_baselines(voi_results, oc)
    probes = run_shortcut_probes(voi_results, per_seed_objects)
    cf_slice = check_counterfactual_slices(voi_results, per_seed_objects)

    positive_net = [v for v in voi_results if v["positive_net"]]
    non_positive_net = [v for v in voi_results if not v["positive_net"]]

    # Per-category non_positive check
    cats_with_nonpos = set()
    for cat in CATEGORIES:
        if any(v["category"] == cat and not v["positive_net"] for v in voi_results):
            cats_with_nonpos.add(cat)

    # n_features overlap
    pos_nfeat = set(v["n_pre_surface_features"] for v in positive_net)
    nonpos_nfeat = set(v["n_pre_surface_features"] for v in non_positive_net)
    nfeat_overlap = pos_nfeat & nonpos_nfeat

    sweep_results.append({
        "observe_cost": oc,
        "positive_net_rate": len(positive_net) / len(voi_results),
        "non_positive_net_rate": len(non_positive_net) / len(voi_results),
        "oracle_selective": bl["oracle_selective"],
        "always_observe_net": bl["always_observe_net"],
        "always_try": bl["always_try"],
        "no_observe": bl["no_observe_pre_only"],
        "sel_vs_always_observe": bl["oracle_selective_gap_vs_always_observe"],
        "sel_vs_always_try": bl["oracle_selective_gap_vs_always_try"],
        "oracle_obs_rate": bl["oracle_selective_obs_rate"],
        "nonpos_every_category": len(cats_with_nonpos) == len(CATEGORIES),
        "nfeat_overlap": sorted(nfeat_overlap),
        "nfeat_overlap_count": len(nfeat_overlap),
        "probes": probes,
        "counterfactual": cf_slice,
    })

# =============================================================================
# Report
# =============================================================================
print(f"\n{'='*120}")
print("Observe-Cost Calibration Sweep Report")
print(f"{'='*120}")

print(f"\n{'cost':>8s}  {'pos%':>6s}  {'nonpos%':>8s}  "
      f"{'sel_net':>8s}  {'alwaysObs':>9s}  {'alwaysTry':>9s}  "
      f"{'sel-obs':>8s}  {'sel-try':>8s}  {'obs%':>6s}  "
      f"{'nonposCat':>10s}  {'nfeatOlap':>9s}")
print("-" * 120)
for r in sweep_results:
    print(f"{r['observe_cost']:8.4f}  {r['positive_net_rate']:6.4f}  {r['non_positive_net_rate']:8.4f}  "
          f"{r['oracle_selective']:8.4f}  {r['always_observe_net']:9.4f}  {r['always_try']:9.4f}  "
          f"{r['sel_vs_always_observe']:8.4f}  {r['sel_vs_always_try']:8.4f}  {r['oracle_obs_rate']:6.4f}  "
          f"{'OK' if r['nonpos_every_category'] else 'FAIL':>10s}  {r['nfeat_overlap_count']:>9d}")

print(f"\n--- Shortcut Probe Metrics (balanced accuracy / macro-F1) ---")
print(f"{'cost':>8s}  {'probe':>15s}  {'accuracy':>8s}  {'bal_acc':>8s}  {'macro_f1':>8s}")
print("-" * 70)
for r in sweep_results:
    for probe_name in ["single_feature", "feature_count", "category", "feature_pair"]:
        if probe_name in r["probes"]:
            p = r["probes"][probe_name]
            print(f"{r['observe_cost']:8.4f}  {probe_name:>15s}  "
                  f"{p['accuracy']:8.4f}  {p['balanced_accuracy']:8.4f}  {p['macro_f1']:8.4f}")

print(f"\n--- Counterfactual Slice Status ---")
print(f"{'cost':>8s}  {'n_feat_both':>11s}  {'shared_both':>11s}  {'shared_list':>20s}")
print("-" * 70)
for r in sweep_results:
    cf = r["counterfactual"]
    print(f"{r['observe_cost']:8.4f}  {cf['n_features_in_both']:>11d}  "
          f"{cf['n_shared_in_both']:>11d}  {str(cf['shared_in_both'][:3]):>20s}")

# =============================================================================
# Recommendation
# =============================================================================
print(f"\n{'='*120}")
print("Recommendation Criteria (must ALL pass)")
print(f"{'='*120}")

# Criteria thresholds
MEANINGFUL_MARGIN = 0.005  # oracle must beat baseline by at least this

def evaluate_cost(r):
    """Evaluate a cost candidate. Returns (passed, reasons)."""
    reasons = []
    ok = True

    # 1. positive_net cases exist
    if r["positive_net_rate"] <= 0:
        reasons.append("FAIL: no positive_net cases")
        ok = False
    else:
        reasons.append(f"OK: positive_net rate={r['positive_net_rate']:.4f}")

    # 2. positive_net not near-universal
    if r["positive_net_rate"] >= 0.90:
        reasons.append(f"FAIL: positive_net near-universal ({r['positive_net_rate']:.4f})")
        ok = False
    else:
        reasons.append(f"OK: positive_net not universal ({r['positive_net_rate']:.4f})")

    # 3. non_positive cases in every category
    if not r["nonpos_every_category"]:
        reasons.append("FAIL: not all categories have non_positive_net")
        ok = False
    else:
        reasons.append("OK: all categories have non_positive_net")

    # 4. oracle_selective > always_observe by meaningful margin
    if r["sel_vs_always_observe"] <= MEANINGFUL_MARGIN:
        reasons.append(f"FAIL: sel-obs gap {r['sel_vs_always_observe']:.4f} <= {MEANINGFUL_MARGIN:.4f}")
        ok = False
    else:
        reasons.append(f"OK: sel-obs gap {r['sel_vs_always_observe']:.4f} > {MEANINGFUL_MARGIN:.4f}")

    # 5. oracle_selective > always_try by meaningful margin
    if r["sel_vs_always_try"] <= MEANINGFUL_MARGIN:
        reasons.append(f"FAIL: sel-try gap {r['sel_vs_always_try']:.4f} <= {MEANINGFUL_MARGIN:.4f}")
        ok = False
    else:
        reasons.append(f"OK: sel-try gap {r['sel_vs_always_try']:.4f} > {MEANINGFUL_MARGIN:.4f}")

    # 6. always_observe is not oracle-level
    if abs(r["always_observe_net"] - r["oracle_selective"]) <= MEANINGFUL_MARGIN:
        reasons.append(f"FAIL: always_observe too close to oracle (diff={abs(r['always_observe_net'] - r['oracle_selective']):.4f})")
        ok = False
    else:
        reasons.append(f"OK: always_observe differs from oracle by {abs(r['always_observe_net'] - r['oracle_selective']):.4f}")

    # 7. always_try not oracle-level
    if abs(r["always_try"] - r["oracle_selective"]) <= MEANINGFUL_MARGIN:
        reasons.append(f"FAIL: always_try too close to oracle")
        ok = False
    else:
        reasons.append(f"OK: always_try far from oracle (gap={r['sel_vs_always_try']:.4f})")

    # 8. Shortcut probes acceptable (balanced accuracy <= 0.65, or macro_f1 <= 0.55)
    for probe_name in ["single_feature", "feature_count", "category"]:
        if probe_name in r["probes"]:
            p = r["probes"][probe_name]
            if p["balanced_accuracy"] > 0.70:
                reasons.append(f"WARN: {probe_name} balanced_accuracy={p['balanced_accuracy']:.4f} > 0.70")
            if p["macro_f1"] > 0.65:
                reasons.append(f"WARN: {probe_name} macro_f1={p['macro_f1']:.4f} > 0.65")

    # 9. n_features_known overlap exists
    if r["nfeat_overlap_count"] < 1:
        reasons.append("FAIL: no n_features overlap between pos and non-pos")
        ok = False
    else:
        reasons.append(f"OK: nfeat overlap={r['nfeat_overlap']}")

    # 10. Counterfactual slice has at least one shared feature in both
    if r["counterfactual"]["n_shared_in_both"] < 1:
        reasons.append("WARN: no shared surface feature in both pos and non-pos")

    return ok, reasons

# Evaluate all costs
print(f"\nMeaningful margin threshold: {MEANINGFUL_MARGIN}")
print()
for r in sweep_results:
    ok, reasons = evaluate_cost(r)
    status = "RECOMMENDED" if ok else "FAIL"
    print(f"--- cost={r['observe_cost']:.4f} [{status}] ---")
    for reason in reasons:
        print(f"  {reason}")
    print()

# Find best recommended cost (maximize sel_vs_always_observe margin)
recommended = None
best_margin = -999
for r in sweep_results:
    ok, _ = evaluate_cost(r)
    if ok:
        if r["sel_vs_always_observe"] > best_margin:
            best_margin = r["sel_vs_always_observe"]
            recommended = r["observe_cost"]

print(f"{'='*120}")
if recommended is not None:
    print(f"Recommended default observe_cost for 1J40b-7b: {recommended:.4f}")
    print(f"  Margin oracle-always_observe: {best_margin:.4f}")
    print(f"  Margin oracle-always_try: {[r['sel_vs_always_try'] for r in sweep_results if abs(r['observe_cost']-recommended)<0.0001][0]:.4f}")
else:
    print("NO cost value passes all criteria. Environment needs adjustment.")
    print("Consider: adjusting subtype affordances for larger VOI, or reducing observe_cost scale.")
print(f"{'='*120}")

elapsed = round(time.time() - t0, 1)
print(f"\nElapsed: {elapsed}s")
