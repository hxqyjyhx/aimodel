"""
Block 1J40b-7f -- Oracle-Belief Upper-Bound Control.

Measures whether the corrected 7e counterfactual distribution contains enough
selective-observe headroom for any policy to exploit.

Core question: Can an oracle-belief policy that knows empirical observe
usefulness by legal pre_obs/signature outperform always_observe on held-out
Test A/B?

oracle-belief is AUDIT-SIDE ONLY, never fed to learner, never written into
model-visible records.
"""

import os, sys, json, math, time, random
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
A3_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5a3_tool_material_transfer")
sys.path.insert(0, A3_DIR)
sys.path.insert(0, CURRENT_DIR)

import config

t0 = time.time()

# =============================================================================
# Config safety (same as 7e)
# =============================================================================
C6_LABEL = "C6_observe_try_counterfactual_v0"
existing_labels = [c["label"] for c in config.CUE_CONDITIONS]
if C6_LABEL not in existing_labels:
    config.CUE_CONDITIONS.append({
        "label": C6_LABEL, "p_target": 0.60, "p_other": 0.40,
        "absent": False, "subtype": True,
        "subtype_config": "observe_try_counterfactual_v0",
        "mixed_source": True, "observe_depth_schedules": True,
    })

# =============================================================================
# Constants (from env3b / 7e)
# =============================================================================
SEEDS = [101, 103, 107, 109, 113]
RIDGE_ALPHA = 1.0
OBSERVE_COSTS = [0.01, 0.03, 0.05, 0.07]

ALL_TRY_AFFORDANCES = [
    "burn_as_fuel", "craft_plank", "eat",
    "mine_by_hand", "mine_with_pickaxe", "use_as_tool",
]
ALL_ACTION_KEYS = ["observe"] + [f"try_{a}" for a in ALL_TRY_AFFORDANCES]

SUCCESS_REWARD = 0.5
FAILURE_PENALTY = 0.1

CATEGORIES = ["wood_log", "stone_block", "apple", "wooden_pickaxe"]
COVERAGE_FRACTIONS = [0.25, 0.50]

# =============================================================================
# Feature Definitions (from env3b)
# =============================================================================
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
    "spotted", "weathered", "coarse_texture",
    "discolored", "has_veins", "pitted",
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
# Instance Subtype Definitions (from 7a/7e)
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
        "post_observe_revealed": ["has_crystal_flecks", "has_granular_surface", "block_like",
                                  "cold_to_touch"],
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
        "post_observe_revealed": ["has_grip_area", "has_shaft_shape", "elongated_with_handle",
                                  "movable"],
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
# Object Generation (from 7a/7e)
# =============================================================================
print("[1/8] Generating env3b objects...")

def generate_env3b_objects(seed):
    rng = random.Random(seed * 100 + 1)
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

            hidden = {}
            for f in ALL_VISIBLE_FEATURE_NAMES:
                if f not in pre_surface and f not in post_revealed:
                    hidden[f] = True

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
                    "hidden_features": dict(hidden),
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


per_seed_objects = {}
per_seed_audit_labels = {}
for seed in SEEDS:
    objects, audit_labels = generate_env3b_objects(seed)
    per_seed_objects[seed] = objects
    per_seed_audit_labels[seed] = audit_labels

print(f"  Objects per seed: {len(per_seed_objects[SEEDS[0]])}")

# =============================================================================
# Global Feature Name Collection (fixed dimension, same as 7e)
# =============================================================================
def collect_all_post_observe_features(per_seed_objects):
    all_feats = set()
    for seed in SEEDS:
        for oid, obj in per_seed_objects[seed].items():
            for f in obj["post_observe_features"]:
                all_feats.add(f)
    return sorted(all_feats)

POST_OBSERVE_FEATURE_NAMES = collect_all_post_observe_features(per_seed_objects)
ALL_MODEL_FEATURE_KEYS = sorted(set(
    [f"feat_{f}" for f in POST_OBSERVE_FEATURE_NAMES] +
    [f"state_{s}" for s in STATE_FEATURE_NAMES] +
    ["prior_observe_count"]
))
N_FEATURES = len(ALL_MODEL_FEATURE_KEYS)
print(f"  Model feature keys: {N_FEATURES}")

# =============================================================================
# Feature Vector / Signature Helpers
# =============================================================================
def try_net_value(success_or_failure):
    if success_or_failure is True:
        return SUCCESS_REWARD - 0.05
    elif success_or_failure is False:
        return -FAILURE_PENALTY - 0.05
    return 0.0

def get_pre_surface_signature(pre_surface_features):
    return tuple(sorted(pre_surface_features.keys()))

def build_feature_vector(known_features, known_states, prior_obs_count):
    """Build a fixed-dimension feature vector dict (same as 7e)."""
    feat = {}
    for fname in POST_OBSERVE_FEATURE_NAMES:
        feat[f"feat_{fname}"] = 1.0 if fname in known_features else 0.0
    for sname in STATE_FEATURE_NAMES:
        feat[f"state_{sname}"] = 1.0 if known_states.get(sname, False) else 0.0
    feat["prior_observe_count"] = float(prior_obs_count)
    return feat

def extract_feature_vector(situation_features):
    """Convert feature dict to fixed-length float list (same as 7e)."""
    return [float(situation_features.get(k, 0.0)) for k in ALL_MODEL_FEATURE_KEYS]

# =============================================================================
# Signature Analysis
# =============================================================================
sig_map = defaultdict(list)
for seed in SEEDS:
    for oid, obj in per_seed_objects[seed].items():
        sig = get_pre_surface_signature(obj["pre_surface_features"])
        sig_map[sig].append((seed, oid))

SIGNATURE_LIST = sorted(sig_map.keys(), key=lambda s: (-len(sig_map[s]), s))
print(f"  Unique signatures: {len(SIGNATURE_LIST)}")

shared_sigs = []
unique_sigs = []
for sig in SIGNATURE_LIST:
    members = sig_map[sig]
    cats_in_sig = set()
    for s2, oid2 in members:
        lbl = per_seed_audit_labels[s2][oid2]
        cats_in_sig.add(lbl["hidden_category"])
    if len(cats_in_sig) >= 2:
        shared_sigs.append(sig)
    else:
        unique_sigs.append(sig)

# =============================================================================
# Model Classes (minimal, from 7e)
# =============================================================================
class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = 0.0
        self._fitted = False

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        X_aug = [row + [1.0] for row in X]
        n_aug = n_features + 1
        XtX = [[0.0] * n_aug for _ in range(n_aug)]
        for i in range(n_aug):
            for j in range(n_aug):
                s = 0.0
                for k in range(n_samples):
                    s += X_aug[k][i] * X_aug[k][j]
                XtX[i][j] = s
        for i in range(n_features):
            XtX[i][i] += self.alpha
        Xty = [0.0] * n_aug
        for i in range(n_aug):
            s = 0.0
            for k in range(n_samples):
                s += X_aug[k][i] * y[k]
            Xty[i] = s
        w_aug = _solve_linear_system(XtX, Xty)
        if w_aug is None:
            self.coef_ = [0.0] * n_features
            self.intercept_ = sum(y) / len(y) if y else 0.0
        else:
            self.coef_ = w_aug[:n_features]
            self.intercept_ = w_aug[n_features]
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return [sum(xi * wi for xi, wi in zip(row, self.coef_)) + self.intercept_
                for row in X]

def _solve_linear_system(A, b):
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[max_row][col]) < 1e-12:
            return None
        M[col], M[max_row] = M[max_row], M[col]
        pivot = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pivot
        for row in range(n):
            if row != col:
                factor = M[row][col]
                for j in range(col, n + 1):
                    M[row][j] -= factor * M[col][j]
    return [M[i][n] for i in range(n)]

class FeatureNormalizer:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X):
        n_feat = len(X[0]) if X else 0
        self.means = [0.0] * n_feat
        self.stds = [1.0] * n_feat
        if not X:
            return self
        n = len(X)
        for j in range(n_feat):
            self.means[j] = sum(row[j] for row in X) / n
            var = sum((row[j] - self.means[j]) ** 2 for row in X) / n
            self.stds[j] = math.sqrt(var) if var > 1e-12 else 1.0
        return self

    def transform(self, X):
        if self.means is None:
            return X
        return [[(row[j] - self.means[j]) / self.stds[j]
                 for j in range(len(row))]
                for row in X]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class PerActionValueEstimator:
    def __init__(self, alpha=RIDGE_ALPHA):
        self.alpha = alpha
        self.models = {}
        self.normalizers = {}

    def fit(self, X_train, y_train, action_keys_train):
        grouped_X = defaultdict(list)
        grouped_y = defaultdict(list)
        for xv, yv, ak in zip(X_train, y_train, action_keys_train):
            grouped_X[ak].append(xv)
            grouped_y[ak].append(yv)

        for ak in ALL_ACTION_KEYS:
            gx = grouped_X.get(ak, [])
            gy = grouped_y.get(ak, [])
            if len(gx) < 3:
                self.models[ak] = RidgeRegression(alpha=self.alpha)
                nf = len(gx[0]) if gx else 0
                self.models[ak].coef_ = [0.0] * nf
                self.models[ak].intercept_ = sum(gy) / len(gy) if gy else 0.0
                self.models[ak]._fitted = True
                self.normalizers[ak] = None
                continue
            normalizer = FeatureNormalizer().fit(gx)
            X_norm = normalizer.transform(gx)
            model = RidgeRegression(alpha=self.alpha).fit(X_norm, gy)
            self.models[ak] = model
            self.normalizers[ak] = normalizer
        return self

    def predict_single(self, x_vec, action_key):
        if action_key not in self.models:
            return 0.0
        model = self.models[action_key]
        normalizer = self.normalizers.get(action_key)
        if normalizer is not None and normalizer.means is not None:
            x_norm = normalizer.transform([x_vec])[0]
        else:
            x_norm = x_vec
        return model.predict([x_norm])[0]

    def predict_all_actions(self, x_vec):
        return {ak: self.predict_single(x_vec, ak) for ak in ALL_ACTION_KEYS}

# =============================================================================
# Stress Test Split Definitions (from 7c/7e)
# =============================================================================
print("[2/8] Defining stress test splits...")

TEST_A_VARIANTS = []
for i in range(min(len(shared_sigs), len(unique_sigs))):
    held_shared = shared_sigs[i]
    held_unique = unique_sigs[i]
    held_sigs = [held_shared, held_unique]
    train_sigs = [s for s in SIGNATURE_LIST if s not in held_sigs]
    train_oids = set()
    for sig in train_sigs:
        for seed, oid in sig_map[sig]:
            train_oids.add((seed, oid))
    test_oids = set()
    for sig in held_sigs:
        for seed, oid in sig_map[sig]:
            test_oids.add((seed, oid))
    test_pos = sum(1 for seed, oid in test_oids
                   if per_seed_audit_labels[seed][oid]["hidden_audit_rationale_class"] == "positive_change")
    test_nonpos = sum(1 for seed, oid in test_oids
                      if per_seed_audit_labels[seed][oid]["hidden_audit_rationale_class"] != "positive_change")
    TEST_A_VARIANTS.append({
        "id": f"A{i+1}", "held_sigs": held_sigs, "train_sigs": train_sigs,
        "train_oids": train_oids, "test_oids": test_oids,
        "n_train": len(train_oids), "n_test": len(test_oids),
        "test_pos": test_pos, "test_nonpos": test_nonpos,
    })

# Test B
apple_n_sig = None
woodlog_n_sig = None
for sig in unique_sigs:
    cats_in = set()
    for seed, oid in sig_map[sig]:
        lbl = per_seed_audit_labels[seed][oid]
        cats_in.add(lbl["hidden_category"])
    if cats_in == {"apple"}:
        apple_n_sig = sig
    if cats_in == {"wood_log"}:
        woodlog_n_sig = sig

ambient_b_only_sig = None
ambient_a_only_sig = None
for sig in shared_sigs:
    some_seed, some_oid = sig_map[sig][0]
    ag = per_seed_audit_labels[some_seed][some_oid]["ambient_group"]
    is_ambient_only = len(sig) == 4
    if ag == "B" and is_ambient_only:
        ambient_b_only_sig = sig
    if ag == "A" and is_ambient_only:
        ambient_a_only_sig = sig

b1_test_sigs = [s for s in [apple_n_sig, ambient_b_only_sig] if s is not None]
b1_train_sigs = [s for s in SIGNATURE_LIST if s not in b1_test_sigs] if b1_test_sigs else []
b2_test_sigs = [s for s in [woodlog_n_sig, ambient_a_only_sig] if s is not None]
b2_train_sigs = [s for s in SIGNATURE_LIST if s not in b2_test_sigs] if b2_test_sigs else []

TEST_B_VARIANTS = []
for bid, test_sigs_b, train_sigs_b in [
    ("B1", b1_test_sigs, b1_train_sigs),
    ("B2", b2_test_sigs, b2_train_sigs),
]:
    if not test_sigs_b or not train_sigs_b:
        continue
    train_oids_b = set()
    for sig in train_sigs_b:
        for seed, oid in sig_map[sig]:
            train_oids_b.add((seed, oid))
    test_oids_b = set()
    for sig in test_sigs_b:
        for seed, oid in sig_map[sig]:
            test_oids_b.add((seed, oid))
    test_pos_b = sum(1 for seed, oid in test_oids_b
                     if per_seed_audit_labels[seed][oid]["hidden_audit_rationale_class"] == "positive_change")
    test_nonpos_b = sum(1 for seed, oid in test_oids_b
                        if per_seed_audit_labels[seed][oid]["hidden_audit_rationale_class"] != "positive_change")
    TEST_B_VARIANTS.append({
        "id": bid, "train_oids": train_oids_b, "test_oids": test_oids_b,
        "n_train": len(train_oids_b), "n_test": len(test_oids_b),
        "test_pos": test_pos_b, "test_nonpos": test_nonpos_b,
    })

# Test C
TEST_C_VARIANTS = []
for heldout_seed in SEEDS:
    train_seeds_c = [s for s in SEEDS if s != heldout_seed]
    train_oids_c = set()
    test_oids_c = set()
    for s2 in train_seeds_c:
        for oid2 in per_seed_objects[s2]:
            train_oids_c.add((s2, oid2))
    for oid2 in per_seed_objects[heldout_seed]:
        test_oids_c.add((heldout_seed, oid2))
    test_pos_c = sum(1 for seed, oid in test_oids_c
                     if per_seed_audit_labels[seed][oid]["hidden_audit_rationale_class"] == "positive_change")
    test_nonpos_c = sum(1 for seed, oid in test_oids_c
                        if per_seed_audit_labels[seed][oid]["hidden_audit_rationale_class"] != "positive_change")
    TEST_C_VARIANTS.append({
        "id": f"C_seed{heldout_seed}", "heldout_seed": heldout_seed,
        "train_oids": train_oids_c, "test_oids": test_oids_c,
        "n_train": len(train_oids_c), "n_test": len(test_oids_c),
        "test_pos": test_pos_c, "test_nonpos": test_nonpos_c,
    })

print(f"  Test A: {len(TEST_A_VARIANTS)} variants, Test B: {len(TEST_B_VARIANTS)} variants, Test C: {len(TEST_C_VARIANTS)} variants")

# =============================================================================
# Counterfactual Intervention Functions (from 7e)
# =============================================================================
AMBIENT_GROUP_CATEGORIES = {
    "A": ["wood_log", "stone_block"],
    "B": ["apple", "wooden_pickaxe"],
}

def get_flat_affordance(all_success=True):
    outcome = "success" if all_success else "fail"
    return {a: outcome for a in ALL_TRY_AFFORDANCES}

def get_lookalike_affordance(seed, oid, rng):
    lbl = per_seed_audit_labels[seed][oid]
    original_cat = lbl["hidden_category"]
    ambient_group = lbl["ambient_group"]
    other_cats = [c for c in AMBIENT_GROUP_CATEGORIES[ambient_group] if c != original_cat]
    if other_cats:
        other_cat = other_cats[0]
        return dict(INSTANCE_SUBTYPE_DEFS[other_cat]["P"]["affordance_profile"]), other_cat
    return get_flat_affordance(True), None

def build_counterfactual_affordance_overrides(train_oids, coverage_fraction):
    rng = random.Random(42)
    train_by_sig = defaultdict(list)
    for seed, oid in train_oids:
        sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
        train_by_sig[sig].append((seed, oid))

    overrides = {}
    intervention_labels = {}
    lookalike_targets = {}

    for sig, instances in sorted(train_by_sig.items()):
        n = len(instances)
        n_flip = max(1, int(n * coverage_fraction))
        flipped = rng.sample(instances, n_flip)

        some_lbl = per_seed_audit_labels[instances[0][0]][instances[0][1]]
        is_shared = some_lbl["hidden_audit_rationale_class"] == "positive_change"

        for seed, oid in flipped:
            if is_shared:
                flat_val = "success" if rng.random() < 0.5 else "fail"
                new_aff = get_flat_affordance(flat_val == "success")
            else:
                new_aff, target_cat = get_lookalike_affordance(seed, oid, rng)
                lookalike_targets[(seed, oid)] = target_cat

            overrides[(seed, oid)] = new_aff
            intervention_labels[(seed, oid)] = (
                "decision_irrelevant_flat" if is_shared
                else "lookalike_deceptive_exception")

    return overrides, intervention_labels, lookalike_targets

def build_lookalike_post_observe_overrides(lookalike_targets):
    overrides = {}
    for (seed, oid), target_cat in lookalike_targets.items():
        obj = per_seed_objects[seed][oid]
        corrected_post = dict(obj["pre_surface_features"])
        for f in CATEGORY_POST_OBSERVE_FEATURES.get(target_cat, []):
            corrected_post[f] = True
        overrides[(seed, oid)] = corrected_post
    return overrides

# =============================================================================
# Core: True/Oracle VOI (from 7e)
# =============================================================================
print("[3/8] Computing true/oracle VOI infrastructure...")

def compute_true_oracle_voi(oid_set, observe_cost, affordance_overrides=None):
    sig_instances = defaultdict(list)
    for seed, oid in oid_set:
        obj = per_seed_objects[seed][oid]
        sig = get_pre_surface_signature(obj["pre_surface_features"])
        if affordance_overrides and (seed, oid) in affordance_overrides:
            aff = affordance_overrides[(seed, oid)]
        else:
            aff = per_seed_audit_labels[seed][oid]["affordance_profile"]
        sig_instances[sig].append((seed, oid, aff))

    sig_expected = {}
    for sig, instances in sig_instances.items():
        n = len(instances)
        action_sums = {a: 0.0 for a in ALL_TRY_AFFORDANCES}
        for seed, oid, aff in instances:
            for a in ALL_TRY_AFFORDANCES:
                outcome = aff.get(a, "fail")
                action_sums[a] += try_net_value(outcome == "success")
        action_exp = {a: action_sums[a] / n for a in ALL_TRY_AFFORDANCES}
        best_action = max(action_exp, key=action_exp.get)
        sig_expected[sig] = {
            "best_action": best_action,
            "best_expected_value": action_exp[best_action],
            "n_instances": n,
        }

    result = {}
    for seed, oid in oid_set:
        obj = per_seed_objects[seed][oid]
        sig = get_pre_surface_signature(obj["pre_surface_features"])
        true_pre_best = sig_expected[sig]["best_expected_value"]

        if affordance_overrides and (seed, oid) in affordance_overrides:
            aff = affordance_overrides[(seed, oid)]
        else:
            aff = per_seed_audit_labels[seed][oid]["affordance_profile"]

        true_post_best = max(
            try_net_value(aff.get(a, "fail") == "success")
            for a in ALL_TRY_AFFORDANCES)

        true_voi = true_post_best - true_pre_best - observe_cost
        result[(seed, oid)] = {
            "true_pre_best": true_pre_best,
            "true_post_best": true_post_best,
            "true_voi": true_voi,
            "true_optimal": true_voi > 0,
        }

    return result

# =============================================================================
# Core: Oracle-Belief Computation (NEW — the heart of 7f)
# =============================================================================
print("[4/8] Computing oracle-belief lookup tables...")

def compute_oracle_belief(train_oids, observe_cost, affordance_overrides=None):
    """Compute empirical P(positive_net | pre_surface_signature) and
    E_observe_net per signature from training data ONLY.

    This is AUDIT-SIDE ONLY — never fed to learner, never written into
    model-visible records.

    Returns dict: sig -> {
        n, n_positive, p_positive, e_observe_net, optimal_decision
    }
    and global aggregates for fallback.
    """
    true_voi = compute_true_oracle_voi(train_oids, observe_cost, affordance_overrides)

    sig_data = defaultdict(lambda: {"vois": [], "n_positive": 0, "n_total": 0})
    for seed, oid in train_oids:
        sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
        voi = true_voi[(seed, oid)]["true_voi"]
        is_pos = true_voi[(seed, oid)]["true_optimal"]
        sig_data[sig]["vois"].append(voi)
        sig_data[sig]["n_total"] += 1
        if is_pos:
            sig_data[sig]["n_positive"] += 1

    belief = {}
    all_vois = []
    all_is_pos = []
    for sig, data in sig_data.items():
        n = data["n_total"]
        n_pos = data["n_positive"]
        p_pos = n_pos / n if n > 0 else 0.0
        e_voi = sum(data["vois"]) / n if n > 0 else 0.0
        optimal = p_pos > 0.5  # majority vote
        belief[sig] = {
            "n": n, "n_positive": n_pos, "p_positive": p_pos,
            "e_observe_net": e_voi, "optimal_decision": optimal,
        }
        all_vois.extend(data["vois"])
        all_is_pos.extend([is_pos for is_pos in [True]*n_pos + [False]*(n-n_pos)])  # approximate

    # Recompute all_is_pos properly
    all_is_pos2 = []
    for seed, oid in train_oids:
        all_is_pos2.append(true_voi[(seed, oid)]["true_optimal"])

    global_p_pos = sum(all_is_pos2) / len(all_is_pos2) if all_is_pos2 else 0.0
    global_e_voi = sum(all_vois) / len(all_vois) if all_vois else 0.0
    global_optimal = global_p_pos > 0.5

    belief["__global__"] = {
        "n": len(train_oids), "n_positive": sum(all_is_pos2),
        "p_positive": global_p_pos, "e_observe_net": global_e_voi,
        "optimal_decision": global_optimal,
    }

    return belief

# =============================================================================
# Test Object Evaluation Data Builder
# =============================================================================
def build_test_obj_eval_data(test_oids):
    """Build evaluation data for test objects (used by ALL policies)."""
    obj_data = {}
    for seed, oid in test_oids:
        obj = per_seed_objects[seed][oid]
        lbl = per_seed_audit_labels[seed][oid]
        pre_features = obj["pre_surface_features"]
        post_features = obj["post_observe_features"]
        sig = get_pre_surface_signature(pre_features)

        true_aff = lbl["affordance_profile"]
        true_returns = {}
        for action in ALL_TRY_AFFORDANCES:
            outcome = true_aff.get(action, "fail")
            true_returns[f"try_{action}"] = try_net_value(outcome == "success")

        obj_data[(seed, oid)] = {
            "seed": seed, "oid": oid,
            "category": lbl["hidden_category"],
            "rationale_class": lbl["hidden_audit_rationale_class"],
            "signature": sig,
            "pre_features": pre_features,
            "true_returns": true_returns,
        }
    return obj_data

# =============================================================================
# Policy Evaluation: ALL 6 Policies on a Split
# =============================================================================
print("[5/8] Running multi-policy evaluation...")

def _make_pre_vec(pre_features):
    """Build pre_vec from pre_surface_features using fixed global keys."""
    sf = build_feature_vector(pre_features, {}, 0)
    return extract_feature_vector(sf)

def _make_post_vec(post_features, state):
    """Build post_vec from post_observe_features + state using fixed global keys."""
    sf = build_feature_vector(post_features, state, 1)
    return extract_feature_vector(sf)

def evaluate_all_policies(train_oids, test_oids, observe_cost,
                          affordance_overrides=None,
                          post_observe_overrides=None,
                          data_label="original"):
    """Evaluate all 6 policies on a single stress split.

    Policies:
    1. always_try (no observe)
    2. always_observe
    3. oracle_selective (test-side oracle, pre_best from signature groups)
    4. current_learned_policy (from corrected 7e D0 approach)
    5. signature_lookup_policy
    6. oracle_belief_policy (primary: E_observe_net > 0)

    All policies use the SAME training data for learned components.
    All feature vectors use the fixed ALL_MODEL_FEATURE_KEYS dimension.
    """
    # ---- Build training records (fixed-dimension vectors) ----
    train_records = []
    record_id = 0
    for seed, oid in train_oids:
        obj = per_seed_objects[seed][oid]
        lbl = per_seed_audit_labels[seed][oid]

        if affordance_overrides and (seed, oid) in affordance_overrides:
            affordance = affordance_overrides[(seed, oid)]
        else:
            affordance = lbl["affordance_profile"]

        pre_features = obj["pre_surface_features"]
        if post_observe_overrides and (seed, oid) in post_observe_overrides:
            post_features = post_observe_overrides[(seed, oid)]
        else:
            post_features = obj["post_observe_features"]
        state = obj["visible_state"]

        pre_vec = _make_pre_vec(pre_features)
        post_vec = _make_post_vec(post_features, state)

        record_id += 1
        train_records.append({
            "record_id": f"r{record_id:06d}", "seed": seed, "oid": oid,
            "feature_vector": pre_vec, "action_key": "observe",
            "net_value": -observe_cost,
        })
        for action in ALL_TRY_AFFORDANCES:
            outcome = affordance.get(action, "fail")
            record_id += 1
            train_records.append({
                "record_id": f"r{record_id:06d}", "seed": seed, "oid": oid,
                "feature_vector": pre_vec, "action_key": f"try_{action}",
                "net_value": try_net_value(outcome == "success"),
            })
        for action in ALL_TRY_AFFORDANCES:
            outcome = affordance.get(action, "fail")
            record_id += 1
            train_records.append({
                "record_id": f"r{record_id:06d}", "seed": seed, "oid": oid,
                "feature_vector": post_vec, "action_key": f"try_{action}",
                "net_value": try_net_value(outcome == "success"),
            })

    # Dimension audit
    train_lens = [len(r["feature_vector"]) for r in train_records]
    dim_train_min = min(train_lens) if train_lens else 0
    dim_train_max = max(train_lens) if train_lens else 0

    # ---- Train shared Q-function ----
    X_train = [r["feature_vector"] for r in train_records]
    y_train = [r["net_value"] for r in train_records]
    ak_train = [r["action_key"] for r in train_records]
    estimator = PerActionValueEstimator(alpha=RIDGE_ALPHA)
    estimator.fit(X_train, y_train, ak_train)

    # ---- Build test object data ----
    obj_data = build_test_obj_eval_data(test_oids)
    test_keys = sorted(obj_data.keys())
    n_test = len(test_keys)

    # ---- Compute oracle-belief from TRAINING only ----
    belief = compute_oracle_belief(train_oids, observe_cost, affordance_overrides)

    # ---- Compute signature_lookup from TRAINING only ----
    sig_lookup = {}
    for sig, bd in belief.items():
        if sig == "__global__":
            continue
        sig_lookup[sig] = bd["optimal_decision"]

    # =====================================================================
    # Policy 1: always_try
    # =====================================================================
    always_try_returns = []
    for key in test_keys:
        od = obj_data[key]
        pre_features = od["pre_features"]
        pre_vec_eval = _make_pre_vec(pre_features)
        qs = estimator.predict_all_actions(pre_vec_eval)
        try_qs = {ak: q for ak, q in qs.items() if ak != "observe"}
        best_action = max(try_qs, key=try_qs.get)
        always_try_returns.append(od["true_returns"].get(best_action, 0.0))
    always_try_net = sum(always_try_returns) / n_test if n_test else 0.0

    # =====================================================================
    # Policy 2: always_observe
    # =====================================================================
    always_obs_returns = []
    for key in test_keys:
        od = obj_data[key]
        post_best = max(od["true_returns"].values())
        always_obs_returns.append(post_best - observe_cost)
    always_obs_net = sum(always_obs_returns) / n_test if n_test else 0.0

    # =====================================================================
    # Policy 3: oracle_selective
    # =====================================================================
    test_sig_groups = defaultdict(list)
    for key in test_keys:
        sig = obj_data[key]["signature"]
        test_sig_groups[sig].append(obj_data[key])

    test_sig_expected = {}
    for sig, members in test_sig_groups.items():
        n = len(members)
        action_expected = {}
        for action in ALL_TRY_AFFORDANCES:
            total = sum(m["true_returns"].get(f"try_{action}", 0.0) for m in members)
            action_expected[action] = total / n
        best_action = max(action_expected, key=action_expected.get)
        test_sig_expected[sig] = {
            "n_members": n,
            "best_action": best_action,
            "best_expected_value": action_expected[best_action],
        }

    oracle_sel_returns = []
    oracle_sel_obs = 0
    for key in test_keys:
        od = obj_data[key]
        true_returns = od["true_returns"]
        post_best = max(true_returns.values())
        sig = od["signature"]
        se = test_sig_expected.get(sig)
        pre_best_expected = se["best_expected_value"] if se else 0.0
        oracle_voi = post_best - pre_best_expected - observe_cost
        if oracle_voi > 0:
            oracle_sel_obs += 1
            oracle_sel_returns.append(post_best - observe_cost)
        else:
            oracle_sel_returns.append(pre_best_expected)
    oracle_sel_net = sum(oracle_sel_returns) / n_test if n_test else 0.0

    # =====================================================================
    # Policy 4: current_learned_policy (from 7e D0)
    # =====================================================================
    train_sig_groups_ov = defaultdict(list)
    for tseed, toid in train_oids:
        tobj = per_seed_objects[tseed][toid]
        tlbl = per_seed_audit_labels[tseed][toid]
        tsig = get_pre_surface_signature(tobj["pre_surface_features"])
        if affordance_overrides and (tseed, toid) in affordance_overrides:
            taff = affordance_overrides[(tseed, toid)]
        else:
            taff = tlbl["affordance_profile"]
        true_rets = {}
        for action in ALL_TRY_AFFORDANCES:
            outcome = taff.get(action, "fail")
            true_rets[action] = try_net_value(outcome == "success")
        train_sig_groups_ov[tsig].append(true_rets)

    train_sig_exp_ov = {}
    for tsig, rets_list in train_sig_groups_ov.items():
        n = len(rets_list)
        act_exp = {}
        for action in ALL_TRY_AFFORDANCES:
            act_exp[action] = sum(r[action] for r in rets_list) / n
        best_act = max(act_exp, key=act_exp.get)
        train_sig_exp_ov[tsig] = act_exp[best_act]

    ov_X = []
    ov_y = []
    for tseed, toid in train_oids:
        tobj = per_seed_objects[tseed][toid]
        tlbl = per_seed_audit_labels[tseed][toid]
        pre_features = tobj["pre_surface_features"]
        if post_observe_overrides and (tseed, toid) in post_observe_overrides:
            post_features = post_observe_overrides[(tseed, toid)]
        else:
            post_features = tobj["post_observe_features"]
        state = tobj["visible_state"]

        pre_vec_ov = _make_pre_vec(pre_features)
        post_vec_ov = _make_post_vec(post_features, state)

        pre_qs = estimator.predict_all_actions(pre_vec_ov)
        post_qs = estimator.predict_all_actions(post_vec_ov)
        pre_try_qs = {ak: q for ak, q in pre_qs.items() if ak != "observe"}
        post_try_qs = {ak: q for ak, q in post_qs.items() if ak != "observe"}
        pre_best = max(pre_try_qs.values()) if pre_try_qs else 0.0
        post_best = max(post_try_qs.values()) if post_try_qs else 0.0

        ov_y.append(post_best - pre_best - observe_cost)
        ov_X.append(list(pre_vec_ov))

    ov_model = None
    ov_normalizer = None
    if len(ov_X) >= 3 and len(set(ov_y)) > 1:
        ov_normalizer = FeatureNormalizer().fit(ov_X)
        ov_X_norm = ov_normalizer.transform(ov_X)
        ov_model = RidgeRegression(alpha=RIDGE_ALPHA).fit(ov_X_norm, ov_y)

    learned_returns = []
    learned_obs = 0
    for key in test_keys:
        od = obj_data[key]
        pre_features = od["pre_features"]
        true_returns = od["true_returns"]

        pre_vec_eval = _make_pre_vec(pre_features)
        pre_qs = estimator.predict_all_actions(pre_vec_eval)
        pre_try_qs = {ak: q for ak, q in pre_qs.items() if ak != "observe"}
        pre_best_action = max(pre_try_qs, key=pre_try_qs.get)

        if ov_model is not None and ov_normalizer is not None:
            ov_input_norm = ov_normalizer.transform([pre_vec_eval])[0]
            predicted_observe_net = ov_model.predict([ov_input_norm])[0]
        else:
            predicted_observe_net = -observe_cost

        decided_observe = predicted_observe_net > 0

        if decided_observe:
            learned_obs += 1
            post_best = max(true_returns.values())
            learned_returns.append(post_best - observe_cost)
        else:
            learned_returns.append(true_returns.get(pre_best_action, 0.0))
    learned_net = sum(learned_returns) / n_test if n_test else 0.0

    # =====================================================================
    # Policy 5: signature_lookup_policy
    # =====================================================================
    sig_lookup_returns = []
    sig_lookup_obs = 0
    sig_lookup_heldout = 0
    global_optimal = belief.get("__global__", {}).get("optimal_decision", True)

    for key in test_keys:
        od = obj_data[key]
        sig = od["signature"]
        true_returns = od["true_returns"]
        pre_features = od["pre_features"]

        pre_vec_eval = _make_pre_vec(pre_features)
        pre_qs = estimator.predict_all_actions(pre_vec_eval)
        pre_try_qs = {ak: q for ak, q in pre_qs.items() if ak != "observe"}
        pre_best_action = max(pre_try_qs, key=pre_try_qs.get)

        if sig in sig_lookup:
            should_observe = sig_lookup[sig]
        else:
            sig_lookup_heldout += 1
            should_observe = global_optimal

        if should_observe:
            sig_lookup_obs += 1
            post_best = max(true_returns.values())
            sig_lookup_returns.append(post_best - observe_cost)
        else:
            sig_lookup_returns.append(true_returns.get(pre_best_action, 0.0))
    sig_lookup_net = sum(sig_lookup_returns) / n_test if n_test else 0.0

    # =====================================================================
    # Policy 6: oracle_belief_policy (A: E_observe_net > 0)
    # =====================================================================
    oracle_belief_returns = []
    oracle_belief_obs = 0
    oracle_belief_heldout = 0
    global_e_voi = belief.get("__global__", {}).get("e_observe_net", 0.0)

    for key in test_keys:
        od = obj_data[key]
        sig = od["signature"]
        true_returns = od["true_returns"]
        pre_features = od["pre_features"]

        pre_vec_eval = _make_pre_vec(pre_features)
        pre_qs = estimator.predict_all_actions(pre_vec_eval)
        pre_try_qs = {ak: q for ak, q in pre_qs.items() if ak != "observe"}
        pre_best_action = max(pre_try_qs, key=pre_try_qs.get)

        if sig in belief:
            e_voi = belief[sig]["e_observe_net"]
        else:
            oracle_belief_heldout += 1
            e_voi = global_e_voi

        should_observe = e_voi > 0

        if should_observe:
            oracle_belief_obs += 1
            post_best = max(true_returns.values())
            oracle_belief_returns.append(post_best - observe_cost)
        else:
            oracle_belief_returns.append(true_returns.get(pre_best_action, 0.0))
    oracle_belief_net = sum(oracle_belief_returns) / n_test if n_test else 0.0

    # =====================================================================
    # Per-rationale breakdowns for oracle_belief
    # =====================================================================
    pos_obs = 0
    pos_total = 0
    nonpos_obs = 0
    nonpos_total = 0

    for key in test_keys:
        od = obj_data[key]
        sig = od["signature"]
        pre_features = od["pre_features"]

        pre_vec_eval = _make_pre_vec(pre_features)
        pre_qs = estimator.predict_all_actions(pre_vec_eval)
        pre_try_qs = {ak: q for ak, q in pre_qs.items() if ak != "observe"}
        pre_best_action = max(pre_try_qs, key=pre_try_qs.get)

        if sig in belief:
            e_voi = belief[sig]["e_observe_net"]
        else:
            e_voi = global_e_voi
        should_observe = e_voi > 0

        if od["rationale_class"] == "positive_change":
            pos_total += 1
            if should_observe:
                pos_obs += 1
        else:
            nonpos_total += 1
            if should_observe:
                nonpos_obs += 1

    # Per-signature belief info for test objects
    test_sig_beliefs = {}
    for key in test_keys:
        od = obj_data[key]
        sig = od["signature"]
        if sig not in test_sig_beliefs:
            bd = belief.get(sig)
            test_sig_beliefs[sig] = {
                "in_training": sig in belief,
                "n_train": bd["n"] if bd else 0,
                "p_positive": bd["p_positive"] if bd else None,
                "e_observe_net": bd["e_observe_net"] if bd else None,
                "optimal_decision": bd["optimal_decision"] if bd else None,
            }

    # Eval dimension audit
    eval_lens = []
    ev_count = 0
    for key in test_keys:
        od = obj_data[key]
        ev = _make_pre_vec(od["pre_features"])
        eval_lens.append(len(ev))
        ev_count += 1
        if ev_count <= 2:  # just sample
            pass

    dim_eval_min = min(eval_lens) if eval_lens else 0
    dim_eval_max = max(eval_lens) if eval_lens else 0

    return {
        "data_label": data_label,
        "n_train": len(train_oids),
        "n_test": n_test,
        "dim_n_features": N_FEATURES,
        "dim_train_min": dim_train_min,
        "dim_train_max": dim_train_max,
        "dim_train_match": dim_train_min == dim_train_max == N_FEATURES,
        "dim_eval_min": dim_eval_min,
        "dim_eval_max": dim_eval_max,
        "dim_eval_match": dim_eval_min == dim_eval_max == N_FEATURES,
        # All 6 policy nets
        "always_try_net": round(always_try_net, 6),
        "always_observe_net": round(always_obs_net, 6),
        "oracle_selective_net": round(oracle_sel_net, 6),
        "learned_policy_net": round(learned_net, 6),
        "signature_lookup_net": round(sig_lookup_net, 6),
        "oracle_belief_net": round(oracle_belief_net, 6),
        # Deltas
        "oracle_belief_vs_always_observe": round(oracle_belief_net - always_obs_net, 6),
        "oracle_belief_vs_always_try": round(oracle_belief_net - always_try_net, 6),
        "oracle_selective_vs_oracle_belief": round(oracle_sel_net - oracle_belief_net, 6),
        "learned_vs_oracle_belief": round(learned_net - oracle_belief_net, 6),
        "signature_lookup_vs_oracle_belief": round(sig_lookup_net - oracle_belief_net, 6),
        # Observe rates
        "oracle_belief_obs_rate": round(oracle_belief_obs / n_test, 6) if n_test else 0.0,
        "oracle_belief_pos_obs_rate": round(pos_obs / pos_total, 6) if pos_total > 0 else 0.0,
        "oracle_belief_nonpos_obs_rate": round(nonpos_obs / nonpos_total, 6) if nonpos_total > 0 else 0.0,
        "learned_obs_rate": round(learned_obs / n_test, 6) if n_test else 0.0,
        "signature_lookup_obs_rate": round(sig_lookup_obs / n_test, 6) if n_test else 0.0,
        "oracle_selective_obs_rate": round(oracle_sel_obs / n_test, 6) if n_test else 0.0,
        # Held-out info
        "oracle_belief_heldout_sigs": oracle_belief_heldout,
        "sig_lookup_heldout_sigs": sig_lookup_heldout,
        # Per-signature belief
        "test_sig_beliefs": test_sig_beliefs,
    }

# =============================================================================
# Experiment Loop
# =============================================================================
print("[6/8] Running experiment loop across all costs, distributions, tests...")

ALL_RESULTS = {}  # keyed by (observe_cost, condition_label, test_group)

for observe_cost in OBSERVE_COSTS:
    print(f"\n  --- observe_cost = {observe_cost} ---")

    for condition_label, condition_builder in [
        ("original", lambda train_oids: (None, None)),
        ("counterfactual_cov0.25", lambda train_oids: (
            build_counterfactual_affordance_overrides(train_oids, 0.25)[0],
            build_lookalike_post_observe_overrides(
                build_counterfactual_affordance_overrides(train_oids, 0.25)[2]))),
        ("counterfactual_cov0.50", lambda train_oids: (
            build_counterfactual_affordance_overrides(train_oids, 0.50)[0],
            build_lookalike_post_observe_overrides(
                build_counterfactual_affordance_overrides(train_oids, 0.50)[2]))),
    ]:
        for test_group, test_variants in [
            ("A", TEST_A_VARIANTS), ("B", TEST_B_VARIANTS), ("C", TEST_C_VARIANTS)
        ]:
            group_results = []
            for tv in test_variants:
                affordance_overrides, post_observe_overrides = condition_builder(tv["train_oids"])
                r = evaluate_all_policies(
                    tv["train_oids"], tv["test_oids"], observe_cost,
                    affordance_overrides=affordance_overrides,
                    post_observe_overrides=post_observe_overrides,
                    data_label=condition_label)
                r["test_variant_id"] = tv["id"]
                r["test_group"] = test_group
                group_results.append(r)

            # Aggregate across variants
            total_n = sum(gr["n_test"] for gr in group_results)
            def wavg(field):
                return round(sum(gr[field] * gr["n_test"] for gr in group_results) / total_n, 6) if total_n > 0 else 0.0

            agg = {
                "observe_cost": observe_cost,
                "condition": condition_label,
                "test_group": test_group,
                "n_total": total_n,
                "always_try_net": wavg("always_try_net"),
                "always_observe_net": wavg("always_observe_net"),
                "oracle_selective_net": wavg("oracle_selective_net"),
                "learned_policy_net": wavg("learned_policy_net"),
                "signature_lookup_net": wavg("signature_lookup_net"),
                "oracle_belief_net": wavg("oracle_belief_net"),
                "oracle_belief_vs_always_observe": wavg("oracle_belief_vs_always_observe"),
                "oracle_belief_vs_always_try": wavg("oracle_belief_vs_always_try"),
                "oracle_selective_vs_oracle_belief": wavg("oracle_selective_vs_oracle_belief"),
                "learned_vs_oracle_belief": wavg("learned_vs_oracle_belief"),
                "oracle_belief_obs_rate": wavg("oracle_belief_obs_rate"),
                "oracle_belief_pos_obs_rate": wavg("oracle_belief_pos_obs_rate"),
                "oracle_belief_nonpos_obs_rate": wavg("oracle_belief_nonpos_obs_rate"),
                "learned_obs_rate": wavg("learned_obs_rate"),
                "variant_details": group_results,
            }
            ALL_RESULTS[(observe_cost, condition_label, test_group)] = agg

            print(f"    {condition_label}/{test_group}: "
                  f"always_obs={agg['always_observe_net']:.4f} "
                  f"oracle_belief={agg['oracle_belief_net']:.4f} "
                  f"vs_always_obs={agg['oracle_belief_vs_always_observe']:+.4f} "
                  f"obs_rate={agg['oracle_belief_obs_rate']:.4f}")

# =============================================================================
# Headroom Conclusion
# =============================================================================
print("\n[7/8] Computing headroom conclusions...")

HEADROOM_VERDICTS = {}

for observe_cost in OBSERVE_COSTS:
    for condition_label in ["original", "counterfactual_cov0.25", "counterfactual_cov0.50"]:
        key_a = (observe_cost, condition_label, "A")
        key_b = (observe_cost, condition_label, "B")
        key_c = (observe_cost, condition_label, "C")

        agg_a = ALL_RESULTS.get(key_a)
        agg_b = ALL_RESULTS.get(key_b)
        agg_c = ALL_RESULTS.get(key_c)

        if not agg_a:
            continue

        ob_vs_always_obs_a = agg_a["oracle_belief_vs_always_observe"]
        ob_vs_always_obs_b = agg_b["oracle_belief_vs_always_observe"] if agg_b else 0.0
        ob_obs_rate_a = agg_a["oracle_belief_obs_rate"]
        ob_obs_rate_b = agg_b["oracle_belief_obs_rate"] if agg_b else 0.0
        ob_pos = agg_a["oracle_belief_pos_obs_rate"]
        ob_nonpos = agg_a["oracle_belief_nonpos_obs_rate"]
        ob_vs_learned_a = agg_a["learned_vs_oracle_belief"]
        oracle_gap = agg_a["oracle_selective_vs_oracle_belief"]

        # Criteria
        beats_always_obs = ob_vs_always_obs_a > 0.005  # meaningful margin
        beats_learned = ob_vs_learned_a < 0  # oracle_belief > learned
        selective = 0.01 < ob_obs_rate_a < 0.99
        discriminates = ob_pos > ob_nonpos
        closer_to_oracle = abs(oracle_gap) < abs(agg_a["oracle_selective_net"] - agg_a["always_observe_net"])

        # PASS: meaningful improvement on Test A and/or B
        a_improves = ob_vs_always_obs_a > 0.005 and selective and discriminates
        b_improves = ob_vs_always_obs_b > 0.005 if agg_b else False

        if a_improves or b_improves:
            verdict = "PASS: structural VOI justified"
        elif ob_vs_always_obs_a > 0.001:
            verdict = "PARTIAL: slight improvement but marginal"
        else:
            verdict = "FAIL: insufficient selective-observe headroom"

        HEADROOM_VERDICTS[(observe_cost, condition_label)] = {
            "verdict": verdict,
            "ob_vs_always_obs_a": ob_vs_always_obs_a,
            "ob_vs_always_obs_b": ob_vs_always_obs_b,
            "ob_obs_rate_a": ob_obs_rate_a,
            "beats_always_obs": beats_always_obs,
            "beats_learned": beats_learned,
            "selective": selective,
            "discriminates": discriminates,
            "closer_to_oracle": closer_to_oracle,
        }

        print(f"  cost={observe_cost} {condition_label}: {verdict}")
        print(f"    ob_vs_always_obs(A)={ob_vs_always_obs_a:+.4f} "
              f"ob_rate={ob_obs_rate_a:.4f} pos_obs={ob_pos:.4f} nonpos_obs={ob_nonpos:.4f}")

# =============================================================================
# Output Generation
# =============================================================================
print("\n[8/8] Writing output files...")

elapsed = time.time() - t0

# --- JSON ---
json_path = os.path.join(CURRENT_DIR, "runs",
    "block1j40b7f_oracle_belief_upper_bound_control.json")

json_output = {
    "block_id": "1J40b-7f",
    "implementation_status": "COMPLETED",
    "elapsed_seconds": round(elapsed, 1),
    "seeds": SEEDS,
    "observe_costs": OBSERVE_COSTS,
    "coverage_fractions": COVERAGE_FRACTIONS,
    "oracle_belief_definition": {
        "primary_rule": "observe if E_observe_net > 0",
        "E_observe_net": "empirical mean(true_oracle_VOI | pre_surface_signature) from training only",
        "P_positive": "empirical P(positive_net | pre_surface_signature) from training only",
        "fallback_method": "global_mean (training-side global E_observe_net / P_positive)",
        "audit_only": True,
        "not_fed_to_learner": True,
    },
    "results": {},
    "headroom_verdicts": {},
}

for (obs_cost, cond, test_group), agg in sorted(ALL_RESULTS.items(),
    key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
    json_output["results"][f"cost{obs_cost}_{cond}_{test_group}"] = {
        k: v for k, v in agg.items() if k != "variant_details"
    }

for (obs_cost, cond), vd in HEADROOM_VERDICTS.items():
    json_output["headroom_verdicts"][f"cost{obs_cost}_{cond}"] = vd

with open(json_path, "w") as f:
    json.dump(json_output, f, indent=2)
print(f"JSON -> {json_path}")

# --- CSV ---
csv_path = os.path.join(CURRENT_DIR, "protocols",
    "block1j40b7f_oracle_belief_upper_bound_control_table.csv")
csv_lines = [
    "observe_cost,condition,test_group,always_try,always_observe,oracle_selective,"
    "learned_policy,signature_lookup,oracle_belief,"
    "ob_vs_always_obs,ob_vs_always_try,oracle_vs_ob,learned_vs_ob,"
    "ob_obs_rate,ob_pos_obs,ob_nonpos_obs,learned_obs_rate"
]
for (obs_cost, cond, test_group), agg in sorted(ALL_RESULTS.items(),
    key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
    csv_lines.append(
        f"{obs_cost},{cond},{test_group},"
        f"{agg['always_try_net']:.6f},{agg['always_observe_net']:.6f},"
        f"{agg['oracle_selective_net']:.6f},{agg['learned_policy_net']:.6f},"
        f"{agg['signature_lookup_net']:.6f},{agg['oracle_belief_net']:.6f},"
        f"{agg['oracle_belief_vs_always_observe']:.6f},"
        f"{agg['oracle_belief_vs_always_try']:.6f},"
        f"{agg['oracle_selective_vs_oracle_belief']:.6f},"
        f"{agg['learned_vs_oracle_belief']:.6f},"
        f"{agg['oracle_belief_obs_rate']:.6f},"
        f"{agg['oracle_belief_pos_obs_rate']:.6f},"
        f"{agg['oracle_belief_nonpos_obs_rate']:.6f},"
        f"{agg['learned_obs_rate']:.6f}"
    )
with open(csv_path, "w") as f:
    f.write("\n".join(csv_lines) + "\n")
print(f"CSV -> {csv_path}")

# --- MD Protocol ---
md_path = os.path.join(CURRENT_DIR, "protocols",
    "block1j40b7f_oracle_belief_upper_bound_control.md")
md = []
md.append("# Block 1J40b-7f: Oracle-Belief Upper-Bound Control")
md.append("")
md.append(f"- **Implementation Status**: COMPLETED")
md.append(f"- **Elapsed**: {elapsed:.1f}s")
md.append(f"- **Seeds**: {SEEDS}")
md.append(f"- **Observe Costs**: {OBSERVE_COSTS}")
md.append(f"- **Coverage Fractions**: {COVERAGE_FRACTIONS}")
md.append("")

md.append("## 1. Files")
md.append("")
md.append("- `_block1j40b7f_oracle_belief_upper_bound_control.py` -- Written")
md.append("- `runs/block1j40b7f_oracle_belief_upper_bound_control.json` -- Generated")
md.append("- `protocols/block1j40b7f_oracle_belief_upper_bound_control.md` -- Generated")
md.append("- `protocols/block1j40b7f_oracle_belief_upper_bound_control_table.csv` -- Generated")
md.append("")

md.append("## 2. Oracle-Belief Definition")
md.append("")
md.append("- **Primary rule**: observe if E_observe_net > 0 (threshold on empirical mean VOI per signature)")
md.append("- **E_observe_net**: mean(true_oracle_VOI | pre_surface_signature) computed from TRAINING ONLY")
md.append("- **P_positive**: empirical P(positive_net | pre_surface_signature) from TRAINING ONLY")
md.append("- **Fallback for held-out signatures**: global mean E_observe_net from training")
md.append("- **Audit-only**: never fed to learner, never written into model-visible records")
md.append("")

md.append("## 3. Policies Compared")
md.append("")
md.append("1. always_try (no observe)")
md.append("2. always_observe")
md.append("3. oracle_selective (test-side oracle, pre_best from signature groups)")
md.append("4. current_learned_policy (7e D0: raw pre_vec observe_value_model)")
md.append("5. signature_lookup_policy (training-side majority optimal per sig)")
md.append("6. oracle_belief_policy (E_observe_net > 0 per sig, global fallback)")
md.append("")

md.append("## 4. Headroom Criteria")
md.append("")
md.append("- **PASS / structural VOI justified**: oracle_belief > always_observe by meaningful margin on Test A/B, selective observe rate, discriminates positive from non-positive")
md.append("- **PARTIAL**: slight improvement but marginal or only one test improves")
md.append("- **FAIL / insufficient headroom**: oracle_belief <= always_observe, also collapses")
md.append("")

# Cost sections
for observe_cost in OBSERVE_COSTS:
    md.append(f"## 5. Results at observe_cost = {observe_cost}")
    md.append("")
    md.append("| Test | Condition | always_try | always_obs | oracle_sel | learned | sig_lookup | oracle_belief | ob-ao | ob-at | osel-ob | learned-ob | ob_rate | pos_obs | nonpos_obs |")
    md.append("|------|-----------|------------|------------|------------|---------|------------|---------------|-------|-------|---------|------------|--------|---------|------------|")

    for condition_label, _ in [
        ("original", None),
        ("counterfactual_cov0.25", None),
        ("counterfactual_cov0.50", None),
    ]:
        for test_group in ["A", "B", "C"]:
            key = (observe_cost, condition_label, test_group)
            if key not in ALL_RESULTS:
                continue
            agg = ALL_RESULTS[key]
            md.append(
                f"| {test_group} | {condition_label} | "
                f"{agg['always_try_net']:.4f} | {agg['always_observe_net']:.4f} | "
                f"{agg['oracle_selective_net']:.4f} | {agg['learned_policy_net']:.4f} | "
                f"{agg['signature_lookup_net']:.4f} | {agg['oracle_belief_net']:.4f} | "
                f"{agg['oracle_belief_vs_always_observe']:+.4f} | {agg['oracle_belief_vs_always_try']:+.4f} | "
                f"{agg['oracle_selective_vs_oracle_belief']:+.4f} | {agg['learned_vs_oracle_belief']:+.4f} | "
                f"{agg['oracle_belief_obs_rate']:.4f} | {agg['oracle_belief_pos_obs_rate']:.4f} | "
                f"{agg['oracle_belief_nonpos_obs_rate']:.4f} |"
            )
    md.append("")

md.append("## 6. Headroom Verdicts")
md.append("")
md.append("| Cost | Condition | Verdict | ob_vs_always_obs(A) | ob_rate(A) | pos_obs | nonpos_obs | Selective? | Discriminates? |")
md.append("|------|-----------|---------|---------------------|------------|---------|------------|------------|----------------|")

for (obs_cost, cond), vd in sorted(HEADROOM_VERDICTS.items()):
    md.append(
        f"| {obs_cost} | {cond} | {vd['verdict']} | "
        f"{vd['ob_vs_always_obs_a']:+.4f} | {vd['ob_obs_rate_a']:.4f} | "
        f"{(ALL_RESULTS.get((obs_cost, cond, 'A'), {}).get('oracle_belief_pos_obs_rate', 0)):.4f} | "
        f"{(ALL_RESULTS.get((obs_cost, cond, 'A'), {}).get('oracle_belief_nonpos_obs_rate', 0)):.4f} | "
        f"{vd['selective']} | {vd['discriminates']} |"
    )
md.append("")

md.append("## 7. Interpretation")
md.append("")

# Check if any verdict is PASS
any_pass = any(vd["verdict"].startswith("PASS") for vd in HEADROOM_VERDICTS.values())
any_partial = any(vd["verdict"].startswith("PARTIAL") for vd in HEADROOM_VERDICTS.values())

if any_pass:
    md.append("**Structural VOI has headroom; 1J40b-8 is justified.**")
    md.append("")
    md.append("The oracle-belief upper bound materially beats always_observe on held-out tests, ")
    md.append("demonstrating that the counterfactual distribution contains sufficient ")
    md.append("selective-observe headroom. The current learned policy's failure is due to ")
    md.append("insufficient feature representation or learning architecture, not inherent ")
    md.append("lack of VOI signal.")
elif any_partial:
    md.append("**Partial headroom; proceed to 1J40b-8 with caution.**")
    md.append("")
    md.append("The oracle-belief upper bound shows marginal improvement over always_observe. ")
    md.append("1J40b-8 may be justified but gains may be small.")
else:
    md.append("**Counterfactual distribution still lacks sufficient selective-observe headroom;** ")
    md.append("**do not implement 1J40b-8 yet.**")
    md.append("")
    md.append("The oracle-belief upper bound does NOT materially beat always_observe on held-out ")
    md.append("tests. Even with oracle knowledge of empirical observe usefulness by signature, ")
    md.append("the counterfactual distribution does not provide enough advantage for selective ")
    md.append("observation to outperform always-observe.")
md.append("")

md.append("## 8. Commands Run")
md.append("")
md.append("```")
md.append('& "D:\\conda\\python.exe" "_block1j40b7f_oracle_belief_upper_bound_control.py"')
md.append("```")
md.append("")
md.append("```")
md.append("[block_done]")
md.append("block_id=1J40b-7f")
md.append("implementation_status=COMPLETED")
md.append("```")

with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")
print(f"MD -> {md_path}")

print(f"\nDone in {elapsed:.1f}s")
