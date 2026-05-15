"""
Block 1J40b-7e -- env3b Distributional Audit + Minimal Counterfactual Coverage.

Part A (Distributional Audit): Characterise the training distribution using
  TRUE/ORACLE observe-value targets under legal pre_signature grouping.
  Q-based targets are reported as a sanity check only.

Part B (Counterfactual Coverage): Create counterfactual training distributions
  (explicitly labeled as interventions, NOT natural env3b redesigns) where
  within each pre_surface signature, instances have BOTH positive and
  non-positive optimal observe decisions.

Part C (Policy Evaluation): Run two-stage D0 policy on original vs
  counterfactual training data for both coverage fractions (0.25, 0.50),
  all stress tests (A/B/C), both observe costs (0.03, 0.05).

Key constraint: test objects are UNCHANGED. Only training affordances modified.
Decision-time information boundary preserved: pre-only input for observe/direct.
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
# Config safety
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
# Constants (from env3b)
# =============================================================================
SEEDS = [101, 103, 107, 109, 113]
RIDGE_ALPHA = 1.0

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
# Instance Subtype Definitions (from 7a)
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
# Object Generation (from 7a)
# =============================================================================
print("[1/12] Generating env3b objects...")

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

n_per_seed = len(per_seed_objects[SEEDS[0]])
print(f"  Objects per seed: {n_per_seed}, total: {n_per_seed * len(SEEDS)}")

# =============================================================================
# Feature Vector Building
# =============================================================================
print("[2/12] Building feature infrastructure...")

def try_net_value(success_or_failure):
    if success_or_failure is True:
        return SUCCESS_REWARD - 0.05
    elif success_or_failure is False:
        return -FAILURE_PENALTY - 0.05
    return 0.0

def get_pre_surface_signature(pre_surface_features):
    return tuple(sorted(pre_surface_features.keys()))

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
print(f"  Post-observe features: {len(POST_OBSERVE_FEATURE_NAMES)}")
print(f"  Model feature keys: {len(ALL_MODEL_FEATURE_KEYS)}")

def build_feature_vector(known_features, known_states, prior_obs_count):
    feat = {}
    for fname in POST_OBSERVE_FEATURE_NAMES:
        feat[f"feat_{fname}"] = 1.0 if fname in known_features else 0.0
    for sname in STATE_FEATURE_NAMES:
        feat[f"state_{sname}"] = 1.0 if known_states.get(sname, False) else 0.0
    feat["prior_observe_count"] = float(prior_obs_count)
    return feat

def extract_feature_vector(situation_features):
    return [float(situation_features.get(k, 0.0)) for k in ALL_MODEL_FEATURE_KEYS]

# =============================================================================
# Signature Analysis
# =============================================================================
print("[3/12] Analyzing signatures...")

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
        print(f"    SHARED sig n={len(members)}: {sig}")
    else:
        unique_sigs.append(sig)
        print(f"    UNIQUE sig n={len(members)}: {sig}")

# =============================================================================
# Model Classes
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
        self._fitted_actions = set()

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
            self._fitted_actions.add(ak)
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
# Stress Split Definitions (from 7c)
# =============================================================================
print("[4/12] Defining stress test splits...")

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

print(f"  Test A: {len(TEST_A_VARIANTS)} variants")
print(f"  Test B: {len(TEST_B_VARIANTS)} variants")
print(f"  Test C: {len(TEST_C_VARIANTS)} variants")

# =============================================================================
# Training Data Helpers
# =============================================================================
print("[5/12] Building training data helpers...")

def build_training_records_from_oids(train_oids, observe_cost, affordance_overrides=None,
                                     post_observe_overrides=None):
    """Build training records using (optionally overridden) affordance profiles
    and post_observe features (for lookalike corrective evidence)."""
    records = []
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

        pre_sf = build_feature_vector(pre_features, {}, 0)
        pre_vec = extract_feature_vector(pre_sf)
        post_sf = build_feature_vector(post_features, state, 1)
        post_vec = extract_feature_vector(post_sf)

        record_id += 1
        records.append({
            "record_id": f"r{record_id:06d}", "seed": seed, "oid": oid,
            "feature_vector": pre_vec, "action_key": "observe",
            "net_value": -observe_cost,
        })
        for action in ALL_TRY_AFFORDANCES:
            outcome = affordance.get(action, "fail")
            record_id += 1
            records.append({
                "record_id": f"r{record_id:06d}", "seed": seed, "oid": oid,
                "feature_vector": pre_vec, "action_key": f"try_{action}",
                "net_value": try_net_value(outcome == "success"),
            })
        for action in ALL_TRY_AFFORDANCES:
            outcome = affordance.get(action, "fail")
            record_id += 1
            records.append({
                "record_id": f"r{record_id:06d}", "seed": seed, "oid": oid,
                "feature_vector": post_vec, "action_key": f"try_{action}",
                "net_value": try_net_value(outcome == "success"),
            })
    return records


def build_obj_eval_vectors(test_oids, affordance_overrides=None):
    """Build evaluation vectors for test objects (never overridden)."""
    obj_data = {}
    for seed, oid in test_oids:
        obj = per_seed_objects[seed][oid]
        lbl = per_seed_audit_labels[seed][oid]
        pre_features = obj["pre_surface_features"]
        post_features = obj["post_observe_features"]
        state = obj["visible_state"]
        sig = get_pre_surface_signature(pre_features)

        pre_sf = build_feature_vector(pre_features, {}, 0)
        pre_vec = extract_feature_vector(pre_sf)
        post_sf = build_feature_vector(post_features, state, 1)
        post_vec = extract_feature_vector(post_sf)

        # True returns always use the ORIGINAL affordance (test objects unchanged)
        true_aff = lbl["affordance_profile"]
        true_returns = {}
        for action in ALL_TRY_AFFORDANCES:
            outcome = true_aff.get(action, "fail")
            true_returns[f"try_{action}"] = try_net_value(outcome == "success")

        obj_data[(seed, oid)] = {
            "seed": seed, "oid": oid,
            "category": lbl["hidden_category"],
            "subtype": lbl["hidden_subtype"],
            "rationale_class": lbl["hidden_audit_rationale_class"],
            "signature": sig,
            "pre_features": pre_features,
            "pre_vec": pre_vec, "post_vec": post_vec,
            "true_returns": true_returns,
            "n_pre_surface_features": len(pre_features),
        }
    return obj_data


# =============================================================================
# Helper: compute true/oracle VOI under legal pre_signature grouping
# =============================================================================
def compute_true_oracle_voi(oid_set, observe_cost, affordance_overrides=None):
    """Compute true/oracle VOI for a set of (seed, oid) instances.

    pre_best is computed from signature-group expected returns
    (legal: uses only pre_surface_signature grouping, NOT hidden category).

    post_best is computed from the individual instance's affordance.

    Returns dict (seed, oid) -> {
        true_pre_best, true_post_best, true_voi, true_optimal
    }
    """
    # Group by signature to compute expected returns
    sig_instances = defaultdict(list)
    for seed, oid in oid_set:
        obj = per_seed_objects[seed][oid]
        sig = get_pre_surface_signature(obj["pre_surface_features"])
        if affordance_overrides and (seed, oid) in affordance_overrides:
            aff = affordance_overrides[(seed, oid)]
        else:
            aff = per_seed_audit_labels[seed][oid]["affordance_profile"]
        sig_instances[sig].append((seed, oid, aff))

    # Per-signature expected return per action
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

    # Per-instance VOI
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


def binary_entropy(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


# =============================================================================
# Part A: Distributional Audit (TRUE/ORACLE primary)
# =============================================================================
print("[6/12] PART A: Distributional audit (true/oracle primary)...")

def run_distributional_audit_true_primary():
    """Characterize whether each pre_surface signature perfectly predicts the
    optimal observe decision, using TRUE/ORACLE VOI as primary.

    Q-based VOI is also computed as a sanity check.
    """
    full_oids = set()
    for seed in SEEDS:
        for oid in per_seed_objects[seed]:
            full_oids.add((seed, oid))

    # True/oracle VOI (primary) at both costs
    true_voi_003 = compute_true_oracle_voi(full_oids, 0.03)
    true_voi_005 = compute_true_oracle_voi(full_oids, 0.05)

    # Q-based VOI (sanity check) -- fit Q on full data in-sample
    train_records_full = build_training_records_from_oids(full_oids, 0.03)
    X_train = [r["feature_vector"] for r in train_records_full]
    y_train = [r["net_value"] for r in train_records_full]
    ak_train = [r["action_key"] for r in train_records_full]
    estimator = PerActionValueEstimator(alpha=RIDGE_ALPHA)
    estimator.fit(X_train, y_train, ak_train)

    q_voi_results = {}
    for seed, oid in full_oids:
        obj = per_seed_objects[seed][oid]
        pre_features = obj["pre_surface_features"]
        post_features = obj["post_observe_features"]
        state = obj["visible_state"]
        pre_sf = build_feature_vector(pre_features, {}, 0)
        pre_vec = extract_feature_vector(pre_sf)
        post_sf = build_feature_vector(post_features, state, 1)
        post_vec = extract_feature_vector(post_sf)

        pre_qs = estimator.predict_all_actions(pre_vec)
        post_qs = estimator.predict_all_actions(post_vec)
        pre_try_qs = {ak: q for ak, q in pre_qs.items() if ak != "observe"}
        post_try_qs = {ak: q for ak, q in post_qs.items() if ak != "observe"}
        pre_best_q = max(pre_try_qs.values()) if pre_try_qs else 0.0
        post_best_q = max(post_try_qs.values()) if post_try_qs else 0.0

        q_voi_results[(seed, oid)] = {
            "q_voi_003": post_best_q - pre_best_q - 0.03,
            "q_voi_005": post_best_q - pre_best_q - 0.05,
        }

    # Per-signature audit (TRUE primary)
    print(f"\n  === ORIGINAL DISTRIBUTION AUDIT (TRUE/ORACLE VOI PRIMARY) ===")
    print(f"  {'Signature':<60s} {'n':>4s} {'+true':>6s} {'-true':>6s} {'frac+':>7s} {'entropy':>8s} {'det?':>5s}")
    print(f"  {'-'*60} {'-'*4} {'-'*6} {'-'*6} {'-'*7} {'-'*8} {'-'*5}")

    audit_results = {}
    total_det_true_003 = 0
    total_det_true_005 = 0
    total_det_q_003 = 0

    for sig in SIGNATURE_LIST:
        instances = [(seed, oid) for seed, oid in sig_map[sig]]
        n = len(instances)

        # True/oracle
        n_true_pos_003 = sum(1 for s, o in instances if true_voi_003[(s, o)]["true_optimal"])
        n_true_neg_003 = n - n_true_pos_003
        frac_true_003 = n_true_pos_003 / n if n > 0 else 0.0
        ent_true_003 = binary_entropy(frac_true_003)
        is_det_true_003 = (n_true_pos_003 == 0 or n_true_pos_003 == n)

        n_true_pos_005 = sum(1 for s, o in instances if true_voi_005[(s, o)]["true_optimal"])
        n_true_neg_005 = n - n_true_pos_005
        frac_true_005 = n_true_pos_005 / n if n > 0 else 0.0
        ent_true_005 = binary_entropy(frac_true_005)
        is_det_true_005 = (n_true_pos_005 == 0 or n_true_pos_005 == n)

        # Q-based sanity
        n_q_pos_003 = sum(1 for s, o in instances if q_voi_results[(s, o)]["q_voi_003"] > 0)
        is_det_q_003 = (n_q_pos_003 == 0 or n_q_pos_003 == n)

        sig_short = str(sig)[:58]
        print(f"  {sig_short:<60s} {n:4d} {n_true_pos_003:6d} {n_true_neg_003:6d} "
              f"{frac_true_003:7.4f} {ent_true_003:8.4f} {str(is_det_true_003):>5s}")

        if is_det_true_003:
            total_det_true_003 += 1
        if is_det_true_005:
            total_det_true_005 += 1
        if is_det_q_003:
            total_det_q_003 += 1

        audit_results[sig] = {
            "n_total": n,
            "n_true_positive_003": n_true_pos_003,
            "n_true_nonpositive_003": n_true_neg_003,
            "fraction_true_positive_003": round(frac_true_003, 4),
            "entropy_true_003": round(ent_true_003, 4),
            "is_true_deterministic_003": is_det_true_003,
            "n_true_positive_005": n_true_pos_005,
            "n_true_nonpositive_005": n_true_neg_005,
            "fraction_true_positive_005": round(frac_true_005, 4),
            "entropy_true_005": round(ent_true_005, 4),
            "is_true_deterministic_005": is_det_true_005,
            "n_q_positive_003": n_q_pos_003,
            "is_q_deterministic_003": is_det_q_003,
        }

    n_sigs = len(SIGNATURE_LIST)
    print(f"\n  TRUE/ORACLE (cost=0.03): {total_det_true_003}/{n_sigs} signatures "
          f"are deterministic ({100*total_det_true_003/n_sigs:.1f}%)")
    print(f"  TRUE/ORACLE (cost=0.05): {total_det_true_005}/{n_sigs} signatures "
          f"are deterministic ({100*total_det_true_005/n_sigs:.1f}%)")
    print(f"  Q-based sanity (cost=0.03): {total_det_q_003}/{n_sigs} signatures "
          f"are Q-deterministic ({100*total_det_q_003/n_sigs:.1f}%)")

    # Verify: Q-based and true-based should agree on determinism
    q_true_agree = sum(1 for sig in SIGNATURE_LIST
                       if audit_results[sig]["is_true_deterministic_003"] == audit_results[sig]["is_q_deterministic_003"])
    print(f"  Q/true agreement on determinism: {q_true_agree}/{n_sigs}")

    return audit_results, estimator


audit_results, full_estimator = run_distributional_audit_true_primary()

# =============================================================================
# Part B: Counterfactual Coverage (INTERVENTION, not natural redesign)
# =============================================================================
print("\n[7/12] PART B: Building counterfactual coverage interventions...")

# Ambient group categories for "lookalike" swap
AMBIENT_GROUP_CATEGORIES = {
    "A": ["wood_log", "stone_block"],
    "B": ["apple", "wooden_pickaxe"],
}

def get_flat_affordance(all_success=True):
    """INTERVENTION: Flat affordance where all try actions have the same outcome.
    This makes observation decision-irrelevant (pre_best == post_best).
    """
    outcome = "success" if all_success else "fail"
    return {a: outcome for a in ALL_TRY_AFFORDANCES}

def get_lookalike_affordance(seed, oid, rng):
    """INTERVENTION: Swap to another category's affordance in the same ambient group.
    This creates a lookalike/deceptive/exception mechanism.
    The pre_surface diagnostic feature suggests the wrong action;
    post-observe evidence may correct it.
    The hidden category/profile is NOT exposed to the model.

    Returns (affordance_dict, target_category).
    """
    lbl = per_seed_audit_labels[seed][oid]
    original_cat = lbl["hidden_category"]
    ambient_group = lbl["ambient_group"]
    other_cats = [c for c in AMBIENT_GROUP_CATEGORIES[ambient_group] if c != original_cat]
    if other_cats:
        other_cat = other_cats[0]
        return dict(INSTANCE_SUBTYPE_DEFS[other_cat]["P"]["affordance_profile"]), other_cat
    return get_flat_affordance(True), None


def build_counterfactual_affordance_overrides(train_oids, coverage_fraction):
    """Build affordance overrides for counterfactual coverage.

    This is an EXPLICIT INTERVENTION on the training distribution, NOT a natural
    env3b redesign.

    For each pre_surface signature, flips coverage_fraction of instances:
      - Shared signatures (normally positive_net): flip to flat affordance
        → observation decision-irrelevant (pre_best == post_best, VOI = -cost).
      - Unique signatures (normally non_positive_net): flip to another category's
        affordance in the same ambient group
        → lookalike/deceptive mechanism. Pre-surface diagnostic feature is
        misleading; post-observe evidence may correct.

    Returns (overrides_dict, intervention_labels_dict).
    """
    rng = random.Random(42)

    # Group by signature
    train_by_sig = defaultdict(list)
    for seed, oid in train_oids:
        sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
        train_by_sig[sig].append((seed, oid))

    overrides = {}
    intervention_labels = {}
    lookalike_targets = {}  # (seed, oid) -> target_category for lookalike interventions
    total_flipped = 0
    total_instances = 0

    for sig, instances in sorted(train_by_sig.items()):
        n = len(instances)
        total_instances += n
        n_flip = max(1, int(n * coverage_fraction))
        flipped = rng.sample(instances, n_flip)

        # Determine intervention type from original rationale
        some_lbl = per_seed_audit_labels[instances[0][0]][instances[0][1]]
        is_shared = some_lbl["hidden_audit_rationale_class"] == "positive_change"

        intervention_type = (
            "decision_irrelevant_flat"
            if is_shared else
            "lookalike_deceptive_exception"
        )

        for seed, oid in flipped:
            if is_shared:
                flat_val = "success" if rng.random() < 0.5 else "fail"
                new_aff = get_flat_affordance(flat_val == "success")
            else:
                new_aff, target_cat = get_lookalike_affordance(seed, oid, rng)
                lookalike_targets[(seed, oid)] = target_cat

            overrides[(seed, oid)] = new_aff
            intervention_labels[(seed, oid)] = intervention_type
            total_flipped += 1

    print(f"  Coverage fraction: {coverage_fraction}")
    print(f"  Total training instances: {total_instances}")
    print(f"  Flipped instances: {total_flipped}")

    # Count by intervention type
    type_counts = defaultdict(int)
    for il in intervention_labels.values():
        type_counts[il] += 1
    print(f"  By intervention type:")
    print(f"    decision_irrelevant_flat (shared -> non-positive): {type_counts.get('decision_irrelevant_flat', 0)}")
    print(f"    lookalike_deceptive_exception (unique -> positive): {type_counts.get('lookalike_deceptive_exception', 0)}")

    return overrides, intervention_labels, lookalike_targets


def build_lookalike_post_observe_overrides(lookalike_targets):
    """Build corrected post_observe_features for lookalike interventions.

    For each flipped unique-sig instance, replace the original category's
    diagnostic post_observe features with the TARGET category's diagnostic
    features.  Pre_surface features are kept unchanged (they are the
    'deceptive' pre-surface signal).

    The result: post_observe reveals target-category-consistent evidence,
    making observation genuinely corrective for the flipped instances.
    """
    overrides = {}
    for (seed, oid), target_cat in lookalike_targets.items():
        obj = per_seed_objects[seed][oid]
        # Start with pre_surface features (unchanged — this is the deceptive part)
        corrected_post = dict(obj["pre_surface_features"])
        # Add target category diagnostic features (the corrective evidence)
        for f in CATEGORY_POST_OBSERVE_FEATURES.get(target_cat, []):
            corrected_post[f] = True
        overrides[(seed, oid)] = corrected_post
    return overrides


# =============================================================================
# Part B continued: Audit counterfactual coverage per signature
# =============================================================================
def audit_counterfactual_coverage(train_oids, coverage_fraction, observe_cost):
    """After counterfactual modification, recompute pre-oracle using legal
    pre_signature groups only. Report per-signature positive_net rate and entropy.
    """
    overrides, _, _ = build_counterfactual_affordance_overrides(train_oids, coverage_fraction)

    # Use true/oracle VOI with counterfactual overrides
    true_voi_cf = compute_true_oracle_voi(train_oids, observe_cost, overrides)

    # Group by signature
    sig_instances = defaultdict(list)
    for seed, oid in train_oids:
        sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
        sig_instances[sig].append((seed, oid))

    cf_audit = {}
    deterministic_count = 0
    total_sigs = 0

    print(f"\n  === COUNTERFACTUAL COVERAGE AUDIT (frac={coverage_fraction}, cost={observe_cost}) ===")
    print(f"  {'Signature':<60s} {'n':>4s} {'+true':>6s} {'-true':>6s} {'frac+':>7s} {'entropy':>8s} {'det?':>5s} {'interv':>8s}")
    print(f"  {'-'*60} {'-'*4} {'-'*6} {'-'*6} {'-'*7} {'-'*8} {'-'*5} {'-'*8}")

    for sig in SIGNATURE_LIST:
        instances = sig_instances.get(sig, [])
        if not instances:
            continue
        n = len(instances)
        total_sigs += 1

        n_pos = sum(1 for s, o in instances if true_voi_cf[(s, o)]["true_optimal"])
        n_neg = n - n_pos
        frac_pos = n_pos / n if n > 0 else 0.0
        ent = binary_entropy(frac_pos)
        is_det = (n_pos == 0 or n_pos == n)
        if is_det:
            deterministic_count += 1

        # Intervention types in this signature
        interv_types = set()
        for s, o in instances:
            if (s, o) in overrides:
                interv_types.add("flat" if per_seed_audit_labels[s][o]["hidden_audit_rationale_class"] == "positive_change" else "lookalike")
        interv_str = ",".join(sorted(interv_types)) if interv_types else "none"

        sig_short = str(sig)[:58]
        print(f"  {sig_short:<60s} {n:4d} {n_pos:6d} {n_neg:6d} "
              f"{frac_pos:7.4f} {ent:8.4f} {str(is_det):>5s} {interv_str:>8s}")

        cf_audit[sig] = {
            "n_total": n,
            "n_true_positive": n_pos,
            "n_true_nonpositive": n_neg,
            "fraction_positive": round(frac_pos, 4),
            "entropy": round(ent, 4),
            "is_deterministic": is_det,
        }

    print(f"\n  After counterfactual coverage (frac={coverage_fraction}, cost={observe_cost}): "
          f"{deterministic_count}/{total_sigs} signatures remain deterministic "
          f"({100*deterministic_count/total_sigs:.1f}%)")
    print(f"  Signatures with mixed VOI: {total_sigs - deterministic_count}/{total_sigs}")

    return cf_audit, overrides, deterministic_count, total_sigs


# =============================================================================
# Corrective Post-Evidence Audit for Lookalike Interventions
# =============================================================================
def audit_corrective_post_evidence(train_oids, coverage_fraction):
    """For every unique-signature -> positive (lookalike) counterfactual intervention,
    verify that post_observe_features contain legal corrective evidence consistent
    with the counterfactual affordance profile.

    If a wood_log-like pre-signature is given stone_block-like affordance, then
    post_observe_features should reveal stone-like diagnostic evidence such as
    has_crystal_flecks / block_like / cold_to_touch, not wood_log-only evidence.

    Returns dict with audit results.
    """
    overrides, intervention_labels, lookalike_targets = \
        build_counterfactual_affordance_overrides(train_oids, coverage_fraction)

    # Build corrected post_observe features (target-category diagnostics)
    corrected_post_features = build_lookalike_post_observe_overrides(lookalike_targets)

    # Identify lookalike interventions only
    lookalike_instances = [(seed, oid) for (seed, oid), label in intervention_labels.items()
                           if label == "lookalike_deceptive_exception"]

    n_lookalike = len(lookalike_instances)
    n_with_corrective = 0
    detail = []

    for seed, oid in lookalike_instances:
        target_cat = lookalike_targets.get((seed, oid))
        obj = per_seed_objects[seed][oid]
        original_cat = per_seed_audit_labels[seed][oid]["hidden_category"]
        # Use CORRECTED post_observe features (with target-category diagnostics)
        post_features = corrected_post_features.get((seed, oid), obj["post_observe_features"])

        # Features that are diagnostic of the target category (from CATEGORY_POST_OBSERVE_FEATURES)
        target_diagnostic_features = set(CATEGORY_POST_OBSERVE_FEATURES.get(target_cat, []))
        original_diagnostic_features = set(CATEGORY_POST_OBSERVE_FEATURES.get(original_cat, []))

        # Post_observe features present on this instance
        post_feature_set = set(post_features.keys())

        # Corrective evidence: post_observe contains features diagnostic of the target category
        # that are NOT also diagnostic of the original category
        corrective_features = target_diagnostic_features & post_feature_set
        original_features_present = original_diagnostic_features & post_feature_set

        has_corrective = len(corrective_features) > 0

        if has_corrective:
            n_with_corrective += 1

        detail.append({
            "seed": seed, "oid": oid,
            "original_category": original_cat,
            "target_category": target_cat,
            "target_diagnostic_features": sorted(target_diagnostic_features),
            "original_diagnostic_features": sorted(original_diagnostic_features),
            "post_features_present": sorted(post_feature_set),
            "corrective_features_found": sorted(corrective_features),
            "original_features_present": sorted(original_features_present),
            "has_corrective_evidence": has_corrective,
        })

    pass_rate = n_with_corrective / n_lookalike if n_lookalike > 0 else 1.0
    design_valid = pass_rate > 0.0

    print(f"\n  === CORRECTIVE POST-EVIDENCE AUDIT (coverage={coverage_fraction}) ===")
    print(f"  Unique-signature -> positive (lookalike) counterfactuals: {n_lookalike}")
    print(f"  With corrective post-evidence: {n_with_corrective}")
    print(f"  Without corrective post-evidence: {n_lookalike - n_with_corrective}")
    print(f"  Corrective post-evidence pass rate: {pass_rate:.4f}")
    print(f"  Design valid (pass_rate > 0): {design_valid}")

    if not design_valid:
        print(f"\n  *** WARNING: Counterfactual coverage design is INVALID. ***")
        print(f"  No lookalike instances have corrective post-observe evidence.")
        print(f"  The post_observe_features come from the original object category,")
        print(f"  not the target category. The 'lookalike' mechanism cannot work")
        print(f"  without corrective post-observe evidence.")

    # Show details for first few and last few
    if detail:
        print(f"\n  Detail (first 3):")
        for d in detail[:3]:
            print(f"    {d['seed']}/{d['oid']}: {d['original_category']} -> {d['target_category']}")
            print(f"      Target diagnostic: {d['target_diagnostic_features']}")
            print(f"      Post features:     {d['post_features_present']}")
            print(f"      Corrective found:  {d['corrective_features_found']} -> {d['has_corrective_evidence']}")
        if len(detail) > 6:
            print(f"    ... ({len(detail) - 6} more) ...")
        if len(detail) > 3:
            print(f"  Detail (last 3):")
            for d in detail[-3:]:
                print(f"    {d['seed']}/{d['oid']}: {d['original_category']} -> {d['target_category']}")
                print(f"      Corrective found: {d['corrective_features_found']} -> {d['has_corrective_evidence']}")

    return {
        "n_lookalike_counterfactuals": n_lookalike,
        "n_with_corrective_post_evidence": n_with_corrective,
        "n_without_corrective_post_evidence": n_lookalike - n_with_corrective,
        "corrective_post_evidence_pass_rate": round(pass_rate, 4),
        "design_valid": design_valid,
        "detail": detail,
    }


# =============================================================================
# Part C: Policy Evaluation
# =============================================================================
print("\n[8/12] PART C: Running policy experiments...")

def run_policy_on_split(train_oids, test_oids, observe_cost, affordance_overrides=None,
                         post_observe_overrides=None, data_label="original"):
    """Two-stage D0 policy (raw pre_vec only) on a single stress split.

    Decision-time information boundary: Stage 1 uses only pre_vec.
    post_observe_features are revealed only after observe is chosen.
    """
    train_records = build_training_records_from_oids(
        train_oids, observe_cost, affordance_overrides, post_observe_overrides)
    obj_data = build_obj_eval_vectors(test_oids)
    test_keys = sorted(obj_data.keys())
    n_test = len(test_keys)

    # Train shared Q-function
    X_train = [r["feature_vector"] for r in train_records]
    y_train = [r["net_value"] for r in train_records]
    ak_train = [r["action_key"] for r in train_records]
    estimator = PerActionValueEstimator(alpha=RIDGE_ALPHA)
    estimator.fit(X_train, y_train, ak_train)

    # ---- Baselines ----
    pre_returns = []
    for key in test_keys:
        od = obj_data[key]
        qs = estimator.predict_all_actions(od["pre_vec"])
        try_qs = {ak: q for ak, q in qs.items() if ak != "observe"}
        best_action = max(try_qs, key=try_qs.get)
        pre_returns.append(od["true_returns"].get(best_action, 0.0))
    pre_mean = sum(pre_returns) / n_test if n_test else 0.0

    post_returns_net = []
    for key in test_keys:
        od = obj_data[key]
        qs = estimator.predict_all_actions(od["post_vec"])
        try_qs = {ak: q for ak, q in qs.items() if ak != "observe"}
        best_action = max(try_qs, key=try_qs.get)
        ret = od["true_returns"].get(best_action, 0.0)
        post_returns_net.append(ret - observe_cost)
    post_mean_net = sum(post_returns_net) / n_test if n_test else 0.0

    # Oracle selective: pre_best from legal pre_signature groups of test objects
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
    oracle_sel_mean = sum(oracle_sel_returns) / n_test if n_test else 0.0

    # ---- Train observe_value_model (D0: raw pre_vec only) ----
    ov_X = []
    ov_y = []
    ov_y_true = []

    # Pre-bests from training signature groups (legal: pre_surface_signature only)
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

    for tseed, toid in train_oids:
        tobj = per_seed_objects[tseed][toid]
        tlbl = per_seed_audit_labels[tseed][toid]
        pre_features = tobj["pre_surface_features"]
        if post_observe_overrides and (tseed, toid) in post_observe_overrides:
            post_features = post_observe_overrides[(tseed, toid)]
        else:
            post_features = tobj["post_observe_features"]
        state = tobj["visible_state"]

        pre_sf = build_feature_vector(pre_features, {}, 0)
        pre_vec = extract_feature_vector(pre_sf)
        post_sf = build_feature_vector(post_features, state, 1)
        post_vec = extract_feature_vector(post_sf)

        pre_qs = estimator.predict_all_actions(pre_vec)
        post_qs = estimator.predict_all_actions(post_vec)
        pre_try_qs = {ak: q for ak, q in pre_qs.items() if ak != "observe"}
        post_try_qs = {ak: q for ak, q in post_qs.items() if ak != "observe"}
        pre_best = max(pre_try_qs.values()) if pre_try_qs else 0.0
        post_best = max(post_try_qs.values()) if post_try_qs else 0.0

        ov_y.append(post_best - pre_best - observe_cost)
        ov_X.append(list(pre_vec))

        # True target using signature-group pre_best
        tsig = get_pre_surface_signature(pre_features)
        pre_best_true = train_sig_exp_ov.get(tsig, 0.0)
        if affordance_overrides and (tseed, toid) in affordance_overrides:
            taff = affordance_overrides[(tseed, toid)]
        else:
            taff = tlbl["affordance_profile"]
        true_rets_obj = {}
        for action in ALL_TRY_AFFORDANCES:
            outcome = taff.get(action, "fail")
            true_rets_obj[action] = try_net_value(outcome == "success")
        post_best_true = max(true_rets_obj.values())
        ov_y_true.append(post_best_true - pre_best_true - observe_cost)

    # Train observe_value_model
    ov_model = None
    ov_normalizer = None
    ov_true_target_corr = None
    ov_q_target_mean = sum(ov_y) / len(ov_y) if ov_y else 0.0
    ov_true_target_mean = sum(ov_y_true) / len(ov_y_true) if ov_y_true else 0.0

    if len(ov_X) >= 3 and len(set(ov_y)) > 1:
        ov_normalizer = FeatureNormalizer().fit(ov_X)
        ov_X_norm = ov_normalizer.transform(ov_X)
        ov_model = RidgeRegression(alpha=RIDGE_ALPHA).fit(ov_X_norm, ov_y)
        if len(ov_y_true) == len(ov_y) and len(ov_y) > 0:
            mean_q = sum(ov_y) / len(ov_y)
            mean_t = sum(ov_y_true) / len(ov_y_true)
            num = sum((ov_y[i] - mean_q) * (ov_y_true[i] - mean_t) for i in range(len(ov_y)))
            den_q = math.sqrt(sum((v - mean_q)**2 for v in ov_y))
            den_t = math.sqrt(sum((v - mean_t)**2 for v in ov_y_true))
            if den_q > 1e-12 and den_t > 1e-12:
                ov_true_target_corr = round(num / (den_q * den_t), 4)

    # ---- Run policy on test objects ----
    policy_returns_net = []
    policy_returns_gross = []
    policy_decisions = []
    policy_obs_count = 0

    for key in test_keys:
        od = obj_data[key]
        pre_vec = od["pre_vec"]
        true_returns = od["true_returns"]

        pre_qs = estimator.predict_all_actions(pre_vec)
        pre_try_qs = {ak: q for ak, q in pre_qs.items() if ak != "observe"}
        pre_best_action = max(pre_try_qs, key=pre_try_qs.get)
        pre_best_value = pre_try_qs[pre_best_action]

        if ov_model is not None and ov_normalizer is not None:
            ov_input_norm = ov_normalizer.transform([pre_vec])[0]
            predicted_observe_net = ov_model.predict([ov_input_norm])[0]
        else:
            predicted_observe_net = -observe_cost

        decided_observe = predicted_observe_net > 0

        if decided_observe:
            policy_obs_count += 1
            post_vec = od["post_vec"]
            post_qs = estimator.predict_all_actions(post_vec)
            post_try_qs = {ak: q for ak, q in post_qs.items() if ak != "observe"}
            post_best_action = max(post_try_qs, key=post_try_qs.get)
            post_best_value = post_try_qs[post_best_action]
            chosen_action = post_best_action
            true_ret_gross = true_returns.get(chosen_action, 0.0)
            true_ret_net = true_ret_gross - observe_cost
        else:
            chosen_action = pre_best_action
            true_ret_gross = true_returns.get(chosen_action, 0.0)
            true_ret_net = true_ret_gross
            post_best_action = None
            post_best_value = 0.0

        policy_returns_gross.append(true_ret_gross)
        policy_returns_net.append(true_ret_net)
        policy_decisions.append({
            "seed": od["seed"], "oid": od["oid"],
            "category": od["category"], "subtype": od["subtype"],
            "rationale_class": od["rationale_class"],
            "signature": od["signature"],
            "decided_observe": decided_observe,
            "predicted_observe_net": predicted_observe_net,
            "pre_best_action": pre_best_action,
            "chosen_action": chosen_action,
            "true_return_gross": true_ret_gross,
            "true_return_net": true_ret_net,
            "n_pre_surface_features": od["n_pre_surface_features"],
        })

    policy_mean_net = sum(policy_returns_net) / n_test if n_test else 0.0
    policy_obs_rate = policy_obs_count / n_test if n_test else 0.0

    # Per-rationale breakdown
    pos_dec = [d for d in policy_decisions if d["rationale_class"] == "positive_change"]
    nonpos_dec = [d for d in policy_decisions if d["rationale_class"] != "positive_change"]
    pos_obs_rate = (sum(1 for d in pos_dec if d["decided_observe"]) /
                    len(pos_dec) if pos_dec else 0.0)
    nonpos_obs_rate = (sum(1 for d in nonpos_dec if d["decided_observe"]) /
                       len(nonpos_dec) if nonpos_dec else 0.0)

    return {
        "data_label": data_label,
        "n_train": len(train_oids),
        "n_test": n_test,
        "pre_mean": round(pre_mean, 4),
        "always_try": round(pre_mean, 4),
        "always_observe_net": round(post_mean_net, 4),
        "oracle_selective": round(oracle_sel_mean, 4),
        "policy_mean_net": round(policy_mean_net, 4),
        "policy_obs_rate": round(policy_obs_rate, 4),
        "pos_obs_rate": round(pos_obs_rate, 4),
        "nonpos_obs_rate": round(nonpos_obs_rate, 4),
        "learned_minus_always_observe": round(policy_mean_net - post_mean_net, 4),
        "learned_minus_always_try": round(policy_mean_net - pre_mean, 4),
        "learned_minus_oracle": round(oracle_sel_mean - policy_mean_net, 4),
        "ov_q_target_mean": round(ov_q_target_mean, 4),
        "ov_true_target_mean": round(ov_true_target_mean, 4),
        "ov_true_target_corr": ov_true_target_corr,
        "policy_decisions": policy_decisions,
    }


def compute_acceptance(result):
    """Four acceptance checks for a single result."""
    checks = {}
    checks["policy_net > always_observe_net"] = (
        result["policy_mean_net"] > result["always_observe_net"])
    checks["policy_net > always_try"] = (
        result["policy_mean_net"] > result["always_try"])
    obs_rate = result["policy_obs_rate"]
    checks["obs_rate between 0 and 1"] = (0 < obs_rate < 1)
    checks["pos_obs > nonpos_obs"] = (result["pos_obs_rate"] > result["nonpos_obs_rate"])
    failures = [k for k, v in checks.items() if not v]
    return {"checks": checks, "failures": failures, "passed": len(failures) == 0}


def aggregate_group(group_results):
    """Aggregate across variants in a test group."""
    total_n = 0
    total_obs = 0
    total_pos = 0
    total_pos_obs = 0
    total_nonpos = 0
    total_nonpos_obs = 0
    weighted_net = 0.0
    weighted_always_obs = 0.0
    weighted_always_try = 0.0
    weighted_oracle = 0.0

    for r in group_results:
        n = r["n_test"]
        weighted_net += r["policy_mean_net"] * n
        weighted_always_obs += r["always_observe_net"] * n
        weighted_always_try += r["always_try"] * n
        weighted_oracle += r["oracle_selective"] * n
        total_n += n
        total_obs += int(r["policy_obs_rate"] * n)
        for d in r["policy_decisions"]:
            if d["rationale_class"] == "positive_change":
                total_pos += 1
                if d["decided_observe"]:
                    total_pos_obs += 1
            else:
                total_nonpos += 1
                if d["decided_observe"]:
                    total_nonpos_obs += 1

    agg_net = weighted_net / total_n if total_n > 0 else 0.0
    agg_always_obs = weighted_always_obs / total_n if total_n > 0 else 0.0
    agg_always_try = weighted_always_try / total_n if total_n > 0 else 0.0
    agg_oracle = weighted_oracle / total_n if total_n > 0 else 0.0

    return {
        "n_test": total_n,
        "policy_net": round(agg_net, 4),
        "always_obs_net": round(agg_always_obs, 4),
        "always_try": round(agg_always_try, 4),
        "oracle_sel": round(agg_oracle, 4),
        "obs_rate": round(total_obs / total_n, 4) if total_n > 0 else 0.0,
        "pos_obs_rate": round(total_pos_obs / total_pos, 4) if total_pos > 0 else 0.0,
        "nonpos_obs_rate": round(total_nonpos_obs / total_nonpos, 4) if total_nonpos > 0 else 0.0,
        "vs_always_obs": round(agg_net - agg_always_obs, 4),
        "vs_oracle": round(agg_oracle - agg_net, 4),
    }


# Run all experiments
print("\n[9/12] Running all policy experiments (original + 2 coverage fractions x 2 costs)...")

ALL_EXPERIMENTS = {}  # keyed by (cost, coverage_fraction_or_"original")

for observe_cost, cost_label in [(0.03, "primary"), (0.05, "robustness")]:
    print(f"\n{'='*70}")
    print(f"EXPERIMENTS at observe_cost = {observe_cost} ({cost_label})")
    print(f"{'='*70}")

    # Original (no counterfactual modifications)
    print(f"\n  --- ORIGINAL TRAINING DISTRIBUTION ---")
    orig_results = {"observe_cost": observe_cost, "groups": {}}
    for group_name, test_variants in [("A", TEST_A_VARIANTS), ("B", TEST_B_VARIANTS), ("C", TEST_C_VARIANTS)]:
        group_results = []
        for tv in test_variants:
            r = run_policy_on_split(tv["train_oids"], tv["test_oids"], observe_cost,
                                     affordance_overrides=None, data_label="original")
            r["test_variant_id"] = tv["id"]
            r["test_group"] = group_name
            checks = compute_acceptance(r)
            r["status"] = "PASS" if checks["passed"] else "FAIL"
            r["failures"] = checks["failures"]
            group_results.append(r)
            print(f"    {group_name}/{tv['id']}: net={r['policy_mean_net']:.4f} "
                  f"obs_rate={r['policy_obs_rate']:.4f} "
                  f"vs_always_obs={r['learned_minus_always_observe']:+.4f} "
                  f"status={r['status']}")
        orig_results["groups"][group_name] = group_results

    ALL_EXPERIMENTS[(observe_cost, "original")] = orig_results

    # Counterfactual coverage variants
    for cov_frac in COVERAGE_FRACTIONS:
        print(f"\n  --- COUNTERFACTUAL TRAINING DISTRIBUTION (coverage={cov_frac}) ---")

        cf_results = {"observe_cost": observe_cost, "coverage_fraction": cov_frac, "groups": {}}
        for group_name, test_variants in [("A", TEST_A_VARIANTS), ("B", TEST_B_VARIANTS), ("C", TEST_C_VARIANTS)]:
            group_results = []
            for tv in test_variants:
                cf_overrides, _, lookalike_targets = build_counterfactual_affordance_overrides(
                    tv["train_oids"], cov_frac)
                post_observe_overrides = build_lookalike_post_observe_overrides(
                    lookalike_targets)

                r = run_policy_on_split(tv["train_oids"], tv["test_oids"], observe_cost,
                                         affordance_overrides=cf_overrides,
                                         post_observe_overrides=post_observe_overrides,
                                         data_label=f"counterfactual_cov{cov_frac}")
                r["test_variant_id"] = tv["id"]
                r["test_group"] = group_name
                r["coverage_fraction"] = cov_frac
                r["n_flipped"] = len(cf_overrides)
                checks = compute_acceptance(r)
                r["status"] = "PASS" if checks["passed"] else "FAIL"
                r["failures"] = checks["failures"]
                group_results.append(r)
                print(f"    {group_name}/{tv['id']}: net={r['policy_mean_net']:.4f} "
                      f"obs_rate={r['policy_obs_rate']:.4f} "
                      f"vs_always_obs={r['learned_minus_always_observe']:+.4f} "
                      f"status={r['status']} "
                      f"flipped={r['n_flipped']}")
            cf_results["groups"][group_name] = group_results

        ALL_EXPERIMENTS[(observe_cost, cov_frac)] = cf_results


# =============================================================================
# Counterfactual coverage audit (per-signature stats after modification)
# =============================================================================
print("\n[10/12] Auditing counterfactual coverage per signature (true/oracle VOI)...")

full_train_oids = set()
for seed in SEEDS:
    for oid in per_seed_objects[seed]:
        full_train_oids.add((seed, oid))

CF_AUDITS = {}  # keyed by (cost, coverage_fraction)

for observe_cost in [0.03, 0.05]:
    for cov_frac in COVERAGE_FRACTIONS:
        cf_audit, cf_overrides, det_count, total_sigs = audit_counterfactual_coverage(
            full_train_oids, cov_frac, observe_cost)
        CF_AUDITS[(observe_cost, cov_frac)] = {
            "audit": cf_audit,
            "deterministic_count": det_count,
            "total_signatures": total_sigs,
            "fraction_deterministic": det_count / total_sigs if total_sigs > 0 else 1.0,
        }

# =============================================================================
# Corrective Post-Evidence Audit (lookalike interventions only)
# =============================================================================
print("\n[10b/12] Auditing corrective post-evidence for lookalike interventions...")

CORRECTIVE_AUDITS = {}  # keyed by coverage_fraction (same across costs)

for cov_frac in COVERAGE_FRACTIONS:
    corrective_result = audit_corrective_post_evidence(full_train_oids, cov_frac)
    CORRECTIVE_AUDITS[cov_frac] = corrective_result


# =============================================================================
# Three-Part Structured Reporting
# =============================================================================
print("\n[11/12] Generating structured reports...")

def print_section_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

# --- Question A: Does counterfactual coverage break signature determinism? ---
print_section_header("QUESTION A: Does counterfactual training coverage "
                      "break signature determinism?")

print("\n  Original distribution (TRUE/ORACLE VOI, cost=0.03):")
n_det_orig = sum(1 for sig in SIGNATURE_LIST if audit_results[sig]["is_true_deterministic_003"])
print(f"    {n_det_orig}/{len(SIGNATURE_LIST)} signatures are deterministic "
      f"(entropy=0, all instances have same optimal observe decision)")
print(f"    All shared signatures: fraction_positive=1.0, entropy=0.0")
print(f"    All unique signatures: fraction_positive=0.0, entropy=0.0")
print(f"    Signature lookup is OPTIMAL in the original training distribution.")

for cov_frac in COVERAGE_FRACTIONS:
    for observe_cost in [0.03, 0.05]:
        ca = CF_AUDITS[(observe_cost, cov_frac)]
        print(f"\n  Counterfactual coverage fraction={cov_frac}, cost={observe_cost}:")
        print(f"    {ca['deterministic_count']}/{ca['total_signatures']} signatures "
              f"remain deterministic ({100*ca['fraction_deterministic']:.1f}%)")
        print(f"    {ca['total_signatures'] - ca['deterministic_count']}/{ca['total_signatures']} "
              f"signatures now have mixed VOI (entropy > 0)")

        # Show which signatures broke
        broken = [sig for sig in SIGNATURE_LIST
                  if sig in ca["audit"] and not ca["audit"][sig]["is_deterministic"]]
        if broken:
            print(f"    Signatures with mixed VOI:")
            for sig in broken:
                a = ca["audit"][sig]
                print(f"      {str(sig)[:50]}: frac+={a['fraction_positive']:.4f} "
                      f"entropy={a['entropy']:.4f}")

# --- Question B: Does the learned policy improve on held-out Test A/B? ---
print_section_header("QUESTION B: Does the learned policy improve on "
                      "original held-out Test A/B?")

for observe_cost, cost_label in [(0.03, "primary"), (0.05, "robustness")]:
    print(f"\n  --- {cost_label} (cost={observe_cost}) ---")

    orig_exp = ALL_EXPERIMENTS[(observe_cost, "original")]
    print(f"\n  {'Test':<6s} {'Condition':<20s} {'Net':>8s} {'ObsRate':>8s} "
          f"{'PosObs':>8s} {'NonposObs':>10s} {'vsAlwaysObs':>11s} {'Status':>6s}")
    print(f"  {'-'*6} {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*11} {'-'*6}")

    for group_name in ["A", "B", "C"]:
        gr = orig_exp["groups"].get(group_name, [])
        if not gr:
            continue

        # Original
        agg_orig = aggregate_group(gr)
        orig_status = "PASS" if (agg_orig["vs_always_obs"] > 0 and
                                  0 < agg_orig["obs_rate"] < 1 and
                                  agg_orig["pos_obs_rate"] > agg_orig["nonpos_obs_rate"]) else "FAIL"
        print(f"  {group_name:<6s} {'original':<20s} {agg_orig['policy_net']:8.4f} "
              f"{agg_orig['obs_rate']:8.4f} {agg_orig['pos_obs_rate']:8.4f} "
              f"{agg_orig['nonpos_obs_rate']:10.4f} {agg_orig['vs_always_obs']:+11.4f} {orig_status:>6s}")

        # Counterfactual variants
        for cov_frac in COVERAGE_FRACTIONS:
            cf_exp = ALL_EXPERIMENTS[(observe_cost, cov_frac)]
            cf_gr = cf_exp["groups"].get(group_name, [])
            if not cf_gr:
                continue
            agg_cf = aggregate_group(cf_gr)

            # Does counterfactual IMPROVE over original?
            delta_net = agg_cf["policy_net"] - agg_orig["policy_net"]
            delta_obs = agg_cf["obs_rate"] - agg_orig["obs_rate"]

            # Collapse check: if obs_rate is still 0 or 1, it collapsed
            collapsed = (agg_cf["obs_rate"] <= 0.0 or agg_cf["obs_rate"] >= 1.0)
            always_obs_collapse = (agg_cf["obs_rate"] >= 1.0)

            cf_status = "PASS" if (agg_cf["vs_always_obs"] > 0 and
                                    0 < agg_cf["obs_rate"] < 1 and
                                    agg_cf["pos_obs_rate"] > agg_cf["nonpos_obs_rate"]) else "FAIL"

            condition_label = f"cf_cov{cov_frac}"
            print(f"  {group_name:<6s} {condition_label:<20s} {agg_cf['policy_net']:8.4f} "
                  f"{agg_cf['obs_rate']:8.4f} {agg_cf['pos_obs_rate']:8.4f} "
                  f"{agg_cf['nonpos_obs_rate']:10.4f} {agg_cf['vs_always_obs']:+11.4f} {cf_status:>6s}")

            # Key question: did CF help on Test A/B?
            if group_name in ("A", "B"):
                if collapsed:
                    print(f"           -> Collapsed to always-observe (obs_rate={agg_cf['obs_rate']:.4f}). "
                          f"Mixed coverage did NOT prevent collapse.")
                elif agg_cf["vs_always_obs"] > 0:
                    print(f"           -> IMPROVED: beats always-observe by {agg_cf['vs_always_obs']:+.4f}")
                else:
                    print(f"           -> Did NOT improve over always-observe "
                          f"(delta_net={delta_net:+.4f}, delta_obs={delta_obs:+.4f})")

# --- Question C: Does signature_lookup_policy remain strong? ---
print_section_header("QUESTION C: Does signature_lookup_policy remain strong?")

print("\n  A 'signature_lookup_policy' is one where the observe_value_model")
print("  predictions are perfectly predictable from signature identity alone")
print("  (i.e., all instances of a signature get the same observe decision).")

for observe_cost, cost_label in [(0.03, "primary"), (0.05, "robustness")]:
    print(f"\n  --- {cost_label} (cost={observe_cost}) ---")

    for cov_frac in [0.0] + COVERAGE_FRACTIONS:  # 0.0 = original
        if cov_frac == 0.0:
            exp_data = ALL_EXPERIMENTS[(observe_cost, "original")]
            label = "original"
        else:
            exp_data = ALL_EXPERIMENTS[(observe_cost, cov_frac)]
            label = f"counterfactual_cov{cov_frac}"

        # Check per-signature observe rate consistency across all test splits
        all_sig_obs = defaultdict(lambda: {"obs": 0, "total": 0})
        for group_name in ["A", "B", "C"]:
            gr = exp_data["groups"].get(group_name, [])
            for entry in gr:
                for d in entry["policy_decisions"]:
                    sig_key = str(d["signature"])
                    all_sig_obs[sig_key]["total"] += 1
                    if d["decided_observe"]:
                        all_sig_obs[sig_key]["obs"] += 1

        # Count signatures where all instances get same decision
        sig_lookup_sigs = 0
        total_sigs_seen = 0
        for sig_key, data in all_sig_obs.items():
            total_sigs_seen += 1
            obs_rate_sig = data["obs"] / data["total"] if data["total"] > 0 else 0.0
            if obs_rate_sig == 0.0 or obs_rate_sig == 1.0:
                sig_lookup_sigs += 1

        print(f"    {label}:")
        print(f"      Signatures with uniform observe decision: "
              f"{sig_lookup_sigs}/{total_sigs_seen} "
              f"({100*sig_lookup_sigs/total_sigs_seen:.1f}%)" if total_sigs_seen > 0 else "      N/A")

        # Overall policy behavior
        for group_name in ["A", "B", "C"]:
            gr = exp_data["groups"].get(group_name, [])
            if not gr:
                continue
            agg = aggregate_group(gr)
            collapsed = agg["obs_rate"] <= 0.0 or agg["obs_rate"] >= 1.0
            collapse_type = ""
            if agg["obs_rate"] >= 1.0:
                collapse_type = "always-observe"
            elif agg["obs_rate"] <= 0.0:
                collapse_type = "always-try"
            print(f"      Test {group_name}: obs_rate={agg['obs_rate']:.4f} "
                  + (f"[COLLAPSED: {collapse_type}]" if collapsed else "[SELECTIVE]"))


# =============================================================================
# Overall Assessment
# =============================================================================
print_section_header("OVERALL ASSESSMENT")

# Key criterion (constraint 8): don't claim success if mixed coverage exists
# but policy still collapses to always-observe.
overall_verdicts = {}

for observe_cost, cost_label in [(0.03, "primary"), (0.05, "robustness")]:
    print(f"\n  --- {cost_label} (cost={observe_cost}) ---")

    for cov_frac in COVERAGE_FRACTIONS:
        cf_exp = ALL_EXPERIMENTS[(observe_cost, cov_frac)]
        orig_exp = ALL_EXPERIMENTS[(observe_cost, "original")]

        # Check (1): did coverage break signature determinism?
        ca = CF_AUDITS[(observe_cost, cov_frac)]
        sigs_broken = ca["total_signatures"] - ca["deterministic_count"]
        determinism_broken = sigs_broken > 0

        # Check (2): does policy improve on Test A?
        agg_a_cf = aggregate_group(cf_exp["groups"]["A"])
        agg_a_orig = aggregate_group(orig_exp["groups"]["A"])
        a_collapsed = agg_a_cf["obs_rate"] <= 0.0 or agg_a_cf["obs_rate"] >= 1.0
        a_beats_always_obs = agg_a_cf["vs_always_obs"] > 0

        # Check (3): does policy improve on Test B?
        agg_b_cf = aggregate_group(cf_exp["groups"]["B"])
        agg_b_orig = aggregate_group(orig_exp["groups"]["B"])
        b_collapsed = agg_b_cf["obs_rate"] <= 0.0 or agg_b_cf["obs_rate"] >= 1.0
        b_beats_always_obs = agg_b_cf["vs_always_obs"] > 0

        # Check (4): Test C still passes?
        agg_c_cf = aggregate_group(cf_exp["groups"]["C"])
        c_passes = (agg_c_cf["vs_always_obs"] > 0 and
                     0 < agg_c_cf["obs_rate"] < 1 and
                     agg_c_cf["pos_obs_rate"] > agg_c_cf["nonpos_obs_rate"])

        # Corrective evidence check
        corrective_valid = CORRECTIVE_AUDITS.get(cov_frac, {}).get("design_valid", True)
        corrective_pass_rate = CORRECTIVE_AUDITS.get(cov_frac, {}).get(
            "corrective_post_evidence_pass_rate", 0.0)

        # Verdict: SUCCESS only if coverage breaks determinism AND policy avoids collapse on A/B
        # AND corrective post-evidence is present (for lookalike mechanism validity)
        if not corrective_valid:
            verdict = "DESIGN_INVALID: Lookalike interventions lack corrective post-observe evidence"
        elif determinism_broken and not a_collapsed and a_beats_always_obs:
            verdict = "SUCCESS: Mixed coverage + policy avoids collapse + beats always-observe on Test A"
        elif determinism_broken and a_collapsed:
            verdict = "FAIL: Mixed coverage exists but policy still collapses to always-observe on Test A"
        elif not determinism_broken:
            verdict = "FAIL: Coverage did not break signature determinism"
        else:
            verdict = f"PARTIAL: determinism_broken={determinism_broken}, a_collapsed={a_collapsed}, a_beats={a_beats_always_obs}"

        print(f"    cov={cov_frac}: sigs_broken={sigs_broken}/{ca['total_signatures']}, "
              f"TestA_collapsed={a_collapsed}, TestA_beats_always_obs={a_beats_always_obs}, "
              f"TestB_collapsed={b_collapsed}, TestC_passes={c_passes}, "
              f"corrective_evidence_pass={corrective_pass_rate:.4f}")
        print(f"      Verdict: {verdict}")

        overall_verdicts[(observe_cost, cov_frac)] = {
            "verdict": verdict,
            "sigs_broken": sigs_broken,
            "total_sigs": ca["total_signatures"],
            "determinism_broken": determinism_broken,
            "test_a_collapsed": a_collapsed,
            "test_a_beats_always_obs": a_beats_always_obs,
            "test_b_collapsed": b_collapsed,
            "test_b_beats_always_obs": b_beats_always_obs,
            "test_c_passes": c_passes,
            "corrective_post_evidence_pass_rate": corrective_pass_rate,
            "design_valid": corrective_valid,
        }


# =============================================================================
# Write Output Files
# =============================================================================
print("\n[12/12] Writing output files...")

elapsed = time.time() - t0

# Determine implementation_status
# Check if any corrective audit found the design invalid
any_design_invalid = any(
    not cr["design_valid"] for cr in CORRECTIVE_AUDITS.values())
if any_design_invalid:
    overall_status = "COMPLETED_DESIGN_INVALID"
    print("\n  *** OVERALL: Counterfactual design marked INVALID due to absent corrective post-evidence ***")
else:
    overall_status = "COMPLETED"

output = {
    "block_id": "1J40b-7e",
    "elapsed_seconds": round(elapsed, 1),
    "implementation_status": overall_status,
    "coverage_fractions": COVERAGE_FRACTIONS,
    "seeds": SEEDS,
    "distributional_audit": {
        "method": "TRUE/ORACLE VOI primary, Q-based sanity check",
        "pre_oracle_method": "legal pre_signature groups only (not hidden category)",
        "cost_003": {
            "n_deterministic_true": sum(1 for sig in SIGNATURE_LIST
                                       if audit_results[sig]["is_true_deterministic_003"]),
            "n_total": len(SIGNATURE_LIST),
            "fraction_deterministic": sum(1 for sig in SIGNATURE_LIST
                                         if audit_results[sig]["is_true_deterministic_003"]) / len(SIGNATURE_LIST),
            "per_signature": {str(sig): {
                "n_total": ar["n_total"],
                "n_true_positive": ar["n_true_positive_003"],
                "n_true_nonpositive": ar["n_true_nonpositive_003"],
                "fraction_true_positive": ar["fraction_true_positive_003"],
                "entropy": ar["entropy_true_003"],
                "is_deterministic": ar["is_true_deterministic_003"],
                "n_q_positive": ar["n_q_positive_003"],
                "is_q_deterministic": ar["is_q_deterministic_003"],
            } for sig, ar in audit_results.items()},
        },
        "cost_005": {
            "n_deterministic_true": sum(1 for sig in SIGNATURE_LIST
                                       if audit_results[sig]["is_true_deterministic_005"]),
            "per_signature": {str(sig): {
                "n_true_positive": ar["n_true_positive_005"],
                "fraction_true_positive": ar["fraction_true_positive_005"],
                "entropy": ar["entropy_true_005"],
                "is_deterministic": ar["is_true_deterministic_005"],
            } for sig, ar in audit_results.items()},
        },
    },
    "counterfactual_coverage": {
        "label": "EXPLICIT INTERVENTION on training distribution, NOT natural env3b redesign",
        "shared_signature_intervention": {
            "mechanism": "Flat affordance (all try actions have same outcome)",
            "effect": "Observation decision-irrelevant (pre_best == post_best, VOI = -cost)",
            "label": "decision_irrelevant_flat",
        },
        "unique_signature_intervention": {
            "mechanism": "Swap to other category's affordance in same ambient group",
            "effect": "Lookalike/deceptive/exception: pre-surface diagnostic feature misleading, post-observe corrective",
            "label": "lookalike_deceptive_exception",
            "note": "Hidden category/profile/audit labels NOT exposed to model",
        },
        "coverage_audits": {},
        "corrective_post_evidence_audit": {},
    },
    "policy_experiments": {},
    "question_reports": {},
    "overall_verdicts": {},
}

# Add counterfactual audits
for (cost, cov_frac), ca_data in CF_AUDITS.items():
    key = f"cost_{cost}_cov_{cov_frac}"
    output["counterfactual_coverage"]["coverage_audits"][key] = {
        "coverage_fraction": cov_frac,
        "observe_cost": cost,
        "deterministic_count": ca_data["deterministic_count"],
        "total_signatures": ca_data["total_signatures"],
        "fraction_deterministic": ca_data["fraction_deterministic"],
        "per_signature": {str(sig): {
            "n_total": a["n_total"],
            "n_true_positive": a["n_true_positive"],
            "fraction_positive": a["fraction_positive"],
            "entropy": a["entropy"],
            "is_deterministic": a["is_deterministic"],
        } for sig, a in ca_data["audit"].items()},
    }

# Add corrective post-evidence audit
for cov_frac, cr in CORRECTIVE_AUDITS.items():
    output["counterfactual_coverage"]["corrective_post_evidence_audit"][f"cov_{cov_frac}"] = {
        "n_lookalike_counterfactuals": cr["n_lookalike_counterfactuals"],
        "n_with_corrective_post_evidence": cr["n_with_corrective_post_evidence"],
        "n_without_corrective_post_evidence": cr["n_without_corrective_post_evidence"],
        "corrective_post_evidence_pass_rate": cr["corrective_post_evidence_pass_rate"],
        "design_valid": cr["design_valid"],
        "detail": cr["detail"],
    }

# Add policy experiment summaries
for (cost, key), exp_data in ALL_EXPERIMENTS.items():
    exp_key = f"cost_{cost}_{key}"
    summary = {}
    for group_name in ["A", "B", "C"]:
        gr = exp_data["groups"].get(group_name, [])
        if gr:
            agg = aggregate_group(gr)
            summary[group_name] = agg
    output["policy_experiments"][exp_key] = summary

# Add verdicts
for (cost, cov_frac), v in overall_verdicts.items():
    output["overall_verdicts"][f"cost_{cost}_cov_{cov_frac}"] = v

# Write JSON
json_path = os.path.join(CURRENT_DIR, "runs",
    "block1j40b7e_env3b_distributional_audit_counterfactual.json")
with open(json_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"JSON -> {json_path}")

# Write CSV
csv_path = os.path.join(CURRENT_DIR, "protocols",
    "block1j40b7e_env3b_distributional_audit_counterfactual_table.csv")
csv_lines = ["test_group,test_variant,condition,policy_net,always_obs,oracle_sel,"
             "obs_rate,pos_obs,nonpos_obs,vs_always_obs,vs_oracle,status,failures"]
for (cost, key), exp_data in sorted(ALL_EXPERIMENTS.items(),
                                      key=lambda kv: (kv[0][0], str(kv[0][1]))):
    for group_name in ["A", "B", "C"]:
        gr = exp_data["groups"].get(group_name, [])
        for entry in gr:
            csv_lines.append(
                f"{group_name},{entry['test_variant_id']},{key}_cost{cost},"
                f"{entry['policy_mean_net']:.4f},{entry['always_observe_net']:.4f},"
                f"{entry['oracle_selective']:.4f},{entry['policy_obs_rate']:.4f},"
                f"{entry['pos_obs_rate']:.4f},{entry['nonpos_obs_rate']:.4f},"
                f"{entry['learned_minus_always_observe']:.4f},"
                f"{entry['learned_minus_oracle']:.4f},"
                f"{entry['status']},\"{entry.get('failures', [])}\"")
with open(csv_path, "w") as f:
    f.write("\n".join(csv_lines) + "\n")
print(f"CSV -> {csv_path}")

# Write MD protocol
md_path = os.path.join(CURRENT_DIR, "protocols",
    "block1j40b7e_env3b_distributional_audit_counterfactual.md")
md_lines = []
md_lines.append("# Block 1J40b-7e: env3b Distributional Audit + Minimal Counterfactual Coverage")
md_lines.append("")
md_lines.append(f"- **Implementation Status**: {overall_status}")
md_lines.append(f"- **Elapsed**: {elapsed:.1f}s")
md_lines.append(f"- **Seeds**: {SEEDS}")
md_lines.append(f"- **Coverage Fractions**: {COVERAGE_FRACTIONS}")
md_lines.append("")

md_lines.append("## 1. Files")
md_lines.append("")
md_lines.append(f"- `_block1j40b7e_env3b_distributional_audit_counterfactual.py` -- Written")
md_lines.append(f"- `runs/block1j40b7e_env3b_distributional_audit_counterfactual.json` -- Generated")
md_lines.append(f"- `protocols/block1j40b7e_env3b_distributional_audit_counterfactual.md` -- Generated")
md_lines.append(f"- `protocols/block1j40b7e_env3b_distributional_audit_counterfactual_table.csv` -- Generated")
md_lines.append("")

# Section 2: Distributional Audit
md_lines.append("## 2. Distributional Audit (TRUE/ORACLE VOI Primary)")
md_lines.append("")
md_lines.append("**Method**: Pre-oracle computed from legal pre_signature groups only "
                "(NOT hidden category or diagnostic feature identity). "
                "Q-based targets reported as sanity check only.")
md_lines.append("")
md_lines.append("### Original Distribution (cost=0.03, TRUE/ORACLE)")
md_lines.append("")
md_lines.append("| Signature | n | +true | -true | frac+ | entropy | Deterministic? |")
md_lines.append("|-----------|----|-------|-------|-------|---------|----------------|")
for sig in SIGNATURE_LIST:
    ar = audit_results[sig]
    md_lines.append(f"| {str(sig)[:50]} | {ar['n_total']} | {ar['n_true_positive_003']} | "
                    f"{ar['n_true_nonpositive_003']} | {ar['fraction_true_positive_003']:.4f} | "
                    f"{ar['entropy_true_003']:.4f} | {ar['is_true_deterministic_003']} |")

n_det_true = sum(1 for sig in SIGNATURE_LIST if audit_results[sig]["is_true_deterministic_003"])
md_lines.append("")
md_lines.append(f"**Result**: {n_det_true}/{len(SIGNATURE_LIST)} signatures are TRUE-deterministic "
                f"at cost=0.03. Every training signature perfectly predicts the optimal observe "
                f"decision. Signature lookup is the optimal training strategy.")
md_lines.append("")

# Section 3: Counterfactual Coverage
md_lines.append("## 3. Counterfactual Coverage Interventions")
md_lines.append("")
md_lines.append("**IMPORTANT**: These are EXPLICIT INTERVENTIONS on the training distribution, "
                "NOT a natural env3b redesign.")
md_lines.append("")
md_lines.append("### Intervention Types")
md_lines.append("")
md_lines.append("| Signature Type | Mechanism | Effect | Label |")
md_lines.append("|---------------|-----------|--------|-------|")
md_lines.append("| Shared (positive_net) | Flat affordance (all try actions same outcome) | "
                "Observation decision-irrelevant (pre_best == post_best, VOI = -cost) | "
                "`decision_irrelevant_flat` |")
md_lines.append("| Unique (non_positive_net) | Swap to other category's affordance in same ambient group | "
                "Lookalike/deceptive/exception: pre-surface diagnostic feature misleading, "
                "post-observe corrective | `lookalike_deceptive_exception` |")
md_lines.append("")

md_lines.append("### Post-Coverage Signature Determinism")
md_lines.append("")
md_lines.append("| Cost | Coverage | Deterministic | Total | Fraction | Signatures Broken |")
md_lines.append("|------|----------|--------------|-------|----------|-------------------|")
for (cost, cov_frac), ca_data in sorted(CF_AUDITS.items()):
    broken = ca_data["total_signatures"] - ca_data["deterministic_count"]
    md_lines.append(f"| {cost} | {cov_frac} | {ca_data['deterministic_count']} | "
                    f"{ca_data['total_signatures']} | {ca_data['fraction_deterministic']:.4f} | {broken} |")
md_lines.append("")

md_lines.append("### Corrective Post-Evidence Audit (Lookalike Interventions)")
md_lines.append("")
md_lines.append("Verifies that unique-signature -> positive counterfactual interventions have "
                "post_observe_features containing diagnostic evidence consistent with the "
                "counterfactual affordance profile.")
md_lines.append("")
md_lines.append("| Coverage | n Lookalike | n Corrective | n Missing | Pass Rate | Design Valid |")
md_lines.append("|----------|-------------|--------------|-----------|-----------|--------------|")
for cov_frac, cr in sorted(CORRECTIVE_AUDITS.items()):
    md_lines.append(f"| {cov_frac} | {cr['n_lookalike_counterfactuals']} | "
                    f"{cr['n_with_corrective_post_evidence']} | "
                    f"{cr['n_without_corrective_post_evidence']} | "
                    f"{cr['corrective_post_evidence_pass_rate']:.4f} | "
                    f"{cr['design_valid']} |")
md_lines.append("")
if any(not cr["design_valid"] for cr in CORRECTIVE_AUDITS.values()):
    md_lines.append("**WARNING: Counterfactual coverage design is INVALID.** "
                    "Lookalike instances lack corrective post-observe evidence. "
                    "The post_observe_features come from the original object category, "
                    "not the target category. The 'lookalike' mechanism cannot work "
                    "without corrective post-observe evidence.")
    md_lines.append("")
    # Show details
    for cov_frac, cr in sorted(CORRECTIVE_AUDITS.items()):
        if cr["detail"]:
            md_lines.append(f"**Coverage {cov_frac} — example details:**")
            md_lines.append("")
            for d in cr["detail"][:3]:
                md_lines.append(f"- `{d['seed']}/{d['oid']}`: {d['original_category']} -> {d['target_category']}")
                md_lines.append(f"  - Target diagnostic: `{d['target_diagnostic_features']}`")
                md_lines.append(f"  - Post features: `{d['post_features_present']}`")
                md_lines.append(f"  - Corrective found: `{d['corrective_features_found']}` -> {d['has_corrective_evidence']}")
            md_lines.append("")
md_lines.append("")

# Section 4: Question Reports
md_lines.append("## 4. Question A: Does counterfactual coverage break signature determinism?")
md_lines.append("")
md_lines.append(f"**Original**: {n_det_true}/{len(SIGNATURE_LIST)} signatures deterministic. "
                "All shared sigs: frac+=1.0. All unique sigs: frac+=0.0. "
                "Entropy=0 for all signatures.")
md_lines.append("")
md_lines.append("**After counterfactual coverage**:")
for (cost, cov_frac), ca_data in sorted(CF_AUDITS.items()):
    broken = ca_data["total_signatures"] - ca_data["deterministic_count"]
    md_lines.append(f"- Cost={cost}, coverage={cov_frac}: {broken}/{ca_data['total_signatures']} "
                    f"signatures have mixed VOI (entropy > 0)")
    broken_sigs = [sig for sig in SIGNATURE_LIST
                   if sig in ca_data["audit"] and not ca_data["audit"][sig]["is_deterministic"]]
    if broken_sigs:
        for sig in broken_sigs:
            a = ca_data["audit"][sig]
            md_lines.append(f"  - {str(sig)[:50]}: frac+={a['fraction_positive']:.4f}, "
                            f"entropy={a['entropy']:.4f}")
md_lines.append("")

md_lines.append("## 5. Question B: Does the learned policy improve on held-out Test A/B?")
md_lines.append("")

for observe_cost, cost_label in [(0.03, "Primary"), (0.05, "Robustness")]:
    md_lines.append(f"### {cost_label} (cost={observe_cost})")
    md_lines.append("")
    md_lines.append("| Test | Condition | Net | ObsRate | PosObs | NonposObs | vsAlwaysObs | vsOracle | Collapsed? |")
    md_lines.append("|------|-----------|-----|---------|--------|-----------|-------------|----------|------------|")

    for group_name in ["A", "B", "C"]:
        for key, label in [("original", "original")] + [(cov_frac, f"cf_cov{cov_frac}") for cov_frac in COVERAGE_FRACTIONS]:
            if key == "original":
                exp_data = ALL_EXPERIMENTS[(observe_cost, "original")]
            else:
                exp_data = ALL_EXPERIMENTS[(observe_cost, key)]
            gr = exp_data["groups"].get(group_name, [])
            if not gr:
                continue
            agg = aggregate_group(gr)
            collapsed = agg["obs_rate"] <= 0.0 or agg["obs_rate"] >= 1.0
            collapse_str = "always-observe" if agg["obs_rate"] >= 1.0 else ("always-try" if agg["obs_rate"] <= 0.0 else "no")
            md_lines.append(f"| {group_name} | {label} | {agg['policy_net']:.4f} | "
                            f"{agg['obs_rate']:.4f} | {agg['pos_obs_rate']:.4f} | "
                            f"{agg['nonpos_obs_rate']:.4f} | {agg['vs_always_obs']:+.4f} | "
                            f"{agg['vs_oracle']:+.4f} | {collapse_str} |")
    md_lines.append("")

md_lines.append("## 6. Question C: Does signature_lookup_policy remain strong?")
md_lines.append("")
md_lines.append("A signature_lookup_policy means the observe_value_model predictions are "
                "perfectly predictable from signature identity alone.")
md_lines.append("")
for observe_cost, cost_label in [(0.03, "Primary"), (0.05, "Robustness")]:
    md_lines.append(f"### {cost_label} (cost={observe_cost})")
    md_lines.append("")
    for cov_frac in [0.0] + COVERAGE_FRACTIONS:
        if cov_frac == 0.0:
            exp_data = ALL_EXPERIMENTS[(observe_cost, "original")]
            label = "original"
        else:
            exp_data = ALL_EXPERIMENTS[(observe_cost, cov_frac)]
            label = f"cf_cov{cov_frac}"
        for group_name in ["A", "B", "C"]:
            gr = exp_data["groups"].get(group_name, [])
            if not gr:
                continue
            agg = aggregate_group(gr)
            collapsed = agg["obs_rate"] <= 0.0 or agg["obs_rate"] >= 1.0
            md_lines.append(f"- {label} Test {group_name}: obs_rate={agg['obs_rate']:.4f} "
                            f"({'COLLAPSED' if collapsed else 'selective'})")
    md_lines.append("")

# Section 7: Overall Verdicts
md_lines.append("## 7. Overall Assessment")
md_lines.append("")
md_lines.append("**Critical constraint (8)**: Do NOT claim success if same-signature mixed "
                "coverage exists but learned_policy still collapses to always_observe.")
md_lines.append("")
md_lines.append("**Corrective evidence constraint**: Lookalike interventions require "
                "corrective post-observe evidence consistent with the counterfactual "
                "affordance profile. If absent, the counterfactual design is INVALID.")
md_lines.append("")
md_lines.append("| Cost | Coverage | Sigs Broken | TestA Collapsed | TestA > AlwaysObs | TestC Passes | Corrective Evid | Design Valid | Verdict |")
md_lines.append("|------|----------|-------------|-----------------|-------------------|--------------|-----------------|--------------|---------|")
for (cost, cov_frac), v in sorted(overall_verdicts.items()):
    corr_rate = v.get("corrective_post_evidence_pass_rate", 0.0)
    design_valid = v.get("design_valid", True)
    md_lines.append(f"| {cost} | {cov_frac} | {v['sigs_broken']}/{v['total_sigs']} | "
                    f"{v['test_a_collapsed']} | {v['test_a_beats_always_obs']} | "
                    f"{v['test_c_passes']} | {corr_rate:.4f} | {design_valid} | {v['verdict']} |")
md_lines.append("")

# Section 8: Commands
md_lines.append("## 8. Commands Run")
md_lines.append("")
md_lines.append('```')
md_lines.append('& "D:\\conda\\python.exe" "_block1j40b7e_env3b_distributional_audit_counterfactual.py"')
md_lines.append('```')
md_lines.append("")
md_lines.append("```")
md_lines.append(f"[block_done]")
md_lines.append(f"block_id=1J40b-7e")
md_lines.append(f"implementation_status={overall_status}")
md_lines.append("```")

with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"MD -> {md_path}")

# =============================================================================
# Final Summary
# =============================================================================
print(f"\n{'='*70}")
print(f"RESULTS SUMMARY")
print(f"{'='*70}")

print(f"\n--- Part A: Distributional Audit ---")
print(f"  TRUE/ORACLE (cost=0.03): {n_det_true}/{len(SIGNATURE_LIST)} signatures deterministic")
print(f"  Finding: Every signature perfectly predicts optimal observe decision.")
print(f"  Signature lookup is optimal in original training distribution.")

print(f"\n--- Part B: Counterfactual Coverage ---")
for (cost, cov_frac), ca_data in sorted(CF_AUDITS.items()):
    broken = ca_data["total_signatures"] - ca_data["deterministic_count"]
    print(f"  cost={cost}, cov={cov_frac}: {broken}/{ca_data['total_signatures']} signatures "
          f"have mixed VOI")

print(f"\n--- Part C: Policy Results ---")
for (cost, key), exp_data in sorted(ALL_EXPERIMENTS.items(),
                                      key=lambda kv: (kv[0][0], str(kv[0][1]))):
    print(f"\n  cost={cost}, condition={key}:")
    for group_name in ["A", "B", "C"]:
        gr = exp_data["groups"].get(group_name, [])
        if gr:
            agg = aggregate_group(gr)
            collapsed = agg["obs_rate"] <= 0.0 or agg["obs_rate"] >= 1.0
            print(f"    Test {group_name}: net={agg['policy_net']:.4f} obs_rate={agg['obs_rate']:.4f} "
                  f"vs_always_obs={agg['vs_always_obs']:+.4f} "
                  f"{'[COLLAPSED]' if collapsed else '[SELECTIVE]'}")

print(f"\n--- Corrective Post-Evidence Audit ---")
for cov_frac, cr in sorted(CORRECTIVE_AUDITS.items()):
    print(f"  cov={cov_frac}: {cr['n_with_corrective_post_evidence']}/{cr['n_lookalike_counterfactuals']} "
          f"lookalike instances have corrective post-evidence "
          f"(pass_rate={cr['corrective_post_evidence_pass_rate']:.4f}, "
          f"design_valid={cr['design_valid']})")
if any(not cr["design_valid"] for cr in CORRECTIVE_AUDITS.values()):
    print(f"  *** DESIGN INVALID: Lookalike interventions lack corrective post-observe evidence ***")

print(f"\n--- Overall Verdicts ---")
for (cost, cov_frac), v in sorted(overall_verdicts.items()):
    print(f"  cost={cost}, cov={cov_frac}: {v['verdict']}")

print(f"\n{'='*70}")
print(f"Block 1J40b-7e complete.")
print(f"  status: {overall_status}")
print(f"  elapsed: {elapsed:.1f}s")
print(f"{'='*70}")
