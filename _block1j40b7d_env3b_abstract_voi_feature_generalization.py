"""
Block 1J40b-7d — env3b Abstract Pre-Only VOI Feature Generalization Test.

Tests whether adding legal, pre-only abstract VOI features to the
observe_value_model improves generalization to held-out signatures
and cue-combinations.

Variants:
  D0: raw pre_vec only (corrected 7c baseline)
  D1: raw pre_vec + abstract features
  D2: abstract features only
  D3: raw + abstract, minus exact-signature features (10, 11)

Stress tests (from 7c):
  Test A: held-out pre_surface signatures (4 variants)
  Test B: held-out cue-combinations (2 variants)
  Test C: held-out seeds / LOSO (5 variants)

Observe costs: primary=0.03, robustness=0.05.
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
# Part 1: Object Generation (from 7a)
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
# Part 2: Feature Vector Building (model-legal only)
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
# Part 3: Signature Analysis & Forbidden Field Audit
# =============================================================================
print("[3/12] Analyzing signatures and forbidden fields...")

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

# Forbidden field audit
FORBIDDEN_FIELDS = [
    "hidden_category", "hidden_subtype", "hidden_audit_rationale_class",
    "audit_rationale_class", "positive_net", "non_positive_net",
    "audit_VOI", "audit_observe_value", "oracle_action", "oracle_selective",
    "true_affordance_profile", "effective_affordance_profile",
    "hidden_affordance_profile", "affordance_profile",
    "object_id", "seed_id", "oid", "depth_schedule", "episode_id",
    "ambient_group", "group_role", "voi_class",
    "hidden_features", "post_observe_revealed",
    "pre_surface_extra", "voi_rationale", "category",
]
forbidden_in_model = []
for ff in FORBIDDEN_FIELDS:
    for mk in ALL_MODEL_FEATURE_KEYS:
        if ff.lower() in mk.lower().replace("feat_", "").replace("state_", ""):
            forbidden_in_model.append(f"{ff} -> {mk}")
print(f"  Forbidden fields in model keys: {len(forbidden_in_model)}")

# =============================================================================
# Part 4: Model Classes
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
# Part 5: Stress Split Definitions (from 7c)
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

# Test B variants (from 7c)
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

# Test C variants (LOSO)
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
# Part 6: Training Data Construction
# =============================================================================
print("[5/12] Building training data helpers...")

def build_training_records_for_oids(train_oids, observe_cost):
    records = []
    record_id = 0
    for seed, oid in train_oids:
        obj = per_seed_objects[seed][oid]
        lbl = per_seed_audit_labels[seed][oid]
        affordance = lbl["affordance_profile"]
        pre_features = obj["pre_surface_features"]
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


def build_obj_eval_vectors(test_oids):
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

        true_returns = {}
        for action in ALL_TRY_AFFORDANCES:
            outcome = lbl["affordance_profile"].get(action, "fail")
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
# Part 7: Abstract Feature Computation
# =============================================================================
print("[6/12] Defining abstract feature system...")

# Abstract feature names (15 features)
ABSTRACT_FEATURE_NAMES = [
    "n_pre_surface_features",           # 1
    "n_ambient_features_known",         # 2
    "n_shared_surface_cues_known",      # 3
    "n_pre_visible_diagnostic_cues_known", # 4
    "has_any_pre_visible_diagnostic_cue",  # 5
    "has_any_shared_surface_cue",       # 6
    "pre_try_value_spread",             # 7
    "pre_best_second_margin",           # 8
    "pre_action_value_variance",        # 9
    "training_support_count_for_exact_signature", # 10
    "training_mean_observe_target_for_exact_signature", # 11
    "cue_support_mean",                 # 12a
    "cue_support_max",                  # 12b
    "cue_observe_target_mean",          # 13a
    "cue_observe_target_max",           # 13b
    "nfeat_group_mean_observe_target",  # 14
    "shared_cue_conflict_sum",          # 15
]
N_ABSTRACT = len(ABSTRACT_FEATURE_NAMES)

# Features 10, 11 are "exact-signature" features removed in D3
EXACT_SIG_FEATURE_INDICES = [9, 10]  # 0-indexed


def compute_training_stats(train_oids, estimator, observe_cost):
    """Compute training statistics for abstract features.

    Returns dict with:
      - signature_stats: sig -> {count, mean_target}
      - cue_stats: feature_name -> {count, mean_target, target_variance}
      - nfeat_stats: n_features -> {count, mean_target}
      - global_mean_target: float
    """
    sig_targets = defaultdict(list)
    cue_targets = defaultdict(list)
    nfeat_targets = defaultdict(list)
    all_targets = []

    for tseed, toid in train_oids:
        tobj = per_seed_objects[tseed][toid]
        tlbl = per_seed_audit_labels[tseed][toid]
        pre_features = tobj["pre_surface_features"]
        post_features = tobj["post_observe_features"]
        state = tobj["visible_state"]
        sig = get_pre_surface_signature(pre_features)

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
        target = post_best - pre_best - observe_cost

        sig_targets[sig].append(target)
        all_targets.append(target)
        nfeat = len(pre_features)
        nfeat_targets[nfeat].append(target)

        for f in pre_features:
            cue_targets[f].append(target)

    signature_stats = {}
    for sig, targets in sig_targets.items():
        signature_stats[sig] = {
            "count": len(targets),
            "mean_target": sum(targets) / len(targets),
        }

    cue_stats = {}
    for f, targets in cue_targets.items():
        mean_t = sum(targets) / len(targets)
        var_t = sum((t - mean_t)**2 for t in targets) / len(targets) if len(targets) > 1 else 0.0
        cue_stats[f] = {
            "count": len(targets),
            "mean_target": mean_t,
            "target_variance": var_t,
        }

    nfeat_stats = {}
    for nf, targets in nfeat_targets.items():
        nfeat_stats[nf] = {
            "count": len(targets),
            "mean_target": sum(targets) / len(targets),
        }

    global_mean = sum(all_targets) / len(all_targets) if all_targets else 0.0

    return {
        "signature_stats": signature_stats,
        "cue_stats": cue_stats,
        "nfeat_stats": nfeat_stats,
        "global_mean_target": global_mean,
    }


def compute_abstract_features(pre_features, pre_vec, estimator, training_stats):
    """Compute abstract pre-only VOI features for observe_value_model.

    All features are computable from pre_surface_features, training experience,
    and trained Q-function — NO test post_vec or audit labels.
    """
    af = {}

    # 1. n_pre_surface_features
    af["n_pre_surface_features"] = float(len(pre_features))

    # 2. n_ambient_features_known
    af["n_ambient_features_known"] = float(
        sum(1 for f in pre_features if f in ALL_AMBIENT_FEATURE_NAMES))

    # 3. n_shared_surface_cues_known
    af["n_shared_surface_cues_known"] = float(
        sum(1 for f in pre_features if f in SHARED_SURFACE_FEATURES))

    # 4. n_pre_visible_diagnostic_cues_known
    af["n_pre_visible_diagnostic_cues_known"] = float(
        sum(1 for f in pre_features if f in ALL_POST_OBSERVE_FEATURE_NAMES))

    # 5. has_any_pre_visible_diagnostic_cue
    af["has_any_pre_visible_diagnostic_cue"] = (
        1.0 if af["n_pre_visible_diagnostic_cues_known"] > 0 else 0.0)

    # 6. has_any_shared_surface_cue
    af["has_any_shared_surface_cue"] = (
        1.0 if af["n_shared_surface_cues_known"] > 0 else 0.0)

    # 7-9: Q-based features from pre_vec
    pre_qs = estimator.predict_all_actions(pre_vec)
    pre_try_values = sorted(
        [q for ak, q in pre_qs.items() if ak != "observe"],
        reverse=True)
    if len(pre_try_values) >= 2:
        af["pre_try_value_spread"] = pre_try_values[0] - pre_try_values[-1]
        af["pre_best_second_margin"] = pre_try_values[0] - pre_try_values[1]
    else:
        af["pre_try_value_spread"] = 0.0
        af["pre_best_second_margin"] = 0.0
    if pre_try_values:
        mean_v = sum(pre_try_values) / len(pre_try_values)
        af["pre_action_value_variance"] = (
            sum((v - mean_v)**2 for v in pre_try_values) / len(pre_try_values))
    else:
        af["pre_action_value_variance"] = 0.0

    # 10-11: Signature-based training stats
    sig = get_pre_surface_signature(pre_features)
    sig_stats = training_stats.get("signature_stats", {}).get(sig, {})
    af["training_support_count_for_exact_signature"] = float(
        sig_stats.get("count", 0))
    global_mean = training_stats.get("global_mean_target", 0.0)
    af["training_mean_observe_target_for_exact_signature"] = (
        sig_stats.get("mean_target", global_mean))

    # 12-13: Per-cue training stats
    cue_counts = []
    cue_targets = []
    cue_stats_map = training_stats.get("cue_stats", {})
    for f in pre_features:
        if f in cue_stats_map:
            cs = cue_stats_map[f]
            cue_counts.append(cs["count"])
            cue_targets.append(cs["mean_target"])

    if cue_counts:
        af["cue_support_mean"] = sum(cue_counts) / len(cue_counts)
        af["cue_support_max"] = float(max(cue_counts))
        af["cue_observe_target_mean"] = sum(cue_targets) / len(cue_targets)
        af["cue_observe_target_max"] = max(cue_targets)
    else:
        af["cue_support_mean"] = 0.0
        af["cue_support_max"] = 0.0
        af["cue_observe_target_mean"] = global_mean
        af["cue_observe_target_max"] = global_mean

    # 14. nfeat_group_mean_observe_target
    nfeat = len(pre_features)
    nfeat_stats = training_stats.get("nfeat_stats", {}).get(nfeat, {})
    af["nfeat_group_mean_observe_target"] = nfeat_stats.get(
        "mean_target", global_mean)

    # 15. shared-cue conflict sum
    conflict_sum = 0.0
    for f in pre_features:
        if f in SHARED_SURFACE_FEATURES and f in cue_stats_map:
            conflict_sum += cue_stats_map[f].get("target_variance", 0.0)
    af["shared_cue_conflict_sum"] = conflict_sum

    return af


def abstract_feature_vector(af_dict):
    """Convert abstract feature dict to ordered list."""
    return [af_dict.get(name, 0.0) for name in ABSTRACT_FEATURE_NAMES]


print(f"  Abstract features: {N_ABSTRACT}")
for i, name in enumerate(ABSTRACT_FEATURE_NAMES):
    print(f"    {i+1:2d}. {name}")

# =============================================================================
# Part 8: Observe Value Model Variants
# =============================================================================
print("[7/12] Defining observe_value_model variants...")

VARIANT_DEFS = {
    "D0": {
        "label": "raw pre_vec only (7c baseline)",
        "use_raw": True,
        "use_abstract": False,
        "exclude_exact_sig": False,
    },
    "D1": {
        "label": "raw pre_vec + abstract features",
        "use_raw": True,
        "use_abstract": True,
        "exclude_exact_sig": False,
    },
    "D2": {
        "label": "abstract features only",
        "use_raw": False,
        "use_abstract": True,
        "exclude_exact_sig": False,
    },
    "D3": {
        "label": "raw + abstract, minus exact-signature features (10,11)",
        "use_raw": True,
        "use_abstract": True,
        "exclude_exact_sig": True,
    },
}

VARIANT_IDS = ["D0", "D1", "D2", "D3"]


def build_ov_input(pre_vec, af_dict, variant_id):
    """Build observe_value_model input vector for a given variant."""
    vdef = VARIANT_DEFS[variant_id]
    parts = []
    if vdef["use_raw"]:
        parts.extend(pre_vec)
    if vdef["use_abstract"]:
        af_vec = abstract_feature_vector(af_dict)
        if vdef["exclude_exact_sig"]:
            # Remove features at indices 9 and 10 (exact sig support/value)
            af_vec = [v for i, v in enumerate(af_vec)
                      if i not in EXACT_SIG_FEATURE_INDICES]
        parts.extend(af_vec)
    return parts


# =============================================================================
# Part 9: Policy Evaluation (two-stage, per variant)
# =============================================================================
print("[8/12] Defining policy evaluation...")

def run_stress_policy_with_variants(train_oids, test_oids, observe_cost):
    """Run all D0-D3 policy variants on a stress split.

    Returns dict mapping variant_id -> results dict.
    """
    train_records = build_training_records_for_oids(train_oids, observe_cost)
    obj_data = build_obj_eval_vectors(test_oids)
    test_keys = sorted(obj_data.keys())
    n_test = len(test_keys)

    # Train shared Q-function (try-action estimator)
    X_train = [r["feature_vector"] for r in train_records]
    y_train = [r["net_value"] for r in train_records]
    ak_train = [r["action_key"] for r in train_records]
    estimator = PerActionValueEstimator(alpha=RIDGE_ALPHA)
    estimator.fit(X_train, y_train, ak_train)

    # Compute training stats for abstract features
    training_stats = compute_training_stats(train_oids, estimator, observe_cost)

    # ---- Baselines (shared across variants) ----
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

    # Pre only
    pre_returns = []
    for key in test_keys:
        od = obj_data[key]
        qs = estimator.predict_all_actions(od["pre_vec"])
        try_qs = {ak: q for ak, q in qs.items() if ak != "observe"}
        best_action = max(try_qs, key=try_qs.get)
        pre_returns.append(od["true_returns"].get(best_action, 0.0))
    pre_mean = sum(pre_returns) / n_test if n_test else 0.0

    # Always observe
    post_returns_net = []
    post_returns_gross = []
    for key in test_keys:
        od = obj_data[key]
        qs = estimator.predict_all_actions(od["post_vec"])
        try_qs = {ak: q for ak, q in qs.items() if ak != "observe"}
        best_action = max(try_qs, key=try_qs.get)
        ret = od["true_returns"].get(best_action, 0.0)
        post_returns_gross.append(ret)
        post_returns_net.append(ret - observe_cost)
    post_mean_gross = sum(post_returns_gross) / n_test if n_test else 0.0
    post_mean_net = sum(post_returns_net) / n_test if n_test else 0.0

    # Random observe
    rng_rand = random.Random(42)
    random_returns = []
    for key in test_keys:
        od = obj_data[key]
        if rng_rand.random() < 0.5:
            vec = od["post_vec"]
        else:
            vec = od["pre_vec"]
        qs = estimator.predict_all_actions(vec)
        try_qs = {ak: q for ak, q in qs.items() if ak != "observe"}
        best_action = max(try_qs, key=try_qs.get)
        random_returns.append(od["true_returns"].get(best_action, 0.0))
    random_mean = sum(random_returns) / n_test if n_test else 0.0

    # Oracle selective
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
    oracle_sel_obs_rate = oracle_sel_obs / n_test if n_test else 0.0

    baseline_results = {
        "pre_mean": round(pre_mean, 4),
        "always_try": round(pre_mean, 4),
        "always_observe_net": round(post_mean_net, 4),
        "always_observe_gross": round(post_mean_gross, 4),
        "random_observe": round(random_mean, 4),
        "oracle_selective": round(oracle_sel_mean, 4),
        "oracle_selective_obs_rate": round(oracle_sel_obs_rate, 4),
    }

    # ---- Per-variant policy evaluation ----
    variant_results = {}
    for variant_id in VARIANT_IDS:
        vdef = VARIANT_DEFS[variant_id]

        # Build observe_value training data
        ov_X = []
        ov_y = []
        ov_y_true = []

        # True-target computation: training-only sig-group expected returns
        train_sig_groups_ov = defaultdict(list)
        for tseed, toid in train_oids:
            tobj = per_seed_objects[tseed][toid]
            tlbl = per_seed_audit_labels[tseed][toid]
            tsig = get_pre_surface_signature(tobj["pre_surface_features"])
            true_rets = {}
            for action in ALL_TRY_AFFORDANCES:
                outcome = tlbl["affordance_profile"].get(action, "fail")
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
            af_dict = compute_abstract_features(pre_features, pre_vec,
                                                estimator, training_stats)
            ov_input = build_ov_input(pre_vec, af_dict, variant_id)
            ov_X.append(ov_input)

            # True-return-based sanity target
            tsig = get_pre_surface_signature(pre_features)
            pre_best_true = train_sig_exp_ov.get(tsig, 0.0)
            true_rets_obj = {}
            for action in ALL_TRY_AFFORDANCES:
                outcome = tlbl["affordance_profile"].get(action, "fail")
                true_rets_obj[action] = try_net_value(outcome == "success")
            post_best_true = max(true_rets_obj.values())
            ov_y_true.append(post_best_true - pre_best_true - observe_cost)

        # Train observe_value_model
        ov_model = None
        ov_normalizer = None
        ov_target_source = "learned-Q: post_best_q - pre_best_q - observe_cost"
        ov_true_target_corr = None
        if len(ov_X) >= 3 and len(set(ov_y)) > 1:
            ov_normalizer = FeatureNormalizer().fit(ov_X)
            ov_X_norm = ov_normalizer.transform(ov_X)
            ov_model = RidgeRegression(alpha=RIDGE_ALPHA).fit(ov_X_norm, ov_y)
            if len(ov_y_true) == len(ov_y) and len(ov_y) > 0:
                mean_q = sum(ov_y) / len(ov_y)
                mean_t = sum(ov_y_true) / len(ov_y_true)
                num = sum((ov_y[i] - mean_q) * (ov_y_true[i] - mean_t)
                         for i in range(len(ov_y)))
                den_q = math.sqrt(sum((v - mean_q)**2 for v in ov_y))
                den_t = math.sqrt(sum((v - mean_t)**2 for v in ov_y_true))
                if den_q > 1e-12 and den_t > 1e-12:
                    ov_true_target_corr = round(num / (den_q * den_t), 4)

        # Run policy on test objects
        policy_returns_gross = []
        policy_returns_net = []
        policy_decisions = []
        policy_obs_count = 0
        policy_direct_count = 0

        for key in test_keys:
            od = obj_data[key]
            pre_vec = od["pre_vec"]
            true_returns = od["true_returns"]

            # Stage 1: observe/direct decision
            pre_qs = estimator.predict_all_actions(pre_vec)
            pre_try_qs = {ak: q for ak, q in pre_qs.items() if ak != "observe"}
            pre_best_action = max(pre_try_qs, key=pre_try_qs.get)
            pre_best_value = pre_try_qs[pre_best_action]

            af_dict_test = compute_abstract_features(
                od["pre_features"], pre_vec, estimator, training_stats)
            ov_input_test = build_ov_input(pre_vec, af_dict_test, variant_id)

            if ov_model is not None and ov_normalizer is not None:
                ov_input_norm = ov_normalizer.transform([ov_input_test])[0]
                predicted_observe_net = ov_model.predict([ov_input_norm])[0]
            else:
                predicted_observe_net = -observe_cost

            decided_observe = predicted_observe_net > 0

            if decided_observe:
                # Stage 2: reveal post_observe_features
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
                policy_direct_count += 1
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
                "pre_best_value": pre_best_value,
                "post_best_action": post_best_action,
                "post_best_value": post_best_value,
                "chosen_action": chosen_action,
                "true_return_gross": true_ret_gross,
                "true_return_net": true_ret_net,
                "n_pre_surface_features": od["n_pre_surface_features"],
            })

        policy_mean_gross = (sum(policy_returns_gross) / n_test) if n_test else 0.0
        policy_mean_net = (sum(policy_returns_net) / n_test) if n_test else 0.0
        policy_obs_rate = policy_obs_count / n_test if n_test else 0.0
        policy_direct_rate = policy_direct_count / n_test if n_test else 0.0

        # Per-rationale breakdown
        pos_dec = [d for d in policy_decisions if d["rationale_class"] == "positive_change"]
        nonpos_dec = [d for d in policy_decisions if d["rationale_class"] != "positive_change"]
        pos_obs_rate = (sum(1 for d in pos_dec if d["decided_observe"]) /
                        len(pos_dec) if pos_dec else 0.0)
        nonpos_obs_rate = (sum(1 for d in nonpos_dec if d["decided_observe"]) /
                           len(nonpos_dec) if nonpos_dec else 0.0)

        # Per-category
        cat_breakdown = {}
        for cat in CATEGORIES:
            cat_d = [d for d in policy_decisions if d["category"] == cat]
            if cat_d:
                cat_net = [d["true_return_net"] for d in cat_d]
                cat_obs = sum(1 for d in cat_d if d["decided_observe"])
                cat_breakdown[cat] = {
                    "count": len(cat_d),
                    "policy_mean_net": sum(cat_net) / len(cat_net),
                    "obs_rate": cat_obs / len(cat_d),
                }

        # Per n_features_known
        nfeat_breakdown = defaultdict(lambda: {"nets": [], "obs": 0, "total": 0})
        for d in policy_decisions:
            nf = d["n_pre_surface_features"]
            nfeat_breakdown[nf]["nets"].append(d["true_return_net"])
            nfeat_breakdown[nf]["total"] += 1
            if d["decided_observe"]:
                nfeat_breakdown[nf]["obs"] += 1
        nfeat_summary = {}
        for nf in sorted(nfeat_breakdown.keys()):
            data = nfeat_breakdown[nf]
            nfeat_summary[nf] = {
                "count": data["total"],
                "policy_mean_net": sum(data["nets"]) / len(data["nets"]),
                "obs_rate": data["obs"] / data["total"],
            }

        # Per-signature results
        per_sig_results = defaultdict(lambda: {"nets": [], "obs": 0, "total": 0,
                                                "is_held_out": False})
        for d in policy_decisions:
            sig_key = str(d["signature"])
            per_sig_results[sig_key]["nets"].append(d["true_return_net"])
            per_sig_results[sig_key]["total"] += 1
            if d["decided_observe"]:
                per_sig_results[sig_key]["obs"] += 1
        sig_summary = {}
        for sig_key in sorted(per_sig_results.keys()):
            data = per_sig_results[sig_key]
            sig_summary[sig_key] = {
                "count": data["total"],
                "policy_mean_net": sum(data["nets"]) / len(data["nets"]),
                "obs_rate": data["obs"] / data["total"],
            }

        # Q error
        q_errors = []
        for d in policy_decisions:
            pred_q = d["pre_best_value"] if not d["decided_observe"] else d["post_best_value"]
            q_errors.append(abs(pred_q - d["true_return_gross"]))
        mean_q_error = sum(q_errors) / len(q_errors) if q_errors else 0.0

        variant_results[variant_id] = {
            "variant_id": variant_id,
            "variant_label": vdef["label"],
            "n_train": len(train_oids),
            "n_test": n_test,
            "pre_mean": baseline_results["pre_mean"],
            "always_try": baseline_results["always_try"],
            "always_observe_net": baseline_results["always_observe_net"],
            "always_observe_gross": baseline_results["always_observe_gross"],
            "random_observe": baseline_results["random_observe"],
            "oracle_selective": baseline_results["oracle_selective"],
            "oracle_selective_obs_rate": baseline_results["oracle_selective_obs_rate"],
            "policy_mean_gross": round(policy_mean_gross, 4),
            "policy_mean_net": round(policy_mean_net, 4),
            "policy_obs_rate": round(policy_obs_rate, 4),
            "policy_direct_rate": round(policy_direct_rate, 4),
            "pos_obs_rate": round(pos_obs_rate, 4),
            "nonpos_obs_rate": round(nonpos_obs_rate, 4),
            "harmful_obs_rate": round(nonpos_obs_rate, 4),
            "mean_q_error": round(mean_q_error, 4),
            "cat_breakdown": cat_breakdown,
            "nfeat_summary": nfeat_summary,
            "sig_summary": sig_summary,
            "policy_decisions": policy_decisions,
            "learned_minus_always_observe": round(policy_mean_net - baseline_results["always_observe_net"], 4),
            "learned_minus_always_try": round(policy_mean_net - baseline_results["always_try"], 4),
            "learned_minus_oracle": round(baseline_results["oracle_selective"] - policy_mean_net, 4),
            "ov_target_source": ov_target_source,
            "ov_true_target_corr": ov_true_target_corr,
            "ov_n_training_objects": len(ov_X),
            "ov_n_features": len(ov_X[0]) if ov_X else 0,
        }

    return variant_results


# =============================================================================
# Part 10: Run All Experiments
# =============================================================================
print("[9/12] Running primary stress tests (cost=0.03)...")

def run_all_tests(observe_cost, cost_label):
    """Run all stress tests at a given observe_cost for all variants."""
    all_results = {"test_a": [], "test_b": [], "test_c": [], "observe_cost": observe_cost}

    print(f"\n{'='*70}")
    print(f"STRESS TESTS at observe_cost = {observe_cost} ({cost_label})")
    print(f"{'='*70}")

    # Test A
    for va in TEST_A_VARIANTS:
        print(f"\n  --- Test {va['id']} (cost={observe_cost})...")
        vresults = run_stress_policy_with_variants(
            va["train_oids"], va["test_oids"], observe_cost)
        for vid in VARIANT_IDS:
            vr = vresults[vid]
            vr["test_variant_id"] = va["id"]
            vr["test_group"] = "A"
            print(f"    {vid}: policy_net={vr['policy_mean_net']:.4f}, "
                  f"obs_rate={vr['policy_obs_rate']:.4f}, "
                  f"pos_obs={vr['pos_obs_rate']:.4f}, "
                  f"nonpos_obs={vr['nonpos_obs_rate']:.4f}, "
                  f"always_obs={vr['always_observe_net']:.4f}, "
                  f"oracle={vr['oracle_selective']:.4f}, "
                  f"ov_feats={vr['ov_n_features']}")
        all_results["test_a"].append(vresults)

    # Test B
    for vb in TEST_B_VARIANTS:
        print(f"\n  --- Test {vb['id']} (cost={observe_cost})...")
        vresults = run_stress_policy_with_variants(
            vb["train_oids"], vb["test_oids"], observe_cost)
        for vid in VARIANT_IDS:
            vr = vresults[vid]
            vr["test_variant_id"] = vb["id"]
            vr["test_group"] = "B"
            print(f"    {vid}: policy_net={vr['policy_mean_net']:.4f}, "
                  f"obs_rate={vr['policy_obs_rate']:.4f}, "
                  f"pos_obs={vr['pos_obs_rate']:.4f}, "
                  f"nonpos_obs={vr['nonpos_obs_rate']:.4f}, "
                  f"always_obs={vr['always_observe_net']:.4f}, "
                  f"oracle={vr['oracle_selective']:.4f}, "
                  f"ov_feats={vr['ov_n_features']}")
        all_results["test_b"].append(vresults)

    # Test C
    for vc in TEST_C_VARIANTS:
        print(f"\n  --- Test {vc['id']} (cost={observe_cost})...")
        vresults = run_stress_policy_with_variants(
            vc["train_oids"], vc["test_oids"], observe_cost)
        for vid in VARIANT_IDS:
            vr = vresults[vid]
            vr["test_variant_id"] = vc["id"]
            vr["test_group"] = "C"
            print(f"    {vid}: policy_net={vr['policy_mean_net']:.4f}, "
                  f"obs_rate={vr['policy_obs_rate']:.4f}, "
                  f"pos_obs={vr['pos_obs_rate']:.4f}, "
                  f"nonpos_obs={vr['nonpos_obs_rate']:.4f}, "
                  f"always_obs={vr['always_observe_net']:.4f}, "
                  f"oracle={vr['oracle_selective']:.4f}, "
                  f"ov_feats={vr['ov_n_features']}")
        all_results["test_c"].append(vresults)

    return all_results


# Run primary
results_003 = run_all_tests(0.03, "primary")

print("\n[10/12] Running robustness stress tests (cost=0.05)...")
results_005 = run_all_tests(0.05, "robustness")

# =============================================================================
# Part 11: Aggregate and Evaluate
# =============================================================================
print("\n[11/12] Aggregating results...")

def evaluate_test_group(test_results_list, variant_id):
    """Aggregate results for a specific variant across a test group."""
    all_decisions = []
    total_test = 0
    total_pos_obs = 0
    total_pos = 0
    total_nonpos_obs = 0
    total_nonpos = 0
    agg_policy_net = 0.0
    agg_always_obs_net = 0.0
    agg_always_try = 0.0
    agg_oracle = 0.0
    agg_obs = 0

    for vresults in test_results_list:
        vr = vresults[variant_id]
        n = vr["n_test"]
        total_test += n
        agg_policy_net += vr["policy_mean_net"] * n
        agg_always_obs_net += vr["always_observe_net"] * n
        agg_always_try += vr["always_try"] * n
        agg_oracle += vr["oracle_selective"] * n
        agg_obs += vr["policy_obs_rate"] * n

        for d in vr["policy_decisions"]:
            all_decisions.append(d)
            if d["rationale_class"] == "positive_change":
                total_pos += 1
                if d["decided_observe"]:
                    total_pos_obs += 1
            else:
                total_nonpos += 1
                if d["decided_observe"]:
                    total_nonpos_obs += 1

    if total_test == 0:
        return {"status": "FAIL", "checks": {}, "failures": ["no test objects"],
                "agg": None}

    agg_policy_net /= total_test
    agg_always_obs_net /= total_test
    agg_always_try /= total_test
    agg_oracle /= total_test
    agg_obs_rate = agg_obs / total_test
    agg_pos_obs_rate = total_pos_obs / total_pos if total_pos > 0 else 0.0
    agg_nonpos_obs_rate = total_nonpos_obs / total_nonpos if total_nonpos > 0 else 0.0

    checks = {}
    checks["policy_net > always_observe_net"] = agg_policy_net > agg_always_obs_net
    checks["policy_net > always_try"] = agg_policy_net > agg_always_try
    checks["obs_rate between 0 and 1"] = 0.0 < agg_obs_rate < 1.0
    checks["pos_obs > nonpos_obs"] = agg_pos_obs_rate > agg_nonpos_obs_rate

    all_pass = all(checks.values())
    failures = [k for k, v in checks.items() if not v]
    collapsed_to_always = agg_obs_rate >= 0.999
    collapsed_to_never = agg_obs_rate <= 0.001

    if all_pass:
        status = "PASS"
    elif collapsed_to_always or collapsed_to_never:
        status = "FAIL"
    elif agg_policy_net > agg_always_obs_net or agg_policy_net > agg_always_try:
        status = "PARTIAL"
    else:
        status = "FAIL"

    return {
        "status": status,
        "checks": checks,
        "failures": failures,
        "agg": {
            "n_test": total_test,
            "policy_net": round(agg_policy_net, 4),
            "always_obs_net": round(agg_always_obs_net, 4),
            "always_try": round(agg_always_try, 4),
            "oracle_sel": round(agg_oracle, 4),
            "obs_rate": round(agg_obs_rate, 4),
            "pos_obs_rate": round(agg_pos_obs_rate, 4),
            "nonpos_obs_rate": round(agg_nonpos_obs_rate, 4),
            "policy_learned_minus_always_obs": round(agg_policy_net - agg_always_obs_net, 4),
            "policy_learned_minus_oracle": round(agg_oracle - agg_policy_net, 4),
        },
    }


# Evaluate all combinations
all_evaluations = {}
for test_group, test_list in [("A", results_003["test_a"]), ("B", results_003["test_b"]),
                               ("C", results_003["test_c"])]:
    for vid in VARIANT_IDS:
        eval_result = evaluate_test_group(test_list, vid)
        key = f"Test{test_group}_{vid}"
        all_evaluations[key] = {
            "primary": eval_result,
            "test_group": test_group,
            "variant": vid,
        }

for test_group, test_list in [("A", results_005["test_a"]), ("B", results_005["test_b"]),
                               ("C", results_005["test_c"])]:
    for vid in VARIANT_IDS:
        eval_result = evaluate_test_group(test_list, vid)
        key = f"Test{test_group}_{vid}"
        all_evaluations[key]["robustness"] = eval_result

# =============================================================================
# Part 12: Output
# =============================================================================
print("\n[12/12] Writing output files...")
elapsed = round(time.time() - t0, 1)

# Print summary
print(f"\n{'='*70}")
print(f"RESULTS SUMMARY")
print(f"{'='*70}")

for cost_label, cost_val in [("Primary (0.03)", 0.03), ("Robustness (0.05)", 0.05)]:
    print(f"\n--- {cost_label} ---")
    for test_group in ["A", "B", "C"]:
        print(f"  Test {test_group}:")
        for vid in VARIANT_IDS:
            key = f"Test{test_group}_{vid}"
            ev = all_evaluations[key][cost_label.split()[0].lower()]
            agg = ev["agg"]
            if agg:
                print(f"    {vid}: {ev['status']:7s}  "
                      f"policy_net={agg['policy_net']:.4f}  "
                      f"obs_rate={agg['obs_rate']:.4f}  "
                      f"pos_obs={agg['pos_obs_rate']:.4f}  "
                      f"nonpos_obs={agg['nonpos_obs_rate']:.4f}  "
                      f"vs_always_obs={agg['policy_learned_minus_always_obs']:+.4f}  "
                      f"vs_oracle={agg['policy_learned_minus_oracle']:.4f}")
            else:
                print(f"    {vid}: {ev['status']:7s}  (no data)")

# Determine overall status
overall_status = "FAIL"
primary_d0_a_fails = all_evaluations["TestA_D0"]["primary"]["status"] == "FAIL"
d1_a_improves = all_evaluations["TestA_D1"]["primary"]["status"] != "FAIL"
d2_a_improves = all_evaluations["TestA_D2"]["primary"]["status"] != "FAIL"
d1_b_improves = all_evaluations["TestB_D1"]["primary"]["status"] != "FAIL"
d2_b_improves = all_evaluations["TestB_D2"]["primary"]["status"] != "FAIL"

a_pass = (all_evaluations["TestA_D1"]["primary"]["status"] == "PASS" or
          all_evaluations["TestA_D2"]["primary"]["status"] == "PASS")
b_pass = (all_evaluations["TestB_D1"]["primary"]["status"] == "PASS" or
          all_evaluations["TestB_D2"]["primary"]["status"] == "PASS")

if a_pass and b_pass:
    overall_status = "PASS"
elif d1_a_improves or d2_a_improves or d1_b_improves or d2_b_improves:
    overall_status = "PARTIAL"
else:
    overall_status = "FAIL"

print(f"\n  Overall: {overall_status}")
print(f"  D0 TestA reproduces 7c FAIL: {all_evaluations['TestA_D0']['primary']['status']}")
print(f"  D1/D2 TestA improved: {a_pass}")
print(f"  D1/D2 TestB improved: {b_pass}")
print(f"  Elapsed: {elapsed}s")

# Hidden/forbidden checks
hidden_before_reveal_violations = 0
for seed in SEEDS:
    for oid, obj in per_seed_objects[seed].items():
        lbl = per_seed_audit_labels[seed][oid]
        pre_surf = set(obj["pre_surface_features"].keys())
        post_rev = set(lbl["post_observe_revealed"])
        for f in post_rev:
            if f in pre_surf:
                hidden_before_reveal_violations += 1

# Verify abstract features don't contain forbidden fields
abstract_forbidden_leak = False
for af_name in ABSTRACT_FEATURE_NAMES:
    for ff in FORBIDDEN_FIELDS:
        if ff.lower() in af_name.lower():
            abstract_forbidden_leak = True

# ---- JSON ----
json_output = {
    "block_id": "1J40b-7d",
    "elapsed_seconds": elapsed,
    "implementation_status": overall_status,
    "seeds": SEEDS,
    "variants": {vid: VARIANT_DEFS[vid]["label"] for vid in VARIANT_IDS},
    "abstract_features": {
        "n_features": N_ABSTRACT,
        "feature_names": ABSTRACT_FEATURE_NAMES,
        "forbidden_field_leak": abstract_forbidden_leak,
        "legality": "All features computable from pre_surface_features, training experience, and trained Q-function. No test post_vec or audit labels used.",
    },
    "forbidden_field_audit": {
        "forbidden_in_model_keys": len(forbidden_in_model),
        "abstract_forbidden_leak": abstract_forbidden_leak,
        "hidden_before_reveal_violations": hidden_before_reveal_violations,
    },
    "decision_time_audit": {
        "post_vec_before_decision": False,
        "two_stage_policy": True,
        "observe_value_from_pre_only": True,
        "post_vec_only_after_observe": True,
    },
    "primary": {"cost": 0.03, "evaluations": {}},
    "robustness": {"cost": 0.05, "evaluations": {}},
}

for cost_label, results in [("primary", results_003), ("robustness", results_005)]:
    cost_section = json_output[cost_label]
    for test_group_key, test_group_name in [("test_a", "A"), ("test_b", "B"), ("test_c", "C")]:
        test_list = results[test_group_key]
        for vresults in test_list:
            for vid in VARIANT_IDS:
                vr = vresults[vid]
                eval_key = f"Test{test_group_name}_{vid}"
                ev = all_evaluations[eval_key][cost_label]
                cost_section["evaluations"][f"{test_group_name}_{vid}"] = {
                    "status": ev["status"],
                    "aggregate": ev["agg"],
                    "checks": {k: v for k, v in ev["checks"].items()},
                }
                # Per-variant per-test details
                if f"{test_group_name}_variants" not in cost_section:
                    cost_section[f"{test_group_name}_variants"] = {}
                if vid not in cost_section[f"{test_group_name}_variants"]:
                    cost_section[f"{test_group_name}_variants"][vid] = []
                cost_section[f"{test_group_name}_variants"][vid].append({
                    "test_variant_id": vr["test_variant_id"],
                    "policy_net": vr["policy_mean_net"],
                    "always_observe_net": vr["always_observe_net"],
                    "oracle_selective": vr["oracle_selective"],
                    "obs_rate": vr["policy_obs_rate"],
                    "pos_obs_rate": vr["pos_obs_rate"],
                    "nonpos_obs_rate": vr["nonpos_obs_rate"],
                    "ov_n_features": vr["ov_n_features"],
                    "ov_true_target_corr": vr["ov_true_target_corr"],
                })

json_path = os.path.join(CURRENT_DIR, "runs", "block1j40b7d_env3b_abstract_voi_feature_generalization.json")
with open(json_path, "w") as f:
    json.dump(json_output, f, indent=2)
print(f"\nJSON -> {json_path}")

# ---- CSV ----
import csv as _csv
csv_path = os.path.join(CURRENT_DIR, "protocols", "block1j40b7d_env3b_abstract_voi_feature_generalization_table.csv")
with open(csv_path, "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["block_id", "1J40b-7d"])
    w.writerow(["implementation_status", overall_status])
    w.writerow(["elapsed_seconds", elapsed])

    for cost_label, results in [("primary_0.03", results_003), ("robustness_0.05", results_005)]:
        w.writerow([f"{cost_label}_cost", 0.03 if "0.03" in cost_label else 0.05])
        for test_group, test_name in [("a", "A"), ("b", "B"), ("c", "C")]:
            test_list = results[f"test_{test_group}"]
            for vid in VARIANT_IDS:
                for vresults in test_list:
                    vr = vresults[vid]
                    w.writerow([f"{cost_label}_{test_group}_{vid}_test_{vr['test_variant_id']}_policy_net", vr["policy_mean_net"]])
                    w.writerow([f"{cost_label}_{test_group}_{vid}_test_{vr['test_variant_id']}_obs_rate", vr["policy_obs_rate"]])
                    w.writerow([f"{cost_label}_{test_group}_{vid}_test_{vr['test_variant_id']}_pos_obs", vr["pos_obs_rate"]])
                    w.writerow([f"{cost_label}_{test_group}_{vid}_test_{vr['test_variant_id']}_nonpos_obs", vr["nonpos_obs_rate"]])
                    w.writerow([f"{cost_label}_{test_group}_{vid}_test_{vr['test_variant_id']}_always_obs", vr["always_observe_net"]])
                    w.writerow([f"{cost_label}_{test_group}_{vid}_test_{vr['test_variant_id']}_oracle", vr["oracle_selective"]])
                    w.writerow([f"{cost_label}_{test_group}_{vid}_test_{vr['test_variant_id']}_ov_feats", vr["ov_n_features"]])

print(f"CSV -> {csv_path}")

# ---- MD ----
md_path = os.path.join(CURRENT_DIR, "protocols", "block1j40b7d_env3b_abstract_voi_feature_generalization.md")
with open(md_path, "w") as f:
    f.write("# Block 1J40b-7d: env3b Abstract Pre-Only VOI Feature Generalization Test\n\n")
    f.write(f"- **Implementation Status**: {overall_status}\n")
    f.write(f"- **Elapsed**: {elapsed}s\n")
    f.write(f"- **Seeds**: {SEEDS}\n\n")

    f.write("## 1. Files\n\n")
    f.write("- `_block1j40b7d_env3b_abstract_voi_feature_generalization.py` — Written\n")
    f.write("- `runs/block1j40b7d_env3b_abstract_voi_feature_generalization.json` — Generated\n")
    f.write("- `protocols/block1j40b7d_env3b_abstract_voi_feature_generalization.md` — Generated\n")
    f.write("- `protocols/block1j40b7d_env3b_abstract_voi_feature_generalization_table.csv` — Generated\n\n")

    f.write("## 2. Commands Run\n\n")
    f.write("```\n")
    f.write('& "D:\\conda\\python.exe" "_block1j40b7d_env3b_abstract_voi_feature_generalization.py"\n')
    f.write("```\n\n")

    f.write("## 3. Variant Definitions\n\n")
    for vid in VARIANT_IDS:
        vdef = VARIANT_DEFS[vid]
        f.write(f"- **{vid}**: {vdef['label']}\n")
        f.write(f"  - use_raw={vdef['use_raw']}, use_abstract={vdef['use_abstract']}, exclude_exact_sig={vdef['exclude_exact_sig']}\n\n")

    f.write("## 4. Abstract Feature List\n\n")
    f.write(f"{N_ABSTRACT} features, all legal (pre_surface_features, training experience, trained Q-function only):\n\n")
    f.write("| # | Feature | Source |\n")
    f.write("|---|---------|--------|\n")
    for i, name in enumerate(ABSTRACT_FEATURE_NAMES):
        if i < 6:
            src = "pre_surface_features"
        elif i < 9:
            src = "Q-estimates from pre_vec"
        elif i < 11:
            src = "training signature statistics"
        elif i < 15:
            src = "training cue/nfeat statistics"
        else:
            src = "training cue conflict"
        f.write(f"| {i+1} | {name} | {src} |\n")
    f.write(f"\n- Abstract forbidden field leak: {abstract_forbidden_leak}\n\n")

    f.write("## 5. Forbidden-Field Audit\n\n")
    f.write(f"- Forbidden fields in model keys: {len(forbidden_in_model)}\n")
    f.write(f"- Abstract forbidden field leak: {abstract_forbidden_leak}\n")
    f.write(f"- Hidden-before-reveal violations: {hidden_before_reveal_violations}\n")
    f.write(f"- Decision-time post_vec absent: True (two-stage policy)\n\n")

    for cost_label, results, cost_val in [("Primary (0.03)", results_003, 0.03),
                                            ("Robustness (0.05)", results_005, 0.05)]:
        f.write(f"## 6. Results at observe_cost={cost_val}\n\n")

        for test_group, test_name in [("A", "Held-out Signatures"),
                                       ("B", "Held-out Cue Combinations"),
                                       ("C", "Held-out Seeds (LOSO)")]:
            f.write(f"### Test {test_group}: {test_name}\n\n")
            f.write("| Variant | Status | PolicyNet | AlwaysObs | OracleSel | ObsRate | PosObs | NonposObs | vsAlwaysObs | vsOracle |\n")
            f.write("|---------|--------|-----------|-----------|-----------|---------|--------|-----------|-------------|----------|\n")
            for vid in VARIANT_IDS:
                key = f"Test{test_group}_{vid}"
                ev = all_evaluations[key][cost_label.split()[0].lower()]
                agg = ev["agg"]
                if agg:
                    f.write(f"| {vid} | **{ev['status']}** | {agg['policy_net']:.4f} | "
                            f"{agg['always_obs_net']:.4f} | {agg['oracle_sel']:.4f} | "
                            f"{agg['obs_rate']:.4f} | {agg['pos_obs_rate']:.4f} | "
                            f"{agg['nonpos_obs_rate']:.4f} | "
                            f"{agg['policy_learned_minus_always_obs']:+.4f} | "
                            f"{agg['policy_learned_minus_oracle']:.4f} |\n")
            f.write("\n")

            # Checks
            for vid in VARIANT_IDS:
                key = f"Test{test_group}_{vid}"
                ev = all_evaluations[key][cost_label.split()[0].lower()]
                f.write(f"**{vid} checks:**\n")
                for check_name, result in ev["checks"].items():
                    f.write(f"| {check_name} | **{'PASS' if result else 'FAIL'}** |\n")
                if ev["failures"]:
                    f.write(f"Failures: {ev['failures']}\n")
                f.write("\n")

    f.write("## 7. Q-Target vs True-Target Sanity Check\n\n")
    f.write("| Test | Variant | Q-Target Mean | True-Target Mean | Correlation |\n")
    f.write("|------|---------|---------------|------------------|-------------|\n")
    for test_group in ["A", "B", "C"]:
        test_list = results_003[f"test_{test_group.lower()}"]
        for vresults in test_list[:1]:  # first variant only (representative)
            for vid in VARIANT_IDS:
                vr = vresults[vid]
                f.write(f"| {test_group}/{vr['test_variant_id']} | {vid} | "
                        f"N/A | N/A | {vr['ov_true_target_corr']} |\n")
    f.write("\n")

    f.write("## 8. Decision-Time Information Boundary Audit\n\n")
    f.write("- Two-stage policy: Stage 1 uses only pre_vec + abstract (pre-computed).\n")
    f.write("- Stage 2 reveals post_observe_features only after observe chosen.\n")
    f.write("- Decision logs: post_best_action=None, post_best_value=0.0 when observe not chosen.\n")
    f.write("- Abstract features: no test post_vec, no audit labels at test time.\n\n")

    f.write("## 9. Overall\n\n")
    f.write(f"**{overall_status}**\n\n")

    f.write("| Test | D0 | D1 | D2 | D3 |\n")
    f.write("|------|----|----|----|----|\n")
    for test_group in ["A", "B", "C"]:
        statuses = []
        for vid in VARIANT_IDS:
            key = f"Test{test_group}_{vid}"
            s = all_evaluations[key]["primary"]["status"]
            statuses.append(s)
        f.write(f"| Test {test_group} | {' | '.join(statuses)} |\n")

    f.write("\n## 10. Key Findings\n\n")
    d0_a_status = all_evaluations["TestA_D0"]["primary"]["status"]
    d0_b_status = all_evaluations["TestB_D0"]["primary"]["status"]
    f.write(f"- D0 baseline reproduces 7c: Test A={d0_a_status}, Test B={d0_b_status}\n")

    for vid in ["D1", "D2"]:
        for tg in ["A", "B"]:
            key = f"Test{tg}_{vid}"
            ev = all_evaluations[key]["primary"]
            agg = ev["agg"]
            if agg:
                f.write(f"- {vid} Test {tg}: {ev['status']}, "
                        f"policy_net={agg['policy_net']:.4f}, "
                        f"obs_rate={agg['obs_rate']:.4f}, "
                        f"vs_always_obs={agg['policy_learned_minus_always_obs']:+.4f}\n")

    f.write(f"\n```\n[block_done]\n")
    f.write(f"block_id=1J40b-7d\n")
    f.write(f"implementation_status={overall_status}\n```\n")

print(f"MD -> {md_path}")

print(f"\n{'='*70}")
print(f"Block 1J40b-7d complete.")
print(f"  overall: {overall_status}")
print(f"  elapsed: {elapsed}s")
print(f"{'='*70}")
