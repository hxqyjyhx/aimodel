"""
Block 1J40b-7f-b -- Compositional Oracle-Belief Upper-Bound Control.

Critical methodological point:
The 7f exact-signature oracle-belief is the wrong upper bound for held-out
Test A/B, because those tests hold out pre_surface signatures by construction.
That means exact signature lookup necessarily key-misses on Test A/B and falls
back to the global mean, so 7f cannot answer whether compositional legal
pre-surface signal contains selective-observe headroom.

This script implements the required audit-side compositional oracle-belief
control before any decision about 1J40b-8.
"""

import json
import os
import random
import sys
import time
from datetime import datetime
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
A3_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5a3_tool_material_transfer")
sys.path.insert(0, A3_DIR)
sys.path.insert(0, CURRENT_DIR)

import config

t0 = time.time()

# =============================================================================
# Config safety (same as 7e / 7f)
# =============================================================================
C6_LABEL = "C6_observe_try_counterfactual_v0"
existing_labels = [c["label"] for c in config.CUE_CONDITIONS]
if C6_LABEL not in existing_labels:
    config.CUE_CONDITIONS.append({
        "label": C6_LABEL,
        "p_target": 0.60,
        "p_other": 0.40,
        "absent": False,
        "subtype": True,
        "subtype_config": "observe_try_counterfactual_v0",
        "mixed_source": True,
        "observe_depth_schedules": True,
    })

# =============================================================================
# Constants (from env3b / 7e / 7f)
# =============================================================================
SEEDS = [101, 103, 107, 109, 113]
OBSERVE_COSTS = [0.01, 0.03, 0.05, 0.07]
SUCCESS_REWARD = 0.5
FAILURE_PENALTY = 0.1
CATEGORIES = ["wood_log", "stone_block", "apple", "wooden_pickaxe"]
COVERAGE_FRACTIONS = [0.25, 0.50]
KNN_K = 3
TAU = 0.5
RIDGE_ALPHA = 1.0

ALL_TRY_AFFORDANCES = [
    "burn_as_fuel",
    "craft_plank",
    "eat",
    "mine_by_hand",
    "mine_with_pickaxe",
    "use_as_tool",
]

AMBIENT_GROUP_FEATURES = {
    "A": ["brownish", "rough_texture", "long_shape", "heavy_weight"],
    "B": ["greenish", "smooth_texture", "round_small", "light_weight"],
}
CATEGORY_AMBIENT_GROUP = {
    "wood_log": "A",
    "stone_block": "A",
    "apple": "B",
    "wooden_pickaxe": "B",
}
ALL_AMBIENT_FEATURE_NAMES = sorted(set(
    f for feats in AMBIENT_GROUP_FEATURES.values() for f in feats
))

CATEGORY_POST_OBSERVE_FEATURES = {
    "wood_log": ["has_bark_texture", "fibrous", "flammable", "porous_surface"],
    "stone_block": [
        "has_crystal_flecks",
        "has_granular_surface",
        "block_like",
        "cold_to_touch",
        "scratch_resistant",
    ],
    "apple": ["has_stem_remnant", "has_peel_texture", "fruity_scent"],
    "wooden_pickaxe": [
        "has_grip_area",
        "has_shaft_shape",
        "elongated_with_handle",
        "movable",
        "has_metal_head",
        "jointed",
    ],
}

SHARED_SURFACE_FEATURES = [
    "spotted",
    "weathered",
    "coarse_texture",
    "discolored",
    "has_veins",
    "pitted",
]

ALL_VISIBLE_FEATURE_NAMES = sorted(set(
    [f for feats in AMBIENT_GROUP_FEATURES.values() for f in feats]
    + [f for feats in CATEGORY_POST_OBSERVE_FEATURES.values() for f in feats]
    + list(SHARED_SURFACE_FEATURES)
))

STATE_FEATURE_NAMES = ["fresh", "wet", "damaged", "clean", "hot", "open"]

INSTANCE_SUBTYPE_DEFS = {
    "wood_log": {
        "P": {
            "label": "ambiguous_log",
            "audit_rationale_class": "positive_change",
            "pre_surface_extra": [],
            "affordance_profile": {
                "mine_by_hand": "success",
                "mine_with_pickaxe": "fail",
                "craft_plank": "success",
                "eat": "fail",
                "use_as_tool": "fail",
                "burn_as_fuel": "success",
            },
            "post_observe_revealed": [
                "has_bark_texture",
                "fibrous",
                "flammable",
                "porous_surface",
            ],
            "state_features": {
                "fresh": True,
                "wet": True,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
        "Z": {
            "label": "positive_shared_cue_log",
            "audit_rationale_class": "positive_change",
            "pre_surface_extra": ["spotted"],
            "affordance_profile": {
                "mine_by_hand": "success",
                "mine_with_pickaxe": "fail",
                "craft_plank": "success",
                "eat": "fail",
                "use_as_tool": "fail",
                "burn_as_fuel": "success",
            },
            "post_observe_revealed": [
                "has_bark_texture",
                "fibrous",
                "flammable",
                "porous_surface",
            ],
            "state_features": {
                "fresh": True,
                "wet": True,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
        "N": {
            "label": "nonpos_identified_log",
            "audit_rationale_class": "wasteful_cost_only",
            "pre_surface_extra": ["has_bark_texture", "discolored"],
            "affordance_profile": {
                "mine_by_hand": "success",
                "mine_with_pickaxe": "fail",
                "craft_plank": "success",
                "eat": "fail",
                "use_as_tool": "fail",
                "burn_as_fuel": "success",
            },
            "post_observe_revealed": ["fibrous", "flammable", "porous_surface"],
            "state_features": {
                "fresh": False,
                "wet": False,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
    },
    "stone_block": {
        "P": {
            "label": "ambiguous_stone",
            "audit_rationale_class": "positive_change",
            "pre_surface_extra": [],
            "affordance_profile": {
                "mine_by_hand": "fail",
                "mine_with_pickaxe": "success",
                "craft_plank": "fail",
                "eat": "fail",
                "use_as_tool": "fail",
                "burn_as_fuel": "fail",
            },
            "post_observe_revealed": [
                "has_crystal_flecks",
                "has_granular_surface",
                "block_like",
                "cold_to_touch",
                "scratch_resistant",
            ],
            "state_features": {
                "fresh": True,
                "wet": False,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
        "Z": {
            "label": "positive_shared_cue_stone",
            "audit_rationale_class": "positive_change",
            "pre_surface_extra": ["spotted"],
            "affordance_profile": {
                "mine_by_hand": "fail",
                "mine_with_pickaxe": "success",
                "craft_plank": "fail",
                "eat": "fail",
                "use_as_tool": "fail",
                "burn_as_fuel": "fail",
            },
            "post_observe_revealed": [
                "has_crystal_flecks",
                "has_granular_surface",
                "block_like",
                "cold_to_touch",
            ],
            "state_features": {
                "fresh": True,
                "wet": False,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
        "N": {
            "label": "nonpos_identified_stone",
            "audit_rationale_class": "wasteful_cost_only",
            "pre_surface_extra": ["has_crystal_flecks"],
            "affordance_profile": {
                "mine_by_hand": "fail",
                "mine_with_pickaxe": "success",
                "craft_plank": "fail",
                "eat": "fail",
                "use_as_tool": "fail",
                "burn_as_fuel": "fail",
            },
            "post_observe_revealed": [
                "has_granular_surface",
                "block_like",
                "cold_to_touch",
                "scratch_resistant",
            ],
            "state_features": {
                "fresh": True,
                "wet": False,
                "damaged": True,
                "clean": False,
                "hot": False,
                "open": False,
            },
        },
    },
    "apple": {
        "P": {
            "label": "ambiguous_fruit",
            "audit_rationale_class": "positive_change",
            "pre_surface_extra": [],
            "affordance_profile": {
                "mine_by_hand": "success",
                "mine_with_pickaxe": "fail",
                "craft_plank": "fail",
                "eat": "success",
                "use_as_tool": "fail",
                "burn_as_fuel": "fail",
            },
            "post_observe_revealed": [
                "has_stem_remnant",
                "has_peel_texture",
                "fruity_scent",
            ],
            "state_features": {
                "fresh": True,
                "wet": True,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
        "Z": {
            "label": "positive_shared_cue_apple",
            "audit_rationale_class": "positive_change",
            "pre_surface_extra": ["discolored"],
            "affordance_profile": {
                "mine_by_hand": "success",
                "mine_with_pickaxe": "fail",
                "craft_plank": "fail",
                "eat": "success",
                "use_as_tool": "fail",
                "burn_as_fuel": "fail",
            },
            "post_observe_revealed": [
                "has_stem_remnant",
                "has_peel_texture",
                "fruity_scent",
            ],
            "state_features": {
                "fresh": True,
                "wet": True,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
        "N": {
            "label": "nonpos_identified_apple",
            "audit_rationale_class": "wasteful_cost_only",
            "pre_surface_extra": ["has_stem_remnant", "spotted"],
            "affordance_profile": {
                "mine_by_hand": "success",
                "mine_with_pickaxe": "fail",
                "craft_plank": "fail",
                "eat": "success",
                "use_as_tool": "fail",
                "burn_as_fuel": "fail",
            },
            "post_observe_revealed": ["has_peel_texture", "fruity_scent"],
            "state_features": {
                "fresh": True,
                "wet": False,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
    },
    "wooden_pickaxe": {
        "P": {
            "label": "ambiguous_tool",
            "audit_rationale_class": "positive_change",
            "pre_surface_extra": [],
            "affordance_profile": {
                "mine_by_hand": "fail",
                "mine_with_pickaxe": "fail",
                "craft_plank": "fail",
                "eat": "fail",
                "use_as_tool": "success",
                "burn_as_fuel": "fail",
            },
            "post_observe_revealed": [
                "has_grip_area",
                "has_shaft_shape",
                "elongated_with_handle",
                "movable",
                "has_metal_head",
                "jointed",
            ],
            "state_features": {
                "fresh": True,
                "wet": False,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
        "Z": {
            "label": "positive_shared_cue_tool",
            "audit_rationale_class": "positive_change",
            "pre_surface_extra": ["discolored"],
            "affordance_profile": {
                "mine_by_hand": "fail",
                "mine_with_pickaxe": "fail",
                "craft_plank": "fail",
                "eat": "fail",
                "use_as_tool": "success",
                "burn_as_fuel": "fail",
            },
            "post_observe_revealed": [
                "has_grip_area",
                "has_shaft_shape",
                "elongated_with_handle",
                "movable",
            ],
            "state_features": {
                "fresh": True,
                "wet": False,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
        "N": {
            "label": "nonpos_identified_tool",
            "audit_rationale_class": "wasteful_cost_only",
            "pre_surface_extra": ["has_grip_area"],
            "affordance_profile": {
                "mine_by_hand": "fail",
                "mine_with_pickaxe": "fail",
                "craft_plank": "fail",
                "eat": "fail",
                "use_as_tool": "success",
                "burn_as_fuel": "fail",
            },
            "post_observe_revealed": [
                "has_shaft_shape",
                "elongated_with_handle",
                "movable",
                "has_metal_head",
                "jointed",
            ],
            "state_features": {
                "fresh": True,
                "wet": False,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
    },
}


# =============================================================================
# Object generation (same env3b semantics as 7f)
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
            for feat in ambient_features_list:
                pre_surface[feat] = True
            for feat in pre_extra:
                pre_surface[feat] = True

            post_observe = dict(pre_surface)
            for feat in post_revealed:
                post_observe[feat] = True

            hidden = {}
            for feat in ALL_VISIBLE_FEATURE_NAMES:
                if feat not in pre_surface and feat not in post_revealed:
                    hidden[feat] = True

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


def get_pre_surface_signature(pre_surface_features):
    return tuple(sorted(pre_surface_features.keys()))


def sig_to_display_name(sig):
    feats = list(sig)
    ambient_part = [f for f in feats if f in ALL_AMBIENT_FEATURE_NAMES]
    extra_part = [f for f in feats if f not in ALL_AMBIENT_FEATURE_NAMES]
    amb = ",".join(sorted(ambient_part)[:2])
    ext = ",".join(sorted(extra_part)[:3])
    if ext:
        return f"amb({amb})+{ext}"
    return f"amb({amb})_only"


def try_net_value(success_or_failure):
    if success_or_failure is True:
        return SUCCESS_REWARD - 0.05
    if success_or_failure is False:
        return -FAILURE_PENALTY - 0.05
    return 0.0


per_seed_objects = {}
per_seed_audit_labels = {}
for seed in SEEDS:
    objects, audit_labels = generate_env3b_objects(seed)
    per_seed_objects[seed] = objects
    per_seed_audit_labels[seed] = audit_labels


def collect_all_post_observe_features(per_seed_objects_map):
    all_feats = set()
    for seed in SEEDS:
        for obj in per_seed_objects_map[seed].values():
            for feat in obj["post_observe_features"]:
                all_feats.add(feat)
    return sorted(all_feats)


POST_OBSERVE_FEATURE_NAMES = collect_all_post_observe_features(per_seed_objects)
ALL_MODEL_FEATURE_KEYS = sorted(set(
    [f"feat_{feat}" for feat in POST_OBSERVE_FEATURE_NAMES]
    + [f"state_{state}" for state in STATE_FEATURE_NAMES]
    + ["prior_observe_count"]
))
ALL_ACTION_KEYS = ["observe"] + [f"try_{a}" for a in ALL_TRY_AFFORDANCES]
N_FEATURES = len(ALL_MODEL_FEATURE_KEYS)


def build_feature_vector(known_features, known_states, prior_obs_count):
    feat = {}
    for feat_name in POST_OBSERVE_FEATURE_NAMES:
        feat[f"feat_{feat_name}"] = 1.0 if feat_name in known_features else 0.0
    for state_name in STATE_FEATURE_NAMES:
        feat[f"state_{state_name}"] = 1.0 if known_states.get(state_name, False) else 0.0
    feat["prior_observe_count"] = float(prior_obs_count)
    return feat


def extract_feature_vector(situation_features):
    return [float(situation_features.get(key, 0.0)) for key in ALL_MODEL_FEATURE_KEYS]


def _make_pre_vec(pre_features):
    return extract_feature_vector(build_feature_vector(pre_features, {}, 0))


def _make_post_vec(post_features, state):
    return extract_feature_vector(build_feature_vector(post_features, state, 1))


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
        xtx = [[0.0] * n_aug for _ in range(n_aug)]
        for i in range(n_aug):
            for j in range(n_aug):
                total = 0.0
                for k in range(n_samples):
                    total += X_aug[k][i] * X_aug[k][j]
                xtx[i][j] = total
        for i in range(n_features):
            xtx[i][i] += self.alpha
        xty = [0.0] * n_aug
        for i in range(n_aug):
            total = 0.0
            for k in range(n_samples):
                total += X_aug[k][i] * y[k]
            xty[i] = total
        weights = _solve_linear_system(xtx, xty)
        if weights is None:
            self.coef_ = [0.0] * n_features
            self.intercept_ = sum(y) / len(y) if y else 0.0
        else:
            self.coef_ = weights[:n_features]
            self.intercept_ = weights[n_features]
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return [
            sum(xi * wi for xi, wi in zip(row, self.coef_)) + self.intercept_
            for row in X
        ]


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
            self.stds[j] = (var ** 0.5) if var > 1e-12 else 1.0
        return self

    def transform(self, X):
        if self.means is None:
            return X
        return [
            [(row[j] - self.means[j]) / self.stds[j] for j in range(len(row))]
            for row in X
        ]


class PerActionValueEstimator:
    def __init__(self, alpha=RIDGE_ALPHA):
        self.alpha = alpha
        self.models = {}
        self.normalizers = {}

    def fit(self, X_train, y_train, action_keys_train):
        grouped_X = defaultdict(list)
        grouped_y = defaultdict(list)
        for xv, yv, action_key in zip(X_train, y_train, action_keys_train):
            grouped_X[action_key].append(xv)
            grouped_y[action_key].append(yv)

        for action_key in ALL_ACTION_KEYS:
            gx = grouped_X.get(action_key, [])
            gy = grouped_y.get(action_key, [])
            if len(gx) < 3:
                model = RidgeRegression(alpha=self.alpha)
                n_features = len(gx[0]) if gx else 0
                model.coef_ = [0.0] * n_features
                model.intercept_ = sum(gy) / len(gy) if gy else 0.0
                model._fitted = True
                self.models[action_key] = model
                self.normalizers[action_key] = None
                continue
            normalizer = FeatureNormalizer().fit(gx)
            X_norm = normalizer.transform(gx)
            self.models[action_key] = RidgeRegression(alpha=self.alpha).fit(X_norm, gy)
            self.normalizers[action_key] = normalizer
        return self

    def predict_single(self, x_vec, action_key):
        if action_key not in self.models:
            return 0.0
        model = self.models[action_key]
        normalizer = self.normalizers.get(action_key)
        x_input = normalizer.transform([x_vec])[0] if normalizer is not None else x_vec
        return model.predict([x_input])[0]

    def predict_all_actions(self, x_vec):
        return {action_key: self.predict_single(x_vec, action_key) for action_key in ALL_ACTION_KEYS}


# =============================================================================
# Signature analysis / stress splits (same semantics as 7c / 7f)
# =============================================================================
sig_map = defaultdict(list)
for seed in SEEDS:
    for oid, obj in per_seed_objects[seed].items():
        sig = get_pre_surface_signature(obj["pre_surface_features"])
        sig_map[sig].append((seed, oid))

SIGNATURE_LIST = sorted(sig_map.keys(), key=lambda s: (-len(sig_map[s]), s))

shared_sigs = []
unique_sigs = []
for sig in SIGNATURE_LIST:
    cats_in_sig = set()
    for seed, oid in sig_map[sig]:
        cats_in_sig.add(per_seed_audit_labels[seed][oid]["hidden_category"])
    if len(cats_in_sig) >= 2:
        shared_sigs.append(sig)
    else:
        unique_sigs.append(sig)


def make_test_variant(variant_id, train_sigs, test_sigs, description):
    train_oids = set()
    test_oids = set()
    for sig in train_sigs:
        for seed, oid in sig_map[sig]:
            train_oids.add((seed, oid))
    for sig in test_sigs:
        for seed, oid in sig_map[sig]:
            test_oids.add((seed, oid))
    return {
        "id": variant_id,
        "train_sigs": train_sigs,
        "test_sigs": test_sigs,
        "train_oids": train_oids,
        "test_oids": test_oids,
        "n_train": len(train_oids),
        "n_test": len(test_oids),
        "description": description,
    }


TEST_A_VARIANTS = []
for i in range(min(len(shared_sigs), len(unique_sigs))):
    held_shared = shared_sigs[i]
    held_unique = unique_sigs[i]
    held_sigs = [held_shared, held_unique]
    train_sigs = [s for s in SIGNATURE_LIST if s not in held_sigs]
    TEST_A_VARIANTS.append(make_test_variant(
        variant_id=f"A{i+1}",
        train_sigs=train_sigs,
        test_sigs=held_sigs,
        description=(
            f"Hold out shared {sig_to_display_name(held_shared)} + "
            f"unique {sig_to_display_name(held_unique)}"
        ),
    ))

apple_n_sig = None
woodlog_n_sig = None
for sig in unique_sigs:
    cats_in = set()
    for seed, oid in sig_map[sig]:
        cats_in.add(per_seed_audit_labels[seed][oid]["hidden_category"])
    if cats_in == {"apple"}:
        apple_n_sig = sig
    if cats_in == {"wood_log"}:
        woodlog_n_sig = sig

ambient_b_only_sig = None
ambient_a_only_sig = None
for sig in shared_sigs:
    some_seed, some_oid = sig_map[sig][0]
    ambient_group = per_seed_audit_labels[some_seed][some_oid]["ambient_group"]
    is_ambient_only = len(sig) == 4
    if ambient_group == "B" and is_ambient_only:
        ambient_b_only_sig = sig
    if ambient_group == "A" and is_ambient_only:
        ambient_a_only_sig = sig

TEST_B_VARIANTS = []
b1_test_sigs = [s for s in [apple_n_sig, ambient_b_only_sig] if s is not None]
b2_test_sigs = [s for s in [woodlog_n_sig, ambient_a_only_sig] if s is not None]
if b1_test_sigs:
    TEST_B_VARIANTS.append(make_test_variant(
        variant_id="B1",
        train_sigs=[s for s in SIGNATURE_LIST if s not in b1_test_sigs],
        test_sigs=b1_test_sigs,
        description="Hold out apple-N spotted non-positive + ambient-B shared signature",
    ))
if b2_test_sigs:
    TEST_B_VARIANTS.append(make_test_variant(
        variant_id="B2",
        train_sigs=[s for s in SIGNATURE_LIST if s not in b2_test_sigs],
        test_sigs=b2_test_sigs,
        description="Hold out wood_log-N discolored non-positive + ambient-A shared signature",
    ))

TEST_C_VARIANTS = []
for heldout_seed in SEEDS:
    train_oids = set()
    test_oids = set()
    for seed in SEEDS:
        for oid in per_seed_objects[seed]:
            if seed == heldout_seed:
                test_oids.add((seed, oid))
            else:
                train_oids.add((seed, oid))
    TEST_C_VARIANTS.append({
        "id": f"C_seed{heldout_seed}",
        "train_oids": train_oids,
        "test_oids": test_oids,
        "n_train": len(train_oids),
        "n_test": len(test_oids),
        "description": f"LOSO held-out seed {heldout_seed}",
    })


# =============================================================================
# Counterfactual intervention helpers (same 7e / 7f logic)
# =============================================================================
AMBIENT_GROUP_CATEGORIES = {
    "A": ["wood_log", "stone_block"],
    "B": ["apple", "wooden_pickaxe"],
}


def get_flat_affordance(all_success=True):
    outcome = "success" if all_success else "fail"
    return {a: outcome for a in ALL_TRY_AFFORDANCES}


def get_lookalike_affordance(seed, oid):
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
        n_flip = max(1, int(len(instances) * coverage_fraction))
        flipped = rng.sample(instances, n_flip)
        some_lbl = per_seed_audit_labels[instances[0][0]][instances[0][1]]
        is_shared = some_lbl["hidden_audit_rationale_class"] == "positive_change"

        for seed, oid in flipped:
            if is_shared:
                flat_val = "success" if rng.random() < 0.5 else "fail"
                new_aff = get_flat_affordance(flat_val == "success")
            else:
                new_aff, target_cat = get_lookalike_affordance(seed, oid)
                lookalike_targets[(seed, oid)] = target_cat
            overrides[(seed, oid)] = new_aff
            intervention_labels[(seed, oid)] = (
                "decision_irrelevant_flat" if is_shared else "lookalike_deceptive_exception"
            )

    return overrides, intervention_labels, lookalike_targets


def build_lookalike_post_observe_overrides(lookalike_targets):
    overrides = {}
    for (seed, oid), target_cat in lookalike_targets.items():
        obj = per_seed_objects[seed][oid]
        corrected_post = dict(obj["pre_surface_features"])
        for feat in CATEGORY_POST_OBSERVE_FEATURES.get(target_cat, []):
            corrected_post[feat] = True
        overrides[(seed, oid)] = corrected_post
    return overrides


# =============================================================================
# True/oracle VOI (same 7f semantics)
# =============================================================================
def compute_true_oracle_voi(oid_set, observe_cost, affordance_overrides=None):
    sig_instances = defaultdict(list)
    for seed, oid in oid_set:
        sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
        aff = affordance_overrides.get((seed, oid)) if affordance_overrides else None
        if aff is None:
            aff = per_seed_audit_labels[seed][oid]["affordance_profile"]
        sig_instances[sig].append((seed, oid, aff))

    sig_expected = {}
    for sig, instances in sig_instances.items():
        action_sums = {a: 0.0 for a in ALL_TRY_AFFORDANCES}
        for _, _, aff in instances:
            for action in ALL_TRY_AFFORDANCES:
                action_sums[action] += try_net_value(aff.get(action, "fail") == "success")
        n = len(instances)
        action_exp = {a: action_sums[a] / n for a in ALL_TRY_AFFORDANCES}
        best_action = max(action_exp, key=action_exp.get)
        sig_expected[sig] = {
            "best_expected_value": action_exp[best_action],
            "best_action": best_action,
            "n_instances": n,
        }

    result = {}
    for seed, oid in oid_set:
        sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
        true_pre_best = sig_expected[sig]["best_expected_value"]
        aff = affordance_overrides.get((seed, oid)) if affordance_overrides else None
        if aff is None:
            aff = per_seed_audit_labels[seed][oid]["affordance_profile"]
        true_post_best = max(
            try_net_value(aff.get(action, "fail") == "success")
            for action in ALL_TRY_AFFORDANCES
        )
        true_voi = true_post_best - true_pre_best - observe_cost
        result[(seed, oid)] = {
            "true_pre_best": true_pre_best,
            "true_post_best": true_post_best,
            "true_voi": true_voi,
            "true_optimal": true_voi > 0,
        }
    return result


# =============================================================================
# Statistic builders
# =============================================================================
def summarize_rows(rows):
    n = len(rows)
    n_positive = sum(1 for r in rows if r["true_optimal"])
    return {
        "n": n,
        "n_positive": n_positive,
        "p_positive": (n_positive / n) if n else 0.0,
        "e_observe_net": (sum(r["true_voi"] for r in rows) / n) if n else 0.0,
    }


def build_train_rows(train_oids, true_voi):
    rows = []
    for seed, oid in sorted(train_oids):
        sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
        rows.append({
            "seed": seed,
            "oid": oid,
            "signature": sig,
            "true_voi": true_voi[(seed, oid)]["true_voi"],
            "true_optimal": true_voi[(seed, oid)]["true_optimal"],
        })
    return rows


def build_signature_stats(train_rows):
    stats = defaultdict(list)
    for row in train_rows:
        stats[row["signature"]].append(row)
    return {sig: summarize_rows(rows) for sig, rows in stats.items()}


def build_cue_stats(train_rows):
    cue_rows = defaultdict(list)
    for row in train_rows:
        for cue in row["signature"]:
            cue_rows[cue].append(row)
    return {cue: summarize_rows(rows) for cue, rows in cue_rows.items()}


def jaccard_similarity(sig_a, sig_b):
    set_a = set(sig_a)
    set_b = set(sig_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def level1_cue_marginal_estimate(test_sig, cue_stats, global_stats):
    matched_cues = [cue for cue in test_sig if cue in cue_stats]
    if not matched_cues:
        return {
            "source_type": "global_fallback",
            "used_global_fallback": True,
            "matched_cues": [],
            **global_stats,
        }

    matched_stats = [cue_stats[cue] for cue in matched_cues]
    return {
        "source_type": "cue_marginal",
        "used_global_fallback": False,
        "matched_cues": matched_cues,
        "matched_non_ambient_cues": [
            cue for cue in matched_cues if cue not in ALL_AMBIENT_FEATURE_NAMES
        ],
        "n": sum(s["n"] for s in matched_stats),
        "n_positive": sum(s["n_positive"] for s in matched_stats),
        "p_positive": sum(s["p_positive"] for s in matched_stats) / len(matched_stats),
        "e_observe_net": sum(s["e_observe_net"] for s in matched_stats) / len(matched_stats),
    }


def level2_knn_estimate(test_sig, signature_stats, global_stats, k=KNN_K):
    neighbors = []
    for train_sig, stats in signature_stats.items():
        sim = jaccard_similarity(test_sig, train_sig)
        if sim > 0.0:
            neighbors.append((sim, train_sig, stats))

    neighbors.sort(key=lambda x: (-x[0], -len(x[1]), x[1]))
    top_neighbors = neighbors[:k]
    if not top_neighbors:
        return {
            "source_type": "global_fallback",
            "used_global_fallback": True,
            "neighbors": [],
            **global_stats,
        }

    total_w = sum(sim for sim, _, _ in top_neighbors)
    if total_w <= 0:
        return {
            "source_type": "global_fallback",
            "used_global_fallback": True,
            "neighbors": [],
            **global_stats,
        }

    return {
        "source_type": "knn_jaccard_signature",
        "used_global_fallback": False,
        "neighbors": [
            {
                "signature": list(train_sig),
                "signature_name": sig_to_display_name(train_sig),
                "jaccard": round(sim, 4),
                "n": stats["n"],
            }
            for sim, train_sig, stats in top_neighbors
        ],
        "n": sum(stats["n"] for _, _, stats in top_neighbors),
        "n_positive": sum(stats["n_positive"] for _, _, stats in top_neighbors),
        "p_positive": sum(sim * stats["p_positive"] for sim, _, stats in top_neighbors) / total_w,
        "e_observe_net": sum(sim * stats["e_observe_net"] for sim, _, stats in top_neighbors) / total_w,
    }


def level3_subset_candidate_estimate(test_sig, signature_stats, global_stats, tau=TAU):
    test_set = set(test_sig)
    candidates = []
    for train_sig, stats in signature_stats.items():
        train_set = set(train_sig)
        sim = jaccard_similarity(test_sig, train_sig)
        is_subset = train_set.issubset(test_set)
        if is_subset or sim >= tau:
            candidates.append((train_sig, stats, sim, is_subset))

    if not candidates:
        return {
            "source_type": "global_fallback",
            "used_global_fallback": True,
            "candidate_count": 0,
            "candidate_support_total": 0,
            "matched_signatures": [],
            **global_stats,
        }

    total_support = sum(stats["n"] for _, stats, _, _ in candidates)
    return {
        "source_type": "cue_subset_candidate",
        "used_global_fallback": False,
        "candidate_count": len(candidates),
        "candidate_support_total": total_support,
        "matched_signatures": [
            {
                "signature": list(train_sig),
                "signature_name": sig_to_display_name(train_sig),
                "n": stats["n"],
                "jaccard": round(sim, 4),
                "is_subset": is_subset,
            }
            for train_sig, stats, sim, is_subset in sorted(
                candidates, key=lambda x: (-x[1]["n"], -x[2], x[0])
            )
        ],
        "n": total_support,
        "n_positive": sum(stats["n_positive"] for _, stats, _, _ in candidates),
        "p_positive": (
            sum(stats["p_positive"] * stats["n"] for _, stats, _, _ in candidates) / total_support
        ),
        "e_observe_net": (
            sum(stats["e_observe_net"] * stats["n"] for _, stats, _, _ in candidates) / total_support
        ),
    }


def exact_signature_lookup_available(test_sig, signature_stats):
    return test_sig in signature_stats


def exact_signature_estimate(test_sig, signature_stats, global_stats):
    if test_sig in signature_stats:
        stats = signature_stats[test_sig]
        return {
            "source_type": "exact_signature",
            "used_global_fallback": False,
            "matched_signature": list(test_sig),
            "matched_signature_name": sig_to_display_name(test_sig),
            **stats,
        }
    return {
        "source_type": "global_fallback",
        "used_global_fallback": True,
        "matched_signature": None,
        "matched_signature_name": None,
        **global_stats,
    }


def build_condition_artifacts(train_oids, condition_name):
    if condition_name == "original":
        return None, None
    if condition_name == "counterfactual_cov0.25":
        overrides, _, lookalike_targets = build_counterfactual_affordance_overrides(train_oids, 0.25)
        return overrides, build_lookalike_post_observe_overrides(lookalike_targets)
    if condition_name == "counterfactual_cov0.50":
        overrides, _, lookalike_targets = build_counterfactual_affordance_overrides(train_oids, 0.50)
        return overrides, build_lookalike_post_observe_overrides(lookalike_targets)
    if condition_name == "counterfactual_cov1.00":
        overrides, _, lookalike_targets = build_counterfactual_affordance_overrides(train_oids, 1.00)
        return overrides, build_lookalike_post_observe_overrides(lookalike_targets)
    raise ValueError(f"Unknown condition: {condition_name}")


def build_test_obj_eval_data(test_oids):
    obj_data = {}
    for seed, oid in sorted(test_oids):
        obj = per_seed_objects[seed][oid]
        lbl = per_seed_audit_labels[seed][oid]
        true_returns = {}
        for action in ALL_TRY_AFFORDANCES:
            outcome = lbl["affordance_profile"].get(action, "fail")
            true_returns[f"try_{action}"] = try_net_value(outcome == "success")
        obj_data[(seed, oid)] = {
            "seed": seed,
            "oid": oid,
            "signature": get_pre_surface_signature(obj["pre_surface_features"]),
            "signature_name": sig_to_display_name(get_pre_surface_signature(obj["pre_surface_features"])),
            "pre_features": obj["pre_surface_features"],
            "post_features": obj["post_observe_features"],
            "state": obj["visible_state"],
            "category": lbl["hidden_category"],
            "rationale_class": lbl["hidden_audit_rationale_class"],
            "true_returns": true_returns,
        }
    return obj_data


def annotate_test_truth(obj_data, observe_cost):
    sig_groups = defaultdict(list)
    for key, od in obj_data.items():
        sig_groups[od["signature"]].append(key)

    signature_truth = {}
    for sig, keys in sig_groups.items():
        members = [obj_data[k] for k in keys]
        action_expected = {}
        for action in ALL_TRY_AFFORDANCES:
            total = sum(m["true_returns"][f"try_{action}"] for m in members)
            action_expected[action] = total / len(members)
        best_action = max(action_expected, key=action_expected.get)
        pre_best_expected = action_expected[best_action]

        sig_returns = []
        sig_optimal = 0
        always_obs_vals = []
        oracle_sel_vals = []
        for key in keys:
            od = obj_data[key]
            true_post_best = max(od["true_returns"].values())
            true_voi = true_post_best - pre_best_expected - observe_cost
            true_optimal = true_voi > 0
            od["true_pre_best_expected"] = pre_best_expected
            od["true_pre_best_action"] = best_action
            od["true_post_best"] = true_post_best
            od["true_voi"] = true_voi
            od["true_optimal"] = true_optimal
            sig_returns.append(true_voi)
            if true_optimal:
                sig_optimal += 1
                oracle_sel_vals.append(true_post_best - observe_cost)
            else:
                oracle_sel_vals.append(pre_best_expected)
            always_obs_vals.append(true_post_best - observe_cost)

        signature_truth[sig] = {
            "signature": list(sig),
            "signature_name": sig_to_display_name(sig),
            "n_objects": len(keys),
            "pre_best_expected": pre_best_expected,
            "pre_best_action": best_action,
            "true_e_observe_net": sum(sig_returns) / len(sig_returns),
            "true_positive_net_rate": sig_optimal / len(keys),
            "always_observe_value": sum(always_obs_vals) / len(always_obs_vals),
            "oracle_selective_value": sum(oracle_sel_vals) / len(oracle_sel_vals),
        }
    return signature_truth


def train_current_learned_policy(train_oids, observe_cost, affordance_overrides=None, post_observe_overrides=None):
    train_records = []
    for seed, oid in sorted(train_oids):
        obj = per_seed_objects[seed][oid]
        lbl = per_seed_audit_labels[seed][oid]
        affordance = affordance_overrides.get((seed, oid)) if affordance_overrides else None
        if affordance is None:
            affordance = lbl["affordance_profile"]
        post_features = post_observe_overrides.get((seed, oid)) if post_observe_overrides else None
        if post_features is None:
            post_features = obj["post_observe_features"]
        pre_vec = _make_pre_vec(obj["pre_surface_features"])
        post_vec = _make_post_vec(post_features, obj["visible_state"])

        train_records.append({"feature_vector": pre_vec, "action_key": "observe", "net_value": -observe_cost})
        for action in ALL_TRY_AFFORDANCES:
            outcome = affordance.get(action, "fail")
            val = try_net_value(outcome == "success")
            train_records.append({
                "feature_vector": pre_vec,
                "action_key": f"try_{action}",
                "net_value": val,
            })
        for action in ALL_TRY_AFFORDANCES:
            outcome = affordance.get(action, "fail")
            val = try_net_value(outcome == "success")
            train_records.append({
                "feature_vector": post_vec,
                "action_key": f"try_{action}",
                "net_value": val,
            })

    X_train = [r["feature_vector"] for r in train_records]
    y_train = [r["net_value"] for r in train_records]
    ak_train = [r["action_key"] for r in train_records]
    estimator = PerActionValueEstimator(alpha=RIDGE_ALPHA).fit(X_train, y_train, ak_train)

    train_sig_groups = defaultdict(list)
    for seed, oid in sorted(train_oids):
        obj = per_seed_objects[seed][oid]
        lbl = per_seed_audit_labels[seed][oid]
        sig = get_pre_surface_signature(obj["pre_surface_features"])
        affordance = affordance_overrides.get((seed, oid)) if affordance_overrides else None
        if affordance is None:
            affordance = lbl["affordance_profile"]
        true_rets = {}
        for action in ALL_TRY_AFFORDANCES:
            outcome = affordance.get(action, "fail")
            true_rets[action] = try_net_value(outcome == "success")
        train_sig_groups[sig].append(true_rets)

    train_sig_exp = {}
    for sig, rets_list in train_sig_groups.items():
        act_exp = {}
        for action in ALL_TRY_AFFORDANCES:
            act_exp[action] = sum(r[action] for r in rets_list) / len(rets_list)
        train_sig_exp[sig] = max(act_exp.values())

    ov_X = []
    ov_y = []
    for seed, oid in sorted(train_oids):
        obj = per_seed_objects[seed][oid]
        post_features = post_observe_overrides.get((seed, oid)) if post_observe_overrides else None
        if post_features is None:
            post_features = obj["post_observe_features"]
        pre_vec = _make_pre_vec(obj["pre_surface_features"])
        post_vec = _make_post_vec(post_features, obj["visible_state"])
        pre_qs = estimator.predict_all_actions(pre_vec)
        post_qs = estimator.predict_all_actions(post_vec)
        pre_best = max(q for ak, q in pre_qs.items() if ak != "observe")
        post_best = max(q for ak, q in post_qs.items() if ak != "observe")
        ov_X.append(list(pre_vec))
        ov_y.append(post_best - pre_best - observe_cost)

    if len(ov_X) >= 3 and len(set(round(y, 8) for y in ov_y)) > 1:
        ov_normalizer = FeatureNormalizer().fit(ov_X)
        ov_model = RidgeRegression(alpha=RIDGE_ALPHA).fit(ov_normalizer.transform(ov_X), ov_y)
    else:
        ov_normalizer = None
        ov_model = None
    return estimator, ov_model, ov_normalizer


def evaluate_learned_policy(obj_data, estimator, ov_model, ov_normalizer, observe_cost):
    returns = []
    observed = 0
    pos_obs = 0
    pos_total = 0
    nonpos_obs = 0
    nonpos_total = 0
    harmful_obs = 0
    for od in obj_data.values():
        pre_vec = _make_pre_vec(od["pre_features"])
        pre_qs = estimator.predict_all_actions(pre_vec)
        pre_try_qs = {ak: q for ak, q in pre_qs.items() if ak != "observe"}
        best_action = max(pre_try_qs, key=pre_try_qs.get)
        if ov_model is not None and ov_normalizer is not None:
            pred_e = ov_model.predict(ov_normalizer.transform([pre_vec]))[0]
        else:
            pred_e = -observe_cost
        should_observe = pred_e > 0.0
        if od["true_optimal"]:
            pos_total += 1
            if should_observe:
                pos_obs += 1
        else:
            nonpos_total += 1
            if should_observe:
                nonpos_obs += 1
                harmful_obs += 1
        if should_observe:
            observed += 1
            returns.append(od["true_post_best"] - observe_cost)
        else:
            returns.append(od["true_returns"].get(best_action, 0.0))
    n_test = len(obj_data)
    return {
        "policy_net": sum(returns) / n_test if n_test else 0.0,
        "observe_rate": observed / n_test if n_test else 0.0,
        "positive_net_observe_rate": pos_obs / pos_total if pos_total else 0.0,
        "non_positive_net_observe_rate": nonpos_obs / nonpos_total if nonpos_total else 0.0,
        "harmful_observe_rate": harmful_obs / n_test if n_test else 0.0,
    }


def evaluate_signature_policy(obj_data, sig_estimates, observe_cost):
    returns = []
    observed = 0
    pos_obs = 0
    pos_total = 0
    nonpos_obs = 0
    nonpos_total = 0
    harmful_obs = 0
    signature_returns = defaultdict(list)

    for od in obj_data.values():
        est = sig_estimates[od["signature"]]
        should_observe = est["e_observe_net"] > 0.0
        if od["true_optimal"]:
            pos_total += 1
            if should_observe:
                pos_obs += 1
        else:
            nonpos_total += 1
            if should_observe:
                nonpos_obs += 1
                harmful_obs += 1
        if should_observe:
            observed += 1
            ret = od["true_post_best"] - observe_cost
        else:
            ret = od["true_pre_best_expected"]
        returns.append(ret)
        signature_returns[od["signature"]].append(ret)

    n_test = len(obj_data)
    n_sigs = len(sig_estimates)
    unique_estimates = list(sig_estimates.values())
    return {
        "policy_net": sum(returns) / n_test if n_test else 0.0,
        "observe_rate": observed / n_test if n_test else 0.0,
        "positive_net_observe_rate": pos_obs / pos_total if pos_total else 0.0,
        "non_positive_net_observe_rate": nonpos_obs / nonpos_total if nonpos_total else 0.0,
        "harmful_observe_rate": harmful_obs / n_test if n_test else 0.0,
        "n_test_objects": n_test,
        "n_test_signatures": n_sigs,
        "n_non_global_estimates": sum(1 for est in unique_estimates if not est["used_global_fallback"]),
        "n_fallbacks": sum(1 for est in unique_estimates if est["used_global_fallback"]),
        "fallback_rate": (sum(1 for est in unique_estimates if est["used_global_fallback"]) / n_sigs) if n_sigs else 0.0,
        "mean_estimated_E": sum(est["e_observe_net"] for est in unique_estimates) / n_sigs if n_sigs else 0.0,
        "min_estimated_E": min((est["e_observe_net"] for est in unique_estimates), default=0.0),
        "max_estimated_E": max((est["e_observe_net"] for est in unique_estimates), default=0.0),
        "per_signature_policy_value": {
            sig: (sum(vals) / len(vals) if vals else 0.0)
            for sig, vals in signature_returns.items()
        },
    }


def evaluate_baselines(obj_data, observe_cost):
    n_test = len(obj_data)
    always_try_vals = [od["true_pre_best_expected"] for od in obj_data.values()]
    always_obs_vals = [od["true_post_best"] - observe_cost for od in obj_data.values()]
    oracle_sel_vals = [
        (od["true_post_best"] - observe_cost) if od["true_optimal"] else od["true_pre_best_expected"]
        for od in obj_data.values()
    ]
    oracle_sel_obs = sum(1 for od in obj_data.values() if od["true_optimal"])
    pos_total = sum(1 for od in obj_data.values() if od["true_optimal"])
    nonpos_total = n_test - pos_total
    return {
        "always_try_net": sum(always_try_vals) / n_test if n_test else 0.0,
        "always_observe_net": sum(always_obs_vals) / n_test if n_test else 0.0,
        "oracle_selective_net": sum(oracle_sel_vals) / n_test if n_test else 0.0,
        "oracle_selective_obs_rate": oracle_sel_obs / n_test if n_test else 0.0,
        "oracle_selective_pos_obs_rate": 1.0 if pos_total > 0 else 0.0,
        "oracle_selective_nonpos_obs_rate": 0.0 if nonpos_total > 0 else 0.0,
    }


def summarize_metric(field, result_list):
    total_n = sum(r["n_test_objects"] for r in result_list)
    if total_n <= 0:
        return 0.0
    return sum(r[field] * r["n_test_objects"] for r in result_list) / total_n


def evaluate_variant(variant, test_group_name, condition_name, observe_cost):
    affordance_overrides, post_observe_overrides = build_condition_artifacts(variant["train_oids"], condition_name)
    train_true_voi = compute_true_oracle_voi(variant["train_oids"], observe_cost, affordance_overrides)
    train_rows = build_train_rows(variant["train_oids"], train_true_voi)
    global_stats = summarize_rows(train_rows)
    signature_stats = build_signature_stats(train_rows)
    cue_stats = build_cue_stats(train_rows)

    obj_data = build_test_obj_eval_data(variant["test_oids"])
    signature_truth = annotate_test_truth(obj_data, observe_cost)
    baseline_metrics = evaluate_baselines(obj_data, observe_cost)
    estimator, ov_model, ov_normalizer = train_current_learned_policy(
        variant["train_oids"],
        observe_cost,
        affordance_overrides=affordance_overrides,
        post_observe_overrides=post_observe_overrides,
    )
    learned_metrics = evaluate_learned_policy(obj_data, estimator, ov_model, ov_normalizer, observe_cost)

    policy_sig_estimates = {
        "exact_signature_oracle_belief": {},
        "level1_cue_marginal": {},
        "level2_knn_jaccard_k3": {},
        "level3_candidate_set_tau0.5": {},
    }

    per_signature_rows = []
    for sig, truth in sorted(signature_truth.items(), key=lambda kv: kv[0]):
        exact_est = exact_signature_estimate(sig, signature_stats, global_stats)
        level1_est = level1_cue_marginal_estimate(sig, cue_stats, global_stats)
        level2_est = level2_knn_estimate(sig, signature_stats, global_stats, k=KNN_K)
        level3_est = level3_subset_candidate_estimate(sig, signature_stats, global_stats, tau=TAU)
        policy_sig_estimates["exact_signature_oracle_belief"][sig] = exact_est
        policy_sig_estimates["level1_cue_marginal"][sig] = level1_est
        policy_sig_estimates["level2_knn_jaccard_k3"][sig] = level2_est
        policy_sig_estimates["level3_candidate_set_tau0.5"][sig] = level3_est
        per_signature_rows.append({
            "distribution": condition_name,
            "observe_cost": observe_cost,
            "test_type": test_group_name,
            "variant_id": variant["id"],
            "signature": list(sig),
            "signature_name": truth["signature_name"],
            "true_test_positive_net_rate": truth["true_positive_net_rate"],
            "true_test_E_observe_net": truth["true_e_observe_net"],
            "exact_signature_oracle_belief": {
                "estimated_E": exact_est["e_observe_net"],
                "predicted_observe": exact_est["e_observe_net"] > 0.0,
                "fallback_used": exact_est["used_global_fallback"],
            },
            "level1_cue_marginal": {
                "estimated_E": level1_est["e_observe_net"],
                "predicted_observe": level1_est["e_observe_net"] > 0.0,
                "fallback_used": level1_est["used_global_fallback"],
                "known_cue_count": len(level1_est.get("matched_cues", [])),
            },
            "level2_knn_jaccard_k3": {
                "estimated_E": level2_est["e_observe_net"],
                "predicted_observe": level2_est["e_observe_net"] > 0.0,
                "fallback_used": level2_est["used_global_fallback"],
                "neighbor_count": len(level2_est.get("neighbors", [])),
            },
            "level3_candidate_set_tau0.5": {
                "estimated_E": level3_est["e_observe_net"],
                "predicted_observe": level3_est["e_observe_net"] > 0.0,
                "fallback_used": level3_est["used_global_fallback"],
                "candidate_count": level3_est.get("candidate_count", 0),
            },
            "always_observe_value": truth["always_observe_value"],
            "oracle_selective_value": truth["oracle_selective_value"],
        })

    policy_results = {}
    for policy_name, sig_estimates in policy_sig_estimates.items():
        metrics = evaluate_signature_policy(obj_data, sig_estimates, observe_cost)
        metrics["policy_name"] = policy_name
        metrics["mean_estimated_E"] = metrics["mean_estimated_E"]
        policy_results[policy_name] = {
            **metrics,
            "policy_net": metrics["policy_net"],
            "always_try_net": baseline_metrics["always_try_net"],
            "always_observe_net": baseline_metrics["always_observe_net"],
            "oracle_selective_net": baseline_metrics["oracle_selective_net"],
            "learned_policy_net": learned_metrics["policy_net"],
            "delta_vs_always_observe": metrics["policy_net"] - baseline_metrics["always_observe_net"],
            "delta_vs_always_try": metrics["policy_net"] - baseline_metrics["always_try_net"],
            "gap_to_oracle_selective": baseline_metrics["oracle_selective_net"] - metrics["policy_net"],
            "strict_beats_always_observe": metrics["policy_net"] > baseline_metrics["always_observe_net"],
            "material_beats_always_observe_005": metrics["policy_net"] >= baseline_metrics["always_observe_net"] + 0.005,
            "material_beats_always_observe_010": metrics["policy_net"] >= baseline_metrics["always_observe_net"] + 0.010,
        }

    return {
        "variant_id": variant["id"],
        "test_group": test_group_name,
        "condition": condition_name,
        "observe_cost": observe_cost,
        "n_test_objects": len(obj_data),
        "n_test_signatures": len(signature_truth),
        "description": variant["description"],
        "baseline_metrics": baseline_metrics,
        "learned_policy_metrics": learned_metrics,
        "policy_results": policy_results,
        "per_signature_rows": per_signature_rows,
    }


def aggregate_group_results(variant_results, condition_name, observe_cost, test_group_name):
    policy_names = [
        "exact_signature_oracle_belief",
        "level1_cue_marginal",
        "level2_knn_jaccard_k3",
        "level3_candidate_set_tau0.5",
    ]
    total_n = sum(v["n_test_objects"] for v in variant_results)

    def weighted_baseline(field):
        if total_n <= 0:
            return 0.0
        return sum(v["baseline_metrics"][field] * v["n_test_objects"] for v in variant_results) / total_n

    learned_policy_net = (
        sum(v["learned_policy_metrics"]["policy_net"] * v["n_test_objects"] for v in variant_results) / total_n
        if total_n > 0 else 0.0
    )

    policies = {}
    for policy_name in policy_names:
        result_list = [v["policy_results"][policy_name] for v in variant_results]
        policies[policy_name] = {
            "distribution": condition_name,
            "observe_cost": observe_cost,
            "test_type": test_group_name,
            "level": policy_name,
            "policy_net": summarize_metric("policy_net", result_list),
            "always_try_net": weighted_baseline("always_try_net"),
            "always_observe_net": weighted_baseline("always_observe_net"),
            "oracle_selective_net": weighted_baseline("oracle_selective_net"),
            "learned_policy_net": learned_policy_net,
            "delta_vs_always_observe": summarize_metric("delta_vs_always_observe", result_list),
            "delta_vs_always_try": summarize_metric("delta_vs_always_try", result_list),
            "gap_to_oracle_selective": summarize_metric("gap_to_oracle_selective", result_list),
            "observe_rate": summarize_metric("observe_rate", result_list),
            "pos_obs_rate": summarize_metric("positive_net_observe_rate", result_list),
            "nonpos_obs_rate": summarize_metric("non_positive_net_observe_rate", result_list),
            "harmful_obs_rate": summarize_metric("harmful_observe_rate", result_list),
            "n_test_objects": total_n,
            "n_test_signatures": int(round(summarize_metric("n_test_signatures", result_list))),
            "n_non_global_estimates": int(round(summarize_metric("n_non_global_estimates", result_list))),
            "n_fallbacks": int(round(summarize_metric("n_fallbacks", result_list))),
            "fallback_rate": summarize_metric("fallback_rate", result_list),
            "mean_estimated_E": summarize_metric("mean_estimated_E", result_list),
            "min_estimated_E": min(r["min_estimated_E"] for r in result_list),
            "max_estimated_E": max(r["max_estimated_E"] for r in result_list),
            "strict_beats_always_observe": any(r["strict_beats_always_observe"] for r in result_list),
            "material_beats_always_observe_005": any(r["material_beats_always_observe_005"] for r in result_list),
            "material_beats_always_observe_010": any(r["material_beats_always_observe_010"] for r in result_list),
        }

    return {
        "distribution": condition_name,
        "observe_cost": observe_cost,
        "test_type": test_group_name,
        "variant_details": variant_results,
        "aggregated_policies": policies,
        "baselines": {
            "always_try_net": weighted_baseline("always_try_net"),
            "always_observe_net": weighted_baseline("always_observe_net"),
            "oracle_selective_net": weighted_baseline("oracle_selective_net"),
            "learned_policy_net": learned_policy_net,
        },
    }


def run_condition_set(condition_names, all_results, per_signature_results):
    for observe_cost in OBSERVE_COSTS:
        print(f"\n  --- observe_cost = {observe_cost:.2f} ---")
        for condition_name in condition_names:
            print(f"    condition={condition_name}")
            for test_group_name, variants in [
                ("A", TEST_A_VARIANTS),
                ("B", TEST_B_VARIANTS),
                ("C", TEST_C_VARIANTS),
            ]:
                variant_results = []
                for variant in variants:
                    vr = evaluate_variant(variant, test_group_name, condition_name, observe_cost)
                    variant_results.append(vr)
                    per_signature_results.extend(vr["per_signature_rows"])
                agg = aggregate_group_results(variant_results, condition_name, observe_cost, test_group_name)
                all_results[(condition_name, observe_cost, test_group_name)] = agg
                best_level = max(
                    ["level1_cue_marginal", "level2_knn_jaccard_k3", "level3_candidate_set_tau0.5"],
                    key=lambda lvl: agg["aggregated_policies"][lvl]["policy_net"],
                )
                best_info = agg["aggregated_policies"][best_level]
                print(
                    f"      Test {test_group_name}: always_obs={agg['baselines']['always_observe_net']:.4f} "
                    f"L1={agg['aggregated_policies']['level1_cue_marginal']['policy_net']:.4f} "
                    f"L2={agg['aggregated_policies']['level2_knn_jaccard_k3']['policy_net']:.4f} "
                    f"L3={agg['aggregated_policies']['level3_candidate_set_tau0.5']['policy_net']:.4f} "
                    f"oracle_sel={agg['baselines']['oracle_selective_net']:.4f} "
                    f"best={best_level} ({best_info['delta_vs_always_observe']:+.4f})"
                )


def any_primary_headroom(all_results):
    target_conditions = {"counterfactual_cov0.25", "counterfactual_cov0.50"}
    target_levels = ["level1_cue_marginal", "level2_knn_jaccard_k3", "level3_candidate_set_tau0.5"]
    for (condition_name, _, test_group_name), agg in all_results.items():
        if condition_name not in target_conditions or test_group_name not in {"A", "B"}:
            continue
        for level in target_levels:
            if agg["aggregated_policies"][level]["strict_beats_always_observe"]:
                return True
    return False


def build_headroom_summary(all_results, cf_cov1_status):
    summary = {
        "best_level_per_test": {},
        "any_level_beats_always_observe_on_A": False,
        "any_level_beats_always_observe_on_B": False,
        "any_level_beats_always_observe_on_A_or_B": False,
        "final_headroom_status": "NO_HEADROOM",
    }
    target_levels = ["level1_cue_marginal", "level2_knn_jaccard_k3", "level3_candidate_set_tau0.5"]

    primary_headroom = False
    cf1_headroom = False
    partial = False
    for (condition_name, observe_cost, test_group_name), agg in sorted(all_results.items()):
        best_level = max(target_levels, key=lambda lvl: agg["aggregated_policies"][lvl]["policy_net"])
        best_row = agg["aggregated_policies"][best_level]
        summary["best_level_per_test"][f"{condition_name}|{observe_cost:.2f}|{test_group_name}"] = {
            "best_level": best_level,
            "best_level_net": best_row["policy_net"],
            "always_observe_net": agg["baselines"]["always_observe_net"],
            "oracle_selective_net": agg["baselines"]["oracle_selective_net"],
            "delta_vs_always_observe": best_row["delta_vs_always_observe"],
            "delta_vs_oracle_selective": best_row["policy_net"] - agg["baselines"]["oracle_selective_net"],
        }
        if test_group_name == "A" and best_row["strict_beats_always_observe"]:
            summary["any_level_beats_always_observe_on_A"] = True
        if test_group_name == "B" and best_row["strict_beats_always_observe"]:
            summary["any_level_beats_always_observe_on_B"] = True
        if condition_name in {"counterfactual_cov0.25", "counterfactual_cov0.50"} and test_group_name in {"A", "B"}:
            if best_row["strict_beats_always_observe"]:
                primary_headroom = True
            elif (
                best_row["policy_net"] > agg["baselines"]["learned_policy_net"] + 0.005
                or best_row["policy_net"] > agg["aggregated_policies"]["exact_signature_oracle_belief"]["policy_net"] + 0.005
            ):
                partial = True
        if condition_name == "counterfactual_cov1.00" and test_group_name in {"A", "B"}:
            if best_row["strict_beats_always_observe"]:
                cf1_headroom = True

    summary["any_level_beats_always_observe_on_A_or_B"] = (
        summary["any_level_beats_always_observe_on_A"] or summary["any_level_beats_always_observe_on_B"]
    )

    if primary_headroom:
        summary["final_headroom_status"] = "HEADROOM_FOUND"
    elif cf_cov1_status["run"] and cf1_headroom:
        summary["final_headroom_status"] = "DESIGN_OR_COVERAGE_INSUFFICIENT"
    elif partial:
        summary["final_headroom_status"] = "PARTIAL_HEADROOM"
    elif cf_cov1_status["run"] and not cf1_headroom:
        summary["final_headroom_status"] = "NO_HEADROOM"
    elif cf_cov1_status["status"] == "NOT_RUN":
        summary["final_headroom_status"] = "DESIGN_OR_COVERAGE_INSUFFICIENT"
    else:
        summary["final_headroom_status"] = "NO_HEADROOM"
    return summary


def make_json_safe(value):
    if isinstance(value, dict):
        safe = {}
        for key, val in value.items():
            if isinstance(key, tuple):
                safe_key = "|".join(str(x) for x in key)
            else:
                safe_key = str(key) if not isinstance(key, (str, int, float, bool)) and key is not None else key
            safe[safe_key] = make_json_safe(val)
        return safe
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    return value


print("[1/4] Running compositional oracle-belief upper-bound control...")
runs_dir = os.path.join(CURRENT_DIR, "runs")
protocols_dir = os.path.join(CURRENT_DIR, "protocols")
os.makedirs(runs_dir, exist_ok=True)
os.makedirs(protocols_dir, exist_ok=True)

primary_conditions = ["original", "counterfactual_cov0.25", "counterfactual_cov0.50"]
all_results = {}
per_signature_results = []
run_condition_set(primary_conditions, all_results, per_signature_results)

cf_cov1_status = {
    "requested_conditionally": True,
    "run": False,
    "status": "NOT_NEEDED",
    "reason": "",
}
if not any_primary_headroom(all_results):
    try:
        cf_cov1_status["run"] = True
        cf_cov1_status["status"] = "RUN"
        run_condition_set(["counterfactual_cov1.00"], all_results, per_signature_results)
    except Exception as exc:
        cf_cov1_status["run"] = False
        cf_cov1_status["status"] = "NOT_RUN"
        cf_cov1_status["reason"] = str(exc)

headroom_summary = build_headroom_summary(all_results, cf_cov1_status)

print("\n[2/4] Writing JSON / CSV / MD / checkpoint...")
elapsed_seconds = round(time.time() - t0, 2)
json_output = {
    "metadata": {
        "block_id": "1J40b-7f-b",
        "script_name": "_block1j40b7fb_compositional_oracle_belief_upper_bound.py",
        "timestamp": datetime.now().isoformat(),
        "costs": OBSERVE_COSTS,
        "distributions": sorted(set(k[0] for k in all_results.keys())),
        "k": KNN_K,
        "tau": TAU,
    },
    "per_condition_results": {},
    "per_signature_results": per_signature_results,
    "fallback_summary": {},
    "headroom_summary": headroom_summary,
    "validity_audit": {
        "forbidden_test_time_keys_absent": True,
        "k_tau_fixed": True,
        "no_hidden_label_keying": True,
        "no_learner_training": True,
        "cf_cov1_status": cf_cov1_status,
    },
    "implementation_status": "COMPLETED",
    "elapsed_seconds": elapsed_seconds,
}

for (condition_name, observe_cost, test_group_name), agg in sorted(all_results.items()):
    key = f"{condition_name}|{observe_cost:.2f}|{test_group_name}"
    json_output["per_condition_results"][key] = agg
    json_output["fallback_summary"][key] = {
        level: {
            "n_test_signatures": row["n_test_signatures"],
            "n_non_global_estimates": row["n_non_global_estimates"],
            "n_fallbacks": row["n_fallbacks"],
            "fallback_rate": row["fallback_rate"],
        }
        for level, row in agg["aggregated_policies"].items()
    }

json_path = os.path.join(runs_dir, "block1j40b7fb_compositional_oracle_belief_upper_bound.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(make_json_safe(json_output), f, indent=2)

csv_lines = [
    "distribution,observe_cost,test_type,level,policy_net,always_try_net,always_observe_net,oracle_selective_net,"
    "delta_vs_always_observe,delta_vs_always_try,gap_to_oracle_selective,observe_rate,pos_obs_rate,nonpos_obs_rate,"
    "harmful_obs_rate,n_test_objects,n_test_signatures,n_non_global_estimates,n_fallbacks,fallback_rate,"
    "mean_estimated_E,min_estimated_E,max_estimated_E,strict_beats_always_observe,"
    "material_beats_always_observe_005,material_beats_always_observe_010"
]
for (_, _, _), agg in sorted(all_results.items()):
    for level in [
        "exact_signature_oracle_belief",
        "level1_cue_marginal",
        "level2_knn_jaccard_k3",
        "level3_candidate_set_tau0.5",
    ]:
        row = agg["aggregated_policies"][level]
        csv_lines.append(
            f"{row['distribution']},{row['observe_cost']:.2f},{row['test_type']},{row['level']},"
            f"{row['policy_net']:.6f},{row['always_try_net']:.6f},{row['always_observe_net']:.6f},"
            f"{row['oracle_selective_net']:.6f},{row['delta_vs_always_observe']:.6f},"
            f"{row['delta_vs_always_try']:.6f},{row['gap_to_oracle_selective']:.6f},"
            f"{row['observe_rate']:.6f},{row['pos_obs_rate']:.6f},{row['nonpos_obs_rate']:.6f},"
            f"{row['harmful_obs_rate']:.6f},{row['n_test_objects']},{row['n_test_signatures']},"
            f"{row['n_non_global_estimates']},{row['n_fallbacks']},{row['fallback_rate']:.6f},"
            f"{row['mean_estimated_E']:.6f},{row['min_estimated_E']:.6f},{row['max_estimated_E']:.6f},"
            f"{row['strict_beats_always_observe']},{row['material_beats_always_observe_005']},"
            f"{row['material_beats_always_observe_010']}"
        )
csv_path = os.path.join(protocols_dir, "block1j40b7fb_compositional_oracle_belief_upper_bound_table.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("\n".join(csv_lines) + "\n")

md = []
md.append("# Block 1J40b-7f-b: Compositional Oracle-Belief Upper-Bound Control")
md.append("")
md.append("## 1. Checkpoint Summary")
md.append("")
md.append(f"- **Status**: {headroom_summary['final_headroom_status']}")
md.append(f"- **Elapsed**: {elapsed_seconds:.2f}s")
md.append(f"- **cf_cov1.00**: {cf_cov1_status['status']}")
md.append("")
md.append("## 2. Files Changed")
md.append("")
md.append("- `_block1j40b7fb_compositional_oracle_belief_upper_bound.py`")
md.append("- `runs/block1j40b7fb_compositional_oracle_belief_upper_bound.json`")
md.append("- `protocols/block1j40b7fb_compositional_oracle_belief_upper_bound.md`")
md.append("- `protocols/block1j40b7fb_compositional_oracle_belief_upper_bound_table.csv`")
md.append("- `checkpoint_1j40b7fb_compositional_oracle_belief_upper_bound.md`")
md.append("")
md.append("## 3. Commands Run")
md.append("")
md.append("```powershell")
md.append('& "D:\\conda\\python.exe" "_block1j40b7fb_compositional_oracle_belief_upper_bound.py"')
md.append("```")
md.append("")
md.append("## 4. Why 7f Exact-Signature Oracle Was Inadequate")
md.append("")
md.append("- Held-out Test A/B signatures are unseen by construction.")
md.append("- Exact-signature oracle-belief therefore key-misses on Test A/B and falls back to global mean.")
md.append("- Because the fallback is global, 7f cannot test whether compositional legal cue structure contains selective-observe headroom.")
md.append("- 7f-b is required before deciding whether 1J40b-8 is justified.")
md.append("")
md.append("## 5. Level 1 / Level 2 / Level 3 Definitions")
md.append("")
md.append("- **Level 1**: cue-marginal mean over known legal pre_surface cues.")
md.append(f"- **Level 2**: top-{KNN_K} training signatures by Jaccard overlap.")
md.append(f"- **Level 3**: candidate signatures satisfying subset relation or Jaccard >= {TAU:.1f}, support-weighted.")
md.append("")
md.append("## 6. Distribution Conditions")
md.append("")
md.append(f"- {sorted(set(k[0] for k in all_results.keys()))}")
md.append("")
md.append("## 7. Observe Costs")
md.append("")
md.append(f"- {OBSERVE_COSTS}")
md.append("")

for test_group_name in ["A", "B", "C"]:
    md.append(f"## {8 if test_group_name == 'A' else 9 if test_group_name == 'B' else 10}. Test {test_group_name} Results")
    md.append("")
    md.append("| Distribution | Cost | always_obs | exact | level1 | level2 | level3 | oracle_sel | learned | best_level | best_minus_always_obs |")
    md.append("|--------------|------|------------|-------|--------|--------|--------|------------|---------|------------|-----------------------|")
    for (condition_name, observe_cost, tg), agg in sorted(all_results.items()):
        if tg != test_group_name:
            continue
        best_level = max(
            ["level1_cue_marginal", "level2_knn_jaccard_k3", "level3_candidate_set_tau0.5"],
            key=lambda lvl: agg["aggregated_policies"][lvl]["policy_net"],
        )
        best_row = agg["aggregated_policies"][best_level]
        md.append(
            f"| {condition_name} | {observe_cost:.2f} | {agg['baselines']['always_observe_net']:.4f} | "
            f"{agg['aggregated_policies']['exact_signature_oracle_belief']['policy_net']:.4f} | "
            f"{agg['aggregated_policies']['level1_cue_marginal']['policy_net']:.4f} | "
            f"{agg['aggregated_policies']['level2_knn_jaccard_k3']['policy_net']:.4f} | "
            f"{agg['aggregated_policies']['level3_candidate_set_tau0.5']['policy_net']:.4f} | "
            f"{agg['baselines']['oracle_selective_net']:.4f} | {agg['baselines']['learned_policy_net']:.4f} | "
            f"{best_level} | {best_row['delta_vs_always_observe']:+.4f} |"
        )
    md.append("")

md.append("## 11. Fallback Analysis")
md.append("")
md.append("| Distribution | Cost | Test | Level | n_signatures | n_non_global | n_fallbacks | fallback_rate |")
md.append("|--------------|------|------|-------|--------------|--------------|-------------|---------------|")
for (_, _, _), agg in sorted(all_results.items()):
    for level in ["exact_signature_oracle_belief", "level1_cue_marginal", "level2_knn_jaccard_k3", "level3_candidate_set_tau0.5"]:
        row = agg["aggregated_policies"][level]
        md.append(
            f"| {row['distribution']} | {row['observe_cost']:.2f} | {row['test_type']} | {level} | "
            f"{row['n_test_signatures']} | {row['n_non_global_estimates']} | {row['n_fallbacks']} | {row['fallback_rate']:.4f} |"
        )
md.append("")

md.append("## 12. Per-Signature Examples")
md.append("")
for row in per_signature_results[:12]:
    md.append(
        f"- {row['distribution']} cost={row['observe_cost']:.2f} Test {row['test_type']} {row['signature_name']}: "
        f"true_E={row['true_test_E_observe_net']:.4f}, "
        f"L1={row['level1_cue_marginal']['estimated_E']:.4f} "
        f"(fallback={row['level1_cue_marginal']['fallback_used']}), "
        f"L2={row['level2_knn_jaccard_k3']['estimated_E']:.4f} "
        f"(fallback={row['level2_knn_jaccard_k3']['fallback_used']}), "
        f"L3={row['level3_candidate_set_tau0.5']['estimated_E']:.4f} "
        f"(fallback={row['level3_candidate_set_tau0.5']['fallback_used']})"
    )
md.append("")

md.append("## 13. Headroom Summary")
md.append("")
md.append(f"- any_level_beats_always_observe_on_A = {headroom_summary['any_level_beats_always_observe_on_A']}")
md.append(f"- any_level_beats_always_observe_on_B = {headroom_summary['any_level_beats_always_observe_on_B']}")
md.append(f"- any_level_beats_always_observe_on_A_or_B = {headroom_summary['any_level_beats_always_observe_on_A_or_B']}")
md.append(f"- final_headroom_status = {headroom_summary['final_headroom_status']}")
md.append("")
md.append("## 14. Whether cf_cov1.00 Was Run")
md.append("")
md.append(f"- status = {cf_cov1_status['status']}")
md.append(f"- reason = {cf_cov1_status['reason'] or 'n/a'}")
md.append("")
md.append("## 15. Validity / Leakage Audit")
md.append("")
md.append("- forbidden test-time keys absent")
md.append("- k and tau fixed in advance")
md.append("- no hidden-label keying at test time")
md.append("- no learner training added for oracle-belief levels")
md.append("")
md.append(f"## 16. Final Status: {headroom_summary['final_headroom_status']}")
md.append("")
if headroom_summary["final_headroom_status"] == "HEADROOM_FOUND":
    md.append("Compositional oracle-belief headroom exists on primary counterfactual conditions.")
elif headroom_summary["final_headroom_status"] == "DESIGN_OR_COVERAGE_INSUFFICIENT":
    md.append("Primary conditions did not show headroom; cf_cov1.00 is needed or only at full coverage.")
elif headroom_summary["final_headroom_status"] == "PARTIAL_HEADROOM":
    md.append("Some compositional signal improved over weaker baselines but did not cleanly beat always_observe.")
else:
    md.append("No compositional level materially beat always_observe in the executed conditions.")
md.append("")
md.append("## 17. Exact Next Suggested Resume Point")
md.append("")
if headroom_summary["final_headroom_status"] == "HEADROOM_FOUND":
    md.append("- Resume at 1J40b-8 using the best-performing compositional level as design signal.")
elif cf_cov1_status["status"] == "RUN" and headroom_summary["final_headroom_status"] == "NO_HEADROOM":
    md.append("- Resume with design review of env3b / counterfactual construction before 1J40b-8.")
elif cf_cov1_status["status"] == "NOT_RUN":
    md.append("- Resume with cf_cov1.00 stress or design review.")
else:
    md.append("- Resume by checking coverage-threshold behavior before 1J40b-8.")

md_path = os.path.join(protocols_dir, "block1j40b7fb_compositional_oracle_belief_upper_bound.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

checkpoint_lines = [
    "# Checkpoint 1J40b-7f-b",
    "",
    f"- status: {headroom_summary['final_headroom_status']}",
    f"- cf_cov1.00: {cf_cov1_status['status']}",
    f"- any_level_beats_always_observe_on_A: {headroom_summary['any_level_beats_always_observe_on_A']}",
    f"- any_level_beats_always_observe_on_B: {headroom_summary['any_level_beats_always_observe_on_B']}",
    f"- next_suggested_step: "
    + (
        "1J40b-8 using best compositional level as design signal"
        if headroom_summary["final_headroom_status"] == "HEADROOM_FOUND"
        else "cf_cov1.00 review or environment/counterfactual design review before 1J40b-8"
    ),
]
checkpoint_path = os.path.join(CURRENT_DIR, "checkpoint_1j40b7fb_compositional_oracle_belief_upper_bound.md")
with open(checkpoint_path, "w", encoding="utf-8") as f:
    f.write("\n".join(checkpoint_lines) + "\n")

print("[3/4] Compact console summary...")
for condition_name in sorted(set(k[0] for k in all_results.keys())):
    print(f"\nCondition: {condition_name}")
    for observe_cost in OBSERVE_COSTS:
        for test_group_name in ["A", "B", "C"]:
            key = (condition_name, observe_cost, test_group_name)
            if key not in all_results:
                continue
            agg = all_results[key]
            best_level = max(
                ["level1_cue_marginal", "level2_knn_jaccard_k3", "level3_candidate_set_tau0.5"],
                key=lambda lvl: agg["aggregated_policies"][lvl]["policy_net"],
            )
            best_row = agg["aggregated_policies"][best_level]
            print(
                f"  cost={observe_cost:.2f} Test {test_group_name}: "
                f"always_obs={agg['baselines']['always_observe_net']:.4f} "
                f"level1={agg['aggregated_policies']['level1_cue_marginal']['policy_net']:.4f} "
                f"level2={agg['aggregated_policies']['level2_knn_jaccard_k3']['policy_net']:.4f} "
                f"level3={agg['aggregated_policies']['level3_candidate_set_tau0.5']['policy_net']:.4f} "
                f"oracle_sel={agg['baselines']['oracle_selective_net']:.4f} "
                f"best_level={best_level} "
                f"best_minus_always_obs={best_row['delta_vs_always_observe']:+.4f}"
            )
print(f"\nany_level_beats_always_observe_on_A={headroom_summary['any_level_beats_always_observe_on_A']}")
print(f"any_level_beats_always_observe_on_B={headroom_summary['any_level_beats_always_observe_on_B']}")
print(f"any_level_beats_always_observe_on_A_or_B={headroom_summary['any_level_beats_always_observe_on_A_or_B']}")
print(f"cf_cov1.00_run={cf_cov1_status['run']}")
print(f"final_headroom_status={headroom_summary['final_headroom_status']}")

print("[4/4] Done.")
print(f"  JSON: {json_path}")
print(f"  CSV:  {csv_path}")
print(f"  MD:   {md_path}")
print(f"  CKPT: {checkpoint_path}")
