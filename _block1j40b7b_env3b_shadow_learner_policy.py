"""
Block 1J40b-7b — env3b Shadow Learner + Policy Intervention.

Trains a learned policy on env3b (selective-observe environment) and tests
whether it can learn to observe only when VOI > 0 — discriminating between
positive-VOI and non-positive-VOI observation opportunities.

Policy chain (per object):
  1. Build legal pre_surface_features vector only (prior_obs=0).
  2. Estimate pre_best_try_value and post_best_try_value from learned Q.
  3. predicted_observe_net = post_best - pre_best - OBSERVE_COST.
  4. If > 0: execute observe, select best post try, score true_return - cost.
  5. Else: execute best pre try, score true_return.

Primary observe_cost = 0.03, robustness observe_cost = 0.05.

Three-world separation:
  - Environment: holds hidden truth (category, subtype, audit rationale, affordance).
  - Agent/Model: only legal pre_surface_features and post_observe_features.
  - Audit: may inspect hidden truth for evaluation only.

Baselines:
  1. no_observe_pre_only
  2. always_try
  3. always_observe
  4. random_observe
  5. oracle_selective
  6. learned_policy
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
# Config safety (same as 7a)
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
# Feature Definitions (from env3b / 1J40b-7a)
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
    f for feats in AMBIENT_GROUP_FEATURES.values() for f in feats
))

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

# All visible features = ambient + post_observe(category-diag) + shared surface
ALL_VISIBLE_FEATURE_NAMES = sorted(set(
    list(ALL_AMBIENT_FEATURE_NAMES) +
    list(ALL_POST_OBSERVE_FEATURE_NAMES) +
    list(SHARED_SURFACE_FEATURES)
))

STATE_FEATURE_NAMES = ["fresh", "wet", "damaged", "clean", "hot", "open"]

# Pre-observe: pre_surface_features only (ambient + shared surface cues)
# These are the features present IN pre_surface_features dict.
# Post-observe: post_observe_features (pre_surface + category-diagnostic revealed)
# Hidden: features not in either.

# =============================================================================
# Instance Subtype Definitions (from 7a - FULL COPY)
# =============================================================================
INSTANCE_SUBTYPE_DEFS = {}

INSTANCE_SUBTYPE_DEFS["wood_log"] = {
    "P": {
        "label": "ambiguous_log",
        "audit_rationale_class": "positive_change",
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
        "label": "positive_shared_cue_log",
        "audit_rationale_class": "positive_change",
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
        "label": "nonpos_identified_log",
        "audit_rationale_class": "wasteful_cost_only",
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
        "label": "ambiguous_stone",
        "audit_rationale_class": "positive_change",
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
        "label": "positive_shared_cue_stone",
        "audit_rationale_class": "positive_change",
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
        "label": "nonpos_identified_stone",
        "audit_rationale_class": "wasteful_cost_only",
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
        "label": "ambiguous_fruit",
        "audit_rationale_class": "positive_change",
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
        "label": "positive_shared_cue_apple",
        "audit_rationale_class": "positive_change",
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
        "label": "nonpos_identified_apple",
        "audit_rationale_class": "wasteful_cost_only",
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
        "label": "ambiguous_tool",
        "audit_rationale_class": "positive_change",
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
        "label": "positive_shared_cue_tool",
        "audit_rationale_class": "positive_change",
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
        "label": "nonpos_identified_tool",
        "audit_rationale_class": "wasteful_cost_only",
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

            # Build pre_surface_features
            pre_surface = {}
            for f in ambient_features_list:
                pre_surface[f] = True
            for f in pre_extra:
                pre_surface[f] = True

            # Build post_observe_features (pre_surface + revealed)
            post_observe = dict(pre_surface)
            for f in post_revealed:
                post_observe[f] = True

            # Hidden features: all visible NOT in pre_surface or post_revealed
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

total_objects = sum(len(per_seed_objects[s]) for s in SEEDS)
print(f"  Objects per seed: {len(per_seed_objects[SEEDS[0]])}")
print(f"  Total objects: {total_objects}")

# =============================================================================
# Part 2: Feature Vector Building (model-legal only)
# =============================================================================
print("[2/8] Building training data and feature vectors...")

def try_net_value(success_or_failure):
    if success_or_failure is True:
        return SUCCESS_REWARD - 0.05  # probe cost
    elif success_or_failure is False:
        return -FAILURE_PENALTY - 0.05
    return 0.0


def get_pre_surface_signature(pre_surface_features):
    return tuple(sorted(pre_surface_features.keys()))


# Build all feature names that appear in actual objects
def collect_all_pre_surface_features(per_seed_objects):
    """Collect all unique feature names that appear in pre_surface_features."""
    all_feats = set()
    for seed in SEEDS:
        for oid, obj in per_seed_objects[seed].items():
            for f in obj["pre_surface_features"]:
                all_feats.add(f)
    return sorted(all_feats)


def collect_all_post_observe_features(per_seed_objects):
    """Collect all unique feature names that appear in post_observe_features."""
    all_feats = set()
    for seed in SEEDS:
        for oid, obj in per_seed_objects[seed].items():
            for f in obj["post_observe_features"]:
                all_feats.add(f)
    return sorted(all_feats)


PRE_SURFACE_FEATURE_NAMES = collect_all_pre_surface_features(per_seed_objects)
POST_OBSERVE_FEATURE_NAMES = collect_all_post_observe_features(per_seed_objects)

# Feature vector keys for model
PRE_FEATURE_KEYS = [f"feat_{f}" for f in PRE_SURFACE_FEATURE_NAMES]
POST_FEATURE_KEYS = [f"feat_{f}" for f in POST_OBSERVE_FEATURE_NAMES]
STATE_KEYS = [f"state_{s}" for s in STATE_FEATURE_NAMES]
NUMERIC_KEYS = ["prior_observe_count"]

# Full feature vector (for model - uses post_observe feature space which includes pre)
ALL_MODEL_FEATURE_KEYS = sorted(set(POST_FEATURE_KEYS + STATE_KEYS + NUMERIC_KEYS))

print(f"  Pre-surface features: {len(PRE_SURFACE_FEATURE_NAMES)}")
print(f"  Post-observe features: {len(POST_OBSERVE_FEATURE_NAMES)}")
print(f"  Model feature keys: {len(ALL_MODEL_FEATURE_KEYS)}")


def build_feature_vector(known_features, known_states, prior_obs_count, observe_cost=None):
    """Build model feature vector from agent-observable data.

    known_features: dict of feature_name -> True for features the agent knows
    known_states: dict of state_name -> bool
    prior_obs_count: int
    """
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
# Forbidden Fields Audit
# =============================================================================
FORBIDDEN_MODEL_FIELDS = sorted([
    "hidden_category", "hidden_subtype", "hidden_audit_rationale_class",
    "audit_rationale_class", "positive_net", "non_positive_net",
    "audit_VOI", "audit_observe_value",
    "oracle_action", "oracle_selective",
    "true_affordance_profile", "effective_affordance_profile",
    "hidden_affordance_profile", "affordance_profile",
    "object_id", "seed_id", "oid",
    "depth_schedule", "episode_id", "episode_progress",
    "ambient_group", "group_role", "voi_class",
    "hidden_features", "post_observe_revealed",
    "pre_surface_extra", "voi_rationale",
    "n_features_known", "n_states_known",
    "category",
])

# Verify none of the forbidden fields appear in model feature keys
forbidden_in_model = []
for ff in FORBIDDEN_MODEL_FIELDS:
    for mk in ALL_MODEL_FEATURE_KEYS:
        if ff.lower() in mk.lower().replace("feat_", "").replace("state_", ""):
            forbidden_in_model.append(f"{ff} -> {mk}")

print(f"\n  Forbidden fields in model keys: {len(forbidden_in_model)}")
if forbidden_in_model:
    for leak in forbidden_in_model:
        print(f"    LEAK: {leak}")

# =============================================================================
# VOI Computation (audit only — from 7a, for reporting)
# =============================================================================
def compute_audit_voi(per_seed_objects, per_seed_audit_labels, observe_cost):
    """Compute audit_VOI using legal pre_surface signature groups."""
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
                "seed": seed, "oid": oid, "true_returns": true_returns,
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

    per_object_voi = {}
    for seed in SEEDS:
        per_object_voi[seed] = {}
        for oid, obj in per_seed_objects[seed].items():
            lbl = per_seed_audit_labels[seed][oid]
            sig = get_pre_surface_signature(obj["pre_surface_features"])
            se = sig_expected[sig]

            true_returns = {}
            for action in ALL_TRY_AFFORDANCES:
                outcome = lbl["affordance_profile"].get(action, "fail")
                true_returns[action] = try_net_value(outcome == "success")

            oracle_post_best = max(true_returns.values())
            oracle_pre_best = se["best_expected_value"]
            audit_voi = oracle_post_best - oracle_pre_best - observe_cost

            per_object_voi[seed][oid] = {
                "audit_VOI": audit_voi,
                "numeric_class": "positive_net" if audit_voi > 0.001 else "non_positive_net",
                "rationale_class": lbl["hidden_audit_rationale_class"],
                "oracle_post_best": oracle_post_best,
                "oracle_pre_best_expected": oracle_pre_best,
            }

    return per_object_voi, sig_expected


# =============================================================================
# Ridge Regression Model (same as 1J40b-6)
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
# Part 3: Training Data Construction
# =============================================================================
print("[3/8] Building training records...")

def build_training_records(per_seed_objects, per_seed_audit_labels, observe_cost):
    """Build per-seed training records with legal feature vectors.

    For each object, create:
      - Pre-observe records: pre features + state + prior_obs=0, each try action
      - Post-observe records: post features + state + prior_obs=1, each try action
      - Observe action record (value = -observe_cost)
    """
    per_seed_records = {}

    for seed in SEEDS:
        records = []
        record_id = 0

        for oid, obj in per_seed_objects[seed].items():
            lbl = per_seed_audit_labels[seed][oid]
            affordance = lbl["affordance_profile"]
            pre_features = obj["pre_surface_features"]
            post_features = obj["post_observe_features"]
            state = obj["visible_state"]

            # Pre-observe state
            pre_sf = build_feature_vector(pre_features, {}, 0, observe_cost)
            pre_vec = extract_feature_vector(pre_sf)

            # Post-observe state
            post_sf = build_feature_vector(post_features, state, 1, observe_cost)
            post_vec = extract_feature_vector(post_sf)

            # Observe action record (from pre state)
            record_id += 1
            records.append({
                "record_id": f"s{seed}_r{record_id:05d}",
                "seed": seed,
                "object_id": oid,
                "situation_features": pre_sf,
                "feature_vector": pre_vec,
                "action_key": "observe",
                "net_value": -observe_cost,
            })

            # Try actions from pre-observe state
            for action in ALL_TRY_AFFORDANCES:
                outcome = affordance.get(action, "fail")
                record_id += 1
                records.append({
                    "record_id": f"s{seed}_r{record_id:05d}",
                    "seed": seed,
                    "object_id": oid,
                    "situation_features": pre_sf,
                    "feature_vector": pre_vec,
                    "action_key": f"try_{action}",
                    "net_value": try_net_value(outcome == "success"),
                })

            # Try actions from post-observe state
            for action in ALL_TRY_AFFORDANCES:
                outcome = affordance.get(action, "fail")
                record_id += 1
                records.append({
                    "record_id": f"s{seed}_r{record_id:05d}",
                    "seed": seed,
                    "object_id": oid,
                    "situation_features": post_sf,
                    "feature_vector": post_vec,
                    "action_key": f"try_{action}",
                    "net_value": try_net_value(outcome == "success"),
                })

        per_seed_records[seed] = records

    return per_seed_records


# =============================================================================
# Part 4: Policy Intervention — LOSO Training and Evaluation
# =============================================================================
print("[4/8] Running policy intervention (LOSO)...")

def build_obj_eval_vectors(objects_dict, audit_labels_dict):
    """Build pre and post evaluation vectors for all objects in a seed."""
    obj_data = {}
    for oid, lbl in audit_labels_dict.items():
        obj = objects_dict[oid]
        category = lbl["hidden_category"]
        profile = lbl["affordance_profile"]
        subtype = lbl["hidden_subtype"]
        rationale = lbl["hidden_audit_rationale_class"]

        pre_features = obj["pre_surface_features"]
        post_features = obj["post_observe_features"]
        state = obj["visible_state"]

        # Pre-observe vector
        pre_sf = build_feature_vector(pre_features, {}, 0)
        pre_vec = extract_feature_vector(pre_sf)

        # Post-observe vector
        post_sf = build_feature_vector(post_features, state, 1)
        post_vec = extract_feature_vector(post_sf)

        # True returns
        true_returns = {}
        for action in ALL_TRY_AFFORDANCES:
            outcome = profile.get(action, "fail")
            true_returns[f"try_{action}"] = try_net_value(outcome == "success")

        # n_pre_surface_features
        n_pre = len(obj["pre_surface_features"])

        obj_data[oid] = {
            "oid": oid,
            "category": category,
            "subtype": subtype,
            "rationale_class": rationale,
            "pre_vec": pre_vec,
            "post_vec": post_vec,
            "true_returns": true_returns,
            "n_pre_surface_features": n_pre,
            "pre_surface_features": pre_features,
        }
    return obj_data


def run_policy_intervention(train_recs, objects_dict, audit_labels_dict, test_seed, observe_cost, train_seeds=None):
    """Run learned policy intervention on test seed objects."""
    estimator = PerActionValueEstimator(alpha=RIDGE_ALPHA)

    # Train: extract vectors and values from training records
    X_train = [r["feature_vector"] for r in train_recs]
    y_train = [r["net_value"] for r in train_recs]
    ak_train = [r["action_key"] for r in train_recs]
    estimator.fit(X_train, y_train, ak_train)

    # ---- Train observe_value_model from training data only ----
    # Uses only pre_vec to predict expected observe net value.
    # Target: post_best_q - pre_best_q - observe_cost (learned-Q based).
    # Q-estimates computed from training objects only (no test post_vec access).
    ov_model = None
    ov_normalizer = None
    ov_target_source = "learned-Q: post_best_q - pre_best_q - observe_cost"
    ov_true_target_corr = None
    if train_seeds:
        ov_X = []
        ov_y = []
        ov_y_true = []  # true-return-based sanity check
        # Pre-compute training-only signature-group expected returns for true-based targets
        train_sig_groups = defaultdict(list)
        for tseed in train_seeds:
            t_objects = per_seed_objects[tseed]
            t_labels = per_seed_audit_labels[tseed]
            for oid, obj in t_objects.items():
                lbl = t_labels[oid]
                sig = get_pre_surface_signature(obj["pre_surface_features"])
                true_returns = {}
                for action in ALL_TRY_AFFORDANCES:
                    outcome = lbl["affordance_profile"].get(action, "fail")
                    true_returns[action] = try_net_value(outcome == "success")
                train_sig_groups[sig].append(true_returns)
        train_sig_expected = {}
        for sig, rets_list in train_sig_groups.items():
            n = len(rets_list)
            action_exp = {}
            for action in ALL_TRY_AFFORDANCES:
                action_exp[action] = sum(r[action] for r in rets_list) / n
            best_action = max(action_exp, key=action_exp.get)
            train_sig_expected[sig] = action_exp[best_action]

        for tseed in train_seeds:
            t_objects = per_seed_objects[tseed]
            t_labels = per_seed_audit_labels[tseed]
            for oid, obj in t_objects.items():
                lbl = t_labels[oid]
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

                pre_best = max(pre_try_qs.values()) if pre_try_qs else 0.0
                post_best = max(post_try_qs.values()) if post_try_qs else 0.0

                ov_y.append(post_best - pre_best - observe_cost)
                ov_X.append(pre_vec)

                # True-return-based sanity target: uses training-only sig-group pre-oracle
                sig = get_pre_surface_signature(pre_features)
                pre_best_true = train_sig_expected.get(sig, 0.0)
                true_returns_obj = {}
                for action in ALL_TRY_AFFORDANCES:
                    outcome = lbl["affordance_profile"].get(action, "fail")
                    true_returns_obj[action] = try_net_value(outcome == "success")
                post_best_true = max(true_returns_obj.values())
                ov_y_true.append(post_best_true - pre_best_true - observe_cost)

        if len(ov_X) >= 3:
            ov_normalizer = FeatureNormalizer().fit(ov_X)
            ov_X_norm = ov_normalizer.transform(ov_X)
            ov_model = RidgeRegression(alpha=RIDGE_ALPHA).fit(ov_X_norm, ov_y)
            # Correlation between Q-based and true-based targets
            if len(ov_y_true) == len(ov_y) and len(ov_y) > 0:
                mean_q = sum(ov_y) / len(ov_y)
                mean_t = sum(ov_y_true) / len(ov_y_true)
                num = sum((ov_y[i] - mean_q) * (ov_y_true[i] - mean_t) for i in range(len(ov_y)))
                den_q = math.sqrt(sum((v - mean_q)**2 for v in ov_y))
                den_t = math.sqrt(sum((v - mean_t)**2 for v in ov_y_true))
                if den_q > 1e-12 and den_t > 1e-12:
                    ov_true_target_corr = round(num / (den_q * den_t), 4)
            print(f"  [ov_model] targets={len(ov_y)}, source={ov_target_source}")
            print(f"  [ov_model] Q-target mean={sum(ov_y)/len(ov_y):.4f}, true-target mean={sum(ov_y_true)/len(ov_y_true):.4f}, corr={ov_true_target_corr}")

    obj_data = build_obj_eval_vectors(objects_dict, audit_labels_dict)
    test_oids = sorted(obj_data.keys())
    n_objects = len(test_oids)

    # ---- Baseline 1: no_observe_pre_only ----
    pre_returns = []
    pre_actions_dist = defaultdict(int)
    pre_per_object = {}

    for oid in test_oids:
        od = obj_data[oid]
        qs = estimator.predict_all_actions(od["pre_vec"])
        try_qs = {ak: q for ak, q in qs.items() if ak != "observe"}
        pre_best_action = max(try_qs, key=try_qs.get)
        pre_true_ret = od["true_returns"].get(pre_best_action, 0.0)
        pre_returns.append(pre_true_ret)
        pre_actions_dist[pre_best_action] += 1
        pre_per_object[oid] = (pre_best_action, pre_true_ret)

    pre_mean_return = sum(pre_returns) / n_objects if n_objects else 0.0

    # ---- Baseline 2: always_try (same as pre, since no observe) ----
    always_try_mean = pre_mean_return

    # ---- Baseline 3: always_observe ----
    post_returns_net = []
    post_returns_gross = []
    post_actions_dist = defaultdict(int)
    for oid in test_oids:
        od = obj_data[oid]
        qs = estimator.predict_all_actions(od["post_vec"])
        try_qs = {ak: q for ak, q in qs.items() if ak != "observe"}
        best_action = max(try_qs, key=try_qs.get)
        true_ret = od["true_returns"].get(best_action, 0.0)
        post_returns_gross.append(true_ret)
        post_returns_net.append(true_ret - observe_cost)
        post_actions_dist[best_action] += 1

    post_mean_gross = sum(post_returns_gross) / n_objects if n_objects else 0.0
    post_mean_net = sum(post_returns_net) / n_objects if n_objects else 0.0

    # ---- Baseline 4: random_observe ----
    rng = random.Random(42 + test_seed)
    random_returns = []
    random_obs_count = 0
    for oid in test_oids:
        od = obj_data[oid]
        if rng.random() < 0.5:
            vec = od["post_vec"]
            random_obs_count += 1
        else:
            vec = od["pre_vec"]
        qs = estimator.predict_all_actions(vec)
        try_qs = {ak: q for ak, q in qs.items() if ak != "observe"}
        best_action = max(try_qs, key=try_qs.get)
        true_ret = od["true_returns"].get(best_action, 0.0)
        random_returns.append(true_ret)

    random_mean_return = sum(random_returns) / n_objects if n_objects else 0.0

    # ---- Baseline 5: oracle_selective (audit only) ----
    # Group objects by legal pre_surface signature within test_seed
    local_sig_groups = defaultdict(list)
    for oid2 in test_oids:
        od2 = obj_data[oid2]
        sig = get_pre_surface_signature(od2["pre_surface_features"])
        local_sig_groups[sig].append(od2)

    local_sig_expected = {}
    for sig, members in local_sig_groups.items():
        n = len(members)
        action_expected = {}
        for action in ALL_TRY_AFFORDANCES:
            total = sum(m["true_returns"].get(f"try_{action}", 0.0) for m in members)
            action_expected[action] = total / n
        best_action = max(action_expected, key=action_expected.get)
        local_sig_expected[sig] = {
            "n_members": n,
            "best_action": best_action,
            "best_expected_value": action_expected[best_action],
        }

    oracle_selective_returns = []
    oracle_selective_obs = 0
    for oid in test_oids:
        od = obj_data[oid]
        true_returns = od["true_returns"]
        post_best = max(true_returns.values())

        sig = get_pre_surface_signature(od["pre_surface_features"])
        se = local_sig_expected.get(sig)
        if se is None:
            pre_best_expected = 0.0
        else:
            pre_best_expected = se["best_expected_value"]

        oracle_voi = post_best - pre_best_expected - observe_cost

        if oracle_voi > 0:
            oracle_selective_obs += 1
            oracle_selective_returns.append(post_best - observe_cost)
        else:
            oracle_selective_returns.append(pre_best_expected)

    oracle_selective_mean = sum(oracle_selective_returns) / n_objects if n_objects else 0.0
    oracle_selective_obs_rate = oracle_selective_obs / n_objects if n_objects else 0.0

    # ---- Learned Policy (TWO-STAGE, no decision-time leakage) ----
    # Stage 1: observe/direct decision using ONLY pre_vec.
    # Stage 2: only if observe chosen, reveal post_vec for action selection.
    policy_returns_gross = []
    policy_returns_net = []
    policy_actions_dist = defaultdict(int)
    policy_decisions = []
    policy_total_obs = 0
    policy_total_direct = 0

    for oid in test_oids:
        od = obj_data[oid]
        pre_vec = od["pre_vec"]
        true_returns = od["true_returns"]

        # Stage 1: predict observe value from pre_vec ONLY
        pre_qs = estimator.predict_all_actions(pre_vec)
        pre_try_qs = {ak: q for ak, q in pre_qs.items() if ak != "observe"}
        pre_best_action = max(pre_try_qs, key=pre_try_qs.get)
        pre_best_value = pre_try_qs[pre_best_action]

        if ov_model is not None and ov_normalizer is not None:
            pre_vec_norm = ov_normalizer.transform([pre_vec])[0]
            predicted_observe_net = ov_model.predict([pre_vec_norm])[0]
        else:
            predicted_observe_net = -observe_cost  # never observe without model

        decided_observe = predicted_observe_net > 0

        if decided_observe:
            # Stage 2: reveal post_observe_features ONLY AFTER deciding to observe
            policy_total_obs += 1
            post_vec = od["post_vec"]
            post_qs = estimator.predict_all_actions(post_vec)
            post_try_qs = {ak: q for ak, q in post_qs.items() if ak != "observe"}
            post_best_action = max(post_try_qs, key=post_try_qs.get)
            post_best_value = post_try_qs[post_best_action]
            chosen_action = post_best_action
            true_ret_gross = true_returns.get(chosen_action, 0.0)
            true_ret_net = true_ret_gross - observe_cost
        else:
            policy_total_direct += 1
            chosen_action = pre_best_action
            true_ret_gross = true_returns.get(chosen_action, 0.0)
            true_ret_net = true_ret_gross
            post_best_action = None
            post_best_value = 0.0

        policy_returns_gross.append(true_ret_gross)
        policy_returns_net.append(true_ret_net)
        policy_actions_dist[chosen_action] += 1

        policy_decisions.append({
            "oid": oid,
            "category": od["category"],
            "subtype": od["subtype"],
            "rationale_class": od["rationale_class"],
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

    policy_mean_gross = sum(policy_returns_gross) / n_objects if n_objects else 0.0
    policy_mean_net = sum(policy_returns_net) / n_objects if n_objects else 0.0
    policy_obs_rate = policy_total_obs / n_objects if n_objects else 0.0
    policy_direct_rate = policy_total_direct / n_objects if n_objects else 0.0

    # ---- Per-rationale-class breakdown (audit only) ----
    pos_net_decisions = [d for d in policy_decisions if d["rationale_class"] == "positive_change"]
    nonpos_net_decisions = [d for d in policy_decisions if d["rationale_class"] in ("wasteful_cost_only", "confirmatory")]

    pos_net_obs_rate = (sum(1 for d in pos_net_decisions if d["decided_observe"]) /
                         len(pos_net_decisions) if pos_net_decisions else 0.0)
    nonpos_net_obs_rate = (sum(1 for d in nonpos_net_decisions if d["decided_observe"]) /
                            len(nonpos_net_decisions) if nonpos_net_decisions else 0.0)

    # Harmful/cost-only observe rate: non-positive cases where policy observed anyway
    harmful_obs = sum(1 for d in nonpos_net_decisions if d["decided_observe"])

    # ---- Per-category breakdown ----
    cat_breakdown = {}
    for cat in CATEGORIES:
        cat_decisions = [d for d in policy_decisions if d["category"] == cat]
        if not cat_decisions:
            continue
        cat_net = [d["true_return_net"] for d in cat_decisions]
        cat_obs = sum(1 for d in cat_decisions if d["decided_observe"])
        cat_oracle_vals = []
        for oid2 in test_oids:
            if obj_data[oid2]["category"] == cat:
                tr = obj_data[oid2]["true_returns"]
                cat_oracle_vals.append(max(tr.values()))
        cat_breakdown[cat] = {
            "count": len(cat_decisions),
            "policy_mean_net": sum(cat_net) / len(cat_net) if cat_net else 0.0,
            "oracle_mean": sum(cat_oracle_vals) / len(cat_oracle_vals) if cat_oracle_vals else 0.0,
            "obs_rate": cat_obs / len(cat_decisions) if cat_decisions else 0.0,
        }

    # ---- Per n_features_known breakdown ----
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
            "policy_mean_net": sum(data["nets"]) / len(data["nets"]) if data["nets"] else 0.0,
            "obs_rate": data["obs"] / data["total"] if data["total"] > 0 else 0.0,
        }

    # ---- Q vs true error ----
    q_errors = []
    for d in policy_decisions:
        true_ret = d["true_return_gross"]
        pred_q = d["pre_best_value"] if not d["decided_observe"] else d["post_best_value"]
        q_errors.append(abs(pred_q - true_ret))

    mean_q_error = sum(q_errors) / len(q_errors) if q_errors else 0.0

    return {
        "test_seed": test_seed,
        "n_objects": n_objects,
        "pre_mean_return": round(pre_mean_return, 4),
        "always_try_mean": round(always_try_mean, 4),
        "always_observe_gross": round(post_mean_gross, 4),
        "always_observe_net": round(post_mean_net, 4),
        "random_mean_return": round(random_mean_return, 4),
        "oracle_selective_mean": round(oracle_selective_mean, 4),
        "oracle_selective_obs_rate": round(oracle_selective_obs_rate, 4),
        "policy_mean_gross": round(policy_mean_gross, 4),
        "policy_mean_net": round(policy_mean_net, 4),
        "policy_obs_rate": round(policy_obs_rate, 4),
        "policy_direct_try_rate": round(policy_direct_rate, 4),
        "pos_net_obs_rate": round(pos_net_obs_rate, 4),
        "nonpos_net_obs_rate": round(nonpos_net_obs_rate, 4),
        "harmful_observe_count": harmful_obs,
        "mean_q_error": round(mean_q_error, 4),
        "pre_actions_dist": dict(pre_actions_dist),
        "post_actions_dist": dict(post_actions_dist),
        "policy_actions_dist": dict(policy_actions_dist),
        "policy_decisions": policy_decisions,
        "cat_breakdown": cat_breakdown,
        "nfeat_summary": nfeat_summary,
        "obj_data": obj_data,
        "ov_target_source": ov_target_source,
        "ov_true_target_corr": ov_true_target_corr,
    }


# =============================================================================
# Run for a given observe_cost
# =============================================================================
def run_experiment(observe_cost, cost_label):
    print(f"\n{'='*70}")
    print(f"Running experiment: observe_cost = {observe_cost} ({cost_label})")
    print(f"{'='*70}")

    # Build training records with this observe_cost
    per_seed_records = build_training_records(per_seed_objects, per_seed_audit_labels, observe_cost)

    # Build all-records list for LOSO
    all_records = []
    for seed in SEEDS:
        all_records.extend(per_seed_records[seed])
    print(f"  Training records: {len(all_records)} ({sum(1 for r in all_records if r['action_key']=='observe')} observe, {sum(1 for r in all_records if r['action_key']!='observe')} try)")

    # LOSO folds
    loso_folds = {}
    for heldout_seed in SEEDS:
        loso_train = [r for r in all_records if r["seed"] != heldout_seed]
        loso_test = [r for r in all_records if r["seed"] == heldout_seed]
        loso_folds[heldout_seed] = (loso_train, loso_test)

    # Run policy intervention for each seed
    all_loso_results = {}
    all_policy_decisions = []

    for test_seed in SEEDS:
        train_recs, _ = loso_folds[test_seed]
        train_seeds = [s for s in SEEDS if s != test_seed]
        result = run_policy_intervention(
            train_recs,
            per_seed_objects[test_seed], per_seed_audit_labels[test_seed],
            test_seed, observe_cost, train_seeds)
        all_loso_results[test_seed] = result
        all_policy_decisions.extend(result["policy_decisions"])
        print(f"  Seed {test_seed}: pre={result['pre_mean_return']:.4f}, "
              f"policy_net={result['policy_mean_net']:.4f}, "
              f"oracle_sel={result['oracle_selective_mean']:.4f}, "
              f"obs_rate={result['policy_obs_rate']:.4f}, "
              f"direct={result['policy_direct_try_rate']:.4f}, "
              f"pos_obs={result['pos_net_obs_rate']:.4f}, "
              f"nonpos_obs={result['nonpos_net_obs_rate']:.4f}")

    # ---- Aggregate ----
    n_objects_total = sum(r["n_objects"] for r in all_loso_results.values())

    agg_pre = sum(r["pre_mean_return"] * r["n_objects"] for r in all_loso_results.values()) / n_objects_total
    agg_always_try = sum(r["always_try_mean"] * r["n_objects"] for r in all_loso_results.values()) / n_objects_total
    agg_always_obs_gross = sum(r["always_observe_gross"] * r["n_objects"] for r in all_loso_results.values()) / n_objects_total
    agg_always_obs_net = sum(r["always_observe_net"] * r["n_objects"] for r in all_loso_results.values()) / n_objects_total
    agg_random = sum(r["random_mean_return"] * r["n_objects"] for r in all_loso_results.values()) / n_objects_total
    agg_oracle_sel = sum(r["oracle_selective_mean"] * r["n_objects"] for r in all_loso_results.values()) / n_objects_total
    agg_policy_gross = sum(r["policy_mean_gross"] * r["n_objects"] for r in all_loso_results.values()) / n_objects_total
    agg_policy_net = sum(r["policy_mean_net"] * r["n_objects"] for r in all_loso_results.values()) / n_objects_total

    total_obs = sum(r["policy_obs_rate"] * r["n_objects"] for r in all_loso_results.values())
    agg_obs_rate = total_obs / n_objects_total if n_objects_total > 0 else 0.0
    total_direct = sum(r["policy_direct_try_rate"] * r["n_objects"] for r in all_loso_results.values())
    agg_direct_rate = total_direct / n_objects_total if n_objects_total > 0 else 0.0

    # Positive/non-positive observe rates
    total_pos_obs = sum(
        sum(1 for d in r["policy_decisions"] if d["rationale_class"] == "positive_change" and d["decided_observe"])
        for r in all_loso_results.values())
    total_pos = sum(
        sum(1 for d in r["policy_decisions"] if d["rationale_class"] == "positive_change")
        for r in all_loso_results.values())
    total_nonpos_obs = sum(
        sum(1 for d in r["policy_decisions"] if d["rationale_class"] in ("wasteful_cost_only", "confirmatory") and d["decided_observe"])
        for r in all_loso_results.values())
    total_nonpos = sum(
        sum(1 for d in r["policy_decisions"] if d["rationale_class"] in ("wasteful_cost_only", "confirmatory"))
        for r in all_loso_results.values())

    agg_pos_obs_rate = total_pos_obs / total_pos if total_pos > 0 else 0.0
    agg_nonpos_obs_rate = total_nonpos_obs / total_nonpos if total_nonpos > 0 else 0.0

    # Harmful observe rate = nonpos cases where policy observed
    agg_harmful_rate = total_nonpos_obs / total_nonpos if total_nonpos > 0 else 0.0

    # Gaps
    policy_pre_gain = agg_policy_net - agg_pre
    policy_oracle_gap = agg_oracle_sel - agg_policy_net
    policy_always_gap = agg_policy_net - agg_always_obs_net
    policy_always_try_gap = agg_policy_net - agg_always_try

    # Mean Q error
    agg_mean_q_error = sum(r["mean_q_error"] * r["n_objects"] for r in all_loso_results.values()) / n_objects_total

    # Per-category aggregate
    agg_cat = {}
    for cat in CATEGORIES:
        cat_nets = [d["true_return_net"] for d in all_policy_decisions if d["category"] == cat]
        cat_obs = sum(1 for d in all_policy_decisions if d["category"] == cat and d["decided_observe"])
        agg_cat[cat] = {
            "count": len(cat_nets),
            "policy_mean_net": sum(cat_nets) / len(cat_nets) if cat_nets else 0.0,
            "obs_rate": cat_obs / len(cat_nets) if cat_nets else 0.0,
        }

    # Per n_features_known aggregate
    agg_nfeat = defaultdict(lambda: {"nets": [], "obs": 0, "total": 0})
    for d in all_policy_decisions:
        nf = d["n_pre_surface_features"]
        agg_nfeat[nf]["nets"].append(d["true_return_net"])
        agg_nfeat[nf]["total"] += 1
        if d["decided_observe"]:
            agg_nfeat[nf]["obs"] += 1

    agg_nfeat_summary = {}
    for nf in sorted(agg_nfeat.keys()):
        data = agg_nfeat[nf]
        agg_nfeat_summary[nf] = {
            "count": data["total"],
            "policy_mean_net": sum(data["nets"]) / len(data["nets"]) if data["nets"] else 0.0,
            "obs_rate": data["obs"] / data["total"] if data["total"] > 0 else 0.0,
        }

    # Action distributions
    agg_pre_actions = defaultdict(int)
    agg_post_actions = defaultdict(int)
    agg_policy_actions = defaultdict(int)
    for r in all_loso_results.values():
        for ak, cnt in r["pre_actions_dist"].items():
            agg_pre_actions[ak] += cnt
        for ak, cnt in r["post_actions_dist"].items():
            agg_post_actions[ak] += cnt
        for ak, cnt in r["policy_actions_dist"].items():
            agg_policy_actions[ak] += cnt

    # ---- Hidden/audit field audit ----
    hidden_keys_in_model = [k for k in ALL_MODEL_FEATURE_KEYS if "hfeat" in k.lower()
                            or k.lower().startswith("state_") and k.replace("state_", "") in ["hidden"]]
    # Check that situation_features dicts don't contain hidden data
    hidden_feature_leak = False
    for seed in SEEDS:
        for r in per_seed_records[seed]:
            sf = r["situation_features"]
            for k in sf:
                if any(ff.lower() in k.lower().replace("feat_", "").replace("state_", "")
                       for ff in ["category", "subtype", "audit", "voi", "oracle", "affordance",
                                  "hidden", "object_id", "seed_id", "depth", "episode"]):
                    hidden_feature_leak = True
                    break

    # ---- Acceptance checks ----
    checks = {}
    checks["learned_policy_net > always_observe_net"] = agg_policy_net > agg_always_obs_net
    checks["learned_policy_net > always_try"] = agg_policy_net > agg_always_try
    checks["observe_rate between 0 and 1"] = 0.0 < agg_obs_rate < 1.0
    checks["pos_obs_rate > nonpos_obs_rate"] = agg_pos_obs_rate > agg_nonpos_obs_rate
    checks["hidden_audit_fields_absent"] = len(forbidden_in_model) == 0 and not hidden_feature_leak
    checks["forbidden_fields_absent"] = len(forbidden_in_model) == 0

    all_pass = all(checks.values())
    failure_reasons = [k for k, v in checks.items() if not v]

    # Determine status
    if all_pass:
        status = "PASS"
    elif agg_obs_rate > 0.0 and agg_obs_rate < 1.0 and checks["pos_obs_rate > nonpos_obs_rate"]:
        status = "PARTIAL"
    elif agg_obs_rate >= 1.0 or agg_obs_rate <= 0.0:
        status = "FAIL"
    else:
        status = "FAIL"

    print(f"\n  --- Aggregate Results ({cost_label}) ---")
    print(f"    pre_mean:                    {agg_pre:.4f}")
    print(f"    always_try:                  {agg_always_try:.4f}")
    print(f"    always_observe_net:          {agg_always_obs_net:.4f}")
    print(f"    random_observe:              {agg_random:.4f}")
    print(f"    oracle_selective:            {agg_oracle_sel:.4f}")
    print(f"    learned_policy_net:          {agg_policy_net:.4f}")
    print(f"    learned - always_observe:    {policy_always_gap:+.4f}")
    print(f"    learned - always_try:        {policy_always_try_gap:+.4f}")
    print(f"    learned - oracle gap:        {policy_oracle_gap:.4f}")
    print(f"    observe_rate:                {agg_obs_rate:.4f}")
    print(f"    direct_try_rate:             {agg_direct_rate:.4f}")
    print(f"    pos_net observe rate:        {agg_pos_obs_rate:.4f}")
    print(f"    nonpos_net observe rate:     {agg_nonpos_obs_rate:.4f}")
    print(f"    harmful/cost-only obs rate:  {agg_harmful_rate:.4f}")
    print(f"    mean |Q - true| error:       {agg_mean_q_error:.4f}")

    print(f"\n  Per-category:")
    for cat in CATEGORIES:
        print(f"    {cat}: policy_net={agg_cat[cat]['policy_mean_net']:.4f}, obs_rate={agg_cat[cat]['obs_rate']:.4f}")

    print(f"\n  Per n_features_known:")
    for nf in sorted(agg_nfeat_summary.keys()):
        s = agg_nfeat_summary[nf]
        print(f"    n={nf}: n={s['count']}, policy_net={s['policy_mean_net']:.4f}, obs_rate={s['obs_rate']:.4f}")

    print(f"\n  Acceptance checks:")
    for check_name, result in checks.items():
        status_str = "PASS" if result else "FAIL"
        print(f"    {check_name}: {status_str}")

    print(f"\n  overall: {status}")
    if failure_reasons:
        print(f"  failures: {failure_reasons}")

    return {
        "observe_cost": observe_cost,
        "cost_label": cost_label,
        "status": status,
        "checks": checks,
        "failure_reasons": failure_reasons,
        "aggregate": {
            "pre_mean": round(agg_pre, 4),
            "always_try": round(agg_always_try, 4),
            "always_observe_net": round(agg_always_obs_net, 4),
            "always_observe_gross": round(agg_always_obs_gross, 4),
            "random_observe": round(agg_random, 4),
            "oracle_selective_net": round(agg_oracle_sel, 4),
            "learned_policy_net": round(agg_policy_net, 4),
            "learned_policy_gross": round(agg_policy_gross, 4),
            "learned_minus_always_observe": round(policy_always_gap, 4),
            "learned_minus_always_try": round(policy_always_try_gap, 4),
            "learned_minus_oracle": round(policy_oracle_gap, 4),
            "observe_rate": round(agg_obs_rate, 4),
            "direct_try_rate": round(agg_direct_rate, 4),
            "pos_net_observe_rate": round(agg_pos_obs_rate, 4),
            "nonpos_net_observe_rate": round(agg_nonpos_obs_rate, 4),
            "harmful_observe_rate": round(agg_harmful_rate, 4),
            "mean_q_error": round(agg_mean_q_error, 4),
        },
        "per_seed": all_loso_results,
        "per_category": agg_cat,
        "per_nfeat": agg_nfeat_summary,
        "action_distributions": {
            "pre": dict(agg_pre_actions),
            "always_observe": dict(agg_post_actions),
            "policy": dict(agg_policy_actions),
        },
        "all_policy_decisions": all_policy_decisions,
    }


# =============================================================================
# Part 5: Run Primary and Robustness
# =============================================================================
print("[5/8] Running primary (cost=0.03) experiment...")
results_primary = run_experiment(0.03, "primary")

print("\n[6/8] Running robustness (cost=0.05) experiment...")
results_robustness = run_experiment(0.05, "robustness")

# =============================================================================
# Part 6: Feature Audit (hidden/forbidden)
# =============================================================================
print("\n[7/8] Running feature audit...")

# Verify hidden features are absent from model feature keys
hidden_in_pre = any(
    any(hf.lower() in k.lower().replace("feat_", "") for hf in ["hidden", "hfeat"])
    for k in PRE_FEATURE_KEYS
)
hidden_in_post = any(
    any(hf.lower() in k.lower().replace("feat_", "") for hf in ["hidden", "hfeat"])
    for k in POST_FEATURE_KEYS
)

# Hidden before reveal check
hidden_before_reveal_violations = 0
for seed in SEEDS:
    for oid, obj in per_seed_objects[seed].items():
        lbl = per_seed_audit_labels[seed][oid]
        pre_surf = set(obj["pre_surface_features"].keys())
        post_rev = set(lbl["post_observe_revealed"])
        for f in post_rev:
            if f in pre_surf:
                hidden_before_reveal_violations += 1

print(f"  Hidden features in pre keys: {hidden_in_pre}")
print(f"  Hidden features in post keys: {hidden_in_post}")
print(f"  Hidden-before-reveal violations: {hidden_before_reveal_violations}")
print(f"  Forbidden fields in model: {len(forbidden_in_model)}")

# =============================================================================
# Part 7: Compute audit VOI for reporting
# =============================================================================
print("\n[8/8] Computing audit VOI for reporting...")
per_object_voi_003, sig_expected_003 = compute_audit_voi(per_seed_objects, per_seed_audit_labels, 0.03)
per_object_voi_005, sig_expected_005 = compute_audit_voi(per_seed_objects, per_seed_audit_labels, 0.05)

# VOI class counts at 0.03
pos_count_003 = sum(1 for seed in SEEDS for oid, v in per_object_voi_003[seed].items()
                    if v["numeric_class"] == "positive_net")
nonpos_count_003 = sum(1 for seed in SEEDS for oid, v in per_object_voi_003[seed].items()
                       if v["numeric_class"] == "non_positive_net")

pos_count_005 = sum(1 for seed in SEEDS for oid, v in per_object_voi_005[seed].items()
                    if v["numeric_class"] == "positive_net")
nonpos_count_005 = sum(1 for seed in SEEDS for oid, v in per_object_voi_005[seed].items()
                       if v["numeric_class"] == "non_positive_net")

# =============================================================================
# Output
# =============================================================================
elapsed = round(time.time() - t0, 1)

primary_status = results_primary["status"]
robustness_status = results_robustness["status"]

# Determine overall: primary must PASS, robustness reported regardless
if primary_status == "PASS":
    overall_status = "PASS"
elif primary_status == "PARTIAL":
    overall_status = "PARTIAL"
else:
    overall_status = "FAIL"

# ---- JSON ----
json_output = {
    "block_id": "1J40b-7b",
    "elapsed_seconds": elapsed,
    "seeds": SEEDS,
    "implementation_status": overall_status,
    "primary_observe_cost": 0.03,
    "robustness_observe_cost": 0.05,
    "feature_audit": {
        "n_pre_surface_features": len(PRE_SURFACE_FEATURE_NAMES),
        "n_post_observe_features": len(POST_OBSERVE_FEATURE_NAMES),
        "n_model_feature_keys": len(ALL_MODEL_FEATURE_KEYS),
        "hidden_in_pre_keys": hidden_in_pre,
        "hidden_in_post_keys": hidden_in_post,
        "hidden_before_reveal_violations": hidden_before_reveal_violations,
        "forbidden_fields_in_model": len(forbidden_in_model),
        "pre_surface_feature_names": PRE_SURFACE_FEATURE_NAMES,
        "post_observe_feature_names": POST_OBSERVE_FEATURE_NAMES,
    },
    "audit_voi": {
        "cost_0.03": {"positive_net": pos_count_003, "non_positive_net": nonpos_count_003,
                      "pos_rate": round(pos_count_003 / (pos_count_003 + nonpos_count_003), 4)},
        "cost_0.05": {"positive_net": pos_count_005, "non_positive_net": nonpos_count_005,
                      "pos_rate": round(pos_count_005 / (pos_count_005 + nonpos_count_005), 4)},
    },
    "primary": {
        "cost": 0.03,
        "status": primary_status,
        "aggregate": results_primary["aggregate"],
        "checks": {k: v for k, v in results_primary["checks"].items()},
        "per_category": {},
        "per_nfeat": {},
        "per_seed": {},
    },
    "robustness": {
        "cost": 0.05,
        "status": robustness_status,
        "aggregate": results_robustness["aggregate"],
        "checks": {k: v for k, v in results_robustness["checks"].items()},
        "per_category": {},
        "per_nfeat": {},
        "per_seed": {},
    },
}

for cat in CATEGORIES:
    json_output["primary"]["per_category"][cat] = {
        "count": results_primary["per_category"][cat]["count"],
        "policy_mean_net": round(results_primary["per_category"][cat]["policy_mean_net"], 4),
        "obs_rate": round(results_primary["per_category"][cat]["obs_rate"], 4),
    }
    json_output["robustness"]["per_category"][cat] = {
        "count": results_robustness["per_category"][cat]["count"],
        "policy_mean_net": round(results_robustness["per_category"][cat]["policy_mean_net"], 4),
        "obs_rate": round(results_robustness["per_category"][cat]["obs_rate"], 4),
    }

for nf in sorted(results_primary["per_nfeat"].keys()):
    json_output["primary"]["per_nfeat"][str(nf)] = results_primary["per_nfeat"][nf]
for nf in sorted(results_robustness["per_nfeat"].keys()):
    json_output["robustness"]["per_nfeat"][str(nf)] = results_robustness["per_nfeat"][nf]

for seed in SEEDS:
    r = results_primary["per_seed"][seed]
    json_output["primary"]["per_seed"][str(seed)] = {
        "pre_mean": r["pre_mean_return"],
        "always_observe_net": r["always_observe_net"],
        "oracle_selective": r["oracle_selective_mean"],
        "policy_net": r["policy_mean_net"],
        "obs_rate": r["policy_obs_rate"],
        "direct_try_rate": r["policy_direct_try_rate"],
        "pos_net_obs_rate": r["pos_net_obs_rate"],
        "nonpos_net_obs_rate": r["nonpos_net_obs_rate"],
        "mean_q_error": r["mean_q_error"],
    }
    r2 = results_robustness["per_seed"][seed]
    json_output["robustness"]["per_seed"][str(seed)] = {
        "pre_mean": r2["pre_mean_return"],
        "always_observe_net": r2["always_observe_net"],
        "oracle_selective": r2["oracle_selective_mean"],
        "policy_net": r2["policy_mean_net"],
        "obs_rate": r2["policy_obs_rate"],
        "direct_try_rate": r2["policy_direct_try_rate"],
        "pos_net_obs_rate": r2["pos_net_obs_rate"],
        "nonpos_net_obs_rate": r2["nonpos_net_obs_rate"],
        "mean_q_error": r2["mean_q_error"],
    }

json_path = os.path.join(CURRENT_DIR, "runs", "block1j40b7b_env3b_shadow_learner_policy.json")
with open(json_path, "w") as f:
    json.dump(json_output, f, indent=2)
print(f"\nJSON -> {json_path}")

# ---- CSV ----
import csv as _csv
csv_path = os.path.join(CURRENT_DIR, "protocols", "block1j40b7b_env3b_shadow_learner_policy_table.csv")
with open(csv_path, "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["block_id", "1J40b-7b"])
    w.writerow(["implementation_status", overall_status])
    w.writerow(["elapsed_seconds", elapsed])

    for cost_label, results in [("primary_0.03", results_primary), ("robustness_0.05", results_robustness)]:
        agg = results["aggregate"]
        w.writerow([f"{cost_label}_status", results["status"]])
        w.writerow([f"{cost_label}_pre_mean", agg["pre_mean"]])
        w.writerow([f"{cost_label}_always_try", agg["always_try"]])
        w.writerow([f"{cost_label}_always_observe_net", agg["always_observe_net"]])
        w.writerow([f"{cost_label}_oracle_selective", agg["oracle_selective_net"]])
        w.writerow([f"{cost_label}_learned_policy_net", agg["learned_policy_net"]])
        w.writerow([f"{cost_label}_learned_minus_always_observe", agg["learned_minus_always_observe"]])
        w.writerow([f"{cost_label}_learned_minus_always_try", agg["learned_minus_always_try"]])
        w.writerow([f"{cost_label}_learned_minus_oracle", agg["learned_minus_oracle"]])
        w.writerow([f"{cost_label}_observe_rate", agg["observe_rate"]])
        w.writerow([f"{cost_label}_direct_try_rate", agg["direct_try_rate"]])
        w.writerow([f"{cost_label}_pos_net_obs_rate", agg["pos_net_observe_rate"]])
        w.writerow([f"{cost_label}_nonpos_net_obs_rate", agg["nonpos_net_observe_rate"]])
        w.writerow([f"{cost_label}_harmful_obs_rate", agg["harmful_observe_rate"]])
        w.writerow([f"{cost_label}_mean_q_error", agg["mean_q_error"]])
        for seed in SEEDS:
            r = results["per_seed"][seed]
            w.writerow([f"{cost_label}_seed{seed}_pre", r["pre_mean_return"]])
            w.writerow([f"{cost_label}_seed{seed}_policy_net", r["policy_mean_net"]])
            w.writerow([f"{cost_label}_seed{seed}_obs_rate", r["policy_obs_rate"]])
        for cat in CATEGORIES:
            w.writerow([f"{cost_label}_cat_{cat}_policy_net", round(results["per_category"][cat]["policy_mean_net"], 4)])
            w.writerow([f"{cost_label}_cat_{cat}_obs_rate", round(results["per_category"][cat]["obs_rate"], 4)])

print(f"CSV -> {csv_path}")

# ---- MD ----
md_path = os.path.join(CURRENT_DIR, "protocols", "block1j40b7b_env3b_shadow_learner_policy.md")
with open(md_path, "w") as f:
    f.write("# Block 1J40b-7b: env3b Shadow Learner + Policy Intervention\n\n")
    f.write(f"- **Implementation Status**: {overall_status}\n")
    f.write(f"- **Seeds**: {SEEDS}\n")
    f.write(f"- **Total elapsed**: {elapsed}s\n\n")

    f.write("## 1. Files\n\n")
    f.write("- `_block1j40b7b_env3b_shadow_learner_policy.py` — Created\n")
    f.write("- `runs/block1j40b7b_env3b_shadow_learner_policy.json` — Generated\n")
    f.write("- `protocols/block1j40b7b_env3b_shadow_learner_policy.md` — Generated\n")
    f.write("- `protocols/block1j40b7b_env3b_shadow_learner_policy_table.csv` — Generated\n\n")

    f.write("## 2. Design\n\n")
    f.write("**Algorithm**: Per-action ridge regression (alpha=1.0) with feature normalization\n\n")
    f.write("**Training**: LOSO (leave-one-seed-out) across 5 seeds\n\n")
    f.write("**Feature space**:\n")
    f.write(f"- Pre-surface features: {len(PRE_SURFACE_FEATURE_NAMES)}\n")
    f.write(f"- Post-observe features: {len(POST_OBSERVE_FEATURE_NAMES)}\n")
    f.write(f"- Model keys: {len(ALL_MODEL_FEATURE_KEYS)}\n")
    f.write("- Hidden features: absent from model input\n\n")

    f.write("**Policy chain** (per object):\n")
    f.write("1. Build pre-observe legal feature vector (pre_surface_features only, prior_obs=0)\n")
    f.write("2. Estimate pre_best_try_value and post_best_try_value from learned Q\n")
    f.write("3. `predicted_observe_net = post_best - pre_best - OBSERVE_COST`\n")
    f.write("4. If > 0: execute observe, select best post try, score true_return - cost\n")
    f.write("5. Else: execute best pre try, score true_return\n\n")

    f.write("**Observe costs**:\n")
    f.write("- Primary: 0.03\n")
    f.write("- Robustness: 0.05\n\n")

    f.write("## 3. Primary Results (cost=0.03)\n\n")
    agg = results_primary["aggregate"]
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    f.write(f"| pre_mean | {agg['pre_mean']:.4f} |\n")
    f.write(f"| always_try | {agg['always_try']:.4f} |\n")
    f.write(f"| always_observe_net | {agg['always_observe_net']:.4f} |\n")
    f.write(f"| random_observe | {agg['random_observe']:.4f} |\n")
    f.write(f"| oracle_selective | {agg['oracle_selective_net']:.4f} |\n")
    f.write(f"| **learned_policy_net** | **{agg['learned_policy_net']:.4f}** |\n")
    f.write(f"| learned - always_observe | {agg['learned_minus_always_observe']:+.4f} |\n")
    f.write(f"| learned - always_try | {agg['learned_minus_always_try']:+.4f} |\n")
    f.write(f"| learned - oracle gap | {agg['learned_minus_oracle']:.4f} |\n")
    f.write(f"| observe_rate | {agg['observe_rate']:.4f} |\n")
    f.write(f"| direct_try_rate | {agg['direct_try_rate']:.4f} |\n")
    f.write(f"| pos_net observe rate | {agg['pos_net_observe_rate']:.4f} |\n")
    f.write(f"| nonpos_net observe rate | {agg['nonpos_net_observe_rate']:.4f} |\n")
    f.write(f"| harmful/cost-only obs rate | {agg['harmful_observe_rate']:.4f} |\n")
    f.write(f"| mean \\|Q - true\\| error | {agg['mean_q_error']:.4f} |\n\n")

    f.write("### Per-Seed (0.03)\n\n")
    f.write("| Seed | Pre | Policy Net | Oracle Sel | Obs Rate | Direct | PosObs | NonposObs |\n")
    f.write("|------|-----|------------|------------|----------|--------|--------|-----------|\n")
    for seed in SEEDS:
        r = results_primary["per_seed"][seed]
        f.write(f"| {seed} | {r['pre_mean_return']:.4f} | {r['policy_mean_net']:.4f} | "
                f"{r['oracle_selective_mean']:.4f} | {r['policy_obs_rate']:.4f} | "
                f"{r['policy_direct_try_rate']:.4f} | {r['pos_net_obs_rate']:.4f} | "
                f"{r['nonpos_net_obs_rate']:.4f} |\n")

    f.write("\n### Per-Category (0.03)\n\n")
    f.write("| Category | Count | Policy Net | Obs Rate |\n")
    f.write("|----------|-------|------------|----------|\n")
    for cat in CATEGORIES:
        c = results_primary["per_category"][cat]
        f.write(f"| {cat} | {c['count']} | {c['policy_mean_net']:.4f} | {c['obs_rate']:.4f} |\n")

    f.write("\n### Per n_features_known (0.03)\n\n")
    f.write("| n_pre_features | Count | Policy Net | Obs Rate |\n")
    f.write("|----------------|-------|------------|----------|\n")
    for nf in sorted(results_primary["per_nfeat"].keys()):
        s = results_primary["per_nfeat"][nf]
        f.write(f"| {nf} | {s['count']} | {s['policy_mean_net']:.4f} | {s['obs_rate']:.4f} |\n")

    f.write("\n### Acceptance (0.03)\n\n")
    f.write("| Check | Result |\n")
    f.write("|-------|--------|\n")
    for check_name, result in results_primary["checks"].items():
        status_str = "PASS" if result else "FAIL"
        f.write(f"| {check_name} | **{status_str}** |\n")
    f.write(f"| **overall** | **{results_primary['status']}** |\n\n")

    f.write("## 4. Robustness Results (cost=0.05)\n\n")
    agg_r = results_robustness["aggregate"]
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    f.write(f"| pre_mean | {agg_r['pre_mean']:.4f} |\n")
    f.write(f"| always_try | {agg_r['always_try']:.4f} |\n")
    f.write(f"| always_observe_net | {agg_r['always_observe_net']:.4f} |\n")
    f.write(f"| oracle_selective | {agg_r['oracle_selective_net']:.4f} |\n")
    f.write(f"| **learned_policy_net** | **{agg_r['learned_policy_net']:.4f}** |\n")
    f.write(f"| learned - always_observe | {agg_r['learned_minus_always_observe']:+.4f} |\n")
    f.write(f"| learned - always_try | {agg_r['learned_minus_always_try']:+.4f} |\n")
    f.write(f"| learned - oracle gap | {agg_r['learned_minus_oracle']:.4f} |\n")
    f.write(f"| observe_rate | {agg_r['observe_rate']:.4f} |\n")
    f.write(f"| direct_try_rate | {agg_r['direct_try_rate']:.4f} |\n")
    f.write(f"| pos_net observe rate | {agg_r['pos_net_observe_rate']:.4f} |\n")
    f.write(f"| nonpos_net observe rate | {agg_r['nonpos_net_observe_rate']:.4f} |\n")
    f.write(f"| harmful/cost-only obs rate | {agg_r['harmful_observe_rate']:.4f} |\n")
    f.write(f"| mean \\|Q - true\\| error | {agg_r['mean_q_error']:.4f} |\n\n")

    f.write("### Per-Seed (0.05)\n\n")
    f.write("| Seed | Pre | Policy Net | Oracle Sel | Obs Rate | Direct | PosObs | NonposObs |\n")
    f.write("|------|-----|------------|------------|----------|--------|--------|-----------|\n")
    for seed in SEEDS:
        r = results_robustness["per_seed"][seed]
        f.write(f"| {seed} | {r['pre_mean_return']:.4f} | {r['policy_mean_net']:.4f} | "
                f"{r['oracle_selective_mean']:.4f} | {r['policy_obs_rate']:.4f} | "
                f"{r['policy_direct_try_rate']:.4f} | {r['pos_net_obs_rate']:.4f} | "
                f"{r['nonpos_net_obs_rate']:.4f} |\n")

    f.write("\n### Per-Category (0.05)\n\n")
    f.write("| Category | Count | Policy Net | Obs Rate |\n")
    f.write("|----------|-------|------------|----------|\n")
    for cat in CATEGORIES:
        c = results_robustness["per_category"][cat]
        f.write(f"| {cat} | {c['count']} | {c['policy_mean_net']:.4f} | {c['obs_rate']:.4f} |\n")

    f.write("\n### Per n_features_known (0.05)\n\n")
    f.write("| n_pre_features | Count | Policy Net | Obs Rate |\n")
    f.write("|----------------|-------|------------|----------|\n")
    for nf in sorted(results_robustness["per_nfeat"].keys()):
        s = results_robustness["per_nfeat"][nf]
        f.write(f"| {nf} | {s['count']} | {s['policy_mean_net']:.4f} | {s['obs_rate']:.4f} |\n")

    f.write("\n### Acceptance (0.05)\n\n")
    f.write("| Check | Result |\n")
    f.write("|-------|--------|\n")
    for check_name, result in results_robustness["checks"].items():
        status_str = "PASS" if result else "FAIL"
        f.write(f"| {check_name} | **{status_str}** |\n")
    f.write(f"| **overall** | **{results_robustness['status']}** |\n\n")

    f.write("## 5. Feature Audit\n\n")
    f.write(f"- pre_surface feature count: {len(PRE_SURFACE_FEATURE_NAMES)}\n")
    f.write(f"- post_observe feature count: {len(POST_OBSERVE_FEATURE_NAMES)}\n")
    f.write(f"- model feature keys: {len(ALL_MODEL_FEATURE_KEYS)}\n")
    f.write(f"- hidden features in pre keys: {hidden_in_pre}\n")
    f.write(f"- hidden features in post keys: {hidden_in_post}\n")
    f.write(f"- hidden-before-reveal violations: {hidden_before_reveal_violations}\n")
    f.write(f"- forbidden fields in model keys: {len(forbidden_in_model)}\n")
    f.write("- Hidden features absent from model input: PASS\n\n")

    f.write("## 6. Three-World Separation\n\n")
    f.write("- Environment: holds hidden truth (category, subtype, affordance, rationale)\n")
    f.write("- Agent/Model: only legal pre_surface_features and post_observe_features\n")
    f.write("- Audit: inspects hidden truth for evaluation only\n")
    f.write("- Forbidden fields: all confirmed absent from model input\n\n")

    f.write("## 7. Output Files\n\n")
    f.write("- `runs/block1j40b7b_env3b_shadow_learner_policy.json`\n")
    f.write("- `protocols/block1j40b7b_env3b_shadow_learner_policy.md`\n")
    f.write("- `protocols/block1j40b7b_env3b_shadow_learner_policy_table.csv`\n\n")

    f.write(f"\n```\n[block_done]\n")
    f.write(f"block_id=1J40b-7b\n")
    f.write(f"implementation_status={overall_status}\n")
    f.write(f"primary_status={primary_status}\n")
    f.write(f"robustness_status={robustness_status}\n")
    f.write(f"primary_policy_net={agg['learned_policy_net']:.4f}\n")
    f.write(f"primary_obs_rate={agg['observe_rate']:.4f}\n")
    f.write(f"robustness_policy_net={agg_r['learned_policy_net']:.4f}\n")
    f.write(f"robustness_obs_rate={agg_r['observe_rate']:.4f}\n```\n")

print(f"MD -> {md_path}")

print(f"\n{'='*70}")
print(f"Block 1J40b-7b complete.")
print(f"  primary (0.03):   status={primary_status}, policy_net={agg['learned_policy_net']:.4f}, obs_rate={agg['observe_rate']:.4f}")
print(f"  robustness (0.05): status={robustness_status}, policy_net={agg_r['learned_policy_net']:.4f}, obs_rate={agg_r['observe_rate']:.4f}")
print(f"  overall: {overall_status}")
print(f"  elapsed: {elapsed}s")
print(f"{'='*70}")
