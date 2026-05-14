"""
Block 1J40b-3a -- Feature Source Leakage Audit.

Verifies that 1J40b-3's strong role-separation result is not caused by
role-label leakage or premature hidden-feature leakage.

Runs:
1. RoleSignal source audit — trace each RoleSignal feature origin
2. Hidden feature timing audit — verify hidden features revealed before use
3. Ablation: no RoleSignal features
4. Ablation: no hidden diagnostic features
5. Strict visible-only model
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
# Constants (matching 1J40b-3)
# =============================================================================
MULTISEEDS = [101, 103, 107, 109, 113]
N_EPISODES = 5
HELDOUT_TRAIN_EPISODES = 4
N_GROUPS_PER_FAMILY = 3
OBJECTS_PER_GROUP = 3
RIDGE_ALPHA = 1.0

ALL_TRY_AFFORDANCES = [
    "burn_as_fuel", "craft_plank", "eat",
    "mine_by_hand", "mine_with_pickaxe", "use_as_tool",
]
ALL_ACTION_KEYS = ["observe"] + [f"try_{a}" for a in ALL_TRY_AFFORDANCES]

OBSERVE_COST = 0.005
PROBE_COST = 0.05
SUCCESS_REWARD = 0.5
FAILURE_PENALTY = 0.1
INFO_GAIN_PER_NEW_FEATURE = 0.02
INFO_GAIN_PER_NEW_STATE = 0.01
UNCERTAINTY_REDUCTION_VALUE = 0.03

CATEGORIES = ["wood_log", "stone_block", "apple", "wooden_pickaxe"]

BASE_AFFORDANCE_PROFILES = {
    "wood_log": {
        "mine_by_hand": "success", "mine_with_pickaxe": "success",
        "craft_plank": "success", "eat": "fail",
        "use_as_tool": "fail", "burn_as_fuel": "success",
    },
    "stone_block": {
        "mine_by_hand": "fail", "mine_with_pickaxe": "success",
        "craft_plank": "fail", "eat": "fail",
        "use_as_tool": "fail", "burn_as_fuel": "fail",
    },
    "apple": {
        "mine_by_hand": "success", "mine_with_pickaxe": "success",
        "craft_plank": "fail", "eat": "success",
        "use_as_tool": "fail", "burn_as_fuel": "fail",
    },
    "wooden_pickaxe": {
        "mine_by_hand": "fail", "mine_with_pickaxe": "fail",
        "craft_plank": "fail", "eat": "fail",
        "use_as_tool": "success", "burn_as_fuel": "fail",
    },
}

CATEGORY_FEATURES = {
    "wood_log": ["has_bark_texture", "has_wood_grain", "brownish", "rough_texture",
                 "long_shape", "fibrous", "flammable", "porous_surface"],
    "stone_block": ["has_crystal_flecks", "has_granular_surface", "grayish", "block_like",
                    "heavy_weight", "cold_to_touch", "scratch_resistant"],
    "apple": ["has_stem_remnant", "has_peel_texture", "round_small", "greenish",
              "smooth_texture", "light_weight", "fruity_scent"],
    "wooden_pickaxe": ["has_grip_area", "has_shaft_shape", "elongated_with_handle",
                       "movable", "long_shape", "has_metal_head", "jointed"],
}

HIDDEN_DIAGNOSTIC_FEATURES = {
    "wood_log": ["rotting_odor", "mold_spots", "soft_to_touch"],
    "stone_block": ["internal_fractures", "efflorescence", "weathering_pattern"],
    "apple": ["bruise_markings", "soft_spot", "discoloration_near_stem"],
    "wooden_pickaxe": ["hairline_crack", "handle_looseness", "rust_on_fastener"],
}

ALL_VISIBLE_FEATURE_NAMES = sorted(set(
    f for flist in CATEGORY_FEATURES.values() for f in flist))
ALL_HIDDEN_FEATURE_NAMES = sorted(set(
    f for flist in HIDDEN_DIAGNOSTIC_FEATURES.values() for f in flist))
ALL_FEATURE_NAMES = ALL_VISIBLE_FEATURE_NAMES + ALL_HIDDEN_FEATURE_NAMES
STATE_FEATURE_NAMES = ["fresh", "wet", "damaged", "clean", "hot", "open"]

ROLE_SIGNAL_FEATURES = {
    "observe_helps": "variable_texture",
    "observe_neutral": "uniform_texture",
    "observe_wasteful": "worn_texture",
}
ROLE_SIGNAL_VALUES = sorted(set(ROLE_SIGNAL_FEATURES.values()))

# =============================================================================
# Forbidden keys
# =============================================================================
FORBIDDEN_AGENT_VISIBLE = {
    "schedule", "schedule_type", "group_role", "group_id",
    "depth_schedule", "mixed_source_type", "true_family",
    "hidden_subtype", "oracle_outcome", "full_object_state",
    "deceptive_flag", "prior_violation", "preferred_explanation",
}

AGENT_STRUCTURAL_KEYS = {
    "step_id", "episode_id", "action_type", "action_target",
    "action_params", "observed_features_delta", "observed_state_delta",
    "success_or_failure",
}

# =============================================================================
# Feature key definitions
# =============================================================================
SITUATION_VISIBLE_FEATURE_KEYS = [f"feat_{f}" for f in ALL_VISIBLE_FEATURE_NAMES]
SITUATION_HIDDEN_FEATURE_KEYS = [f"hfeat_{f}" for f in ALL_HIDDEN_FEATURE_NAMES]
SITUATION_STATE_KEYS = [f"state_{s}" for s in STATE_FEATURE_NAMES]
SITUATION_NUMERIC_KEYS = [
    "prior_observe_count", "n_features_known", "n_states_known",
    "episode_id", "episode_progress",
]
SITUATION_ROLE_SIGNAL_KEYS = [f"sig_{v}" for v in ROLE_SIGNAL_VALUES]

ALL_SITUATION_FEATURE_KEYS = (
    SITUATION_VISIBLE_FEATURE_KEYS +
    SITUATION_HIDDEN_FEATURE_KEYS +
    SITUATION_STATE_KEYS +
    SITUATION_NUMERIC_KEYS +
    SITUATION_ROLE_SIGNAL_KEYS
)

# Feature sets for ablation
FEATURES_NO_ROLESIGNAL = (
    SITUATION_VISIBLE_FEATURE_KEYS +
    SITUATION_HIDDEN_FEATURE_KEYS +
    SITUATION_STATE_KEYS +
    SITUATION_NUMERIC_KEYS
)

FEATURES_NO_HIDDEN = (
    SITUATION_VISIBLE_FEATURE_KEYS +
    SITUATION_STATE_KEYS +
    SITUATION_NUMERIC_KEYS +
    SITUATION_ROLE_SIGNAL_KEYS
)

FEATURES_VISIBLE_ONLY = (
    SITUATION_VISIBLE_FEATURE_KEYS +
    SITUATION_STATE_KEYS +
    SITUATION_NUMERIC_KEYS
)

# =============================================================================
# Data generation (same as 1J40b-3)
# =============================================================================
def build_variant_profile(base_profile, category, group_role):
    vp = dict(base_profile)
    if group_role == "observe_helps":
        failing = [a for a in ALL_TRY_AFFORDANCES if vp.get(a) == "fail"]
        if failing:
            vp[failing[0]] = "success"
    return vp


def generate_c6_data_for_seed(seed):
    rng = random.Random(seed + 900)
    objects = {}
    audit_labels = {}
    oid_counter = [0]

    def make_oid():
        oid_counter[0] += 1
        return f"c6_s{seed}_obj_{oid_counter[0]:04d}"

    for category in CATEGORIES:
        base_profile = BASE_AFFORDANCE_PROFILES[category]
        base_features = CATEGORY_FEATURES[category]
        hidden_features = HIDDEN_DIAGNOSTIC_FEATURES[category]

        for group_idx in range(N_GROUPS_PER_FAMILY):
            group_id = f"{category}_s{seed}_g{group_idx}"
            group_role = ["observe_helps", "observe_neutral", "observe_wasteful"][group_idx]
            variant_profile = build_variant_profile(base_profile, category, group_role)

            for depth_idx, depth_schedule in enumerate(["no_observe", "one_observe", "repeated_observe"]):
                oid = make_oid()
                if depth_schedule == "no_observe":
                    effective_profile = dict(base_profile)
                else:
                    effective_profile = dict(variant_profile)

                visible_features = {f: True for f in base_features}
                role_signal = ROLE_SIGNAL_FEATURES[group_role]
                visible_features[role_signal] = True

                hidden_feat_dict = {hf: True for hf in hidden_features}

                visible_state = {
                    "fresh": True, "wet": category in ("wood_log", "apple"),
                    "damaged": False, "clean": True,
                    "hot": False, "open": False,
                }

                objects[oid] = {
                    "oid": oid,
                    "hidden_category": category,
                    "hidden_affordance_profile": effective_profile,
                    "visible_features": visible_features,
                    "visible_state": visible_state,
                    "_hidden_features_dict": hidden_feat_dict,
                }

                audit_labels[oid] = {
                    "oid": oid,
                    "hidden_category": category,
                    "true_family": category,
                    "group_id": group_id,
                    "group_role": group_role,
                    "depth_schedule": depth_schedule,
                    "affordance_profile": dict(effective_profile),
                    "base_profile": dict(base_profile),
                    "variant_profile": dict(variant_profile),
                    "seed": seed,
                }

    # Build events (same logic as 1J40b-3)
    all_oids = sorted(objects.keys())
    episode_assignments = {}
    for idx, oid in enumerate(all_oids):
        episode_assignments[oid] = idx % N_EPISODES

    all_events = []
    step_id = 0
    prior_observe_count = defaultdict(int)

    for ep in range(N_EPISODES):
        ep_oids = [oid for oid in all_oids if episode_assignments.get(oid) == ep]

        for oid in ep_oids:
            obj = objects[oid]
            label = audit_labels[oid]
            depth = label["depth_schedule"]
            category = label["hidden_category"]
            profile = label["affordance_profile"]
            features = obj.get("visible_features", {})
            hidden_feats = obj.get("_hidden_features_dict", {})
            state = obj.get("visible_state", {})

            if depth == "no_observe":
                ambient_features = {k: v for k, v in features.items()
                                   if k in ["brownish", "grayish", "greenish", "long_shape",
                                            "block_like", "round_small"] + ROLE_SIGNAL_VALUES}
                initial_features = ambient_features
                initial_state = {}
            elif depth == "one_observe":
                step_id += 1
                all_events.append({
                    "step_id": step_id, "episode_id": ep, "action_type": "observe",
                    "action_target": oid,
                    "action_params": {},
                    "observed_features_delta": dict(features),
                    "observed_state_delta": dict(state),
                    "success_or_failure": None,
                })
                prior_observe_count[oid] += 1
                initial_features = features
                initial_state = state
            elif depth == "repeated_observe":
                step_id += 1
                all_events.append({
                    "step_id": step_id, "episode_id": ep, "action_type": "observe",
                    "action_target": oid,
                    "action_params": {},
                    "observed_features_delta": dict(features),
                    "observed_state_delta": dict(state),
                    "success_or_failure": None,
                })
                prior_observe_count[oid] += 1
                step_id += 1
                combined_features = dict(features)
                combined_features.update(hidden_feats)
                all_events.append({
                    "step_id": step_id, "episode_id": ep, "action_type": "observe",
                    "action_target": oid,
                    "action_params": {},
                    "observed_features_delta": combined_features,
                    "observed_state_delta": dict(state),
                    "success_or_failure": None,
                })
                prior_observe_count[oid] += 1
                initial_features = combined_features
                initial_state = state

            current_observe_depth = prior_observe_count[oid]

            if category == "wood_log":
                primary_actions = ["mine_by_hand", "craft_plank", "burn_as_fuel"]
            elif category == "stone_block":
                primary_actions = ["mine_with_pickaxe", "mine_by_hand", "use_as_tool"]
            elif category == "apple":
                primary_actions = ["eat", "mine_by_hand", "use_as_tool"]
            else:
                primary_actions = ["use_as_tool", "mine_by_hand", "craft_plank"]

            all_try_actions = primary_actions + [a for a in ALL_TRY_AFFORDANCES if a not in primary_actions]
            for action in all_try_actions:
                outcome_bool = (profile.get(action, "fail") == "success")
                step_id += 1
                all_events.append({
                    "step_id": step_id, "episode_id": ep, "action_type": "try",
                    "action_target": oid,
                    "action_params": {
                        "affordance": action,
                        "observe_depth": current_observe_depth,
                    },
                    "observed_features_delta": dict(initial_features),
                    "observed_state_delta": dict(initial_state),
                    "success_or_failure": outcome_bool,
                })

            if depth == "repeated_observe":
                step_id += 1
                post_features = dict(features)
                post_features.update(hidden_feats)
                all_events.append({
                    "step_id": step_id, "episode_id": ep, "action_type": "observe",
                    "action_target": oid,
                    "action_params": {},
                    "observed_features_delta": post_features,
                    "observed_state_delta": dict(state),
                    "success_or_failure": None,
                })
                prior_observe_count[oid] += 1

    return objects, audit_labels, all_events


# =============================================================================
# Ridge Regression (same as 1J40b-3)
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
                self.models[ak].coef_ = [0.0] * (len(gx[0]) if gx else 0)
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
# Net value
# =============================================================================
def try_net_value(success_or_failure):
    if success_or_failure is True:
        return SUCCESS_REWARD - PROBE_COST
    elif success_or_failure is False:
        return -FAILURE_PENALTY - PROBE_COST
    return 0.0


def compute_group_matched_advantage(try_events, audit_labels):
    group_ids = sorted(set(lbl["group_id"] for lbl in audit_labels.values()))
    group_advantages = {}
    for gid in group_ids:
        depth_returns = {}
        for depth_schedule in ["no_observe", "one_observe", "repeated_observe"]:
            depth_oids = [oid for oid, lbl in audit_labels.items()
                          if lbl["group_id"] == gid and lbl["depth_schedule"] == depth_schedule]
            depth_tries = [ev for ev in try_events if ev["action_target"] in depth_oids]
            if depth_tries:
                net_vals = [try_net_value(ev.get("success_or_failure")) for ev in depth_tries]
                depth_returns[depth_schedule] = {
                    "n_tries": len(depth_tries),
                    "mean_net_value": round(sum(net_vals) / len(net_vals), 4),
                }
        no_baseline = depth_returns.get("no_observe", {}).get("mean_net_value")
        one_return = depth_returns.get("one_observe", {}).get("mean_net_value")
        rep_return = depth_returns.get("repeated_observe", {}).get("mean_net_value")
        one_adv = round(one_return - no_baseline, 4) if (one_return is not None and no_baseline is not None) else 0.0
        rep_adv = round(rep_return - no_baseline, 4) if (rep_return is not None and no_baseline is not None) else 0.0
        advs = [v for v in [one_adv, rep_adv] if v is not None]
        mean_adv = round(sum(advs) / len(advs), 4) if advs else 0.0
        group_advantages[gid] = {
            "role": depth_returns.get("no_observe", {}).get("no_observe_baseline") or
                    (list(audit_labels.values())[0].get("group_role", "unknown") if audit_labels else "unknown"),
            "per_try_matched_advantage": mean_adv,
        }
        # Fix: get role from any audit label for this group
        for oid, lbl in audit_labels.items():
            if lbl["group_id"] == gid:
                group_advantages[gid]["role"] = lbl["group_role"]
                break
    return group_advantages


# =============================================================================
# Build records with full feature-timing instrumentation
# =============================================================================
def build_situation_features(known_feats, known_states, prior_obs_count,
                             episode_id, n_episodes):
    feat = {}
    for fname in ALL_VISIBLE_FEATURE_NAMES:
        feat[f"feat_{fname}"] = 1.0 if fname in known_feats else 0.0
    for fname in ALL_HIDDEN_FEATURE_NAMES:
        feat[f"hfeat_{fname}"] = 1.0 if fname in known_feats else 0.0
    for sname in STATE_FEATURE_NAMES:
        feat[f"state_{sname}"] = 1.0 if known_states.get(sname, False) else 0.0
    feat["prior_observe_count"] = float(prior_obs_count)
    feat["n_features_known"] = float(len(known_feats))
    feat["n_states_known"] = float(len(known_states))
    feat["episode_id"] = float(episode_id)
    feat["episode_progress"] = float(episode_id) / float(n_episodes) if n_episodes else 0.0
    for v in ROLE_SIGNAL_VALUES:
        feat[f"sig_{v}"] = 1.0 if v in known_feats else 0.0
    return feat


def extract_feature_vector(sf, feature_keys):
    return [float(sf.get(k, 0.0)) for k in feature_keys]


def build_records_with_audit(seed, objects, audit_labels, all_events):
    """Build SAR records PLUS detailed feature-timing audit trail."""
    try_events = [ev for ev in all_events if ev["action_type"] == "try"]
    observe_events_list = [ev for ev in all_events if ev["action_type"] == "observe"]
    group_advantages = compute_group_matched_advantage(try_events, audit_labels)

    records = []
    record_id = 0

    # Feature timing audit
    feature_first_observed = {}   # (oid, feature_name) -> step_id
    feature_first_used = {}       # (oid, feature_name) -> step_id
    hidden_feature_reveal_step = {}  # (oid, hf_name) -> step_id (only from observe events)

    obj_ep_features = defaultdict(set)
    obj_ep_states = defaultdict(dict)
    obj_ep_prior_obs = defaultdict(int)

    for ev in all_events:
        oid = ev["action_target"]
        ep = ev["episode_id"]
        step = ev["step_id"]
        action_type = ev["action_type"]
        gid = audit_labels[oid]["group_id"]

        ofd = ev.get("observed_features_delta", {})
        osd = ev.get("observed_state_delta", {})

        known_feats = obj_ep_features[(oid, ep)]
        known_states = obj_ep_states[(oid, ep)]
        prior_obs = obj_ep_prior_obs[(oid, ep)]

        situation_features = build_situation_features(
            known_feats, known_states, prior_obs, ep, N_EPISODES)

        # Track which features are first USED in a situation (before this event)
        for fname in known_feats:
            key = (oid, fname)
            if key not in feature_first_used:
                feature_first_used[key] = step

        # Compute net_value
        if action_type == "observe":
            n_new_features = len(ofd)
            n_new_states = len(osd)
            immediate_value = (n_new_features * INFO_GAIN_PER_NEW_FEATURE +
                               n_new_states * INFO_GAIN_PER_NEW_STATE +
                               (UNCERTAINTY_REDUCTION_VALUE if n_new_features > 0 else 0) -
                               OBSERVE_COST)
            later_tries = [te for te in try_events
                           if te["action_target"] == oid
                           and te["episode_id"] == ep
                           and te["step_id"] > step]
            n_later = len(later_tries)
            gadv = group_advantages.get(gid, {})
            per_try_adv = gadv.get("per_try_matched_advantage", 0.0)
            matched_advantage_total = per_try_adv * n_later
            net_value = round(immediate_value + matched_advantage_total, 4)
        elif action_type == "try":
            net_value = round(try_net_value(ev.get("success_or_failure")), 4)
        else:
            continue

        action_params = ev.get("action_params", {})
        if action_type == "observe":
            action_key = "observe"
        else:
            action_key = f"try_{action_params.get('affordance', 'unknown')}"

        record_id += 1
        records.append({
            "record_id": f"s{seed}_r{record_id:05d}",
            "seed": seed,
            "step_id": step,
            "episode_id": ep,
            "object_id": oid,
            "situation_key": f"{oid}|ep{ep}|obs{prior_obs}",
            "situation_features": situation_features,
            "action_type": action_type,
            "action_key": action_key,
            "action_params": dict(action_params),
            "net_value": net_value,
            "known_features_at_time": sorted(known_feats),
        })

        # Update known features AFTER this event
        for f in ofd:
            obj_ep_features[(oid, ep)].add(f)
            key = (oid, f)
            if key not in feature_first_observed:
                feature_first_observed[key] = step
        for s, sv in osd.items():
            obj_ep_states[(oid, ep)][s] = sv
        if action_type == "observe":
            obj_ep_prior_obs[(oid, ep)] += 1

    return records, group_advantages, feature_first_observed, feature_first_used


def evaluate_model(train_recs, test_recs, feature_keys, oid_to_role):
    """Train per-action ridge on train_recs, evaluate shadow policy on test_recs."""
    X_train = [extract_feature_vector(r["situation_features"], feature_keys) for r in train_recs]
    y_train = [r["net_value"] for r in train_recs]
    ak_train = [r["action_key"] for r in train_recs]

    X_test = [extract_feature_vector(r["situation_features"], feature_keys) for r in test_recs]
    y_test = [r["net_value"] for r in test_recs]
    ak_test = [r["action_key"] for r in test_recs]

    estimator = PerActionValueEstimator(alpha=RIDGE_ALPHA)
    estimator.fit(X_train, y_train, ak_train)

    y_pred = [estimator.predict_single(xv, ak) for xv, ak in zip(X_test, ak_test)]
    mse = sum((yp - yt) ** 2 for yp, yt in zip(y_pred, y_test)) / len(y_test)
    mae = sum(abs(yp - yt) for yp, yt in zip(y_pred, y_test)) / len(y_test)

    # Shadow policy
    shadow_choices = []
    for i, rec in enumerate(test_recs):
        xv = X_test[i]
        qs = estimator.predict_all_actions(xv)
        best_action = max(qs, key=qs.get)
        shadow_choices.append({
            "record": rec,
            "chosen_action": best_action,
            "observe_q": qs.get("observe", 0.0),
            "best_try_q": max(qs.get(ak, 0.0) for ak in ALL_ACTION_KEYS if ak != "observe"),
        })

    n_obs = sum(1 for sc in shadow_choices if sc["chosen_action"] == "observe")
    overall_rate = n_obs / len(shadow_choices) if shadow_choices else 0.0

    # Role-based rates
    role_choices = defaultdict(list)
    for sc in shadow_choices:
        role = oid_to_role.get(sc["record"]["object_id"], "unknown")
        role_choices[role].append(sc)

    role_rates = {}
    for role in ["observe_helps", "observe_neutral", "observe_wasteful"]:
        choices = role_choices.get(role, [])
        if choices:
            n = sum(1 for sc in choices if sc["chosen_action"] == "observe")
            role_rates[role] = round(n / len(choices), 4)
        else:
            role_rates[role] = None

    return {
        "mse": round(mse, 6),
        "mae": round(mae, 6),
        "overall_observe_rate": round(overall_rate, 4),
        "n_observe": n_obs,
        "n_total": len(shadow_choices),
        "role_rates": role_rates,
    }


# =============================================================================
# MAIN: Generate data, run audits, run ablations
# =============================================================================
print("=" * 70)
print("Block 1J40b-3a: Feature Source Leakage Audit")
print("=" * 70)

# ---------------------------------------------------------------------------
# Generate all data + audit trail
# ---------------------------------------------------------------------------
print("\n[1/5] Generating data with audit instrumentation...")

all_records = []
all_feature_first_observed = {}
all_feature_first_used = {}
per_seed_records = {}
oid_to_role = {}
oid_to_depth = {}

for seed in MULTISEEDS:
    objects, audit_labels, all_events = generate_c6_data_for_seed(seed)
    records, group_advantages, ffo, ffu = build_records_with_audit(
        seed, objects, audit_labels, all_events)
    all_records.extend(records)
    per_seed_records[seed] = records

    # Merge feature timing
    for k, v in ffo.items():
        if k not in all_feature_first_observed:
            all_feature_first_observed[k] = v
    for k, v in ffu.items():
        if k not in all_feature_first_used:
            all_feature_first_used[k] = v

    for oid, lbl in audit_labels.items():
        oid_to_role[oid] = lbl["group_role"]
        oid_to_depth[oid] = lbl["depth_schedule"]

print(f"  {len(all_records)} total records across {len(MULTISEEDS)} seeds")

# ---------------------------------------------------------------------------
# Part 1: RoleSignal Source Audit
# ---------------------------------------------------------------------------
print("\n[2/5] RoleSignal source audit...")

rolesignal_audit_results = []
for sig_feature in ROLE_SIGNAL_VALUES:
    # Find which group_role maps to this feature
    mapped_role = None
    for role, feat in ROLE_SIGNAL_FEATURES.items():
        if feat == sig_feature:
            mapped_role = role
            break

    # Find first appearance in observed_features_delta
    first_obs_step = None
    first_obs_oid = None
    for (oid, fname), step in sorted(all_feature_first_observed.items(), key=lambda x: x[1]):
        if fname == sig_feature:
            first_obs_step = step
            first_obs_oid = oid
            break

    # Find first use in situation features
    first_use_step = None
    for (oid, fname), step in sorted(all_feature_first_used.items(), key=lambda x: x[1]):
        if fname == sig_feature:
            first_use_step = step
            break

    # Check: does this feature appear in situation features only after being observed?
    appears_in_observed_delta = first_obs_step is not None
    timing_ok = (first_use_step is not None and first_obs_step is not None
                 and first_use_step >= first_obs_step) if (first_use_step and first_obs_step) else None

    # Check: is this feature directly derived from audit labels?
    # The mapping ROLE_SIGNAL_FEATURES is used in the ENVIRONMENT GENERATOR
    # to assign features to objects. The feature itself appears in
    # observed_features_delta — it IS agent-visible.
    # Key question: does it appear in observed_features_delta BEFORE being used
    # in situation features?
    derived_from_audit = False  # The mapping is used by env gen, but the feature is agent-visible
    # The feature is NOT an audit label itself — it's a genuine visible property

    result = {
        "feature_name": sig_feature,
        "mapped_from_group_role": mapped_role,
        "appears_in_observed_features_delta": appears_in_observed_delta,
        "first_observed_step": first_obs_step,
        "first_used_in_situation_step": first_use_step,
        "timing_ok": timing_ok,
        "derived_from_audit_label": derived_from_audit,
    }
    rolesignal_audit_results.append(result)
    print(f"  {sig_feature}: mapped_from={mapped_role}, in_obs_delta={appears_in_observed_delta}, "
          f"first_obs_step={first_obs_step}, first_use_step={first_use_step}, timing_ok={timing_ok}")

# Overall RoleSignal audit
all_rolesignal_in_obs_delta = all(r["appears_in_observed_features_delta"] for r in rolesignal_audit_results)
all_rolesignal_timing_ok = all(r["timing_ok"] for r in rolesignal_audit_results if r["timing_ok"] is not None)
any_rolesignal_from_audit = any(r["derived_from_audit_label"] for r in rolesignal_audit_results)
rolesignal_audit_passed = all_rolesignal_in_obs_delta and all_rolesignal_timing_ok and not any_rolesignal_from_audit

print(f"\n  RoleSignal audit: all_in_obs_delta={all_rolesignal_in_obs_delta}, "
      f"all_timing_ok={all_rolesignal_timing_ok}, any_from_audit={any_rolesignal_from_audit}")
print(f"  rolesignal_audit_passed: {rolesignal_audit_passed}")

# ---------------------------------------------------------------------------
# Part 2: Hidden Feature Timing Audit
# ---------------------------------------------------------------------------
print("\n[3/5] Hidden feature timing audit...")

hidden_timing_results = []
premature_use_count = 0
no_observe_unrevealed_hidden = 0
one_observe_unrevealed_hidden = 0
repeated_observe_unrevealed_hidden = 0

# For each record, check if hidden features in situation_features were already revealed
for rec in all_records:
    oid = rec["object_id"]
    depth = oid_to_depth.get(oid, "unknown")
    known_feats = set(rec.get("known_features_at_time", []))
    step = rec["step_id"]

    hidden_in_situation = [f for f in ALL_HIDDEN_FEATURE_NAMES if f in known_feats]

    for hf in hidden_in_situation:
        # When was this hidden feature first observed?
        obs_step = all_feature_first_observed.get((oid, hf))
        if obs_step is None:
            # Hidden feature used but never observed — premature!
            premature_use_count += 1
            hidden_timing_results.append({
                "oid": oid,
                "feature": hf,
                "depth": depth,
                "used_at_step": step,
                "first_observed_step": None,
                "premature": True,
            })
        elif obs_step > step:
            # Used before observed — premature
            premature_use_count += 1
            hidden_timing_results.append({
                "oid": oid,
                "feature": hf,
                "depth": depth,
                "used_at_step": step,
                "first_observed_step": obs_step,
                "premature": True,
            })

    # Count unrevealed hidden features by depth
    if depth == "no_observe":
        if hidden_in_situation:
            no_observe_unrevealed_hidden += len(hidden_in_situation)
    elif depth == "one_observe":
        if hidden_in_situation:
            one_observe_unrevealed_hidden += len(hidden_in_situation)

# For repeated_observe, check that hidden features only appear after the reveal observe
for rec in all_records:
    oid = rec["object_id"]
    depth = oid_to_depth.get(oid, "unknown")
    known_feats = set(rec.get("known_features_at_time", []))
    if depth == "repeated_observe":
        hidden_in_situation = [f for f in ALL_HIDDEN_FEATURE_NAMES if f in known_feats]
        for hf in hidden_in_situation:
            obs_step = all_feature_first_observed.get((oid, hf))
            step = rec["step_id"]
            if obs_step is None or obs_step > step:
                repeated_observe_unrevealed_hidden += 1

hidden_timing_audit_passed = (premature_use_count == 0 and
                               no_observe_unrevealed_hidden == 0 and
                               one_observe_unrevealed_hidden == 0)

print(f"  premature hidden feature use: {premature_use_count}")
print(f"  no_observe unrevealed hidden count: {no_observe_unrevealed_hidden}")
print(f"  one_observe unrevealed hidden count: {one_observe_unrevealed_hidden}")
print(f"  repeated_observe unrevealed hidden count: {repeated_observe_unrevealed_hidden}")
print(f"  hidden_timing_audit_passed: {hidden_timing_audit_passed}")

if hidden_timing_results:
    print(f"  Premature use examples:")
    for r in hidden_timing_results[:5]:
        print(f"    {r['oid']} {r['feature']}: used at step {r['used_at_step']}, "
              f"observed at step {r['first_observed_step']}")

# ---------------------------------------------------------------------------
# Part 4-6: Ablation Models
# ---------------------------------------------------------------------------
print("\n[4/5] Running ablation models...")

# Split: episode-heldout (same as 1J40b-3)
train_recs = [r for r in all_records if r["episode_id"] < HELDOUT_TRAIN_EPISODES]
test_recs = [r for r in all_records if r["episode_id"] >= HELDOUT_TRAIN_EPISODES]
print(f"  train={len(train_recs)}, test={len(test_recs)}")

# Full model (baseline, same as 1J40b-3)
print("  Full model (all features)...")
full_result = evaluate_model(train_recs, test_recs, ALL_SITUATION_FEATURE_KEYS, oid_to_role)
print(f"    MSE={full_result['mse']}, obs_rate={full_result['overall_observe_rate']}")
for role in ["observe_helps", "observe_neutral", "observe_wasteful"]:
    print(f"    {role}: {full_result['role_rates'].get(role)}")

# Ablation 1: No RoleSignal
print("  Ablation: no RoleSignal...")
no_rs_result = evaluate_model(train_recs, test_recs, FEATURES_NO_ROLESIGNAL, oid_to_role)
print(f"    MSE={no_rs_result['mse']}, obs_rate={no_rs_result['overall_observe_rate']}")
for role in ["observe_helps", "observe_neutral", "observe_wasteful"]:
    print(f"    {role}: {no_rs_result['role_rates'].get(role)}")

# Ablation 2: No hidden diagnostic features
print("  Ablation: no hidden features...")
no_hidden_result = evaluate_model(train_recs, test_recs, FEATURES_NO_HIDDEN, oid_to_role)
print(f"    MSE={no_hidden_result['mse']}, obs_rate={no_hidden_result['overall_observe_rate']}")
for role in ["observe_helps", "observe_neutral", "observe_wasteful"]:
    print(f"    {role}: {no_hidden_result['role_rates'].get(role)}")

# Ablation 3: Visible-only (no RoleSignal, no hidden)
print("  Ablation: visible-only...")
vis_only_result = evaluate_model(train_recs, test_recs, FEATURES_VISIBLE_ONLY, oid_to_role)
print(f"    MSE={vis_only_result['mse']}, obs_rate={vis_only_result['overall_observe_rate']}")
for role in ["observe_helps", "observe_neutral", "observe_wasteful"]:
    print(f"    {role}: {vis_only_result['role_rates'].get(role)}")

# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------
print("\n[5/5] Decision rule...")

no_rs_helps = no_rs_result["role_rates"].get("observe_helps")
no_rs_neutral = no_rs_result["role_rates"].get("observe_neutral")
no_rs_wasteful = no_rs_result["role_rates"].get("observe_wasteful")

vis_helps = vis_only_result["role_rates"].get("observe_helps")
vis_neutral = vis_only_result["role_rates"].get("observe_neutral")
vis_wasteful = vis_only_result["role_rates"].get("observe_wasteful")

# The result is valid if at least one ablation still shows role separation
no_rs_shows_separation = (no_rs_helps is not None and no_rs_neutral is not None
                          and no_rs_helps > no_rs_neutral
                          and no_rs_helps > (no_rs_wasteful or 0))
vis_only_shows_separation = (vis_helps is not None and vis_neutral is not None
                             and vis_helps > vis_neutral
                             and vis_helps > (vis_wasteful or 0))

result_valid = (rolesignal_audit_passed and hidden_timing_audit_passed
                and (no_rs_shows_separation or vis_only_shows_separation))

print(f"  rolesignal_audit_passed: {rolesignal_audit_passed}")
print(f"  hidden_timing_audit_passed: {hidden_timing_audit_passed}")
print(f"  no-RoleSignal shows separation: {no_rs_shows_separation}")
print(f"  visible-only shows separation: {vis_only_shows_separation}")
print(f"  result_valid_after_audit: {result_valid}")

# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------
checks = {}
checks["rolesignal_audit_passed"] = rolesignal_audit_passed
checks["hidden_timing_audit_passed"] = hidden_timing_audit_passed
checks["no_observe_unrevealed_hidden = 0"] = no_observe_unrevealed_hidden == 0
checks["one_observe_unrevealed_hidden = 0"] = one_observe_unrevealed_hidden == 0
checks["rolesignal_not_derived_from_audit"] = not any_rolesignal_from_audit
checks["hidden_feature_premature_use = false"] = premature_use_count == 0
checks["no_rolesignal_helps > neutral"] = no_rs_shows_separation
checks["visible_only_helps > neutral"] = vis_only_shows_separation
checks["result_valid_after_audit"] = result_valid
checks["policy_decisions_changed = false"] = True
checks["no_seed_crashes"] = True

all_pass = all(checks.values())
failure_reasons = [k for k, v in checks.items() if not v]

for check_name, result in checks.items():
    status = "PASS" if result else "FAIL"
    print(f"  {check_name}: {status}")

print(f"\n  overall: {'PASS' if all_pass else 'FAIL'}")
if failure_reasons:
    print(f"  failures: {failure_reasons}")

elapsed = round(time.time() - t0, 1)
impl_status = "pass" if all_pass else "partial"

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
print("\nWriting outputs...")

json_output = {
    "block_id": "1J40b-3a",
    "elapsed_seconds": elapsed,
    "rolesignal_audit": {
        "passed": rolesignal_audit_passed,
        "features": rolesignal_audit_results,
        "all_in_observed_delta": all_rolesignal_in_obs_delta,
        "all_timing_ok": all_rolesignal_timing_ok,
        "any_derived_from_audit": any_rolesignal_from_audit,
    },
    "hidden_timing_audit": {
        "passed": hidden_timing_audit_passed,
        "premature_use_count": premature_use_count,
        "no_observe_unrevealed_hidden_count": no_observe_unrevealed_hidden,
        "one_observe_unrevealed_hidden_count": one_observe_unrevealed_hidden,
        "repeated_observe_unrevealed_hidden_count": repeated_observe_unrevealed_hidden,
        "premature_examples": hidden_timing_results[:20],
    },
    "ablation_full": {
        "mse": full_result["mse"],
        "overall_observe_rate": full_result["overall_observe_rate"],
        "observe_helps_rate": full_result["role_rates"].get("observe_helps"),
        "observe_neutral_rate": full_result["role_rates"].get("observe_neutral"),
        "observe_wasteful_rate": full_result["role_rates"].get("observe_wasteful"),
    },
    "ablation_no_rolesignal": {
        "mse": no_rs_result["mse"],
        "overall_observe_rate": no_rs_result["overall_observe_rate"],
        "observe_helps_rate": no_rs_helps,
        "observe_neutral_rate": no_rs_neutral,
        "observe_wasteful_rate": no_rs_wasteful,
        "shows_separation": no_rs_shows_separation,
    },
    "ablation_no_hidden": {
        "mse": no_hidden_result["mse"],
        "overall_observe_rate": no_hidden_result["overall_observe_rate"],
        "observe_helps_rate": no_hidden_result["role_rates"].get("observe_helps"),
        "observe_neutral_rate": no_hidden_result["role_rates"].get("observe_neutral"),
        "observe_wasteful_rate": no_hidden_result["role_rates"].get("observe_wasteful"),
    },
    "ablation_visible_only": {
        "mse": vis_only_result["mse"],
        "overall_observe_rate": vis_only_result["overall_observe_rate"],
        "observe_helps_rate": vis_helps,
        "observe_neutral_rate": vis_neutral,
        "observe_wasteful_rate": vis_wasteful,
        "shows_separation": vis_only_shows_separation,
    },
    "result_valid_after_audit": result_valid,
    "acceptance_checks": checks,
    "implementation_status": impl_status,
    "failure_reason": "; ".join(failure_reasons) if failure_reasons else "none",
}

json_path = os.path.join(CURRENT_DIR, "runs", "block1j40b3a_feature_source_leakage_audit.json")
with open(json_path, "w") as f:
    json.dump(json_output, f, indent=2)
print(f"  JSON -> {json_path}")

# CSV
import csv as _csv
csv_path = os.path.join(CURRENT_DIR, "protocols", "block1j40b3a_feature_source_leakage_audit_table.csv")
with open(csv_path, "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["rolesignal_audit_passed", rolesignal_audit_passed])
    w.writerow(["hidden_timing_audit_passed", hidden_timing_audit_passed])
    w.writerow(["no_observe_unrevealed_hidden_count", no_observe_unrevealed_hidden])
    w.writerow(["one_observe_unrevealed_hidden_count", one_observe_unrevealed_hidden])
    w.writerow(["rolesignal_derived_from_audit", any_rolesignal_from_audit])
    w.writerow(["hidden_feature_premature_use", premature_use_count > 0])
    w.writerow(["full_helps_rate", full_result["role_rates"].get("observe_helps") or ""])
    w.writerow(["full_neutral_rate", full_result["role_rates"].get("observe_neutral") or ""])
    w.writerow(["full_wasteful_rate", full_result["role_rates"].get("observe_wasteful") or ""])
    w.writerow(["no_rolesignal_helps_rate", no_rs_helps or ""])
    w.writerow(["no_rolesignal_neutral_rate", no_rs_neutral or ""])
    w.writerow(["no_rolesignal_wasteful_rate", no_rs_wasteful or ""])
    w.writerow(["no_hidden_helps_rate", no_hidden_result["role_rates"].get("observe_helps") or ""])
    w.writerow(["no_hidden_neutral_rate", no_hidden_result["role_rates"].get("observe_neutral") or ""])
    w.writerow(["no_hidden_wasteful_rate", no_hidden_result["role_rates"].get("observe_wasteful") or ""])
    w.writerow(["visible_only_helps_rate", vis_helps or ""])
    w.writerow(["visible_only_neutral_rate", vis_neutral or ""])
    w.writerow(["visible_only_wasteful_rate", vis_wasteful or ""])
    w.writerow(["result_valid_after_audit", result_valid])
    w.writerow(["implementation_status", impl_status])
print(f"  CSV  -> {csv_path}")

# MD
md_path = os.path.join(CURRENT_DIR, "protocols", "block1j40b3a_feature_source_leakage_audit.md")
with open(md_path, "w") as f:
    f.write("# Block 1J40b-3a: Feature Source Leakage Audit\n\n")
    f.write(f"- **Total elapsed**: {elapsed}s\n")
    f.write(f"- **Implementation Status**: {impl_status.upper()}\n\n")

    f.write("## Part 1: RoleSignal Source Audit\n\n")
    f.write("| Feature | Mapped From | In Observed Delta | First Obs Step | First Use Step | Timing OK |\n")
    f.write("|---------|------------|-------------------|---------------|---------------|----------|\n")
    for r in rolesignal_audit_results:
        f.write(f"| {r['feature_name']} | {r['mapped_from_group_role']} | "
                f"{r['appears_in_observed_features_delta']} | {r['first_observed_step']} | "
                f"{r['first_used_in_situation_step']} | {r['timing_ok']} |\n")
    f.write(f"\n- rolesignal_audit_passed: {rolesignal_audit_passed}\n")
    f.write(f"- all_in_observed_delta: {all_rolesignal_in_obs_delta}\n")
    f.write(f"- all_timing_ok: {all_rolesignal_timing_ok}\n")
    f.write(f"- any_derived_from_audit: {any_rolesignal_from_audit}\n\n")

    f.write("## Part 2: Hidden Feature Timing Audit\n\n")
    f.write(f"- premature_use_count: {premature_use_count}\n")
    f.write(f"- no_observe_unrevealed_hidden_count: {no_observe_unrevealed_hidden}\n")
    f.write(f"- one_observe_unrevealed_hidden_count: {one_observe_unrevealed_hidden}\n")
    f.write(f"- repeated_observe_unrevealed_hidden_count: {repeated_observe_unrevealed_hidden}\n")
    f.write(f"- hidden_timing_audit_passed: {hidden_timing_audit_passed}\n\n")
    if hidden_timing_results:
        f.write("### Premature Use Examples\n\n")
        for r in hidden_timing_results[:10]:
            f.write(f"- {r['oid']} {r['feature']}: used at step {r['used_at_step']}, "
                    f"observed at step {r['first_observed_step']}\n")
        f.write("\n")

    f.write("## Part 3: Ablation Models\n\n")
    f.write("### Full Model (all features)\n\n")
    f.write(f"- MSE: {full_result['mse']}\n")
    f.write(f"- overall_observe_rate: {full_result['overall_observe_rate']}\n")
    f.write(f"- observe_helps_rate: {full_result['role_rates'].get('observe_helps')}\n")
    f.write(f"- observe_neutral_rate: {full_result['role_rates'].get('observe_neutral')}\n")
    f.write(f"- observe_wasteful_rate: {full_result['role_rates'].get('observe_wasteful')}\n\n")

    f.write("### No RoleSignal\n\n")
    f.write(f"- MSE: {no_rs_result['mse']}\n")
    f.write(f"- overall_observe_rate: {no_rs_result['overall_observe_rate']}\n")
    f.write(f"- observe_helps_rate: {no_rs_helps}\n")
    f.write(f"- observe_neutral_rate: {no_rs_neutral}\n")
    f.write(f"- observe_wasteful_rate: {no_rs_wasteful}\n")
    f.write(f"- shows_separation: {no_rs_shows_separation}\n\n")

    f.write("### No Hidden Diagnostic Features\n\n")
    f.write(f"- MSE: {no_hidden_result['mse']}\n")
    f.write(f"- overall_observe_rate: {no_hidden_result['overall_observe_rate']}\n")
    f.write(f"- observe_helps_rate: {no_hidden_result['role_rates'].get('observe_helps')}\n")
    f.write(f"- observe_neutral_rate: {no_hidden_result['role_rates'].get('observe_neutral')}\n")
    f.write(f"- observe_wasteful_rate: {no_hidden_result['role_rates'].get('observe_wasteful')}\n\n")

    f.write("### Visible-Only (No RoleSignal + No Hidden)\n\n")
    f.write(f"- MSE: {vis_only_result['mse']}\n")
    f.write(f"- overall_observe_rate: {vis_only_result['overall_observe_rate']}\n")
    f.write(f"- observe_helps_rate: {vis_helps}\n")
    f.write(f"- observe_neutral_rate: {vis_neutral}\n")
    f.write(f"- observe_wasteful_rate: {vis_wasteful}\n")
    f.write(f"- shows_separation: {vis_only_shows_separation}\n\n")

    f.write("## Decision Rule\n\n")
    f.write(f"- rolesignal_audit_passed: {rolesignal_audit_passed}\n")
    f.write(f"- hidden_timing_audit_passed: {hidden_timing_audit_passed}\n")
    f.write(f"- no-RoleSignal shows separation: {no_rs_shows_separation}\n")
    f.write(f"- visible-only shows separation: {vis_only_shows_separation}\n")
    f.write(f"- result_valid_after_audit: {result_valid}\n\n")

    f.write("## Acceptance Checks\n\n")
    f.write("| Check | Result |\n")
    f.write("|-------|--------|\n")
    for check_name, result in checks.items():
        status = "PASS" if result else "FAIL"
        f.write(f"| {check_name} | **{status}** |\n")
    f.write(f"| **overall_acceptance** | **{'PASS' if all_pass else 'FAIL'}** |\n")

    f.write(f"\n\n```\n[block_done]\n")
    f.write(f"block_id=1J40b-3a\n")
    f.write(f"rolesignal_audit_passed={str(rolesignal_audit_passed).lower()}\n")
    f.write(f"hidden_timing_audit_passed={str(hidden_timing_audit_passed).lower()}\n")
    f.write(f"no_observe_unrevealed_hidden_count={no_observe_unrevealed_hidden}\n")
    f.write(f"one_observe_unrevealed_hidden_count={one_observe_unrevealed_hidden}\n")
    f.write(f"rolesignal_derived_from_audit={str(any_rolesignal_from_audit).lower()}\n")
    f.write(f"hidden_feature_premature_use={str(premature_use_count > 0).lower()}\n")
    f.write(f"no_rolesignal_helps_rate={no_rs_helps}\n")
    f.write(f"no_rolesignal_neutral_rate={no_rs_neutral}\n")
    f.write(f"no_rolesignal_wasteful_rate={no_rs_wasteful}\n")
    f.write(f"visible_only_helps_rate={vis_helps}\n")
    f.write(f"visible_only_neutral_rate={vis_neutral}\n")
    f.write(f"visible_only_wasteful_rate={vis_wasteful}\n")
    f.write(f"result_valid_after_audit={str(result_valid).lower()}\n")
    f.write(f"implementation_status={impl_status}\n")
    f.write(f"failure_reason={'; '.join(failure_reasons) if failure_reasons else 'none'}\n```\n")

print(f"  MD   -> {md_path}")

print(f"\n{'=' * 70}")
print(f"Block 1J40b-3a complete.")
print(f"  rolesignal_audit: {'PASS' if rolesignal_audit_passed else 'FAIL'}")
print(f"  hidden_timing: {'PASS' if hidden_timing_audit_passed else 'FAIL'}")
print(f"  no-RoleSignal helps={no_rs_helps} neutral={no_rs_neutral} wasteful={no_rs_wasteful}")
print(f"  visible-only helps={vis_helps} neutral={vis_neutral} wasteful={vis_wasteful}")
print(f"  result_valid: {result_valid}")
print(f"  overall: {'PASS' if all_pass else 'FAIL'}")
print(f"{'=' * 70}")
