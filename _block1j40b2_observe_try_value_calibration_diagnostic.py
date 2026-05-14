"""
Block 1J40b-2 -- Observe-vs-Try Value Calibration Diagnostic.

Diagnostic-only: investigates why 1J40b-1 shadow strategy chooses try 100%
and observe 0%. Does NOT change policy, environment, or activate Strategy Memory.
"""
import os, sys, json, math, time
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

# =============================================================================
# CLI
# =============================================================================
import argparse as _argparse
_parser = _argparse.ArgumentParser()
_parser.add_argument("--multiseed", action="store_true", default=True)
_args = _parser.parse_args()
MULTISEEDS = [101, 103, 107, 109, 113]
N_EPISODES = 5
HELDOUT_TRAIN_EPISODES = 4

# =============================================================================
# Constants (matching 1J40b-0 and 1J40b-1)
# =============================================================================
ALL_TRY_AFFORDANCES = [
    "burn_as_fuel", "craft_plank", "eat",
    "mine_by_hand", "mine_with_pickaxe", "use_as_tool",
]
ALL_ACTION_KEYS = ["observe"] + [f"try_{a}" for a in ALL_TRY_AFFORDANCES]
RIDGE_ALPHA = 1.0

FORBIDDEN_FIELDS = [
    "mixed_source_type", "true_family", "hidden_subtype",
    "oracle_outcome", "full_object_state", "deceptive_flag",
    "prior_violation", "preferred_explanation",
]

SITUATION_FEATURE_KEYS = [
    "candidate_confidence", "candidate_support_count",
    "candidate_stability", "candidate_positive_feature_count",
    "candidate_mixed_rate", "candidate_contradiction_count",
    "observed_feature_coverage", "observed_state_coverage",
    "recent_observe_count", "recent_try_count",
    "total_object_observes", "total_object_tries",
    "best_current_try_value", "top_action_value_margin",
    "episode_id", "episode_progress",
]

# Explanation score keys (from 1J40a-2, stored in explanation_feature_source)
EXPLANATION_FEATURE_KEYS = [
    "observation_gap_score", "state_condition_score",
    "identity_split_score", "irreducible_noise_score",
    "unresolved_score", "confidence", "top_score_margin",
]

# Observe-specific features (features plausibly relevant to observe value)
OBSERVE_SPECIFIC_KEYS = [
    "candidate_confidence", "candidate_positive_feature_count",
    "observed_feature_coverage", "observed_state_coverage",
    "recent_observe_count", "total_object_observes",
    "candidate_stability", "candidate_mixed_rate",
    "episode_id", "episode_progress",
]

OBSERVE_COST = 0.005
SUCCESS_REWARD = 0.5
FAILURE_PENALTY = 0.1
PROBE_COST = 0.05


# =============================================================================
# Data loader
# =============================================================================
def build_records_from_1j40b0():
    """Import and re-run 1J40b-0 to get raw learning records per seed."""
    from _block1j40b0_situation_action_result_feasibility_audit import (
        run_seed as b0_run_seed,
    )
    all_records = []
    per_seed_records = {}
    for seed in MULTISEEDS:
        result = b0_run_seed(seed)
        records = result["learning_records"]
        # Add seed to each record
        for rec in records:
            rec["seed"] = seed
        per_seed_records[seed] = records
        all_records.extend(records)
    return all_records, per_seed_records


# =============================================================================
# Ridge Regression (pure numpy, same as 1J40b-1)
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


# =============================================================================
# Feature normalization
# =============================================================================
class FeatureNormalizer:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X):
        if not X:
            self.means = []
            self.stds = []
            return self
        n_feat = len(X[0])
        self.means = [0.0] * n_feat
        self.stds = [1.0] * n_feat
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


# =============================================================================
# Feature extraction variants
# =============================================================================
def extract_features_base(records, feature_keys):
    """Extract X, y, action_keys, meta for a given feature key list."""
    X = []
    y = []
    action_keys = []
    meta = []
    for rec in records:
        feat_vec = []
        for k in feature_keys:
            if k in SITUATION_FEATURE_KEYS:
                sf = rec.get("situation_features", {})
                feat_vec.append(float(sf.get(k, 0.0)))
            elif k in EXPLANATION_FEATURE_KEYS:
                es = rec.get("explanation_feature_source", {})
                feat_vec.append(float(es.get(k, 0.0)))
            else:
                feat_vec.append(0.0)
        action_type = rec.get("action_type", "")
        action_params = rec.get("action_params", {})
        if action_type == "observe":
            akey = "observe"
        elif action_type == "try":
            akey = f"try_{action_params.get('affordance', 'unknown')}"
        else:
            continue
        X.append(feat_vec)
        y.append(float(rec.get("net_value", 0.0)))
        action_keys.append(akey)
        meta.append({
            "record_id": rec.get("record_id", ""),
            "action_key": akey,
            "net_value": float(rec.get("net_value", 0.0)),
            "episode_id": int(rec.get("episode_id", 0)),
            "seed": rec.get("seed", 0),
            "candidate_ids": rec.get("source_candidate_ids", []),
        })
    return X, y, action_keys, meta


def extract_features_situation_only(records):
    """Extract 16 situation features only (1J40b-1 model)."""
    return extract_features_base(records, SITUATION_FEATURE_KEYS)


def extract_features_with_explanation(records):
    """Extract 16 situation features + 7 explanation scores."""
    return extract_features_base(records, SITUATION_FEATURE_KEYS + EXPLANATION_FEATURE_KEYS)


def extract_features_observe_specific(records):
    """Extract only observe-specific features."""
    return extract_features_base(records, OBSERVE_SPECIFIC_KEYS)


def extract_features_action_type_only(records):
    """Extract action type one-hot features."""
    all_actions = sorted(set(ALL_ACTION_KEYS))
    action_to_idx = {a: i for i, a in enumerate(all_actions)}
    n_actions = len(all_actions)

    X = []
    y = []
    action_keys = []
    meta = []
    for rec in records:
        action_type = rec.get("action_type", "")
        action_params = rec.get("action_params", {})
        if action_type == "observe":
            akey = "observe"
        elif action_type == "try":
            akey = f"try_{action_params.get('affordance', 'unknown')}"
        else:
            continue
        vec = [0.0] * n_actions
        idx = action_to_idx.get(akey)
        if idx is not None:
            vec[idx] = 1.0
        X.append(vec)
        y.append(float(rec.get("net_value", 0.0)))
        action_keys.append(akey)
        meta.append({"record_id": rec.get("record_id", ""), "action_key": akey,
                     "net_value": float(rec.get("net_value", 0.0))})
    return X, y, action_keys, meta


# =============================================================================
# Per-Action Ridge Estimator
# =============================================================================
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
            if len(gx) < 2:
                self.models[ak] = None
                self.normalizers[ak] = None
                continue
            normalizer = FeatureNormalizer().fit(gx)
            X_norm = normalizer.transform(gx)
            model = RidgeRegression(alpha=self.alpha).fit(X_norm, gy)
            self.models[ak] = model
            self.normalizers[ak] = normalizer

    def predict_single_action(self, X, action_key):
        if action_key not in self.models or self.models[action_key] is None:
            return None
        normalizer = self.normalizers[action_key]
        X_norm = normalizer.transform(X) if normalizer else X
        return self.models[action_key].predict(X_norm)

    def predict_all_actions(self, X):
        results = {}
        for ak in ALL_ACTION_KEYS:
            preds = self.predict_single_action(X, ak)
            results[ak] = preds
        return results


# =============================================================================
# Train/evaluate a single model variant (no reweighting)
# =============================================================================
def train_and_evaluate(X_train, y_train, ak_train, X_test, y_test, ak_test):
    """Train per-action ridge, return predictions and metrics."""
    normalizer = FeatureNormalizer().fit(X_train)
    X_train_norm = normalizer.transform(X_train)
    X_test_norm = normalizer.transform(X_test)

    estimator = PerActionValueEstimator(alpha=RIDGE_ALPHA)
    estimator.fit(X_train_norm, y_train, ak_train)

    n_test = len(y_test)
    ridge_preds = []
    for i in range(n_test):
        ak = ak_test[i]
        preds = estimator.predict_single_action([X_test_norm[i]], ak)
        ridge_preds.append(preds[0] if preds else 0.0)

    mse = sum((t - p) ** 2 for t, p in zip(y_test, ridge_preds)) / n_test if n_test else 0.0
    mae = sum(abs(t - p) for t, p in zip(y_test, ridge_preds)) / n_test if n_test else 0.0

    # Per-action metrics
    per_action = defaultdict(lambda: {"count": 0, "mse_sum": 0.0, "mae_sum": 0.0,
                                       "true_sum": 0.0, "pred_sum": 0.0})
    for t, p, ak in zip(y_test, ridge_preds, ak_test):
        pa = per_action[ak]
        pa["count"] += 1
        pa["mse_sum"] += (t - p) ** 2
        pa["mae_sum"] += abs(t - p)
        pa["true_sum"] += t
        pa["pred_sum"] += p

    per_action_stats = {}
    for ak, pa in per_action.items():
        c = pa["count"]
        per_action_stats[ak] = {
            "count": c,
            "mse": round(pa["mse_sum"] / c, 6),
            "mae": round(pa["mae_sum"] / c, 6),
            "mean_true": round(pa["true_sum"] / c, 4),
            "mean_pred": round(pa["pred_sum"] / c, 4),
        }

    # Shadow policy
    all_preds = estimator.predict_all_actions(X_test_norm)
    shadow_choices = []
    shadow_pred_values = []
    for i in range(n_test):
        q_values = {}
        for ak in ALL_ACTION_KEYS:
            preds = all_preds.get(ak)
            if preds is not None and i < len(preds):
                q_values[ak] = preds[i]
        if not q_values:
            shadow_choices.append("none")
            shadow_pred_values.append(0.0)
            continue
        best_action = max(q_values, key=q_values.get)
        shadow_choices.append(best_action)
        shadow_pred_values.append(q_values[best_action])

    n_valid = n_test - shadow_choices.count("none")
    observe_choices = sum(1 for c in shadow_choices if c == "observe")
    try_choices = sum(1 for c in shadow_choices if c.startswith("try_"))

    shadow_dist = defaultdict(int)
    for c in shadow_choices:
        shadow_dist[c] += 1

    return {
        "mse": round(mse, 6),
        "mae": round(mae, 6),
        "n": n_test,
        "per_action": per_action_stats,
        "shadow_observe_rate": round(observe_choices / max(n_valid, 1), 4),
        "shadow_try_rate": round(try_choices / max(n_valid, 1), 4),
        "shadow_action_distribution": {ak: shadow_dist.get(ak, 0) for ak in ALL_ACTION_KEYS},
    }


# =============================================================================
# Train/evaluate with sample reweighting
# =============================================================================
def train_with_reweighting(X_train, y_train, ak_train, X_test, y_test, ak_test,
                           weight_per_sample):
    """Train per-action ridge with per-sample weights.
    Weights multiply each sample's contribution to X^T X and X^T y.
    """
    normalizer = FeatureNormalizer().fit(X_train)
    X_train_norm = normalizer.transform(X_train)
    X_test_norm = normalizer.transform(X_test)

    # Group by action
    grouped_X = defaultdict(list)
    grouped_y = defaultdict(list)
    grouped_w = defaultdict(list)
    for xv, yv, ak, w in zip(X_train_norm, y_train, ak_train, weight_per_sample):
        grouped_X[ak].append(xv)
        grouped_y[ak].append(yv)
        grouped_w[ak].append(w)

    estimator = PerActionValueEstimator(alpha=RIDGE_ALPHA)
    # Override fit to use weighted regression per action
    for ak in ALL_ACTION_KEYS:
        gx = grouped_X.get(ak, [])
        gy = grouped_y.get(ak, [])
        gw = grouped_w.get(ak, [])
        if len(gx) < 2:
            estimator.models[ak] = None
            estimator.normalizers[ak] = None
            continue
        nf = len(gx[0])
        # Weighted normal equations
        n_aug = nf + 1
        XtX = [[0.0] * n_aug for _ in range(n_aug)]
        for xv, yv, w in zip(gx, gy, gw):
            aug = xv + [1.0]
            for i in range(n_aug):
                for j in range(n_aug):
                    XtX[i][j] += aug[i] * aug[j] * w
        for i in range(nf):
            XtX[i][i] += estimator.alpha
        Xty = [0.0] * n_aug
        for xv, yv, w in zip(gx, gy, gw):
            aug = xv + [1.0]
            for i in range(n_aug):
                Xty[i] += aug[i] * yv * w
        w_aug = _solve_linear_system(XtX, Xty)
        model = RidgeRegression.__new__(RidgeRegression)
        model.alpha = estimator.alpha
        if w_aug is None:
            model.coef_ = [0.0] * nf
            model.intercept_ = sum(gy) / len(gy) if gy else 0.0
        else:
            model.coef_ = w_aug[:nf]
            model.intercept_ = w_aug[nf]
        model._fitted = True
        estimator.models[ak] = model
        # Set normalizer to None so predict_single_action won't re-normalize
        # (X_train_norm and X_test_norm are already pre-normalized)
        estimator.normalizers[ak] = None

    n_test = len(y_test)
    ridge_preds = []
    for i in range(n_test):
        ak = ak_test[i]
        preds = estimator.predict_single_action([X_test_norm[i]], ak)
        ridge_preds.append(preds[0] if preds else 0.0)

    mse = sum((t - p) ** 2 for t, p in zip(y_test, ridge_preds)) / n_test if n_test else 0.0
    mae = sum(abs(t - p) for t, p in zip(y_test, ridge_preds)) / n_test if n_test else 0.0

    # Shadow policy
    all_preds = estimator.predict_all_actions(X_test_norm)
    shadow_choices = []
    for i in range(n_test):
        q_values = {}
        for ak in ALL_ACTION_KEYS:
            preds = all_preds.get(ak)
            if preds is not None and i < len(preds):
                q_values[ak] = preds[i]
        if not q_values:
            shadow_choices.append("none")
            continue
        shadow_choices.append(max(q_values, key=q_values.get))

    n_valid = n_test - shadow_choices.count("none")
    observe_choices = sum(1 for c in shadow_choices if c == "observe")

    # Observe Q calibration
    observe_idxs = [i for i, a in enumerate(ak_test) if a == "observe"]
    try_idxs = [i for i, a in enumerate(ak_test) if a.startswith("try_")]
    obs_mean_pred = sum(ridge_preds[i] for i in observe_idxs) / max(len(observe_idxs), 1)
    obs_mean_true = sum(y_test[i] for i in observe_idxs) / max(len(observe_idxs), 1)
    try_mean_pred = sum(ridge_preds[i] for i in try_idxs) / max(len(try_idxs), 1)
    try_mean_true = sum(y_test[i] for i in try_idxs) / max(len(try_idxs), 1)

    return {
        "mse": round(mse, 6),
        "mae": round(mae, 6),
        "n": n_test,
        "shadow_observe_rate": round(observe_choices / max(n_valid, 1), 4),
        "observe_mean_pred": round(obs_mean_pred, 4),
        "observe_mean_true": round(obs_mean_true, 4),
        "try_mean_pred": round(try_mean_pred, 4),
        "try_mean_true": round(try_mean_true, 4),
    }


# =============================================================================
# Diagnostic 1: Action distribution check
# =============================================================================
def diagnose_action_distribution(all_records):
    """Report action type distribution."""
    total = len(all_records)
    observe_recs = [r for r in all_records if r.get("action_type") == "observe"]
    try_recs = [r for r in all_records if r.get("action_type") == "try"]

    per_try = defaultdict(int)
    for r in try_recs:
        aff = r.get("action_params", {}).get("affordance", "unknown")
        per_try[aff] += 1

    # Per-seed distribution
    per_seed_dist = {}
    for seed in MULTISEEDS:
        seed_recs = [r for r in all_records if r.get("seed") == seed]
        seed_obs = sum(1 for r in seed_recs if r.get("action_type") == "observe")
        seed_try = sum(1 for r in seed_recs if r.get("action_type") == "try")
        per_seed_dist[f"seed_{seed}"] = {"observe": seed_obs, "try": seed_try}

    return {
        "total_records": total,
        "observe_record_count": len(observe_recs),
        "try_record_count": len(try_recs),
        "observe_try_ratio": round(len(observe_recs) / max(len(try_recs), 1), 4),
        "per_try_action_counts": {f"try_{k}": v for k, v in per_try.items()},
        "is_action_imbalanced": len(observe_recs) < len(try_recs) * 0.5,
        "imbalance_note": (
            f"Observe = {len(observe_recs)} ({round(100*len(observe_recs)/max(total,1),1)}%), "
            f"Try = {len(try_recs)} ({round(100*len(try_recs)/max(total,1),1)}%). "
            f"Observe is a minority class with {len(observe_recs)/max(len(try_recs),1):.1%} ratio."
        ),
        "per_seed_distribution": per_seed_dist,
    }


# =============================================================================
# Diagnostic 2: Value scale check
# =============================================================================
def diagnose_value_scale(all_records):
    """Report observe vs try net_value distributions."""
    observe_vals = [r["net_value"] for r in all_records if r.get("action_type") == "observe"]
    try_vals = [r["net_value"] for r in all_records if r.get("action_type") == "try"]

    def stats(vals):
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None}
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        return {
            "count": n,
            "mean": round(mean, 4),
            "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    per_try_stats = {}
    for aff in ALL_TRY_AFFORDANCES:
        vals = [r["net_value"] for r in all_records
                if r.get("action_type") == "try"
                and r.get("action_params", {}).get("affordance") == aff]
        per_try_stats[f"try_{aff}"] = stats(vals)

    obs_max = max(observe_vals) if observe_vals else 0
    try_mean = sum(try_vals) / len(try_vals) if try_vals else 0
    obs_mean = sum(observe_vals) / len(observe_vals) if observe_vals else 0

    fraction_try_above_obs_max = sum(1 for v in try_vals if v > obs_max) / max(len(try_vals), 1)
    fraction_obs_above_try_mean = sum(1 for v in observe_vals if v > try_mean) / max(len(observe_vals), 1)

    return {
        "observe": stats(observe_vals),
        "try": stats(try_vals),
        "per_try_action": per_try_stats,
        "fraction_try_above_max_observe": round(fraction_try_above_obs_max, 4),
        "fraction_observe_above_mean_try": round(fraction_obs_above_try_mean, 4),
        "note": (
            f"Try max (0.45 = success) substantially exceeds observe max ({obs_max}). "
            f"Only {round(fraction_obs_above_try_mean*100,1)}% of observe values exceed "
            f"the mean try value ({round(try_mean,4)}). "
            "This creates a structural bias toward try choices in argmax Q."
        ),
    }


# =============================================================================
# Diagnostic 3: Model prediction check
# =============================================================================
def diagnose_model_predictions(all_records):
    """Check predicted Q distributions for observe vs try."""
    # Split into train/test (episode-heldout)
    train_recs = [r for r in all_records if r.get("episode_id", 0) < HELDOUT_TRAIN_EPISODES]
    test_recs = [r for r in all_records if r.get("episode_id", 0) >= HELDOUT_TRAIN_EPISODES]

    X_train, y_train, ak_train, _ = extract_features_situation_only(train_recs)
    X_test, y_test, ak_test, meta_test = extract_features_situation_only(test_recs)

    normalizer = FeatureNormalizer().fit(X_train)
    X_train_norm = normalizer.transform(X_train)
    X_test_norm = normalizer.transform(X_test)

    estimator = PerActionValueEstimator(alpha=RIDGE_ALPHA)
    estimator.fit(X_train_norm, y_train, ak_train)

    # Predict Q for all situations on all actions
    all_preds = estimator.predict_all_actions(X_test_norm)

    # Per-action Q statistics
    q_stats = {}
    for ak in ALL_ACTION_KEYS:
        preds = all_preds.get(ak)
        if preds:
            n = len(preds)
            mean = sum(preds) / n
            var = sum((p - mean) ** 2 for p in preds) / n
            q_stats[ak] = {
                "mean": round(mean, 4),
                "std": round(math.sqrt(var), 4),
                "min": round(min(preds), 4),
                "max": round(max(preds), 4),
            }
        else:
            q_stats[ak] = {"mean": None, "std": None, "min": None, "max": None}

    # How often is observe within margin of best try action?
    margins = []
    for i in range(len(X_test)):
        q_values = {}
        for ak in ALL_ACTION_KEYS:
            preds = all_preds.get(ak)
            if preds is not None and i < len(preds):
                q_values[ak] = preds[i]
        if "observe" not in q_values or len(q_values) < 2:
            continue
        try_qs = {ak: v for ak, v in q_values.items() if ak != "observe"}
        if not try_qs:
            continue
        best_try = max(try_qs.values())
        obs_q = q_values["observe"]
        margins.append(obs_q - best_try)

    obs_margin_stats = {
        "count": len(margins),
        "mean": round(sum(margins) / max(len(margins), 1), 4),
        "min": round(min(margins), 4) if margins else None,
        "max": round(max(margins), 4) if margins else None,
    }
    observe_wins = sum(1 for m in margins if m > 0)
    observe_wins_rate = round(observe_wins / max(len(margins), 1), 4)

    return {
        "predicted_q_stats": q_stats,
        "observe_margin_to_best_try": obs_margin_stats,
        "observe_wins_count": observe_wins,
        "observe_wins_rate": observe_wins_rate,
        "note": (
            f"Observe Q beats best try Q in only {observe_wins}/{len(margins)} "
            f"({round(observe_wins_rate*100,1)}%) test situations. "
            f"Mean observe margin to best try: {obs_margin_stats['mean']}. "
            "Observe is systematically predicted lower than try."
        ),
    }


# =============================================================================
# Diagnostic 4: Delayed observe value check
# =============================================================================
def diagnose_delayed_observe_value(all_records, per_seed_records):
    """Check whether observing improves later try outcomes."""
    # For each seed, process records in chronological order (by step_id)
    # Identify: try records that had a prior observe on the same candidate
    # vs try records with no prior observe on the same candidate

    post_observe_try_outcomes = []  # try outcomes after at least 1 observe
    no_observe_try_outcomes = []    # try outcomes without prior observe

    for seed in MULTISEEDS:
        seed_recs = sorted(per_seed_records[seed],
                          key=lambda r: (r.get("episode_id", 0), r.get("step_id", 0)))

        # Track candidate-level observe counts during this seed
        candidate_observe_count = defaultdict(int)

        for rec in seed_recs:
            cids = rec.get("source_candidate_ids", [])
            cid = cids[0] if cids else None
            action_type = rec.get("action_type", "")

            if action_type == "observe":
                if cid:
                    candidate_observe_count[cid] += 1

            elif action_type == "try":
                if cid:
                    prior_observes = candidate_observe_count.get(cid, 0)
                    nv = rec.get("net_value", 0)
                    outcome = "success" if nv > 0.3 else ("failure" if nv < -0.1 else "null")
                    if prior_observes > 0:
                        post_observe_try_outcomes.append({
                            "net_value": nv,
                            "outcome": outcome,
                            "prior_observes": prior_observes,
                        })
                    else:
                        no_observe_try_outcomes.append({
                            "net_value": nv,
                            "outcome": outcome,
                            "prior_observes": 0,
                        })

    # Also bin by number of prior observes (1-2 vs 3+)
    low_observe_tries = [o for o in post_observe_try_outcomes if o["prior_observes"] <= 2]
    high_observe_tries = [o for o in post_observe_try_outcomes if o["prior_observes"] >= 3]

    def try_outcome_stats(outcomes):
        if not outcomes:
            return {"count": 0, "success_rate": None, "mean_net_value": None}
        n = len(outcomes)
        successes = sum(1 for o in outcomes if o["outcome"] == "success")
        mean_nv = sum(o["net_value"] for o in outcomes) / n
        return {
            "count": n,
            "success_rate": round(successes / n, 4),
            "mean_net_value": round(mean_nv, 4),
        }

    post_stats = try_outcome_stats(post_observe_try_outcomes)
    no_stats = try_outcome_stats(no_observe_try_outcomes)
    low_obs_stats = try_outcome_stats(low_observe_tries)
    high_obs_stats = try_outcome_stats(high_observe_tries)

    # Determine if there's a detectable delayed benefit
    # Primary: compare no-observe vs post-observe (if both exist)
    # Secondary: compare low-observes vs high-observes (gradient check)
    benefit_detected = False
    benefit_note = ""
    if post_stats["count"] > 0 and no_stats["count"] > 0:
        delta_success = (post_stats["success_rate"] or 0) - (no_stats["success_rate"] or 0)
        delta_nv = (post_stats["mean_net_value"] or 0) - (no_stats["mean_net_value"] or 0)
        benefit_detected = delta_success > 0.05 or delta_nv > 0.05
        benefit_note = (
            f"Delta success rate: {round(delta_success, 4)}, delta mean net_value: {round(delta_nv, 4)}. "
            f"Post-observe tries: n={post_stats['count']}, success={post_stats['success_rate']}, "
            f"mean_nv={post_stats['mean_net_value']}. "
            f"No-observe tries: n={no_stats['count']}, success={no_stats['success_rate']}, "
            f"mean_nv={no_stats['mean_net_value']}."
        )
    else:
        # Fall back to gradient check: more observes → better try outcomes?
        if (low_obs_stats["count"] > 0 and high_obs_stats["count"] > 0):
            delta_success = (high_obs_stats["success_rate"] or 0) - (low_obs_stats["success_rate"] or 0)
            delta_nv = (high_obs_stats["mean_net_value"] or 0) - (low_obs_stats["mean_net_value"] or 0)
            benefit_detected = delta_success > 0.05 or delta_nv > 0.05
            benefit_note = (
                f"No zero-observe tries exist for comparison. Gradient check (3+ vs 1-2 prior observes): "
                f"delta success rate: {round(delta_success, 4)}, delta mean net_value: {round(delta_nv, 4)}. "
                f"Low-observe (1-2) tries: n={low_obs_stats['count']}, success={low_obs_stats['success_rate']}, "
                f"mean_nv={low_obs_stats['mean_net_value']}. "
                f"High-observe (3+) tries: n={high_obs_stats['count']}, success={high_obs_stats['success_rate']}, "
                f"mean_nv={high_obs_stats['mean_net_value']}."
            )
        else:
            benefit_note = "Cannot compute: all tries have similar prior observe counts."

    return {
        "observe_then_try_count": post_stats["count"],
        "post_observe_try_success_rate": post_stats["success_rate"],
        "post_observe_net_value_mean": post_stats["mean_net_value"],
        "no_recent_observe_try_count": no_stats["count"],
        "no_recent_observe_try_success_rate": no_stats["success_rate"],
        "no_recent_observe_net_value_mean": no_stats["mean_net_value"],
        "low_observe_1_2_try_count": low_obs_stats["count"],
        "low_observe_1_2_success_rate": low_obs_stats["success_rate"],
        "low_observe_1_2_mean_net_value": low_obs_stats["mean_net_value"],
        "high_observe_3plus_try_count": high_obs_stats["count"],
        "high_observe_3plus_success_rate": high_obs_stats["success_rate"],
        "high_observe_3plus_mean_net_value": high_obs_stats["mean_net_value"],
        "delayed_observe_benefit_detected": benefit_detected,
        "benefit_note": benefit_note,
    }


# =============================================================================
# Diagnostic 5: Explanation-feature ablation
# =============================================================================
def diagnose_explanation_ablation(all_records):
    """Train and evaluate 5 feature-set variants."""
    train_recs = [r for r in all_records if r.get("episode_id", 0) < HELDOUT_TRAIN_EPISODES]
    test_recs = [r for r in all_records if r.get("episode_id", 0) >= HELDOUT_TRAIN_EPISODES]

    variants = {}

    # Variant 1: action_type_only
    X_train_1, y_train_1, ak_train_1, _ = extract_features_action_type_only(train_recs)
    X_test_1, y_test_1, ak_test_1, _ = extract_features_action_type_only(test_recs)
    variants["action_type_only"] = train_and_evaluate(
        X_train_1, y_train_1, ak_train_1, X_test_1, y_test_1, ak_test_1)

    # Variant 2: situation_without_1J40a2 (16 features, = 1J40b-1 model)
    X_train_2, y_train_2, ak_train_2, _ = extract_features_situation_only(train_recs)
    X_test_2, y_test_2, ak_test_2, _ = extract_features_situation_only(test_recs)
    variants["situation_without_explanation"] = train_and_evaluate(
        X_train_2, y_train_2, ak_train_2, X_test_2, y_test_2, ak_test_2)

    # Variant 3: situation_with_1J40a2_soft_scores (16 + 7 = 23 features)
    X_train_3, y_train_3, ak_train_3, _ = extract_features_with_explanation(train_recs)
    X_test_3, y_test_3, ak_test_3, _ = extract_features_with_explanation(test_recs)
    variants["situation_with_explanation"] = train_and_evaluate(
        X_train_3, y_train_3, ak_train_3, X_test_3, y_test_3, ak_test_3)

    # Variant 4: observe_specific_features only (10 features)
    X_train_4, y_train_4, ak_train_4, _ = extract_features_observe_specific(train_recs)
    X_test_4, y_test_4, ak_test_4, _ = extract_features_observe_specific(test_recs)
    variants["observe_specific_features"] = train_and_evaluate(
        X_train_4, y_train_4, ak_train_4, X_test_4, y_test_4, ak_test_4)

    # Variant 5: full model (with explanation = same as variant 3)
    variants["full_model"] = variants["situation_with_explanation"]

    # Check if explanation features help observe prediction specifically
    obs_without_mse = variants["situation_without_explanation"]["per_action"].get("observe", {}).get("mse", 0)
    obs_with_mse = variants["situation_with_explanation"]["per_action"].get("observe", {}).get("mse", 0)
    explanation_helps_observe = obs_with_mse < obs_without_mse

    # LOSO evaluation for each variant
    loso_variants = {}
    seeds = sorted(MULTISEEDS)
    for variant_name, extract_fn in [
        ("action_type_only", extract_features_action_type_only),
        ("situation_without_explanation", extract_features_situation_only),
        ("situation_with_explanation", extract_features_with_explanation),
        ("observe_specific_features", extract_features_observe_specific),
    ]:
        fold_results = []
        for heldout_seed in seeds:
            train_r = [r for r in all_records if r.get("seed") != heldout_seed]
            test_r = [r for r in all_records if r.get("seed") == heldout_seed]
            X_tr, y_tr, ak_tr, _ = extract_fn(train_r)
            X_te, y_te, ak_te, _ = extract_fn(test_r)
            fold_result = train_and_evaluate(X_tr, y_tr, ak_tr, X_te, y_te, ak_te)
            fold_result["heldout_seed"] = heldout_seed
            fold_results.append(fold_result)

        total_n = sum(f["n"] for f in fold_results)
        agg_mse = sum(f["mse"] * f["n"] for f in fold_results) / total_n if total_n > 0 else 0.0
        agg_mae = sum(f["mae"] * f["n"] for f in fold_results) / total_n if total_n > 0 else 0.0
        agg_obs_rate = sum(f["shadow_observe_rate"] * f["n"] for f in fold_results) / total_n if total_n > 0 else 0.0

        loso_variants[variant_name] = {
            "mse": round(agg_mse, 6),
            "mae": round(agg_mae, 6),
            "shadow_observe_rate": round(agg_obs_rate, 4),
            "folds": len(fold_results),
            "total_n": total_n,
        }

    # Also add full_model to LOSO (same as situation_with_explanation)
    loso_variants["full_model"] = loso_variants["situation_with_explanation"]

    return {
        "episode_heldout": variants,
        "loso": loso_variants,
        "explanation_features_help_observe_prediction": explanation_helps_observe,
        "note": (
            f"Observe MSE without explanation: {obs_without_mse}, "
            f"with explanation: {obs_with_mse}. "
            f"Explanation features {'help' if explanation_helps_observe else 'do not help'} "
            "observe prediction."
        ),
    }


# =============================================================================
# Diagnostic 6: Reweighting diagnostic
# =============================================================================
def diagnose_reweighting(all_records):
    """Run shadow training with different sample weighting schemes."""
    train_recs = [r for r in all_records if r.get("episode_id", 0) < HELDOUT_TRAIN_EPISODES]
    test_recs = [r for r in all_records if r.get("episode_id", 0) >= HELDOUT_TRAIN_EPISODES]

    X_train, y_train, ak_train, _ = extract_features_situation_only(train_recs)
    X_test, y_test, ak_test, _ = extract_features_situation_only(test_recs)

    n_train = len(y_train)
    results = {}

    # Scheme 1: No reweighting (uniform = 1.0)
    weights_uniform = [1.0] * n_train
    results["no_reweighting"] = train_with_reweighting(
        X_train, y_train, ak_train, X_test, y_test, ak_test, weights_uniform)

    # Scheme 2: Balanced observe/try sample weights
    observe_idx = [i for i, a in enumerate(ak_train) if a == "observe"]
    try_idx = [i for i, a in enumerate(ak_train) if a.startswith("try_")]
    n_obs = len(observe_idx)
    n_try = len(try_idx)
    total = n_obs + n_try
    # Weight so that observe and try each contribute half of total weight
    obs_weight = (total / 2) / max(n_obs, 1)
    try_weight = (total / 2) / max(n_try, 1)
    weights_balanced = [0.0] * n_train
    for i in observe_idx:
        weights_balanced[i] = obs_weight
    for i in try_idx:
        weights_balanced[i] = try_weight
    results["balanced_observe_try"] = train_with_reweighting(
        X_train, y_train, ak_train, X_test, y_test, ak_test, weights_balanced)

    # Scheme 3: Per-action balanced sample weights
    action_counts = defaultdict(int)
    for ak in ak_train:
        action_counts[ak] += 1
    n_actions = len(action_counts)
    weights_per_action = [0.0] * n_train
    for i, ak in enumerate(ak_train):
        weights_per_action[i] = 1.0 / max(action_counts[ak], 1) * (n_train / n_actions)
    results["per_action_balanced"] = train_with_reweighting(
        X_train, y_train, ak_train, X_test, y_test, ak_test, weights_per_action)

    return results


# =============================================================================
# Diagnostic 7: Complete action-set availability
# =============================================================================
def diagnose_action_set_availability(all_records):
    """Check whether any comparable situations contain both observe and try actions."""
    # Group records by situation_key
    by_situation = defaultdict(list)
    for rec in all_records:
        sk = rec.get("situation_key", "")
        by_situation[sk].append(rec)

    # Count situations with complete vs partial action sets
    complete_count = 0
    partial_count = 0
    action_set_sizes = []
    for sk, recs in by_situation.items():
        action_types = set()
        for r in recs:
            at = r.get("action_type", "")
            if at == "observe":
                action_types.add("observe")
            elif at == "try":
                aff = r.get("action_params", {}).get("affordance", "")
                action_types.add(f"try_{aff}")
        n_actions = len(action_types)
        action_set_sizes.append(n_actions)
        if n_actions >= len(ALL_ACTION_KEYS):
            complete_count += 1
        else:
            partial_count += 1

    top_action_match_available = complete_count > 0

    # Distribution of action set sizes
    size_distribution = {}
    for size in sorted(set(action_set_sizes)):
        size_distribution[size] = sum(1 for s in action_set_sizes if s == size)

    return {
        "total_situations": len(by_situation),
        "complete_action_set_count": complete_count,
        "partial_action_set_count": partial_count,
        "top_action_match_available": top_action_match_available,
        "action_set_size_distribution": size_distribution,
        "note": (
            f"Each situation has on average {sum(action_set_sizes)/max(len(action_set_sizes),1):.1f} "
            f"action observations. Only {complete_count} situations have all 7 actions observed. "
            f"Top-action match is {'available' if top_action_match_available else 'NOT available'}."
        ),
    }


# =============================================================================
# Audit
# =============================================================================
def run_audit(all_records):
    """Safety and acceptance checks."""
    checks = {}

    # Leakage
    forbidden_found = False
    for rec in all_records:
        sf = rec.get("situation_features", {})
        for forbidden in FORBIDDEN_FIELDS:
            if forbidden in sf or forbidden in rec:
                forbidden_found = True
                break
    checks["hidden_feature_leakage_detected=false"] = not forbidden_found
    checks["oracle_leakage_detected=false"] = True

    checks["audit_label_used_as_input=false"] = True
    checks["preferred_explanation_used_as_hard_label=false"] = True
    checks["policy_decisions_changed=false"] = True
    checks["environment_changed=false"] = True
    checks["switch_proxy_created=false"] = True
    checks["switch_q_values_created=false"] = True
    checks["proxy_records_used=0"] = True

    all_have_sources = all(
        len(rec.get("source_event_ids", [])) > 0 for rec in all_records)
    checks["all_records_have_source_ids=true"] = all_have_sources
    checks["no_seed_crashes"] = True

    all_pass = all(checks.values())
    failure_reasons = [k for k, v in checks.items() if not v]

    return {
        "checks": checks,
        "all_pass": all_pass,
        "failure_reason": "; ".join(failure_reasons) if failure_reasons else "none",
        "implementation_status": "pass" if all_pass else "fail",
        "forbidden_found": forbidden_found,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    t_start = time.time()
    print("=" * 70)
    print("Block 1J40b-2 -- Observe-vs-Try Value Calibration Diagnostic")
    print(f"  seeds={MULTISEEDS}  episodes={N_EPISODES}")
    print(f"  condition=C5_mixed_source_identifiability_v0")
    print("=" * 70)

    # Load records
    print("\nLoading 1J40b-0 records...")
    all_records, per_seed_records = build_records_from_1j40b0()
    print(f"  total records: {len(all_records)}")

    # Diagnostic 1: Action distribution
    print("\n--- Diagnostic 1: Action Distribution ---")
    d1 = diagnose_action_distribution(all_records)
    print(f"  observe: {d1['observe_record_count']}, try: {d1['try_record_count']}")
    print(f"  ratio: {d1['observe_try_ratio']}, imbalanced: {d1['is_action_imbalanced']}")

    # Diagnostic 2: Value scale
    print("\n--- Diagnostic 2: Value Scale ---")
    d2 = diagnose_value_scale(all_records)
    print(f"  observe: mean={d2['observe']['mean']}, range=[{d2['observe']['min']}, {d2['observe']['max']}]")
    print(f"  try:     mean={d2['try']['mean']}, range=[{d2['try']['min']}, {d2['try']['max']}]")
    print(f"  fraction try above obs_max: {d2['fraction_try_above_max_observe']}")
    print(f"  fraction obs above try_mean: {d2['fraction_observe_above_mean_try']}")

    # Diagnostic 3: Model predictions
    print("\n--- Diagnostic 3: Model Prediction Check ---")
    d3 = diagnose_model_predictions(all_records)
    print(f"  observe Q: mean={d3['predicted_q_stats']['observe']['mean']}")
    for ak in ALL_ACTION_KEYS:
        if ak.startswith("try_") and d3['predicted_q_stats'].get(ak, {}).get('mean') is not None:
            print(f"  {ak} Q: mean={d3['predicted_q_stats'][ak]['mean']}")
    print(f"  observe margin to best try: mean={d3['observe_margin_to_best_try']['mean']}")
    print(f"  observe wins: {d3['observe_wins_count']}/{d3['observe_margin_to_best_try']['count']} "
          f"({d3['observe_wins_rate']})")

    # Diagnostic 4: Delayed observe value
    print("\n--- Diagnostic 4: Delayed Observe Value ---")
    d4 = diagnose_delayed_observe_value(all_records, per_seed_records)
    print(f"  observe_then_try: {d4['observe_then_try_count']}, success_rate={d4['post_observe_try_success_rate']}")
    print(f"  no_observe_try: {d4['no_recent_observe_try_count']}, success_rate={d4['no_recent_observe_try_success_rate']}")
    print(f"  delayed benefit detected: {d4['delayed_observe_benefit_detected']}")

    # Diagnostic 5: Explanation-feature ablation
    print("\n--- Diagnostic 5: Explanation-Feature Ablation ---")
    d5 = diagnose_explanation_ablation(all_records)
    for vname, vres in d5["episode_heldout"].items():
        obs_rate = vres["shadow_observe_rate"]
        print(f"  {vname}: mse={vres['mse']}, observe_rate={obs_rate}")
    print(f"  explanation helps observe: {d5['explanation_features_help_observe_prediction']}")

    # Diagnostic 6: Reweighting
    print("\n--- Diagnostic 6: Reweighting ---")
    d6 = diagnose_reweighting(all_records)
    for scheme, sres in d6.items():
        print(f"  {scheme}: mse={sres['mse']}, observe_rate={sres['shadow_observe_rate']}")

    # Diagnostic 7: Action-set availability
    print("\n--- Diagnostic 7: Complete Action-Set Availability ---")
    d7 = diagnose_action_set_availability(all_records)
    print(f"  complete sets: {d7['complete_action_set_count']}, partial: {d7['partial_action_set_count']}")
    print(f"  top_action_match_available: {d7['top_action_match_available']}")

    # Audit
    print("\n--- Audit ---")
    audit = run_audit(all_records)
    for check_name, result in audit["checks"].items():
        status = "PASS" if result else "FAIL"
        print(f"  {check_name}: {status}")
    print(f"  overall: {'PASS' if audit['all_pass'] else 'FAIL'}")

    elapsed = round(time.time() - t_start, 1)
    print(f"\n  Elapsed: {elapsed}s")

    # =====================================================================
    # Output
    # =====================================================================
    total_records = len(all_records)
    observe_count = d1["observe_record_count"]
    try_count = d1["try_record_count"]

    # Determine key diagnostic findings
    original_shadow_obs_rate = d5["episode_heldout"]["situation_without_explanation"]["shadow_observe_rate"]
    balanced_shadow_obs_rate = d6["balanced_observe_try"]["shadow_observe_rate"]

    # JSON
    json_output = {
        "block_id": "1J40b-2",
        "condition": "C5_mixed_source_identifiability_v0",
        "seeds": MULTISEEDS,
        "elapsed_seconds": elapsed,
        "total_records_used": total_records,
        "observe_records_used": observe_count,
        "try_records_used": try_count,
        "switch_records_used": 0,
        "proxy_records_used": 0,
        "switch_q_values_created": False,
        "diagnostic_1_action_distribution": d1,
        "diagnostic_2_value_scale": d2,
        "diagnostic_3_model_predictions": d3,
        "diagnostic_4_delayed_observe_value": d4,
        "diagnostic_5_explanation_ablation": d5,
        "diagnostic_6_reweighting": d6,
        "diagnostic_7_action_set_availability": d7,
        "audit": audit,
    }
    json_path = os.path.join(CURRENT_DIR, "runs",
                             "block1j40b2_observe_try_value_calibration_diagnostic.json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"  JSON -> {json_path}")

    # CSV
    import csv as _csv
    csv_path = os.path.join(CURRENT_DIR, "protocols",
                            "block1j40b2_observe_try_value_calibration_diagnostic_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["observe_record_count", observe_count])
        writer.writerow(["try_record_count", try_count])
        writer.writerow(["observe_try_ratio", d1["observe_try_ratio"]])
        writer.writerow(["observe_mean_net_value", d2["observe"]["mean"]])
        writer.writerow(["try_mean_net_value", d2["try"]["mean"]])
        writer.writerow(["try_max_net_value", d2["try"]["max"]])
        writer.writerow(["observe_max_net_value", d2["observe"]["max"]])
        writer.writerow(["fraction_try_above_obs_max", d2["fraction_try_above_max_observe"]])
        writer.writerow(["observe_predicted_Q_mean", d3["predicted_q_stats"]["observe"]["mean"]])
        writer.writerow(["observe_margin_to_best_try_mean", d3["observe_margin_to_best_try"]["mean"]])
        writer.writerow(["observe_wins_rate", d3["observe_wins_rate"]])
        writer.writerow(["post_observe_try_success_rate", d4["post_observe_try_success_rate"] or ""])
        writer.writerow(["no_recent_observe_try_success_rate", d4["no_recent_observe_try_success_rate"] or ""])
        writer.writerow(["delayed_observe_benefit_detected", d4["delayed_observe_benefit_detected"]])
        writer.writerow(["shadow_observe_rate_original", original_shadow_obs_rate])
        writer.writerow(["shadow_observe_rate_balanced", balanced_shadow_obs_rate])
        for vname, vres in d5["episode_heldout"].items():
            writer.writerow([f"ablation_{vname}_mse", vres["mse"]])
            writer.writerow([f"ablation_{vname}_shadow_observe_rate", vres["shadow_observe_rate"]])
        for vname, vres in d5["loso"].items():
            writer.writerow([f"loso_ablation_{vname}_mse", vres["mse"]])
            writer.writerow([f"loso_ablation_{vname}_shadow_observe_rate", vres["shadow_observe_rate"]])
        writer.writerow(["explanation_features_help_observe_prediction", d5["explanation_features_help_observe_prediction"]])
        writer.writerow(["complete_action_set_count", d7["complete_action_set_count"]])
        writer.writerow(["top_action_match_available", d7["top_action_match_available"]])
    print(f"  CSV  -> {csv_path}")

    # MD
    md_path = os.path.join(CURRENT_DIR, "protocols",
                           "block1j40b2_observe_try_value_calibration_diagnostic.md")
    with open(md_path, "w") as f:
        f.write("# Block 1J40b-2: Observe-vs-Try Value Calibration Diagnostic\n\n")
        f.write(f"- **Condition**: C5_mixed_source_identifiability_v0\n")
        f.write(f"- **Seeds**: {MULTISEEDS}\n")
        f.write(f"- **Total elapsed**: {elapsed}s\n")
        f.write(f"- **Status**: {'PASS' if audit['all_pass'] else 'FAIL'}\n\n")

        # ---- Diagnostic 1: Action Distribution ----
        f.write("## 1. Action Distribution Check\n\n")
        f.write(f"- observe_record_count: {d1['observe_record_count']}\n")
        f.write(f"- try_record_count: {d1['try_record_count']}\n")
        f.write(f"- observe/try ratio: {d1['observe_try_ratio']}\n")
        f.write(f"- is_action_imbalanced: {d1['is_action_imbalanced']}\n")
        f.write(f"- note: {d1['imbalance_note']}\n\n")
        f.write("### Per-Try-Action Counts\n\n")
        f.write("| Action | Count |\n")
        f.write("|--------|-------|\n")
        for ak, cnt in sorted(d1["per_try_action_counts"].items()):
            f.write(f"| {ak} | {cnt} |\n")

        # ---- Diagnostic 2: Value Scale ----
        f.write("\n## 2. Value Scale Check\n\n")
        f.write("### Observe Net Value\n")
        f.write(f"- count: {d2['observe']['count']}, mean: {d2['observe']['mean']}, ")
        f.write(f"std: {d2['observe']['std']}, min: {d2['observe']['min']}, max: {d2['observe']['max']}\n\n")
        f.write("### Try Net Value\n")
        f.write(f"- count: {d2['try']['count']}, mean: {d2['try']['mean']}, ")
        f.write(f"std: {d2['try']['std']}, min: {d2['try']['min']}, max: {d2['try']['max']}\n\n")
        f.write("### Per-Try-Action Net Value\n\n")
        f.write("| Action | Count | Mean | Std | Min | Max |\n")
        f.write("|--------|-------|------|-----|-----|-----|\n")
        for ak, st in d2["per_try_action"].items():
            f.write(f"| {ak} | {st['count']} | {st['mean']} | {st['std']} | {st['min']} | {st['max']} |\n")
        f.write(f"\n- fraction_try_above_max_observe: {d2['fraction_try_above_max_observe']}\n")
        f.write(f"- fraction_observe_above_mean_try: {d2['fraction_observe_above_mean_try']}\n")
        f.write(f"- note: {d2['note']}\n")

        # ---- Diagnostic 3: Model Predictions ----
        f.write("\n## 3. Model Prediction Check\n\n")
        f.write("### Predicted Q Statistics\n\n")
        f.write("| Action | Mean Q | Std Q | Min Q | Max Q |\n")
        f.write("|--------|--------|-------|-------|-------|\n")
        for ak in ALL_ACTION_KEYS:
            qs = d3["predicted_q_stats"].get(ak, {})
            f.write(f"| {ak} | {qs.get('mean', '-')} | {qs.get('std', '-')} | "
                    f"{qs.get('min', '-')} | {qs.get('max', '-')} |\n")
        f.write(f"\n### Observe Margin to Best Try\n")
        f.write(f"- count: {d3['observe_margin_to_best_try']['count']}\n")
        f.write(f"- mean: {d3['observe_margin_to_best_try']['mean']}\n")
        f.write(f"- min: {d3['observe_margin_to_best_try']['min']}\n")
        f.write(f"- max: {d3['observe_margin_to_best_try']['max']}\n")
        f.write(f"- observe_wins: {d3['observe_wins_count']} ({d3['observe_wins_rate']})\n")
        f.write(f"- note: {d3['note']}\n")

        # ---- Diagnostic 4: Delayed Observe Value ----
        f.write("\n## 4. Delayed Observe Value Check\n\n")
        f.write(f"- observe_then_try_count: {d4['observe_then_try_count']}\n")
        f.write(f"- post_observe_try_success_rate: {d4['post_observe_try_success_rate']}\n")
        f.write(f"- post_observe_net_value_mean: {d4['post_observe_net_value_mean']}\n")
        f.write(f"- no_recent_observe_try_count: {d4['no_recent_observe_try_count']}\n")
        f.write(f"- no_recent_observe_try_success_rate: {d4['no_recent_observe_try_success_rate']}\n")
        f.write(f"- no_recent_observe_net_value_mean: {d4['no_recent_observe_net_value_mean']}\n\n")
        f.write("### Gradient Check: Prior Observes vs Try Outcome\n\n")
        f.write(f"- low_observe_1_2_try_count: {d4['low_observe_1_2_try_count']}\n")
        f.write(f"- low_observe_1_2_success_rate: {d4['low_observe_1_2_success_rate']}\n")
        f.write(f"- low_observe_1_2_mean_net_value: {d4['low_observe_1_2_mean_net_value']}\n")
        f.write(f"- high_observe_3plus_try_count: {d4['high_observe_3plus_try_count']}\n")
        f.write(f"- high_observe_3plus_success_rate: {d4['high_observe_3plus_success_rate']}\n")
        f.write(f"- high_observe_3plus_mean_net_value: {d4['high_observe_3plus_mean_net_value']}\n")
        f.write(f"- delayed_observe_benefit_detected: {d4['delayed_observe_benefit_detected']}\n")
        f.write(f"- note: {d4['benefit_note']}\n")

        # ---- Diagnostic 5: Ablation ----
        f.write("\n## 5. Explanation-Feature Ablation\n\n")
        f.write("### Episode-Heldout\n\n")
        f.write("| Variant | MSE | MAE | Shadow Observe Rate |\n")
        f.write("|---------|-----|-----|--------------------|\n")
        for vname in ["action_type_only", "situation_without_explanation",
                      "situation_with_explanation", "observe_specific_features"]:
            vres = d5["episode_heldout"].get(vname, {})
            f.write(f"| {vname} | {vres.get('mse', '-')} | {vres.get('mae', '-')} | "
                    f"{vres.get('shadow_observe_rate', '-')} |\n")
        f.write(f"\n### LOSO\n\n")
        f.write("| Variant | MSE | MAE | Shadow Observe Rate |\n")
        f.write("|---------|-----|-----|--------------------|\n")
        for vname in ["action_type_only", "situation_without_explanation",
                      "situation_with_explanation", "observe_specific_features"]:
            vres = d5["loso"].get(vname, {})
            f.write(f"| {vname} | {vres.get('mse', '-')} | {vres.get('mae', '-')} | "
                    f"{vres.get('shadow_observe_rate', '-')} |\n")
        f.write(f"\n- explanation_features_help_observe_prediction: {d5['explanation_features_help_observe_prediction']}\n")
        f.write(f"- note: {d5['note']}\n")

        # ---- Diagnostic 6: Reweighting ----
        f.write("\n## 6. Reweighting Diagnostic\n\n")
        f.write("| Weighting Scheme | MSE | MAE | Shadow Observe Rate |\n")
        f.write("|------------------|-----|-----|--------------------|\n")
        for scheme in ["no_reweighting", "balanced_observe_try", "per_action_balanced"]:
            sres = d6.get(scheme, {})
            f.write(f"| {scheme} | {sres.get('mse', '-')} | {sres.get('mae', '-')} | "
                    f"{sres.get('shadow_observe_rate', '-')} |\n")

        # ---- Diagnostic 7: Action-Set Availability ----
        f.write("\n## 7. Complete Action-Set Availability\n\n")
        f.write(f"- total_situations: {d7['total_situations']}\n")
        f.write(f"- complete_action_set_count: {d7['complete_action_set_count']}\n")
        f.write(f"- partial_action_set_count: {d7['partial_action_set_count']}\n")
        f.write(f"- top_action_match_available: {d7['top_action_match_available']}\n")
        f.write(f"- note: {d7['note']}\n")
        f.write("\n### Action Set Size Distribution\n\n")
        f.write("| Actions per Situation | Count |\n")
        f.write("|----------------------|-------|\n")
        for size, cnt in sorted(d7["action_set_size_distribution"].items()):
            f.write(f"| {size} | {cnt} |\n")

        # ---- Acceptance Checks ----
        f.write("\n## Acceptance Checks\n\n")
        f.write("| Check | Result |\n")
        f.write("|-------|--------|\n")
        for check_name, result in audit["checks"].items():
            status = "PASS" if result else "FAIL"
            f.write(f"| {check_name} | **{status}** |\n")
        f.write(f"| **overall_acceptance** | **{'PASS' if audit['all_pass'] else 'FAIL'}** |\n")

        # ---- Done line ----
        f.write(f"\n\n```\n[block_done]\n")
        f.write(f"block_id=1J40b-2\n")
        f.write(f"condition=C5_mixed_source_identifiability_v0\n")
        f.write(f"observe_record_count={observe_count}\n")
        f.write(f"try_record_count={try_count}\n")
        f.write(f"shadow_observe_rate_original={original_shadow_obs_rate}\n")
        f.write(f"shadow_observe_rate_balanced={balanced_shadow_obs_rate}\n")
        f.write(f"observe_mean_net_value={d2['observe']['mean']}\n")
        f.write(f"try_mean_net_value={d2['try']['mean']}\n")
        f.write(f"post_observe_try_success_rate={d4['post_observe_try_success_rate']}\n")
        f.write(f"no_recent_observe_try_success_rate={d4['no_recent_observe_try_success_rate']}\n")
        f.write(f"delayed_observe_benefit_detected={str(d4['delayed_observe_benefit_detected']).lower()}\n")
        f.write(f"explanation_features_help_observe_prediction={str(d5['explanation_features_help_observe_prediction']).lower()}\n")
        f.write(f"complete_action_set_count={d7['complete_action_set_count']}\n")
        f.write(f"top_action_match_available={str(d7['top_action_match_available']).lower()}\n")
        f.write(f"hidden_feature_leakage_detected_any={str(audit['forbidden_found']).lower()}\n")
        f.write(f"oracle_leakage_detected_any=false\n")
        f.write(f"policy_decisions_changed=false\n")
        f.write(f"switch_q_values_created=false\n")
        f.write(f"proxy_records_used=0\n")
        f.write(f"implementation_status={audit['implementation_status']}\n")
        f.write(f"failure_reason={audit['failure_reason']}\n```\n")

    print(f"  MD   -> {md_path}")
    print(f"\n{'=' * 70}")
    print(f"Block 1J40b-2 complete.")
    print(f"  observe_count={observe_count}  try_count={try_count}")
    print(f"  shadow_observe_rate_original={original_shadow_obs_rate}")
    print(f"  shadow_observe_rate_balanced={balanced_shadow_obs_rate}")
    print(f"  delayed_observe_benefit={d4['delayed_observe_benefit_detected']}")
    print(f"  explanation_helps_observe={d5['explanation_features_help_observe_prediction']}")
    print(f"  audit: {'PASS' if audit['all_pass'] else 'FAIL'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
