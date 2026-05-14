"""Mini-MC v0 — All constants."""

# ---- Grid layout ----
GRID_ROWS = 7
GRID_COLS = 9
AGENT_START = (3, 4)  # center; avg Manhattan dist ~3.93

# ---- Cost parameters (abstract budget units) ----
REACH_COST_PER_UNIT = 0.01
OBSERVE_COST = 0.005
PROBE_COST = 0.05

# ---- Budget ----
DEFAULT_INITIAL_BUDGET = 3.0

# ---- Experiment seeds ----
SEEDS = [101, 202, 303, 404, 505]

# ---- Sparse outcome coverage ----
COVERAGE = 0.5

# ---- Instance memory hyperparameters ----
INSTANCE_K = 10
SIMILARITY_POWER = 1.0
SIMILARITY_MODE = "nb_likelihood"

# ---- C13 cost weights (multiplies normalized_step_cost) ----
COST_WEIGHTS = [0.0, 0.5, 1.0, 1.5, 2.0]
COST_WEIGHT_LABELS = {0.0: "CW00", 0.5: "CW05", 1.0: "CW10",
                      1.5: "CW15", 2.0: "CW20"}

# ---- Cue conditions ----
CUE_CONDITIONS = [
    {"label": "C3_P060_O040", "p_target": 0.60, "p_other": 0.40, "absent": False},
    {"label": "C4_P055_O045", "p_target": 0.55, "p_other": 0.45, "absent": False},
    {"label": "C5_absent",    "p_target": 0.50, "p_other": 0.50, "absent": True},
    {"label": "C4_instance_subtype_cued_v1", "p_target": 0.60, "p_other": 0.40,
     "absent": False, "subtype": True,
     "subtype_config": "instance_subtype_cued_v1"},
    {"label": "C5_mixed_source_identifiability_v0", "p_target": 0.60, "p_other": 0.40,
     "absent": False, "subtype": True,
     "subtype_config": "mixed_source_identifiability_v0",
     "mixed_source": True},
]

# Default cue condition (set after calibration)
FIXED_CONDITION = CUE_CONDITIONS[0]

# ---- Main policy config IDs ----
POLICY_CONFIGS = [
    "C0a_no_interaction",
    "C0b_observe_only",
    "C2_cost_gate",
    "C8_cost_gate",
    "C13_cost_gate_CW00",
    "C13_cost_gate_CW05",
    "C13_cost_gate_CW10",
    "C13_cost_gate_CW15",
    "C13_cost_gate_CW20",
    "C14b_budgeted_oracle_probe",
]

# C14a is truth-answer oracle (no budget), run separately, not in main sweep
ORACLE_TRUTH_CONFIG = "C14a_truth_answer_oracle"

MAIN_NON_ORACLE_CONFIGS = [c for c in POLICY_CONFIGS if "C14" not in c]
ORACLE_CONFIGS = ["C14b_budgeted_oracle_probe"]

# ---- Diagnostic configs ----
DIAGNOSTIC_CONFIGS = [
    "C2_forced_budget",
    "C8_forced_budget",
    "C13_forced_visit_probe_gate_CW05",
]

# ---- Calibration grid ----
CALIBRATION_BUDGETS = [1.0, 1.5, 2.0, 3.0]
CALIBRATION_CW_SWEEP = [0.5, 1.0, 1.5, 2.0]  # C13 cost weights to test
CALIBRATION_CONFIGS = [
    "C0a_no_interaction",
    "C0b_observe_only",
    "C2_cost_gate",
    "C2_forced_budget",
    "C8_cost_gate",
    "C8_forced_budget",
    "C13_cost_gate",          # will sweep cost_weight
    "C13_forced_visit_probe_gate",  # will sweep cost_weight
    "C14a_truth_answer_oracle",
    "C14b_budgeted_oracle_probe",
]

# Calibration non-C13 configs (don't need cost_weight sweep)
CALIBRATION_FIXED_CONFIGS = [
    c for c in CALIBRATION_CONFIGS
    if "C13" not in c
]

# ---- Utility beta values for normalized utility ----
UTILITY_BETAS = [0.1, 0.25, 0.5]

# ---- Results ----
RESULTS_DIR = None  # set in run.py relative to this file's directory
