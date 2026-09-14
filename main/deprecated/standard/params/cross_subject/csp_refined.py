# ============================================================
# Cross-subject CSP refined experiment
# ============================================================

# Goal:
#   Re-evaluate CSP + Logistic Regression using the preprocessing
#   setup that substantially improved the EEGNet baseline.
#
# Dataset:
#       BCI Competition IV 2a
#
# Scenario:
#       Cross-subject
#
# Protocol:
#       Leave one subject out as target.
#       Train on all remaining source subjects.
#
# Preprocessing:
#       1-38 Hz
#       250 Hz
#       0-4 s motor-imagery window


# ============================================================
# Execution
# ============================================================

EXECUTION_PARAMS = {
    "max_workers": 1,
}


# ============================================================
# Preprocessing
# ============================================================

COMMON_CLASSES = [
    "left_hand_imagery",
    "right_hand_imagery",
    "both_feet_imagery",
    "tongue_imagery",
]

CHANNELS = [
    "Fz",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

PREPROCESSING_PARAMS = [
    {
        "dataset": "bci2a",
        "root_gdf": "datasets/bci2a/gdf",
        "root_mat": "datasets/bci2a/mat",
        "name": "bci2a_cross_subject_csp_1_38_250hz",
        "representation": "signal",
        "loader": {
            "channels": CHANNELS,
            "classes": COMMON_CLASSES,
            "tmin": 0.0,
            "tmax": 3.996,
        },
        "filter": {
            "bandpass": {
                "enabled": True,
                "bands": [(1, 38)],
                "order": 5,
                "stack_bands": True,
            },
            "resample": {
                "enabled": False,
            },
        },
        "show_progress": False,
    },
]


# ============================================================
# Scenario
# ============================================================

SCENARIO = "cross_subject"

SCENARIO_PARAMS = {
    "source_counts": {
        "bci_iv_2a": ["all"],
    },
    "target_fractions": [1.0],
    "max_source_combinations": 1,
    "seed": 0,
}


# ============================================================
# CSP
# ============================================================

CSP_COMPONENTS = [6]

FEATURE_SELECTION_PARAMS = [
    {
        "method": "csp",
        "config_label": f"csp_{n}_standard",
        "params": {
            "n_components": n,
            "reg": 1e-6,
            "post_scaler": "standard",
        },
    }
    for n in CSP_COMPONENTS
]

# ============================================================
# Models
# ============================================================

MLP_PARAMS = {
    "hidden_dims": (32, 16),
    "activation": "relu",
    "dropout": 0.5,
    "batch_norm": True,
}

_NEURAL_BASE_PARAMS = {
    "epochs": 200,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "device": "mps",
    "seed": 0,
}


# ============================================================
# Training
# ============================================================

TRAINING_PARAMS = [
    {
        "name": "mlp_erm",
        "learning": "neural_erm",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {**_NEURAL_BASE_PARAMS},
    },
]


# ============================================================
# Evaluation
# ============================================================

MODEL_EVALUATION_PARAMS = {}


# ============================================================
# Benchmark tables
# ============================================================

BENCHMARK_TABLES_PARAMS = {
    "method_display": [
        {
            "learning_method": "neural_erm",
            "model_name": "mlp",
            "regime": "ERM",
            "method": "CSP + MLP",
        },
    ],
    "tables": [
        {
            "name": "cross_subject_csp_refined",
            "scenario": "cross_subject",
            "setting_column": "Dataset",
            "output_name": "cross_subject_csp_refined_table.csv",
            "include_discrepancy": False,
            "filters": {
                "target_fraction": 1.0,
                "n_target_super_domains": 0,
                "use_max_source_domains": True,
            },
        },
    ],
}