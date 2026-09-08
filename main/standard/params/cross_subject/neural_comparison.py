# ============================================================
# Cross-subject development experiment
# ============================================================

# Goal:
#   Compare source-only and adaptation methods using the
#   preprocessing/feature representation validated intra-subject.
#
# Methods:
#       - MLP ERM
#       - Deep CORAL
#       - DANN
#       - Importance Weighting
#
# Dataset:
#       BCI Competition IV 2a
#
# Scenario:
#       Cross-subject adaptation
#
# Target protocol:
#       100% of target samples available unlabeled as calibration.


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

BAND_CONFIGS = {
    #"8_30": [(8, 30)],
    "mu_beta": [(8, 12), (13, 30)],
}

PREPROCESSING_PARAMS = [
    {
        "dataset": "bci2a",
        "root_gdf": "datasets/bci2a/gdf",
        "root_mat": "datasets/bci2a/mat",
        "name": f"bci2a_cross_csp_{name}",
        "representation": "signal",
        "loader": {
            "channels": CHANNELS,
            "classes": COMMON_CLASSES,
        },
        "filter": {
            "bandpass": {
                "enabled": True,
                "bands": bands,
                "order": 5,
                "stack_bands": True,
            },
            "resample": {
                "enabled": True,
                "new_fs": 128.0,
            },
        },
        "show_progress": False,
    }
    for name, bands in BAND_CONFIGS.items()
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
# Feature selection
# ============================================================

#CSP_COMPONENTS = [2, 4, 6, 8]
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
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "device": "mps",
    "seed": 0,
}


# ============================================================
# Training methods
# ============================================================

NEURAL_EPOCHS = 200

TRAINING_PARAMS = [
    {
        "name": "mlp_small_reg",
        "learning": "neural_erm__mlp_small_reg",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "epochs": NEURAL_EPOCHS,
            "optimizer": "adam",
            "validation_fraction": 0.0,
        },
    },
    {
        "name": "mlp_deep_coral",
        "learning": "deep_coral",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "epochs": NEURAL_EPOCHS,
            "coral_lambda": 1.0,
        },
    },
    {
        "name": "mlp_dann",
        "learning": "dann",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "epochs": NEURAL_EPOCHS,
            "dann_lambda": 1.0,
            "domain_hidden_dim": 32,
        },
    },
    {
        "name": "mlp_importance_weighting",
        "learning": "importance_weighting",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "epochs": NEURAL_EPOCHS,
            "estimator": "ulsif",
            "estimator_params": {
                "regularization": 1e-3,
                "n_centers": 100,
            },
            "normalize_weights": True,
            "max_weight": 10.0,
        },
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
            "learning_method": "neural_erm__mlp_small_reg",
            "model_name": "mlp",
            "regime": "ERM",
            "method": "MLP ERM",
        },
        {
            "learning_method": "deep_coral",
            "model_name": "mlp",
            "regime": "UDA",
            "method": "Deep CORAL",
        },
        {
            "learning_method": "dann",
            "model_name": "mlp",
            "regime": "UDA",
            "method": "DANN",
        },
        {
            "learning_method": "importance_weighting",
            "model_name": "mlp",
            "regime": "UDA",
            "method": "Importance Weighting",
        },
    ],
    "tables": [
        {
            "name": "cross_subject",
            "scenario": "cross_subject",
            "setting_column": "Dataset",
            "output_name": "cross_subject_table.csv",
            "include_discrepancy": False,
            "filters": {
                "target_fraction": 1.0,
                "n_target_super_domains": 0,
                "use_max_source_domains": True,
            },
        },
    ],
}