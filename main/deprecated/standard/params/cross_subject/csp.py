# ============================================================
# Cross-subject development experiment
# ============================================================

# Goal:
#   Compare source-only and adaptation methods using the
#   preprocessing/feature representation validated intra-subject.
#
# Methods:
#       - Logistic Regression
#       - MLP ERM
#       - Positive-Negative Learning
#       - Standard Importance Weighting
#       - Structural Importance Weighting V1
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

LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "max_iter": 5000,
}

SVM_PARAMS = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale",
    "probability": True,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 1,
    "random_state": 0,
    "n_jobs": 1,
}

MLP_PARAMS = {
    "hidden_dims": (128, 64, 32),
    "activation": "relu",
    "dropout": 0.3,
    "batch_norm": True,
}

_NEURAL_BASE_PARAMS = {
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "optimizer": "adamw",
    "device": "cpu",
    "seed": 0,
}


# ============================================================
# Training methods
# ============================================================

TRAINING_PARAMS = [
    {
        "name": "logistic_regression",
        "learning": "sklearn_erm",
        "model": "logistic_regression",
        "model_params": LOGISTIC_REGRESSION_PARAMS,
        "training_params": {},
    },
    {
        "name": "svm",
        "learning": "sklearn_erm",
        "model": "svm",
        "model_params": SVM_PARAMS,
        "training_params": {},
    },
    {
        "name": "random_forest",
        "learning": "sklearn_erm",
        "model": "random_forest",
        "model_params": RANDOM_FOREST_PARAMS,
        "training_params": {},
    },
    {
        "name": "mlp_erm",
        "learning": "neural_erm",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {**_NEURAL_BASE_PARAMS},
    },
    {
        "name": "coral",
        "learning": "coral",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "coral_lambda": 1.0,
        },
    },
    {
        "name": "mmd",
        "learning": "mmd",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "mmd_lambda": 1.0,
            "gamma": None,
        },
    },
    {
        "name": "dev_positive_negative",
        "learning": "dev_positive_negative",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "beta": 1.0,
            "fusion_lambda": 1.0,
        },
    },
    {
        "name": "importance_weighting",
        "learning": "importance_weighting",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "estimator": "kliep",
            "estimator_params": {"n_centers": 100},
            "normalize_weights": True,
        },
    },
    {
        "name": "dev_structural_weighting_v1",
        "learning": "dev_structural_weighting_v1",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "estimator": "kliep",
            "estimator_params": {
                "n_centers": 100,
                "standardize": False,
            },
            "normalize_weights": True,
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
            "learning_method": "sklearn_erm",
            "model_name": "logistic_regression",
            "regime": "Classical",
            "method": "Logistic Regression",
        },
        {
            "learning_method": "sklearn_erm",
            "model_name": "svm",
            "regime": "Classical",
            "method": "SVM",
        },
        {
            "learning_method": "sklearn_erm",
            "model_name": "random_forest",
            "regime": "Classical",
            "method": "Random Forest",
        },
        {
            "learning_method": "neural_erm",
            "model_name": "mlp",
            "regime": "Classical",
            "method": "MLP ERM",
        },
        {
            "learning_method": "coral",
            "model_name": "mlp",
            "regime": "DG",
            "method": "CORAL",
        },
        {
            "learning_method": "mmd",
            "model_name": "mlp",
            "regime": "DG",
            "method": "MMD",
        },
        {
            "learning_method": "dev_positive_negative",
            "model_name": "mlp",
            "regime": "Development",
            "method": "Positive-Negative",
        },
        {
            "learning_method": "importance_weighting",
            "model_name": "mlp",
            "regime": "UDA",
            "method": "Importance Weighting",
        },
        {
            "learning_method": "dev_structural_weighting_v1",
            "model_name": "mlp",
            "regime": "UDA Development",
            "method": "Structural IW V1",
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