# ============================================================
# Intra-subject development experiment
# ============================================================

# Goal:
#   Verify that the EEG preprocessing/features are genuinely
#   learnable before evaluating cross-subject adaptation.
#
# Methods:
#       - Logistic Regression
#       - MLP ERM
#       - Positive-Negative Learning
#
# Dataset:
#       BCI Competition IV 2a
#
# Scenario:
#       Intra-subject
#
# Protocol:
#       80% train / 20% test within each subject.


# ============================================================
# Preprocessing
# ============================================================

COMMON_CLASSES = [
    "left_hand_imagery",
    "right_hand_imagery",
    "both_feet_imagery",
    "tongue_imagery",
]


FILTER_CONFIG = {
    "bandpass": {
        "enabled": True,
        "bands": [
            (8, 12),
            (13, 30),
        ],
        "order": 5,
        "stack_bands": True,
    },

    "resample": {
        "enabled": True,
        "new_fs": 128.0,
    },
}


FEATURE_CONFIG = {
    # Strongest / most relevant
    "logvar": {},
    "logcov": {},
    "std": {},
    "eig": {},
    "mom": {},

    # Useful but somewhat redundant
    # "cov": {},
    # "bandpower": {},

    # Lower priority
    # "q_stats": {},
    # "h_diff": {},
    # "mean": {},
    # "min": {},
    # "max": {},

    # Lowest priority in current implementation
    # "fft": {},
}


# ------------------------------------------------------------
# Motor-region channel configuration
# ------------------------------------------------------------

CHANNELS = [
    "Fz",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2",
    "POz",
]


PREPROCESSING_PARAMS = [
    {
        "dataset": "bci2a",

        "root_gdf": "datasets/bci2a/gdf",
        "root_mat": "datasets/bci2a/mat",

        "name": "bci2a_intra_dev",

        "loader": {
            "channels": CHANNELS,
            "classes": COMMON_CLASSES,
        },

        "filter": FILTER_CONFIG,
        "features": FEATURE_CONFIG,

        "show_progress": False,
    }
]


# ============================================================
# Scenario
# ============================================================

SCENARIO = "intra_subject"


SCENARIO_PARAMS = {
    "train_fraction": 0.8,
    "seed": 0,
}


# ============================================================
# Feature selection
# ============================================================

# ANOVA is supervised, deterministic, and substantially more
# meaningful here than random feature selection.

FS_K = 256
FEATURE_SELECTION_PARAMS = [
    {
        "method": "anova",
        "config_label": f"anova_k{FS_K}_standard",
        "params": {
            "k": FS_K,
            "pre_scaler": None,
            "post_scaler": "standard",
        },
    }
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
    "n_jobs": -1,
}

MLP_PARAMS = {
    "hidden_dims": (128, 64, 32),
    "activation": "relu",
    "dropout": 0.3,
    "batch_norm": True,
}

# ============================================================
# Shared neural training parameters
# ============================================================

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

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------
    {
        "name": "logistic_regression",
        "learning": "sklearn_erm",
        "model": "logistic_regression",
        "model_params": LOGISTIC_REGRESSION_PARAMS,
        "training_params": {},
    },

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------
    {
        "name": "svm",
        "learning": "sklearn_erm",
        "model": "svm",
        "model_params": SVM_PARAMS,
        "training_params": {},
    },

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------
    {
        "name": "random_forest",
        "learning": "sklearn_erm",
        "model": "random_forest",
        "model_params": RANDOM_FOREST_PARAMS,
        "training_params": {},
    },

    # --------------------------------------------------------
    # MLP ERM
    # --------------------------------------------------------
    {
        "name": "mlp_erm",
        "learning": "neural_erm",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
        },
    },

    # --------------------------------------------------------
    # Positive-Negative Learning
    # --------------------------------------------------------
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
]

# ============================================================
# Evaluation
# ============================================================

MODEL_EVALUATION_PARAMS = {}


# ============================================================
# Benchmark tables
# ============================================================

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
            "learning_method": "dev_positive_negative",
            "model_name": "mlp",
            "regime": "Development",
            "method": "Positive-Negative",
        },
    ],

    "tables": [
        {
            "name": "intra_subject",
            "scenario": "intra_subject",
            "setting_column": "Dataset",
            "output_name": "intra_subject_table.csv",
            "include_discrepancy": False,

            "filters": {
                "target_fraction": 0.8,
            },
        },
    ],
}