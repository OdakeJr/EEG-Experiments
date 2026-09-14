# main/feature_grid/params/intra_subject/all_features.py

# ============================================================
# Intra-subject fused handcrafted-feature grid
# ============================================================

EXECUTION_PARAMS = {"max_workers": 1}


# ============================================================
# Dataset
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


# ============================================================
# Features
# ============================================================

# All deterministic/sample-wise features are extracted together.
# Split-dependent transformations remain in FEATURE_SELECTION_PARAMS.

FEATURE_CONFIGS = {
    # Statistical
    "mean": {},
    "std": {},
    "var": {},
    "logvar": {},
    "skew": {},
    "kurtosis": {},
    "min": {},
    "max": {},
    "rms": {},
    "ptp": {},

    # Temporal
    "line_length": {},
    "hjorth_activity": {},
    "hjorth_mobility": {},
    "hjorth_complexity": {},
    "zero_crossing": {},
    "ar": {},

    # Spectral
    "bandpower": {"sfreq": 250.0},
    "relative_bandpower": {"sfreq": 250.0, "total_band": (1, 38)},
    "psd_stats": {"sfreq": 250.0, "band": (1, 38)},
    "spectral_entropy": {"sfreq": 250.0, "band": (1, 38)},
    "differential_entropy": {},

    # Nonlinear
    "sample_entropy": {},
    "permutation_entropy": {},
    "higuchi_fd": {},
    "petrosian_fd": {},

    # Covariance
    "cov": {},
    "logcov": {},
    "eig": {},

    # Time-frequency
    "wavelet_energy": {},
    "wavelet_entropy": {},
}


# ============================================================
# Frequency preprocessing
# ============================================================

BAND_CONFIGS = {
    "1_38": [(1, 38)],
    # "4_38": [(4, 38)],
    # "8_30": [(8, 30)],
    # "mu_beta": [(8, 12), (13, 30)],
}


# ============================================================
# Preprocessing
# ============================================================

PREPROCESSING_PARAMS = [
    {
        "dataset": "bci2a",
        "root_gdf": "datasets/bci2a/gdf",
        "root_mat": "datasets/bci2a/mat",
        "name": f"bci2a_intra_all_{band_name}",
        "loader": {
            "channels": CHANNELS,
            "classes": COMMON_CLASSES,
            "tmin": 0.0,
            "tmax": 3.996,
        },
        "filter": {
            "bandpass": {
                "enabled": True,
                "bands": bands,
                "order": 5,
                "stack_bands": True,
            },
            "resample": {
                "enabled": False,
            },
        },
        "features": FEATURE_CONFIGS,
        "show_progress": False,
    }
    for band_name, bands in BAND_CONFIGS.items()
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

FS_K = [50, 100, 250]

FEATURE_SELECTION_PARAMS = [
    # All fused features: only remove constants.
    {
        "method": "variance",
        "config_label": "all_standard",
        "params": {
            "threshold": 0.0,
            "post_scaler": "standard",
        },
    },

    # Supervised univariate selection.
    *[
        {
            "method": "anova",
            "config_label": f"anova_{k}_standard",
            "params": {
                "k": k,
                "post_scaler": "standard",
            },
        }
        for k in FS_K
    ],

    # Nonlinear feature relevance.
    *[
        {
            "method": "mutual_information",
            "config_label": f"mi_{k}_standard",
            "params": {
                "k": k,
                "post_scaler": "standard",
            },
        }
        for k in FS_K
    ],
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

MLP_SMALL_PARAMS = {
    "hidden_dims": (64, 32),
    "activation": "relu",
    "dropout": 0.2,
    "batch_norm": True,
}

MLP_MEDIUM_PARAMS = {
    "hidden_dims": (128, 64, 32),
    "activation": "relu",
    "dropout": 0.2,
    "batch_norm": True,
}

MLP_LARGE_PARAMS = {
    "hidden_dims": (256, 128, 64),
    "activation": "relu",
    "dropout": 0.2,
    "batch_norm": True,
}

_NEURAL_BASE_PARAMS = {
    "epochs": 500,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "optimizer": "adam",
    "device": "mps",
    "seed": 0,
}


# ============================================================
# Training
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
        "name": "mlp_small",
        "learning": "neural_erm__mlp_small",
        "model": "mlp",
        "model_params": MLP_SMALL_PARAMS,
        "training_params": {**_NEURAL_BASE_PARAMS},
    },
    {
        "name": "mlp_medium",
        "learning": "neural_erm__mlp_medium",
        "model": "mlp",
        "model_params": MLP_MEDIUM_PARAMS,
        "training_params": {**_NEURAL_BASE_PARAMS},
    },
    {
        "name": "mlp_large",
        "learning": "neural_erm__mlp_large",
        "model": "mlp",
        "model_params": MLP_LARGE_PARAMS,
        "training_params": {**_NEURAL_BASE_PARAMS},
    },
]


# ============================================================
# Evaluation
# ============================================================

MODEL_EVALUATION_PARAMS = {}


# ============================================================
# Benchmark
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
            "learning_method": "neural_erm__mlp_small",
            "model_name": "mlp",
            "regime": "Neural",
            "method": "MLP Small",
        },
        {
            "learning_method": "neural_erm__mlp_medium",
            "model_name": "mlp",
            "regime": "Neural",
            "method": "MLP Medium",
        },
        {
            "learning_method": "neural_erm__mlp_large",
            "model_name": "mlp",
            "regime": "Neural",
            "method": "MLP Large",
        },
    ],
    "tables": [
        {
            "name": "intra_all_features",
            "scenario": "intra_subject",
            "setting_column": "Dataset",
            "output_name": "intra_all_features_table.csv",
            "include_discrepancy": False,
            "filters": {"target_fraction": 0.8},
        },
    ],
}