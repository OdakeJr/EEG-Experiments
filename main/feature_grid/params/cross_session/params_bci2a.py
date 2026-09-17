# main/feature_grid/params/cross_session/params.py

EXECUTION_PARAMS = {"max_workers": 2}

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
    "broad_1_38": [(1, 38)],
    # "delta_1_4": [(1, 4)],
    "theta_4_8": [(4, 8)],
    # "alpha_8_12": [(8, 12)],
    # "beta_12_30": [(12, 30)],
    # "gamma_30_38": [(30, 38)],
}

PREPROCESSING_PARAMS = [
    {
        "dataset": "bci2a",
        "root_gdf": "datasets/bci2a/gdf",
        "root_mat": "datasets/bci2a/mat",
        "name": f"bci2a_cross_session_{band_name}",
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
                "bands": bands,
                "order": 5,
                "stack_bands": True,
            },
            "resample": {"enabled": False},
        },
        "show_progress": False,
    }
    for band_name, bands in BAND_CONFIGS.items()
]

SCENARIO = "cross_session"

SCENARIO_PARAMS = {
    "source_counts": ["all"],
    "target_fractions": [0.0],
    "seed": 0,
}


# ============================================================
# Feature extraction: signal -> features
# ============================================================

STATISTICAL_FEATURES = {
    # "mean": {},
    # "std": {},
    # "var": {},
    "logvar": {},
    # "skew": {},
    # "kurtosis": {},
    # "min": {},
    # "max": {},
    # "rms": {},
    # "ptp": {},
}

TEMPORAL_FEATURES = {
    "line_length": {},
    # "hjorth_activity": {},
    # "hjorth_mobility": {},
    # "hjorth_complexity": {},
    # "zero_crossing": {},
    # "ar": {},
}

SPECTRAL_FEATURES = {
    "bandpower": {"sfreq": 250.0},
    # "relative_bandpower": {"sfreq": 250.0, "total_band": (1, 38)},
    # "psd_stats": {"sfreq": 250.0, "band": (1, 38)},
    # "spectral_entropy": {"sfreq": 250.0, "band": (1, 38)},
    # "differential_entropy": {},
}

NONLINEAR_FEATURES = {
    # "sample_entropy": {},
    # "permutation_entropy": {},
    # "higuchi_fd": {},
    # "petrosian_fd": {},
}

WAVELET_FEATURES = {
    # "wavelet_energy": {},
    # "wavelet_entropy": {},
}

ALL_HANDCRAFTED_FEATURES = {
    **STATISTICAL_FEATURES,
    **TEMPORAL_FEATURES,
    **SPECTRAL_FEATURES,
    **NONLINEAR_FEATURES,
    "cov": {},
    # "logcov": {},
    # "eig": {},
    **WAVELET_FEATURES,
}

FEATURE_EXTRACTION_PARAMS = [
    {
        "name": "statistical",
        "method": "handcrafted",
        "params": STATISTICAL_FEATURES,
    },
    # {
    #     "name": "temporal",
    #     "method": "handcrafted",
    #     "params": TEMPORAL_FEATURES,
    # },
    # {
    #     "name": "spectral",
    #     "method": "handcrafted",
    #     "params": SPECTRAL_FEATURES,
    # },
    # {
    #     "name": "nonlinear",
    #     "method": "handcrafted",
    #     "params": NONLINEAR_FEATURES,
    # },
    # {
    #     "name": "wavelet",
    #     "method": "handcrafted",
    #     "params": WAVELET_FEATURES,
    # },
    # {
    #     "name": "cov",
    #     "method": "handcrafted",
    #     "params": {"cov": {}},
    # },
    # {
    #     "name": "logcov",
    #     "method": "handcrafted",
    #     "params": {"logcov": {}},
    # },
    # {
    #     "name": "eig",
    #     "method": "handcrafted",
    #     "params": {"eig": {}},
    # },
    # {
    #     "name": "csp_4",
    #     "method": "csp",
    #     "params": {"n_components": 4, "reg": 1e-6},
    # },
    {
        "name": "csp_6",
        "method": "csp",
        "params": {"n_components": 6, "reg": 1e-6},
    },
    # {
    #     "name": "csp_8",
    #     "method": "csp",
    #     "params": {"n_components": 8, "reg": 1e-6},
    # },
    # {
    #     "name": "rcsp_4",
    #     "method": "rcsp",
    #     "params": {"n_components": 4, "alpha": 0.1, "reg": 1e-6},
    # },
    # {
    #     "name": "rcsp_6",
    #     "method": "rcsp",
    #     "params": {"n_components": 6, "alpha": 0.1, "reg": 1e-6},
    # },
    # {
    #     "name": "rcsp_8",
    #     "method": "rcsp",
    #     "params": {"n_components": 8, "alpha": 0.1, "reg": 1e-6},
    # },
    # {
    #     "name": "riemann",
    #     "method": "riemann",
    #     "params": {"reg": 1e-6},
    # },
    {
        "name": "all_fused",
        "extractors": [
            {
                "method": "handcrafted",
                "params": ALL_HANDCRAFTED_FEATURES,
            },
            {
                "method": "csp",
                "params": {"n_components": 6, "reg": 1e-6},
            },
            # {
            #     "method": "rcsp",
            #     "params": {"n_components": 6, "alpha": 0.1, "reg": 1e-6},
            # },
            # {
            #     "method": "riemann",
            #     "params": {"reg": 1e-6},
            # },
        ],
    },
]


# ============================================================
# Feature selection: features -> features
# ============================================================

FEATURE_SELECTION_PARAMS = [
    {
        "name": "all_standard",
        "method": "variance",
        "params": {
            "threshold": 0.0,
            "post_scaler": "standard",
        },
    },
]


# ============================================================
# Deep representation: signal -> signal
# ============================================================

SIGNAL_TRANSFORM_PARAMS = [
    {
        "name": "standardized_signal",
        "method": "standardize_signal",
        "params": {
            "mode": "channel",
            "scale": 1e6,
        },
    },
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

EEGNET_PARAMS = {
    "F1": 16,
    "D": 2,
    "F2": 32,
    "kernel_length": 64,
    "drop_prob": 0.25,
    "pool_mode": "mean",
}

NEURAL_PARAMS = {
    "epochs": 1,  # 300 final
    "batch_size": 64,
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
    # {
    #     "name": "svm",
    #     "learning": "sklearn_erm",
    #     "model": "svm",
    #     "model_params": SVM_PARAMS,
    #     "training_params": {},
    # },
    # {
    #     "name": "random_forest",
    #     "learning": "sklearn_erm",
    #     "model": "random_forest",
    #     "model_params": RANDOM_FOREST_PARAMS,
    #     "training_params": {},
    # },
    {
        "name": "mlp_small",
        "learning": "neural_erm__mlp_small",
        "model": "mlp",
        "model_params": MLP_SMALL_PARAMS,
        "training_params": NEURAL_PARAMS,
    },
    # {
    #     "name": "mlp_medium",
    #     "learning": "neural_erm__mlp_medium",
    #     "model": "mlp",
    #     "model_params": MLP_MEDIUM_PARAMS,
    #     "training_params": NEURAL_PARAMS,
    # },
    # {
    #     "name": "mlp_large",
    #     "learning": "neural_erm__mlp_large",
    #     "model": "mlp",
    #     "model_params": MLP_LARGE_PARAMS,
    #     "training_params": NEURAL_PARAMS,
    # },
    {
        "name": "eegnet",
        "learning": "neural_erm__eegnet",
        "model": "eegnet",
        "model_params": EEGNET_PARAMS,
        "training_params": NEURAL_PARAMS,
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
        {
            "learning_method": "neural_erm__eegnet",
            "model_name": "eegnet",
            "regime": "Deep",
            "method": "EEGNet",
        },
    ],
    "tables": [
        {
            "name": "cross_session_paper1",
            "scenario": "cross_session",
            "setting_column": "Dataset",
            "output_name": "cross_session_paper1_table.csv",
            "include_discrepancy": False,
            "filters": {
                "n_target_super_domains": 0,
                "target_fraction": 0.0,
                "use_max_source_domains": True,
            },
        },
    ],
}


# ============================================================
# Paper 1 analysis
# ============================================================

PAPER_ANALYSIS_PARAMS = {
    "name": "paper1",
    "collection": "paper1",
    "required_scenarios": [
        "intra_subject",
        "cross_session",
        "cross_subject",
    ],
}