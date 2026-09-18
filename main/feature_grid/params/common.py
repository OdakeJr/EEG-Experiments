# main/feature_grid/params/common.py

# ============================================================
# Execution
# ============================================================

EXECUTION_PARAMS = {"max_workers": 1}

SPECTRAL_BAND = (1, 38)

BAND_CONFIGS = {
    "broad_1_38": [(1, 38)],
    "delta_1_4": [(1, 4)],
    "theta_4_8": [(4, 8)],
    #"alpha_8_12": [(8, 12)],
    #"beta_12_30": [(12, 30)],
    #"gamma_30_38": [(30, 38)],
}
# ============================================================
# Feature extraction
# ============================================================

STATISTICAL_FEATURES = {
    "logvar": {},
    "skew": {},
    "kurtosis": {},
}

TEMPORAL_FEATURES = {
    "line_length": {},
    "hjorth_mobility": {},
    "hjorth_complexity": {},
}

NONLINEAR_FEATURES = {
    "sample_entropy": {},
    "permutation_entropy": {},
    "higuchi_fd": {},
}

WAVELET_FEATURES = {
    "wavelet_energy": {},
    "wavelet_entropy": {},
}

COVARIANCE_FEATURES = {
    "logcov": {},
}

CSP_CONFIGS = {
    "csp_6": {"n_components": 6, "reg": 1e-6},
}

RIEMANN_CONFIGS = {
    "riemann": {"reg": 1e-6},
}


# ============================================================
# Active representations
# ============================================================

ACTIVE_REPRESENTATIONS = {
    "statistical": True,
    "temporal": True,
    "spectral": False,
    "nonlinear": False,
    "wavelet": False,
    "covariance": False,
    "csp": True,
    "riemann": False,
    "all_fused": False,
}


def make_feature_extraction_params(sfreq):
    spectral_features = {
        "bandpower": {"sfreq": sfreq},
        "spectral_entropy": {"sfreq": sfreq, "band": SPECTRAL_BAND},
    }

    handcrafted_families = {
        "statistical": STATISTICAL_FEATURES,
        "temporal": TEMPORAL_FEATURES,
        "spectral": spectral_features,
        "nonlinear": NONLINEAR_FEATURES,
        "wavelet": WAVELET_FEATURES,
        "covariance": COVARIANCE_FEATURES,
    }

    configs = []

    for name, features in handcrafted_families.items():
        if ACTIVE_REPRESENTATIONS[name]:
            configs.append({
                "name": name,
                "method": "handcrafted",
                "params": features,
            })

    if ACTIVE_REPRESENTATIONS["csp"]:
        configs += [
            {"name": name, "method": "csp", "params": params}
            for name, params in CSP_CONFIGS.items()
        ]

    if ACTIVE_REPRESENTATIONS["riemann"]:
        configs += [
            {"name": name, "method": "riemann", "params": params}
            for name, params in RIEMANN_CONFIGS.items()
        ]

    if ACTIVE_REPRESENTATIONS["all_fused"]:
        fused_handcrafted = {
            feature: params
            for family, features in handcrafted_families.items()
            if ACTIVE_REPRESENTATIONS[family]
            for feature, params in features.items()
        }

        extractors = []

        if fused_handcrafted:
            extractors.append({
                "method": "handcrafted",
                "params": fused_handcrafted,
            })

        if ACTIVE_REPRESENTATIONS["csp"]:
            extractors += [
                {"method": "csp", "params": params}
                for params in CSP_CONFIGS.values()
            ]

        if ACTIVE_REPRESENTATIONS["riemann"]:
            extractors += [
                {"method": "riemann", "params": params}
                for params in RIEMANN_CONFIGS.values()
            ]

        configs.append({
            "name": "all_fused",
            "extractors": extractors,
        })

    return configs


# ============================================================
# Feature selection
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
# Signal representation
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

SHALLOW_FBCSP_PARAMS = {
    "n_filters_time": 40,
    "filter_time_length": 25,
    "n_filters_spat": 40,
    "pool_time_length": 75,
    "pool_time_stride": 15,
    "final_conv_length": "auto",
    "pool_mode": "mean",
    "batch_norm": True,
    "drop_prob": 0.5,
}

EEG_TCNET_PARAMS = {
    "depth_multiplier": 2,
    "filter_1": 8,
    "kern_length": 64,
    "depth": 2,
    "kernel_size": 4,
    "filters": 12,
    "max_norm_const": 0.25,
    "drop_prob_eeg": 0.2,
    "drop_prob_tcn": 0.3,
    "tcn_batch_norm": True,
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

TRAINING_PARAMS_SMOKE = [
    {
        "name": "logistic_regression",
        "learning": "sklearn_erm",
        "model": "logistic_regression",
        "model_params": LOGISTIC_REGRESSION_PARAMS,
        "training_params": {},
    },
    {
        "name": "eegnet",
        "learning": "neural_erm__eegnet",
        "model": "eegnet",
        "model_params": EEGNET_PARAMS,
        "training_params": NEURAL_PARAMS,
    },
]

TRAINING_PARAMS_FULL = [
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
        "training_params": NEURAL_PARAMS,
    },
    {
        "name": "mlp_medium",
        "learning": "neural_erm__mlp_medium",
        "model": "mlp",
        "model_params": MLP_MEDIUM_PARAMS,
        "training_params": NEURAL_PARAMS,
    },
    {
        "name": "mlp_large",
        "learning": "neural_erm__mlp_large",
        "model": "mlp",
        "model_params": MLP_LARGE_PARAMS,
        "training_params": NEURAL_PARAMS,
    },
    {
        "name": "eegnet",
        "learning": "neural_erm__eegnet",
        "model": "eegnet",
        "model_params": EEGNET_PARAMS,
        "training_params": NEURAL_PARAMS,
    },
    {
        "name": "shallow_fbcsp",
        "learning": "neural_erm__shallow_fbcsp",
        "model": "shallow_fbcsp",
        "model_params": SHALLOW_FBCSP_PARAMS,
        "training_params": NEURAL_PARAMS,
    },
    {
        "name": "eeg_tcnet",
        "learning": "neural_erm__eeg_tcnet",
        "model": "eeg_tcnet",
        "model_params": EEG_TCNET_PARAMS,
        "training_params": NEURAL_PARAMS,
    },
]

# Change only this line when moving from smoke test to final experiments.
TRAINING_PARAMS = TRAINING_PARAMS_SMOKE


# ============================================================
# Evaluation
# ============================================================

MODEL_EVALUATION_PARAMS = {}


# ============================================================
# Benchmark display
# ============================================================

METHOD_DISPLAY = [
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
    {
        "learning_method": "neural_erm__shallow_fbcsp",
        "model_name": "shallow_fbcsp",
        "regime": "Deep",
        "method": "ShallowFBCSPNet",
    },
    {
        "learning_method": "neural_erm__eeg_tcnet",
        "model_name": "eeg_tcnet",
        "regime": "Deep",
        "method": "EEG-TCNet",
    },
]


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