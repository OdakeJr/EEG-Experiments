
# main/domain_generalization/params/common.py

# ============================================================
# Execution
# ============================================================

EXECUTION_PARAMS = {
    "max_workers": 8,
    "evaluation_workers": 4,
}


# ============================================================
# Frequency bands
# ============================================================

BAND_CONFIGS = {
    "broad_1_38": [(1, 38)],
}


# ============================================================
# Feature extraction and selection
# ============================================================

def make_feature_extraction_params(sfreq):
    return []


FEATURE_SELECTION_PARAMS = []


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

EEGNET_PARAMS = {
    "F1": 16,
    "D": 2,
    "F2": 32,
    "kernel_length": 64,
    "drop_prob": 0.25,
    "pool_mode": "mean",
}

"""
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
"""

"""
NEURAL_PARAMS = {
    "epochs": 300,  # 300 final
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "optimizer": "adam",
    "device": "cpu",
    "seed": 0,
}
"""

NEURAL_PARAMS = {
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "optimizer": "adam",
    "device": "cpu",
    "seed": 0,
    "validation_fraction": 0.1,
    "patience": 20,
    "min_delta": 0.0,
}

# ============================================================
# Training
# ============================================================

_TRAIN_EEGNET = {
    "name": "eegnet_erm",
    "learning": "neural_erm__eegnet",
    "model": "eegnet",
    "model_params": EEGNET_PARAMS,
    "training_params": NEURAL_PARAMS,
}

DG_METHODS = {
    "irm": {"irm_lambda": 100.0},
    "vrex": {"vrex_lambda": 10.0},
    "groupdro": {"eta": 0.01},
    "coral": {"coral_lambda": 1.0},
     "mmd": {"mmd_lambda": 1.0},
     "dg_dann": {"dann_lambda": 1.0, "dann_gamma": 10.0},
     "mldg": {"mldg_alpha": 1e-3, "mldg_beta": 1.0},
    #"residual_moments__linear": {"moment_lambda": 1.0, "moment_type": "linear", "normalize_features": True},
    #"residual_moments__quadratic": {"moment_lambda": 1.0, "moment_type": "quadratic", "normalize_features": True},
}

TRAINING_PARAMS = [
    _TRAIN_EEGNET,
    *[
        {
            "name": f"eegnet_{method}",
            "learning": f"{method}__eegnet",
            "model": "eegnet",
            "model_params": EEGNET_PARAMS,
            "training_params": {
                **NEURAL_PARAMS,
                **method_params,
            },
        }
        for method, method_params in DG_METHODS.items()
    ],
]

# ============================================================
# Evaluation
# ============================================================

MODEL_EVALUATION_PARAMS = {}


# ============================================================
# Benchmark display
# ============================================================

DG_LABELS = {
    "irm": "IRM",
    "vrex": "VREx",
    "groupdro": "GroupDRO",
    "coral": "CORAL",
    "mmd": "MMD",
    "dg_dann": "DG-DANN",
    "mldg": "MLDG",
    
    "residual_moments__linear": "Residual Moments (Linear)",
    "residual_moments__quadratic": "Residual Moments (Quadratic)",
}

METHOD_DISPLAY = [
    {
        "learning_method": "neural_erm__eegnet",
        "model_name": "eegnet",
        "regime": "Baseline",
        "method": "ERM",
    },
    *[
        {
            "learning_method": f"{method}__eegnet",
            "model_name": "eegnet",
            "regime": "Domain Generalization",
            "method": DG_LABELS[method],
        }
        for method in DG_METHODS
    ],
]


# ============================================================
# Benchmark
# ============================================================

BENCHMARK_TABLES_PARAMS = {
    "method_display": METHOD_DISPLAY,
    "tables": [
        {
            "name": "cross_subject_paper2",
            "scenario": "cross_subject",
            "setting_column": "Dataset",
            "output_name": "cross_subject_paper2_table.csv",
            "include_discrepancy": False,
            "filters": {"target_fraction": 0.0},
        },
    ],
}


# ============================================================
# Paper 2 analysis
# ============================================================

PAPER2_ANALYSIS_PARAMS = {}
PAPER_ANALYSIS_PARAMS = PAPER2_ANALYSIS_PARAMS