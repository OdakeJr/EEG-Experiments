# ============================================================
# Cross-session EEGNet development experiment
# ============================================================

# Goal:
#   Evaluate EEGNet as a modern EEG-specific deep baseline
#   directly from preprocessed EEG signals.
#
# Dataset:
#       BCI Competition IV 2a
#
# Scenario:
#       Cross-session
#
# Protocol:
#       Train on one session and evaluate on the other session
#       within the same subject.
#
#       With two sessions, both directions are generated:
#           session_01 -> session_02
#           session_02 -> session_01


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
        "name": "bci2a_cross_session_eegnet_4_38",
        "representation": "signal",
        "loader": {
            "channels": CHANNELS,
            "classes": COMMON_CLASSES,
        },
        "filter": {
            "bandpass": {
                "enabled": True,
                "bands": [(4, 38)],
                "order": 5,
                "stack_bands": True,
            },
            "resample": {
                "enabled": True,
                "new_fs": 128.0,
            },
        },
        "show_progress": False,
    },
]


# ============================================================
# Scenario
# ============================================================

SCENARIO = "cross_session"

SCENARIO_PARAMS = {
    "source_counts": ["all"],
    "target_fractions": [0.0],
    "seed": 0,
}


# ============================================================
# Feature transformation
# ============================================================

FEATURE_SELECTION_PARAMS = [
    {
        "method": "standardize_signal",
        "config_label": "channel_standard",
        "params": {
            "mode": "channel",
            "scale": 1e6,
        },
    },
]


# ============================================================
# EEGNet
# ============================================================

EEGNET_PARAMS = {
    "F1": 16,
    "D": 2,
    "F2": 32,
    "kernel_length": 64,
    "drop_prob": 0.25,
    "pool_mode": "mean",
}

_NEURAL_BASE_PARAMS = {
    "epochs": 300,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "optimizer": "adam",
    # "device": "cpu",
    "device": "mps",
    "seed": 0,
    "validation_fraction": 0.0,
    "patience": 20,
}


# ============================================================
# Training
# ============================================================

TRAINING_PARAMS = [
    {
        "name": "eegnet_erm",
        "learning": "neural_erm",
        "model": "eegnet",
        "model_params": EEGNET_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
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
            "learning_method": "neural_erm",
            "model_name": "eegnet",
            "regime": "Deep Learning",
            "method": "EEGNet",
        },
    ],
    "tables": [
        {
            "name": "cross_session_eegnet",
            "scenario": "cross_session",
            "setting_column": "Dataset",
            "output_name": "cross_session_eegnet_table.csv",
            "include_discrepancy": False,
            "filters": {
                "target_fraction": 0.0,
            },
        },
    ],
}