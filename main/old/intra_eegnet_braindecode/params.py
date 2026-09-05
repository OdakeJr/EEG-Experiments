# ============================================================
# Intra-subject EEGNet development experiment
# ============================================================

# Goal:
#   Evaluate EEGNet as a modern EEG-specific deep baseline
#   directly from preprocessed EEG signals.
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
# Intra-subject EEGNet development experiment
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
# Preprocessing
# ============================================================

PREPROCESSING_PARAMS = [
    {
        "dataset": "bci2a",
        "root_gdf": "datasets/bci2a/gdf",
        "root_mat": "datasets/bci2a/mat",
        "name": "bci2a_intra_eegnet_braindecode",
        "representation": "signal",

        "loader": {
            "channels": CHANNELS,
            "classes": COMMON_CLASSES,
        },

        "continuous_preprocessing": {
            "scale": 1e6,
            "bandpass": {
                "enabled": True,
                "l_freq": 4.0,
                "h_freq": 38.0,
            },
            "exponential_standardize": {
                "enabled": True,
                "factor_new": 1e-3,
                "init_block_size": 1000,
                "eps": 1e-4,
            },
        },

        "filter": {
            "bandpass": {
                "enabled": False,
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

SCENARIO = "intra_subject"

SCENARIO_PARAMS = {
    "train_fraction": 0.8,
    "seed": 0,
}


# ============================================================
# Feature transformation
# ============================================================

FEATURE_SELECTION_PARAMS = [
    {
        "method": "identity_signal",
        "config_label": "identity_signal",
        "params": {},
    },
]


# ============================================================
# EEGNet
# ============================================================

EEGNET_PARAMS = {
    "F1": 8,
    "D": 2,
    "F2": 16,
    "kernel_length": 64,
    "drop_prob": 0.25,
    "pool_mode": "mean",
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
            "name": "intra_subject_eegnet",
            "scenario": "intra_subject",
            "setting_column": "Dataset",
            "output_name": "intra_subject_eegnet_table.csv",
            "include_discrepancy": False,
            "filters": {
                "target_fraction": 0.8,
            },
        },
    ],
}