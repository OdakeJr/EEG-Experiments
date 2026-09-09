# ============================================================
# Cross-subject EEGNet development experiment
# ============================================================

# Goal:
#   Evaluate EEGNet + ERM under subject distribution shift,
#   using a preprocessing setup aligned with published
#   cross-subject BCI Competition IV 2a experiments.
#
# Dataset:
#       BCI Competition IV 2a
#
# Scenario:
#       Cross-subject
#
# Protocol:
#       Leave one subject out as target.
#       Train ERM on all remaining source subjects.
#
#       100% of target samples are available as unlabeled
#       calibration data. ERM does not use these samples,
#       but later UDA methods can use them under the same
#       experimental protocol.


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
        "name": "bci2a_cross_subject_eegnet_1_38_250hz",
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
    "F1": 8,
    "D": 2,
    "F2": 16,
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
    #"device": "cpu",
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
        "name": "eegnet_canonical_low_dropout",
        "learning": "neural_erm__eegnet_canonical_low_dropout",
        "model": "eegnet",
        "model_params": EEGNET_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
        },
    },
    {
        "name": "eegnet_deep_coral",
        "learning": "deep_coral",
        "model": "eegnet",
        "model_params": EEGNET_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "coral_lambda": 1.0,
        },
    },
    {
        "name": "eegnet_dann",
        "learning": "dann",
        "model": "eegnet",
        "model_params": EEGNET_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "dann_lambda": 1.0,
            "domain_hidden_dim": 64,
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
            "learning_method": "neural_erm__eegnet_canonical_low_dropout",
            "model_name": "eegnet",
            "regime": "ERM",
            "method": "EEGNet ERM",
        },
        {
            "learning_method": "deep_coral",
            "model_name": "eegnet",
            "regime": "UDA",
            "method": "Deep CORAL",
        },
        {
            "learning_method": "dann",
            "model_name": "eegnet",
            "regime": "UDA",
            "method": "DANN",
        },
    ],
    "tables": [
        {
            "name": "cross_subject_eegnet",
            "scenario": "cross_subject",
            "setting_column": "Dataset",
            "output_name": "cross_subject_eegnet_table.csv",
            "include_discrepancy": False,
            "filters": {
                "target_fraction": 1.0,
                "n_target_super_domains": 0,
                "use_max_source_domains": True,
            },
        },
    ],
}