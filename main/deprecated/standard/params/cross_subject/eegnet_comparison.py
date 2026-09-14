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
    "device": "cpu",
    #"device": "mps",
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
        "training_params": {**_NEURAL_BASE_PARAMS},
    },

    # --------------------------------------------------------
    # DANN
    # --------------------------------------------------------

    {
        "name": "eegnet_dann_0_1",
        "learning": "dann__progressive_0_1",
        "model": "eegnet",
        "model_params": EEGNET_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "dann_lambda": 0.1,
            "dann_gamma": 10.0,
            "domain_hidden_dim": 64,
        },
    },
    {
        "name": "eegnet_dann_0_3",
        "learning": "dann__progressive_0_3",
        "model": "eegnet",
        "model_params": EEGNET_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "dann_lambda": 0.3,
            "dann_gamma": 10.0,
            "domain_hidden_dim": 64,
        },
    },
    {
        "name": "eegnet_dann_1_0",
        "learning": "dann__progressive_1_0",
        "model": "eegnet",
        "model_params": EEGNET_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "dann_lambda": 1.0,
            "dann_gamma": 10.0,
            "domain_hidden_dim": 64,
        },
    },

    # --------------------------------------------------------
    # Deep CORAL
    # --------------------------------------------------------

    {
        "name": "eegnet_coral_0_1",
        "learning": "deep_coral__single_forward_0_1",
        "model": "eegnet",
        "model_params": EEGNET_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "coral_lambda": 0.1,
        },
    },
    {
        "name": "eegnet_coral_1_0",
        "learning": "deep_coral__single_forward_1_0",
        "model": "eegnet",
        "model_params": EEGNET_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "coral_lambda": 1.0,
        },
    },
    {
        "name": "eegnet_coral_10_0",
        "learning": "deep_coral__single_forward_10_0",
        "model": "eegnet",
        "model_params": EEGNET_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "coral_lambda": 10.0,
        },
    },

    # --------------------------------------------------------
    # Deep MMD
    # --------------------------------------------------------

    {
        "name": "eegnet_mmd_0_1",
        "learning": "deep_mmd__lambda_0_1",
        "model": "eegnet",
        "model_params": EEGNET_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "mmd_lambda": 0.1,
            "gamma": None,
        },
    },
    {
        "name": "eegnet_mmd_1_0",
        "learning": "deep_mmd__lambda_1_0",
        "model": "eegnet",
        "model_params": EEGNET_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "mmd_lambda": 1.0,
            "gamma": None,
        },
    },
    {
        "name": "eegnet_mmd_10_0",
        "learning": "deep_mmd__lambda_10_0",
        "model": "eegnet",
        "model_params": EEGNET_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "mmd_lambda": 10.0,
            "gamma": None,
        },
    },

    # --------------------------------------------------------
    # MCD
    # --------------------------------------------------------

    {
        "name": "eegnet_mcd_0_1",
        "learning": "mcd__lambda_0_1_g4",
        "model": "eegnet",
        "model_params": EEGNET_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "mcd_lambda": 0.1,
            "generator_steps": 4,
        },
    },
    {
        "name": "eegnet_mcd_0_5",
        "learning": "mcd__lambda_0_5_g4",
        "model": "eegnet",
        "model_params": EEGNET_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "mcd_lambda": 0.5,
            "generator_steps": 4,
        },
    },
    {
        "name": "eegnet_mcd_1_0",
        "learning": "mcd__lambda_1_0_g4",
        "model": "eegnet",
        "model_params": EEGNET_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "mcd_lambda": 1.0,
            "generator_steps": 4,
        },
    },
]


# ============================================================
# Evaluation
# ============================================================

MODEL_EVALUATION_PARAMS = {
    "device": "auto",
}


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

        # DANN
        {
            "learning_method": "dann__progressive_0_1",
            "model_name": "eegnet",
            "regime": "UDA",
            "method": "DANN λ=0.1",
        },
        {
            "learning_method": "dann__progressive_0_3",
            "model_name": "eegnet",
            "regime": "UDA",
            "method": "DANN λ=0.3",
        },
        {
            "learning_method": "dann__progressive_1_0",
            "model_name": "eegnet",
            "regime": "UDA",
            "method": "DANN λ=1.0",
        },

        # Deep CORAL
        {
            "learning_method": "deep_coral__single_forward_0_1",
            "model_name": "eegnet",
            "regime": "UDA",
            "method": "Deep CORAL λ=0.1",
        },
        {
            "learning_method": "deep_coral__single_forward_1_0",
            "model_name": "eegnet",
            "regime": "UDA",
            "method": "Deep CORAL λ=1.0",
        },
        {
            "learning_method": "deep_coral__single_forward_10_0",
            "model_name": "eegnet",
            "regime": "UDA",
            "method": "Deep CORAL λ=10",
        },

        # Deep MMD
        {
            "learning_method": "deep_mmd__lambda_0_1",
            "model_name": "eegnet",
            "regime": "UDA",
            "method": "Deep MMD λ=0.1",
        },
        {
            "learning_method": "deep_mmd__lambda_1_0",
            "model_name": "eegnet",
            "regime": "UDA",
            "method": "Deep MMD λ=1.0",
        },
        {
            "learning_method": "deep_mmd__lambda_10_0",
            "model_name": "eegnet",
            "regime": "UDA",
            "method": "Deep MMD λ=10",
        },

        # MCD
        {
            "learning_method": "mcd__lambda_0_1_g4",
            "model_name": "eegnet",
            "regime": "UDA",
            "method": "MCD λ=0.1",
        },
        {
            "learning_method": "mcd__lambda_0_5_g4",
            "model_name": "eegnet",
            "regime": "UDA",
            "method": "MCD λ=0.5",
        },
        {
            "learning_method": "mcd__lambda_1_0_g4",
            "model_name": "eegnet",
            "regime": "UDA",
            "method": "MCD λ=1.0",
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