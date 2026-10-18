# main/feature_grid/params/intra_subject/params_test.py

EXECUTION_PARAMS = {"max_workers": 1}

COMMON_CLASSES = [
    "left_hand_imagery", "right_hand_imagery",
    "both_feet_imagery", "tongue_imagery",
]

CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

PREPROCESSING_PARAMS = [{
    "dataset": "bci2a",
    "root_gdf": "datasets/bci2a/gdf",
    "root_mat": "datasets/bci2a/mat",
    "name": "bci2a_intra_test",
    "representation": "signal",
    "loader": {
        "subjects": [1, 2],
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
        "resample": {"enabled": False},
    },
    "show_progress": False,
}]

SCENARIO = "intra_subject"
SCENARIO_PARAMS = {"train_fraction": 0.8, "seed": 0}


# ============================================================
# Representation
# ============================================================

FEATURE_EXTRACTION_PARAMS = [
    {
        "name": "logvar",
        "method": "handcrafted",
        "params": {"logvar": {}},
    },
    {
        "name": "csp_4",
        "method": "csp",
        "params": {"n_components": 4, "reg": 1e-6},
    },
    {
        "name": "logvar_csp",
        "extractors": [
            {"method": "handcrafted", "params": {"logvar": {}}},
            {"method": "csp", "params": {"n_components": 4, "reg": 1e-6}},
        ],
    },
]

FEATURE_SELECTION_PARAMS = [{
    "name": "all_standard",
    "method": "variance",
    "params": {"threshold": 0.0, "post_scaler": "standard"},
}]

SIGNAL_TRANSFORM_PARAMS = [{
    "name": "standardized_signal",
    "method": "standardize_signal",
    "params": {"mode": "channel", "scale": 1e6},
}]


# ============================================================
# Models
# ============================================================

TRAINING_PARAMS = [
    {
        "name": "logistic_regression",
        "learning": "sklearn_erm",
        "model": "logistic_regression",
        "model_params": {"C": 1.0, "max_iter": 1000},
        "training_params": {},
    },
    {
        "name": "eegnet",
        "learning": "neural_erm__eegnet",
        "model": "eegnet",
        "model_params": {
            "F1": 16, "D": 2, "F2": 32,
            "kernel_length": 64,
            "drop_prob": 0.25,
            "pool_mode": "mean",
        },
        "training_params": {
            "epochs": 1,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "optimizer": "adam",
            "device": "mps",
            "seed": 0,
        },
    },
]


# ============================================================
# Evaluation / benchmark
# ============================================================

MODEL_EVALUATION_PARAMS = {}

BENCHMARK_TABLES_PARAMS = {
    "method_display": [
        {
            "learning_method": "sklearn_erm",
            "model_name": "logistic_regression",
            "regime": "Classical",
            "method": "Logistic Regression",
        },
        {
            "learning_method": "neural_erm__eegnet",
            "model_name": "eegnet",
            "regime": "Deep",
            "method": "EEGNet",
        },
    ],
    "tables": [{
        "name": "intra_subject_test",
        "scenario": "intra_subject",
        "setting_column": "Dataset",
        "output_name": "intra_subject_test_table.csv",
        "include_discrepancy": False,
        "filters": {"target_fraction": 0.8},
    }],
}