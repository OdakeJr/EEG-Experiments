# ============================================================
# Development experiment
# ============================================================

# Goal:
#   Fast initial comparison of:
#       - ERM
#       - Positive-Negative Learning
#       - Standard Importance Weighting
#       - Structural Importance Weighting V1
#
# Dataset:
#   BCI Competition IV 2a
#
# Scenario:
#   Cross-subject adaptation
#
# Target protocol:
#   100% of target samples available unlabeled as calibration data.


# ============================================================
# Preprocessing
# ============================================================

COMMON_CLASSES = [
    "left_hand_imagery",
    "right_hand_imagery",
    "both_feet_imagery",
]


FILTER_CONFIG = {
    "bandpass": {
        "bands": [
            (8, 12),
            (13, 30),
        ],
    },
}


FEATURE_CONFIG = {
    "logvar": {},
    "cov": {},
}


CHANNELS = [
    "C3",
    "Cz",
    "C4",
]


PREPROCESSING_PARAMS = [
    {
        "dataset": "bci2a",

        "root_gdf": "data/bci2a/gdf",
        "root_mat": "data/bci2a/mat",

        "name": "bci2a_dev",

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

SCENARIO = "cross_subject"


SCENARIO_PARAMS = {
    "source_counts": {
        # Use all remaining subjects as sources.
        # This gives IW methods enough data for the first test.
        "bci_iv_2a": [
            "all",
        ],
    },

    # Entire target domain is available without labels.
    "target_fractions": [
        1.0,
    ],

    "max_source_combinations": 1,

    "seed": 0,
}


# ============================================================
# Feature selection
# ============================================================

# For the first method test, avoid introducing feature-selection
# variability. Keep the full extracted representation.

FEATURE_SELECTION_PARAMS = [
    {
        "method": "random",

        "config_label": "all_features",

        "params": {
            "ratio": 1.0,
            "seed": 0,
        },
    }
]


# ============================================================
# Model
# ============================================================

MLP_PARAMS = {
    "hidden_dims": (
        128,
        64,
    ),
}


# ============================================================
# Shared training parameters
# ============================================================

_NEURAL_BASE_PARAMS = {
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "device": "cpu",
    "seed": 0,
}


# ============================================================
# Training methods
# ============================================================

TRAINING_PARAMS = [

    # --------------------------------------------------------
    # ERM baseline
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


    # --------------------------------------------------------
    # Standard Importance Weighting
    # --------------------------------------------------------

    {
        "name": "importance_weighting",
        "learning": "importance_weighting",

        "model": "mlp",
        "model_params": MLP_PARAMS,

        "training_params": {
            **_NEURAL_BASE_PARAMS,

            "estimator": "kliep",

            "estimator_params": {
                "n_centers": 100,
            },

            "normalize_weights": True,
        },
    },


    # --------------------------------------------------------
    # Structural Importance Weighting V1
    # --------------------------------------------------------

    {
        "name": "dev_structural_weighting_v1",
        "learning": "dev_structural_weighting_v1",

        "model": "mlp",
        "model_params": MLP_PARAMS,

        "training_params": {
            **_NEURAL_BASE_PARAMS,

            "estimator": "kliep",

            "estimator_params": {
                "n_centers": 100,

                # Initial formulation:
                # pooled centering but no variance scaling.
                "standardize": False,
            },

            "normalize_weights": True,
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
            "model_name": "mlp",
            "regime": "Classical",
            "method": "ERM",
        },
        {
            "learning_method": "dev_positive_negative",
            "model_name": "mlp",
            "regime": "Classical",
            "method": "Positive-Negative",
        },
        {
            "learning_method": "importance_weighting",
            "model_name": "mlp",
            "regime": "UDA",
            "method": "Importance Weighting",
        },
        {
            "learning_method": "dev_structural_weighting_v1",
            "model_name": "mlp",
            "regime": "UDA",
            "method": "Structural IW V1",
        },
    ],

    "tables": [
        {
            "name": "cross_subject",
            "scenario": "cross_subject",
            "setting_column": "Dataset",
            "output_name": "cross_subject_table.csv",

            # We are not running domain evaluation in this
            # development experiment.
            "include_discrepancy": False,

            "filters": {
                "target_fraction": 1.0,
                "n_target_super_domains": 0,
                "use_max_source_domains": True,
            },
        },
    ],
}