# ============================================================
# Preprocessing
# ============================================================

COMMON_CLASSES = [
    "left_hand_imagery",
    "right_hand_imagery",
    "both_feet_imagery",
]


# ------------------------------------------------------------
# Fixed preprocessing configuration
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Channel configurations
# ------------------------------------------------------------

CHANNEL_CONFIGS = {
    "ch01": ["C3", "Cz", "C4"],

    "ch02": ["FC3", "FCz", "FC4"],
    "ch03": ["CP3", "CPz", "CP4"],

    "ch04": ["FC3", "C3", "Cz", "C4"],
    "ch05": ["C3", "Cz", "C4", "CP3"],

    "ch06": ["FC3", "FC4", "C3", "Cz", "C4"],
    "ch07": ["C3", "Cz", "C4", "CP3", "CP4"],

    "ch08": ["FC3", "FCz", "FC4", "C3", "Cz", "C4"],

    "ch09": ["C3", "Cz", "C4", "CP3", "CPz", "CP4"],

    "ch10": ["FC3", "FC4", "C3", "C4", "CP3", "CP4"],
}


# ------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------

DATASET_CONFIGS = [
    {
        "dataset": "bci2a",
        "root_gdf": "datasets/bci2a/gdf",
        "root_mat": "datasets/bci2a/mat",
    },
    {
        "dataset": "eegmmidb",
        "root": "datasets/eegmmidb",
    },
    {
        "dataset": "weibo",
        "root": "datasets/weibo",
    },
    {
        "dataset": "zhou",
        "root": "datasets/zhou",
    },
]


# ------------------------------------------------------------
# Preprocessing experiments
# ------------------------------------------------------------

PREPROCESSING_PARAMS = []

for channel_name, channels in CHANNEL_CONFIGS.items():

    for dataset_config in DATASET_CONFIGS:

        PREPROCESSING_PARAMS.append({
            **dataset_config,

            "name": channel_name,

            "loader": {
                "channels": channels,
                "classes": COMMON_CLASSES,
            },

            "filter": FILTER_CONFIG,
            "features": FEATURE_CONFIG,

            "show_progress": False,
        })


# ============================================================
# Dataset combination
# ============================================================

COMBINE_PARAMS = {}


# ============================================================
# Scenarios
# ============================================================

INTRA_SUBJECT_PARAMS = {
    "train_fraction": 0.8,
    "seed": 0,
}


CROSS_SESSION_PARAMS = {
    "source_counts": {
        # BCI IV 2a: 2 sessions per subject
        "bci_iv_2a": [1],

        # Zhou2016: 3 sessions per subject
        "zhou2016": [1, "all"],
    },

    "target_fractions": [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
    ],

    "max_source_combinations": 10,
    "seed": 0,
}


CROSS_SUBJECT_PARAMS = {
    "source_counts": {
        # BCI IV 2a: 9 subjects
        "bci_iv_2a": [1, 3, 5, 7, "all"],

        # EEGMMIDB: 109 subjects
        "eegmmidb": [1, 20, 40, 60, 80, "all"],

        # Weibo2014: 10 subjects
        "weibo2014": [1, 3, 5, 7, "all"],

        # Zhou2016: 4 subjects
        "zhou2016": [1, 2, "all"],
    },

    "target_fractions": [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
    ],

    "max_source_combinations": 10,
    "seed": 0,
}


CROSS_DATASET_PARAMS = {
    "source_dataset_counts": [
        1,
        2,
        "all",
    ],

    "target_dataset_subject_counts": [
        0,
    ],

    "target_subject_fractions": [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
    ],

    "max_source_combinations": 10,
    "seed": 0,
}


SCENARIO_PARAMS = {
    "intra_subject": INTRA_SUBJECT_PARAMS,
    "cross_session": CROSS_SESSION_PARAMS,
    "cross_subject": CROSS_SUBJECT_PARAMS,
    "cross_dataset": CROSS_DATASET_PARAMS,
}


# ============================================================
# Feature selection
# ============================================================

FEATURE_SELECTION_PARAMS = [
    {
        "method": "random",
        "config_label": f"random_20_seed{seed}",
        "params": {
            "ratio": 0.2,
            "seed": seed,
        },
    }
    for seed in range(10)
]

# ============================================================
# Training
# ============================================================

TRAINING_SEEDS = [
    0,
]


# ------------------------------------------------------------
# Shared model parameters
# ------------------------------------------------------------

LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
}

SVM_PARAMS = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale",
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "n_jobs": -1,
}

MLP_PARAMS = {
    "hidden_dims": (128, 64),
}


# ------------------------------------------------------------
# Shared training parameters
# ------------------------------------------------------------

_NEURAL_BASE_PARAMS = {
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "device": "cpu",
}

_SOURCE_FREE_BASE_PARAMS = {
    "source_epochs": 100,
    "source_learning_rate": 1e-3,
    "batch_size": 64,
    "weight_decay": 0.0,
    "device": "cpu",
}


def _expand_seeds(configs):
    return [
        {
            **config,
            "name": f"{config['name']}_seed{seed}",
            "training_params": {
                **config["training_params"],
                "seed": seed,
            },
        }
        for config in configs
        for seed in TRAINING_SEEDS
    ]


# ------------------------------------------------------------
# Classical
# ------------------------------------------------------------

_CLASSICAL_METHODS = [
    {
        "name": "logistic_regression",
        "learning": "sklearn_erm",
        "model": "logistic_regression",
        "model_params": LOGISTIC_REGRESSION_PARAMS,
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
        "name": "svm",
        "learning": "sklearn_erm",
        "model": "svm",
        "model_params": SVM_PARAMS,
        "training_params": {},
    },
    {
        "name": "mlp_erm",
        "learning": "neural_erm",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
        },
    },
]

CLASSICAL_PARAMS = _expand_seeds(
    _CLASSICAL_METHODS
)


# ------------------------------------------------------------
# Domain generalization
# ------------------------------------------------------------

_DOMAIN_GENERALIZATION_METHODS = [
    {
        "name": "irm",
        "learning": "irm",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "irm_lambda": 100.0,
            "penalty_anneal_epochs": 10,
        },
    },
    {
        "name": "groupdro",
        "learning": "groupdro",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "eta": 0.01,
        },
    },
    {
        "name": "vrex",
        "learning": "vrex",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "vrex_lambda": 1.0,
            "penalty_anneal_epochs": 0,
        },
    },
    {
        "name": "coral",
        "learning": "coral",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "coral_lambda": 1.0,
        },
    },
    {
        "name": "mmd",
        "learning": "mmd",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "mmd_lambda": 1.0,
            "gamma": None,
        },
    },
]

DOMAIN_GENERALIZATION_PARAMS = _expand_seeds(
    _DOMAIN_GENERALIZATION_METHODS
)


# ------------------------------------------------------------
# Domain adaptation - unlabeled target
# ------------------------------------------------------------

_DOMAIN_ADAPTATION_UNLABELED_METHODS = [
    {
        "name": "deep_coral",
        "learning": "deep_coral",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "coral_lambda": 1.0,
        },
    },
    {
        "name": "dann",
        "learning": "dann",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "dann_lambda": 1.0,
        },
    },
    {
        "name": "mcd",
        "learning": "mcd",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
        },
    },
    {
        "name": "importance_weighting",
        "learning": "importance_weighting",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "estimator": "kliep",
        },
    },
]

DOMAIN_ADAPTATION_UNLABELED_PARAMS = _expand_seeds(
    _DOMAIN_ADAPTATION_UNLABELED_METHODS
)


# ------------------------------------------------------------
# Domain adaptation - labeled target
# ------------------------------------------------------------

_DOMAIN_ADAPTATION_LABELED_METHODS = [
    {
        "name": "joint_supervised",
        "learning": "joint_supervised",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
        },
    },
    {
        "name": "supervised_dann",
        "learning": "supervised_dann",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_NEURAL_BASE_PARAMS,
            "dann_lambda": 1.0,
        },
    },
]

DOMAIN_ADAPTATION_LABELED_PARAMS = _expand_seeds(
    _DOMAIN_ADAPTATION_LABELED_METHODS
)


# ------------------------------------------------------------
# Source-free adaptation - unlabeled target
# ------------------------------------------------------------

_SOURCE_FREE_UNLABELED_METHODS = [
    {
        "name": "shot",
        "learning": "shot",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_SOURCE_FREE_BASE_PARAMS,
            "adaptation_epochs": 50,
            "adaptation_lr": 1e-4,
            "pseudo_label_weight": 1.0,
            "entropy_weight": 1.0,
            "diversity_weight": 1.0,
        },
    },
    {
        "name": "nrc",
        "learning": "nrc",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_SOURCE_FREE_BASE_PARAMS,
            "adaptation_epochs": 50,
            "adaptation_lr": 1e-4,
            "k": 5,
            "kk": 5,
            "nonreciprocal_weight": 0.1,
            "self_weight": 1.0,
            "diversity_weight": 1.0,
        },
    },
    {
        "name": "sfda_de",
        "learning": "sfda_de",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_SOURCE_FREE_BASE_PARAMS,
            "adaptation_epochs": 50,
            "adaptation_lr": 1e-4,
            "confidence_threshold": 0.5,
            "covariance_gamma": 1.0,
            "kernel_gamma": None,
            "samples_per_class": 16,
            "steps_per_epoch": 10,
            "kmeans_iterations": 10,
            "covariance_eps": 1e-5,
        },
    },
]

SOURCE_FREE_UNLABELED_PARAMS = _expand_seeds(
    _SOURCE_FREE_UNLABELED_METHODS
)


# ------------------------------------------------------------
# Source-free adaptation - labeled target
# ------------------------------------------------------------

_SOURCE_FREE_LABELED_METHODS = [
    {
        "name": "linear_probe",
        "learning": "linear_probe",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_SOURCE_FREE_BASE_PARAMS,
            "adaptation_epochs": 50,
            "adaptation_learning_rate": 1e-3,
        },
    },
    {
        "name": "fine_tuning",
        "learning": "fine_tuning",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_SOURCE_FREE_BASE_PARAMS,
            "adaptation_epochs": 50,
            "adaptation_learning_rate": 1e-4,
        },
    },
    {
        "name": "lp_ft",
        "learning": "lp_ft",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_SOURCE_FREE_BASE_PARAMS,
            "probe_epochs": 25,
            "fine_tune_epochs": 25,
            "probe_learning_rate": 1e-3,
            "fine_tune_learning_rate": 1e-4,
        },
    },
    {
        "name": "l2_sp",
        "learning": "l2_sp",
        "model": "mlp",
        "model_params": MLP_PARAMS,
        "training_params": {
            **_SOURCE_FREE_BASE_PARAMS,
            "adaptation_epochs": 50,
            "adaptation_learning_rate": 1e-4,
            "l2sp_lambda": 1e-3,
        },
    },
]

SOURCE_FREE_LABELED_PARAMS = _expand_seeds(
    _SOURCE_FREE_LABELED_METHODS
)


# ------------------------------------------------------------
# Final training configurations
# ------------------------------------------------------------

TRAINING_PARAMS = (
    CLASSICAL_PARAMS
    + DOMAIN_GENERALIZATION_PARAMS
    + DOMAIN_ADAPTATION_UNLABELED_PARAMS
    + DOMAIN_ADAPTATION_LABELED_PARAMS
    + SOURCE_FREE_UNLABELED_PARAMS
    + SOURCE_FREE_LABELED_PARAMS
)


# ============================================================
# Model evaluation
# ============================================================

MODEL_EVALUATION_PARAMS = {}


# ============================================================
# Domain evaluation
# ============================================================

DOMAIN_EVALUATION_PARAMS = {
    "metrics": [
        {
            "name": "mmd",
            "params": {},
        },
    ],

    "representations": [
        "marginal",
        "class",
    ],

    "standardize": True,

    "min_samples_per_side": 2,
}


# ============================================================
# Analysis - Benchmark tables
# ============================================================

BENCHMARK_TABLES_PARAMS = {}


# ============================================================
# Analysis - Source-domain effect
# ============================================================

SOURCE_DOMAIN_EFFECT_PARAMS = {}


# ============================================================
# Analysis - Target-fraction effect
# ============================================================

TARGET_FRACTION_EFFECT_PARAMS = {}


# ============================================================
# Analysis - Discrepancy analysis
# ============================================================

DISCREPANCY_ANALYSIS_PARAMS = {}


# ============================================================
# Analysis - Variability decomposition
# ============================================================

VARIABILITY_DECOMPOSITION_PARAMS = {}


# ============================================================
# Analysis - Method ranking
# ============================================================

METHOD_RANKING_PARAMS = {}