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
        "root_gdf": "data/bci2a/gdf",
        "root_mat": "data/bci2a/mat",
    },
    {
        "dataset": "eegmmidb",
        "root": "data/eegmmidb",
    },
    {
        "dataset": "weibo",
        "root": "data/weibo",
    },
    {
        "dataset": "zhou",
        "root": "data/zhou",
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

LOGISTIC_REGRESSION_PARAMS = {}

SVM_PARAMS = {}

RANDOM_FOREST_PARAMS = {}

MLP_PARAMS = {}


# ------------------------------------------------------------
# Classical
# ------------------------------------------------------------

CLASSICAL_PARAMS = [
    # Logistic Regression
    # Random Forest
    # RBF-SVM
    # MLP / ERM
]


# ------------------------------------------------------------
# Domain generalization
# ------------------------------------------------------------

DOMAIN_GENERALIZATION_PARAMS = [
    # IRM
    # GroupDRO
    # VREx
    # CORAL
    # MMD
]


# ------------------------------------------------------------
# Domain adaptation - unlabeled target
# ------------------------------------------------------------

DOMAIN_ADAPTATION_UNLABELED_PARAMS = [
    # Deep CORAL
    # DANN
    # MCD
    # Importance Weighting
]


# ------------------------------------------------------------
# Domain adaptation - labeled target
# ------------------------------------------------------------

DOMAIN_ADAPTATION_LABELED_PARAMS = [
    # Joint Supervised
    # Supervised DANN
    # MME
    # LIRR
]


# ------------------------------------------------------------
# Source-free adaptation - unlabeled target
# ------------------------------------------------------------

SOURCE_FREE_UNLABELED_PARAMS = [
    # Frozen Source
    # SHOT
    # NRC
    # SFDA-DE
]


# ------------------------------------------------------------
# Source-free adaptation - labeled target
# ------------------------------------------------------------

SOURCE_FREE_LABELED_PARAMS = [
    # Linear Probe
    # Fine-Tuning
    # LP-FT
    # L2-SP
]


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

DOMAIN_EVALUATION_PARAMS = {}


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