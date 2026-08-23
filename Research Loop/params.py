# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

PREPROCESSING_PARAMS = [
    {
        "dataset": "bci2a",
        "filter": [8, 30],
        "features": "covariance",
    },
    # more variants...
]

# --------------------------------------------------
# Combine
# --------------------------------------------------

COMBINE_PARAMS = {
    "name": "all_datasets",
}

# --------------------------------------------------
# Scenarios
# --------------------------------------------------

INTRA_SUBJECT_PARAMS = {
    "train_fraction": 0.8,
}

CROSS_SESSION_PARAMS = {
    "source_counts": [1, "all"],
    "target_fractions": [0.0, 0.1, 0.25],
}

CROSS_SUBJECT_PARAMS = {
    "source_counts": [1, 2, 4, "all"],
    "target_fractions": [0.0, 0.1, 0.25],
}

CROSS_DATASET_PARAMS = {
    "source_dataset_counts": [1, 2, "all"],
    "target_dataset_subject_counts": [0, 1, "all"],
    "target_subject_fractions": [0.0, 0.1, 0.25],
}

SCENARIO_PARAMS = {
    "intra_subject": INTRA_SUBJECT_PARAMS,
    "cross_session": CROSS_SESSION_PARAMS,
    "cross_subject": CROSS_SUBJECT_PARAMS,
    "cross_dataset": CROSS_DATASET_PARAMS,
}


# --------------------------------------------------
# Domain analysis
# --------------------------------------------------

DOMAIN_ANALYSIS_PARAMS = {
    "methods": ["mmd", "energy", "wasserstein"]
}


# --------------------------------------------------
# Feature selection / transformations
# --------------------------------------------------

FEATURE_SELECTION_PARAMS = [
    {"method": "none"},
    {"method": "mutual_information", "k": 20},
]


# --------------------------------------------------
# Learning
# --------------------------------------------------

TRAINING_PARAMS = [
    {"method": "svm"},
    {"method": "random_forest"},
    {"method": "mlp"},
]


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

EVALUATION_PARAMS = {
    "metrics": [
        "balanced_accuracy",
        "accuracy",
        "macro_f1",
    ]
}


