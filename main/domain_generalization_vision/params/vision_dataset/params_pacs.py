
# main/domain_generalization_vision/params/vision_dataset/params_pacs.py

from copy import deepcopy

from params.common import (
    EXECUTION_PARAMS,
    FEATURE_EXTRACTION_PARAMS,
    FEATURE_SELECTION_PARAMS,
    SIGNAL_TRANSFORM_PARAMS,
    IMAGE_TRANSFORM_PARAMS,
    TRAINING_PARAMS as COMMON_TRAINING_PARAMS,
    MODEL_EVALUATION_PARAMS,
    BENCHMARK_TABLES_PARAMS,
)


# ============================================================
# Preprocessing
# ============================================================

PREPROCESSING_PARAMS = [
    {
        "dataset": "pacs",
        "name": "pacs_cross_domain",
        "root": "datasets/pacs",
        "representation": "image",
    },
]


# ============================================================
# Scenario
# ============================================================

SCENARIO = "cross_domain"

SCENARIO_PARAMS = {
    "seed": 0,
}


# ============================================================
# Training
# ============================================================

# Initial test: ERM only
TRAINING_PARAMS = deepcopy(COMMON_TRAINING_PARAMS[:1])

# Full experiments:
# TRAINING_PARAMS = deepcopy(COMMON_TRAINING_PARAMS)

for config in TRAINING_PARAMS:
    config["training_params"].update({
        "epochs": 2,
        "patience": None,
    })
