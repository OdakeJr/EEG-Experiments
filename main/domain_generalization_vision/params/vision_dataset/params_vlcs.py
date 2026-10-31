# main/feature_grid/params/cross_subject/params_zhou.py

from params.common import (
    BAND_CONFIGS,
    EXECUTION_PARAMS,
    FEATURE_SELECTION_PARAMS,
    METHOD_DISPLAY,
    MODEL_EVALUATION_PARAMS,
    PAPER_ANALYSIS_PARAMS,
    SIGNAL_TRANSFORM_PARAMS,
    TRAINING_PARAMS,
    BENCHMARK_TABLES_PARAMS,
    make_feature_extraction_params,
)


SFREQ = 250.0

COMMON_CLASSES = [
    "left_hand_imagery",
    "right_hand_imagery",
    "both_feet_imagery",
]

CHANNELS = None


# ============================================================
# Preprocessing
# ============================================================

PREPROCESSING_PARAMS = [
    {
        "dataset": "zhou",
        "root": "datasets/zhou",
        "name": f"zhou_cross_subject_{band_name}",
        "representation": "signal",
        "loader": {
            "channels": CHANNELS,
            "classes": COMMON_CLASSES,
            "tmin": 0.5,
            "tmax": 3.5,
        },
        "filter": {
            "bandpass": {
                "enabled": True,
                "bands": bands,
                "order": 5,
                "stack_bands": True,
            },
            "resample": {"enabled": False},
        },
        "show_progress": False,
    }
    for band_name, bands in BAND_CONFIGS.items()
]


# ============================================================
# Scenario
# ============================================================

SCENARIO = "cross_subject"

SCENARIO_PARAMS = {
    "source_counts": ["all"],
    "target_fractions": [0.0],
    "seed": 0,
}


# ============================================================
# Representations
# ============================================================

FEATURE_EXTRACTION_PARAMS = make_feature_extraction_params(SFREQ)