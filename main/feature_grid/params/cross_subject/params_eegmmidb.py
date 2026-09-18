# main/feature_grid/params/cross_subject/params_eegmmidb.py

from params.common import (
    BAND_CONFIGS,
    EXECUTION_PARAMS,
    FEATURE_SELECTION_PARAMS,
    METHOD_DISPLAY,
    MODEL_EVALUATION_PARAMS,
    PAPER_ANALYSIS_PARAMS,
    SIGNAL_TRANSFORM_PARAMS,
    TRAINING_PARAMS,
    make_feature_extraction_params,
)


SFREQ = 160.0

COMMON_CLASSES = [
    "left_hand_imagery",
    "right_hand_imagery",
    "both_hands_imagery",
    "both_feet_imagery",
]

CHANNELS = None
EEGMMIDB_RUNS = [4, 6, 8, 10, 12, 14]


# ============================================================
# Preprocessing
# ============================================================

PREPROCESSING_PARAMS = [
    {
        "dataset": "eegmmidb",
        "root_dir": "datasets/eegmmidb",
        "name": f"eegmmidb_cross_subject_{band_name}",
        "representation": "signal",
        "loader": {
            "runs": EEGMMIDB_RUNS,
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


# ============================================================
# Benchmark
# ============================================================

BENCHMARK_TABLES_PARAMS = {
    "method_display": METHOD_DISPLAY,
    "tables": [
        {
            "name": "cross_subject_paper1",
            "scenario": "cross_subject",
            "setting_column": "Dataset",
            "output_name": "cross_subject_paper1_table.csv",
            "include_discrepancy": False,
            "filters": {"target_fraction": 0.0},
        },
    ],
}