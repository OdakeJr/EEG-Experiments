# main/feature_grid/params/intra_subject/params_bci2a.py

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


SFREQ = 250.0

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


# ============================================================
# Preprocessing
# ============================================================

PREPROCESSING_PARAMS = [
    {
        "dataset": "bci2a",
        "root_gdf": "datasets/bci2a/gdf",
        "root_mat": "datasets/bci2a/mat",
        "name": f"bci2a_intra_{band_name}",
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

SCENARIO = "intra_subject"

SCENARIO_PARAMS = {
    "train_fraction": 0.8,
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
            "name": "intra_subject_paper1",
            "scenario": "intra_subject",
            "setting_column": "Dataset",
            "output_name": "intra_subject_paper1_table.csv",
            "include_discrepancy": False,
            "filters": {"target_fraction": 0.8},
        },
    ],
}