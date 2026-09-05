import os
import sys
from pathlib import Path


# ============================================================
# Paths
# ============================================================

EXPERIMENT_DIR = Path(
    __file__
).resolve().parent

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


# Make sure this experiment's params.py is imported.
sys.path.insert(
    0,
    str(EXPERIMENT_DIR),
)

# Make project packages available.
sys.path.append(
    str(PROJECT_ROOT),
)

# Keep all relative project paths relative to project root.
os.chdir(
    PROJECT_ROOT
)


# ============================================================
# Parameters
# ============================================================

from main.old.development_tests_cross_sub.params import (
    PREPROCESSING_PARAMS,
    SCENARIO,
    SCENARIO_PARAMS,
    FEATURE_SELECTION_PARAMS,
    TRAINING_PARAMS,
    MODEL_EVALUATION_PARAMS,
    BENCHMARK_TABLES_PARAMS,
)


# ============================================================
# Pipeline
# ============================================================

from pipeline.process_data import (
    run_preprocessing,
)

from pipeline.scenarios import (
    run_scenario,
)

from pipeline.feature_selection import (
    run_feature_selection,
)

from pipeline.training import (
    run_training,
)

from pipeline.evaluation.model_results import (
    run_model_evaluation,
)

from pipeline.analysis.benchmark_tables import (
    run_benchmark_tables,
)


# ============================================================
# Preprocessing
# ============================================================

def preprocessing_stage():
    artifacts = []

    for params in PREPROCESSING_PARAMS:

        view = run_preprocessing(
            params
        )

        artifacts.append({
            "group": params["name"],
            "view": view,
        })

    return artifacts


# ============================================================
# Scenario
# ============================================================

def scenario_stage(
    preprocessing,
):
    artifacts = []

    for item in preprocessing:

        splits = run_scenario(
            item["view"],
            SCENARIO,
            SCENARIO_PARAMS,
        )

        artifacts.append({
            **item,
            "splits": splits,
        })

    return artifacts


# ============================================================
# Feature selection
# ============================================================

def feature_selection_stage(
    scenarios,
):
    artifacts = []

    for item in scenarios:

        for split in item["splits"]:

            for params in FEATURE_SELECTION_PARAMS:

                fs = run_feature_selection(
                    split,
                    item["view"],
                    params,
                    group=item["group"],
                )

                artifacts.append({
                    "group": item["group"],
                    "view": item["view"],
                    "split": split,
                    "fs_artifact": fs,
                })

    return artifacts


# ============================================================
# Training
# ============================================================

def training_stage(
    features,
):
    artifacts = []

    for item in features:

        models = []

        for params in TRAINING_PARAMS:

            model = run_training(
                item["split"],
                item["view"],
                item["fs_artifact"],
                params,
                group=item["group"],
            )

            models.append(
                model
            )

        artifacts.append({
            **item,
            "artifacts": models,
        })

    return artifacts


# ============================================================
# Evaluation
# ============================================================

def evaluation_stage(
    models,
):
    model_results = run_model_evaluation(
        {
            SCENARIO: models,
        },
        MODEL_EVALUATION_PARAMS,
    )

    return model_results


# ============================================================
# Analysis
# ============================================================

def analysis_stage(
    model_results,
):
    benchmark_tables = run_benchmark_tables(
        model_results,
        None,
        BENCHMARK_TABLES_PARAMS,
    )

    return benchmark_tables


# ============================================================
# Main
# ============================================================

def main():

    preprocessing = (
        preprocessing_stage()
    )

    scenarios = scenario_stage(
        preprocessing
    )

    features = feature_selection_stage(
        scenarios
    )

    models = training_stage(
        features
    )

    model_results = evaluation_stage(
        models
    )

    benchmark_tables = analysis_stage(
        model_results
    )

    return benchmark_tables


if __name__ == "__main__":
    main()