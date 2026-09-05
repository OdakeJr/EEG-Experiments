import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


# ============================================================
# Paths
# ============================================================

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.append(str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


# ============================================================
# Parameters
# ============================================================

from params import (
    EXECUTION_PARAMS,
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

from pipeline.process_data import run_preprocessing
from pipeline.scenarios import run_scenario
from pipeline.feature_selection import run_feature_selection
from pipeline.training import run_training
from pipeline.evaluation.model_results import run_model_evaluation
from pipeline.analysis.benchmark_tables import run_benchmark_tables


# ============================================================
# Execution helpers
# ============================================================

def _run_stage(name, function, *args):
    print(f"\n[{name}] Starting...", flush=True)
    start = time.perf_counter()
    result = function(*args)
    print(f"[{name}] Done in {time.perf_counter() - start:.1f}s", flush=True)
    return result


def _run_tasks(function, tasks, max_workers):
    if max_workers <= 1:
        return [function(task) for task in tasks]

    results = [None] * len(tasks)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(function, task): i
            for i, task in enumerate(tasks)
        }

        try:
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        except Exception:
            for future in futures:
                future.cancel()
            raise

    return results


def _feature_selection_task(task):
    group, view, split, params = task

    fs = run_feature_selection(
        split, view, params, group=group
    )

    return {
        "group": group,
        "view": view,
        "split": split,
        "fs_artifact": fs,
    }


def _training_task(task):
    item_idx, model_idx, group, view, split, fs_artifact, params = task

    model = run_training(
        split, view, fs_artifact, params, group=group
    )

    return item_idx, model_idx, model


# ============================================================
# Stages
# ============================================================

def preprocessing_stage():
    return [
        {
            "group": params["name"],
            "view": run_preprocessing(params),
        }
        for params in PREPROCESSING_PARAMS
    ]


def scenario_stage(preprocessing):
    return [
        {
            **item,
            "splits": run_scenario(
                item["view"], SCENARIO, SCENARIO_PARAMS
            ),
        }
        for item in preprocessing
    ]


def feature_selection_stage(scenarios, max_workers=1):
    tasks = [
        (item["group"], item["view"], split, params)
        for item in scenarios
        for split in item["splits"]
        for params in FEATURE_SELECTION_PARAMS
    ]

    return _run_tasks(
        _feature_selection_task,
        tasks,
        max_workers,
    )


def training_stage(features, max_workers=1):
    artifacts = [
        {
            **item,
            "artifacts": [None] * len(TRAINING_PARAMS),
        }
        for item in features
    ]

    tasks = [
        (
            item_idx,
            model_idx,
            item["group"],
            item["view"],
            item["split"],
            item["fs_artifact"],
            params,
        )
        for item_idx, item in enumerate(features)
        for model_idx, params in enumerate(TRAINING_PARAMS)
    ]

    for item_idx, model_idx, model in _run_tasks(
        _training_task,
        tasks,
        max_workers,
    ):
        artifacts[item_idx]["artifacts"][model_idx] = model

    return artifacts


def evaluation_stage(models):
    return run_model_evaluation(
        {SCENARIO: models},
        MODEL_EVALUATION_PARAMS,
    )


def analysis_stage(model_results):
    return run_benchmark_tables(
        model_results,
        None,
        BENCHMARK_TABLES_PARAMS,
    )


# ============================================================
# Main
# ============================================================

def main():
    max_workers = EXECUTION_PARAMS.get("max_workers", 1)
    start = time.perf_counter()

    print(
        f"\n[Pipeline] Starting | scenario={SCENARIO} | workers={max_workers}",
        flush=True,
    )

    preprocessing = _run_stage("Preprocessing", preprocessing_stage)
    scenarios = _run_stage("Scenarios", scenario_stage, preprocessing)
    features = _run_stage(
        "Feature selection", feature_selection_stage, scenarios, max_workers
    )
    models = _run_stage("Training", training_stage, features, max_workers)
    model_results = _run_stage("Evaluation", evaluation_stage, models)
    results = _run_stage("Analysis", analysis_stage, model_results)

    print(
        f"\n[Pipeline] Finished in {time.perf_counter() - start:.1f}s",
        flush=True,
    )

    return results


if __name__ == "__main__":
    main()