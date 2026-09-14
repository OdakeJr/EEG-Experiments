import argparse
import importlib.util
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
# Pipeline
# ============================================================

from pipeline.process_data import run_preprocessing
from pipeline.scenarios import run_scenario
from pipeline.representation import run_feature_selection
from pipeline.training import run_training
from pipeline.evaluation.model_results import run_model_evaluation
from pipeline.analysis.benchmark_tables import run_benchmark_tables


# ============================================================
# Parameters
# ============================================================

def _load_params(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path = path.resolve()
    spec = importlib.util.spec_from_file_location("experiment_params", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module, path.relative_to(PROJECT_ROOT)


# ============================================================
# Execution helpers
# ============================================================

def _log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def _run_stage(name, function, *args):
    _log(f"[{name}] Starting...")
    start = time.perf_counter()
    result = function(*args)
    _log(f"[{name}] Done | elapsed={_format_time(time.perf_counter() - start)}")
    return result


def _run_tasks(function, tasks, max_workers):
    total = len(tasks)
    label = function.__name__.removeprefix("_").removesuffix("_task").replace("_", " ").title()

    if total == 0:
        _log(f"[{label}] No tasks")
        return []

    _log(f"[{label}] Tasks={total} | workers={max_workers}")
    start = time.perf_counter()

    def log_progress(completed):
        elapsed = time.perf_counter() - start
        remaining = total - completed
        eta = elapsed / completed * remaining if completed else 0
        _log(
            f"[{label}] Progress {completed}/{total} | remaining={remaining} | "
            f"elapsed={_format_time(elapsed)} | eta≈{_format_time(eta)}"
        )

    if max_workers <= 1:
        results = []
        for completed, task in enumerate(tasks, 1):
            results.append(function(task))
            log_progress(completed)
        return results

    results = [None] * total

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(function, task): i for i, task in enumerate(tasks)}

        try:
            completed = 0
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                completed += 1
                log_progress(completed)
        except Exception:
            _log(f"[{label}] ERROR | cancelling remaining tasks")
            for future in futures:
                future.cancel()
            raise

    return results


def _feature_selection_task(task):
    group, view, split, params = task
    fs = run_feature_selection(split, view, params, group=group)

    return {
        "group": group,
        "view": view,
        "split": split,
        "fs_artifact": fs,
    }


def _training_task(task):
    item_idx, model_idx, group, view, split, fs_artifact, params = task
    name = params.get("name", params.get("learning", "unknown"))
    pid = os.getpid()

    _log(f"[Training task] Starting | pid={pid} | item={item_idx + 1} | model={name}")
    start = time.perf_counter()

    try:
        model = run_training(split, view, fs_artifact, params, group=group)
    except Exception as exc:
        _log(
            f"[Training task] FAILED | pid={pid} | item={item_idx + 1} | "
            f"model={name} | {type(exc).__name__}: {exc}"
        )
        raise

    _log(
        f"[Training task] Done | pid={pid} | item={item_idx + 1} | "
        f"model={name} | elapsed={_format_time(time.perf_counter() - start)}"
    )

    return item_idx, model_idx, model


# ============================================================
# Stages
# ============================================================

def preprocessing_stage(params):
    return [
        {
            "group": config["name"],
            "view": run_preprocessing(config),
        }
        for config in params
    ]


def scenario_stage(preprocessing, scenario, params):
    return [
        {
            **item,
            "splits": run_scenario(item["view"], scenario, params),
        }
        for item in preprocessing
    ]


def feature_selection_stage(scenarios, params, max_workers=1):
    tasks = [
        (item["group"], item["view"], split, config)
        for item in scenarios
        for split in item["splits"]
        for config in params
    ]

    return _run_tasks(_feature_selection_task, tasks, max_workers)


def training_stage(features, params, max_workers=1):
    artifacts = [
        {
            **item,
            "artifacts": [None] * len(params),
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
            config,
        )
        for item_idx, item in enumerate(features)
        for model_idx, config in enumerate(params)
    ]

    for item_idx, model_idx, model in _run_tasks(_training_task, tasks, max_workers):
        artifacts[item_idx]["artifacts"][model_idx] = model

    return artifacts


def evaluation_stage(models, scenario, params):
    return run_model_evaluation({scenario: models}, params)


def analysis_stage(model_results, params):
    return run_benchmark_tables(model_results, None, params)


# ============================================================
# Main
# ============================================================

def main(params_path):
    params, params_path = _load_params(params_path)

    max_workers = params.EXECUTION_PARAMS.get("max_workers", 1)
    start = time.perf_counter()

    _log(
        f"[Pipeline] Starting | pid={os.getpid()} | "
        f"scenario={params.SCENARIO} | workers={max_workers}"
    )
    _log(f"[Pipeline] Params | {params_path}")

    preprocessing = _run_stage(
        "Preprocessing", preprocessing_stage,
        params.PREPROCESSING_PARAMS,
    )

    scenarios = _run_stage(
        "Scenarios", scenario_stage,
        preprocessing, params.SCENARIO, params.SCENARIO_PARAMS,
    )

    features = _run_stage(
        "Feature selection", feature_selection_stage,
        scenarios, params.FEATURE_SELECTION_PARAMS, max_workers,
    )

    models = _run_stage(
        "Training", training_stage,
        features, params.TRAINING_PARAMS, max_workers,
    )

    model_results = _run_stage(
        "Evaluation", evaluation_stage,
        models, params.SCENARIO, params.MODEL_EVALUATION_PARAMS,
    )

    results = _run_stage(
        "Analysis", analysis_stage,
        model_results, params.BENCHMARK_TABLES_PARAMS,
    )

    _log(f"[Pipeline] Finished | total={_format_time(time.perf_counter() - start)}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    args = parser.parse_args()
    main(args.params)