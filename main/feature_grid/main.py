# main/feature_grid/main.py

import argparse
import importlib.util
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.append(str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from ml.models.registry import MODELS
from pipeline.analysis.benchmark_tables import run_benchmark_tables
from pipeline.evaluation.model_results import run_model_evaluation
from pipeline.process_data import run_preprocessing
from pipeline.representation import build_representation_configs, run_representation
from pipeline.scenarios import run_scenario
from pipeline.training import run_training


def _load_params(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    spec = importlib.util.spec_from_file_location("experiment_params", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path.relative_to(PROJECT_ROOT)


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
        _log(f"[{label}] Progress {completed}/{total} | remaining={remaining} | "
             f"elapsed={_format_time(elapsed)} | eta≈{_format_time(eta)}")

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


def _representation_task(task):
    group, view, split, params = task
    artifact = run_representation(split, view, params, group=group)
    return {"group": group, "view": view, "split": split, "representation_artifact": artifact}


def _training_task(task):
    item_idx, model_idx, group, view, split, representation_artifact, params = task
    name = params.get("name", params.get("learning", "unknown"))
    pid = os.getpid()

    _log(f"[Training task] Starting | pid={pid} | item={item_idx + 1} | model={name}")
    start = time.perf_counter()
    try:
        model = run_training(split, view, representation_artifact, params, group=group)
    except Exception as exc:
        _log(f"[Training task] FAILED | pid={pid} | item={item_idx + 1} | "
             f"model={name} | {type(exc).__name__}: {exc}")
        raise

    _log(f"[Training task] Done | pid={pid} | item={item_idx + 1} | model={name} | "
         f"elapsed={_format_time(time.perf_counter() - start)}")
    return item_idx, model_idx, model


def _model_input_representation(params):
    model_name = params["model"]
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return getattr(MODELS[model_name], "input_representation", "features")


def _is_compatible(representation_artifact, training_params):
    produced = representation_artifact.output_representation
    expected = _model_input_representation(training_params)
    return expected == "any" or expected == produced


def preprocessing_stage(params):
    return [{"group": config["name"], "view": run_preprocessing(config)} for config in params]


def scenario_stage(preprocessing, scenario, params):
    return [{**item, "splits": run_scenario(item["view"], scenario, params)}
            for item in preprocessing]


def representation_stage(scenarios, params, max_workers=1):
    tasks = [(item["group"], item["view"], split, config)
             for item in scenarios for split in item["splits"] for config in params]
    return _run_tasks(_representation_task, tasks, max_workers)


def training_stage(representations, params, max_workers=1):
    artifacts = [{**item, "artifacts": [None] * len(params)} for item in representations]
    tasks = [
        (item_idx, model_idx, item["group"], item["view"], item["split"],
         item["representation_artifact"], config)
        for item_idx, item in enumerate(representations)
        for model_idx, config in enumerate(params)
        if _is_compatible(item["representation_artifact"], config)
    ]

    for item_idx, model_idx, model in _run_tasks(_training_task, tasks, max_workers):
        artifacts[item_idx]["artifacts"][model_idx] = model

    for item in artifacts:
        item["artifacts"] = [artifact for artifact in item["artifacts"] if artifact is not None]

    return [item for item in artifacts if item["artifacts"]]


def evaluation_stage(models, scenario, params):
    return run_model_evaluation({scenario: models}, params)


def analysis_stage(model_results, params):
    return run_benchmark_tables(model_results, None, params)


def main(params_path):
    params, params_path = _load_params(params_path)
    max_workers = params.EXECUTION_PARAMS.get("max_workers", 1)
    start = time.perf_counter()

    _log(f"[Pipeline] Starting | pid={os.getpid()} | scenario={params.SCENARIO} | workers={max_workers}")
    _log(f"[Pipeline] Params | {params_path}")

    preprocessing = _run_stage("Preprocessing", preprocessing_stage, params.PREPROCESSING_PARAMS)
    scenarios = _run_stage("Scenarios", scenario_stage, preprocessing, params.SCENARIO,
                           params.SCENARIO_PARAMS)

    representation_params = build_representation_configs(
        params.FEATURE_EXTRACTION_PARAMS,
        params.FEATURE_SELECTION_PARAMS,
        params.SIGNAL_TRANSFORM_PARAMS,
    )

    representations = _run_stage("Representation", representation_stage, scenarios,
                                 representation_params, max_workers)
    models = _run_stage("Training", training_stage, representations,
                        params.TRAINING_PARAMS, max_workers)
    model_results = _run_stage("Evaluation", evaluation_stage, models, params.SCENARIO,
                               params.MODEL_EVALUATION_PARAMS)
    results = _run_stage("Analysis", analysis_stage, model_results,
                         params.BENCHMARK_TABLES_PARAMS)

    _log(f"[Pipeline] Finished | total={_format_time(time.perf_counter() - start)}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    main(parser.parse_args().params)