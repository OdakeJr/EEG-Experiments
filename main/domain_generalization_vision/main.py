
# main/domain_generalization_vision/main.py

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
    label = function.__name__.strip("_").removesuffix("_task").replace("_", " ").title()

    if not total:
        _log(f"[{label}] No tasks")
        return []

    _log(f"[{label}] Tasks={total} | workers={max_workers}")
    start = time.perf_counter()

    def progress(completed):
        elapsed = time.perf_counter() - start
        remaining = total - completed
        eta = elapsed / completed * remaining
        _log(
            f"[{label}] Progress {completed}/{total} | remaining={remaining} | "
            f"elapsed={_format_time(elapsed)} | eta≈{_format_time(eta)}"
        )

    if max_workers <= 1:
        results = []
        for i, task in enumerate(tasks, 1):
            results.append(function(task))
            progress(i)
        return results

    results = [None] * total

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(function, task): i
            for i, task in enumerate(tasks)
        }

        try:
            for completed, future in enumerate(as_completed(futures), 1):
                results[futures[future]] = future.result()
                progress(completed)
        except Exception:
            _log(f"[{label}] ERROR | cancelling remaining tasks")
            for future in futures:
                future.cancel()
            raise

    return results


def _representation_task(task):
    group, view, split, params = task
    name = params.get("name", "unknown")
    start = time.perf_counter()

    _log(f"[Representation] Starting | split={split.id} | config={name}")

    try:
        artifact = run_representation(split, view, params, group=group)
    except Exception as error:
        _log(f"[Representation] FAILED | split={split.id} | {error}")
        raise

    _log(
        f"[Representation] Done | split={split.id} | config={name} | "
        f"elapsed={_format_time(time.perf_counter() - start)}"
    )

    return {
        "group": group,
        "view": view,
        "split": split,
        "representation_artifact": artifact,
    }


def _training_task(task):
    item_idx, model_idx, group, view, split, representation, params = task
    name = params.get("name", params.get("learning", "unknown"))
    start = time.perf_counter()

    _log(f"[Training] Starting | split={split.id} | model={name}")

    try:
        model = run_training(
            split, view, representation, params, group=group
        )
    except Exception as error:
        _log(f"[Training] FAILED | split={split.id} | model={name} | {error}")
        raise

    _log(
        f"[Training] Done | split={split.id} | model={name} | "
        f"elapsed={_format_time(time.perf_counter() - start)}"
    )

    return item_idx, model_idx, model


def _is_compatible(representation, training_params):
    model_name = training_params["model"]

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    expected = getattr(MODELS[model_name], "input_representation", "features")
    produced = representation.output_representation

    return expected == "any" or expected == produced


# ============================================================
# Pipeline stages
# ============================================================

def preprocessing_stage(params):
    return [
        {"group": config["name"], "view": run_preprocessing(config)}
        for config in params
    ]


def scenario_stage(preprocessing, scenario, params):
    return [
        {**item, "splits": run_scenario(item["view"], scenario, params)}
        for item in preprocessing
    ]


def representation_stage(scenarios, params, max_workers=1):
    tasks = [
        (item["group"], item["view"], split, config)
        for item in scenarios
        for split in item["splits"]
        for config in params
    ]
    return _run_tasks(_representation_task, tasks, max_workers)


def training_stage(representations, params, max_workers=1):
    artifacts = [
        {**item, "artifacts": [None] * len(params)}
        for item in representations
    ]

    tasks = [
        (
            item_idx, model_idx,
            item["group"], item["view"], item["split"],
            item["representation_artifact"], config,
        )
        for item_idx, item in enumerate(representations)
        for model_idx, config in enumerate(params)
        if _is_compatible(item["representation_artifact"], config)
    ]

    for item_idx, model_idx, model in _run_tasks(
        _training_task, tasks, max_workers
    ):
        artifacts[item_idx]["artifacts"][model_idx] = model

    for item in artifacts:
        item["artifacts"] = [
            model for model in item["artifacts"] if model is not None
        ]

    return [item for item in artifacts if item["artifacts"]]


def evaluation_stage(models, scenario, params, max_workers=1):
    return run_model_evaluation(
        {scenario: models},
        params,
        max_workers=max_workers,
    )


def benchmark_stage(model_results, params):
    from pipeline.analysis.benchmark_tables import run_benchmark_tables
    return run_benchmark_tables(model_results, None, params)


# ============================================================
# Main
# ============================================================

def main(params_path):
    params, params_path = _load_params(params_path)

    max_workers = params.EXECUTION_PARAMS.get("max_workers", 1)
    evaluation_workers = params.EXECUTION_PARAMS.get("evaluation_workers", 1)
    start = time.perf_counter()

    _log(
        f"[Pipeline] Starting | scenario={params.SCENARIO} | "
        f"workers={max_workers} | evaluation_workers={evaluation_workers}"
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

    representation_params = build_representation_configs(
        getattr(params, "FEATURE_EXTRACTION_PARAMS", []),
        getattr(params, "FEATURE_SELECTION_PARAMS", []),
        getattr(params, "SIGNAL_TRANSFORM_PARAMS", []),
        getattr(params, "IMAGE_TRANSFORM_PARAMS", []),
    )

    representations = _run_stage(
        "Representation", representation_stage,
        scenarios, representation_params, max_workers,
    )

    models = _run_stage(
        "Training", training_stage,
        representations, params.TRAINING_PARAMS, max_workers,
    )

    model_results = _run_stage(
        "Evaluation", evaluation_stage,
        models, params.SCENARIO,
        getattr(params, "MODEL_EVALUATION_PARAMS", {}),
        evaluation_workers,
    )

    benchmark_params = getattr(params, "BENCHMARK_TABLES_PARAMS", None)
    results = model_results

    if benchmark_params is not None:
        results = _run_stage(
            "Benchmark", benchmark_stage,
            model_results, benchmark_params,
        )

    _log(
        f"[Pipeline] Finished | total={_format_time(time.perf_counter() - start)}"
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    main(parser.parse_args().params)
