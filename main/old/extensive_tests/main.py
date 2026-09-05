import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from main.old.extensive_tests.params import (
    PREPROCESSING_PARAMS,
    COMBINE_PARAMS,
    SCENARIO_PARAMS,
    FEATURE_SELECTION_PARAMS,
    TRAINING_PARAMS,
    MODEL_EVALUATION_PARAMS,
    DOMAIN_EVALUATION_PARAMS,
    BENCHMARK_TABLES_PARAMS,
    SOURCE_DOMAIN_EFFECT_PARAMS,
    TARGET_FRACTION_EFFECT_PARAMS,
    DISCREPANCY_ANALYSIS_PARAMS,
    VARIABILITY_DECOMPOSITION_PARAMS,
    METHOD_RANKING_PARAMS,
)

from pipeline.process_data import run_preprocessing
from pipeline.combine_data import combine_datasets
from pipeline.scenarios import run_scenario
from pipeline.feature_selection import run_feature_selection
from pipeline.training import run_training
from pipeline.evaluation.model_results import run_model_evaluation
from pipeline.evaluation.domain_results import run_domain_evaluation

from pipeline.analysis.benchmark_tables import run_benchmark_tables
from pipeline.analysis.source_domain_effect import run_source_domain_effect
from pipeline.analysis.target_fraction_effect import run_target_fraction_effect
from pipeline.analysis.discrepancy_analysis import run_discrepancy_analysis
from pipeline.analysis.variability_decomposition import run_variability_decomposition
from pipeline.analysis.method_ranking import run_method_ranking


def preprocessing_stage():
    groups = {}

    for params in PREPROCESSING_PARAMS:
        view = run_preprocessing(params)
        groups.setdefault(params["name"], []).append(view)

    return [
        {"name": name, "views": views}
        for name, views in groups.items()
    ]


def combination_stage(preprocessing):
    combined = []

    for group in preprocessing:
        if len(group["views"]) < 2:
            continue

        params = {
            **COMBINE_PARAMS,
            "name": group["name"],
            "preprocessing_config_label": group["name"],
        }

        combined.append({
            "name": group["name"],
            "view": combine_datasets(group["views"], params),
        })

    return combined


def scenario_stage(preprocessing, combined):
    artifacts = {name: [] for name in SCENARIO_PARAMS}

    for group in preprocessing:
        for view in group["views"]:
            for scenario in ("intra_subject", "cross_session", "cross_subject"):
                artifacts[scenario].append({
                    "group": group["name"],
                    "view": view,
                    "splits": run_scenario(
                        view,
                        scenario,
                        SCENARIO_PARAMS[scenario],
                    ),
                })

    for group in combined:
        view = group["view"]

        artifacts["cross_dataset"].append({
            "group": group["name"],
            "view": view,
            "splits": run_scenario(
                view,
                "cross_dataset",
                SCENARIO_PARAMS["cross_dataset"],
            ),
        })

    return artifacts


def feature_selection_stage(scenarios):
    artifacts = {name: [] for name in scenarios}

    for scenario, groups in scenarios.items():
        for group in groups:
            for split in group["splits"]:
                fs = [
                    run_feature_selection(
                        split,
                        group["view"],
                        params,
                        group=group["group"],
                    )
                    for params in FEATURE_SELECTION_PARAMS
                ]

                artifacts[scenario].append({
                    "group": group["group"],
                    "view": group["view"],
                    "split": split,
                    "artifacts": fs,
                })

    return artifacts


def training_stage(fs_artifacts):
    artifacts = {name: [] for name in fs_artifacts}

    for scenario, items in fs_artifacts.items():
        for item in items:
            for fs in item["artifacts"]:
                models = [
                    run_training(
                        item["split"],
                        item["view"],
                        fs,
                        params,
                        group=item["group"],
                    )
                    for params in TRAINING_PARAMS
                ]

                artifacts[scenario].append({
                    **item,
                    "fs_artifact": fs,
                    "artifacts": models,
                })

    return artifacts


def evaluation_stage(scenarios, models):
    return (
        run_model_evaluation(models, MODEL_EVALUATION_PARAMS),
        run_domain_evaluation(scenarios, DOMAIN_EVALUATION_PARAMS),
    )


def analysis_stage(model_results, domain_results):
    return {
        "benchmark_tables": run_benchmark_tables(
            model_results, domain_results, BENCHMARK_TABLES_PARAMS
        ),
        "source_domain_effect": run_source_domain_effect(
            model_results, SOURCE_DOMAIN_EFFECT_PARAMS
        ),
        "target_fraction_effect": run_target_fraction_effect(
            model_results, TARGET_FRACTION_EFFECT_PARAMS
        ),
        "discrepancy_analysis": run_discrepancy_analysis(
            model_results, domain_results, DISCREPANCY_ANALYSIS_PARAMS
        ),
        "variability_decomposition": run_variability_decomposition(
            model_results, VARIABILITY_DECOMPOSITION_PARAMS
        ),
        "method_ranking": run_method_ranking(
            model_results, METHOD_RANKING_PARAMS
        ),
    }


def main():
    preprocessing = preprocessing_stage()
    combined = combination_stage(preprocessing)
    scenarios = scenario_stage(preprocessing, combined)
    features = feature_selection_stage(scenarios)
    models = training_stage(features)
    model_results, domain_results = evaluation_stage(scenarios, models)
    analysis_stage(model_results, domain_results)


if __name__ == "__main__":
    main()
