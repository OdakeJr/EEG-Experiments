from params import (
    PREPROCESSING_PARAMS,
    COMBINE_PARAMS,
    INTRA_SUBJECT_PARAMS,
    CROSS_SESSION_PARAMS,
    CROSS_SUBJECT_PARAMS,
    CROSS_DATASET_PARAMS,
    DOMAIN_ANALYSIS_PARAMS,
    FEATURE_SELECTION_PARAMS,
    TRAINING_PARAMS,
    EVALUATION_PARAMS,
)

from pipeline.process_data import run_preprocessing
from pipeline.combine_data import combine_datasets
from pipeline.scenarios import run_scenario
from pipeline.domain_analysis import run_domain_analysis
from pipeline.feature_selection import run_feature_selection
from pipeline.training import run_training
from pipeline.evaluation import run_evaluation


def main():

    # --------------------------------------------------
    # 1. PREPROCESSING
    # --------------------------------------------------

    preprocessing_artifacts = []

    for group_params in PREPROCESSING_PARAMS:

        dataset_views = []

        for params in group_params["datasets"]:

            view = run_preprocessing(
                params
            )

            dataset_views.append(
                view
            )

        preprocessing_artifacts.append({
            "name": group_params["name"],
            "views": dataset_views,
        })

    # --------------------------------------------------
    # 2. COMBINE DATASETS
    # --------------------------------------------------

    combined_artifacts = []

    for artifact in preprocessing_artifacts:

        combined_view = combine_datasets(
            artifact["views"],
            {
                **COMBINE_PARAMS,
                "name": artifact["name"],
            },
        )

        combined_artifacts.append({
            "name": artifact["name"],
            "view": combined_view,
        })

    # --------------------------------------------------
    # 3. SCENARIOS
    # --------------------------------------------------

    scenario_artifacts = {
        "intra_subject": [],
        "cross_session": [],
        "cross_subject": [],
        "cross_dataset": [],
    }


    for artifact in preprocessing_artifacts:

        group_name = artifact["name"]

        for view in artifact["views"]:

            scenario_artifacts["intra_subject"].append({
                "group": group_name,
                "view": view,
                "splits": run_scenario(
                    view,
                    "intra_subject",
                    INTRA_SUBJECT_PARAMS,
                ),
            })

            scenario_artifacts["cross_session"].append({
                "group": group_name,
                "view": view,
                "splits": run_scenario(
                    view,
                    "cross_session",
                    CROSS_SESSION_PARAMS,
                ),
            })

            scenario_artifacts["cross_subject"].append({
                "group": group_name,
                "view": view,
                "splits": run_scenario(
                    view,
                    "cross_subject",
                    CROSS_SUBJECT_PARAMS,
                ),
            })


    for artifact in combined_artifacts:

        scenario_artifacts["cross_dataset"].append({
            "group": artifact["name"],
            "view": artifact["view"],
            "splits": run_scenario(
                artifact["view"],
                "cross_dataset",
                CROSS_DATASET_PARAMS,
            ),
        })

    # --------------------------------------------------
    # 4. FEATURE SELECTION / TRANSFORMATION
    # --------------------------------------------------

    fs_artifacts = {
        scenario: []
        for scenario in scenario_artifacts
    }

    for scenario, artifacts in scenario_artifacts.items():

        for artifact in artifacts:

            group_name = artifact["group"]
            view = artifact["view"]

            for split in artifact["splits"]:

                split_fs_artifacts = []

                for fs_params in FEATURE_SELECTION_PARAMS:

                    fs_artifact = run_feature_selection(
                        split,
                        view,
                        fs_params,
                        group=group_name,
                    )

                    split_fs_artifacts.append(
                        fs_artifact
                    )

                fs_artifacts[scenario].append({
                    "group": group_name,
                    "view": view,
                    "split": split,
                    "artifacts": split_fs_artifacts,
                })

    # --------------------------------------------------
    # 5. TRAINING
    # --------------------------------------------------

    model_artifacts = {
        scenario: []
        for scenario in fs_artifacts
    }

    for scenario, artifacts in fs_artifacts.items():
        for artifact in artifacts:
            group_name = artifact["group"]
            view = artifact["view"]
            split = artifact["split"]

            for fs_artifact in artifact["artifacts"]:
                split_model_artifacts = []

                for training_params in TRAINING_PARAMS:
                    model_artifact = run_training(
                        split,
                        view,
                        fs_artifact,
                        training_params,
                        group=group_name,
                    )

                    split_model_artifacts.append(
                        model_artifact
                    )

                model_artifacts[scenario].append({
                    "group": group_name,
                    "view": view,
                    "split": split,
                    "fs_artifact": fs_artifact,
                    "artifacts": split_model_artifacts,
                })

    # --------------------------------------------------
    # 6. DOMAIN ANALYSIS
    # --------------------------------------------------

    run_domain_analysis(
        scenario_artifacts,
        DOMAIN_ANALYSIS_PARAMS,
    )
        
    # --------------------------------------------------
    # 7. EVALUATION
    # --------------------------------------------------

    run_evaluation(
        scenario_artifacts,
        model_artifacts,
        EVALUATION_PARAMS,
    )


if __name__ == "__main__":
    main()
