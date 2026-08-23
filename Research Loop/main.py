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

    dataset_views = []

    for params in PREPROCESSING_PARAMS:
        view = run_preprocessing(params)
        dataset_views.append(view)

    # --------------------------------------------------
    # 2. COMBINE DATASETS
    # --------------------------------------------------

    combined_view = combine_datasets(
        dataset_views,
        COMBINE_PARAMS,
    )

    # --------------------------------------------------
    # 3. SCENARIOS
    # --------------------------------------------------

    scenario_artifacts = {
        "intra_subject": [],
        "cross_session": [],
        "cross_subject": [],
        "cross_dataset": [],
    }


    for view, preprocessing_params in zip(
        dataset_views,
        PREPROCESSING_PARAMS,
    ):

        dataset = preprocessing_params["dataset"]

        scenario_artifacts["intra_subject"].append({
            "dataset": dataset,
            "view": view,
            "splits": run_scenario(
                view,
                "intra_subject",
                INTRA_SUBJECT_PARAMS,
            ),
        })

        scenario_artifacts["cross_session"].append({
            "dataset": dataset,
            "view": view,
            "splits": run_scenario(
                view,
                "cross_session",
                CROSS_SESSION_PARAMS,
            ),
        })

        scenario_artifacts["cross_subject"].append({
            "dataset": dataset,
            "view": view,
            "splits": run_scenario(
                view,
                "cross_subject",
                CROSS_SUBJECT_PARAMS,
            ),
        })


    scenario_artifacts["cross_dataset"].append({
        "dataset": "combined",
        "view": combined_view,
        "splits": run_scenario(
            combined_view,
            "cross_dataset",
            CROSS_DATASET_PARAMS,
        ),
    })

    # --------------------------------------------------
    # 4. FEATURE SELECTION / TRANSFORMATION
    # --------------------------------------------------

    fs_artifacts = run_feature_selection(
        scenario_artifacts,
        FEATURE_SELECTION_PARAMS,
    )

    # --------------------------------------------------
    # 5. TRAINING
    # --------------------------------------------------

    model_artifacts = run_training(
        scenario_artifacts,
        fs_artifacts,
        TRAINING_PARAMS,
    )

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
