# pipeline/evaluation/model_results.py

import time
import json
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from models.model_result import ModelResult
from models.model_results_artifact import ModelResultsArtifact

from utils.storage import (
    exists,
    load_manifest,
    load_pickle,
    save_data,
    save_manifest,
)

from utils.status import (
    is_done,
    make_manifest,
    make_signature,
)


OUTPUT_ROOT = Path("outputs/model_results")


# --------------------------------------------------
# Helper: JSON-safe copy
# --------------------------------------------------

def _json_copy(value):
    try:
        return json.loads(
            json.dumps(
                value,
                default=str,
            )
        )
    except TypeError:
        return deepcopy(
            value
        )


def _json_string(value):
    if value is None:
        return None

    try:
        return json.dumps(
            value,
            sort_keys=True,
            default=str,
        )
    except TypeError:
        return str(
            value
        )


# --------------------------------------------------
# Helper: propagated trace
# --------------------------------------------------

def _get_feature_selection_trace(
    fs_artifact,
):
    method = getattr(
        fs_artifact,
        "feature_selection_method",
        None,
    )

    params = getattr(
        fs_artifact,
        "feature_selection_params",
        None,
    )

    config_label = getattr(
        fs_artifact,
        "feature_selection_config_label",
        None,
    )

    if (
        method is not None
        and config_label is not None
    ):
        return {
            "feature_selection_method": method,
            "feature_selection_params": _json_copy(
                params
            ),
            "feature_selection_config_label": config_label,
        }

    try:
        manifest = load_manifest(
            fs_artifact.manifest_path
        )

        output = manifest.get(
            "output",
            {},
        )

        return {
            "feature_selection_method": output.get(
                "feature_selection_method",
                method,
            ),
            "feature_selection_params": _json_copy(
                output.get(
                    "feature_selection_params",
                    params,
                )
            ),
            "feature_selection_config_label": output.get(
                "feature_selection_config_label",
                config_label,
            ),
        }

    except Exception:
        return {
            "feature_selection_method": method,
            "feature_selection_params": _json_copy(
                params
            ),
            "feature_selection_config_label": config_label,
        }


def _get_preprocessing_trace(
    view,
    model_artifact=None,
):
    preprocessing_signature = getattr(
        model_artifact,
        "preprocessing_signature",
        None,
    )

    preprocessing_config_label = getattr(
        model_artifact,
        "preprocessing_config_label",
        None,
    )

    if preprocessing_signature is None:
        preprocessing_signature = getattr(
            view,
            "preprocessing_signature",
            None,
        )

    if preprocessing_config_label is None:
        preprocessing_config_label = getattr(
            view,
            "preprocessing_config_label",
            None,
        )

    if (
        preprocessing_signature is not None
        and preprocessing_config_label is not None
    ):
        return {
            "preprocessing_signature": preprocessing_signature,
            "preprocessing_config_label": preprocessing_config_label,
        }

    try:
        manifest = load_manifest(
            view.manifest_path
        )

        output = manifest.get(
            "output",
            {},
        )

        return {
            "preprocessing_signature": output.get(
                "preprocessing_signature",
                preprocessing_signature,
            ),
            "preprocessing_config_label": output.get(
                "preprocessing_config_label",
                preprocessing_config_label,
            ),
        }

    except Exception:
        return {
            "preprocessing_signature": preprocessing_signature,
            "preprocessing_config_label": preprocessing_config_label,
        }


def _get_model_trace(
    model_artifact,
):
    method = getattr(
        model_artifact,
        "feature_selection_method",
        None,
    )

    params = getattr(
        model_artifact,
        "feature_selection_params",
        None,
    )

    config_label = getattr(
        model_artifact,
        "feature_selection_config_label",
        None,
    )

    preprocessing_signature = getattr(
        model_artifact,
        "preprocessing_signature",
        None,
    )

    preprocessing_config_label = getattr(
        model_artifact,
        "preprocessing_config_label",
        None,
    )

    if (
        config_label is not None
        or preprocessing_config_label is not None
    ):
        return {
            "feature_selection_method": method,
            "feature_selection_params": _json_copy(
                params
            ),
            "feature_selection_config_label": config_label,
            "preprocessing_signature": preprocessing_signature,
            "preprocessing_config_label": preprocessing_config_label,
        }

    try:
        manifest = load_manifest(
            model_artifact.manifest_path
        )

        output = manifest.get(
            "output",
            {},
        )

        return {
            "feature_selection_method": output.get(
                "feature_selection_method",
                method,
            ),
            "feature_selection_params": _json_copy(
                output.get(
                    "feature_selection_params",
                    params,
                )
            ),
            "feature_selection_config_label": output.get(
                "feature_selection_config_label",
                config_label,
            ),
            "preprocessing_signature": output.get(
                "preprocessing_signature",
                preprocessing_signature,
            ),
            "preprocessing_config_label": output.get(
                "preprocessing_config_label",
                preprocessing_config_label,
            ),
        }

    except Exception:
        return {
            "feature_selection_method": method,
            "feature_selection_params": _json_copy(
                params
            ),
            "feature_selection_config_label": config_label,
            "preprocessing_signature": preprocessing_signature,
            "preprocessing_config_label": preprocessing_config_label,
        }


# --------------------------------------------------
# Helper: class order
# --------------------------------------------------

def _get_classes(learner):
    """
    Recover the class order associated with
    predict_proba().
    """

    if hasattr(learner, "classes_"):
        classes = learner.classes_

        if classes is not None:
            return np.asarray(classes)

    if (
        hasattr(learner, "model")
        and hasattr(learner.model, "classes_")
    ):
        return np.asarray(
            learner.model.classes_
        )

    return None


# --------------------------------------------------
# Helper: AUC
# --------------------------------------------------

def _compute_auc(
    y,
    probabilities,
    classes,
):
    """
    Compute binary or multiclass AUC when possible.
    """

    if probabilities is None:
        return None

    try:
        unique_classes = np.unique(y)

        if len(unique_classes) < 2:
            return None

        # ------------------------------------------
        # Binary classification
        # ------------------------------------------

        if probabilities.shape[1] == 2:

            if classes is None:
                return None

            positive_class = classes[1]

            y_binary = (
                np.asarray(y)
                == positive_class
            ).astype(int)

            return roc_auc_score(
                y_binary,
                probabilities[:, 1],
            )

        # ------------------------------------------
        # Multiclass classification
        # ------------------------------------------

        if classes is None:
            return None

        # AUC is not directly comparable if the
        # evaluation partition lacks trained classes.
        if not set(classes).issubset(
            set(unique_classes)
        ):
            return None

        return roc_auc_score(
            y,
            probabilities,
            labels=classes,
            multi_class="ovr",
            average="macro",
        )

    except ValueError:
        return None


# --------------------------------------------------
# Helper: parameter count
# --------------------------------------------------

def _count_parameters(learner):
    """
    Count trainable parameters for neural models.
    """

    model = getattr(
        learner,
        "model",
        None,
    )

    if model is None:
        return None

    if not hasattr(
        model,
        "parameters",
    ):
        return None

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# --------------------------------------------------
# Helper: training seed
# --------------------------------------------------

def _get_training_seed(manifest):
    """
    Recover the training random seed when available.
    """

    params = manifest.get(
        "params",
        {},
    )

    training_params = params.get(
        "training_params",
        {},
    )

    if "seed" in training_params:
        return training_params["seed"]

    model_params = params.get(
        "model_params",
        {},
    )

    return model_params.get(
        "random_state"
    )


# --------------------------------------------------
# Helper: iterate evaluation partitions
# --------------------------------------------------

def _iter_evaluation_sets(data):
    """
    Yield every available group / partition pair.

    Examples
    --------
    source / train
    target_super_domain / calibration
    target_elementary_domain / calibration
    target_elementary_domain / test
    """

    groups = {
        "source": data.source,
        "target_super_domain": (
            data.target_super_domain
        ),
        "target_elementary_domain": (
            data.target_elementary_domain
        ),
    }

    for group_name, group in groups.items():

        if group is None:
            continue

        for partition in np.unique(
            group.partitions
        ):

            mask = (
                group.partitions
                == partition
            )

            if not np.any(mask):
                continue

            yield (
                group_name,
                str(partition),
                group.X[mask],
                group.y[mask],
                group.elementary_domains[mask],
                (
                    None
                    if group.super_domains is None
                    else group.super_domains[mask]
                ),
            )


# --------------------------------------------------
# Helper: evaluate one partition
# --------------------------------------------------

def _evaluate_partition(
    learner,
    transformer,
    X,
    y,
    domains,
    super_domains,
):
    """
    Apply the fitted feature transformation and
    evaluate one fitted learning system.
    """

    X = transformer.transform(
        X,
        domains,
    )

    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------

    start = time.perf_counter()

    predictions = learner.predict(
        X,
        domains,
        super_domains,
    )

    inference_time = (
        time.perf_counter()
        - start
    )

    # ----------------------------------------------
    # Probabilities
    # ----------------------------------------------

    probabilities = None

    try:
        probabilities = (
            learner.predict_proba(
                X,
                domains,
                super_domains,
            )
        )

    except (
        NotImplementedError,
        AttributeError,
    ):
        pass

    classes = _get_classes(
        learner
    )

    # ----------------------------------------------
    # Metrics
    # ----------------------------------------------

    return {
        "accuracy": accuracy_score(
            y,
            predictions,
        ),

        "balanced_accuracy": (
            balanced_accuracy_score(
                y,
                predictions,
            )
        ),

        "macro_f1": f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "auc": _compute_auc(
            y,
            probabilities,
            classes,
        ),

        "inference_time": (
            inference_time
        ),

        "inference_time_per_sample": (
            inference_time
            / len(y)
        ),
    }


# --------------------------------------------------
# Public pipeline
# --------------------------------------------------

def run_model_evaluation(
    model_artifacts,
    params=None,
):
    """
    Evaluate all trained learning systems and
    generate the canonical model-results table.
    """

    params = params or {}

    # --------------------------------------------------
    # Evaluation identity
    # --------------------------------------------------

    model_signatures = []

    for artifacts in model_artifacts.values():

        for artifact in artifacts:

            for model_artifact in (
                artifact["artifacts"]
            ):
                model_signatures.append(
                    model_artifact.signature
                )

    effective_params = {
        "models": sorted(
            model_signatures
        ),
        "params": params,
    }

    signature = make_signature(
        effective_params
    )

    # --------------------------------------------------
    # Output paths
    # --------------------------------------------------

    output_dir = (
        OUTPUT_ROOT
        / signature[:12]
    )

    results_path = (
        output_dir
        / "model_results.csv"
    )

    manifest_path = (
        output_dir
        / "manifest.json"
    )

    # --------------------------------------------------
    # Resume
    # --------------------------------------------------

    if (
        exists(results_path)
        and is_done(
            manifest_path,
            effective_params,
        )
    ):
        manifest = load_manifest(
            manifest_path
        )

        return ModelResultsArtifact(
            path=str(
                results_path
            ),
            manifest_path=str(
                manifest_path
            ),
            signature=signature,
            n_rows=manifest[
                "output"
            ]["n_rows"],
        )

    # --------------------------------------------------
    # Start execution
    # --------------------------------------------------

    start = time.time()

    save_manifest(
        make_manifest(
            "running",
            effective_params,
        ),
        manifest_path,
    )

    rows = []

    try:

        # --------------------------------------------------
        # Iterate scenarios
        # --------------------------------------------------

        for scenario, artifacts in (
            model_artifacts.items()
        ):

            # ----------------------------------------------
            # Iterate split + FS combinations
            # ----------------------------------------------

            for artifact in artifacts:

                group_name = artifact[
                    "group"
                ]

                view = artifact[
                    "view"
                ]

                split = artifact[
                    "split"
                ]

                fs_artifact = artifact[
                    "fs_artifact"
                ]

                fs_trace = _get_feature_selection_trace(
                    fs_artifact
                )

                # ------------------------------------------
                # Materialize scenario once
                # ------------------------------------------

                data = split.materialize(
                    view
                )

                # ------------------------------------------
                # Load fitted feature transformer once
                # ------------------------------------------

                transformer = load_pickle(
                    fs_artifact.transformer_path
                )

                # ------------------------------------------
                # Iterate trained models
                # ------------------------------------------

                for model_artifact in (
                    artifact["artifacts"]
                ):

                    learner = load_pickle(
                        model_artifact.model_path
                    )

                    model_manifest = (
                        load_manifest(
                            model_artifact.manifest_path
                        )
                    )

                    model_trace = _get_model_trace(
                        model_artifact
                    )

                    preprocessing_trace = (
                        _get_preprocessing_trace(
                            view=view,
                            model_artifact=model_artifact,
                        )
                    )

                    feature_selection_method = (
                        model_trace.get(
                            "feature_selection_method"
                        )
                        or fs_trace.get(
                            "feature_selection_method"
                        )
                    )

                    feature_selection_params = (
                        model_trace.get(
                            "feature_selection_params"
                        )
                        or fs_trace.get(
                            "feature_selection_params"
                        )
                    )

                    feature_selection_config_label = (
                        model_trace.get(
                            "feature_selection_config_label"
                        )
                        or fs_trace.get(
                            "feature_selection_config_label"
                        )
                    )

                    preprocessing_signature = (
                        model_trace.get(
                            "preprocessing_signature"
                        )
                        or preprocessing_trace.get(
                            "preprocessing_signature"
                        )
                    )

                    preprocessing_config_label = (
                        model_trace.get(
                            "preprocessing_config_label"
                        )
                        or preprocessing_trace.get(
                            "preprocessing_config_label"
                        )
                    )

                    # --------------------------------------
                    # Training information
                    # --------------------------------------

                    training_time = (
                        model_manifest.get(
                            "execution_time"
                        )
                    )

                    training_seed = (
                        _get_training_seed(
                            model_manifest
                        )
                    )

                    model_size_bytes = (
                        Path(
                            model_artifact.model_path
                        ).stat().st_size
                    )

                    n_parameters = (
                        _count_parameters(
                            learner
                        )
                    )

                    # --------------------------------------
                    # Evaluate every available partition
                    # --------------------------------------

                    for (
                        evaluation_group,
                        partition,
                        X,
                        y,
                        domains,
                        super_domains,
                    ) in _iter_evaluation_sets(
                        data
                    ):

                        metrics = (
                            _evaluate_partition(
                                learner,
                                transformer,
                                X,
                                y,
                                domains,
                                super_domains,
                            )
                        )

                        # ----------------------------------
                        # Canonical result row
                        # ----------------------------------

                        result = ModelResult(

                            # Experiment identity
                            split_id=split.id,
                            scenario=scenario,
                            group=group_name,

                            # Scenario configuration
                            n_source_domains=len(
                                split.source_elementary_domains
                            ),

                            n_target_super_domains=len(
                                split.target_super_domain_elementary_domains
                            ),

                            target_fraction=(
                                split.target_fraction
                            ),

                            split_seed=split.seed,

                            # Domain identity
                            source_domains=";".join(
                                split.source_elementary_domains
                            ),

                            target_super_domains=";".join(
                                split.target_super_domain_elementary_domains
                            ),

                            target_domains=";".join(
                                split.target_elementary_domains
                            ),

                            # Training configuration
                            feature_selection_signature=(
                                fs_artifact.signature
                            ),

                            learning_method=(
                                model_artifact.learning_method
                            ),

                            model_name=(
                                model_artifact.model_name
                            ),

                            model_signature=(
                                model_artifact.signature
                            ),

                            training_seed=(
                                training_seed
                            ),

                            # Evaluation location
                            evaluation_group=(
                                evaluation_group
                            ),

                            partition=partition,

                            n_samples=len(y),

                            # Predictive performance
                            accuracy=metrics[
                                "accuracy"
                            ],

                            balanced_accuracy=metrics[
                                "balanced_accuracy"
                            ],

                            macro_f1=metrics[
                                "macro_f1"
                            ],

                            auc=metrics[
                                "auc"
                            ],

                            # Computational information
                            training_time=(
                                training_time
                            ),

                            inference_time=metrics[
                                "inference_time"
                            ],

                            inference_time_per_sample=(
                                metrics[
                                    "inference_time_per_sample"
                                ]
                            ),

                            model_size_bytes=(
                                model_size_bytes
                            ),

                            n_parameters=(
                                n_parameters
                            ),
                        )

                        row = result.to_dict()

                        # ----------------------------------
                        # Clean propagated trace columns
                        # ----------------------------------

                        row.update(
                            {
                                "feature_selection_method": (
                                    feature_selection_method
                                ),

                                "feature_selection_params": (
                                    _json_string(
                                        feature_selection_params
                                    )
                                ),

                                "feature_selection_config_label": (
                                    feature_selection_config_label
                                ),

                                "preprocessing_signature": (
                                    preprocessing_signature
                                ),

                                "preprocessing_config_label": (
                                    preprocessing_config_label
                                ),
                            }
                        )

                        rows.append(
                            row
                        )

        # --------------------------------------------------
        # Build canonical table
        # --------------------------------------------------

        dataframe = pd.DataFrame(
            rows
        )

        save_data(
            dataframe,
            results_path,
        )

        # --------------------------------------------------
        # Manifest
        # --------------------------------------------------

        execution_time = (
            time.time()
            - start
        )

        manifest = make_manifest(
            "done",
            effective_params,
            execution_time=execution_time,
        )

        manifest["output"] = {
            "path": str(
                results_path
            ),
            "n_rows": len(
                dataframe
            ),
        }

        save_manifest(
            manifest,
            manifest_path,
        )

    except Exception as error:

        execution_time = (
            time.time()
            - start
        )

        save_manifest(
            make_manifest(
                "failed",
                effective_params,
                execution_time=execution_time,
                error=str(error),
            ),
            manifest_path,
        )

        raise

    # --------------------------------------------------
    # Return table artifact
    # --------------------------------------------------

    return ModelResultsArtifact(
        path=str(
            results_path
        ),
        manifest_path=str(
            manifest_path
        ),
        signature=signature,
        n_rows=len(rows),
    )