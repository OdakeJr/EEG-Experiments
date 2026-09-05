# pipeline/evaluation/model_results.py

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from models.model_result import ModelResult
from models.model_results_artifact import ModelResultsArtifact
from utils.status import is_done, make_manifest, make_signature
from utils.storage import exists, load_manifest, load_pickle, save_data, save_manifest


OUTPUT_ROOT = Path("outputs/model_results")


def _json_copy(value):
    return json.loads(json.dumps(value, default=str))


def _json_string(value):
    return None if value is None else json.dumps(value, sort_keys=True, default=str)


def _coalesce(*values):
    return next((value for value in values if value is not None), None)


def _manifest_output(artifact):
    try:
        return load_manifest(artifact.manifest_path).get("output", {})
    except Exception:
        return {}


def _get_feature_selection_trace(artifact):
    output = _manifest_output(artifact)
    return {
        "feature_selection_method": _coalesce(
            getattr(artifact, "feature_selection_method", None),
            output.get("feature_selection_method"),
        ),
        "feature_selection_params": _json_copy(_coalesce(
            getattr(artifact, "feature_selection_params", None),
            output.get("feature_selection_params"),
        )),
        "feature_selection_config_label": _coalesce(
            getattr(artifact, "feature_selection_config_label", None),
            output.get("feature_selection_config_label"),
        ),
    }


def _get_model_trace(artifact):
    output = _manifest_output(artifact)
    return {
        **_get_feature_selection_trace(artifact),
        "preprocessing_signature": _coalesce(
            getattr(artifact, "preprocessing_signature", None),
            output.get("preprocessing_signature"),
        ),
        "preprocessing_config_label": _coalesce(
            getattr(artifact, "preprocessing_config_label", None),
            output.get("preprocessing_config_label"),
        ),
    }


def _get_preprocessing_trace(view, model_artifact=None):
    output = _manifest_output(view)
    return {
        "preprocessing_signature": _coalesce(
            getattr(model_artifact, "preprocessing_signature", None),
            getattr(view, "preprocessing_signature", None),
            output.get("preprocessing_signature"),
        ),
        "preprocessing_config_label": _coalesce(
            getattr(model_artifact, "preprocessing_config_label", None),
            getattr(view, "preprocessing_config_label", None),
            output.get("preprocessing_config_label"),
        ),
    }


def _get_classes(learner):
    if getattr(learner, "classes_", None) is not None:
        return np.asarray(learner.classes_)
    if getattr(getattr(learner, "model", None), "classes_", None) is not None:
        return np.asarray(learner.model.classes_)
    return None


def _compute_auc(y, probabilities, classes):
    if probabilities is None:
        return None

    try:
        unique = np.unique(y)
        if len(unique) < 2 or classes is None:
            return None

        if probabilities.shape[1] == 2:
            y_binary = (np.asarray(y) == classes[1]).astype(int)
            return roc_auc_score(y_binary, probabilities[:, 1])

        if not set(classes).issubset(set(unique)):
            return None

        return roc_auc_score(
            y, probabilities, labels=classes,
            multi_class="ovr", average="macro",
        )
    except ValueError:
        return None


def _count_parameters(learner):
    model = getattr(learner, "model", None)
    if model is None or not hasattr(model, "parameters"):
        return None
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _get_training_seed(manifest):
    params = manifest.get("params", {})
    training_params = params.get("training_params", {})
    if "seed" in training_params:
        return training_params["seed"]
    return params.get("model_params", {}).get("random_state")


def _iter_evaluation_sets(data):
    groups = {
        "source": data.source,
        "target_super_domain": data.target_super_domain,
        "target_elementary_domain": data.target_elementary_domain,
    }

    for group_name, group in groups.items():
        if group is None:
            continue

        for partition in np.unique(group.partitions):
            mask = group.partitions == partition
            if not np.any(mask):
                continue

            yield (
                group_name, str(partition), group.X[mask], group.y[mask],
                group.elementary_domains[mask],
                None if group.super_domains is None else group.super_domains[mask],
            )


def _evaluate_partition(learner, transformer, X, y, domains, super_domains):
    X = transformer.transform(X, domains)

    start = time.perf_counter()
    predictions = learner.predict(X, domains, super_domains)
    inference_time = time.perf_counter() - start

    try:
        probabilities = learner.predict_proba(X, domains, super_domains)
    except (NotImplementedError, AttributeError):
        probabilities = None

    return {
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, average="macro", zero_division=0),
        "auc": _compute_auc(y, probabilities, _get_classes(learner)),
        "inference_time": inference_time,
        "inference_time_per_sample": inference_time / len(y),
    }


def _slug(value):
    return re.sub(r"[^a-zA-Z0-9]+", "-", str(value)).strip("-").lower()


def _output_dir(model_artifacts, signature):
    scenarios = "+".join(_slug(x) for x in sorted(model_artifacts))

    methods = sorted({
        model.learning_method
        for artifacts in model_artifacts.values()
        for artifact in artifacts
        for model in artifact["artifacts"]
    })
    methods = "-".join(_slug(x) for x in methods)

    return OUTPUT_ROOT / scenarios / f"{methods}__{signature[:12]}"


def run_model_evaluation(model_artifacts, params=None):
    params = params or {}

    model_signatures = sorted(
        model.signature
        for artifacts in model_artifacts.values()
        for artifact in artifacts
        for model in artifact["artifacts"]
    )
    effective_params = {"models": model_signatures, "params": params}
    signature = make_signature(effective_params)

    output_dir = _output_dir(model_artifacts, signature)
    results_path = output_dir / "model_results.csv"
    manifest_path = output_dir / "manifest.json"

    if exists(results_path) and is_done(manifest_path, effective_params):
        manifest = load_manifest(manifest_path)
        return ModelResultsArtifact(
            path=str(results_path), manifest_path=str(manifest_path),
            signature=signature, n_rows=manifest["output"]["n_rows"],
        )

    start = time.time()
    save_manifest(make_manifest("running", effective_params), manifest_path)
    rows = []

    try:
        for scenario, artifacts in model_artifacts.items():
            for artifact in artifacts:
                group_name, view, split = artifact["group"], artifact["view"], artifact["split"]
                fs_artifact = artifact["fs_artifact"]
                fs_trace = _get_feature_selection_trace(fs_artifact)

                data = split.materialize(view)
                transformer = load_pickle(fs_artifact.transformer_path)

                for model_artifact in artifact["artifacts"]:
                    learner = load_pickle(model_artifact.model_path)
                    model_manifest = load_manifest(model_artifact.manifest_path)
                    model_trace = _get_model_trace(model_artifact)
                    preprocessing_trace = _get_preprocessing_trace(view, model_artifact)

                    fs_method = _coalesce(
                        model_trace["feature_selection_method"],
                        fs_trace["feature_selection_method"],
                    )
                    fs_params = _coalesce(
                        model_trace["feature_selection_params"],
                        fs_trace["feature_selection_params"],
                    )
                    fs_label = _coalesce(
                        model_trace["feature_selection_config_label"],
                        fs_trace["feature_selection_config_label"],
                    )
                    preprocessing_signature = _coalesce(
                        model_trace["preprocessing_signature"],
                        preprocessing_trace["preprocessing_signature"],
                    )
                    preprocessing_label = _coalesce(
                        model_trace["preprocessing_config_label"],
                        preprocessing_trace["preprocessing_config_label"],
                    )

                    training_time = model_manifest.get("execution_time")
                    training_seed = _get_training_seed(model_manifest)
                    model_size = Path(model_artifact.model_path).stat().st_size
                    n_parameters = _count_parameters(learner)

                    for evaluation_group, partition, X, y, domains, super_domains in _iter_evaluation_sets(data):
                        metrics = _evaluate_partition(
                            learner, transformer, X, y, domains, super_domains
                        )

                        result = ModelResult(
                            split_id=split.id,
                            scenario=scenario,
                            group=group_name,

                            n_source_domains=len(split.source_elementary_domains),
                            n_target_super_domains=len(split.target_super_domain_elementary_domains),
                            target_fraction=split.target_fraction,
                            split_seed=split.seed,

                            source_domains=";".join(map(str, split.source_elementary_domains)),
                            target_super_domains=";".join(map(str, split.target_super_domain_elementary_domains)),
                            target_domains=";".join(map(str, split.target_elementary_domains)),

                            feature_selection_signature=fs_artifact.signature,
                            learning_method=model_artifact.learning_method,
                            model_name=model_artifact.model_name,
                            model_signature=model_artifact.signature,
                            training_seed=training_seed,

                            evaluation_group=evaluation_group,
                            partition=partition,
                            n_samples=len(y),

                            accuracy=metrics["accuracy"],
                            balanced_accuracy=metrics["balanced_accuracy"],
                            macro_f1=metrics["macro_f1"],
                            auc=metrics["auc"],

                            training_time=training_time,
                            inference_time=metrics["inference_time"],
                            inference_time_per_sample=metrics["inference_time_per_sample"],
                            model_size_bytes=model_size,
                            n_parameters=n_parameters,
                        )

                        row = result.to_dict()
                        row.update({
                            "feature_selection_method": fs_method,
                            "feature_selection_params": _json_string(fs_params),
                            "feature_selection_config_label": fs_label,
                            "preprocessing_signature": preprocessing_signature,
                            "preprocessing_config_label": preprocessing_label,
                        })
                        rows.append(row)

        dataframe = pd.DataFrame(rows)
        save_data(dataframe, results_path)

        manifest = make_manifest(
            "done", effective_params, execution_time=time.time() - start
        )
        manifest["output"] = {"path": str(results_path), "n_rows": len(dataframe)}
        save_manifest(manifest, manifest_path)

    except Exception as error:
        save_manifest(make_manifest(
            "failed", effective_params,
            execution_time=time.time() - start,
            error=str(error),
        ), manifest_path)
        raise

    return ModelResultsArtifact(
        path=str(results_path), manifest_path=str(manifest_path),
        signature=signature, n_rows=len(rows),
    )