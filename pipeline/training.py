import time
import json
from pathlib import Path
from copy import deepcopy

import numpy as np

from ml.models.registry import get_model
from ml.learning.registry import get_learning_algorithm

from models.model_artifact import ModelArtifact
from models.scenario_data import DataGroup, ScenarioData

from utils.storage import exists, load_manifest, load_pickle, save_manifest
from utils.status import is_done, make_manifest, make_signature


OUTPUT_ROOT = Path("outputs/training")


# ============================================================
# Trace helpers
# ============================================================

def _json_copy(value):
    try:
        return json.loads(json.dumps(value, default=str))
    except TypeError:
        return deepcopy(value)


def _get_preprocessing_trace(view):
    signature = getattr(view, "preprocessing_signature", None)
    label = getattr(view, "preprocessing_config_label", None)

    if signature is not None and label is not None:
        return {
            "preprocessing_signature": signature,
            "preprocessing_config_label": label,
        }

    try:
        output = load_manifest(view.manifest_path).get("output", {})
        return {
            "preprocessing_signature": output.get(
                "preprocessing_signature", signature
            ),
            "preprocessing_config_label": output.get(
                "preprocessing_config_label", label
            ),
        }
    except Exception:
        return {
            "preprocessing_signature": signature,
            "preprocessing_config_label": label,
        }


def _get_feature_selection_trace(fs_artifact):
    method = getattr(fs_artifact, "feature_selection_method", None)
    params = getattr(fs_artifact, "feature_selection_params", None)
    label = getattr(fs_artifact, "feature_selection_config_label", None)

    if method is not None and label is not None:
        return {
            "feature_selection_method": method,
            "feature_selection_params": _json_copy(params),
            "feature_selection_config_label": label,
        }

    try:
        output = load_manifest(fs_artifact.manifest_path).get("output", {})
        return {
            "feature_selection_method": output.get(
                "feature_selection_method", method
            ),
            "feature_selection_params": _json_copy(
                output.get("feature_selection_params", params)
            ),
            "feature_selection_config_label": output.get(
                "feature_selection_config_label", label
            ),
        }
    except Exception:
        return {
            "feature_selection_method": method,
            "feature_selection_params": _json_copy(params),
            "feature_selection_config_label": label,
        }


def _make_model_artifact(
    split,
    fs_artifact,
    learning_method,
    model_name,
    model_path,
    manifest_path,
    signature,
    preprocessing_trace,
    feature_selection_trace,
):
    return ModelArtifact(
        split_id=split.id,
        feature_selection_signature=fs_artifact.signature,
        learning_method=learning_method,
        model_name=model_name,
        model_path=str(model_path),
        manifest_path=str(manifest_path),
        signature=signature,
        feature_selection_method=feature_selection_trace.get(
            "feature_selection_method"
        ),
        feature_selection_params=feature_selection_trace.get(
            "feature_selection_params"
        ),
        feature_selection_config_label=feature_selection_trace.get(
            "feature_selection_config_label"
        ),
        preprocessing_signature=preprocessing_trace.get(
            "preprocessing_signature"
        ),
        preprocessing_config_label=preprocessing_trace.get(
            "preprocessing_config_label"
        ),
    )


# ============================================================
# Data transformation
# ============================================================

def _transform_group(group, transformer):
    if group is None:
        return None

    return DataGroup(
        X=transformer.transform(group.X, group.elementary_domains),
        y=group.y,
        elementary_domains=group.elementary_domains,
        partitions=group.partitions,
        super_domains=group.super_domains,
    )


def _transform_data(data, transformer):
    return ScenarioData(
        source=_transform_group(data.source, transformer),
        target_super_domain=_transform_group(
            data.target_super_domain, transformer
        ),
        target_elementary_domain=_transform_group(
            data.target_elementary_domain, transformer
        ),
    )


# ============================================================
# Model context
# ============================================================

def _get_input_shape(data):
    for group in [
        data.source,
        data.target_super_domain,
        data.target_elementary_domain,
    ]:
        if group is not None and len(group.X) > 0:
            return tuple(group.X.shape[1:])

    raise ValueError("No samples found.")


def _get_output_dim(data):
    labels = []

    for group in [
        data.source,
        data.target_super_domain,
        data.target_elementary_domain,
    ]:
        if group is None:
            continue

        mask = group.partitions == "train"

        if np.any(mask):
            labels.append(group.y[mask])

    if not labels:
        raise ValueError("No training labels found.")

    return len(np.unique(np.concatenate(labels)))


# ============================================================
# Paths
# ============================================================

def _get_output_paths(
    scenario,
    group,
    split_id,
    fs_signature,
    name,
    signature,
):
    output_dir = (
        OUTPUT_ROOT
        / scenario
        / group
        / split_id
        / fs_signature[:12]
        / f"{name}_{signature[:12]}"
    )

    return (
        output_dir / "model.pkl",
        output_dir / "manifest.json",
    )


# ============================================================
# Public function
# ============================================================

def run_training(
    split,
    view,
    fs_artifact,
    training_params,
    group="default",
):
    learning_method = training_params["learning"]
    model_name = training_params["model"]

    learning_params = training_params.get("learning_params", {})
    model_params = training_params.get("model_params", {})
    fit_params = training_params.get("training_params", {})

    name = training_params.get(
        "name",
        f"{learning_method}_{model_name}",
    )

    input_manifest = load_manifest(view.manifest_path)
    preprocessing_trace = _get_preprocessing_trace(view)
    feature_selection_trace = _get_feature_selection_trace(fs_artifact)

    effective_params = {
        "split": split.to_dict(),
        "input_signature": input_manifest["signature"],
        "feature_selection_signature": fs_artifact.signature,
        "feature_selection_config_label": feature_selection_trace.get(
            "feature_selection_config_label"
        ),
        "preprocessing_signature": preprocessing_trace.get(
            "preprocessing_signature"
        ),
        "preprocessing_config_label": preprocessing_trace.get(
            "preprocessing_config_label"
        ),
        "learning_method": learning_method,
        "learning_params": learning_params,
        "model": model_name,
        "model_params": model_params,
        "training_params": fit_params,
    }

    signature = make_signature(effective_params)

    model_path, manifest_path = _get_output_paths(
        split.scenario,
        group,
        split.id,
        fs_artifact.signature,
        name,
        signature,
    )

    if exists(model_path) and is_done(manifest_path, effective_params):
        return _make_model_artifact(
            split=split,
            fs_artifact=fs_artifact,
            learning_method=learning_method,
            model_name=model_name,
            model_path=model_path,
            manifest_path=manifest_path,
            signature=signature,
            preprocessing_trace=preprocessing_trace,
            feature_selection_trace=feature_selection_trace,
        )

    start = time.time()

    save_manifest(
        make_manifest("running", effective_params),
        manifest_path,
    )

    try:
        # Materialize scenario
        data = split.materialize(view)

        # Apply fitted transformation
        transformer = load_pickle(fs_artifact.transformer_path)
        data = _transform_data(data, transformer)

        # Data-dependent model context
        model_context = {
            "input_shape": _get_input_shape(data),
            "output_dim": _get_output_dim(data),
        }

        # Create model
        model = get_model(
            model_name,
            model_params,
            **model_context,
        )

        # Create learner
        learner = get_learning_algorithm(
            learning_method,
            learning_params,
        )

        # Train
        learner.fit(
            model=model,
            source=data.source,
            target_super_domain=data.target_super_domain,
            target_elementary_domain=data.target_elementary_domain,
            **fit_params,
        )

        learner.save(model_path)

        execution_time = time.time() - start

        manifest = make_manifest(
            "done",
            effective_params,
            execution_time=execution_time,
        )

        manifest["output"] = {
            "model_path": str(model_path),
            "model_context": _json_copy(model_context),
            "model_params": _json_copy(model_params),
            "feature_selection_signature": fs_artifact.signature,
            "feature_selection_method": feature_selection_trace.get(
                "feature_selection_method"
            ),
            "feature_selection_params": feature_selection_trace.get(
                "feature_selection_params"
            ),
            "feature_selection_config_label": feature_selection_trace.get(
                "feature_selection_config_label"
            ),
            "preprocessing_signature": preprocessing_trace.get(
                "preprocessing_signature"
            ),
            "preprocessing_config_label": preprocessing_trace.get(
                "preprocessing_config_label"
            ),
        }

        save_manifest(manifest, manifest_path)

    except Exception as error:
        execution_time = time.time() - start

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

    return _make_model_artifact(
        split=split,
        fs_artifact=fs_artifact,
        learning_method=learning_method,
        model_name=model_name,
        model_path=model_path,
        manifest_path=manifest_path,
        signature=signature,
        preprocessing_trace=preprocessing_trace,
        feature_selection_trace=feature_selection_trace,
    )