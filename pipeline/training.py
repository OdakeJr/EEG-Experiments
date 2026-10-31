
# pipeline/training.py

import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from ml.learning.lib.data import class_mapping, make_loader, split_indices
from ml.learning.registry import get_learning_algorithm
from ml.models.registry import MODELS, get_model
from models.model_artifact import ModelArtifact
from models.scenario_data import DataGroup, ScenarioData
from utils.storage import exists, load_manifest, load_pickle, save_manifest
from utils.status import is_done, make_manifest, make_signature


OUTPUT_ROOT = Path("outputs/training")

TRACE_FIELDS = (
    "input_representation", "output_representation",
    "representation_method", "representation_params",
    "representation_config_label",
    "signal_transform_method", "signal_transform_params",
    "signal_transform_config_label",
    "feature_extraction_method", "feature_extraction_params",
    "feature_extraction_config_label",
    "feature_selection_method", "feature_selection_params",
    "feature_selection_config_label",
    "preprocessing_signature", "preprocessing_config_label",
)


# ============================================================
# Helpers
# ============================================================

def _json_copy(value):
    try:
        return json.loads(json.dumps(value, default=str))
    except TypeError:
        return deepcopy(value)


def _get_preprocessing_trace(view):
    return {
        "preprocessing_signature": getattr(view, "preprocessing_signature", None),
        "preprocessing_config_label": getattr(view, "preprocessing_config_label", None),
    }


def _get_representation_trace(artifact):
    trace = {field: getattr(artifact, field, None) for field in TRACE_FIELDS}

    try:
        output = load_manifest(artifact.manifest_path).get("output", {})
        for field in TRACE_FIELDS:
            if trace[field] is None:
                trace[field] = output.get(field)
    except Exception:
        pass

    return {key: _json_copy(value) for key, value in trace.items()}


def _get_model_input_representation(model_name):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(MODELS)}")
    return getattr(MODELS[model_name], "input_representation", "features")


def _validate_model_representation(model_name, representation):
    expected = _get_model_input_representation(model_name)
    if expected != "any" and expected != representation:
        raise ValueError(
            f"Model '{model_name}' expects '{expected}', got '{representation}'."
        )
    return expected


def _make_model_artifact(split, representation_artifact, learning_method,
                         model_name, model_input_representation, model_path,
                         manifest_path, signature, preprocessing_trace,
                         representation_trace):
    trace = {
        key: representation_trace.get(key)
        for key in TRACE_FIELDS
        if key not in preprocessing_trace
    }

    return ModelArtifact(
        split_id=split.id,
        representation_signature=representation_artifact.signature,
        learning_method=learning_method,
        model_name=model_name,
        model_path=str(model_path),
        manifest_path=str(manifest_path),
        signature=signature,
        model_input_representation=model_input_representation,
        **trace,
        **preprocessing_trace,
    )


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
        target_super_domain=_transform_group(data.target_super_domain, transformer),
        target_elementary_domain=_transform_group(data.target_elementary_domain, transformer),
    )


def _get_input_shape(data):
    for group in (data.source, data.target_super_domain, data.target_elementary_domain):
        if group is not None and len(group.X):
            return tuple(group.X.shape[1:])
    raise ValueError("No samples found.")


def _get_output_dim(data):
    labels = [
        group.y[group.partitions == "train"]
        for group in (data.source, data.target_super_domain, data.target_elementary_domain)
        if group is not None and np.any(group.partitions == "train")
    ]
    if not labels:
        raise ValueError("No training labels found.")
    return len(np.unique(np.concatenate(labels)))


def _get_output_paths(scenario, group, split_id, representation_signature,
                      name, signature):
    root = (OUTPUT_ROOT / scenario / group / split_id /
            representation_signature[:12] / f"{name}_{signature[:12]}")
    return root / "model.pkl", root / "manifest.json"


# ============================================================
# Image training
# ============================================================

def _prepare_image_training(source, representation_artifact, fit_params):
    from vision.lib.transforms import make_image_transforms

    if source is None:
        raise ValueError("Image training requires source data.")

    indices = np.flatnonzero(np.asarray(source.partitions) == "train")
    if not len(indices):
        raise ValueError("No source training images found.")

    _, class_to_idx = class_mapping(source.y[indices])

    config = (
        getattr(representation_artifact, "representation_params", None) or {}
    )
    image_config = config.get("image_transform") or {}
    image_size = int(image_config.get("params", {}).get("image_size", 224))

    train_transform, eval_transform = make_image_transforms(
        image_size=image_size
    )

    # ----------------------------------------------------
    # Source-only train/validation split
    # ----------------------------------------------------

    fraction = float(fit_params.get("validation_fraction", 0.1))
    seed = fit_params.get("seed", 42)

    train_local, val_local = split_indices(
        source.y[indices],
        validation_fraction=fraction,
        seed=seed,
        domains=source.elementary_domains[indices],
    )

    partitions = np.full(len(source.y), "unused", dtype=object)
    partitions[indices[train_local]] = "train"

    if val_local is not None:
        partitions[indices[val_local]] = "validation"

    group = DataGroup(
        X=source.X,
        y=source.y,
        elementary_domains=source.elementary_domains,
        partitions=partitions,
        super_domains=source.super_domains,
    )

    # ----------------------------------------------------
    # Shared DataLoaders
    # ----------------------------------------------------

    common = {
        "group": group,
        "class_to_idx": class_to_idx,
        "batch_size": int(fit_params.get("batch_size", 32)),
        "num_workers": int(fit_params.get("num_workers", 0)),
        "seed": seed,
    }

    train_loader = make_loader(
        **common,
        partition="train",
        transform=train_transform,
        shuffle=True,
    )

    val_loader = (
        make_loader(
            **common,
            partition="validation",
            transform=eval_transform,
        )
        if val_local is not None else None
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "class_to_idx": class_to_idx,
        "eval_transform": eval_transform,
    }, (3, image_size, image_size)


# ============================================================
# Main training
# ============================================================

def run_training(split, view, representation_artifact, training_params, group="default"):
    learning_method = training_params["learning"]
    model_name = training_params["model"]
    learning_params = training_params.get("learning_params", {})
    model_params = training_params.get("model_params", {})
    fit_params = training_params.get("training_params", {})
    name = training_params.get("name", f"{learning_method}_{model_name}")

    signature_fit_params = deepcopy(fit_params)
    signature_fit_params.pop("device", None)
    runtime = {"device": fit_params.get("device")}

    input_manifest = load_manifest(view.manifest_path)
    preprocessing_trace = _get_preprocessing_trace(view)
    representation_trace = _get_representation_trace(representation_artifact)
    output_representation = representation_trace.get("output_representation")
    model_input_representation = _validate_model_representation(
        model_name, output_representation
    )

    effective_params = {
        "split": split.to_dict(),
        "input_signature": input_manifest["signature"],
        "representation_signature": representation_artifact.signature,
        "representation_config_label": representation_trace.get("representation_config_label"),
        "output_representation": output_representation,
        **preprocessing_trace,
        "learning_method": learning_method,
        "learning_params": learning_params,
        "model": model_name,
        "model_params": model_params,
        "training_params": signature_fit_params,
    }

    signature = make_signature(effective_params)
    model_path, manifest_path = _get_output_paths(
        split.scenario, group, split.id,
        representation_artifact.signature, name, signature
    )

    def artifact():
        return _make_model_artifact(
            split, representation_artifact, learning_method, model_name,
            model_input_representation, model_path, manifest_path,
            signature, preprocessing_trace, representation_trace
        )

    if exists(model_path) and is_done(manifest_path, effective_params):
        return artifact()

    start = time.time()
    manifest = make_manifest("running", effective_params)
    manifest["runtime"] = runtime
    save_manifest(manifest, manifest_path)

    try:
        data = split.materialize(view)
        transformer = load_pickle(representation_artifact.transformer_path)
        data = _transform_data(data, transformer)

        model_context = {
            "input_shape": _get_input_shape(data),
            "output_dim": _get_output_dim(data),
        }

        # ----------------------------------------------------
        # Image-specific preparation
        # ----------------------------------------------------

        image_kwargs = {}

        if output_representation == "image":
            image_kwargs, model_context["input_shape"] = _prepare_image_training(
                data.source, representation_artifact, fit_params
            )

        # ----------------------------------------------------
        # Model training
        # ----------------------------------------------------

        model = get_model(model_name, model_params, **model_context)
        learner = get_learning_algorithm(learning_method, learning_params)

        learner.fit(
            model=model,
            source=data.source,
            target_super_domain=data.target_super_domain,
            target_elementary_domain=data.target_elementary_domain,
            **fit_params,
            **image_kwargs,
        )

        learner.save(model_path)

        # ----------------------------------------------------
        # Save successful results
        # ----------------------------------------------------

        manifest = make_manifest(
            "done", effective_params,
            execution_time=time.time() - start
        )
        manifest["runtime"] = runtime
        manifest["output"] = {
            "model_path": str(model_path),
            "model_context": _json_copy(model_context),
            "model_params": _json_copy(model_params),
            "representation_signature": representation_artifact.signature,
            "model_input_representation": model_input_representation,
            **representation_trace,
            **preprocessing_trace,
        }
        save_manifest(manifest, manifest_path)

    except Exception as error:
        manifest = make_manifest(
            "failed", effective_params,
            execution_time=time.time() - start,
            error=str(error)
        )
        manifest["runtime"] = runtime
        save_manifest(manifest, manifest_path)
        raise

    return artifact()
