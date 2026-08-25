import time
from pathlib import Path

import numpy as np

from ml.models.registry import get_model
from ml.learning.registry import get_learning_algorithm

from models.model_artifact import ModelArtifact
from models.scenario_data import DataGroup, ScenarioData

from utils.storage import (
    exists,
    load_manifest,
    load_pickle,
    save_manifest,
)
from utils.status import (
    is_done,
    make_manifest,
    make_signature,
)


OUTPUT_ROOT = Path("outputs/training")


def _transform_group(group, transformer):
    if group is None:
        return None

    return DataGroup(
        X=transformer.transform(
            group.X,
            group.elementary_domains,
        ),
        y=group.y,
        elementary_domains=group.elementary_domains,
        partitions=group.partitions,
        super_domains=group.super_domains,
    )


def _transform_data(data, transformer):
    return ScenarioData(
        source=_transform_group(
            data.source,
            transformer,
        ),
        target_super_domain=_transform_group(
            data.target_super_domain,
            transformer,
        ),
        target_elementary_domain=_transform_group(
            data.target_elementary_domain,
            transformer,
        ),
    )


def _get_input_dim(data):
    for group in [
        data.source,
        data.target_super_domain,
        data.target_elementary_domain,
    ]:
        if group is not None and len(group.X) > 0:
            return group.X.shape[1]

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
        raise ValueError(
            "No training labels found."
        )

    y = np.concatenate(labels)

    return len(np.unique(y))


def _prepare_model_params(
    model_name,
    model_params,
    data,
):
    params = dict(model_params)

    if model_name == "mlp":
        params.setdefault(
            "input_dim",
            _get_input_dim(data),
        )
        params.setdefault(
            "output_dim",
            _get_output_dim(data),
        )

    return params


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

    model_path = output_dir / "model.pkl"
    manifest_path = output_dir / "manifest.json"

    return model_path, manifest_path


def run_training(
    split,
    view,
    fs_artifact,
    training_params,
    group="default",
):
    learning_method = training_params["learning"]
    model_name = training_params["model"]

    learning_params = training_params.get(
        "learning_params",
        {},
    )

    model_params = training_params.get(
        "model_params",
        {},
    )

    fit_params = training_params.get(
        "training_params",
        {},
    )

    name = training_params.get(
        "name",
        f"{learning_method}_{model_name}",
    )

    input_manifest = load_manifest(
        view.manifest_path
    )

    effective_params = {
        "split": split.to_dict(),
        "input_signature": input_manifest["signature"],
        "feature_selection_signature": fs_artifact.signature,
        "learning_method": learning_method,
        "learning_params": learning_params,
        "model": model_name,
        "model_params": model_params,
        "training_params": fit_params,
    }

    signature = make_signature(
        effective_params
    )

    model_path, manifest_path = (
        _get_output_paths(
            split.scenario,
            group,
            split.id,
            fs_artifact.signature,
            name,
            signature,
        )
    )

    if (
        exists(model_path)
        and is_done(
            manifest_path,
            effective_params,
        )
    ):
        return ModelArtifact(
            split_id=split.id,
            feature_selection_signature=fs_artifact.signature,
            learning_method=learning_method,
            model_name=model_name,
            model_path=str(model_path),
            manifest_path=str(manifest_path),
            signature=signature,
        )

    start = time.time()

    save_manifest(
        make_manifest(
            "running",
            effective_params,
        ),
        manifest_path,
    )

    try:
        # ----------------------------------------
        # Materialize scenario
        # ----------------------------------------

        data = split.materialize(view)

        # ----------------------------------------
        # Apply fitted feature transformation
        # ----------------------------------------

        transformer = load_pickle(
            fs_artifact.transformer_path
        )

        data = _transform_data(
            data,
            transformer,
        )

        # ----------------------------------------
        # Create model
        # ----------------------------------------

        resolved_model_params = (
            _prepare_model_params(
                model_name,
                model_params,
                data,
            )
        )

        model = get_model(
            model_name,
            resolved_model_params,
        )

        # ----------------------------------------
        # Create learning algorithm
        # ----------------------------------------

        learner = get_learning_algorithm(
            learning_method,
            learning_params,
        )

        # ----------------------------------------
        # Train
        # ----------------------------------------

        learner.fit(
            model=model,
            source=data.source,
            target_super_domain=(
                data.target_super_domain
            ),
            target_elementary_domain=(
                data.target_elementary_domain
            ),
            **fit_params,
        )

        # Save the complete fitted learning system.
        learner.save(model_path)

        execution_time = (
            time.time() - start
        )

        manifest = make_manifest(
            "done",
            effective_params,
            execution_time=execution_time,
        )

        manifest["output"] = {
            "model_path": str(model_path),
            "resolved_model_params": (
                resolved_model_params
            ),
        }

        save_manifest(
            manifest,
            manifest_path,
        )

    except Exception as error:

        execution_time = (
            time.time() - start
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

    return ModelArtifact(
        split_id=split.id,
        feature_selection_signature=fs_artifact.signature,
        learning_method=learning_method,
        model_name=model_name,
        model_path=str(model_path),
        manifest_path=str(manifest_path),
        signature=signature,
    )