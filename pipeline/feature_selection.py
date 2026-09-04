# pipeline/feature_selection.py

import json
import re
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from ml.feature_selection.registry import get_feature_transformer
from models.feature_selection_artifact import FeatureSelectionArtifact

from utils.storage import (
    exists,
    load_manifest,
    save_manifest,
    save_pickle,
)
from utils.status import (
    is_done,
    make_manifest,
    make_signature,
)


OUTPUT_ROOT = Path("outputs/feature_selection")


# ============================================================
# Config labels
# ============================================================

def _safe_label(text):
    text = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(text),
    )

    return text.strip("_")


def _params_label(params):
    if not params:
        return ""

    parts = []

    for key in sorted(params):
        value = params[key]

        if isinstance(value, float):
            value = f"{value:g}"

        parts.append(
            f"{key}_{value}"
        )

    return "_".join(parts)


def _feature_selection_config_label(
    fs_params,
    method,
    method_params,
):
    if "feature_selection_config_label" in fs_params:
        return _safe_label(
            fs_params["feature_selection_config_label"]
        )

    if "config_label" in fs_params:
        return _safe_label(
            fs_params["config_label"]
        )

    if (
        "name" in fs_params
        and fs_params["name"] != method
    ):
        return _safe_label(
            fs_params["name"]
        )

    params_text = _params_label(
        method_params
    )

    if params_text:
        return _safe_label(
            f"{method}_{params_text}"
        )

    return _safe_label(method)


def _json_copy(value):
    try:
        return json.loads(
            json.dumps(
                value,
                default=str,
            )
        )

    except TypeError:
        return deepcopy(value)


# ============================================================
# Fitting data
# ============================================================

def _get_fit_data(
    data,
    fit_partitions,
):
    """
    Collect samples allowed to fit the transformer.

    X may be:
        [samples, features]
        [samples, channels, time]
        [samples, bands, channels, time]
    """

    X_parts = []
    y_parts = []
    domain_parts = []

    groups = [
        data.source,
        data.target_super_domain,
        data.target_elementary_domain,
    ]

    for group in groups:

        if group is None:
            continue

        mask = np.isin(
            group.partitions,
            fit_partitions,
        )

        if not np.any(mask):
            continue

        X_parts.append(
            group.X[mask]
        )

        y_parts.append(
            group.y[mask]
        )

        domain_parts.append(
            group.elementary_domains[mask]
        )

    if not X_parts:
        raise ValueError(
            "No samples available to fit feature transformation."
        )

    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(domain_parts, axis=0),
    )


# ============================================================
# Representation compatibility
# ============================================================

def _validate_representation(
    transformer,
    representation,
):
    """
    Check whether the transformer accepts the current input.

    Existing transformers default to feature input until they
    explicitly declare otherwise.
    """

    expected = getattr(
        transformer,
        "input_representation",
        "features",
    )

    if expected == "any":
        return

    if expected != representation:
        raise ValueError(
            f"Transformer expects '{expected}' input, "
            f"but preprocessing produced "
            f"'{representation}'."
        )


# ============================================================
# Paths
# ============================================================

def _get_output_paths(
    scenario,
    group,
    split_id,
    name,
    signature,
):
    output_dir = (
        OUTPUT_ROOT
        / scenario
        / group
        / split_id
        / f"{name}_{signature[:12]}"
    )

    return (
        output_dir / "transformer.pkl",
        output_dir / "manifest.json",
    )


# ============================================================
# Artifact
# ============================================================

def _make_feature_selection_artifact(
    split,
    transformer_path,
    manifest_path,
    signature,
    method,
    method_params,
    config_label,
    view,
):
    return FeatureSelectionArtifact(
        split_id=split.id,
        method=method,

        transformer_path=str(
            transformer_path
        ),

        manifest_path=str(
            manifest_path
        ),

        signature=signature,

        feature_selection_method=method,

        feature_selection_params=_json_copy(
            method_params
        ),

        feature_selection_config_label=config_label,

        preprocessing_signature=getattr(
            view,
            "preprocessing_signature",
            None,
        ),

        preprocessing_config_label=getattr(
            view,
            "preprocessing_config_label",
            None,
        ),
    )


# ============================================================
# Public
# ============================================================

def run_feature_selection(
    split,
    view,
    fs_params,
    group="default",
):
    """
    Fit or reuse one feature transformer for one ScenarioSplit.
    """

    method = fs_params["method"]

    method_params = fs_params.get(
        "params",
        {},
    )

    config_label = _feature_selection_config_label(
        fs_params,
        method,
        method_params,
    )

    name = fs_params.get(
        "name",
        config_label,
    )

    fit_partitions = fs_params.get(
        "fit_partitions",
        ["train"],
    )

    representation = getattr(
        view,
        "representation",
        "features",
    )

    # --------------------------------------------------------
    # Signature
    # --------------------------------------------------------

    input_manifest = load_manifest(
        view.manifest_path
    )

    effective_params = {
        "split": split.to_dict(),
        "input_signature": input_manifest["signature"],
        "input_representation": representation,
        "method": method,
        "params": method_params,
        "feature_selection_config_label": config_label,
        "fit_partitions": fit_partitions,
    }

    signature = make_signature(
        effective_params
    )

    transformer_path, manifest_path = (
        _get_output_paths(
            scenario=split.scenario,
            group=group,
            split_id=split.id,
            name=name,
            signature=signature,
        )
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if (
        exists(transformer_path)
        and is_done(
            manifest_path,
            effective_params,
        )
    ):
        return _make_feature_selection_artifact(
            split,
            transformer_path,
            manifest_path,
            signature,
            method,
            method_params,
            config_label,
            view,
        )

    save_manifest(
        make_manifest(
            status="running",
            params=effective_params,
        ),
        manifest_path,
    )

    start = time.perf_counter()

    try:

        # ----------------------------------------------------
        # Materialize split
        # ----------------------------------------------------

        data = split.materialize(
            view
        )

        X, y, domains = _get_fit_data(
            data,
            fit_partitions,
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        transformer = get_feature_transformer(
            method,
            method_params,
        )

        _validate_representation(
            transformer,
            representation,
        )

        transformer.fit(
            X,
            y,
            domains,
        )

        save_pickle(
            transformer,
            transformer_path,
        )

        execution_time = (
            time.perf_counter()
            - start
        )

        # ----------------------------------------------------
        # Manifest
        # ----------------------------------------------------

        manifest = make_manifest(
            status="done",
            params=effective_params,
            execution_time=execution_time,
        )

        manifest["output"] = {
            "transformer_path": str(
                transformer_path
            ),

            "input_representation": representation,

            "input_shape": [
                int(value)
                for value in X.shape[1:]
            ],

            "n_fit_samples": int(
                X.shape[0]
            ),

            "feature_selection_method": method,

            "feature_selection_params": _json_copy(
                method_params
            ),

            "feature_selection_config_label": config_label,

            "preprocessing_signature": getattr(
                view,
                "preprocessing_signature",
                None,
            ),

            "preprocessing_config_label": getattr(
                view,
                "preprocessing_config_label",
                None,
            ),
        }

        save_manifest(
            manifest,
            manifest_path,
        )

    except Exception as error:

        save_manifest(
            make_manifest(
                status="failed",
                params=effective_params,
                execution_time=(
                    time.perf_counter()
                    - start
                ),
                error=str(error),
            ),
            manifest_path,
        )

        raise

    return _make_feature_selection_artifact(
        split,
        transformer_path,
        manifest_path,
        signature,
        method,
        method_params,
        config_label,
        view,
    )