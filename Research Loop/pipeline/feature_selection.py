# pipeline/feature_selection.py

import time
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
# Fitting data
# ============================================================

def _get_fit_data(data, fit_partitions):
    """
    Collect samples allowed to fit the feature transformer.
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
            "No samples available to fit feature selection."
        )

    return (
        np.concatenate(X_parts),
        np.concatenate(y_parts),
        np.concatenate(domain_parts),
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
# Public function
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

    name = fs_params.get(
        "name",
        method,
    )

    method_params = fs_params.get(
        "params",
        {},
    )

    fit_partitions = fs_params.get(
        "fit_partitions",
        ["train"],
    )

    # --------------------------------------------------
    # Signature
    # --------------------------------------------------

    input_manifest = load_manifest(
        view.manifest_path
    )

    effective_params = {
        "split": split.to_dict(),
        "input_signature": input_manifest["signature"],
        "method": method,
        "params": method_params,
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

    # --------------------------------------------------
    # Resume
    # --------------------------------------------------

    if (
        exists(transformer_path)
        and is_done(
            manifest_path,
            effective_params,
        )
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
        )

    # --------------------------------------------------
    # Running manifest
    # --------------------------------------------------

    save_manifest(
        make_manifest(
            status="running",
            params=effective_params,
        ),
        manifest_path,
    )

    start = time.perf_counter()

    try:

        # --------------------------------------------------
        # Materialize split
        # --------------------------------------------------

        data = split.materialize(
            view
        )

        X, y, domains = _get_fit_data(
            data,
            fit_partitions,
        )

        # --------------------------------------------------
        # Fit transformer
        # --------------------------------------------------

        transformer = get_feature_transformer(
            method,
            method_params,
        )

        transformer.fit(
            X,
            y,
            domains,
        )

        # --------------------------------------------------
        # Save transformer
        # --------------------------------------------------

        save_pickle(
            transformer,
            transformer_path,
        )

        execution_time = (
            time.perf_counter()
            - start
        )

        manifest = make_manifest(
            status="done",
            params=effective_params,
            execution_time=execution_time,
        )

        manifest["output"] = {
            "transformer_path": str(
                transformer_path
            ),
            "n_input_features": int(
                X.shape[1]
            ),
            "n_fit_samples": int(
                X.shape[0]
            ),
        }

        save_manifest(
            manifest,
            manifest_path,
        )

    except Exception as error:

        execution_time = (
            time.perf_counter()
            - start
        )

        save_manifest(
            make_manifest(
                status="failed",
                params=effective_params,
                execution_time=execution_time,
                error=str(error),
            ),
            manifest_path,
        )

        raise

    # --------------------------------------------------
    # Artifact
    # --------------------------------------------------

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
    )