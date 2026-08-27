import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from ml.domain_discrepancy.registry import (
    get_domain_discrepancy,
)

from models.domain_result import DomainResult
from models.domain_results_artifact import DomainResultsArtifact

from utils.storage import (
    exists,
    load_manifest,
    save_data,
    save_manifest,
)

from utils.status import (
    is_done,
    make_manifest,
    make_signature,
)


OUTPUT_ROOT = Path("outputs/domain_results")


# --------------------------------------------------
# Helper: comparison definitions
# --------------------------------------------------

def _iter_comparisons(data):
    """
    Yield all meaningful pairwise domain comparisons
    available in one ScenarioData object.
    """

    # source <-> final target domain
    if (
        data.source is not None
        and data.target_elementary_domain is not None
    ):
        yield (
            "source_to_target_elementary",
            "source",
            data.source,
            "target_elementary_domain",
            data.target_elementary_domain,
        )

    # source <-> target super-domain
    if (
        data.source is not None
        and data.target_super_domain is not None
    ):
        yield (
            "source_to_target_super",
            "source",
            data.source,
            "target_super_domain",
            data.target_super_domain,
        )

    # target super-domain <-> final target domain
    if (
        data.target_super_domain is not None
        and data.target_elementary_domain is not None
    ):
        yield (
            "target_super_to_target_elementary",
            "target_super_domain",
            data.target_super_domain,
            "target_elementary_domain",
            data.target_elementary_domain,
        )


# --------------------------------------------------
# Helper: domain identity
# --------------------------------------------------

def _domain_identity(group):
    """
    Recover elementary-domain identities represented
    inside a DataGroup.
    """

    domains = sorted(
        str(domain)
        for domain in np.unique(
            group.elementary_domains
        )
    )

    return (
        ";".join(domains),
        len(domains),
    )


# --------------------------------------------------
# Helper: standardization
# --------------------------------------------------

def _standardize_pair(
    X_left,
    X_right,
):
    """
    Standardize both groups using statistics from
    their pooled feature space.

    The same transformation is used for both sides.
    """

    X_left = np.asarray(
        X_left,
        dtype=float,
    )

    X_right = np.asarray(
        X_right,
        dtype=float,
    )

    pooled = np.vstack([
        X_left,
        X_right,
    ])

    mean = pooled.mean(
        axis=0,
    )

    std = pooled.std(
        axis=0,
    )

    std = np.where(
        std < 1e-12,
        1.0,
        std,
    )

    return (
        (X_left - mean) / std,
        (X_right - mean) / std,
    )


# --------------------------------------------------
# Helper: metric configurations
# --------------------------------------------------

def _normalize_metrics(metrics):
    """
    Accept either:

    ["mmd", "energy"]

    or:

    [
        {"name": "mmd", "params": {...}},
        {"name": "energy", "params": {}},
    ]
    """

    normalized = []

    for metric in metrics:

        if isinstance(
            metric,
            str,
        ):
            normalized.append({
                "name": metric,
                "params": {},
            })

        else:
            normalized.append({
                "name": metric["name"],
                "params": metric.get(
                    "params",
                    {},
                ),
            })

    return normalized


# --------------------------------------------------
# Helper: one discrepancy calculation
# --------------------------------------------------

def _compute_result(
    metric_name,
    metric_params,
    X_left,
    X_right,
):
    """
    Compute one configured discrepancy.
    """

    metric = get_domain_discrepancy(
        metric_name
    )

    return float(
        metric(
            X_left,
            X_right,
            **metric_params,
        )
    )


# --------------------------------------------------
# Helper: evaluate one domain pair
# --------------------------------------------------

def _evaluate_pair(
    split,
    scenario,
    group_name,
    comparison,
    left_group_name,
    left_group,
    right_group_name,
    right_group,
    metric_configs,
    representations,
    standardize,
    min_samples_per_side,
):
    """
    Generate all DomainResult rows for one pair
    of domain groups.
    """

    rows = []

    X_left = np.asarray(
        left_group.X
    )

    X_right = np.asarray(
        right_group.X
    )

    y_left = np.asarray(
        left_group.y
    )

    y_right = np.asarray(
        right_group.y
    )

    # --------------------------------------------------
    # Domain identities
    # --------------------------------------------------

    (
        left_domains,
        n_left_domains,
    ) = _domain_identity(
        left_group
    )

    (
        right_domains,
        n_right_domains,
    ) = _domain_identity(
        right_group
    )

    # --------------------------------------------------
    # Optional common scaling
    # --------------------------------------------------

    if standardize:
        X_left, X_right = (
            _standardize_pair(
                X_left,
                X_right,
            )
        )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    for metric_config in metric_configs:

        metric_name = (
            metric_config["name"]
        )

        metric_params = (
            metric_config["params"]
        )

        metric_signature = (
            make_signature({
                "metric": metric_name,
                "params": metric_params,
                "standardize": standardize,
            })
        )

        # --------------------------------------------------
        # Marginal discrepancy
        # --------------------------------------------------

        if "marginal" in representations:

            if (
                len(X_left)
                >= min_samples_per_side
                and len(X_right)
                >= min_samples_per_side
            ):
                value = _compute_result(
                    metric_name,
                    metric_params,
                    X_left,
                    X_right,
                )

                rows.append(
                    DomainResult(
                        split_id=split.id,
                        scenario=scenario,
                        group=group_name,
                        seed=split.seed,

                        target_fraction=(
                            split.target_fraction
                        ),

                        comparison=comparison,

                        left_group=(
                            left_group_name
                        ),

                        right_group=(
                            right_group_name
                        ),

                        left_domains=(
                            left_domains
                        ),

                        right_domains=(
                            right_domains
                        ),

                        n_left_domains=(
                            n_left_domains
                        ),

                        n_right_domains=(
                            n_right_domains
                        ),

                        metric=metric_name,

                        metric_signature=(
                            metric_signature
                        ),

                        representation=(
                            "marginal"
                        ),

                        class_label=None,

                        value=value,

                        n_left_samples=len(
                            X_left
                        ),

                        n_right_samples=len(
                            X_right
                        ),
                    ).to_dict()
                )

        # --------------------------------------------------
        # Class-specific discrepancy
        # --------------------------------------------------

        if "class" in representations:

            common_classes = sorted(
                set(
                    np.unique(
                        y_left
                    )
                ).intersection(
                    np.unique(
                        y_right
                    )
                ),
                key=str,
            )

            for class_label in (
                common_classes
            ):

                left_mask = (
                    y_left == class_label
                )

                right_mask = (
                    y_right == class_label
                )

                X_left_class = (
                    X_left[left_mask]
                )

                X_right_class = (
                    X_right[right_mask]
                )

                if (
                    len(X_left_class)
                    < min_samples_per_side
                    or len(X_right_class)
                    < min_samples_per_side
                ):
                    continue

                value = _compute_result(
                    metric_name,
                    metric_params,
                    X_left_class,
                    X_right_class,
                )

                rows.append(
                    DomainResult(
                        split_id=split.id,
                        scenario=scenario,
                        group=group_name,
                        seed=split.seed,

                        target_fraction=(
                            split.target_fraction
                        ),

                        comparison=comparison,

                        left_group=(
                            left_group_name
                        ),

                        right_group=(
                            right_group_name
                        ),

                        left_domains=(
                            left_domains
                        ),

                        right_domains=(
                            right_domains
                        ),

                        n_left_domains=(
                            n_left_domains
                        ),

                        n_right_domains=(
                            n_right_domains
                        ),

                        metric=metric_name,

                        metric_signature=(
                            metric_signature
                        ),

                        representation=(
                            "class"
                        ),

                        class_label=str(
                            class_label
                        ),

                        value=value,

                        n_left_samples=len(
                            X_left_class
                        ),

                        n_right_samples=len(
                            X_right_class
                        ),
                    ).to_dict()
                )

    return rows


# --------------------------------------------------
# Public pipeline
# --------------------------------------------------

def run_domain_evaluation(
    scenario_artifacts,
    params=None,
):
    """
    Evaluate domain discrepancy across all scenario
    splits and generate the canonical domain-results
    table.
    """

    params = params or {}

    metric_configs = (
        _normalize_metrics(
            params.get(
                "metrics",
                ["mmd", "energy"],
            )
        )
    )

    representations = (
        params.get(
            "representations",
            [
                "marginal",
                "class",
            ],
        )
    )

    standardize = params.get(
        "standardize",
        True,
    )

    min_samples_per_side = (
        params.get(
            "min_samples_per_side",
            2,
        )
    )

    # --------------------------------------------------
    # Input identity
    # --------------------------------------------------

    inputs = []

    for scenario, artifacts in (
        scenario_artifacts.items()
    ):

        for artifact in artifacts:

            view = artifact["view"]

            input_manifest = (
                load_manifest(
                    view.manifest_path
                )
            )

            input_signature = (
                input_manifest.get(
                    "signature"
                )
            )

            for split in artifact[
                "splits"
            ]:
                inputs.append({
                    "scenario": scenario,
                    "group": artifact[
                        "group"
                    ],
                    "split": (
                        split.to_dict()
                    ),
                    "input_signature": (
                        input_signature
                    ),
                })

    effective_params = {
        "inputs": inputs,
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
        / "domain_results.csv"
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

        return DomainResultsArtifact(
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
    # Execute
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
        # Scenarios
        # --------------------------------------------------

        for scenario, artifacts in (
            scenario_artifacts.items()
        ):

            for artifact in artifacts:

                group_name = artifact[
                    "group"
                ]

                view = artifact[
                    "view"
                ]

                # ------------------------------------------
                # Splits
                # ------------------------------------------

                for split in artifact[
                    "splits"
                ]:

                    data = (
                        split.materialize(
                            view
                        )
                    )

                    # --------------------------------------
                    # Valid pairwise comparisons
                    # --------------------------------------

                    for (
                        comparison,
                        left_group_name,
                        left_group,
                        right_group_name,
                        right_group,
                    ) in _iter_comparisons(
                        data
                    ):

                        pair_rows = (
                            _evaluate_pair(
                                split=split,
                                scenario=scenario,
                                group_name=group_name,
                                comparison=comparison,
                                left_group_name=(
                                    left_group_name
                                ),
                                left_group=(
                                    left_group
                                ),
                                right_group_name=(
                                    right_group_name
                                ),
                                right_group=(
                                    right_group
                                ),
                                metric_configs=(
                                    metric_configs
                                ),
                                representations=(
                                    representations
                                ),
                                standardize=(
                                    standardize
                                ),
                                min_samples_per_side=(
                                    min_samples_per_side
                                ),
                            )
                        )

                        rows.extend(
                            pair_rows
                        )

        # --------------------------------------------------
        # Canonical table
        # --------------------------------------------------

        column_names = [
            field.name
            for field in fields(
                DomainResult
            )
        ]

        dataframe = pd.DataFrame(
            rows,
            columns=column_names,
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
    # Return aggregate artifact
    # --------------------------------------------------

    return DomainResultsArtifact(
        path=str(
            results_path
        ),
        manifest_path=str(
            manifest_path
        ),
        signature=signature,
        n_rows=len(rows),
    )