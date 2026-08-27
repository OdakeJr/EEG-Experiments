# pipeline/analysis/domain_variability_qualitative.py

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from models.analysis_artifact import (
    AnalysisArtifact,
)

from utils.storage import (
    load_data,
    load_manifest,
    save_manifest,
)

from utils.status import (
    is_done,
    make_manifest,
    make_signature,
)


OUTPUT_ROOT = Path(
    "outputs/analysis/domain_variability_qualitative"
)


# ============================================================
# UMAP
# ============================================================

def _get_umap():

    try:
        from umap import UMAP

    except ImportError as error:
        raise ImportError(
            "UMAP is required for the qualitative "
            "domain-variability analysis. Install "
            "'umap-learn' in the active environment."
        ) from error

    return UMAP


# ============================================================
# Load standardized datasets
# ============================================================

def _load_views(
    dataset_views,
):
    """
    Load all standardized datasets.

    Expected input:

        {
            "bci2a": DatasetView,
            "eegmmidb": DatasetView,
            "weibo": DatasetView,
            "zhou": DatasetView,
        }
    """

    datasets = {}

    for name, view in (
        dataset_views.items()
    ):

        dataframe = load_data(
            view.path
        )

        datasets[name] = {
            "view": view,
            "data": dataframe,
        }

    return datasets


# ============================================================
# Feature compatibility
# ============================================================

def _validate_feature_compatibility(
    datasets,
):
    """
    Cross-dataset projections require all datasets to
    share the same feature representation.
    """

    reference_features = None

    for name, item in (
        datasets.items()
    ):

        features = list(
            item["view"].feature_columns
        )

        if reference_features is None:
            reference_features = features
            continue

        if features != reference_features:
            raise ValueError(
                "Qualitative cross-dataset visualization "
                "requires compatible feature columns. "
                f"Dataset '{name}' has a different "
                "feature representation."
            )


# ============================================================
# General helpers
# ============================================================

def _sorted_unique(
    values,
):

    return sorted(
        pd.Series(
            values
        )
        .dropna()
        .unique(),
        key=str,
    )


def _filter_classes(
    dataframe,
    classes,
):

    return dataframe[
        dataframe["label"].isin(
            classes
        )
    ].copy()


# ============================================================
# Balanced sampling
# ============================================================

def _sample_per_class(
    dataframe,
    classes,
    n_per_class,
    seed,
):
    """
    Sample approximately the same number of trials
    from every requested class.
    """

    rng = np.random.default_rng(
        seed
    )

    sampled = []

    for class_label in classes:

        subset = dataframe[
            dataframe["label"]
            == class_label
        ]

        if subset.empty:
            continue

        n_samples = min(
            len(subset),
            n_per_class,
        )

        indices = rng.choice(
            subset.index.to_numpy(),
            size=n_samples,
            replace=False,
        )

        sampled.append(
            subset.loc[
                indices
            ]
        )

    if not sampled:

        return pd.DataFrame(
            columns=dataframe.columns
        )

    return pd.concat(
        sampled,
        ignore_index=True,
    )


# ============================================================
# Cross-subject
#
# Four independent subject-pair comparisons.
# ============================================================

def _select_subject_pairs(
    dataframe,
    params,
):

    requested_pairs = params.get(
        "pairs"
    )

    available_subjects = (
        _sorted_unique(
            dataframe["subject"]
        )
    )

    if requested_pairs is not None:

        return [
            tuple(pair)
            for pair in requested_pairs
        ]

    n_pairs = params.get(
        "n_pairs",
        4,
    )

    required_subjects = (
        2 * n_pairs
    )

    if len(
        available_subjects
    ) < required_subjects:

        raise ValueError(
            "Not enough subjects for "
            f"{n_pairs} cross-subject pairs."
        )

    selected = (
        available_subjects[
            :required_subjects
        ]
    )

    return [
        (
            selected[index],
            selected[index + 1],
        )
        for index in range(
            0,
            required_subjects,
            2,
        )
    ]


def _prepare_cross_subject_panels(
    dataset,
    params,
    classes,
    seed,
):

    dataframe = dataset[
        "data"
    ]

    view = dataset[
        "view"
    ]

    dataframe = _filter_classes(
        dataframe,
        classes,
    )

    pairs = _select_subject_pairs(
        dataframe,
        params,
    )

    n_per_class = params.get(
        "samples_per_subject_class",
        60,
    )

    panels = []

    for panel_index, (
        subject_left,
        subject_right,
    ) in enumerate(
        pairs
    ):

        left = dataframe[
            dataframe["subject"]
            == subject_left
        ]

        right = dataframe[
            dataframe["subject"]
            == subject_right
        ]

        left = _sample_per_class(
            left,
            classes,
            n_per_class,
            seed + panel_index * 10,
        )

        right = _sample_per_class(
            right,
            classes,
            n_per_class,
            seed + panel_index * 10 + 1,
        )

        if (
            left.empty
            or right.empty
        ):
            continue

        combined = pd.concat(
            [
                left.assign(
                    plot_domain=str(
                        subject_left
                    )
                ),

                right.assign(
                    plot_domain=str(
                        subject_right
                    )
                ),
            ],
            ignore_index=True,
        )

        panels.append({

            "title": (
                f"{subject_left} vs "
                f"{subject_right}"
            ),

            "data": combined,

            "features": list(
                view.feature_columns
            ),

            "class_label": None,

            "selection": [
                {
                    "scenario": (
                        "cross_subject"
                    ),

                    "dataset": params[
                        "dataset"
                    ],

                    "panel": panel_index,

                    "subject": (
                        subject_left
                    ),

                    "session": None,

                    "n_samples": len(
                        left
                    ),
                },

                {
                    "scenario": (
                        "cross_subject"
                    ),

                    "dataset": params[
                        "dataset"
                    ],

                    "panel": panel_index,

                    "subject": (
                        subject_right
                    ),

                    "session": None,

                    "n_samples": len(
                        right
                    ),
                },
            ],
        })

    return panels


# ============================================================
# Cross-session
#
# Four subjects, with two sessions compared within each.
# ============================================================

def _subjects_with_multiple_sessions(
    dataframe,
):

    counts = (
        dataframe
        .groupby(
            "subject"
        )[
            "session"
        ]
        .nunique()
    )

    return sorted(
        counts[
            counts >= 2
        ].index.tolist(),
        key=str,
    )


def _prepare_cross_session_panels(
    dataset,
    params,
    classes,
    seed,
):

    dataframe = dataset[
        "data"
    ]

    view = dataset[
        "view"
    ]

    dataframe = _filter_classes(
        dataframe,
        classes,
    )

    subjects = params.get(
        "subjects"
    )

    if subjects is None:

        candidates = (
            _subjects_with_multiple_sessions(
                dataframe
            )
        )

        n_subjects = params.get(
            "n_subjects",
            4,
        )

        subjects = candidates[
            :n_subjects
        ]

    if len(subjects) < 1:

        raise ValueError(
            "No subjects with multiple sessions "
            "were found."
        )

    requested_sessions = params.get(
        "sessions"
    )

    n_per_class = params.get(
        "samples_per_session_class",
        60,
    )

    panels = []

    for panel_index, subject in enumerate(
        subjects
    ):

        subject_data = dataframe[
            dataframe["subject"]
            == subject
        ]

        available_sessions = (
            _sorted_unique(
                subject_data[
                    "session"
                ]
            )
        )

        if requested_sessions is None:

            if len(
                available_sessions
            ) < 2:
                continue

            session_left = (
                available_sessions[0]
            )

            session_right = (
                available_sessions[1]
            )

        else:

            session_left = (
                requested_sessions[0]
            )

            session_right = (
                requested_sessions[1]
            )

        left = subject_data[
            subject_data["session"]
            == session_left
        ]

        right = subject_data[
            subject_data["session"]
            == session_right
        ]

        if (
            left.empty
            or right.empty
        ):
            continue

        left = _sample_per_class(
            left,
            classes,
            n_per_class,
            seed + panel_index * 10,
        )

        right = _sample_per_class(
            right,
            classes,
            n_per_class,
            seed + panel_index * 10 + 1,
        )

        if (
            left.empty
            or right.empty
        ):
            continue

        combined = pd.concat(
            [
                left.assign(
                    plot_domain=str(
                        session_left
                    )
                ),

                right.assign(
                    plot_domain=str(
                        session_right
                    )
                ),
            ],
            ignore_index=True,
        )

        panels.append({

            "title": str(
                subject
            ),

            "data": combined,

            "features": list(
                view.feature_columns
            ),

            "class_label": None,

            "selection": [
                {
                    "scenario": (
                        "cross_session"
                    ),

                    "dataset": params[
                        "dataset"
                    ],

                    "panel": panel_index,

                    "subject": subject,

                    "session": (
                        session_left
                    ),

                    "n_samples": len(
                        left
                    ),
                },

                {
                    "scenario": (
                        "cross_session"
                    ),

                    "dataset": params[
                        "dataset"
                    ],

                    "panel": panel_index,

                    "subject": subject,

                    "session": (
                        session_right
                    ),

                    "n_samples": len(
                        right
                    ),
                },
            ],
        })

    if not panels:

        raise ValueError(
            "No valid cross-session panels "
            "could be generated."
        )

    return panels


# ============================================================
# Cross-dataset
#
# Four panels:
#
#   1. All classes
#   2. Left hand only
#   3. Right hand only
#   4. Feet only
#
# Dataset is always represented by color.
# ============================================================

def _prepare_cross_dataset_panels(
    datasets,
    params,
    classes,
    seed,
):

    n_subjects = params.get(
        "subjects_per_dataset",
        4,
    )

    explicit_subjects = params.get(
        "subjects",
        {},
    )

    n_per_subject_class = params.get(
        "samples_per_subject_class",
        20,
    )

    all_data = []
    selection = []

    feature_columns = None

    # --------------------------------------------------
    # Sample each dataset
    # --------------------------------------------------

    for dataset_index, (
        dataset_name,
        item,
    ) in enumerate(
        datasets.items()
    ):

        dataframe = _filter_classes(
            item["data"],
            classes,
        )

        view = item[
            "view"
        ]

        if feature_columns is None:

            feature_columns = list(
                view.feature_columns
            )

        # --------------------------------------------------
        # Subjects
        # --------------------------------------------------

        if dataset_name in explicit_subjects:

            subjects = explicit_subjects[
                dataset_name
            ]

        else:

            subjects = (
                _sorted_unique(
                    dataframe[
                        "subject"
                    ]
                )[
                    :n_subjects
                ]
            )

        dataset_parts = []

        # --------------------------------------------------
        # Balanced subject / class sampling
        # --------------------------------------------------

        for subject_index, subject in enumerate(
            subjects
        ):

            subject_data = dataframe[
                dataframe["subject"]
                == subject
            ]

            sampled = _sample_per_class(
                subject_data,
                classes,
                n_per_subject_class,
                (
                    seed
                    + dataset_index * 1000
                    + subject_index
                ),
            )

            if sampled.empty:
                continue

            dataset_parts.append(
                sampled
            )

            selection.append({
                "scenario": (
                    "cross_dataset"
                ),

                "dataset": (
                    dataset_name
                ),

                "panel": "all",

                "subject": (
                    subject
                ),

                "session": None,

                "n_samples": len(
                    sampled
                ),
            })

        if not dataset_parts:
            continue

        dataset_data = pd.concat(
            dataset_parts,
            ignore_index=True,
        )

        dataset_data[
            "plot_domain"
        ] = dataset_name

        all_data.append(
            dataset_data
        )

    if len(
        all_data
    ) < 2:

        raise ValueError(
            "Cross-dataset visualization "
            "requires at least two datasets."
        )

    combined = pd.concat(
        all_data,
        ignore_index=True,
    )

    # --------------------------------------------------
    # All-class panel
    # --------------------------------------------------

    panels = [
        {
            "title": (
                "All classes"
            ),

            "data": combined,

            "features": (
                feature_columns
            ),

            "class_label": None,

            "selection": (
                selection
            ),
        }
    ]

    # --------------------------------------------------
    # Individual class panels
    # --------------------------------------------------

    class_titles = {
        "left_hand_imagery": (
            "Left hand"
        ),

        "right_hand_imagery": (
            "Right hand"
        ),

        "both_feet_imagery": (
            "Feet"
        ),
    }

    for class_label in classes:

        class_data = combined[
            combined["label"]
            == class_label
        ].copy()

        if class_data.empty:
            continue

        panels.append({

            "title": (
                class_titles.get(
                    class_label,
                    str(class_label),
                )
            ),

            "data": (
                class_data
            ),

            "features": (
                feature_columns
            ),

            "class_label": (
                class_label
            ),

            "selection": [],
        })

    return panels


# ============================================================
# Projection
#
# Every panel gets its own joint projection.
# ============================================================

def _project(
    panel,
    method,
    seed,
    standardize,
    pca_params,
    umap_params,
):

    dataframe = panel[
        "data"
    ]

    features = panel[
        "features"
    ]

    X = dataframe[
        features
    ].to_numpy(
        dtype=float
    )

    if standardize:

        X = StandardScaler(
        ).fit_transform(
            X
        )

    # --------------------------------------------------
    # PCA
    # --------------------------------------------------

    if method == "pca":

        model = PCA(
            n_components=2,
            **pca_params,
        )

    # --------------------------------------------------
    # UMAP
    # --------------------------------------------------

    elif method == "umap":

        UMAP = _get_umap()

        model = UMAP(
            n_components=2,
            random_state=seed,
            **umap_params,
        )

    else:

        raise ValueError(
            f"Unknown projection method: "
            f"{method}"
        )

    embedding = (
        model.fit_transform(
            X
        )
    )

    return (
        embedding,
        dataframe,
    )


# ============================================================
# Plot utilities
# ============================================================

MARKERS = [
    "o",
    "s",
    "^",
    "D",
    "P",
    "X",
]


def _plot_panel(
    ax,
    embedding,
    dataframe,
    classes,
    title,
    show_legend=False,
):
    """
    Color  = domain
    Marker = class
    """

    domains = _sorted_unique(
        dataframe[
            "plot_domain"
        ]
    )

    color_cycle = (
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()[
            "color"
        ]
    )

    domain_colors = {
        domain: color_cycle[
            index
            % len(color_cycle)
        ]
        for index, domain
        in enumerate(domains)
    }

    class_markers = {
        class_label: MARKERS[
            index
            % len(MARKERS)
        ]
        for index, class_label
        in enumerate(classes)
    }

    # --------------------------------------------------
    # Samples
    # --------------------------------------------------

    for domain in domains:

        for class_label in classes:

            mask = (
                dataframe[
                    "plot_domain"
                ].eq(
                    domain
                )
                & dataframe[
                    "label"
                ].eq(
                    class_label
                )
            ).to_numpy()

            if not np.any(
                mask
            ):
                continue

            ax.scatter(
                embedding[
                    mask,
                    0,
                ],

                embedding[
                    mask,
                    1,
                ],

                s=12,

                alpha=0.55,

                marker=(
                    class_markers[
                        class_label
                    ]
                ),

                color=(
                    domain_colors[
                        domain
                    ]
                ),
            )

    ax.set_title(
        title,
        fontsize=8,
    )

    ax.set_xticks([])
    ax.set_yticks([])

    # --------------------------------------------------
    # Legend
    # --------------------------------------------------

    if show_legend:

        domain_handles = [
            plt.Line2D(
                [0],
                [0],

                marker="o",
                linestyle="",

                label=str(
                    domain
                ),

                markerfacecolor=(
                    color
                ),

                markeredgecolor=(
                    color
                ),

                markersize=5,
            )

            for domain, color
            in domain_colors.items()
        ]

        class_handles = [
            plt.Line2D(
                [0],
                [0],

                marker=marker,
                linestyle="",

                label=str(
                    class_label
                ),

                markerfacecolor=(
                    "black"
                ),

                markeredgecolor=(
                    "black"
                ),

                markersize=5,
            )

            for class_label, marker
            in class_markers.items()
        ]

        domain_legend = ax.legend(
            handles=domain_handles,
            title="Domain",
            fontsize=5,
            title_fontsize=6,
            loc="upper left",
        )

        ax.add_artist(
            domain_legend
        )

        # Only useful when more than one class exists
        if len(
            class_handles
        ) > 1:

            ax.legend(
                handles=class_handles,
                title="Class",
                fontsize=5,
                title_fontsize=6,
                loc="lower left",
            )


# ============================================================
# Generic 2 x 2 mini-grid
#
# Used for cross-subject and cross-session.
# ============================================================

def _plot_mini_grid(
    figure,
    parent_spec,
    panels,
    method,
    classes,
    seed,
    standardize,
    pca_params,
    umap_params,
):

    subgrid = (
        parent_spec.subgridspec(
            2,
            2,
            wspace=0.08,
            hspace=0.18,
        )
    )

    for index in range(
        4
    ):

        ax = figure.add_subplot(
            subgrid[
                index // 2,
                index % 2,
            ]
        )

        if index >= len(
            panels
        ):

            ax.set_axis_off()
            continue

        panel = panels[
            index
        ]

        embedding, dataframe = (
            _project(
                panel=panel,
                method=method,
                seed=seed + index,
                standardize=standardize,
                pca_params=pca_params,
                umap_params=umap_params,
            )
        )

        _plot_panel(
            ax=ax,
            embedding=embedding,
            dataframe=dataframe,
            classes=classes,
            title=panel[
                "title"
            ],
            show_legend=False,
        )


# ============================================================
# Cross-dataset 2 x 2 mini-grid
#
#   All classes | Left hand
#   Right hand  | Feet
# ============================================================

def _plot_cross_dataset_grid(
    figure,
    parent_spec,
    panels,
    method,
    classes,
    seed,
    standardize,
    pca_params,
    umap_params,
):

    subgrid = (
        parent_spec.subgridspec(
            2,
            2,
            wspace=0.08,
            hspace=0.18,
        )
    )

    for index in range(
        4
    ):

        ax = figure.add_subplot(
            subgrid[
                index // 2,
                index % 2,
            ]
        )

        if index >= len(
            panels
        ):

            ax.set_axis_off()
            continue

        panel = panels[
            index
        ]

        embedding, dataframe = (
            _project(
                panel=panel,
                method=method,
                seed=seed + index,
                standardize=standardize,
                pca_params=pca_params,
                umap_params=umap_params,
            )
        )

        # --------------------------------------------------
        # All-class panel
        # --------------------------------------------------

        if panel[
            "class_label"
        ] is None:

            panel_classes = (
                classes
            )

        # --------------------------------------------------
        # Class-specific panel
        # --------------------------------------------------

        else:

            panel_classes = [
                panel[
                    "class_label"
                ]
            ]

        _plot_panel(
            ax=ax,
            embedding=embedding,
            dataframe=dataframe,
            classes=panel_classes,
            title=panel[
                "title"
            ],

            # Domain legend only once per PCA / UMAP grid
            show_legend=(
                index == 0
            ),
        )


# ============================================================
# Complete qualitative figure
#
#                  Cross-subject   Cross-session   Cross-dataset
#
# PCA                2 x 2            2 x 2           2 x 2
#
# UMAP               2 x 2            2 x 2           2 x 2
#
# Cross-dataset 2 x 2:
#
#   All classes | Left hand
#   Right hand  | Feet
# ============================================================

def _create_figure(
    cross_subject_panels,
    cross_session_panels,
    cross_dataset_panels,
    classes,
    output_path,
    seed,
    standardize,
    pca_params,
    umap_params,
):

    figure = plt.figure(
        figsize=(
            16,
            9,
        )
    )

    grid = (
        figure.add_gridspec(
            2,
            3,

            width_ratios=[
                1,
                1,
                1.1,
            ],

            hspace=0.18,
            wspace=0.12,
        )
    )

    # ==================================================
    # PCA ROW
    # ==================================================

    # --------------------------------------------------
    # Cross-subject PCA
    # --------------------------------------------------

    _plot_mini_grid(
        figure=figure,
        parent_spec=grid[
            0,
            0,
        ],
        panels=(
            cross_subject_panels
        ),
        method="pca",
        classes=classes,
        seed=seed,
        standardize=standardize,
        pca_params=pca_params,
        umap_params=umap_params,
    )

    # --------------------------------------------------
    # Cross-session PCA
    # --------------------------------------------------

    _plot_mini_grid(
        figure=figure,
        parent_spec=grid[
            0,
            1,
        ],
        panels=(
            cross_session_panels
        ),
        method="pca",
        classes=classes,
        seed=seed + 100,
        standardize=standardize,
        pca_params=pca_params,
        umap_params=umap_params,
    )

    # --------------------------------------------------
    # Cross-dataset PCA
    # --------------------------------------------------

    _plot_cross_dataset_grid(
        figure=figure,
        parent_spec=grid[
            0,
            2,
        ],
        panels=(
            cross_dataset_panels
        ),
        method="pca",
        classes=classes,
        seed=seed + 200,
        standardize=standardize,
        pca_params=pca_params,
        umap_params=umap_params,
    )

    # ==================================================
    # UMAP ROW
    # ==================================================

    # --------------------------------------------------
    # Cross-subject UMAP
    # --------------------------------------------------

    _plot_mini_grid(
        figure=figure,
        parent_spec=grid[
            1,
            0,
        ],
        panels=(
            cross_subject_panels
        ),
        method="umap",
        classes=classes,
        seed=seed + 300,
        standardize=standardize,
        pca_params=pca_params,
        umap_params=umap_params,
    )

    # --------------------------------------------------
    # Cross-session UMAP
    # --------------------------------------------------

    _plot_mini_grid(
        figure=figure,
        parent_spec=grid[
            1,
            1,
        ],
        panels=(
            cross_session_panels
        ),
        method="umap",
        classes=classes,
        seed=seed + 400,
        standardize=standardize,
        pca_params=pca_params,
        umap_params=umap_params,
    )

    # --------------------------------------------------
    # Cross-dataset UMAP
    # --------------------------------------------------

    _plot_cross_dataset_grid(
        figure=figure,
        parent_spec=grid[
            1,
            2,
        ],
        panels=(
            cross_dataset_panels
        ),
        method="umap",
        classes=classes,
        seed=seed + 500,
        standardize=standardize,
        pca_params=pca_params,
        umap_params=umap_params,
    )

    # ==================================================
    # Outer labels
    # ==================================================

    figure.text(
        0.18,
        0.97,
        "Cross-subject",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    figure.text(
        0.50,
        0.97,
        "Cross-session",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    figure.text(
        0.82,
        0.97,
        "Cross-dataset",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    figure.text(
        0.015,
        0.72,
        "PCA",
        rotation=90,
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    figure.text(
        0.015,
        0.28,
        "UMAP",
        rotation=90,
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# Public pipeline
#
# This qualitative analysis works directly on standardized
# DatasetViews rather than experimental scenario splits.
#
# Cross-subject:
#   Four independent subject-pair comparisons.
#
# Cross-session:
#   Four subjects, each comparing two sessions.
#
# Cross-dataset:
#   Four subjects per dataset, using all compatible datasets.
#
#   Four visualization panels are produced:
#       all classes
#       left hand
#       right hand
#       feet
#
# PCA and UMAP are generated for all three shift levels.
# ============================================================

def run_domain_variability_qualitative(
    dataset_views,
    params=None,
):

    params = params or {}

    # --------------------------------------------------
    # General configuration
    # --------------------------------------------------

    classes = params.get(
        "classes",
        [
            "left_hand_imagery",
            "right_hand_imagery",
            "both_feet_imagery",
        ],
    )

    seed = params.get(
        "seed",
        42,
    )

    standardize = params.get(
        "standardize",
        True,
    )

    pca_params = params.get(
        "pca_params",
        {},
    )

    umap_params = params.get(
        "umap_params",
        {
            "n_neighbors": 15,
            "min_dist": 0.1,
        },
    )

    cross_subject_params = (
        params[
            "cross_subject"
        ]
    )

    cross_session_params = (
        params[
            "cross_session"
        ]
    )

    cross_dataset_params = (
        params[
            "cross_dataset"
        ]
    )

    # ==================================================
    # Input identity
    # ==================================================

    input_signatures = {}

    for name, view in (
        dataset_views.items()
    ):

        manifest = load_manifest(
            view.manifest_path
        )

        input_signatures[
            name
        ] = manifest.get(
            "signature"
        )

    effective_params = {
        "inputs": (
            input_signatures
        ),
        "params": params,
    }

    signature = make_signature(
        effective_params
    )

    # ==================================================
    # Output paths
    # ==================================================

    output_dir = (
        OUTPUT_ROOT
        / signature[
            :12
        ]
    )

    figure_path = (
        output_dir
        / "qualitative_domain_variability.pdf"
    )

    selection_path = (
        output_dir
        / "qualitative_selection.csv"
    )

    manifest_path = (
        output_dir
        / "manifest.json"
    )

    # ==================================================
    # Resume
    # ==================================================

    if (
        figure_path.exists()
        and selection_path.exists()
        and is_done(
            manifest_path,
            effective_params,
        )
    ):

        return AnalysisArtifact(
            name=(
                "domain_variability_qualitative"
            ),

            output_dir=str(
                output_dir
            ),

            tables={
                "qualitative_selection": str(
                    selection_path
                ),
            },

            figures={
                "qualitative_domain_variability": str(
                    figure_path
                ),
            },

            manifest_path=str(
                manifest_path
            ),

            signature=signature,
        )

    # ==================================================
    # Start
    # ==================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
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

        # ==================================================
        # Load datasets
        # ==================================================

        datasets = _load_views(
            dataset_views
        )

        # --------------------------------------------------
        # Cross-dataset requires compatible representation
        # --------------------------------------------------

        _validate_feature_compatibility(
            datasets
        )

        # ==================================================
        # Cross-subject
        # ==================================================

        subject_dataset_name = (
            cross_subject_params[
                "dataset"
            ]
        )

        if subject_dataset_name not in (
            datasets
        ):

            raise ValueError(
                "Unknown cross-subject dataset: "
                f"{subject_dataset_name}"
            )

        cross_subject_panels = (
            _prepare_cross_subject_panels(
                dataset=(
                    datasets[
                        subject_dataset_name
                    ]
                ),

                params=(
                    cross_subject_params
                ),

                classes=classes,

                seed=seed,
            )
        )

        if len(
            cross_subject_panels
        ) < 4:

            raise ValueError(
                "The qualitative cross-subject "
                "analysis requires four valid panels."
            )

        # ==================================================
        # Cross-session
        # ==================================================

        session_dataset_name = (
            cross_session_params[
                "dataset"
            ]
        )

        if session_dataset_name not in (
            datasets
        ):

            raise ValueError(
                "Unknown cross-session dataset: "
                f"{session_dataset_name}"
            )

        cross_session_panels = (
            _prepare_cross_session_panels(
                dataset=(
                    datasets[
                        session_dataset_name
                    ]
                ),

                params=(
                    cross_session_params
                ),

                classes=classes,

                seed=seed + 100,
            )
        )

        if len(
            cross_session_panels
        ) < 4:

            raise ValueError(
                "The qualitative cross-session "
                "analysis requires four valid panels."
            )

        # ==================================================
        # Cross-dataset
        # ==================================================

        cross_dataset_panels = (
            _prepare_cross_dataset_panels(
                datasets=datasets,

                params=(
                    cross_dataset_params
                ),

                classes=classes,

                seed=seed + 200,
            )
        )

        if len(
            cross_dataset_panels
        ) != (
            1 + len(
                classes
            )
        ):

            raise ValueError(
                "Cross-dataset visualization should "
                "contain one all-class panel plus one "
                "panel for each requested class."
            )

        # ==================================================
        # Save exact visualization selection
        # ==================================================

        selection_rows = []

        # --------------------------------------------------
        # Cross-subject selections
        # --------------------------------------------------

        for panel in (
            cross_subject_panels
        ):

            selection_rows.extend(
                panel[
                    "selection"
                ]
            )

        # --------------------------------------------------
        # Cross-session selections
        # --------------------------------------------------

        for panel in (
            cross_session_panels
        ):

            selection_rows.extend(
                panel[
                    "selection"
                ]
            )

        # --------------------------------------------------
        # Cross-dataset selection is stored in the
        # all-class panel because class-specific panels use
        # subsets of exactly the same sampled observations.
        # --------------------------------------------------

        selection_rows.extend(
            cross_dataset_panels[
                0
            ][
                "selection"
            ]
        )

        selection_dataframe = (
            pd.DataFrame(
                selection_rows
            )
        )

        selection_dataframe.to_csv(
            selection_path,
            index=False,
        )

        # ==================================================
        # Generate figure
        # ==================================================

        _create_figure(
            cross_subject_panels=(
                cross_subject_panels
            ),

            cross_session_panels=(
                cross_session_panels
            ),

            cross_dataset_panels=(
                cross_dataset_panels
            ),

            classes=classes,

            output_path=(
                figure_path
            ),

            seed=seed,

            standardize=(
                standardize
            ),

            pca_params=(
                pca_params
            ),

            umap_params=(
                umap_params
            ),
        )

        # ==================================================
        # Manifest
        # ==================================================

        execution_time = (
            time.time()
            - start
        )

        cross_dataset_names = (
            selection_dataframe[
                selection_dataframe[
                    "scenario"
                ]
                == "cross_dataset"
            ][
                "dataset"
            ]
            .nunique()
        )

        manifest = make_manifest(
            "done",
            effective_params,
            execution_time=(
                execution_time
            ),
        )

        manifest[
            "output"
        ] = {

            "figure": str(
                figure_path
            ),

            "selection": str(
                selection_path
            ),

            "n_cross_subject_panels": len(
                cross_subject_panels
            ),

            "n_cross_session_panels": len(
                cross_session_panels
            ),

            "n_cross_dataset_panels": len(
                cross_dataset_panels
            ),

            "n_cross_dataset_datasets": int(
                cross_dataset_names
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
                execution_time=(
                    execution_time
                ),
                error=str(
                    error
                ),
            ),
            manifest_path,
        )

        raise

    # ==================================================
    # Artifact
    # ==================================================

    return AnalysisArtifact(
        name=(
            "domain_variability_qualitative"
        ),

        output_dir=str(
            output_dir
        ),

        tables={
            "qualitative_selection": str(
                selection_path
            ),
        },

        figures={
            "qualitative_domain_variability": str(
                figure_path
            ),
        },

        manifest_path=str(
            manifest_path
        ),

        signature=signature,
    )