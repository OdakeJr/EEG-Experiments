from pathlib import Path

import pytest

import pipeline.process_data as process_data
import pipeline.combine_data as combine_data
import pipeline.scenarios as scenarios

from utils.storage import (
    exists,
    load_data,
)


# ============================================================
# Test configuration
# ============================================================

TEST_ROOT = Path(
    "tests/output/scenarios"
)

COMMON_CLASSES = [
    "left_hand_imagery",
    "right_hand_imagery",
    "both_feet_imagery",
]

CHANNELS = [
    "C3",
    "Cz",
    "C4",
]

FILTER_CONFIG = {
    "bandpass": {
        "bands": [
            (8, 12),
            (13, 30),
        ],
    },
}

FEATURE_CONFIG = {
    "logvar": {},
    "cov": {},
}


# ============================================================
# Preprocessing parameters
# ============================================================

PREPROCESSING_TEST_PARAMS = [

    # --------------------------------------------------------
    # BCI Competition IV 2a
    # --------------------------------------------------------

    {
        "dataset": "bci2a",
        "name": "scenario_test_bci2a",

        "root_gdf": "data/bci2a/gdf",
        "root_mat": "data/bci2a/mat",

        "loader": {
            "subjects": [
                1, 2, 3, 4, 5
            ],

            "channels": CHANNELS,

            "classes": COMMON_CLASSES,
        },

        "filter": FILTER_CONFIG,

        "features": FEATURE_CONFIG,

        "subject_batch_size": 2,

        "show_progress": False,
    },

    # --------------------------------------------------------
    # PhysioNet EEGMMIDB
    # --------------------------------------------------------

    {
        "dataset": "eegmmidb",
        "name": "scenario_test_eegmmidb",

        "root_dir": "data/eegmmidb",

        "loader": {
            "subjects": [
                1, 2, 3, 4, 5
            ],

            "runs": {
                4: {
                    "name": "run_04",

                    "label_map": {
                        "T1": "left_hand_imagery",
                        "T2": "right_hand_imagery",
                    },
                },

                6: {
                    "name": "run_06",

                    "label_map": {
                        "T1": "both_hands_imagery",
                        "T2": "both_feet_imagery",
                    },
                },
            },

            "channels": CHANNELS,

            "classes": COMMON_CLASSES,
        },

        "filter": FILTER_CONFIG,

        "features": FEATURE_CONFIG,

        "subject_batch_size": 2,

        "show_progress": False,
    },

    # --------------------------------------------------------
    # Weibo2014
    # --------------------------------------------------------

    {
        "dataset": "weibo",
        "name": "scenario_test_weibo",

        "root": "data/weibo",

        "loader": {
            "subjects": [
                1, 2, 3, 4, 5
            ],

            "channels": CHANNELS,

            "classes": COMMON_CLASSES,
        },

        "filter": FILTER_CONFIG,

        "features": FEATURE_CONFIG,

        "subject_batch_size": 2,

        "show_progress": False,
    },

    # --------------------------------------------------------
    # Zhou2016
    # --------------------------------------------------------

    {
        "dataset": "zhou",
        "name": "scenario_test_zhou",

        "root": "data/zhou",

        # Zhou2016 has four subjects.
        "loader": {
            "subjects": [
                1, 2, 3, 4
            ],

            "channels": CHANNELS,

            "classes": COMMON_CLASSES,
        },

        "filter": FILTER_CONFIG,

        "features": FEATURE_CONFIG,

        "subject_batch_size": 2,

        "show_progress": False,
    },
]


# ============================================================
# Scenario parameters
# ============================================================

INTRA_TEST_PARAMS = {
    "train_fraction": 0.5,
    "seed": 42,
}


SESSION_TEST_PARAMS = {
    "source_counts": [
        1,
        "all",
    ],

    "target_fractions": [
        0.5,
    ],

    "seed": 42,
}


SUBJECT_TEST_PARAMS = {
    "source_counts": [
        1,
        2,
        "all",
    ],

    "target_fractions": [
        0.5,
    ],

    "max_source_combinations": 3,

    "seed": 42,
}


DATASET_TEST_PARAMS = {

    # Important:
    # 2 lets us explicitly test multiple source
    # super-domains.
    "source_dataset_counts": [
        1,
        2,
    ],

    "target_dataset_subject_counts": [
        0,
        1,
    ],

    "target_subject_fractions": [
        0.5,
    ],

    "max_source_combinations": 3,

    "max_target_domain_combinations": 3,

    "seed": 42,
}


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(
    scope="module"
)
def pipeline_data():

    # --------------------------------------------------------
    # Test output locations
    # --------------------------------------------------------

    process_data.OUTPUT_ROOT = (
        TEST_ROOT
        / "preprocessing"
    )

    combine_data.OUTPUT_ROOT = (
        TEST_ROOT
        / "combined"
    )

    scenarios.OUTPUT_ROOT = (
        TEST_ROOT
        / "splits"
    )

    # --------------------------------------------------------
    # Check raw dataset paths
    # --------------------------------------------------------

    assert Path(
        "data/bci2a/gdf"
    ).exists()

    assert Path(
        "data/bci2a/mat"
    ).exists()

    assert Path(
        "data/eegmmidb"
    ).exists()

    assert Path(
        "data/weibo"
    ).exists()

    assert Path(
        "data/zhou"
    ).exists()

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    views = {}

    for params in (
        PREPROCESSING_TEST_PARAMS
    ):

        view = (
            process_data.run_preprocessing(
                params
            )
        )

        assert exists(
            view.path
        )

        dataframe = load_data(
            view.path
        )

        assert not dataframe.empty

        assert set(
            dataframe[
                "label"
            ].unique()
        ).issubset(
            COMMON_CLASSES
        )

        views[
            params["dataset"]
        ] = view

    # --------------------------------------------------------
    # Combine all four compatible datasets
    # --------------------------------------------------------

    combined_view = (
        combine_data.combine_datasets(
            list(
                views.values()
            ),
            {
                "name": (
                    "scenario_test_all_datasets"
                )
            },
        )
    )

    assert exists(
        combined_view.path
    )

    combined_dataframe = load_data(
        combined_view.path
    )

    assert (
        combined_dataframe[
            "dataset"
        ].nunique()
        == 4
    )

    return {
        "views": views,
        "combined_view": combined_view,
    }


# ============================================================
# Intra-subject
# ============================================================

def test_intra_subject(
    pipeline_data,
):

    for dataset, view in (
        pipeline_data[
            "views"
        ].items()
    ):

        splits = scenarios.run_scenario(
            view,
            "intra_subject",
            INTRA_TEST_PARAMS,
        )

        assert len(
            splits
        ) > 0

        split = splits[
            0
        ]

        data = split.materialize(
            view
        )

        # No previous adaptation stages.
        assert data.source is None

        assert (
            data.target_super_domain
            is None
        )

        target = (
            data.target_elementary_domain
        )

        assert target is not None

        # Intra-subject:
        # train + test.
        assert set(
            target.partitions
        ) == {
            "train",
            "test",
        }

        # No super-domain hierarchy here.
        assert (
            target.super_domains
            is None
        )

        assert (
            len(
                set(
                    target.elementary_domains
                )
            )
            == 1
        )


# ============================================================
# Cross-session
# ============================================================

def test_cross_session(
    pipeline_data,
):

    for dataset, view in (
        pipeline_data[
            "views"
        ].items()
    ):

        dataframe = load_data(
            view.path
        )

        splits = scenarios.run_scenario(
            view,
            "cross_session",
            SESSION_TEST_PARAMS,
        )

        # ----------------------------------------------------
        # Determine whether this dataset actually has
        # multiple sessions per subject.
        # ----------------------------------------------------

        session_counts = (
            dataframe
            .groupby(
                [
                    "dataset",
                    "subject",
                ]
            )[
                "session"
            ]
            .nunique()
        )

        has_multiple_sessions = (
            session_counts.max()
            > 1
        )

        if not has_multiple_sessions:

            assert len(
                splits
            ) == 0

            continue

        # ----------------------------------------------------
        # Materialize one valid split
        # ----------------------------------------------------

        assert len(
            splits
        ) > 0

        data = splits[
            0
        ].materialize(
            view
        )

        assert data.source is not None

        assert (
            data.target_super_domain
            is None
        )

        target = (
            data.target_elementary_domain
        )

        assert target is not None

        # Source sessions are training data.
        assert set(
            data.source.partitions
        ) == {
            "train",
        }

        # Final session is calibration + test.
        assert set(
            target.partitions
        ) == {
            "calibration",
            "test",
        }

        # Source and target sessions cannot overlap.
        assert set(
            data.source.elementary_domains
        ).isdisjoint(
            set(
                target.elementary_domains
            )
        )

        assert (
            data.source.super_domains
            is None
        )

        assert (
            target.super_domains
            is None
        )


# ============================================================
# Cross-subject
# ============================================================

def test_cross_subject(
    pipeline_data,
):

    for dataset, view in (
        pipeline_data[
            "views"
        ].items()
    ):

        splits = scenarios.run_scenario(
            view,
            "cross_subject",
            SUBJECT_TEST_PARAMS,
        )

        assert len(
            splits
        ) > 0

        data = splits[
            0
        ].materialize(
            view
        )

        assert data.source is not None

        assert (
            data.target_super_domain
            is None
        )

        target = (
            data.target_elementary_domain
        )

        assert target is not None

        assert set(
            data.source.partitions
        ) == {
            "train",
        }

        assert set(
            target.partitions
        ) == {
            "calibration",
            "test",
        }

        # Source subjects cannot contain
        # the held-out target subject.
        assert set(
            data.source.elementary_domains
        ).isdisjoint(
            set(
                target.elementary_domains
            )
        )

        # No super-domain level in the
        # within-dataset experiment.
        assert (
            data.source.super_domains
            is None
        )

        assert (
            target.super_domains
            is None
        )


# ============================================================
# Cross-dataset
# ============================================================

def test_cross_dataset(
    pipeline_data,
):

    view = pipeline_data[
        "combined_view"
    ]

    splits = scenarios.run_scenario(
        view,
        "cross_dataset",
        DATASET_TEST_PARAMS,
    )

    assert len(
        splits
    ) > 0

    # --------------------------------------------------------
    # Find a split containing:
    #
    # 2 source datasets
    # +
    # target-super-domain calibration
    #
    # This directly tests the hierarchy that motivated
    # the super-domain representation.
    # --------------------------------------------------------

    selected_split = None

    for split in splits:

        source_datasets = {
            domain.split(
                "|",
                1,
            )[0]

            for domain
            in split.source_elementary_domains
        }

        if (
            len(source_datasets) == 2
            and len(
                split
                .target_super_domain_elementary_domains
            ) > 0
        ):

            selected_split = split
            break

    assert (
        selected_split
        is not None
    )

    # --------------------------------------------------------
    # Materialize
    # --------------------------------------------------------

    data = selected_split.materialize(
        view
    )

    assert data.source is not None

    assert (
        data.target_super_domain
        is not None
    )

    assert (
        data.target_elementary_domain
        is not None
    )

    # --------------------------------------------------------
    # Super-domain hierarchy
    # --------------------------------------------------------

    source_super_domains = set(
        data.source.super_domains
    )

    target_super_domains = set(
        data
        .target_super_domain
        .super_domains
    )

    final_target_super_domains = set(
        data
        .target_elementary_domain
        .super_domains
    )

    # Exactly two source datasets.
    assert len(
        source_super_domains
    ) == 2

    # One target dataset.
    assert len(
        target_super_domains
    ) == 1

    assert len(
        final_target_super_domains
    ) == 1

    # Source datasets cannot include
    # the target dataset.
    assert (
        source_super_domains
        .isdisjoint(
            final_target_super_domains
        )
    )

    # Both target stages belong to the
    # same target dataset.
    assert (
        target_super_domains
        == final_target_super_domains
    )

    # --------------------------------------------------------
    # Elementary domains
    # --------------------------------------------------------

    source_elementary = set(
        data.source.elementary_domains
    )

    target_super_elementary = set(
        data
        .target_super_domain
        .elementary_domains
    )

    target_elementary = set(
        data
        .target_elementary_domain
        .elementary_domains
    )

    assert (
        source_elementary
        .isdisjoint(
            target_elementary
        )
    )

    assert (
        target_super_elementary
        .isdisjoint(
            target_elementary
        )
    )

    # --------------------------------------------------------
    # Partitions
    # --------------------------------------------------------

    assert set(
        data.source.partitions
    ) == {
        "train",
    }

    assert set(
        data
        .target_super_domain
        .partitions
    ) == {
        "calibration",
    }

    assert set(
        data
        .target_elementary_domain
        .partitions
    ) == {
        "calibration",
        "test",
    }


# ============================================================
# Scenario resume
# ============================================================

def test_scenario_resume(
    pipeline_data,
):

    view = pipeline_data[
        "combined_view"
    ]

    # --------------------------------------------------------
    # First call
    # --------------------------------------------------------

    first_splits = (
        scenarios.run_scenario(
            view,
            "cross_dataset",
            DATASET_TEST_PARAMS,
        )
    )

    scenario_directory = (
        scenarios.OUTPUT_ROOT
        / "cross_dataset"
    )

    split_files = list(
        scenario_directory.rglob(
            "splits.json"
        )
    )

    assert split_files

    before = {
        path: path.stat().st_mtime_ns
        for path in split_files
    }

    # --------------------------------------------------------
    # Second call
    # --------------------------------------------------------

    second_splits = (
        scenarios.run_scenario(
            view,
            "cross_dataset",
            DATASET_TEST_PARAMS,
        )
    )

    after = {
        path: path.stat().st_mtime_ns
        for path in split_files
    }

    # Definitions should have been reused,
    # not regenerated.
    assert before == after

    assert [
        split.id
        for split in first_splits
    ] == [
        split.id
        for split in second_splits
    ]