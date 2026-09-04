# eeg/datasets/weibo.py

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from eeg.lib.preparation import prepare_eeg_dataframe
from eeg.lib.filtering import bands
import eeg.lib.feature_extraction as fe


# ============================================================
# Dataset constants
# ============================================================

DATASET_NAME = "weibo2014"
ORIGINAL_SAMPLING_RATE = 200.0


# ============================================================
# Label definitions
# ============================================================

WEIBO_LABEL_MAP = {
    "left_hand": "left_hand_imagery",
    "right_hand": "right_hand_imagery",
    "feet": "both_feet_imagery",
    "rest": "rest",
    "both_hands": "both_hands_imagery",
}


# ============================================================
# Default loading configuration
# ============================================================

DEFAULT_LOAD_CONFIG = {
    "subjects": list(range(1, 11)),

    "sessions": [
        "session_01",
    ],

    "tmin": 0.5,
    "tmax": 3.5,

    # None keeps every available class.
    "classes": None,

    # None keeps every available EEG channel.
    "channels": None,

    "verbose": False,
}


# ============================================================
# Configuration
# ============================================================

def _merge_config(user_config=None):

    config = deepcopy(
        DEFAULT_LOAD_CONFIG
    )

    if user_config is not None:
        config.update(
            user_config
        )

    return config


# ============================================================
# Channel handling
# ============================================================

def _prepare_weibo_channels(
    data,
    channel_names,
    channels=None,
):
    """
    Select and reorder Weibo2014 EEG channels.
    """

    channel_names = list(
        channel_names
    )

    if channels is None:
        selected_channels = (
            channel_names.copy()
        )

    else:
        selected_channels = list(
            channels
        )

    if not selected_channels:
        raise ValueError(
            "The channel list cannot be empty."
        )

    if len(selected_channels) != len(
        set(selected_channels)
    ):
        raise ValueError(
            "The channel selection contains "
            "duplicate names."
        )

    missing_channels = [
        channel
        for channel in selected_channels
        if channel not in channel_names
    ]

    if missing_channels:
        raise ValueError(
            "Requested channels are not available "
            f"in Weibo2014: {missing_channels}."
        )

    indices = [
        channel_names.index(channel)
        for channel in selected_channels
    ]

    data = data[
        :,
        indices,
    ]

    return (
        data,
        selected_channels,
    )


# ============================================================
# Label handling
# ============================================================

def _prepare_weibo_labels(
    metadata,
    selected_classes=None,
):
    """
    Map dataset-specific stimulus codes to canonical labels.
    """

    yaml_labels = metadata[
        "stim"
    ][
        "labels"
    ]

    unknown_labels = [
        label_name
        for label_name in yaml_labels
        if label_name not in WEIBO_LABEL_MAP
    ]

    if unknown_labels:
        raise ValueError(
            "Unknown Weibo2014 labels: "
            f"{unknown_labels}"
        )

    code_to_label = {
        int(code): WEIBO_LABEL_MAP[
            label_name
        ]
        for label_name, code
        in yaml_labels.items()
    }

    if selected_classes is None:
        return code_to_label

    selected_classes = list(
        selected_classes
    )

    valid_classes = set(
        WEIBO_LABEL_MAP.values()
    )

    unknown_classes = [
        label
        for label in selected_classes
        if label not in valid_classes
    ]

    if unknown_classes:
        raise ValueError(
            "Requested classes are not available "
            f"in Weibo2014: {unknown_classes}."
        )

    return {
        code: label
        for code, label
        in code_to_label.items()
        if label in selected_classes
    }


# ============================================================
# Dataset-specific loader
# ============================================================

def load_weibo2014_data(
    root,
    config=None,
):
    """
    Load Weibo2014 into the standardized intermediate
    EEG representation.

    Returns
    -------
    dict

        dataset[subject][session] = {
            "X": ...,
            "y": ...,
            "channel_names": ...,
            "sampling_rate": ...
        }
    """

    config = _merge_config(
        config
    )

    root = Path(
        root
    )

    subjects = config[
        "subjects"
    ]

    sessions = config[
        "sessions"
    ]

    tmin = config[
        "tmin"
    ]

    tmax = config[
        "tmax"
    ]

    classes = config[
        "classes"
    ]

    channels = config[
        "channels"
    ]

    verbose = config[
        "verbose"
    ]

    if not root.exists():
        raise FileNotFoundError(
            f"Weibo2014 directory not found: "
            f"{root}"
        )

    if tmin < 0:
        raise ValueError(
            "'tmin' must be greater than or "
            "equal to zero."
        )

    if tmax <= tmin:
        raise ValueError(
            "'tmax' must be greater than 'tmin'."
        )

    all_data = {}

    for subject in subjects:

        subject_id = (
            f"subject_{subject:02d}"
        )

        subject_data = {}

        for session_name in sessions:

            npz_path = (
                root
                / f"{subject_id}_{session_name}.npz"
            )

            yaml_path = (
                root
                / f"{subject_id}_{session_name}.yml"
            )

            if not npz_path.exists():
                continue

            if not yaml_path.exists():
                continue

            # --------------------------------------------------
            # Load data
            # --------------------------------------------------

            npz_data = np.load(
                npz_path
            )

            if "data" not in npz_data:
                raise KeyError(
                    f"'data' not found in {npz_path}"
                )

            if "stim" not in npz_data:
                raise KeyError(
                    f"'stim' not found in {npz_path}"
                )

            data = npz_data[
                "data"
            ]

            stim = npz_data[
                "stim"
            ]

            with open(
                yaml_path,
                "r",
            ) as file:

                metadata = yaml.safe_load(
                    file
                )

            sampling_rate = float(
                metadata[
                    "acquisition"
                ][
                    "samplingrate"
                ]
            )

            channel_names = list(
                metadata[
                    "acquisition"
                ][
                    "sensors"
                ]
            )

            offset = int(
                metadata[
                    "stim"
                ][
                    "offset"
                ]
            )

            window_length = int(
                metadata[
                    "stim"
                ][
                    "windowlength"
                ]
            )

            # --------------------------------------------------
            # Validation
            # --------------------------------------------------

            if data.ndim != 2:
                raise ValueError(
                    f"Expected 2-D EEG data in "
                    f"{npz_path}, found {data.shape}."
                )

            if stim.ndim != 1:
                raise ValueError(
                    f"Expected 1-D stimulus vector in "
                    f"{npz_path}, found {stim.shape}."
                )

            if len(data) != len(stim):
                raise ValueError(
                    "Data/stim length mismatch in "
                    f"{npz_path}."
                )

            if data.shape[1] != len(
                channel_names
            ):
                raise ValueError(
                    "Signal/metadata channel mismatch "
                    f"in {npz_path}."
                )

            # --------------------------------------------------
            # Channels
            # --------------------------------------------------

            data, selected_channels = (
                _prepare_weibo_channels(
                    data=data,
                    channel_names=channel_names,
                    channels=channels,
                )
            )

            # --------------------------------------------------
            # Labels
            # --------------------------------------------------

            code_to_label = (
                _prepare_weibo_labels(
                    metadata=metadata,
                    selected_classes=classes,
                )
            )

            # --------------------------------------------------
            # Analysis interval
            # --------------------------------------------------

            start_offset = (
                offset
                + int(
                    round(
                        tmin
                        * sampling_rate
                    )
                )
            )

            end_offset = (
                offset
                + int(
                    round(
                        tmax
                        * sampling_rate
                    )
                )
            )

            if (
                end_offset
                > offset + window_length
            ):
                raise ValueError(
                    f"Requested interval "
                    f"[{tmin}, {tmax}] s extends "
                    "beyond the motor-imagery window."
                )

            expected_samples = (
                end_offset
                - start_offset
            )

            # --------------------------------------------------
            # Trial extraction
            # --------------------------------------------------

            event_samples = (
                np.flatnonzero(
                    stim != 0
                )
            )

            X = []
            y = []

            for event_sample in event_samples:

                event_code = int(
                    stim[
                        event_sample
                    ]
                )

                if event_code not in code_to_label:
                    continue

                start = (
                    event_sample
                    + start_offset
                )

                end = (
                    event_sample
                    + end_offset
                )

                if end > len(data):
                    continue

                trial = data[
                    start:end,
                    :
                ].T

                if (
                    trial.shape[1]
                    != expected_samples
                ):
                    continue

                # Weibo/FII values are stored in µV.
                # Convert to volts.
                trial = (
                    trial.astype(
                        np.float32,
                        copy=False,
                    )
                    * 1e-6
                )

                X.append(
                    trial
                )

                y.append(
                    code_to_label[
                        event_code
                    ]
                )

            if not X:

                if verbose:
                    print(
                        "No usable trials found in "
                        f"{npz_path.name}."
                    )

                continue

            X = np.stack(
                X,
                axis=0,
            ).astype(
                np.float32,
                copy=False,
            )

            y = np.asarray(
                y,
                dtype=str,
            )

            # --------------------------------------------------
            # Store session
            # --------------------------------------------------

            subject_data[
                session_name
            ] = {
                "X": X,
                "y": y,

                "channel_names": (
                    selected_channels.copy()
                ),

                "sampling_rate": (
                    sampling_rate
                ),
            }

        if subject_data:

            all_data[
                subject_id
            ] = subject_data

    return all_data


# ============================================================
# Dataset preparation
# ============================================================

def prepare_weibo(params):
    """
    Prepare Weibo2014.

    representation:
        "features" -> handcrafted feature representation
        "signal"   -> processed EEG signal representation
    """

    root = params.get("root")

    if root is None:
        raise ValueError(
            "'root' must be specified for Weibo2014."
        )

    representation = params.get(
        "representation",
        "features",
    )

    loader_config = deepcopy(
        params.get("loader", {})
    )

    subjects = loader_config.get(
        "subjects",
        DEFAULT_LOAD_CONFIG["subjects"],
    )

    filter_config = deepcopy(
        params.get("filter", {})
    )

    filter_config["original_fs"] = (
        ORIGINAL_SAMPLING_RATE
    )

    band_labels = (
        filter_config
        .get("bandpass", {})
        .get("bands", bands)
    )

    feature_config = deepcopy(
        params.get(
            "features",
            fe.DEFAULT_FEATURE_CONFIG,
        )
    )

    subject_batch_size = params.get(
        "subject_batch_size",
        5,
    )

    metadata = deepcopy(
        params.get("metadata", {})
    )

    metadata.setdefault(
        "dataset",
        DATASET_NAME,
    )

    data, info = prepare_eeg_dataframe(
        loader=load_weibo2014_data,

        loader_kwargs={
            "root": root,
        },

        loader_config=loader_config,
        filter_config=filter_config,
        feature_config=feature_config,
        dataset_name=DATASET_NAME,
        representation=representation,

        subjects=subjects,
        subject_batch_size=subject_batch_size,
        band_labels=band_labels,

        # Sessions are already created by the loader.
        session_name=None,

        metadata=metadata,

        show_progress=params.get(
            "show_progress",
            False,
        ),
    )

    return data, info