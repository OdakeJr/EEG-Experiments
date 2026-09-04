# eeg/datasets/eegmmidb.py

from copy import deepcopy
from pathlib import Path

import mne
import numpy as np
from tqdm.auto import tqdm

from eeg.lib.preparation import prepare_eeg_dataframe
from eeg.lib.filtering import bands
from eeg.lib.feature_extraction import DEFAULT_FEATURE_CONFIG


# ============================================================
# Dataset constants
# ============================================================

DATASET_NAME = "eegmmidb"
DEFAULT_SESSION_NAME = "session_01"
ORIGINAL_SAMPLING_RATE = 160.0


# ============================================================
# Default loading configuration
# ============================================================

DEFAULT_LOAD_CONFIG = {
    "subjects": list(range(1, 110)),

    "runs": {
        # ----------------------------------------------------
        # Left / right fist execution
        # ----------------------------------------------------

        3: {
            "name": "run_03",
            "label_map": {
                "T1": "left_hand_execution",
                "T2": "right_hand_execution",
            },
        },
        7: {
            "name": "run_07",
            "label_map": {
                "T1": "left_hand_execution",
                "T2": "right_hand_execution",
            },
        },
        11: {
            "name": "run_11",
            "label_map": {
                "T1": "left_hand_execution",
                "T2": "right_hand_execution",
            },
        },

        # ----------------------------------------------------
        # Left / right hand imagery
        # ----------------------------------------------------

        4: {
            "name": "run_04",
            "label_map": {
                "T1": "left_hand_imagery",
                "T2": "right_hand_imagery",
            },
        },
        8: {
            "name": "run_08",
            "label_map": {
                "T1": "left_hand_imagery",
                "T2": "right_hand_imagery",
            },
        },
        12: {
            "name": "run_12",
            "label_map": {
                "T1": "left_hand_imagery",
                "T2": "right_hand_imagery",
            },
        },

        # ----------------------------------------------------
        # Both hands / feet execution
        # ----------------------------------------------------

        5: {
            "name": "run_05",
            "label_map": {
                "T1": "both_hands_execution",
                "T2": "both_feet_execution",
            },
        },
        9: {
            "name": "run_09",
            "label_map": {
                "T1": "both_hands_execution",
                "T2": "both_feet_execution",
            },
        },
        13: {
            "name": "run_13",
            "label_map": {
                "T1": "both_hands_execution",
                "T2": "both_feet_execution",
            },
        },

        # ----------------------------------------------------
        # Both hands / feet imagery
        # ----------------------------------------------------

        6: {
            "name": "run_06",
            "label_map": {
                "T1": "both_hands_imagery",
                "T2": "both_feet_imagery",
            },
        },
        10: {
            "name": "run_10",
            "label_map": {
                "T1": "both_hands_imagery",
                "T2": "both_feet_imagery",
            },
        },
        14: {
            "name": "run_14",
            "label_map": {
                "T1": "both_hands_imagery",
                "T2": "both_feet_imagery",
            },
        },
    },

    "tmin": 0.5,
    "tmax": 3.5,
    "baseline": None,

    "montage": "standard_1020",
    "on_missing": "ignore",

    # None keeps all available classes.
    "classes": None,

    # None keeps all EEG channels.
    "channels": None,

    "verbose": False,
}


# ============================================================
# Configuration
# ============================================================

def _merge_config(user_config=None):
    config = deepcopy(DEFAULT_LOAD_CONFIG)

    if user_config is not None:
        config.update(user_config)

    return config


# ============================================================
# Class handling
# ============================================================

def _get_available_classes(runs):
    """
    Return all canonical labels available in the selected runs.
    """

    return {
        label
        for run_config in runs.values()
        for label in run_config["label_map"].values()
    }


def _validate_classes(
    classes,
    runs,
):
    """
    Validate requested canonical class labels.
    """

    if classes is None:
        return None

    classes = list(classes)

    available_classes = (
        _get_available_classes(
            runs
        )
    )

    unknown_classes = [
        label
        for label in classes
        if label not in available_classes
    ]

    if unknown_classes:
        raise ValueError(
            "Requested classes are not available "
            f"in EEGMMIDB: {unknown_classes}."
        )

    return classes


# ============================================================
# Channel handling
# ============================================================

def _standardize_channel_names(
    raw,
    montage_name,
):
    """
    Standardize EEG channel names using an MNE montage.
    """

    montage = mne.channels.make_standard_montage(
        montage_name
    )

    montage_lookup = {
        channel_name.lower(): channel_name
        for channel_name in montage.ch_names
    }

    rename_map = {}

    for original_name in raw.ch_names:

        cleaned_name = (
            original_name
            .strip()
            .rstrip(".")
        )

        canonical_name = montage_lookup.get(
            cleaned_name.lower(),
            cleaned_name,
        )

        rename_map[
            original_name
        ] = canonical_name

    raw.rename_channels(
        rename_map
    )

    return montage


def _select_physionet_channels(
    raw,
    channels=None,
):
    """
    Select and order requested EEGMMIDB electrodes.
    """

    if channels is None:
        return raw

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
            "The channel selection contains duplicate names."
        )

    missing_channels = [
        channel
        for channel in selected_channels
        if channel not in raw.ch_names
    ]

    if missing_channels:
        raise ValueError(
            "Requested channels are not available "
            f"in EEGMMIDB: {missing_channels}."
        )

    raw.pick(
        selected_channels
    )

    raw.reorder_channels(
        selected_channels
    )

    return raw


# ============================================================
# Dataset loader
# ============================================================

def load_eegmmidb_data(
    root_dir,
    config=None,
):
    """
    Load EEGMMIDB into the standardized intermediate EEG
    representation.

    Returns
    -------
    dict

        dataset[subject][run] = {
            "X": ...,
            "y": ...,
            "channel_names": ...,
            "sampling_rate": ...
        }
    """

    config = _merge_config(
        config
    )

    root_dir = Path(
        root_dir
    )

    subjects = config["subjects"]
    runs = config["runs"]

    tmin = config["tmin"]
    tmax = config["tmax"]
    baseline = config["baseline"]

    montage_name = config["montage"]
    on_missing = config["on_missing"]

    classes = _validate_classes(
        config["classes"],
        runs,
    )

    channels = config["channels"]
    verbose = config["verbose"]

    if not root_dir.exists():
        raise FileNotFoundError(
            f"EEGMMIDB directory not found: "
            f"{root_dir}"
        )

    all_data = {}

    for subject in tqdm(
        subjects,
        desc="Loading EEGMMIDB",
        unit="subject",
        disable=not verbose,
    ):

        subject_id = f"S{subject:03d}"

        subject_path = (
            root_dir
            / subject_id
        )

        if not subject_path.exists():
            continue

        subject_data = {}

        for run_number, run_config in runs.items():

            run_name = run_config[
                "name"
            ]

            label_map = run_config[
                "label_map"
            ]

            edf_path = (
                subject_path
                / f"{subject_id}R{run_number:02d}.edf"
            )

            if not edf_path.exists():
                continue

            # --------------------------------------------------
            # Load EEG
            # --------------------------------------------------

            raw = mne.io.read_raw_edf(
                edf_path,
                preload=True,
                verbose=verbose,
            )

            raw.pick("eeg")

            # --------------------------------------------------
            # Standardize electrodes
            # --------------------------------------------------

            montage = _standardize_channel_names(
                raw=raw,
                montage_name=montage_name,
            )

            raw.set_montage(
                montage,
                on_missing=on_missing,
                verbose=verbose,
            )

            raw = _select_physionet_channels(
                raw=raw,
                channels=channels,
            )

            # --------------------------------------------------
            # Events
            # --------------------------------------------------

            events, event_id = (
                mne.events_from_annotations(
                    raw,
                    verbose=verbose,
                )
            )

            missing_events = [
                event_name
                for event_name in label_map
                if event_name not in event_id
            ]

            if missing_events:
                continue

            selected_event_id = {
                event_name: event_id[
                    event_name
                ]
                for event_name in label_map
            }

            # --------------------------------------------------
            # Epochs
            # --------------------------------------------------

            epochs = mne.Epochs(
                raw,
                events,
                event_id=selected_event_id,
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                preload=True,
                verbose=verbose,
            )

            X = epochs.get_data().astype(
                np.float32,
                copy=False,
            )

            # --------------------------------------------------
            # Canonical labels
            # --------------------------------------------------

            event_code_to_label = {
                event_id[event_name]: label
                for event_name, label
                in label_map.items()
            }

            y = np.asarray(
                [
                    event_code_to_label[
                        event_code
                    ]
                    for event_code
                    in epochs.events[:, -1]
                ],
                dtype=str,
            )

            # --------------------------------------------------
            # Safety
            # --------------------------------------------------

            if len(X) != len(y):

                minimum_length = min(
                    len(X),
                    len(y),
                )

                X = X[
                    :minimum_length
                ]

                y = y[
                    :minimum_length
                ]

            # --------------------------------------------------
            # Class selection
            # --------------------------------------------------

            if classes is not None:

                class_mask = np.isin(
                    y,
                    classes,
                )

                X = X[
                    class_mask
                ]

                y = y[
                    class_mask
                ]

            if len(X) == 0:
                continue

            # --------------------------------------------------
            # Standardized run
            # --------------------------------------------------

            subject_data[
                run_name
            ] = {
                "X": X,
                "y": y,
                "channel_names": (
                    epochs.ch_names.copy()
                ),
                "sampling_rate": float(
                    epochs.info["sfreq"]
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

def prepare_eegmmidb(params):
    """
    Prepare EEGMMIDB.

    representation:
        "features" -> handcrafted feature representation
        "signal"   -> processed EEG signal representation
    """

    root_dir = params.get("root_dir")

    if root_dir is None:
        raise ValueError(
            "'root_dir' must be specified for EEGMMIDB."
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
            DEFAULT_FEATURE_CONFIG,
        )
    )

    subject_batch_size = params.get(
        "subject_batch_size",
        5,
    )

    session_name = params.get(
        "session_name",
        DEFAULT_SESSION_NAME,
    )

    metadata = deepcopy(
        params.get("metadata", {})
    )

    metadata.setdefault(
        "dataset",
        DATASET_NAME,
    )

    data, info = prepare_eeg_dataframe(
        loader=load_eegmmidb_data,

        loader_kwargs={
            "root_dir": root_dir,
        },

        loader_config=loader_config,
        filter_config=filter_config,
        feature_config=feature_config,
        dataset_name=DATASET_NAME,
        representation=representation,

        subjects=subjects,
        subject_batch_size=subject_batch_size,
        band_labels=band_labels,

        # EEGMMIDB runs are treated as one acquisition session.
        session_name=session_name,

        metadata=metadata,

        show_progress=params.get(
            "show_progress",
            False,
        ),
    )

    return data, info