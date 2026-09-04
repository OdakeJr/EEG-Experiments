# eeg/datasets/bci2a.py

from copy import deepcopy
from pathlib import Path
import warnings

import mne
import numpy as np
from scipy.io import loadmat

from eeg.lib.preparation import prepare_eeg_dataframe
from eeg.lib.filtering import bands
from eeg.lib.feature_extraction import DEFAULT_FEATURE_CONFIG


# ============================================================
# Dataset constants
# ============================================================

DATASET_NAME = "bci_iv_2a"
ORIGINAL_SAMPLING_RATE = 250.0


# ============================================================
# BCI Competition IV 2a definitions
# ============================================================

BCI_CHANNEL_MAP = {
    "EEG-Fz": "Fz",
    "EEG-0": "FC3",
    "EEG-1": "FC1",
    "EEG-2": "FCz",
    "EEG-3": "FC2",
    "EEG-4": "FC4",
    "EEG-5": "C5",
    "EEG-C3": "C3",
    "EEG-6": "C1",
    "EEG-Cz": "Cz",
    "EEG-7": "C2",
    "EEG-C4": "C4",
    "EEG-8": "C6",
    "EEG-9": "CP3",
    "EEG-10": "CP1",
    "EEG-11": "CPz",
    "EEG-12": "CP2",
    "EEG-13": "CP4",
    "EEG-14": "P1",
    "EEG-Pz": "Pz",
    "EEG-15": "P2",
    "EEG-16": "POz",
}


BCI_CHANNEL_ORDER = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
]


# Fixed mapping from native BCI labels to canonical labels.
BCI_LABEL_MAP = {
    1: "left_hand_imagery",
    2: "right_hand_imagery",
    3: "both_feet_imagery",
    4: "tongue_imagery",
}


# ============================================================
# Default loader configuration
# ============================================================

DEFAULT_LOAD_CONFIG = {
    "subjects": list(range(1, 10)),

    "sessions": [
        ("session_01", "T"),
        ("session_02", "E"),
    ],

    "mi_codes": [
        "769",
        "770",
        "771",
        "772",
        "783",
    ],

    "tmin": 0.5,
    "tmax": 3.5,
    "baseline": None,

    # None keeps all available classes.
    "classes": None,

    # None keeps all available EEG channels.
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
# Channel handling
# ============================================================

def _prepare_bci_channels(
    raw,
    channels=None,
):
    """
    Map BCI-specific channel names to canonical electrode names,
    then select and order the requested electrodes.
    """

    missing_original_channels = [
        channel
        for channel in BCI_CHANNEL_MAP
        if channel not in raw.ch_names
    ]

    if missing_original_channels:
        raise ValueError(
            "Expected BCI EEG channels were not found: "
            f"{missing_original_channels}."
        )

    raw.rename_channels(
        BCI_CHANNEL_MAP
    )

    if channels is None:
        selected_channels = (
            BCI_CHANNEL_ORDER.copy()
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
            "The channel selection contains duplicate names."
        )

    missing_channels = [
        channel
        for channel in selected_channels
        if channel not in BCI_CHANNEL_ORDER
    ]

    if missing_channels:
        raise ValueError(
            "Requested channels are not available "
            f"in BCI IV 2a: {missing_channels}."
        )

    raw.pick(
        selected_channels
    )

    raw.reorder_channels(
        selected_channels
    )

    return raw


# ============================================================
# Class handling
# ============================================================

def _validate_classes(classes):
    """
    Validate requested canonical class labels.
    """

    if classes is None:
        return None

    classes = list(classes)

    valid_classes = set(
        BCI_LABEL_MAP.values()
    )

    unknown_classes = [
        label
        for label in classes
        if label not in valid_classes
    ]

    if unknown_classes:
        raise ValueError(
            "Requested classes are not available "
            f"in BCI IV 2a: {unknown_classes}."
        )

    return classes


# ============================================================
# Dataset loader
# ============================================================

def load_bci_iv_2a_data(
    root_gdf,
    root_mat,
    config=None,
):
    """
    Load BCI Competition IV 2a into the standardized
    intermediate EEG representation.

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

    root_gdf = Path(
        root_gdf
    )

    root_mat = Path(
        root_mat
    )

    subjects = config["subjects"]
    sessions = config["sessions"]
    mi_codes = config["mi_codes"]

    tmin = config["tmin"]
    tmax = config["tmax"]
    baseline = config["baseline"]

    classes = _validate_classes(
        config["classes"]
    )

    channels = config["channels"]
    verbose = config["verbose"]

    if not root_gdf.exists():
        raise FileNotFoundError(
            f"BCI GDF directory not found: "
            f"{root_gdf}"
        )

    if not root_mat.exists():
        raise FileNotFoundError(
            f"BCI MAT directory not found: "
            f"{root_mat}"
        )

    all_data = {}

    for subject in subjects:

        subject_id = f"A{subject:02d}"
        subject_data = {}

        for session_name, suffix in sessions:

            gdf_path = (
                root_gdf
                / f"{subject_id}{suffix}.gdf"
            )

            mat_path = (
                root_mat
                / f"{subject_id}{suffix}.mat"
            )

            if not gdf_path.exists():
                continue

            if not mat_path.exists():
                continue

            # --------------------------------------------------
            # Load recording
            # --------------------------------------------------

            #raw = mne.io.read_raw_gdf(
            #    gdf_path,
            #    preload=True,
            #    verbose=verbose,
            #)
            
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Channel names are not unique.*",
                    category=RuntimeWarning,
                )
                raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=verbose)

            raw = _prepare_bci_channels(
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

            mi_event_id = {
                event_name: event_code
                for event_name, event_code
                in event_id.items()
                if event_name in mi_codes
            }

            if not mi_event_id:
                continue

            # --------------------------------------------------
            # Epochs
            # --------------------------------------------------

            epochs = mne.Epochs(
                raw,
                events,
                event_id=mi_event_id,
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
            # Labels
            # --------------------------------------------------

            mat_data = loadmat(
                mat_path
            )

            if "classlabel" not in mat_data:
                raise KeyError(
                    f"'classlabel' not found "
                    f"in {mat_path}"
                )

            numeric_labels = (
                mat_data["classlabel"]
                .squeeze()
                .astype(int)
            )

            unknown_labels = sorted(
                set(numeric_labels)
                - set(BCI_LABEL_MAP)
            )

            if unknown_labels:
                raise ValueError(
                    "Unexpected BCI class labels: "
                    f"{unknown_labels}"
                )

            y = np.asarray(
                [
                    BCI_LABEL_MAP[label]
                    for label in numeric_labels
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
            # Standardized session
            # --------------------------------------------------

            subject_data[
                session_name
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

def prepare_bci2a(params):
    """
    Prepare BCI Competition IV 2a.

    representation:
        "features" -> handcrafted feature representation
        "signal"   -> processed EEG signal representation
    """

    root_gdf = params.get("root_gdf")
    root_mat = params.get("root_mat")

    if root_gdf is None:
        raise ValueError("'root_gdf' must be specified.")

    if root_mat is None:
        raise ValueError("'root_mat' must be specified.")

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

    metadata = deepcopy(
        params.get("metadata", {})
    )

    metadata.setdefault(
        "dataset",
        DATASET_NAME,
    )

    data, info = prepare_eeg_dataframe(
        loader=load_bci_iv_2a_data,

        loader_kwargs={
            "root_gdf": root_gdf,
            "root_mat": root_mat,
        },

        loader_config=loader_config,
        filter_config=filter_config,
        feature_config=feature_config,
        dataset_name=DATASET_NAME,
        representation=representation,

        subjects=subjects,

        subject_batch_size=params.get(
            "subject_batch_size"
        ),

        band_labels=band_labels,

        # Keep session_01 / session_02.
        session_name=None,

        metadata=metadata,

        show_progress=params.get(
            "show_progress",
            False,
        ),
    )

    return data, info