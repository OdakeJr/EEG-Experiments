# preprocessing.py

from copy import deepcopy
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat


# ============================================================
# BCI Competition IV 2a channel definitions
# ============================================================

# The GDF files contain generic names such as EEG-0 and EEG-1.
# This dictionary maps them to their actual electrode positions.
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


# Canonical order of all 22 EEG electrodes.
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


# ============================================================
# Default loading configuration
# ============================================================

DEFAULT_LOAD_CONFIG = {
    "subjects": list(range(1, 10)),
    "sessions": [
        ("session1", "T"),
        ("session2", "E"),
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
    "label_offset": 1,

    # None keeps all 22 EEG electrodes.
    # Example subset: ["C3", "Cz", "C4"]
    "channels": None,

    "verbose": False,
}


def _merge_config(user_config=None):
    """
    Merge user configuration with the default configuration.
    """
    config = deepcopy(DEFAULT_LOAD_CONFIG)

    if user_config is not None:
        config.update(user_config)

    return config


def _prepare_bci_channels(raw, channels=None):
    """
    Map the original BCI channel names to canonical electrode names
    and optionally select a subset of electrodes.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw BCI IV 2a recording.

    channels : list of str or None
        Canonical channel names to retain. When None, all 22 EEG
        electrodes are retained.

    Returns
    -------
    raw : mne.io.BaseRaw
        Recording containing the selected EEG channels in the
        requested order.
    """

    # Check that the expected original EEG channels are present.
    missing_original_channels = [
        channel
        for channel in BCI_CHANNEL_MAP
        if channel not in raw.ch_names
    ]

    if missing_original_channels:
        raise ValueError(
            "Expected BCI EEG channels were not found: "
            f"{missing_original_channels}. "
            f"Available channels: {raw.ch_names}"
        )

    # Convert dataset-specific names into canonical names.
    raw.rename_channels(BCI_CHANNEL_MAP)

    # Keep all EEG channels unless a subset was requested.
    if channels is None:
        selected_channels = BCI_CHANNEL_ORDER.copy()
    else:
        selected_channels = list(channels)

    if not selected_channels:
        raise ValueError(
            "The channel list cannot be empty."
        )

    # Check the requested canonical channel names.
    missing_selected_channels = [
        channel
        for channel in selected_channels
        if channel not in BCI_CHANNEL_ORDER
    ]

    if missing_selected_channels:
        raise ValueError(
            "Requested channels are not available in BCI IV 2a: "
            f"{missing_selected_channels}. "
            f"Available EEG channels: {BCI_CHANNEL_ORDER}"
        )

    if len(selected_channels) != len(set(selected_channels)):
        raise ValueError(
            "The channel selection contains duplicate names."
        )

    # This removes the EOG channels and any unselected EEG channels.
    raw.pick(selected_channels)

    # Guarantee the exact requested order.
    raw.reorder_channels(selected_channels)

    return raw


def load_bci_iv_2a_data(
    root_gdf,
    root_mat,
    config=None,
):
    """
    Load BCI Competition IV 2a data into a nested dictionary.

    Parameters
    ----------
    root_gdf : str or Path
        Directory containing the GDF files.

    root_mat : str or Path
        Directory containing the MAT label files.

    config : dict, optional
        Loading configuration. Missing keys are filled using
        DEFAULT_LOAD_CONFIG.

    Returns
    -------
    all_data : dict
        Nested structure:

        all_data[subject_id][session_name] = {
            "X": ndarray of shape
                 (n_trials, n_channels, n_samples),

            "y": ndarray of shape
                 (n_trials,),

            "channel_names": list of canonical electrode names,

            "sampling_rate": sampling frequency in Hz
        }
    """
    config = _merge_config(config)

    root_gdf = Path(root_gdf)
    root_mat = Path(root_mat)

    subjects = config["subjects"]
    sessions = config["sessions"]
    mi_codes = config["mi_codes"]
    tmin = config["tmin"]
    tmax = config["tmax"]
    baseline = config["baseline"]
    label_offset = config["label_offset"]
    channels = config["channels"]
    verbose = config["verbose"]

    all_data = {}

    for subject in subjects:
        subject_id = f"A{subject:02d}"
        all_data[subject_id] = {}

        for session_name, suffix in sessions:
            gdf_path = root_gdf / f"{subject_id}{suffix}.gdf"
            mat_path = root_mat / f"{subject_id}{suffix}.mat"

            if not gdf_path.exists():
                print(f"Warning: file not found: {gdf_path}")
                continue

            if not mat_path.exists():
                print(f"Warning: file not found: {mat_path}")
                continue

            # ==================================================
            # Load raw recording
            # ==================================================

            raw = mne.io.read_raw_gdf(
                gdf_path,
                preload=True,
                verbose=verbose,
            )

            raw = _prepare_bci_channels(
                raw=raw,
                channels=channels,
            )

            # ==================================================
            # Extract motor-imagery events
            # ==================================================

            events, event_id = mne.events_from_annotations(
                raw,
                verbose=verbose,
            )

            mi_event_id = {
                event_name: event_code
                for event_name, event_code in event_id.items()
                if event_name in mi_codes
            }

            if not mi_event_id:
                print(
                    f"Warning: no motor-imagery events found "
                    f"in {gdf_path.name}"
                )
                continue

            # ==================================================
            # Create epochs
            # ==================================================

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

            # ==================================================
            # Load labels
            # ==================================================

            mat_data = loadmat(mat_path)

            if "classlabel" not in mat_data:
                raise KeyError(
                    f"'classlabel' not found in {mat_path}"
                )

            y = (
                mat_data["classlabel"]
                .squeeze()
                .astype(int)
            )

            y = y - label_offset

            # ==================================================
            # Safety check
            # ==================================================

            if len(X) != len(y):
                minimum_length = min(
                    len(X),
                    len(y),
                )

                print(
                    f"Warning: mismatch in {subject_id} "
                    f"{session_name}: {len(X)} epochs versus "
                    f"{len(y)} labels. Truncating to "
                    f"{minimum_length}."
                )

                X = X[:minimum_length]
                y = y[:minimum_length]

            # ==================================================
            # Store session
            # ==================================================

            all_data[subject_id][session_name] = {
                "X": X,
                "y": y,
                "channel_names": epochs.ch_names.copy(),
                "sampling_rate": float(
                    epochs.info["sfreq"]
                ),
            }

    return all_data