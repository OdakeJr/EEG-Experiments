# preprocessing.py

from copy import deepcopy
from pathlib import Path

import mne
import numpy as np
from tqdm.auto import tqdm


# ============================================================
# Default loading configuration
# ============================================================

DEFAULT_LOAD_CONFIG = {
    "subjects": list(range(1, 110)),

    "runs": {
        # ----------------------------------------------------
        # Execution: left fist versus right fist
        # ----------------------------------------------------
        3: {
            "name": "run_03",
            "label_map": {
                "T1": "left_fist_execution",
                "T2": "right_fist_execution",
            },
        },
        7: {
            "name": "run_07",
            "label_map": {
                "T1": "left_fist_execution",
                "T2": "right_fist_execution",
            },
        },
        11: {
            "name": "run_11",
            "label_map": {
                "T1": "left_fist_execution",
                "T2": "right_fist_execution",
            },
        },

        # ----------------------------------------------------
        # Imagery: left fist versus right fist
        # ----------------------------------------------------
        4: {
            "name": "run_04",
            "label_map": {
                "T1": "left_fist_imagery",
                "T2": "right_fist_imagery",
            },
        },
        8: {
            "name": "run_08",
            "label_map": {
                "T1": "left_fist_imagery",
                "T2": "right_fist_imagery",
            },
        },
        12: {
            "name": "run_12",
            "label_map": {
                "T1": "left_fist_imagery",
                "T2": "right_fist_imagery",
            },
        },

        # ----------------------------------------------------
        # Execution: both fists versus both feet
        # ----------------------------------------------------
        5: {
            "name": "run_05",
            "label_map": {
                "T1": "both_fists_execution",
                "T2": "both_feet_execution",
            },
        },
        9: {
            "name": "run_09",
            "label_map": {
                "T1": "both_fists_execution",
                "T2": "both_feet_execution",
            },
        },
        13: {
            "name": "run_13",
            "label_map": {
                "T1": "both_fists_execution",
                "T2": "both_feet_execution",
            },
        },

        # ----------------------------------------------------
        # Imagery: both fists versus both feet
        # ----------------------------------------------------
        6: {
            "name": "run_06",
            "label_map": {
                "T1": "both_fists_imagery",
                "T2": "both_feet_imagery",
            },
        },
        10: {
            "name": "run_10",
            "label_map": {
                "T1": "both_fists_imagery",
                "T2": "both_feet_imagery",
            },
        },
        14: {
            "name": "run_14",
            "label_map": {
                "T1": "both_fists_imagery",
                "T2": "both_feet_imagery",
            },
        },
    },

    "tmin": 0.5,
    "tmax": 3.5,
    "baseline": None,

    "montage": "standard_1020",
    "on_missing": "ignore",

    # None keeps all available EEG channels.
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


def _standardize_channel_names(raw, montage_name):
    """
    Standardize PhysioNet channel names using the capitalization
    defined by an MNE standard montage.

    Examples
    --------
    Fc3 -> FC3
    Fcz -> FCz
    Cp3 -> CP3
    Poz -> POz
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

        rename_map[original_name] = canonical_name

    raw.rename_channels(rename_map)

    return montage


def _select_physionet_channels(raw, channels=None):
    """
    Optionally select and reorder PhysioNet EEG channels.
    """
    if channels is None:
        return raw

    selected_channels = list(channels)

    if not selected_channels:
        raise ValueError(
            "The channel list cannot be empty."
        )

    if len(selected_channels) != len(set(selected_channels)):
        raise ValueError(
            "The channel selection contains duplicate names."
        )

    missing_channels = [
        channel_name
        for channel_name in selected_channels
        if channel_name not in raw.ch_names
    ]

    if missing_channels:
        raise ValueError(
            "Requested channels are not available in "
            f"PhysioNet EEGMMIDB: {missing_channels}. "
            f"Available channels: {raw.ch_names}"
        )

    raw.pick(selected_channels)
    raw.reorder_channels(selected_channels)

    return raw


def load_physionet_eegmmidb_data(
    root_dir,
    config=None,
):
    """
    Load the PhysioNet EEG Motor Movement/Imagery Database.

    By default, the function loads all movement and imagery runs:

        Runs 3, 7, 11:
            executed left fist versus executed right fist

        Runs 4, 8, 12:
            imagined left fist versus imagined right fist

        Runs 5, 9, 13:
            executed both fists versus executed both feet

        Runs 6, 10, 14:
            imagined both fists versus imagined both feet

    Labels are stored as dataset-specific semantic strings.

    Parameters
    ----------
    root_dir : str or Path
        Directory containing folders S001, ..., S109.

    config : dict, optional
        Loading configuration. Missing keys are filled using
        DEFAULT_LOAD_CONFIG.

    Returns
    -------
    all_data : dict
        Nested structure:

        all_data[subject_id][run_name] = {
            "X": ndarray of shape
                 (n_trials, n_channels, n_samples),

            "y": ndarray of semantic string labels,

            "channel_names": list of canonical electrode names,

            "sampling_rate": sampling frequency in Hz
        }
    """
    config = _merge_config(config)

    root_dir = Path(root_dir)

    subjects = config["subjects"]
    runs = config["runs"]
    tmin = config["tmin"]
    tmax = config["tmax"]
    baseline = config["baseline"]

    montage_name = config["montage"]
    on_missing = config["on_missing"]
    channels = config["channels"]
    verbose = config["verbose"]

    if not root_dir.exists():
        raise FileNotFoundError(
            f"PhysioNet EEGMMIDB directory not found: "
            f"{root_dir}"
        )

    all_data = {}

    for subject in tqdm(
        subjects,
        desc="Loading PhysioNet EEGMMIDB",
        unit="subject",
    ):
        subject_id = f"S{subject:03d}"
        subject_path = root_dir / subject_id

        if not subject_path.exists():
            print(
                f"Warning: subject directory not found: "
                f"{subject_path}"
            )
            continue

        subject_data = {}

        for run_number, run_config in runs.items():
            run_name = run_config["name"]
            label_map = run_config["label_map"]

            edf_path = (
                subject_path
                / f"{subject_id}R{run_number:02d}.edf"
            )

            if not edf_path.exists():
                print(
                    f"Warning: file not found: {edf_path}"
                )
                continue

            # ==================================================
            # Load raw EEG
            # ==================================================

            raw = mne.io.read_raw_edf(
                edf_path,
                preload=True,
                verbose=verbose,
            )

            raw.pick("eeg")

            # ==================================================
            # Standardize electrode names
            # ==================================================

            montage = _standardize_channel_names(
                raw=raw,
                montage_name=montage_name,
            )

            raw.set_montage(
                montage,
                on_missing=on_missing,
                verbose=verbose,
            )

            # ==================================================
            # Optional electrode selection
            # ==================================================

            raw = _select_physionet_channels(
                raw=raw,
                channels=channels,
            )

            # ==================================================
            # Extract events
            # ==================================================

            events, event_id = mne.events_from_annotations(
                raw,
                verbose=verbose,
            )

            missing_events = [
                event_name
                for event_name in label_map
                if event_name not in event_id
            ]

            if missing_events:
                print(
                    f"Warning: missing events {missing_events} "
                    f"in {edf_path.name}"
                )
                continue

            selected_event_id = {
                event_name: event_id[event_name]
                for event_name in label_map
            }

            # ==================================================
            # Create epochs
            # ==================================================

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

            # ==================================================
            # Map T1 and T2 to semantic labels
            # ==================================================

            event_code_to_label = {
                event_id[event_name]: semantic_label
                for event_name, semantic_label
                in label_map.items()
            }

            y = np.asarray(
                [
                    event_code_to_label[event_code]
                    for event_code
                    in epochs.events[:, -1]
                ],
                dtype=str,
            )

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
                    f"{run_name}: {len(X)} epochs versus "
                    f"{len(y)} labels. Truncating to "
                    f"{minimum_length}."
                )

                X = X[:minimum_length]
                y = y[:minimum_length]

            # ==================================================
            # Store run
            # ==================================================

            subject_data[run_name] = {
                "X": X,
                "y": y,
                "channel_names": epochs.ch_names.copy(),
                "sampling_rate": float(
                    epochs.info["sfreq"]
                ),
            }

        if subject_data:
            all_data[subject_id] = subject_data

    return all_data