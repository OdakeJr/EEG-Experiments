# preprocessing.py

from pathlib import Path
from copy import deepcopy

import mne
import numpy as np
from tqdm.auto import tqdm


DEFAULT_LOAD_CONFIG = {
    "subjects": list(range(1, 110)),
    "runs": {
        4: {
            "name": "run_04",
            "task": "left_right",
            "label_map": {"T1": 0, "T2": 1},
        },
        8: {
            "name": "run_08",
            "task": "left_right",
            "label_map": {"T1": 0, "T2": 1},
        },
        12: {
            "name": "run_12",
            "task": "left_right",
            "label_map": {"T1": 0, "T2": 1},
        },
        6: {
            "name": "run_06",
            "task": "fists_feet",
            "label_map": {"T1": 2, "T2": 3},
        },
        10: {
            "name": "run_10",
            "task": "fists_feet",
            "label_map": {"T1": 2, "T2": 3},
        },
        14: {
            "name": "run_14",
            "task": "fists_feet",
            "label_map": {"T1": 2, "T2": 3},
        },
    },
    "tmin": 0.5,
    "tmax": 3.5,
    "baseline": None,
    "montage": "standard_1020",
    "on_missing": "ignore",
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
    Clean EDF channel names and match their capitalization to an
    MNE standard montage.
    """
    montage = mne.channels.make_standard_montage(montage_name)

    montage_lookup = {
        channel.lower(): channel
        for channel in montage.ch_names
    }

    rename_map = {}

    for channel in raw.ch_names:
        cleaned = channel.strip().rstrip(".")
        standardized = montage_lookup.get(
            cleaned.lower(),
            cleaned,
        )
        rename_map[channel] = standardized

    raw.rename_channels(rename_map)

    return montage


def load_physionet_eegmmidb_data(root_dir, config=None):
    """
    Load the PhysioNet EEG Motor Movement/Imagery Database.

    By default, the function loads:

        Runs 4, 8, 12:
            imagined left fist versus imagined right fist

        Runs 6, 10, 14:
            imagined both fists versus imagined both feet

    Label mapping:

        0 -> left fist
        1 -> right fist
        2 -> both fists
        3 -> both feet

    Parameters
    ----------
    root_dir : str or Path
        Directory containing folders S001, ..., S109.

    config : dict, optional
        Configuration dictionary. Missing keys are filled using
        DEFAULT_LOAD_CONFIG.

    Returns
    -------
    all_data : dict
        Nested dictionary with structure:

        all_data[subj_id][run_name] = {
            "X": ndarray with shape
                 (n_trials, n_channels, n_samples),
            "y": ndarray with shape (n_trials,)
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
    verbose = config["verbose"]

    if not root_dir.exists():
        raise FileNotFoundError(
            f"PhysioNet EEGMMIDB directory not found: {root_dir}"
        )

    all_data = {}

    for subject in tqdm(
        subjects,
        desc="Loading PhysioNet EEGMMIDB",
        unit="subject",
    ):
        subj_id = f"S{subject:03d}"
        subject_path = root_dir / subj_id

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
                / f"{subj_id}R{run_number:02d}.edf"
            )

            if not edf_path.exists():
                print(f"Warning: file not found: {edf_path}")
                continue

            # --------------------------------------------------
            # Load raw EEG
            # --------------------------------------------------
            raw = mne.io.read_raw_edf(
                edf_path,
                preload=True,
                verbose=verbose,
            )

            raw.pick("eeg")

            # --------------------------------------------------
            # Standardize channel names and assign montage
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

            # --------------------------------------------------
            # Extract events
            # --------------------------------------------------
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

            mi_event_id = {
                event_name: event_id[event_name]
                for event_name in label_map
            }

            # --------------------------------------------------
            # Create epochs
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

            X = epochs.get_data()

            # --------------------------------------------------
            # Map T1 and T2 according to the run type
            # --------------------------------------------------
            event_code_to_label = {
                event_id[event_name]: class_label
                for event_name, class_label in label_map.items()
            }

            y = np.asarray(
                [
                    event_code_to_label[event_code]
                    for event_code in epochs.events[:, -1]
                ],
                dtype=int,
            )

            # --------------------------------------------------
            # Safety check
            # --------------------------------------------------
            if len(X) != len(y):
                min_len = min(len(X), len(y))

                print(
                    f"Warning: mismatch in {subj_id} "
                    f"{run_name}: {len(X)} epochs versus "
                    f"{len(y)} labels. Truncating to {min_len}."
                )

                X = X[:min_len]
                y = y[:min_len]

            subject_data[run_name] = {
                "X": X,
                "y": y,
            }

        if subject_data:
            all_data[subj_id] = subject_data

    return all_data