# preprocessing.py

from pathlib import Path
from copy import deepcopy

import mne
import numpy as np
from scipy.io import loadmat


DEFAULT_LOAD_CONFIG = {
    "subjects": list(range(1, 10)),
    "sessions": [("session1", "T"), ("session2", "E")],
    "mi_codes": ['769', '770', '771', '772', '783'],
    "tmin": 1.0,
    "tmax": 4.0,
    "baseline": None,
    "label_offset": 1,
    "n_eeg_channels": 22,
    "verbose": False,
}


def _merge_config(user_config=None):
    """
    Merge user config with default config.
    """
    config = deepcopy(DEFAULT_LOAD_CONFIG)
    if user_config is not None:
        config.update(user_config)
    return config


def load_bci_iv_2a_data(root_gdf, root_mat, config=None):
    """
    Load BCI Competition IV 2a data into a nested dictionary.

    Parameters
    ----------
    root_gdf : str or Path
        Directory containing the .gdf files.
    root_mat : str or Path
        Directory containing the .mat label files.
    config : dict, optional
        Configuration dictionary. Missing keys are filled from
        DEFAULT_LOAD_CONFIG.

    Returns
    -------
    all_data : dict
        Nested dictionary with structure:

        all_data[subj_id][session_name] = {
            "X": ndarray of shape (n_trials, n_channels, n_samples),
            "y": ndarray of shape (n_trials,)
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
    n_eeg_channels = config["n_eeg_channels"]
    verbose = config["verbose"]

    all_data = {}

    for subj in subjects:
        subj_id = f"A{subj:02d}"
        all_data[subj_id] = {}

        for sess_name, suffix in sessions:
            gdf_path = root_gdf / f"{subj_id}{suffix}.gdf"
            mat_path = root_mat / f"{subj_id}{suffix}.mat"

            if not gdf_path.exists():
                print(f"Warning: file not found: {gdf_path}")
                continue

            if not mat_path.exists():
                print(f"Warning: file not found: {mat_path}")
                continue

            # --------------------------------------------------
            # Load raw EEG
            # --------------------------------------------------
            raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=verbose)

            ch_names = raw.info["ch_names"]
            eeg_names = ch_names[:n_eeg_channels]
            raw.pick(eeg_names)

            # --------------------------------------------------
            # Extract events
            # --------------------------------------------------
            events, event_id = mne.events_from_annotations(raw, verbose=verbose)

            mi_event_id = {
                key: event_id[key]
                for key in event_id
                if key in mi_codes
            }

            if not mi_event_id:
                print(f"Warning: no MI events found in {gdf_path.name}")
                continue

            # --------------------------------------------------
            # Epoch data
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
            # Load labels
            # --------------------------------------------------
            mat_data = loadmat(mat_path)

            if "classlabel" not in mat_data:
                raise KeyError(f"'classlabel' not found in {mat_path}")

            y = mat_data["classlabel"].squeeze()
            y = y.astype(int) - label_offset

            # --------------------------------------------------
            # Safety check
            # --------------------------------------------------
            if len(X) != len(y):
                min_len = min(len(X), len(y))
                print(
                    f"Warning: mismatch in {subj_id} {sess_name}: "
                    f"{len(X)} epochs vs {len(y)} labels. Truncating to {min_len}."
                )
                X = X[:min_len]
                y = y[:min_len]

            all_data[subj_id][sess_name] = {
                "X": X,
                "y": y,
            }

    return all_data