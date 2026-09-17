# eeg/dataset_handlers/bci2a.py

from copy import deepcopy
from pathlib import Path
import warnings

import mne
import numpy as np
from scipy.io import loadmat

from eeg.lib.feature_extraction import DEFAULT_FEATURE_CONFIG
from eeg.lib.filtering import bands
from eeg.lib.preparation import prepare_eeg_dataframe


# ============================================================
# Dataset definitions
# ============================================================

DATASET_NAME = "bci_iv_2a"
ORIGINAL_SAMPLING_RATE = 250.0

BCI_CHANNEL_MAP = {
    "EEG-Fz": "Fz", "EEG-0": "FC3", "EEG-1": "FC1", "EEG-2": "FCz",
    "EEG-3": "FC2", "EEG-4": "FC4", "EEG-5": "C5", "EEG-C3": "C3",
    "EEG-6": "C1", "EEG-Cz": "Cz", "EEG-7": "C2", "EEG-C4": "C4",
    "EEG-8": "C6", "EEG-9": "CP3", "EEG-10": "CP1", "EEG-11": "CPz",
    "EEG-12": "CP2", "EEG-13": "CP4", "EEG-14": "P1", "EEG-Pz": "Pz",
    "EEG-15": "P2", "EEG-16": "POz",
}

BCI_CHANNEL_ORDER = [
    "Fz",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

BCI_LABEL_MAP = {
    1: "left_hand_imagery",
    2: "right_hand_imagery",
    3: "both_feet_imagery",
    4: "tongue_imagery",
}

DEFAULT_LOAD_CONFIG = {
    "subjects": list(range(1, 10)),
    "sessions": [("session_01", "T"), ("session_02", "E")],
    "mi_codes": ["769", "770", "771", "772", "783"],
    "tmin": 0.5,
    "tmax": 3.5,
    "baseline": None,
    "classes": None,
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
# Validation
# ============================================================

def _prepare_bci_channels(raw, channels=None):
    missing = [ch for ch in BCI_CHANNEL_MAP if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"Expected BCI EEG channels were not found: {missing}.")

    raw.rename_channels(BCI_CHANNEL_MAP)
    selected = BCI_CHANNEL_ORDER.copy() if channels is None else list(channels)

    if not selected:
        raise ValueError("The channel list cannot be empty.")
    if len(selected) != len(set(selected)):
        raise ValueError("The channel selection contains duplicate names.")

    missing = [ch for ch in selected if ch not in BCI_CHANNEL_ORDER]
    if missing:
        raise ValueError(
            f"Requested channels are not available in BCI IV 2a: {missing}."
        )

    raw.pick(selected)
    raw.reorder_channels(selected)
    return raw


def _validate_classes(classes):
    if classes is None:
        return None

    classes = list(classes)
    unknown = [label for label in classes if label not in BCI_LABEL_MAP.values()]
    if unknown:
        raise ValueError(
            f"Requested classes are not available in BCI IV 2a: {unknown}."
        )

    return classes


# ============================================================
# Dataset loader
# ============================================================

def load_bci_iv_2a_data(root_gdf, root_mat, config=None):
    config = _merge_config(config)
    root_gdf, root_mat = Path(root_gdf), Path(root_mat)

    if not root_gdf.exists():
        raise FileNotFoundError(f"BCI GDF directory not found: {root_gdf}")
    if not root_mat.exists():
        raise FileNotFoundError(f"BCI MAT directory not found: {root_mat}")

    subjects = config["subjects"]
    sessions = config["sessions"]
    mi_codes = config["mi_codes"]
    tmin, tmax = config["tmin"], config["tmax"]
    baseline = config["baseline"]
    classes = _validate_classes(config["classes"])
    channels = config["channels"]
    verbose = config["verbose"]

    all_data = {}

    for subject in subjects:
        subject_id = f"A{subject:02d}"
        subject_data = {}

        for session_name, suffix in sessions:
            gdf_path = root_gdf / f"{subject_id}{suffix}.gdf"
            mat_path = root_mat / f"{subject_id}{suffix}.mat"

            if not gdf_path.exists() or not mat_path.exists():
                continue

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Channel names are not unique.*",
                    category=RuntimeWarning,
                )
                raw = mne.io.read_raw_gdf(
                    gdf_path, preload=True, verbose=verbose
                )

            raw = _prepare_bci_channels(raw, channels)

            events, event_id = mne.events_from_annotations(raw, verbose=verbose)
            mi_event_id = {
                name: code for name, code in event_id.items() if name in mi_codes
            }

            if not mi_event_id:
                continue

            mi_events = events[
                np.isin(events[:, 2], list(mi_event_id.values()))
            ]

            mat_data = loadmat(mat_path)
            if "classlabel" not in mat_data:
                raise KeyError(f"'classlabel' not found in {mat_path}")

            numeric_labels = mat_data["classlabel"].squeeze().astype(int)
            unknown = sorted(set(numeric_labels) - set(BCI_LABEL_MAP))

            if unknown:
                raise ValueError(f"Unexpected BCI class labels: {unknown}")
            if len(mi_events) != len(numeric_labels):
                raise ValueError(
                    f"Event/label mismatch in {subject_id}{suffix}: "
                    f"{len(mi_events)} MI events vs {len(numeric_labels)} labels."
                )

            epochs = mne.Epochs(
                raw, mi_events, event_id=mi_event_id,
                tmin=tmin, tmax=tmax, baseline=baseline,
                preload=True, verbose=verbose,
            )

            X = epochs.get_data().astype(np.float32, copy=False)
            numeric_labels = numeric_labels[epochs.selection]
            y = np.asarray(
                [BCI_LABEL_MAP[label] for label in numeric_labels],
                dtype=str,
            )

            if len(X) != len(y):
                raise ValueError(
                    f"Epoch/label mismatch in {subject_id}{suffix}: "
                    f"{len(X)} epochs vs {len(y)} labels."
                )

            if classes is not None:
                mask = np.isin(y, classes)
                X, y = X[mask], y[mask]

            if len(X) == 0:
                continue

            subject_data[session_name] = {
                "X": X,
                "y": y,
                "channel_names": epochs.ch_names.copy(),
                "sampling_rate": float(epochs.info["sfreq"]),
            }

        if subject_data:
            all_data[subject_id] = subject_data

    return all_data


# ============================================================
# Dataset preparation
# ============================================================

def prepare_bci2a(params):
    root_gdf = params.get("root_gdf")
    root_mat = params.get("root_mat")

    if root_gdf is None:
        raise ValueError("'root_gdf' must be specified.")
    if root_mat is None:
        raise ValueError("'root_mat' must be specified.")

    representation = params.get("representation", "features")
    loader_config = deepcopy(params.get("loader", {}))
    subjects = loader_config.get("subjects", DEFAULT_LOAD_CONFIG["subjects"])

    filter_config = deepcopy(params.get("filter", {}))
    filter_config["original_fs"] = ORIGINAL_SAMPLING_RATE
    band_labels = filter_config.get("bandpass", {}).get("bands", bands)

    feature_config = deepcopy(
        params.get("features", DEFAULT_FEATURE_CONFIG)
    )

    metadata = deepcopy(params.get("metadata", {}))
    metadata.setdefault("dataset", DATASET_NAME)

    return prepare_eeg_dataframe(
        loader=load_bci_iv_2a_data,
        loader_kwargs={"root_gdf": root_gdf, "root_mat": root_mat},
        loader_config=loader_config,
        filter_config=filter_config,
        feature_config=feature_config,
        dataset_name=DATASET_NAME,
        representation=representation,
        subjects=subjects,
        subject_batch_size=params.get("subject_batch_size"),
        band_labels=band_labels,
        session_name=None,
        metadata=metadata,
        show_progress=params.get("show_progress", False),
    )