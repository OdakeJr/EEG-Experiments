# eeg/dataset_handlers/eegmmidb.py

from copy import deepcopy
from pathlib import Path

import mne
import numpy as np
from tqdm.auto import tqdm

from eeg.lib.feature_extraction import DEFAULT_FEATURE_CONFIG
from eeg.lib.filtering import bands
from eeg.lib.preparation import prepare_eeg_dataframe


# ============================================================
# Dataset definitions
# ============================================================

DATASET_NAME = "eegmmidb"
DEFAULT_SESSION_NAME = "session_01"
ORIGINAL_SAMPLING_RATE = 160.0

EEGMMIDB_RUNS = {
    # Left/right fist execution
    3: {"name": "run_03", "label_map": {
        "T1": "left_hand_execution", "T2": "right_hand_execution"}},
    7: {"name": "run_07", "label_map": {
        "T1": "left_hand_execution", "T2": "right_hand_execution"}},
    11: {"name": "run_11", "label_map": {
        "T1": "left_hand_execution", "T2": "right_hand_execution"}},

    # Left/right hand imagery
    4: {"name": "run_04", "label_map": {
        "T1": "left_hand_imagery", "T2": "right_hand_imagery"}},
    8: {"name": "run_08", "label_map": {
        "T1": "left_hand_imagery", "T2": "right_hand_imagery"}},
    12: {"name": "run_12", "label_map": {
        "T1": "left_hand_imagery", "T2": "right_hand_imagery"}},

    # Both hands/feet execution
    5: {"name": "run_05", "label_map": {
        "T1": "both_hands_execution", "T2": "both_feet_execution"}},
    9: {"name": "run_09", "label_map": {
        "T1": "both_hands_execution", "T2": "both_feet_execution"}},
    13: {"name": "run_13", "label_map": {
        "T1": "both_hands_execution", "T2": "both_feet_execution"}},

    # Both hands/feet imagery
    6: {"name": "run_06", "label_map": {
        "T1": "both_hands_imagery", "T2": "both_feet_imagery"}},
    10: {"name": "run_10", "label_map": {
        "T1": "both_hands_imagery", "T2": "both_feet_imagery"}},
    14: {"name": "run_14", "label_map": {
        "T1": "both_hands_imagery", "T2": "both_feet_imagery"}},
}

DEFAULT_LOAD_CONFIG = {
    "subjects": list(range(1, 110)),
    "runs": list(EEGMMIDB_RUNS),
    "tmin": 0.5,
    "tmax": 3.5,
    "baseline": None,
    "montage": "standard_1020",
    "on_missing": "ignore",
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


def _resolve_runs(runs):
    if isinstance(runs, dict):
        return deepcopy(runs)

    runs = list(runs)
    unknown = [run for run in runs if run not in EEGMMIDB_RUNS]

    if unknown:
        raise ValueError(f"Unknown EEGMMIDB runs: {unknown}.")

    return {run: deepcopy(EEGMMIDB_RUNS[run]) for run in runs}


# ============================================================
# Class handling
# ============================================================

def _get_available_classes(runs):
    return {
        label
        for run_config in runs.values()
        for label in run_config["label_map"].values()
    }


def _validate_classes(classes, runs):
    if classes is None:
        return None

    classes = list(classes)
    available = _get_available_classes(runs)
    unknown = [label for label in classes if label not in available]

    if unknown:
        raise ValueError(
            f"Requested classes are not available in EEGMMIDB: {unknown}."
        )

    return classes


# ============================================================
# Channel handling
# ============================================================

def _standardize_channel_names(raw, montage_name):
    montage = mne.channels.make_standard_montage(montage_name)
    lookup = {name.lower(): name for name in montage.ch_names}

    rename_map = {}
    for original in raw.ch_names:
        cleaned = original.strip().rstrip(".")
        rename_map[original] = lookup.get(cleaned.lower(), cleaned)

    raw.rename_channels(rename_map)
    return montage


def _select_physionet_channels(raw, channels=None):
    if channels is None:
        return raw

    selected = list(channels)

    if not selected:
        raise ValueError("The channel list cannot be empty.")
    if len(selected) != len(set(selected)):
        raise ValueError("The channel selection contains duplicate names.")

    missing = [channel for channel in selected if channel not in raw.ch_names]
    if missing:
        raise ValueError(
            f"Requested channels are not available in EEGMMIDB: {missing}."
        )

    raw.pick(selected)
    raw.reorder_channels(selected)
    return raw


# ============================================================
# Dataset loader
# ============================================================

def load_eegmmidb_data(root_dir, config=None):
    config = _merge_config(config)
    root_dir = Path(root_dir)

    if not root_dir.exists():
        raise FileNotFoundError(f"EEGMMIDB directory not found: {root_dir}")

    subjects = config["subjects"]
    runs = _resolve_runs(config["runs"])
    tmin, tmax = config["tmin"], config["tmax"]
    baseline = config["baseline"]
    montage_name = config["montage"]
    on_missing = config["on_missing"]
    classes = _validate_classes(config["classes"], runs)
    channels = config["channels"]
    verbose = config["verbose"]

    all_data = {}

    for subject in tqdm(
        subjects,
        desc="Loading EEGMMIDB",
        unit="subject",
        disable=not verbose,
    ):
        subject_id = f"S{subject:03d}"
        subject_path = root_dir / subject_id

        if not subject_path.exists():
            continue

        subject_data = {}

        for run_number, run_config in runs.items():
            run_name = run_config["name"]
            label_map = run_config["label_map"]

            if classes is not None and not set(label_map.values()) & set(classes):
                continue

            edf_path = subject_path / f"{subject_id}R{run_number:02d}.edf"
            if not edf_path.exists():
                continue

            raw = mne.io.read_raw_edf(
                edf_path, preload=True, verbose=verbose
            )
            raw.pick("eeg")

            # Some EEGMMIDB recordings have a different sampling rate.
            # Standardize all recordings before event extraction and epoching.
            if not np.isclose(raw.info["sfreq"], ORIGINAL_SAMPLING_RATE):
                if verbose:
                    print(
                        f"Resampling {edf_path.name}: "
                        f"{raw.info['sfreq']:.1f} -> {ORIGINAL_SAMPLING_RATE:.1f} Hz"
                    )
                raw.resample(
                    ORIGINAL_SAMPLING_RATE,
                    npad="auto",
                    verbose=verbose,
                )

            montage = _standardize_channel_names(raw, montage_name)
            raw.set_montage(
                montage, on_missing=on_missing, verbose=verbose
            )
            raw = _select_physionet_channels(raw, channels)

            events, event_id = mne.events_from_annotations(
                raw, verbose=verbose
            )

            missing_events = [
                event for event in label_map if event not in event_id
            ]
            if missing_events:
                continue

            selected_event_id = {
                event: event_id[event] for event in label_map
            }

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

            X = epochs.get_data().astype(np.float32, copy=False)

            code_to_label = {
                event_id[event]: label
                for event, label in label_map.items()
            }
            y = np.asarray(
                [code_to_label[code] for code in epochs.events[:, -1]],
                dtype=str,
            )

            if len(X) != len(y):
                raise ValueError(
                    f"Epoch/label mismatch in {subject_id} run {run_number}: "
                    f"{len(X)} epochs vs {len(y)} labels."
                )

            if classes is not None:
                mask = np.isin(y, classes)
                X, y = X[mask], y[mask]

            if len(X) == 0:
                continue

            subject_data[run_name] = {
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

def prepare_eegmmidb(params):
    root_dir = params.get("root_dir")

    if root_dir is None:
        raise ValueError("'root_dir' must be specified for EEGMMIDB.")

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
        loader=load_eegmmidb_data,
        loader_kwargs={"root_dir": root_dir},
        loader_config=loader_config,
        filter_config=filter_config,
        feature_config=feature_config,
        dataset_name=DATASET_NAME,
        representation=representation,
        subjects=subjects,
        subject_batch_size=params.get("subject_batch_size", 5),
        band_labels=band_labels,
        session_name=params.get("session_name", DEFAULT_SESSION_NAME),
        metadata=metadata,
        show_progress=params.get("show_progress", False),
    )