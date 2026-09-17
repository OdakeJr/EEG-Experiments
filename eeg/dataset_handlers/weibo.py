# eeg/dataset_handlers/weibo.py

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from eeg.lib.feature_extraction import DEFAULT_FEATURE_CONFIG
from eeg.lib.filtering import bands
from eeg.lib.preparation import prepare_eeg_dataframe


# ============================================================
# Dataset definitions
# ============================================================

DATASET_NAME = "weibo2014"
ORIGINAL_SAMPLING_RATE = 200.0

WEIBO_LABEL_MAP = {
    "left_hand": "left_hand_imagery",
    "right_hand": "right_hand_imagery",
    "feet": "both_feet_imagery",
    "rest": "rest",
    "both_hands": "both_hands_imagery",
}

DEFAULT_LOAD_CONFIG = {
    "subjects": list(range(1, 11)),
    "sessions": ["session_01"],
    "tmin": 0.5,
    "tmax": 3.5,
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
# Channel handling
# ============================================================

def _prepare_weibo_channels(data, channel_names, channels=None):
    channel_names = list(channel_names)
    selected = channel_names.copy() if channels is None else list(channels)

    if not selected:
        raise ValueError("The channel list cannot be empty.")
    if len(selected) != len(set(selected)):
        raise ValueError("The channel selection contains duplicate names.")

    missing = [channel for channel in selected if channel not in channel_names]
    if missing:
        raise ValueError(
            f"Requested channels are not available in Weibo2014: {missing}."
        )

    indices = [channel_names.index(channel) for channel in selected]
    return data[:, indices], selected


# ============================================================
# Label handling
# ============================================================

def _prepare_weibo_labels(metadata, selected_classes=None):
    yaml_labels = metadata["stim"]["labels"]

    unknown = [
        label for label in yaml_labels
        if label not in WEIBO_LABEL_MAP
    ]
    if unknown:
        raise ValueError(f"Unknown Weibo2014 labels: {unknown}")

    code_to_label = {
        int(code): WEIBO_LABEL_MAP[label]
        for label, code in yaml_labels.items()
    }

    if selected_classes is None:
        return code_to_label

    selected_classes = list(selected_classes)
    valid = set(WEIBO_LABEL_MAP.values())
    unknown = [label for label in selected_classes if label not in valid]

    if unknown:
        raise ValueError(
            f"Requested classes are not available in Weibo2014: {unknown}."
        )

    return {
        code: label
        for code, label in code_to_label.items()
        if label in selected_classes
    }


# ============================================================
# Dataset loader
# ============================================================

def load_weibo2014_data(root, config=None):
    config = _merge_config(config)
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"Weibo2014 directory not found: {root}")

    subjects = config["subjects"]
    sessions = config["sessions"]
    tmin, tmax = config["tmin"], config["tmax"]
    classes = config["classes"]
    channels = config["channels"]
    verbose = config["verbose"]

    if tmin < 0:
        raise ValueError("'tmin' must be greater than or equal to zero.")
    if tmax <= tmin:
        raise ValueError("'tmax' must be greater than 'tmin'.")

    all_data = {}

    for subject in subjects:
        subject_id = f"subject_{subject:02d}"
        subject_data = {}

        for session_name in sessions:
            npz_path = root / f"{subject_id}_{session_name}.npz"
            yaml_path = root / f"{subject_id}_{session_name}.yml"

            if not npz_path.exists() or not yaml_path.exists():
                continue

            with np.load(npz_path) as npz_data:
                if "data" not in npz_data:
                    raise KeyError(f"'data' not found in {npz_path}")
                if "stim" not in npz_data:
                    raise KeyError(f"'stim' not found in {npz_path}")

                data = np.asarray(npz_data["data"])
                stim = np.asarray(npz_data["stim"])

            with open(yaml_path, "r") as file:
                metadata = yaml.safe_load(file)

            acquisition = metadata["acquisition"]
            stim_metadata = metadata["stim"]

            sampling_rate = float(acquisition["samplingrate"])
            channel_names = list(acquisition["sensors"])
            offset = int(stim_metadata["offset"])
            window_length = int(stim_metadata["windowlength"])

            if data.ndim != 2:
                raise ValueError(
                    f"Expected 2-D EEG data in {npz_path}, found {data.shape}."
                )
            if stim.ndim != 1:
                raise ValueError(
                    f"Expected 1-D stimulus vector in {npz_path}, found {stim.shape}."
                )
            if len(data) != len(stim):
                raise ValueError(f"Data/stim length mismatch in {npz_path}.")
            if data.shape[1] != len(channel_names):
                raise ValueError(
                    f"Signal/metadata channel mismatch in {npz_path}."
                )

            data, selected_channels = _prepare_weibo_channels(
                data, channel_names, channels
            )
            code_to_label = _prepare_weibo_labels(metadata, classes)

            start_offset = offset + int(round(tmin * sampling_rate))
            end_offset = offset + int(round(tmax * sampling_rate))

            if end_offset > offset + window_length:
                raise ValueError(
                    f"Requested interval [{tmin}, {tmax}] s extends "
                    "beyond the motor-imagery window."
                )

            expected_samples = end_offset - start_offset
            X, y = [], []

            for event_sample in np.flatnonzero(stim != 0):
                event_code = int(stim[event_sample])

                if event_code not in code_to_label:
                    continue

                start = event_sample + start_offset
                end = event_sample + end_offset

                if end > len(data):
                    continue

                trial = data[start:end].T

                if trial.shape[1] != expected_samples:
                    continue

                # Dataset values are stored in µV; convert to volts.
                X.append(
                    trial.astype(np.float32, copy=False) * 1e-6
                )
                y.append(code_to_label[event_code])

            if not X:
                if verbose:
                    print(f"No usable trials found in {npz_path.name}.")
                continue

            X = np.stack(X).astype(np.float32, copy=False)
            y = np.asarray(y, dtype=str)

            if len(X) != len(y):
                raise ValueError(
                    f"Trial/label mismatch in {subject_id} {session_name}: "
                    f"{len(X)} trials vs {len(y)} labels."
                )

            subject_data[session_name] = {
                "X": X,
                "y": y,
                "channel_names": selected_channels.copy(),
                "sampling_rate": sampling_rate,
            }

        if subject_data:
            all_data[subject_id] = subject_data

    return all_data


# ============================================================
# Dataset preparation
# ============================================================

def prepare_weibo(params):
    root = params.get("root")

    if root is None:
        raise ValueError("'root' must be specified for Weibo2014.")

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
        loader=load_weibo2014_data,
        loader_kwargs={"root": root},
        loader_config=loader_config,
        filter_config=filter_config,
        feature_config=feature_config,
        dataset_name=DATASET_NAME,
        representation=representation,
        subjects=subjects,
        subject_batch_size=params.get("subject_batch_size", 5),
        band_labels=band_labels,
        session_name=None,
        metadata=metadata,
        show_progress=params.get("show_progress", False),
    )