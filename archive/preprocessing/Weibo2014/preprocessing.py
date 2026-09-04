# preprocessing.py

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml


# ============================================================
# Weibo2014 label definitions
# ============================================================

# Labels used in the FII representation are converted into the
# same semantic naming convention used by the other datasets.
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

    # Analysis window relative to the beginning of the
    # motor-imagery interval defined by the YAML offset.
    "tmin": 0.5,
    "tmax": 3.5,

    # None keeps all available classes.
    # Example common subset:
    # [
    #     "left_hand_imagery",
    #     "right_hand_imagery",
    #     "both_feet_imagery",
    # ]
    "classes": None,

    # None keeps all EEG electrodes.
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


def _prepare_weibo_channels(
    data,
    channel_names,
    channels=None,
):
    """
    Optionally select a subset of Weibo2014 EEG electrodes.

    Parameters
    ----------
    data : ndarray
        Continuous EEG data with shape
        (n_samples, n_channels).

    channel_names : list of str
        Channel names stored in the YAML metadata.

    channels : list of str or None
        Channels to retain. When None, all EEG electrodes
        are retained.

    Returns
    -------
    data : ndarray
        EEG data containing only the selected channels.

    selected_channels : list of str
        Channel names in the returned order.
    """
    channel_names = list(channel_names)

    if channels is None:
        selected_channels = channel_names.copy()
    else:
        selected_channels = list(channels)

    if not selected_channels:
        raise ValueError(
            "The channel list cannot be empty."
        )

    missing_channels = [
        channel
        for channel in selected_channels
        if channel not in channel_names
    ]

    if missing_channels:
        raise ValueError(
            "Requested channels are not available in Weibo2014: "
            f"{missing_channels}. "
            f"Available EEG channels: {channel_names}"
        )

    if len(selected_channels) != len(set(selected_channels)):
        raise ValueError(
            "The channel selection contains duplicate names."
        )

    indices = [
        channel_names.index(channel)
        for channel in selected_channels
    ]

    data = data[:, indices]

    return data, selected_channels


def _prepare_weibo_labels(
    metadata,
    selected_classes=None,
):
    """
    Build the mapping between stimulus codes and semantic labels.

    Parameters
    ----------
    metadata : dict
        YAML metadata associated with the recording.

    selected_classes : list of str or None
        Semantic classes to retain. When None, all available
        Weibo2014 classes are retained.

    Returns
    -------
    code_to_label : dict
        Mapping from numeric stimulus code to semantic label.
    """
    yaml_labels = metadata["stim"]["labels"]

    unknown_labels = [
        label_name
        for label_name in yaml_labels
        if label_name not in WEIBO_LABEL_MAP
    ]

    if unknown_labels:
        raise ValueError(
            "Unknown Weibo2014 labels found in metadata: "
            f"{unknown_labels}"
        )

    code_to_label = {
        int(code): WEIBO_LABEL_MAP[label_name]
        for label_name, code in yaml_labels.items()
    }

    if selected_classes is None:
        return code_to_label

    selected_classes = list(selected_classes)

    valid_classes = set(WEIBO_LABEL_MAP.values())

    unknown_classes = [
        label
        for label in selected_classes
        if label not in valid_classes
    ]

    if unknown_classes:
        raise ValueError(
            "Requested classes are not available in Weibo2014: "
            f"{unknown_classes}. "
            f"Available classes: {sorted(valid_classes)}"
        )

    code_to_label = {
        code: label
        for code, label in code_to_label.items()
        if label in selected_classes
    }

    return code_to_label


def load_weibo2014_data(
    root,
    config=None,
):
    """
    Load Weibo2014 data into a nested dictionary.

    Labels are stored as dataset-specific semantic strings:

        left_hand_imagery
        right_hand_imagery
        both_feet_imagery
        both_hands_imagery
        rest

    Parameters
    ----------
    root : str or Path
        Directory containing the NPZ and YAML files.

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

            "y": ndarray of semantic string labels,

            "channel_names": list of electrode names,

            "sampling_rate": sampling frequency in Hz
        }
    """
    config = _merge_config(config)

    root = Path(root)

    subjects = config["subjects"]
    sessions = config["sessions"]
    tmin = config["tmin"]
    tmax = config["tmax"]
    classes = config["classes"]
    channels = config["channels"]
    verbose = config["verbose"]

    if not root.exists():
        raise FileNotFoundError(
            f"Weibo2014 directory not found: {root}"
        )

    if tmin < 0:
        raise ValueError(
            "tmin must be greater than or equal to zero."
        )

    if tmax <= tmin:
        raise ValueError(
            "tmax must be greater than tmin."
        )

    all_data = {}

    for subject in subjects:

        subject_id = f"subject_{subject:02d}"
        subject_data = {}

        for session_name in sessions:

            npz_path = (
                root /
                f"{subject_id}_{session_name}.npz"
            )

            yaml_path = (
                root /
                f"{subject_id}_{session_name}.yml"
            )

            if not npz_path.exists():
                print(
                    f"Warning: file not found: {npz_path}"
                )
                continue

            if not yaml_path.exists():
                print(
                    f"Warning: file not found: {yaml_path}"
                )
                continue

            # ==================================================
            # Load recording and metadata
            # ==================================================

            npz_data = np.load(npz_path)

            if "data" not in npz_data:
                raise KeyError(
                    f"'data' not found in {npz_path}"
                )

            if "stim" not in npz_data:
                raise KeyError(
                    f"'stim' not found in {npz_path}"
                )

            data = npz_data["data"]
            stim = npz_data["stim"]

            with open(yaml_path, "r") as file:
                metadata = yaml.safe_load(file)

            sampling_rate = float(
                metadata["acquisition"]["samplingrate"]
            )

            channel_names = list(
                metadata["acquisition"]["sensors"]
            )

            offset = int(
                metadata["stim"]["offset"]
            )

            window_length = int(
                metadata["stim"]["windowlength"]
            )

            # ==================================================
            # Safety checks
            # ==================================================

            if data.ndim != 2:
                raise ValueError(
                    f"Expected 2-D EEG data in {npz_path}, "
                    f"but found shape {data.shape}"
                )

            if stim.ndim != 1:
                raise ValueError(
                    f"Expected 1-D stimulus vector in {npz_path}, "
                    f"but found shape {stim.shape}"
                )

            if len(data) != len(stim):
                raise ValueError(
                    f"Data/stim length mismatch in {npz_path}: "
                    f"{len(data)} versus {len(stim)}"
                )

            if data.shape[1] != len(channel_names):
                raise ValueError(
                    f"Channel mismatch in {npz_path}: "
                    f"{data.shape[1]} signal channels versus "
                    f"{len(channel_names)} metadata channels."
                )

            # ==================================================
            # Select channels
            # ==================================================

            data, selected_channels = (
                _prepare_weibo_channels(
                    data=data,
                    channel_names=channel_names,
                    channels=channels,
                )
            )

            # ==================================================
            # Prepare semantic labels
            # ==================================================

            code_to_label = _prepare_weibo_labels(
                metadata=metadata,
                selected_classes=classes,
            )

            # ==================================================
            # Determine analysis window
            # ==================================================

            start_offset = (
                offset +
                int(round(tmin * sampling_rate))
            )

            end_offset = (
                offset +
                int(round(tmax * sampling_rate))
            )

            if end_offset > offset + window_length:
                raise ValueError(
                    f"Requested interval [{tmin}, {tmax}] s "
                    f"extends beyond the motor-imagery window "
                    f"in {yaml_path.name}."
                )

            expected_samples = (
                end_offset - start_offset
            )

            # ==================================================
            # Extract trials
            # ==================================================

            event_samples = np.flatnonzero(
                stim != 0
            )

            X = []
            y = []

            for event_sample in event_samples:

                event_code = int(
                    stim[event_sample]
                )

                if event_code not in code_to_label:
                    continue

                start = (
                    event_sample +
                    start_offset
                )

                end = (
                    event_sample +
                    end_offset
                )

                if end > len(data):
                    if verbose:
                        print(
                            f"Warning: incomplete trial at "
                            f"sample {event_sample} in "
                            f"{npz_path.name}"
                        )
                    continue

                trial = data[
                    start:end,
                    :
                ].T

                if trial.shape[1] != expected_samples:
                    if verbose:
                        print(
                            f"Warning: invalid trial shape "
                            f"{trial.shape} in "
                            f"{npz_path.name}"
                        )
                    continue

                # FII stores EEG values in microvolts.
                # Convert to volts to match MNE-loaded datasets.
                trial = (
                    trial.astype(
                        np.float32,
                        copy=False,
                    )
                    * 1e-6
                )

                X.append(trial)

                y.append(
                    code_to_label[event_code]
                )

            # ==================================================
            # Store session
            # ==================================================

            if not X:
                print(
                    f"Warning: no usable trials found in "
                    f"{npz_path.name}"
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

            subject_data[session_name] = {
                "X": X,
                "y": y,
                "channel_names": selected_channels.copy(),
                "sampling_rate": sampling_rate,
            }

        if subject_data:
            all_data[subject_id] = subject_data

    return all_data