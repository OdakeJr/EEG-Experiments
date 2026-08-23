# plugins/eeg/zhou.py

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from plugins.pipe import Pipe, ExperimentRecord
from plugins.eeg.lib.preparation import prepare_eeg_dataframe
from plugins.eeg.lib.filtering import bands
import plugins.eeg.lib.feature_extraction as fe


# ============================================================
# Dataset constants
# ============================================================

DATASET_NAME = "zhou2016"
ORIGINAL_SAMPLING_RATE = 250.0


# ============================================================
# Label definitions
# ============================================================

ZHOU_LABEL_MAP = {
    "left_hand": "left_hand_imagery",
    "right_hand": "right_hand_imagery",
    "feet": "both_feet_imagery",
}


# ============================================================
# Default loading configuration
# ============================================================

DEFAULT_LOAD_CONFIG = {
    "subjects": list(range(1, 5)),

    "sessions": [
        "session_01",
        "session_02",
        "session_03",
    ],

    "tmin": 0.5,
    "tmax": 3.5,

    # None keeps all available classes.
    "classes": None,

    # None keeps all EEG channels.
    "channels": None,

    "verbose": False,
}


# ============================================================
# Feature configuration
# ============================================================

FEATURE_FUNCTIONS = {
    "mean": fe.extract_mean,
    "std": fe.extract_std,
    "mom": fe.extract_moments,
    "min": fe.extract_min,
    "max": fe.extract_max,
    "cov": fe.extract_covariance,
    "eig": fe.extract_eigenvalues,
    "logcov": fe.extract_logcov,
    "fft": fe.extract_fft,
    "h_diff": fe.extract_halves_diff,
    "q_stats": fe.extract_quarters_stats,
    "logvar": fe.extract_logvar,
    "bandpower": fe.extract_bandpower,
}


DEFAULT_FEATURE_CONFIG = {
    "mean": {},
    "std": {},
    "mom": {},
    "min": {},
    "max": {},
    "cov": {},
    "eig": {},
    "h_diff": {},
    "q_stats": {},
    "logvar": {},
}


# ============================================================
# Configuration helpers
# ============================================================

def _merge_config(user_config=None):
    config = deepcopy(DEFAULT_LOAD_CONFIG)

    if user_config is not None:
        config.update(user_config)

    return config


def _build_extract_config(feature_config=None):
    """
    Convert JSON-friendly feature configuration into the
    configuration expected by feature_extraction.py.
    """

    if feature_config is None:
        feature_config = DEFAULT_FEATURE_CONFIG

    extract_config = {}

    for feature_name, feature_params in feature_config.items():

        if feature_name not in FEATURE_FUNCTIONS:
            raise ValueError(
                f"Unknown EEG feature '{feature_name}'. "
                f"Available features: "
                f"{sorted(FEATURE_FUNCTIONS)}"
            )

        if feature_params is False:
            continue

        if feature_params is None:
            feature_params = {}

        if not isinstance(feature_params, dict):
            raise TypeError(
                f"Parameters for feature '{feature_name}' "
                "must be a dictionary."
            )

        extract_config[feature_name] = {
            "function": FEATURE_FUNCTIONS[feature_name],
            "params": deepcopy(feature_params),
        }

    if not extract_config:
        raise ValueError(
            "At least one feature must be enabled."
        )

    return extract_config


# ============================================================
# Channel handling
# ============================================================

def _prepare_zhou_channels(
    data,
    channel_names,
    channels=None,
):
    """
    Select and reorder Zhou2016 EEG channels.
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

    if len(selected_channels) != len(
        set(selected_channels)
    ):
        raise ValueError(
            "The channel selection contains duplicate names."
        )

    missing_channels = [
        channel
        for channel in selected_channels
        if channel not in channel_names
    ]

    if missing_channels:
        raise ValueError(
            "Requested channels are not available "
            f"in Zhou2016: {missing_channels}."
        )

    indices = [
        channel_names.index(channel)
        for channel in selected_channels
    ]

    data = data[:, indices]

    return data, selected_channels


# ============================================================
# Label handling
# ============================================================

def _prepare_zhou_labels(
    metadata,
    selected_classes=None,
):
    """
    Map Zhou2016 stimulus codes to semantic labels.
    """

    yaml_labels = metadata["stim"]["labels"]

    unknown_labels = [
        label_name
        for label_name in yaml_labels
        if label_name not in ZHOU_LABEL_MAP
    ]

    if unknown_labels:
        raise ValueError(
            "Unknown Zhou2016 labels: "
            f"{unknown_labels}"
        )

    code_to_label = {
        int(code): ZHOU_LABEL_MAP[label_name]
        for label_name, code
        in yaml_labels.items()
    }

    if selected_classes is None:
        return code_to_label

    selected_classes = list(
        selected_classes
    )

    valid_classes = set(
        ZHOU_LABEL_MAP.values()
    )

    unknown_classes = [
        label
        for label in selected_classes
        if label not in valid_classes
    ]

    if unknown_classes:
        raise ValueError(
            "Requested classes are not available "
            f"in Zhou2016: {unknown_classes}."
        )

    return {
        code: label
        for code, label in code_to_label.items()
        if label in selected_classes
    }


# ============================================================
# Dataset-specific loader
# ============================================================

def load_zhou2016_data(
    root,
    config=None,
):
    """
    Load Zhou2016 into the standardized intermediate EEG
    representation.

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
            f"Zhou2016 directory not found: {root}"
        )

    if tmin < 0:
        raise ValueError(
            "'tmin' must be greater than or equal to zero."
        )

    if tmax <= tmin:
        raise ValueError(
            "'tmax' must be greater than 'tmin'."
        )

    all_data = {}

    for subject in subjects:

        subject_id = f"subject_{subject:02d}"
        subject_data = {}

        for session_name in sessions:

            npz_path = (
                root
                / f"{subject_id}_{session_name}.npz"
            )

            yaml_path = (
                root
                / f"{subject_id}_{session_name}.yml"
            )

            if not npz_path.exists():
                continue

            if not yaml_path.exists():
                continue

            # --------------------------------------------------
            # Load recording and metadata
            # --------------------------------------------------

            npz_data = np.load(
                npz_path
            )

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

            with open(
                yaml_path,
                "r",
            ) as file:
                metadata = yaml.safe_load(file)

            sampling_rate = float(
                metadata[
                    "acquisition"
                ][
                    "samplingrate"
                ]
            )

            channel_names = list(
                metadata[
                    "acquisition"
                ][
                    "sensors"
                ]
            )

            offset = int(
                metadata["stim"]["offset"]
            )

            window_length = int(
                metadata[
                    "stim"
                ][
                    "windowlength"
                ]
            )

            # --------------------------------------------------
            # Safety checks
            # --------------------------------------------------

            if data.ndim != 2:
                raise ValueError(
                    f"Expected 2-D EEG data in "
                    f"{npz_path}, found {data.shape}."
                )

            if stim.ndim != 1:
                raise ValueError(
                    f"Expected 1-D stimulus vector in "
                    f"{npz_path}, found {stim.shape}."
                )

            if len(data) != len(stim):
                raise ValueError(
                    "Data/stim length mismatch in "
                    f"{npz_path}."
                )

            if data.shape[1] != len(
                channel_names
            ):
                raise ValueError(
                    "Signal/metadata channel mismatch "
                    f"in {npz_path}."
                )

            # --------------------------------------------------
            # Channels
            # --------------------------------------------------

            data, selected_channels = (
                _prepare_zhou_channels(
                    data=data,
                    channel_names=channel_names,
                    channels=channels,
                )
            )

            # --------------------------------------------------
            # Labels
            # --------------------------------------------------

            code_to_label = (
                _prepare_zhou_labels(
                    metadata=metadata,
                    selected_classes=classes,
                )
            )

            # --------------------------------------------------
            # Analysis window
            # --------------------------------------------------

            start_offset = (
                offset
                + int(
                    round(
                        tmin
                        * sampling_rate
                    )
                )
            )

            end_offset = (
                offset
                + int(
                    round(
                        tmax
                        * sampling_rate
                    )
                )
            )

            if (
                end_offset
                > offset + window_length
            ):
                raise ValueError(
                    f"Requested interval "
                    f"[{tmin}, {tmax}] s extends "
                    "beyond the motor-imagery window."
                )

            expected_samples = (
                end_offset
                - start_offset
            )

            # --------------------------------------------------
            # Extract trials
            # --------------------------------------------------

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
                    event_sample
                    + start_offset
                )

                end = (
                    event_sample
                    + end_offset
                )

                if end > len(data):
                    continue

                trial = data[
                    start:end,
                    :
                ].T

                if (
                    trial.shape[1]
                    != expected_samples
                ):
                    continue

                # FII representation stores values in µV.
                # Convert to volts.
                trial = (
                    trial.astype(
                        np.float32,
                        copy=False,
                    )
                    * 1e-6
                )

                X.append(trial)

                y.append(
                    code_to_label[
                        event_code
                    ]
                )

            if not X:
                if verbose:
                    print(
                        "No usable trials found in "
                        f"{npz_path.name}."
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

            # --------------------------------------------------
            # Store session
            # --------------------------------------------------

            subject_data[
                session_name
            ] = {
                "X": X,
                "y": y,
                "channel_names": (
                    selected_channels.copy()
                ),
                "sampling_rate": (
                    sampling_rate
                ),
            }

        if subject_data:
            all_data[
                subject_id
            ] = subject_data

    return all_data


# ============================================================
# Pipe
# ============================================================

class Zhou2016Pipe(Pipe):
    """
    Prepare the Zhou2016 EEG dataset.

    NPZ/YAML
        -> trial extraction
        -> channel/label harmonization
        -> filtering/resampling
        -> feature extraction
        -> EEGDataFrame
    """

    def expand(
        self,
        input_nodes,
        params,
    ):
        return [
            {
                "inputs": [],
                "params": deepcopy(params),
            }
        ]

    def run(
        self,
        inputs,
        params,
    ):

        if inputs:
            raise ValueError(
                "Zhou2016Pipe is a source pipe and "
                "does not accept upstream inputs."
            )

        # ====================================================
        # Path
        # ====================================================

        root = params.get("root")

        if root is None:
            raise ValueError(
                "'root' must be specified "
                "for Zhou2016."
            )

        # ====================================================
        # Loader configuration
        # ====================================================

        loader_config = deepcopy(
            params.get(
                "loader",
                {},
            )
        )

        subjects = loader_config.get(
            "subjects",
            DEFAULT_LOAD_CONFIG["subjects"],
        )

        # ====================================================
        # Filtering
        # ====================================================

        filter_config = deepcopy(
            params.get(
                "filter",
                {},
            )
        )

        filter_config["original_fs"] = (
            ORIGINAL_SAMPLING_RATE
        )

        band_labels = (
            filter_config
            .get("bandpass", {})
            .get("bands", bands)
        )

        # ====================================================
        # Features
        # ====================================================

        feature_config = deepcopy(
            params.get(
                "features",
                DEFAULT_FEATURE_CONFIG,
            )
        )

        extract_config = (
            _build_extract_config(
                feature_config
            )
        )

        # ====================================================
        # Other configuration
        # ====================================================

        subject_batch_size = params.get(
            "subject_batch_size",
            2,
        )

        metadata = deepcopy(
            params.get(
                "metadata",
                {},
            )
        )

        metadata.setdefault(
            "dataset",
            DATASET_NAME,
        )

        # ====================================================
        # Shared preparation
        # ====================================================

        eeg_data = prepare_eeg_dataframe(
            loader=load_zhou2016_data,

            loader_kwargs={
                "root": root,
            },

            loader_config=loader_config,

            filter_config=filter_config,

            extract_config=extract_config,

            dataset_name=DATASET_NAME,

            subjects=subjects,

            subject_batch_size=(
                subject_batch_size
            ),

            band_labels=band_labels,

            # Preserve the three actual sessions.
            session_name=None,

            metadata=metadata,

            show_progress=params.get(
                "show_progress",
                False,
            ),
        )

        # ====================================================
        # Experiment record
        # ====================================================

        record = ExperimentRecord()

        record.set(
            "dataset",
            {
                "name": DATASET_NAME,
                "subjects": list(subjects),
                "channels": list(
                    eeg_data.channels
                ),
                "original_sampling_rate": (
                    ORIGINAL_SAMPLING_RATE
                ),
                "sampling_rate": (
                    eeg_data.sampling_rate
                ),
                "classes": deepcopy(
                    loader_config.get(
                        "classes",
                        DEFAULT_LOAD_CONFIG["classes"],
                    )
                ),
                "sessions": deepcopy(
                    loader_config.get(
                        "sessions",
                        DEFAULT_LOAD_CONFIG["sessions"],
                    )
                ),
                "n_rows": len(
                    eeg_data.data
                ),
                "n_features": len(
                    eeg_data.feature_columns or []
                ),
            },
        )

        record.set(
            "preparation",
            {
                "loader": deepcopy(
                    loader_config
                ),
                "filter": deepcopy(
                    filter_config
                ),
                "features": deepcopy(
                    feature_config
                ),
            },
        )

        # ====================================================
        # Output
        # ====================================================

        return self.make_result(
            value=eeg_data,
            record=record,
        )