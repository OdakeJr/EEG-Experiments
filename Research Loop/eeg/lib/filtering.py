from copy import deepcopy
from fractions import Fraction

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


# Frequency bands used during feature extraction
bands = [
    (8, 12),   # Mu / alpha
    (13, 30),  # Beta
]


DEFAULT_FILTER_CONFIG = {
    # Must be specified for each dataset:
    # BCI IV 2a = 250 Hz
    # PhysioNet EEGMMIDB = 160 Hz
    "original_fs": None,

    "bandpass": {
        "enabled": True,
        "bands": bands,
        "order": 5,
        "stack_bands": True,
    },

    "resample": {
        "enabled": True,
        "new_fs": 128.0,
    },

    # Reduce memory consumption
    "output_dtype": np.float32,
}


def _merge_filter_config(user_config=None):
    """
    Merge a user configuration with DEFAULT_FILTER_CONFIG.
    """
    config = deepcopy(DEFAULT_FILTER_CONFIG)

    if user_config is None:
        return config

    for key, value in user_config.items():
        if (
            isinstance(value, dict)
            and key in config
            and isinstance(config[key], dict)
        ):
            config[key].update(value)
        else:
            config[key] = value

    return config


def bandpass_filter(data, fs, band, order=5):
    """
    Apply a zero-phase Butterworth band-pass filter.

    Parameters
    ----------
    data : ndarray
        EEG data with time on the last axis.
    fs : float
        Sampling frequency in Hz.
    band : tuple
        Lower and upper cutoff frequencies in Hz.
    order : int
        Butterworth filter order.

    Returns
    -------
    ndarray
        Filtered EEG data.
    """
    fs = float(fs)
    low_frequency, high_frequency = band
    nyquist = fs / 2.0

    if low_frequency <= 0:
        raise ValueError(
            f"Lower cutoff must be greater than zero: {band}"
        )

    if high_frequency >= nyquist:
        raise ValueError(
            f"Upper cutoff {high_frequency} Hz must be below "
            f"the Nyquist frequency {nyquist} Hz."
        )

    if low_frequency >= high_frequency:
        raise ValueError(
            f"Invalid frequency band: {band}"
        )

    sos = butter(
        N=order,
        Wn=[low_frequency, high_frequency],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    return sosfiltfilt(
        sos,
        data,
        axis=-1,
    )


def resample_data(data, old_fs, new_fs):
    """
    Resample EEG data using polyphase filtering.

    Parameters
    ----------
    data : ndarray
        EEG data with time on the last axis.
    old_fs : float
        Original sampling frequency.
    new_fs : float
        Desired sampling frequency.

    Returns
    -------
    ndarray
        Resampled EEG data.
    """
    old_fs = float(old_fs)
    new_fs = float(new_fs)

    if old_fs <= 0 or new_fs <= 0:
        raise ValueError(
            "Sampling frequencies must be greater than zero."
        )

    if np.isclose(old_fs, new_fs):
        return data

    ratio = Fraction(
        new_fs / old_fs
    ).limit_denominator(1000)

    return resample_poly(
        data,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=-1,
    )


def apply_filters_to_array(X, config=None):
    """
    Apply band-pass filtering and optional resampling to an EEG array.

    Processing order:
        1. Filter using the original sampling frequency.
        2. Resample to the common sampling frequency.
        3. Convert to the configured output dtype.
    """
    config = _merge_filter_config(config)

    original_fs = config.get("original_fs")

    if original_fs is None:
        raise ValueError(
            "'original_fs' must be specified. "
            "Use 250 Hz for BCI IV 2a and 160 Hz for PhysioNet EEGMMIDB."
        )

    X = np.asarray(X)
    bandpass_config = config["bandpass"]
    resample_config = config["resample"]
    output_dtype = config.get("output_dtype", np.float32)

    # ========================================================
    # Band-pass filtering
    # ========================================================

    if bandpass_config.get("enabled", True):
        selected_bands = bandpass_config.get(
            "bands",
            bands,
        )
        order = bandpass_config.get(
            "order",
            5,
        )
        stack_bands = bandpass_config.get(
            "stack_bands",
            True,
        )

        filtered_bands = [
            bandpass_filter(
                data=X,
                fs=original_fs,
                band=band,
                order=order,
            )
            for band in selected_bands
        ]

        if len(filtered_bands) == 1:
            X_out = filtered_bands[0]

        elif stack_bands:
            X_out = np.stack(
                filtered_bands,
                axis=0,
            )

        else:
            X_out = filtered_bands

    else:
        X_out = X.copy()

    # ========================================================
    # Resampling
    # ========================================================

    if resample_config.get("enabled", True):
        new_fs = resample_config.get(
            "new_fs",
            128.0,
        )

        if isinstance(X_out, list):
            X_out = [
                resample_data(
                    data=band_data,
                    old_fs=original_fs,
                    new_fs=new_fs,
                )
                for band_data in X_out
            ]

        else:
            X_out = resample_data(
                data=X_out,
                old_fs=original_fs,
                new_fs=new_fs,
            )

    # ========================================================
    # Reduce memory usage
    # ========================================================

    if isinstance(X_out, list):
        X_out = [
            band_data.astype(
                output_dtype,
                copy=False,
            )
            for band_data in X_out
        ]

    else:
        X_out = X_out.astype(
            output_dtype,
            copy=False,
        )

    return X_out


def apply_filters_to_dataset(dataset, config=None):
    """
    Apply filtering and resampling to every subject/session in a dataset.
    """
    config = _merge_filter_config(config)
    filtered_dataset = deepcopy(dataset)

    original_fs = config["original_fs"]
    resample_config = config["resample"]

    if resample_config.get("enabled", True):
        final_fs = resample_config.get(
            "new_fs",
            128.0,
        )
    else:
        final_fs = original_fs

    for subject_id, sessions in filtered_dataset.items():
        for session_name, session_data in sessions.items():

            X = session_data["X"]

            filtered_dataset[subject_id][session_name]["X"] = (
                apply_filters_to_array(
                    X=X,
                    config=config,
                )
            )

            filtered_dataset[subject_id][session_name][
                "sampling_rate"
            ] = float(final_fs)

    return filtered_dataset