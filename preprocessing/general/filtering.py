from copy import deepcopy
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample

bands = [(8,12), (13,30)]

DEFAULT_FILTER_CONFIG = {
    "bandpass": {
        "enabled": True,
        "fs": 250.0,
        "bands": bands,
        "order": 5,
        "stack_bands": True,
    },
    "downsample": {
        "enabled": False,
        "old_fs": 250,
        "new_fs": 128,
    }
}

def _merge_filter_config(user_config=None):
    config = deepcopy(DEFAULT_FILTER_CONFIG)
    if user_config is None:
        return config
    for key, value in user_config.items():
        if isinstance(value, dict) and key in config:
            config[key].update(value)
        else:
            config[key] = value
    return config

def bandpass_filter(data, fs=250.0, band=(4, 40), order=5):
    nyq = fs / 2.0
    low = band[0] / nyq
    high = band[1] / nyq
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=-1)

def downsample_data(data, old_fs=250, new_fs=128):
    n_new = int(data.shape[-1] * new_fs / old_fs)
    return resample(data, n_new, axis=-1)

def apply_filters_to_array(X, config=None):
    config = _merge_filter_config(config)
    X_out = X.copy()

    downsample_cfg = config.get("downsample", {})
    if downsample_cfg.get("enabled", False):
        X_out = downsample_data(
            X_out,
            old_fs=downsample_cfg.get("old_fs", 250),
            new_fs=downsample_cfg.get("new_fs", 128),
        )

    bandpass_cfg = config.get("bandpass", {})
    if bandpass_cfg.get("enabled", False):
        bands = bandpass_cfg.get("bands", [(4, 40)])
        fs = bandpass_cfg.get("fs", 250.0)
        order = bandpass_cfg.get("order", 5)
        stack_bands = bandpass_cfg.get("stack_bands", True)

        filtered_bands = [
            bandpass_filter(X_out, fs=fs, band=band, order=order)
            for band in bands
        ]

        if len(filtered_bands) == 1:
            X_out = filtered_bands[0]
        elif stack_bands:
            X_out = np.stack(filtered_bands, axis=0)
        else:
            X_out = filtered_bands

    return X_out

def apply_filters_to_dataset(dataset, config=None):
    filtered_dataset = deepcopy(dataset)

    for subj_id, sessions in filtered_dataset.items():
        for sess_name, sess_data in sessions.items():
            X = sess_data["X"]
            filtered_dataset[subj_id][sess_name]["X"] = apply_filters_to_array(
                X, config=config
            )

    return filtered_dataset