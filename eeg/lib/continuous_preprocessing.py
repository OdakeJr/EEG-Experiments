import numpy as np


def apply_continuous_preprocessing(raw, config=None, verbose=False):
    config = config or {}

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scale = float(config.get("scale", 1.0))

    if scale != 1.0:
        raw.apply_function(
            lambda X: X * scale,
            picks="eeg",
            channel_wise=False,
            verbose=verbose,
        )

    # --------------------------------------------------------
    # Bandpass
    # --------------------------------------------------------

    bandpass = config.get("bandpass", {})

    if bandpass.get("enabled", False):
        raw.filter(
            l_freq=bandpass.get("l_freq"),
            h_freq=bandpass.get("h_freq"),
            picks="eeg",
            verbose=verbose,
        )

    # --------------------------------------------------------
    # Exponential moving standardization
    # --------------------------------------------------------

    exp = config.get("exponential_standardize", {})

    if exp.get("enabled", False):
        try:
            from braindecode.preprocessing import exponential_moving_standardize
        except ImportError as e:
            raise ImportError(
                "Exponential standardization requires Braindecode."
            ) from e

        raw.apply_function(
            exponential_moving_standardize,
            picks="eeg",
            channel_wise=False,
            verbose=verbose,
            factor_new=exp.get("factor_new", 1e-3),
            init_block_size=exp.get("init_block_size", 1000),
            eps=exp.get("eps", 1e-4),
        )

    return raw