# ml/representation/feature_extraction/handcrafted.py

import numpy as np

from eeg.lib.feature_extraction import build_extract_config, extract_features_from_trial
from ml.representation.feature_extraction.base import FeatureExtractor


class HandcraftedFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        features=None,
        channel_names=None,
        band_labels=None,
        pre_scaler=None,
        post_scaler=None,
    ):
        if pre_scaler is not None:
            raise ValueError("Handcrafted feature extraction does not support pre_scaling.")

        super().__init__(pre_scaler=None, post_scaler=post_scaler)
        self.features = features
        self.channel_names = channel_names
        self.band_labels = band_labels
        self.extract_config_ = None
        self.feature_names_ = None

    def _as_trials(self, X):
        X = np.asarray(X)

        if X.ndim not in {3, 4}:
            raise ValueError(
                f"Handcrafted features expect [N,C,T] or [N,B,C,T], got {X.shape}."
            )

        return X

    def _extract_trial(self, trial):
        values, names = extract_features_from_trial(
            trial,
            self.extract_config_,
            channel_names=self.channel_names,
            band_labels=self.band_labels,
        )
        return np.asarray(values, dtype=np.float64), names

    def _fit(self, X, y=None, domains=None):
        X = self._as_trials(X)
        self.extract_config_ = build_extract_config(self.features)

        values, names = self._extract_trial(X[0])
        self.feature_names_ = names

        if not np.isfinite(values).all():
            raise ValueError("Non-finite handcrafted feature values found during fit.")

        return self

    def _transform(self, X, domains=None):
        if self.extract_config_ is None:
            raise RuntimeError("HandcraftedFeatureExtractor must be fitted before transform().")

        rows = []
        for trial in self._as_trials(X):
            values, names = self._extract_trial(trial)

            if names != self.feature_names_:
                raise ValueError("Handcrafted feature names changed between trials.")
            if not np.isfinite(values).all():
                raise ValueError("Non-finite handcrafted feature values found.")

            rows.append(values)

        return np.vstack(rows)