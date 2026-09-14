# ml/representation/signal_transform/standardize.py

import numpy as np

from ml.representation.base import RepresentationTransformer


class StandardizeSignalTransformer(RepresentationTransformer):
    input_representation = "signal"
    output_representation = "signal"

    def __init__(self, mode="channel", scale=1.0, eps=1e-8):
        super().__init__(pre_scaler=None, post_scaler=None)

        if mode not in {"channel", "trial"}:
            raise ValueError("mode must be 'channel' or 'trial'.")

        self.mode = mode
        self.scale = scale
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def _fit(self, X, y=None, domains=None):
        X = np.asarray(X, dtype=np.float32) * self.scale

        if self.mode == "channel":
            self.mean_ = X.mean(axis=(0, -1), keepdims=True)
            self.std_ = np.maximum(X.std(axis=(0, -1), keepdims=True), self.eps)

        return self

    def _transform(self, X, domains=None):
        X = np.asarray(X, dtype=np.float32) * self.scale

        if self.mode == "channel":
            return (X - self.mean_) / self.std_

        mean = X.mean(axis=-1, keepdims=True)
        std = np.maximum(X.std(axis=-1, keepdims=True), self.eps)

        return (X - mean) / std