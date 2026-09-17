# ml/models/eeg_tcnet.py

import torch.nn as nn


class EEGTCNet(nn.Module):
    input_representation = "signal"

    def __init__(self, input_shape, output_dim, **params):
        super().__init__()

        if len(input_shape) == 2:
            n_chans, n_times = input_shape
        elif len(input_shape) == 3 and input_shape[0] == 1:
            _, n_chans, n_times = input_shape
        else:
            raise ValueError(
                f"EEGTCNet expects [C,T] or [1,C,T], got {input_shape}."
            )

        try:
            from braindecode.models import EEGTCNet as BraindecodeEEGTCNet
        except ImportError as e:
            raise ImportError(
                "EEGTCNet requires a Braindecode version with EEGTCNet support."
            ) from e

        self.model = BraindecodeEEGTCNet(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=output_dim,
            **params,
        )

    def _prepare_input(self, X):
        if X.ndim == 4:
            if X.shape[1] != 1:
                raise ValueError(
                    f"EEGTCNet requires one signal band, got {tuple(X.shape)}."
                )
            X = X[:, 0]
        return X

    def extract_feature_map(self, X):
        X = self._prepare_input(X)
        X = self.model.arrange_dim_input(X)
        X = self.model.eegnet_tc(X)
        X = self.model.arrange_dim_eegnet(X)
        X = self.model.tcn_block(X)
        return X[:, -1, :]

    def extract_features(self, X):
        return self.extract_feature_map(X).flatten(1)

    def classify_feature_map(self, X):
        return self.model.final_layer(X)

    @property
    def classifier(self):
        return self.model.final_layer

    def forward(self, X):
        return self.model(self._prepare_input(X))