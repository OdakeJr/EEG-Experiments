import torch.nn as nn


class EEGNet(nn.Module):
    input_representation = "signal"

    def __init__(self, input_shape, output_dim, **params):
        super().__init__()

        if len(input_shape) == 2:
            n_chans, n_times = input_shape
        elif len(input_shape) == 3 and input_shape[0] == 1:
            _, n_chans, n_times = input_shape
        else:
            raise ValueError(
                f"EEGNet expects [C,T] or [1,C,T], got {input_shape}."
            )

        try:
            from braindecode.models import EEGNetv4 as BraindecodeEEGNet
        except ImportError as e:
            raise ImportError(
                "EEGNet requires Braindecode with EEGNetv4 support."
            ) from e

        self.model = BraindecodeEEGNet(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=output_dim,
            **params,
        )

    def _prepare_input(self, X):
        if X.ndim == 4:
            if X.shape[1] != 1:
                raise ValueError(
                    f"EEGNet requires one signal band, got {tuple(X.shape)}."
                )
            X = X[:, 0]

        return X

    def extract_features(self, X):
        X = self._prepare_input(X)

        for name, layer in self.model.named_children():
            if name == "final_layer":
                break
            X = layer(X)

        return X.flatten(1)

    def forward(self, X):
        return self.model(self._prepare_input(X))