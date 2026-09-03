import torch.nn as nn


ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=(128, 64),
        activation="relu",
        dropout=0.0,
        batch_norm=False,
    ):
        super().__init__()

        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Available: {sorted(ACTIVATIONS)}"
            )

        activation_class = ACTIVATIONS[activation]

        layers = []
        dims = [input_dim, *hidden_dims]

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):

            layers.append(
                nn.Linear(in_dim, out_dim)
            )

            if batch_norm:
                layers.append(
                    nn.BatchNorm1d(out_dim)
                )

            layers.append(
                activation_class()
            )

            if dropout > 0:
                layers.append(
                    nn.Dropout(dropout)
                )

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Linear(
            dims[-1],
            output_dim,
        )

    def extract_features(self, X):
        return self.features(X)

    def forward(self, X):
        features = self.extract_features(X)
        return self.classifier(features)