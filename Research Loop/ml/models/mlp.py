import torch.nn as nn


class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=(128, 64),
    ):
        super().__init__()

        layers = []
        dims = [input_dim, *hidden_dims]

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
            ])

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