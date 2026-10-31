
# ml/models/mobilenet.py

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetV3Small(nn.Module):
    input_representation = "image"

    def __init__(self, input_shape, output_dim, pretrained=True, **params):
        super().__init__()

        if input_shape is not None and (
            len(input_shape) != 3 or input_shape[0] != 3
        ):
            raise ValueError(f"Expected RGB image input, got {input_shape}.")

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.model = mobilenet_v3_small(weights=weights)

        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, output_dim)

    def extract_feature_map(self, X):
        X = self.model.features(X)
        return torch.flatten(self.model.avgpool(X), 1)

    def extract_features(self, X):
        return self.extract_feature_map(X)

    def classify_feature_map(self, X):
        return self.model.classifier(X)

    def forward(self, X):
        return self.classify_feature_map(self.extract_feature_map(X))
