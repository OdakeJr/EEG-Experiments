from ml.feature_selection.base import FeatureTransformer


class IdentitySignalTransformer(FeatureTransformer):
    input_representation = "signal"
    output_representation = "signal"

    def __init__(self):
        super().__init__(pre_scaler=None, post_scaler=None)

    def _fit(self, X, y=None, domains=None):
        return self

    def _transform(self, X, domains=None):
        return X