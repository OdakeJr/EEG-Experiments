# ml/feature_selection/base.py

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    Normalizer,
)


SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "normalize": Normalizer,
}


class FeatureTransformer:

    def __init__(
        self,
        pre_scaler=None,
        post_scaler=None,
    ):
        self.pre_scaler = self._build_scaler(pre_scaler)
        self.post_scaler = self._build_scaler(post_scaler)

    def _build_scaler(self, config):
        if config is None:
            return None

        if isinstance(config, str):
            method = config
            params = {}
        else:
            method = config["method"]
            params = config.get("params", {})

        return SCALERS[method](**params)

    def fit(self, X, y=None, domains=None):

        if self.pre_scaler is not None:
            X = self.pre_scaler.fit_transform(X)

        self._fit(X, y, domains)

        X = self._transform(X, domains)

        if self.post_scaler is not None:
            self.post_scaler.fit(X)

        return self

    def transform(self, X, domains=None):

        if self.pre_scaler is not None:
            X = self.pre_scaler.transform(X)

        X = self._transform(X, domains)

        if self.post_scaler is not None:
            X = self.post_scaler.transform(X)

        return X

    def fit_transform(self, X, y=None, domains=None):
        self.fit(X, y, domains)
        return self.transform(X, domains)

    def _fit(self, X, y=None, domains=None):
        raise NotImplementedError

    def _transform(self, X, domains=None):
        raise NotImplementedError