# ml/feature_selection/base.py

from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler


SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "normalize": Normalizer,
}


class FeatureTransformer:
    input_representation = "features"
    output_representation = "features"

    def __init__(self, pre_scaler=None, post_scaler=None):
        self.pre_scaler = self._build_scaler(pre_scaler)
        self.post_scaler = self._build_scaler(post_scaler)

    def _build_scaler(self, config):
        if config is None:
            return None

        if isinstance(config, str):
            method, params = config, {}
        else:
            method = config["method"]
            params = config.get("params", {})

        if method not in SCALERS:
            raise ValueError(f"Unknown scaler: {method}")

        return SCALERS[method](**params)

    def _check_scaler_input(self, X, scaler_name):
        if X.ndim != 2:
            raise ValueError(
                f"{scaler_name} requires 2D feature input, got shape {X.shape}."
            )

    def fit(self, X, y=None, domains=None):
        if self.pre_scaler is not None:
            self._check_scaler_input(X, "pre_scaler")
            X = self.pre_scaler.fit_transform(X)

        self._fit(X, y, domains)
        X_transformed = self._transform(X, domains)

        if self.post_scaler is not None:
            self._check_scaler_input(X_transformed, "post_scaler")
            self.post_scaler.fit(X_transformed)

        return self

    def transform(self, X, domains=None):
        if self.pre_scaler is not None:
            self._check_scaler_input(X, "pre_scaler")
            X = self.pre_scaler.transform(X)

        X = self._transform(X, domains)

        if self.post_scaler is not None:
            self._check_scaler_input(X, "post_scaler")
            X = self.post_scaler.transform(X)

        return X

    def fit_transform(self, X, y=None, domains=None):
        self.fit(X, y, domains)
        return self.transform(X, domains)

    def _fit(self, X, y=None, domains=None):
        raise NotImplementedError

    def _transform(self, X, domains=None):
        raise NotImplementedError