# ml/representation/feature_selection/classical.py

from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, mutual_info_classif

from ml.representation.feature_selection.base import FeatureSelector


class VarianceSelector(FeatureSelector):
    def __init__(self, threshold=0.0, pre_scaler=None, post_scaler=None):
        super().__init__(pre_scaler=pre_scaler, post_scaler=post_scaler)
        self.selector = VarianceThreshold(threshold=threshold)

    def _fit(self, X, y=None, domains=None):
        self.selector.fit(X)
        return self

    def _transform(self, X, domains=None):
        return self.selector.transform(X)


class ANOVASelector(FeatureSelector):
    def __init__(self, k=10, pre_scaler=None, post_scaler=None):
        super().__init__(pre_scaler=pre_scaler, post_scaler=post_scaler)
        self.selector = SelectKBest(score_func=f_classif, k=k)

    def _fit(self, X, y=None, domains=None):
        if y is None:
            raise ValueError("ANOVASelector requires labels.")
        self.selector.fit(X, y)
        return self

    def _transform(self, X, domains=None):
        return self.selector.transform(X)


class MutualInformationSelector(FeatureSelector):
    def __init__(self, k=10, pre_scaler=None, post_scaler=None):
        super().__init__(pre_scaler=pre_scaler, post_scaler=post_scaler)
        self.selector = SelectKBest(score_func=mutual_info_classif, k=k)

    def _fit(self, X, y=None, domains=None):
        if y is None:
            raise ValueError("MutualInformationSelector requires labels.")
        self.selector.fit(X, y)
        return self

    def _transform(self, X, domains=None):
        return self.selector.transform(X)