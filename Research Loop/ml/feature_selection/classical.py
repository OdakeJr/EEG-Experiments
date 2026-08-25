# ml/feature_selection/classical.py

from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)

from ml.feature_selection.base import FeatureTransformer


class VarianceSelector(FeatureTransformer):

    def __init__(self, threshold=0.0):
        self.selector = VarianceThreshold(threshold=threshold)

    def fit(self, X, y=None, domains=None):
        self.selector.fit(X)
        return self

    def transform(self, X, domains=None):
        return self.selector.transform(X)


class ANOVASelector(FeatureTransformer):

    def __init__(self, k=10):
        self.selector = SelectKBest(score_func=f_classif, k=k)

    def fit(self, X, y=None, domains=None):
        self.selector.fit(X, y)
        return self

    def transform(self, X, domains=None):
        return self.selector.transform(X)


class MutualInformationSelector(FeatureTransformer):

    def __init__(self, k=10):
        self.selector = SelectKBest(score_func=mutual_info_classif, k=k)

    def fit(self, X, y=None, domains=None):
        self.selector.fit(X, y)
        return self

    def transform(self, X, domains=None):
        return self.selector.transform(X)