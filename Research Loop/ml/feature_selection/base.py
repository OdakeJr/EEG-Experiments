# ml/feature_selection/base.py

class FeatureTransformer:

    def fit(self, X, y=None, domains=None):
        raise NotImplementedError

    def transform(self, X, domains=None):
        raise NotImplementedError

    def fit_transform(self, X, y=None, domains=None):
        self.fit(X, y, domains)
        return self.transform(X, domains)