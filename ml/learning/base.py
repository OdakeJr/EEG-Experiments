# ml/learning/base.py


class BaseLearningAlgorithm:

    def fit(
        self,
        model,
        source=None,
        target_super_domain=None,
        target_elementary_domain=None,
        **training_params,
    ):
        raise NotImplementedError

    def predict(self, X, domains=None, super_domains=None):
        raise NotImplementedError

    def predict_proba(self, X, domains=None, super_domains=None):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        raise NotImplementedError