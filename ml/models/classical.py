# ml/models/classical.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def logistic_regression(input_shape=None, output_dim=None, **params):
    return LogisticRegression(**params)


def random_forest(input_shape=None, output_dim=None, **params):
    return RandomForestClassifier(**params)


def svm(input_shape=None, output_dim=None, **params):
    params.setdefault("probability", True)
    return SVC(**params)


logistic_regression.input_representation = "features"
random_forest.input_representation = "features"
svm.input_representation = "features"