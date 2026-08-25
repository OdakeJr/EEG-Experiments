from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def logistic_regression(**params):
    return LogisticRegression(**params)


def random_forest(**params):
    return RandomForestClassifier(**params)


def svm(**params):
    params.setdefault("probability", True)
    return SVC(**params)