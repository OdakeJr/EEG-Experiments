# ml/representation/feature_selection/base.py

from ml.representation.base import RepresentationTransformer


class FeatureSelector(RepresentationTransformer):
    input_representation = "features"
    output_representation = "features"