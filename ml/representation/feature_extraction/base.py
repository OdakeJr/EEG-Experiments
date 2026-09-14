# ml/representation/feature_extraction/base.py

from ml.representation.base import RepresentationTransformer


class FeatureExtractor(RepresentationTransformer):
    input_representation = "signal"
    output_representation = "features"