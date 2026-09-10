# ml/feature_selection/signal_to_feature/base.py

from ml.feature_selection.base import FeatureTransformer


class SignalToFeatureTransformer(FeatureTransformer):
    input_representation = "signal"
    output_representation = "features"