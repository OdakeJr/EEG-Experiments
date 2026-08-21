# plugins/machine_learning/feature_selection/filter.py

from abc import abstractmethod

import numpy as np
from sklearn.feature_selection import mutual_info_classif, f_classif

from plugins.pipe import Pipe
from plugins.machine_learning.feature_selection.result import (
    FeatureSelectionResult
)


# ============================================================
# Base filter feature-selection pipe
# ============================================================

class FilterFeatureSelectionPipe(Pipe):
    """Base class for filter-based feature-selection methods."""

    def expand(self, input_nodes, params):
        """Create one feature-selection node for each input node."""

        return [
            {
                "inputs": [node.id],
                "params": params.copy()
            }
            for node in input_nodes
        ]

    def run(self, inputs, params):
        """Apply feature selection to one input dataset."""

        data = inputs[0]

        X_source = data.X_source
        y_source = data.y_source

        X_target = data.X_target
        y_target = data.y_target

        source_domains = data.source_domains
        target_domains = data.target_domains

        selected_features, artifacts = self._select_features(
            X_source,
            y_source,
            params
        )

        X_source_selected = X_source[:, selected_features]

        X_target_selected = None
        if X_target is not None:
            X_target_selected = X_target[:, selected_features]

        return FeatureSelectionResult(
            X_source=X_source_selected,
            y_source=y_source,
            X_target=X_target_selected,
            y_target=y_target,
            source_domains=source_domains,
            target_domains=target_domains,
            selected_features=selected_features,
            selector=None,
            artifacts=artifacts
        )

    @abstractmethod
    def _select_features(self, X_source, y_source, params):
        """Return selected feature indices and method-specific artifacts."""
        pass


# ============================================================
# 1. Mutual Information
# ============================================================

class MutualInformationPipe(FilterFeatureSelectionPipe):
    """Select features using mutual information with the target."""

    def _select_features(self, X_source, y_source, params):

        n_features = params.get("n_features", 128)
        n_features = min(n_features, X_source.shape[1])

        scores = mutual_info_classif(
            X_source,
            y_source,
            discrete_features=False
        )

        selected_features = np.argsort(scores)[::-1][:n_features]

        return selected_features, {
            "method": "mutual_information",
            "scores": scores
        }


# ============================================================
# 2. ANOVA F-test
# ============================================================

class ANOVAPipe(FilterFeatureSelectionPipe):
    """Select features using the ANOVA F-test."""

    def _select_features(self, X_source, y_source, params):

        n_features = params.get("n_features", 128)
        n_features = min(n_features, X_source.shape[1])

        scores, p_values = f_classif(
            X_source,
            y_source
        )

        scores = np.nan_to_num(
            scores,
            nan=-np.inf,
            posinf=-np.inf,
            neginf=-np.inf
        )

        selected_features = np.argsort(scores)[::-1][:n_features]

        return selected_features, {
            "method": "anova",
            "scores": scores,
            "p_values": p_values
        }


# ============================================================
# 3. Variance
# ============================================================

class VariancePipe(FilterFeatureSelectionPipe):
    """Select features with the highest source-data variance."""

    def _select_features(self, X_source, y_source, params):

        n_features = params.get("n_features", 128)
        n_features = min(n_features, X_source.shape[1])

        scores = np.var(
            X_source,
            axis=0
        )

        selected_features = np.argsort(scores)[::-1][:n_features]

        return selected_features, {
            "method": "variance",
            "scores": scores
        }