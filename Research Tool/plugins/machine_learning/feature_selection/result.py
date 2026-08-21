# plugins/machine_learning/feature_selection/result.py

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FeatureSelectionResult:
    """Standard representation of feature-selection output."""

    # --------------------------------------------------------
    # Transformed data
    # --------------------------------------------------------

    X_source: Any
    y_source: Any

    X_target: Optional[Any] = None
    y_target: Optional[Any] = None

    # --------------------------------------------------------
    # Domain information
    # --------------------------------------------------------

    source_domains: Optional[Any] = None
    target_domains: Optional[Any] = None

    # --------------------------------------------------------
    # Feature-selection information
    # --------------------------------------------------------

    selected_features: Optional[Any] = None
    selector: Optional[Any] = None

    # Method-specific information such as scores or rankings.
    artifacts: dict = field(default_factory=dict)