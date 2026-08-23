# plugins/pipe.py

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import pandas as pd


# ============================================================
# Experiment record
# ============================================================

@dataclass
class ExperimentRecord:
    """
    Flexible lightweight record that travels through one
    experimental branch.

    Its internal structure is intentionally not imposed by the
    engine. Each pipeline or Pipe may add whatever information
    is useful for later analysis.

    Large artifacts should generally be stored externally and
    referenced from this record rather than embedded directly.
    """

    data: Dict[str, Any] = field(default_factory=dict)

    def copy(self):
        """Return an independent copy for a new branch."""
        return ExperimentRecord(
            data=deepcopy(self.data)
        )

    def get(self, key, default=None):
        return self.data.get(
            key,
            default,
        )

    def set(self, key, value):
        self.data[key] = value

    def update(self, values):
        self.data.update(values)


# ============================================================
# Standard Pipe output
# ============================================================

@dataclass
class PipeResult:
    """
    Standard output of a concrete Pipe execution.

    value
        Node-specific result.

    record
        ExperimentRecord propagated through this branch.
    """

    value: Any

    record: ExperimentRecord = field(
        default_factory=ExperimentRecord
    )


# ============================================================
# Dataset abstractions
# ============================================================

@dataclass
class DatasetReference:
    """
    Lightweight reference to a persisted tabular dataset.
    """

    path: str

    feature_columns: List[str]
    target_column: str
    metadata_columns: List[str]

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class DatasetView:
    """
    Lightweight description of how a persisted dataset should
    currently be interpreted.

    The physical dataset itself is not copied.
    """

    dataset: DatasetReference

    # Named sample subsets.
    #
    # Examples:
    # source
    # target
    # source_train
    # source_test
    # target_adapt
    # target_test
    indices: Dict[str, List[int]] = field(
        default_factory=dict
    )

    # Currently active features.
    feature_columns: Optional[List[str]] = None

    # Columns currently used to define domains.
    domain_columns: List[str] = field(
        default_factory=list
    )

    # Additional information specific to this view.
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# Base Pipe
# ============================================================

class Pipe(ABC):

    @abstractmethod
    def expand(self, input_nodes, params):
        """
        Define the concrete nodes that should be created.

        Returns a list of dictionaries containing:
        - inputs: IDs of the input nodes
        - params: resolved parameters for this concrete node
        """
        pass

    @abstractmethod
    def run(self, inputs, params) -> PipeResult:
        """
        Execute one concrete node.

        The standard return value is PipeResult:
        - value: node-specific output
        - record: branch ExperimentRecord
        """
        pass

    # ========================================================
    # PipeResult helpers
    # ========================================================

    def get_value(self, result):
        """
        Extract the node-specific value from an upstream result.

        Raw values are temporarily accepted to simplify
        migration of existing Pipes.
        """

        if isinstance(result, PipeResult):
            return result.value

        return result

    def get_record(self, result):
        """
        Extract an independent ExperimentRecord from one
        upstream result.

        An empty record is created for legacy/raw inputs.
        """

        if isinstance(result, PipeResult):
            return result.record.copy()

        return ExperimentRecord()

    def get_records(self, inputs):
        """
        Extract independent records from multiple inputs.

        No automatic merge is performed because merge semantics
        depend on the particular many-to-one Pipe.
        """

        return [
            self.get_record(result)
            for result in inputs
        ]

    def make_result(
        self,
        value,
        record=None,
    ):
        """
        Create the standardized Pipe output.
        """

        if record is None:
            record = ExperimentRecord()

        return PipeResult(
            value=value,
            record=record,
        )


# ============================================================
# Dataset-aware Pipe
# ============================================================

class DatasetPipe(Pipe):

    def load_dataset(
        self,
        obj: Union[
            DatasetReference,
            DatasetView,
        ],
    ):
        """
        Load the physical dataset.

        This can later delegate to a DatasetStore/cache without
        changing the downstream Pipe implementations.
        """

        if isinstance(obj, DatasetView):
            ref = obj.dataset

        elif isinstance(obj, DatasetReference):
            ref = obj

        else:
            raise TypeError(
                "Expected DatasetReference or DatasetView."
            )

        return pd.read_parquet(
            ref.path
        )

    def save_dataset(
        self,
        dataframe,
        path,
        feature_columns,
        target_column,
        metadata_columns,
        metadata=None,
    ):
        """
        Persist a tabular dataset and return its lightweight
        reference.
        """

        dataframe.to_parquet(
            path,
            index=False,
        )

        return DatasetReference(
            path=str(path),

            feature_columns=list(
                feature_columns
            ),

            target_column=target_column,

            metadata_columns=list(
                metadata_columns
            ),

            metadata=metadata or {},
        )

    def get_rows(
        self,
        dataframe,
        view,
        subset,
    ):
        """
        Return one named subset using positional row indices.
        """

        if subset not in view.indices:
            raise KeyError(
                f"DatasetView does not contain "
                f"subset '{subset}'."
            )

        return dataframe.iloc[
            view.indices[subset]
        ]

    def get_X(
        self,
        dataframe,
        view,
        subset=None,
    ):
        """
        Return the currently active feature matrix.
        """

        feature_columns = (
            view.feature_columns
            if view.feature_columns is not None
            else view.dataset.feature_columns
        )

        rows = (
            dataframe
            if subset is None
            else self.get_rows(
                dataframe,
                view,
                subset,
            )
        )

        return rows[
            feature_columns
        ]

    def get_y(
        self,
        dataframe,
        view,
        subset=None,
    ):
        """
        Return target values.
        """

        rows = (
            dataframe
            if subset is None
            else self.get_rows(
                dataframe,
                view,
                subset,
            )
        )

        return rows[
            view.dataset.target_column
        ]

    def get_metadata(
        self,
        dataframe,
        view,
        subset=None,
    ):
        """
        Return dataset metadata columns.
        """

        rows = (
            dataframe
            if subset is None
            else self.get_rows(
                dataframe,
                view,
                subset,
            )
        )

        return rows[
            view.dataset.metadata_columns
        ]