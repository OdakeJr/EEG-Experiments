# plugins/data_manipulation/composition.py

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from plugins.pipe import (
    DatasetPipe,
    DatasetView,
    ExperimentRecord,
)
from plugins.eeg.eeg_dataframe import EEGDataFrame


class EEGConcatPipe(DatasetPipe):
    """
    Concatenate one or more compatible EEGDataFrame objects,
    persist the standardized tabular dataset, and create the
    initial DatasetView used by downstream experimental nodes.

    Expected input
    --------------
    One or more PipeResult objects whose values are
    EEGDataFrame objects.

    Parameters
    ----------
    output_path : str
        Path where the Parquet dataset will be stored.

    dataset_name : str, optional
        Logical name of the resulting persisted dataset.

    Output
    ------
    PipeResult
        value:
            Initial DatasetView.

        record:
            ExperimentRecord containing the current DatasetView,
            which is the data state required by downstream nodes.
    """

    def expand(
        self,
        input_nodes,
        params,
    ):
        if not input_nodes:
            raise ValueError(
                "EEGConcatPipe requires at least one input node."
            )

        return [
            {
                "inputs": [
                    node.id
                    for node in input_nodes
                ],
                "params": deepcopy(params),
            }
        ]

    def run(
        self,
        inputs,
        params,
    ):
        if not inputs:
            raise ValueError(
                "EEGConcatPipe received no inputs."
            )

        if "output_path" not in params:
            raise ValueError(
                "EEGConcatPipe requires "
                "'output_path' in params."
            )

        # ====================================================
        # Extract node-specific values
        # ====================================================

        data_inputs = [
            self.get_value(result)
            for result in inputs
        ]

        # ====================================================
        # Validate inputs
        # ====================================================

        for data in data_inputs:

            if not isinstance(
                data,
                EEGDataFrame,
            ):
                raise TypeError(
                    "EEGConcatPipe expects "
                    "EEGDataFrame values."
                )

        reference = data_inputs[0]

        if reference.feature_columns is None:
            raise ValueError(
                "EEGConcatPipe requires "
                "feature_columns to be defined."
            )

        # ====================================================
        # Compatibility checks
        # ====================================================

        for data in data_inputs[1:]:

            if (
                data.feature_columns
                != reference.feature_columns
            ):
                raise ValueError(
                    "Cannot concatenate EEG datasets "
                    "with different feature columns."
                )

            if (
                data.label_column
                != reference.label_column
            ):
                raise ValueError(
                    "Cannot concatenate EEG datasets "
                    "with different label columns."
                )

            if (
                data.channels
                != reference.channels
            ):
                raise ValueError(
                    "Cannot concatenate EEG datasets "
                    "with different channel configurations."
                )

            if not np.isclose(
                data.sampling_rate,
                reference.sampling_rate,
            ):
                raise ValueError(
                    "Cannot concatenate EEG datasets "
                    "with different sampling rates."
                )

        # ====================================================
        # Concatenate
        # ====================================================

        dataframe = pd.concat(
            [
                data.data
                for data in data_inputs
            ],
            axis=0,
            ignore_index=True,
        )

        dataset_names = [
            data.dataset_name
            for data in data_inputs
        ]

        # ====================================================
        # Standardized dataset schema
        # ====================================================

        feature_columns = list(
            reference.feature_columns
        )

        target_column = (
            reference.label_column
        )

        metadata_columns = [
            column
            for column in dataframe.columns
            if column not in feature_columns
            and column != target_column
        ]

        # ====================================================
        # Persist physical dataset
        # ====================================================

        output_path = Path(
            params["output_path"]
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset_name = params.get(
            "dataset_name",
            "combined",
        )

        dataset_ref = self.save_dataset(
            dataframe=dataframe,

            path=output_path,

            feature_columns=feature_columns,

            target_column=target_column,

            metadata_columns=metadata_columns,

            # Information describing this physical dataset
            # belongs to the DatasetReference, not to the
            # ExperimentRecord.
            metadata={
                "dataset_name": dataset_name,
                "source_datasets": (
                    dataset_names
                ),
                "n_datasets": len(
                    data_inputs
                ),
                "n_rows": len(
                    dataframe
                ),
                "n_features": len(
                    feature_columns
                ),
                "channels": list(
                    reference.channels
                ),
                "sampling_rate": (
                    reference.sampling_rate
                ),
            },
        )

        # ====================================================
        # Initial dataset view
        # ====================================================

        view = DatasetView(
            dataset=dataset_ref,

            # Source/target and train/test subsets will be
            # introduced later.
            indices={},

            # All features are initially active.
            feature_columns=(
                feature_columns.copy()
            ),

            # Domain interpretation is introduced later.
            domain_columns=[],

            metadata={
                "stage": "composition",
            },
        )

        # ====================================================
        # Experiment record
        # ====================================================

        record = ExperimentRecord()

        # This is the important state that downstream nodes
        # actually need to carry through the branch.
        record.set(
            "dataset_view",
            view,
        )

        # ====================================================
        # Output
        # ====================================================

        return self.make_result(
            value=view,
            record=record,
        )