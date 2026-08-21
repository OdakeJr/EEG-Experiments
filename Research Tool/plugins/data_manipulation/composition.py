# plugins/data_manipulation/composition.py

from copy import deepcopy

import numpy as np
import pandas as pd

from plugins.pipe import Pipe
from plugins.eeg.eeg_dataframe import EEGDataFrame


class EEGConcatPipe(Pipe):
    """
    Concatenate multiple compatible EEGDataFrame outputs into
    one EEGDataFrame.

    Expected input
    --------------
    Multiple EEGDataFrame objects.

    Output
    ------
    One EEGDataFrame containing all rows from the inputs.
    """

    def expand(self, input_nodes, params):
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

    def run(self, inputs, params):
        if not inputs:
            raise ValueError(
                "EEGConcatPipe received no inputs."
            )

        for data in inputs:
            if not isinstance(data, EEGDataFrame):
                raise TypeError(
                    "EEGConcatPipe expects EEGDataFrame inputs."
                )

        reference = inputs[0]

        # ====================================================
        # Compatibility checks
        # ====================================================

        for data in inputs[1:]:

            if data.feature_columns != reference.feature_columns:
                raise ValueError(
                    "Cannot concatenate EEG datasets with "
                    "different feature columns."
                )

            if data.label_column != reference.label_column:
                raise ValueError(
                    "Cannot concatenate EEG datasets with "
                    "different label columns."
                )

            if data.channels != reference.channels:
                raise ValueError(
                    "Cannot concatenate EEG datasets with "
                    "different channel configurations."
                )

            if not np.isclose(
                data.sampling_rate,
                reference.sampling_rate,
            ):
                raise ValueError(
                    "Cannot concatenate EEG datasets with "
                    "different sampling rates."
                )

        # ====================================================
        # Concatenate
        # ====================================================

        dataframe = pd.concat(
            [
                data.data
                for data in inputs
            ],
            axis=0,
            ignore_index=True,
        )

        dataset_names = [
            data.dataset_name
            for data in inputs
        ]

        # ====================================================
        # Output
        # ====================================================

        return EEGDataFrame(
            data=dataframe,

            dataset_name=params.get(
                "dataset_name",
                "combined",
            ),

            channels=reference.channels.copy(),

            sampling_rate=reference.sampling_rate,

            feature_columns=(
                reference.feature_columns.copy()
                if reference.feature_columns is not None
                else None
            ),

            label_column=reference.label_column,

            metadata={
                "source_datasets": dataset_names,
                "n_datasets": len(inputs),
                "n_rows": len(dataframe),
            },
        )