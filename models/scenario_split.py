
# models/scenario_split.py

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from models.scenario_data import DataGroup, ScenarioData
from utils.storage import load_data


@dataclass
class ScenarioSplit:
    id: str
    scenario: str
    source_elementary_domains: list[str]
    target_super_domain_elementary_domains: list[str]
    target_elementary_domains: list[str]
    target_fraction: float
    seed: int

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def materialize(self, dataset_view):
        X, y, metadata = _load_view_data(dataset_view)
        domains = _build_elementary_domains(metadata, self.scenario)
        super_domains = _build_super_domains(metadata, self.scenario)

        source = _build_group(
            X, y, domains, super_domains,
            self.source_elementary_domains, "train"
        )
        target_super = _build_group(
            X, y, domains, super_domains,
            self.target_super_domain_elementary_domains, "calibration"
        )
        target = _build_target_group(
            X, y, domains, super_domains,
            self.target_elementary_domains,
            self.target_fraction, self.seed, self.scenario
        )

        return ScenarioData(
            source=source,
            target_super_domain=target_super,
            target_elementary_domain=target,
        )


# ============================================================
# Data loading
# ============================================================

def _load_view_data(view):
    data = load_data(view.path)
    representation = getattr(view, "representation", "features")

    if representation == "features":
        return (
            data[view.feature_columns].to_numpy(),
            data[view.label_column].to_numpy(),
            data,
        )

    if representation == "image":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Image metadata must be a DataFrame.")

        path_column = getattr(view, "image_path_column", None) or "image_path"
        return (
            data[path_column].to_numpy(dtype=str),
            data[view.label_column].to_numpy(),
            data,
        )

    if representation == "signal":
        X = np.asarray(data["X"])
        y = np.asarray(data[view.label_column])
        columns = list(dict.fromkeys(
            view.domain_columns + view.metadata_columns + [view.label_column]
        ))
        metadata = pd.DataFrame({col: data[col] for col in columns})

        if len(X) != len(metadata):
            raise ValueError("Signal and metadata sample counts do not match.")

        return X, y, metadata

    raise ValueError(f"Unknown representation: {representation}")


# ============================================================
# Domain construction
# ============================================================

def _build_elementary_domains(metadata, scenario):
    if scenario == "cross_session":
        columns = ["dataset", "subject", "session"]
    elif scenario == "cross_domain":
        columns = ["dataset", "domain"]
    else:
        columns = ["dataset", "subject"]

    return metadata[columns].astype(str).agg("|".join, axis=1).to_numpy()


def _build_super_domains(metadata, scenario):
    if scenario != "cross_dataset":
        return None
    return metadata["dataset"].astype(str).to_numpy()


# ============================================================
# DataGroup helpers
# ============================================================

def _build_group(X, y, domains, super_domains, selected_domains, partition):
    if not selected_domains:
        return None

    mask = np.isin(domains, selected_domains)
    if not mask.any():
        return None

    return DataGroup(
        X=X[mask],
        y=y[mask],
        elementary_domains=domains[mask],
        partitions=np.full(mask.sum(), partition, dtype=object),
        super_domains=None if super_domains is None else super_domains[mask],
    )


def _build_target_group(X, y, domains, super_domains, selected_domains,
                        fraction, seed, scenario):
    if not selected_domains:
        return None
    if not 0 <= fraction <= 1:
        raise ValueError("'target_fraction' must be between 0 and 1.")

    indices = np.flatnonzero(np.isin(domains, selected_domains))
    if not len(indices):
        return None

    if fraction == 0:
        selected = np.array([], dtype=int)
    elif fraction == 1:
        selected = indices
    else:
        try:
            selected, _ = train_test_split(
                indices, train_size=fraction,
                random_state=seed, stratify=y[indices]
            )
        except ValueError:
            selected, _ = train_test_split(
                indices, train_size=fraction, random_state=seed
            )

    partition = "train" if scenario == "intra_subject" else "calibration"
    partitions = np.where(np.isin(indices, selected), partition, "test")

    return DataGroup(
        X=X[indices],
        y=y[indices],
        elementary_domains=domains[indices],
        partitions=partitions,
        super_domains=None if super_domains is None else super_domains[indices],
    )
