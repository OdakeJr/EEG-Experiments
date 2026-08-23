# utils/storage.py

import json
import pickle
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

DATA_FORMAT = "csv"


# ============================================================
# Tabular data
# ============================================================

def _data_path(path):
    """
    Add the configured data extension.
    """
    path = Path(path)

    if path.suffix:
        return path

    return path.with_suffix(
        f".{DATA_FORMAT}"
    )


def save_data(data, path):
    """
    Save tabular data using the configured format.
    """
    path = _data_path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    if DATA_FORMAT == "csv":
        data.to_csv(
            path,
            index=False,
        )

    elif DATA_FORMAT == "pkl":
        data.to_pickle(path)

    else:
        raise ValueError(
            f"Unknown data format: {DATA_FORMAT}"
        )

    return path


def load_data(path):
    """
    Load tabular data.
    """
    path = _data_path(path)

    if DATA_FORMAT == "csv":
        return pd.read_csv(path)

    elif DATA_FORMAT == "pkl":
        return pd.read_pickle(path)

    raise ValueError(
        f"Unknown data format: {DATA_FORMAT}"
    )


# ============================================================
# JSON
# ============================================================

def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
        )


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# Pickle
# ============================================================

def save_pickle(data, path):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "wb",
    ) as file:
        pickle.dump(
            data,
            file,
        )


def load_pickle(path):
    with open(
        path,
        "rb",
    ) as file:
        return pickle.load(file)


# ============================================================
# Manifest
# ============================================================

def save_manifest(manifest, path):
    save_json(
        manifest,
        path,
    )


def load_manifest(path):
    return load_json(path)


# ============================================================
# General
# ============================================================

def exists(path):
    path = Path(path)

    if path.suffix:
        return path.exists()

    return _data_path(
        path
    ).exists()