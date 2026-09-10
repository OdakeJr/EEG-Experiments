# utils/storage.py

import io
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

TABULAR_DATA_FORMAT = "csv"
ARRAY_DATA_FORMAT = "npz"


# ============================================================
# Paths
# ============================================================

def _tabular_data_path(path):
    path = Path(path)
    return path if path.suffix else path.with_suffix(f".{TABULAR_DATA_FORMAT}")


def _array_data_path(path):
    path = Path(path)
    return path if path.suffix else path.with_suffix(f".{ARRAY_DATA_FORMAT}")


def _resolve_data_path(path):
    path = Path(path)

    if path.suffix:
        if path.exists():
            return path
        raise FileNotFoundError(f"Data file does not exist: {path}")

    candidates = [_tabular_data_path(path), _array_data_path(path)]
    existing = [candidate for candidate in candidates if candidate.exists()]

    if not existing:
        raise FileNotFoundError(f"No stored data found for: {path}")

    if len(existing) > 1:
        raise RuntimeError(
            f"Multiple stored representations found for '{path}': {existing}"
        )

    return existing[0]


# ============================================================
# Data type detection
# ============================================================

def _contains_multidimensional_array(data):
    if not isinstance(data, dict):
        return False
    return any(np.asarray(value).ndim > 1 for value in data.values())


# ============================================================
# Tabular data
# ============================================================

def _save_tabular_data(data, path):
    path = _tabular_data_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    if TABULAR_DATA_FORMAT == "csv":
        data.to_csv(path, index=False)
    elif TABULAR_DATA_FORMAT == "pkl":
        data.to_pickle(path)
    else:
        raise ValueError(f"Unknown tabular format: {TABULAR_DATA_FORMAT}")

    return path


def _load_tabular_data(path, keys=None):
    suffix = Path(path).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, usecols=keys)

    if suffix in {".pkl", ".pickle"}:
        data = pd.read_pickle(path)
        return data if keys is None else data[keys]

    raise ValueError(f"Unsupported tabular file: {path}")


# ============================================================
# Multidimensional data
# ============================================================

def _save_array_data(data, path):
    if not isinstance(data, dict):
        raise TypeError("Array data must be provided as a dictionary.")

    path = _array_data_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(path, **{key: np.asarray(value) for key, value in data.items()})
    return path


def _load_array_data(path, keys=None):
    with np.load(path, allow_pickle=False) as stored:
        selected_keys = stored.files if keys is None else keys
        missing = [key for key in selected_keys if key not in stored.files]

        if missing:
            raise KeyError(f"Arrays not found in '{path}': {missing}")

        return {key: stored[key] for key in selected_keys}


# ============================================================
# Public data interface
# ============================================================

def save_data(data, path):
    """Save either tabular or multidimensional data."""
    if _contains_multidimensional_array(data):
        return _save_array_data(data, path)
    return _save_tabular_data(data, path)


def load_data(path, keys=None):
    """Load stored tabular or multidimensional data."""
    path = _resolve_data_path(path)
    suffix = path.suffix.lower()

    if suffix in {".csv", ".pkl", ".pickle"}:
        return _load_tabular_data(path, keys=keys)

    if suffix == ".npz":
        return _load_array_data(path, keys=keys)

    raise ValueError(f"Unsupported data format: {suffix}")


# ============================================================
# JSON
# ============================================================

def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# Pickle
# ============================================================

class _DeviceUnpickler(pickle.Unpickler):
    def __init__(self, file, map_location):
        super().__init__(file)
        self.map_location = map_location

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch
            return lambda data: torch.load(
                io.BytesIO(data),
                map_location=self.map_location,
            )
        return super().find_class(module, name)


def save_pickle(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as file:
        pickle.dump(data, file)


def load_pickle(path, map_location=None):
    with open(path, "rb") as file:
        if map_location is None:
            return pickle.load(file)
        return _DeviceUnpickler(file, map_location).load()


# ============================================================
# Manifest
# ============================================================

def save_manifest(manifest, path):
    save_json(manifest, path)


def load_manifest(path):
    return load_json(path)


# ============================================================
# General
# ============================================================

def exists(path):
    path = Path(path)

    if path.suffix:
        return path.exists()

    return _tabular_data_path(path).exists() or _array_data_path(path).exists()