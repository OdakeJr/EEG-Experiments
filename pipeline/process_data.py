
# pipeline/process_data.py

import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from eeg.dataset_handlers.bci2a import prepare_bci2a
from eeg.dataset_handlers.eegmmidb import prepare_eegmmidb
from eeg.dataset_handlers.weibo import prepare_weibo
from eeg.dataset_handlers.zhou import prepare_zhou
from models.dataset_view import DatasetView
from utils.storage import exists, load_manifest, save_data, save_manifest
from utils.status import is_done, make_manifest, make_signature


OUTPUT_ROOT = Path("outputs/preprocessing")


def _prepare_pacs(params):
    from vision.dataset_handlers.pacs import prepare_pacs
    return prepare_pacs(params)


DATASET_PREPARERS = {
    "bci2a": prepare_bci2a,
    "eegmmidb": prepare_eegmmidb,
    "weibo": prepare_weibo,
    "zhou": prepare_zhou,
    "pacs": _prepare_pacs,
}

SUPPORTED_REPRESENTATIONS = {"features", "signal", "image"}


# ============================================================
# Helpers
# ============================================================

def _get_representation(params):
    default = "image" if params["dataset"] == "pacs" else "features"
    representation = params.get("representation", default)

    if representation not in SUPPORTED_REPRESENTATIONS:
        raise ValueError(f"Unknown representation: {representation}")
    return representation


def _add_preprocessing_trace(info, params, representation):
    info = deepcopy(info)
    signature = make_signature(params)
    info.update({
        "representation": representation,
        "preprocessing_signature": signature,
        "preprocessing_config_label": params.get(
            "name", f"{params['dataset']}_{signature[:8]}"
        ),
        "preprocessing_params": deepcopy(params),
    })
    return info


def _get_output_paths(params):
    name = params.get("name", make_signature(params)[:8])
    root = OUTPUT_ROOT / params["dataset"] / name
    return root / "data", root / "manifest.json"


# ============================================================
# Prepared data normalization
# ============================================================

def _prepare_feature_data(data, info):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Features must be a DataFrame.")

    columns = list(info.get("feature_columns", []))
    info["feature_columns"] = columns
    info["input_shape"] = (len(columns),)

    return {col: data[col].to_numpy() for col in data.columns}, info


def _prepare_signal_data(data, info):
    if not isinstance(data, dict) or "X" not in data:
        raise TypeError("Signal data must be a dictionary containing X.")

    X = np.asarray(data["X"])
    if X.ndim < 3:
        raise ValueError("Signal X must have at least 3 dimensions.")

    for col in info.get("metadata_columns", []):
        if col not in data or len(data[col]) != len(X):
            raise ValueError(f"Invalid signal metadata: {col}")

    info["feature_columns"] = None
    info["input_shape"] = tuple(X.shape[1:])
    return {key: np.asarray(value) for key, value in data.items()}, info


def _prepare_image_data(data, info):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Image data must be a DataFrame.")

    required = {
        "image_path", info["label_column"],
        *info["domain_columns"], *info["metadata_columns"],
    }
    missing = required - set(data.columns)
    if missing or data.empty or data["image_path"].isna().any():
        raise ValueError(f"Invalid image data. Missing columns: {sorted(missing)}")

    info["feature_columns"] = None
    info["input_shape"] = None
    return data, info


def _prepare_output_data(data, info, representation):
    info = deepcopy(info)
    preparers = {
        "features": _prepare_feature_data,
        "signal": _prepare_signal_data,
        "image": _prepare_image_data,
    }
    return preparers[representation](data, info)


# ============================================================
# DatasetView
# ============================================================

def _make_dataset_view(data_path, manifest_path, info):
    shape = info.get("input_shape")
    bands = info.get("band_labels")

    return DatasetView(
        path=str(data_path),
        representation=info.get("representation", "features"),
        feature_columns=info.get("feature_columns"),
        label_column=info["label_column"],
        domain_columns=info["domain_columns"],
        metadata_columns=info["metadata_columns"],
        manifest_path=str(manifest_path),
        input_shape=tuple(shape) if shape is not None else None,
        channel_names=info.get("channel_names"),
        band_labels=[tuple(b) for b in bands] if bands is not None else None,
        preprocessing_signature=info.get("preprocessing_signature"),
        preprocessing_config_label=info.get("preprocessing_config_label"),
        preprocessing_params=info.get("preprocessing_params"),
    )


# ============================================================
# Main preprocessing
# ============================================================

def run_preprocessing(params):
    representation = _get_representation(params)
    dataset_name = params["dataset"]

    if dataset_name not in DATASET_PREPARERS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data_path, manifest_path = _get_output_paths(params)

    if exists(data_path) and is_done(manifest_path, params):
        manifest = load_manifest(manifest_path)
        info = _add_preprocessing_trace(
            manifest["output"], params, representation
        )
        return _make_dataset_view(data_path, manifest_path, info)

    save_manifest(make_manifest(status="running", params=params), manifest_path)
    start = time.time()

    try:
        data, info = DATASET_PREPARERS[dataset_name](params)
        data, info = _prepare_output_data(data, info, representation)
        info = _add_preprocessing_trace(info, params, representation)

        save_data(data, data_path)

        manifest = make_manifest(
            status="done", params=params,
            execution_time=time.time() - start,
        )
        manifest["output"] = info
        save_manifest(manifest, manifest_path)

    except Exception as error:
        save_manifest(make_manifest(
            status="failed", params=params,
            execution_time=time.time() - start,
            error=str(error),
        ), manifest_path)
        raise

    return _make_dataset_view(data_path, manifest_path, info)
