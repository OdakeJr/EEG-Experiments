# pipeline/representation.py

import json
import re
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from ml.representation.registry import get_representation_transformer
from models.representation_artifact import RepresentationArtifact
from utils.storage import exists, load_manifest, save_manifest, save_pickle
from utils.status import is_done, make_manifest, make_signature


OUTPUT_ROOT = Path("outputs/representation")

STEP_KINDS = ("signal_transform", "feature_extraction", "feature_selection")


def _safe_label(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def _params_label(params):
    if not params:
        return ""

    parts = []
    for key in sorted(params):
        value = params[key]
        if isinstance(value, float):
            value = f"{value:g}"
        parts.append(f"{key}_{value}")

    return "_".join(parts)


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, default=str))
    except TypeError:
        return deepcopy(value)


def _infer_step_kind(method):
    if method in {"identity_signal", "standardize_signal"}:
        return "signal_transform"
    if method in {"handcrafted", "csp", "rcsp", "riemann", "riemannian", "combined"}:
        return "feature_extraction"
    if method in {"variance", "anova", "mutual_information", "random"}:
        return "feature_selection"
    return None


def _step_label(config):
    method = config["method"]
    params = config.get("params", {})

    if "config_label" in config:
        return _safe_label(config["config_label"])
    if "name" in config and config["name"] != method:
        return _safe_label(config["name"])

    params_text = _params_label(params)
    return _safe_label(f"{method}_{params_text}" if params_text else method)


def _normalize_step(config, kind=None):
    if config is None or config == "none":
        return None

    if isinstance(config, str):
        config = {"method": config}

    config = deepcopy(config)
    config.setdefault("params", {})
    config.setdefault("kind", kind or _infer_step_kind(config["method"]))
    config["config_label"] = _step_label(config)
    return config


def _build_step_configs(params, view):
    if "steps" in params:
        steps = [_normalize_step(step) for step in params["steps"]]
    elif "method" in params:
        steps = [_normalize_step(params)]
    else:
        steps = [_normalize_step(params.get(kind), kind) for kind in STEP_KINDS]

    steps = [step for step in steps if step is not None]

    if not steps:
        raise ValueError("No representation steps were defined.")

    for step in steps:
        method = step["method"]
        step_params = step["params"]

        if method == "handcrafted":
            step_params.setdefault("channel_names", getattr(view, "channel_names", None))
            step_params.setdefault("band_labels", getattr(view, "band_labels", None))

    return steps


def _representation_label(params, steps):
    if "representation_config_label" in params:
        return _safe_label(params["representation_config_label"])
    if "config_label" in params and "method" not in params:
        return _safe_label(params["config_label"])
    if "name" in params:
        return _safe_label(params["name"])

    return _safe_label("__".join(step["config_label"] for step in steps))


def _get_fit_data(data, fit_partitions):
    X_parts, y_parts, domain_parts = [], [], []

    for group in (data.source, data.target_super_domain, data.target_elementary_domain):
        if group is None:
            continue

        mask = np.isin(group.partitions, fit_partitions)
        if not np.any(mask):
            continue

        X_parts.append(group.X[mask])
        y_parts.append(group.y[mask])
        domain_parts.append(group.elementary_domains[mask])

    if not X_parts:
        raise ValueError("No samples available to fit representation.")

    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(domain_parts, axis=0),
    )


def _validate_input(transformer, representation):
    expected = getattr(transformer, "input_representation", "features")

    if expected != "any" and expected != representation:
        raise ValueError(f"Transformer expects '{expected}', got '{representation}'.")


class RepresentationPipeline:
    def __init__(self, step_configs):
        self.step_configs = step_configs
        self.steps = [
            get_representation_transformer(step["method"], step.get("params", {}))
            for step in step_configs
        ]
        self.input_representation = None
        self.output_representation = None
        self.output_shape_ = None

    def fit(self, X, y=None, domains=None, input_representation="signal"):
        representation = input_representation
        self.input_representation = representation

        for transformer in self.steps:
            _validate_input(transformer, representation)
            X = transformer.fit_transform(X, y, domains)
            representation = transformer.output_representation

        self.output_representation = representation
        self.output_shape_ = tuple(X.shape[1:])
        return self

    def transform(self, X, domains=None):
        representation = self.input_representation

        for transformer in self.steps:
            _validate_input(transformer, representation)
            X = transformer.transform(X, domains)
            representation = transformer.output_representation

        return X


def _get_output_paths(scenario, group, split_id, name, signature):
    output_dir = OUTPUT_ROOT / scenario / group / split_id / f"{name}_{signature[:12]}"
    return output_dir / "transformer.pkl", output_dir / "manifest.json"


def _step_info(steps, kind):
    matches = [step for step in steps if step.get("kind") == kind]

    if not matches:
        return None, None, None

    method = "__".join(step["method"] for step in matches)
    label = "__".join(step["config_label"] for step in matches)
    params = {step["method"]: step.get("params", {}) for step in matches}

    return method, _json_copy(params), _safe_label(label)


def _make_artifact(
    split,
    transformer_path,
    manifest_path,
    signature,
    representation_method,
    representation_params,
    representation_label,
    input_representation,
    output_representation,
    steps,
    view,
):
    st_method, st_params, st_label = _step_info(steps, "signal_transform")
    fe_method, fe_params, fe_label = _step_info(steps, "feature_extraction")
    fs_method, fs_params, fs_label = _step_info(steps, "feature_selection")

    return RepresentationArtifact(
        split_id=split.id,
        method=representation_method,
        transformer_path=str(transformer_path),
        manifest_path=str(manifest_path),
        signature=signature,
        input_representation=input_representation,
        output_representation=output_representation,
        representation_method=representation_method,
        representation_params=_json_copy(representation_params),
        representation_config_label=representation_label,
        signal_transform_method=st_method,
        signal_transform_params=st_params,
        signal_transform_config_label=st_label,
        feature_extraction_method=fe_method,
        feature_extraction_params=fe_params,
        feature_extraction_config_label=fe_label,
        feature_selection_method=fs_method,
        feature_selection_params=fs_params,
        feature_selection_config_label=fs_label,
        preprocessing_signature=getattr(view, "preprocessing_signature", None),
        preprocessing_config_label=getattr(view, "preprocessing_config_label", None),
    )


def run_representation(split, view, representation_params, group="default"):
    steps = _build_step_configs(representation_params, view)
    input_representation = getattr(view, "representation", "features")
    representation_label = _representation_label(representation_params, steps)
    representation_method = "pipeline" if len(steps) > 1 else steps[0]["method"]
    fit_partitions = representation_params.get("fit_partitions", ["train"])
    name = representation_params.get("name", representation_label)

    input_manifest = load_manifest(view.manifest_path)

    effective_params = {
        "split": split.to_dict(),
        "input_signature": input_manifest["signature"],
        "input_representation": input_representation,
        "representation_method": representation_method,
        "representation_config_label": representation_label,
        "steps": _json_copy(steps),
        "fit_partitions": fit_partitions,
    }

    signature = make_signature(effective_params)
    transformer_path, manifest_path = _get_output_paths(
        split.scenario, group, split.id, name, signature
    )

    if exists(transformer_path) and is_done(manifest_path, effective_params):
        manifest = load_manifest(manifest_path)
        output = manifest["output"]

        return _make_artifact(
            split,
            transformer_path,
            manifest_path,
            signature,
            representation_method,
            representation_params,
            representation_label,
            output["input_representation"],
            output["output_representation"],
            steps,
            view,
        )

    save_manifest(make_manifest(status="running", params=effective_params), manifest_path)
    start = time.perf_counter()

    try:
        data = split.materialize(view)
        X, y, domains = _get_fit_data(data, fit_partitions)

        transformer = RepresentationPipeline(steps)
        transformer.fit(X, y, domains, input_representation=input_representation)
        save_pickle(transformer, transformer_path)

        manifest = make_manifest(
            status="done",
            params=effective_params,
            execution_time=time.perf_counter() - start,
        )

        manifest["output"] = {
            "transformer_path": str(transformer_path),
            "input_representation": input_representation,
            "output_representation": transformer.output_representation,
            "input_shape": [int(v) for v in X.shape[1:]],
            "output_shape": [int(v) for v in transformer.output_shape_],
            "n_fit_samples": int(X.shape[0]),
            "representation_method": representation_method,
            "representation_params": _json_copy(representation_params),
            "representation_config_label": representation_label,
            "steps": _json_copy(steps),
            "preprocessing_signature": getattr(view, "preprocessing_signature", None),
            "preprocessing_config_label": getattr(view, "preprocessing_config_label", None),
        }

        save_manifest(manifest, manifest_path)

    except Exception as error:
        save_manifest(
            make_manifest(
                status="failed",
                params=effective_params,
                execution_time=time.perf_counter() - start,
                error=str(error),
            ),
            manifest_path,
        )
        raise

    return _make_artifact(
        split,
        transformer_path,
        manifest_path,
        signature,
        representation_method,
        representation_params,
        representation_label,
        input_representation,
        transformer.output_representation,
        steps,
        view,
    )