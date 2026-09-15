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


def _safe_label(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, default=str))
    except TypeError:
        return deepcopy(value)


def build_representation_configs(feature_extraction_params, feature_selection_params,
                                 signal_transform_params):
    configs = [
        {
            "name": f"{fe['name']}__{fs['name']}",
            "route": "features",
            "feature_extraction": deepcopy(fe),
            "feature_selection": deepcopy(fs),
        }
        for fe in feature_extraction_params for fs in feature_selection_params
    ]

    configs += [
        {"name": st["name"], "route": "signal", "signal_transform": deepcopy(st)}
        for st in signal_transform_params
    ]
    return configs


def _prepare_extractor(config, view):
    method = config["method"]
    params = deepcopy(config.get("params", {}))

    if method == "handcrafted":
        params = {
            "features": params,
            "channel_names": getattr(view, "channel_names", None),
            "band_labels": getattr(view, "band_labels", None),
        }

    return {"method": method, "params": params}


def _feature_extraction_step(config, view):
    raw_extractors = deepcopy(config.get("extractors"))
    if raw_extractors is None:
        raw_extractors = [{"method": config["method"], "params": config.get("params", {})}]

    extractors = [_prepare_extractor(extractor, view) for extractor in raw_extractors]
    trace_params = raw_extractors if "extractors" in config else deepcopy(config.get("params", {}))

    return {
        "kind": "feature_extraction",
        "config_label": _safe_label(config["name"]),
        "extractors": extractors,
        "trace_method": "__".join(extractor["method"] for extractor in raw_extractors),
        "trace_params": trace_params,
    }


def _feature_selection_step(config):
    return {
        "kind": "feature_selection",
        "method": config["method"],
        "config_label": _safe_label(config["name"]),
        "params": deepcopy(config.get("params", {})),
        "trace_method": config["method"],
        "trace_params": deepcopy(config.get("params", {})),
    }


def _signal_transform_step(config):
    return {
        "kind": "signal_transform",
        "method": config["method"],
        "config_label": _safe_label(config["name"]),
        "params": deepcopy(config.get("params", {})),
        "trace_method": config["method"],
        "trace_params": deepcopy(config.get("params", {})),
    }


def _build_steps(config, view):
    if config["route"] == "features":
        return [
            _feature_extraction_step(config["feature_extraction"], view),
            _feature_selection_step(config["feature_selection"]),
        ]
    if config["route"] == "signal":
        return [_signal_transform_step(config["signal_transform"])]
    raise ValueError(f"Unknown representation route: {config['route']}")


def _validate_input(transformer, representation):
    expected = getattr(transformer, "input_representation", "features")
    if expected != "any" and expected != representation:
        raise ValueError(f"Transformer expects '{expected}', got '{representation}'.")


class RepresentationPipeline:
    def __init__(self, steps):
        self.step_configs = steps
        self.steps = []

        for step in steps:
            if step["kind"] == "feature_extraction":
                self.steps.append([
                    get_representation_transformer(x["method"], x["params"])
                    for x in step["extractors"]
                ])
            else:
                self.steps.append(get_representation_transformer(step["method"], step["params"]))

        self.input_representation = None
        self.output_representation = None
        self.output_shape_ = None

    def fit(self, X, y=None, domains=None, input_representation="signal"):
        representation = input_representation
        self.input_representation = representation

        for config, transformer in zip(self.step_configs, self.steps):
            if config["kind"] == "feature_extraction":
                outputs = []
                for extractor in transformer:
                    _validate_input(extractor, representation)
                    output = extractor.fit_transform(X, y, domains)
                    if extractor.output_representation != "features":
                        raise ValueError("Feature extractor must output 'features'.")
                    outputs.append(output)
                X = np.concatenate(outputs, axis=1)
                representation = "features"
            else:
                _validate_input(transformer, representation)
                X = transformer.fit_transform(X, y, domains)
                representation = transformer.output_representation

        self.output_representation = representation
        self.output_shape_ = tuple(X.shape[1:])
        return self

    def transform(self, X, domains=None):
        representation = self.input_representation

        for config, transformer in zip(self.step_configs, self.steps):
            if config["kind"] == "feature_extraction":
                outputs = []
                for extractor in transformer:
                    _validate_input(extractor, representation)
                    outputs.append(extractor.transform(X, domains))
                X = np.concatenate(outputs, axis=1)
                representation = "features"
            else:
                _validate_input(transformer, representation)
                X = transformer.transform(X, domains)
                representation = transformer.output_representation

        return X


def _get_fit_data(data, fit_partitions):
    X, y, domains = [], [], []

    for group in (data.source, data.target_super_domain, data.target_elementary_domain):
        if group is None:
            continue
        mask = np.isin(group.partitions, fit_partitions)
        if np.any(mask):
            X.append(group.X[mask])
            y.append(group.y[mask])
            domains.append(group.elementary_domains[mask])

    if not X:
        raise ValueError("No samples available to fit representation.")

    return np.concatenate(X), np.concatenate(y), np.concatenate(domains)


def _step_info(steps, kind):
    step = next((step for step in steps if step["kind"] == kind), None)
    if step is None:
        return None, None, None
    return step["trace_method"], _json_copy(step["trace_params"]), step["config_label"]


def _trace(steps):
    st = _step_info(steps, "signal_transform")
    fe = _step_info(steps, "feature_extraction")
    fs = _step_info(steps, "feature_selection")

    return {
        "signal_transform_method": st[0],
        "signal_transform_params": st[1],
        "signal_transform_config_label": st[2],
        "feature_extraction_method": fe[0],
        "feature_extraction_params": fe[1],
        "feature_extraction_config_label": fe[2],
        "feature_selection_method": fs[0],
        "feature_selection_params": fs[1],
        "feature_selection_config_label": fs[2],
    }


def _get_output_paths(scenario, group, split_id, name, signature):
    output_dir = OUTPUT_ROOT / scenario / group / split_id / f"{name}_{signature[:12]}"
    return output_dir / "transformer.pkl", output_dir / "manifest.json"


def _make_artifact(split, transformer_path, manifest_path, signature, config,
                   input_representation, output_representation, steps, view):
    return RepresentationArtifact(
        split_id=split.id,
        method=config["route"],
        transformer_path=str(transformer_path),
        manifest_path=str(manifest_path),
        signature=signature,
        input_representation=input_representation,
        output_representation=output_representation,
        representation_method=config["route"],
        representation_params=_json_copy(config),
        representation_config_label=_safe_label(config["name"]),
        **_trace(steps),
        preprocessing_signature=getattr(view, "preprocessing_signature", None),
        preprocessing_config_label=getattr(view, "preprocessing_config_label", None),
    )


def run_representation(split, view, representation_params, group="default"):
    steps = _build_steps(representation_params, view)
    input_representation = getattr(view, "representation", "signal")
    fit_partitions = representation_params.get("fit_partitions", ["train"])
    name = _safe_label(representation_params["name"])
    input_manifest = load_manifest(view.manifest_path)

    effective_params = {
        "split": split.to_dict(),
        "input_signature": input_manifest["signature"],
        "input_representation": input_representation,
        "representation": _json_copy(representation_params),
        "fit_partitions": fit_partitions,
    }

    signature = make_signature(effective_params)
    transformer_path, manifest_path = _get_output_paths(
        split.scenario, group, split.id, name, signature
    )

    if exists(transformer_path) and is_done(manifest_path, effective_params):
        output = load_manifest(manifest_path)["output"]
        return _make_artifact(
            split, transformer_path, manifest_path, signature, representation_params,
            output["input_representation"], output["output_representation"], steps, view
        )

    save_manifest(make_manifest("running", effective_params), manifest_path)
    start = time.perf_counter()

    try:
        data = split.materialize(view)
        X, y, domains = _get_fit_data(data, fit_partitions)

        transformer = RepresentationPipeline(steps)
        transformer.fit(X, y, domains, input_representation)
        save_pickle(transformer, transformer_path)

        manifest = make_manifest(
            "done", effective_params, execution_time=time.perf_counter() - start
        )
        manifest["output"] = {
            "transformer_path": str(transformer_path),
            "input_representation": input_representation,
            "output_representation": transformer.output_representation,
            "input_shape": [int(v) for v in X.shape[1:]],
            "output_shape": [int(v) for v in transformer.output_shape_],
            "n_fit_samples": int(X.shape[0]),
            "representation_method": representation_params["route"],
            "representation_params": _json_copy(representation_params),
            "representation_config_label": name,
            **_trace(steps),
            "preprocessing_signature": getattr(view, "preprocessing_signature", None),
            "preprocessing_config_label": getattr(view, "preprocessing_config_label", None),
        }
        save_manifest(manifest, manifest_path)

    except Exception as error:
        save_manifest(
            make_manifest("failed", effective_params,
                          execution_time=time.perf_counter() - start, error=str(error)),
            manifest_path,
        )
        raise

    return _make_artifact(
        split, transformer_path, manifest_path, signature, representation_params,
        input_representation, transformer.output_representation, steps, view
    )