import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from ml.domain_discrepancy.registry import get_domain_discrepancy
from models.domain_result import DomainResult
from models.domain_results_artifact import DomainResultsArtifact
from utils.status import is_done, make_manifest, make_signature
from utils.storage import exists, load_manifest, save_data, save_manifest


OUTPUT_ROOT = Path("outputs/domain_results")


def _iter_comparisons(data):
    pairs = [
        ("source_to_target_elementary", "source", data.source,
         "target_elementary_domain", data.target_elementary_domain),
        ("source_to_target_super", "source", data.source,
         "target_super_domain", data.target_super_domain),
        ("target_super_to_target_elementary", "target_super_domain", data.target_super_domain,
         "target_elementary_domain", data.target_elementary_domain),
    ]
    for pair in pairs:
        if pair[2] is not None and pair[4] is not None:
            yield pair


def _domain_identity(group):
    domains = sorted(str(domain) for domain in np.unique(group.elementary_domains))
    return ";".join(domains), len(domains)


def _standardize_pair(X_left, X_right):
    X_left, X_right = np.asarray(X_left, dtype=float), np.asarray(X_right, dtype=float)
    pooled = np.vstack([X_left, X_right])
    mean, std = pooled.mean(axis=0), pooled.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (X_left - mean) / std, (X_right - mean) / std


def _normalize_metrics(metrics):
    return [
        {"name": metric, "params": {}} if isinstance(metric, str)
        else {"name": metric["name"], "params": metric.get("params", {})}
        for metric in metrics
    ]


def _compute_result(name, params, X_left, X_right):
    return float(get_domain_discrepancy(name)(X_left, X_right, **params))


def _output_dir(scenario_artifacts, metric_configs, representations, standardize, signature):
    scenarios = "+".join(sorted(scenario_artifacts))
    metrics = "-".join(config["name"] for config in metric_configs)
    reps = "-".join(representations)
    scaling = "std" if standardize else "raw"
    return OUTPUT_ROOT / scenarios / f"{metrics}__{reps}__{scaling}__{signature[:12]}"


def _evaluate_pair(split, scenario, group_name, comparison,
                   left_group_name, left_group, right_group_name, right_group,
                   metric_configs, representations, standardize, min_samples_per_side):
    X_left, X_right = np.asarray(left_group.X), np.asarray(right_group.X)
    y_left, y_right = np.asarray(left_group.y), np.asarray(right_group.y)
    left_domains, n_left_domains = _domain_identity(left_group)
    right_domains, n_right_domains = _domain_identity(right_group)

    if standardize:
        X_left, X_right = _standardize_pair(X_left, X_right)

    base = dict(
        split_id=split.id, scenario=scenario, group=group_name, seed=split.seed,
        target_fraction=split.target_fraction, comparison=comparison,
        left_group=left_group_name, right_group=right_group_name,
        left_domains=left_domains, right_domains=right_domains,
        n_left_domains=n_left_domains, n_right_domains=n_right_domains,
    )

    rows = []
    for config in metric_configs:
        name, params = config["name"], config["params"]
        metric_signature = make_signature({
            "metric": name, "params": params, "standardize": standardize,
        })

        if "marginal" in representations and min(len(X_left), len(X_right)) >= min_samples_per_side:
            value = _compute_result(name, params, X_left, X_right)
            rows.append(DomainResult(
                **base, metric=name, metric_signature=metric_signature,
                representation="marginal", class_label=None, value=value,
                n_left_samples=len(X_left), n_right_samples=len(X_right),
            ).to_dict())

        if "class" not in representations:
            continue

        common_classes = sorted(set(np.unique(y_left)) & set(np.unique(y_right)), key=str)
        for class_label in common_classes:
            X_l, X_r = X_left[y_left == class_label], X_right[y_right == class_label]
            if min(len(X_l), len(X_r)) < min_samples_per_side:
                continue

            value = _compute_result(name, params, X_l, X_r)
            rows.append(DomainResult(
                **base, metric=name, metric_signature=metric_signature,
                representation="class", class_label=str(class_label), value=value,
                n_left_samples=len(X_l), n_right_samples=len(X_r),
            ).to_dict())

    return rows


def run_domain_evaluation(scenario_artifacts, params=None):
    params = params or {}
    metric_configs = _normalize_metrics(params.get("metrics", ["mmd", "energy"]))
    representations = params.get("representations", ["marginal", "class"])
    standardize = params.get("standardize", True)
    min_samples_per_side = params.get("min_samples_per_side", 2)

    inputs = []
    for scenario, artifacts in scenario_artifacts.items():
        for artifact in artifacts:
            view = artifact["view"]
            input_signature = load_manifest(view.manifest_path).get("signature")
            for split in artifact["splits"]:
                inputs.append({
                    "scenario": scenario,
                    "group": artifact["group"],
                    "split": split.to_dict(),
                    "input_signature": input_signature,
                })

    config = {
        "metrics": metric_configs,
        "representations": representations,
        "standardize": standardize,
        "min_samples_per_side": min_samples_per_side,
    }
    effective_params = {"inputs": inputs, "params": config}
    signature = make_signature(effective_params)

    output_dir = _output_dir(
        scenario_artifacts, metric_configs, representations, standardize, signature
    )
    results_path = output_dir / "domain_results.csv"
    manifest_path = output_dir / "manifest.json"

    if exists(results_path) and is_done(manifest_path, effective_params):
        manifest = load_manifest(manifest_path)
        return DomainResultsArtifact(
            path=str(results_path), manifest_path=str(manifest_path),
            signature=signature, n_rows=manifest["output"]["n_rows"],
        )

    start = time.time()
    save_manifest(make_manifest("running", effective_params), manifest_path)
    rows = []

    try:
        for scenario, artifacts in scenario_artifacts.items():
            for artifact in artifacts:
                group_name, view = artifact["group"], artifact["view"]

                for split in artifact["splits"]:
                    data = split.materialize(view)

                    for comparison, left_name, left, right_name, right in _iter_comparisons(data):
                        rows.extend(_evaluate_pair(
                            split, scenario, group_name, comparison,
                            left_name, left, right_name, right,
                            metric_configs, representations, standardize, min_samples_per_side,
                        ))

        dataframe = pd.DataFrame(rows, columns=[field.name for field in fields(DomainResult)])
        save_data(dataframe, results_path)

        manifest = make_manifest(
            "done", effective_params, execution_time=time.time() - start
        )
        manifest["output"] = {"path": str(results_path), "n_rows": len(dataframe)}
        save_manifest(manifest, manifest_path)

    except Exception as error:
        save_manifest(make_manifest(
            "failed", effective_params,
            execution_time=time.time() - start, error=str(error),
        ), manifest_path)
        raise

    return DomainResultsArtifact(
        path=str(results_path), manifest_path=str(manifest_path),
        signature=signature, n_rows=len(rows),
    )