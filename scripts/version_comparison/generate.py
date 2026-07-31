"""Generate TINTOlib images from a version-neutral comparison config."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import platform
import time
from pathlib import Path

import pandas as pd


METHODS = {
    "TINTO": ("TINTOlib.tinto", "TINTO"),
    "IGTD": ("TINTOlib.igtd", "IGTD"),
    "REFINED": ("TINTOlib.refined", "REFINED"),
    "BarGraph": ("TINTOlib.barGraph", "BarGraph"),
    "DistanceMatrix": ("TINTOlib.distanceMatrix", "DistanceMatrix"),
    "Combination": ("TINTOlib.combination", "Combination"),
    "SuperTML": ("TINTOlib.supertml", "SuperTML"),
    "FeatureWrap": ("TINTOlib.featureWrap", "FeatureWrap"),
    "BIE": ("TINTOlib.bie", "BIE"),
    "DeepInsight": ("TINTOlib.deepInsight", "DeepInsight"),
    "Fotomics": ("TINTOlib.fotomics", "Fotomics"),
    "Clusters": ("TINTOlib.clusters", "Clusters"),
}

CONTROL_KEYS = {"method", "auto_scale", "auto_size"}


def load_method(name: str):
    try:
        module_name, class_name = METHODS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown method {name!r}") from exc
    return getattr(importlib.import_module(module_name), class_name)


def prepare_parameters(method_class, raw_params, problem, feature_count):
    params = {k: v for k, v in raw_params.items() if k not in CONTROL_KEYS}
    signature = inspect.signature(method_class.__init__).parameters

    # v1.0.6.1 calls classification tasks "supervised"; current releases use
    # "classification". Regression and unsupervised names are unchanged.
    params["problem"] = (
        "supervised"
        if "normalize" in signature and problem == "classification"
        else problem
    )

    # Input splits are already preprocessed. Disable each version's internal
    # preprocessing so both versions receive exactly the same numbers.
    if "transformer" in signature:
        params["transformer"] = None
    elif "normalize" in signature:
        params["normalize"] = False

    if raw_params.get("auto_scale") and "scale" in signature:
        side = math.ceil(math.sqrt(feature_count))
        params["scale"] = [side, side]

    if raw_params.get("auto_size") and "size" in signature:
        bins = int(params.get("bins", 10))
        side = math.ceil(math.sqrt(math.ceil(feature_count * bins / 8)))
        params["size"] = [side, side]

    unsupported = sorted(set(params) - set(signature))
    if unsupported:
        raise TypeError(
            f"{method_class.__name__} does not support parameters: {unsupported}"
        )
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--problem",
        required=True,
        choices=("classification", "regression", "unsupervised"),
    )
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8-sig") as handle:
        config = json.load(handle)

    splits = {
        name: pd.read_csv(args.splits / f"{name}.csv")
        for name in ("train", "val", "test")
    }
    feature_count = splits["train"].shape[1]
    if args.problem != "unsupervised":
        feature_count -= 1

    root = args.output / args.label
    root.mkdir(parents=True, exist_ok=True)
    results = []

    for alias, raw_params in config["parameters"].items():
        method_name = raw_params.get("method", alias)
        method_class = load_method(method_name)
        params = prepare_parameters(
            method_class, raw_params, args.problem, feature_count
        )
        method_root = root / alias
        started = time.perf_counter()
        status = "passed"
        error = None
        try:
            transformer = method_class(**params)
            transformer.fit(splits["train"])
            for split_name, data in splits.items():
                destination = method_root / split_name
                destination.mkdir(parents=True, exist_ok=True)
                transformer.transform(data, str(destination))
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"

        result = {
            "alias": alias,
            "method": method_name,
            "status": status,
            "seconds": time.perf_counter() - started,
            "error": error,
        }
        results.append(result)
        print(f"{status.upper():6} {alias}: {error or 'ok'}", flush=True)

    summary = {
        "label": args.label,
        "python": platform.python_version(),
        "problem": args.problem,
        "feature_count": feature_count,
        "config": str(args.config),
        "results": results,
    }
    with (root / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if any(result["status"] == "failed" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
