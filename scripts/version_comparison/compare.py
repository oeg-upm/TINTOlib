"""Compare image outputs produced by two TINTOlib versions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def files_below(root: Path):
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def compare_pair(before: Path, after: Path):
    with Image.open(before) as left_image, Image.open(after) as right_image:
        left = np.asarray(left_image.convert("RGB"), dtype=np.float64)
        right = np.asarray(right_image.convert("RGB"), dtype=np.float64)
    if left.shape != right.shape:
        return {"same_shape": False, "before_shape": left.shape, "after_shape": right.shape}
    delta = np.abs(left - right)
    return {
        "same_shape": True,
        "exact": bool(np.array_equal(left, right)),
        "mae": float(delta.mean()),
        "max_abs_error": float(delta.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    before_files = files_below(args.before)
    after_files = files_below(args.after)
    common = sorted(before_files.keys() & after_files.keys())
    comparisons = {
        relative: compare_pair(before_files[relative], after_files[relative])
        for relative in common
    }
    same_shape = [value for value in comparisons.values() if value["same_shape"]]
    report = {
        "before": str(args.before),
        "after": str(args.after),
        "before_images": len(before_files),
        "after_images": len(after_files),
        "common_images": len(common),
        "missing_after": sorted(before_files.keys() - after_files.keys()),
        "new_after": sorted(after_files.keys() - before_files.keys()),
        "shape_mismatches": sum(not value["same_shape"] for value in comparisons.values()),
        "exact_matches": sum(value.get("exact", False) for value in comparisons.values()),
        "mean_mae": (
            float(np.mean([value["mae"] for value in same_shape]))
            if same_shape
            else None
        ),
        "max_abs_error": (
            float(max(value["max_abs_error"] for value in same_shape))
            if same_shape
            else None
        ),
        "files": comparisons,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
