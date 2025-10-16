#!/usr/bin/env python3
"""
Generate quick_check YAML summaries from docs/_data/datasets/*/*/dataset.yml

Output files:
  docs/_data/quick_check/regression.yml
  docs/_data/quick_check/binary.yml
  docs/_data/quick_check/multiclass.yml

Logic:
  - For each dataset.yml, read leaderboard_rows and select best_classical among families ['Trees','MLP']
    and best_transformed among other families using the primary metric per task.
  - Primary metric mapping: regression -> test_rmse (lower is better), binary -> test_accuracy (higher), multiclass -> test_accuracy (higher)
  - Parse mean from strings like "15.02703 ± 0.62945" (take the number before ±). If absent, skip numeric comparison and pick first matching row.

This script requires PyYAML.
"""
import re
import glob
import os
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS_GLOB = ROOT / 'docs' / '_data' / 'datasets' / '*' / '*' / 'dataset.yml'
OUT_DIR = ROOT / 'docs' / '_data' / 'quick_check'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSICAL = {'Trees', 'MLP'}

def parse_mean(s):
    if not s:
        return None
    # match float at start
    m = re.search(r"([-+]?[0-9]*\.?[0-9]+)", str(s))
    if not m:
        return None
    try:
        return float(m.group(1).replace(',', ''))
    except Exception:
        return None

def primary_metric_for_task(task):
    if task == 'regression':
        return 'test_rmse', 'test_r2'
    if task == 'binary':
        return 'test_accuracy', 'test_roc_auc'
    if task == 'multiclass':
        return 'test_accuracy', 'test_f1'
    return 'test_rmse', None

def pick_best(rows, key, higher_is_better=True):
    best = None
    best_val = None
    for r in rows:
        v_raw = r.get(key)
        mean = parse_mean(v_raw) if isinstance(v_raw, str) else None
        if mean is None:
            # fallback: if no numeric value, pick first
            if best is None:
                best = r
            continue
        if best is None:
            best = r
            best_val = mean
            continue
        if higher_is_better:
            if mean > best_val:
                best = r
                best_val = mean
        else:
            if mean < best_val:
                best = r
                best_val = mean
    return best

def process():
    tasks = {'regression': [], 'binary': [], 'multiclass': []}
    for path in glob.glob(str(DATASETS_GLOB)):
        p = Path(path)
        with open(p, 'r', encoding='utf-8') as fh:
            data = yaml.safe_load(fh)
        meta = data.get('meta', {})
        task = meta.get('task')
        if task not in tasks:
            continue
        rows = data.get('leaderboard_rows', [])
        primary, secondary = primary_metric_for_task(task)
        higher = True
        if primary == 'test_rmse':
            higher = False

        # find classical rows
        classical_rows = [r for r in rows if r.get('family') in CLASSICAL or r.get('method') in CLASSICAL]
        other_rows = [r for r in rows if (r.get('family') not in CLASSICAL and r.get('method') not in CLASSICAL)]

        best_classical = pick_best(classical_rows, primary, higher_is_better=higher) if classical_rows else None
        best_other = pick_best(other_rows, primary, higher_is_better=higher) if other_rows else None

        def row_summary(r):
            if not r:
                return None
            # find variant name
            variant = r.get('best_variant') or r.get('method') or r.get('model')
            return {
                'family': r.get('family') or r.get('method') or 'N/A',
                'best_variant': variant,
                'primary_raw': r.get(primary),
                'primary_mean': parse_mean(r.get(primary)) if isinstance(r.get(primary), str) else r.get(primary),
                'secondary_raw': r.get(secondary) if secondary else None,
                'secondary_mean': parse_mean(r.get(secondary)) if secondary and isinstance(r.get(secondary), str) else None,
            }

        tasks[task].append({
            'dataset': meta.get('name'),
            'meta': {
                'task': task,
                'target': meta.get('target_variable'),
                'sample_size': meta.get('sample_size') or meta.get('instances'),
                'n_features': meta.get('n_features') or meta.get('features'),
                'n_num_features': meta.get('n_num_features'),
                'n_cat_features': meta.get('n_cat_features'),
                'categorical_pct': meta.get('cat_frac_est') or 0.0,
                'missing_pct': meta.get('missing_pct_estimated_from_file') or 0.0,
                'n_classes': meta.get('n_classes'),
                'imbalance_ratio': meta.get('imbalance_ratio'),
                'source': meta.get('source')
            },
            'best_classical': row_summary(best_classical),
            'best_transformed': row_summary(best_other)
        })

    # write outputs
    for task, items in tasks.items():
        out = OUT_DIR / f"{task}.yml"
        with open(out, 'w', encoding='utf-8') as fh:
            yaml.safe_dump(items, fh, sort_keys=False, allow_unicode=True)
        print(f"Wrote {out} with {len(items)} entries")

if __name__ == '__main__':
    process()
