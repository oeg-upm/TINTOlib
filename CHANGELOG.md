# Changelog

## 1.4.0 - 2026-11-20

TINTOlib 1.4.0 integrates correlated Binary Image Encoding (cBIE, from Javier López's PR #30) into `BIE` with an explicit opt-out to the original BIE ordering, while retaining all TINTOlib 1.3.4 fixes and declaring Python 3.14 in package classifiers.

### Changed

- **Behavior change in `BIE()` (supervised default):** `BIE()` and `BIE(correlated=True)` now use the Pearson/greedy feature-ordering algorithm for supervised `classification` and `regression`. Feature ordering is learned only during `fit` and reused during `transform` as a deterministic transformation of observations (it does not generate new observations and does not claim default predictive-performance gains).
- Added final parameter `correlated: bool = True` to `BIE(...)`, preserving all previous parameter positions. Pass `BIE(correlated=False)` to retain the original IEEE-754 feature order without computing Pearson correlations.
- `BIE(problem="unsupervised")` always preserves the original feature order and X-only (`y=None`) export regardless of `correlated`.
- Callers remain responsible for numeric preprocessing of `X` and external numeric encoding of classification targets `y`; no internal label encoder, no new feature-schema validation, and no value imputation are introduced. Undefined/non-finite Pearson correlations (e.g., constant training features or target) raise a descriptive `ValueError` only in supervised `correlated=True` mode.
- Added `"Programming Language :: Python :: 3.14"` to `pyproject.toml` classifiers. Optional runtime requirements (notably MPI for `REFINED`) remain platform-dependent.

### Fixed

- Aligned transformed `X` and `y` positionally (`reset_index(drop=True)`) during supervised `BIE._fitAlg` correlation computation so non-consecutive `DataFrame` indices do not misalign features and targets when using `MinMaxScaler`.
- Preserved the historical random MDS initialization in `REFINED` across scikit-learn versions, passing `MDS(init="random")` only where supported to avoid the MDS initialization deprecation warning.

## 1.3.4 - 2026-11-17

TINTOlib 1.3.4 is a maintenance release for collaborator testing. It includes
corrections to tabular-to-image transformations and updates dependency constraints
relative to TINTOlib 1.3.3.

### Changed

- Updated REFINED's parallel Hill-climbing helper to use NumPy reductions. No
  quantitative runtime improvement is claimed.
- Updated optional dependency constraints for `greedy`, `refined` and `all` by
  Python version, including NumPy `>=2.5,<2.6` for REFINED on Python 3.14 and
  `<2.4` on earlier versions.

### Fixed

- Prevented sharing mutable default preprocessing transformers between model
  instances of IGTD, BIE, BarGraph, DistanceMatrix, Combination, FeatureWrap,
  SuperTML and REFINED.
- Corrected missing-label handling in supported unsupervised export workflows;
  SuperTML VF explicitly requires labels.
- Adjusted REFINED's image-pixel assignment to extract a scalar tabular value.

### Installation

For this GitHub pre-release, download the version 1.3.4 wheel from the release
assets and install that local wheel. Version 1.3.4 is not being uploaded to PyPI
as part of the collaborator-testing phase. REFINED requires a system MPI runtime
in addition to `mpi4py`; SuperTML requires an available Arial font.

## 1.3.3 - 2026-11-12

TINTOlib 1.3.3 fixes TINTO and IGTD image generation and feature-coordinate exports, isolates default preprocessing between models, and strengthens Clusters image encoding and saved-state validation.

### Changed

- Clusters clips transformed intensities outside the training range instead of wrapping during `uint8` conversion, so affected image pixels can change.
- Retrain legacy fitted Clusters models that lack their fitted scaler; that preprocessing state cannot be recovered from the saved file.
- Corrected TINTO zoom and non-submatrix intensity scaling can change image dimensions or intensities relative to earlier artifacts.

### Deprecated

- Emit a deprecation warning when Clusters receives `normalize`; it has no effect because Clusters always applies `StandardScaler`.

### Fixed

- Give each `TINTO`, `DeepInsight`, and `Fotomics` instance its own default transformer, preventing one model's fit from changing another model's preprocessing. Explicit `transformer=None` remains supported.
- Allow DeepInsight and Fotomics to export images for unsupervised problems without attempting to read missing target values.
- Round and clip Clusters pixel intensities to the range 0–255 before converting to `uint8`, and reject NaN or infinite intensities.
- Validate Clusters image inputs and target lengths before export, including multi-channel input.
- Reject fitted Clusters saved states without an individual fitted `StandardScaler`, before modifying the receiving model.
- Fix IGTD feature-to-pixel coordinates by inverting the optimized feature permutation before exporting the mapping CSV.
- Apply TINTO zoom once instead of twice in the submatrix image-generation path.
- Correct TINTO feature-coordinate exports when `submatrix=False` and remove the extra multiplication by 255 in that image-generation path.
- Raise `ValueError` when IGTD's pixel grid cannot hold all features, and replace removed NumPy `np.str` usage with Python `str` for DataFrame sample identifiers.

### Installation

After this version is published to PyPI:

```bash
pip install TINTOlib==1.3.3
```

## 1.3.2 - 2026-11-03

TINTOlib 1.3.2 fixes Clusters preprocessing and image export, handles identical training rows in RBF encoding, and updates installation dependencies and compatibility constraints.

### Fixed

- Give each `Clusters` instance its own fitted `StandardScaler` and preserve it in newly saved model state.
- Fix `Clusters` unsupervised image export and validate the dimensions and length of label arrays.
- Handle zero-bandwidth RBF encoding for identical training rows. Matching rows have similarity 1 and different rows have similarity 0; these values map to intensities 255 and 0 without division by zero or negative intensities. Positive-bandwidth encoding retains its existing calculation and fitted intensity range.
- Correct the Factor Analysis description, method parameters, classification examples, installation guidance, and README badge links.

### Compatibility and dependencies

- Declare `scikit-image` as a core dependency for automatic cluster selection.
- Restrict NumPy to versions below 2.4 for the `refined` and `all` extras to retain REFINED compatibility. The core package keeps its existing NumPy range.
- Synchronize package, citation, and CodeMeta versions and release links.
- Include the changelog, citation file, and CodeMeta file in the source distribution.

### Validation

- Add regression tests for zero-bandwidth RBF encoding, new-data intensity scaling, automatic cluster selection, and fitted-state save/load.

### Installation

```bash
pip install TINTOlib==1.3.2
```

## 1.3.1 - 2026-08-26

Previous release. See the [v1.3.1 source](https://github.com/oeg-upm/TINTOlib/tree/v1.3.1).
