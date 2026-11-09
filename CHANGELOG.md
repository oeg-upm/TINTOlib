# Changelog

## 1.3.3 - 2026-11-09

TINTOlib 1.3.3 isolates default preprocessing between models, fixes unsupervised image export in DeepInsight and Fotomics, and strengthens Clusters image encoding and saved-state validation.

### Fixed

- Give each `TINTO`, `DeepInsight`, and `Fotomics` instance its own default transformer, preventing one model's fit from changing another model's preprocessing. Explicit `transformer=None` remains supported.
- Allow DeepInsight and Fotomics to export images for unsupervised problems without attempting to read missing target values.
- Round and clip Clusters pixel intensities to the range 0–255 before converting to `uint8`, and reject NaN or infinite intensities.
- Validate Clusters image inputs and target lengths before export, including multi-channel input.
- Reject fitted Clusters saved states without an individual fitted `StandardScaler`, before modifying the receiving model.

### Compatibility and dependencies

- Emit a deprecation warning when Clusters receives `normalize`; it has no effect because Clusters always applies `StandardScaler`.
- Retrain legacy fitted Clusters models that lack their fitted scaler; that preprocessing state cannot be recovered from the saved file.
- Clusters images may change when transformed intensities fall outside the training range: intensities are now clipped instead of wrapping during `uint8` conversion.

### Validation

- All five existing Clusters regression tests passed.
- Eight additional targeted diagnostic checks passed, covering independent default transformers, unsupervised export, safe intensity conversion, and rejection of incompatible fitted saved states. These diagnostics are not part of the committed test suite.
- A reduced comparison against the pre-fix commit `8d30d8b`, using synthetic classification data and four methods, generated 144 images per version: 141 exact matches, three Clusters differences consistent with intensity clipping, no missing or additional images, and no shape mismatches.
- The full method/dataset comparison matrix and the PowerShell wrapper were not executed.

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
