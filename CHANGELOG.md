# Changelog

## 1.3.3 - 2026-11-03

### Metadata

- Add Salvador Martínez-Moreno to the software authors, without an email address, in the package metadata, citation file, CodeMeta, and README.
- Retain all runtime fixes and dependency constraints from 1.3.2; no algorithm changes.

## 1.3.2 - 2026-11-03

### Fixed

- Give each `Clusters` instance its own fitted `StandardScaler` and preserve it in newly saved model state.
- Fix `Clusters` unsupervised image export and validate the dimensions and length of label arrays.
- Handle zero-bandwidth RBF encoding for identical training rows. Matching rows have similarity 1 and different rows have similarity 0; these values map to intensities 255 and 0 without division by zero or negative intensities. Positive-bandwidth encoding retains its existing calculation and fitted intensity range.
- Correct the Factor Analysis description, method parameters, classification examples, installation guidance, and README badge links.

### Packaging

- Declare `scikit-image` as a core dependency for automatic cluster selection.
- Restrict NumPy to versions below 2.4 for the `refined` and `all` extras to retain REFINED compatibility. The core package keeps its existing NumPy range.
- Synchronize package, citation, and CodeMeta versions and release links.
- Include the changelog, citation file, and CodeMeta file in the source distribution.

### Validation

- Add regression tests for zero-bandwidth RBF encoding, new-data intensity scaling, automatic cluster selection, and fitted-state save/load.

## 1.3.1 - 2026-08-26

Previous release. See the [v1.3.1 source](https://github.com/oeg-upm/TINTOlib/tree/v1.3.1).
