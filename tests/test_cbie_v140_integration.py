"""TINTOlib 1.4.0 cBIE integration regression tests.

Synthetic fixtures only. Prepared under G1 without executing tests.
Run only in a new, explicitly authorised G2 on the integrated source tree.
Historical R1/G2 gates and their locks must never be rerun.
"""

import tempfile
import unittest
from pathlib import Path

import bitstring
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

from TINTOlib.bie import BIE


def _training_frame():
    rng = np.random.default_rng(140)
    y = np.tile([0, 1, 2], 15)
    f0 = y.astype(float) + rng.normal(0.0, 0.05, y.size)
    f1 = rng.normal(0.0, 1.0, y.size)
    f2 = f0 * 0.8 + rng.normal(0.0, 0.1, y.size)
    f3 = rng.normal(0.0, 0.5, y.size)
    return pd.DataFrame({"f0": f0, "f1": f1, "f2": f2, "f3": f3, "target": y})


def _images(folder):
    result = {}
    for path in sorted(Path(folder).rglob("*.png")):
        with Image.open(path) as image:
            result[path.name] = np.asarray(image.convert("L")).copy()
    return result


class CBIEIntegratedContractTests(unittest.TestCase):
    def setUp(self):
        self.frame = _training_frame()

    def test_correlated_true_is_default_and_classic_is_opt_in(self):
        self.assertTrue(BIE().correlated)
        self.assertTrue(BIE(correlated=True).correlated)
        self.assertFalse(BIE(correlated=False).correlated)
        with self.assertRaisesRegex(TypeError, "correlated must be of type bool"):
            BIE(correlated="yes")

    def test_classic_mode_preserves_original_order_without_correlations(self):
        model = BIE(problem="classification", transformer=None, correlated=False)
        model.fit(self.frame)
        self.assertEqual(model._BIE__columns_sorted, [])
        with tempfile.TemporaryDirectory() as tmp:
            model.transform(self.frame.iloc[:4], tmp)
            self.assertEqual(len(_images(tmp)), 4)

    def test_classic_pixels_follow_original_ieee754_feature_order(self):
        model = BIE(problem="classification", transformer=None, correlated=False, precision=32)
        data = self.frame.iloc[:3].copy()
        with tempfile.TemporaryDirectory() as tmp:
            model.fit_transform(data, tmp)
            output = _images(tmp)
            for row_index, (_, row) in enumerate(data.iterrows()):
                expected = np.asarray([
                    [255 if b == "1" else 0 for b in bitstring.BitArray(float=float(row[col]), length=32).bin]
                    for col in ("f0", "f1", "f2", "f3")
                ], dtype=np.uint8)
                np.testing.assert_array_equal(output[f"{row_index:06d}.png"], expected)

    def test_classic_mode_does_not_require_defined_pearson_correlations(self):
        frame = self.frame.copy()
        frame["f2"] = 7.0  # Constant features do not affect classic BIE layout.
        model = BIE(problem="classification", transformer=None, correlated=False)
        model.fit(frame)
        self.assertEqual(model._BIE__columns_sorted, [])

    def test_classic_flag_survives_save_and_load(self):
        model = BIE(problem="classification", transformer=None, correlated=False)
        model.fit(self.frame)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "classic.pkl"
            model.saveHyperparameters(str(path))
            restored = BIE(problem="classification", transformer=None)
            restored.loadHyperparameters(str(path))
            self.assertIs(restored.correlated, False)
            self.assertEqual(restored._BIE__columns_sorted, [])

    def test_default_scaler_not_shared_with_another_instance(self):
        first = BIE(problem="classification")
        second = BIE(problem="classification")
        self.assertIsInstance(first.transformer, MinMaxScaler)
        self.assertIsInstance(second.transformer, MinMaxScaler)
        self.assertIsNot(first.transformer, second.transformer)

    def test_supervised_order_is_learned_and_reused(self):
        model = BIE(problem="classification", transformer=None, precision=32)
        model.fit(self.frame)
        order = model._BIE__columns_sorted
        self.assertEqual(set(order), {"f0", "f1", "f2", "f3"})
        self.assertEqual(len(order), 4)

        batch = self.frame.iloc[:6].copy()
        shuffled = batch[["f3", "f1", "f0", "f2", "target"]]
        changed_y = batch.copy()
        changed_y["target"] = (changed_y["target"] + 1) % 3
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            reorder = Path(tmp) / "reorder"
            other_y = Path(tmp) / "other_y"
            model.transform(batch, str(base))
            model.transform(shuffled, str(reorder))
            model.transform(changed_y, str(other_y))
            images = _images(base)
            self.assertEqual(images.keys(), _images(reorder).keys())
            self.assertEqual(images.keys(), _images(other_y).keys())
            for key, image in images.items():
                np.testing.assert_array_equal(image, _images(reorder)[key])
                np.testing.assert_array_equal(image, _images(other_y)[key])

    def test_unsupervised_export_keeps_134_fix(self):
        model = BIE(problem="unsupervised", transformer=None, precision=32, correlated=True)
        with tempfile.TemporaryDirectory() as tmp:
            model.fit_transform(self.frame.drop(columns=["target"]), tmp)
            self.assertEqual(len(_images(tmp)), len(self.frame))
            self.assertEqual(model._BIE__columns_sorted, [])

    def test_persistence_retains_order(self):
        model = BIE(problem="classification", transformer=None)
        model.fit(self.frame)
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.pkl"
            model.saveHyperparameters(str(state))
            loaded = BIE(problem="classification", transformer=None)
            loaded.loadHyperparameters(str(state))
            self.assertEqual(model._BIE__columns_sorted, loaded._BIE__columns_sorted)
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            model.transform(self.frame.iloc[:5], str(first))
            loaded.transform(self.frame.iloc[:5], str(second))
            for key, pixels in _images(first).items():
                np.testing.assert_array_equal(pixels, _images(second)[key])


    def test_default_scaler_correlated_layout_is_index_invariant(self):
        # A transformed X gets a fresh RangeIndex in AbstractImageMethod.
        # A y Series can retain original nonconsecutive indices.
        shifted = self.frame.copy()
        shifted.index = np.arange(101, 101 + len(shifted))
        baseline = BIE(problem="classification")  # default MinMaxScaler
        alternate = BIE(problem="classification")
        baseline.fit(self.frame)
        alternate.fit(shifted)
        self.assertEqual(
            baseline._BIE__columns_sorted,
            alternate._BIE__columns_sorted,
        )

    def test_undefined_pearson_fails_before_learning_order(self):
        for scenario in ("constant_feature", "constant_target"):
            with self.subTest(scenario=scenario):
                frame = self.frame.copy()
                if scenario == "constant_feature":
                    frame["f2"] = 3.0
                else:
                    frame["target"] = 1
                model = BIE(problem="classification", transformer=None)
                with self.assertRaisesRegex(ValueError, "finite, defined Pearson"):
                    model.fit(frame)

    def test_failed_correlated_refit_invalidates_prior_model(self):
        """Do not silently export a classic layout after failed reordering."""
        model = BIE(problem="classification", transformer=None)
        model.fit(self.frame)
        self.assertTrue(model._fitted)
        self.assertEqual(len(model._BIE__columns_sorted), 4)
        bad = self.frame.copy()
        bad["f2"] = 7.0
        with self.assertRaisesRegex(ValueError, "finite, defined Pearson"):
            model.fit(bad)
        self.assertFalse(model._fitted)
        self.assertEqual(model._BIE__columns_sorted, [])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                model.transform(self.frame, tmp)

    def test_failed_fit_transform_invalidates_prior_model(self):
        """Combined fit-transform must not retain stale fitted state on error."""
        model = BIE(problem="classification", transformer=None)
        model.fit(self.frame)
        bad = self.frame.copy()
        bad["target"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "finite, defined Pearson"):
                model.fit_transform(bad, tmp)
        self.assertFalse(model._fitted)

    def test_failed_input_validation_refit_invalidates_prior_model(self):
        """A failure before _fitAlg also invalidates the prior fitted model."""
        model = BIE(problem="classification", transformer=None)
        model.fit(self.frame)
        bad = self.frame.copy()
        bad.loc[0, "f0"] = np.nan
        with self.assertRaisesRegex(TypeError, "missing values"):
            model.fit(bad)
        self.assertFalse(model._fitted)

    def test_nan_values_are_rejected_by_existing_base_loader(self):
        frame = self.frame.copy()
        frame.loc[0, "f0"] = np.nan
        model = BIE(problem="classification", transformer=None)
        with self.assertRaisesRegex(TypeError, "missing values"):
            model.fit(frame)


    def test_regression_continuous_target_correlated_layout_reused(self):
        """cBIE regression learns from continuous TRAIN labels, not transform labels."""
        train = self.frame.copy()
        train["target"] = (
            1.7 * train["f0"] - 0.45 * train["f1"] + 0.2 * train["f3"]
        )
        self.assertTrue(pd.api.types.is_float_dtype(train["target"]))
        self.assertGreater(train["target"].nunique(), 3)
        model = BIE(problem="regression", transformer=None, correlated=True, precision=32)
        model.fit(train)
        trained_order = tuple(model._BIE__columns_sorted)
        self.assertEqual(set(trained_order), {"f0", "f1", "f2", "f3"})
        self.assertEqual(len(trained_order), 4)

        batch = train.iloc[5:12].copy()
        different_targets = batch.copy()
        different_targets["target"] = different_targets["target"] + 100.25
        with tempfile.TemporaryDirectory() as tmp:
            original_dir = Path(tmp) / "original_labels"
            modified_dir = Path(tmp) / "modified_labels"
            model.transform(batch, str(original_dir))
            model.transform(different_targets, str(modified_dir))
            original = _images(original_dir)
            changed = _images(modified_dir)
            self.assertEqual(len(original), len(batch))
            self.assertEqual(original.keys(), changed.keys())
            for name, pixels in original.items():
                np.testing.assert_array_equal(pixels, changed[name])
        self.assertEqual(tuple(model._BIE__columns_sorted), trained_order)

    def test_correlated_default_fit_transform_equals_separate_fit_and_transform(self):
        """Default scaler and cBIE give identical PNGs for equivalent fit flows."""
        train = self.frame.copy()
        combined = BIE(problem="classification", precision=32)
        separated = BIE(problem="classification", precision=32)
        self.assertTrue(combined.correlated)
        self.assertTrue(separated.correlated)

        with tempfile.TemporaryDirectory() as tmp:
            combined_dir = Path(tmp) / "combined"
            separated_dir = Path(tmp) / "separated"
            combined.fit_transform(train, str(combined_dir))
            separated.fit(train)
            separated.transform(train, str(separated_dir))
            self.assertEqual(
                combined._BIE__columns_sorted,
                separated._BIE__columns_sorted,
            )
            from_combined = _images(combined_dir)
            from_separated = _images(separated_dir)
            self.assertEqual(len(from_combined), len(train))
            self.assertEqual(from_combined.keys(), from_separated.keys())
            for name, pixels in from_combined.items():
                np.testing.assert_array_equal(pixels, from_separated[name])

if __name__ == "__main__":
    unittest.main()
