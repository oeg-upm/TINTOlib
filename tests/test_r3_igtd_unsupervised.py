"""R3 regression tests (v1.3.4 candidate) - IGTD unsupervised contract and IGTD non-regression.

STATUS: Authored statically during R3.
In the subsequent historical R4 gate executed on macOS Terminal, the documented
aggregate result across the test campaign was 56 PASS / 1 SKIP, with one
REFINED/MPI case deferred. This historical aggregate figure is not a claim
that each test in this module received additional individual validation.
In R13-S9A no tests are executed, and no new claims regarding Python 3.14
support are asserted here.

Contract under test (derived from abstractImageMethod.py / igtd.py / mappingMethod.py, v1.3.3):
    problem="unsupervised" ->
      <folder>/images/000000.<fmt> ... 00000N.<fmt>      (no class sub-folders)
      <folder>/unsupervised.csv   columns: images, values, <original features...>
                                  ('values' is empty because there is no target;
                                   no invented column such as 'target' may appear)
      <folder>/features_positions.csv  columns: feature, row, column

Requirements: numpy, pandas, scikit-learn, matplotlib, Pillow, scipy, tqdm (core deps).
No real data, no MPI, no fonts, no network.  Every test uses its own TemporaryDirectory.

Seeds: synthetic data from numpy.random.default_rng(SEED_*) (declared below); IGTD itself is
seeded through its ``random_seed`` parameter (left at its default of 1 unless stated).

Counting: NEW tests, tallied separately from the 15 R1 kit tests and the 5 upstream
tests/test_clusters_rbf.py tests.
"""
import inspect
import os
import re
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import TINTOlib.igtd as igtd_module
from TINTOlib.igtd import IGTD

SEED_UNSUP = 20261121
SEED_CLASS = 20261122
SEED_REG = 20261123

N_SAMPLES = 16
N_FEATURES = 4
SCALE = [2, 2]
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def make_features(seed, n_samples=N_SAMPLES, n_features=N_FEATURES, low=0.0, high=1.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.uniform(low, high, size=(n_samples, n_features)),
        columns=[f"f{j}" for j in range(n_features)],
    )


def list_files(folder):
    return sorted(os.listdir(folder))


class IGTDUnsupervisedContractTests(unittest.TestCase):
    def setUp(self):
        self.df = make_features(SEED_UNSUP)
        self.feature_names = list(self.df.columns)

    # ------------------------------------------------------------------ helpers
    def _assert_unsupervised_contract(self, folder, fmt, df):
        n = len(df)
        # --- layout: only "images/" + the two csv files; no class sub-folders
        self.assertEqual(
            set(os.listdir(folder)),
            {"images", "unsupervised.csv", "features_positions.csv"},
        )
        images_dir = os.path.join(folder, "images")
        expected_files = [f"{i:06d}.{fmt}" for i in range(n)]
        self.assertEqual(list_files(images_dir), expected_files)

        # --- unsupervised.csv: images, values, original variables (in order)
        table = pd.read_csv(os.path.join(folder, "unsupervised.csv"))
        self.assertEqual(list(table.columns), ["images", "values"] + list(df.columns))
        self.assertNotIn("target", table.columns)
        self.assertEqual(len(table), n)
        self.assertEqual(
            table["images"].tolist(),
            [os.path.join("images", name) for name in expected_files],
        )
        self.assertTrue(table["values"].isna().all())
        # original (un-scaled) variables are exported unchanged, row aligned
        np.testing.assert_allclose(
            table[list(df.columns)].to_numpy(), df.to_numpy(), rtol=1e-12, atol=0
        )

        # --- features_positions.csv: feature,row,column; feature order = input order
        positions = pd.read_csv(os.path.join(folder, "features_positions.csv"))
        self.assertEqual(list(positions.columns), ["feature", "row", "column"])
        self.assertEqual(positions["feature"].tolist(), list(df.columns))
        self.assertTrue(positions["row"].between(0, SCALE[0] - 1).all())
        self.assertTrue(positions["column"].between(0, SCALE[1] - 1).all())
        cells = set(zip(positions["row"].tolist(), positions["column"].tolist()))
        self.assertEqual(len(cells), len(df.columns))  # one distinct pixel per feature
        return table, positions

    # -------------------------------------------------------------------- tests
    def test_fit_transform_npy_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = IGTD(problem="unsupervised", scale=SCALE, format="npy")
            model.fit_transform(self.df, tmp)
            self._assert_unsupervised_contract(tmp, "npy", self.df)
            for i in range(N_SAMPLES):
                array = np.load(os.path.join(tmp, "images", f"{i:06d}.npy"))
                self.assertEqual(array.shape, tuple(SCALE))
                self.assertTrue(np.isfinite(array).all())
                self.assertTrue(((array >= 0) & (array <= 255)).all())

    def test_fit_transform_png_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = IGTD(problem="unsupervised", scale=SCALE)  # default format: png
            model.fit_transform(self.df, tmp)
            self._assert_unsupervised_contract(tmp, "png", self.df)
            for i in range(N_SAMPLES):
                path = os.path.join(tmp, "images", f"{i:06d}.png")
                with open(path, "rb") as handle:
                    self.assertEqual(handle.read(8), PNG_MAGIC)
                with Image.open(path) as image:
                    self.assertEqual(image.size, (SCALE[1], SCALE[0]))

    def test_separate_fit_then_transform_produces_same_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = IGTD(problem="unsupervised", scale=SCALE, format="npy")
            model.fit(self.df)
            model.transform(self.df, tmp)
            self._assert_unsupervised_contract(tmp, "npy", self.df)

    def test_non_default_dataframe_index_does_not_misalign_exports(self):
        shifted = self.df.copy()
        shifted.index = np.arange(100, 100 + N_SAMPLES)
        with tempfile.TemporaryDirectory() as tmp:
            model = IGTD(problem="unsupervised", scale=SCALE, format="npy")
            model.fit_transform(shifted, tmp)
            self._assert_unsupervised_contract(tmp, "npy", shifted.reset_index(drop=True))

    def test_image_values_follow_features_positions_contract(self):
        """Order contract: the pixel of feature j in image i holds the scaled value of
        feature j for sample i, mapped to [0, 255] as in IGTD.__generate_image_data
        (255 - (v - min) / (max - min) * 255, global min/max).  The pixel position is read
        from features_positions.csv, so this also guards the I1 mapping fix."""
        with tempfile.TemporaryDirectory() as tmp:
            model = IGTD(problem="unsupervised", scale=SCALE, format="npy")
            model.fit_transform(self.df, tmp)
            positions = pd.read_csv(os.path.join(tmp, "features_positions.csv"))

            scaled = MinMaxScaler().fit_transform(self.df)
            gmin, gmax = scaled.min(), scaled.max()
            expected_values = 255 - (scaled - gmin) / (gmax - gmin) * 255

            for i in range(N_SAMPLES):
                array = np.load(os.path.join(tmp, "images", f"{i:06d}.npy"))
                for j, name in enumerate(self.feature_names):
                    row = int(positions.loc[positions["feature"] == name, "row"].iloc[0])
                    col = int(positions.loc[positions["feature"] == name, "column"].iloc[0])
                    self.assertAlmostEqual(
                        array[row, col], expected_values[i, j], places=9,
                        msg=f"image {i}, feature {name} at ({row},{col})",
                    )

    def test_same_seed_gives_identical_outputs_in_independent_folders(self):
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            IGTD(problem="unsupervised", scale=SCALE, format="npy", random_seed=1).fit_transform(self.df, tmp_a)
            IGTD(problem="unsupervised", scale=SCALE, format="npy", random_seed=1).fit_transform(self.df, tmp_b)
            for i in range(N_SAMPLES):
                np.testing.assert_array_equal(
                    np.load(os.path.join(tmp_a, "images", f"{i:06d}.npy")),
                    np.load(os.path.join(tmp_b, "images", f"{i:06d}.npy")),
                )
            pd.testing.assert_frame_equal(
                pd.read_csv(os.path.join(tmp_a, "features_positions.csv")),
                pd.read_csv(os.path.join(tmp_b, "features_positions.csv")),
            )

    def test_unsupervised_with_none_and_custom_transformer(self):
        for label, transformer in (("none", None), ("custom_standard_scaler", StandardScaler())):
            with self.subTest(transformer=label):
                with tempfile.TemporaryDirectory() as tmp:
                    model = IGTD(problem="unsupervised", scale=SCALE, format="npy", transformer=transformer)
                    model.fit_transform(self.df, tmp)
                    self._assert_unsupervised_contract(tmp, "npy", self.df)


class IGTDSupervisedNonRegressionTests(unittest.TestCase):
    """The ``target = None if labels is None else labels[i]`` guard must not alter the
    supervised paths (classification / regression)."""

    def test_classification_layout_and_values_unchanged(self):
        frame = make_features(SEED_CLASS, n_samples=18)
        frame["label"] = np.arange(18) % 3  # integer classes 0,1,2
        with tempfile.TemporaryDirectory() as tmp:
            IGTD(problem="classification", scale=SCALE, format="npy").fit_transform(frame, tmp)
            self.assertEqual(
                set(os.listdir(tmp)),
                {"00", "01", "02", "classification.csv", "features_positions.csv"},
            )
            table = pd.read_csv(os.path.join(tmp, "classification.csv"))
            self.assertEqual(list(table.columns), ["images", "values", "f0", "f1", "f2", "f3"])
            self.assertEqual(table["values"].tolist(), (np.arange(18) % 3).tolist())
            for i in range(18):
                cls = str(i % 3).zfill(2)
                self.assertTrue(os.path.isfile(os.path.join(tmp, cls, f"{i:06d}.npy")))

    def test_regression_layout_and_values_unchanged(self):
        frame = make_features(SEED_REG, n_samples=12)
        frame["y"] = np.linspace(-1.0, 1.0, 12)
        with tempfile.TemporaryDirectory() as tmp:
            IGTD(problem="regression", scale=SCALE, format="npy").fit_transform(frame, tmp)
            self.assertEqual(
                set(os.listdir(tmp)),
                {"images", "regression.csv", "features_positions.csv"},
            )
            table = pd.read_csv(os.path.join(tmp, "regression.csv"))
            np.testing.assert_allclose(table["values"].to_numpy(), np.linspace(-1.0, 1.0, 12), rtol=1e-12)
            self.assertEqual(
                list_files(os.path.join(tmp, "images")), [f"{i:06d}.npy" for i in range(12)]
            )

    def test_features_positions_matches_inverse_of_index_I1(self):
        """T6 - I1: features_positions.csv equals argsort of the optimised index."""
        frame = make_features(SEED_CLASS, n_samples=18)
        frame["label"] = np.arange(18) % 2
        with tempfile.TemporaryDirectory() as tmp:
            model = IGTD(problem="classification", scale=SCALE, format="npy")
            model.fit_transform(frame, tmp)
            positions = pd.read_csv(os.path.join(tmp, "features_positions.csv"))
        inverse = np.argsort(model.index[model.min_id, :].astype(int))
        self.assertEqual(positions["feature"].tolist(), list(frame.columns[:-1]))
        self.assertEqual(positions["row"].tolist(), (inverse // SCALE[1]).tolist())
        self.assertEqual(positions["column"].tolist(), (inverse % SCALE[1]).tolist())

    def test_scale_too_small_raises_value_error_I8(self):
        frame = make_features(SEED_CLASS, n_samples=8)
        frame["label"] = np.arange(8) % 2
        too_small = [1, 2]  # 2 pixels < 4 features
        with self.subTest(entry="fit"):
            with self.assertRaises(ValueError) as ctx:
                IGTD(problem="classification", scale=too_small).fit(frame)
            self.assertIn("scale", str(ctx.exception))
        with self.subTest(entry="fit_transform"):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    IGTD(problem="classification", scale=too_small).fit_transform(frame, tmp)
                self.assertEqual(os.listdir(tmp), [])  # nothing written before the error
        with self.subTest(entry="unsupervised_fit_transform"):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    IGTD(problem="unsupervised", scale=too_small).fit_transform(frame.iloc[:, :-1], tmp)

    def test_no_deprecated_np_str_alias_L1(self):
        source = inspect.getsource(igtd_module)
        self.assertIsNone(re.search(r"\bnp\.str(?![_\w])", source), "deprecated np.str alias found")


if __name__ == "__main__":
    unittest.main()
