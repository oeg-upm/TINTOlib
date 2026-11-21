"""R3 regression tests (v1.3.4 candidate) - ``y=None`` (problem="unsupervised") exports of the
non-IGTD methods, SuperTML EF/VF contract, and REFINED (deferred).

STATUS: Authored statically during R3.
In the subsequent historical R4 gate executed on macOS Terminal, the documented
aggregate result across the test campaign was 56 PASS / 1 SKIP, with one
REFINED/MPI case deferred. This historical aggregate figure is not a claim
that each test in this module received additional individual validation.
In R13-S9A no tests are executed, and no new claims regarding Python 3.14
support are asserted here.

What a ``None`` guard does and does NOT mean
--------------------------------------------
Guarding the *export* step (``target = None if y is None else y[i]``) only removes an incidental
TypeError at ``y[i]``.  It does NOT make every configuration of every method unsupervised: other
phases may still need labels (e.g. SuperTML-VF needs targets during FIT).  The cases below cover
ONLY the configurations that were statically checked to be label-free:

  BIE, BarGraph, DistanceMatrix, Combination, FeatureWrap  -> unsupervised export expected to work
  SuperTML(feature_importance=False)  (EF)                  -> expected to work (needs a system font)
  SuperTML(feature_importance=True)   (VF) with y=None      -> early, explicit ValueError expected
  REFINED                                                   -> DEFERRED to R4-MPI (needs mpirun)

External dependencies / preconditions
-------------------------------------
* SuperTML renders text with ``ImageFont.truetype("arial.ttf", ...)``.  If that font cannot be
  resolved on the machine, the EF export test is reported as an EXPLICIT skip prefixed
  ``HOLD_COVERAGE`` (it is a coverage HOLD, never a PASS).  The VF ValueError test needs no font.
* REFINED is only constructed elsewhere; fitting needs MPI/subprocess and is NOT exercised here.
* No real data, no network.  Each test uses its own TemporaryDirectory.

Seeds: synthetic data from numpy.random.default_rng(SEED_*).

Counting: NEW tests, tallied separately from the 15 R1 kit tests and the 5 upstream
tests/test_clusters_rbf.py tests.  ``subTest`` scenarios are counted as scenarios.
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image, ImageFont

from TINTOlib.barGraph import BarGraph
from TINTOlib.bie import BIE
from TINTOlib.combination import Combination
from TINTOlib.distanceMatrix import DistanceMatrix
from TINTOlib.featureWrap import FeatureWrap
from TINTOlib.supertml import SuperTML

SEED_UNSUP = 20261131
SEED_CLASS = 20261132

N_SAMPLES = 8
N_FEATURES = 6  # >= 3 is required by BarGraph/Combination (bar area = n - 2 rows)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def make_features(seed, n_samples=N_SAMPLES, n_features=N_FEATURES):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.uniform(0.0, 1.0, size=(n_samples, n_features)),
        columns=[f"f{j}" for j in range(n_features)],
    )


def arial_available():
    try:
        ImageFont.truetype("arial.ttf", 10)
        return True
    except OSError:
        return False


# name -> (factory(problem), expected PIL image size (width, height))
# Sizes derive from each method's _transformAlg / _img_to_file (v1.3.3 sources).
LABEL_FREE_METHODS = {
    "BIE": (lambda problem, **kw: BIE(problem=problem, **kw), (32, N_FEATURES)),       # N_FEATURES rows x 32 bits
    "BarGraph": (lambda problem, **kw: BarGraph(problem=problem, **kw), (N_FEATURES, N_FEATURES)),
    "DistanceMatrix": (lambda problem, **kw: DistanceMatrix(problem=problem, **kw), (N_FEATURES, N_FEATURES)),
    "Combination": (lambda problem, **kw: Combination(problem=problem, **kw), (N_FEATURES, N_FEATURES)),
    "FeatureWrap": (lambda problem, **kw: FeatureWrap(problem=problem, **kw), (8, 8)),  # default size (8, 8)
}


class UnsupervisedExportTests(unittest.TestCase):
    def setUp(self):
        self.df = make_features(SEED_UNSUP)

    def _assert_unsupervised_export(self, folder, size):
        self.assertEqual(set(os.listdir(folder)), {"images", "unsupervised.csv"})
        names = [f"{i:06d}.png" for i in range(N_SAMPLES)]
        self.assertEqual(sorted(os.listdir(os.path.join(folder, "images"))), names)

        table = pd.read_csv(os.path.join(folder, "unsupervised.csv"))
        self.assertEqual(list(table.columns), ["images", "values"] + list(self.df.columns))
        self.assertNotIn("target", table.columns)
        self.assertEqual(len(table), N_SAMPLES)
        self.assertEqual(table["images"].tolist(), [os.path.join("images", n) for n in names])
        self.assertTrue(table["values"].isna().all())
        np.testing.assert_allclose(
            table[list(self.df.columns)].to_numpy(), self.df.to_numpy(), rtol=1e-12, atol=0
        )
        for name in names:
            path = os.path.join(folder, "images", name)
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(8), PNG_MAGIC)
            with Image.open(path) as image:
                self.assertEqual(image.size, size, msg=path)

    def test_label_free_methods_fit_transform_without_targets(self):
        for name, (factory, size) in LABEL_FREE_METHODS.items():
            with self.subTest(method=name):
                with tempfile.TemporaryDirectory() as tmp:
                    factory("unsupervised").fit_transform(self.df, tmp)
                    self._assert_unsupervised_export(tmp, size)

    def test_label_free_methods_separate_fit_then_transform(self):
        for name, (factory, size) in LABEL_FREE_METHODS.items():
            with self.subTest(method=name):
                with tempfile.TemporaryDirectory() as tmp:
                    model = factory("unsupervised")
                    model.fit(self.df)
                    model.transform(self.df, tmp)
                    self._assert_unsupervised_export(tmp, size)

    def test_label_free_methods_unsupervised_with_none_transformer(self):
        # Data already lies in [0, 1], so disabling preprocessing is legitimate here.
        for name, (factory, size) in LABEL_FREE_METHODS.items():
            with self.subTest(method=name):
                model = factory("unsupervised", transformer=None)
                self.assertIsNone(model.transformer)
                with tempfile.TemporaryDirectory() as tmp:
                    model.fit_transform(self.df, tmp)
                    self._assert_unsupervised_export(tmp, size)


class SupervisedNonRegressionTests(unittest.TestCase):
    """The ``None`` guards must leave the classification layout untouched."""

    def test_classification_layout_unchanged(self):
        frame = make_features(SEED_CLASS)
        frame["label"] = np.arange(N_SAMPLES) % 2
        for name, (factory, size) in LABEL_FREE_METHODS.items():
            with self.subTest(method=name):
                with tempfile.TemporaryDirectory() as tmp:
                    factory("classification").fit_transform(frame, tmp)
                    self.assertEqual(set(os.listdir(tmp)), {"00", "01", "classification.csv"})
                    table = pd.read_csv(os.path.join(tmp, "classification.csv"))
                    self.assertEqual(table["values"].tolist(), (np.arange(N_SAMPLES) % 2).tolist())
                    for i in range(N_SAMPLES):
                        path = os.path.join(tmp, str(i % 2).zfill(2), f"{i:06d}.png")
                        self.assertTrue(os.path.isfile(path), msg=path)
                        with Image.open(path) as image:
                            self.assertEqual(image.size, size)


class SuperTMLUnsupervisedTests(unittest.TestCase):
    def setUp(self):
        self.df = make_features(SEED_UNSUP)

    def test_ef_without_targets_exports_images(self):
        """SuperTML-EF (feature_importance=False) needs no labels.  Requires an Arial-resolvable
        font; otherwise this is an explicit HOLD_COVERAGE (not a PASS)."""
        if not arial_available():
            self.skipTest("HOLD_COVERAGE: 'arial.ttf' not resolvable by Pillow on this machine; "
                          "SuperTML-EF unsupervised export is UNVERIFIED (register as coverage HOLD in R4).")
        with tempfile.TemporaryDirectory() as tmp:
            SuperTML(problem="unsupervised", feature_importance=False, pixels=64, font_size=8).fit_transform(self.df, tmp)
            self.assertEqual(set(os.listdir(tmp)), {"images", "unsupervised.csv"})
            names = [f"{i:06d}.png" for i in range(N_SAMPLES)]
            self.assertEqual(sorted(os.listdir(os.path.join(tmp, "images"))), names)
            table = pd.read_csv(os.path.join(tmp, "unsupervised.csv"))
            self.assertEqual(list(table.columns), ["images", "values"] + list(self.df.columns))
            self.assertTrue(table["values"].isna().all())
            with Image.open(os.path.join(tmp, "images", names[0])) as image:
                self.assertEqual(image.size, (64, 64))

    def test_vf_without_targets_is_rejected_early_with_value_error(self):
        """SuperTML-VF needs targets during FIT: explicit, reproducible ValueError, no files written,
        no fabricated feature importances."""
        with self.subTest(entry="fit_transform"):
            model = SuperTML(problem="unsupervised", feature_importance=True)
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError) as ctx:
                    model.fit_transform(self.df, tmp)
                self.assertIn("feature_importance", str(ctx.exception))
                self.assertEqual(os.listdir(tmp), [])
            self.assertFalse(hasattr(model, "feature_importances"))
            self.assertFalse(model._fitted)
        with self.subTest(entry="fit"):
            model = SuperTML(problem="unsupervised", feature_importance=True)
            with self.assertRaises(ValueError):
                model.fit(self.df)
            self.assertFalse(hasattr(model, "feature_importances"))
            self.assertFalse(model._fitted)

    def test_vf_with_targets_still_fits_feature_importances(self):
        """Behaviour with targets present must be unchanged (fit only: no font is needed)."""
        frame = make_features(SEED_CLASS, n_samples=16)
        frame["label"] = np.arange(16) % 2
        for problem in ("classification", "regression"):
            with self.subTest(problem=problem):
                model = SuperTML(problem=problem, feature_importance=True, random_seed=1)
                model.fit(frame)
                self.assertTrue(model._fitted)
                self.assertEqual(model.feature_importances.shape, (N_FEATURES,))
                self.assertTrue(np.isfinite(model.feature_importances).all())


class REFINEDDeferredR4MPITests(unittest.TestCase):
    """REFINED relies on mpirun/subprocess and temporary pickle/csv artifacts in the current
    working directory.  Nothing here may run it.  Coverage is DEFERRED to the isolated R4-MPI gate."""

    @unittest.skip("HOLD_COVERAGE (deferred to R4-MPI): REFINED unsupervised export "
                   "(target=None guard in __saveImages) requires mpirun; not authorised in R3.")
    def test_refined_unsupervised_export_TODO_R4_MPI(self):
        # TODO(R4-MPI): with mpi4py/mpirun available and authorised, run
        #   REFINED(problem="unsupervised", hcIterations=1, n_processors=2, format="npy",
        #          random_seed=1).fit_transform(<synthetic frame, 6+ features>, <tmp>)
        # from inside a temporary working directory (REFINED writes Init_MDS_Euc.pickle,
        # Mapping_REFINED_subprocess.pickle and REFINED_Evolve_subprocess.csv to the CWD) and assert:
        #   images/000000.npy ..., unsupervised.csv (images, values, features), features_positions.csv.
        raise NotImplementedError("deferred to R4-MPI")


if __name__ == "__main__":
    unittest.main()
