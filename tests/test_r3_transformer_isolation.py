"""R3 regression tests (v1.3.4 candidate) - shared default ``transformer`` isolation.

STATUS: Authored statically during R3.
In the subsequent historical R4 gate executed on macOS Terminal, the documented
aggregate result across the test campaign was 56 PASS / 1 SKIP, with one
REFINED/MPI case deferred. This historical aggregate figure is not a claim
that each test in this module received additional individual validation.
In R13-S9A no tests are executed, and no new claims regarding Python 3.14
support are asserted here.

Scope
-----
* Identity isolation of the default ``MinMaxScaler`` for the 8 constructors patched in
  R3 (IGTD, BIE, BarGraph, DistanceMatrix, Combination, SuperTML, FeatureWrap, REFINED).
* Explicit ``transformer=None`` and user supplied (custom) transformers.
* Signature / parameter-order preservation by introspection.
* Non-regression of the v1.3.3 sentinel isolation in TINTO, DeepInsight and Fotomics
  (their implementation is NOT touched by R3).
* IGTD FIT(A) -> FIT(B) -> TRANSFORM(A) sequence.
* Persistence (saveHyperparameters / loadHyperparameters) of fitted and custom transformers.

Requirements: numpy, pandas, scikit-learn, matplotlib, Pillow, scipy, tqdm, bitstring
(core dependencies of TINTOlib).  No real data, no MPI, no external fonts, no network.
REFINED is only *constructed* here (never fitted); its MPI integration is deferred to R4-MPI.

Determinism: every synthetic dataset comes from ``numpy.random.default_rng(SEED)``
with the seeds declared below; every test writes into its own ``TemporaryDirectory``.

Counting: these tests are NEW and must be tallied separately from the 15 R1 kit tests
and the 5 upstream ``tests/test_clusters_rbf.py`` tests.  Parametrised scenarios are
implemented with ``subTest`` (one test method = several scenarios) and are counted
as scenarios, not as additional test methods.
"""
import inspect
import os
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

import TINTOlib.bie as bie_module
import TINTOlib.barGraph as bargraph_module
import TINTOlib.combination as combination_module
import TINTOlib.distanceMatrix as distancematrix_module
import TINTOlib.featureWrap as featurewrap_module
import TINTOlib.igtd as igtd_module
import TINTOlib.refined as refined_module
import TINTOlib.supertml as supertml_module
from TINTOlib.barGraph import BarGraph
from TINTOlib.bie import BIE
from TINTOlib.combination import Combination
from TINTOlib.distanceMatrix import DistanceMatrix
from TINTOlib.featureWrap import FeatureWrap
from TINTOlib.igtd import IGTD
from TINTOlib.refined import REFINED
from TINTOlib.supertml import SuperTML

# Seeds (declared explicitly for reproducibility)
SEED_DF1 = 20261112
SEED_DF2 = 20261113
SEED_PERSIST = 20261114

# (name, class, module, expected positional parameter order after ``self``,
#  expected defaults for every parameter except ``transformer``)
# Orders/defaults transcribed from the v1.3.3 sources (commit 307f1918...).
PATCHED = [
    (
        "IGTD", IGTD, igtd_module,
        ["problem", "transformer", "verbose", "scale", "fea_dist_method", "image_dist_method",
         "error", "max_step", "val_step", "switch_t", "min_gain", "zoom", "format", "cmap",
         "random_seed"],
        dict(problem=None, verbose=None, scale=[6, 6], fea_dist_method="Pearson",
             image_dist_method="Euclidean", error="squared", max_step=1000, val_step=50,
             switch_t=0, min_gain=0.00001, zoom=1, format="png", cmap="gray", random_seed=1),
    ),
    (
        "BIE", BIE, bie_module,
        ["problem", "transformer", "verbose", "precision", "zoom", "correlated"],
        dict(problem=None, verbose=None, precision=32, zoom=1, correlated=True),
    ),
    (
        "BarGraph", BarGraph, bargraph_module,
        ["problem", "transformer", "verbose", "pixel_width", "gap", "zoom"],
        dict(problem=None, verbose=None, pixel_width=1, gap=0, zoom=1),
    ),
    (
        "DistanceMatrix", DistanceMatrix, distancematrix_module,
        ["problem", "transformer", "verbose", "zoom"],
        dict(problem=None, verbose=None, zoom=1),
    ),
    (
        "Combination", Combination, combination_module,
        ["problem", "transformer", "verbose", "zoom"],
        dict(problem=None, verbose=None, zoom=1),
    ),
    (
        "SuperTML", SuperTML, supertml_module,
        ["problem", "transformer", "verbose", "pixels", "feature_importance", "font_size",
         "random_seed"],
        dict(problem=None, verbose=None, pixels=224, feature_importance=False, font_size=10,
             random_seed=1),
    ),
    (
        "FeatureWrap", FeatureWrap, featurewrap_module,
        ["problem", "transformer", "verbose", "size", "bins", "zoom"],
        dict(problem=None, verbose=None, size=(8, 8), bins=10, zoom=1),
    ),
    (
        "REFINED", REFINED, refined_module,
        ["problem", "transformer", "verbose", "hcIterations", "n_processors", "zoom", "format",
         "cmap", "random_seed"],
        dict(problem=None, verbose=None, hcIterations=5, n_processors=8, zoom=1, format="png",
             cmap="viridis", random_seed=1),
    ),
]


def make_features(seed, n_samples, n_features, low, high):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.uniform(low, high, size=(n_samples, n_features)),
        columns=[f"f{j}" for j in range(n_features)],
    )


def make_classification_frame(seed, n_samples, n_features, low, high):
    frame = make_features(seed, n_samples, n_features, low, high)
    frame["label"] = np.arange(n_samples) % 2  # integer labels 0/1, deterministic
    return frame


class DefaultTransformerIsolationTests(unittest.TestCase):
    """T1 - object identity isolation for the 8 patched constructors."""

    def test_default_instances_do_not_share_scaler(self):
        for name, cls, _module, _order, _defaults in PATCHED:
            with self.subTest(model=name):
                a = cls()
                b = cls()
                self.assertIsInstance(a.transformer, MinMaxScaler)
                self.assertIsInstance(b.transformer, MinMaxScaler)
                self.assertIsNot(a.transformer, b.transformer)
                self.assertIsNot(a, b)
                # a never-fitted scaler: no fitted attribute may exist on a new instance
                self.assertFalse(hasattr(a.transformer, "data_min_"))
                self.assertFalse(hasattr(b.transformer, "data_min_"))

    def test_many_default_instances_are_all_distinct(self):
        for name, cls, _module, _order, _defaults in PATCHED:
            with self.subTest(model=name):
                scalers = [cls().transformer for _ in range(5)]
                self.assertEqual(len({id(s) for s in scalers}), 5)

    def test_explicit_none_disables_preprocessing(self):
        for name, cls, _module, _order, _defaults in PATCHED:
            with self.subTest(model=name):
                self.assertIsNone(cls(transformer=None).transformer)

    def test_custom_transformer_is_kept_by_identity_without_cloning(self):
        for name, cls, _module, _order, _defaults in PATCHED:
            for factory in (StandardScaler, RobustScaler):
                with self.subTest(model=name, transformer=factory.__name__):
                    custom = factory()
                    model = cls(transformer=custom)
                    self.assertIs(model.transformer, custom)
                    # R3 must not introduce silent cloning: a custom object that the user
                    # shares between two models is still the very same object.
                    other = cls(transformer=custom)
                    self.assertIs(other.transformer, custom)

    def test_custom_minmax_scaler_with_non_default_range_is_kept(self):
        for name, cls, _module, _order, _defaults in PATCHED:
            with self.subTest(model=name):
                custom = MinMaxScaler(feature_range=(-1, 1))
                model = cls(transformer=custom)
                self.assertIs(model.transformer, custom)
                self.assertEqual(model.transformer.feature_range, (-1, 1))

    def test_module_level_private_sentinel_is_the_signature_default(self):
        """Introspection-observable change (intended in R3, to be reviewed in R4):
        the default in the signature is a module-level private sentinel, not a MinMaxScaler."""
        for name, cls, module, _order, _defaults in PATCHED:
            with self.subTest(model=name):
                sentinel = getattr(module, "_DEFAULT_TRANSFORMER", None)
                self.assertIsNotNone(sentinel, "module must define _DEFAULT_TRANSFORMER")
                self.assertIs(type(sentinel), object)
                default = inspect.signature(cls.__init__).parameters["transformer"].default
                self.assertIs(default, sentinel)
                self.assertNotIsInstance(default, MinMaxScaler)

    def test_instance_state_never_holds_the_sentinel(self):
        """The sentinel must not leak into instance attributes (and hence into pickles)."""
        for name, cls, module, _order, _defaults in PATCHED:
            with self.subTest(model=name):
                sentinel = module._DEFAULT_TRANSFORMER
                model = cls()
                self.assertFalse(any(value is sentinel for value in vars(model).values()))

    def test_parameter_order_is_preserved(self):
        for name, cls, _module, order, _defaults in PATCHED:
            with self.subTest(model=name):
                names = list(inspect.signature(cls.__init__).parameters)
                self.assertEqual(names[0], "self")
                self.assertEqual(names[1:], order)

    def test_other_parameter_defaults_are_preserved(self):
        for name, cls, _module, _order, defaults in PATCHED:
            with self.subTest(model=name):
                params = inspect.signature(cls.__init__).parameters
                for pname, expected in defaults.items():
                    self.assertEqual(params[pname].default, expected, msg=f"{name}.{pname}")


class ApprovedModelsNonRegressionTests(unittest.TestCase):
    """Non-regression of the v1.3.3 sentinel isolation in TINTO, DeepInsight and Fotomics.

    Only constructors are exercised; implementation files are untouched by R3.
    """

    @staticmethod
    def _builders():
        from TINTOlib.deepInsight import DeepInsight
        from TINTOlib.fotomics import Fotomics
        from TINTOlib.tinto import TINTO
        from TINTOlib.utils.LogScaler import LogScaler

        return [
            ("TINTO", lambda **kw: TINTO(**kw), MinMaxScaler),
            ("DeepInsight", lambda **kw: DeepInsight(image_dim=4, **kw), MinMaxScaler),
            ("Fotomics", lambda **kw: Fotomics(image_dim=4, **kw), LogScaler),
        ]

    def test_default_instances_do_not_share_scaler(self):
        for name, build, expected_type in self._builders():
            with self.subTest(model=name):
                a, b = build(), build()
                self.assertIsInstance(a.transformer, expected_type)
                self.assertIsNot(a.transformer, b.transformer)

    def test_explicit_none_and_custom_transformer_are_respected(self):
        for name, build, _expected_type in self._builders():
            with self.subTest(model=name, case="none"):
                self.assertIsNone(build(transformer=None).transformer)
            with self.subTest(model=name, case="custom"):
                custom = StandardScaler()
                self.assertIs(build(transformer=custom).transformer, custom)


class IGTDCrossFitIsolationTests(unittest.TestCase):
    """T2 - IGTD FIT(A) -> FIT(B) -> TRANSFORM(A).

    Note on causality: this test asserts (1) that the scaler *state* of A is unchanged by
    fitting B, and (2) that A's output equals an independent expectation computed with a
    private scaler fitted only on df1.  It demonstrates absence of cross-contamination
    after the patch; it does NOT prove that the shared scaler was the only possible cause
    of any past discrepancy.
    """

    SCALE = [2, 2]
    N_FEATURES = 4
    N_SAMPLES = 16

    def setUp(self):
        self.df1 = make_classification_frame(SEED_DF1, self.N_SAMPLES, self.N_FEATURES, 0.0, 1.0)
        self.df2 = make_classification_frame(SEED_DF2, self.N_SAMPLES, self.N_FEATURES, 100.0, 200.0)

    def test_fit_b_does_not_modify_scaler_state_of_a(self):
        a = IGTD(scale=self.SCALE)
        b = IGTD(scale=self.SCALE)
        self.assertIsNot(a.transformer, b.transformer)

        a.fit(self.df1)
        snapshot = {
            "data_min_": a.transformer.data_min_.copy(),
            "data_max_": a.transformer.data_max_.copy(),
            "scale_": a.transformer.scale_.copy(),
            "n_samples_seen_": int(a.transformer.n_samples_seen_),
        }
        features1 = self.df1.iloc[:, :-1]
        np.testing.assert_allclose(snapshot["data_min_"], features1.min().to_numpy())
        np.testing.assert_allclose(snapshot["data_max_"], features1.max().to_numpy())

        b.fit(self.df2)
        # b's scaler really learnt a different range ([100, 200])
        self.assertTrue(np.all(b.transformer.data_min_ >= 100.0))

        # a's scaler is bit-for-bit what it was before fitting b
        np.testing.assert_array_equal(a.transformer.data_min_, snapshot["data_min_"])
        np.testing.assert_array_equal(a.transformer.data_max_, snapshot["data_max_"])
        np.testing.assert_array_equal(a.transformer.scale_, snapshot["scale_"])
        self.assertEqual(int(a.transformer.n_samples_seen_), snapshot["n_samples_seen_"])

    def test_transform_a_after_fit_b_matches_independent_expectation(self):
        a = IGTD(scale=self.SCALE, format="npy")
        b = IGTD(scale=self.SCALE, format="npy")
        a.fit(self.df1)
        b.fit(self.df2)

        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "out_a")
            a.transform(self.df1, folder)

            # Independent expectation: private scaler fitted only on df1 features, then the
            # same arithmetic as IGTD.__generate_image_data (documented in igtd.py).
            features1 = self.df1.iloc[:, :-1]
            scaled = MinMaxScaler().fit_transform(features1)
            index = a.index[a.min_id, :]
            data_2 = scaled[:, index]
            max_v, min_v = np.max(data_2), np.min(data_2)
            data_2 = 255 - (data_2 - min_v) / (max_v - min_v) * 255

            labels = self.df1["label"].to_numpy()
            for i in range(self.N_SAMPLES):
                expected = np.full(tuple(self.SCALE), np.nan)
                expected[a.coordinate] = data_2[i, :]
                expected[np.isnan(expected)] = 255
                path = os.path.join(folder, str(int(labels[i])).zfill(2), f"{i:06d}.npy")
                self.assertTrue(os.path.isfile(path), msg=path)
                np.testing.assert_allclose(np.load(path), expected, rtol=0, atol=1e-9, err_msg=path)

            # Classification CSV exists and keeps one row per sample
            csv_path = os.path.join(folder, "classification.csv")
            self.assertTrue(os.path.isfile(csv_path))
            self.assertEqual(len(pd.read_csv(csv_path)), self.N_SAMPLES)


class PersistenceTests(unittest.TestCase):
    """T3 - saveHyperparameters / loadHyperparameters with fitted and custom transformers."""

    def setUp(self):
        self.df = make_classification_frame(SEED_PERSIST, 16, 4, 0.0, 10.0)

    def test_fitted_default_scaler_roundtrip_is_isolated_from_new_instances(self):
        a = IGTD(scale=[2, 2], format="npy")
        a.fit(self.df)
        with tempfile.TemporaryDirectory() as tmp:
            pkl = os.path.join(tmp, "igtd.pkl")
            a.saveHyperparameters(pkl)

            b = IGTD(scale=[2, 2], format="npy")
            b.loadHyperparameters(pkl)

            self.assertTrue(b._fitted)
            self.assertIsInstance(b.transformer, MinMaxScaler)
            self.assertIsNot(b.transformer, a.transformer)  # unpickled copy, not an alias
            np.testing.assert_array_equal(b.transformer.data_min_, a.transformer.data_min_)
            np.testing.assert_array_equal(b.transformer.data_max_, a.transformer.data_max_)

            # A brand new default instance must not inherit the loaded/fitted scaler.
            c = IGTD()
            self.assertIsNot(c.transformer, b.transformer)
            self.assertIsNot(c.transformer, a.transformer)
            self.assertFalse(hasattr(c.transformer, "data_min_"))

            # The reloaded model reproduces A's images.
            out_a = os.path.join(tmp, "a")
            out_b = os.path.join(tmp, "b")
            a.transform(self.df, out_a)
            b.transform(self.df, out_b)
            for i in range(len(self.df)):
                label = str(int(self.df["label"].iloc[i])).zfill(2)
                np.testing.assert_array_equal(
                    np.load(os.path.join(out_a, label, f"{i:06d}.npy")),
                    np.load(os.path.join(out_b, label, f"{i:06d}.npy")),
                )

    def test_custom_estimator_roundtrip_keeps_type_and_state(self):
        a = IGTD(scale=[2, 2], transformer=StandardScaler())
        a.fit(self.df)
        with tempfile.TemporaryDirectory() as tmp:
            pkl = os.path.join(tmp, "igtd_custom.pkl")
            a.saveHyperparameters(pkl)
            b = IGTD(scale=[2, 2])
            b.loadHyperparameters(pkl)
        self.assertIsInstance(b.transformer, StandardScaler)
        self.assertIsNot(b.transformer, a.transformer)
        np.testing.assert_array_equal(b.transformer.mean_, a.transformer.mean_)
        np.testing.assert_array_equal(b.transformer.scale_, a.transformer.scale_)

    def test_explicit_none_survives_roundtrip(self):
        a = IGTD(scale=[2, 2], transformer=None)
        a.fit(self.df)
        with tempfile.TemporaryDirectory() as tmp:
            pkl = os.path.join(tmp, "igtd_none.pkl")
            a.saveHyperparameters(pkl)
            b = IGTD(scale=[2, 2])
            self.assertIsInstance(b.transformer, MinMaxScaler)
            b.loadHyperparameters(pkl)
        self.assertIsNone(b.transformer)

    def test_saved_state_does_not_contain_the_sentinel(self):
        a = IGTD(scale=[2, 2])
        a.fit(self.df)
        with tempfile.TemporaryDirectory() as tmp:
            pkl = os.path.join(tmp, "igtd_state.pkl")
            a.saveHyperparameters(pkl)
            with open(pkl, "rb") as handle:
                state = pickle.load(handle)
        # After unpickling, identity with the module-level sentinel is meaningless; a leaked
        # sentinel would instead show up as a bare ``object()`` instance in the saved state.
        self.assertFalse(any(type(value) is object for value in state.values()))
        self.assertIn("transformer", state)


if __name__ == "__main__":
    unittest.main()
