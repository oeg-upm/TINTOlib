import importlib.util
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.exceptions import ConvergenceWarning

from TINTOlib.clusters import Clusters


class ClustersRBFTests(unittest.TestCase):
    @staticmethod
    def parameters(algorithm, n_clusters=2):
        params = dict(
            algorithm=algorithm, n_clusters=n_clusters,
            RBFMethod=True, random_seed=42, max_iter=20,
        )
        if algorithm in ('kmeans', 'mixMethod'):
            params['n_init'] = 2
        if algorithm == 'mixMethod':
            params['ensamMethod'] = ['kmeans', 'kmedoids']
        return params

    def test_zero_bandwidth_uses_gaussian_limit(self):
        distances = np.array([[0.0, 1.0], [2.0, 0.0]])
        with np.errstate(divide='raise', invalid='raise'):
            actual = Clusters._rbf_transform(distances, 0.0)
        np.testing.assert_array_equal(actual, [[1.0, 0.0], [0.0, 1.0]])

    def test_positive_bandwidth_preserves_existing_values(self):
        distances = np.array([[0.0, 0.5, 2.0], [1.0, 3.0, 4.0]])
        sigma = np.mean(distances)
        expected = np.exp(-(distances**2) / (2 * sigma**2))
        np.testing.assert_array_equal(
            Clusters._rbf_transform(distances, sigma), expected,
        )

    def check_constant_rows(self, algorithm, n_clusters=2):
        data = pd.DataFrame(np.ones((16, 3)), columns=list('abc'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = Clusters(
                problem='unsupervised', **self.parameters(algorithm, n_clusters),
            )
            with warnings.catch_warnings(), np.errstate(divide='raise', invalid='raise'):
                # Identical data necessarily has fewer distinct clusters than requested.
                warnings.simplefilter('ignore', ConvergenceWarning)
                model.fit_transform(data, str(root / 'first'))
                scalers = model.minmaxList if algorithm == 'mixMethod' else [model.minmax]
                for scaler in scalers:
                    self.assertTrue(np.isfinite(scaler.data_min_).all())
                    self.assertTrue(np.isfinite(scaler.data_max_).all())
                    np.testing.assert_array_equal(scaler.data_min_, 1.0)
                    np.testing.assert_array_equal(scaler.data_max_, 1.0)

                filename = root / 'model.pkl'
                model.saveHyperparameters(str(filename))
                restored = Clusters(problem='unsupervised', **self.parameters(algorithm))
                restored.loadHyperparameters(str(filename))
                restored.transform(data, str(root / 'restored'))

            first = sorted((root / 'first').rglob('*.png'))
            self.assertEqual(len(first), len(data))
            for path in first:
                other = root / 'restored' / path.relative_to(root / 'first')
                with Image.open(path) as left, Image.open(other) as right:
                    pixels = np.array(left)
                    np.testing.assert_array_equal(pixels, np.array(right))
                    np.testing.assert_array_equal(pixels, 0)
            self.assertEqual(
                (root / 'first' / 'unsupervised.csv').read_bytes(),
                (root / 'restored' / 'unsupervised.csv').read_bytes(),
            )

    def test_identical_rows_fit_transform_and_round_trip(self):
        for algorithm in ('kmeans', 'kmedoids', 'mixMethod'):
            with self.subTest(algorithm=algorithm):
                self.check_constant_rows(algorithm)

    @unittest.skipUnless(importlib.util.find_spec('skimage'), 'requires scikit-image')
    def test_identical_rows_during_cluster_selection(self):
        for algorithm in ('kmeans', 'kmedoids', 'mixMethod'):
            with self.subTest(algorithm=algorithm):
                self.check_constant_rows(algorithm, n_clusters=[9])


if __name__ == '__main__':
    unittest.main()
