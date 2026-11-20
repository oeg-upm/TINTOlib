# Standard library imports
import os
import shutil

# Third-party library imports
import bitstring
import matplotlib
import matplotlib.image
import numpy as np
import pandas as pd

# Typing imports
from typing import Iterator, List, Union

from sklearn.preprocessing import MinMaxScaler

# Local application/library imports
from TINTOlib.abstractImageMethod import AbstractImageMethod

###########################################################
################    Binary Image Encoding    ##############
###########################################################

default_precision = 32
default_zoom = 1

_DEFAULT_TRANSFORMER = object()

class BIE(AbstractImageMethod):
    """
    BIE: Generates 1-channel images by encoding the floating-point representation of numeric values.

    Each feature's value is converted to its binary representation, which is then used to create rows in the image. 
    This method preserves precise numeric information but may not effectively capture relationships between features.

    Parameters:
    ----------
    problem : str, optional
        The type of problem, defining how the images are grouped. 
        Default is 'classification'. Valid values: ['classification', 'unsupervised', 'regression'].
    transformer : CustomTransformer, optional
        Preprocessing transformations like scaling, normalization,etc.
        Default is MinMaxScaler.
        Valid: Scikit Learn transformers or custom implementation using inheritance over CustomTransformer class.
    verbose : bool, optional
        Show execution details in the terminal. 
        Default is False. Valid values: [True, False].
    precision : int, optional
        Determines the precision of the binary encoding. 
        Default is 32. Valid values: [32, 64].
    zoom : int, optional
        Multiplication factor determining the size of the saved image relative to the original size. 
        Default is 1. Valid values: integer > 0.
    correlated : bool, optional
        If True (default), learn a Pearson/greedy correlated feature order for supervised
        BIE. If False, preserve the original BIE feature order. In unsupervised
        mode the original input feature order is always used.
    """
    ###### Default values ###############
    default_precision = 32  # Precision for binary encoding
    default_zoom = 1  # Rescale factor for saving the image

    def __init__(
        self,
        problem = None,
        transformer=_DEFAULT_TRANSFORMER,
        verbose = None,
        precision: int = default_precision,
        zoom: int = default_zoom,
        correlated: bool = True
    ):
        if transformer is _DEFAULT_TRANSFORMER:
            transformer = MinMaxScaler()

        super().__init__(problem=problem, verbose=verbose, transformer=transformer)

        if not isinstance(precision, int):
            raise TypeError(f"precision must be of type int (got {type(precision)})")
        configurable_precisions = [32, 64]
        if precision not in configurable_precisions:
            raise ValueError(f"precision must have one of this values {configurable_precisions}. Instead, got {precision}")
        
        if not isinstance(zoom, int):
            raise TypeError(f"zoom must be of type int (got {type(zoom)})")
        if zoom <= 0:
            raise ValueError(f"zoom must be positive. Instead, got {zoom}")
        
        if not isinstance(correlated, bool):
            raise TypeError(f"correlated must be of type bool (got {type(correlated)})")

        self.correlated = correlated
        self.precision = precision

        self.zoom = zoom
        
        self.ones, self.zeros = 255, 0

        self.__columns_sorted = []

    def __convert_samples_to_binary(self, data: np.ndarray) -> Iterator[List[List[int]]]:
        def process_sample(sample):
            return [[self.ones if b=='1' else self.zeros for b in bitstring.BitArray(float=feat, length=self.precision).bin] for feat in sample]
        return map(process_sample, data)

    def __save_images(self, matrices: Iterator[List[List[int]]], y, num_elems):
        for (i,matrix) in enumerate(matrices):
            # Scale the matrix
            matrix = np.repeat(np.repeat(matrix, self.zoom, axis=0), self.zoom, axis=1)
            target = None if y is None else y[i]
            self._save_image(matrix,target,i)

    def _fitAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        """Learn Pearson/greedy order only when correlated=True and a target exists."""
        self.__columns_sorted = []
        if self.correlated and y is not None:
            # Match each transformed feature row to its target by position.
            df = pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
            corr = df.corr().abs().to_numpy()
            n_features = x.shape[1]
            # Pearson can be undefined for constant features or a constant target
            # even when the original input contains no NaNs. Never impute values.
            if not np.isfinite(corr).all():
                raise ValueError(
                    "cBIE requires finite, defined Pearson correlations during fit; "
                    "check constant training features or target and preprocess them "
                    "before fitting. No input values were replaced."
                )
            # Javier PR #30: original Pearson/greedy ordering for defined inputs.
            # After concat the target is the last column
            target_col = n_features
            # Start from the feature most correlated with the target
            order = [int(np.argmax(corr[:n_features, target_col]))]
            visited = np.zeros(n_features, dtype=bool)
            visited[order[0]] = True
            feat_corr = corr[:n_features, :n_features]
            for _ in range(n_features - 1):
                last = order[-1]
                cand = feat_corr[:, last].copy()
                cand[visited] = -1.0
                order.append(int(np.argmax(cand)))
                visited[order[-1]] = True
            self.__columns_sorted = [x.columns[i] for i in order]
        return self

    def _transformAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        if self.correlated and len(self.__columns_sorted) > 0:
            x = x[self.__columns_sorted]
        x = x.values
        y = y.values if y is not None else None

        matrices = self.__convert_samples_to_binary(x)
        self.__save_images(matrices, y, num_elems=x.shape[0])

    def _img_to_file(self, image_matrix, file):
        matplotlib.image.imsave(file, image_matrix, cmap='gray', format='png', dpi=self.zoom, vmin=0, vmax=1)
