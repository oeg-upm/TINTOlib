# Standard library imports
import os
import shutil

# Third-party library imports
import numpy as np
import pandas as pd
from PIL import Image

# Typing imports
from typing import Iterator, List, Union

from sklearn.preprocessing import MinMaxScaler

# Local application/library imports
from TINTOlib.abstractImageMethod import AbstractImageMethod

###########################################################
################    Combination    ########################
###########################################################

class Combination(AbstractImageMethod):
    """
    Combination: Combines the Distance Matrix and BarGraph representations.

    This method generates a hybrid visualization by combining the outputs of the 
    Distance Matrix and BarGraph methods into a single image representation.

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
    zoom : int, optional
        Multiplication factor determining the size of the saved image relative to the original size. 
        Default is 1. Valid values: integer > 0.
    """
    ###### Default values ###############
    default_zoom = 1  # Rescale factor for saving the image

    def __init__(
        self,
        problem = None,
        transformer=MinMaxScaler(),
        verbose = None,
        zoom=default_zoom,
    ):
        super().__init__(problem=problem, verbose=verbose, transformer=transformer)

        self.zoom = zoom

    def _img_to_file(self, image_matrix, file):
        img = Image.fromarray(np.uint8(np.squeeze(image_matrix) * 255))
        img.save(file)

    
    def _fitAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        """
        Fit method for stateless transformers. Does nothing and returns self.
        """
        return self
    
    def _transformAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        X = x.values
        Y = y.values if y is not None else None

        imagesRoutesArr = []

        n_columns = X.shape[1]
        # Matrix for level 1 calculations
        imgI = np.empty((n_columns, n_columns))

        pixel_width, gap = 1, 0
        assert (pixel_width*n_columns + (n_columns+1)*gap) == n_columns
        top_padding, bottom_padding = pixel_width, pixel_width
        max_bar_height = n_columns - (bottom_padding + top_padding)
        step_column = gap + pixel_width

        # Matrix for level 2 calculations
        imgage = np.zeros([n_columns, n_columns, 1])

        # Matrix for level 3 calculations
        lvl3img = np.empty((n_columns, n_columns))

        # for each instance
        for ins,dataInstance in enumerate(X):
            """LEVEL - 1 (MATRIX)"""
            for i in range(n_columns):
                for j in range(n_columns):
                    imgI[i][j] = dataInstance[i] - dataInstance[j]
            # Normalize matrix
            image_norm = (imgI - np.min(imgI)) / (np.max(imgI) - np.min(imgI))
            # Apply zoom
            imgI1 = np.repeat(np.repeat(image_norm, self.zoom, axis=0), self.zoom, axis=1)

            """LEVEL - 2 (BARS)"""
            # TODO: reorder the columns
            imgage = np.zeros([n_columns, n_columns, 1])
            bar_heights = np.floor(X[ins] * max_bar_height).astype(np.int64)
            for i_bar,val_bar in enumerate(bar_heights):
                imgage[
                    top_padding : (top_padding + val_bar),                              # The height of the column
                    (gap+(step_column*i_bar)) : (gap+(step_column*i_bar)) + pixel_width # The width of the column
                ] = 1
            # Apply zoom
            imgI2 = np.repeat(np.repeat(imgage, self.zoom, axis=0), self.zoom, axis=1)

            """LEVEL - 3"""
            # Fill the matrix
            lvl3img[:, :] = dataInstance[:, np.newaxis]
            # Normalize the matrix
            lvl3 = (lvl3img - np.min(lvl3img)) / (np.max(lvl3img) - np.min(lvl3img))
            # Apply zoom
            imgI3 = np.repeat(np.repeat(lvl3, self.zoom, axis=0), self.zoom, axis=1)

            # Combine ALL img
            imgFinal = np.dstack((imgI1,imgI2,imgI3))

            self._save_image(imgFinal,Y[ins],ins)
