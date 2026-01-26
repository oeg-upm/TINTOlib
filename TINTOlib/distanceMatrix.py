# Standard library imports
import os

# Third-party library imports
import numpy as np
import pandas as pd
from PIL import Image

# Typing imports
from typing import Union

from sklearn.preprocessing import MinMaxScaler

# Local application/library imports
from TINTOlib.abstractImageMethod import AbstractImageMethod

###########################################################
################    Distance Matrix    ####################
###########################################################

class DistanceMatrix(AbstractImageMethod):
    """
    DistanceMatrix: Represents a distance matrix of all normalized variables within the range [0, 1].

    This method constructs a distance matrix for the given data and represents it as an image. 
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
    default_zoom = 1  # Rescale factor for saving the image              

    def __init__(
        self,
        problem = None,
        transformer=MinMaxScaler(),
        verbose = None,
        zoom: int = default_zoom,
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
        N,d=X.shape

        #Create matrix (only once, then reuse it)
        imgI = np.empty((d,d))

        #For each instance
        for ins,dataInstance in enumerate(X):
            for i in range(d):
                for j in range(d):
                    imgI[i][j] = dataInstance[i]-dataInstance[j]

            #Normalize matrix
            image_norm = (imgI - np.min(imgI)) / (np.max(imgI) - np.min(imgI))
            image = np.repeat(np.repeat(image_norm, self.zoom, axis=0), self.zoom, axis=1)

            self._save_image(image,Y[ins],ins)