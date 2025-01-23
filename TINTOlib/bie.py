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

# Local application/library imports
from TINTOlib.abstractImageMethod import AbstractImageMethod

###########################################################
################    Binary Image Encoding    ##############
###########################################################

default_precision = 32
default_zoom = 1

class BIE(AbstractImageMethod):
    """
    BIE: Generates 1-channel images by encoding the floating-point representation of numeric values.

    Each feature's value is converted to its binary representation, which is then used to create rows in the image. 
    This method preserves precise numeric information but may not effectively capture relationships between features.

    Parameters:
    ----------
    problem : str, optional
        The type of problem, defining how the images are grouped. 
        Default is 'supervised'. Valid values: ['supervised', 'unsupervised', 'regression'].
    normalize : bool, optional
        If True, normalizes input data using MinMaxScaler. 
        Default is True. Valid values: [True, False].
    verbose : bool, optional
        Show execution details in the terminal. 
        Default is False. Valid values: [True, False].
    precision : int, optional
        Determines the precision of the binary encoding. 
        Default is 32. Valid values: [32, 64].
    zoom : int, optional
        Multiplication factor determining the size of the saved image relative to the original size. 
        Default is 1. Valid values: integer > 0.
    """
    ###### Default values ###############
    default_precision = 32  # Precision for binary encoding
    default_zoom = 1  # Rescale factor for saving the image

    def __init__(
        self,
        problem = None,
        normalize=None,
        verbose = None,
        precision: int = default_precision,
        zoom: int = default_zoom
    ):
        super().__init__(problem=problem, verbose=verbose, normalize=normalize)

        if not isinstance(precision, int):
            raise TypeError(f"precision must be of type int (got {type(precision)})")
        configurable_precisions = [32, 64]
        if precision not in configurable_precisions:
            raise ValueError(f"precision must have one of this values {configurable_precisions}. Instead, got {precision}")
        
        if not isinstance(zoom, int):
            raise TypeError(f"zoom must be of type int (got {type(zoom)})")
        if zoom <= 0:
            raise ValueError(f"zoom must be positive. Instead, got {zoom}")
        
        self.precision = precision

        self.zoom = zoom
        
        self.ones, self.zeros = 255, 0

    def __convert_samples_to_binary(self, data: np.ndarray) -> Iterator[List[List[int]]]:
        def process_sample(sample):
            return [[self.ones if b=='1' else self.zeros for b in bitstring.BitArray(float=feat, length=self.precision).bin] for feat in sample]
        return map(process_sample, data)
    
    def __saveSupervised(self, matrix: np.ndarray, i: int, y):
        extension = 'png'
        subfolder = str(int(y)).zfill(2)  # subfolder for grouping the results of each class
        name_image = str(i).zfill(6)
        route = os.path.join(self.folder, subfolder)
        route_complete = os.path.join(route, name_image + '.' + extension)

        if not os.path.exists(route):
            os.makedirs(route)

        matplotlib.image.imsave(route_complete, matrix, cmap='gray', format=extension, dpi=self.zoom, vmin=0, vmax=1)

        route_relative = os.path.join(subfolder, name_image+ '.' + extension)
        return route_relative

    def __saveRegressionOrUnsupervised(self, matrix: np.ndarray, i: int):
        extension = 'png'  # eps o pdf
        subfolder = "images"
        name_image = str(i).zfill(6) + '.' + extension
        route = os.path.join(self.folder, subfolder)
        route_complete = os.path.join(route, name_image)

        if not os.path.exists(route):
            os.makedirs(route)

        matplotlib.image.imsave(route_complete, matrix, cmap='gray', format=extension, dpi=self.zoom, vmin=0, vmax=1)

        route_relative = os.path.join(subfolder, name_image)
        return route_relative

    def __save_images(self, matrices: Iterator[List[List[int]]], y, num_elems):
        imagesRoutesArr=[]

        if os.path.exists(self.folder):
            shutil.rmtree(self.folder)
        os.makedirs(self.folder)

        for (i,matrix) in enumerate(matrices):
            # Scale the matrix
            matrix = np.repeat(np.repeat(matrix, self.zoom, axis=0), self.zoom, axis=1)

            if self.problem == "supervised":
                route=self.__saveSupervised(matrix, i, y[i])
            elif self.problem == "unsupervised" or self.problem == "regression":
                route = self.__saveRegressionOrUnsupervised(matrix, i)
        
            imagesRoutesArr.append(route)

            if self.verbose:
                print("Created ", str(i + 1), "/", int(num_elems))
        
        if self.problem == "supervised":
            data = {'images': imagesRoutesArr, 'class': y}
            supervisedCSV = pd.DataFrame(data=data)
            supervisedCSV.to_csv(self.folder + "/supervised.csv", index=False)
        elif self.problem == "unsupervised":
            data = {'images': imagesRoutesArr}
            unsupervisedCSV = pd.DataFrame(data=data)
            unsupervisedCSV.to_csv(self.folder + "/unsupervised.csv", index=False)
        elif self.problem == "regression":
            data = {'images': imagesRoutesArr, 'values': y}
            regressionCSV = pd.DataFrame(data=data)
            regressionCSV.to_csv(self.folder + "/regression.csv", index=False)

    def _fitAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        """
        Fit method for stateless transformers. Does nothing and returns self.
        """
        return self
    
    def _transformAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        x = x.values
        y = y.values if y is not None else None

        matrices = self.__convert_samples_to_binary(x)
        self.__save_images(matrices, y, num_elems=x.shape[0])
