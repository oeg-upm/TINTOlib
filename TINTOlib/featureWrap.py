# Standard library imports
import os
import shutil

# Third-party library imports
import matplotlib.image
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, OneHotEncoder

# Typing imports
from typing import List, Union

# Local application/library imports
from TINTOlib.abstractImageMethod import AbstractImageMethod

###########################################################
################    FeatureWrap    ########################
###########################################################

class FeatureWrap(AbstractImageMethod):
    """
    FeatureWrap: A method that transforms feature values into binary vectors and arranges them into 2D images.

    This method converts features into fixed-length binary strings, representing both numerical and categorical 
    data, which are then arranged to form a 2D image.

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
    size : tuple of int, optional
        The width and height of the final image, in pixels (rows x columns). 
        Default is (8, 8). Valid values: tuple of two positive integers.
    bins : int, optional
        The number of bins or intervals used for grouping numeric data. 
        Default is 10. Valid values: integer > 1.
    zoom : int, optional
        Multiplication factor determining the size of the saved image relative to the original size. 
        Default is 1. Valid values: integer > 0.

    Attributes:
    ----------
    bits_per_pixel : int
        Fixed at 8 bits for binary representation.
    """
    ###### Default values ###############
    default_size = (8, 8)  # Image dimensions (rows x columns)

    default_bins = 10  # Number of bins for grouping numeric data

    default_zoom = 1  # Rescale factor for saving the image        

    def __init__(
        self,
        problem = None,
        transformer=MinMaxScaler(),
        verbose = None,
        size: tuple = default_size,
        bins: int = default_bins,
        zoom: int = default_zoom
    ):
        super().__init__(problem=problem, verbose=verbose, transformer=transformer)

        # Validation for `size`
        try:
            _ = len(size)
            _ = size[0]
        except (TypeError, IndexError, AttributeError):
            raise TypeError(f"`size` must be a tuple (or similar) of with len == 2")
        if len(size) != 2:
            raise ValueError(f"`size` must have a length of 2 (got {len(size)})")
        for elem in size:
            if not isinstance(elem, int):
                raise TypeError(f"The elements in `size` parameter must be of type int (got {type(elem)}).")
            if elem <= 0:
                raise ValueError(f"The elements in `size` must be positive (got {elem})")
        
        # Validation for `bins`
        if not isinstance(bins, int) or bins <= 1:
            raise ValueError("`bins` must be an integer greater than 1.")
        
        # Validation for `zoom`
        if not isinstance(zoom, int) or zoom <= 0:
            raise ValueError("`zoom` must be a positive integer.")

        self.normalize = False # FeatureWrap has normalization implemented in the method

        self.size = tuple(size[::])

        self.bins = bins
        self.bits_per_pixel = 8

        self.zoom = zoom

    def __save_images(self, matrices, y, num_elems):
        for (i,matrix) in enumerate(matrices):
            # Scale the matrix
            matrix = np.repeat(np.repeat(matrix, self.zoom, axis=0), self.zoom, axis=1)
            self._save_image(matrix,y[i],i)

    def _img_to_file(self,image,file):
        matplotlib.image.imsave(file, image, cmap='gray', format='png', dpi=self.zoom, vmin=0, vmax=255)

    def __binary_vector_to_matrix(self, binary_vector):
        # Calculate the total number of pixels in the image
        total_pixels = self.size[0] * self.size[1]
        required_size = total_pixels * self.bits_per_pixel

        if required_size - binary_vector.shape[0] < 0:
            raise Exception("the current size is too small for the amount of embeddings. Consider increasing `size` or reducing the value of `bins`.")

        # Pad the vector with zeros
        padded_vector = np.pad(
            array = binary_vector,
            pad_width = (0, required_size - binary_vector.shape[0]),
            mode = 'constant',
            constant_values = 0   
        )
        # Group the bits to form bytes
        grouped_elems = padded_vector.reshape(-1, self.bits_per_pixel)
        # Transform the group of bits into numbers
        numbers = grouped_elems.dot(2**np.arange(self.bits_per_pixel)[::-1])
        # Reshape the numbers into the desired shape
        matrix = np.array(numbers).reshape(self.size)
        return matrix

    def __preprocess_samples(self, X: pd.DataFrame, is_categorical: List[bool]):
        # Convert original data into the binary space
        categorical_ohe = OneHotEncoder(sparse_output=False)
        numerical_scaler = MinMaxScaler()
        numerical_discretizer = KBinsDiscretizer(n_bins=self.bins, encode='ordinal', strategy='uniform', subsample=None)
        numerical_ohe = OneHotEncoder(sparse_output=False, categories=[range(self.bins)])

        binary_vectors = []
        for column_name,is_caterical_column in zip(X.columns, is_categorical):
            column_values = X[column_name].to_numpy().reshape(-1, 1)
            if is_caterical_column:
                # Convert categorical into one-hot
                one_hot_column_values = categorical_ohe.fit_transform(column_values)
            else:
                # Scale the values of in the column
                scaled_data = numerical_scaler.fit_transform(column_values)
                # # Discretize into self.bins intervals
                discretized_data = numerical_discretizer.fit_transform(scaled_data)
                # Transform discretized intervals into one-hot encodings
                one_hot_column_values = numerical_ohe.fit_transform(discretized_data)

            binary_vectors.append(one_hot_column_values)
        binary_vectors = np.hstack(binary_vectors)

        # Turn binary vectors into images
        matrices = map(self.__binary_vector_to_matrix, binary_vectors)
        return matrices
    
    def _fitAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        """
        Fit method for stateless transformers. Does nothing and returns self.
        """
        return self
    
    def _transformAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        X = x.values 
        y = y.values if y is not None else None
        
        is_categorical = [pd.api.types.is_string_dtype(x[col]) for col in x]
        matrices = self.__preprocess_samples(x, is_categorical)
        self.__save_images(matrices, y, num_elems=x.shape[0])
