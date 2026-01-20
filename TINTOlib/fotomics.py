import numpy as np
from TINTOlib.abstractImageMethod import AbstractImageMethod
from TINTOlib.utils.LogScaler import LogScaler
from TINTOlib.utils.geometry import get_minimum_rectangle
from TINTOlib.paramImageMethod import ParamImageMethod
import TINTOlib.utils.constants as constants

###########################################################
################    Fotomics    ##############################
###########################################################

default_random_seed = 1,
default_zoom=1
default_cmap = 'gray'  # Default cmap image output
outliers_treatment_allowed=[constants.zero_option,constants.max_min_option,constants.avg_option]

class Fotomics(ParamImageMethod):
    """
    This class implements Fotomics process to transform tidy data in synthetic images using fourier transform. The method was published in the paper:
    Fotomics: fourier transform‑based omics imagification for deep learning‑based cell‑identity mapping using single‑cell omics profiles.
    The original implementations GitHub repository is https://github.com/VafaeeLab/Fotomics-Imagification/blob/main/Fotomics.py
    This implementation extend the original behavior allowing diferent methods to mapping features and pixels, to treat outliers and to calculate pixel values.

    Parameters:
    ----------
    image_dim : int
        Order size for a square matrix image
    problem : str, optional
        The type of problem, defining how the images are grouped.
        Default is 'classification'. Valid values: ['classification', 'unsupervised', 'regression'].
   transformer : CustomTransformer, optional
        Preprocessing transformations like scaling, normalization,etc.
        Default is LogScaler custom transformer.
        Valid: Scikit Learn transformers or custom implementation using inheritance over CustomTransformer class.
    verbose : bool, optional
        Show execution details in the terminal.
        Default is False. Valid values: [True, False].
    outliers : bool, optional
        Treat outliers after fourier transform. Default is True.
    min_percentile : int, optional
        Minimum percentile for outlier detection. Default is 10.
    max_percentile : int, optional
        Maximum percentile for outlier detection. Default is 90.
    outliers_treatment: str, optional
        Using to apply different techniques to treat outlier. Default is avg
    assignment_method : str, optional
        Using to apply different techniques to mapping features with pixels. Default is 'bin'.
    relocate: bool, optional
        Relocate features so that each pixel can represent a single feature. Use the relevance of features and pixels shift to build a cost function
    algorithm_opt : str, optional
        Optimization algorithm that could be apply in pixels assignment stage.
    group_method : str, optional
        Using to apply different techniques to calculate pixels values that share multiples features. Default is 'avg'.
    zoom : int, optional
        Multiplication factor determining the size of the saved image relative to the original size.
        Default is 1. Valid values: integer > 0.
    format : str, optional
        Output format using images with matplotlib with [0,255] range for pixel or using npy format.
        Default is images with format 'png'.
    cmap : str, optional
        color map to use with matplotlib.
        Default is gray
    random_seed : int, optional
        Seed for reproducibility.
        Default is 1. Valid values: integer.
    """

    def __init__(self,
            image_dim,
            problem=None,
            transformer=LogScaler(),
            verbose=False,
            outliers=True,
            min_percentile=10,
            max_percentile=90,
            outliers_treatment=constants.zero_option,
            assignment_method=constants.bin_digitize_assigner,
            relocate=False,
            algorithm_opt=constants.linear_sum_assigner,
            group_method=constants.avg_option,
            zoom=default_zoom,
            format=constants.png_format,
            cmap=default_cmap,
            random_seed=default_random_seed
            ):
        super().__init__(image_dim,problem,transformer,verbose,assignment_method,relocate,algorithm_opt,group_method,zoom,format,cmap,random_seed)
        self.__min_percentile = min_percentile
        self.__max_percentile = max_percentile
        self.__outliers = outliers

        if (outliers_treatment not in outliers_treatment_allowed):
            raise ValueError(f"Type of outliers treatment must be in {outliers_treatment_allowed}")
        else:
            self.__outliers_treatment = outliers_treatment

    def __fourier_transform(self,X):
        """
        This method computes the Fourier Transform of an input set. Then we average the real and imaginary values across all samples by feature.
        Finally return a matrix with averages values by feature as a features coordinates.
        Args:
            X: Input set

        Returns:
            features_coord: Features coordinates matrix
        """
        # Apply fourier transform and shift
        x_fourier = np.fft.fftshift(np.fft.fft(X.to_numpy(), axis=0), axes=0)

        #Split real and imaginary part to different matrices
        x_real = np.real(x_fourier)
        x_imaginary = np.imag(x_fourier)

        #Clean outliers
        if(self.__outliers):
            x_real=self.__clean_outliers(x_real)
            x_imaginary=self.__clean_outliers(x_imaginary)

        #Averaging real and imaginary part across all samples by feature
        x_real_means = np.mean(x_real, axis=1)
        x_imaginary_means = np.mean(x_imaginary, axis=1)

        #Create features coordinate matrix
        features_coord = np.column_stack((x_real_means, x_imaginary_means))
        return features_coord



    def __clean_outliers(self,x):
        """
        This method adjust outliers using avg, max/min or zero substitutions methods for features coordinates
        Args:
            x: features real/imaginary values

        Returns:
            features real/imaginary values with outliers adjusted
        """
        percentiles = np.percentile(x,[self.__min_percentile,self.__max_percentile],axis=0).T
        iqrs = (percentiles[:, 1] - percentiles[:, 0])
        iqrs = iqrs * 1.5
        lower_limits = (percentiles[:, 0] - iqrs).reshape(-1, 1)
        upper_limits = (percentiles[:, 1] + iqrs).reshape(-1, 1)
        x_t = x.T
        match self.__outliers_treatment:
            case constants.avg_option:
                subs = np.tile(np.mean(x, axis=0), (x.shape[0], 1)).T
                x_t = np.where(x_t >= lower_limits, x_t, subs)
                x_t = np.where(x_t <= upper_limits, x_t, subs)
            case constants.max_min_option:
                max = np.tile(lower_limits.T, (x.shape[0], 1)).T
                min = np.tile(upper_limits.T, (x.shape[0], 1)).T
                x_t = np.where(x_t >= lower_limits, x_t, max)
                x_t = np.where(x_t <= upper_limits, x_t, min)
            case constants.zero_option:
                x_t = np.where(x_t >= lower_limits, x_t, 0)
                x_t = np.where(x_t <= upper_limits, x_t, 0)
        return x_t.T

    def _get_features_coords(self,x):
        """
        This method computes the features coordinates matrix
        Args:
            x: features tabular data

        Returns:
            features coordinate matrix
        """
        features_coord = self.__fourier_transform(x)
        # Apply Convex Hull algorithm to get limit points, calculate mimimun rectangle area and rotate
        rotmat, rect_coords, limit_points = get_minimum_rectangle(features_coord)
        return np.dot(rotmat, features_coord.T).T

    def _compute_relevance(self, x,features_coord):
        """

        Args:
            x: features dataset not transposed
            features_coord: features coordinates retrieved using a features extraction method
        Returns:
            Array that contains features relevance
        """
        return np.linalg.norm(features_coord, axis=1).reshape(-1, 1) + 1




