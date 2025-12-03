"""

"""
import numpy as np

from TINTOlib.utils.CustomTransformer import CustomTransformer


class LogScaler(CustomTransformer):
    """
    This class implements a custom transformer following scikit-learn structure. Extends from CustomTransformer class
    and implements the fit and transform methods. Perform a ln transformation and normalize to range [0,1].
    """
    def __init__(self):
        self.__min_by_feature=None
        self.__max_by_feature_normalized=None
        self.__fitted=False

    def fit(self, X,y=None):
        """
        Calculate the minimum value by feature and de maximum value of the log transformed matrix
        Args:
            X: Independent variables
            y: Class variable

        Returns:

        """

        self.__min_by_feature = X.min(axis=0)
        self.__x_normalized = np.log(X+np.abs(self.__min_by_feature)+1)
        self.__max_by_feature_normalized = np.max(self.__x_normalized)

        self.__fitted = True
        return self

    def transform(self, X,y=None):
        """
        Normalized independent variables data
        Args:
            X: Independent variables
            y: Class variable

        Returns:

        """
        if(self.__fitted == False):
            raise ValueError(f"Fit process is mandatory before transform")


        if(X.shape[1] != self.__min_by_feature.shape[0]):
            raise ValueError(f"The number of features not match with fitting set")

        x_norm_clipped = self.__x_normalized.clip(0,None)
        return (x_norm_clipped / self.__max_by_feature_normalized).clip(0,1)
