from abc import abstractmethod

from sklearn.base import BaseEstimator,TransformerMixin


class CustomTransformer(BaseEstimator, TransformerMixin):
    """
    Abstract class implementing a custom transformer following scikit-learn structure.
    """
    @abstractmethod
    def fit(self, x, y=None):
        raise NotImplementedError("Custom Transformers must implement fit.")

    @abstractmethod
    def transform(self, x, y=None):
        raise NotImplementedError("Custom Transformers must implement transform.")
