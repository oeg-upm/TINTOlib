# Standard library imports
import pickle
from abc import ABC, abstractmethod

# Third-party library imports
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Typing imports
from typing import Optional, Union

# Default configuration values
default_problem = "supervised"  # Define the type of task [supervised, unsupervised, regression]
default_verbose = False         # Verbose: if True, shows the compilation text
default_normalize = True
default_hyperparameters_filename = 'objs.pkl'

class AbstractImageMethod(ABC):
    """
    Abstract class that other classes must inherit from and implement abstract methods.
    Provides utility methods for saving/loading hyperparameters and data transformations.
    """
    def __init__(
        self,
        problem: Optional[str], 
        verbose: Optional[bool],
        normalize: Optional[bool],
    ):
        # Validate `problem`
        if problem is None:
            problem = default_problem
        allowed_values_for_problem = ["supervised", "unsupervised", "regression"]
        if not isinstance(problem, str):
            raise TypeError(f"problem must be of type str (got {type(problem)})")
        if problem not in allowed_values_for_problem:
            raise ValueError(f"Allowed values for problem are {allowed_values_for_problem}. Instead got {problem}")
        
        # Validate `verbose`
        if verbose is None:
            verbose = default_verbose
        if not isinstance(verbose, bool):
            raise TypeError(f"verbose must be of type bool (got {type(verbose)})")
        
        # Validate `normalize`
        if normalize is None:
            normalize = default_normalize
        if not isinstance(normalize, bool):
            raise TypeError(f"verbose must be of type bool (got {type(normalize)})")

        self.problem = problem
        self.verbose = verbose
        self._fitted = False  # Tracks if fit has been called

        # Normalize data
        self.normalize = normalize  # Whether to normalize data
        self.scaler = MinMaxScaler() if normalize else None  # Initialize scaler if needed


    def saveHyperparameters(self, filename=default_hyperparameters_filename):
        """
        This function allows SAVING the transformation options to images in a Pickle object.
        This point is basically to be able to reproduce the experiments or reuse the transformation
        on unlabelled data.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        if self.verbose:
            print(f"Hyperparameters successfully saved in {filename}.")

    def loadHyperparameters(self, filename=default_hyperparameters_filename):
        """
        This function allows LOADING the transformation options to images from a Pickle object.
        This point is basically to be able to reproduce the experiments or reuse the transformation
        on unlabelled data.
        """
        with open(filename, 'rb') as f:
            variables = pickle.load(f)
        
        for key, val in variables.items():
            setattr(self, key, val)

        if self.verbose:
            print(f"Hyperparameters successfully loaded from {filename}.")
        
    def fit(self, data):
        """
        Fits the model to the tabular data.

        Parameters:
        - data: Path to CSV file or a pandas DataFrame containing data and targets.
        """

        dataset = self._load_data(data)
        x, y = self._split_features_targets(dataset)

        # Normalize features if required
        if self.normalize:
            x = pd.DataFrame(self.scaler.fit_transform(x), columns=x.columns)

        # Call the training function
        self._fitAlg(x, y)

        self._fitted = True  # Mark as fitted

        if self.verbose:
            print("Fit process completed.")

    def transform(self, data, folder):
        """
        Generate and saves the synthetic images in the specified folder.
        
        Parameters:
        - data: Path to CSV file or a pandas DataFrame containing data and targets.
        - folder: Path to folder where the images will be saved.
        """
        if not self._fitted:
            raise RuntimeError("The model must be fitted before calling 'transform'. Please call 'fit' first.")

        dataset = self._load_data(data)
        x, y = self._split_features_targets(dataset)

        # Normalize features if required
        if self.normalize:
            x = pd.DataFrame(self.scaler.transform(x), columns=x.columns)

        self.folder = folder
        self._transformAlg(x, y)

        if self.verbose:
            print("Transform process completed.")

    def fit_transform(self, data, folder):
        """
        Fits the model to the tabular data and then generate and saves the synthetic images in the specified folder.
        
        Parameters:
        - data: Path to CSV file or a pandas DataFrame containing data and targets.
        - folder: Path to folder where the images will be saved.
        """
        dataset = self._load_data(data)
        x, y = self._split_features_targets(dataset)

        # Normalize features if required
        if self.normalize:
            x = pd.DataFrame(self.scaler.fit_transform(x), columns=x.columns)

        self.folder = folder
        self._fitAlg(x, y)
        self._transformAlg(x, y)
        self._fitted = True  # Mark as fitted after both operations
    
        if self.verbose:
            print("Fit-Transform process completed.")

    def _load_data(self, data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Loads data from a file or returns the DataFrame directly.
        """
        if isinstance(data, str):
            dataset = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            dataset = data
        else:
            raise TypeError("data must be a string (file path) or a pandas DataFrame.")

        if self.verbose:
            print("Data successfully loaded.")

        return dataset

    def _split_features_targets(self, dataset: pd.DataFrame):
        """
        Splits dataset into features and targets based on the problem type.
        """
        if self.problem in ["supervised", "regression"]:
            x = dataset.drop(columns=dataset.columns[-1])
            y = dataset[dataset.columns[-1]]
        else:
            x = dataset
            y = None

        return x, y

    @abstractmethod
    def _fitAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        """
        Abstract method for fitting the algorithm. Must be implemented by subclasses.
        This method is not to be called from the outside.
        """
        raise NotImplementedError("Subclasses must implement _fit_alg.")

    @abstractmethod
    def _transformAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        """
        Abstract method for transforming the data. Must be implemented by subclasses.
        This method is not to be called from the outside.
        """
        raise NotImplementedError("Subclasses must implement _transform_alg.")
    