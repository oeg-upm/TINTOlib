from abc import ABC, abstractmethod
import pickle
from typing import Optional, Union
import pandas as pd

default_problem = "supervised"  # Define the type of dataset [supervised, unsupervised, regression]
default_verbose = False         # Verbose: if it's true, show the compilation text
default_hyperparameters_filename = 'objs.pkl'

class AbstractImageMethod(ABC):
    """Abstract class that all the other classes must inherit from and implement the abstract functions"""

    def __init__(
        self,
        problem: Optional[str], 
        verbose: Optional[bool],
    ):
        if problem is None:
            problem = default_problem
        if not isinstance(problem, str):
            raise TypeError(f"problem must be of type str (got {type(problem)})")
        allowed_values_for_problem = ["supervised", "unsupervised", "regression"]
        if problem not in allowed_values_for_problem:
            raise ValueError(f"Allowed values for problem {allowed_values_for_problem}. Instead got {problem}")
        
        if verbose is None:
            verbose = default_verbose
        if not isinstance(verbose, bool):
            raise TypeError(f"verbose must be of type bool (got {type(verbose)})")

        self.problem = problem
        self.verbose = verbose

    def saveHyperparameters(self, filename=default_hyperparameters_filename):
        """
        This function allows SAVING the transformation options to images in a Pickle object.
        This point is basically to be able to reproduce the experiments or reuse the transformation
        on unlabelled data.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        if self.verbose:
            print("It has been successfully saved in " + filename)

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
            print("It has been successfully loaded from " + filename)
        
    def generateImages_fit(self, data, folder):
        """
        This function generates and saves the synthetic images in folders.
        Arguments
        ---------
        data: str or Dataframe
            The data and targets
        folder: str
            The folder where the images are created
        """
        self.folder = folder

        # Get the data
        if type(data) == str:
            dataset = pd.read_csv(data)
        elif isinstance(data,pd.DataFrame) :
            dataset = data

        if self.verbose:
            print("Loaded data")

        # Separate targets from data
        if self.problem=="supervised"  or  self.problem=="regression":
            # The data includes the targets
            x = dataset.drop(dataset.columns[-1], axis="columns")
            y = dataset[dataset.columns[-1]]
        else:
            # The data doesn't include the targets
            x = dataset
            y = None

        # Call the training function
        self._trainingAlg(x, y)

        if self.verbose:
            print("End")

    def generateImages_pred(self, data, folder):
      """
      This function generate and save the synthetic images in folders.
      - data : data CSV or pandas Dataframe
      - folder : the folder where the images are created
      """
      # Read the CSV
      self.folder = folder
      if type(data)==str:
        dataset = pd.read_csv(data)
      elif isinstance(data, pd.DataFrame):
        dataset = data

      # Separate targets from data
      if self.problem=="supervised"  or  self.problem=="regression":
          # The data includes the targets
          x = dataset.drop(dataset.columns[-1], axis="columns")
          y = dataset[dataset.columns[-1]]
      else:
          # The data doesn't include the targets
          x = dataset
          y = None

      self._testAlg(x, y)

      if self.verbose: print("End")


    @abstractmethod
    def _trainingAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        """This method is not to be called from the outside."""
        raise NotImplementedError()

    @abstractmethod
    def _testAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        """This method is not to be called from the outside."""
        raise NotImplementedError()
    