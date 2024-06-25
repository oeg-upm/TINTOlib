from abc import ABC, abstractmethod
import pickle
from typing import Optional

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
        
    @abstractmethod
    def generateImages(self, data, folder):
        pass
