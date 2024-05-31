from abc import ABC, abstractmethod
import pickle
from typing import Optional

default_problem = "supervised"  # Define the type of dataset [supervised, unsupervised, regression]
default_verbose = False         # Verbose: if it's true, show the compilation text

class AbstractImageMethod(ABC):
    """Abstract class that all the other classes must inherit from and implement the abstract functions"""

    def __init__(
        self,
        problem: Optional[str], 
        verbose: Optional[bool],
    ):
        if problem is None:
            problem = default_problem
        if verbose is None:
            verbose = default_verbose

        self.problem = problem
        self.verbose = verbose

        print("INIT", type(self))

    def saveHyperparameters(self, filename='objs'):
        """
        This function allows SAVING the transformation options to images in a Pickle object.
        This point is basically to be able to reproduce the experiments or reuse the transformation
        on unlabelled data.
        """
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self.__dict__, f)
        if self.verbose:
            print("It has been successfully saved in " + filename)

    def loadHyperparameters(self, filename='objs.pkl'):
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
